use crate::graphical_core::{
    camera::{create_uniform_buffer, destroy_uniform_buffer, update_uniform_buffer, Camera, EyeMatrices, UniformBufferObject},
    commands::{allocate_command_buffers, create_command_pool, create_frame_buffers, create_sync_objects, record_mesh_shader_command_buffer},
    compute_cull::{CullPushConstants, DepthPyramidResources},
    depth::{create_depth_image, create_depth_pyramid, destroy_depth_image, destroy_depth_pyramid},
    descriptors,
    frustum::Frustum,
    gpu::choose_gpu,
    instance::{create_instance, create_logical_device},
    pipeline::create_sky_pipeline,
    render_pass::create_render_pass,
    swapchain::{create_swapchain, create_swapchain_image_views},
    texture_mapping::{create_texture_image, destroy_textures},
    ui_pipeline::UiPipeline,
    voxel_pool::VoxelPool,
    MAX_FRAMES_IN_FLIGHT,
};
use crate::voxel::chunk::CHUNK_SIZE;
use crate::voxel::grid::ChunkPos;
use crate::voxel::world::World;
use crate::VALIDATION_ENABLED;
use anyhow::anyhow;
use vk::Handle;
use vulkan_rust::{vk, Device, Entry, Instance, LibloadingLoader};
use winit::window::Window;

#[derive(Clone, Debug, Default)]
pub struct VulkanApplicationData {
    pub surface: vk::SurfaceKHR,
    pub debug_messenger: vk::DebugUtilsMessengerEXT,
    pub physical_device: vk::PhysicalDevice,
    pub graphics_queue: vk::Queue,
    pub presentation_queue: vk::Queue,
    pub swapchain_format: vk::Format,
    pub swapchain_extent: vk::Extent2D,
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_image_views: Vec<vk::ImageView>,
    pub render_pass: vk::RenderPass,
    pub render_pass_load: vk::RenderPass,
    pub framebuffers: Vec<vk::Framebuffer>,
    pub command_pool: vk::CommandPool,
    pub command_buffers: Vec<vk::CommandBuffer>,
    pub image_available_semaphores: Vec<vk::Semaphore>,
    pub render_finished_semaphores: Vec<vk::Semaphore>,
    pub(crate) in_flight_fences: Vec<vk::Fence>,
    pub(crate) images_in_flight: Vec<vk::Fence>,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub texture_image: vk::Image,
    pub texture_memory: vk::DeviceMemory,
    pub texture_image_view: vk::ImageView,
    pub texture_sampler: vk::Sampler,
    pub descriptor_set: vk::DescriptorSet,
    pub uniform_buffer: vk::Buffer,
    pub uniform_buffer_memory: vk::DeviceMemory,
    pub uniform_buffer_ptr: *mut UniformBufferObject,
    pub depth_image: vk::Image,
    pub depth_image_memory: vk::DeviceMemory,
    pub depth_image_view: vk::ImageView,
    pub depth_pyramid_image: vk::Image,
    pub depth_pyramid_memory: vk::DeviceMemory,
    pub depth_pyramid_mip_views: Vec<vk::ImageView>,
    pub depth_pyramid_full_view: vk::ImageView,
    pub depth_pyramid_sampler: vk::Sampler,
    pub depth_pyramid_mip_count: u32,
    pub palette_buffer: vk::Buffer,
    pub palette_buffer_memory: vk::DeviceMemory,
    pub sky_pipeline: vk::Pipeline,
    pub sky_pipeline_layout: vk::PipelineLayout,
}

/// World generates and streams terrain out to this distance around the player.
pub const WORLD_DISTANCE: i32 = 10;
/// Sized to hold the working set: a `(2·WORLD_DISTANCE+1)²` column window
/// around the player times the radial column height, with generous slack.
/// At `WORLD_DISTANCE=36` and 48-chunk-tall columns the bound is
/// `73² × 48 ≈ 256k` worst case; we round to a power of two for slot
/// arithmetic. Pinned by `world_resident_set_is_bounded_by_render_distance`.
const MAX_MESH_CHUNKS: usize = 262144;
/// See `CullPushConstants::planet_radius` at its use site in `render_frame`.
const NO_HORIZON_CULL_RADIUS: f32 = 1.0e9;

/// World-specific resources created when entering a world, destroyed when returning to menu.
pub struct WorldResources {
    pub world: World,
    voxel_pool: VoxelPool,
    face_gen_pipeline: crate::graphical_core::face_gen_pipeline::FaceGenPipeline,
    vertex_pull_pipeline: crate::graphical_core::vertex_pull_pipeline::VertexPullPipeline,
    cull_compact_pipeline: crate::graphical_core::cull_compact::CullCompactPipeline,
    last_player_chunk: [i32; 3],
    seed: u32,
}

pub struct VulkanApplication {
    _vulkan_entry_point: Entry,
    vulkan_instance: Instance,
    vulkan_application_data: VulkanApplicationData,
    device: Device,
    frame: usize,
    pub(crate) resized: bool,
    depth_pyramid_pipeline: DepthPyramidResources,
    depth_pyramid_needs_init: bool,
    /// None when in the menu, Some when a world is loaded.
    wr: Option<WorldResources>,
    pub ui: UiPipeline,
    /// Single-slot timestamp query pool for per-stage GPU timing.
    /// Size = `TIMING_QUERY_COUNT`. Read back synchronously each frame.
    timing_query_pool: vk::QueryPool,
    timing_period_ns: f64,
}

/// Number of timestamp queries written per frame in `record_mesh_shader_command_buffer`.
/// Slot meanings (set by the recording code):
/// 0 = start, 1 = after sky, 2 = after phase1 mesh, 3 = after depth pyramid,
/// 4 = after phase2 mesh, 5 = unused, 6 = after ui (= end).
pub const TIMING_QUERY_COUNT: u32 = 7;

/// True iff every one of `cp`'s six axis-neighbors is uniform-opaque (or out
/// of generated range, which the world treats as solid below the terrain
/// layer). A uniform-opaque chunk with this property is buried — none of its
/// faces touch air, so it emits no geometry and can be skipped at upload.
fn neighbors_all_opaque(world: &World, cp: ChunkPos) -> bool {
    let neighbor_solid = |nb: ChunkPos| -> bool {
        match world.get_chunk_at(nb) {
            Some(chunk) => chunk.contains_no_air(),
            None => {
                // Missing chunk → solid only if it's inside the radial terrain
                // band. Above the band is sky.
                (crate::voxel::world::TERRAIN_MIN_CY..=crate::voxel::world::TERRAIN_MAX_CY).contains(&nb.y)
            }
        }
    };
    let neighbors = [
        cp.offset(1, 0, 0),
        cp.offset(-1, 0, 0),
        cp.offset(0, 1, 0),
        cp.offset(0, -1, 0),
        cp.offset(0, 0, 1),
        cp.offset(0, 0, -1),
    ];
    neighbors.iter().all(|&n| neighbor_solid(n))
}

impl VulkanApplication {
    /// Returns the world if one is loaded.
    pub fn world(&self) -> Option<&World> {
        self.wr.as_ref().map(|wr| &wr.world)
    }

    /// Returns true if a world is currently loaded.
    pub fn has_world(&self) -> bool {
        self.wr.is_some()
    }

    /// True if LOD generation has settled. There is no separate LOD tier
    /// beyond the mesh pool right now, so this is trivially always true.
    pub fn lod_settled(&self) -> bool {
        true
    }

    pub fn swapchain_extent(&self) -> vk::Extent2D {
        self.vulkan_application_data.swapchain_extent
    }
}

impl VulkanApplication {
    /// Creates the core Vulkan renderer without loading a world.
    /// Call `enter_world()` to load a world before rendering game frames.
    ///
    /// # Safety
    /// Calls unsafe Vulkan APIs. The caller must call [`destroy_vulkan_application`]
    /// before dropping the returned value or closing the window.
    pub unsafe fn create_vulkan_application(user_window: &Window) -> anyhow::Result<Self> {
        let CoreInfrastructure {
            entry,
            instance,
            device,
            mut data,
        } = create_core_infrastructure(user_window)?;
        create_presentation_pipeline(user_window, &instance, &device, &mut data)?;
        create_resources(&device, &instance, &mut data)?;
        allocate_command_buffers(&device, &mut data)?;
        create_sync_objects(&device, &mut data)?;
        let depth_pyramid_pipeline = crate::graphical_core::compute_cull::create_depth_pyramid_pipeline(&device, &data)?;
        let ui = UiPipeline::create(&device, &instance, &mut data)?;

        // Per-stage GPU timestamp query pool. One pool with TIMING_QUERY_COUNT
        // slots — we read it back synchronously after each frame, so no
        // double-buffering is needed.
        let qp_info = vk::QueryPoolCreateInfo::builder()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(TIMING_QUERY_COUNT);
        let timing_query_pool = device.create_query_pool(&qp_info, None)?;
        let props = instance.get_physical_device_properties(data.physical_device);
        let timing_period_ns = props.limits.timestamp_period as f64;
        // Clear debug.log on startup so the user can see whether the new
        // perf line ever gets written this run.
        let _ = std::fs::write("debug.log", format!("[startup] timestamp_period={} ns\n", timing_period_ns));

        Ok(Self {
            _vulkan_entry_point: entry,
            vulkan_instance: instance,
            vulkan_application_data: data,
            device,
            frame: 0,
            resized: false,
            depth_pyramid_pipeline,
            depth_pyramid_needs_init: true,
            wr: None,
            ui,
            timing_query_pool,
            timing_period_ns,
        })
    }

    /// Load a world and create all GPU resources for rendering it.
    ///
    /// # Safety
    /// Calls unsafe Vulkan APIs.
    pub unsafe fn enter_world(&mut self, _world_dir: &std::path::Path, seed: u32) -> anyhow::Result<()> {
        let mut world = World::new(seed);
        let spawn = glam::DVec3::new(0.0, 90.0, 0.0);
        world.update(spawn, WORLD_DISTANCE);

        let voxel_pool = VoxelPool::new(
            MAX_MESH_CHUNKS as u32,
            &self.device,
            &self.vulkan_instance,
            &mut self.vulkan_application_data,
        )?;
        // Note: chunks are uploaded per-frame in `update_chunks_inner` as the
        // chunk generator threads finish — not here. Here the world is empty.
        let face_gen_pipeline = crate::graphical_core::face_gen_pipeline::FaceGenPipeline::create(&self.device, &voxel_pool)?;
        let vertex_pull_pipeline =
            crate::graphical_core::vertex_pull_pipeline::VertexPullPipeline::create(&self.device, &self.vulkan_application_data, &voxel_pool)?;
        let cull_compact_pipeline =
            crate::graphical_core::cull_compact::CullCompactPipeline::create(&self.device, &self.vulkan_application_data, &voxel_pool)?;

        self.wr = Some(WorldResources {
            world,
            voxel_pool,
            face_gen_pipeline,
            vertex_pull_pipeline,
            cull_compact_pipeline,
            last_player_chunk: [0, 0, 0],
            seed,
        });
        self.depth_pyramid_needs_init = true;
        Ok(())
    }

    /// Unload the current world and return to menu state.
    ///
    /// # Safety
    /// Calls unsafe Vulkan APIs.
    pub unsafe fn exit_world(&mut self) {
        self.device.device_wait_idle().unwrap();
        if let Some(mut wr) = self.wr.take() {
            wr.cull_compact_pipeline.destroy(&self.device);
            wr.vertex_pull_pipeline.destroy(&self.device);
            wr.face_gen_pipeline.destroy(&self.device);
            wr.voxel_pool.destroy(&self.device);
        }
    }
}

struct CoreInfrastructure {
    entry: Entry,
    instance: Instance,
    device: Device,
    data: VulkanApplicationData,
}

unsafe fn create_core_infrastructure(window: &Window) -> anyhow::Result<CoreInfrastructure> {
    let loader = LibloadingLoader::new().map_err(|e| anyhow!("{}", e))?;
    let entry = unsafe { Entry::new(loader) }.map_err(|b| anyhow!("{}", b))?;
    let mut data = VulkanApplicationData::default();
    let instance = create_instance(window, &entry, &mut data)?;
    data.surface = instance.create_surface(&window, &window, None).map_err(|e| anyhow!("{}", e))?;

    choose_gpu(&instance, &mut data, None)?;
    let device = create_logical_device(&entry, &instance, &mut data)?;

    Ok(CoreInfrastructure { entry, instance, device, data })
}

unsafe fn create_presentation_pipeline(
    window: &Window,
    instance: &Instance,
    device: &Device,
    data: &mut VulkanApplicationData,
) -> anyhow::Result<()> {
    create_swapchain(window, instance, device, data)?;
    create_swapchain_image_views(device, data)?;
    create_depth_image(device, instance, data)?;
    create_depth_pyramid(device, instance, data)?;
    create_render_pass(instance, device, data)?;
    descriptors::create_layout(device, data)?;
    create_sky_pipeline(device, data)?;
    create_frame_buffers(device, data)?;
    Ok(())
}

unsafe fn create_resources(device: &Device, instance: &Instance, data: &mut VulkanApplicationData) -> anyhow::Result<()> {
    create_command_pool(instance, device, data)?;
    create_uniform_buffer(device, instance, data)?;
    let (texture_image, texture_memory, texture_image_view, texture_sampler) = create_texture_image(device, instance, data)?;
    descriptors::create_pool(device, data)?;
    let descriptor_sets = descriptors::allocate_set(device, data.descriptor_pool, data.descriptor_set_layout)?;
    let descriptor_set = descriptor_sets
        .first()
        .copied()
        .ok_or_else(|| anyhow!("Failed to allocate descriptor set"))?;
    descriptors::update_set(
        device,
        descriptor_set,
        texture_image_view,
        texture_sampler,
        data.uniform_buffer,
    );

    data.texture_image = texture_image;
    data.texture_memory = texture_memory;
    data.texture_image_view = texture_image_view;
    data.texture_sampler = texture_sampler;
    data.descriptor_set = descriptor_set;

    Ok(())
}

impl VulkanApplication {
    /// Acquires a swapchain image, submits the command buffer, and presents the result.
    ///
    /// # Safety
    /// Calls unsafe Vulkan queue and synchronization APIs.
    pub unsafe fn render_frame(&mut self, window: &Window, camera: &Camera, eyes: &EyeMatrices) -> anyhow::Result<()> {
        let t_total = std::time::Instant::now();

        let t0 = std::time::Instant::now();
        let wr = self.wr.as_mut().expect("render_frame called without a loaded world");
        Self::update_chunks_inner(wr, camera)?;
        let dt_update_chunks = t0.elapsed();
        let resident_count = wr.voxel_pool.chunk_count();

        let t1 = std::time::Instant::now();
        let image_index = match self.acquire_next_image(window)? {
            Some(index) => index,
            None => return Ok(()),
        };
        let dt_acquire = t1.elapsed();
        update_uniform_buffer(&self.vulkan_application_data, eyes)?;

        let wr = self.wr.as_ref().unwrap();
        let frustum = if eyes.is_stereo() {
            Frustum::combined_stereo(&eyes.view_projection[0], &eyes.view_projection[1])
        } else {
            Frustum::from_view_projection(&eyes.primary_vp())
        };
        let cull_push = CullPushConstants {
            planes: [
                frustum.plane(0),
                frustum.plane(1),
                frustum.plane(2),
                frustum.plane(3),
                frustum.plane(4),
                frustum.plane(5),
            ],
            camera_pos: camera.position.to_array(),
            chunk_count: wr.voxel_pool.chunk_count(),
            screen_size: [
                self.vulkan_application_data.swapchain_extent.width as f32,
                self.vulkan_application_data.swapchain_extent.height as f32,
            ],
            phase: 1,
            draw_offset: crate::voxel::block::BlockType::opaque_mask(),
            // No planet in the flat-world renderer; a radius this large keeps
            // the shader's horizon-culling branch (`cam_dist <= planet_radius`)
            // always true, i.e. a no-op.
            planet_radius: NO_HORIZON_CULL_RADIUS,
            stereo: if eyes.is_stereo() { 1 } else { 0 },
            _pad: [0.0; 2],
        };

        let t2 = std::time::Instant::now();
        record_mesh_shader_command_buffer(
            &self.device,
            &self.vulkan_application_data,
            image_index,
            &wr.face_gen_pipeline,
            &wr.vertex_pull_pipeline,
            &wr.cull_compact_pipeline,
            &wr.voxel_pool,
            &self.depth_pyramid_pipeline,
            &cull_push,
            self.depth_pyramid_needs_init,
            &self.ui,
            self.timing_query_pool,
        )?;
        self.depth_pyramid_needs_init = false;
        let dt_record = t2.elapsed();

        let t3 = std::time::Instant::now();
        self.submit_command_buffer(image_index)?;
        let dt_submit = t3.elapsed();

        let t4 = std::time::Instant::now();
        self.present_frame(image_index, window)?;
        let dt_present = t4.elapsed();

        // Read back GPU timestamps from THIS frame (blocks until done; the
        // CPU is normally idle during this window because acquire blocks
        // anyway). Convert tick deltas → ms via timestamp_period.
        let mut ts = [0u64; TIMING_QUERY_COUNT as usize];
        let _ = self.device.get_query_pool_results(
            self.timing_query_pool,
            0,
            TIMING_QUERY_COUNT,
            std::mem::size_of_val(&ts),
            ts.as_mut_ptr() as *mut std::ffi::c_void,
            std::mem::size_of::<u64>() as u64,
            vk::QueryResultFlags::_64 | vk::QueryResultFlags::WAIT,
        );
        let to_ms = |ticks: u64| -> f64 { (ticks as f64) * self.timing_period_ns / 1_000_000.0 };
        let gpu_sky = to_ms(ts[1] - ts[0]);
        let gpu_phase1 = to_ms(ts[2] - ts[1]);
        let gpu_pyramid = to_ms(ts[3] - ts[2]);
        let gpu_phase2 = to_ms(ts[4] - ts[3]);
        let gpu_ui = to_ms(ts[6] - ts[5]);
        let gpu_total = to_ms(ts[6] - ts[0]);

        self.frame = (self.frame + 1) % MAX_FRAMES_IN_FLIGHT;
        let dt_total = t_total.elapsed();

        // Rolling 60-frame average; write to debug.log once per second.
        struct PerfAccum {
            update_chunks: u128,
            acquire: u128,
            record: u128,
            submit: u128,
            present: u128,
            total: u128,
            gpu_sky: f64,
            gpu_phase1: f64,
            gpu_pyramid: f64,
            gpu_phase2: f64,
            gpu_ui: f64,
            gpu_total: f64,
            n: u32,
        }
        static PERF: std::sync::Mutex<PerfAccum> = std::sync::Mutex::new(PerfAccum {
            update_chunks: 0,
            acquire: 0,
            record: 0,
            submit: 0,
            present: 0,
            total: 0,
            gpu_sky: 0.0,
            gpu_phase1: 0.0,
            gpu_pyramid: 0.0,
            gpu_phase2: 0.0,
            gpu_ui: 0.0,
            gpu_total: 0.0,
            n: 0,
        });
        let mut p = PERF.lock().unwrap();
        p.update_chunks += dt_update_chunks.as_micros();
        p.acquire += dt_acquire.as_micros();
        p.record += dt_record.as_micros();
        p.submit += dt_submit.as_micros();
        p.present += dt_present.as_micros();
        p.total += dt_total.as_micros();
        p.gpu_sky += gpu_sky;
        p.gpu_phase1 += gpu_phase1;
        p.gpu_pyramid += gpu_pyramid;
        p.gpu_phase2 += gpu_phase2;
        p.gpu_ui += gpu_ui;
        p.gpu_total += gpu_total;
        p.n += 1;
        if p.n >= 60 {
            let n = p.n as f64;
            let resident = resident_count;
            let msg = format!(
                "[perf avg over {}f] cpu_total={:.1}ms acquire={:.1} record={:.1} submit={:.1} present={:.1} update={:.1} | gpu_total={:.1}ms sky={:.1} phase1={:.1} pyramid={:.1} phase2={:.1} ui={:.1} | resident={}\n",
                p.n,
                p.total as f64 / n / 1000.0,
                p.acquire as f64 / n / 1000.0,
                p.record as f64 / n / 1000.0,
                p.submit as f64 / n / 1000.0,
                p.present as f64 / n / 1000.0,
                p.update_chunks as f64 / n / 1000.0,
                p.gpu_total / n,
                p.gpu_sky / n,
                p.gpu_phase1 / n,
                p.gpu_pyramid / n,
                p.gpu_phase2 / n,
                p.gpu_ui / n,
                resident,
            );
            let _ = std::fs::write("debug.log", &msg);
            *p = PerfAccum {
                update_chunks: 0,
                acquire: 0,
                record: 0,
                submit: 0,
                present: 0,
                total: 0,
                gpu_sky: 0.0,
                gpu_phase1: 0.0,
                gpu_pyramid: 0.0,
                gpu_phase2: 0.0,
                gpu_ui: 0.0,
                gpu_total: 0.0,
                n: 0,
            };
        }
        Ok(())
    }

    /// Render a menu frame (sky background + UI overlay).
    pub unsafe fn render_menu_frame(&mut self, window: &Window, eyes: &EyeMatrices) -> anyhow::Result<()> {
        let image_index = match self.acquire_next_image(window)? {
            Some(index) => index,
            None => return Ok(()),
        };
        update_uniform_buffer(&self.vulkan_application_data, eyes)?;

        let cmd = self.vulkan_application_data.command_buffers[image_index];
        self.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
        self.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::builder())?;

        crate::graphical_core::commands::begin_render_pass(&self.device, cmd, &self.vulkan_application_data, image_index);
        crate::graphical_core::commands::draw_sky(&self.device, cmd, &self.vulkan_application_data);
        let screen = [
            self.vulkan_application_data.swapchain_extent.width as f32,
            self.vulkan_application_data.swapchain_extent.height as f32,
        ];
        self.ui.record(&self.device, cmd, screen);
        self.device.cmd_end_render_pass(cmd);
        self.device.end_command_buffer(cmd)?;

        self.submit_command_buffer(image_index)?;
        self.present_frame(image_index, window)?;
        self.frame = (self.frame + 1) % MAX_FRAMES_IN_FLIGHT;
        Ok(())
    }

    /// Pump chunk loading without rendering. Call during pre-generation to
    /// stream chunks into the mesh pool ahead of world entry.
    pub unsafe fn update_world(&mut self, camera: &Camera) -> anyhow::Result<()> {
        if let Some(wr) = self.wr.as_mut() {
            Self::update_chunks_inner(wr, camera)?;
        }
        Ok(())
    }

    /// Loads/unloads chunks into the mesh pool as the player moves.
    unsafe fn update_chunks_inner(wr: &mut WorldResources, camera: &Camera) -> anyhow::Result<()> {
        let player_world = camera.position.as_dvec3();
        let player_cx = (camera.position.x / CHUNK_SIZE as f32).floor() as i32;
        let player_cy = (camera.position.y / CHUNK_SIZE as f32).floor() as i32;
        let player_cz = (camera.position.z / CHUNK_SIZE as f32).floor() as i32;
        let delta = wr.world.update(player_world, WORLD_DISTANCE);

        for pos in &delta.unloaded_chunks {
            wr.voxel_pool.invalidate_neighbor_boundaries(*pos, &wr.world);
            wr.voxel_pool.remove_chunk(pos);
        }
        // Track which newly-loaded chunks were uniform-opaque so we can
        // re-check their neighbors for "now buried" after this batch is done.
        let mut newly_opaque: Vec<ChunkPos> = Vec::new();
        for pos in &delta.loaded_chunks {
            if wr.voxel_pool.has_chunk(pos) {
                continue;
            }
            if let Some(chunk) = wr.world.get_chunk_at(*pos) {
                if chunk.contains_only_air() {
                    continue;
                }
                if chunk.contains_no_air() {
                    newly_opaque.push(*pos);
                    if neighbors_all_opaque(&wr.world, *pos) {
                        continue;
                    }
                }
                let chunk_ptr = chunk as *const _;
                wr.voxel_pool.upload_chunk(*pos, &*chunk_ptr, &wr.world);
                wr.voxel_pool.invalidate_neighbor_boundaries(*pos, &wr.world);
            }
        }
        // Second sweep: each newly-opaque chunk may have just turned a
        // previously-uploaded neighbor into a buried chunk. Re-check the
        // 6 axis-neighbors of every newly-opaque chunk and evict any that
        // are now buried.
        let mut to_evict: std::collections::HashSet<ChunkPos> = Default::default();
        for &cp in &newly_opaque {
            let neighbors = [
                cp.offset(1, 0, 0),
                cp.offset(-1, 0, 0),
                cp.offset(0, 1, 0),
                cp.offset(0, -1, 0),
                cp.offset(0, 0, 1),
                cp.offset(0, 0, -1),
            ];
            for neighbor in neighbors {
                if !wr.voxel_pool.has_chunk(&neighbor) {
                    continue;
                }
                let Some(neighbor_chunk) = wr.world.get_chunk_at(neighbor) else { continue };
                if neighbor_chunk.contains_no_air() && neighbors_all_opaque(&wr.world, neighbor) {
                    to_evict.insert(neighbor);
                }
            }
        }
        for cp in to_evict {
            wr.voxel_pool.invalidate_neighbor_boundaries(cp, &wr.world);
            wr.voxel_pool.remove_chunk(&cp);
        }

        wr.last_player_chunk = [player_cx, player_cy, player_cz];
        Ok(())
    }
}

impl VulkanApplication {
    /// Waits for the current frame's fence, then acquires the next swapchain image.
    /// Returns `None` if the swapchain was out of date and had to be recreated.
    unsafe fn acquire_next_image(&mut self, window: &Window) -> anyhow::Result<Option<usize>> {
        let data = &self.vulkan_application_data;
        self.device.wait_for_fences(&[data.in_flight_fences[self.frame]], true, u64::MAX)?;

        let result = self
            .device
            .acquire_next_image_khr(data.swapchain, u64::MAX, data.image_available_semaphores[self.frame], vk::Fence::null());
        let image_index = match result {
            Ok(index) => index as usize,
            Err(e) if e == vk::Result::ERROR_OUT_OF_DATE => {
                self.recreate_swapchain(window)?;
                return Ok(None);
            }
            Err(e) => return Err(anyhow!("{e:?}")),
        };

        if !self.vulkan_application_data.images_in_flight[image_index].is_null() {
            self.device
                .wait_for_fences(&[self.vulkan_application_data.images_in_flight[image_index]], true, u64::MAX)?;
        }
        self.vulkan_application_data.images_in_flight[image_index] = self.vulkan_application_data.in_flight_fences[self.frame];

        Ok(Some(image_index))
    }

    unsafe fn submit_command_buffer(&self, image_index: usize) -> anyhow::Result<()> {
        let data = &self.vulkan_application_data;
        let wait_semaphores = &[data.image_available_semaphores[self.frame]];
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = &[data.command_buffers[image_index]];
        let signal_semaphores = &[data.render_finished_semaphores[self.frame]];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores);

        self.device.reset_fences(&[data.in_flight_fences[self.frame]])?;
        self.device
            .queue_submit(data.graphics_queue, &[*submit_info], data.in_flight_fences[self.frame])?;
        Ok(())
    }

    unsafe fn present_frame(&mut self, image_index: usize, window: &Window) -> anyhow::Result<()> {
        let data = &self.vulkan_application_data;
        let signal_semaphores = &[data.render_finished_semaphores[self.frame]];
        let swapchains = &[data.swapchain];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);

        let result = self.device.queue_present_khr(data.presentation_queue, &present_info);

        if result == Err(vk::Result::ERROR_OUT_OF_DATE) {
            self.recreate_swapchain(window)?;
        }
        Ok(())
    }

    /// Destroys and rebuilds the swapchain and all dependent resources.
    ///
    /// Required when the window resizes or the swapchain becomes suboptimal,
    /// because most pipeline resources reference swapchain dimensions or format.
    ///
    /// # Safety
    /// Calls unsafe Vulkan destruction and creation APIs.
    pub unsafe fn recreate_swapchain(&mut self, user_window: &Window) -> anyhow::Result<()> {
        use crate::graphical_core::compute_cull;
        self.device.device_wait_idle()?;
        compute_cull::destroy_depth_pyramid_pipeline(&self.device, &self.depth_pyramid_pipeline);
        self.destroy_swapchain();
        create_swapchain(user_window, &self.vulkan_instance, &self.device, &mut self.vulkan_application_data)?;
        create_swapchain_image_views(&self.device, &mut self.vulkan_application_data)?;
        create_depth_image(&self.device, &self.vulkan_instance, &mut self.vulkan_application_data)?;
        create_depth_pyramid(&self.device, &self.vulkan_instance, &mut self.vulkan_application_data)?;
        self.depth_pyramid_pipeline = compute_cull::create_depth_pyramid_pipeline(&self.device, &self.vulkan_application_data)?;
        self.depth_pyramid_needs_init = true;
        create_render_pass(&self.vulkan_instance, &self.device, &mut self.vulkan_application_data)?;
        create_sky_pipeline(&self.device, &mut self.vulkan_application_data)?;
        create_frame_buffers(&self.device, &mut self.vulkan_application_data)?;
        allocate_command_buffers(&self.device, &mut self.vulkan_application_data)?;
        self.vulkan_application_data
            .images_in_flight
            .resize(self.vulkan_application_data.swapchain_images.len(), vk::Fence::null());
        // Re-bind depth pyramid descriptors for world pipelines (handles were invalidated)
        if let Some(wr) = &self.wr {
            wr.cull_compact_pipeline.update_depth_pyramid(&self.device, &self.vulkan_application_data);
        }
        Ok(())
    }

    /// Destroys the swapchain and all resources that depend on it.
    ///
    /// # Safety
    /// Calls unsafe Vulkan destruction APIs. The GPU must be idle before calling.
    pub unsafe fn destroy_swapchain(&mut self) {
        self.vulkan_application_data
            .framebuffers
            .iter()
            .for_each(|framebuffer| self.device.destroy_framebuffer(*framebuffer, None));
        self.device
            .free_command_buffers(self.vulkan_application_data.command_pool, &self.vulkan_application_data.command_buffers);
        self.device.destroy_pipeline(self.vulkan_application_data.sky_pipeline, None);
        self.device
            .destroy_pipeline_layout(self.vulkan_application_data.sky_pipeline_layout, None);
        self.device.destroy_render_pass(self.vulkan_application_data.render_pass, None);
        self.device.destroy_render_pass(self.vulkan_application_data.render_pass_load, None);
        destroy_depth_pyramid(&self.device, &mut self.vulkan_application_data);
        destroy_depth_image(&self.device, &mut self.vulkan_application_data);
        self.vulkan_application_data
            .swapchain_image_views
            .iter()
            .for_each(|image_view| self.device.destroy_image_view(*image_view, None));
        self.device.destroy_swapchain_khr(self.vulkan_application_data.swapchain, None);
    }

    /// Destroys all Vulkan resources. Must be called exactly once before the
    /// window closes, because `Drop` cannot guarantee the required destruction order.
    ///
    /// # Safety
    /// Calls unsafe Vulkan destruction APIs. No rendering is possible after this call.
    pub unsafe fn destroy_vulkan_application(&mut self) {
        self.device.device_wait_idle().unwrap();
        self.destroy_resources();
        self.destroy_swapchain();
        self.destroy_sync_objects();
        self.destroy_core_infrastructure();
    }

    unsafe fn destroy_resources(&mut self) {
        use crate::graphical_core::compute_cull;
        self.ui.destroy(&self.device);
        if let Some(mut wr) = self.wr.take() {
            wr.cull_compact_pipeline.destroy(&self.device);
            wr.vertex_pull_pipeline.destroy(&self.device);
            wr.face_gen_pipeline.destroy(&self.device);
            wr.voxel_pool.destroy(&self.device);
        }
        compute_cull::destroy_depth_pyramid_pipeline(&self.device, &self.depth_pyramid_pipeline);
        destroy_textures(&self.device, &mut self.vulkan_application_data);
        destroy_uniform_buffer(&self.device, &mut self.vulkan_application_data);
        self.device.destroy_descriptor_pool(self.vulkan_application_data.descriptor_pool, None);
        self.device
            .destroy_descriptor_set_layout(self.vulkan_application_data.descriptor_set_layout, None);
    }

    unsafe fn destroy_sync_objects(&self) {
        self.vulkan_application_data
            .in_flight_fences
            .iter()
            .for_each(|f| self.device.destroy_fence(*f, None));
        self.vulkan_application_data
            .render_finished_semaphores
            .iter()
            .for_each(|s| self.device.destroy_semaphore(*s, None));
        self.vulkan_application_data
            .image_available_semaphores
            .iter()
            .for_each(|s| self.device.destroy_semaphore(*s, None));
        self.device.destroy_command_pool(self.vulkan_application_data.command_pool, None);
    }

    unsafe fn destroy_core_infrastructure(&mut self) {
        self.device.destroy_device(None);
        self.vulkan_instance.destroy_surface(self.vulkan_application_data.surface, None);
        if VALIDATION_ENABLED {
            self.vulkan_instance
                .destroy_debug_utils_messenger_ext(self.vulkan_application_data.debug_messenger, None);
        }
        self.vulkan_instance.destroy_instance(None);
    }
}
