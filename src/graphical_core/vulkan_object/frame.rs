//! Per-frame rendering: world frames, menu frames, and chunk streaming.

use super::{VulkanApplication, WorldResources, TIMING_QUERY_COUNT, WORLD_DISTANCE};
use crate::graphical_core::{
    camera::{update_uniform_buffer, Camera, EyeMatrices},
    commands::record_vertex_pull_command_buffer,
    compute_cull::CullPushConstants,
    frustum::Frustum,
    MAX_FRAMES_IN_FLIGHT,
};
use crate::voxel::chunk::CHUNK_SIZE;
use crate::voxel::grid::ChunkPos;
use crate::voxel::world::{TERRAIN_MAX_CY, TERRAIN_MIN_CY, World};
use vulkan_rust::vk;
use winit::window::Window;

fn neighbors_all_completely_solid(world: &World, chunk_pos: ChunkPos) -> bool {
    let neighbor_solid = |neighbor_pos: ChunkPos| -> bool {
        match world.get_chunk_at(neighbor_pos) {
            Some(chunk) => chunk.contains_no_air(),
            None => {
                (TERRAIN_MIN_CY..=TERRAIN_MAX_CY).contains(&neighbor_pos.y)
            }
        }
    };
    let neighbors = [
        chunk_pos.offset(1, 0, 0),
        chunk_pos.offset(-1, 0, 0),
        chunk_pos.offset(0, 1, 0),
        chunk_pos.offset(0, -1, 0),
        chunk_pos.offset(0, 0, 1),
        chunk_pos.offset(0, 0, -1),
    ];
    neighbors.iter().all(|&neighbor_pos| neighbor_solid(neighbor_pos))
}

impl VulkanApplication {
    /// Acquires a swapchain image, submits the command buffer, and presents the result. 
    /// Also keeps check of performance of every specific step in the rendering.
    /// 
    /// # Safety
    /// Calls unsafe Vulkan queue and synchronization APIs.
    pub unsafe fn render_frame(&mut self, window: &Window, camera: &Camera, eyes: &EyeMatrices) -> anyhow::Result<()> {
        let t_total = std::time::Instant::now();

        let t0 = std::time::Instant::now();
        let world_resources = self.world_resources.as_mut().expect("render_frame called without a loaded world");
        Self::update_chunks_inner(world_resources, camera)?;
        let dt_update_chunks = t0.elapsed();
        let resident_count = world_resources.voxel_pool.chunk_count();

        let t1 = std::time::Instant::now();
        let image_index = match self.acquire_next_image(window)? {
            Some(index) => index,
            None => return Ok(()),
        };
        let dt_acquire = t1.elapsed();
        update_uniform_buffer(&self.vulkan_application_data, eyes)?;

        let world_resources = self.world_resources.as_ref().unwrap();
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
            chunk_count: world_resources.voxel_pool.chunk_count(),
            screen_size: [
                self.vulkan_application_data.swapchain_extent.width as f32,
                self.vulkan_application_data.swapchain_extent.height as f32,
            ],
            phase: 1,
            draw_offset: crate::voxel::block::BlockType::opaque_mask(),
            stereo: if eyes.is_stereo() { 1 } else { 0 },
            _pad: [0.0; 2],
        };

        let t2 = std::time::Instant::now();
        record_vertex_pull_command_buffer(
            &self.device,
            &self.vulkan_application_data,
            image_index,
            &world_resources.face_gen_pipeline,
            &world_resources.vertex_pull_pipeline,
            &world_resources.cull_compact_pipeline,
            &world_resources.voxel_pool,
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

    /// Pump chunk loading without rendering.
    pub unsafe fn update_world(&mut self, camera: &Camera) -> anyhow::Result<()> {
        if let Some(wr) = self.world_resources.as_mut() {
            Self::update_chunks_inner(wr, camera)?;
        }
        Ok(())
    }

    /// Loads/unloads chunks as the player moves.
    unsafe fn update_chunks_inner(world_resources: &mut WorldResources, camera: &Camera) -> anyhow::Result<()> {
        let player_position = camera.position.as_dvec3();
        let player_chunk_x = (camera.position.x / CHUNK_SIZE as f32).floor() as i32;
        let player_chunk_y = (camera.position.y / CHUNK_SIZE as f32).floor() as i32;
        let player_chunk_z = (camera.position.z / CHUNK_SIZE as f32).floor() as i32;
        let changed_chunks = world_resources.world.update(player_position, WORLD_DISTANCE);

        for chunk_pos in &changed_chunks.unloaded_chunks {
            world_resources.voxel_pool.invalidate_neighbor_boundaries(*chunk_pos, &world_resources.world);
            world_resources.voxel_pool.remove_chunk(chunk_pos);
        }
        let mut newly_solid_chunks: Vec<ChunkPos> = Vec::new();
        for pos in &changed_chunks.loaded_chunks {
            if world_resources.voxel_pool.has_chunk(pos) {
                continue;
            }
            if let Some(chunk) = world_resources.world.get_chunk_at(*pos) {
                if chunk.contains_only_air() {
                    continue;
                }
                if chunk.contains_no_air() {
                    newly_solid_chunks.push(*pos);
                    if neighbors_all_completely_solid(&world_resources.world, *pos) {
                        continue;
                    }
                }
                let chunk_ptr = chunk as *const _;
                world_resources.voxel_pool.upload_chunk(*pos, &*chunk_ptr, &world_resources.world);
                world_resources.voxel_pool.invalidate_neighbor_boundaries(*pos, &world_resources.world);
            }
        }
        // Second sweep: each newly-opaque chunk may have just turned a
        // previously-uploaded neighbor into a buried chunk. Re-check the
        // 6 axis-neighbors of every newly-opaque chunk and evict any that
        // are now buried.
        let mut to_evict: std::collections::HashSet<ChunkPos> = Default::default();
        for &chunk_pos in &newly_solid_chunks {
            let neighbors = [
                chunk_pos.offset(1, 0, 0),
                chunk_pos.offset(-1, 0, 0),
                chunk_pos.offset(0, 1, 0),
                chunk_pos.offset(0, -1, 0),
                chunk_pos.offset(0, 0, 1),
                chunk_pos.offset(0, 0, -1),
            ];
            for neighbor in neighbors {
                if !world_resources.voxel_pool.has_chunk(&neighbor) {
                    continue;
                }
                let Some(neighbor_chunk) = world_resources.world.get_chunk_at(neighbor) else {
                    continue;
                };
                if neighbor_chunk.contains_no_air() && neighbors_all_completely_solid(&world_resources.world, neighbor) {
                    to_evict.insert(neighbor);
                }
            }
        }
        for chunk_pos in to_evict {
            world_resources.voxel_pool.invalidate_neighbor_boundaries(chunk_pos, &world_resources.world);
            world_resources.voxel_pool.remove_chunk(&chunk_pos);
        }

        world_resources.last_player_chunk = [player_chunk_x, player_chunk_y, player_chunk_z];
        Ok(())
    }
}

fn identify_solid_chunks() {
    let mut newly_solid_chunks: Vec<ChunkPos> = Vec::new();
    todo!()
}
