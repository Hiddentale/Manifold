//! Creation, world entry/exit, and full teardown of `VulkanApplication`.

use super::{VulkanApplication, VulkanApplicationData, WorldResources, TIMING_QUERY_COUNT, VOXEL_POOL_CAPACITY, WORLD_DISTANCE};
use crate::graphical_core::{
    camera::{create_uniform_buffer, destroy_uniform_buffer},
    commands::{allocate_command_buffers, create_command_pool, create_frame_buffers, create_sync_objects},
    compute_cull,
    depth::{create_depth_image, create_depth_pyramid},
    descriptors,
    gpu::choose_gpu,
    instance::{create_instance, create_logical_device},
    pipeline::create_sky_pipeline,
    render_pass::create_render_pass,
    texture_mapping::{create_texture_image, destroy_textures},
    ui_pipeline::UiPipeline,
    voxel_pool::VoxelPool,
};
use crate::voxel::world::World;
use crate::VALIDATION_ENABLED;
use anyhow::anyhow;
use vulkan_rust::{vk, Device, Entry, Instance, LibloadingLoader};
use winit::window::Window;

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

    Ok(CoreInfrastructure {
        entry,
        instance,
        device,
        data,
    })
}

unsafe fn create_presentation_pipeline(
    window: &Window,
    instance: &Instance,
    device: &Device,
    data: &mut VulkanApplicationData,
) -> anyhow::Result<()> {
    crate::graphical_core::swapchain::create_swapchain(window, instance, device, data)?;
    crate::graphical_core::swapchain::create_swapchain_image_views(device, data)?;
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
    descriptors::update_set(device, descriptor_set, texture_image_view, texture_sampler, data.uniform_buffer);

    data.texture_image = texture_image;
    data.texture_memory = texture_memory;
    data.texture_image_view = texture_image_view;
    data.texture_sampler = texture_sampler;
    data.descriptor_set = descriptor_set;

    Ok(())
}

impl VulkanApplication {
    /// Creates the core Vulkan renderer without loading a world.
    /// Call `enter_world()` to load a world before rendering game frames.
    ///
    /// # Safety
    /// Calls unsafe Vulkan APIs. The caller must call [`Self::destroy_vulkan_application`]
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
        let depth_pyramid_pipeline = compute_cull::create_depth_pyramid_pipeline(&device, &data)?;
        let ui = UiPipeline::create(&device, &instance, &mut data)?;
        let query_pool_info = vk::QueryPoolCreateInfo::builder()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(TIMING_QUERY_COUNT);
        let timing_query_pool = device.create_query_pool(&query_pool_info, None)?;
        let properties = instance.get_physical_device_properties(data.physical_device);
        let timing_period_ns = properties.limits.timestamp_period as f64;

        Ok(Self {
            _vulkan_entry_point: entry,
            vulkan_instance: instance,
            vulkan_application_data: data,
            device,
            frame: 0,
            resized: false,
            depth_pyramid_pipeline,
            depth_pyramid_needs_init: true,
            world_resources: None,
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
            VOXEL_POOL_CAPACITY as u32,
            &self.device,
            &self.vulkan_instance,
            &mut self.vulkan_application_data,
        )?;
        let face_gen_pipeline = crate::graphical_core::face_gen_pipeline::FaceGenPipeline::create(&self.device, &voxel_pool)?;
        let vertex_pull_pipeline =
            crate::graphical_core::vertex_pull_pipeline::VertexPullPipeline::create(&self.device, &self.vulkan_application_data, &voxel_pool)?;
        let cull_compact_pipeline =
            crate::graphical_core::cull_compact::CullCompactPipeline::create(&self.device, &self.vulkan_application_data, &voxel_pool)?;

        self.world_resources = Some(WorldResources {
            world,
            voxel_pool,
            face_gen_pipeline,
            vertex_pull_pipeline,
            cull_compact_pipeline,
            last_player_chunk: [0, 0, 0],
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
        if let Some(mut wr) = self.world_resources.take() {
            wr.cull_compact_pipeline.destroy(&self.device);
            wr.vertex_pull_pipeline.destroy(&self.device);
            wr.face_gen_pipeline.destroy(&self.device);
            wr.voxel_pool.destroy(&self.device);
        }
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
        self.ui.destroy(&self.device);
        if let Some(mut wr) = self.world_resources.take() {
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
