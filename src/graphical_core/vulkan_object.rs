mod frame;
mod lifecycle;
mod presentation;

use crate::graphical_core::camera::UniformBufferObject;
use crate::graphical_core::compute_cull::DepthPyramidResources;
use crate::graphical_core::ui_pipeline::UiPipeline;
use crate::graphical_core::voxel_pool::VoxelPool;
use crate::voxel::world::World;
use vulkan_rust::{vk, Device, Entry, Instance};

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
    pub sky_pipeline: vk::Pipeline,
    pub sky_pipeline_layout: vk::PipelineLayout,
}

/// World generates and streams terrain out to this distance around the player.
pub const WORLD_DISTANCE: i32 = 10;
/// Number of timestamp queries written per frame in `record_vertex_pull_command_buffer`.
/// Slot meanings (set by the recording code):
/// 0 = start, 1 = after sky, 2 = after phase1 mesh, 3 = after depth pyramid,
/// 4 = after phase2 mesh, 5 = before ui, 6 = after ui (= end).
pub const TIMING_QUERY_COUNT: u32 = 7;
const VOXEL_POOL_CAPACITY: usize = (i32::pow(2 * WORLD_DISTANCE + 1, 2) * 48) as usize;

/// World-specific resources created when entering a world, destroyed when returning to menu.
pub struct WorldResources {
    pub world: World,
    voxel_pool: VoxelPool,
    face_gen_pipeline: crate::graphical_core::face_gen_pipeline::FaceGenPipeline,
    vertex_pull_pipeline: crate::graphical_core::vertex_pull_pipeline::VertexPullPipeline,
    cull_compact_pipeline: crate::graphical_core::cull_compact::CullCompactPipeline,
    last_player_chunk: [i32; 3],
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
    world_resources: Option<WorldResources>,
    pub ui: UiPipeline,
    /// Single-slot timestamp query pool for per-stage GPU timing.
    /// Size = `TIMING_QUERY_COUNT`. Read back synchronously each frame.
    timing_query_pool: vk::QueryPool,
    timing_period_ns: f64,
}

impl VulkanApplication {
    /// Returns the world if one is loaded.
    pub fn world(&self) -> Option<&World> {
        self.world_resources.as_ref().map(|wr| &wr.world)
    }

    pub fn has_loaded_world(&self) -> bool {
        self.world_resources.is_some()
    }

    pub fn swapchain_extent(&self) -> vk::Extent2D {
        self.vulkan_application_data.swapchain_extent
    }
}
