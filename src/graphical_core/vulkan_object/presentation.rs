//! Swapchain image acquire/submit/present and swapchain (re)creation.

use super::VulkanApplication;
use crate::graphical_core::{
    commands::{allocate_command_buffers, create_frame_buffers},
    compute_cull,
    depth::{create_depth_image, create_depth_pyramid, destroy_depth_image, destroy_depth_pyramid},
    pipeline::create_sky_pipeline,
    render_pass::create_render_pass,
    swapchain::{create_swapchain, create_swapchain_image_views},
};
use anyhow::anyhow;
use vk::Handle;
use vulkan_rust::vk;
use winit::window::Window;

impl VulkanApplication {
    /// Waits for the current frame's fence, then acquires the next swapchain image.
    /// Returns `None` if the swapchain was out of date and had to be recreated.
    pub(super) unsafe fn acquire_next_image(&mut self, window: &Window) -> anyhow::Result<Option<usize>> {
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

    pub(super) unsafe fn submit_command_buffer(&self, image_index: usize) -> anyhow::Result<()> {
        let data = &self.vulkan_application_data;
        let wait_semaphores = &[data.image_available_semaphores[self.frame]];
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = &[data.command_buffers[image_index]];
        let signal_semaphores = &[data.render_finished_semaphores[image_index]];
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

    pub(super) unsafe fn present_frame(&mut self, image_index: usize, window: &Window) -> anyhow::Result<()> {
        let data = &self.vulkan_application_data;
        let signal_semaphores = &[data.render_finished_semaphores[image_index]];
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
        if let Some(wr) = &self.world_resources {
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
}
