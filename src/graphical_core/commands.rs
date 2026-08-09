use crate::graphical_core::compute_cull::{CullPushConstants, DepthPyramidResources, DepthReducePush};
use crate::graphical_core::cull_compact::CullCompactPipeline;
use crate::graphical_core::face_gen_pipeline::FaceGenPipeline;
use crate::graphical_core::vertex_pull_pipeline::VertexPullPipeline;
use crate::graphical_core::voxel_pool::VoxelPool;
use crate::graphical_core::vulkan_object::VulkanApplicationData;
use crate::graphical_core::{self, MAX_FRAMES_IN_FLIGHT};
use vk::Handle;
use vulkan_rust::{vk, Device, Instance};

/// Creates a framebuffer for each swapchain image view, attaching color and depth.
pub unsafe fn create_frame_buffers(device: &Device, data: &mut VulkanApplicationData) -> anyhow::Result<()> {
    data.framebuffers = data
        .swapchain_image_views
        .iter()
        .map(|i| {
            let attachments = &[*i, data.depth_image_view];
            let create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(data.render_pass)
                .attachments(attachments)
                .width(data.swapchain_extent.width)
                .height(data.swapchain_extent.height)
                .layers(1);

            device.create_framebuffer(&create_info, None)
        })
        .collect::<anyhow::Result<Vec<_>, _>>()?;

    Ok(())
}

/// Creates a command pool for the graphics queue family.
pub unsafe fn create_command_pool(instance: &Instance, device: &Device, data: &mut VulkanApplicationData) -> anyhow::Result<()> {
    let indices = graphical_core::queue_families::RequiredQueueFamilies::get(instance, data, data.physical_device)?;
    let info = vk::CommandPoolCreateInfo::builder()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(indices.graphics_queue_index);

    data.command_pool = device.create_command_pool(&info, None)?;
    Ok(())
}

/// Allocates one command buffer per framebuffer without recording.
pub unsafe fn allocate_command_buffers(device: &Device, data: &mut VulkanApplicationData) -> anyhow::Result<()> {
    let allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(data.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(data.framebuffers.len() as u32);

    data.command_buffers = device.allocate_command_buffers(&allocate_info)?;
    Ok(())
}

unsafe fn transition_pyramid_undefined_to_general(device: &Device, cmd: vk::CommandBuffer, data: &VulkanApplicationData) {
    let barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::SHADER_READ)
        .image(data.depth_pyramid_image)
        .subresource_range(super::subresource_range(vk::ImageAspectFlags::COLOR, data.depth_pyramid_mip_count));
    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[*barrier],
    );
}

/// Draws a fullscreen triangle with the sky shader before voxel geometry.
pub unsafe fn draw_sky(device: &Device, cmd: vk::CommandBuffer, data: &VulkanApplicationData) {
    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, data.sky_pipeline);
    device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::GRAPHICS,
        data.sky_pipeline_layout,
        0,
        &[data.descriptor_set],
        &[],
    );
    let screen_size: [f32; 2] = [data.swapchain_extent.width as f32, data.swapchain_extent.height as f32];
    let push_bytes: &[u8] = std::slice::from_raw_parts(screen_size.as_ptr() as *const u8, std::mem::size_of::<[f32; 2]>());
    device.cmd_push_constants(cmd, data.sky_pipeline_layout, vk::ShaderStageFlags::FRAGMENT, 0, push_bytes);
    device.cmd_draw(cmd, 3, 1, 0, 0);
}

pub unsafe fn begin_render_pass(device: &Device, cmd: vk::CommandBuffer, data: &VulkanApplicationData, framebuffer_index: usize) {
    // Reverse-Z: clear depth to 0.0 (= far plane in reverse-Z NDC).
    let clear_values = &[vk::ClearValue::color_f32([0.0, 0.0, 0.0, 1.0]), vk::ClearValue::depth_stencil(0.0, 0)];
    let info = vk::RenderPassBeginInfo::builder()
        .render_pass(data.render_pass)
        .framebuffer(data.framebuffers[framebuffer_index])
        .render_area(vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: data.swapchain_extent,
        })
        .clear_values(clear_values);
    device.cmd_begin_render_pass(cmd, &info, vk::SubpassContents::INLINE);
}

unsafe fn begin_render_pass_no_clear(device: &Device, cmd: vk::CommandBuffer, data: &VulkanApplicationData, framebuffer_index: usize) {
    let info = vk::RenderPassBeginInfo::builder()
        .render_pass(data.render_pass_load)
        .framebuffer(data.framebuffers[framebuffer_index])
        .render_area(vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: data.swapchain_extent,
        });
    device.cmd_begin_render_pass(cmd, &info, vk::SubpassContents::INLINE);
}

/// Generates the depth pyramid from the depth buffer after the render pass.
/// Transitions the depth buffer to shader-read, then dispatches one reduction per mip level.
unsafe fn record_depth_pyramid_generation(device: &Device, cmd: vk::CommandBuffer, data: &VulkanApplicationData, pyramid: &DepthPyramidResources) {
    let mip_count = data.depth_pyramid_mip_count;
    let extent = data.swapchain_extent;

    // Transition depth buffer: DEPTH_ATTACHMENT → SHADER_READ_ONLY
    let depth_barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        .new_layout(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)
        .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ)
        .image(data.depth_image)
        .subresource_range(super::subresource_range(vk::ImageAspectFlags::DEPTH, 1));
    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[*depth_barrier],
    );

    // Pyramid is already in GENERAL layout; ensure prior reads complete before writes
    let pyramid_barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(vk::ImageLayout::GENERAL)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_access_mask(vk::AccessFlags::SHADER_READ)
        .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
        .image(data.depth_pyramid_image)
        .subresource_range(super::subresource_range(vk::ImageAspectFlags::COLOR, mip_count));
    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[*pyramid_barrier],
    );

    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pyramid.pipeline);

    for mip in 0..mip_count {
        let dst_width = (extent.width >> mip).max(1);
        let dst_height = (extent.height >> mip).max(1);

        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            pyramid.pipeline_layout,
            0,
            &[pyramid.descriptor_sets[mip as usize]],
            &[],
        );

        let push = DepthReducePush {
            dst_size: [dst_width, dst_height],
            is_copy: if mip == 0 { 1 } else { 0 },
            _pad: 0,
        };
        let push_bytes: &[u8] = std::slice::from_raw_parts(&push as *const DepthReducePush as *const u8, std::mem::size_of::<DepthReducePush>());
        device.cmd_push_constants(cmd, pyramid.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, push_bytes);

        let wg_x = dst_width.div_ceil(16);
        let wg_y = dst_height.div_ceil(16);
        device.cmd_dispatch(cmd, wg_x, wg_y, 1);

        // Barrier between mip passes: previous write must complete before next read
        if mip + 1 < mip_count {
            let mip_barrier = vk::ImageMemoryBarrier::builder()
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .image(data.depth_pyramid_image)
                .subresource_range(super::subresource_range_mip(vk::ImageAspectFlags::COLOR, mip, 1));
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[] as &[vk::MemoryBarrier],
                &[] as &[vk::BufferMemoryBarrier],
                &[*mip_barrier],
            );
        }
    }

    // Transition depth buffer back to DEPTH_ATTACHMENT for next frame
    let depth_restore = vk::ImageMemoryBarrier::builder()
        .old_layout(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)
        .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        .src_access_mask(vk::AccessFlags::SHADER_READ)
        .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
        .image(data.depth_image)
        .subresource_range(super::subresource_range(vk::ImageAspectFlags::DEPTH, 1));
    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[*depth_restore],
    );
}

/// Records the two-phase vertex-pulling rendering pipeline:
/// 1. Phase 1: previously visible chunks (frustum cull only)
/// 2. Build depth pyramid
/// 3. Phase 2: previously invisible chunks (frustum + Hi-Z occlusion)
pub unsafe fn record_mesh_shader_command_buffer(
    device: &Device,
    data: &VulkanApplicationData,
    image_index: usize,
    face_gen: &FaceGenPipeline,
    vertex_pull: &VertexPullPipeline,
    cull_compact: &CullCompactPipeline,
    voxel_pool: &VoxelPool,
    depth_pyramid: &DepthPyramidResources,
    cull_push: &CullPushConstants,
    pyramid_needs_init: bool,
    ui: &crate::graphical_core::ui_pipeline::UiPipeline,
    timing_query_pool: vk::QueryPool,
) -> anyhow::Result<()> {
    let cmd = data.command_buffers[image_index];
    device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
    device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::builder())?;

    // Reset timing queries for this frame.
    device.cmd_reset_query_pool(cmd, timing_query_pool, 0, crate::graphical_core::vulkan_object::TIMING_QUERY_COUNT);
    device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, timing_query_pool, 0);

    if pyramid_needs_init {
        transition_pyramid_undefined_to_general(device, cmd, data);
    }

    // === Phase 1 cull compact (outside any render pass) ===
    record_cull_compact_pass(device, cmd, cull_compact, voxel_pool, cull_push, 1);

    // === Phase 1 face gen: decide visible faces, reserve draw slots ===
    record_face_gen_pass(device, cmd, face_gen, voxel_pool, cull_push, 1);

    // === Phase 1 vertex-pull draw: previously visible chunks (no occlusion test) ===
    begin_render_pass(device, cmd, data, image_index);
    draw_sky(device, cmd, data);
    device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, timing_query_pool, 1);

    bind_vertex_pull_pipeline_and_draw_indirect(device, cmd, vertex_pull, voxel_pool, 1);

    device.cmd_end_render_pass(cmd);
    device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, timing_query_pool, 2);

    // === Build depth pyramid from phase 1 depth ===
    record_depth_pyramid_generation(device, cmd, data, depth_pyramid);
    device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, timing_query_pool, 3);

    // === Phase 2 cull compact ===
    record_cull_compact_pass(device, cmd, cull_compact, voxel_pool, cull_push, 2);

    // === Phase 2 face gen ===
    record_face_gen_pass(device, cmd, face_gen, voxel_pool, cull_push, 2);

    // === Phase 2 vertex-pull draw: previously invisible chunks (with occlusion test) ===
    begin_render_pass_no_clear(device, cmd, data, image_index);

    bind_vertex_pull_pipeline_and_draw_indirect(device, cmd, vertex_pull, voxel_pool, 2);

    device.cmd_end_render_pass(cmd);
    device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, timing_query_pool, 4);
    device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, timing_query_pool, 5);

    // UI overlay — drawn last so it's on top of everything
    let screen = [data.swapchain_extent.width as f32, data.swapchain_extent.height as f32];
    begin_render_pass_no_clear(device, cmd, data, image_index);
    ui.record(device, cmd, screen);
    device.cmd_end_render_pass(cmd);
    device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, timing_query_pool, 6);

    device.end_command_buffer(cmd)?;
    Ok(())
}

/// Reset the indirect args x-field, run `chunk_cull_compact.comp` for one
/// phase, and barrier the writes against both the subsequent compute read
/// (visible_chunks SSBO consumed by the task shader) and the indirect dispatch
/// fetch (DRAW_INDIRECT_BIT — the validation trap if omitted).
pub(crate) unsafe fn record_cull_compact_pass(
    device: &Device,
    cmd: vk::CommandBuffer,
    cull_compact: &CullCompactPipeline,
    voxel_pool: &VoxelPool,
    cull_push: &CullPushConstants,
    phase: u32,
) {
    let phase_idx = (phase - 1) as usize;
    let args_buf = voxel_pool.indirect_args_buffer[phase_idx];
    let visible_buf = voxel_pool.visible_chunks_buffer[phase_idx];

    // Clear groupCountX (offset 0, 4 bytes). Y/Z stay at 1 from init.
    device.cmd_fill_buffer(cmd, args_buf, 0, 4, 0);
    let fill_barrier = *vk::BufferMemoryBarrier::builder()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
        .buffer(args_buf)
        .size(vk::WHOLE_SIZE);
    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[fill_barrier],
        &[] as &[vk::ImageMemoryBarrier],
    );

    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, cull_compact.pipeline);
    device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::COMPUTE,
        cull_compact.pipeline_layout,
        0,
        &[cull_compact.descriptor_sets[phase_idx]],
        &[],
    );
    let mut push = *cull_push;
    push.phase = phase;
    let push_bytes: &[u8] = std::slice::from_raw_parts(&push as *const CullPushConstants as *const u8, std::mem::size_of::<CullPushConstants>());
    device.cmd_push_constants(cmd, cull_compact.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, push_bytes);
    let workgroups = cull_push.chunk_count.div_ceil(64);
    device.cmd_dispatch(cmd, workgroups, 1, 1);

    // Barrier: cull writes → indirect args fetch + task shader visible-list read.
    let args_after = *vk::BufferMemoryBarrier::builder()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::INDIRECT_COMMAND_READ)
        .buffer(args_buf)
        .size(vk::WHOLE_SIZE);
    let visible_after = *vk::BufferMemoryBarrier::builder()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ)
        .buffer(visible_buf)
        .size(vk::WHOLE_SIZE);
    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        // ALL_GRAPHICS covers the task-shader stage (vulkan-rust 0.10 doesn't
        // expose TASK_SHADER_EXT as a constant). DRAW_INDIRECT covers the
        // indirect args fetch — the "#1 validation trap" if omitted.
        vk::PipelineStageFlags::DRAW_INDIRECT | vk::PipelineStageFlags::ALL_GRAPHICS,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[args_after, visible_after],
        &[] as &[vk::ImageMemoryBarrier],
    );
}

/// Resets `draw_args_buffer[phase]`'s vertexCount, dispatches `face_gen.comp`
/// indirectly off the same args `chunk_cull_compact.comp` already produced
/// for this phase, and barriers its writes against the subsequent indirect
/// draw (draw_args) and vertex shader read (faces). See
/// vertex_pulling_guide.md Step 5.
pub(crate) unsafe fn record_face_gen_pass(
    device: &Device,
    cmd: vk::CommandBuffer,
    face_gen: &FaceGenPipeline,
    voxel_pool: &VoxelPool,
    cull_push: &CullPushConstants,
    phase: u32,
) {
    let phase_idx = (phase - 1) as usize;
    let dispatch_args_buf = voxel_pool.indirect_args_buffer[phase_idx];
    let draw_args_buf = voxel_pool.draw_args_buffer[phase_idx];
    let faces_buf = voxel_pool.faces_buffer[phase_idx];

    // Clear vertexCount only (offset 0, 4 bytes); instanceCount/firstVertex/
    // firstInstance stay at their init values.
    device.cmd_fill_buffer(cmd, draw_args_buf, 0, 4, 0);
    let fill_barrier = *vk::BufferMemoryBarrier::builder()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
        .buffer(draw_args_buf)
        .size(vk::WHOLE_SIZE);
    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[fill_barrier],
        &[] as &[vk::ImageMemoryBarrier],
    );

    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, face_gen.pipeline);
    device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::COMPUTE,
        face_gen.pipeline_layout,
        0,
        &[face_gen.descriptor_sets[phase_idx]],
        &[],
    );
    let mut push = *cull_push;
    push.phase = phase;
    let push_bytes: &[u8] = std::slice::from_raw_parts(&push as *const CullPushConstants as *const u8, std::mem::size_of::<CullPushConstants>());
    device.cmd_push_constants(cmd, face_gen.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, push_bytes);

    // Same indirect args `chunk_cull_compact.comp` already produced for this
    // phase this frame — VkDispatchIndirectCommand is byte-identical to the
    // VkDrawMeshTasksIndirectCommandEXT layout stored there, and
    // record_cull_compact_pass's trailing barrier already covers the
    // DRAW_INDIRECT-stage read this dispatch performs.
    device.cmd_dispatch_indirect(cmd, dispatch_args_buf, 0);

    // Barrier: face_gen writes -> draw_args consumed as indirect draw args,
    // faces consumed by the vertex shader.
    let draw_args_after = *vk::BufferMemoryBarrier::builder()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::INDIRECT_COMMAND_READ)
        .buffer(draw_args_buf)
        .size(vk::WHOLE_SIZE);
    let faces_after = *vk::BufferMemoryBarrier::builder()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ)
        .buffer(faces_buf)
        .size(vk::WHOLE_SIZE);
    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::DRAW_INDIRECT | vk::PipelineStageFlags::VERTEX_SHADER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[draw_args_after, faces_after],
        &[] as &[vk::ImageMemoryBarrier],
    );
}

pub(crate) unsafe fn bind_vertex_pull_pipeline_and_draw_indirect(
    device: &Device,
    cmd: vk::CommandBuffer,
    vertex_pull: &VertexPullPipeline,
    voxel_pool: &VoxelPool,
    phase: u32,
) {
    let phase_idx = (phase - 1) as usize;
    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, vertex_pull.pipeline);
    device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::GRAPHICS,
        vertex_pull.pipeline_layout,
        0,
        &[vertex_pull.descriptor_sets[phase_idx]],
        &[],
    );
    let draw_args_buf = voxel_pool.draw_args_buffer[phase_idx];
    device.cmd_draw_indirect(cmd, draw_args_buf, 0, 1, 16);
}


/// Creates semaphores and fences for each frame in flight.
pub unsafe fn create_sync_objects(device: &Device, data: &mut VulkanApplicationData) -> anyhow::Result<()> {
    let semaphore_info = vk::SemaphoreCreateInfo::builder();
    let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

    for _ in 0..MAX_FRAMES_IN_FLIGHT {
        data.image_available_semaphores.push(device.create_semaphore(&semaphore_info, None)?);
        data.in_flight_fences.push(device.create_fence(&fence_info, None)?);
    }
    for _ in 0..data.swapchain_images.len() {
        data.render_finished_semaphores.push(device.create_semaphore(&semaphore_info, None)?);
    }
    data.images_in_flight = data.swapchain_images.iter().map(|_| vk::Fence::null()).collect();
    Ok(())
}
