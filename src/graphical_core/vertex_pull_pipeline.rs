use crate::graphical_core::{
    camera::UniformBufferObject,
    shaders::create_shader_module,
    voxel_pool::VoxelPool,
    vulkan_object::VulkanApplicationData
};
use vk::Handle;
use vulkan_rust::{vk, Device};

pub struct VertexPullPipeline {
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_sets: [vk::DescriptorSet; 2],
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
}

impl VertexPullPipeline {
    pub unsafe fn create(device: &Device, data: &VulkanApplicationData, voxel_pool: &VoxelPool) -> anyhow::Result<Self> {
        let descriptor_set_layout = create_descriptor_layout(device)?;
        let descriptor_pool = create_descriptor_pool(device)?;
        let descriptor_sets = allocate_descriptor_sets(device, descriptor_pool, descriptor_set_layout)?;

        write_descriptors(device, descriptor_sets, data, voxel_pool);

        let set_layouts = [descriptor_set_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::builder().set_layouts(&set_layouts);
        let pipeline_layout = device.create_pipeline_layout(&layout_info, None)?;
        let pipeline = create_graphics_pipeline(device, data, pipeline_layout)?;

        Ok(Self {
            descriptor_set_layout,
            descriptor_pool,
            descriptor_sets,
            pipeline_layout,
            pipeline,
        })
    }

    pub unsafe fn destroy(&self, device: &Device) {
        device.destroy_pipeline(self.pipeline, None);
        device.destroy_pipeline_layout(self.pipeline_layout, None);
        device.destroy_descriptor_pool(self.descriptor_pool, None);
        device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
    }
}

unsafe fn create_descriptor_layout(device: &Device) -> anyhow::Result<vk::DescriptorSetLayout> {
    let bindings = [
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(3)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::VERTEX),
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(4)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::VERTEX),
    ];
    let create_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
    Ok(device.create_descriptor_set_layout(&create_info, None)?)
}

unsafe fn create_descriptor_pool(device: &Device) -> anyhow::Result<vk::DescriptorPool> {
    let pool_sizes = [
        *vk::DescriptorPoolSize::builder()
            .descriptor_count(2) // texture, x2 sets
            .r#type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
        *vk::DescriptorPoolSize::builder()
            .descriptor_count(2) // CameraUBO, x2 sets
            .r#type(vk::DescriptorType::UNIFORM_BUFFER),
        *vk::DescriptorPoolSize::builder()
            .descriptor_count(4) // faces + chunk_info, x2 sets
            .r#type(vk::DescriptorType::STORAGE_BUFFER),
    ];
    let pool_info = vk::DescriptorPoolCreateInfo::builder().max_sets(2).pool_sizes(&pool_sizes);
    Ok(device.create_descriptor_pool(&pool_info, None)?)
}

unsafe fn allocate_descriptor_sets(
    device: &Device,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
) -> anyhow::Result<[vk::DescriptorSet; 2]> {
    let layouts = [layout, layout];
    let alloc_info = vk::DescriptorSetAllocateInfo::builder().descriptor_pool(pool).set_layouts(&layouts);
    let sets = device.allocate_descriptor_sets(&alloc_info)?;
    Ok([sets[0], sets[1]])
}

unsafe fn write_descriptors(device: &Device, sets: [vk::DescriptorSet; 2], data: &VulkanApplicationData, pool: &VoxelPool) {
    for (phase, &set) in sets.iter().enumerate() {
        let image_info = [*vk::DescriptorImageInfo::builder()
            .image_view(data.texture_image_view)
            .sampler(data.texture_sampler)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];

        let ubo_info = [*vk::DescriptorBufferInfo::builder()
            .buffer(data.uniform_buffer)
            .range(std::mem::size_of::<UniformBufferObject>() as u64)];

        let faces_info = [*vk::DescriptorBufferInfo::builder().buffer(pool.faces_buffer[phase]).range(vk::WHOLE_SIZE)];

        let chunk_info = [*vk::DescriptorBufferInfo::builder().buffer(pool.chunk_info_buffer).range(vk::WHOLE_SIZE)];

        let writes = [
            *vk::WriteDescriptorSet::builder()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&image_info),
            *vk::WriteDescriptorSet::builder()
                .dst_set(set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&ubo_info),
            *vk::WriteDescriptorSet::builder()
                .dst_set(set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&faces_info),
            *vk::WriteDescriptorSet::builder()
                .dst_set(set)
                .dst_binding(4)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&chunk_info),
        ];
        device.update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);
    }
}

unsafe fn create_graphics_pipeline(
    device: &Device,
    data: &VulkanApplicationData,
    pipeline_layout: vk::PipelineLayout,
) -> anyhow::Result<vk::Pipeline> {
    let vert_module = create_shader_module(device, include_bytes!("../shaders/voxel_pull.vert.spv"))?;
    let frag_module = create_shader_module(device, include_bytes!("../shaders/shader.frag.spv"))?;

    let vert_stage = *vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vert_module)
        .name(c"main");
    let frag_stage = *vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::FRAGMENT)
        .module(frag_module)
        .name(c"main");

    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder();
    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder().topology(vk::PrimitiveTopology::TRIANGLE_LIST);

    let viewport = *vk::Viewport::builder()
        .width(data.swapchain_extent.width as f32)
        .height(data.swapchain_extent.height as f32)
        .min_depth(0.0)
        .max_depth(1.0);
    let scissor = *vk::Rect2D::builder().extent(data.swapchain_extent);

    let viewports = [viewport];
    let scissors = [scissor];
    let viewport_state = vk::PipelineViewportStateCreateInfo::builder().viewports(&viewports).scissors(&scissors);

    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::CLOCKWISE);

    let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder().rasterization_samples(vk::SampleCountFlags::_1);

    let blend_attachment = *vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::all())
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD);

    let blend_attachments = [blend_attachment];
    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder().attachments(&blend_attachments);

    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::GREATER_OR_EQUAL);

    let stages = [vert_stage, frag_stage];
    let info = vk::GraphicsPipelineCreateInfo::builder()
        .stages(&stages)
        .vertex_input_state(&vertex_input_state)
        .input_assembly_state(&input_assembly_state)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization_state)
        .multisample_state(&multisample_state)
        .depth_stencil_state(&depth_stencil_state)
        .color_blend_state(&color_blend_state)
        .layout(pipeline_layout)
        .render_pass(data.render_pass)
        .subpass(0);

    let pipeline = device.create_graphics_pipeline(vk::PipelineCache::null(), &info, None)?;

    device.destroy_shader_module(vert_module, None);
    device.destroy_shader_module(frag_module, None);

    Ok(pipeline)
}
