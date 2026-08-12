use crate::graphical_core::{compute_cull::CullPushConstants, shaders::create_shader_module, voxel_pool::VoxelPool};
use vk::Handle;
use vulkan_rust::{vk, Device};

pub struct FaceGenPipeline {
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_sets: [vk::DescriptorSet; 2],
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
}

impl FaceGenPipeline {
    pub unsafe fn create(device: &Device, voxel_pool: &VoxelPool) -> anyhow::Result<Self> {
        let descriptor_set_layout = create_layout(device)?;
        let descriptor_pool = create_pool(device)?;
        let descriptor_sets = allocate_sets(device, descriptor_pool, descriptor_set_layout)?;
        write_descriptors(device, descriptor_sets, voxel_pool);

        let push_range = *vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<CullPushConstants>() as u32);
        let set_layouts = [descriptor_set_layout];
        let push_ranges = [push_range];
        let layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_ranges);
        let pipeline_layout = device.create_pipeline_layout(&layout_info, None)?;

        let module = create_shader_module(device, include_bytes!("../shaders/face_gen.comp.spv"))?;
        let stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main");
        let pipeline_info = vk::ComputePipelineCreateInfo::builder().stage(*stage).layout(pipeline_layout);
        let pipeline = device.create_compute_pipeline(vk::PipelineCache::null(), &pipeline_info, None)?;
        device.destroy_shader_module(module, None);

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

unsafe fn create_layout(device: &Device) -> anyhow::Result<vk::DescriptorSetLayout> {
    let bindings = [
        // 0: chunk info (read)
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        // 1: voxel data (read)
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        // 2: boundary data (read)
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(2)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        // 3: visible chunks, phase 1 (read)
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(3)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        // 4: visible chunks, phase 2 (read)
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(4)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        // 5: faces out (read/write)
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(5)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        // 6: draw args / vertexCount atomic (read/write)
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(6)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
    ];
    let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
    Ok(device.create_descriptor_set_layout(&info, None)?)
}

unsafe fn create_pool(device: &Device) -> anyhow::Result<vk::DescriptorPool> {
    // 2 sets x 7 storage buffer bindings each.
    let sizes = [*vk::DescriptorPoolSize::builder()
        .descriptor_count(14)
        .r#type(vk::DescriptorType::STORAGE_BUFFER)];
    let info = vk::DescriptorPoolCreateInfo::builder().max_sets(2).pool_sizes(&sizes);
    Ok(device.create_descriptor_pool(&info, None)?)
}

unsafe fn allocate_sets(device: &Device, pool: vk::DescriptorPool, layout: vk::DescriptorSetLayout) -> anyhow::Result<[vk::DescriptorSet; 2]> {
    let layouts = [layout, layout];
    let info = vk::DescriptorSetAllocateInfo::builder().descriptor_pool(pool).set_layouts(&layouts);
    let sets = device.allocate_descriptor_sets(&info)?;
    Ok([sets[0], sets[1]])
}

unsafe fn write_descriptors(device: &Device, sets: [vk::DescriptorSet; 2], pool: &VoxelPool) {
    for (phase, &set) in sets.iter().enumerate() {
        let chunk_info = [*vk::DescriptorBufferInfo::builder().buffer(pool.chunk_info_buffer).range(vk::WHOLE_SIZE)];
        let voxel = [*vk::DescriptorBufferInfo::builder().buffer(pool.voxel_buffer).range(vk::WHOLE_SIZE)];
        let boundary = [*vk::DescriptorBufferInfo::builder().buffer(pool.boundary_buffer).range(vk::WHOLE_SIZE)];
        let visible1 = [*vk::DescriptorBufferInfo::builder()
            .buffer(pool.visible_chunks_buffer[0])
            .range(vk::WHOLE_SIZE)];
        let visible2 = [*vk::DescriptorBufferInfo::builder()
            .buffer(pool.visible_chunks_buffer[1])
            .range(vk::WHOLE_SIZE)];
        let faces = [*vk::DescriptorBufferInfo::builder().buffer(pool.faces_buffer[phase]).range(vk::WHOLE_SIZE)];
        let draw_args = [*vk::DescriptorBufferInfo::builder()
            .buffer(pool.draw_args_buffer[phase])
            .range(vk::WHOLE_SIZE)];

        let writes = [
            *vk::WriteDescriptorSet::builder()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&chunk_info),
            *vk::WriteDescriptorSet::builder()
                .dst_set(set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&voxel),
            *vk::WriteDescriptorSet::builder()
                .dst_set(set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&boundary),
            *vk::WriteDescriptorSet::builder()
                .dst_set(set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&visible1),
            *vk::WriteDescriptorSet::builder()
                .dst_set(set)
                .dst_binding(4)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&visible2),
            *vk::WriteDescriptorSet::builder()
                .dst_set(set)
                .dst_binding(5)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&faces),
            *vk::WriteDescriptorSet::builder()
                .dst_set(set)
                .dst_binding(6)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&draw_args),
        ];
        device.update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);
    }
}
