#![allow(dead_code)] // VoxelPool is wired up in Phase 1 step 7
use crate::graphical_core::buffers::allocate_buffer;
use crate::graphical_core::vulkan_object::VulkanApplicationData;
use crate::voxel::chunk::{Chunk, CHUNK_SIZE};
use crate::voxel::grid::ChunkPos;
use crate::voxel::world::World;
use std::collections::HashMap;
use vulkan_rust::{vk, Device, Instance};

const VOXEL_CHUNK_BYTES: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE; // 4096
const BOUNDARY_FACES: usize = 6;
const BOUNDARY_FACE_BYTES: usize = CHUNK_SIZE * CHUNK_SIZE; // 256
const BOUNDARY_CHUNK_BYTES: usize = BOUNDARY_FACES * BOUNDARY_FACE_BYTES; // 1536

/// Bounded face budget for `face_gen.comp`'s vertex-pulling output — see
/// vertex_pulling_guide.md Step 4. Matches the constant of the same name
/// in `face_gen.comp`.
const MAX_FACES: u64 = 4_194_304;
const FACE_RECORD_BYTES: u64 = 8; // uvec2 per face
const DRAW_ARGS_BYTES: u64 = 16; // VkDrawIndirectCommand: 4 x u32

/// GPU-side chunk info for the task shader. Must match GLSL layout (std430).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct GpuMeshChunkInfo {
    pub aabb_min: [f32; 3],
    pub voxel_slot: u32,
    pub aabb_max: [f32; 3],
    pub boundary_slot: u32,
    pub chunk_pos: [i32; 3], // chunk-space (x, y, z)
    /// Vestigial from the cube-sphere renderer; always 0 in the flat-world build.
    pub face_id: u32,
}

/// Manages GPU SSBOs for raw voxel data, boundary slices, and chunk info.
/// Uses slot-based allocation so chunks can be added/removed without rebuilding.
pub struct VoxelPool {
    pub voxel_buffer: vk::Buffer,
    voxel_memory: vk::DeviceMemory,
    voxel_ptr: *mut u8,

    pub boundary_buffer: vk::Buffer,
    boundary_memory: vk::DeviceMemory,
    boundary_ptr: *mut u8,

    pub chunk_info_buffer: vk::Buffer,
    chunk_info_memory: vk::DeviceMemory,
    chunk_info_ptr: *mut GpuMeshChunkInfo,

    pub visibility_buffer: vk::Buffer,
    visibility_memory: vk::DeviceMemory,
    visibility_ptr: *mut u32,

    pub visible_chunks_buffer: [vk::Buffer; 2],
    visible_chunks_memory: [vk::DeviceMemory; 2],

    pub indirect_args_buffer: [vk::Buffer; 2],
    indirect_args_memory: [vk::DeviceMemory; 2],

    pub faces_buffer: [vk::Buffer; 2],
    faces_memory: [vk::DeviceMemory; 2],

    pub draw_args_buffer: [vk::Buffer; 2],
    draw_args_memory: [vk::DeviceMemory; 2],

    free_slots: Vec<u32>,
    next_slot: u32,
    max_slots: u32,
    chunk_slots: HashMap<ChunkPos, u32>,

    chunk_info_count: u32,
    slot_to_info_index: HashMap<u32, u32>,
    info_index_to_slot: Vec<u32>,
}

impl VoxelPool {
    pub unsafe fn new(max_slots: u32, device: &Device, instance: &Instance, data: &mut VulkanApplicationData) -> anyhow::Result<Self> {
        let host_visible = super::host_visible_coherent();
        let ssbo_flags = vk::BufferUsageFlags::STORAGE_BUFFER;
        let indirect_flags = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;

        let voxel_size = (max_slots as usize * VOXEL_CHUNK_BYTES) as u64;
        let (voxel_buffer, voxel_memory, voxel_ptr) = allocate_buffer::<u8>(voxel_size, ssbo_flags, device, instance, data, host_visible)?;

        let boundary_size = (max_slots as usize * BOUNDARY_CHUNK_BYTES) as u64;
        let (boundary_buffer, boundary_memory, boundary_ptr) =
            allocate_buffer::<u8>(boundary_size, ssbo_flags, device, instance, data, host_visible)?;

        let chunk_info_size = (max_slots as usize * std::mem::size_of::<GpuMeshChunkInfo>()) as u64;
        let (chunk_info_buffer, chunk_info_memory, chunk_info_ptr) =
            allocate_buffer::<GpuMeshChunkInfo>(chunk_info_size, ssbo_flags, device, instance, data, host_visible)?;

        let visibility_size = (max_slots as u64) * 4;
        let (visibility_buffer, visibility_memory, visibility_ptr) =
            allocate_buffer::<u32>(visibility_size, ssbo_flags, device, instance, data, host_visible)?;

        // Zero visibility buffer
        std::ptr::write_bytes(visibility_ptr, 0, max_slots as usize);

        // Per-phase visible chunk lists (one u32 per max_slot).
        let visible_size = (max_slots as u64) * 4;
        let (vc0_buf, vc0_mem, _vc0_ptr) = allocate_buffer::<u32>(visible_size, ssbo_flags, device, instance, data, host_visible)?;
        let (vc1_buf, vc1_mem, _vc1_ptr) = allocate_buffer::<u32>(visible_size, ssbo_flags, device, instance, data, host_visible)?;

        // Per-phase indirect args buffers (12 bytes each, layout matches
        // VkDrawMeshTasksIndirectCommandEXT). Pre-init Y/Z to 1; the cull
        // compact pass only touches X, and per-frame cmd_fill_buffer only
        // resets X to 0.
        let args_size: u64 = 12;
        let (a0_buf, a0_mem, a0_ptr) = allocate_buffer::<u32>(args_size, indirect_flags, device, instance, data, host_visible)?;
        let (a1_buf, a1_mem, a1_ptr) = allocate_buffer::<u32>(args_size, indirect_flags, device, instance, data, host_visible)?;
        let init_args = [0u32, 1u32, 1u32];
        std::ptr::copy_nonoverlapping(init_args.as_ptr(), a0_ptr, 3);
        std::ptr::copy_nonoverlapping(init_args.as_ptr(), a1_ptr, 3);

        // Per-phase faces SSBO (MAX_FACES * 8 bytes each).
        let faces_size = MAX_FACES * FACE_RECORD_BYTES;
        let (f0_buf, f0_mem, _f0_ptr) = allocate_buffer::<u8>(faces_size, ssbo_flags, device, instance, data, host_visible)?;
        let (f1_buf, f1_mem, _f1_ptr) = allocate_buffer::<u8>(faces_size, ssbo_flags, device, instance, data, host_visible)?;

        // Per-phase draw args (VkDrawIndirectCommand). Pre-init
        // instanceCount/firstVertex/firstInstance to (1, 0, 0); only
        // vertexCount is touched per frame.
        let (d0_buf, d0_mem, d0_ptr) = allocate_buffer::<u32>(DRAW_ARGS_BYTES, indirect_flags, device, instance, data, host_visible)?;
        let (d1_buf, d1_mem, d1_ptr) = allocate_buffer::<u32>(DRAW_ARGS_BYTES, indirect_flags, device, instance, data, host_visible)?;
        let init_draw_args = [0u32, 1u32, 0u32, 0u32];
        std::ptr::copy_nonoverlapping(init_draw_args.as_ptr(), d0_ptr, 4);
        std::ptr::copy_nonoverlapping(init_draw_args.as_ptr(), d1_ptr, 4);

        Ok(Self {
            voxel_buffer,
            voxel_memory,
            voxel_ptr,
            boundary_buffer,
            boundary_memory,
            boundary_ptr,
            chunk_info_buffer,
            chunk_info_memory,
            chunk_info_ptr,
            visibility_buffer,
            visibility_memory,
            visibility_ptr,
            visible_chunks_buffer: [vc0_buf, vc1_buf],
            visible_chunks_memory: [vc0_mem, vc1_mem],
            indirect_args_buffer: [a0_buf, a1_buf],
            indirect_args_memory: [a0_mem, a1_mem],
            faces_buffer: [f0_buf, f1_buf],
            faces_memory: [f0_mem, f1_mem],
            draw_args_buffer: [d0_buf, d1_buf],
            draw_args_memory: [d0_mem, d1_mem],
            free_slots: Vec::new(),
            next_slot: 0,
            max_slots,
            chunk_slots: HashMap::new(),
            chunk_info_count: 0,
            slot_to_info_index: HashMap::new(),
            info_index_to_slot: Vec::new(),
        })
    }

    /// Uploads a chunk's voxel data and boundary slices to GPU.
    pub unsafe fn upload_chunk(&mut self, pos: ChunkPos, chunk: &Chunk, world: &World) {
        let slot = self.allocate_slot(pos);

        // Write voxel data
        let voxel_offset = slot as usize * VOXEL_CHUNK_BYTES;
        std::ptr::copy_nonoverlapping(chunk.as_bytes().as_ptr(), self.voxel_ptr.add(voxel_offset), VOXEL_CHUNK_BYTES);

        // Write boundary slices
        self.write_boundary(slot, pos, world);

        // Write chunk info.
        let (aabb_min, aabb_max) = crate::voxel::grid::chunk_world_aabb(pos);
        let info = GpuMeshChunkInfo {
            aabb_min,
            voxel_slot: slot,
            aabb_max,
            boundary_slot: slot,
            chunk_pos: [pos.x, pos.y, pos.z],
            face_id: 0,
        };
        let info_index = self.chunk_info_count;
        std::ptr::write(self.chunk_info_ptr.add(info_index as usize), info);
        self.slot_to_info_index.insert(slot, info_index);
        self.info_index_to_slot.push(slot);
        self.chunk_info_count += 1;
    }

    /// Re-uploads voxel data for a chunk that is already in the pool.
    /// Used after in-place block edits. Does not allocate a new slot.
    pub unsafe fn reupload_chunk(&mut self, pos: ChunkPos, chunk: &Chunk, world: &World) {
        let slot = match self.chunk_slots.get(&pos) {
            Some(&s) => s,
            None => return,
        };
        let voxel_offset = slot as usize * VOXEL_CHUNK_BYTES;
        std::ptr::copy_nonoverlapping(chunk.as_bytes().as_ptr(), self.voxel_ptr.add(voxel_offset), VOXEL_CHUNK_BYTES);
        self.write_boundary(slot, pos, world);
    }

    /// Removes a chunk from the pool, returning its slot for reuse.
    pub unsafe fn remove_chunk(&mut self, pos: &ChunkPos) {
        let slot = match self.chunk_slots.remove(pos) {
            Some(s) => s,
            None => return,
        };
        self.free_slots.push(slot);

        // Swap-remove from chunk info array
        if let Some(&info_index) = self.slot_to_info_index.get(&slot) {
            let last_index = self.chunk_info_count - 1;
            if info_index != last_index {
                // Copy last entry into the removed slot
                let last_info = std::ptr::read(self.chunk_info_ptr.add(last_index as usize));
                std::ptr::write(self.chunk_info_ptr.add(info_index as usize), last_info);

                // Update tracking for the moved entry
                let moved_slot = self.info_index_to_slot[last_index as usize];
                self.slot_to_info_index.insert(moved_slot, info_index);
                self.info_index_to_slot[info_index as usize] = moved_slot;
            }
            self.slot_to_info_index.remove(&slot);
            self.info_index_to_slot.pop();
            self.chunk_info_count -= 1;

            // Reset visibility for the swapped index
            std::ptr::write(self.visibility_ptr.add(info_index as usize), 0);
        }
    }

    /// Updates boundary data for a chunk's neighbors (call when a chunk is loaded/unloaded).
    pub unsafe fn invalidate_neighbor_boundaries(&mut self, pos: ChunkPos, world: &World) {
        let neighbors = [
            pos.offset(1, 0, 0),
            pos.offset(-1, 0, 0),
            pos.offset(0, 1, 0),
            pos.offset(0, -1, 0),
            pos.offset(0, 0, 1),
            pos.offset(0, 0, -1),
        ];
        for neighbor_pos in neighbors {
            if let Some(&slot) = self.chunk_slots.get(&neighbor_pos) {
                self.write_boundary(slot, neighbor_pos, world);
            }
        }
    }

    pub fn chunk_count(&self) -> u32 {
        self.chunk_info_count
    }

    pub fn has_chunk(&self, pos: &ChunkPos) -> bool {
        self.chunk_slots.contains_key(pos)
    }

    pub fn chunk_positions(&self) -> Vec<ChunkPos> {
        self.chunk_slots.keys().copied().collect()
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        device.unmap_memory(self.voxel_memory);
        device.destroy_buffer(self.voxel_buffer, None);
        device.free_memory(self.voxel_memory, None);

        device.unmap_memory(self.boundary_memory);
        device.destroy_buffer(self.boundary_buffer, None);
        device.free_memory(self.boundary_memory, None);

        device.unmap_memory(self.chunk_info_memory);
        device.destroy_buffer(self.chunk_info_buffer, None);
        device.free_memory(self.chunk_info_memory, None);

        device.unmap_memory(self.visibility_memory);
        device.destroy_buffer(self.visibility_buffer, None);
        device.free_memory(self.visibility_memory, None);

        for i in 0..2 {
            device.unmap_memory(self.visible_chunks_memory[i]);
            device.destroy_buffer(self.visible_chunks_buffer[i], None);
            device.free_memory(self.visible_chunks_memory[i], None);

            device.unmap_memory(self.indirect_args_memory[i]);
            device.destroy_buffer(self.indirect_args_buffer[i], None);
            device.free_memory(self.indirect_args_memory[i], None);

            device.unmap_memory(self.faces_memory[i]);
            device.destroy_buffer(self.faces_buffer[i], None);
            device.free_memory(self.faces_memory[i], None);

            device.unmap_memory(self.draw_args_memory[i]);
            device.destroy_buffer(self.draw_args_buffer[i], None);
            device.free_memory(self.draw_args_memory[i], None);
        }
    }

    fn allocate_slot(&mut self, pos: ChunkPos) -> u32 {
        let slot = self.free_slots.pop().unwrap_or_else(|| {
            let s = self.next_slot;
            self.next_slot += 1;
            assert!(s < self.max_slots, "VoxelPool: exceeded max slot count");
            s
        });
        self.chunk_slots.insert(pos, slot);
        slot
    }

    unsafe fn write_boundary(&self, slot: u32, pos: ChunkPos, world: &World) {
        let base = slot as usize * BOUNDARY_CHUNK_BYTES;

        self.write_boundary_face(base, 0, world.get_chunk_at(pos.offset(1, 0, 0)), |c, u, v| c.get_block_at(0, v, u));
        self.write_boundary_face(base, 1, world.get_chunk_at(pos.offset(-1, 0, 0)), |c, u, v| {
            c.get_block_at(CHUNK_SIZE - 1, v, u)
        });
        self.write_boundary_face(base, 2, world.get_chunk_at(pos.offset(0, 1, 0)), |c, u, v| c.get_block_at(u, 0, v));
        self.write_boundary_face(base, 3, world.get_chunk_at(pos.offset(0, -1, 0)), |c, u, v| {
            c.get_block_at(u, CHUNK_SIZE - 1, v)
        });
        self.write_boundary_face(base, 4, world.get_chunk_at(pos.offset(0, 0, 1)), |c, u, v| c.get_block_at(u, v, 0));
        self.write_boundary_face(base, 5, world.get_chunk_at(pos.offset(0, 0, -1)), |c, u, v| {
            c.get_block_at(u, v, CHUNK_SIZE - 1)
        });
    }

    unsafe fn write_boundary_face(
        &self,
        base_offset: usize,
        face: usize,
        neighbor: Option<&Chunk>,
        read_block: impl Fn(&Chunk, usize, usize) -> crate::voxel::block::BlockType,
    ) {
        let offset = base_offset + face * BOUNDARY_FACE_BYTES;
        match neighbor {
            Some(chunk) => {
                for v in 0..CHUNK_SIZE {
                    for u in 0..CHUNK_SIZE {
                        let block = read_block(chunk, u, v);
                        *self.boundary_ptr.add(offset + u + v * CHUNK_SIZE) = block as u8;
                    }
                }
            }
            None => {
                // No neighbor loaded — fill with Air (0) so boundary faces are emitted
                std::ptr::write_bytes(self.boundary_ptr.add(offset), 0, BOUNDARY_FACE_BYTES);
            }
        }
    }
}
