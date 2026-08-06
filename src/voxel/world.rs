use super::{
    block::BlockType,
    chunk::Chunk,
    chunk_generator::ChunkGenerator,
    grid::{world_to_chunk_local, ChunkPos},
    metric::MetricField,
};
use glam::DVec3;
use std::collections::{HashMap, HashSet};

pub const TERRAIN_MIN_CY: i32 = 0;
pub const TERRAIN_MAX_CY: i32 = 47;

pub struct World {
    chunks: HashMap<ChunkPos, Chunk>,
    generator: ChunkGenerator,
    pub metric: MetricField,
}

pub struct ChunkChanges {
    pub loaded_chunks: Vec<ChunkPos>,
    pub unloaded_chunks: Vec<ChunkPos>,
}

impl World {
    pub fn new(seed: u32) -> Self {
        Self {
            chunks: HashMap::new(),
            generator: ChunkGenerator::new(seed),
            metric: MetricField::new(),
        }
    }

    pub fn update(&mut self, player_position: DVec3, render_distance: i32) -> ChunkChanges {
        let mut loaded_chunks: Vec<ChunkPos> = Vec::new();
        let mut unloaded_chunks: Vec<ChunkPos> = Vec::new();
        let (active_chunk, _, _, _) = world_to_chunk_local(player_position);
        let render_area: HashSet<(i32, i32)> = calculate_render_area(active_chunk, render_distance);
        self.evict_chunks_outside_render_area(&render_area, &mut unloaded_chunks);
        self.request_pending_chunks(&render_area);
        self.render_pending_chunks(&render_area, &mut loaded_chunks);

        ChunkChanges {
            loaded_chunks,
            unloaded_chunks,
        }
    }

    fn evict_chunks_outside_render_area(&mut self, render_area: &HashSet<(i32, i32)>, unloaded_chunks: &mut Vec<ChunkPos>) {
        let chunks: Vec<ChunkPos> = self.chunks.keys().copied().collect();
        for chunk in chunks {
            if !chunk_in_render_area(render_area, chunk) {
                self.chunks.remove(&chunk);
                unloaded_chunks.push(chunk);
            }
        }
    }

    fn request_pending_chunks(&mut self, render_area: &HashSet<(i32, i32)>) {
        for &(chunk_x, chunk_z) in render_area {
            let any_loaded = (TERRAIN_MIN_CY..=TERRAIN_MAX_CY).any(|cy| self.chunks.contains_key(&ChunkPos::new(chunk_x, cy, chunk_z)));
            if !any_loaded && !self.generator.is_pending(chunk_x, chunk_z) {
                self.generator.request(chunk_x, chunk_z);
            }
        }
    }

    fn render_pending_chunks(&mut self, render_area: &HashSet<(i32, i32)>, loaded_chunks: &mut Vec<ChunkPos>) {
        for chunk_column in self.generator.receive() {
            for (i, chunk) in chunk_column.chunks.into_iter().enumerate() {
                let chunk_y = TERRAIN_MIN_CY + i as i32;
                let chunk_pos = ChunkPos::new(chunk_column.chunk_x, chunk_y, chunk_column.chunk_z);
                if !chunk_in_render_area(render_area, chunk_pos) {
                    continue;
                }
                loaded_chunks.push(chunk_pos);
                self.chunks.insert(chunk_pos, chunk);
            }
        }
    }

    pub fn block_solid_at(&self, world_x: i32, world_y: i32, world_z: i32) -> bool {
        let (chunk_pos, local_x, local_y, local_z) = 
            convert_world_coordinates_to_chunk_coordinates(world_x, world_y, world_z);
        match self.chunks.get(&chunk_pos) {
            Some(chunk) => chunk.get(local_x, local_y, local_z).is_opaque(),
            None => false,
        }
    }

    /// Direct cube-space block read.
    pub fn get_block_at(&self, world_x: i32, world_y: i32, world_z: i32) -> BlockType {
        let (chunk_pos, local_x, local_y, local_z) = 
            convert_world_coordinates_to_chunk_coordinates(world_x, world_y, world_z);
        match self.chunks.get(&chunk_pos) {
            Some(chunk) => chunk.get(local_x, local_y, local_z),
            None => BlockType::Air,
        }
    }

    /// Direct cube-space block write.
    pub fn set_block_at(&mut self, world_x: i32, world_y: i32, world_z: i32, block: BlockType) -> bool {
        let (chunk_pos, local_x, local_y, local_z) = 
            convert_world_coordinates_to_chunk_coordinates(world_x, world_y, world_z);
        match self.chunks.get_mut(&chunk_pos) {
            Some(chunk) => { 
                chunk.set(local_x, local_y, local_z, block);
                true
            }
            None => false,
        }
    }

    pub fn get_chunk(&self, chunk_x: i32, chunk_y: i32, chunk_z: i32) -> Option<&Chunk> {
        self.chunks.get(&ChunkPos::new(chunk_x, chunk_y, chunk_z))
    }

    pub fn get_chunk_at(&self, chunk_pos: ChunkPos) -> Option<&Chunk> {
        self.chunks.get(&chunk_pos)
    }

    pub fn chunk_positions(&self) -> impl Iterator<Item = ChunkPos> + '_ {
        self.chunks.keys().copied()
    }
}

fn calculate_render_area(active_chunk: ChunkPos, render_distance: i32) -> HashSet<(i32, i32)> {
    let mut target_columns: HashSet<(i32, i32)> = HashSet::new();
    if render_distance >= 0 {
        for dz in -render_distance..=render_distance {
            for dx in -render_distance..=render_distance {
                let chunk_x = active_chunk.x + dx;
                let chunk_z = active_chunk.z + dz;
                target_columns.insert((chunk_x, chunk_z));
            }
        }
    }
    target_columns
}

fn chunk_in_render_area(render_area: &HashSet<(i32, i32)>, chunk: ChunkPos) -> bool {
    render_area.contains(&(chunk.x, chunk.z))
}

fn convert_world_coordinates_to_chunk_coordinates(world_x: i32, world_y: i32, world_z: i32) -> (ChunkPos, usize, usize, usize){
    let (chunk_pos, local_x, local_y, local_z) = world_to_chunk_local(
           DVec3::new(world_x as f64, world_y as f64, world_z as f64)
       );
    return (chunk_pos, local_x, local_y, local_z)
}

#[cfg(test)]
mod tests {
    use super::super::chunk::CHUNK_SIZE;
    use super::*;
    use crate::voxel::grid::ChunkPos;

    // Total number of columns should equal (2r+1)^2, where r is the render distance.
    #[test]
    fn render_distance_one_returns_nine_columns() {
        let active_chunk = ChunkPos { x: 0, y: 0, z: 0 };
        let render_area = calculate_render_area(active_chunk, 1);
        assert_eq!(render_area.len(), 9); // 3x3
    }

    #[test]
    fn render_distance_three_returns_forty_nine_columns() {
        let active_chunk = ChunkPos { x: 0, y: 0, z: 0 };
        let render_area = calculate_render_area(active_chunk, 3);
        assert_eq!(render_area.len(), 49); // 7x7
    }

    #[test]
    fn zero_render_distance_returns_active_chunk() {
        let active_chunk = ChunkPos { x: 5, y: 0, z: -3 };
        let render_distance = 0;
        let render_area = calculate_render_area(active_chunk, render_distance);
        assert_eq!(render_area, HashSet::from([(5, -3)]));
    }

    #[test]
    fn negative_render_distance_returns_empty() {
        let active_chunk = ChunkPos { x: 5, y: 0, z: -3 };
        let render_distance = -1;
        let render_area = calculate_render_area(active_chunk, render_distance);
        assert!(render_area.is_empty());
    }

    #[test]
    fn chunk_outside_render_area_is_rejected() {
        let active_chunk = ChunkPos { x: 0, y: 0, z: 0 };
        let render_distance = 2;
        let render_area = calculate_render_area(active_chunk, render_distance);
        let chunk = ChunkPos { x: 3, y: 0, z: 5 };
        assert!(!chunk_in_render_area(&render_area, chunk));
    }

    #[test]
    fn chunk_inside_render_area_is_accepted() {
        let active_chunk = ChunkPos { x: 0, y: 0, z: 0 };
        let render_distance = 2;
        let render_area = calculate_render_area(active_chunk, render_distance);
        let chunk = ChunkPos { x: 1, y: 0, z: 1 };
        assert!(chunk_in_render_area(&render_area, chunk));
    }

    #[test]
    fn fully_loaded_column_is_not_rerequested() {
        let mut world = World::new(0);
        for cy in TERRAIN_MIN_CY..=TERRAIN_MAX_CY {
            world.chunks.insert(ChunkPos { x: 0, y: cy, z: 0 }, Chunk::new(BlockType::Air));
        }
        let render_distance = 2;
        let _ = world.update(DVec3::new(0.0, 0.0, 0.0), render_distance);
        assert!(!world.generator.is_pending(0, 0));
    }

    #[test]
    fn mid_generation_column_is_not_rerequested() {}

    #[test]
    fn chunk_finished_after_player_moved_away_is_not_inserted() {}

    #[test]
    fn previously_loaded_column_outside_render_distance_is_unloaded() {
        let mut world = World::new(0);
        let column = ChunkPos { x: 0, y: 0, z: 0 };
        for cy in TERRAIN_MIN_CY..=TERRAIN_MAX_CY {
            world.chunks.insert(ChunkPos { x: 0, y: cy, z: 0 }, Chunk::new(BlockType::Air));
        }
        let render_distance = 2;
        let far_chunk_x = render_distance + 5;
        let chunk_changes = world.update(DVec3::new((far_chunk_x * CHUNK_SIZE as i32) as f64, 0.0, 0.0), render_distance);
        assert!(chunk_changes.unloaded_chunks.contains(&column));
    }
}
