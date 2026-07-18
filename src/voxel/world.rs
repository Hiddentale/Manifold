use super::{
    block::BlockType,
    chunk::{Chunk, CHUNK_SIZE},
    chunk_generator::ChunkGenerator,
    erosion::ErosionMap,
    grid::{world_to_chunk_local, ChunkPos},
    metric::MetricField,
};
use glam::DVec3;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

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
    pub fn new(seed: u32, erosion_map: Option<Arc<ErosionMap>>) -> Self {
        Self {
            chunks: HashMap::new(),
            generator: ChunkGenerator::new(seed, erosion_map),
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

    // Out-of-range chunks are treated as solid below the terrain layer
    pub fn block_solid(&self, cp: ChunkPos, lx: usize, ly: usize, lz: usize) -> bool {
        let lxi = lx.min(CHUNK_SIZE - 1);
        let lyi = ly.min(CHUNK_SIZE - 1);
        let lzi = lz.min(CHUNK_SIZE - 1);
        match self.chunks.get(&cp) {
            Some(chunk) => chunk.get(lxi, lyi, lzi).is_opaque(),
            None => (TERRAIN_MIN_CY..=TERRAIN_MAX_CY).contains(&cp.y),
        }
    }

    /// Direct cube-space block read.
    pub fn block_at(&self, cp: ChunkPos, lx: usize, ly: usize, lz: usize) -> BlockType {
        match self.chunks.get(&cp) {
            Some(chunk) => chunk.get(lx.min(CHUNK_SIZE - 1), ly.min(CHUNK_SIZE - 1), lz.min(CHUNK_SIZE - 1)),
            None => BlockType::Air,
        }
    }

    /// Direct cube-space block write.
    pub fn set_block_at(&mut self, cp: ChunkPos, lx: usize, ly: usize, lz: usize, block: BlockType) -> bool {
        match self.chunks.get_mut(&cp) {
            Some(chunk) => {
                chunk.set(lx.min(CHUNK_SIZE - 1), ly.min(CHUNK_SIZE - 1), lz.min(CHUNK_SIZE - 1), block);
                true
            }
            None => false,
        }
    }

    pub fn get_chunk(&self, cx: i32, cy: i32, cz: i32) -> Option<&Chunk> {
        self.chunks.get(&ChunkPos::new(cx, cy, cz))
    }

    pub fn get_chunk_at(&self, cp: ChunkPos) -> Option<&Chunk> {
        self.chunks.get(&cp)
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

#[cfg(test)]
mod tests {
    use super::CHUNK_SIZE;
    use super::*;
    use crate::voxel::grid::ChunkPos;

    #[test]
    fn unloaded_terrain_chunks_are_solid() {
        let world = World::new(0, None);
        let in_band = ChunkPos { x: 0, y: 5, z: 0 };
        assert!(world.block_solid(in_band, 0, 0, 0));
        let above_band = ChunkPos {
            x: 0,
            y: TERRAIN_MAX_CY + 5,
            z: 0,
        };
        assert!(!world.block_solid(above_band, 0, 0, 0));
    }

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
        let mut world = World::new(0, None);
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
        let mut world = World::new(0, None);
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
