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
        for chunk_pos in chunks {
            if !chunk_in_render_area(render_area, chunk_pos) {
                self.chunks.remove(&chunk_pos);
                unloaded_chunks.push(chunk_pos);
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

fn chunk_in_render_area(render_area: &HashSet<(i32, i32)>, chunk_pos: ChunkPos) -> bool {
    render_area.contains(&(chunk_pos.x, chunk_pos.z))
}

#[cfg(test)]
mod tests {
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

    #[test]
    fn calculate_render_area_returns_expected_number_of_colums() {}

    #[test]
    fn null_render_distance_returns_active_chunk() {}

    #[test]
    fn negative_render_distance_returns_empty() {}

    #[test]
    fn illegal_chunk_is_rejected() {}

    #[test]
    fn legal_chunk_is_accepted() {}

    #[test]
    fn fully_loaded_column_is_not_rerequested() {}

    #[test]
    fn mid_generation_column_is_not_rerequested() {}

    #[test]
    fn chunk_finished_after_player_moved_away_is_not_inserted() {}

    #[test]
    fn previously_loaded_column_outside_render_distance_is_unloaded() {}
}
