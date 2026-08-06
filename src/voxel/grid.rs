//! Flat world coordinate space.

#![allow(dead_code)]

use super::chunk::CHUNK_SIZE;
use glam::{DVec3, Vec3};

pub const SEA_LEVEL_BLOCKS: i32 = 64;
pub const WORLD_HEIGHT_CHUNKS: i32 = 96;
const ACCEPTABLE_MACHINE_ERROR: f32 = 1e-4;
const DIAGONAL_FACTOR: f32 = 3.0_f32;

/// The position of a Chunk in the Z^3 lattice, coordinates
/// are the corner of the Chunk where all three coordinates are simultaneously smallest.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct ChunkPos {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl ChunkPos {
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    pub const fn get_coords(self) -> [i32; 3] {
        [self.x, self.y, self.z]
    }

    /// Chunk offset by an integer step on each axis.
    pub const fn offset(self, dx: i32, dy: i32, dz: i32) -> Self {
        Self {
            x: self.x + dx,
            y: self.y + dy,
            z: self.z + dz,
        }
    }
}

/// The chunk that owns the given world-space coordinate.
pub fn block_to_chunk(world_x: i32, world_y: i32, world_z: i32) -> ChunkPos {
    let size = CHUNK_SIZE as i32;
    ChunkPos::new(world_x.div_euclid(size), world_y.div_euclid(size), world_z.div_euclid(size))
}

/// Cartesian world position of a chunk.
pub fn chunk_to_world(chunk_position: ChunkPos, local: Vec3) -> DVec3 {
    let chunk_size = CHUNK_SIZE as f64;
    DVec3::new(
        chunk_position.x as f64 * chunk_size + local.x as f64,
        chunk_position.y as f64 * chunk_size + local.y as f64,
        chunk_position.z as f64 * chunk_size + local.z as f64,
    )
}

/// The chunk that owns the given world-space coordinate and the specific coordinate in that chunk in
/// local chunk coordinates.
pub fn world_to_chunk_local(world_coordinates: DVec3) -> (ChunkPos, usize, usize, usize) {
    let chunk_size = CHUNK_SIZE as f64;
    let chunk_x = world_coordinates.x.div_euclid(chunk_size) as i32;
    let chunk_y = world_coordinates.y.div_euclid(chunk_size) as i32;
    let chunk_z = world_coordinates.z.div_euclid(chunk_size) as i32;
    let local_x = world_coordinates.x.rem_euclid(chunk_size) as usize;
    let local_y = world_coordinates.y.rem_euclid(chunk_size) as usize;
    let local_z = world_coordinates.z.rem_euclid(chunk_size) as usize;
    (ChunkPos::new(chunk_x, chunk_y, chunk_z), local_x, local_y, local_z)
}

/// Axis-aligned bounding box of a chunk in world space, as `(min, max)`.
pub fn chunk_world_aabb(chunk_position: ChunkPos) -> ([f32; 3], [f32; 3]) {
    let chunk_size = CHUNK_SIZE as f32;
    let min = [
        chunk_position.x as f32 * chunk_size,
        chunk_position.y as f32 * chunk_size,
        chunk_position.z as f32 * chunk_size,
    ];
    let max = [min[0] + chunk_size, min[1] + chunk_size, min[2] + chunk_size];
    (min, max)
}

/// Bounding sphere of a chunk in world space, as `(center, radius)`.
pub fn chunk_bounding_sphere(chunk_position: ChunkPos) -> (Vec3, f32) {
    let chunk_size = CHUNK_SIZE as f32;
    let half = chunk_size * 0.5;
    let center = Vec3::new(
        chunk_position.x as f32 * chunk_size + half,
        chunk_position.y as f32 * chunk_size + half,
        chunk_position.z as f32 * chunk_size + half,
    );
    (center, half * DIAGONAL_FACTOR.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_to_chunk_local_handles_negative_coordinates() {
        let chunk_size = CHUNK_SIZE as f64;
        let (chunk_pos, local_x, _, _) = world_to_chunk_local(DVec3::new(-1.0, 0.0, 0.0));
        assert_eq!(chunk_pos.x, -1);
        assert!((local_x as f64 - (chunk_size - 1.0)).abs() < ACCEPTABLE_MACHINE_ERROR as f64);
    }

    #[test]
    fn local_offset_stays_within_chunk_bounds() {
        let chunk_size = CHUNK_SIZE as f32;
        let (_, local_x, local_y, local_z) = world_to_chunk_local(DVec3::new(100.0, 200.5, -37.0));
        for local_coordinate in [local_x, local_y, local_z] {
            assert!((0.0..chunk_size).contains(local_coordinate));
        }
    }

    #[test]
    fn block_to_chunk_floors_toward_negative_infinity() {
        assert_eq!(block_to_chunk(-1, 0, 0), ChunkPos::new(-1, 0, 0));
        assert_eq!(block_to_chunk(0, 0, 0), ChunkPos::new(0, 0, 0));
        assert_eq!(block_to_chunk(CHUNK_SIZE as i32, 0, 0), ChunkPos::new(1, 0, 0));
    }

    #[test]
    fn offset_moves_by_integer_steps_on_each_axis() {
        assert_eq!(ChunkPos::new(0, 0, 0).offset(1, -2, 3), ChunkPos::new(1, -2, 3));
    }

    #[test]
    fn chunk_world_aabb_spans_one_chunk() {
        let chunk_size = CHUNK_SIZE as f32;
        let (min, max) = chunk_world_aabb(ChunkPos::new(2, 0, -1));
        assert_eq!(min, [2.0 * chunk_size, 0.0, -chunk_size]);
        assert_eq!(max, [3.0 * chunk_size, chunk_size, 0.0]);
    }

    #[test]
    fn bounding_sphere_center_is_chunk_midpoint() {
        let chunk_size = CHUNK_SIZE as f32;
        let (center, radius) = chunk_bounding_sphere(ChunkPos::new(0, 0, 0));
        assert_eq!(center, Vec3::splat(chunk_size * 0.5));
        assert!((radius - chunk_size * 0.5 * DIAGONAL_FACTOR.sqrt()).abs() < ACCEPTABLE_MACHINE_ERROR);
    }
}
