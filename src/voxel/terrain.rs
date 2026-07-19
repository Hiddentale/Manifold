use super::biome::{self, Biome};
use super::block::BlockType;
use super::chunk::{Chunk, CHUNK_SIZE};
use super::world::{TERRAIN_MAX_CY, TERRAIN_MIN_CY};
use noise::{Fbm, MultiFractal, NoiseFn, Perlin, RidgedMulti};

pub(crate) const SEA_LEVEL: usize = 64;
const DIRT_DEPTH: usize = 4;
const MIN_HEIGHT: usize = 4;
const MAX_HEIGHT: usize = 700;
const CAVE_THRESHOLD: f64 = 0.55;
const CAVE_MIN_DEPTH: usize = 20;
const CHUNK_LAYERS: usize = (TERRAIN_MAX_CY - TERRAIN_MIN_CY + 1) as usize;

const CONTINENTALNESS_SCALE: f64 = 0.0008;
const EROSION_SCALE: f64 = 0.002;
const WEIRDNESS_SCALE: f64 = 0.004;
const DETAIL_SCALE: f64 = 0.02;
const MOUNTAIN_SCALE: f64 = 0.005;
const CAVE_SCALE: f64 = 0.05;
const TEMPERATURE_SCALE: f64 = 0.001;
const HUMIDITY_SCALE: f64 = 0.001;
const WARP_SCALE: f64 = 0.003;
pub(crate) const WARP_STRENGTH: f64 = 80.0;

const OVERHANG_SCALE: f64 = 0.04;
const OVERHANG_STRENGTH: f64 = 1.5;
const OVERHANG_BAND: usize = 20;

const MOUNTAIN_AMPLITUDE: f64 = 25.0;
const DETAIL_AMPLITUDE: f64 = 4.0;
const WEIRDNESS_AMPLITUDE: f64 = 10.0;

pub(crate) struct WorldNoises {
    pub(crate) continentalness: Fbm<Perlin>,
    pub(crate) erosion_noise: Fbm<Perlin>,
    pub(crate) weirdness: Fbm<Perlin>,
    detail: Fbm<Perlin>,
    mountain: RidgedMulti<Perlin>,
    cave: Perlin,
    temperature: Fbm<Perlin>,
    humidity: Fbm<Perlin>,
    pub(crate) warp_x: Fbm<Perlin>,
    pub(crate) warp_y: Fbm<Perlin>,
    pub(crate) warp_z: Fbm<Perlin>,
    overhang: Perlin,
}

impl WorldNoises {
    pub(crate) fn new(seed: u32) -> Self {
        Self {
            continentalness: Fbm::<Perlin>::new(seed)
                .set_frequency(CONTINENTALNESS_SCALE)
                .set_octaves(5)
                .set_persistence(0.5)
                .set_lacunarity(2.0),
            erosion_noise: Fbm::<Perlin>::new(seed + 9)
                .set_frequency(EROSION_SCALE)
                .set_octaves(4)
                .set_persistence(0.5)
                .set_lacunarity(2.0),
            weirdness: Fbm::<Perlin>::new(seed + 10)
                .set_frequency(WEIRDNESS_SCALE)
                .set_octaves(3)
                .set_persistence(0.5)
                .set_lacunarity(2.0),
            detail: Fbm::<Perlin>::new(seed + 1)
                .set_frequency(DETAIL_SCALE)
                .set_octaves(3)
                .set_persistence(0.5)
                .set_lacunarity(2.0),
            mountain: RidgedMulti::<Perlin>::new(seed + 2).set_frequency(MOUNTAIN_SCALE).set_octaves(4),
            cave: Perlin::new(seed + 3),
            temperature: Fbm::<Perlin>::new(seed + 4)
                .set_frequency(TEMPERATURE_SCALE)
                .set_octaves(3)
                .set_persistence(0.5)
                .set_lacunarity(2.0),
            humidity: Fbm::<Perlin>::new(seed + 5)
                .set_frequency(HUMIDITY_SCALE)
                .set_octaves(3)
                .set_persistence(0.5)
                .set_lacunarity(2.0),
            warp_x: Fbm::<Perlin>::new(seed + 6)
                .set_frequency(WARP_SCALE)
                .set_octaves(3)
                .set_persistence(0.5)
                .set_lacunarity(2.0),
            warp_y: Fbm::<Perlin>::new(seed + 11)
                .set_frequency(WARP_SCALE)
                .set_octaves(3)
                .set_persistence(0.5)
                .set_lacunarity(2.0),
            warp_z: Fbm::<Perlin>::new(seed + 7)
                .set_frequency(WARP_SCALE)
                .set_octaves(3)
                .set_persistence(0.5)
                .set_lacunarity(2.0),
            overhang: Perlin::new(seed + 8),
        }
    }
}

/// All noise router parameters for a single (x, z) position.
#[allow(dead_code)]
struct TerrainParams {
    continentalness: f64,
    erosion: f64,
    weirdness: f64,
    temperature: f64,
    humidity: f64,
    world_y: usize,
    biome: Biome,
}

/// Sample all terrain parameters at flat world (x, z) coordinates, applying
/// domain warping.
fn sample_params(noises: &WorldNoises, world_x: f64, world_z: f64, erosion_map: Option<&super::erosion::ErosionMap>) -> TerrainParams {
    let initial_coordinates = [world_x, 0.0, world_z];
    let warped_x = world_x + noises.warp_x.get(initial_coordinates) * WARP_STRENGTH;
    let warped_z = world_z + noises.warp_z.get(initial_coordinates) * WARP_STRENGTH;
    let warped_coordinates = [warped_x, 0.0, warped_z];

    let continentalness = noises.continentalness.get(warped_coordinates);
    let erosion = noises.erosion_noise.get(warped_coordinates);
    let weirdness = noises.weirdness.get(warped_coordinates);
    let temperature = noises.temperature.get(warped_coordinates);
    let humidity = noises.humidity.get(warped_coordinates);
    let mut world_y = compute_height_from_params(noises, warped_x, warped_z, continentalness, erosion, weirdness);

    if let Some(emap) = erosion_map {
        let delta = emap.sample(world_x, world_z);
        world_y = (world_y as f64 + delta).clamp(MIN_HEIGHT as f64, MAX_HEIGHT as f64) as usize;
    }

    let biome = biome::determine_biome(continentalness, temperature, humidity, erosion, weirdness, world_y, SEA_LEVEL);

    TerrainParams {
        continentalness,
        erosion,
        weirdness,
        temperature,
        humidity,
        world_y,
        biome,
    }
}

/// Maps continentalness [-1, 1] to a base height offset from sea level.
/// Piecewise linear: deep ocean → shelf → coast → lowland → highland.
fn continental_curve(c: f64) -> f64 {
    if c < -0.4 {
        // Deep ocean: -40 at c=-1.0 to -10 at c=-0.4
        lerp(-40.0, -10.0, (c + 1.0) / 0.6)
    } else if c < -0.2 {
        // Ocean shelf: -10 to 0
        lerp(-10.0, 0.0, (c + 0.4) / 0.2)
    } else if c < 0.0 {
        // Coast: 0 to +5
        lerp(0.0, 5.0, (c + 0.2) / 0.2)
    } else if c < 0.5 {
        // Lowland: +5 to +30
        lerp(5.0, 30.0, c / 0.5)
    } else {
        // Highland: +30 to +80
        lerp(30.0, 80.0, (c - 0.5) / 0.5)
    }
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

pub(crate) fn compute_height_from_params(noises: &WorldNoises, u: f64, v: f64, continentalness: f64, erosion: f64, weirdness: f64) -> usize {
    let base = continental_curve(continentalness);
    let p = [u, 0.0, v];

    // Erosion controls terrain roughness: high erosion = full mountains, low = flat
    let erosion_factor = (0.3 + erosion * 0.7).clamp(0.3, 1.0);
    let mountain = noises.mountain.get(p) * MOUNTAIN_AMPLITUDE * erosion_factor;
    let detail = noises.detail.get(p) * DETAIL_AMPLITUDE * erosion_factor;
    let weirdness_offset = weirdness * WEIRDNESS_AMPLITUDE;

    let height = SEA_LEVEL as f64 + base + mountain + detail + weirdness_offset;
    height.clamp(MIN_HEIGHT as f64, MAX_HEIGHT as f64) as usize
}

/// Generates a full column of chunks by evaluating a density function at
/// every block. Density > 0 is solid; density <= 0 with `y <= SEA_LEVEL` is
/// water; otherwise air.
pub fn generate_column(chunk_x: i32, chunk_z: i32, seed: u32, erosion_map: Option<&super::erosion::ErosionMap>) -> Vec<Chunk> {
    let noises = WorldNoises::new(seed);
    let mut chunks: Vec<Chunk> = (0..CHUNK_LAYERS).map(|_| Chunk::new(BlockType::Air)).collect();

    for z in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            fill_density_column(&mut chunks, chunk_x, chunk_z, x, z, &noises, erosion_map);
        }
    }

    chunks
}

/// Per-(x, z) column fill: walks every vertical layer, evaluates density at
/// the block center, and writes the resulting block type. The per-column
/// noise (continentalness, mountain, biome, …) is sampled ONCE for the whole
/// column — it depends only on (x, z), which is constant as `ly` varies.
/// Only the 3D cave noise samples per block.
fn fill_density_column(
    chunks: &mut [Chunk],
    chunk_x: i32,
    chunk_z: i32,
    x: usize,
    z: usize,
    noises: &WorldNoises,
    erosion_map: Option<&super::erosion::ErosionMap>,
) {
    let wx = chunk_x as f64 * CHUNK_SIZE as f64 + x as f64 + 0.5;
    let wz = chunk_z as f64 * CHUNK_SIZE as f64 + z as f64 + 0.5;
    let params = sample_params(noises, wx, wz, erosion_map);
    let surface_block = biome::surface_block(params.biome);
    let subsurface_block = biome::subsurface_block(params.biome);

    for (cy, chunk) in chunks.iter_mut().enumerate().take(CHUNK_LAYERS) {
        for ly in 0..CHUNK_SIZE {
            let wy = cy * CHUNK_SIZE + ly;
            let block = sample_density_block(wy, params.height, surface_block, subsurface_block, noises, wx, wz);
            if block != BlockType::Air {
                chunk.set(x, ly, z, block);
            }
        }
    }
}

/// Per-block density evaluation. Column-dependent values are passed in.
///
/// **Surface contract**: a block is solid iff `y <= height`. There is no 3D
/// overhang noise carving the surface — that would create a per-block height
/// field that diverges from the analytical height, breaking LOD parity with
/// the heightmap tile path. Caves are still allowed strictly below the
/// surface (`depth_from_surface > CAVE_MIN_DEPTH`) so they never punch
/// through the visible top.
///
/// Pinned by `heightmap_top_matches_chunked_top_within_one_block` in
/// `voxel::heightmap_generator::tests`.
fn sample_density_block(y: usize, height: usize, surface: BlockType, subsurface: BlockType, noises: &WorldNoises, wx: f64, wz: f64) -> BlockType {
    if y > height {
        return if y <= SEA_LEVEL { BlockType::Water } else { BlockType::Air };
    }

    // At or below the surface — pick stone / subsurface / surface based on depth.
    let depth_from_surface = height - y;
    let block = if depth_from_surface < 1 {
        surface
    } else if depth_from_surface < DIRT_DEPTH {
        subsurface
    } else {
        BlockType::Stone
    };

    // 3D cave carving — spheres of air punched out of the solid mass.
    // Caves must stay well below the surface so they don't expose tall walls
    // when an adjacent column happens to be solid right where this column
    // has a cave. With CAVE_SCALE=0.05 (period ~20 blocks) the cave features
    // are ~10 blocks across; a depth-from-surface threshold of CAVE_MIN_DEPTH
    // ensures even the topmost cave block sits well under the surface band.
    if depth_from_surface > CAVE_MIN_DEPTH {
        let cave_val = noises.cave.get([wx * CAVE_SCALE, y as f64 * CAVE_SCALE, wz * CAVE_SCALE]);
        if cave_val > CAVE_THRESHOLD {
            return BlockType::Air;
        }
    }

    block
}

/// Generate a 64³ LOD super-chunk by sampling terrain noise at `voxel_size` spacing.
pub fn generate_lod_super_chunk(origin: [i32; 3], voxel_size: u32, seed: u32, erosion_map: Option<&super::erosion::ErosionMap>) -> LodVoxelGrid {
    let noises = WorldNoises::new(seed);
    let vs = voxel_size as f64;
    let grid_size = CHUNK_SIZE * 4; // 64
    let mut blocks = vec![BlockType::Air; grid_size * grid_size * grid_size];

    for gz in 0..grid_size {
        for gx in 0..grid_size {
            let wx = origin[0] as f64 + gx as f64 * vs;
            let wz = origin[2] as f64 + gz as f64 * vs;
            let params = sample_params(&noises, wx, wz, erosion_map);
            let surface = biome::surface_block(params.biome);
            let subsurface = biome::subsurface_block(params.biome);

            for gy in 0..grid_size {
                let wy = origin[1] as f64 + gy as f64 * vs;
                let y_top = (wy + vs - 1.0) as usize;
                let block = sample_block(y_top, params.height, surface, subsurface, &noises, wx, wy + vs * 0.5, wz);
                blocks[gx + gz * grid_size + gy * grid_size * grid_size] = block;
            }
        }
    }

    // Strip underground: keep only top SURFACE_DEPTH solid voxels per column.
    const SURFACE_DEPTH: usize = 2;
    for gz in 0..grid_size {
        for gx in 0..grid_size {
            let col = gx + gz * grid_size;
            let mut top = 0;
            for gy in (0..grid_size).rev() {
                if blocks[col + gy * grid_size * grid_size] != BlockType::Air {
                    top = gy;
                    break;
                }
            }
            if top >= SURFACE_DEPTH {
                for gy in 0..top - SURFACE_DEPTH {
                    blocks[col + gy * grid_size * grid_size] = BlockType::Air;
                }
            }
        }
    }

    LodVoxelGrid { blocks, size: grid_size }
}

fn sample_block(y: usize, height: usize, surface: BlockType, subsurface: BlockType, noises: &WorldNoises, wx: f64, wy: f64, wz: f64) -> BlockType {
    if y > height && y <= SEA_LEVEL {
        return BlockType::Water;
    }
    if y > height + OVERHANG_BAND {
        return BlockType::Air;
    }

    let band_bottom = height.saturating_sub(OVERHANG_BAND);
    let band_top = height + OVERHANG_BAND;
    if y >= band_bottom && y <= band_top {
        let base_density = (height as f64 - y as f64) / OVERHANG_BAND as f64;
        let noise_val = noises.overhang.get([wx * OVERHANG_SCALE, wy * OVERHANG_SCALE, wz * OVERHANG_SCALE]);
        let density = base_density + noise_val * (OVERHANG_STRENGTH / OVERHANG_BAND as f64);
        if density <= 0.0 {
            return BlockType::Air;
        }
    }

    let block = if y >= height {
        surface
    } else if y + DIRT_DEPTH > height {
        subsurface
    } else {
        BlockType::Stone
    };

    if y >= 1 && y + 5 <= height {
        let cave_val = noises.cave.get([wx * CAVE_SCALE, wy * CAVE_SCALE, wz * CAVE_SCALE]);
        if cave_val > CAVE_THRESHOLD {
            return BlockType::Air;
        }
    }

    block
}

/// A flat 64³ voxel grid for LOD super-chunk generation.
pub struct LodVoxelGrid {
    blocks: Vec<BlockType>,
    size: usize,
}

impl super::svdag::VoxelSource for LodVoxelGrid {
    fn get(&self, x: usize, y: usize, z: usize) -> BlockType {
        self.blocks[x + z * self.size + y * self.size * self.size]
    }
}
