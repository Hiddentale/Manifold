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

#[allow(dead_code)]
struct TerrainParams {
    continentalness: f64,
    erosion: f64,
    weirdness: f64,
    temperature: f64,
    humidity: f64,
    surface_height: usize,
    biome: Biome,
}

fn sample_params(noises: &WorldNoises, world_x: f64, world_z: f64) -> TerrainParams {
    let initial_coordinates = [world_x, 0.0, world_z];
    let warped_x = world_x + noises.warp_x.get(initial_coordinates) * WARP_STRENGTH;
    let warped_z = world_z + noises.warp_z.get(initial_coordinates) * WARP_STRENGTH;
    let warped_coordinates = [warped_x, 0.0, warped_z];

    let continentalness = noises.continentalness.get(warped_coordinates);
    let erosion = noises.erosion_noise.get(warped_coordinates);
    let weirdness = noises.weirdness.get(warped_coordinates);
    let temperature = noises.temperature.get(warped_coordinates);
    let humidity = noises.humidity.get(warped_coordinates);
    let surface_height = compute_height_from_params(noises, warped_x, warped_z, continentalness, erosion, weirdness);

    let biome = biome::determine_biome(continentalness, temperature, humidity, erosion, weirdness, surface_height, SEA_LEVEL);

    TerrainParams {
        continentalness,
        erosion,
        weirdness,
        temperature,
        humidity,
        surface_height,
        biome,
    }
}

const DEEP_OCEAN_OFFSET: f64 = -0.4;
const OCEAN_SHELF_OFFSET: f64 = -0.2;
const COAST_OFFSET: f64 = 0.0;
const LOWLAND_OFFSET: f64 = 0.5;

const DEEP_OCEAN_HEIGHT_MIN: f64 = -40.0;
const DEEP_OCEAN_HEIGHT_MAX: f64 = -10.0;
const DEEP_OCEAN_START: f64 = -1.0;
const DEEP_OCEAN_END: f64 = -0.4;
const DEEP_OCEAN_RANGE: f64 = DEEP_OCEAN_END - DEEP_OCEAN_START;

const OCEAN_SHELF_HEIGHT_MIN: f64 = -10.0;
const OCEAN_SHELF_HEIGHT_MAX: f64 = 0.0;
const OCEAN_SHELF_START: f64 = -0.4;
const OCEAN_SHELF_END: f64 = -0.2;
const OCEAN_SHELF_RANGE: f64 = OCEAN_SHELF_END - OCEAN_SHELF_START;

const COAST_HEIGHT_MIN: f64 = 0.0;
const COAST_HEIGHT_MAX: f64 = 5.0;
const COAST_START: f64 = -0.2;
const COAST_END: f64 = 0.0;
const COAST_RANGE: f64 = COAST_END - COAST_START;

const LOWLAND_HEIGHT_MIN: f64 = 5.0;
const LOWLAND_HEIGHT_MAX: f64 = 30.0;
const LOWLAND_START: f64 = 0.0;
const LOWLAND_END: f64 = 0.5;
const LOWLAND_RANGE: f64 = LOWLAND_END - LOWLAND_START;

const HIGHLAND_HEIGHT_MIN: f64 = 30.0;
const HIGHLAND_HEIGHT_MAX: f64 = 80.0;
const HIGHLAND_START: f64 = 0.5;
const HIGHLAND_END: f64 = 1.0;
const HIGHLAND_RANGE: f64 = HIGHLAND_END - HIGHLAND_START;

fn map_continental_curve(continentalness: f64) -> f64 {
    if continentalness < DEEP_OCEAN_OFFSET {
        lerp(DEEP_OCEAN_HEIGHT_MIN, DEEP_OCEAN_HEIGHT_MAX, (continentalness - DEEP_OCEAN_START) / DEEP_OCEAN_RANGE)
    } else if continentalness < OCEAN_SHELF_OFFSET {
        lerp(OCEAN_SHELF_HEIGHT_MIN, OCEAN_SHELF_HEIGHT_MAX, (continentalness - OCEAN_SHELF_START) / OCEAN_SHELF_RANGE)
    } else if continentalness < COAST_OFFSET {
        lerp(COAST_HEIGHT_MIN, COAST_HEIGHT_MAX, (continentalness - COAST_START) / COAST_RANGE)
    } else if continentalness < LOWLAND_OFFSET {
        lerp(LOWLAND_HEIGHT_MIN, LOWLAND_HEIGHT_MAX, (continentalness - LOWLAND_START) / LOWLAND_RANGE)
    } else {
        lerp(HIGHLAND_HEIGHT_MIN, HIGHLAND_HEIGHT_MAX, (continentalness - HIGHLAND_START) / HIGHLAND_RANGE)
    }
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

const MINIMUM_EROSION_FACTOR: f64 = 0.3;
const MAXIMUM_EROSION_FACTOR: f64 = 1.0;
const EROSION_RANGE: f64 = MAXIMUM_EROSION_FACTOR - MINIMUM_EROSION_FACTOR;


/// Calculates the height of a certain (x,z) position.
pub(crate) fn compute_height_from_params(noises: &WorldNoises, x: f64, z: f64, continentalness: f64, erosion: f64, weirdness: f64) -> usize {
    let average_continental_height = map_continental_curve(continentalness);
    let coordinates = [x, 0.0, z];
    
    let erosion_factor = (MINIMUM_EROSION_FACTOR + erosion * EROSION_RANGE).clamp(MINIMUM_EROSION_FACTOR, MAXIMUM_EROSION_FACTOR);
    let mountain = noises.mountain.get(coordinates) * MOUNTAIN_AMPLITUDE * erosion_factor;
    let detail = noises.detail.get(coordinates) * DETAIL_AMPLITUDE * erosion_factor;
    let weirdness_offset = weirdness * WEIRDNESS_AMPLITUDE;

    let height = SEA_LEVEL as f64 + average_continental_height + mountain + detail + weirdness_offset;
    height.clamp(MIN_HEIGHT as f64, MAX_HEIGHT as f64) as usize
}

/// Generates a full column of chunks by evaluating a density function at
/// every block.
pub fn generate_column(chunk_x: i32, chunk_z: i32, seed: u32) -> Vec<Chunk> {
    let noises = WorldNoises::new(seed);
    let mut chunks: Vec<Chunk> = (0..CHUNK_LAYERS).map(|_| Chunk::new(BlockType::Air)).collect();

    for z in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            fill_density_column(&mut chunks, chunk_x, chunk_z, x, z, &noises);
        }
    }

    chunks
}

fn fill_density_column(
    chunks: &mut [Chunk],
    chunk_x: i32,
    chunk_z: i32,
    x: usize,
    z: usize,
    noises: &WorldNoises,
) {
    let world_x = chunk_x as f64 * CHUNK_SIZE as f64 + x as f64 + 0.5;
    let world_z = chunk_z as f64 * CHUNK_SIZE as f64 + z as f64 + 0.5;
    let params = sample_params(noises, world_x, world_z);
    let surface_block = biome::surface_block(params.biome);
    let subsurface_block = biome::subsurface_block(params.biome);

    for (chunk_y, chunk) in chunks.iter_mut().enumerate().take(CHUNK_LAYERS) {
        for local_y in 0..CHUNK_SIZE {
            let world_y = (chunk_y * CHUNK_SIZE + local_y) as f64;
            let block = find_blocktype(world_x, world_y, world_z, params.surface_height, surface_block, subsurface_block, noises, );
            if block != BlockType::Air {
                chunk.set(x, local_y, z, block);
            }
        }
    }
}

const SURFACE_DEPTH: usize = 1;

fn find_blocktype(world_x: f64, world_y: f64, world_z: f64, height: usize, surface: BlockType, subsurface: BlockType, noises: &WorldNoises) -> BlockType {
    if world_y > height as f64 {
        if world_y <= SEA_LEVEL as f64 {
            return BlockType::Water 
        } 
        else {
            return BlockType::Air 
        };
    }
    
    let depth_from_surface = (height as f64 - world_y).max(0.0).round() as usize;
    let block = if depth_from_surface < SURFACE_DEPTH {
        surface
    } else if depth_from_surface < DIRT_DEPTH {
        subsurface
    } else {
        BlockType::Stone
    };

    if depth_from_surface > CAVE_MIN_DEPTH {
        let cave_val = noises.cave.get([world_x * CAVE_SCALE, world_y as f64 * CAVE_SCALE, world_z * CAVE_SCALE]);
        if cave_val > CAVE_THRESHOLD {
            return BlockType::Air;
        }
    }

    block
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
