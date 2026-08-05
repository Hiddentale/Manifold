//! Player state.

use crate::voxel::grid::ChunkPos; 
use super::world::World;
use glam::Vec3;

const GRAVITY: f32 = 20.0;
const JUMP_VELOCITY: f32 = 8.0;
const PLAYER_HEIGHT: f32 = 1.7;
const PLAYER_HALF_WIDTH: f32 = 0.3;
const CAMERA_FORWARD_OFFSET: f32 = 0.25;
pub const MAX_PITCH: f32 = 89.0_f32 * (std::f32::consts::PI / 180.0);

pub struct Player {
    pub chunk_x: i32,
    pub chunk_y: i32,
    pub chunk_z: i32,
    pub local_x: f32,
    pub local_y: f32,
    pub local_z: f32,
    pub forward: Vec3,
    pub right: Vec3,
    pub radial_velocity: f32,
    pub on_ground: bool,
    pub fly_mode: bool,
}

impl Player {
    pub fn new() -> Self {
        let forward = Vec3::new(1.0, 0.0, 0.0);
        let right = Vec3::new(0.0, 0.0, -1.0);
        Self {
            chunk_x: 0,
            chunk_y: 90,
            chunk_z: 0,
            local_x: 8.0,
            local_y: 8.0,
            local_z: 8.0,
            forward,
            right,
            radial_velocity: 0.0,
            on_ground: false,
            fly_mode: true,
        }
    }

    pub fn chunk_pos(&self) -> ChunkPos {
        ChunkPos {
            x: self.chunk_x,
            y: self.chunk_y,
            z: self.chunk_z,
        }
    }

    /// Cartesian body-center position. Always derived; never stored.
    pub fn world_pos(&self) -> Vec3 {
        super::grid::chunk_to_world(self.chunk_pos(), Vec3::new(self.lx, self.ly, self.lz)).as_vec3()
    }

    /// Cartesian eye position: body center plus a small forward offset so the
    /// camera can approach walls more closely than the body's half-width.
    pub fn eye_pos(&self) -> Vec3 {
        self.world_pos() + self.forward * CAMERA_FORWARD_OFFSET
    }

    /// Radial outward direction at the player.
    pub fn up(&self) -> Vec3 {
        self.world_pos().normalize_or(Vec3::Y)
    }

    pub fn right(&self) -> Vec3 {
        self.right
    }

    /// True if any integer block index in the player's capsule footprint
    /// equals `(target_chunk, ix, iy, iz)`. Used to refuse a block placement
    /// that would embed the player.
    pub fn overlaps_block(&self, target_chunk: ChunkPos, ix: usize, iy: usize, iz: usize) -> bool {
        let hw = PLAYER_HALF_WIDTH;
        let lx_min = (self.lx - hw).floor() as i32;
        let lx_max = (self.lx + hw).floor() as i32;
        let lz_min = (self.lz - hw).floor() as i32;
        let lz_max = (self.lz + hw).floor() as i32;
        let ly_min = (self.ly - PLAYER_HEIGHT + 0.01).floor() as i32;
        let ly_max = (self.ly - 0.01).floor() as i32;
        for sx in lx_min..=lx_max {
            for sy in ly_min..=ly_max {
                for sz in lz_min..=lz_max {
                    if same_block(self.face, self.cx, self.cy, self.cz, sx, sy, sz, target_chunk, ix, iy, iz) {
                        return true;
                    }
                }
            }
        }
        false
    }

    pub fn jump(&mut self) {
        if self.on_ground && !self.fly_mode {
            self.radial_velocity = JUMP_VELOCITY;
            self.on_ground = false;
        }
    }

    pub fn toggle_fly_mode(&mut self) {
        self.fly_mode = !self.fly_mode;
        if !self.fly_mode {
            self.snap_walk_basis();
        }
    }
    
    /// Walk a tangent-plane displacement (`tangent_world`) of length in
    /// blocks. Splits into forward and right components in the local tangent
    /// plane and applies each as a world-space step with collision. World-
    /// space integration is required so that motion at off-center positions
    /// (corners, edges) follows the actual sphere tangent rather than the
    /// face's flat cube basis — otherwise pressing W near a corner produces
    /// a diagonal slide.
    pub fn walk(&mut self, tangent_world: Vec3, world: &World) {
        let up = self.up();
        let forward_h = (self.forward - up * self.forward.dot(up)).normalize_or(Vec3::ZERO);
        if forward_h == Vec3::ZERO {
            return;
        }
        let right_h = forward_h.cross(up).normalize_or(Vec3::ZERO);
        let forward_amt = tangent_world.dot(forward_h);
        let right_amt = tangent_world.dot(right_h);
        self.try_world_step(forward_h * forward_amt, world);
        self.try_world_step(right_h * right_amt, world);
    }

    /// Fly with full 6DoF in world space. Adds the displacement to the
    /// player's cartesian position and re-derives the cube-space coords
    /// via the inverse projection. Decoupled from the source face's flat
    /// basis so motion follows the screen, not the underlying cube.
    pub fn fly_move(&mut self, displacement_world: Vec3) {
        let cur = sphere::chunk_to_world(self.chunk_pos(), Vec3::new(self.lx, self.ly, self.lz));
        let new_world = cur + displacement_world.as_dvec3();
        let unit_eps = FACE_HYSTERESIS / sphere::CUBE_HALF_BLOCKS;
        if let Some((cp, lx, ly, lz)) = sphere::world_to_chunk_local_hysteretic(new_world, Some(self.face), unit_eps) {
            let n = sphere::FACE_SIDE_CHUNKS;
            if cp.cy >= 0 {
                self.face = cp.face;
                self.cx = cp.cx.clamp(0, n - 1);
                self.cy = cp.cy;
                self.cz = cp.cz.clamp(0, n - 1);
                self.lx = lx;
                self.ly = ly;
                self.lz = lz;
                let up_old = cur.as_vec3().normalize_or(Vec3::Y);
                let up_new = self.world_pos().normalize_or(Vec3::Y);
                self.forward = parallel_transport(self.forward, up_old, up_new);
                self.right = parallel_transport(self.right, up_old, up_new);
                self.reorthonormalize_basis();
            }
        }
    }

    /// Apply radial gravity. Sticky `on_ground` is verified each frame by
    /// probing the block below the feet.
    pub fn apply_physics(&mut self, dt: f32, world: &World) {
        if self.fly_mode {
            return;
        }
        if self.on_ground {
            if ground_below(self, world) {
                return; // standing — no motion this frame
            }
            self.on_ground = false;
        }

        self.radial_velocity -= GRAVITY * dt;
        let dd = self.radial_velocity * dt;
        // Try the radial step with collision; if blocked, zero velocity and
        // mark grounded. Avoid the previous "embed then lift up" approach,
        // which could push the player INTO blocks above them.
        let moved = self.try_axis(0.0, dd, 0.0, world);
        if !moved {
            self.radial_velocity = 0.0;
            self.on_ground = true;
        }
    }

    /// Apply a single-axis displacement and revert that axis on collision.
    /// Returns true if the move succeeded.
    fn try_axis(&mut self, du: f32, dd: f32, dv: f32, world: &World) -> bool {
        let (old_face, old_cx, old_cy, old_cz) = (self.face, self.cx, self.cy, self.cz);
        let (old_lx, old_ly, old_lz) = (self.lx, self.ly, self.lz);
        self.lx += du;
        self.ly += dd;
        self.lz += dv;
        self.carry();
        if capsule_collides(self, world) {
            self.face = old_face;
            self.cx = old_cx;
            self.cy = old_cy;
            self.cz = old_cz;
            self.lx = old_lx;
            self.ly = old_ly;
            self.lz = old_lz;
            false
        } else {
            true
        }
    }
}

/// Iterate every integer block the player's capsule overlaps. For a
/// PLAYER_HEIGHT × 2*HALF_WIDTH × 2*HALF_WIDTH box this is at most
/// 3 × 2 × 2 = 12 lookups, all direct chunk array reads.
fn capsule_collides(player: &Player, world: &World) -> bool {
    let hw = PLAYER_HALF_WIDTH;
    let lx_min = (player.lx - hw).floor() as i32;
    let lx_max = (player.lx + hw).floor() as i32;
    let lz_min = (player.lz - hw).floor() as i32;
    let lz_max = (player.lz + hw).floor() as i32;
    let ly_min = (player.ly - PLAYER_HEIGHT + 0.01).floor() as i32;
    let ly_max = (player.ly - 0.01).floor() as i32;
    for ix in lx_min..=lx_max {
        for iy in ly_min..=ly_max {
            for iz in lz_min..=lz_max {
                if sample_block_solid(player.face, player.cx, player.cy, player.cz, ix, iy, iz, world) {
                    return true;
                }
            }
        }
    }
    false
}

fn ground_below(player: &Player, world: &World) -> bool {
    let hw = PLAYER_HALF_WIDTH;
    let lx_min = (player.lx - hw).floor() as i32;
    let lx_max = (player.lx + hw).floor() as i32;
    let lz_min = (player.lz - hw).floor() as i32;
    let lz_max = (player.lz + hw).floor() as i32;
    let iy = (player.ly - PLAYER_HEIGHT - 0.05).floor() as i32;
    for ix in lx_min..=lx_max {
        for iz in lz_min..=lz_max {
            if sample_block_solid(player.face, player.cx, player.cy, player.cz, ix, iy, iz, world) {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
}
