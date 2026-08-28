//! Player state.

use crate::voxel::world::World;
use glam::Vec3;

const GRAVITY: f32 = 20.0;
const JUMP_VELOCITY: Vec3 = Vec3::new(0.0, 8.0, 0.0);
const CAMERA_FORWARD_OFFSET: f32 = 0.25;
const WORLD_UP_VECTOR: Vec3 = Vec3::new(0.0, 1.0, 0.0);
/// Keeps pitch just short of ±90° so forward/right never go parallel to WORLD_UP.
const PITCH_LIMIT: f32 = 1.5533; // ~89 degrees

pub struct Player {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub height: f32,
    pub half_width: f32,
    pub forward: Vec3,
    pub right: Vec3,
    pub velocity: Vec3,
    pub on_ground: bool,
    pub fly_mode: bool,
    yaw: f32,
    pitch: f32,
}

impl Player {
    pub fn new() -> Self {
        let mut player = Self {
            x: 0.0,
            y: 90.0,
            z: 0.0,
            height: 1.7,
            half_width: 0.3,
            forward: Vec3::ZERO,
            right: Vec3::ZERO,
            velocity: Vec3::ZERO,
            on_ground: true,
            fly_mode: false,
            yaw: 0.0,
            pitch: 0.0,
        };
        player.update_look_vectors();
        player
    }

    /// Cartesian body-center position.
    pub fn world_position(&self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }

    /// Cartesian eye position.
    pub fn camera_position(&self) -> Vec3 {
        const EYE_HEIGHT: f32 = 1.6; // Make this depend on height in future
        self.world_position() + Vec3::new(0.0, EYE_HEIGHT, 0.0) + self.forward * CAMERA_FORWARD_OFFSET
    }

    /// Cartesian right direction relative to player.
    pub fn right_vector(&self) -> Vec3 {
        self.right
    }

    /// Cartesian unit vector in the up direction.
    pub fn up(&self) -> Vec3 {
        WORLD_UP_VECTOR
    }

    /// Turn left/right around world-up.
    pub fn rotate_yaw(&mut self, delta: f32) {
        self.yaw -= delta;
        self.update_look_vectors();
    }

    /// Look up/down, clamped to ±~89° so the camera can't flip over.
    pub fn rotate_pitch(&mut self, delta: f32) {
        self.pitch = (self.pitch + delta).clamp(-PITCH_LIMIT, PITCH_LIMIT);
        self.update_look_vectors();
    }

    /// Compute the player's forward and right direction vectors from the current yaw
    /// and pitch angles.
    fn update_look_vectors(&mut self) {
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        self.forward = Vec3::new(cos_yaw * cos_pitch, sin_pitch, sin_yaw * cos_pitch).normalize();
        self.right = self.forward.cross(WORLD_UP_VECTOR).normalize();
    }

    /// Collisionless moving
    pub fn fly_move(&mut self, delta: Vec3) {
        self.x += delta.x;
        self.y += delta.y;
        self.z += delta.z;
    }

    pub fn apply_physics(&mut self, dt: f32, world: &World) {
        if !self.fly_mode {
            self.velocity.y -= GRAVITY * dt;
            if !self.can_move(0.0, self.velocity.y * dt, 0.0, world) {
                if self.velocity.y <= 0.0 {
                    self.on_ground = true;
                }
                self.velocity.y = 0.0;
            }
        }
    }

    pub fn can_move(&mut self, dx: f32, dy: f32, dz: f32, world: &World) -> bool {
        let old_x = self.x;
        let old_y = self.y;
        let old_z = self.z;

        self.x += dx;
        self.y += dy;
        self.z += dz;

        if self.capsule_collides(world) {
            self.x = old_x;
            self.y = old_y;
            self.z = old_z;
            return false;
        }
        true
    }

    /// Walk-mode horizontal move with wall sliding: if the combined (x, z)
    /// move is blocked, retry each axis independently.
    pub fn walk(&mut self, direction: Vec3, world: &World) {
        if self.can_move(direction.x, 0.0, direction.z, world) {
            return;
        }
        self.can_move(direction.x, 0.0, 0.0, world);
        self.can_move(0.0, 0.0, direction.z, world);
    }

    pub fn swept_AABB(&self, world: &World) {
        // Step 1: Find time and distance until collision and for how long collision happens
        //  on each axis separately.
        // So need to find first block that gets hit by raycast.
        // 
        // Step 2: Find which axis collides first (by time)
        // if all axes agree on the collision, we know an actual collision has happened this frame.
        if () {

        }
        else {
            // Step 3: if there was a collision, calculate the normal of the edge that was collided with  
        }



        todo!()
    }

    pub fn capsule_collides(&self, world: &World) -> bool {
        let player_min_x = (self.x - self.half_width).floor() as i32;
        let player_min_y = self.y.floor() as i32;
        let player_min_z = (self.z - self.half_width).floor() as i32;
        let player_max_x = (self.x + self.half_width).floor() as i32;
        let player_max_y = (self.y + self.height).floor() as i32;
        let player_max_z = (self.z + self.half_width).floor() as i32;

        for block_x in player_min_x..=player_max_x {
            for block_y in player_min_y..=player_max_y {
                for block_z in player_min_z..=player_max_z {
                    if block_is_solid(
                        block_x,
                        block_y,
                        block_z,
                        world,
                    ) {
                        return true;
                    }
                }
            }
        }
        false
    }

    pub fn jump(&mut self) {
        if self.on_ground && !self.fly_mode {
            self.velocity += JUMP_VELOCITY;
            self.on_ground = false;
        }
    }

    pub fn toggle_fly_mode(&mut self) {
        self.fly_mode = !self.fly_mode;
        self.velocity = Vec3::ZERO;
    }
}

fn block_is_solid(block_x: i32, block_y: i32, block_z: i32, world: &World) -> bool {
    world.block_solid_at(block_x, block_y, block_z)
}

#[cfg(test)]
mod tests {
    // TODO
}
