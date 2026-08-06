//! Player state.

use crate::voxel::{grid::ChunkPos, world::World}; 
use glam::Vec3;

const GRAVITY: f32 = 20.0;
const JUMP_VELOCITY: Vec3 = Vec3::new(0.0, 8.0, 0.0);
const PLAYER_HEIGHT: f32 = 1.7;
const PLAYER_HALF_WIDTH: f32 = 0.3;
const CAMERA_FORWARD_OFFSET: f32 = 0.25;

pub struct Player {
    pub chunk_x: i32,
    pub chunk_y: i32,
    pub chunk_z: i32,
    pub local_x: f32,
    pub local_y: f32,
    pub local_z: f32,
    pub forward: Vec3,
    pub right: Vec3,
    pub velocity: Vec3,
    pub on_ground: bool,
    pub fly_mode: bool,
}

impl Player {
    pub fn new() -> Self {
        let forward = Vec3::new(1.0, 0.0, 0.0);
        let right = Vec3::new(0.0, 0.0, -1.0);
        let velocity = Vec3::new(0.0, 0.0, 0.0);
        Self {
            chunk_x: 0,
            chunk_y: 90,
            chunk_z: 0,
            local_x: 8.0,
            local_y: 8.0,
            local_z: 8.0,
            forward,
            right,
            velocity,
            on_ground: true,
            fly_mode: false,
        }
    }

    pub fn chunk_pos(&self) -> ChunkPos {
        ChunkPos {
            x: self.chunk_x,
            y: self.chunk_y,
            z: self.chunk_z,
        }
    }

    /// Cartesian body-center position.
    pub fn world_position(&self) -> Vec3 {
        super::grid::chunk_to_world(
            self.chunk_pos(),
            Vec3::new(self.local_x, self.local_y, self.local_z)).as_vec3()
    }

    /// Cartesian eye position.
    pub fn camera_position(&self) -> Vec3 {
        self.world_position() + self.forward * CAMERA_FORWARD_OFFSET
    }
    
    pub fn right_vector(&self) -> Vec3 {
        self.right
    }

    pub fn apply_physics(&mut self, dt: f32, world: &World) {
        todo!()
    }

    pub fn walk(&mut self, direction: Vec3, world: &World) {
        todo!()
    }

    pub fn try_axis(&mut self, dx: f32, dy: f32, dz: f32, world: &World) -> bool {
        todo!()
    }

    pub fn sweep_capsule(&self, delta: Vec3, world: &World) -> bool {
        todo!()
    }

    pub fn capsule_collides(player: &Player, world: &World) -> bool {
        todo!()
    }

    fn sample_block_solid() -> bool {
        todo!()
    }

    pub fn jump(&mut self) {
        if self.on_ground && !self.fly_mode {
            self.velocity += JUMP_VELOCITY;
            self.on_ground = false;
        }
    }

    pub fn toggle_fly_mode(&mut self) {
        todo!()
    }

}

#[cfg(test)]
mod tests {
    use super::*;
}
