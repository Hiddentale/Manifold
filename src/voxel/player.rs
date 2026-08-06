//! Player state.

use crate::voxel::world::World; 
use glam::Vec3;

const GRAVITY: f32 = 20.0;
const JUMP_VELOCITY: Vec3 = Vec3::new(0.0, 8.0, 0.0);
const CAMERA_FORWARD_OFFSET: f32 = 0.25;

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
}

impl Player {
    pub fn new() -> Self {
        let forward = Vec3::new(1.0, 0.0, 0.0);
        let right = Vec3::new(0.0, 0.0, -1.0);
        let velocity = Vec3::new(0.0, 0.0, 0.0);
        Self {
            x: 0.0,
            y: 90.0,
            z: 0.0,
            height: 1.7,
            half_width: 0.3,
            forward,
            right,
            velocity,
            on_ground: true,
            fly_mode: false,
        }
    }

    /// Cartesian body-center position.
    pub fn world_position(&self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }

    /// Cartesian eye position.
    pub fn camera_position(&self) -> Vec3 {
        self.world_position() + self.forward * CAMERA_FORWARD_OFFSET
    }
    
    pub fn right_vector(&self) -> Vec3 {
        self.right
    }

    pub fn apply_physics(&mut self, dt: f32, player: &Player, world: &World) {
        self.velocity.y -= GRAVITY * dt;
        // Might have issue here where if we jump and hit a ceiling, can_move will give us false?
        if !self.can_move(0.0, self.velocity.y * dt, 0.0, player, world) {
            self.velocity.y = 0.0;
            self.on_ground = true;
        }
    }

    pub fn can_move(&mut self, dx: f32, dy: f32, dz: f32, player: &Player, world: &World) -> bool {
            let old_x = self.x;
            let old_y = self.y;
            let old_z = self.z;
            
            self.x += dx;
            self.y += dy;
            self.z += dz;
            
            if self.capsule_collides(player, world) {
                self.x = old_x;
                self.y = old_y;
                self.z = old_z;
                return false;
            }
            true
    }

    pub fn walk(&mut self, direction: Vec3, world: &World) {
        todo!()
    }

    pub fn sweep_capsule(&self, delta: Vec3, world: &World) -> bool {
        todo!()
    }

    pub fn capsule_collides(&self, player: &Player, world: &World) -> bool {
        let angles = [0.0, std::f32::consts::PI / 2.0, std::f32::consts::PI, 3.0 * std::f32::consts::PI / 2.0];
            
        for y_offset in [0.0, 0.85, player.height] {
            for &angle in &angles {
                let dx = (angle.cos() * player.half_width).round() as i32;
                let dz = (angle.sin() * player.half_width).round() as i32;
                    
                if block_is_solid(player.x as i32 + dx, (player.y + y_offset) as i32, player.z as i32 + dz, world) {
                    return true;
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
        todo!()
    }

}


fn block_is_solid(block_x: i32, block_y: i32, block_z: i32, world: &World) -> bool {
    world.block_solid_at(block_x, block_y, block_z)
}

#[cfg(test)]
mod tests {
    use super::*;
}
