use crate::{
    event_handling::EventInfo,
    graphical_core::camera::EyeMatrices
};

pub struct FpsCounter {
    frames: u32,
    elapsed: f32,
    display_fps: u32,
}

impl FpsCounter {
    pub fn new() -> Self {
        Self {
            frames: 0,
            elapsed: 0.0,
            display_fps: 0,
        }
    }

    pub fn tick(&mut self, dt: f32) {
        self.frames += 1;
        self.elapsed += dt;
        if self.elapsed >= 0.5 {
            self.display_fps = (self.frames as f32 / self.elapsed) as u32;
            self.frames = 0;
            self.elapsed = 0.0;
        }
    }

    fn display(&self) -> String {
        format!("{} fps", self.display_fps)
    }
}

pub fn draw_ui(event_info: &mut EventInfo, eyes: EyeMatrices) -> anyhow::Result<()> {
    event_info.application.ui.begin_frame();
    event_info
        .application
        .ui
        .draw_text(&event_info.fps_counter.display(), 4.0, 4.0, 16.0, [1.0, 1.0, 1.0, 0.8]);
    let pos_text = format!("pos=({:.1},{:.1},{:.1})", event_info.player.x, event_info.player.y, event_info.player.z);
    event_info.application.ui.draw_text(&pos_text, 4.0, 22.0, 14.0, [1.0, 1.0, 1.0, 0.8]);
    unsafe { event_info.application.render_frame(&event_info.user_window, &event_info.camera, &eyes) }
}
