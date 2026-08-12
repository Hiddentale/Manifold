use crate::{
    game_state::GameState,
    graphical_core::{ui_pipeline::UiPipeline, vulkan_object::WORLD_DISTANCE},
    storage::world_meta::{create_world, list_worlds},
    utils::rand_seed,
};

const BUTTON_W: f32 = 300.0;
const BUTTON_H: f32 = 40.0;
const TEXT_SIZE: f32 = 20.0;

const BUTTON_COLOR: [f32; 4] = [0.2, 0.2, 0.3, 0.85];
const BUTTON_HOVER: [f32; 4] = [0.3, 0.3, 0.5, 0.9];
const TEXT_COLOR: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
const DIM_TEXT: [f32; 4] = [0.7, 0.7, 0.7, 1.0];

/// Draw a button, returns true if clicked.
fn button_clicked(ui: &mut UiPipeline, label: &str, x: f32, y: f32, cursor: [f32; 2], clicked: bool) -> bool {
    let hovered = cursor[0] >= x && cursor[0] <= x + BUTTON_W && cursor[1] >= y && cursor[1] <= y + BUTTON_H;
    let color = if hovered { BUTTON_HOVER } else { BUTTON_COLOR };
    ui.draw_rect(x, y, BUTTON_W, BUTTON_H, color);
    let tw = UiPipeline::text_width(label, TEXT_SIZE);
    let tx = x + (BUTTON_W - tw) / 2.0;
    let ty = y + (BUTTON_H - TEXT_SIZE) / 2.0;
    ui.draw_text(label, tx, ty, TEXT_SIZE, TEXT_COLOR);
    hovered && clicked
}

/// Draw the start menu of the game.
pub fn draw_menu(ui: &mut UiPipeline, state: &mut GameState, screen_width: f32, screen_height: f32, cursor: [f32; 2], clicked: bool) {
    const TITLE_SIZE: f32 = 48.0;
    match state {
        GameState::TitleScreen => {
            let title = "MANIFOLD";
            let text_width = UiPipeline::text_width(title, TITLE_SIZE);
            ui.draw_text(title, (screen_width - text_width) / 2.0, screen_height * 0.2, TITLE_SIZE, TEXT_COLOR);

            let cx = (screen_width - BUTTON_W) / 2.0;
            if button_clicked(ui, "Singleplayer", cx, screen_height * 0.45, cursor, clicked) {
                *state = GameState::WorldSelect { worlds: list_worlds() };
            }
            if button_clicked(ui, "Quit", cx, screen_height * 0.55, cursor, clicked) {
                std::process::exit(0);
            }
        }
        GameState::WorldSelect { worlds } => {
            let title = "Select World";
            let text_width = UiPipeline::text_width(title, TEXT_SIZE * 1.5);
            ui.draw_text(title, (screen_width - text_width) / 2.0, 40.0, TEXT_SIZE * 1.5, TEXT_COLOR);

            let cx = (screen_width - BUTTON_W) / 2.0;
            let mut y = 100.0;
            let mut selected = None;
            for (i, (_, meta)) in worlds.iter().enumerate() {
                let label = format!("{} (seed: {})", meta.name, meta.seed);
                if button_clicked(ui, &label, cx, y, cursor, clicked) {
                    selected = Some(i);
                }
                y += BUTTON_H + 10.0;
            }
            if let Some(i) = selected {
                let (dir, meta) = &worlds[i];
                *state = GameState::EnteringWorld {
                    world_dir: dir.clone(),
                    seed: meta.seed,
                };
                return;
            }

            y += 20.0;
            if button_clicked(ui, "Create New World", cx, y, cursor, clicked) {
                *state = GameState::CreateWorld {
                    name: String::new(),
                    seed_text: String::new(),
                };
                return;
            }
            if button_clicked(ui, "Back", cx, y + BUTTON_H + 10.0, cursor, clicked) {
                *state = GameState::TitleScreen;
            }
        }
        GameState::CreateWorld { name, seed_text } => {
            let title = "Create New World";
            let text_width = UiPipeline::text_width(title, TEXT_SIZE * 1.5);
            ui.draw_text(title, (screen_width - text_width) / 2.0, 40.0, TEXT_SIZE * 1.5, TEXT_COLOR);

            let cx = (screen_width - BUTTON_W) / 2.0;
            let mut y = 120.0;

            ui.draw_text("Name:", cx, y, TEXT_SIZE, DIM_TEXT);
            y += 25.0;
            ui.draw_rect(cx, y, BUTTON_W, BUTTON_H, [0.15, 0.15, 0.2, 0.9]);
            let name_empty = name.is_empty();
            let display_name: &str = if name_empty { "type a name..." } else { name };
            let name_color = if name_empty { DIM_TEXT } else { TEXT_COLOR };
            ui.draw_text(display_name, cx + 10.0, y + 10.0, TEXT_SIZE, name_color);
            y += BUTTON_H + 20.0;

            ui.draw_text("Seed (digits, optional):", cx, y, TEXT_SIZE, DIM_TEXT);
            y += 25.0;
            ui.draw_rect(cx, y, BUTTON_W, BUTTON_H, [0.15, 0.15, 0.2, 0.9]);
            let seed_empty = seed_text.is_empty();
            let display_seed: &str = if seed_empty { "random" } else { seed_text };
            let seed_color = if seed_empty { DIM_TEXT } else { TEXT_COLOR };
            ui.draw_text(display_seed, cx + 10.0, y + 10.0, TEXT_SIZE, seed_color);
            y += BUTTON_H + 30.0;

            if !name.is_empty() {
                if button_clicked(ui, "Create & Play", cx, y, cursor, clicked) {
                    let seed: u32 = seed_text.parse().unwrap_or_else(|_| rand_seed());
                    match create_world(name, seed) {
                        Ok(dir) => {
                            let stream_rd = WORLD_DISTANCE;
                            let side = (2 * stream_rd + 1) as usize;
                            let total = side * side;
                            *state = GameState::PreGenerating {
                                world_dir: dir,
                                seed,
                                loaded: 0,
                                total,
                            };
                        }
                        Err(e) => eprintln!("Failed to create world: {e}"),
                    }
                    return;
                }
                y += BUTTON_H + 10.0;
            }
            if button_clicked(ui, "Back", cx, y, cursor, clicked) {
                *state = GameState::TitleScreen;
            }
        }
        GameState::PreGenerating { loaded, total, .. } => {
            let title = if *loaded >= *total { "Finishing up..." } else { "Generating terrain..." };
            let text_width = UiPipeline::text_width(title, TEXT_SIZE * 1.5);
            ui.draw_text(
                title,
                (screen_width - text_width) / 2.0,
                screen_height * 0.35,
                TEXT_SIZE * 1.5,
                TEXT_COLOR,
            );

            let bar_w = 400.0;
            let bar_h = 30.0;
            let bx = (screen_width - bar_w) / 2.0;
            let by = screen_height * 0.5;
            let progress = if *total > 0 { *loaded as f32 / *total as f32 } else { 0.0 };
            ui.draw_rect(bx, by, bar_w, bar_h, [0.15, 0.15, 0.2, 0.9]);
            ui.draw_rect(bx + 2.0, by + 2.0, (bar_w - 4.0) * progress.min(1.0), bar_h - 4.0, [0.3, 0.7, 0.3, 1.0]);

            let progress_counter = format!("{}%", (progress * 100.0).min(100.0) as u32);
            let progress_counter_width = UiPipeline::text_width(&progress_counter, TEXT_SIZE);
            ui.draw_text(
                &progress_counter,
                (screen_width - progress_counter_width) / 2.0,
                by + 5.0,
                TEXT_SIZE,
                TEXT_COLOR,
            );
        }
        GameState::EnteringWorld { .. } | GameState::Playing => {}
    }
}
