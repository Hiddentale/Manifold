use crate::{
    game_state::GameState,
    graphical_core::{
        camera::{Camera, EyeMatrices},
        input::InputState,
        vulkan_object::{VulkanApplication, WORLD_DISTANCE},
    },
    hud::{draw_ui, FpsCounter},
    menu::draw_menu,
    storage::world_meta::create_world,
    utils::rand_seed,
    voxel::player::Player,
};
use std::time::Instant;
use winit::{
    dpi::LogicalSize,
    event::{DeviceEvent, ElementState, KeyEvent, MouseButton, WindowEvent},
    event_loop::EventLoopWindowTarget,
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowBuilder},
};

pub struct EventInfo {
    game_state: GameState,
    input: InputState,
    pub player: Player,
    pub camera: Camera,
    pub fps_counter: FpsCounter,
    last_frame: Instant,
    pub application: VulkanApplication,
    pub user_window: Window,
    cursor_position: [f32; 2],
    physics_accumulator: f32,
    minimized: bool,
    menu_click: bool,
    destroy_application: bool,
}

pub fn initialize_event_info(event_handler: &EventLoopWindowTarget<()>) -> anyhow::Result<EventInfo> {
    let user_window = WindowBuilder::new()
        .with_title("Manifold")
        .with_inner_size(LogicalSize::new(1024, 768))
        .build(event_handler)?;

    let application = unsafe { VulkanApplication::create_vulkan_application(&user_window) }?;

    let event_info = EventInfo {
        game_state: GameState::TitleScreen,
        input: InputState::new(),
        player: Player::new(),
        camera: Camera::default(),
        fps_counter: FpsCounter::new(),
        last_frame: Instant::now(),
        application,
        user_window,
        cursor_position: [0.0; 2],
        physics_accumulator: 0.0,
        minimized: false,
        menu_click: false,
        destroy_application: false,
    };
    Ok(event_info)
}

pub fn handle_window_event(event: WindowEvent, event_info: &mut EventInfo, current_window: &EventLoopWindowTarget<()>) -> anyhow::Result<()> {
    match event {
        WindowEvent::CloseRequested => {
            exit_program(&mut event_info.destroy_application, current_window, &mut event_info.application);
        }
        WindowEvent::Resized(size) => {
            if size.width == 0 || size.height == 0 {
                event_info.minimized = true;
            } else {
                event_info.minimized = false;
                event_info.application.resized = true;
            }
        }
        WindowEvent::CursorMoved { position, .. } => {
            event_info.cursor_position = [position.x as f32, position.y as f32];
        }
        WindowEvent::KeyboardInput { event: key_event, .. } => handle_keyboard_input(key_event, current_window, event_info),
        WindowEvent::MouseInput {
            state: ElementState::Pressed,
            button: MouseButton::Left,
            ..
        } => {
            if event_info.game_state.is_menu() {
                event_info.menu_click = true;
            } else {
                event_info.input.mouse_pressed(MouseButton::Left);
            }
        }
        WindowEvent::MouseInput {
            state: ElementState::Pressed,
            button,
            ..
        } if !event_info.game_state.is_menu() => {
            event_info.input.mouse_pressed(button);
        }
        WindowEvent::RedrawRequested => {
            if event_info.destroy_application || event_info.minimized {
                return Ok(());
            }
            let eyes = EyeMatrices::from_camera(&event_info.camera, event_info.application.swapchain_extent());
            let result = match &event_info.game_state {
                GameState::Playing => draw_ui(event_info, eyes),
                _ => {
                    let extent = event_info.application.swapchain_extent();
                    let screen_width = extent.width as f32;
                    let screen_height = extent.height as f32;
                    let clicked = event_info.menu_click;
                    event_info.menu_click = false;

                    event_info.application.ui.begin_frame();
                    draw_menu(
                        &mut event_info.application.ui,
                        &mut event_info.game_state,
                        screen_width,
                        screen_height,
                        event_info.cursor_position,
                        clicked,
                    );
                    unsafe { event_info.application.render_menu_frame(&event_info.user_window, &eyes) }
                }
            };
            if let Err(e) = result {
                eprintln!("Render error: {e}");
                exit_program(&mut event_info.destroy_application, current_window, &mut event_info.application);
            }
        }
        _ => {}
    }
    Ok(())
}

pub fn handle_device_event(event: DeviceEvent, event_info: &mut EventInfo) {
    if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
        if matches!(event_info.game_state, GameState::Playing) {
            event_info.input.accumulate_mouse_delta(dx, dy);
        }
    }
}

fn handle_keyboard_input(key_event: KeyEvent, current_window: &EventLoopWindowTarget<()>, event_info: &mut EventInfo) {
    if let PhysicalKey::Code(key_code) = key_event.physical_key {
        match &mut event_info.game_state {
            GameState::Playing => match key_event.state {
                ElementState::Pressed => {
                    if key_code == KeyCode::Escape {
                        unsafe { event_info.application.exit_world() };
                        event_info.game_state = GameState::TitleScreen;
                        release_cursor(&event_info.user_window);
                        return;
                    }
                    if key_code == KeyCode::KeyF && !key_event.repeat {
                        event_info.player.toggle_fly_mode();
                    }
                    if key_code == KeyCode::Space {
                        event_info.player.jump();
                    }
                    event_info.input.key_pressed(key_code);
                }
                ElementState::Released => event_info.input.key_released(key_code),
            },
            GameState::CreateWorld { name, seed_text } => {
                if key_event.state == ElementState::Pressed {
                    if key_code == KeyCode::Escape {
                        event_info.game_state = GameState::TitleScreen;
                        return;
                    }
                    if key_code == KeyCode::Backspace {
                        if !seed_text.is_empty() {
                            seed_text.pop();
                        } else {
                            name.pop();
                        }
                        return;
                    }
                    if key_code == KeyCode::Enter && !name.is_empty() {
                        let seed: u32 = seed_text.parse().unwrap_or_else(|_| rand_seed());
                        match create_world(name, seed) {
                            Ok(dir) => {
                                let stream_rd = WORLD_DISTANCE;
                                let side = (2 * stream_rd + 1) as usize;
                                let total = side * side;
                                event_info.game_state = GameState::PreGenerating {
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
                    if let Some(text) = &key_event.text {
                        for char in text.chars() {
                            if char.is_ascii_digit() {
                                if seed_text.len() < 10 {
                                    seed_text.push(char);
                                }
                            } else if (char.is_ascii_alphanumeric() || char == ' ' || char == '-' || char == '_') && name.len() < 24 {
                                name.push(char);
                            }
                        }
                    }
                }
            }
            _ => {
                if key_event.state == ElementState::Pressed && key_code == KeyCode::Escape {
                    match &event_info.game_state {
                        GameState::WorldSelect { .. } => event_info.game_state = GameState::TitleScreen,
                        GameState::TitleScreen => {
                            exit_program(&mut event_info.destroy_application, current_window, &mut event_info.application);
                        }
                        _ => {}
                    }
                }
            }
        }
    }
}

pub fn handle_wait_event(event_info: &mut EventInfo) {
    const MAX_PHYSICS_CATCHUP: f32 = 0.25;
    const PHYSICS_TICK: f32 = 1.0 / 60.0;
    let now = Instant::now();
    let delta_time = (now - event_info.last_frame).as_secs_f32();
    event_info.last_frame = now;
    event_info.fps_counter.tick(delta_time);
    match &mut event_info.game_state {
        GameState::Playing => {
            event_info.input.apply_camera_rotation(&mut event_info.player);
            event_info.physics_accumulator = (event_info.physics_accumulator + delta_time).min(MAX_PHYSICS_CATCHUP);
            while event_info.physics_accumulator >= PHYSICS_TICK {
                if let Some(world) = event_info.application.world() {
                    let local_p = world.metric.sample_metric_at_pos(event_info.player.world_position()).minkowski_exponent;
                    event_info.input.tick_movement(&mut event_info.player, world, PHYSICS_TICK, local_p);
                }
                event_info.physics_accumulator -= PHYSICS_TICK;
            }
            event_info.camera.sync_from_player(&event_info.player);
            event_info.input.take_left_click();
            event_info.input.take_right_click();
        }
        GameState::PreGenerating { .. } => {
            unsafe { event_info.application.update_world(&event_info.camera).ok() };
            tick_pregen(
                &mut event_info.game_state,
                &mut event_info.application,
                &event_info.user_window,
                &mut event_info.camera,
                &mut event_info.player,
            );
        }
        GameState::EnteringWorld { .. } => {
            if let GameState::EnteringWorld { world_dir, seed } = std::mem::replace(&mut event_info.game_state, GameState::Playing) {
                if let Err(e) = unsafe { event_info.application.enter_world(&world_dir, seed) } {
                    eprintln!("Failed to enter world: {e}");
                    event_info.game_state = GameState::TitleScreen;
                } else {
                    grab_cursor(&event_info.user_window);
                    event_info.camera = Camera::default();
                    event_info.player = Player::new();
                }
            }
        }
        _ => {}
    }
    event_info.user_window.request_redraw();
}

fn tick_pregen(state: &mut GameState, application: &mut VulkanApplication, window: &winit::window::Window, camera: &mut Camera, player: &mut Player) {
    let GameState::PreGenerating {
        world_dir,
        seed,
        loaded,
        total,
    } = state
    else {
        return;
    };

    if !application.has_loaded_world() {
        let dir = world_dir.clone();
        let s = *seed;
        if let Err(e) = unsafe { application.enter_world(&dir, s) } {
            eprintln!("Failed to enter world: {e}");
            *state = GameState::TitleScreen;
            return;
        }
    }
    let columns_count = if let Some(world) = application.world() {
        let mut columns_seen = std::collections::HashSet::new();
        for chunk_pos in world.chunk_positions() {
            columns_seen.insert((chunk_pos.x, chunk_pos.z));
        }
        columns_seen.len()
    } else {
        0
    };
    *loaded = columns_count;
    if columns_count >= *total {
        *state = GameState::Playing;
        grab_cursor(window);
        *camera = Camera::default();
        *player = Player::new();
    }
}

fn grab_cursor(window: &winit::window::Window) {
    if window.set_cursor_grab(CursorGrabMode::Confined).is_err() {
        let _ = window.set_cursor_grab(CursorGrabMode::Locked);
    }
    window.set_cursor_visible(false);
}

pub fn release_cursor(window: &winit::window::Window) {
    let _ = window.set_cursor_grab(CursorGrabMode::None);
    window.set_cursor_visible(true);
}

fn exit_program(destroy_application: &mut bool, current_window: &EventLoopWindowTarget<()>, application: &mut VulkanApplication) {
    *destroy_application = true;
    current_window.exit();
    unsafe { application.destroy_vulkan_application() }
}
