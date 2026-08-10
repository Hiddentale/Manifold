mod event_handling;
mod game_state;
mod graphical_core;
mod hud;
mod menu;
mod storage;
mod utils;
mod voxel;
use anyhow::Result;
use event_handling::{handle_device_event, handle_wait_event, handle_window_event, initialize_event_info, release_cursor};
use vulkan_rust::{vk, Version};
use winit::{event::Event, event_loop::EventLoop};

const PORTABILITY_MACOS_VERSION: Version = Version::new(1, 3, 216);
const VALIDATION_ENABLED: bool = cfg!(debug_assertions);

const VALIDATION_LAYER: &std::ffi::CStr = c"VK_LAYER_KHRONOS_validation";
const DEVICE_EXTENSIONS: &[&std::ffi::CStr] = &[vk::extension_names::KHR_SWAPCHAIN_EXTENSION_NAME];

fn main() -> Result<()> {
    initialize_error_handler();
    let event_handler = EventLoop::new()?;
    let mut event_info = initialize_event_info(&event_handler)?;
    release_cursor(&event_info.user_window);

    event_handler
        .run(move |event, current_window| match event {
            Event::WindowEvent { event, .. } => {
                let _ = handle_window_event(event, &mut event_info, current_window);
            }
            Event::DeviceEvent { event, .. } => {
                handle_device_event(event, &mut event_info);
            }
            Event::AboutToWait => {
                handle_wait_event(&mut event_info);
            }
            _ => (),
        })
        .expect("Main function crashed!");
    Ok(())
}

fn initialize_error_handler() {
    pretty_env_logger::init();
}
