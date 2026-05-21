mod backend_runner;
mod pty;

use backend_runner::BackendManager;
use pty::TerminalManager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .manage(BackendManager::new())
        .manage(TerminalManager::new())
        .invoke_handler(tauri::generate_handler![
            backend_runner::stop_backend,
            pty::terminal_create,
            pty::terminal_write,
            pty::terminal_resize,
            pty::terminal_kill,
        ])
        .setup(|app| {
            let handle = app.handle().clone();
            std::thread::spawn(move || {
                backend_runner::spawn_and_monitor(handle);
            });
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
