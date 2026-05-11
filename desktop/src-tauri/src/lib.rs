use tauri::Manager;

#[tauri::command]
fn get_server_port(state: tauri::State<'_, ServerState>) -> Result<u16, String> {
    state
        .port
        .load(std::sync::atomic::Ordering::SeqCst)
        .try_into()
        .map_err(|e| format!("端口读取失败: {}", e))
}

struct ServerPort(std::sync::atomic::AtomicU16);

type ServerState = std::sync::Arc<ServerPort>;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let server_port = std::sync::Arc::new(ServerPort(std::sync::atomic::AtomicU16::new(0)));

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(server_port.clone())
        .invoke_handler(tauri::generate_handler![get_server_port])
        .setup(move |app| {
            use std::process::{Command, Stdio};
            use std::io::BufRead;

            let handle = app.handle().clone();
            let port_atomic = server_port.clone();

            // Spawn Python backend server
            std::thread::spawn(move || {
                let mut child = Command::new("cyber-agent")
                    .args(["ide-server", "--port", "0"])
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .spawn()
                    .expect("无法启动 IDE 后端服务。请确认 cyber-agent 已安装。");

                let stdout = child.stdout.take().expect("无法捕获后端输出");
                let reader = std::io::BufReader::new(stdout);

                for line in reader.lines() {
                    if let Ok(line) = line {
                        if line.starts_with("IDE_SERVER_PORT=") {
                            if let Ok(port) = line
                                .trim_start_matches("IDE_SERVER_PORT=")
                                .parse::<u16>()
                            {
                                port_atomic.0.store(port, std::sync::atomic::Ordering::SeqCst);
                                // Emit event to frontend
                                let _ = handle.emit("backend-ready", port);
                            }
                            break;
                        }
                    }
                }

                // Keep stderr draining
                if let Some(stderr) = child.stderr.take() {
                    let stderr_reader = std::io::BufReader::new(stderr);
                    for line in stderr_reader.lines() {
                        if let Ok(line) = line {
                            eprintln!("[backend] {}", line);
                        }
                    }
                }

                let _ = child.wait();
            });

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("启动 IDE 时出错");
}
