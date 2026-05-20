use tauri::Emitter;

#[tauri::command]
fn get_server_port(state: tauri::State<'_, ServerState>) -> Result<u16, String> {
    state
        .0
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
            let handle = app.handle().clone();
            let port_atomic = server_port.clone();

            // 优先使用环境变量传入的端口（由 cyber-agent ide 设置）
            if let Ok(port_str) = std::env::var("CYBER_IDE_PORT") {
                if let Ok(port) = port_str.parse::<u16>() {
                    port_atomic.0.store(port, std::sync::atomic::Ordering::SeqCst);
                    let _ = handle.emit("backend-ready", port);
                    return Ok(());
                }
            }

            // 回退：自行启动后端服务器
            std::thread::spawn(move || {
                use std::process::{Command, Stdio};
                use std::io::BufRead;

                // 尝试多种方式定位 cyber-agent
                let cmds = vec![
                    vec!["cyber-agent".to_string(), "ide-server".to_string(), "--port".to_string(), "0".to_string()],
                ];
                for args in &cmds {
                    let mut cmd = Command::new(&args[0]);
                    cmd.args(&args[1..]);
                    cmd.stdout(Stdio::piped());
                    cmd.stderr(Stdio::piped());

                    if let Ok(mut child) = cmd.spawn() {
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
                                        let _ = handle.emit("backend-ready", port);
                                    }
                                    break;
                                }
                            }
                        }

                        // Drain stderr
                        if let Some(stderr) = child.stderr.take() {
                            let r = std::io::BufReader::new(stderr);
                            for line in r.lines() {
                                if let Ok(l) = line {
                                    eprintln!("[backend] {}", l);
                                }
                            }
                        }

                        let _ = child.wait();
                        return;
                    }
                }

                eprintln!("无法启动 IDE 后端服务。请确认 cyber-agent 已安装。");
            });

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("启动 IDE 时出错");
}
