use tauri::Emitter;
use std::io::Write;

const LOG_DIR: &str = "/tmp/cyber-agent-ide";

fn ide_log(msg: &str) {
    let _ = std::fs::create_dir_all(LOG_DIR);
    if let Ok(mut f) = std::fs::OpenOptions::new().append(true).create(true).open(format!("{LOG_DIR}/tauri.log")) {
        let _ = writeln!(f, "[tauri] {msg}");
    }
    eprintln!("[cyber-ide-tauri] {msg}");
}

#[tauri::command]
fn get_server_port(state: tauri::State<'_, ServerState>) -> Result<u16, String> {
    let port = state.0.load(std::sync::atomic::Ordering::SeqCst);
    ide_log(&format!("invoke get_server_port → {port}"));
    port.try_into().map_err(|e| format!("端口读取失败: {}", e))
}

struct ServerPort(std::sync::atomic::AtomicU16);

type ServerState = std::sync::Arc<ServerPort>;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let server_port = std::sync::Arc::new(ServerPort(std::sync::atomic::AtomicU16::new(0)));

    ide_log("Tauri run() 启动");

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(server_port.clone())
        .invoke_handler(tauri::generate_handler![get_server_port])
        .setup(move |app| {
            ide_log("Tauri setup 开始");

            let handle = app.handle().clone();
            let port_atomic = server_port.clone();

            // 优先使用环境变量传入的端口（由 cyber-agent ide 设置）
            if let Ok(port_str) = std::env::var("CYBER_IDE_PORT") {
                ide_log(&format!("读取到 CYBER_IDE_PORT={port_str}"));
                if let Ok(port) = port_str.parse::<u16>() {
                    port_atomic.0.store(port, std::sync::atomic::Ordering::SeqCst);
                    ide_log(&format!("端口已存储: {port}, 发送 backend-ready 事件"));
                    let _ = handle.emit("backend-ready", port);
                    return Ok(());
                } else {
                    ide_log(&format!("CYBER_IDE_PORT 解析失败: {port_str}"));
                }
            } else {
                ide_log("未找到 CYBER_IDE_PORT 环境变量，回退自行启动后端");
            }

            // 回退：自行启动后端服务器
            std::thread::spawn(move || {
                ide_log("回退线程: 尝试启动 cyber-agent ide-server");

                use std::process::{Command, Stdio};
                use std::io::BufRead;

                let cmds = vec![
                    vec!["cyber-agent".to_string(), "ide-server".to_string(), "--port".to_string(), "0".to_string()],
                ];
                for args in &cmds {
                    ide_log(&format!("回退线程: 执行命令 {:?}", args));
                    let mut cmd = Command::new(&args[0]);
                    cmd.args(&args[1..]);
                    cmd.stdout(Stdio::piped());
                    cmd.stderr(Stdio::piped());

                    match cmd.spawn() {
                        Ok(mut child) => {
                            ide_log("回退线程: 子进程启动成功，等待端口...");
                            let stdout = match child.stdout.take() {
                                Some(s) => s,
                                None => {
                                    ide_log("回退线程: 无法捕获 stdout");
                                    continue;
                                }
                            };
                            let reader = std::io::BufReader::new(stdout);

                            for line in reader.lines() {
                                if let Ok(line) = line {
                                    ide_log(&format!("回退线程: stdout: {line}"));
                                    if line.starts_with("IDE_SERVER_PORT=") {
                                        if let Ok(port) = line
                                            .trim_start_matches("IDE_SERVER_PORT=")
                                            .parse::<u16>()
                                        {
                                            port_atomic.0.store(port, std::sync::atomic::Ordering::SeqCst);
                                            let _ = handle.emit("backend-ready", port);
                                            ide_log(&format!("回退线程: 端口 {port} 已就绪，事件已发送"));
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
                                        ide_log(&format!("[backend-stderr] {l}"));
                                    }
                                }
                            }

                            let _ = child.wait();
                            ide_log("回退线程: 子进程已退出");
                            return;
                        }
                        Err(e) => {
                            ide_log(&format!("回退线程: 启动失败: {e}"));
                        }
                    }
                }

                ide_log("回退线程: 所有启动尝试均失败！");
            });

            ide_log("Tauri setup 完成");
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("启动 IDE 时出错");
}
