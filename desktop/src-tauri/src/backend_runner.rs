use std::process::{Child, Command, Stdio};
use std::io::{BufRead, BufReader};
use std::sync::Mutex;
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, State};

#[derive(Serialize, Clone)]
pub struct BackendStatusPayload {
    pub ready: bool,
    pub port: u16,
}

pub struct BackendManager {
    process: Mutex<Option<Child>>,
}

impl BackendManager {
    pub fn new() -> Self {
        BackendManager {
            process: Mutex::new(None),
        }
    }
}

pub fn spawn_and_monitor(app: AppHandle) {
    // 若 Python 启动器已经拉起后端，这里只转发已有端口。
    if let Ok(port_str) = std::env::var("CYBER_AGENT_BACKEND_PORT") {
        if let Ok(port) = port_str.parse::<u16>() {
            println!("[backend] using pre-started backend on port {}", port);
            let _ = app.emit(
                "backend:status",
                BackendStatusPayload {
                    ready: true,
                    port,
                },
            );
            return;
        }
    }

    let python_cmd = if cfg!(target_os = "windows") {
        "python"
    } else {
        "python3"
    };

    // 解析项目根目录；开发模式下当前目录通常是 desktop/。
    let project_root = std::env::current_dir().unwrap_or_default();

    let mut child = match Command::new(python_cmd)
        .args([
            "-m",
            "cyber_agent.cli.ide_server",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
        ])
        .current_dir(&project_root)
        .env("PYTHONUNBUFFERED", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[backend] failed to spawn: {}", e);
            let _ = app.emit(
                "backend:status",
                BackendStatusPayload {
                    ready: false,
                    port: 0,
                },
            );
            return;
        }
    };

    let pid = child.id();
    println!("[backend] spawned Python backend (pid={})", pid);

    let stdout = child.stdout.take().expect("failed to capture stdout");
    let stderr = child.stderr.take().expect("failed to capture stderr");

    // 保存子进程句柄，便于退出时清理。
    if let Ok(mut proc) = app.state::<BackendManager>().process.lock() {
        *proc = Some(child);
    }

    let app_clone = app.clone();

    // 从 stdout 读取后端随机分配的端口。
    std::thread::spawn(move || {
        let reader = BufReader::new(stdout);
        let mut found_port = false;
        for line in reader.lines() {
            match line {
                Ok(text) => {
                    println!("[backend] {}", text);
                    if let Some(port_str) = text.strip_prefix("IDE_SERVER_PORT=") {
                        if let Ok(port) = port_str.trim().parse::<u16>() {
                            found_port = true;
                            let _ = app_clone.emit(
                                "backend:status",
                                BackendStatusPayload {
                                    ready: true,
                                    port,
                                },
                            );
                            break;
                        }
                    }
                }
                Err(_) => break,
            }
        }
        if !found_port {
            let _ = app_clone.emit(
                "backend:status",
                BackendStatusPayload {
                    ready: false,
                    port: 0,
                },
            );
        }
    });

    // 后台转发 stderr，便于排查后端启动错误。
    std::thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines() {
            if let Ok(text) = line {
                eprintln!("[backend:err] {}", text);
            }
        }
    });
}

#[tauri::command]
pub fn stop_backend(state: State<'_, BackendManager>) -> Result<(), String> {
    let mut proc_guard = state.process.lock().map_err(|e| e.to_string())?;
    if let Some(mut child) = proc_guard.take() {
        let pid = child.id();
        println!("[backend] stopping process (pid={})", pid);

        // 先尝试终止进程。
        #[cfg(unix)]
        {
            unsafe {
                libc::kill(child.id() as i32, libc::SIGTERM);
            }
        }
        #[cfg(not(unix))]
        {
            let _ = child.kill();
        }

        // 最多等待 5 秒，仍未退出则强制清理。
        let start = std::time::Instant::now();
        let mut exited = false;
        while start.elapsed() < std::time::Duration::from_secs(5) {
            match child.try_wait() {
                Ok(Some(_)) => {
                    exited = true;
                    break;
                }
                Ok(None) => std::thread::sleep(std::time::Duration::from_millis(200)),
                Err(_) => break,
            }
        }
        if !exited {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
    Ok(())
}
