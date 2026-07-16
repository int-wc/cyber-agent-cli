use std::collections::HashMap;
use std::io::{Read, Write};
use std::sync::Mutex;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, State};

#[derive(Serialize, Clone)]
pub struct TerminalOutputPayload {
    pub session_id: String,
    pub data: String,
}

#[derive(Serialize)]
pub struct SessionInfo {
    pub session_id: String,
    pub pid: u32,
}

#[derive(Deserialize)]
pub struct CreateOptions {
    pub shell: Option<String>,
}

#[derive(Deserialize)]
pub struct WriteData {
    #[serde(rename = "sessionId")]
    pub session_id: String,
    pub data: String,
}

#[derive(Deserialize)]
pub struct ResizeData {
    #[serde(rename = "sessionId")]
    pub session_id: String,
    pub cols: u16,
    pub rows: u16,
}

#[derive(Deserialize)]
pub struct KillData {
    #[serde(rename = "sessionId")]
    pub session_id: String,
}

struct Session {
    writer: Box<dyn Write + Send>,
}

pub struct TerminalManager {
    sessions: Mutex<HashMap<String, Session>>,
    counter: Mutex<u32>,
}

impl TerminalManager {
    pub fn new() -> Self {
        TerminalManager {
            sessions: Mutex::new(HashMap::new()),
            counter: Mutex::new(0),
        }
    }
}

fn get_shell() -> String {
    if cfg!(target_os = "windows") {
        std::env::var("COMSPEC").unwrap_or_else(|_| "powershell.exe".to_string())
    } else {
        std::env::var("SHELL").unwrap_or_else(|_| "/bin/bash".to_string())
    }
}

#[tauri::command]
pub fn terminal_create(
    app: AppHandle,
    state: State<'_, TerminalManager>,
    options: Option<CreateOptions>,
) -> Result<SessionInfo, String> {
    let shell_cmd = options
        .and_then(|o| o.shell)
        .unwrap_or_else(get_shell);

    let mut cmd = std::process::Command::new(&shell_cmd);
    cmd.stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());

    // 在 Unix 平台使用登录 shell，保持用户环境变量和启动脚本一致。
    if !cfg!(target_os = "windows") {
        cmd.arg("-l");
    }

    let mut child = cmd.spawn().map_err(|e| format!("Failed to spawn {}: {}", shell_cmd, e))?;
    let pid = child.id();

    let session_id = {
        let mut counter = state.counter.lock().map_err(|e| e.to_string())?;
        *counter += 1;
        format!("term-{}", *counter)
    };

    let child_stdin = child
        .stdin
        .take()
        .ok_or_else(|| "Failed to open stdin".to_string())?;

    let child_stdout = child
        .stdout
        .take()
        .ok_or_else(|| "Failed to open stdout".to_string())?;

    let session = Session {
        writer: Box::new(child_stdin),
    };

    state
        .sessions
        .lock()
        .map_err(|e| e.to_string())?
        .insert(session_id.clone(), session);

    // 后台线程读取 stdout，并转发为前端终端事件。
    let app_handle = app.clone();
    let sid = session_id.clone();
    std::thread::spawn(move || {
        let mut reader = std::io::BufReader::new(child_stdout);
        let mut buf = [0u8; 4096];
        loop {
            match reader.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    let data = String::from_utf8_lossy(&buf[..n]).to_string();
                    let _ = app_handle.emit(
                        "terminal:output",
                        TerminalOutputPayload {
                            session_id: sid.clone(),
                            data,
                        },
                    );
                }
                Err(e) => {
                    eprintln!("[pty] read error for {}: {}", sid, e);
                    break;
                }
            }
        }
        let _ = app_handle.emit(
            "terminal:output",
            TerminalOutputPayload {
                session_id: sid.clone(),
                data: "\r\n[Session ended]\r\n".to_string(),
            },
        );
    });

    Ok(SessionInfo {
        session_id,
        pid,
    })
}

#[tauri::command]
pub fn terminal_write(
    state: State<'_, TerminalManager>,
    data: WriteData,
) -> Result<(), String> {
    let mut sessions = state.sessions.lock().map_err(|e| e.to_string())?;
    let session = sessions
        .get_mut(&data.session_id)
        .ok_or_else(|| format!("Session {} not found", data.session_id))?;
    session
        .writer
        .write_all(data.data.as_bytes())
        .map_err(|e| format!("Write error: {}", e))?;
    session
        .writer
        .flush()
        .map_err(|e| format!("Flush error: {}", e))?;
    Ok(())
}

#[tauri::command]
pub fn terminal_resize(
    _state: State<'_, TerminalManager>,
    data: ResizeData,
) -> Result<(), String> {
    let _ = (&data.session_id, data.cols, data.rows);
    // 调整 PTY 尺寸需要真实的 PTY master fd。
    // 当前简单模式使用管道 stdin/stdout，因此暂不支持 resize。
    // 后续若接入 portable-pty，可通过 ioctl(TIOCSWINSZ) 实现。
    Ok(())
}

#[tauri::command]
pub fn terminal_kill(
    state: State<'_, TerminalManager>,
    data: KillData,
) -> Result<bool, String> {
    let mut sessions = state.sessions.lock().map_err(|e| e.to_string())?;
    sessions.remove(&data.session_id);
    Ok(true)
}
