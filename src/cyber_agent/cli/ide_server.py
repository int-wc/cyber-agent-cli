"""FastAPI + WebSocket 服务器，为桌面 IDE 提供 AI 代理和文件操作后端。"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import socket
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..agent.approval import ApprovalDecision, ApprovalPolicy
from ..agent.mode import AgentMode
from ..agent.runner import AgentRunner, extract_text_content
from ..config import settings
from .app import (
    build_runtime_context,
    create_runner,
    ensure_runtime_capabilities,
)

_AGENT_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agent")
_GIT_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="git")

_RUNNER: AgentRunner | None = None
_RUNTIME_CTX: dict[str, object] = {}


def _load_fastapi():
    import fastapi
    from fastapi.middleware.cors import CORSMiddleware

    return fastapi, CORSMiddleware


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _is_path_allowed(path: Path) -> bool:
    if _RUNNER is None:
        return True
    for root in _RUNNER.allowed_roots:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


class AgentEventBridge:
    """将同步 AgentRunner 事件桥接到异步 WebSocket。"""

    def __init__(self) -> None:
        self.event_queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        self._pending_approvals: dict[str, asyncio.Future[ApprovalDecision]] = {}
        self._loop: asyncio.AbstractEventLoop = asyncio.get_event_loop()
        self._stop_requested = False

    # ── 从 AgentRunner 线程调用 ──

    def event_handler(self, event_type: str, payload: object) -> None:
        if self._loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(
            self.event_queue.put({"type": event_type, "payload": payload}),
            self._loop,
        )

    def approval_handler(self, tool, tool_call: dict) -> ApprovalDecision:
        call_id = str(tool_call.get("id", ""))
        future: asyncio.Future[ApprovalDecision] = asyncio.Future()
        self._pending_approvals[call_id] = future
        self.event_handler("approval_request", {
            "tool_name": tool.name,
            "tool_call_id": call_id,
            "tool_call": tool_call,
            "risk": str((tool.metadata or {}).get("risk", "read")),
        })
        try:
            return asyncio.run_coroutine_threadsafe(
                self._wait_for_approval(call_id, timeout=300), self._loop
            ).result()
        except Exception:
            return ApprovalDecision(False, "审批超时或已取消")

    async def _wait_for_approval(self, call_id: str, timeout: int) -> ApprovalDecision:
        future = self._pending_approvals.pop(call_id, None)
        if future is None:
            return ApprovalDecision(False, "审批请求已失效")
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            return ApprovalDecision(False, "审批超时")

    # ── 从 async 事件循环调用 ──

    async def get_next_event(self) -> dict[str, object] | None:
        try:
            return await asyncio.wait_for(self.event_queue.get(), timeout=0.05)
        except asyncio.TimeoutError:
            return None

    def resolve_approval(self, call_id: str, approved: bool, reason: str = "") -> None:
        future = self._pending_approvals.pop(call_id, None)
        if future is not None and not future.done():
            future.set_result(ApprovalDecision(approved, reason))


def _run_agent_sync(bridge: AgentEventBridge, user_input: str) -> str:
    if _RUNNER is None:
        raise RuntimeError("AgentRunner 未初始化")
    return _RUNNER.run(
        user_input,
        verbose=False,
        event_handler=bridge.event_handler,
        approval_handler=bridge.approval_handler,
    )


# ── FastAPI 应用 ──


def create_app() -> Any:
    fastapi, CORSMiddleware = _load_fastapi()
    app = fastapi.FastAPI(title="Cyber Agent IDE Server", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    _register_routes(app)
    return app


def _register_routes(app: Any) -> None:

    @app.get("/api/health")
    async def health():
        return {
            "status": "ok",
            "version": "0.1.0",
            "session": bool(_RUNNER),
            "service": str(_RUNTIME_CTX.get("service_name", "")),
            "model": str(_RUNTIME_CTX.get("model_name", "")),
        }

    # ── 会话管理 ──

    @app.get("/api/session/status")
    async def session_status():
        if _RUNNER is None:
            return {"error": "会话未就绪"}
        diag = _RUNNER.get_context_diagnostics()
        return {
            "mode": _RUNNER.mode.value,
            "service": _RUNNER.service,
            "model": _RUNNER.model_name,
            "turn_count": _RUNNER.get_turn_count(),
            "history_length": diag["history_message_count"],
            "model_message_count": diag["model_message_count"],
            "approval_policy": str(_RUNTIME_CTX.get("approval_policy", "")),
        }

    @app.post("/api/session/reset")
    async def session_reset():
        if _RUNNER is None:
            return {"error": "会话未就绪"}
        _RUNNER.reset()
        return {"status": "ok"}

    @app.post("/api/session/mode")
    async def session_mode(req: dict[str, str]):
        if _RUNNER is None:
            return {"error": "会话未就绪"}
        try:
            target = AgentMode(req["mode"])
        except (KeyError, ValueError) as exc:
            return {"error": str(exc)}
        _RUNNER.switch_mode(target)
        return {"status": "ok", "mode": _RUNNER.mode.value}

    @app.post("/api/session/approval-policy")
    async def session_approval_policy(req: dict[str, str]):
        try:
            policy = ApprovalPolicy(req["policy"])
        except (KeyError, ValueError) as exc:
            return {"error": str(exc)}
        _RUNTIME_CTX["approval_policy"] = policy
        return {"status": "ok", "policy": policy.value}

    @app.post("/api/session/switch-model")
    async def session_switch_model(req: dict[str, str]):
        if _RUNNER is None:
            return {"error": "会话未就绪"}
        service_name = req.get("service", _RUNNER.service)
        model_name = req.get("model", _RUNNER.model_name)
        _RUNNER.update_llm_config(service_name=service_name, model_name=model_name)
        return {"status": "ok", "service": _RUNNER.service, "model": _RUNNER.model_name}

    # ── 文件系统操作 ──

    @app.get("/api/fs/list")
    async def fs_list(path: str = "", hidden: bool = False):
        target = Path(path).expanduser().resolve() if path else Path.cwd()
        if not _is_path_allowed(target):
            return {"error": "路径不在允许范围内"}
        if not target.exists():
            return {"error": "路径不存在"}
        if not target.is_dir():
            return {"error": "路径不是目录"}
        entries = []
        try:
            for entry in sorted(target.iterdir()):
                name = entry.name
                if not hidden and name.startswith("."):
                    continue
                try:
                    st = entry.stat()
                    entries.append({
                        "name": name,
                        "path": str(entry.resolve()),
                        "type": "dir" if entry.is_dir() else "file",
                        "size": st.st_size if entry.is_file() else None,
                        "modified": st.st_mtime,
                    })
                except PermissionError:
                    entries.append({
                        "name": name, "path": str(entry.resolve()),
                        "type": "unknown", "size": None, "modified": None,
                    })
        except PermissionError:
            return {"error": "无权限访问该目录"}
        return {"entries": entries}

    @app.get("/api/fs/read")
    async def fs_read(path: str = ""):
        target = Path(path).expanduser().resolve()
        if not _is_path_allowed(target):
            return {"error": "路径不在允许范围内"}
        if not target.exists():
            return {"error": "文件不存在"}
        if target.is_dir():
            return {"error": "路径是目录"}
        if target.stat().st_size > 5 * 1024 * 1024:
            try:
                content = target.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return {"error": "二进制文件过大，无法读取"}
            return {"content": content, "size": target.stat().st_size, "encoding": "utf-8", "truncated": True}
        try:
            content = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return {"error": "无法以 UTF-8 解码该文件"}
        return {"content": content, "size": target.stat().st_size, "encoding": "utf-8"}

    @app.post("/api/fs/write")
    async def fs_write(req: dict[str, str]):
        target = Path(req["path"]).expanduser().resolve()
        if not _is_path_allowed(target):
            return {"error": "路径不在允许范围内"}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(req["content"], encoding="utf-8")
        return {"status": "ok"}

    @app.post("/api/fs/create-dir")
    async def fs_create_dir(req: dict[str, str]):
        target = Path(req["path"]).expanduser().resolve()
        if not _is_path_allowed(target):
            return {"error": "路径不在允许范围内"}
        target.mkdir(parents=True, exist_ok=True)
        return {"status": "ok"}

    @app.delete("/api/fs/delete")
    async def fs_delete(path: str = "", recursive: bool = False):
        target = Path(path).expanduser().resolve()
        if not _is_path_allowed(target):
            return {"error": "路径不在允许范围内"}
        if not target.exists():
            return {"error": "路径不存在"}
        if target.is_dir():
            if recursive:
                import shutil
                shutil.rmtree(target)
            else:
                target.rmdir()
        else:
            target.unlink()
        return {"status": "ok"}

    @app.post("/api/fs/rename")
    async def fs_rename(req: dict[str, str]):
        old = Path(req["old_path"]).expanduser().resolve()
        new = Path(req["new_path"]).expanduser().resolve()
        if not _is_path_allowed(old) or not _is_path_allowed(new):
            return {"error": "路径不在允许范围内"}
        old.rename(new)
        return {"status": "ok"}

    # ── Git 操作 ──

    @app.get("/api/git/status")
    async def git_status():
        return await asyncio.get_event_loop().run_in_executor(_GIT_EXECUTOR, _git_status_sync)

    @app.get("/api/git/diff")
    async def git_diff(path: str = "", staged: bool = False):
        return await asyncio.get_event_loop().run_in_executor(
            _GIT_EXECUTOR, _git_diff_sync, path, staged
        )

    @app.post("/api/git/stage")
    async def git_stage(req: dict[str, list[str]]):
        return await asyncio.get_event_loop().run_in_executor(
            _GIT_EXECUTOR, _git_stage_sync, req.get("paths", [])
        )

    @app.post("/api/git/unstage")
    async def git_unstage(req: dict[str, list[str]]):
        return await asyncio.get_event_loop().run_in_executor(
            _GIT_EXECUTOR, _git_unstage_sync, req.get("paths", [])
        )

    @app.post("/api/git/commit")
    async def git_commit(req: dict[str, str]):
        return await asyncio.get_event_loop().run_in_executor(
            _GIT_EXECUTOR, _git_commit_sync, req.get("message", "")
        )

    @app.get("/api/git/log")
    async def git_log(limit: int = 20):
        return await asyncio.get_event_loop().run_in_executor(
            _GIT_EXECUTOR, _git_log_sync, limit
        )

    @app.get("/api/git/branches")
    async def git_branches():
        return await asyncio.get_event_loop().run_in_executor(_GIT_EXECUTOR, _git_branches_sync)

    # ── 配置 ──

    @app.get("/api/config")
    async def get_config():
        return {
            "service": str(_RUNTIME_CTX.get("service_name", "")),
            "model": str(_RUNTIME_CTX.get("model_name", "")),
            "mode": _RUNNER.mode.value if _RUNNER else "standard",
            "approval_policy": str(_RUNTIME_CTX.get("approval_policy", "")),
            "cwd": str(Path.cwd()),
            "base_url": str(_RUNTIME_CTX.get("base_url", "")),
        }

    @app.get("/api/config/providers")
    async def get_providers():
        from ..config import DEFAULT_MODELS
        return {"providers": list(DEFAULT_MODELS.keys()), "models": dict(DEFAULT_MODELS)}

    # ── 终端命令执行 ──

    @app.post("/api/terminal/exec")
    async def terminal_exec(req: dict[str, str]):
        command = req.get("command", "")
        cwd = req.get("cwd", str(Path.cwd()))
        if not command.strip():
            return {"error": "命令为空"}
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=30
            )
            return {
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
                "exit_code": proc.returncode or 0,
            }
        except asyncio.TimeoutError:
            return {"error": "命令执行超时 (30s)"}

    # ── WebSocket ──

    @app.websocket("/ws/chat")
    async def ws_chat(ws: Any):
        await ws.accept()
        bridge = AgentEventBridge()
        run_task: asyncio.Task[Any] | None = None
        loop = asyncio.get_event_loop()
        await ws.send_json({"type": "connected", "session_id": str(_RUNTIME_CTX.get("session_id", ""))})

        async def send_queued_events():
            while True:
                event = await bridge.get_next_event()
                if event is None:
                    break
                # 跳过内部事件，审批请求会在 agent 循环中直接发送
                try:
                    await ws.send_json(event)
                except Exception:
                    break

        try:
            while True:
                try:
                    raw = await asyncio.wait_for(ws.receive_text(), timeout=30)
                except asyncio.TimeoutError:
                    continue
                try:
                    msg: dict[str, object] = json.loads(raw)
                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "message": "无效 JSON"})
                    continue

                msg_type = str(msg.get("type", ""))

                if msg_type == "ping":
                    await ws.send_json({"type": "pong"})

                elif msg_type == "user_message":
                    content = str(msg.get("content", ""))
                    if not content.strip():
                        continue
                    if _RUNNER is None:
                        await ws.send_json({"type": "error", "message": "AgentRunner 未就绪"})
                        continue

                    await ws.send_json({"type": "turn_start", "input": content})
                    bridge._stop_requested = False

                    async def run_and_drain():
                        try:
                            result = await loop.run_in_executor(
                                _AGENT_EXECUTOR, _run_agent_sync, bridge, content
                            )
                        except Exception as exc:
                            await ws.send_json({"type": "error", "message": str(exc)})
                            return
                        # 排空事件队列
                        await send_queued_events()

                    run_task = asyncio.create_task(run_and_drain())
                    # 持续发送队列中的事件直到 agent 结束
                    while run_task is not None and not run_task.done():
                        event = await bridge.get_next_event()
                        if event is not None:
                            try:
                                await ws.send_json(event)
                            except Exception:
                                break
                        else:
                            await asyncio.sleep(0.01)
                    if run_task is not None:
                        await run_task

                elif msg_type == "stop":
                    if _RUNNER is not None and _RUNTIME_CTX.get("execution_controller") is not None:
                        from ..execution_control import ExecutionController
                        ctrl: ExecutionController = _RUNTIME_CTX["execution_controller"]
                        ctrl.request_stop("用户在 IDE 中点击停止")
                    bridge._stop_requested = True
                    await ws.send_json({"type": "stopped"})

                elif msg_type == "approve":
                    call_id = str(msg.get("tool_call_id", ""))
                    bridge.resolve_approval(call_id, True, "用户已批准")
                    await ws.send_json({"type": "approval_result", "tool_call_id": call_id, "approved": True})

                elif msg_type == "reject":
                    call_id = str(msg.get("tool_call_id", ""))
                    reason = str(msg.get("reason", "用户已拒绝"))
                    bridge.resolve_approval(call_id, False, reason)
                    await ws.send_json({"type": "approval_result", "tool_call_id": call_id, "approved": False, "reason": reason})

                # 处理完用户消息后发送所有排队事件
                await send_queued_events()

        except Exception:
            pass
        finally:
            if run_task is not None and not run_task.done():
                run_task.cancel()


# ── Git 同步操作 ──


def _run_git(args: list[str]) -> tuple[str, str, int]:
    try:
        proc = subprocess.run(
            ["git", *args],
            capture_output=True, text=True, timeout=30,
        )
        return proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        return "", "Git 操作超时", -1
    except FileNotFoundError:
        return "", "Git 未安装或不在 PATH 中", -1


def _git_status_sync() -> dict[str, object]:
    branch = ""
    stdout, _, _ = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    branch = stdout.strip()

    staged: list[dict[str, str]] = []
    unstaged: list[dict[str, str]] = []
    untracked: list[str] = []

    stdout, _, _ = _run_git(["status", "--porcelain"])
    for line in stdout.splitlines():
        if not line.strip():
            continue
        status = line[:2]
        filename = line[3:].strip()
        if status[0] in "MRC":
            staged.append({"path": filename, "status": status[0]})
        if status[1] in "MD":
            unstaged.append({"path": filename, "status": status[1]})
        if status == "??":
            untracked.append(filename)

    return {
        "branch": branch,
        "staged": staged,
        "unstaged": unstaged,
        "untracked": untracked,
    }


def _git_diff_sync(path: str, staged: bool) -> dict[str, object]:
    args = ["diff"]
    if staged:
        args.append("--staged")
    if path:
        args.extend(["--", path])
    stdout, stderr, rc = _run_git(args)
    return {"diff": stdout, "error": stderr if rc != 0 else ""}


def _git_stage_sync(paths: list[str]) -> dict[str, object]:
    _, stderr, rc = _run_git(["add", *paths])
    return {"status": "ok" if rc == 0 else "error", "error": stderr}


def _git_unstage_sync(paths: list[str]) -> dict[str, object]:
    _, stderr, rc = _run_git(["reset", "--", *paths])
    return {"status": "ok" if rc == 0 else "error", "error": stderr}


def _git_commit_sync(message: str) -> dict[str, object]:
    if not message.strip():
        return {"error": "提交信息不能为空"}
    _, stderr, rc = _run_git(["commit", "-m", message])
    if rc == 0:
        stdout, _, _ = _run_git(["rev-parse", "HEAD"])
        return {"status": "ok", "commit_hash": stdout.strip()}
    return {"status": "error", "error": stderr}


def _git_log_sync(limit: int) -> dict[str, object]:
    stdout, _, rc = _run_git([
        "log", f"-{max(1, min(limit, 100))}",
        "--format=%H||%s||%an||%ad",
        "--date=short",
    ])
    commits = []
    for line in stdout.splitlines():
        parts = line.split("||", 3)
        if len(parts) == 4:
            commits.append({
                "hash": parts[0][:8], "message": parts[1],
                "author": parts[2], "date": parts[3],
            })
    return {"commits": commits}


def _git_branches_sync() -> dict[str, object]:
    current = ""
    stdo, _, _ = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    current = stdo.strip()
    stdout, _, _ = _run_git(["branch", "--format=%(refname:short)"])
    branches = [b.strip() for b in stdout.splitlines() if b.strip()]
    return {"current": current, "branches": branches}


# ── 服务器启动 ──


def run_ide_server(
    runtime_context: dict[str, object],
    host: str = "127.0.0.1",
    port: int = 0,
) -> None:
    """启动 IDE API 服务器。"""
    global _RUNNER, _RUNTIME_CTX
    _RUNTIME_CTX = runtime_context

    ensure_runtime_capabilities(runtime_context)
    _RUNNER = create_runner(runtime_context)

    app = create_app()

    if port == 0:
        port = _find_free_port()

    print(f"IDE_SERVER_PORT={port}", flush=True)
    print(f"IDE server starting on http://{host}:{port}", file=sys.stderr, flush=True)

    try:
        import uvicorn
    except ImportError:
        print("错误：需要安装 uvicorn。运行 pip install uvicorn[standard]", file=sys.stderr)
        raise

    uvicorn.run(app, host=host, port=port, log_level="warning")
