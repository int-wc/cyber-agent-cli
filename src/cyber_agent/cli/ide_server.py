"""IDE 后端：FastAPI + WebSocket 服务器，桥接 AgentRunner 与 Electron 前端。"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from ..agent.approval import ApprovalDecision, ApprovalPolicy, parse_approval_policy
from ..agent.mode import AgentMode, parse_agent_mode
from ..agent.runner import AgentRunner, extract_text_content
from ..execution_control import ExecutionController
from ..tools.filesystem import (
    MAX_DIRECTORY_ENTRIES,
    MAX_FILE_READ_CHARS,
    normalize_allowed_roots,
    resolve_permitted_path,
)

_RUNNER: AgentRunner | None = None
_RUNTIME_CTX: dict[str, object] = {}
_SERVER_PORT: int = 0
_SERVER_READY: threading.Event = threading.Event()

AGENT_QUEUE_POLL_TIMEOUT = 0.05
WS_PING_INTERVAL = 30
APPROVAL_TIMEOUT = 30.0


class AgentEventBridge:
    """桥接同步 AgentRunner 线程与异步 WebSocket 协程。"""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._pending_approval: asyncio.Future[ApprovalDecision] | None = None
        self._approval_event = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def push_event(self, event: dict[str, Any]) -> None:
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._queue.put_nowait, event)

    async def get_event(self) -> dict[str, Any] | None:
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=AGENT_QUEUE_POLL_TIMEOUT)
        except asyncio.TimeoutError:
            return None

    def create_event_handler(self):
        def handler(event_type: str, payload: object) -> None:
            self.push_event({"type": event_type, "payload": payload})
        return handler

    def create_approval_handler(self):
        def handler(tool, tool_call: dict) -> ApprovalDecision:
            return self._wait_for_approval(tool, tool_call)
        return handler

    def _wait_for_approval(self, tool, tool_call: dict) -> ApprovalDecision:
        loop = self._loop
        if loop is None:
            return ApprovalDecision(False, "后端事件循环未就绪")

        self._approval_event.clear()
        future = asyncio.run_coroutine_threadsafe(
            self._create_approval_future(tool, tool_call), loop
        )
        try:
            approval_future = future.result(timeout=5)
        except TimeoutError:
            return ApprovalDecision(False, "创建审批超时")

        self.push_event({
            "type": "approval_request",
            "payload": {
                "tool_name": getattr(tool, "name", "unknown"),
                "tool_call_id": tool_call.get("id", ""),
                "tool_call": tool_call,
                "risk": getattr(tool, "metadata", {}).get("risk", "unknown") if hasattr(tool, "metadata") else "unknown",
            },
        })

        try:
            return approval_future.result(timeout=APPROVAL_TIMEOUT)
        except TimeoutError:
            return ApprovalDecision(False, "审批超时自动拒绝")

    async def _create_approval_future(self, tool, tool_call: dict) -> asyncio.Future:
        loop = asyncio.get_event_loop()
        f = loop.create_future()
        self._pending_approval = f
        return f

    def resolve_approval(self, decision: ApprovalDecision) -> None:
        if self._pending_approval and not self._pending_approval.done():
            self._pending_approval.set_result(decision)

    def reject_pending_approval(self, reason: str = "已取消") -> None:
        if self._pending_approval and not self._pending_approval.done():
            self._pending_approval.set_result(ApprovalDecision(False, reason))


_bridge: AgentEventBridge | None = None


def _get_runner() -> AgentRunner:
    if _RUNNER is None:
        raise HTTPException(status_code=503, detail="Agent 运行器尚未初始化")
    return _RUNNER


def _get_bridge() -> AgentEventBridge:
    if _bridge is None:
        raise HTTPException(status_code=503, detail="事件桥未初始化")
    return _bridge


def _is_path_allowed(path_str: str) -> bool:
    runner = _RUNNER
    if runner is None:
        return False
    try:
        resolve_permitted_path(path_str, runner.allowed_roots)
        return True
    except (ValueError, FileNotFoundError):
        return False


# ── FastAPI 应用工厂 ────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(title="Cyber Agent IDE Server", version="0.1.0")

    # ── 健康检查 ──

    @app.get("/api/health")
    async def health():
        runner = _RUNNER
        return {
            "status": "ok" if runner else "initializing",
            "mode": runner.mode.value if runner else None,
            "service": runner.service if runner else None,
            "model": runner.model_name if runner else None,
            "session_id": _RUNTIME_CTX.get("session_id", ""),
        }

    # ── 文件系统 ──

    @app.get("/api/fs/list")
    async def fs_list(path: str = Query(default=".")):
        runner = _get_runner()
        try:
            target = resolve_permitted_path(path, runner.allowed_roots)
        except (ValueError, FileNotFoundError):
            raise HTTPException(status_code=403, detail=f"路径访问被拒绝: {path}")
        if not target.is_dir():
            raise HTTPException(status_code=400, detail="路径不是目录")
        entries = []
        try:
            for p in sorted(target.iterdir())[:MAX_DIRECTORY_ENTRIES]:
                try:
                    stat = p.stat()
                    entries.append({
                        "name": p.name, "path": str(p),
                        "is_dir": p.is_dir(),
                        "size": stat.st_size if not p.is_dir() else 0,
                        "modified": stat.st_mtime,
                    })
                except OSError:
                    entries.append({
                        "name": p.name, "path": str(p),
                        "is_dir": p.is_dir(), "size": 0, "modified": 0,
                    })
        except PermissionError:
            raise HTTPException(status_code=403, detail="无权限读取目录")
        return {"path": str(target), "entries": entries}

    @app.get("/api/fs/read")
    async def fs_read(path: str = Query(...)):
        runner = _get_runner()
        try:
            target = resolve_permitted_path(path, runner.allowed_roots)
        except (ValueError, FileNotFoundError):
            raise HTTPException(status_code=403, detail=f"路径访问被拒绝: {path}")
        if not target.is_file():
            raise HTTPException(status_code=400, detail="路径不是文件")
        try:
            content = target.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"读取文件失败: {e}")
        truncated = len(content) > MAX_FILE_READ_CHARS
        if truncated:
            content = content[:MAX_FILE_READ_CHARS]
        return {
            "path": str(target), "content": content,
            "size": target.stat().st_size, "truncated": truncated,
        }

    @app.post("/api/fs/write")
    async def fs_write(req: dict):
        path = req.get("path", "")
        content = req.get("content", "")
        runner = _get_runner()
        target = Path(path).expanduser().resolve()
        for root in runner.allowed_roots:
            try:
                target.relative_to(root)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                return {"path": str(target), "written": True}
            except ValueError:
                continue
        raise HTTPException(status_code=403, detail=f"路径访问被拒绝: {path}")

    @app.post("/api/fs/create-dir")
    async def fs_create_dir(req: dict):
        path = req.get("path", "")
        runner = _get_runner()
        target = Path(path).expanduser().resolve()
        for root in runner.allowed_roots:
            try:
                target.relative_to(root)
                target.mkdir(parents=True, exist_ok=True)
                return {"path": str(target), "created": True}
            except ValueError:
                continue
        raise HTTPException(status_code=403, detail=f"路径访问被拒绝: {path}")

    @app.delete("/api/fs/delete")
    async def fs_delete(path: str = Query(...)):
        runner = _get_runner()
        try:
            target = resolve_permitted_path(path, runner.allowed_roots)
        except (ValueError, FileNotFoundError):
            raise HTTPException(status_code=403, detail=f"路径访问被拒绝: {path}")
        try:
            if target.is_dir():
                import shutil
                shutil.rmtree(target)
            else:
                target.unlink()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"删除失败: {e}")
        return {"path": str(target), "deleted": True}

    @app.post("/api/fs/rename")
    async def fs_rename(req: dict):
        old_path = req.get("old_path", "")
        new_path = req.get("new_path", "")
        runner = _get_runner()
        try:
            src = resolve_permitted_path(old_path, runner.allowed_roots)
        except (ValueError, FileNotFoundError):
            raise HTTPException(status_code=403, detail=f"源路径访问被拒绝: {old_path}")
        dst = Path(new_path).expanduser().resolve()
        allowed = False
        for root in runner.allowed_roots:
            try:
                dst.relative_to(root)
                allowed = True
                break
            except ValueError:
                continue
        if not allowed:
            raise HTTPException(status_code=403, detail=f"目标路径访问被拒绝: {new_path}")
        try:
            src.rename(dst)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"重命名失败: {e}")
        return {"old_path": str(src), "new_path": str(dst), "renamed": True}

    @app.get("/api/fs/search")
    async def fs_search(query: str = Query(...), path: str = Query(default=".")):
        runner = _get_runner()
        try:
            base = resolve_permitted_path(path, runner.allowed_roots)
        except (ValueError, FileNotFoundError):
            raise HTTPException(status_code=403, detail=f"路径访问被拒绝: {path}")
        results = []
        try:
            for p in base.rglob("*"):
                if p.is_file() and p.suffix not in (".pyc", ".pyo", ".so", ".o", ".class"):
                    try:
                        if query.lower() in p.name.lower():
                            results.append({"path": str(p), "name": p.name, "match": "filename"})
                        if len(results) >= 50:
                            break
                    except OSError:
                        continue
        except PermissionError:
            pass
        return {"query": query, "results": results[:50]}

    # ── Git ──

    def _run_git(args: list[str], cwd: Path | None = None) -> dict:
        try:
            result = subprocess.run(
                ["git"] + args,
                capture_output=True, text=True, timeout=30,
                cwd=str(cwd) if cwd else None,
            )
            return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}
        except Exception as e:
            return {"stdout": "", "stderr": str(e), "returncode": -1}

    @app.get("/api/git/status")
    async def git_status():
        return _run_git(["status", "--porcelain"], cwd=Path.cwd())

    @app.get("/api/git/diff")
    async def git_diff(staged: bool = Query(default=False)):
        args = ["diff"];
        if staged: args.append("--staged")
        return _run_git(args, cwd=Path.cwd())

    @app.post("/api/git/stage")
    async def git_stage(req: dict):
        files = req.get("files", ["."])
        return _run_git(["add"] + files, cwd=Path.cwd())

    @app.post("/api/git/unstage")
    async def git_unstage(req: dict):
        files = req.get("files", ["."])
        return _run_git(["reset", "HEAD"] + files, cwd=Path.cwd())

    @app.post("/api/git/commit")
    async def git_commit(req: dict):
        message = req.get("message", "")
        if not message:
            raise HTTPException(status_code=400, detail="commit message 不能为空")
        return _run_git(["commit", "-m", message], cwd=Path.cwd())

    @app.get("/api/git/log")
    async def git_log(max_count: int = Query(default=50)):
        return _run_git(["log", "--oneline", f"-n{max_count}"], cwd=Path.cwd())

    @app.get("/api/git/branches")
    async def git_branches():
        return _run_git(["branch", "-a"], cwd=Path.cwd())

    @app.post("/api/git/branch-create")
    async def git_branch_create(req: dict):
        name = req.get("name", "")
        if not name: raise HTTPException(status_code=400, detail="分支名不能为空")
        return _run_git(["checkout", "-b", name], cwd=Path.cwd())

    @app.post("/api/git/branch-checkout")
    async def git_branch_checkout(req: dict):
        name = req.get("name", "")
        if not name: raise HTTPException(status_code=400, detail="分支名不能为空")
        return _run_git(["checkout", name], cwd=Path.cwd())

    # ── 会话管理 ──

    @app.get("/api/session/status")
    async def session_status():
        runner = _get_runner()
        return {
            "mode": runner.mode.value,
            "service": runner.service,
            "model": runner.model_name,
            "allowed_roots": [str(r) for r in runner.allowed_roots],
            "message_count": len(runner.history),
            "session_id": _RUNTIME_CTX.get("session_id", ""),
        }

    @app.post("/api/session/reset")
    async def session_reset():
        _get_runner().reset()
        return {"status": "ok", "message": "会话上下文已重置"}

    @app.post("/api/session/mode")
    async def session_mode(req: dict):
        mode_str = req.get("mode", "standard")
        runner = _get_runner()
        try:
            new_mode = parse_agent_mode(mode_str)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        runner.switch_mode(new_mode)
        return {"mode": runner.mode.value}

    @app.post("/api/session/approval-policy")
    async def session_approval_policy(req: dict):
        policy_str = req.get("policy", "prompt")
        try:
            parse_approval_policy(policy_str)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"policy": policy_str}

    @app.post("/api/session/switch-model")
    async def session_switch_model(req: dict):
        service_name = req.get("service", "")
        model_name = req.get("model", "")
        runner = _get_runner()
        if service_name: runner.service = service_name
        if model_name: runner.model_name = model_name
        runner.update_llm_config(service=runner.service, model=runner.model_name)
        return {"service": runner.service, "model": runner.model_name}

    @app.post("/api/session/stop")
    async def session_stop():
        _get_runner().execution_controller.request_stop("IDE 用户点击停止")
        _get_bridge().reject_pending_approval("已停止")
        return {"status": "stopped"}

    # ── 配置 ──

    @app.get("/api/config")
    async def config_get():
        from ..config import settings
        return {
            "service": settings.get_service(),
            "model": settings.get_model_name(),
            "base_url": settings.resolve_base_url(),
        }

    @app.get("/api/config/providers")
    async def config_providers():
        from ..config import settings
        providers = getattr(settings, "provider_configs", {})
        return {"providers": list(providers.keys()) if providers else ["openai", "deepseek", "claude", "mimo"]}

    # ── 工具列表 ──

    @app.get("/api/tools")
    async def tools_list():
        runner = _get_runner()
        return {
            "tools": [
                {
                    "name": getattr(t, "name", str(t)),
                    "description": getattr(t, "description", ""),
                    "risk": getattr(t, "metadata", {}).get("risk", "unknown") if hasattr(t, "metadata") else "unknown",
                }
                for t in runner.tools
            ]
        }

    # ── 历史会话 ──

    @app.get("/api/history/list")
    async def history_list():
        from ..session_store import list_stored_sessions
        sessions = list_stored_sessions()
        return {
            "sessions": [
                {"session_id": s.session_id, "title": s.title,
                 "created_at": s.created_at, "updated_at": s.updated_at,
                 "mode": s.mode, "turn_count": s.turn_count}
                for s in sessions
            ]
        }

    # ── WebSocket ──

    @app.websocket("/ws/chat")
    async def ws_chat(ws: WebSocket):
        global _bridge
        await ws.accept()

        bridge = _get_bridge()
        bridge.set_loop(asyncio.get_event_loop())

        runner = _get_runner()
        await ws.send_json({
            "type": "connected",
            "session_id": _RUNTIME_CTX.get("session_id", ""),
            "mode": runner.mode.value,
            "service": runner.service,
            "model": runner.model_name,
        })

        agent_task: asyncio.Task | None = None
        ping_task = asyncio.create_task(_ws_ping_loop(ws))

        try:
            while True:
                # 从 WebSocket 读取客户端消息
                try:
                    raw = await asyncio.wait_for(ws.receive_text(), timeout=AGENT_QUEUE_POLL_TIMEOUT)
                except asyncio.TimeoutError:
                    pass
                else:
                    msg = json.loads(raw)
                    msg_type = msg.get("type", "")

                    if msg_type == "user_message":
                        content = msg.get("content", "")
                        if content.strip():
                            if agent_task and not agent_task.done():
                                await ws.send_json({"type": "error", "payload": {"message": "已有任务正在运行"}})
                            else:
                                agent_task = asyncio.create_task(
                                    _run_agent_in_thread(runner, bridge, content)
                                )
                    elif msg_type == "stop":
                        runner.execution_controller.request_stop("前端用户点击停止")
                        bridge.reject_pending_approval("用户停止")
                    elif msg_type == "approve":
                        bridge.resolve_approval(ApprovalDecision(True, "用户批准"))
                    elif msg_type == "reject":
                        bridge.resolve_approval(ApprovalDecision(False, msg.get("reason", "用户拒绝")))
                    elif msg_type == "ping":
                        await ws.send_json({"type": "pong"})

                # 发送事件队列中的待处理事件
                event = await bridge.get_event()
                if event is not None:
                    try:
                        await ws.send_json(event)
                    except Exception:
                        break

                if agent_task and agent_task.done():
                    try:
                        await agent_task
                    except Exception:
                        pass
                    agent_task = None

        except WebSocketDisconnect:
            pass
        except Exception:
            traceback.print_exc()
        finally:
            ping_task.cancel()
            if agent_task and not agent_task.done():
                agent_task.cancel()
            bridge.reject_pending_approval("连接断开")
            try:
                await ws.close()
            except Exception:
                pass

    return app


async def _ws_ping_loop(ws: WebSocket) -> None:
    while True:
        await asyncio.sleep(WS_PING_INTERVAL)
        try:
            await ws.send_json({"type": "pong"})
        except Exception:
            break


async def _run_agent_in_thread(runner: AgentRunner, bridge: AgentEventBridge, user_input: str) -> None:
    event_handler = bridge.create_event_handler()
    approval_handler = bridge.create_approval_handler()

    def _run():
        try:
            result = runner.run(
                user_input, verbose=False,
                event_handler=event_handler,
                approval_handler=approval_handler,
            )
            bridge.push_event({"type": "run_complete", "payload": {"content": result}})
        except Exception as e:
            bridge.push_event({"type": "error", "payload": {"message": str(e)}})

    await asyncio.get_event_loop().run_in_executor(None, _run)


# ── 初始化函数（供 ide_launcher 内联调用） ──

def build_ide_runtime_context(
    mode: str = "standard",
    allow_paths: list[str] | None = None,
    approval_policy: str = "prompt",
    service_name: str | None = None,
    model_name: str | None = None,
) -> dict[str, object]:
    """构建 IDE 运行上下文（独立于 CLI flags）。"""
    from ..config import settings
    from datetime import datetime
    from ..session_store import create_session_id

    agent_mode = parse_agent_mode(mode)
    policy = parse_approval_policy(approval_policy)

    resolved_service = service_name or settings.get_service()
    resolved_model = model_name or settings.get_model_name(service_name=resolved_service)

    # IDE 模式下将 OS 根目录加入 allowed_roots，确保文件树可浏览整个磁盘
    allowed_roots = [Path.cwd().resolve()]
    if sys.platform == "win32":
        for drive in ("C:\\", "D:\\", "E:\\"):
            p = Path(drive)
            if p.exists():
                allowed_roots.append(p)
    else:
        allowed_roots.append(Path("/"))
    allowed_roots = normalize_allowed_roots(allowed_roots)
    extra_paths = [Path(p).expanduser() for p in (allow_paths or [])]
    if agent_mode is AgentMode.AUTHORIZED and extra_paths:
        allowed_roots = normalize_allowed_roots(allowed_roots + extra_paths)

    execution_controller = ExecutionController()

    return {
        "mode": agent_mode,
        "extra_allowed_paths": extra_paths,
        "allowed_roots": allowed_roots,
        "configured_registry": {},
        "command_registry": {} if agent_mode == AgentMode.AUTHORIZED else {},
        "tools": [],
        "approval_policy": policy,
        "execution_controller": execution_controller,
        "capability_registry": None,
        "runtime_capabilities_loaded": False,
        "file_skills": [],
        "service_name": resolved_service,
        "model_name": resolved_model,
        "base_url": settings.resolve_base_url(resolved_service),
        "api_key": settings.get_api_key(resolved_service),
        "session_id": create_session_id(datetime.now().astimezone()),
        "session_source_id": None,
    }


def init_runner(runtime_context: dict[str, object]) -> AgentRunner:
    """从 IDE 运行上下文初始化 AgentRunner（内部使用，返回 runner）。"""
    global _RUNNER, _RUNTIME_CTX, _bridge

    from ..agent.runner import AgentRunner as AR
    from ..capability_registry import CapabilityRegistry

    _RUNTIME_CTX = runtime_context

    cap_registry = CapabilityRegistry(
        execution_controller=runtime_context["execution_controller"],
        service_name=str(runtime_context["service_name"]),
        model_name=str(runtime_context["model_name"]),
        api_key=str(runtime_context["api_key"]),
        base_url=str(runtime_context.get("base_url", "")),
    )

    from ..tools import get_default_tools
    tools = get_default_tools(
        runtime_context["mode"],
        runtime_context["extra_allowed_paths"],
        runtime_context["configured_registry"],
        runtime_context["execution_controller"],
        cap_registry,
    )

    runtime_context["tools"] = tools
    runtime_context["capability_registry"] = cap_registry
    runtime_context["runtime_capabilities_loaded"] = True

    runner = AR(
        tools,
        mode=runtime_context["mode"],
        allowed_roots=runtime_context["allowed_roots"],
        command_registry=runtime_context["command_registry"],
        extra_allowed_paths=runtime_context["extra_allowed_paths"],
        configured_registry=runtime_context["configured_registry"],
        execution_controller=runtime_context["execution_controller"],
        capability_registry=cap_registry,
        file_skills=runtime_context.get("file_skills", []),
        service_name=runtime_context["service_name"],
        model_name=runtime_context["model_name"],
        api_key=runtime_context["api_key"],
        base_url=runtime_context["base_url"],
    )

    cap_registry.register_refresh_callback(lambda: runner._refresh_runtime_scope())

    _RUNNER = runner
    _bridge = AgentEventBridge()

    return runner


# ── 启动入口（供独立子进程使用 / python -m cyber_agent.cli.ide_server） ──

def run_ide_server(
    host: str = "127.0.0.1",
    port: int = 0,
    mode: str = "standard",
    allow_paths: list[str] | None = None,
    approval_policy: str = "prompt",
    service_name: str | None = None,
    model_name: str | None = None,
) -> None:
    """启动 IDE 后端服务器（阻塞，供子进程入口使用）。"""
    import uvicorn

    runtime_context = build_ide_runtime_context(
        mode=mode, allow_paths=allow_paths,
        approval_policy=approval_policy,
        service_name=service_name, model_name=model_name,
    )
    init_runner(runtime_context)

    app = create_app()

    if port == 0:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

    global _SERVER_PORT
    _SERVER_PORT = port

    print(f"IDE_SERVER_PORT={port}", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="info")
