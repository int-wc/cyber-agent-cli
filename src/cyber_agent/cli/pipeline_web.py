"""四柱管线 Web 仪表盘：FastAPI + SSE 实时可视化。

提供 REST API 和 SSE 实时事件推送，前端用嵌入式单页 HTML
以「机器人领任务→思考→接取→传递」的生动形象展示 10 个角色的
任务分配流程。
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from ..agent.roles import AgentRole, get_role_label

# ── 常量 ──

PIPELINE_TRACE_DIR = Path.home() / ".cyber-agent-cli-traces"
SSE_KEEPALIVE_INTERVAL = 15
MAX_TRACE_EVENTS = 5000
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8318

# ── 角色元数据 ──

PHASE1_ROLES: list[dict[str, Any]] = [
    {
        "id": "analyst", "label": "分析者", "en_name": "ANALYST",
        "motto": "分析为底", "desc": "对任务进行多维度深度分析",
        "color": "#7C3AED", "icon": "🔍",
    },
    {
        "id": "diffuser", "label": "扩散者", "en_name": "DIFFUSER",
        "motto": "扩展为路", "desc": "探索一切可能的路径",
        "color": "#2563EB", "icon": "🌊",
    },
    {
        "id": "jumper", "label": "迁跃者", "en_name": "JUMPER",
        "motto": "迁跃为辅", "desc": "不受常规约束，创造性跨越",
        "color": "#D97706", "icon": "🚀",
    },
    {
        "id": "reflector_p1", "label": "反思者", "en_name": "REFLECTOR",
        "motto": "反思为主", "desc": "综合审视全部输出，制定执行计划",
        "color": "#059669", "icon": "🪞",
    },
]

PHASE2_ROLES: list[dict[str, Any]] = [
    {
        "id": "decision_maker", "label": "决策者", "en_name": "DECISION_MAKER",
        "motto": "分解子任务", "desc": "将执行计划分解为可操作子任务",
        "color": "#DC2626", "icon": "🎯",
    },
    {
        "id": "thinker", "label": "思考者", "en_name": "THINKER",
        "motto": "评估选择", "desc": "分析子任务合理性，做出选择决策",
        "color": "#7C3AED", "icon": "💭",
    },
    {
        "id": "runner", "label": "执行者", "en_name": "RUNNER",
        "motto": "直接执行", "desc": "调用工具直接完成任务",
        "color": "#2563EB", "icon": "⚡",
    },
    {
        "id": "reader", "label": "阅读者", "en_name": "READER",
        "motto": "信息获取", "desc": "获取并提取文件、目录、网页信息",
        "color": "#0891B2", "icon": "📖",
    },
    {
        "id": "builder", "label": "构建者", "en_name": "BUILDER",
        "motto": "方案落地", "desc": "创建目录、编写代码、生成配置",
        "color": "#D97706", "icon": "🏗️",
    },
    {
        "id": "checker", "label": "审计者", "en_name": "CHECKER",
        "motto": "验证质量", "desc": "逐项检查执行结果，发现遗漏错误",
        "color": "#059669", "icon": "✅",
    },
    {
        "id": "reflector_p2", "label": "反思者", "en_name": "REFLECTOR",
        "motto": "闭环判定", "desc": "审视结果，决定继续迭代或结束",
        "color": "#059669", "icon": "🪞",
    },
]

_ALL_ROLES: dict[str, dict[str, Any]] = {}
for r in PHASE1_ROLES:
    _ALL_ROLES[r["id"]] = r
for r in PHASE2_ROLES:
    _ALL_ROLES[r["id"]] = r

# ── SSE 事件管道 ──

_sse_clients: list[asyncio.Queue] = []
_sse_lock = threading.Lock()
_last_pipeline_state: dict[str, Any] | None = None


def broadcast_pipeline_event(event_type: str, data: dict[str, Any]) -> None:
    global _last_pipeline_state
    payload = json.dumps({"type": event_type, "data": data, "timestamp": time.time()})
    if event_type == "state_update":
        _last_pipeline_state = data
    with _sse_lock:
        dead: list[asyncio.Queue] = []
        for q in _sse_clients:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            _sse_clients.remove(q)


def clear_sse_clients() -> None:
    with _sse_lock:
        _sse_clients.clear()


# ═══════════════════════════════════════════════════════════════
# FastAPI 应用
# ═══════════════════════════════════════════════════════════════

def create_app() -> FastAPI:
    app = FastAPI(title="四柱管线仪表盘", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return _render_dashboard()

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        return _render_dashboard()

    @app.get("/api/health")
    async def health():
        return {
            "status": "ok",
            "time": datetime.now().isoformat(),
            "trace_dir": str(PIPELINE_TRACE_DIR),
            "trace_count": len(list(PIPELINE_TRACE_DIR.glob("*.trace.json"))) if PIPELINE_TRACE_DIR.exists() else 0,
        }

    @app.get("/api/roles")
    async def get_roles():
        return {"phase1": PHASE1_ROLES, "phase2": PHASE2_ROLES, "all_roles": _ALL_ROLES}

    @app.get("/api/traces")
    async def list_traces():
        if not PIPELINE_TRACE_DIR.exists():
            return {"traces": []}
        files = sorted(PIPELINE_TRACE_DIR.glob("*.trace.json"), reverse=True)[:50]
        traces = []
        for f in files:
            try:
                stat = f.stat()
                data = json.loads(f.read_text(encoding="utf-8"))
                events = data if isinstance(data, list) else data.get("events", [])
                summary = {"角色": 0, "工具调用": 0, "子任务": 0, "迭代": 0}
                roles_seen: set[str] = set()
                for ev in events:
                    e = ev.get("event", "")
                    if e.startswith("role_"):
                        summary["角色"] += 1
                        roles_seen.add(e.replace("role_", "", 1))
                    elif e == "tool_call":
                        summary["工具调用"] += 1
                    elif e == "subtask_start":
                        summary["子任务"] += 1
                    elif e == "iteration_start":
                        summary["迭代"] += 1
                traces.append({
                    "session_id": f.stem.replace(".trace", ""),
                    "time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "events": len(events),
                    "summary": summary,
                    "roles": list(roles_seen),
                    "size": stat.st_size,
                })
            except Exception:
                continue
        return {"traces": traces}

    @app.get("/api/traces/{session_id}")
    async def get_trace(session_id: str):
        trace_file = PIPELINE_TRACE_DIR / f"{session_id}.trace.json"
        if not trace_file.exists():
            trace_file = PIPELINE_TRACE_DIR / f"{session_id.replace('.trace', '')}.trace.json"
        if not trace_file.exists():
            raise HTTPException(status_code=404, detail=f"轨迹文件不存在：{session_id}")
        try:
            data = json.loads(trace_file.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"读取轨迹失败：{exc}")
        events = data if isinstance(data, list) else data.get("events", [])
        return {
            "session_id": session_id,
            "events": events[:MAX_TRACE_EVENTS],
            "total_events": len(events),
            "truncated": len(events) > MAX_TRACE_EVENTS,
        }

    @app.get("/api/state")
    async def get_current_state():
        global _last_pipeline_state
        if _last_pipeline_state:
            return {"running": True, "state": _last_pipeline_state}
        return {"running": False, "state": None}

    @app.get("/api/events")
    async def event_stream(request: Request):
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        with _sse_lock:
            _sse_clients.append(queue)

        async def _generate():
            try:
                if _last_pipeline_state:
                    yield f"data: {json.dumps({'type': 'state_update', 'data': _last_pipeline_state, 'timestamp': time.time()})}\n\n"
                yield f"data: {json.dumps({'type': 'connected', 'data': {'time': datetime.now().isoformat()}})}\n\n"
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        payload = await asyncio.wait_for(queue.get(), timeout=SSE_KEEPALIVE_INTERVAL)
                        yield f"data: {payload}\n\n"
                    except asyncio.TimeoutError:
                        yield f": keepalive\n\n"
            finally:
                with _sse_lock:
                    if queue in _sse_clients:
                        _sse_clients.remove(queue)

        from fastapi.responses import StreamingResponse
        return StreamingResponse(
            _generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    return app


def _render_dashboard() -> str:
    return HTML_DASHBOARD


def run_pipeline_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    import uvicorn
    if port == 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, 0))
            port = s.getsockname()[1]
    print(f"🌐 四柱管线仪表盘 → http://{host}:{port}")
    print(f"   SSE 实时事件 → http://{host}:{port}/api/events")
    print(f"   REST API     → http://{host}:{port}/api/health")
    print(f"   轨迹目录     → {PIPELINE_TRACE_DIR}")
    print()
    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="info")


def push_trace_to_dashboard(trace_events: list[dict[str, Any]]) -> None:
    for ev in trace_events:
        broadcast_pipeline_event("pipeline_event", ev)


# ═══════════════════════════════════════════════════════════════
# 仪表盘 HTML — 机器人领任务 → 思考 → 接取 → 传递 动画
# ═══════════════════════════════════════════════════════════════

HTML_DASHBOARD = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>四柱管线 · 智能体任务分配</title>
<style>
  :root {
    --bg: #0a0c14;
    --bg2: #111520;
    --card-bg: #161b2b;
    --card-border: #252b3d;
    --text: #d6dae5;
    --text-dim: #7a7f94;
    --text-bright: #f0f2f8;
    --success: #22c55e;
    --error: #ef4444;
    --warn: #f59e0b;
    --running: #3b82f6;
    --thinking-color: #f59e0b;
    --p1: #7C3AED;
    --p2: #DC2626;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    line-height: 1.5;
    overflow-x: hidden;
  }
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-thumb { background: var(--card-border); border-radius: 2px; }

  /* ── 顶栏 ── */
  .header {
    background: linear-gradient(135deg, #111520 0%, #0a0c14 100%);
    border-bottom: 1px solid var(--card-border);
    padding: 16px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 10px;
  }
  .header-left { display: flex; align-items: center; gap: 14px; }
  .header-logo {
    font-size: 28px;
    line-height: 1;
    animation: float 3s ease-in-out infinite;
  }
  @keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-4px)} }
  .header h1 { font-size: 20px; font-weight: 700; }
  .header h1 span:nth-child(1) { color: var(--p1); }
  .header h1 span:nth-child(2) { color: var(--running); }
  .header h1 span:nth-child(3) { color: var(--success); }
  .header .subtitle { font-size: 12px; color: var(--text-dim); }
  .header-right { display: flex; gap: 10px; align-items: center; }
  .conn-indicator {
    display: flex; align-items: center; gap: 6px;
    font-size: 12px; color: var(--text-dim);
  }
  .conn-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--text-dim); transition: all 0.3s;
  }
  .conn-dot.on { background: var(--success); box-shadow: 0 0 10px rgba(34,197,94,0.5); }
  .conn-dot.busy { background: var(--running); box-shadow: 0 0 10px rgba(59,130,246,0.5); animation: pulse-dot 1.2s infinite; }
  @keyframes pulse-dot { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.5;transform:scale(1.2)} }
  .btn {
    padding: 5px 14px; border: 1px solid var(--card-border); border-radius: 6px;
    background: var(--card-bg); color: var(--text); cursor: pointer;
    font-size: 12px; transition: all 0.2s;
  }
  .btn:hover { border-color: var(--p1); background: rgba(124,58,237,0.12); }

  .container { max-width: 1440px; margin: 0 auto; padding: 20px 28px 64px; }

  /* ── 统计条 ── */
  .stats-bar {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 10px; margin-bottom: 20px;
  }
  .stat-pill {
    background: var(--card-bg); border: 1px solid var(--card-border);
    border-radius: 10px; padding: 12px 16px;
    display: flex; align-items: center; gap: 12px;
    transition: border-color 0.3s;
  }
  .stat-pill:hover { border-color: rgba(124,58,237,0.3); }
  .stat-pill .s-icon { font-size: 20px; }
  .stat-pill .s-num { font-size: 20px; font-weight: 700; color: var(--text-bright); line-height: 1.2; }
  .stat-pill .s-lbl { font-size: 11px; color: var(--text-dim); }

  /* ── 轨迹选择 ── */
  .trace-bar {
    display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0 16px;
    padding: 10px 14px; background: var(--card-bg); border: 1px solid var(--card-border);
    border-radius: 10px; align-items: center;
  }
  .trace-bar .tb-label { font-size: 11px; color: var(--text-dim); margin-right: 6px; white-space: nowrap; }
  .trace-btn {
    padding: 3px 10px; border: 1px solid var(--card-border); border-radius: 5px;
    background: transparent; color: var(--text-dim); cursor: pointer;
    font-size: 11px; transition: all 0.2s;
  }
  .trace-btn:hover { border-color: var(--p1); color: var(--text); }
  .trace-btn.cur { background: rgba(124,58,237,0.15); border-color: var(--p1); color: #a78bfa; }

  /* ── 阶段标题 ── */
  .section-head {
    display: flex; align-items: center; gap: 10px;
    margin: 24px 0 14px; position: relative;
  }
  .section-head .badge {
    padding: 3px 12px; border-radius: 14px; font-size: 11px;
    font-weight: 600; letter-spacing: 0.3px;
  }
  .badge.p1 { background: rgba(124,58,237,0.18); color: #a78bfa; border: 1px solid rgba(124,58,237,0.25); }
  .badge.p2 { background: rgba(220,38,38,0.18); color: #fca5a5; border: 1px solid rgba(220,38,38,0.25); }
  .section-head h2 { font-size: 16px; font-weight: 600; }
  .section-head .hint { font-size: 12px; color: var(--text-dim); }
  .section-head .line { flex:1; height:1px; background: linear-gradient(90deg, var(--card-border), transparent); }

  /* ── 角色轨道 ── */
  .pipeline-track {
    display: flex; align-items: stretch; gap: 0;
    margin-bottom: 14px; flex-wrap: wrap;
    position: relative;
  }

  /* ── 智能体角色卡 ── */
  .agent {
    flex: 1; min-width: 140px; max-width: 260px;
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 14px;
    padding: 14px 14px 12px;
    position: relative;
    transition: all 0.35s cubic-bezier(0.34, 1.56, 0.64, 1);
    cursor: default;
    overflow: hidden;
  }
  .agent::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: var(--card-border);
    transition: background 0.4s;
  }
  .agent:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
  }

  /* ── 机器人头像 ── */
  .agent-avatar {
    width: 52px; height: 52px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 24px;
    margin-bottom: 8px;
    position: relative;
    transition: all 0.3s;
    border: 2px solid var(--card-border);
    background: rgba(0,0,0,0.25);
  }
  .agent-avatar .ring {
    position: absolute; inset: -4px;
    border-radius: 50%;
    border: 2px solid transparent;
    transition: all 0.4s;
  }
  .agent.thinking .agent-avatar { animation: bob 1s ease-in-out infinite; }
  .agent.thinking .agent-avatar .ring {
    border-color: var(--thinking-color);
    box-shadow: 0 0 16px rgba(245,158,11,0.3);
    animation: spin-ring 2s linear infinite;
  }
  .agent.running .agent-avatar .ring {
    border-color: var(--running);
    box-shadow: 0 0 16px rgba(59,130,246,0.3);
    animation: spin-ring 1.5s linear infinite;
  }
  .agent.done .agent-avatar .ring {
    border-color: var(--success);
    box-shadow: 0 0 12px rgba(34,197,94,0.3);
  }
  .agent.failed .agent-avatar .ring {
    border-color: var(--error);
    box-shadow: 0 0 12px rgba(239,68,68,0.3);
  }
  @keyframes bob { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-6px)} }
  @keyframes spin-ring { to { transform: rotate(360deg); } }

  .agent .agent-row1 {
    display: flex; align-items: center; gap: 10px; margin-bottom: 6px;
  }
  .agent .agent-info { flex: 1; min-width: 0; }
  .agent .agent-name { font-size: 14px; font-weight: 600; color: var(--text-bright); }
  .agent .agent-en { font-size: 9px; color: var(--text-dim); letter-spacing: 0.5px; text-transform: uppercase; }
  .agent .agent-motto {
    display: inline-block;
    font-size: 10px; font-weight: 500; padding: 1px 8px; border-radius: 4px;
    margin: 2px 0 4px;
  }
  .agent .agent-desc { font-size: 11px; color: var(--text-dim); line-height: 1.3; }
  .agent .status-tag {
    font-size: 10px; padding: 2px 8px; border-radius: 8px;
    font-weight: 500; white-space: nowrap;
    transition: all 0.3s;
    flex-shrink: 0;
  }
  .status-tag.idle { background: rgba(122,127,148,0.12); color: var(--text-dim); }
  .status-tag.thinking { background: rgba(245,158,11,0.18); color: #fbbf24; }
  .status-tag.running { background: rgba(59,130,246,0.18); color: #60a5fa; }
  .status-tag.done { background: rgba(34,197,94,0.18); color: #4ade80; }
  .status-tag.failed { background: rgba(239,68,68,0.18); color: #f87171; }

  /* ── 思考气泡 ── */
  .thought-bubble {
    display: none;
    margin-top: 8px; padding: 8px 10px;
    background: rgba(0,0,0,0.35);
    border-radius: 8px;
    font-size: 10px; color: var(--text-dim);
    max-height: 48px; overflow-y: auto; line-height: 1.4;
    border-left: 2px solid var(--thinking-color);
  }
  .thought-bubble.show { display: block; animation: fadeIn 0.3s; }
  @keyframes fadeIn { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:translateY(0)} }

  /* ── 工具调用标签 ── */
  .tool-chip {
    display: inline-block;
    font-family: monospace; font-size: 9px;
    padding: 1px 6px; border-radius: 3px;
    background: rgba(59,130,246,0.15); color: #60a5fa;
    margin: 2px 2px 0 0;
  }

  /* ── 任务飞行物 ── */
  .task-fly {
    position: fixed;
    z-index: 999;
    pointer-events: none;
    font-size: 14px;
    padding: 4px 10px;
    background: rgba(124,58,237,0.85);
    border-radius: 8px;
    color: #fff;
    box-shadow: 0 4px 20px rgba(124,58,237,0.4);
    white-space: nowrap;
    transition: opacity 0.3s;
  }
  .task-fly .fly-icon { margin-right: 4px; }

  /* ── 连接箭头 ── */
  .pipe-arrow {
    display: flex; align-items: center; justify-content: center;
    min-width: 28px; flex-shrink: 0;
    position: relative;
  }
  .pipe-arrow .pa-body {
    width: 20px; height: 2px;
    background: var(--card-border);
    position: relative; transition: all 0.3s;
  }
  .pipe-arrow .pa-head {
    position: absolute; right: -4px; top: -3px;
    width:0; height:0;
    border-left: 6px solid var(--card-border);
    border-top: 4px solid transparent;
    border-bottom: 4px solid transparent;
    transition: border-left-color 0.3s;
  }
  .pipe-arrow.active .pa-body {
    background: var(--p1); height: 3px;
    box-shadow: 0 0 10px rgba(124,58,237,0.4);
  }
  .pipe-arrow.active .pa-head { border-left-color: var(--p1); }
  .pipe-arrow .pa-label {
    position: absolute; top: -14px; left: 50%; transform: translateX(-50%);
    font-size: 8px; color: var(--text-dim); white-space: nowrap;
  }

  /* ── P2 布线（用 flex wrap 自然排列，加回路指示） ── */
  .p2-track {
    display: flex; align-items: stretch; gap: 0;
    flex-wrap: wrap; justify-content: center;
    position: relative;
  }
  .p2-track .agent { max-width: 200px; min-width: 120px; }

  /* ── 循环指示器 ── */
  .loop-ring {
    display: flex; align-items: center; gap: 8px;
    padding: 6px 16px;
    background: rgba(5,150,105,0.06);
    border: 1px dashed rgba(5,150,105,0.25);
    border-radius: 20px;
    margin: 8px auto;
    font-size: 11px; color: #6ee7b7;
  }
  .loop-ring .lr-icon { animation: spin-ring 3s linear infinite; display: inline-block; }

  /* ── 时间线 ── */
  .timeline-wrap {
    margin-top: 28px;
    background: var(--card-bg); border: 1px solid var(--card-border);
    border-radius: 12px; overflow: hidden;
  }
  .tl-header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 12px 18px; border-bottom: 1px solid var(--card-border);
    cursor: pointer;
  }
  .tl-header h3 { font-size: 14px; font-weight: 600; }
  .tl-header .tl-count { font-size: 11px; color: var(--text-dim); }
  .tl-body {
    max-height: 360px; overflow-y: auto; padding: 4px 0;
  }
  .tl-row {
    display: flex; align-items: flex-start; gap: 8px;
    padding: 4px 18px; font-size: 11px;
    transition: background 0.1s;
  }
  .tl-row:hover { background: rgba(255,255,255,0.02); }
  .tl-row .t { font-family: monospace; color: var(--text-dim); min-width: 60px; font-size: 10px; flex-shrink:0; }
  .tl-row .m { color: var(--text); word-break: break-all; }
  .tag-r { background: rgba(124,58,237,0.2); color: #a78bfa; }
  .tag-t { background: rgba(59,130,246,0.2); color: #60a5fa; }
  .tag-i { background: rgba(245,158,11,0.2); color: #fbbf24; }
  .tag-s { background: rgba(34,197,94,0.2); color: #4ade80; }
  .tag-f { background: rgba(239,68,68,0.2); color: #f87171; }
  .tl-tag {
    display: inline-block; font-size: 9px; padding: 0 5px; border-radius: 3px;
    font-family: monospace; margin-right: 2px;
  }

  .empty-state { text-align: center; padding: 32px; color: var(--text-dim); }
  .empty-state .e { font-size: 36px; margin-bottom: 8px; }

  /* ── Toast ── */
  .toast {
    position: fixed; bottom: 20px; right: 20px;
    padding: 10px 18px; border-radius: 8px;
    background: var(--card-bg); border: 1px solid var(--card-border);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    font-size: 12px;
    transform: translateY(80px); opacity: 0;
    transition: all 0.35s cubic-bezier(0.34, 1.56, 0.64, 1);
    z-index: 1000;
    display: flex; align-items: center; gap: 8px;
  }
  .toast.show { transform: translateY(0); opacity: 1; }

  /* ── 响应式 ── */
  @media (max-width: 900px) {
    .container { padding: 12px 14px; }
    .agent { min-width: 100px; padding: 10px; }
    .agent-avatar { width: 40px; height: 40px; font-size: 18px; }
    .pipe-arrow { min-width: 16px; }
    .pipe-arrow .pa-body { width: 10px; }
    .header { padding: 12px 16px; }
    .header h1 { font-size: 16px; }
  }
</style>
</head>
<body>

<!-- 顶栏 -->
<div class="header">
  <div class="header-left">
    <div class="header-logo">🤖</div>
    <div>
      <h1><span>四柱</span><span>管线</span><span> · Agent Flow</span></h1>
      <div class="subtitle">智能体任务分配 · 领取 → 思考 → 接取 → 传递 · 全流程可视化</div>
    </div>
  </div>
  <div class="header-right">
    <div class="conn-indicator">
      <span class="conn-dot" id="connDot"></span>
      <span id="connLabel">未连接</span>
    </div>
    <button class="btn" onclick="loadTraces()">📂 轨迹</button>
  </div>
</div>

<div class="container">

  <!-- 统计 -->
  <div class="stats-bar" id="statsBar">
    <div class="stat-pill"><span class="s-icon">🤖</span><div><div class="s-num">10</div><div class="s-lbl">智能体</div></div></div>
    <div class="stat-pill"><span class="s-icon">🔄</span><div><div class="s-num" id="sIters">0</div><div class="s-lbl">迭代</div></div></div>
    <div class="stat-pill"><span class="s-icon">📋</span><div><div class="s-num" id="sTasks">0</div><div class="s-lbl">子任务</div></div></div>
    <div class="stat-pill"><span class="s-icon">🔧</span><div><div class="s-num" id="sTools">0</div><div class="s-lbl">工具调用</div></div></div>
  </div>

  <!-- 轨迹选择 -->
  <div class="trace-bar" id="traceBar" style="display:none;">
    <span class="tb-label">📁 轨迹:</span>
    <div id="traceBtns" style="display:flex;flex-wrap:wrap;gap:4px;"></div>
  </div>

  <!-- ▸ Phase 1 -->
  <div class="section-head">
    <span class="badge p1">Phase 1</span>
    <h2>🧠 四柱思考</h2>
    <span class="hint">纯 LLM 推演 · 按序传递上下文</span>
    <span class="line"></span>
  </div>
  <div class="pipeline-track" id="trackP1"></div>

  <!-- 过渡 -->
  <div style="text-align:center;margin:4px 0 10px;">
    <span style="font-size:20px;">⬇</span>
    <div style="font-size:10px;color:var(--text-dim);">四柱思考完成 → 进入执行循环</div>
  </div>

  <!-- ▸ Phase 2 -->
  <div class="section-head">
    <span class="badge p2">Phase 2</span>
    <h2>⚡ 执行循环</h2>
    <span class="hint">反思闭环 · 默认最多 20 轮 · 支持并行子任务</span>
    <span class="line"></span>
  </div>
  <div class="pipeline-track p2-track" id="trackP2"></div>

  <!-- 时间线 -->
  <div class="timeline-wrap">
    <div class="tl-header" onclick="toggleTL()">
      <h3>📜 执行记录</h3>
      <span class="tl-count" id="tlCount">0 条</span>
    </div>
    <div class="tl-body" id="tlBody">
      <div class="empty-state">
        <div class="e">⏳</div>
        <p>等待管线执行…<br>事件将实时出现在这里</p>
      </div>
    </div>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
// ═══════════════════════════════════════════════════════════════
//  角色定义
// ═══════════════════════════════════════════════════════════════

const PHASE1 = [
  { id:'analyst',        label:'分析者',   en:'ANALYST',        icon:'🔍', motto:'分析为底', color:'#7C3AED', desc:'多维度深度分析' },
  { id:'diffuser',       label:'扩散者',   en:'DIFFUSER',       icon:'🌊', motto:'扩展为路', color:'#2563EB', desc:'探索一切可能路径' },
  { id:'jumper',         label:'迁跃者',   en:'JUMPER',         icon:'🚀', motto:'迁跃为辅', color:'#D97706', desc:'创造性跨越突破' },
  { id:'reflector_p1',   label:'反思者',   en:'REFLECTOR',      icon:'🪞', motto:'反思为主', color:'#059669', desc:'综合审视制定计划' },
];

const PHASE2 = [
  { id:'decision_maker', label:'决策者',   en:'DECISION_MAKER', icon:'🎯', motto:'分解任务', color:'#DC2626', desc:'分解为可操作子任务' },
  { id:'thinker',        label:'思考者',   en:'THINKER',        icon:'💭', motto:'评估选择', color:'#7C3AED', desc:'评估合理性做出选择' },
  { id:'runner',         label:'执行者',   en:'RUNNER',         icon:'⚡', motto:'直接执行', color:'#2563EB', desc:'调用工具完成任务' },
  { id:'reader',         label:'阅读者',   en:'READER',         icon:'📖', motto:'信息获取', color:'#0891B2', desc:'读取文件网页信息' },
  { id:'builder',        label:'构建者',   en:'BUILDER',        icon:'🏗️', motto:'方案落地', color:'#D97706', desc:'创建目录编写代码' },
  { id:'checker',        label:'审计者',   en:'CHECKER',        icon:'✅', motto:'验证质量', color:'#059669', desc:'逐项检查执行结果' },
  { id:'reflector_p2',   label:'反思者',   en:'REFLECTOR',      icon:'🪞', motto:'闭环判定', color:'#059669', desc:'审视结果决定迭代' },
];

// ── 状态 ──
let tlOpen = true;
let currentTrace = null;

// ═══════════════════════════════════════════════════════════════
//  渲染智能体
// ═══════════════════════════════════════════════════════════════

function renderAgent(role, idx, phaseId) {
  const el = document.createElement('div');
  el.className = 'agent idle';
  el.id = `a-${role.id}`;
  el.dataset.phase = phaseId;
  el.dataset.roleId = role.id;
  el.style.borderColor = `${role.color}22`;
  el.innerHTML = `
    <div class="agent-row1">
      <div class="agent-avatar" style="border-color:${role.color}44;">
        <div class="ring"></div>
        <span>${role.icon}</span>
      </div>
      <div class="agent-info">
        <div class="agent-name">${role.label}</div>
        <div class="agent-en">${role.en}</div>
        <div class="agent-motto" style="color:${role.color};background:${role.color}15;">${role.motto}</div>
      </div>
      <span class="status-tag idle" id="st-${role.id}">💤 等待</span>
    </div>
    <div class="agent-desc">${role.desc}</div>
    <div class="thought-bubble" id="tb-${role.id}"></div>
  `;
  return el;
}

function renderArrow(phaseId, idx) {
  const el = document.createElement('div');
  el.className = 'pipe-arrow';
  el.id = `ar-${phaseId}-${idx}`;
  el.innerHTML = `<div class="pa-body"><span class="pa-head"></span></div>`;
  return el;
}

function renderTrack(containerId, roles, phaseId) {
  const c = document.getElementById(containerId);
  c.innerHTML = '';
  roles.forEach((role, i) => {
    if (i > 0) c.appendChild(renderArrow(phaseId, i));
    c.appendChild(renderAgent(role, i, phaseId));
  });
  // P2 加回路指示
  if (phaseId === 'p2') {
    const loop = document.createElement('div');
    loop.style.cssText = 'width:100%;display:flex;justify-content:center;margin:4px 0;';
    loop.innerHTML = '<div class="loop-ring"><span class="lr-icon">🔄</span><span>反思闭环 · 默认最多 20 轮迭代</span></div>';
    c.appendChild(loop);
  }
}

// ═══════════════════════════════════════════════════════════════
//  动画效果
// ═══════════════════════════════════════════════════════════════

function setAgentStatus(roleId, status, thought) {
  const card = document.getElementById(`a-${roleId}`);
  const tag = document.getElementById(`st-${roleId}`);
  const bubble = document.getElementById(`tb-${roleId}`);
  if (!card || !tag) return;

  // 清除旧状态
  card.className = 'agent';
  card.classList.add(status);

  const map = {
    idle:     { label: '💤 等待',     cls: 'idle' },
    thinking: { label: '💭 思考中…', cls: 'thinking' },
    running:  { label: '⚡ 执行中',   cls: 'running' },
    done:     { label: '✅ 完成',     cls: 'done' },
    failed:   { label: '❌ 失败',     cls: 'failed' },
  };
  const s = map[status] || map.idle;
  tag.textContent = s.label;
  tag.className = `status-tag ${s.cls}`;

  if (thought != null) {
    bubble.textContent = thought;
    bubble.classList.add('show');
  } else {
    bubble.classList.remove('show');
  }
}

function arrowActive(phaseId, idx, active) {
  const el = document.getElementById(`ar-${phaseId}-${idx}`);
  if (!el) return;
  if (active) el.classList.add('active');
  else el.classList.remove('active');
}

// ── 任务飞行物动画 ──
function flyTask(fromId, toId, label) {
  const from = document.getElementById(`a-${fromId}`);
  const to = document.getElementById(`a-${toId}`);
  if (!from || !to) return;
  const fr = from.getBoundingClientRect();
  const tr = to.getBoundingClientRect();
  const fly = document.createElement('div');
  fly.className = 'task-fly';
  fly.innerHTML = `<span class="fly-icon">📄</span>${label || '任务'}`;
  const sx = fr.right;
  const sy = fr.top + fr.height / 2;
  const ex = tr.left;
  const ey = tr.top + tr.height / 2;
  fly.style.left = sx + 'px';
  fly.style.top = sy + 'px';
  document.body.appendChild(fly);
  // 关键帧动画
  const dur = 600;
  fly.animate([
    { left: sx + 'px', top: sy + 'px', opacity: 1, transform: 'scale(1)' },
    { left: (sx+ex)/2 + 'px', top: (sy+ey)/2 - 30 + 'px', opacity: 1, transform: 'scale(1.15)' },
    { left: ex + 'px', top: ey + 'px', opacity: 0, transform: 'scale(0.8)' },
  ], { duration: dur, easing: 'cubic-bezier(0.34, 1.56, 0.64, 1)' });
  setTimeout(() => fly.remove(), dur);
}

// ── 任务领取动画 ──
function claimTask(roleId, taskName) {
  const card = document.getElementById(`a-${roleId}`);
  if (!card) return;
  setAgentStatus(roleId, 'running', `📥 领取任务: ${taskName}`);
  const rect = card.getBoundingClientRect();
  const spark = document.createElement('div');
  spark.style.cssText = `position:fixed;left:${rect.left+rect.width/2}px;top:${rect.top+rect.height/2}px;font-size:24px;pointer-events:none;z-index:999;transition:all 0.6s ease-out;opacity:1;transform:scale(0.5);`;
  spark.textContent = '📥';
  document.body.appendChild(spark);
  spark.animate([
    { transform: 'scale(0.5) translateY(-20px)', opacity: 1 },
    { transform: 'scale(1.2) translateY(-60px)', opacity: 0 },
  ], { duration: 600, easing: 'ease-out' });
  setTimeout(() => spark.remove(), 600);
}

// ── 思考动画 ──
function startThinking(roleId) {
  setAgentStatus(roleId, 'thinking', '🤔 正在深入思考…');
}
function showThought(roleId, text) {
  setAgentStatus(roleId, 'thinking', text);
}

// ── 传递任务 ──
function passTask(fromId, toId, taskName) {
  flyTask(fromId, toId, taskName);
  setAgentStatus(fromId, 'done', `📤 已传递给下一站`);
  setAgentStatus(toId, 'thinking', `📥 收到新任务: ${taskName}`);
}

// ═══════════════════════════════════════════════════════════════
//  事件处理
// ═══════════════════════════════════════════════════════════════

function handlePipelineEvent(ev) {
  addTimelineEvent(ev);
  updateAgentFromEvent(ev);
}

function updateAgentFromEvent(ev) {
  const e = ev.event || '';
  const det = ev.detail || '';

  // 阶段 1：按顺序触发
  const p1seq = [
    { event:'role_analyst',   id:'analyst',      arrow:1 },
    { event:'role_diffuser',  id:'diffuser',     arrow:2 },
    { event:'role_jumper',    id:'jumper',       arrow:3 },
    { event:'role_reflector', id:'reflector_p1', arrow:-1 },
  ];
  for (const p of p1seq) {
    if (e === p.event) {
      setAgentStatus(p.id, 'done', det.substring(0, 200) || '分析完成');
      if (p.arrow > 0) arrowActive('p1', p.arrow, true);
      // 找下一个角色并飞行
      const next = p1seq.find(x => x.arrow === p.arrow + 1);
      if (next) passTask(p.id, next.id, '分析结果');
      break;
    }
  }

  // 检查器
  if (e === 'role_checker') {
    const failed = det.includes('失败') || det.includes('不通过');
    setAgentStatus('checker', failed ? 'failed' : 'done', det.substring(0, 200));
  }

  // 工具调用：找最近的活跃角色
  if (e === 'tool_call') {
    const toolName = ev.metadata?.tool || '?';
    // 找第一个 running/thinking 的角色显示工具调用
    document.querySelectorAll('.agent').forEach(c => {
      if (c.classList.contains('running') || c.classList.contains('thinking')) {
        const bid = c.id.replace('a-', '');
        const bubble = document.getElementById(`tb-${bid}`);
        if (bubble) {
          const chip = `<span class="tool-chip">🔧 ${toolName}</span> `;
          bubble.innerHTML = (bubble.innerHTML || '') + chip;
          bubble.classList.add('show');
        }
      }
    });
  }

  // 生命周期
  if (e === 'pipeline_start') {
    resetAll();
    showToast('🚀 管线启动', 'info');
  }
  if (e === 'pipeline_complete') {
    showToast('🎉 管线执行完成', 'success');
    document.querySelectorAll('.agent.done').forEach(c => {
      c.style.borderColor = 'rgba(34,197,94,0.25)';
    });
  }
  if (e === 'pipeline_abort') {
    showToast('⛔ 管线中止', 'error');
  }
  if (e === 'iteration_start') {
    const n = det.match(/\d+/)?.[0] || '?';
    document.getElementById('sIters').textContent = n;
    showToast(`🔄 第 ${n} 轮迭代`, 'info');
    // 清空 P2 角色状态
    PHASE2.forEach(r => {
      if (r.id !== 'reflector_p2') setAgentStatus(r.id, 'idle');
    });
  }
  if (e === 'subtask_start') {
    document.getElementById('sTasks').textContent =
      parseInt(document.getElementById('sTasks').textContent || '0') + 1;
  }
}

function resetAll() {
  document.querySelectorAll('.agent').forEach(c => {
    c.className = 'agent idle';
    c.style.borderColor = '';
  });
  document.querySelectorAll('.status-tag').forEach(t => {
    t.textContent = '💤 等待';
    t.className = 'status-tag idle';
  });
  document.querySelectorAll('.thought-bubble').forEach(b => {
    b.classList.remove('show');
    b.innerHTML = '';
  });
  document.querySelectorAll('.pipe-arrow').forEach(a => a.classList.remove('active'));
  const body = document.getElementById('tlBody');
  body.innerHTML = '';
  document.getElementById('tlCount').textContent = '0 条';
  document.getElementById('sIters').textContent = '0';
  document.getElementById('sTasks').textContent = '0';
  document.getElementById('sTools').textContent = '0';
}

// ═══════════════════════════════════════════════════════════════
//  时间线
// ═══════════════════════════════════════════════════════════════

function addTimelineEvent(ev) {
  const body = document.getElementById('tlBody');
  const empty = body.querySelector('.empty-state');
  if (empty) body.innerHTML = '';

  const ts = (ev.timestamp || '').slice(11,19) || new Date().toISOString().slice(11,19);
  const e = ev.event || '?';
  const d = ev.detail || ev.detail === '' ? ev.detail : '';

  let icon = '▫️', cls = '', msg = '';
  if (e.startsWith('role_'))     { icon = '🎭'; cls='tag-r'; msg = `${e.replace('role_','')} 完成`; }
  else if (e==='tool_call')     { icon = '🔧'; cls='tag-t'; const tn = ev.metadata?.tool||'?'; msg = `${tn}(${(d||'').substring(0,100)})`; }
  else if (e==='tool_result')   { icon = ev.metadata?.status==='失败'?'❌':'✅'; cls='tag-'+(ev.metadata?.status==='失败'?'f':'s'); msg = `${ev.metadata?.tool||''} → ${ev.metadata?.status||'完成'}`; }
  else if (e==='subtask_start') { icon = '📋'; cls='tag-i'; msg = d.substring(0,100); }
  else if (e==='subtask_complete') { icon='✅'; cls='tag-s'; msg='完成'; }
  else if (e==='subtask_timeout')  { icon='⏰'; cls='tag-f'; msg=`超时 ${d.substring(0,60)}`; }
  else if (e==='subtask_error')    { icon='❌'; cls='tag-f'; msg=`失败 ${d.substring(0,60)}`; }
  else if (e==='iteration_start')  { icon='🔄'; cls='tag-i'; const n=d.match(/\d+/)?.[0]||'?'; msg=`第 ${n} 轮迭代`; }
  else if (e==='iteration_done')   { icon='🏁'; cls='tag-s'; msg = d||'执行完成'; }
  else if (e==='pipeline_start')   { icon='🚀'; cls=''; msg = `管线启动`; }
  else if (e==='pipeline_complete'){ icon='🎉'; cls='tag-s'; msg='管线执行完成'; }
  else if (e==='pipeline_abort')   { icon='⛔'; cls='tag-f'; msg=`管线中止: ${d.substring(0,80)}`; }
  else if (e==='parallel_batch_start'){ icon='⚡'; cls='tag-i'; msg=`并行 ${d}`; }
  else if (e==='parallel_batch_end')  { icon='✅'; cls='tag-s'; msg='并行完成'; }
  else { icon='📌'; cls=''; msg=`${e} ${(d||'').substring(0,100)}`; }

  // 工具调用数
  if (e === 'tool_call') {
    document.getElementById('sTools').textContent =
      parseInt(document.getElementById('sTools').textContent || '0') + 1;
  }

  const row = document.createElement('div');
  row.className = 'tl-row';
  const tagHtml = cls ? `<span class="tl-tag ${cls}">${e.replace('role_','').replace('_',' ')}</span>` : '';
  row.innerHTML = `<span class="t">${ts}</span><span>${icon}</span><span class="m">${tagHtml} ${msg}</span>`;
  body.appendChild(row);
  body.scrollTop = body.scrollHeight;
  document.getElementById('tlCount').textContent = `${body.querySelectorAll('.tl-row').length} 条`;
}

// ═══════════════════════════════════════════════════════════════
//  加载轨迹
// ═══════════════════════════════════════════════════════════════

async function loadTraces() {
  const bar = document.getElementById('traceBar');
  bar.style.display = 'flex';
  const btns = document.getElementById('traceBtns');
  btns.innerHTML = '<span style="font-size:11px;color:var(--text-dim);">加载中…</span>';
  try {
    const r = await fetch('/api/traces');
    const d = await r.json();
    btns.innerHTML = '';
    if (!d.traces || !d.traces.length) {
      btns.innerHTML = '<span style="font-size:11px;color:var(--text-dim);">暂无轨迹</span>';
      return;
    }
    d.traces.forEach(t => {
      const b = document.createElement('button');
      b.className = 'trace-btn' + (t.session_id===currentTrace ? ' cur' : '');
      const ts = (t.time||'').slice(0,16).replace('T',' ');
      b.textContent = `${ts} (${t.events}事)`;
      b.onclick = () => loadTraceDetail(t.session_id);
      btns.appendChild(b);
    });
  } catch (_) {
    btns.innerHTML = '<span style="font-size:11px;color:var(--error);">加载失败</span>';
  }
}

async function loadTraceDetail(sid) {
  try {
    const r = await fetch(`/api/traces/${sid}`);
    const d = await r.json();
    currentTrace = sid;
    document.querySelectorAll('.trace-btn').forEach(b => b.classList.remove('cur'));
    document.querySelectorAll('.trace-btn').forEach(b => {
      if (b.textContent.includes(sid.slice(0,6))) b.classList.add('cur');
    });
    resetAll();
    d.events.forEach(ev => {
      addTimelineEvent(ev);
      updateAgentFromEvent(ev);
    });
    showToast(`📂 已加载轨迹 (${d.events.length} 事件)`, 'success');
  } catch (_) {
    showToast('加载轨迹失败', 'error');
  }
}

// ═══════════════════════════════════════════════════════════════
//  SSE / 连接
// ═══════════════════════════════════════════════════════════════

function connectSSE() {
  const dot = document.getElementById('connDot');
  const lbl = document.getElementById('connLabel');
  fetch('/api/health').then(r=>r.json()).then(()=>{
    dot.className = 'conn-dot on'; lbl.textContent = '已连接';
  }).catch(()=>{ dot.className='conn-dot'; lbl.textContent='离线'; });

  const es = new EventSource('/api/events');
  es.onmessage = e => {
    try {
      const m = JSON.parse(e.data);
      if (m.type === 'connected') { dot.className='conn-dot on'; lbl.textContent='实时连接'; }
      else if (m.type === 'pipeline_event') handlePipelineEvent(m.data);
      else if (m.type === 'state_update') {
        if (m.data && m.data.events) m.data.events.forEach(ev => handlePipelineEvent(ev));
      }
    } catch(_) {}
  };
  es.onerror = () => { dot.className='conn-dot'; lbl.textContent='断开(自动重连)'; };
}

function toggleTL() {
  const b = document.getElementById('tlBody');
  tlOpen = !tlOpen;
  b.style.display = tlOpen ? 'block' : 'none';
}

// ── Toast ──
function showToast(msg, type) {
  const t = document.getElementById('toast');
  t.innerHTML = msg;
  t.className = 'toast show';
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.remove('show'), 3500);
}

// ═══════════════════════════════════════════════════════════════
//  初始化
// ═══════════════════════════════════════════════════════════════

renderTrack('trackP1', PHASE1, 'p1');
renderTrack('trackP2', PHASE2, 'p2');
connectSSE();

// 自动加载最近轨迹
fetch('/api/traces').then(r=>r.json()).then(d=>{
  if (d.traces && d.traces.length) {
    document.getElementById('traceBar').style.display = 'flex';
    loadTraces();
    loadTraceDetail(d.traces[0].session_id);
  }
}).catch(()=>{});
</script>
</body>
</html>"""
