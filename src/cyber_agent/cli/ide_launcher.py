"""IDE 一键启动器：环境检测 → 依赖安装 → 后端启动 → 前端启动。"""

from __future__ import annotations

import json
import os
import queue
import re
import socket
import subprocess
import sys
import threading
import time
import unicodedata
import urllib.request
from pathlib import Path

# ── ANSI 颜色 ──

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_RED    = "\033[31m";  _GREEN  = "\033[32m"; _YELLOW = "\033[33m"
_BLUE   = "\033[34m";  _MAGENTA= "\033[35m"; _CYAN   = "\033[36m"; _WHITE  = "\033[37m"
_BBLACK = "\033[90m";  _BRED   = "\033[91m"; _BYELLOW= "\033[93m"
_BBLUE  = "\033[94m";  _BMAG   = "\033[95m"; _BCYAN  = "\033[96m"; _BWHITE = "\033[97m"

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")
_BOX_INNER = 36
_LABEL_W   = 20

def _c(color: str, text: str) -> str:
    return f"{color}{text}{_RESET}"

def _ok(t: str = "✓") -> str:   return _c(_GREEN, t)
def _err(t: str = "✗") -> str:  return _c(_RED, t)
def _warn(t: str) -> str:       return _c(_YELLOW, t)
def _info(t: str) -> str:       return _c(_CYAN, t)
def _label(t: str) -> str:      return _c(_DIM, t)
def _ide() -> str:              return _c(_BMAG, "[IDE]")
def _phase(n: str) -> str:      return _c(_BOLD + _BBLUE, n)

def _display_width(text: str) -> int:
    clean = _ANSI_RE.sub("", text)
    w = 0
    for ch in clean:
        w += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return w

def _box_line(text: str) -> str:
    dw = _display_width(text)
    left = max(0, (_BOX_INNER - dw) // 2)
    right = _BOX_INNER - dw - left
    b = _c(_BCYAN, "║")
    return f"{b}{' ' * left}{text}{' ' * right}{b}"

def _row(label: str, value: str, ok_: bool | None = None) -> None:
    """固定宽度行。ok_=True→末尾绿色✓, False→红色✗, None→无标记。"""
    mark = ""
    if ok_ is True:   mark = f"  {_ok()}"
    elif ok_ is False: mark = f"  {_err()}"
    pad = max(1, _LABEL_W - _display_width(label))
    print(f" {_ide()}  {_label(label)}{' ' * pad}{value}{mark}")

def _sub(label: str, value: str) -> None:
    """次级详情行：缩进 + 灰色标签。"""
    print(f" {_ide()}    {_label(label)}  {value}")

def _section(n: str, title: str) -> None:
    print(f"\n {_phase(n)} {_c(_BOLD + _BWHITE, title)}")
    print(f" {_label('─' * (_BOX_INNER + 2))}")

# ── 常量 ──

DEFAULT_IDE_HOST = "127.0.0.1"
BACKEND_STARTUP_TIMEOUT = 60
HEALTH_POLL_INTERVAL = 0.3
NPM_INSTALL_TIMEOUT = 300


# ── 工具函数 ──

def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

def _which(cmd: str) -> str | None:
    for p in os.environ.get("PATH", "").split(os.pathsep):
        for name in (cmd, f"{cmd}.exe", f"{cmd}.cmd"):
            candidate = Path(p) / name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
    return None

def _get_desktop_dir() -> Path:
    return (Path(__file__).resolve().parent.parent.parent.parent / "desktop").resolve()


# ── Phase 1: 环境检测 ──

class EnvReport:
    def __init__(self):
        self.node_bin: str | None = None
        self.node_version = ""
        self.npm_bin: str | None = None
        self.npm_version = ""
        self.python_bin = sys.executable
        self.os_name = sys.platform
        self.desktop_dir = _get_desktop_dir()
        self.node_modules_exists = False
        self.electron_bin: str | None = None
        self.vite_dev_server_running = False
        self.errors: list[str] = []
        self.warnings: list[str] = []


def _check_single(env: EnvReport, label: str, checker) -> bool:
    ok_ = checker(env)
    if ok_:
        _row(label, "已检测", True)
    else:
        _row(label, "缺失", False)
    return ok_


def _detect_node(env: EnvReport) -> bool:
    node = _which("node")
    if not node:
        env.errors.append("未检测到 Node.js。请安装 Node.js >= 18: https://nodejs.org")
        return False
    env.node_bin = node
    try:
        r = subprocess.run([node, "-v"], capture_output=True, text=True, timeout=10)
        env.node_version = r.stdout.strip()
        return True
    except Exception:
        env.errors.append("Node.js 无法执行。")
        return False

def _detect_npm(env: EnvReport) -> bool:
    npm = _which("npm")
    if not npm:
        env.errors.append("未检测到 npm（通常随 Node.js 安装）。")
        return False
    env.npm_bin = npm
    try:
        r = subprocess.run([npm, "-v"], capture_output=True, text=True, timeout=10)
        env.npm_version = r.stdout.strip()
        return True
    except Exception:
        env.errors.append("npm 无法执行。")
        return False

def _detect_desktop(env: EnvReport) -> bool:
    if not env.desktop_dir.exists():
        env.errors.append(f"desktop/ 目录不存在: {env.desktop_dir}")
        return False
    if not (env.desktop_dir / "package.json").exists():
        env.errors.append("desktop/package.json 不存在。")
        return False
    return True

def _detect_node_modules(env: EnvReport) -> bool:
    nm = env.desktop_dir / "node_modules"
    if nm.is_dir():
        electron = env.desktop_dir / "node_modules" / ".bin" / "electron"
        vite = env.desktop_dir / "node_modules" / ".bin" / "vite"
        if electron.exists() or vite.exists():
            env.node_modules_exists = True
            env.electron_bin = str(electron) if electron.exists() else None
            return True
    env.warnings.append("前端依赖未安装。")
    return False

def _detect_vite(env: EnvReport) -> bool:
    try:
        resp = urllib.request.urlopen("http://localhost:5173", timeout=1)
        env.vite_dev_server_running = resp.status == 200
        return True if env.vite_dev_server_running else False
    except Exception:
        return False

def _detect_wayland(env: EnvReport) -> bool:
    return bool(os.environ.get("WAYLAND_DISPLAY") or os.environ.get("XDG_SESSION_TYPE") == "wayland")

def detect_environment(env: EnvReport | None = None) -> EnvReport:
    if env is None:
        env = EnvReport()
    _row("系统", _c(_BWHITE, env.os_name))
    _sub("可执行文件", _c(_BCYAN, env.python_bin))
    _row("Python", _c(_BWHITE, sys.version.split()[0]))
    _check_single(env, "Node.js", _detect_node)
    if env.node_version:
        _sub("版本", _c(_BWHITE, env.node_version))
        _sub("路径", _c(_BCYAN, str(env.node_bin)))
    _check_single(env, "npm", _detect_npm)
    if env.npm_version:
        _sub("版本", _c(_BWHITE, env.npm_version))
        _sub("路径", _c(_BCYAN, str(env.npm_bin)))
    _check_single(env, "desktop/", _detect_desktop)
    _sub("路径", _c(_BCYAN, str(env.desktop_dir)))
    _check_single(env, "node_modules", _detect_node_modules)
    if env.node_modules_exists:
        _row("Electron", _c(_BWHITE, str(env.electron_bin)))
    else:
        _row("Electron", _warn("○ 待安装"))
    _check_single(env, "Vite dev server", _detect_vite)
    if env.vite_dev_server_running:
        _sub("地址", _c(_BCYAN, "http://localhost:5173"))
    _row("Wayland", _ok("是") if _detect_wayland(env) else _label("否 (X11)"))

    if env.warnings:
        for w in env.warnings:
            print(f" {_ide()}  {_warn('⚠ ' + w)}")
    if env.errors:
        for e in env.errors:
            print(f" {_ide()}  {_err('✗ ' + e)}")
    return env


# ── Phase 2: 依赖安装 ──

def install_dependencies(env: EnvReport, force: bool = False) -> bool:
    if env.node_modules_exists and not force:
        _row("依赖状态", _c(_BWHITE, "已安装"), True)
        return True

    if not env.npm_bin:
        _row("npm", _err("不可用"), False)
        return False

    _row("包管理器", _c(_BWHITE, f"npm {env.npm_version}"))
    _row("安装目录", _c(_BCYAN, str(env.desktop_dir)))
    _row("Electron 镜像", _c(_CYAN, os.environ.get("ELECTRON_MIRROR", "https://npmmirror.com/mirrors/electron/")))

    print(f" {_ide()}  {_label('正在运行 npm install ...')}")
    print(f" {_ide()}  {_label('（首次约 180MB，请耐心等待）')}")

    t0 = time.time()
    try:
        result = subprocess.run(
            [env.npm_bin, "install"],
            cwd=str(env.desktop_dir),
            capture_output=False,
            text=True,
            timeout=NPM_INSTALL_TIMEOUT,
            env={**os.environ, "ELECTRON_MIRROR": os.environ.get("ELECTRON_MIRROR", "https://npmmirror.com/mirrors/electron/")},
        )
        elapsed = time.time() - t0
        if result.returncode != 0:
            _row("npm install", _err("失败"), False)
            _sub("手动安装", _c(_CYAN, "cd desktop && npm install"))
            return False
    except subprocess.TimeoutExpired:
        _row("npm install", _err("超时"), False)
        return False

    nm = env.desktop_dir / "node_modules"
    if nm.is_dir():
        env.node_modules_exists = True
        electron = env.desktop_dir / "node_modules" / ".bin" / "electron"
        if electron.exists():
            env.electron_bin = str(electron)

    _row("安装耗时", _c(_BWHITE, f"{elapsed:.1f}s"))
    _row("依赖安装", _c(_BWHITE, "完成"), True)
    return True


# ── Phase 3: 后端启动（in-process daemon thread + 进度队列） ──

_backend_thread: threading.Thread | None = None
_backend_port = 0
_backend_ready = threading.Event()
_progress_q: queue.Queue[dict] = queue.Queue()


def _progress(label: str, value: str, ok_: bool | None = None) -> None:
    """从后端线程推送进度到队列。"""
    _progress_q.put({"label": label, "value": value, "ok": ok_})


def _start_backend_in_thread(
    host: str = "127.0.0.1",
    port: int = 0,
    mode: str = "standard",
    allow_paths: list[str] | None = None,
    approval_policy: str = "prompt",
    service_name: str | None = None,
    model_name: str | None = None,
) -> int:
    global _backend_thread, _backend_port, _backend_ready

    if port == 0:
        port = _find_free_port()
    _backend_port = port
    _backend_ready.clear()
    # 清空队列
    while not _progress_q.empty():
        _progress_q.get_nowait()

    def _run():
        import uvicorn
        from ..config import settings as _cfg
        from .ide_server import build_ide_runtime_context, init_runner, create_app

        # Step 1: 读取配置
        svc = service_name or _cfg.get_service()
        mdl = model_name or _cfg.get_model_name(service_name=svc)
        _progress("读取配置", f"service={svc}  model={mdl}", None)
        _progress("网关地址", str(_cfg.resolve_base_url(svc)), None)
        _progress("审批策略", approval_policy, None)
        _progress("运行模式", mode, None)

        # Step 2: 构建运行上下文
        _progress("构建运行上下文", "build_ide_runtime_context()", None)
        rt = build_ide_runtime_context(
            mode=mode, allow_paths=allow_paths,
            approval_policy=approval_policy,
            service_name=svc, model_name=mdl,
        )
        _progress("构建运行上下文", "完成", True)

        # Step 3: 初始化 AgentRunner + 加载工具
        _progress("初始化 AgentRunner", "init_runner() ...", None)
        runner = init_runner(rt)
        n_tools = len(runner.tools)
        _progress("AgentRunner 就绪", f"已加载 {n_tools} 个工具", True)
        for t in runner.tools[:6]:
            name = getattr(t, "name", str(t))
            risk = "?"
            if hasattr(t, "metadata") and isinstance(t.metadata, dict):
                risk = t.metadata.get("risk", "?")
            _sub("工具", f"{name}  [{risk}]")
        if n_tools > 6:
            _sub("...", f"还有 {n_tools - 6} 个工具")

        # Step 4: 创建 FastAPI 应用
        _progress("创建 API 应用", "create_app()", None)
        app = create_app()
        n_routes = len(app.routes)
        _progress("API 路由注册", f"{n_routes} 条路由", True)

        # Step 5: 启动 uvicorn
        class ReadyServer(uvicorn.Server):
            def started(self):
                _progress("uvicorn 启动", "socket 绑定完成", True)
                _backend_ready.set()

        config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        server = ReadyServer(config)
        _progress("启动 HTTP 服务器", f"uvicorn.run({host}:{port})", None)
        try:
            server.run()
        except Exception as exc:
            _progress("uvicorn 异常", str(exc), False)
            _backend_ready.set()  # 避免主线程死等

    _backend_thread = threading.Thread(target=_run, daemon=True)
    _backend_thread.start()

    return _wait_backend_ready(host, port)


def _wait_backend_ready(host: str, port: int) -> int:
    """消费进度队列 + 健康检查双通道等待。"""
    start = time.time()
    seen: set[str] = set()

    _row("启动端口", _c(_BCYAN, f"{host}:{port}"))

    while True:
        # 消费队列中所有进度事件
        drained = False
        while True:
            try:
                p = _progress_q.get_nowait()
            except queue.Empty:
                break
            drained = True
            key = p["label"]
            # 不输出重复行（同名 label 只更新值）
            if key in seen:
                continue
            seen.add(key)
            _row(key, _c(_BWHITE, str(p["value"])), p["ok"])

        if not drained:
            time.sleep(0.08)  # 避免空转

        # 检查 uvicorn 是否就绪
        if _backend_ready.is_set():
            _row("服务器就绪", _ok("socket 监听中"), True)
            break

        if time.time() - start > BACKEND_STARTUP_TIMEOUT:
            _row("uvicorn 启动", _err("超时"), False)
            return 0

    # HTTP 健康检查
    health_url = f"http://{host}:{port}/api/health"

    # 先快速尝试几次
    for _ in range(5):
        try:
            resp = urllib.request.urlopen(urllib.request.Request(health_url), timeout=1)
            if resp.status == 200:
                body = json.loads(resp.read())
                svc = body.get("service", "?")
                mdl = body.get("model", "?")
                _row("健康检查", _ok("通过"), True)
                _sub("Service", _c(_BWHITE, str(svc)))
                _sub("Model", _c(_BWHITE, str(mdl)))
                _sub("Session", _c(_BCYAN, str(body.get("session_id", ""))))
                _backend_port = port
                return port
        except Exception:
            time.sleep(0.2)

    # 轮询等待
    attempts = 5
    while time.time() - start < BACKEND_STARTUP_TIMEOUT:
        attempts += 1
        try:
            resp = urllib.request.urlopen(urllib.request.Request(health_url), timeout=1)
            if resp.status == 200:
                body = json.loads(resp.read())
                svc = body.get("service", "?")
                mdl = body.get("model", "?")
                _row("健康检查", _ok("通过"), True)
                _sub("Service", _c(_BWHITE, str(svc)))
                _sub("Model", _c(_BWHITE, str(mdl)))
                _sub("Session", _c(_BCYAN, str(body.get("session_id", ""))))
                _backend_port = port
                return port
        except Exception:
            pass
        if attempts % 40 == 1:
            _row("健康检查", _label(f"等待中（{attempts} 次）..."))
        time.sleep(HEALTH_POLL_INTERVAL)

    _row("健康检查", _err("超时"), False)
    return 0


# ── Phase 4: Electron 启动 ──

def _find_electron(env: EnvReport) -> str | None:
    if env.electron_bin and Path(env.electron_bin).exists():
        return env.electron_bin
    candidate = env.desktop_dir / "node_modules" / ".bin" / "electron"
    if candidate.exists():
        env.electron_bin = str(candidate)
        return str(candidate)
    se = _which("electron")
    if se:
        env.electron_bin = se
        return se
    if env.npm_bin:
        npx = str(Path(env.npm_bin).parent / "npx")
        if Path(npx).exists():
            return npx
    return None

def _get_electron_flags() -> list[str]:
    flags = []
    if sys.platform == "linux":
        if os.environ.get("WAYLAND_DISPLAY") or os.environ.get("XDG_SESSION_TYPE") == "wayland":
            flags.extend([
                "--enable-features=UseOzonePlatform",
                "--ozone-platform=wayland",
                "--enable-gpu-rasterization",
            ])
    return flags


# ── 主入口 ──

def launch_ide(
    mode: str = "standard",
    allow_paths: list[str] | None = None,
    approval_policy: str = "prompt",
    service_name: str | None = None,
    model_name: str | None = None,
    dev: bool = False,
    skip_install: bool = False,
) -> int:
    # 横幅
    print()
    print(_c(_BCYAN, "╔" + "═" * _BOX_INNER + "╗"))
    print(_box_line(_c(_BOLD + _BWHITE, "Cyber Agent IDE")))
    print(_box_line(_c(_BCYAN, "Liquid Glass · 一键启动")))
    print(_c(_BCYAN, "╚" + "═" * _BOX_INNER + "╝"))

    # ── Phase 1 ──
    _section("[1/4]", "环境检测")
    env = EnvReport()
    detect_environment(env)
    if env.errors:
        print(f"\n {_ide()} {_err('环境检测未通过，请修复后重试。')}")
        return 1

    # ── Phase 2 ──
    _section("[2/4]", "前端依赖")
    if not skip_install and not env.node_modules_exists:
        if not install_dependencies(env):
            print(f"\n {_ide()} {_err('依赖安装失败。')}")
            return 1
        env = EnvReport()
        detect_environment(env)
    else:
        _row("依赖状态", _c(_BWHITE, "已安装"), True)

    # ── Phase 3 ──
    _section("[3/4]", "启动后端服务器")
    backend_port = _start_backend_in_thread(
        host=DEFAULT_IDE_HOST, mode=mode,
        allow_paths=allow_paths, approval_policy=approval_policy,
        service_name=service_name, model_name=model_name,
    )
    if backend_port == 0:
        print(f"\n {_ide()} {_err('后端启动失败。')}")
        return 1

    backend_url = _c(_BCYAN, f"http://{DEFAULT_IDE_HOST}:{backend_port}")
    _row("后端地址", backend_url)

    # ── Phase 4 ──
    _section("[4/4]", "启动前端")
    is_dev = dev or not (env.desktop_dir / "dist" / "renderer" / "index.html").exists()

    if is_dev and not env.vite_dev_server_running:
        _row("模式", _warn("开发模式 (Vite HMR)"))
        _row("Vite 状态", _label("未运行"))
        _sub("启动命令", _c(_CYAN, f"cd {env.desktop_dir} && npx vite"))
        _sub("后端已就绪", backend_url)
        _sub("提示", "按 Ctrl+C 退出")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            return 0

    electron_bin = _find_electron(env)
    if not electron_bin or electron_bin == "npx":
        _row("Electron", _warn("未找到，使用 npx"))
        electron_bin = "npx"
        electron_args = ["electron"]
    else:
        _row("Electron", _c(_BWHITE, electron_bin))
        electron_args = []

    electron_args.extend([str(env.desktop_dir), f"--backend-port={backend_port}"])
    eflags = _get_electron_flags()
    electron_args.extend(eflags)

    _row("窗口协议", _ok("Wayland") if eflags else _label("X11 / 默认"))
    _row("启动 Electron", _label("正在启动 …"))
    print()

    try:
        ep = subprocess.Popen(
            [electron_bin] + electron_args,
            cwd=str(env.desktop_dir),
            env={**os.environ, "CYBER_AGENT_BACKEND_PORT": str(backend_port),
                 "ELECTRON_OZONE_PLATFORM_HINT": os.environ.get("XDG_SESSION_TYPE", "")},
        )
        return ep.wait()
    except FileNotFoundError:
        _row("Electron", _err("未找到可执行文件"), False)
        print(f" {_ide()}  {_label('请运行: cd desktop && npm install')}")
        return 1
    except KeyboardInterrupt:
        print(f"\n {_ide()} {_label('已退出。')}")
        return 0


# ── 独立后端入口 ──

def run_ide_server_main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Cyber Agent IDE Server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--mode", default="standard")
    p.add_argument("--approval-policy", default="prompt")
    p.add_argument("--allow-path", action="append", dest="allow_paths", default=None)
    p.add_argument("--service", default=None)
    p.add_argument("--model", default=None)
    args = p.parse_args()

    from .ide_server import run_ide_server
    run_ide_server(
        host=args.host, port=args.port, mode=args.mode,
        allow_paths=args.allow_paths, approval_policy=args.approval_policy,
        service_name=args.service, model_name=args.model,
    )

if __name__ == "__main__":
    run_ide_server_main()
