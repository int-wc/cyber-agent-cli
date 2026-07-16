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
        tauri_cli = env.desktop_dir / "node_modules" / ".bin" / "tauri"
        vite = env.desktop_dir / "node_modules" / ".bin" / "vite"
        if tauri_cli.exists() or vite.exists():
            env.node_modules_exists = True
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
        _row("Tauri CLI", _ok("已安装"))
    else:
        _row("Tauri CLI", _warn("○ 待安装"))
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
            env={**os.environ},
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

    _row("安装耗时", _c(_BWHITE, f"{elapsed:.1f}s"))
    _row("依赖安装", _c(_BWHITE, "完成"), True)
    return True


# ── Phase 3: 后端启动（in-process daemon thread + 进度队列） ──

_backend_thread: threading.Thread | None = None
_backend_port = 0
_backend_ready = threading.Event()
_progress_q: queue.Queue[dict] = queue.Queue()
_boot_error_q: queue.Queue[str] = queue.Queue()


def _progress(label: str, value: str, ok_: bool | None = None) -> None:
    _progress_q.put({"label": label, "value": value, "ok": ok_, "sub": False})

def _progress_sub(label: str, value: str) -> None:
    _progress_q.put({"label": label, "value": value, "ok": None, "sub": True})


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
    _boot_error_q.queue.clear()
    while not _progress_q.empty():
        _progress_q.get_nowait()

    def _run():
        try:
            import uvicorn
        except ImportError as e:
            _boot_error_q.put(f"无法导入 uvicorn: {e}")
            _backend_ready.set()
            return

        try:
            from ..config import settings as _cfg
            from .ide_server import build_ide_runtime_context, init_runner, create_app
        except Exception as e:
            _boot_error_q.put(f"导入 ide_server 失败: {e}")
            _backend_ready.set()
            return

        # 步骤 1：读取配置
        svc = service_name or _cfg.get_service()
        mdl = model_name or _cfg.get_model_name(service_name=svc)
        _progress("读取配置", f"service={svc}  model={mdl}", None)
        _progress("网关地址", str(_cfg.resolve_base_url(svc)), None)
        _progress("审批策略", approval_policy, None)
        _progress("运行模式", mode, None)

        # 步骤 2：构建运行上下文
        _progress("构建运行上下文", "build_ide_runtime_context()", None)
        try:
            rt = build_ide_runtime_context(
                mode=mode, allow_paths=allow_paths,
                approval_policy=approval_policy,
                service_name=svc, model_name=mdl,
            )
        except Exception as e:
            _boot_error_q.put(f"构建运行上下文失败: {e}")
            _backend_ready.set()
            return
        _progress("构建运行上下文", "完成", True)

        # 步骤 3：初始化 AgentRunner 并加载工具
        _progress("初始化 AgentRunner", "init_runner() ...", None)
        try:
            runner = init_runner(rt)
        except Exception as e:
            _boot_error_q.put(f"init_runner 失败: {e}")
            _backend_ready.set()
            return
        n_tools = len(runner.tools)
        _progress("AgentRunner 就绪", f"已加载 {n_tools} 个工具", True)
        for t in runner.tools[:6]:
            name = getattr(t, "name", str(t))
            risk = "?"
            if hasattr(t, "metadata") and isinstance(t.metadata, dict):
                risk = t.metadata.get("risk", "?")
            _progress_sub("工具", f"{name}  [{risk}]")
        if n_tools > 6:
            _progress_sub("...", f"还有 {n_tools - 6} 个工具")

        # 步骤 4：创建 FastAPI 应用
        try:
            _progress("创建 API 应用", "create_app()", None)
            app = create_app()
            n_routes = len(app.routes)
            _progress("API 路由注册", f"{n_routes} 条路由", True)
        except Exception as e:
            _boot_error_q.put(f"create_app 失败: {e}")
            _backend_ready.set()
            return

        # 步骤 5：启动 uvicorn（不依赖 started() 回调，用健康检查轮询）
        try:
            config = uvicorn.Config(app, host=host, port=port, log_level="error")
            server = uvicorn.Server(config)
        except Exception as e:
            _boot_error_q.put(f"uvicorn.Config 失败: {e}")
            return

        _progress("启动 HTTP 服务器", f"uvicorn.run({host}:{port})", None)

        # 延迟一小段时间再启动，让主线程先进入轮询状态
        def _delayed_run():
            time.sleep(0.3)
            try:
                server.run()
            except Exception as exc:
                _boot_error_q.put(f"uvicorn.run 异常: {exc}")

        threading.Thread(target=_delayed_run, daemon=True).start()

    _backend_thread = threading.Thread(target=_run, daemon=True)
    _backend_thread.start()

    return _wait_backend_ready(host, port)


def _drain_progress_q() -> None:
    """消费进度队列：主行用 _row，次级行用 _sub。"""
    pending_rows: dict[str, dict] = {}
    pending_subs: list[dict] = []

    while True:
        try:
            p = _progress_q.get_nowait()
        except queue.Empty:
            break
        if p.get("sub"):
            pending_subs.append(p)
        else:
            pending_rows[p["label"]] = p

    for p in pending_rows.values():
        _row(p["label"], _c(_BWHITE, str(p["value"])), p["ok"])
    for p in pending_subs:
        if p["label"] == "...":
            _sub(p["label"], _label(p["value"]))
        else:
            _sub(p["label"], _c(_BWHITE, str(p["value"])))


def _wait_backend_ready(host: str, port: int) -> int:
    """纯 HTTP 健康检查轮询（不依赖 uvicorn 内部回调）。"""
    start = time.time()

    _row("启动端口", _c(_BCYAN, f"{host}:{port}"))
    _drain_progress_q()

    # 初始等待：给 uvicorn 启动时间
    _row("等待 uvicorn", _label("启动中 …"))
    time.sleep(1.5)

    health_url = f"http://{host}:{port}/api/health"
    attempts = 0

    while time.time() - start < BACKEND_STARTUP_TIMEOUT:
        _drain_progress_q()

        # 检查启动错误
        try:
            err = _boot_error_q.get_nowait()
            _row("启动错误", _err(str(err)[:100]), False)
            return 0
        except queue.Empty:
            pass

        attempts += 1
        try:
            resp = urllib.request.urlopen(urllib.request.Request(health_url), timeout=1)
            if resp.status == 200:
                body = json.loads(resp.read())
                svc = body.get("service", "?")
                mdl = body.get("model", "?")
                _row("HTTP 服务器", _ok("已响应"), True)
                _row("健康检查", _ok("通过"), True)
                _sub("Service", _c(_BWHITE, str(svc)))
                _sub("Model", _c(_BWHITE, str(mdl)))
                _sub("Session", _c(_BCYAN, str(body.get("session_id", ""))))
                _backend_port = port
                return port
        except Exception:
            pass

        if attempts == 1 or attempts % 30 == 0:
            _row("健康检查", _label(f"轮询中（{attempts} 次）…"))

        time.sleep(HEALTH_POLL_INTERVAL)

    _row("健康检查", _err("超时"), False)
    # 尝试获取错误信息
    try:
        err = _boot_error_q.get_nowait()
        _row("错误详情", _err(str(err)[:120]), False)
    except queue.Empty:
        pass
    return 0


# ── Phase 4: Tauri 启动 ──

def _find_tauri_cli(env: EnvReport) -> str | None:
    candidate = env.desktop_dir / "node_modules" / ".bin" / "tauri"
    if candidate.exists():
        return str(candidate)
    return None


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
    _section("[4/4]", "启动前端 (Tauri)")

    tauri_cli = _find_tauri_cli(env)
    vite_bin = str(env.desktop_dir / "node_modules" / ".bin" / "vite")
    if not tauri_cli or not Path(vite_bin).exists():
        _row("依赖", _err("缺失"), False)
        _sub("修复", _c(_CYAN, f"cd {env.desktop_dir} && npm install"))
        return 1

    _row("Tauri CLI", _c(_BWHITE, tauri_cli))
    _row("窗口协议", _ok("Wayland") if _detect_wayland(env) else _label("X11 / 默认"))

    # 步骤 1：后台启动 Vite 开发服务器，tauri dev 需要它先就绪。
    _row("Vite dev server", _label("启动 localhost:5173 …"))
    try:
        vite_proc = subprocess.Popen(
            [vite_bin],
            cwd=str(env.desktop_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={**os.environ, "CYBER_AGENT_BACKEND_PORT": str(backend_port)},
        )
    except Exception as e:
        _row("Vite", _err(f"启动失败: {e}"), False)
        return 1

    # 等待 Vite 就绪。
    vite_ready = False
    for _ in range(30):
        try:
            resp = urllib.request.urlopen("http://localhost:5173", timeout=0.5)
            if resp.status == 200:
                vite_ready = True
                break
        except Exception:
            time.sleep(0.3)
    if vite_ready:
        _row("Vite dev server", _ok("已就绪"), True)
    else:
        _row("Vite dev server", _err("超时"), False)
        vite_proc.kill()
        return 1

    # 步骤 2：启动 Tauri。
    _row("启动 Tauri", _label("正在编译并启动 …"))
    print()

    try:
        tp = subprocess.Popen(
            [tauri_cli, "dev"],
            cwd=str(env.desktop_dir),
            env={**os.environ, "CYBER_AGENT_BACKEND_PORT": str(backend_port)},
        )
        rc = tp.wait()
        vite_proc.terminate()
        vite_proc.wait()
        return rc
    except FileNotFoundError:
        _row("Tauri", _err("未找到可执行文件"), False)
        vite_proc.terminate()
        print(f" {_ide()}  {_label('请运行: cd desktop && npm install')}")
        return 1
    except KeyboardInterrupt:
        vite_proc.terminate()
        vite_proc.wait()
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
