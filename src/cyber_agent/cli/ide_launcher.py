"""IDE 一键启动器：环境检测 → 依赖安装 → 后端启动 → 前端启动。"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# ── 常量 ──

DEFAULT_IDE_HOST = "127.0.0.1"
BACKEND_STARTUP_TIMEOUT = 60  # AgentRunner 初始化可能较慢（模型导入等）
HEALTH_POLL_INTERVAL = 0.3
NPM_INSTALL_TIMEOUT = 300  # 5 分钟


# ── 工具函数 ──

def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _which(cmd: str) -> str | None:
    """查找可执行文件路径。"""
    for p in os.environ.get("PATH", "").split(os.pathsep):
        for name in (cmd, f"{cmd}.exe", f"{cmd}.cmd"):
            candidate = Path(p) / name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
    return None


def _get_desktop_dir() -> Path:
    """获取 desktop/ 目录的绝对路径。"""
    return (Path(__file__).resolve().parent.parent.parent.parent / "desktop").resolve()


def _get_project_root() -> Path:
    """获取项目根目录。"""
    return Path(__file__).resolve().parent.parent.parent.parent


# ── Phase 1: 环境检测 ──

class EnvReport:
    def __init__(self):
        self.node_bin: str | None = None
        self.node_version: str = ""
        self.npm_bin: str | None = None
        self.npm_version: str = ""
        self.python_bin: str = sys.executable
        self.os_name: str = sys.platform
        self.desktop_dir: Path = _get_desktop_dir()
        self.node_modules_exists: bool = False
        self.electron_bin: str | None = None
        self.vite_dev_server_running: bool = False
        self.errors: list[str] = []
        self.warnings: list[str] = []


def _check_node(env: EnvReport) -> bool:
    """检测 Node.js 运行时。"""
    node = _which("node")
    if not node:
        env.errors.append("未检测到 Node.js。请安装 Node.js >= 18: https://nodejs.org")
        return False
    env.node_bin = node
    try:
        result = subprocess.run([node, "-v"], capture_output=True, text=True, timeout=10)
        env.node_version = result.stdout.strip()
    except Exception:
        env.errors.append("Node.js 无法执行。")
        return False
    return True


def _check_npm(env: EnvReport) -> bool:
    """检测 npm。"""
    npm = _which("npm")
    if not npm:
        env.errors.append("未检测到 npm（通常随 Node.js 一起安装）。")
        return False
    env.npm_bin = npm
    try:
        result = subprocess.run([npm, "-v"], capture_output=True, text=True, timeout=10)
        env.npm_version = result.stdout.strip()
    except Exception:
        env.errors.append("npm 无法执行。")
        return False
    return True


def _check_desktop_dir(env: EnvReport) -> bool:
    """检测 desktop/ 目录和 package.json。"""
    if not env.desktop_dir.exists():
        env.errors.append(f"desktop/ 目录不存在: {env.desktop_dir}")
        return False
    pkg = env.desktop_dir / "package.json"
    if not pkg.exists():
        env.errors.append(f"desktop/package.json 不存在，请先初始化前端项目。")
        return False
    return True


def _check_node_modules(env: EnvReport) -> bool:
    """检测依赖是否已安装。"""
    nm = env.desktop_dir / "node_modules"
    if nm.is_dir():
        # 检查关键依赖是否存在
        electron = env.desktop_dir / "node_modules" / ".bin" / "electron"
        vite = env.desktop_dir / "node_modules" / ".bin" / "vite"
        if electron.exists() or vite.exists():
            env.node_modules_exists = True
            env.electron_bin = str(electron) if electron.exists() else None
            return True
    env.warnings.append("前端依赖未安装。")
    return False


def _check_vite_dev(env: EnvReport) -> bool:
    """检测 Vite 开发服务器是否已在运行。"""
    try:
        resp = urllib.request.urlopen("http://localhost:5173", timeout=1)
        env.vite_dev_server_running = resp.status == 200
        return True
    except Exception:
        return False


def detect_environment() -> EnvReport:
    """Phase 1: 完整环境检测。"""
    env = EnvReport()
    _check_node(env)
    _check_npm(env)
    _check_desktop_dir(env)
    _check_node_modules(env)
    _check_vite_dev(env)
    return env


def print_env_report(env: EnvReport) -> None:
    """打印环境检测报告。"""
    print("╔══════════════════════════════════════╗")
    print("║   Cyber Agent IDE — 环境检测         ║")
    print("╚══════════════════════════════════════╝")
    print(f"  系统:        {env.os_name}")
    print(f"  Python:      {sys.version.split()[0]}")
    print(f"  Node.js:     {env.node_version or '❌ 未安装'}")
    print(f"  npm:         {env.npm_version or '❌ 未安装'}")
    print(f"  desktop/:    {'✓' if env.desktop_dir.exists() else '✗ 不存在'}")
    print(f"  依赖:        {'✓ 已安装' if env.node_modules_exists else '○ 待安装'}")
    print(f"  Electron:    {env.electron_bin or '○ 待安装'}")
    print(f"  Vite Dev:    {'✓ 运行中' if env.vite_dev_server_running else '○ 未启动'}")
    if env.warnings:
        for w in env.warnings:
            print(f"  ⚠ {w}")
    if env.errors:
        for e in env.errors:
            print(f"  ✗ {e}")
    print()


# ── Phase 2: 依赖安装 ──

def install_dependencies(env: EnvReport, force: bool = False) -> bool:
    """Phase 2: 自动安装前端依赖 (npm install)。"""
    if env.node_modules_exists and not force:
        print("[IDE] 依赖已安装，跳过 npm install。")
        return True

    if not env.npm_bin:
        print("[IDE] ✗ 未找到 npm，无法安装依赖。")
        return False

    print(f"[IDE] 正在安装前端依赖...")
    print(f"[IDE] 目录: {env.desktop_dir}")
    print(f"[IDE] 这可能需要几分钟（首次安装需下载 Electron ~180MB）...")

    try:
        result = subprocess.run(
            [env.npm_bin, "install"],
            cwd=str(env.desktop_dir),
            capture_output=False,  # 实时输出
            text=True,
            timeout=NPM_INSTALL_TIMEOUT,
            env={
                **os.environ,
                "ELECTRON_MIRROR": os.environ.get(
                    "ELECTRON_MIRROR",
                    "https://npmmirror.com/mirrors/electron/",
                ),
            },
        )
        if result.returncode != 0:
            print("[IDE] ✗ npm install 失败。")
            print("[IDE] 请手动执行: cd desktop && npm install")
            return False
    except subprocess.TimeoutExpired:
        print("[IDE] ✗ npm install 超时。请手动执行: cd desktop && npm install")
        return False
    except FileNotFoundError:
        print("[IDE] ✗ 未找到 npm。")
        return False

    # 验证安装
    if not env.node_modules_exists:
        nm = env.desktop_dir / "node_modules"
        if nm.is_dir():
            env.node_modules_exists = True
            electron = env.desktop_dir / "node_modules" / ".bin" / "electron"
            if electron.exists():
                env.electron_bin = str(electron)

    print("[IDE] ✓ 依赖安装完成。")
    return True


# ── Phase 3: 后端启动（in-process daemon thread） ──

_backend_thread: threading.Thread | None = None
_backend_port: int = 0
_backend_ready: threading.Event = threading.Event()


def _start_backend_in_thread(
    host: str = "127.0.0.1",
    port: int = 0,
    mode: str = "standard",
    allow_paths: list[str] | None = None,
    approval_policy: str = "prompt",
    service_name: str | None = None,
    model_name: str | None = None,
) -> int:
    """在 daemon 线程中启动 FastAPI 后端，返回实际端口。"""
    global _backend_thread, _backend_port, _backend_ready

    if port == 0:
        port = _find_free_port()

    _backend_port = port
    _backend_ready.clear()

    def _run():
        import uvicorn
        from .ide_server import build_ide_runtime_context, init_runner, create_app

        runtime_context = build_ide_runtime_context(
            mode=mode, allow_paths=allow_paths,
            approval_policy=approval_policy,
            service_name=service_name, model_name=model_name,
        )
        init_runner(runtime_context)
        app = create_app()

        # 向 uvicorn 注入启动回调
        class ReadyServer(uvicorn.Server):
            def started(self):
                _backend_ready.set()

        config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        server = ReadyServer(config)

        try:
            server.run()
        except Exception:
            pass

    _backend_thread = threading.Thread(target=_run, daemon=True)
    _backend_thread.start()

    # 等待就绪（基于事件 + HTTP 健康检查双重确认）
    return _wait_backend_ready(host, port)


def _wait_backend_ready(host: str, port: int) -> int:
    """等待后端就绪（双重确认：事件 + HTTP 健康检查）。"""
    start_time = time.time()

    # 先等 uvicorn started 事件
    if not _backend_ready.wait(timeout=BACKEND_STARTUP_TIMEOUT):
        print("[IDE] ✗ 后端服务器启动超时（uvicorn 未就绪）。")
        return 0

    # 再通过 HTTP 健康检查确认 AgentRunner 已就绪
    health_url = f"http://{host}:{port}/api/health"
    while time.time() - start_time < BACKEND_STARTUP_TIMEOUT:
        try:
            req = urllib.request.Request(health_url)
            resp = urllib.request.urlopen(req, timeout=1)
            if resp.status == 200:
                _backend_port = port
                return port
        except Exception:
            pass
        time.sleep(HEALTH_POLL_INTERVAL)

    print("[IDE] ✗ 后端启动超时（HTTP 健康检查未通过）。")
    return 0


def _stop_backend():
    """停止后台运行的后端。"""
    global _backend_thread
    # daemon 线程会在主进程退出时自动终止
    # 但我们可以尝试优雅关闭 uvicorn
    if _backend_thread and _backend_thread.is_alive():
        # daemon 线程会自动清理
        pass


# ── Phase 4: Electron 启动 ──

def _find_electron(env: EnvReport) -> str | None:
    """多策略查找 Electron 可执行文件。"""
    # 1. 已缓存的路径
    if env.electron_bin and Path(env.electron_bin).exists():
        return env.electron_bin

    # 2. node_modules/.bin/
    candidate = env.desktop_dir / "node_modules" / ".bin" / "electron"
    if candidate.exists():
        env.electron_bin = str(candidate)
        return str(candidate)

    # 3. 系统 PATH
    system_electron = _which("electron")
    if system_electron:
        env.electron_bin = system_electron
        return system_electron

    # 4. npx electron
    if env.npm_bin:
        npx = str(Path(env.npm_bin).parent / "npx")
        if Path(npx).exists():
            env.electron_bin = npx
            return npx

    return None


def _get_electron_flags() -> list[str]:
    """获取平台特定的 Electron 启动参数。"""
    flags = []

    if sys.platform == "linux":
        # Wayland 检测：XDG_SESSION_TYPE 或 WAYLAND_DISPLAY
        if os.environ.get("WAYLAND_DISPLAY") or os.environ.get("XDG_SESSION_TYPE") == "wayland":
            flags.extend([
                "--enable-features=UseOzonePlatform",
                "--ozone-platform=wayland",
                "--enable-gpu-rasterization",
            ])

    return flags


def _build_electron_env(backend_port: int) -> dict:
    """构建 Electron 子进程环境变量。"""
    return {
        **os.environ,
        "CYBER_AGENT_BACKEND_PORT": str(backend_port),
        "ELECTRON_OZONE_PLATFORM_HINT": os.environ.get("XDG_SESSION_TYPE", ""),
    }


def launch_ide(
    mode: str = "standard",
    allow_paths: list[str] | None = None,
    approval_policy: str = "prompt",
    service_name: str | None = None,
    model_name: str | None = None,
    dev: bool = False,
    skip_install: bool = False,
) -> int:
    """一键启动 IDE。

    流程: 环境检测 → 依赖安装 → 后端启动 → 前端启动

    Args:
        mode: Agent 运行模式 (standard / authorized)
        allow_paths: 授权模式下的额外路径
        approval_policy: 审批策略 (prompt / auto / never)
        service_name: 模型服务商
        model_name: 模型名称
        dev: 开发模式（使用 Vite dev server 而非构建产物）
        skip_install: 跳过依赖安装

    Returns:
        进程退出码 (0 = 正常退出)
    """
    print()
    print("╔══════════════════════════════════════╗")
    print("║   Cyber Agent IDE                     ║")
    print("║   Liquid Glass · 一键启动              ║")
    print("╚══════════════════════════════════════╝")
    print()

    # ── Phase 1: 环境检测 ──
    print("[1/4] 环境检测...")
    env = detect_environment()
    print_env_report(env)

    if env.errors:
        print("[IDE] ✗ 环境检测失败，请修复上述错误后重试。")
        return 1

    # ── Phase 2: 依赖安装 ──
    if not skip_install and not env.node_modules_exists:
        print("[2/4] 安装前端依赖...")
        if not install_dependencies(env):
            print("[IDE] ✗ 依赖安装失败。")
            return 1
        # 刷新检测结果
        env = detect_environment()
    else:
        print(f"[2/4] 依赖: {'✓ 已安装' if env.node_modules_exists else '○ 已跳过'}")

    # ── Phase 3: 启动后端 ──
    print("[3/4] 启动后端服务器...")
    backend_port = _start_backend_in_thread(
        host=DEFAULT_IDE_HOST,
        mode=mode,
        allow_paths=allow_paths,
        approval_policy=approval_policy,
        service_name=service_name,
        model_name=model_name,
    )

    if backend_port == 0:
        print("[IDE] ✗ 后端启动失败。")
        return 1

    print(f"[IDE] ✓ 后端就绪: http://{DEFAULT_IDE_HOST}:{backend_port}")

    # ── Phase 4: 启动前端 ──
    print("[4/4] 启动前端...")

    is_dev = dev or (not (env.desktop_dir / "dist" / "renderer" / "index.html").exists())

    if is_dev:
        # 开发模式：检测 Vite 是否在运行
        if not env.vite_dev_server_running:
            print("[IDE] 开发模式: 启动 Vite 开发服务器...")
            print(f"[IDE] 请在另一个终端执行: cd {env.desktop_dir} && npx vite")
            print(f"[IDE] 后端运行中: http://{DEFAULT_IDE_HOST}:{backend_port}")
            print("[IDE] 按 Ctrl+C 退出...")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                return 0

    # 查找 Electron
    electron_bin = _find_electron(env)
    if not electron_bin or electron_bin == "npx":
        print("[IDE] 未找到 Electron，尝试 npx electron...")
        electron_bin = "npx"
        electron_args = ["electron"]
    else:
        electron_args = []

    electron_args.extend([
        str(env.desktop_dir),
        f"--backend-port={backend_port}",
    ])
    electron_args.extend(_get_electron_flags())

    electron_env = _build_electron_env(backend_port)

    print(f"[IDE] 启动 Electron...")
    print(f"[IDE] 后端: http://{DEFAULT_IDE_HOST}:{backend_port}")
    print(f"[IDE] 按 Ctrl+C 退出。")
    print()

    try:
        electron_process = subprocess.Popen(
            [electron_bin] + electron_args,
            cwd=str(env.desktop_dir),
            env=electron_env,
        )
        # 等待 Electron 退出
        exit_code = electron_process.wait()
        return exit_code
    except FileNotFoundError:
        print(f"[IDE] ✗ 未找到 Electron。请运行: cd desktop && npm install")
        return 1
    except KeyboardInterrupt:
        print("\n[IDE] 正在退出...")
        return 0


# ── 独立后端入口 (python -m cyber_agent.cli.ide_server) ──

def run_ide_server_main() -> None:
    """独立的 IDE 后端入口。"""
    import argparse

    parser = argparse.ArgumentParser(description="Cyber Agent IDE Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--mode", default="standard")
    parser.add_argument("--approval-policy", default="prompt")
    parser.add_argument("--allow-path", action="append", dest="allow_paths", default=None)
    parser.add_argument("--service", default=None)
    parser.add_argument("--model", default=None)

    args = parser.parse_args()

    from .ide_server import run_ide_server

    run_ide_server(
        host=args.host,
        port=args.port,
        mode=args.mode,
        allow_paths=args.allow_paths,
        approval_policy=args.approval_policy,
        service_name=args.service,
        model_name=args.model,
    )


if __name__ == "__main__":
    run_ide_server_main()
