"""IDE 启动引导器 —— 按步骤检查依赖、安装、构建并启动桌面 IDE。"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if sys.stdout.isatty():
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
        from rich import box
        _RICH = True
        _console = Console()
    except ImportError:
        _RICH = False
        _console = None  # type: ignore[assignment]
else:
    _RICH = False
    _console = None

DESKTOP_DIR = Path(__file__).resolve().parent.parent.parent.parent / "desktop"
REQUIRED_PYTHON_PKGS = ["fastapi", "uvicorn"]

CHECK_MARK = "✓"
CROSS_MARK = "✗"
GEAR_MARK = "⚙"
ROCKET_MARK = "🚀"


def _status(step: str, symbol: str, detail: str, ok: bool = True):
    if _RICH and _console:
        color = "green" if ok else "red"
        text = Text()
        text.append(f"  {symbol}  ", style=f"bold {color}")
        text.append(f"{step:<24} ", style="bold")
        text.append(detail, style="dim")
        _console.print(text)
    else:
        print(f"  {symbol}  {step:<24} {detail}")


def _header(title: str):
    if _RICH and _console:
        _console.print()
        _console.print(Panel(title, box=box.HEAVY, border_style="bold cyan", padding=(0, 3)))
        _console.print()
    else:
        print(f"\n{'='*50}")
        print(f"  {title}")
        print(f"{'='*50}\n")


def _hint(text: str):
    if _RICH and _console:
        _console.print(f"    {text}", style="dim yellow")
    else:
        print(f"    {text}")


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


# ═══════════════════════════════════════════════════════════════
#  Step 1: 检查 Python 依赖
# ═══════════════════════════════════════════════════════════════

def check_python_deps() -> bool:
    _header("Cyber Agent IDE — 启动引导")
    missing: list[str] = []
    for pkg in REQUIRED_PYTHON_PKGS:
        try:
            __import__(pkg)
            _status(f"Python: {pkg}", CHECK_MARK, "已安装")
        except ImportError:
            _status(f"Python: {pkg}", CROSS_MARK, "未安装", ok=False)
            missing.append(pkg)

    if not missing:
        return True

    _hint(f"缺少 {len(missing)} 个 Python 包，正在自动安装...")

    pip_cmd = [sys.executable, "-m", "pip", "install", "--break-system-packages", *missing]
    result = _run(pip_cmd)
    if result.returncode != 0:
        # 尝试不用 break-system-packages
        result = _run([sys.executable, "-m", "pip", "install", *missing])

    if result.returncode == 0:
        for pkg in missing:
            _status(f"Python: {pkg}", CHECK_MARK, "安装完成")
        return True
    else:
        for pkg in missing:
            _status(f"Python: {pkg}", CROSS_MARK, f"安装失败: {result.stderr.strip()[-100:]}", ok=False)
        _hint(f"请手动运行: pip install {' '.join(missing)}")
        return False


# ═══════════════════════════════════════════════════════════════
#  Step 2: 检查 Node.js
# ═══════════════════════════════════════════════════════════════

def check_nodejs() -> bool:
    node_path = shutil.which("node")
    npm_path = shutil.which("npm")

    if node_path:
        result = _run([node_path, "--version"])
        version = result.stdout.strip()
        _status("Node.js", CHECK_MARK, f"v{version} ({node_path})")
    else:
        _status("Node.js", CROSS_MARK, "未安装", ok=False)
        _hint("请安装 Node.js: https://nodejs.org/ 或 sudo pacman -S nodejs npm")
        return False

    if npm_path:
        result = _run([npm_path, "--version"])
        version = result.stdout.strip()
        _status("npm", CHECK_MARK, f"v{version}")
    else:
        _status("npm", CROSS_MARK, "未安装", ok=False)
        return False

    return True


# ═══════════════════════════════════════════════════════════════
#  Step 3: 检查前端依赖
# ═══════════════════════════════════════════════════════════════

def check_frontend_deps() -> bool:
    node_modules = DESKTOP_DIR / "node_modules"
    package_json = DESKTOP_DIR / "package.json"

    if not package_json.exists():
        _status("前端项目", CROSS_MARK, f"缺少 {DESKTOP_DIR}/package.json", ok=False)
        _hint("请确认 desktop/ 目录存在且包含 package.json")
        return False

    if node_modules.exists() and list(node_modules.iterdir()):
        _status("前端依赖 (node_modules)", CHECK_MARK, "已安装")
        return True

    _status("前端依赖 (node_modules)", CROSS_MARK, "未安装", ok=False)
    _hint("正在运行 npm install (首次安装可能需要几分钟)...")

    result = _run(["npm", "install"], cwd=str(DESKTOP_DIR))
    if result.returncode == 0:
        _status("前端依赖", CHECK_MARK, "安装完成")
        return True
    else:
        _status("前端依赖", CROSS_MARK, f"安装失败: {result.stderr.strip()[-100:]}", ok=False)
        _hint(f"请手动运行: cd {DESKTOP_DIR} && npm install")
        return False


# ═══════════════════════════════════════════════════════════════
#  Step 4: 检查 Rust 工具链
# ═══════════════════════════════════════════════════════════════

def check_rust() -> bool:
    cargo_path = shutil.which("cargo")
    rustc_path = shutil.which("rustc")

    if rustc_path:
        result = _run([rustc_path, "--version"])
        _status("Rust (rustc)", CHECK_MARK, result.stdout.strip().split("(")[0].strip())
    else:
        _status("Rust (rustc)", CROSS_MARK, "未安装", ok=False)
        _hint("请安装 Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh")
        return False

    if cargo_path:
        _status("Cargo", CHECK_MARK, f"已就绪 ({cargo_path})")
    else:
        _status("Cargo", CROSS_MARK, "未安装", ok=False)
        return False

    return True


# ═══════════════════════════════════════════════════════════════
#  Step 5: 检查 Tauri CLI
# ═══════════════════════════════════════════════════════════════

def check_tauri_cli() -> bool:
    result = _run(["npx", "tauri", "--version"], cwd=str(DESKTOP_DIR))
    if result.returncode == 0:
        _status("Tauri CLI", CHECK_MARK, result.stdout.strip())
        return True

    # Check if it's in node_modules
    tauri_cli = DESKTOP_DIR / "node_modules" / ".bin" / "tauri"
    if tauri_cli.exists():
        _status("Tauri CLI", CHECK_MARK, f"已安装 ({tauri_cli})")
        return True

    _status("Tauri CLI", GEAR_MARK, "正在安装 @tauri-apps/cli...")
    result = _run(["npm", "install", "-D", "@tauri-apps/cli@^2"], cwd=str(DESKTOP_DIR))
    if result.returncode == 0:
        _status("Tauri CLI", CHECK_MARK, "安装完成")
        return True
    else:
        _status("Tauri CLI", CROSS_MARK, f"安装失败", ok=False)
        _hint(f"请手动运行: cd {DESKTOP_DIR} && npm install -D @tauri-apps/cli")
        return False


# ═══════════════════════════════════════════════════════════════
#  Step 6: 检查系统依赖（仅 Linux）
# ═══════════════════════════════════════════════════════════════

def check_system_deps() -> bool:
    if not sys.platform.startswith("linux"):
        _status("系统依赖", CHECK_MARK, f"平台: {sys.platform}")
        return True

    missing_sys: list[str] = []
    for lib, pkg_name in [
        ("libwebkit2gtk", "webkit2gtk-4.1"),
        ("libgtk-3", "gtk3"),
        ("libayatana-appindicator", "libayatana-appindicator3"),
    ]:
        found = False
        for base in ["/usr/lib", "/usr/lib64", "/usr/lib/x86_64-linux-gnu"]:
            if list(Path(base).glob(f"{lib}*.so*")):
                found = True
                break
        if found:
            _status(f"系统: {pkg_name}", CHECK_MARK, "已安装")
        else:
            _status(f"系统: {pkg_name}", CROSS_MARK, "未安装", ok=False)
            missing_sys.append(pkg_name)

    if missing_sys:
        _hint(f"缺少系统库。Arch Linux: sudo pacman -S {' '.join(missing_sys)}")
        _hint(f"Ubuntu/Debian: sudo apt install {' '.join(m.replace('-4.1','-4.0-dev') for m in missing_sys)}")
        return False

    return True


# ═══════════════════════════════════════════════════════════════
#  Step 7: 构建前端 + Tauri 应用
# ═══════════════════════════════════════════════════════════════

def build_desktop_app() -> Path | None:
    """构建 Tauri 桌面应用。返回二进制路径或 None。"""
    tauri_target = DESKTOP_DIR / "src-tauri" / "target"

    # 1) 构建 Vite 前端
    _status("前端构建 (Vite)", GEAR_MARK, "正在构建...")
    result = _run(["npm", "run", "build"], cwd=str(DESKTOP_DIR))
    if result.returncode != 0:
        _status("前端构建 (Vite)", CROSS_MARK, "构建失败", ok=False)
        _hint(f"错误: {result.stderr.strip()[-200:]}")
        return None
    _status("前端构建 (Vite)", CHECK_MARK, "完成")

    # 2) 构建 Tauri
    _status("Tauri 构建", GEAR_MARK, "正在编译 Rust 桌面应用（首次编译需要几分钟）...")
    result = _run(
        ["npx", "tauri", "build", "--ci"],
        cwd=str(DESKTOP_DIR),
        env={**os.environ, "TAURI_PRIVATE_KEY": "", "TAURI_KEY_PASSWORD": ""},
    )
    if result.returncode != 0:
        # Check if binary already exists from a previous build
        _status("Tauri 构建", CROSS_MARK, "构建失败，尝试使用已有二进制...", ok=False)
    else:
        _status("Tauri 构建", CHECK_MARK, "完成")

    # Find the binary
    candidates = [
        tauri_target / "release" / "cyber-agent-ide",
        tauri_target / "debug" / "cyber-agent-ide",
        tauri_target / "release" / "cyber-agent-ide.exe",
        tauri_target / "debug" / "cyber-agent-ide.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            _status("桌面应用二进制", CHECK_MARK, str(candidate))
            return candidate

    _status("桌面应用二进制", CROSS_MARK, "未找到", ok=False)
    _hint("请手动运行: cd desktop && npx tauri build")
    return None


# ═══════════════════════════════════════════════════════════════
#  Step 8: 启动
# ═══════════════════════════════════════════════════════════════

def launch_ide(
    runtime_context: dict[str, object],
    *,
    skip_build: bool = False,
    require_rust: bool = True,
) -> None:
    """完整的 IDE 启动流程。"""

    # ── 依赖检查阶段 ──
    if not check_python_deps():
        sys.exit(1)

    if require_rust:
        if not check_rust():
            _hint("跳过 Rust 检查，仅启动后端服务器（前端需独立开发运行）")
            require_rust = False

    if require_rust:
        if not check_nodejs():
            sys.exit(1)
        if not check_system_deps():
            _hint("系统库缺失，IDE 窗口可能无法正常显示")
        if not check_frontend_deps():
            sys.exit(1)
        if not check_tauri_cli():
            sys.exit(1)

    # ── 导入后端模块 ──
    _status("Python 后端", GEAR_MARK, "正在初始化 AgentRunner...")
    try:
        from .ide_server import (
            run_ide_server,
            create_app,
            _RUNNER,
            _RUNTIME_CTX,
            _AGENT_EXECUTOR,
            _find_free_port,
        )
    except ImportError as e:
        _status("Python 后端", CROSS_MARK, f"导入失败: {e}", ok=False)
        sys.exit(1)

    # ── 初始化后端 ──
    from .app import ensure_runtime_capabilities, create_runner

    _RUNTIME_CTX_REF = runtime_context  # noqa: F841 (referenced via import)
    ensure_runtime_capabilities(runtime_context)
    runner = create_runner(runtime_context)

    # 注入到 ide_server 的全局变量
    import cyber_agent.cli.ide_server as ide_mod
    ide_mod._RUNTIME_CTX = runtime_context
    ide_mod._RUNNER = runner

    port = _find_free_port()
    _status("API 服务器", CHECK_MARK, f"端口 {port}")

    # ── 构建 + 启动 ──
    if require_rust and not skip_build:
        app_binary = build_desktop_app()
        if app_binary is None:
            _hint("桌面应用构建失败，回退为仅启动 API 服务器")
            require_rust = False

    _header(f"{ROCKET_MARK}  启动 IDE")
    _status("API 后端", GEAR_MARK, f"http://127.0.0.1:{port}/api")
    _status("WebSocket", GEAR_MARK, f"ws://127.0.0.1:{port}/ws/chat")

    if require_rust:
        # 在后台启动 Python API 服务器，然后启动 Tauri 应用
        import threading
        import uvicorn

        app = create_app()

        def run_backend():
            uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")

        backend_thread = threading.Thread(target=run_backend, daemon=True)
        backend_thread.start()
        time.sleep(1.5)

        _status("桌面应用", ROCKET_MARK, "正在启动...")
        result = _run(
            [str(app_binary)] if isinstance(app_binary, Path) else ["cargo", "run", "--release"],
            cwd=str(DESKTOP_DIR / "src-tauri"),
        )
        if result.returncode != 0:
            _status("桌面应用", CROSS_MARK, f"退出码: {result.returncode}", ok=False)
    else:
        # 仅启动 API 服务器
        _status("启动方式", GEAR_MARK, "仅 API 服务器模式")
        _hint("请在新终端运行: cd desktop && npm run dev")
        _hint("然后打开浏览器访问: http://localhost:1420")
        _hint("按 Ctrl+C 停止服务器")
        print()
        run_ide_server(runtime_context, host="127.0.0.1", port=port)
