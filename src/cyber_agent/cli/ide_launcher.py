"""IDE 启动引导器 —— 检查依赖、构建前端、启动后端并在浏览器打开。"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any

if sys.stdout.isatty():
    try:
        from rich.console import Console
        from rich.panel import Panel
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
        color = "green" if ok else "red" if symbol == CROSS_MARK else "yellow"
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


# ═══════════════════════════════════════════════════════════════
#  Step 1: Python 包
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
    if missing:
        _hint(f"缺少 {len(missing)} 个包，自动安装...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--break-system-packages", *missing],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", *missing],
                capture_output=True, text=True,
            )
        if result.returncode != 0:
            for pkg in missing:
                _status(f"Python: {pkg}", CROSS_MARK, "安装失败", ok=False)
            return False
    return True


# ═══════════════════════════════════════════════════════════════
#  Step 2: Node.js
# ═══════════════════════════════════════════════════════════════

def check_nodejs() -> bool:
    node = shutil.which("node")
    npm = shutil.which("npm")
    if node:
        v = subprocess.run([node, "--version"], capture_output=True, text=True).stdout.strip()
        _status("Node.js", CHECK_MARK, f"v{v}")
    else:
        _status("Node.js", CROSS_MARK, "未安装（构建前端需要）", ok=False)
        _hint("sudo pacman -S nodejs npm  或  https://nodejs.org")
        return False
    if npm:
        v = subprocess.run([npm, "--version"], capture_output=True, text=True).stdout.strip()
        _status("npm", CHECK_MARK, f"v{v}")
    return bool(node and npm)


# ═══════════════════════════════════════════════════════════════
#  Step 3: 前端依赖
# ═══════════════════════════════════════════════════════════════

def check_frontend_deps() -> bool:
    node_modules = DESKTOP_DIR / "node_modules"
    if not (DESKTOP_DIR / "package.json").exists():
        _status("前端项目", CROSS_MARK, "缺少 package.json", ok=False)
        return False
    if node_modules.exists() and list(node_modules.iterdir()):
        _status("前端依赖", CHECK_MARK, "已安装")
        return True
    _status("前端依赖", CROSS_MARK, "未安装，运行 npm install...", ok=False)
    result = subprocess.run(["npm", "install"], cwd=str(DESKTOP_DIR), capture_output=True, text=True)
    if result.returncode == 0:
        _status("前端依赖", CHECK_MARK, "安装完成")
        return True
    _status("前端依赖", CROSS_MARK, "安装失败", ok=False)
    return False


# ═══════════════════════════════════════════════════════════════
#  Step 4: 构建前端
# ═══════════════════════════════════════════════════════════════

def build_frontend() -> bool:
    """Vite 构建前端 dist/"""
    _status("前端构建", GEAR_MARK, "npm run build ...")
    result = subprocess.run(["npm", "run", "build"], cwd=str(DESKTOP_DIR), capture_output=True, text=True)
    if result.returncode != 0:
        _status("前端构建", CROSS_MARK, "失败", ok=False)
        _hint(result.stderr.strip()[-300:])
        return False
    _status("前端构建", CHECK_MARK, "完成")
    return True


# ═══════════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════════

def launch_ide(
    runtime_context: dict[str, object],
    *,
    skip_build: bool = False,
    open_browser: bool = True,
) -> None:
    # 依赖检查
    if not check_python_deps():
        sys.exit(1)

    node_ok = check_nodejs()

    if node_ok:
        if not check_frontend_deps():
            _hint(f"请手动: cd {DESKTOP_DIR} && npm install")
            node_ok = False

    if node_ok and not skip_build:
        if not build_frontend():
            node_ok = False

    if not node_ok:
        _hint("跳过前端构建。如前端未构建，请手动: cd desktop && npm run build")

    # 导入并初始化后端
    _status("Python 后端", GEAR_MARK, "初始化 AgentRunner...")
    from .ide_server import run_ide_server, _find_free_port
    from .app import ensure_runtime_capabilities, create_runner
    import cyber_agent.cli.ide_server as ide_mod

    ensure_runtime_capabilities(runtime_context)
    runner = create_runner(runtime_context)
    ide_mod._RUNTIME_CTX = runtime_context
    ide_mod._RUNNER = runner

    port = _find_free_port()
    _status("API 服务器", CHECK_MARK, f"端口 {port}")

    _header(f"{ROCKET_MARK}  启动 IDE")
    url = f"http://127.0.0.1:{port}"
    _status("IDE 地址", GEAR_MARK, url)
    _status("WebSocket", GEAR_MARK, f"ws://127.0.0.1:{port}/ws/chat")

    if open_browser:
        _status("浏览器", GEAR_MARK, "正在打开...")
        time.sleep(1)
        webbrowser.open(url)

    _hint("按 Ctrl+C 停止服务器")
    print()
    run_ide_server(runtime_context, host="127.0.0.1", port=port)
