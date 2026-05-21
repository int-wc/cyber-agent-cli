"""IDE 启动引导器：管理 FastAPI 后端 + Electron 前端的启动流程。"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

DEFAULT_IDE_HOST = "127.0.0.1"
DEFAULT_IDE_PORT = 0  # OS 自动分配
BACKEND_STARTUP_TIMEOUT = 30
ELECTRON_STARTUP_TIMEOUT = 60


def _find_free_port() -> int:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _find_electron_binary() -> str | None:
    """查找 Electron 可执行文件（开发模式 vs 打包模式）。"""
    # 打包模式：Electron 与 Python 包一起安装
    package_dir = Path(__file__).resolve().parent.parent.parent.parent
    packaged_electron = package_dir / "desktop" / "node_modules" / ".bin" / "electron"
    if packaged_electron.exists():
        return str(packaged_electron)

    # 开发模式：在项目根目录查找
    project_root = package_dir
    dev_electron = project_root / "desktop" / "node_modules" / ".bin" / "electron"
    if dev_electron.exists():
        return str(dev_electron)

    # 系统级 electron
    for path in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(path) / "electron"
        if candidate.exists():
            return str(candidate)

    return None


def _find_npx() -> str:
    """查找 npx 可执行文件。"""
    for path in os.environ.get("PATH", "").split(os.pathsep):
        for name in ("npx", "npx.cmd"):
            candidate = Path(path) / name
            if candidate.exists():
                return str(candidate)
    return "npx"


def _is_dev_mode() -> bool:
    """检测是否处于开发模式（存在 desktop/package.json）。"""
    package_dir = Path(__file__).resolve().parent.parent.parent.parent
    return (package_dir / "desktop" / "package.json").exists()


def launch_ide(
    mode: str = "standard",
    allow_paths: list[str] | None = None,
    approval_policy: str = "prompt",
    service_name: str | None = None,
    model_name: str | None = None,
    dev: bool = False,
) -> int:
    """启动完整的 IDE（后端 + 前端）。

    Returns:
        进程退出码。
    """
    # 1. 启动 FastAPI 后端
    backend_port = _find_free_port()

    python = sys.executable
    backend_args = [
        python, "-m", "cyber_agent.cli.ide_server",
        "--host", DEFAULT_IDE_HOST,
        "--port", str(backend_port),
        "--mode", mode,
        "--approval-policy", approval_policy,
    ]
    if service_name:
        backend_args.extend(["--service", service_name])
    if model_name:
        backend_args.extend(["--model", model_name])
    if allow_paths:
        backend_args.extend(["--allow-path"] + allow_paths)

    print(f"[IDE] 启动后端服务器 (127.0.0.1:{backend_port})...")
    backend_process = subprocess.Popen(
        backend_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )

    # 等待后端就绪（解析 IDE_SERVER_PORT 或健康检查）
    backend_ready = False
    start_time = time.time()
    while time.time() - start_time < BACKEND_STARTUP_TIMEOUT:
        if backend_process.poll() is not None:
            stderr_output = backend_process.stderr.read() if backend_process.stderr else ""
            print(f"[IDE] 后端进程意外退出 (code={backend_process.returncode})")
            if stderr_output:
                print(f"[IDE] stderr: {stderr_output[:500]}")
            return 1

        line = backend_process.stdout.readline()
        if line:
            line = line.strip()
            if line.startswith("IDE_SERVER_PORT="):
                actual_port = int(line.split("=", 1)[1])
                if actual_port != backend_port:
                    backend_port = actual_port
                backend_ready = True
                break

    if not backend_ready:
        print("[IDE] 后端启动超时")
        backend_process.terminate()
        return 1

    print(f"[IDE] 后端就绪: http://{DEFAULT_IDE_HOST}:{backend_port}")

    # 2. 启动 Electron 前端
    electron_bin = _find_electron_binary()
    is_dev = dev or _is_dev_mode()

    if electron_bin is None:
        print("[IDE] 未找到 Electron，尝试使用 npx electron...")
        electron_bin = "electron"

    package_dir = Path(__file__).resolve().parent.parent.parent.parent
    desktop_dir = package_dir / "desktop"

    if not desktop_dir.exists():
        print("[IDE] desktop/ 目录不存在，请先初始化前端项目")
        backend_process.terminate()
        return 1

    if is_dev:
        # 开发模式：需要先启动 Vite dev server，然后 Electron 加载 localhost
        print("[IDE] 开发模式：启动 Vite + Electron")
        os.environ["CYBER_AGENT_BACKEND_PORT"] = str(backend_port)

        # 检查 Vite 是否已经在运行
        import urllib.request
        vite_ready = False
        try:
            urllib.request.urlopen("http://localhost:5173", timeout=1)
            vite_ready = True
        except Exception:
            pass

        if not vite_ready and not dev:
            print("[IDE] 请先在 desktop/ 目录运行: npm run dev")
            print(f"[IDE] 后端运行中: http://{DEFAULT_IDE_HOST}:{backend_port}")
            print("[IDE] 按 Ctrl+C 停止后端")
            try:
                backend_process.wait()
            except KeyboardInterrupt:
                pass
            backend_process.terminate()
            return 0

        electron_args = [
            electron_bin,
            str(desktop_dir),
            f"--backend-port={backend_port}",
        ]
    else:
        # 打包模式
        os.environ["CYBER_AGENT_BACKEND_PORT"] = str(backend_port)
        electron_args = [
            electron_bin,
            str(desktop_dir),
            f"--backend-port={backend_port}",
        ]

    print(f"[IDE] 启动 Electron: {' '.join(electron_args)}")
    try:
        electron_process = subprocess.Popen(
            electron_args,
            cwd=str(desktop_dir),
            env={**os.environ, "CYBER_AGENT_BACKEND_PORT": str(backend_port)},
        )

        # 等待 Electron 退出
        electron_process.wait()
    except FileNotFoundError:
        print(f"[IDE] 未找到 Electron，请安装: cd desktop && npm install")
        backend_process.terminate()
        return 1
    except KeyboardInterrupt:
        pass
    finally:
        backend_process.terminate()
        try:
            backend_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            backend_process.kill()

    return 0


def run_ide_server_main() -> None:
    """独立的 IDE 后端入口（python -m cyber_agent.cli.ide_server）。"""
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
