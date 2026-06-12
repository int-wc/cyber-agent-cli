import os
import shutil
import subprocess
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

from ..execution_control import (
    ExecutionController,
    ExecutionInterruptedError,
    terminate_process_tree,
)
from .filesystem import normalize_allowed_roots, resolve_permitted_path
from .metadata import attach_tool_risk

MAX_COMMAND_OUTPUT_CHARS = 4000
MAX_TOOL_TIMEOUT_SECONDS = 120
PROCESS_POLL_INTERVAL_SECONDS = 0.1
# 输出静默超时：进程曾产生过输出，但此后连续无新输出的秒数。
# 用于检测后台进程 + 阻塞式后续命令（tail -f、无 timeout 的 curl 等）
# 导致的 shell 进程无法退出的情况。
_MAX_QUIET_SECONDS = 60


# ── 后台进程管理器 ──

class _ManagedProcess:
    """一个正在运行或已完成的后台进程，输出由 reader 线程持续捕获。"""

    __slots__ = (
        "handle", "label", "command", "cwd", "process",
        "started_at", "stdout_chunks", "stderr_chunks",
        "_stdout_thread", "_stderr_thread",
    )

    def __init__(
        self, handle: str, label: str, command: str,
        cwd: Path, process: subprocess.Popen,
    ) -> None:
        self.handle = handle
        self.label = label
        self.command = command
        self.cwd = cwd
        self.process = process
        self.started_at = time.monotonic()
        self.stdout_chunks, self._stdout_thread = _start_stream_reader(process.stdout)
        self.stderr_chunks, self._stderr_thread = _start_stream_reader(process.stderr)

    @property
    def is_running(self) -> bool:
        return self.process.poll() is None

    @property
    def exit_code(self) -> int | None:
        return self.process.returncode

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_at

    def read_output(self, tail: int | None = None) -> str:
        """返回累积输出，可选仅最近 N 行。"""
        stdout = "".join(self.stdout_chunks)
        stderr = "".join(self.stderr_chunks)
        combined = stdout + stderr
        if tail and tail > 0:
            lines = combined.splitlines()
            if len(lines) > tail:
                combined = "\n".join(lines[-tail:])
        return combined

    def close_pipes(self) -> None:
        """关闭管道，释放 reader 线程。"""
        if self.process.stdout and not self.process.stdout.closed:
            self.process.stdout.close()
        if self.process.stderr and not self.process.stderr.closed:
            self.process.stderr.close()

    def terminate(self) -> None:
        """终止进程树。"""
        if not self.is_running:
            return
        terminate_process_tree(self.process)
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
        self.close_pipes()


class ProcessManager:
    """全局后台进程注册表，跨子任务持久追踪。"""

    def __init__(self) -> None:
        self._processes: dict[str, _ManagedProcess] = {}
        self._lock = threading.Lock()

    # ── 公开接口 ──

    def start(
        self,
        command: str,
        cwd: Path,
        *,
        label: str = "",
    ) -> str:
        """启动一个后台进程，返回句柄。"""
        handle = f"proc:{label or 'bg'}:{uuid.uuid4().hex[:8]}"
        process = subprocess.Popen(
            _build_shell_command(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=cwd,
            shell=False,
            **_build_subprocess_options(),
        )
        mp = _ManagedProcess(handle, label or handle, command, cwd, process)
        with self._lock:
            self._processes[handle] = mp
        return handle

    def read_output(self, handle: str, tail: int | None = 50) -> str | None:
        """读取指定进程的累积输出。句柄不存在返回 None。"""
        mp = self._get(handle)
        if mp is None:
            return None
        return mp.read_output(tail)

    def get_status(self, handle: str) -> dict | None:
        """获取进程状态。"""
        mp = self._get(handle)
        if mp is None:
            return None
        return {
            "handle": mp.handle,
            "label": mp.label,
            "command": mp.command[:200],
            "cwd": str(mp.cwd),
            "running": mp.is_running,
            "exit_code": mp.exit_code,
            "elapsed_seconds": round(mp.elapsed_seconds, 1),
            "output_bytes": sum(len(c) for c in mp.stdout_chunks)
                          + sum(len(c) for c in mp.stderr_chunks),
        }

    def stop(self, handle: str) -> bool:
        """终止进程并从注册表中移除。"""
        mp = self._pop(handle)
        if mp is None:
            return False
        mp.terminate()
        return True

    def list_summary(self) -> list[dict]:
        """返回所有进程的摘要。"""
        with self._lock:
            return [
                {
                    "handle": mp.handle,
                    "label": mp.label,
                    "running": mp.is_running,
                    "exit_code": mp.exit_code,
                    "elapsed_seconds": round(mp.elapsed_seconds, 1),
                }
                for mp in self._processes.values()
            ]

    def cleanup_completed(self, max_age: float = 300.0) -> int:
        """清理已退出且建立超过 max_age 秒的进程记录。"""
        now = time.monotonic()
        stale = []
        with self._lock:
            for handle, mp in self._processes.items():
                if not mp.is_running and now - mp.started_at > max_age:
                    stale.append(handle)
            for h in stale:
                mp = self._processes.pop(h)
                mp.close_pipes()
        return len(stale)

    # ── 内部 ──

    def _get(self, handle: str) -> _ManagedProcess | None:
        with self._lock:
            return self._processes.get(handle)

    def _pop(self, handle: str) -> _ManagedProcess | None:
        with self._lock:
            return self._processes.pop(handle, None)


# 模块级单例
_process_manager = ProcessManager()


def get_process_manager() -> ProcessManager:
    """获取全局进程管理器实例。"""
    return _process_manager


def normalize_command_registry(
    command_registry: Mapping[str, Path | str],
) -> dict[str, Path]:
    """规范化外部工具注册表。"""
    normalized_registry: dict[str, Path] = {}
    for tool_name, executable_path in command_registry.items():
        normalized_registry[tool_name] = Path(executable_path).expanduser().resolve()
    return normalized_registry


def describe_command_registry(command_registry: Mapping[str, Path]) -> list[str]:
    """返回适合状态展示的外部工具列表。"""
    return [
        f"{tool_name}={executable_path}"
        for tool_name, executable_path in command_registry.items()
    ]


def _truncate_output(output: str) -> str:
    truncated_output = output[:MAX_COMMAND_OUTPUT_CHARS]
    if len(output) > MAX_COMMAND_OUTPUT_CHARS:
        truncated_output += "\n... 输出过长，已截断。"
    return truncated_output


def _format_completed_process_output(
    *,
    command_description: str,
    working_directory: Path,
    completed_process: subprocess.CompletedProcess[str],
) -> str:
    stdout = completed_process.stdout.strip()
    stderr = completed_process.stderr.strip()
    combined_output = "\n".join(
        part for part in [stdout, stderr] if part
    ).strip() or "无输出。"

    return (
        f"{command_description}\n"
        f"工作目录: {working_directory}\n"
        f"退出码: {completed_process.returncode}\n"
        f"执行权限: 继承当前 CLI 进程权限，不会自动提权。\n"
        f"输出:\n{_truncate_output(combined_output)}"
    )


def _build_shell_command(command: str) -> list[str]:
    if os.name == "nt":
        powershell_executable = shutil.which("pwsh") or shutil.which("powershell")
        if powershell_executable is None:
            raise FileNotFoundError("未找到可用的 PowerShell 可执行文件。")
        return [
            powershell_executable,
            "-NoLogo",
            "-NoProfile",
            "-Command",
            command,
        ]

    return ["/bin/sh", "-lc", command]


def _build_subprocess_options() -> dict[str, object]:
    """为外部进程构建适合后续中断的启动参数。"""
    if os.name == "nt":
        creation_flag = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        return {"creationflags": creation_flag}
    return {"start_new_session": True}


def _start_stream_reader(stream: object) -> tuple[list[str], threading.Thread]:
    """异步持续读取子进程输出，避免主线程轮询时管道写满阻塞。"""
    chunks: list[str] = []

    def reader() -> None:
        if not hasattr(stream, "read"):
            return
        try:
            while True:
                try:
                    chunk = stream.read(1024)
                except ValueError:
                    # 另一线程关闭了管道，终止读取
                    break
                if not chunk:
                    break
                chunks.append(chunk)
        finally:
            if hasattr(stream, "close"):
                stream.close()

    reader_thread = threading.Thread(target=reader, daemon=True)
    reader_thread.start()
    return chunks, reader_thread


def _collect_stream_output(
    stdout_chunks: list[str],
    stderr_chunks: list[str],
    stdout_thread: threading.Thread,
    stderr_thread: threading.Thread,
) -> tuple[str, str]:
    """等待输出读取线程收尾，并归并标准输出与标准错误。"""
    stdout_thread.join(timeout=1)
    stderr_thread.join(timeout=1)
    return "".join(stdout_chunks), "".join(stderr_chunks)


def _run_process_with_controller(
    command: list[str],
    *,
    working_directory: Path,
    timeout_seconds: int,
    execution_controller: ExecutionController | None,
) -> subprocess.CompletedProcess[str]:
    """以可轮询方式执行外部进程，便于 /stop 中途终止。"""
    process: subprocess.Popen[str] | None = None
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    stdout_thread: threading.Thread | None = None
    stderr_thread: threading.Thread | None = None
    started_at = time.monotonic()

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=working_directory,
            shell=False,
            **_build_subprocess_options(),
        )
        if execution_controller is not None:
            execution_controller.register_process(process)
        stdout_chunks, stdout_thread = _start_stream_reader(process.stdout)
        stderr_chunks, stderr_thread = _start_stream_reader(process.stderr)

        # 输出静默检测状态
        _last_chunk_count = 0
        _had_output = False
        _quiet_since: float | None = None

        while True:
            if execution_controller is not None:
                execution_controller.ensure_not_cancelled()

            # ── 输出静默检测 ──
            chunk_count = len(stdout_chunks) + len(stderr_chunks)
            if chunk_count > _last_chunk_count:
                _had_output = True
                _quiet_since = None
                _last_chunk_count = chunk_count
            elif _had_output and _quiet_since is None:
                _quiet_since = time.monotonic()

            if (
                _had_output
                and _quiet_since is not None
                and time.monotonic() - _quiet_since >= _MAX_QUIET_SECONDS
                and process.poll() is None  # 进程仍在运行但无输出
            ):
                terminate_process_tree(process)
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                if process.stdout is not None and not process.stdout.closed:
                    process.stdout.close()
                if process.stderr is not None and not process.stderr.closed:
                    process.stderr.close()
                stdout, stderr = _collect_stream_output(
                    stdout_chunks, stderr_chunks, stdout_thread, stderr_thread,
                )
                hint = (f"[静默超时] 进程存活但超过 {_MAX_QUIET_SECONDS}s 无输出，已终止。"
                        f" 如需运行持续服务，请使用 run_shell_command(detach=True, ...)。")
                raise subprocess.TimeoutExpired(
                    command, timeout_seconds,
                    output=f"{stdout}\n{hint}",
                    stderr=stderr,
                ) from None

            if process.poll() is not None:
                # 关闭管道写端：后台子进程（如 java -jar ... &）可能仍持有写端
                # 导致 reader 线程的 stream.read(1024) 永远等不到 EOF 而死锁
                if process.stdout is not None and not process.stdout.closed:
                    process.stdout.close()
                if process.stderr is not None and not process.stderr.closed:
                    process.stderr.close()
                stdout, stderr = _collect_stream_output(
                    stdout_chunks,
                    stderr_chunks,
                    stdout_thread,
                    stderr_thread,
                )
                return subprocess.CompletedProcess(
                    process.args,
                    process.returncode,
                    stdout,
                    stderr,
                )
            if time.monotonic() - started_at >= timeout_seconds:
                terminate_process_tree(process)
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                stdout, stderr = _collect_stream_output(
                    stdout_chunks,
                    stderr_chunks,
                    stdout_thread,
                    stderr_thread,
                )
                raise subprocess.TimeoutExpired(
                    command,
                    timeout_seconds,
                    output=stdout,
                    stderr=stderr,
                ) from None
            time.sleep(PROCESS_POLL_INTERVAL_SECONDS)
    except ExecutionInterruptedError:
        if process is not None and process.poll() is None:
            terminate_process_tree(process)
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
        raise
    finally:
        if process is not None:
            if process.stdout is not None and not process.stdout.closed:
                process.stdout.close()
            if process.stderr is not None and not process.stderr.closed:
                process.stderr.close()
        if process is not None and execution_controller is not None:
            execution_controller.unregister_process(process)


def create_run_shell_command_tool(
    allowed_roots: Sequence[Path],
    execution_controller: ExecutionController | None = None,
):
    """创建受工作目录范围约束的 Shell 命令执行工具。"""
    normalized_roots = normalize_allowed_roots(allowed_roots)

    @tool("run_shell_command")
    def run_shell_command(
        command: str,
        working_directory: str = ".",
        timeout_seconds: int = 60,
        detach: bool = False,
        label: str = "",
    ) -> str:
        """
        执行 shell 命令。

        参数：
          command: 要执行的 shell 命令（如 "ls -la"）。
          working_directory: 工作目录（默认 "."）。
          timeout_seconds: 超时秒数（默认 60，最大 120）。
          detach: 是否分离运行。为 true 时立即返回进程句柄，不等待完成。
                  适合启动需要持续运行的服务（java -jar、docker compose up 等）。
                  后续可用 read_process_output / list_processes / stop_process 管理。
          label: detach 模式下的进程标签，便于识别。
        """
        try:
            resolved_working_directory = resolve_permitted_path(
                working_directory,
                normalized_roots,
            )
        except ValueError as exc:
            return f"❌ {exc}"

        if not resolved_working_directory.exists():
            return f"❌ 工作目录不存在：{working_directory}"
        if not resolved_working_directory.is_dir():
            return f"❌ 工作目录不是目录：{working_directory}"

        if detach:
            pm = get_process_manager()
            handle = pm.start(command, resolved_working_directory, label=label)
            status = pm.get_status(handle)
            return (
                f"✅ 后台进程已启动\n"
                f"   句柄: {handle}\n"
                f"   标签: {label or '未命名'}\n"
                f"   工作目录: {resolved_working_directory}\n"
                f"   命令: {command}\n\n"
                f"使用 read_process_output(handle=\"{handle}\") 读取输出。\n"
                f"使用 stop_process(handle=\"{handle}\") 停止进程。"
            )

        safe_timeout_seconds = max(1, min(timeout_seconds, MAX_TOOL_TIMEOUT_SECONDS))
        try:
            completed_process = _run_process_with_controller(
                _build_shell_command(command),
                working_directory=resolved_working_directory,
                timeout_seconds=safe_timeout_seconds,
                execution_controller=execution_controller,
            )
        except FileNotFoundError as exc:
            return f"❌ 命令执行环境不可用：{exc}"
        except subprocess.TimeoutExpired as exc:
            detail = ""
            if exc.output:
                out = exc.output.strip()
                if out:
                    # 输出最后三行，包含静默超时等诊断信息
                    detail = "\n" + "\n".join(out.splitlines()[-3:])
            return f"❌ 命令执行超时：{command}{detail}"
        except ExecutionInterruptedError:
            raise
        except Exception as exc:
            # 需要捕获所有工具执行异常，转换为用户可读错误而非崩溃
            return f"❌ 执行命令时发生错误：{exc}"

        return _format_completed_process_output(
            command_description=f"命令: {command}",
            working_directory=resolved_working_directory,
            completed_process=completed_process,
        )

    return attach_tool_risk(run_shell_command, "execute")


def create_read_process_output_tool() -> object:
    """创建读取后台进程输出的工具。"""
    @tool("read_process_output")
    def read_process_output(
        handle: str,
        tail: Optional[int] = 50,
    ) -> str:
        """
        读取后台进程的累积输出。句柄由 run_shell_command(detach=True) 返回。
        可选 tail 参数（默认 50）限制只返回末尾 N 行，传 None 返回全部。
        """
        pm = get_process_manager()
        status = pm.get_status(handle)
        if status is None:
            return f"❌ 找不到进程句柄：{handle}"

        output = pm.read_output(handle, tail=tail)
        running_icon = "🟢 运行中" if status["running"] else "🔴 已退出"
        exit_info = f", 退出码 {status['exit_code']}" if status["exit_code"] is not None else ""
        status_line = f"{running_icon} (已运行 {status['elapsed_seconds']:.0f}s{exit_info})"
        if not output:
            return f"[进程: {status['label']}]\n{status_line}\n(尚无输出)"
        return f"[进程: {status['label']}]\n{status_line}\n\n{output}"

    return attach_tool_risk(read_process_output, "read")


def create_list_processes_tool() -> object:
    """创建列出所有后台进程的工具。"""
    @tool("list_processes")
    def list_processes() -> str:
        """
        列出所有通过 run_shell_command(detach=True) 启动的后台进程。
        """
        pm = get_process_manager()
        processes = pm.list_summary()
        if not processes:
            return "当前没有后台进程。"
        lines = [f"{'句柄':<40} {'标签':<20} {'状态':<10} {'已运行':<10}"]
        lines.append("-" * 80)
        for p in processes:
            state = "🟢 运行中" if p["running"] else f"🔴 退出({p['exit_code']})"
            elapsed = f"{p['elapsed_seconds']:.0f}s"
            lines.append(
                f"{p['handle']:<40} {p['label']:<20} {state:<10} {elapsed:<10}"
            )
        return "\n".join(lines)

    return attach_tool_risk(list_processes, "read")


def create_stop_process_tool() -> object:
    """创建停止后台进程的工具。"""
    @tool("stop_process")
    def stop_process(handle: str) -> str:
        """
        停止一个后台进程。句柄由 run_shell_command(detach=True) 返回。
        """
        pm = get_process_manager()
        status = pm.get_status(handle)
        if status is None:
            return f"❌ 找不到进程句柄：{handle}"
        if not status["running"]:
            return f"进程 {handle} 已退出（退出码 {status['exit_code']}）。"

        pm.stop(handle)
        return f"✅ 进程 {handle} 已终止。"

    return attach_tool_risk(stop_process, "write")


def create_run_registered_tool_tool(
    command_registry: Mapping[str, Path | str],
    execution_controller: ExecutionController | None = None,
):
    """创建只允许执行已注册外部工具的命令工具。"""
    normalized_registry = normalize_command_registry(command_registry)
    registered_tool_names = ", ".join(sorted(normalized_registry)) or "无"

    @tool("run_registered_tool")
    def run_registered_tool(
        tool_name: str,
        arguments: list[str] | None = None,
        timeout_seconds: int = 30,
    ) -> str:
        """
        执行已注册的外部工具。
        该工具只允许调用显式注册过的绝对路径程序，
        命令会继承当前 CLI 进程权限，不会自动使用 sudo 或其他提权方式。
        """
        command_path = normalized_registry.get(tool_name)
        if command_path is None:
            return (
                f"❌ 未注册的外部工具：{tool_name}。"
                f"当前已注册工具有：{registered_tool_names}"
            )

        safe_timeout_seconds = max(1, min(timeout_seconds, MAX_TOOL_TIMEOUT_SECONDS))
        command_arguments = [str(argument) for argument in (arguments or [])]
        command = [str(command_path), *command_arguments]

        try:
            completed_process = _run_process_with_controller(
                command,
                working_directory=Path.cwd(),
                timeout_seconds=safe_timeout_seconds,
                execution_controller=execution_controller,
            )
        except FileNotFoundError:
            return f"❌ 工具路径不存在：{command_path}"
        except subprocess.TimeoutExpired:
            return f"❌ 工具执行超时：{tool_name}"
        except ExecutionInterruptedError:
            raise
        except Exception as exc:
            # 需要捕获所有工具执行异常，转换为用户可读错误而非崩溃
            return f"❌ 调用外部工具时发生错误：{exc}"

        return _format_completed_process_output(
            command_description=f"工具: {tool_name}\n路径: {command_path}",
            working_directory=Path.cwd(),
            completed_process=completed_process,
        )

    return attach_tool_risk(run_registered_tool, "execute")
