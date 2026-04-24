from __future__ import annotations

import os
import signal
import subprocess
import threading

DEFAULT_STOP_MESSAGE = "当前任务已被 /stop 中断。"


class ExecutionInterruptedError(RuntimeError):
    """表示当前运行任务已被用户主动中断。"""


def terminate_process_tree(process: subprocess.Popen[str]) -> None:
    """尽量终止进程及其子进程，确保 /stop 能清理外部命令。"""
    if process.poll() is not None:
        return

    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                check=False,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError:
            process.kill()
            return
        if process.poll() is not None:
            return
        try:
            process.wait(timeout=1)
            return
        except subprocess.TimeoutExpired:
            process.kill()
            return

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError:
        process.kill()


class ExecutionController:
    """维护当前运行任务的取消状态与外部进程句柄。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._active_processes: set[subprocess.Popen[str]] = set()
        self._is_running = False
        self._stop_reason = ""

    def begin_run(self) -> None:
        """开始一轮新任务前清理上一轮状态。"""
        with self._lock:
            self._cancel_event.clear()
            self._active_processes.clear()
            self._is_running = True
            self._stop_reason = ""

    def finish_run(self) -> None:
        """结束当前任务并重置控制状态。"""
        with self._lock:
            self._active_processes.clear()
            self._is_running = False
            self._cancel_event.clear()
            self._stop_reason = ""

    def is_running(self) -> bool:
        """返回当前是否存在仍在执行中的任务。"""
        with self._lock:
            return self._is_running

    def is_cancel_requested(self) -> bool:
        """返回当前任务是否已收到取消请求。"""
        return self._cancel_event.is_set()

    def request_stop(self, reason: str = "用户输入 /stop") -> bool:
        """请求中断当前任务，并尝试终止所有已登记的外部进程。"""
        with self._lock:
            was_running = self._is_running
            self._cancel_event.set()
            self._stop_reason = reason
            active_processes = list(self._active_processes)

        for process in active_processes:
            terminate_process_tree(process)

        return was_running

    def ensure_not_cancelled(self) -> None:
        """在关键执行边界检查取消状态，必要时立刻抛出中断异常。"""
        if self._cancel_event.is_set():
            raise ExecutionInterruptedError(self.get_stop_message())

    def register_process(self, process: subprocess.Popen[str]) -> None:
        """登记当前活跃外部进程，供 /stop 统一终止。"""
        with self._lock:
            self._active_processes.add(process)
            cancel_requested = self._cancel_event.is_set()

        if cancel_requested:
            terminate_process_tree(process)
            self.unregister_process(process)
            self.ensure_not_cancelled()

    def unregister_process(self, process: subprocess.Popen[str]) -> None:
        """移除已结束或已清理的外部进程登记。"""
        with self._lock:
            self._active_processes.discard(process)

    def get_stop_message(self) -> str:
        """返回适合展示给用户的中断说明。"""
        with self._lock:
            stop_reason = self._stop_reason.strip()
        if not stop_reason:
            return DEFAULT_STOP_MESSAGE
        return f"{DEFAULT_STOP_MESSAGE} 原因：{stop_reason}"
