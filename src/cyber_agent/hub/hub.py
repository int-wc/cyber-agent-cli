from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from langchain_core.messages import AIMessage

from ..agent.events import AgentEventType
from ..agent.mode import AgentMode
from ..execution_control import ExecutionInterruptedError
from ..logging import log_warning
from ..session_store import (
    append_session_event,
    create_session_id,
    load_session_history,
    save_session_history,
)

HubEventSubscriber = Callable[["HubEvent"], None]


@dataclass(slots=True, frozen=True)
class HubTaskSource:
    """描述一个 Hub 输入来源。"""

    kind: str
    name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class HubTask:
    """排队进入 Hub 的用户输入或控制命令。"""

    text: str
    source: HubTaskSource
    created_at: float = field(default_factory=time.time)


@dataclass(slots=True, frozen=True)
class HubEvent:
    """广播给所有前端的 Hub 事件。"""

    type: str
    payload: Any = None
    source: HubTaskSource | None = None


class CyberAgentHub:
    """单 runner、串行任务队列、多前端事件广播的本地 Hub。"""

    def __init__(
        self,
        *,
        runner: Any,
        runtime_context: dict[str, object],
        approval_handler_factory: Callable[[dict[str, object]], Any],
        detect_task_complexity: Callable[[str], bool],
        run_multi_agent_turn: Callable[[str, Any, dict[str, object]], None],
        renderless_event_handler_factory: Callable[
            [Any, dict[str, object], Callable[[str | AgentEventType, object], None]],
            Callable[[str | AgentEventType, object], None],
        ],
        base_dir: Path | None = None,
    ) -> None:
        self.runner = runner
        self.runtime_context = runtime_context
        self.approval_handler_factory = approval_handler_factory
        self.detect_task_complexity = detect_task_complexity
        self.run_multi_agent_turn = run_multi_agent_turn
        self.renderless_event_handler_factory = renderless_event_handler_factory
        self.base_dir = base_dir
        if base_dir is not None:
            self.runtime_context["session_base_dir"] = base_dir

        self._queue: Queue[HubTask] = Queue()
        self._subscribers: list[HubEventSubscriber] = []
        self._subscriber_lock = threading.RLock()
        self._session_lock = threading.RLock()
        self._worker_thread: threading.Thread | None = None
        self._stop_worker = threading.Event()
        self._session_command_pending = threading.Event()

    def start(self) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._stop_worker.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="cyber-agent-hub-worker",
            daemon=True,
        )
        self._worker_thread.start()
        self.broadcast("hub_started", {"session_id": self.session_id})

    def stop(self) -> None:
        self._stop_worker.set()
        self.request_stop("Hub 正在停止")
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)
        remaining = self._queue.qsize()
        if remaining:
            log_warning("hub", f"Hub 停止时仍有 {remaining} 个未处理任务。")
        self._persist_current_session()
        self.broadcast("hub_stopped", {"session_id": self.session_id})

    @property
    def session_id(self) -> str:
        return str(self.runtime_context.get("session_id") or "")

    def subscribe(self, subscriber: HubEventSubscriber) -> Callable[[], None]:
        with self._subscriber_lock:
            self._subscribers.append(subscriber)

        def unsubscribe() -> None:
            with self._subscriber_lock:
                if subscriber in self._subscribers:
                    self._subscribers.remove(subscriber)

        return unsubscribe

    def broadcast(
        self,
        event_type: str | AgentEventType,
        payload: Any = None,
        *,
        source: HubTaskSource | None = None,
    ) -> None:
        event = HubEvent(str(event_type), payload, source)
        with self._subscriber_lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            try:
                subscriber(event)
            except Exception as exc:
                log_warning("hub", f"Hub 事件订阅器执行失败：{exc}")
                continue

    def submit(
        self,
        text: str,
        *,
        source: HubTaskSource | None = None,
    ) -> None:
        normalized_text = text.strip()
        if not normalized_text:
            return
        resolved_source = source or HubTaskSource("cli", "cli")
        if normalized_text.lower() == "/stop":
            self.request_stop(f"{resolved_source.kind} 请求停止当前任务")
            self.broadcast(
                "task_stop_requested",
                {"reason": f"{resolved_source.kind} 请求停止当前任务"},
                source=resolved_source,
            )
            return
        if self._is_immediate_session_command(normalized_text):
            self._handle_session_command(normalized_text, resolved_source)
            return
        self._queue.put(HubTask(normalized_text, resolved_source))
        self.broadcast(
            "task_queued",
            {"text": normalized_text, "queue_size": self._queue.qsize()},
            source=resolved_source,
        )

    def request_stop(self, reason: str) -> None:
        controller = getattr(self.runner, "execution_controller", None)
        for method_name in ("request_stop", "request_cancel", "cancel"):
            method = getattr(controller, method_name, None)
            if callable(method):
                try:
                    method(reason)
                except TypeError:
                    method()
                return

    def wait_until_idle(self, timeout_seconds: float = 5.0) -> bool:
        deadline = time.time() + max(timeout_seconds, 0.0)
        while time.time() <= deadline:
            if self._queue.unfinished_tasks == 0:
                return True
            time.sleep(0.01)
        return self._queue.unfinished_tasks == 0

    def _worker_loop(self) -> None:
        while not self._stop_worker.is_set():
            if self._session_command_pending.is_set():
                time.sleep(0.05)
                continue
            try:
                task = self._queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                self._run_task(task)
            finally:
                self._queue.task_done()

    def _run_task(self, task: HubTask) -> None:
        with self._session_lock:
            self.broadcast(
                "task_started",
                {"text": task.text, "session_id": self.session_id},
                source=task.source,
            )

            recent = self.runtime_context.setdefault("_recent_inputs", [])
            if isinstance(recent, list):
                recent.append(task.text)
                if len(recent) > 50:
                    del recent[:-50]
            self.broadcast(
                "user_input_received",
                {"input": task.text, "source": task.source.kind},
                source=task.source,
            )
            self._append_current_session_event(
                "user_input_received",
                {"input": task.text, "source": task.source.kind},
            )

            def inner_event_handler(
                event_type: str | AgentEventType,
                payload: object,
            ) -> None:
                self.broadcast(event_type, payload, source=task.source)

            event_handler = self.renderless_event_handler_factory(
                self.runner,
                self.runtime_context,
                inner_event_handler,
            )
            reply_text = ""
            try:
                multi_setting = self.runtime_context.get("multi_agent_enabled", "auto")
                if multi_setting is True or (
                    multi_setting == "auto" and self.detect_task_complexity(task.text)
                ):
                    self.run_multi_agent_turn(task.text, self.runner, self.runtime_context)
                    reply_text = self._last_ai_text()
                else:
                    reply_text = str(
                        self.runner.run(
                            task.text,
                            verbose=False,
                            event_handler=event_handler,
                            approval_handler=self.approval_handler_factory(
                                self.runtime_context
                            ),
                        )
                        or ""
                    )
                    if not reply_text:
                        reply_text = self._last_ai_text()
                self._persist_current_session()
                self.broadcast(
                    "task_finished",
                    {
                        "text": task.text,
                        "session_id": self.session_id,
                        "reply_text": reply_text,
                    },
                    source=task.source,
                )
            except ExecutionInterruptedError as exc:
                message = str(exc) or "当前任务已被停止。"
                self.runner.history.append(AIMessage(content=message))
                self._persist_current_session()
                self.broadcast(
                    AgentEventType.HISTORY_UPDATED,
                    {"message_type": "ai", "content": message},
                    source=task.source,
                )
                self.broadcast(
                    "task_interrupted",
                    {
                        "message": message,
                        "session_id": self.session_id,
                        "reply_text": message,
                    },
                    source=task.source,
                )
            except Exception as exc:  # noqa: BLE001 - Hub 需要把真实错误广播给前端
                message = f"运行失败：{exc}"
                self.runner.history.append(AIMessage(content=message))
                self._persist_current_session()
                self.broadcast(
                    AgentEventType.HISTORY_UPDATED,
                    {"message_type": "ai", "content": message},
                    source=task.source,
                )
                self.broadcast(
                    "task_error",
                    {
                        "message": message,
                        "session_id": self.session_id,
                        "reply_text": message,
                    },
                    source=task.source,
                )

    @staticmethod
    def _is_immediate_session_command(text: str) -> bool:
        normalized = text.strip().lower()
        return (
            normalized in {"/new", "/clear", "/session new"}
            or normalized.startswith("/session use ")
            or normalized.startswith("/session load ")
        )

    def _handle_session_command(self, text: str, source: HubTaskSource) -> None:
        self._session_command_pending.set()
        self.request_stop(f"{source.kind} 请求切换会话")
        try:
            with self._session_lock:
                drained = self._drain_pending_tasks_locked()
                if drained:
                    self.broadcast(
                        "task_queue_drained",
                        {"count": drained, "reason": "session_switch"},
                        source=source,
                    )
                normalized = text.strip()
                lowered = normalized.lower()
                if lowered in {"/new", "/clear", "/session new"}:
                    self._persist_current_session()
                    new_session_id = create_session_id()
                    self.runtime_context["session_id"] = new_session_id
                    self.runtime_context["session_source_id"] = None
                    self.runtime_context["_recent_inputs"] = []
                    self.runner.reset()
                    self._persist_current_session()
                    self.broadcast(
                        "session_switched",
                        {"session_id": new_session_id, "reason": "new"},
                        source=source,
                    )
                    return

                if lowered.startswith("/session use "):
                    session_id = normalized[len("/session use ") :].strip()
                else:
                    session_id = normalized[len("/session load ") :].strip()
                if not session_id:
                    self.broadcast(
                        "session_switch_failed",
                        {"reason": "缺少会话 ID。"},
                        source=source,
                    )
                    return
                try:
                    stored = load_session_history(session_id, base_dir=self.base_dir)
                except Exception as exc:
                    self.broadcast(
                        "session_switch_failed",
                        {"session_id": session_id, "reason": str(exc)},
                        source=source,
                    )
                    return
                self._persist_current_session()
                self.runner.restore_history(stored.messages)
                self.runtime_context["session_id"] = stored.summary.session_id
                self.runtime_context["session_source_id"] = stored.summary.source_session_id
                self.runtime_context["_recent_inputs"] = list(stored.recent_inputs or [])
                self._persist_current_session()
                self.broadcast(
                    "session_switched",
                    {"session_id": stored.summary.session_id, "reason": "load"},
                    source=source,
                )
        finally:
            self._session_command_pending.clear()

    def _drain_pending_tasks_locked(self) -> int:
        drained = 0
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                return drained
            self._queue.task_done()
            drained += 1

    def _last_ai_text(self) -> str:
        for message in reversed(self.runner.get_history_snapshot()):
            if isinstance(message, AIMessage):
                content = message.content
                if isinstance(content, str):
                    return content
                return str(content)
        return ""

    def _persist_current_session(self) -> None:
        session_id = self.session_id
        if not session_id:
            return
        approval_policy = self.runtime_context.get("approval_policy", "prompt")
        approval_value = getattr(approval_policy, "value", str(approval_policy))
        mode = getattr(getattr(self.runner, "mode", AgentMode.STANDARD), "value", "standard")
        session_path = save_session_history(
            session_id,
            self.runner.get_history_snapshot(),
            mode=mode,
            approval_policy=approval_value,
            source_session_id=self.runtime_context.get("session_source_id"),
            recent_inputs=self.runtime_context.get("_recent_inputs"),
            base_dir=self.base_dir,
        )
        self.runtime_context["session_storage_dir"] = session_path.parent

    def _append_current_session_event(
        self,
        event_type: str | AgentEventType,
        payload: object = None,
    ) -> None:
        session_id = self.session_id
        if not session_id:
            return
        try:
            event_path = append_session_event(
                session_id,
                str(event_type),
                payload=payload,
                base_dir=self.base_dir,
            )
            self.runtime_context["session_event_log"] = event_path
        except Exception as exc:
            log_warning("hub", f"Hub 会话事件落盘失败：{exc}")
