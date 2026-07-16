from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..agent.events import AgentEventType

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

    from ..agent.runner import AgentRunner


SESSION_STORAGE_DIRNAME = ".cyber-agent-cli-sessions"


def _load_session_store_support():
    """按需加载历史会话存储，避免帮助和版本命令导入 LangChain 消息类型。"""
    from ..session_store import (
        append_session_event,
        clear_interrupt_checkpoint,
        create_session_id,
        export_session_history,
        get_session_storage_dir,
        has_interrupt_checkpoint,
        list_stored_sessions,
        load_interrupt_checkpoint,
        load_session_history,
        save_interrupt_checkpoint,
        save_session_history,
        search_stored_sessions,
    )

    return {
        "append_session_event": append_session_event,
        "clear_interrupt_checkpoint": clear_interrupt_checkpoint,
        "create_session_id": create_session_id,
        "export_session_history": export_session_history,
        "get_session_storage_dir": get_session_storage_dir,
        "has_interrupt_checkpoint": has_interrupt_checkpoint,
        "list_stored_sessions": list_stored_sessions,
        "load_interrupt_checkpoint": load_interrupt_checkpoint,
        "load_session_history": load_session_history,
        "save_interrupt_checkpoint": save_interrupt_checkpoint,
        "save_session_history": save_session_history,
        "search_stored_sessions": search_stored_sessions,
    }


def create_runtime_session_id(now: datetime | None = None) -> str:
    """轻量生成会话 ID，避免启动阶段导入完整历史存储模块。"""
    resolved_now = now or datetime.now().astimezone()
    return resolved_now.strftime("%Y%m%d-%H%M%S-%f")


def get_runtime_session_storage_dir(base_dir: Path | None = None) -> Path:
    """轻量计算历史目录路径，支持任意目录启动时回溯查找。"""
    from ..local_config import find_data_dir

    return find_data_dir(SESSION_STORAGE_DIRNAME, base_dir)


def start_new_runtime_session(
    runtime_context: dict[str, object],
    *,
    source_session_id: str | None = None,
) -> str:
    """为当前运行上下文分配新的会话标识，避免覆盖既有历史。"""
    session_id = create_runtime_session_id()
    runtime_context["session_id"] = session_id
    runtime_context["session_source_id"] = source_session_id
    runtime_context["_stop_input_buffer"] = ""
    return session_id


def start_fresh_visible_runtime_session(
    runtime_context: dict[str, object],
    *,
    source_session_id: str | None = None,
) -> str:
    """开始一个对用户可见也全新的会话窗口。"""
    session_id = start_new_runtime_session(
        runtime_context,
        source_session_id=source_session_id,
    )
    runtime_context["_recent_inputs"] = []
    runtime_context["__clear_visible_session"] = True
    try:
        session_store = _load_session_store_support()
        session_store["clear_interrupt_checkpoint"]()
    except (OSError, ValueError) as exc:
        from ..logging import log_warning

        log_warning("app", f"清理中断快照失败：{exc}")
    return session_id


def _try_persist(
    runner: AgentRunner,
    runtime_context: dict[str, object],
    *,
    force: bool = False,
) -> None:
    """安全保存会话，失败时仅记录日志不影响主流程。"""
    try:
        persist_runtime_session(runner, runtime_context, force=force)
    except (OSError, TypeError, ValueError) as exc:
        from ..logging import log_warning

        log_warning("app", f"会话持久化失败：{exc}")


def _get_runtime_session_base_dir(
    runtime_context: dict[str, object],
) -> Path | None:
    """读取运行期指定的会话存储基准目录；CLI 默认使用当前目录发现规则。"""
    raw_base_dir = runtime_context.get("session_base_dir")
    if raw_base_dir is None:
        return None
    return Path(str(raw_base_dir)).expanduser()


def persist_runtime_session(
    runner: AgentRunner,
    runtime_context: dict[str, object],
    *,
    force: bool = False,
) -> Path | None:
    """按当前工作目录自动保存会话历史，供后续 /history 访问。"""
    history = runner.get_history_snapshot()
    if not force and len(history) <= 1 and runner.get_turn_count() == 0:
        return None

    session_store = _load_session_store_support()
    session_path = session_store["save_session_history"](
        str(runtime_context["session_id"]),
        history,
        mode=runner.mode.value,
        approval_policy=runtime_context["approval_policy"].value,
        source_session_id=runtime_context.get("session_source_id"),
        recent_inputs=runtime_context.get("_recent_inputs"),
        base_dir=_get_runtime_session_base_dir(runtime_context),
    )
    runtime_context["session_storage_dir"] = session_path.parent
    return session_path


def append_runtime_session_event(
    runtime_context: dict[str, object],
    event_type: str | AgentEventType,
    payload: object = None,
) -> Path | None:
    """把运行期事件追加到当前会话的 JSONL 事件流。"""
    session_id = runtime_context.get("session_id")
    if not session_id:
        return None
    try:
        session_store = _load_session_store_support()
        event_path = session_store["append_session_event"](
            str(session_id),
            str(event_type),
            payload=payload,
            base_dir=_get_runtime_session_base_dir(runtime_context),
        )
        runtime_context["session_event_log"] = event_path
        return event_path
    except (OSError, TypeError, ValueError) as exc:
        from ..logging import log_warning

        log_warning("app", f"会话事件落盘失败：{exc}")
        return None


def create_persisting_event_handler(
    runner: AgentRunner,
    runtime_context: dict[str, object],
    inner_handler: Any | None,
) -> Any:
    """包装运行器事件处理器：保留原展示逻辑，同时实时写事件和会话快照。"""
    semantic_events = {
        AgentEventType.TURN_START,
        AgentEventType.RESPONSE_END,
        AgentEventType.RESPONSE_RETRY,
        AgentEventType.TOOL_CALL,
        AgentEventType.TOOL_RESULT,
        AgentEventType.APPROVAL_REQUEST,
        AgentEventType.APPROVAL_RESULT,
        AgentEventType.TURN_END,
        AgentEventType.HISTORY_UPDATED,
    }

    def handler(event_type: str | AgentEventType, payload: object) -> None:
        if inner_handler is not None:
            inner_handler(event_type, payload)
        try:
            normalized_event: str | AgentEventType = AgentEventType(event_type)
        except ValueError:
            normalized_event = str(event_type)
        if normalized_event in semantic_events:
            append_runtime_session_event(runtime_context, normalized_event, payload)
        if normalized_event == AgentEventType.HISTORY_UPDATED:
            _try_persist(runner, runtime_context, force=True)

    return handler


def _save_interrupt_checkpoint(
    runner: AgentRunner,
    runtime_context: dict[str, object],
) -> None:
    """会话异常中断时保存续传快照，下次启动可恢复。"""
    try:
        session_store = _load_session_store_support()
        session_store["save_interrupt_checkpoint"](
            str(runtime_context["session_id"]),
            runner.get_history_snapshot(),
            mode=runner.mode.value,
            approval_policy=runtime_context["approval_policy"].value,
        )
    except (OSError, TypeError, ValueError) as exc:
        from ..logging import log_warning

        log_warning("app", f"中断快照保存失败：{exc}")


def _resolve_resume_session(
    runtime_context: dict[str, object],
) -> tuple[str, list[BaseMessage], str, str] | None:
    """检查是否存在可续传的中断快照，返回会话 ID、消息、模式与审批策略。"""
    _ = runtime_context
    session_store = _load_session_store_support()
    checkpoint = session_store["load_interrupt_checkpoint"]()
    if checkpoint is None:
        return None

    try:
        raw_messages = checkpoint.get("messages", [])
        if not isinstance(raw_messages, list) or not raw_messages:
            return None
        from langchain_core.messages import messages_from_dict

        messages = messages_from_dict(raw_messages)
    except Exception as exc:  # noqa: BLE001 - LangChain 反序列化跨版本错误类型不稳定
        from ..logging import log_warning

        log_warning("app", f"中断快照消息反序列化失败：{exc}")
        return None

    session_id = str(checkpoint.get("session_id", ""))
    mode = str(checkpoint.get("mode", "standard"))
    approval_policy = str(checkpoint.get("approval_policy", "prompt"))
    return session_id, messages, mode, approval_policy


def _has_pending_checkpoint() -> bool:
    """是否存在待恢复的中断快照。"""
    session_store = _load_session_store_support()
    return session_store["has_interrupt_checkpoint"]()
