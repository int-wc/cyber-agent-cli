from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, messages_from_dict, messages_to_dict

SESSION_STORAGE_DIRNAME = ".cyber-agent-cli-sessions"


@dataclass(slots=True, frozen=True)
class StoredSessionSummary:
    """描述一个已落盘会话的摘要信息。"""

    session_id: str
    title: str
    created_at: str
    updated_at: str
    mode: str
    approval_policy: str
    turn_count: int
    message_count: int
    source_session_id: str | None


@dataclass(slots=True)
class StoredSession:
    """描述一个完整的历史会话。"""

    summary: StoredSessionSummary
    messages: list[BaseMessage]


def get_session_storage_dir(base_dir: Path | None = None) -> Path:
    """返回当前工作目录下的历史会话目录。"""
    resolved_base_dir = (base_dir or Path.cwd()).resolve()
    return resolved_base_dir / SESSION_STORAGE_DIRNAME


def create_session_id(now: datetime | None = None) -> str:
    """生成适合文件名与人工查阅的会话标识。"""
    resolved_now = now or datetime.now().astimezone()
    return resolved_now.strftime("%Y%m%d-%H%M%S-%f")


def _extract_message_text(message: BaseMessage) -> str:
    """将 LangChain 消息内容归一化为纯文本，便于生成标题和展示摘要。"""
    if isinstance(message.content, str):
        return message.content

    parts: list[str] = []
    for item in message.content:
        if isinstance(item, str):
            parts.append(item)
            continue
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text", "")))
    return "".join(parts)


def _build_session_title(messages: list[BaseMessage]) -> str:
    """优先使用首条用户消息作为会话标题，便于 /history 快速定位。"""
    for message in messages:
        if not isinstance(message, HumanMessage):
            continue
        first_line = _extract_message_text(message).strip().splitlines()
        title = first_line[0].strip() if first_line else ""
        if not title:
            continue
        if len(title) <= 40:
            return title
        return f"{title[:40]}..."
    return "空会话"


def _build_summary(raw_data: dict, session_id: str) -> StoredSessionSummary:
    """从磁盘记录中提取会话摘要。"""
    return StoredSessionSummary(
        session_id=session_id,
        title=str(raw_data.get("title", "空会话")),
        created_at=str(raw_data.get("created_at", "")),
        updated_at=str(raw_data.get("updated_at", "")),
        mode=str(raw_data.get("mode", "standard")),
        approval_policy=str(raw_data.get("approval_policy", "prompt")),
        turn_count=int(raw_data.get("turn_count", 0)),
        message_count=int(raw_data.get("message_count", 0)),
        source_session_id=(
            str(raw_data.get("source_session_id"))
            if raw_data.get("source_session_id")
            else None
        ),
    )


def save_session_history(
    session_id: str,
    messages: list[BaseMessage],
    *,
    mode: str,
    approval_policy: str,
    source_session_id: str | None = None,
    base_dir: Path | None = None,
) -> Path:
    """将当前会话历史保存到工作目录下的独立会话文件。"""
    storage_dir = get_session_storage_dir(base_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    session_path = storage_dir / f"{session_id}.json"

    created_at = ""
    if session_path.exists():
        try:
            raw_existing_data = json.loads(session_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raw_existing_data = {}
        if isinstance(raw_existing_data, dict):
            created_at = str(raw_existing_data.get("created_at", ""))

    timestamp = datetime.now().astimezone().isoformat()
    serialized_messages = messages_to_dict(messages)
    payload = {
        "session_id": session_id,
        "title": _build_session_title(messages),
        "created_at": created_at or timestamp,
        "updated_at": timestamp,
        "mode": mode,
        "approval_policy": approval_policy,
        "source_session_id": source_session_id,
        "turn_count": sum(isinstance(message, HumanMessage) for message in messages),
        "message_count": len(messages),
        "messages": serialized_messages,
    }
    session_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return session_path


def list_stored_sessions(base_dir: Path | None = None) -> list[StoredSessionSummary]:
    """列出当前工作目录下已保存的历史会话摘要。"""
    storage_dir = get_session_storage_dir(base_dir)
    if not storage_dir.exists():
        return []

    summaries: list[StoredSessionSummary] = []
    for session_path in storage_dir.glob("*.json"):
        try:
            raw_data = json.loads(session_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(raw_data, dict):
            continue
        session_id = str(raw_data.get("session_id") or session_path.stem)
        summaries.append(_build_summary(raw_data, session_id))

    return sorted(
        summaries,
        key=lambda item: (item.updated_at, item.session_id),
        reverse=True,
    )


def load_session_history(
    session_id: str,
    *,
    base_dir: Path | None = None,
) -> StoredSession:
    """读取指定历史会话，并还原为可继续复用的 LangChain 消息列表。"""
    session_path = get_session_storage_dir(base_dir) / f"{session_id}.json"
    if not session_path.exists():
        raise ValueError(f"未找到历史会话：{session_id}")

    try:
        raw_data = json.loads(session_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"历史会话文件不是合法 JSON：{session_path}") from exc

    if not isinstance(raw_data, dict):
        raise ValueError(f"历史会话文件内容必须为对象：{session_path}")

    raw_messages = raw_data.get("messages")
    if not isinstance(raw_messages, list):
        raise ValueError(f"历史会话缺少合法的消息数组：{session_path}")

    return StoredSession(
        summary=_build_summary(raw_data, session_id),
        messages=messages_from_dict(raw_messages),
    )
