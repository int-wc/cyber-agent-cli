from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    messages_from_dict,
    messages_to_dict,
)

SESSION_STORAGE_DIRNAME = ".cyber-agent-cli-sessions"
DEFAULT_HISTORY_SEARCH_LIMIT = 20
DEFAULT_HISTORY_SEARCH_EXCERPT_LENGTH = 120


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


@dataclass(slots=True, frozen=True)
class StoredSessionSearchResult:
    """描述一次历史会话检索命中的摘要结果。"""

    session_id: str
    title: str
    updated_at: str
    mode: str
    approval_policy: str
    source_session_id: str | None
    matched_message_count: int
    excerpts: list[str]


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


def _get_message_role_label(message: BaseMessage) -> str:
    """生成适合历史检索与导出的消息角色标签。"""
    if isinstance(message, HumanMessage):
        return "用户"
    if isinstance(message, AIMessage):
        return "助手"
    if isinstance(message, ToolMessage):
        return f"工具({message.name or 'unknown'})"
    if isinstance(message, SystemMessage):
        return "系统"
    return message.__class__.__name__


def _extract_searchable_message_text(message: BaseMessage) -> str:
    """生成适合历史检索的消息文本，兼容工具调用型 AI 消息。"""
    content = _extract_message_text(message).strip()
    if isinstance(message, AIMessage) and message.tool_calls and not content:
        return f"工具调用：{json.dumps(message.tool_calls, ensure_ascii=False)}"
    return content


def _normalize_search_excerpt(text: str, query: str) -> str:
    """围绕命中关键词生成更短的摘要片段，便于终端快速定位。"""
    normalized_text = " ".join(part.strip() for part in text.splitlines() if part.strip())
    if len(normalized_text) <= DEFAULT_HISTORY_SEARCH_EXCERPT_LENGTH:
        return normalized_text

    lowered_text = normalized_text.lower()
    lowered_query = query.lower()
    match_index = lowered_text.find(lowered_query)
    if match_index < 0:
        return f"{normalized_text[:DEFAULT_HISTORY_SEARCH_EXCERPT_LENGTH]}..."

    window_radius = max((DEFAULT_HISTORY_SEARCH_EXCERPT_LENGTH - len(query)) // 2, 20)
    start_index = max(match_index - window_radius, 0)
    end_index = min(match_index + len(query) + window_radius, len(normalized_text))
    excerpt = normalized_text[start_index:end_index]
    if start_index > 0:
        excerpt = f"...{excerpt}"
    if end_index < len(normalized_text):
        excerpt = f"{excerpt}..."
    return excerpt


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


def _load_session_payload(
    session_path: Path,
    *,
    strict: bool,
) -> dict[str, object] | None:
    """读取单个会话文件的 JSON 对象；检索场景下允许跳过损坏文件。"""
    try:
        raw_data = json.loads(session_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        if strict:
            raise ValueError(f"历史会话文件不是合法 JSON：{session_path}") from exc
        return None

    if not isinstance(raw_data, dict):
        if strict:
            raise ValueError(f"历史会话文件内容必须为对象：{session_path}")
        return None

    return raw_data


def _load_messages_from_payload(
    raw_data: dict[str, object],
    *,
    session_path: Path,
    strict: bool,
) -> list[BaseMessage] | None:
    """从已解析的会话对象中恢复消息列表；无效记录在检索时直接跳过。"""
    raw_messages = raw_data.get("messages")
    if not isinstance(raw_messages, list):
        if strict:
            raise ValueError(f"历史会话缺少合法的消息数组：{session_path}")
        return None

    try:
        return messages_from_dict(raw_messages)
    except Exception as exc:  # noqa: BLE001 - 需要兼容不同版本 LangChain 的消息反序列化错误
        if strict:
            raise ValueError(f"历史会话消息无法反序列化：{session_path}") from exc
        return None


def _list_session_payloads(
    base_dir: Path | None = None,
) -> list[tuple[StoredSessionSummary, dict[str, object]]]:
    """一次性读取并排序会话摘要与原始对象，避免检索时重复读盘。"""
    storage_dir = get_session_storage_dir(base_dir)
    if not storage_dir.exists():
        return []

    session_items: list[tuple[StoredSessionSummary, dict[str, object]]] = []
    for session_path in storage_dir.glob("*.json"):
        raw_data = _load_session_payload(session_path, strict=False)
        if raw_data is None:
            continue
        session_id = str(raw_data.get("session_id") or session_path.stem)
        session_items.append((_build_summary(raw_data, session_id), raw_data))

    return sorted(
        session_items,
        key=lambda item: (item[0].updated_at, item[0].session_id),
        reverse=True,
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
        raw_existing_data = _load_session_payload(session_path, strict=False) or {}
        if raw_existing_data:
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
    return [summary for summary, _ in _list_session_payloads(base_dir)]


def load_session_history(
    session_id: str,
    *,
    base_dir: Path | None = None,
) -> StoredSession:
    """读取指定历史会话，并还原为可继续复用的 LangChain 消息列表。"""
    session_path = get_session_storage_dir(base_dir) / f"{session_id}.json"
    if not session_path.exists():
        raise ValueError(f"未找到历史会话：{session_id}")

    raw_data = _load_session_payload(session_path, strict=True)
    assert raw_data is not None
    messages = _load_messages_from_payload(
        raw_data,
        session_path=session_path,
        strict=True,
    )
    assert messages is not None

    return StoredSession(
        summary=_build_summary(raw_data, session_id),
        messages=messages,
    )


def search_stored_sessions(
    query: str,
    *,
    base_dir: Path | None = None,
    limit: int = DEFAULT_HISTORY_SEARCH_LIMIT,
) -> list[StoredSessionSearchResult]:
    """按关键词检索当前工作目录下的历史会话内容。"""
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("历史检索关键词不能为空。")

    results: list[StoredSessionSearchResult] = []
    for summary, raw_data in _list_session_payloads(base_dir):
        session_path = get_session_storage_dir(base_dir) / f"{summary.session_id}.json"
        messages = _load_messages_from_payload(
            raw_data,
            session_path=session_path,
            strict=False,
        )
        if messages is None:
            continue
        excerpts: list[str] = []
        matched_message_count = 0

        for index, message in enumerate(messages, start=1):
            searchable_text = _extract_searchable_message_text(message).strip()
            if not searchable_text:
                continue
            if normalized_query.lower() not in searchable_text.lower():
                continue

            matched_message_count += 1
            if len(excerpts) >= 3:
                continue
            excerpts.append(
                f"{index}. {_get_message_role_label(message)}: "
                f"{_normalize_search_excerpt(searchable_text, normalized_query)}"
            )

        if matched_message_count <= 0:
            continue

        results.append(
            StoredSessionSearchResult(
                session_id=summary.session_id,
                title=summary.title,
                updated_at=summary.updated_at,
                mode=summary.mode,
                approval_policy=summary.approval_policy,
                source_session_id=summary.source_session_id,
                matched_message_count=matched_message_count,
                excerpts=excerpts,
            )
        )
        if len(results) >= limit:
            break

    return results


def _serialize_stored_session(stored_session: StoredSession) -> dict[str, object]:
    """将已加载的历史会话重新组织为适合导出的结构。"""
    return {
        "session_id": stored_session.summary.session_id,
        "title": stored_session.summary.title,
        "created_at": stored_session.summary.created_at,
        "updated_at": stored_session.summary.updated_at,
        "mode": stored_session.summary.mode,
        "approval_policy": stored_session.summary.approval_policy,
        "turn_count": stored_session.summary.turn_count,
        "message_count": stored_session.summary.message_count,
        "source_session_id": stored_session.summary.source_session_id,
        "messages": messages_to_dict(stored_session.messages),
    }


def _render_stored_session_markdown(stored_session: StoredSession) -> str:
    """将历史会话渲染为适合人工排查的 Markdown 文档。"""
    summary = stored_session.summary
    lines = [
        "# 历史会话导出",
        "",
        f"- 会话 ID: {summary.session_id}",
        f"- 标题: {summary.title}",
        f"- 创建时间: {summary.created_at}",
        f"- 更新时间: {summary.updated_at}",
        f"- 模式: {summary.mode}",
        f"- 审批策略: {summary.approval_policy}",
        f"- 用户轮数: {summary.turn_count}",
        f"- 消息数: {summary.message_count}",
        f"- 来源会话: {summary.source_session_id or '无'}",
        "",
        "## 消息记录",
        "",
    ]
    for index, message in enumerate(stored_session.messages, start=1):
        lines.append(f"### {index}. {_get_message_role_label(message)}")
        lines.append("")
        content = _extract_searchable_message_text(message).strip() or "（空内容）"
        lines.append(content)
        lines.append("")
    return "\n".join(lines)


def export_session_history(
    session_id: str,
    *,
    output_path: Path | None = None,
    base_dir: Path | None = None,
) -> Path:
    """将指定历史会话导出为 Markdown 或 JSON 文件。"""
    stored_session = load_session_history(session_id, base_dir=base_dir)
    if output_path is None:
        target_path = get_session_storage_dir(base_dir) / f"{session_id}.md"
    else:
        target_path = Path(output_path).expanduser()

    if not target_path.suffix:
        target_path = target_path.with_suffix(".md")
    resolved_target_path = target_path.resolve()
    resolved_target_path.parent.mkdir(parents=True, exist_ok=True)

    export_suffix = resolved_target_path.suffix.lower()
    if export_suffix == ".json":
        exported_content = json.dumps(
            _serialize_stored_session(stored_session),
            ensure_ascii=False,
            indent=2,
        ) + "\n"
    elif export_suffix == ".md":
        exported_content = _render_stored_session_markdown(stored_session) + "\n"
    else:
        raise ValueError("历史会话导出仅支持 .md 或 .json 文件。")

    resolved_target_path.write_text(exported_content, encoding="utf-8")
    return resolved_target_path
