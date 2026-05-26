"""上下文窗口管理的辅助函数和常量。

从 runner.py 中提取的纯函数，不依赖 AgentRunner 实例状态。
"""
from __future__ import annotations

import json

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

LOCAL_CONTEXT_COMPACT_MIN_CHARS = 800
LOCAL_CONTEXT_COMPACT_MAX_CHARS = 12000


def format_message_for_context_summary(message: BaseMessage) -> str:
    """将消息压缩为可用于上下文摘要和调试的文本。"""
    role_label = "system"
    if isinstance(message, HumanMessage):
        role_label = "user"
    elif isinstance(message, AIMessage):
        role_label = "assistant"
    elif isinstance(message, ToolMessage):
        role_label = f"tool:{message.name or 'unknown'}"

    content = _extract_text_content(message.content).strip()
    if isinstance(message, AIMessage) and message.tool_calls and not content:
        content = f"工具调用: {json.dumps(message.tool_calls, ensure_ascii=False)}"
    if not content:
        content = "（空内容）"
    return f"{role_label}: {content}"


def _extract_text_content(content: str | list[str | dict]) -> str:
    """从 LangChain 消息内容结构中提取纯文本。"""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text", "")))
    return "".join(parts)


def estimate_message_token_count(message: BaseMessage) -> int:
    """保守估算消息 token 数，避免没有 tokenizer 时低估超长中文或 JSON。"""
    return max(1, len(format_message_for_context_summary(message)))


def compact_text_for_model_context(text: str, max_chars: int) -> str:
    """将超长单条消息压缩为首尾片段，完整原文仍保存在本地历史中。"""
    normalized_max_chars = max(LOCAL_CONTEXT_COMPACT_MIN_CHARS, max_chars)
    if len(text) <= normalized_max_chars:
        return text

    marker = (
        "[上下文保护] 该条消息过长，已在发送给模型前做本地压缩；"
        f"原始长度约 {len(text)} 字符，完整内容仍保存在本地历史中。"
    )
    marker_budget = len(marker) + 32
    slice_budget = max(LOCAL_CONTEXT_COMPACT_MIN_CHARS, normalized_max_chars - marker_budget)
    head_chars = max(1, slice_budget * 2 // 3)
    tail_chars = max(1, slice_budget - head_chars)
    return (
        f"{marker}\n\n"
        "【开头片段】\n"
        f"{text[:head_chars]}\n\n"
        "【结尾片段】\n"
        f"{text[-tail_chars:]}"
    )


def copy_message_with_content(message: BaseMessage, content: str) -> BaseMessage:
    """复制 LangChain 消息并替换 content，保留 tool_call_id 等结构字段。"""
    model_copy = getattr(message, "model_copy", None)
    if callable(model_copy):
        return model_copy(update={"content": content})
    return message.copy(update={"content": content})
