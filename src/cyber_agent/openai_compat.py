from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, BaseMessageChunk


def ensure_deepseek_reasoning_content_compat() -> None:
    """修补当前 langchain-openai 对 DeepSeek reasoning_content 的透传缺口。

    仅在 langchain-openai 内部 API 匹配预期签名时才执行 patch；
    若上游已升级或移除相关函数，则跳过并记录警告。
    """
    try:
        from langchain_openai.chat_models import base as openai_base
    except ModuleNotFoundError:
        return

    if getattr(openai_base, "_cyber_agent_deepseek_reasoning_patch", False):
        return

    required_functions = (
        "_convert_delta_to_message_chunk",
        "_convert_dict_to_message",
        "_convert_message_to_dict",
    )
    missing = [fn for fn in required_functions if not callable(getattr(openai_base, fn, None))]
    if missing:
        import warnings
        warnings.warn(
            f"langchain-openai 内部 API 已变更，跳过 DeepSeek reasoning_content 兼容 patch。"
            f"缺失函数：{', '.join(missing)}",
            RuntimeWarning,
        )
        return

    original_convert_delta = openai_base._convert_delta_to_message_chunk
    original_convert_dict = openai_base._convert_dict_to_message
    original_convert_message = openai_base._convert_message_to_dict

    def patched_convert_delta_to_message_chunk(
        payload: Mapping[str, Any],
        default_class: type[BaseMessageChunk],
    ) -> BaseMessageChunk:
        chunk = original_convert_delta(payload, default_class)
        reasoning_content = payload.get("reasoning_content")
        if reasoning_content is not None and isinstance(chunk, AIMessageChunk):
            chunk.additional_kwargs["reasoning_content"] = str(reasoning_content)
        return chunk

    def patched_convert_dict_to_message(payload: Mapping[str, Any]) -> BaseMessage:
        message = original_convert_dict(payload)
        reasoning_content = payload.get("reasoning_content")
        if reasoning_content is not None and isinstance(message, AIMessage):
            message.additional_kwargs["reasoning_content"] = str(reasoning_content)
        return message

    def patched_convert_message_to_dict(message: BaseMessage) -> dict[str, Any]:
        message_dict = original_convert_message(message)
        if isinstance(message, AIMessage):
            reasoning_content = message.additional_kwargs.get("reasoning_content")
            # 只透传非空值：空字符串或 None 不添加，防止网关创建
            # {"type": "thinking"}（缺少 thinking 字段）的 content block
            if reasoning_content:
                message_dict["reasoning_content"] = str(reasoning_content)
        return message_dict

    openai_base._convert_delta_to_message_chunk = patched_convert_delta_to_message_chunk
    openai_base._convert_dict_to_message = patched_convert_dict_to_message
    openai_base._convert_message_to_dict = patched_convert_message_to_dict
    openai_base._cyber_agent_deepseek_reasoning_patch = True


def prepare_messages_for_openai_compatible_service(
    messages: Sequence[BaseMessage],
    service_name: str,
    *,
    deepseek_thinking_enabled: bool = False,
) -> list[BaseMessage]:
    """按服务商整理消息。DeepSeek 始终保留 reasoning_content（API 要求回传），
    非 DeepSeek 服务剥除以避免上游拒绝。"""
    if service_name == "deepseek":
        if deepseek_thinking_enabled:
            return [_ensure_deepseek_reasoning_content(message) for message in messages]
        # thinking 关闭时仍需透传模型返回的 reasoning_content
        return list(messages)
    return [_strip_reasoning_content(message) for message in messages]


def _extract_text_from_content(
    content: str | list[str | dict],
) -> str:
    """从 LangChain 消息 content 字段中提取纯文本，兼容 str 和 list[dict] 格式。"""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text", "")))
    return "".join(parts)


def _ensure_deepseek_reasoning_content(message: BaseMessage) -> BaseMessage:
    """将 DeepSeek 格式的 reasoning_content 嵌入消息纯文本中。

    不再创建 list[dict] content blocks（type:thinking + type:text），
    因为 clauses:
      - 通过本地网关转发到 Claude 时，网关会自行处理 reasoning_content
      - 直连 DeepSeek anthropic 端点时不支持 type:thinking block
      - 直连 DeepSeek /v1 端点时 content 应为纯文本

    改为将 reasoning 作为 <thinking> 标记嵌入文本开头，并剥离
    additional_kwargs 中的 reasoning_content，避免 serialization 时
    作为顶层字段残留导致不稳定。
    """
    if not isinstance(message, AIMessage):
        return message

    reasoning_content = message.additional_kwargs.get("reasoning_content")

    if reasoning_content is not None:
        text_part = _extract_text_from_content(message.content)
        # 将 reasoning 作为标记嵌入纯文本，不创建 content block
        if reasoning_content:
            merged = f"<thinking>\n{reasoning_content}\n</thinking>\n\n{text_part}" if text_part else f"<thinking>\n{reasoning_content}\n</thinking>"
        else:
            merged = text_part

        additional_kwargs = {
            k: v for k, v in message.additional_kwargs.items()
            if k != "reasoning_content"
        }
        return AIMessage(
            content=merged,
            additional_kwargs=additional_kwargs,
            tool_calls=list(message.tool_calls) if message.tool_calls else [],
            id=message.id,
        )

    # 工具调用消息：确保没有空 reasoning_content 残留
    if message.tool_calls or "tool_calls" in message.additional_kwargs:
        additional_kwargs = dict(message.additional_kwargs)
        additional_kwargs.pop("reasoning_content", None)
        return message.model_copy(update={"additional_kwargs": additional_kwargs})

    return message


def _strip_reasoning_content(message: BaseMessage) -> BaseMessage:
    """切回 OpenAI 等服务时移除 DeepSeek 专属字段，避免上游拒绝请求。"""
    if not isinstance(message, AIMessage):
        return message
    if "reasoning_content" not in message.additional_kwargs:
        return message
    additional_kwargs = dict(message.additional_kwargs)
    additional_kwargs.pop("reasoning_content", None)
    return message.model_copy(update={"additional_kwargs": additional_kwargs})
