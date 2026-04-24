from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, BaseMessageChunk


def ensure_deepseek_reasoning_content_compat() -> None:
    """修补当前 langchain-openai 对 DeepSeek reasoning_content 的透传缺口。"""
    try:
        from langchain_openai.chat_models import base as openai_base
    except ModuleNotFoundError:
        return

    if getattr(openai_base, "_cyber_agent_deepseek_reasoning_patch", False):
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
            if reasoning_content is not None:
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
    """按服务商整理消息，DeepSeek thinking 工具调用轮次需携带 reasoning_content。"""
    if service_name == "deepseek" and deepseek_thinking_enabled:
        return [_ensure_deepseek_reasoning_content(message) for message in messages]
    return [_strip_reasoning_content(message) for message in messages]


def _ensure_deepseek_reasoning_content(message: BaseMessage) -> BaseMessage:
    """兼容旧历史：DeepSeek 工具调用 assistant 消息至少要有 reasoning_content 字段。"""
    if not isinstance(message, AIMessage):
        return message
    if not message.tool_calls and "tool_calls" not in message.additional_kwargs:
        return message
    if "reasoning_content" in message.additional_kwargs:
        return message
    additional_kwargs = dict(message.additional_kwargs)
    additional_kwargs["reasoning_content"] = ""
    return message.model_copy(update={"additional_kwargs": additional_kwargs})


def _strip_reasoning_content(message: BaseMessage) -> BaseMessage:
    """切回 OpenAI 等服务时移除 DeepSeek 专属字段，避免上游拒绝请求。"""
    if not isinstance(message, AIMessage):
        return message
    if "reasoning_content" not in message.additional_kwargs:
        return message
    additional_kwargs = dict(message.additional_kwargs)
    additional_kwargs.pop("reasoning_content", None)
    return message.model_copy(update={"additional_kwargs": additional_kwargs})
