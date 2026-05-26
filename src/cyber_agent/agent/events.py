"""Agent 运行器事件类型枚举，替代分散在代码中的裸字符串。"""
from __future__ import annotations

from enum import StrEnum


class AgentEventType(StrEnum):
    """AgentRunner 在运行过程中产生的事件类型。"""

    TURN_START = "turn_start"
    RESPONSE_BEGIN = "response_begin"
    REASONING_TOKEN = "reasoning_token"
    RESPONSE_TOKEN = "response_token"
    RESPONSE_END = "response_end"
    RESPONSE_RETRY = "response_retry"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    APPROVAL_REQUEST = "approval_request"
    APPROVAL_RESULT = "approval_result"
    TURN_END = "turn_end"
