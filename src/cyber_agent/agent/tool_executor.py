"""共享工具调用逻辑，供降级执行器与主运行器共用，避免维护分歧。"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool


def normalize_tool_call_args(tool_call: dict[str, Any]) -> dict[str, Any]:
    """将模型返回的工具参数规范化为字典，兼容字符串形式的 JSON 参数。
    委托给 tools 模块的统一实现。"""
    from ..tools import normalize_tool_args as _normalize

    return _normalize(tool_call)


def invoke_tool_simple(
    tool_call: dict[str, Any],
    tool_registry: dict[str, BaseTool],
) -> ToolMessage:
    """执行单次工具调用并统一返回 ToolMessage（无审批、无执行控制器检查）。

    供 _FallbackCompiledGraph 等简化执行路径使用。
    """
    tool_name = str(tool_call.get("name", "")) or "unknown"
    tool_call_id = str(tool_call.get("id", ""))
    tool = tool_registry.get(tool_name)
    if tool is None:
        return ToolMessage(
            content=f"❌ 未知工具：{tool_name}",
            name=tool_name,
            tool_call_id=tool_call_id,
        )

    try:
        tool_result = str(tool.invoke(normalize_tool_call_args(tool_call)))
    except ValueError as exc:
        tool_result = f"❌ 工具参数错误：{exc}"
    except Exception as exc:
        tool_result = f"❌ 工具执行异常：{exc}"

    return ToolMessage(
        content=tool_result,
        name=tool_name,
        tool_call_id=tool_call_id,
    )
