from __future__ import annotations

import json
import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool

try:
    from langgraph.graph import StateGraph
    from langgraph.prebuilt import ToolNode, tools_condition

    LANGGRAPH_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - 是否安装依赖由运行环境决定
    StateGraph = None
    ToolNode = None
    tools_condition = None
    LANGGRAPH_AVAILABLE = False

MAX_FALLBACK_TOOL_ROUNDS = 24


class AgentState(TypedDict):
    """用于在图节点之间传递消息列表。"""

    messages: Annotated[list[BaseMessage], operator.add]


def agent(state: AgentState, llm: Any, tools: list[BaseTool]) -> dict[str, list[AIMessage]]:
    """调用模型，让其决定是直接回复还是发起工具调用。"""
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def _normalize_tool_args(tool_call: dict[str, Any]) -> dict[str, Any]:
    """兼容字典和 JSON 字符串两种工具参数格式。"""
    raw_args = tool_call.get("args", {})
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        stripped_args = raw_args.strip()
        if not stripped_args:
            return {}
        parsed_args = json.loads(stripped_args)
        if not isinstance(parsed_args, dict):
            raise ValueError("工具参数必须是 JSON 对象。")
        return parsed_args
    raise ValueError("工具参数格式无效，必须是对象或 JSON 字符串。")


class _FallbackCompiledGraph:
    """LangGraph 不可用时的最小执行器，保证基础工具链路仍可验证。"""

    def __init__(self, llm: Any, tools: list[BaseTool]) -> None:
        self._llm = llm
        self._tools = list(tools)
        self._tool_registry = {tool.name: tool for tool in tools}

    def invoke(self, state: AgentState) -> AgentState:
        """模拟 agent -> tool -> agent 的循环，直到模型返回最终文本。"""
        messages = list(state.get("messages", []))
        if not messages:
            raise ValueError("初始状态缺少 messages。")

        for _ in range(MAX_FALLBACK_TOOL_ROUNDS):
            ai_message = agent({"messages": messages}, self._llm, self._tools)["messages"][0]
            messages.append(ai_message)

            tool_calls = list(ai_message.tool_calls or [])
            if not tool_calls:
                return {"messages": messages}

            for tool_call in tool_calls:
                messages.append(self._invoke_tool(tool_call))

        raise RuntimeError(
            "工具调用轮数超过降级执行器的安全上限，"
            f"当前已停止在 {MAX_FALLBACK_TOOL_ROUNDS} 轮。"
        )

    def _invoke_tool(self, tool_call: dict[str, Any]) -> ToolMessage:
        """执行单次工具调用，并统一返回 ToolMessage。"""
        tool_name = str(tool_call.get("name", "")) or "unknown"
        tool_call_id = str(tool_call.get("id", ""))
        tool = self._tool_registry.get(tool_name)
        if tool is None:
            tool_result = f"❌ 未知工具：{tool_name}"
        else:
            try:
                tool_result = str(tool.invoke(_normalize_tool_args(tool_call)))
            except ValueError as exc:
                tool_result = f"❌ 工具参数错误：{exc}"
            except Exception as exc:  # noqa: BLE001 - 需要把真实工具错误回传到消息链
                tool_result = f"❌ 工具执行异常：{exc}"

        return ToolMessage(
            content=tool_result,
            name=tool_name,
            tool_call_id=tool_call_id,
        )


def create_agent_graph(llm: Any, tools: list[BaseTool]):
    """优先使用 LangGraph；缺依赖时回退到内置最小执行器。"""
    if not LANGGRAPH_AVAILABLE or StateGraph is None or ToolNode is None or tools_condition is None:
        return _FallbackCompiledGraph(llm, tools)

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", lambda state: agent(state, llm, tools))
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")
    return workflow.compile()
