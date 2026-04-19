import unittest
from typing import Any
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from cyber_agent.agent.core import create_agent_graph


class FakeToolCallingLLM:
    """
    用于测试的假模型：
    1. 首次收到用户消息时，返回工具调用请求。
    2. 收到工具执行结果后，返回最终回复。
    """

    def __init__(self) -> None:
        self.bound_tools: list[Any] = []

    def bind_tools(self, tools: list[Any]) -> "FakeToolCallingLLM":
        self.bound_tools = tools
        return self

    def invoke(self, messages: list[Any]) -> AIMessage:
        last_message = messages[-1]
        if isinstance(last_message, HumanMessage):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "echo_tool",
                        "args": {"text": last_message.content},
                        "id": "call_echo_tool",
                        "type": "tool_call",
                    }
                ],
            )

        if isinstance(last_message, ToolMessage):
            return AIMessage(content=f"final:{last_message.content}")

        raise AssertionError(f"未处理的消息类型: {type(last_message)!r}")


class AgentToolCallTestCase(unittest.TestCase):
    def test_create_agent_graph_can_call_tool_and_return_final_message(self) -> None:
        """
        测试：图中的 agent 节点能够发起工具调用，工具节点执行后再返回最终回复。
        """
        tool_inputs: list[str] = []

        @tool
        def echo_tool(text: str) -> str:
            """回显文本，便于验证工具是否真的被执行。"""
            tool_inputs.append(text)
            return f"processed:{text}"

        llm = FakeToolCallingLLM()
        app = create_agent_graph(llm, [echo_tool])

        final_state = app.invoke({"messages": [HumanMessage(content="hello-tool")]})
        messages = final_state["messages"]

        # 断言模型已经绑定工具，且工具确实被执行过。
        self.assertEqual(len(llm.bound_tools), 1)
        self.assertEqual(tool_inputs, ["hello-tool"])

        # 断言消息链路完整经过“用户 -> 工具调用 -> 工具结果 -> 最终回复”。
        self.assertIsInstance(messages[0], HumanMessage)
        self.assertIsInstance(messages[1], AIMessage)
        self.assertEqual(messages[1].tool_calls[0]["name"], "echo_tool")
        self.assertIsInstance(messages[2], ToolMessage)
        self.assertEqual(messages[2].content, "processed:hello-tool")
        self.assertIsInstance(messages[3], AIMessage)
        self.assertEqual(messages[3].content, "final:processed:hello-tool")

    def test_create_agent_graph_fallback_runner_can_execute_tool_chain(self) -> None:
        """
        测试：即使禁用 LangGraph，也应由内置降级执行器完成工具调用链路。
        """
        tool_inputs: list[str] = []

        @tool
        def echo_tool(text: str) -> str:
            """回显文本，便于验证降级执行器是否真的执行了工具。"""
            tool_inputs.append(text)
            return f"processed:{text}"

        llm = FakeToolCallingLLM()
        with patch("cyber_agent.agent.core.LANGGRAPH_AVAILABLE", False):
            app = create_agent_graph(llm, [echo_tool])

        final_state = app.invoke({"messages": [HumanMessage(content="hello-fallback")]})
        messages = final_state["messages"]

        self.assertEqual(tool_inputs, ["hello-fallback"])
        self.assertIsInstance(messages[0], HumanMessage)
        self.assertIsInstance(messages[1], AIMessage)
        self.assertIsInstance(messages[2], ToolMessage)
        self.assertEqual(messages[2].content, "processed:hello-fallback")
        self.assertIsInstance(messages[3], AIMessage)
        self.assertEqual(messages[3].content, "final:processed:hello-fallback")


if __name__ == "__main__":
    unittest.main()
