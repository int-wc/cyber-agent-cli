import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.tools import tool

from cyber_agent.agent.approval import ApprovalDecision
from cyber_agent.agent.mode import AgentMode
from cyber_agent.agent.runner import AgentRunner


class FakeChatOpenAI:
    """
    用于测试 AgentRunner 的假模型。
    通过 stream 返回 token 片段和工具调用片段，模拟真实流式行为。
    """

    human_turn_message_counts: list[int] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.bound_tools = []

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = tools
        return self

    def stream(self, messages):
        last_message = messages[-1]
        if isinstance(last_message, HumanMessage):
            self.__class__.human_turn_message_counts.append(len(messages))
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {
                        "name": "echo_tool",
                        "args": f'{{"text":"{last_message.content}"}}',
                        "id": f"call_{last_message.content}",
                        "index": 0,
                        "type": "tool_call_chunk",
                    }
                ],
            )
            return

        if isinstance(last_message, ToolMessage):
            yield AIMessageChunk(content="final:")
            yield AIMessageChunk(content=str(last_message.content))
            return

        raise AssertionError(f"未处理的消息类型: {type(last_message)!r}")


class ApprovalFakeChatOpenAI:
    """
    用于审批测试的假模型。
    第一次请求写工具，第二次根据审批结果输出最终结论。
    """

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.bound_tools = []

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = tools
        return self

    def stream(self, messages):
        last_message = messages[-1]
        if isinstance(last_message, HumanMessage):
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {
                        "name": "write_text_file",
                        "args": '{"path":"note.txt","content":"unsafe"}',
                        "id": "call_write",
                        "index": 0,
                        "type": "tool_call_chunk",
                    }
                ],
            )
            return

        if isinstance(last_message, ToolMessage):
            yield AIMessageChunk(content="审批结果:")
            yield AIMessageChunk(content=str(last_message.content))
            return

        raise AssertionError(f"未处理的消息类型: {type(last_message)!r}")


class AgentRunnerTestCase(unittest.TestCase):
    def test_runner_keeps_history_and_does_not_duplicate_tool_execution(self) -> None:
        """
        测试：运行器应保留会话历史，且每轮工具只执行一次。
        """
        FakeChatOpenAI.human_turn_message_counts = []
        tool_inputs: list[str] = []

        @tool
        def echo_tool(text: str) -> str:
            """回显文本，便于测试工具执行次数。"""
            tool_inputs.append(text)
            return f"processed:{text}"

        with patch("cyber_agent.agent.runner.ChatOpenAI", FakeChatOpenAI):
            runner = AgentRunner([echo_tool])
            first_response = runner.run("first", verbose=False)
            second_response = runner.run("second", verbose=False)

        self.assertEqual(first_response, "final:processed:first")
        self.assertEqual(second_response, "final:processed:second")
        self.assertEqual(tool_inputs, ["first", "second"])
        self.assertEqual(FakeChatOpenAI.human_turn_message_counts, [2, 6])
        self.assertEqual(runner.get_turn_count(), 2)

        runner.reset()
        self.assertEqual(runner.get_turn_count(), 0)

    def test_runner_can_switch_mode_and_reset_history(self) -> None:
        """
        测试：切换模式后应更新模式并重置上下文。
        """
        with patch("cyber_agent.agent.runner.ChatOpenAI", FakeChatOpenAI):
            runner = AgentRunner([], mode=AgentMode.STANDARD)
            runner.history.append(HumanMessage(content="keep"))
            runner.switch_mode(AgentMode.AUTHORIZED)

        self.assertEqual(runner.mode, AgentMode.AUTHORIZED)
        self.assertEqual(runner.get_turn_count(), 0)
        self.assertEqual(len(runner.history), 1)
        self.assertIn("授权模式", runner.history[0].content)

    def test_runner_respects_approval_handler_for_high_risk_tools(self) -> None:
        """
        测试：高风险工具若未通过审批，应将拒绝结果回填给模型。
        """

        @tool("write_text_file", extras={"risk": "write"})
        def write_text_file(path: str, content: str) -> str:
            """写文件。"""
            raise AssertionError("审批拒绝后不应真正执行写文件工具。")

        def reject_all(tool, tool_call):
            return ApprovalDecision(False, "测试中拒绝所有写入。")

        with patch("cyber_agent.agent.runner.ChatOpenAI", ApprovalFakeChatOpenAI):
            runner = AgentRunner([write_text_file])
            response = runner.run(
                "请写文件",
                verbose=False,
                approval_handler=reject_all,
            )

        self.assertIn("审批结果:", response)
        self.assertIn("测试中拒绝所有写入", response)


if __name__ == "__main__":
    unittest.main()
