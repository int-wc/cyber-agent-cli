import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage
from typer.testing import CliRunner

from cyber_agent.cli.app import app


class FakeChatOpenAI:
    """
    用于 CLI 端到端测试的假模型。
    首轮返回工具调用请求，收到工具结果后流式输出最终回复。
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
                        "name": "scan_port",
                        "args": '{"target":"127.0.0.1","port":80}',
                        "id": "call_scan_port",
                        "index": 0,
                        "type": "tool_call_chunk",
                    }
                ],
            )
            return

        if isinstance(last_message, ToolMessage):
            yield AIMessageChunk(content="根据")
            yield AIMessageChunk(content=f"工具结果回复: {last_message.content}")
            return

        raise AssertionError(f"未处理的消息类型: {type(last_message)!r}")


class FakeSocket:
    """
    伪造底层 socket，避免测试访问真实网络。
    使用类变量记录调用，便于断言工具确实被执行。
    """

    connect_targets: list[tuple[str, int]] = []
    timeouts: list[float] = []

    def __init__(self, *args, **kwargs) -> None:
        pass

    def settimeout(self, timeout: float) -> None:
        self.__class__.timeouts.append(timeout)

    def connect_ex(self, target: tuple[str, int]) -> int:
        self.__class__.connect_targets.append(target)
        return 0

    def close(self) -> None:
        return None


class ApprovalCliFakeChatOpenAI:
    """
    用于审批链路端到端测试的假模型。
    首轮请求写文件，收到工具结果后输出最终总结。
    """

    target_path: Path | None = None

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.bound_tools = []

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = tools
        return self

    def stream(self, messages):
        last_message = messages[-1]
        if isinstance(last_message, HumanMessage):
            if self.__class__.target_path is None:
                raise AssertionError("测试未设置目标写入路径。")
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {
                        "name": "write_text_file",
                        "args": json.dumps(
                            {
                                "path": str(self.__class__.target_path),
                                "content": "auto-approved content",
                            },
                            ensure_ascii=False,
                        ),
                        "id": "call_write_file",
                        "index": 0,
                        "type": "tool_call_chunk",
                    }
                ],
            )
            return

        if isinstance(last_message, ToolMessage):
            yield AIMessageChunk(content="写入完成。")
            yield AIMessageChunk(content=str(last_message.content))
            return

        raise AssertionError(f"未处理的消息类型: {type(last_message)!r}")


class CliChatE2ETestCase(unittest.TestCase):
    def test_chat_command_can_run_agent_and_print_streaming_tool_result(self) -> None:
        """
        测试：CLI 交互模式能够显示流式输出、工具调用和工具结果。
        """
        FakeSocket.connect_targets = []
        FakeSocket.timeouts = []
        cli_runner = CliRunner()

        with (
            patch("cyber_agent.agent.runner.ChatOpenAI", FakeChatOpenAI),
            patch("cyber_agent.tools.security.socket.socket", FakeSocket),
        ):
            result = cli_runner.invoke(app, [], input="扫描一下 127.0.0.1 80\nquit\n")

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Cyber Agent CLI", result.output)
        self.assertIn("工具调用", result.output)
        self.assertIn("工具结果", result.output)
        self.assertIn("Agent", result.output)
        self.assertIn("根据工具结果回复", result.output)
        self.assertIn("127.0.0.1", result.output)
        self.assertIn("80", result.output)
        self.assertIn("再见", result.output)

        self.assertEqual(len(FakeSocket.connect_targets), 1)
        self.assertTrue(
            all(target == ("127.0.0.1", 80) for target in FakeSocket.connect_targets)
        )
        self.assertEqual(len(FakeSocket.timeouts), 1)
        self.assertTrue(all(timeout == 2.0 for timeout in FakeSocket.timeouts))

    def test_chat_command_can_auto_approve_high_risk_write_tool(self) -> None:
        """
        测试：CLI 可在自动审批策略下完成高风险写文件工具调用。
        """
        cli_runner = CliRunner()
        with TemporaryDirectory() as temp_dir:
            target_path = Path(temp_dir) / "approval-note.txt"
            ApprovalCliFakeChatOpenAI.target_path = target_path

            with patch("cyber_agent.agent.runner.ChatOpenAI", ApprovalCliFakeChatOpenAI):
                result = cli_runner.invoke(
                    app,
                    [
                        "--mode",
                        "authorized",
                        "--approval-policy",
                        "auto",
                        "--allow-path",
                        temp_dir,
                    ],
                    input="请写一个测试文件\nquit\n",
                )

            self.assertEqual(result.exit_code, 0)
            self.assertIn("审批请求", result.output)
            self.assertIn("已批准", result.output)
            self.assertIn("write_text_file", result.output)
            self.assertIn("工具结果", result.output)
            self.assertIn("写入完成", result.output)
            self.assertTrue(target_path.exists())
            self.assertEqual(
                target_path.read_text(encoding="utf-8"),
                "auto-approved content",
            )


if __name__ == "__main__":
    unittest.main()
