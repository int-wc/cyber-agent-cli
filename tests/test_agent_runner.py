import json
import unittest
from time import sleep
from unittest.mock import patch

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.tools import tool

from cyber_agent.agent.approval import ApprovalDecision
from cyber_agent.agent.mode import AgentMode
from cyber_agent.agent.runner import AgentRunner, iter_stream_characters
from cyber_agent.execution_control import ExecutionInterruptedError
from cyber_agent.tools.metadata import attach_tool_risk


class FakeChatOpenAI:
    """
    用于测试 AgentRunner 的假模型。
    通过 stream 返回 token 片段和工具调用片段，模拟真实流式行为。
    """

    human_turn_message_counts: list[int] = []
    init_kwargs_history: list[dict] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.bound_tools = []
        self.__class__.init_kwargs_history.append(kwargs)

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


class InterruptibleFakeChatOpenAI:
    """
    用于中断测试的假模型。
    首轮连续返回多个分片，便于测试 /stop 对流式生成的打断。
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
            yield AIMessageChunk(content="A")
            sleep(0.02)
            yield AIMessageChunk(content="B")
            sleep(0.02)
            yield AIMessageChunk(content="C")
            return

        raise AssertionError(f"未处理的消息类型: {type(last_message)!r}")


class LongSequenceFakeChatOpenAI:
    """
    用于长链路测试的假模型。
    会连续发起多轮不同参数的工具调用，最后再给出最终答复。
    """

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.bound_tools = []

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = tools
        return self

    def stream(self, messages):
        last_message = messages[-1]
        completed_steps = sum(
            1
            for message in messages
            if isinstance(message, ToolMessage) and message.name == "echo_tool"
        )
        next_step = completed_steps + 1

        if isinstance(last_message, HumanMessage):
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {
                        "name": "echo_tool",
                        "args": json.dumps({"text": f"step-{next_step}"}),
                        "id": f"call_step_{next_step}",
                        "index": 0,
                        "type": "tool_call_chunk",
                    }
                ],
            )
            return

        if isinstance(last_message, ToolMessage):
            if next_step <= 14:
                yield AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": "echo_tool",
                            "args": json.dumps({"text": f"step-{next_step}"}),
                            "id": f"call_step_{next_step}",
                            "index": 0,
                            "type": "tool_call_chunk",
                        }
                    ],
                )
                return

            yield AIMessageChunk(content="长链路完成")
            return

        raise AssertionError(f"未处理的消息类型: {type(last_message)!r}")


class RepeatedLoopFakeChatOpenAI:
    """
    用于循环检测测试的假模型。
    无论前文如何，都会重复请求同一个工具调用。
    """

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.bound_tools = []

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = tools
        return self

    def stream(self, messages):
        last_message = messages[-1]
        if isinstance(last_message, HumanMessage | ToolMessage):
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {
                        "name": "echo_tool",
                        "args": json.dumps({"text": "same-loop"}),
                        "id": "call_loop",
                        "index": 0,
                        "type": "tool_call_chunk",
                    }
                ],
            )
            return

        raise AssertionError(f"未处理的消息类型: {type(last_message)!r}")


class TransientStreamStartErrorFakeChatOpenAI:
    """
    用于测试模型在首个分片前断流时，运行器会做一次安全重试。
    首轮先请求工具；工具结果回填后，第一次续跑抛出上游 empty_stream，
    第二次再正常给出最终回答。
    """

    tool_message_attempt_count = 0

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
                        "name": "echo_tool",
                        "args": json.dumps({"text": "retry-me"}),
                        "id": "call_retry_once",
                        "index": 0,
                        "type": "tool_call_chunk",
                    }
                ],
            )
            return

        if isinstance(last_message, ToolMessage):
            self.__class__.tool_message_attempt_count += 1
            if self.__class__.tool_message_attempt_count == 1:
                raise RuntimeError(
                    "Error code: 500 - {'error': {'message': "
                    "'empty_stream: upstream stream closed before first payload', "
                    "'type': 'server_error', 'code': 'internal_server_error'}}"
                )
            yield AIMessageChunk(content="final:")
            yield AIMessageChunk(content=str(last_message.content))
            return

        raise AssertionError(f"未处理的消息类型: {type(last_message)!r}")


class EmptyFinalReplyFakeChatOpenAI:
    """
    用于测试模型在工具执行完成后第一次返回空最终消息时，运行器会补一次重试。
    """

    tool_message_attempt_count = 0

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
                        "name": "echo_tool",
                        "args": json.dumps({"text": "empty-final"}),
                        "id": "call_empty_final",
                        "index": 0,
                        "type": "tool_call_chunk",
                    }
                ],
            )
            return

        if isinstance(last_message, ToolMessage):
            self.__class__.tool_message_attempt_count += 1
            if self.__class__.tool_message_attempt_count == 1:
                yield AIMessageChunk(content="")
                return
            yield AIMessageChunk(content="最终总结:")
            yield AIMessageChunk(content=str(last_message.content))
            return

        raise AssertionError(f"未处理的消息类型: {type(last_message)!r}")


class AlwaysEmptyFinalReplyFakeChatOpenAI:
    """
    用于测试模型连续返回空最终消息时，运行器会明确报错而不是发送空回复。
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
                        "name": "echo_tool",
                        "args": json.dumps({"text": "always-empty-final"}),
                        "id": "call_always_empty_final",
                        "index": 0,
                        "type": "tool_call_chunk",
                    }
                ],
            )
            return

        if isinstance(last_message, ToolMessage):
            yield AIMessageChunk(content="")
            return

        raise AssertionError(f"未处理的消息类型: {type(last_message)!r}")


class AgentRunnerTestCase(unittest.TestCase):
    def test_iter_stream_characters_splits_stream_chunk_into_single_characters(self) -> None:
        """
        测试：统一流式分片会被拆成单字符，供 CLI 和 TUI 复用逐字输出。
        """

        self.assertEqual(
            iter_stream_characters("根据工具结果回复"),
            list("根据工具结果回复"),
        )

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

    def test_runner_emits_response_token_events_character_by_character(self) -> None:
        """
        测试：流式事件回调按字符触发，避免界面层受模型原始分片粒度影响。
        """

        @tool
        def echo_tool(text: str) -> str:
            """回显文本。"""
            return f"processed:{text}"

        token_events: list[str] = []

        def capture_event(event_type: str, payload: object) -> None:
            if event_type == "response_token":
                token_events.append(str(payload))

        with patch("cyber_agent.agent.runner.ChatOpenAI", FakeChatOpenAI):
            runner = AgentRunner([echo_tool])
            response = runner.run(
                "first",
                verbose=False,
                event_handler=capture_event,
            )

        self.assertEqual(response, "final:processed:first")
        self.assertEqual(token_events, list("final:processed:first"))

    def test_runner_can_be_interrupted_during_streaming_response(self) -> None:
        """
        测试：收到停止请求后，运行器会在流式生成过程中抛出中断异常。
        """

        token_events: list[str] = []
        runner: AgentRunner | None = None

        def capture_event(event_type: str, payload: object) -> None:
            if event_type != "response_token":
                return
            token_events.append(str(payload))
            if runner is not None and len(token_events) == 1:
                runner.execution_controller.request_stop("测试中主动停止")

        with patch("cyber_agent.agent.runner.ChatOpenAI", InterruptibleFakeChatOpenAI):
            runner = AgentRunner([])
            with self.assertRaises(ExecutionInterruptedError):
                runner.run(
                    "interrupt-me",
                    verbose=False,
                    event_handler=capture_event,
                )

        self.assertEqual(token_events, ["A"])

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

    def test_runner_compresses_older_messages_when_context_is_too_large(self) -> None:
        """
        测试：当完整历史超过上下文阈值时，会保留最近消息并插入压缩摘要。
        """
        with patch("cyber_agent.agent.runner.ChatOpenAI", FakeChatOpenAI):
            runner = AgentRunner(
                [],
                max_context_chars=80,
                context_keep_recent_messages=2,
            )
            runner.history.extend(
                [
                    HumanMessage(content="a" * 40),
                    AIMessage(content="b" * 40),
                    HumanMessage(content="c" * 40),
                    AIMessage(content="d" * 40),
                ]
            )

            with patch.object(
                runner,
                "_summarize_messages_for_context",
                return_value="压缩摘要",
            ) as mock_summarize:
                model_messages = runner.get_model_context_snapshot()

        self.assertEqual(runner.compressed_summary, "压缩摘要")
        self.assertEqual(runner.compressed_message_count, 2)
        self.assertEqual(mock_summarize.call_count, 1)
        self.assertEqual(len(model_messages), 4)
        self.assertIn("压缩摘要", model_messages[1].content)
        self.assertEqual(model_messages[-2].content, "c" * 40)
        self.assertEqual(model_messages[-1].content, "d" * 40)

    def test_runner_compression_boundary_can_skip_orphan_tool_messages(self) -> None:
        """
        测试：压缩边界若落在工具回合中间，应自动前移，避免模型上下文里留下孤立 ToolMessage。
        """
        with patch("cyber_agent.agent.runner.ChatOpenAI", FakeChatOpenAI):
            runner = AgentRunner(
                [],
                max_context_chars=40,
                context_keep_recent_messages=4,
            )
            runner.history.extend(
                [
                    HumanMessage(content="旧问题"),
                    AIMessage(content="", tool_calls=[{"id": "call-001", "name": "echo_tool", "args": {"text": "hi"}}]),
                    ToolMessage(content="工具结果", name="echo_tool", tool_call_id="call-001"),
                    AIMessage(content="旧回答"),
                    HumanMessage(content="新问题"),
                    AIMessage(content="新回答"),
                ]
            )

            with patch.object(
                runner,
                "_summarize_messages_for_context",
                return_value="压缩摘要",
            ):
                model_messages = runner.get_model_context_snapshot()

        self.assertEqual(runner.compressed_summary, "压缩摘要")
        self.assertEqual(runner.compressed_message_count, 3)
        self.assertEqual(len(model_messages), 5)
        self.assertIn("压缩摘要", model_messages[1].content)
        self.assertIsInstance(model_messages[2], AIMessage)
        self.assertEqual(model_messages[2].content, "旧回答")
        self.assertFalse(any(isinstance(message, ToolMessage) for message in model_messages[2:3]))

    def test_runner_can_use_and_switch_openai_compatible_service_config(self) -> None:
        """
        测试：运行器支持非 openai 服务商，并可在当前会话内切换模型配置。
        """
        FakeChatOpenAI.init_kwargs_history = []

        with patch("cyber_agent.agent.runner.ChatOpenAI", FakeChatOpenAI):
            runner = AgentRunner(
                [],
                service_name="deepseek",
                model_name="deepseek-chat",
            )
            self.assertEqual(FakeChatOpenAI.init_kwargs_history, [])
            runner._get_llm()
            runner.update_llm_config(
                service_name="openai",
                model_name="gpt-5.4-mini",
                base_url="https://example.test/v1",
            )
            self.assertIsNone(runner.llm)
            runner._get_llm()

        self.assertEqual(runner.service, "openai")
        self.assertEqual(runner.model_name, "gpt-5.4-mini")
        self.assertEqual(runner.base_url, "http://localhost:8317/")
        self.assertEqual(FakeChatOpenAI.init_kwargs_history[0]["model"], "deepseek-chat")
        self.assertEqual(
            FakeChatOpenAI.init_kwargs_history[0]["base_url"],
            "http://localhost:8317/",
        )
        self.assertEqual(
            FakeChatOpenAI.init_kwargs_history[0]["extra_body"],
            {"provider": "deepseek", "thinking": {"type": "disabled"}},
        )
        self.assertEqual(FakeChatOpenAI.init_kwargs_history[-1]["model"], "gpt-5.4-mini")
        self.assertEqual(
            FakeChatOpenAI.init_kwargs_history[-1]["base_url"],
            "http://localhost:8317/",
        )
        self.assertEqual(
            FakeChatOpenAI.init_kwargs_history[-1]["extra_body"],
            {"provider": "openai"},
        )

    def test_runner_allows_long_non_repeating_tool_sequences(self) -> None:
        """
        测试：正常推进且每轮都在变化的长工具链，不应被旧的低轮数上限提前终止。
        """

        @tool
        def echo_tool(text: str) -> str:
            """回显文本。"""
            return f"processed:{text}"

        with patch("cyber_agent.agent.runner.ChatOpenAI", LongSequenceFakeChatOpenAI):
            runner = AgentRunner([echo_tool])
            response = runner.run("请执行一个较长的工具链", verbose=False)

        self.assertEqual(response, "长链路完成")

    def test_runner_stops_identical_tool_loop_with_explicit_reason(self) -> None:
        """
        测试：只有出现重复工具调用循环时，才会被提前终止并返回明确原因。
        """

        @tool
        def echo_tool(text: str) -> str:
            """回显文本。"""
            return f"processed:{text}"

        with patch("cyber_agent.agent.runner.ChatOpenAI", RepeatedLoopFakeChatOpenAI):
            runner = AgentRunner([echo_tool])
            with self.assertRaises(RuntimeError) as captured:
                runner.run("进入循环", verbose=False)

        self.assertIn("重复工具调用循环", str(captured.exception))

    def test_runner_retries_transient_empty_stream_before_first_payload(self) -> None:
        """
        测试：若模型在首个分片前遇到上游 empty_stream，运行器会重试一次，
        且不会重复执行已经完成的工具调用。
        """

        tool_inputs: list[str] = []
        TransientStreamStartErrorFakeChatOpenAI.tool_message_attempt_count = 0

        @tool
        def echo_tool(text: str) -> str:
            """回显文本，用于确认重试不会重复执行工具。"""
            tool_inputs.append(text)
            return f"processed:{text}"

        with patch("cyber_agent.agent.runner.ChatOpenAI", TransientStreamStartErrorFakeChatOpenAI):
            runner = AgentRunner([echo_tool])
            response = runner.run("请继续", verbose=False)

        self.assertEqual(response, "final:processed:retry-me")
        self.assertEqual(tool_inputs, ["retry-me"])
        self.assertEqual(
            TransientStreamStartErrorFakeChatOpenAI.tool_message_attempt_count,
            2,
        )

    def test_runner_retries_empty_final_response_after_tools(self) -> None:
        """
        测试：工具执行完成后模型第一次返回空最终消息时，会自动重试并拿到总结。
        """
        tool_inputs: list[str] = []
        retry_events: list[object] = []
        EmptyFinalReplyFakeChatOpenAI.tool_message_attempt_count = 0

        @tool
        def echo_tool(text: str) -> str:
            """返回输入文本，便于验证重试不会重复执行工具。"""
            tool_inputs.append(text)
            return f"processed:{text}"

        def capture_event(event_type: str, payload: object) -> None:
            if event_type == "response_retry":
                retry_events.append(payload)

        with patch("cyber_agent.agent.runner.ChatOpenAI", EmptyFinalReplyFakeChatOpenAI):
            runner = AgentRunner([echo_tool])
            response = runner.run(
                "请执行后总结",
                verbose=False,
                event_handler=capture_event,
            )

        self.assertEqual(response, "最终总结:processed:empty-final")
        self.assertEqual(tool_inputs, ["empty-final"])
        self.assertEqual(EmptyFinalReplyFakeChatOpenAI.tool_message_attempt_count, 2)
        self.assertEqual(len(retry_events), 1)
        self.assertIsInstance(runner.history[-1], AIMessage)
        self.assertEqual(runner.history[-1].content, "最终总结:processed:empty-final")

    def test_runner_rejects_repeated_empty_final_response_after_retry(self) -> None:
        """
        测试：模型连续返回空最终消息时，仍会明确报错且不会保存空 AI 消息。
        """
        tool_inputs: list[str] = []

        @tool
        def echo_tool(text: str) -> str:
            """返回输入文本，便于验证工具只执行一次。"""
            tool_inputs.append(text)
            return f"processed:{text}"

        with patch("cyber_agent.agent.runner.ChatOpenAI", AlwaysEmptyFinalReplyFakeChatOpenAI):
            runner = AgentRunner([echo_tool])
            with self.assertRaises(RuntimeError) as captured:
                runner.run("请执行后总结", verbose=False)

        self.assertIn("模型返回空最终回复", str(captured.exception))
        self.assertEqual(tool_inputs, ["always-empty-final"])
        self.assertIsInstance(runner.history[-1], ToolMessage)
        self.assertEqual(runner.history[-1].content, "processed:always-empty-final")

    def test_runner_respects_approval_handler_for_high_risk_tools(self) -> None:
        """
        测试：高风险工具若未通过审批，应将拒绝结果回填给模型。
        """

        @tool("write_text_file")
        def write_text_file(path: str, content: str) -> str:
            """写文件。"""
            raise AssertionError("审批拒绝后不应真正执行写文件工具。")

        write_text_file = attach_tool_risk(write_text_file, "write")

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


    def test_runner_can_be_created_without_langchain_openai_until_model_is_used(self) -> None:
        """
        测试：缺少 langchain_openai 时，非模型路径仍可初始化运行器，真正调用模型时再报错。
        """
        import_error = ModuleNotFoundError("No module named 'langchain_openai'")

        with (
            patch("cyber_agent.agent.runner.ChatOpenAI", None),
            patch("cyber_agent.agent.runner.LANGCHAIN_OPENAI_IMPORT_ERROR", import_error),
        ):
            runner = AgentRunner([])

            self.assertEqual(runner.get_turn_count(), 0)
            self.assertEqual(runner.get_context_diagnostics()["history_message_count"], 1)

            with self.assertRaises(ModuleNotFoundError) as captured:
                runner.run("hello", verbose=False)

        self.assertIn("langchain_openai", str(captured.exception))


if __name__ == "__main__":
    unittest.main()
