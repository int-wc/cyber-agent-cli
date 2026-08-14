"""事件钩子系统测试：订阅/发布/拦截/异常隔离。"""

from __future__ import annotations

import unittest

from cyber_agent.agent.event_bus import EventBus


class EventBusTestCase(unittest.TestCase):
    """事件总线订阅/发布。"""

    def setUp(self) -> None:
        self.bus = EventBus()

    def test_subscribe_and_publish(self) -> None:
        seen: list[tuple[str, dict]] = []

        def handler(event_type: str, payload: dict | None) -> None:
            seen.append((event_type, payload))

        self.bus.subscribe("tool_call", handler)
        self.bus.publish("tool_call", {"name": "scan_port"})
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][0], "tool_call")
        self.assertEqual(seen[0][1]["name"], "scan_port")

    def test_unsubscribe(self) -> None:
        seen: list[str] = []

        def handler(event_type: str, payload: dict | None) -> None:
            seen.append(event_type)

        unsubscribe = self.bus.subscribe("tool_result", handler)
        self.bus.publish("tool_result", None)
        unsubscribe()
        self.bus.publish("tool_result", None)
        self.assertEqual(len(seen), 1)

    def test_wildcard_subscribe(self) -> None:
        seen: list[str] = []
        self.bus.subscribe("", lambda et, p: seen.append(et))
        self.bus.publish("tool_call", None)
        self.bus.publish("tool_result", None)
        self.assertEqual(seen, ["tool_call", "tool_result"])

    def test_interceptor_can_rewrite_payload(self) -> None:
        """监听器返回 dict 可改写 payload（拦截语义）。"""

        def interceptor(event_type: str, payload: dict | None) -> dict:
            assert payload is not None
            return {**payload, "rewritten": True}

        final = self.bus.publish("tool_result", {"content": "原始"})
        # 无监听器时返回原始
        self.assertEqual(final, {"content": "原始"})

        self.bus.subscribe("tool_result", interceptor)
        final = self.bus.publish("tool_result", {"content": "原始"})
        self.assertTrue(final.get("rewritten"))

    def test_listener_exception_does_not_break_publish(self) -> None:
        def bad_handler(event_type: str, payload: dict | None) -> None:
            raise RuntimeError("boom")

        seen: list[str] = []

        def good_handler(event_type: str, payload: dict | None) -> None:
            seen.append(event_type)

        self.bus.subscribe("tool_call", bad_handler)
        self.bus.subscribe("tool_call", good_handler)
        self.bus.publish("tool_call", None)
        self.assertEqual(seen, ["tool_call"])

    def test_listener_count(self) -> None:
        self.bus.subscribe("a", lambda et, p: None)
        self.bus.subscribe("a", lambda et, p: None)
        self.bus.subscribe("b", lambda et, p: None)
        self.assertEqual(self.bus.listener_count("a"), 2)
        self.assertEqual(self.bus.listener_count("b"), 1)
        self.assertEqual(self.bus.listener_count("c"), 0)

    def test_subscribe_requires_callable(self) -> None:
        with self.assertRaises(TypeError):
            self.bus.subscribe("a", "not-callable")  # type: ignore[arg-type]


class RunnerEventBusIntegrationTestCase(unittest.TestCase):
    """AgentRunner 发布事件到总线的集成验证。"""

    def test_runner_publishes_tool_events_to_injected_bus(self) -> None:
        """runner 用注入的 EventBus 发布 TOOL_CALL/TOOL_RESULT。"""
        from unittest.mock import patch

        from langchain_core.messages import AIMessage, AIMessageChunk
        from langchain_core.tools import tool as lc_tool

        from cyber_agent.agent.mode import AgentMode
        from cyber_agent.agent.runner import AgentRunner

        bus = EventBus()
        seen: list[str] = []
        bus.subscribe("", lambda et, p: seen.append(et))

        @lc_tool("test_ping")
        def test_ping() -> str:
            """测试工具。"""
            return "pong"

        runner = AgentRunner(
            tools=[test_ping],
            mode=AgentMode.STANDARD,
            event_bus=bus,
        )

        # 流式 fake LLM：第一轮返回 tool_call，第二轮返回纯文本
        calls = [
            AIMessage(
                content="",
                tool_calls=[{"id": "call_1", "name": "test_ping", "args": {}}],
            ),
            AIMessage(content="完成"),
        ]

        class _FakeStreamLLM:
            def __init__(self, msgs: list) -> None:
                self._msgs = msgs
                self._idx = 0

            def bind_tools(self, tools, **kwargs):
                return self

            def stream(self, messages):
                if self._idx < len(self._msgs):
                    msg = self._msgs[self._idx]
                    self._idx += 1
                    yield AIMessageChunk(
                        content=msg.content,
                        tool_call_chunks=(
                            [{"index": 0, "id": "call_1", "name": "test_ping", "args": "{}"}]
                            if msg.tool_calls else []
                        ),
                    )
                else:
                    yield AIMessageChunk(content="完成")

        runner.llm = _FakeStreamLLM(calls)

        with patch.object(
            runner, "_build_tool_registry", return_value={"test_ping": test_ping}
        ):
            runner.run("ping一下")

        self.assertIn("tool_call", seen)
        self.assertIn("tool_result", seen)

        self.assertIn("tool_call", seen)
        self.assertIn("tool_result", seen)


if __name__ == "__main__":
    unittest.main()
