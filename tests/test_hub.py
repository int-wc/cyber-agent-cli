from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from cyber_agent.agent.approval import ApprovalPolicy
from cyber_agent.agent.events import AgentEventType
from cyber_agent.agent.mode import AgentMode
from cyber_agent.execution_control import ExecutionInterruptedError
from cyber_agent.cli.app import (
    _is_hub_control_command,
    _parse_feishu_broadcast_chat_ids,
    infer_execution_profile,
)
from cyber_agent.cli.webhook_feishu import _save_feishu_session_state
from cyber_agent.hub import CyberAgentHub, HubTaskSource
from cyber_agent.session_store import load_session_history, save_session_history


class _FakeExecutionController:
    def __init__(self) -> None:
        self.stop_reasons: list[str] = []
        self.stop_event = threading.Event()

    def request_stop(self, reason: str) -> bool:
        self.stop_reasons.append(reason)
        self.stop_event.set()
        return True


class _FakeRunner:
    def __init__(self) -> None:
        self.mode = AgentMode.STANDARD
        self.history: list = []
        self.execution_controller = _FakeExecutionController()
        self.run_inputs: list[str] = []
        self.reset_count = 0
        self.restored_messages: list | None = None

    def run(self, text, *, verbose, event_handler, approval_handler):
        _ = verbose, approval_handler
        self.run_inputs.append(text)
        self.history.append(HumanMessage(content=text))
        event_handler(AgentEventType.TURN_START, {"input": text})
        event_handler(AgentEventType.RESPONSE_BEGIN, {})
        event_handler(AgentEventType.RESPONSE_TOKEN, "reply")
        reply = f"reply: {text}"
        self.history.append(AIMessage(content=reply))
        event_handler(
            AgentEventType.RESPONSE_END,
            {"content": reply, "has_tool_calls": False},
        )
        event_handler(AgentEventType.HISTORY_UPDATED, {"message_type": "ai"})
        event_handler(AgentEventType.TURN_END, {"input_tokens": 1, "output_tokens": 1})
        return reply

    def get_history_snapshot(self):
        return list(self.history)

    def get_turn_count(self):
        return len(self.run_inputs)

    def reset(self):
        self.reset_count += 1
        self.history = []

    def restore_history(self, messages):
        self.restored_messages = list(messages)
        self.history = list(messages)


class _BlockingRunner(_FakeRunner):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()

    def run(self, text, *, verbose, event_handler, approval_handler):
        _ = verbose, approval_handler
        self.run_inputs.append(text)
        self.history.append(HumanMessage(content=text))
        self.started.set()
        if not self.execution_controller.stop_event.wait(timeout=2.0):
            raise AssertionError("blocking runner was not stopped")
        raise ExecutionInterruptedError("stopped for session switch")


def _build_hub(base_dir: Path) -> tuple[CyberAgentHub, _FakeRunner, dict[str, object]]:
    runner = _FakeRunner()
    runtime_context: dict[str, object] = {
        "session_id": "test-session",
        "session_source_id": None,
        "approval_policy": ApprovalPolicy.AUTO,
        "_recent_inputs": [],
        "multi_agent_enabled": False,
    }
    hub = CyberAgentHub(
        runner=runner,
        runtime_context=runtime_context,
        approval_handler_factory=lambda context: None,
        detect_task_complexity=lambda text: False,
        run_multi_agent_turn=lambda text, runner, context, event_handler=None: None,
        renderless_event_handler_factory=lambda runner, context, inner: inner,
        base_dir=base_dir,
    )
    return hub, runner, runtime_context


def _build_hub_with_runner(
    base_dir: Path,
    runner: _FakeRunner,
) -> tuple[CyberAgentHub, dict[str, object]]:
    runtime_context: dict[str, object] = {
        "session_id": "test-session",
        "session_source_id": None,
        "approval_policy": ApprovalPolicy.AUTO,
        "_recent_inputs": [],
        "multi_agent_enabled": False,
    }
    hub = CyberAgentHub(
        runner=runner,
        runtime_context=runtime_context,
        approval_handler_factory=lambda context: None,
        detect_task_complexity=lambda text: False,
        run_multi_agent_turn=lambda text, runner, context, event_handler=None: None,
        renderless_event_handler_factory=lambda runner, context, inner: inner,
        base_dir=base_dir,
    )
    return hub, runtime_context


class CyberAgentHubTestCase(unittest.TestCase):
    def test_hub_control_command_classifier_keeps_session_list_builtin(self) -> None:
        self.assertTrue(_is_hub_control_command("/stop"))
        self.assertTrue(_is_hub_control_command("/new"))
        self.assertTrue(_is_hub_control_command("/session new"))
        self.assertTrue(_is_hub_control_command("/session load abc"))
        self.assertTrue(_is_hub_control_command("/session use abc"))
        self.assertFalse(_is_hub_control_command("/session"))
        self.assertFalse(_is_hub_control_command("/session list"))
        self.assertFalse(_is_hub_control_command("/tools"))

    def test_execution_profile_auto_infers_aggressive_from_full_authorization(self) -> None:
        context = {
            "mode": AgentMode.AUTHORIZED,
            "approval_policy": ApprovalPolicy.AUTO,
            "auto_decision": True,
            "allowed_roots": [Path("/")],
            "execution_profile": "auto",
        }

        self.assertEqual(infer_execution_profile(context), "aggressive")

    def test_execution_profile_auto_stays_conservative_without_root_access(self) -> None:
        context = {
            "mode": AgentMode.AUTHORIZED,
            "approval_policy": ApprovalPolicy.AUTO,
            "auto_decision": True,
            "allowed_roots": [Path("/tmp/project")],
            "execution_profile": "auto",
        }

        self.assertEqual(infer_execution_profile(context), "conservative")

    def test_hub_dispatches_message_broadcasts_and_persists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hub, runner, runtime_context = _build_hub(Path(tmp))
            events = []
            hub.subscribe(events.append)

            hub.start()
            hub.submit("hello", source=HubTaskSource("cli", "cli"))

            self.assertTrue(hub.wait_until_idle(2.0))
            hub.stop()

            self.assertEqual(runner.run_inputs, ["hello"])
            event_types = [event.type for event in events]
            self.assertIn("task_queued", event_types)
            self.assertIn("task_started", event_types)
            self.assertIn(AgentEventType.RESPONSE_TOKEN.value, event_types)
            self.assertIn("task_finished", event_types)
            finished = next(event for event in events if event.type == "task_finished")
            self.assertEqual(finished.payload["reply_text"], "reply: hello")

            stored = load_session_history("test-session", base_dir=Path(tmp))
            self.assertEqual(stored.summary.turn_count, 1)
            self.assertEqual(runtime_context["_recent_inputs"], ["hello"])
            event_log = Path(tmp) / ".cyber-agent-cli-sessions" / "test-session.events.jsonl"
            self.assertTrue(event_log.exists())
            self.assertIn("user_input_received", event_log.read_text(encoding="utf-8"))

    def test_stop_is_immediate_and_not_queued(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hub, runner, _ = _build_hub(Path(tmp))
            events = []
            hub.subscribe(events.append)

            hub.submit("/stop", source=HubTaskSource("feishu", "chat"))

            self.assertEqual(runner.run_inputs, [])
            self.assertEqual(runner.execution_controller.stop_reasons, ["feishu 请求停止当前任务"])
            self.assertIn("task_stop_requested", [event.type for event in events])

    def test_session_new_resets_runner_and_broadcasts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hub, runner, runtime_context = _build_hub(Path(tmp))
            events = []
            hub.subscribe(events.append)
            runner.history.append(AIMessage(content="old"))

            hub.submit("/new", source=HubTaskSource("cli", "cli"))

            self.assertEqual(runner.reset_count, 1)
            self.assertNotEqual(runtime_context["session_id"], "test-session")
            self.assertEqual(runtime_context["_recent_inputs"], [])
            self.assertIn("session_switched", [event.type for event in events])

    def test_session_use_restores_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            save_session_history(
                "stored-session",
                [HumanMessage(content="old question"), AIMessage(content="old answer")],
                mode=AgentMode.STANDARD.value,
                approval_policy=ApprovalPolicy.AUTO.value,
                recent_inputs=["old question"],
                base_dir=base_dir,
            )
            hub, runner, runtime_context = _build_hub(base_dir)

            hub.submit("/session use stored-session", source=HubTaskSource("cli", "cli"))

            self.assertEqual(runtime_context["session_id"], "stored-session")
            self.assertEqual(runtime_context["_recent_inputs"], ["old question"])
            self.assertIsNotNone(runner.restored_messages)
            self.assertEqual(len(runner.history), 2)

    def test_session_new_during_active_task_interrupts_and_drains_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = _BlockingRunner()
            hub, runtime_context = _build_hub_with_runner(Path(tmp), runner)
            events = []
            hub.subscribe(events.append)
            hub.start()

            hub.submit("long task", source=HubTaskSource("cli", "cli"))
            self.assertTrue(runner.started.wait(timeout=2.0))
            hub.submit("queued before switch", source=HubTaskSource("cli", "cli"))

            switch_thread = threading.Thread(
                target=lambda: hub.submit("/new", source=HubTaskSource("feishu", "chat")),
            )
            switch_thread.start()
            switch_thread.join(timeout=2.0)
            self.assertFalse(switch_thread.is_alive())
            hub.stop()

            self.assertEqual(runner.run_inputs, ["long task"])
            self.assertEqual(runner.reset_count, 1)
            self.assertNotEqual(runtime_context["session_id"], "test-session")
            event_types = [event.type for event in events]
            self.assertIn("task_interrupted", event_types)
            self.assertIn("task_queue_drained", event_types)
            drained = next(event for event in events if event.type == "task_queue_drained")
            self.assertEqual(drained.payload["count"], 1)

    def test_feishu_broadcast_chat_ids_merge_config_and_persisted_state(self) -> None:
        class Route:
            provider_options = {
                "hub_broadcast_chat_ids": "oc_config_a, oc_config_b",
            }

        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            _save_feishu_session_state(
                {
                    "version": 1,
                    "chats": {
                        "oc_config_a": {},
                        "oc_persisted": {},
                    },
                },
                base_dir,
            )

            chat_ids = _parse_feishu_broadcast_chat_ids(
                Route(),
                ["oc_option"],
                base_dir=base_dir,
            )

            self.assertEqual(
                chat_ids,
                ["oc_option", "oc_config_a", "oc_config_b", "oc_persisted"],
            )

    def test_multi_agent_pipeline_events_are_broadcast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = _FakeRunner()
            runtime_context: dict[str, object] = {
                "session_id": "test-session",
                "session_source_id": None,
                "approval_policy": ApprovalPolicy.AUTO,
                "_recent_inputs": [],
                "multi_agent_enabled": True,
            }

            def fake_multi_agent_turn(text, runner, context, event_handler=None):
                _ = context
                runner.history.append(HumanMessage(content=text))
                if event_handler is not None:
                    event_handler(
                        "pipeline.subtask_status",
                        {
                            "event": "subtask_status",
                            "detail": "执行测试子任务",
                            "metadata": {
                                "index": 0,
                                "role": "runner",
                                "agent_label": "runner Agent",
                                "status": "start",
                                "status_label": "开始",
                                "mode": "顺序",
                            },
                        },
                    )
                runner.history.append(AIMessage(content="pipeline reply"))

            hub = CyberAgentHub(
                runner=runner,
                runtime_context=runtime_context,
                approval_handler_factory=lambda context: None,
                detect_task_complexity=lambda text: True,
                run_multi_agent_turn=fake_multi_agent_turn,
                renderless_event_handler_factory=lambda runner, context, inner: inner,
                base_dir=Path(tmp),
            )
            events = []
            hub.subscribe(events.append)

            hub.start()
            hub.submit("complex task", source=HubTaskSource("cli", "cli"))

            self.assertTrue(hub.wait_until_idle(2.0))
            hub.stop()

            event_types = [event.type for event in events]
            self.assertIn("pipeline.subtask_status", event_types)
            finished = next(event for event in events if event.type == "task_finished")
            self.assertEqual(finished.payload["reply_text"], "pipeline reply")


if __name__ == "__main__":
    unittest.main()
