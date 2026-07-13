"""并行子任务执行框架测试。

验证内容：
1. _build_subtask_prompt 按 role/desc/context/reasoning 正确组装 prompt
2. _create_subtask_runner 克隆的 runner 拥有独立 ExecutionController
3. _run_parallel_batch 正确并发执行多条子任务并按序收集结果
4. 管线循环中 parallel=true 的子任务被归组为并行批次
"""
from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from cyber_agent.agent.mode import AgentMode
from cyber_agent.agent.pipeline import (
    BASE_SUBTASK_TIMEOUT,
    CIRCUIT_BREAKER_CONSECUTIVE_FAILS,
    MAX_TIMEOUT_ESCALATIONS,
    TIMEOUT_ESCALATION_STEP,
    FourPillarPipeline,
)


class BuildSubtaskPromptTestCase(unittest.TestCase):
    """测试 _build_subtask_prompt 静态方法。"""

    def test_basic_prompt(self):
        """基础 prompt 包含角色标签和子任务描述。"""
        prompt = FourPillarPipeline._build_subtask_prompt(
            "执行者", "读取 /etc/hosts 文件",
        )
        self.assertIn("你是执行者", prompt)
        self.assertIn("读取 /etc/hosts 文件", prompt)
        self.assertIn("请直接调用工具完成", prompt)
        self.assertIn("任务边界", prompt)
        self.assertIn("不要为了寻找线索去读取 cyber-agent 本地源码", prompt)

    def test_with_context(self):
        """传入 ctx 时嵌入上下文段落。"""
        prompt = FourPillarPipeline._build_subtask_prompt(
            "阅读者", "分析日志", ctx="日志路径: /var/log/syslog",
        )
        self.assertIn("上下文: 日志路径: /var/log/syslog", prompt)

    def test_with_reasoning(self):
        """传入 reasoning 时嵌入整体背景，且截断到 300 字。"""
        long_reasoning = "背景 " * 200  # 远超 300 字
        prompt = FourPillarPipeline._build_subtask_prompt(
            "构建者", "部署服务", reasoning=long_reasoning,
        )
        self.assertIn("整体背景:", prompt)
        # 截断标记：reasoning[:300] 保证嵌入内容不大于 300
        reasoning_start = prompt.index("整体背景: ") + len("整体背景: ")
        reasoning_end = prompt.index("\n", reasoning_start)
        reasoning_in_prompt = prompt[reasoning_start:reasoning_end]
        self.assertLessEqual(len(reasoning_in_prompt), 300)

    def test_aggressive_prompt_discourages_unnecessary_user_confirmation(self):
        """激进执行下，子任务 prompt 明确要求直接执行而非路径选择。"""
        prompt = FourPillarPipeline._build_subtask_prompt(
            "执行者",
            "连接 VPN 并按 CHALLENGES_API.md 调用 API",
            aggressive=True,
        )

        self.assertIn("激进授权执行要求", prompt)
        self.assertIn("不要把已可由工具验证的事项改成询问用户", prompt)
        self.assertIn("不要输出“请选择 A/B/C 路径”后停止", prompt)
        self.assertIn("连接 VPN", prompt)

    def test_benchmark_profile_prompt_adds_score_first_constraints(self):
        """Benchmark aggressive 子任务 prompt 强调提交、止损和切题。"""
        prompt = FourPillarPipeline._build_subtask_prompt(
            "执行者",
            "解 xben-001-24 并提交 flag",
            benchmark_profile="aggressive",
        )

        self.assertIn("Benchmark aggressive 子任务要求", prompt)
        self.assertIn("发现疑似 flag 立即调用 submit", prompt)
        self.assertIn("close 当前题并切换下一题", prompt)

    def test_execution_summary_keeps_full_subtask_body(self):
        """四柱最终总结应保留完整子任务正文，不再只展示短摘要。"""
        long_body = "完整输出-" + ("细节" * 120)
        summary = FourPillarPipeline._build_execution_summary(
            [[f"[runner] 长任务\n{long_body}"]],
            iteration=1,
        )

        self.assertIn(long_body, summary)
        self.assertNotIn("完整输出-" + ("细节" * 50) + "…", summary)


class PipelineSessionPersistenceTestCase(unittest.TestCase):
    """测试四柱管线会把外壳对话写回主 runner history。"""

    def test_pipeline_run_appends_user_and_summary_messages_to_main_history(self):
        class FakeRunner:
            def __init__(self) -> None:
                self.mode = AgentMode.STANDARD
                self.history = [SystemMessage(content="system")]

            def get_turn_count(self) -> int:
                return sum(isinstance(message, HumanMessage) for message in self.history)

            def get_history_snapshot(self):
                return list(self.history)

        runner = FakeRunner()
        renderer = MagicMock()
        renderer.add_token_usage = MagicMock()
        pipeline = FourPillarPipeline(
            runner=runner,
            runtime_context={
                "session_id": "pipeline-session",
                "approval_policy": "prompt",
            },
            renderer=renderer,
        )

        def _fake_run_phases(_user_input: str, _auto_decision: bool) -> None:
            pipeline._final_summary = "完整四柱总结\n包含所有子任务结果"

        with (
            patch.object(pipeline, "_run_phases", side_effect=_fake_run_phases),
            patch.object(pipeline, "_save_trace"),
            patch("cyber_agent.session_store.append_session_event"),
            patch("cyber_agent.session_store.save_session_history"),
        ):
            pipeline.run("链接VPN并启动题目")

        self.assertIsInstance(runner.history[1], HumanMessage)
        self.assertEqual(runner.history[1].content, "链接VPN并启动题目")
        self.assertIsInstance(runner.history[2], AIMessage)
        self.assertIn("完整四柱总结", runner.history[2].content)


class SubtaskChecklistTestCase(unittest.TestCase):
    """测试子 Agent 任务清单展示。"""

    def setUp(self):
        self.runner = MagicMock()
        self.renderer = MagicMock()
        self.renderer.console.print = MagicMock()
        self.pipeline = FourPillarPipeline(
            runner=self.runner,
            runtime_context={
                "service_name": "deepseek",
                "model_name": "deepseek-chat",
                "api_key": "sk-test",
                "base_url": "http://test:8000/v1",
            },
            renderer=self.renderer,
        )

    def test_format_subtask_checklist_line(self):
        line = FourPillarPipeline._format_subtask_checklist_line(
            1,
            {
                "role": "reader",
                "task_description": "读取题目列表",
                "parallel": True,
            },
            selected=True,
        )

        self.assertIn("#02", line)
        self.assertIn("reader Agent", line)
        self.assertIn("并行", line)
        self.assertIn("待执行", line)
        self.assertIn("读取题目列表", line)

    def test_print_subtask_checklist_includes_all_tasks(self):
        subtasks = [
            {"role": "runner", "task_description": "VPN 预检"},
            {"role": "builder", "task_description": "构建队列", "parallel": True},
        ]

        self.pipeline._print_subtask_checklist(
            subtasks,
            [0],
            iteration=2,
        )

        printed = "\n".join(
            str(call.args[0])
            for call in self.renderer.console.print.call_args_list
            if call.args
        )
        self.assertIn("子 Agent 任务清单", printed)
        self.assertIn("第 2 轮", printed)
        self.assertIn("#01", printed)
        self.assertIn("runner Agent", printed)
        self.assertIn("待执行", printed)
        self.assertIn("#02", printed)
        self.assertIn("builder Agent", printed)
        self.assertIn("未选择", printed)


class PipelineIterationLimitTestCase(unittest.TestCase):
    """测试四柱执行循环的轮数配置。"""

    def test_default_iteration_limit_is_long_task_friendly(self):
        self.assertGreaterEqual(FourPillarPipeline._resolve_max_iterations(), 20)

    def test_iteration_limit_is_clamped_to_positive_range(self):
        from cyber_agent.config import settings

        original = settings.pipeline_max_iterations
        try:
            settings.pipeline_max_iterations = 0
            self.assertEqual(FourPillarPipeline._resolve_max_iterations(), 1)

            settings.pipeline_max_iterations = 101
            self.assertEqual(FourPillarPipeline._resolve_max_iterations(), 100)
        finally:
            settings.pipeline_max_iterations = original


class CreateSubtaskRunnerTestCase(unittest.TestCase):
    """测试 _create_subtask_runner 克隆逻辑。"""

    def setUp(self):
        self.mock_runner = MagicMock()
        self.mock_runner.tools = []
        self.mock_runner.mode = "standard"
        self.mock_runner.allowed_roots = []
        self.mock_runner.command_registry = {}
        self.mock_runner.extra_allowed_paths = []
        self.mock_runner.configured_registry = {}
        self.mock_runner.capability_registry = None
        self.mock_runner.file_skills = []
        self.mock_runner.system_prompt = "测试系统提示"
        self.mock_runner.max_context_chars = None
        self.mock_runner.max_context_tokens = None
        self.mock_runner.context_keep_recent_messages = None
        self.mock_runner.context_summary_max_chars = None

        self.runtime_context = {
            "service_name": "deepseek",
            "model_name": "deepseek-chat",
            "api_key": "sk-test",
            "base_url": "http://test:8000/v1",
        }
        self.pipeline = FourPillarPipeline(
            runner=self.mock_runner,
            runtime_context=self.runtime_context,
            renderer=MagicMock(),
        )

    @patch("cyber_agent.agent.pipeline.ExecutionController")
    def test_runner_has_independent_controller(self, mock_ec):
        """克隆的 runner 使用新的 ExecutionController，与原 runner 不同。"""
        sub = self.pipeline._create_subtask_runner()
        self.assertIsNotNone(sub)
        # ExecutionController 应被构造一次（克隆的 runner 使用新实例）
        mock_ec.assert_called_once()


class RunParallelBatchTestCase(unittest.TestCase):
    """测试 _run_parallel_batch 并发执行与结果收集。"""

    def setUp(self):
        self.mock_runner = MagicMock()
        self.mock_runner.tools = []
        self.mock_runner.mode = "standard"
        self.mock_runner.allowed_roots = []
        self.mock_runner.command_registry = {}
        self.mock_runner.extra_allowed_paths = []
        self.mock_runner.configured_registry = {}
        self.mock_runner.capability_registry = None
        self.mock_runner.file_skills = []
        self.mock_runner.system_prompt = "系统提示"
        self.mock_runner.execution_controller = MagicMock()
        self.mock_runner.max_context_chars = None
        self.mock_runner.max_context_tokens = None
        self.mock_runner.context_keep_recent_messages = None
        self.mock_runner.context_summary_max_chars = None

        self.renderer = MagicMock()
        self.renderer.console.print = MagicMock()

        self.pipeline = FourPillarPipeline(
            runner=self.mock_runner,
            runtime_context={
                "service_name": "deepseek",
                "model_name": "deepseek-chat",
                "api_key": "sk-test",
                "base_url": "http://test:8000/v1",
            },
            renderer=self.renderer,
        )

    def test_parallel_batch_returns_results_in_order(self):
        """_run_parallel_batch 按原始 seq 顺序返回结果。"""
        # 创建一个可验证的 _create_subtask_runner: 返回带 run 的 mock
        def _fake_create():
            r = MagicMock()
            r.run.return_value = f"结果来自 {id(r)}"
            return r

        with patch.object(self.pipeline, "_create_subtask_runner", _fake_create):
            batch = [
                {"role": "runner", "task_description": "任务A", "parallel": True},
                {"role": "reader", "task_description": "任务B", "parallel": True},
            ]
            results = self.pipeline._run_parallel_batch(
                batch,
                user_input="测试用户输入",
                reasoning="测试推理",
                additional_context="",
            )
            self.assertEqual(len(results), 2)
            # 按顺序：任务A 在前，任务B 在后
            self.assertIn("任务A", results[0])
            self.assertIn("任务B", results[1])

    def test_parallel_batch_single_task(self):
        """单任务并行批次也能正常工作。"""
        def _fake_create():
            r = MagicMock()
            r.run.return_value = "单任务结果"
            return r

        with patch.object(self.pipeline, "_create_subtask_runner", _fake_create):
            batch = [
                {"role": "builder", "task_description": "单任务", "parallel": True},
            ]
            results = self.pipeline._run_parallel_batch(
                batch,
                user_input="测试",
                reasoning="",
                additional_context="",
            )
            self.assertEqual(len(results), 1)
            self.assertIn("单任务", results[0])

    def test_parallel_batch_handles_failure_gracefully(self):
        """某条子任务失败时不影响其他子任务的结果收集。"""
        fail_flag = {"count": 0}

        def _fake_create():
            r = MagicMock()
            fail_flag["count"] += 1
            if fail_flag["count"] == 2:  # 第二条子任务模拟失败
                r.run.side_effect = RuntimeError("模拟失败")
            else:
                r.run.return_value = "成功结果"
            return r

        with patch.object(self.pipeline, "_create_subtask_runner", _fake_create):
            batch = [
                {"role": "runner", "task_description": "任务A", "parallel": True},
                {"role": "runner", "task_description": "任务B", "parallel": True},
            ]
            results = self.pipeline._run_parallel_batch(
                batch,
                user_input="测试",
                reasoning="",
                additional_context="",
            )
            self.assertEqual(len(results), 2)
            # 第一条成功，第二条失败标记
            self.assertIn("任务A", results[0])
            self.assertIn("失败", results[1])

    def test_parallel_batch_uses_independent_event_handlers(self):
        """每个并行子任务创建独立的事件处理器。"""
        created_handlers: list[object] = []

        original_make_handler = self.pipeline._make_subtask_event_handler

        def tracking_make_handler(renderer):
            handler = original_make_handler(renderer)
            created_handlers.append(handler)
            return handler

        self.pipeline._make_subtask_event_handler = tracking_make_handler

        def _fake_create():
            r = MagicMock()
            r.run.return_value = "handler test"
            return r

        with patch.object(self.pipeline, "_create_subtask_runner", _fake_create):
            batch = [
                {"role": "runner", "task_description": "T1", "parallel": True},
                {"role": "reader", "task_description": "T2", "parallel": True},
            ]
            self.pipeline._run_parallel_batch(
                batch,
                user_input="测试",
                reasoning="",
                additional_context="",
            )
            # 每个子任务应创建独立 handler
            self.assertEqual(len(created_handlers), 2)

    def test_parallel_batch_respects_max_subagents(self):
        """并行批次使用 runtime_context 中配置的最大子 Agent 数。"""
        self.pipeline._runtime_context["max_subagents"] = 2

        def _fake_create():
            r = MagicMock()
            r.run.return_value = "ok"
            return r

        with patch.object(self.pipeline, "_create_subtask_runner", _fake_create):
            batch = [
                {"role": "runner", "task_description": f"T{i}", "parallel": True}
                for i in range(5)
            ]
            self.pipeline._run_parallel_batch(
                batch,
                user_input="测试",
                reasoning="",
                additional_context="",
            )

        scheduled = [
            event for event in self.pipeline._trace
            if event["event"] == "parallel_batch_scheduled"
        ]
        self.assertEqual(scheduled[-1]["metadata"]["max_workers"], 2)


class SubtaskSchedulerTestCase(unittest.TestCase):
    """测试四柱子任务并发调度规则。"""

    def setUp(self):
        runner = MagicMock()
        runner.tools = []
        runner.mode = AgentMode.STANDARD
        runner.allowed_roots = []
        runner.command_registry = {}
        runner.extra_allowed_paths = []
        runner.configured_registry = {}
        runner.capability_registry = None
        runner.file_skills = []
        renderer = MagicMock()
        renderer.console.print = MagicMock()
        self.pipeline = FourPillarPipeline(
            runner=runner,
            runtime_context={
                "service_name": "deepseek",
                "model_name": "deepseek-chat",
                "api_key": "sk-test",
                "base_url": "http://test:8000/v1",
            },
            renderer=renderer,
        )

    def test_concurrency_off_rejects_parallel_tasks(self):
        self.pipeline._runtime_context["subtask_concurrency"] = "off"

        allowed, reason = self.pipeline._subtask_parallel_decision(
            {"task_description": "读取两个独立文件", "parallel": True}
        )

        self.assertFalse(allowed)
        self.assertEqual(reason, "concurrency_off")

    def test_auto_requires_llm_parallel_marker(self):
        self.pipeline._runtime_context["subtask_concurrency"] = "auto"

        allowed, reason = self.pipeline._subtask_parallel_decision(
            {"task_description": "读取独立文件"}
        )

        self.assertFalse(allowed)
        self.assertEqual(reason, "not_marked_parallel")

    def test_force_parallelizes_non_sensitive_tasks(self):
        self.pipeline._runtime_context["subtask_concurrency"] = "force"

        allowed, reason = self.pipeline._subtask_parallel_decision(
            {"task_description": "读取独立文件"}
        )

        self.assertTrue(allowed)
        self.assertEqual(reason, "force")

    def test_sensitive_benchmark_operations_stay_sequential(self):
        self.pipeline._runtime_context["subtask_concurrency"] = "force"

        allowed, reason = self.pipeline._subtask_parallel_decision(
            {
                "task_description": "POST submit xben-013-24 的 flag",
                "parallel": True,
            }
        )

        self.assertFalse(allowed)
        self.assertEqual(reason, "sensitive_operation")

    def test_resource_keys_extract_challenge_host_and_file(self):
        keys = self.pipeline._extract_subtask_resource_keys(
            {
                "task_description": (
                    "分析 xben-013-24，访问 10.0.166.193:80，"
                    "读取 /home/my/cyber/test/challenges.json"
                )
            }
        )

        self.assertIn("challenge:xben-013-24", keys)
        self.assertIn("host:10.0.166.193:80", keys)
        self.assertIn("file:/home/my/cyber/test/challenges.json", keys)

    def test_max_subagents_one_disables_concurrency(self):
        self.pipeline._runtime_context["subtask_concurrency"] = "force"
        self.pipeline._runtime_context["max_subagents"] = 1

        self.assertEqual(self.pipeline._resolve_subtask_concurrency(), "off")

    def test_benchmark_stale_detector_forces_close_and_switch(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True

        directive = ""
        for _ in range(3):
            directive = self.pipeline._update_benchmark_stale_state(
                ["## [runner] 探索 xben-001-24\n测试 SQLi/SSTI，未发现有效结果。"]
            )

        self.assertIn("close?unique_code=xben-001-24", directive)
        self.assertIn("选择下一道未完成", directive)
        events = [item["event"] for item in self.pipeline._trace]
        self.assertIn("benchmark_stale_detected", events)

    def test_benchmark_stale_detector_resets_on_submit_signal(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True

        self.pipeline._update_benchmark_stale_state(
            ["## [runner] 探索 xben-001-24\n未发现有效结果。"]
        )
        directive = self.pipeline._update_benchmark_stale_state(
            [
                "POST /openapi/v1/challenges/submit "
                '{"correct":true,"correct_flag_count":1}'
            ]
        )

        self.assertEqual(directive, "")
        self.assertEqual(self.pipeline._benchmark_stale_rounds, 0)

    def test_benchmark_stale_detector_ignores_api_docs_field_names(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True

        directive = ""
        for _ in range(3):
            directive = self.pipeline._update_benchmark_stale_state(
                [
                    "继续探索 xben-001-24。API 文档说明字段 "
                    "`correct_flag_count`、`cumulative_score`，但没有提交结果。"
                ]
            )

        self.assertIn("close?unique_code=xben-001-24", directive)


class SequentialSubtaskIsolationTestCase(unittest.TestCase):
    """测试顺序子任务不会污染主 runner 的对话历史。"""

    def setUp(self):
        self.main_runner = MagicMock()
        self.main_runner.run.side_effect = AssertionError(
            "sequential subtasks must not use the main runner"
        )
        self.main_runner.history = [
            HumanMessage(content="当前文件夹有什么"),
            HumanMessage(content="请你链接VPN"),
        ]
        self.main_runner.tools = []
        self.main_runner.mode = "standard"
        self.main_runner.allowed_roots = []
        self.main_runner.command_registry = {}
        self.main_runner.extra_allowed_paths = []
        self.main_runner.configured_registry = {}
        self.main_runner.capability_registry = None
        self.main_runner.file_skills = []
        self.main_runner.system_prompt = "系统提示"
        self.main_runner.execution_controller = MagicMock()
        self.main_runner.max_context_chars = None
        self.main_runner.max_context_tokens = None
        self.main_runner.context_keep_recent_messages = None
        self.main_runner.context_summary_max_chars = None

        self.renderer = MagicMock()
        self.renderer.console.print = MagicMock()
        self.pipeline = FourPillarPipeline(
            runner=self.main_runner,
            runtime_context={
                "service_name": "deepseek",
                "model_name": "deepseek-chat",
                "api_key": "sk-test",
                "base_url": "http://test:8000/v1",
            },
            renderer=self.renderer,
        )

    def test_sequential_subtask_uses_isolated_runner_history(self):
        sub_runner = MagicMock()
        sub_runner.execution_controller = None
        sub_runner.history = []

        def _sub_run(user_input, **_kwargs):
            sub_runner.history.append(HumanMessage(content=user_input))
            return "子任务结果"

        sub_runner.run.side_effect = _sub_run
        prompt = FourPillarPipeline._build_subtask_prompt(
            "runner", "执行VPN联通预检",
        )

        with patch.object(
            self.pipeline, "_create_subtask_runner", return_value=sub_runner,
        ):
            result = self.pipeline._run_subtask_with_escalating_timeout(
                prompt,
                "执行者",
                "执行VPN联通预检",
            )

        self.assertEqual(result, "子任务结果")
        self.main_runner.run.assert_not_called()
        self.assertEqual(len(sub_runner.history), 1)
        self.assertIn("请完成以下子任务", sub_runner.history[0].content)
        self.assertFalse(
            any("请完成以下子任务" in msg.content for msg in self.main_runner.history)
        )


class SubtaskBoundaryApprovalTestCase(unittest.TestCase):
    """测试内部子任务的任务边界审批。"""

    def setUp(self):
        self.runner = MagicMock()
        self.runner.tools = []
        self.runner.mode = "standard"
        self.runner.allowed_roots = []
        self.runner.command_registry = {}
        self.runner.extra_allowed_paths = []
        self.runner.configured_registry = {}
        self.runner.capability_registry = None
        self.runner.file_skills = []
        self.runner.system_prompt = "系统提示"
        self.runner.max_context_chars = None
        self.runner.max_context_tokens = None
        self.runner.context_keep_recent_messages = None
        self.runner.context_summary_max_chars = None
        self.pipeline = FourPillarPipeline(
            runner=self.runner,
            runtime_context={
                "service_name": "deepseek",
                "model_name": "deepseek-chat",
                "api_key": "sk-test",
                "base_url": "http://test:8000/v1",
            },
            renderer=MagicMock(),
        )

    def test_boundary_approval_blocks_unrelated_project_probe(self):
        handler = self.pipeline._make_subtask_approval_handler(
            "TSec Benchmark 启动容器",
        )
        decision = handler(
            MagicMock(),
            {
                "name": "run_shell_command",
                "args": {
                    "command": (
                        "grep -rn 'api' "
                        "/home/my/cyber/pentest/cyber-agent-cli/src/cyber_agent"
                    ),
                },
            },
        )

        self.assertFalse(decision.approved)
        self.assertIn("禁止", decision.reason)

    def test_boundary_approval_blocks_secret_probe(self):
        handler = self.pipeline._make_subtask_approval_handler(
            "TSec Benchmark 获取题目列表",
        )
        decision = handler(
            MagicMock(),
            {
                "name": "run_shell_command",
                "args": {
                    "command": "cat /home/my/cyber/pentest/cyber-agent-cli/.env",
                },
            },
        )

        self.assertFalse(decision.approved)
        self.assertIn(".env", decision.reason)

    def test_boundary_approval_blocks_reading_root_cyber_claude_file(self):
        handler = self.pipeline._make_subtask_approval_handler(
            "TSec Benchmark 获取题目列表",
        )
        decision = handler(
            MagicMock(),
            {
                "name": "read_text_file",
                "args": {"path": "/home/my/cyber/CLAUDE.md"},
            },
        )

        self.assertFalse(decision.approved)
        self.assertIn("cyber-agent 本地源码", decision.reason)

    def test_boundary_approval_blocks_environment_dump(self):
        handler = self.pipeline._make_subtask_approval_handler(
            "TSec Benchmark 获取题目列表",
        )
        decision = handler(
            MagicMock(),
            {
                "name": "run_shell_command",
                "args": {"command": "env | sort"},
            },
        )

        self.assertFalse(decision.approved)
        self.assertIn("环境变量", decision.reason)

    def test_boundary_approval_does_not_treat_session_as_debug_task(self):
        handler = self.pipeline._make_subtask_approval_handler(
            "获取 session 题目列表并读取 challenge history",
        )
        decision = handler(
            MagicMock(),
            {
                "name": "run_shell_command",
                "args": {
                    "command": (
                        "grep -rn 'BENCHMARK_TOKEN' "
                        "/home/my/cyber/pentest/cyber-agent-cli/.cyber/sessions"
                    ),
                },
            },
        )

        self.assertFalse(decision.approved)
        self.assertIn("禁止", decision.reason)

    def test_boundary_prompt_text_does_not_enable_project_probe(self):
        prompt = FourPillarPipeline._build_subtask_prompt(
            "runner",
            "启动 xben-013-24 靶场容器",
        )
        handler = self.pipeline._make_subtask_approval_handler(prompt)
        decision = handler(
            MagicMock(),
            {
                "name": "read_text_file",
                "args": {
                    "path": "/home/my/cyber/pentest/cyber-agent-cli/.env",
                },
            },
        )

        self.assertFalse(decision.approved)
        self.assertIn(".env", decision.reason)

    def test_cyber_agent_mention_without_debug_intent_is_not_debug_task(self):
        self.assertFalse(
            FourPillarPipeline._is_cyber_agent_debug_task(
                "不要为了寻找线索去读取 cyber-agent 本地源码"
            )
        )
        self.assertTrue(
            FourPillarPipeline._is_cyber_agent_debug_task(
                "调试 cyber-agent 的 FourPillarPipeline history 污染问题"
            )
        )

    def test_boundary_approval_allows_cyber_agent_debug_task(self):
        handler = self.pipeline._make_subtask_approval_handler(
            "调试 cyber-agent 的 FourPillarPipeline history 污染问题",
        )
        decision = handler(
            MagicMock(),
            {
                "name": "run_shell_command",
                "args": {
                    "command": (
                        "grep -rn 'run(' "
                        "/home/my/cyber/pentest/cyber-agent-cli/src/cyber_agent"
                    ),
                },
            },
        )

        self.assertTrue(decision.approved)


class SubtaskGroupingTestCase(unittest.TestCase):
    """验证管线循环正确归组 parallel 子任务。

    通过仿造 _run_phases 中的归组逻辑来验证。
    """

    def test_consecutive_parallel_tasks_grouped(self):
        """连续 parallel=true 的子任务应被归为同一批次。

        模拟 selected_indices 和 subtasks，验证归组逻辑：
        [seq0(parallel), seq1(parallel), seq2(not), seq3(parallel)] →
        batch1=[seq0, seq1], single=[seq2], batch3=[seq3]
        """
        subtasks = [
            {"role": "runner", "task_description": "P1", "parallel": True},
            {"role": "reader", "task_description": "P2", "parallel": True},
            {"role": "builder", "task_description": "S1", "parallel": False},
            {"role": "runner", "task_description": "P3", "parallel": True},
        ]
        selected_indices = [0, 1, 2, 3]

        # 模拟归组逻辑
        groups: list[list[int]] = []
        i = 0
        while i < len(selected_indices):
            idx = selected_indices[i]
            if idx >= len(subtasks):
                i += 1
                continue
            task = subtasks[idx]
            if task.get("parallel", False):
                batch: list[int] = []
                while i < len(selected_indices):
                    pidx = selected_indices[i]
                    if pidx >= len(subtasks):
                        i += 1
                        continue
                    ptask = subtasks[pidx]
                    if not ptask.get("parallel", False):
                        break
                    batch.append(pidx)
                    i += 1
                groups.append(batch)
            else:
                groups.append([idx])
                i += 1

        self.assertEqual(len(groups), 3)
        self.assertEqual(groups[0], [0, 1])   # 并行批次
        self.assertEqual(groups[1], [2])      # 单条顺序
        self.assertEqual(groups[2], [3])      # 单条顺序（只有一条 parallel 单独也是一批）

    def test_all_sequential_no_grouping(self):
        """全部未标记 parallel 时保持原状——每条都是独立批次。"""
        subtasks = [
            {"role": "runner", "task_description": "S1"},
            {"role": "reader", "task_description": "S2"},
            {"role": "builder", "task_description": "S3"},
        ]
        selected_indices = [0, 1, 2]

        groups: list[list[int]] = []
        i = 0
        while i < len(selected_indices):
            idx = selected_indices[i]
            task = subtasks[idx]
            if task.get("parallel", False):
                batch: list[int] = []
                while i < len(selected_indices):
                    pidx = selected_indices[i]
                    ptask = subtasks[pidx]
                    if not ptask.get("parallel", False):
                        break
                    batch.append(pidx)
                    i += 1
                groups.append(batch)
            else:
                groups.append([idx])
                i += 1

        self.assertEqual(len(groups), 3)
        for g in groups:
            self.assertEqual(len(g), 1)


if __name__ == "__main__":
    unittest.main()
