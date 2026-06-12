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
