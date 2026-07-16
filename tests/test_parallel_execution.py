"""并行子任务执行框架测试。

验证内容：
1. _build_subtask_prompt 按 role/desc/context/reasoning 正确组装 prompt
2. _create_subtask_runner 克隆的 runner 拥有独立 ExecutionController
3. _run_parallel_batch 正确并发执行多条子任务并按序收集结果
4. 管线循环中 parallel=true 的子任务被归组为并行批次
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import ANY, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from cyber_agent.agent.approval import ApprovalPolicy
from cyber_agent.agent.mode import AgentMode
from cyber_agent.agent.pipeline import (
    BASE_SUBTASK_TIMEOUT,
    BENCHMARK_LOW_VALUE_SIGNAL_LIMIT,
    BENCHMARK_SUBTASK_TIMEOUT,
    CIRCUIT_BREAKER_CONSECUTIVE_FAILS,
    MAX_TIMEOUT_ESCALATIONS,
    TIMEOUT_ESCALATION_STEP,
    FourPillarPipeline,
)
from cyber_agent.cli.agent_executor import create_approval_handler


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
        self.assertIn("硬预算约 90 秒", prompt)
        self.assertIn("close 当前题并切换下一题", prompt)
        self.assertIn("默认禁止调用 hint API", prompt)
        self.assertIn("立即结束当前子任务", prompt)
        self.assertIn("下一次工具调用必须是 submit", prompt)

    def test_benchmark_profile_prompt_includes_runtime_state(self):
        """Benchmark 子任务 prompt 应携带已确认运行态，减少重复前置步骤。"""
        prompt = FourPillarPipeline._build_subtask_prompt(
            "执行者",
            "继续解当前题",
            benchmark_profile="aggressive",
            benchmark_state_context=(
                "## Benchmark 已确认运行态（必须信任并复用）\n"
                "VPN：已连接 tun0 10.254.0.10，"
                "不要重复启动 OpenVPN。"
            ),
        )

        self.assertIn("Benchmark 已确认运行态", prompt)
        self.assertIn("不要重复启动 OpenVPN", prompt)

    def test_benchmark_task_detection_avoids_validation_code_prefix(self):
        self.assertFalse(FourPillarPipeline._looks_like_benchmark_task("解 xben-001-24"))
        self.assertTrue(
            FourPillarPipeline._looks_like_benchmark_task(
                "读取 CHALLENGES_API.md 后用 unique_code start"
            )
        )

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


class BenchmarkFastPathTestCase(unittest.TestCase):
    """测试 Benchmark aggressive 的固定刷题快路径。"""

    def setUp(self):
        self.runner = MagicMock()
        self.runner.history = []
        self.renderer = MagicMock()
        self.renderer.console.print = MagicMock()
        self.pipeline = FourPillarPipeline(
            runner=self.runner,
            runtime_context={
                "benchmark_profile": "aggressive",
                "benchmark_target_score": 4000,
            },
            renderer=self.renderer,
        )
        self.pipeline._benchmark_profile_active = True

    def test_benchmark_aggressive_run_phases_uses_fast_path(self):
        with (
            patch.object(
                self.pipeline,
                "_run_benchmark_fast_phases",
                return_value=False,
            ) as fast,
            patch.object(self.pipeline, "_call_role_with_timeout") as role_call,
        ):
            self.pipeline._run_phases("TSec Benchmark 跑分", auto_decision=True)

        fast.assert_called_once_with("TSec Benchmark 跑分")
        role_call.assert_not_called()

    def test_benchmark_fast_path_handoff_runs_standard_roles(self):
        self.pipeline._runtime_context["benchmark_target_score"] = 0
        with (
            patch.object(
                self.pipeline,
                "_run_benchmark_fast_phases",
                return_value=True,
            ) as fast,
            patch.object(self.pipeline, "_auto_select", return_value=([], "")),
            patch.object(self.pipeline, "_call_role_with_timeout") as role_call,
        ):
            role_call.side_effect = [
                "analysis",
                "diffusion",
                "jump",
                "reflection",
                '{"reasoning":"done","subtasks":[]}',
            ]
            self.pipeline._run_phases("TSec Benchmark 跑分", auto_decision=True)

        fast.assert_called_once_with("TSec Benchmark 跑分")
        self.assertGreaterEqual(role_call.call_count, 5)

    def test_benchmark_fast_cycle_uses_two_score_first_runner_tasks(self):
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["vpn_connected"] = True
            self.pipeline._benchmark_state["api_interface"] = "enp0s20f0u3u4"

        subtasks = self.pipeline._benchmark_fast_cycle_subtasks()

        self.assertEqual(len(subtasks), 2)
        self.assertTrue(all(task["role"] == "runner" for task in subtasks))
        self.assertEqual(
            [task["benchmark_action"] for task in subtasks],
            ["schedule", "probe"],
        )
        joined = "\n".join(task["task_description"] for task in subtasks)
        self.assertIn("POST start", joined)
        self.assertIn("45 秒快速指纹", joined)
        self.assertIn("立即 submit", joined)
        self.assertIn("保留 active", joined)
        self.assertIn("连续不可达达到 2 次才 close", joined)
        self.assertIn("不 hint", joined)

    def test_benchmark_fast_cycle_inserts_setup_when_state_unknown(self):
        subtasks = self.pipeline._benchmark_fast_cycle_subtasks()

        self.assertEqual(len(subtasks), 3)
        self.assertEqual(
            [task["benchmark_action"] for task in subtasks],
            ["setup", "schedule", "probe"],
        )
        self.assertIn("Benchmark fast setup", subtasks[0]["task_description"])
        self.assertIn("CHALLENGES_API.md", subtasks[0]["task_description"])
        self.assertIn("不要 start", subtasks[0]["task_description"])

    def test_benchmark_execution_control_policy_adjusts_fast_loop_breadth(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "benchmark-profiles.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "execution_control_policy": {
                            "max_probe_paths": 2,
                            "max_probe_urls": 12,
                            "max_authenticated_urls": 8,
                            "max_payloads_per_param": 1,
                            "max_flag_paths": 3,
                            "fast_probe_seconds": 30,
                            "max_subagents": 5,
                            "subtask_concurrency": "force",
                        },
                        "probe_paths": ["healthz", "metrics", "debug"],
                    }
                ),
                encoding="utf-8",
            )
            self.pipeline._runtime_context["benchmark_profiles_path"] = str(profile_path)
            self.pipeline._runtime_context.pop("max_subagents", None)
            self.pipeline._runtime_context.pop("subtask_concurrency", None)
            with self.pipeline._benchmark_state_lock:
                self.pipeline._benchmark_state["vpn_connected"] = True
                self.pipeline._benchmark_state["api_interface"] = "enp0s20f0u3u4"

            subtasks = self.pipeline._benchmark_fast_cycle_subtasks()
            joined = "\n".join(task["task_description"] for task in subtasks)

            self.assertEqual(self.pipeline._benchmark_probe_paths(), ["", "robots.txt"])
            self.assertEqual(len(self.pipeline._benchmark_flag_paths()), 3)
            self.assertEqual(
                self.pipeline._benchmark_payloads_for_param("file"),
                ["../flag"],
            )
            self.assertEqual(self.pipeline._resolve_max_subagents(), 5)
            self.assertEqual(self.pipeline._resolve_subtask_concurrency(), "force")
            self.assertIn("30 秒快速指纹", joined)
            self.assertIn("最多 12 个容器 URL", joined)
            self.assertIn("认证后最多 8 个 URL", joined)

    def test_benchmark_runtime_execution_control_overrides_profile_policy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "benchmark-profiles.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "execution_control_policy": {
                            "max_probe_urls": 12,
                            "fast_probe_seconds": 30,
                        }
                    }
                ),
                encoding="utf-8",
            )
            self.pipeline._runtime_context["benchmark_profiles_path"] = str(profile_path)
            self.pipeline._runtime_context["execution_control_policy"] = {
                "max_probe_urls": 25,
                "fast_probe_seconds": 60,
            }
            with self.pipeline._benchmark_state_lock:
                self.pipeline._benchmark_state["vpn_connected"] = True
                self.pipeline._benchmark_state["api_interface"] = "enp0s20f0u3u4"

            policy = self.pipeline._benchmark_execution_control_policy()
            subtasks = self.pipeline._benchmark_fast_cycle_subtasks()
            joined = "\n".join(task["task_description"] for task in subtasks)

            self.assertEqual(policy["max_probe_urls"], 25)
            self.assertEqual(policy["fast_probe_seconds"], 60)
            self.assertIn("60 秒快速指纹", joined)
            self.assertIn("最多 25 个容器 URL", joined)

    def test_benchmark_timeout_and_low_value_thresholds_are_fast(self):
        self.assertEqual(BENCHMARK_SUBTASK_TIMEOUT, 90)
        self.assertEqual(BENCHMARK_LOW_VALUE_SIGNAL_LIMIT, 4)
        self.assertEqual(self.pipeline._subtask_timeout_config()[0], 90)

    def test_benchmark_fast_path_only_when_easy_candidates_exist(self):
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["last_challenges_snapshot"] = [
                {
                    "unique_code": "xben-001-24",
                    "difficulty": "medium",
                    "is_completed": False,
                    "container_status": "stopped",
                },
                {
                    "unique_code": "xben-002-24",
                    "difficulty": "hard",
                    "is_completed": False,
                    "container_status": "stopped",
                },
            ]

        should_fast, reason = self.pipeline._benchmark_should_use_fast_path()

        self.assertFalse(should_fast)
        self.assertIn("切回四柱", reason)

    def test_benchmark_fast_path_handles_current_easy(self):
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["current_challenge"] = "xben-038-24"
            self.pipeline._benchmark_state["last_challenges_snapshot"] = [
                {
                    "unique_code": "xben-038-24",
                    "difficulty": "easy",
                    "is_completed": False,
                    "container_status": "available",
                }
            ]

        should_fast, reason = self.pipeline._benchmark_should_use_fast_path()

        self.assertTrue(should_fast)
        self.assertIn("easy", reason)

    def test_benchmark_fast_path_does_not_recover_abandoned_easy(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["last_challenges_snapshot"] = [
                {
                    "unique_code": "xben-020-24",
                    "difficulty": "easy",
                    "level": 1,
                    "total_score": 200,
                    "is_completed": False,
                    "container_status": "stopped",
                }
            ]
            self.pipeline._benchmark_state["abandoned_challenges"] = {"xben-020-24"}

        should_fast, reason = self.pipeline._benchmark_should_use_fast_path()
        selected = self.pipeline._benchmark_select_next_easy(
            self.pipeline._benchmark_state["last_challenges_snapshot"]
        )

        self.assertFalse(should_fast)
        self.assertIn("切回四柱", reason)
        self.assertIsNone(selected)
        state = self.pipeline._benchmark_state_snapshot()
        self.assertIn("xben-020-24", state["abandoned_challenges"])
        self.assertNotIn("xben-020-24", state["recovery_attempted_challenges"])

    def test_benchmark_fast_path_recovers_closed_easy_once_when_platform_stopped(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["last_challenges_snapshot"] = [
                {
                    "unique_code": "xben-006-24",
                    "difficulty": "easy",
                    "level": 1,
                    "total_score": 200,
                    "is_completed": False,
                    "container_status": "stopped",
                }
            ]
            self.pipeline._benchmark_state["closed_challenges"] = {"xben-006-24"}

        should_fast, reason = self.pipeline._benchmark_should_use_fast_path()
        selected = self.pipeline._benchmark_select_next_easy(
            self.pipeline._benchmark_state["last_challenges_snapshot"]
        )

        self.assertTrue(should_fast)
        self.assertIn("恢复 fast path", reason)
        self.assertEqual(selected["unique_code"], "xben-006-24")
        state = self.pipeline._benchmark_state_snapshot()
        self.assertNotIn("xben-006-24", state["closed_challenges"])
        self.assertIn("xben-006-24", state["recovery_attempted_challenges"])

    def test_benchmark_fast_path_recovers_abandoned_easy_only_once(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["last_challenges_snapshot"] = [
                {
                    "unique_code": "xben-071-24",
                    "difficulty": "easy",
                    "level": 1,
                    "total_score": 200,
                    "is_completed": False,
                    "container_status": "stopped",
                }
            ]
            self.pipeline._benchmark_state["abandoned_challenges"] = {"xben-071-24"}
            self.pipeline._benchmark_state["recovery_attempted_challenges"] = {"xben-071-24"}

        should_fast, reason = self.pipeline._benchmark_should_use_fast_path()
        selected = self.pipeline._benchmark_select_next_easy(
            self.pipeline._benchmark_state["last_challenges_snapshot"]
        )

        self.assertFalse(should_fast)
        self.assertIn("切回四柱", reason)
        self.assertIsNone(selected)

    def test_benchmark_fast_path_rejects_current_medium(self):
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["current_challenge"] = "xben-001-24"
            self.pipeline._benchmark_state["last_challenges_snapshot"] = [
                {
                    "unique_code": "xben-001-24",
                    "difficulty": "medium",
                    "is_completed": False,
                    "container_status": "available",
                },
                {
                    "unique_code": "xben-038-24",
                    "difficulty": "easy",
                    "is_completed": False,
                    "container_status": "stopped",
                },
            ]

        should_fast, reason = self.pipeline._benchmark_should_use_fast_path()

        self.assertFalse(should_fast)
        self.assertIn("medium", reason)

    def test_benchmark_select_next_candidate_falls_through_to_medium(self):
        snapshot = [
            {
                "unique_code": "xben-006-24",
                "difficulty": "easy",
                "level": 1,
                "total_score": 100,
                "is_completed": True,
                "container_status": "stopped",
            },
            {
                "unique_code": "e1-01",
                "difficulty": "medium",
                "level": 1,
                "total_score": 250,
                "is_completed": False,
                "container_status": "stopped",
            },
            {
                "unique_code": "e1-03",
                "difficulty": "hard",
                "level": 1,
                "total_score": 250,
                "is_completed": False,
                "container_status": "stopped",
            },
        ]

        selected = self.pipeline._benchmark_select_next_candidate(snapshot)

        self.assertEqual(selected["unique_code"], "e1-01")

    def test_benchmark_selection_policy_changes_difficulty_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "benchmark-profiles.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "selection_policy": {
                            "difficulty_order": ["medium", "easy", "hard"],
                            "fast_path_difficulties": ["easy"],
                        }
                    }
                ),
                encoding="utf-8",
            )
            self.pipeline._runtime_context["benchmark_profiles_path"] = str(profile_path)

            snapshot = [
                {
                    "unique_code": "easy-1",
                    "difficulty": "easy",
                    "level": 1,
                    "total_score": 100,
                    "is_completed": False,
                    "container_status": "stopped",
                },
                {
                    "unique_code": "medium-1",
                    "difficulty": "medium",
                    "level": 1,
                    "total_score": 200,
                    "is_completed": False,
                    "container_status": "stopped",
                },
            ]

            selected = self.pipeline._benchmark_select_next_candidate(snapshot)

        self.assertEqual(selected["unique_code"], "medium-1")

    def test_benchmark_fast_path_difficulties_are_profile_driven(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "benchmark-profiles.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "selection_policy": {
                            "fast_path_difficulties": ["easy", "medium"],
                            "handoff_difficulties": ["hard"],
                        }
                    }
                ),
                encoding="utf-8",
            )
            self.pipeline._runtime_context["benchmark_profiles_path"] = str(profile_path)
            with self.pipeline._benchmark_state_lock:
                self.pipeline._benchmark_state["current_challenge"] = "medium-1"
                self.pipeline._benchmark_state["last_challenges_snapshot"] = [
                    {
                        "unique_code": "medium-1",
                        "difficulty": "medium",
                        "is_completed": False,
                        "container_status": "available",
                    }
                ]

            should_fast, reason = self.pipeline._benchmark_should_use_fast_path()

        self.assertTrue(should_fast)
        self.assertIn("medium", reason)

    def test_benchmark_invalid_selection_policy_keeps_default_fast_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "benchmark-profiles.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "selection_policy": {
                            "difficulty_order": ["invalid"],
                            "fast_path_difficulties": ["invalid"],
                        }
                    }
                ),
                encoding="utf-8",
            )
            self.pipeline._runtime_context["benchmark_profiles_path"] = str(profile_path)
            with self.pipeline._benchmark_state_lock:
                self.pipeline._benchmark_state["current_challenge"] = "medium-1"
                self.pipeline._benchmark_state["last_challenges_snapshot"] = [
                    {
                        "unique_code": "medium-1",
                        "difficulty": "medium",
                        "is_completed": False,
                        "container_status": "available",
                    }
                ]

            should_fast, reason = self.pipeline._benchmark_should_use_fast_path()

        self.assertFalse(should_fast)
        self.assertIn("medium", reason)

    def test_benchmark_planning_instruction_uses_policy_estimated_score(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "benchmark-profiles.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "selection_policy": {
                            "fast_path_difficulties": ["medium"],
                            "handoff_difficulties": ["hard"],
                            "estimated_fast_score": 500,
                        }
                    }
                ),
                encoding="utf-8",
            )
            self.pipeline._runtime_context["benchmark_profiles_path"] = str(profile_path)
            self.pipeline._runtime_context["benchmark_target_score"] = 2000

            instruction = self.pipeline._benchmark_planning_instruction()
            max_iterations = self.pipeline._resolve_effective_max_iterations()

        self.assertIn("约 500 分", instruction)
        self.assertIn("约 4 道 medium/低 level", instruction)
        self.assertIn("medium 题优先 deterministic/adaptive fast path", instruction)
        self.assertGreaterEqual(max_iterations, 9)

    def test_benchmark_select_next_candidate_recovers_closed_easy_before_medium(self):
        snapshot = [
            {
                "unique_code": "c-06",
                "difficulty": "easy",
                "level": 1,
                "total_score": 100,
                "is_completed": False,
                "container_status": "stopped",
            },
            {
                "unique_code": "e1-01",
                "difficulty": "medium",
                "level": 1,
                "total_score": 250,
                "is_completed": False,
                "container_status": "stopped",
            },
        ]
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["closed_challenges"] = {"c-06"}

        selected = self.pipeline._benchmark_select_next_candidate(snapshot)

        self.assertEqual(selected["unique_code"], "c-06")
        state = self.pipeline._benchmark_state_snapshot()
        self.assertNotIn("c-06", state["closed_challenges"])
        self.assertIn("c-06", state["recovery_attempted_challenges"])

    def test_benchmark_select_next_candidate_skips_abandoned_easy_before_medium(self):
        snapshot = [
            {
                "unique_code": "d-01",
                "difficulty": "easy",
                "level": 1,
                "total_score": 100,
                "is_completed": False,
                "container_status": "stopped",
            },
            {
                "unique_code": "e1-01",
                "difficulty": "medium",
                "level": 1,
                "total_score": 250,
                "is_completed": False,
                "container_status": "stopped",
            },
        ]
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["abandoned_challenges"] = {"d-01"}

        selected = self.pipeline._benchmark_select_next_candidate(snapshot)

        self.assertEqual(selected["unique_code"], "e1-01")
        state = self.pipeline._benchmark_state_snapshot()
        self.assertIn("d-01", state["abandoned_challenges"])
        self.assertNotIn("d-01", state["recovery_attempted_challenges"])

    def test_benchmark_fast_path_hands_off_reasoning_easy(self):
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["current_challenge"] = "xben-038-24"
            self.pipeline._benchmark_state["active_containers"] = {
                "xben-038-24": ["10.0.1.2:3000"],
            }
            self.pipeline._benchmark_state["reasoning_challenges"] = {"xben-038-24"}
            self.pipeline._benchmark_state["last_challenges_snapshot"] = [
                {
                    "unique_code": "xben-038-24",
                    "difficulty": "easy",
                    "is_completed": False,
                    "container_status": "available",
                }
            ]

        should_fast, reason = self.pipeline._benchmark_should_use_fast_path()

        self.assertFalse(should_fast)
        self.assertIn("深挖", reason)

    def test_benchmark_pauses_same_round_generic_plan_after_reasoning_mark(self):
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["current_challenge"] = "xben-038-24"
            self.pipeline._benchmark_state["active_containers"] = {
                "xben-038-24": ["10.0.1.2:3000"],
            }
            self.pipeline._benchmark_state["reasoning_challenges"] = {"xben-038-24"}

        pause, reason = (
            self.pipeline._benchmark_should_pause_generic_plan_after_deterministic(
                "快速指纹当前容器并寻找明显 flag"
            )
        )

        self.assertTrue(pause)
        self.assertIn("xben-038-24", reason)
        self.assertIn("reasoning_handoff", reason)

    def test_benchmark_pauses_same_round_generic_plan_after_active_schedule(self):
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["current_challenge"] = "c-06"
            self.pipeline._benchmark_state["active_containers"] = {
                "c-06": ["10.0.1.2:8080"],
            }

        pause, reason = (
            self.pipeline._benchmark_should_pause_generic_plan_after_deterministic(
                "如果上一题不可做，则 start 下一道未完成题"
            )
        )

        self.assertTrue(pause)
        self.assertIn("active", reason)
        self.assertIn("c-06", reason)

    def test_benchmark_does_not_pause_same_round_handoff_plan(self):
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["current_challenge"] = "xben-038-24"
            self.pipeline._benchmark_state["active_containers"] = {
                "xben-038-24": ["10.0.1.2:3000"],
            }
            self.pipeline._benchmark_state["reasoning_challenges"] = {"xben-038-24"}

        pause, reason = (
            self.pipeline._benchmark_should_pause_generic_plan_after_deterministic(
                "Benchmark handoff step 1：只深挖当前 active 题"
            )
        )

        self.assertFalse(pause)
        self.assertEqual(reason, "")

    def test_benchmark_handoff_selection_restores_close_step(self):
        subtasks = [
            {"task_description": "Benchmark handoff step 1：只深挖当前 active"},
            {"task_description": "Benchmark handoff step 2：尝试最高置信利用"},
            {"task_description": "Benchmark handoff step 3：无 flag 则 close"},
        ]

        selected, note = self.pipeline._benchmark_normalize_selected_indices(
            subtasks,
            [0, 1],
        )

        self.assertEqual(selected, [0, 1, 2])
        self.assertIn("完整执行", note)

    def test_benchmark_non_handoff_selection_is_not_forced(self):
        subtasks = [
            {"task_description": "读取题目列表"},
            {"task_description": "选择下一题"},
            {"task_description": "探测容器"},
        ]

        selected, note = self.pipeline._benchmark_normalize_selected_indices(
            subtasks,
            [0, 2],
        )

        self.assertEqual(selected, [0, 2])
        self.assertEqual(note, "")

    def test_benchmark_recovers_closed_easy_before_medium_is_untried(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        snapshot = [
            {
                "unique_code": "c-03",
                "difficulty": "easy",
                "level": 1,
                "is_completed": False,
                "container_status": "stopped",
            },
            {
                "unique_code": "m-01",
                "difficulty": "medium",
                "level": 2,
                "is_completed": False,
                "container_status": "stopped",
            },
        ]
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["last_challenges_snapshot"] = snapshot
            self.pipeline._benchmark_state["closed_challenges"] = {"c-03"}

        should_fast, reason = self.pipeline._benchmark_should_use_fast_path()
        selected = self.pipeline._benchmark_select_next_easy(snapshot)

        self.assertTrue(should_fast)
        self.assertIn("恢复 fast path", reason)
        self.assertIsNone(selected)


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

    def test_benchmark_guard_blocks_repeated_openvpn(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["vpn_connected"] = True
            self.pipeline._benchmark_state["tun_interface"] = "tun0"
            self.pipeline._benchmark_state["tun_ip"] = "10.254.0.10"

        handler = self.pipeline._make_subtask_approval_handler("TSec Benchmark")
        decision = handler(
            MagicMock(),
            {
                "name": "run_shell_command",
                "args": {
                    "command": (
                        "sudo openvpn --config "
                        "/home/my/cyber/benchmark_test/task.ovpn --daemon"
                    ),
                },
            },
        )

        self.assertFalse(decision.approved)
        self.assertIn("VPN 已连接", decision.reason)

    def test_benchmark_guard_blocks_platform_api_on_tun0(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["api_interface"] = "enp0s20f0u3u4"

        handler = self.pipeline._make_subtask_approval_handler("TSec Benchmark")
        decision = handler(
            MagicMock(),
            {
                "name": "run_shell_command",
                "args": {
                    "command": (
                        "curl --interface tun0 "
                        "https://tsecbench.zc.tencent.com/openapi/v1/challenges"
                    ),
                },
            },
        )

        self.assertFalse(decision.approved)
        self.assertIn("平台 API 禁止走 tun0", decision.reason)

    def test_benchmark_guard_blocks_platform_api_on_any_tun(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["api_interface"] = "enp0s20f0u3u4"

        handler = self.pipeline._make_subtask_approval_handler("TSec Benchmark")
        decision = handler(
            MagicMock(),
            {
                "name": "run_shell_command",
                "args": {
                    "command": (
                        "curl --interface tun1 "
                        "https://tsecbench.zc.tencent.com/openapi/v1/challenges"
                    ),
                },
            },
        )

        self.assertFalse(decision.approved)
        self.assertIn("平台 API 禁止走 tun1", decision.reason)

    def test_benchmark_guard_blocks_container_on_physical_interface(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["tun_interface"] = "tun0"

        handler = self.pipeline._make_subtask_approval_handler("runner\nprobe")
        decision = handler(
            MagicMock(),
            {
                "name": "run_shell_command",
                "args": {
                    "command": (
                        "curl --interface enp0s20f0u3u4 "
                        "http://10.0.180.232:80/"
                    )
                },
            },
        )

        self.assertFalse(decision.approved)
        self.assertIn("--interface tun0", decision.reason)

    def test_benchmark_guard_blocks_fetch_web_page_for_container(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["tun_interface"] = "tun0"

        handler = self.pipeline._make_subtask_approval_handler("runner\nprobe")
        decision = handler(
            MagicMock(),
            {
                "name": "fetch_web_page",
                "args": {"url": "http://10.0.180.232:80/"},
            },
        )

        self.assertFalse(decision.approved)
        self.assertIn("curl --interface tun0", decision.reason)

    def test_benchmark_guard_blocks_platform_api_without_known_interface(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["api_interface"] = "enp0s20f0u3u4"

        handler = self.pipeline._make_subtask_approval_handler("TSec Benchmark")
        decision = handler(
            MagicMock(),
            {
                "name": "run_shell_command",
                "args": {
                    "command": (
                        "curl https://tsecbench.zc.tencent.com/"
                        "openapi/v1/challenges"
                    ),
                },
            },
        )

        self.assertFalse(decision.approved)
        self.assertIn("--interface enp0s20f0u3u4", decision.reason)

    def test_benchmark_guard_blocks_authorization_bearer_for_platform(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True

        handler = self.pipeline._make_subtask_approval_handler("TSec Benchmark")
        decision = handler(
            MagicMock(),
            {
                "name": "run_shell_command",
                "args": {
                    "command": (
                        "curl --interface enp0s20f0u3u4 "
                        "-H 'Authorization: Bearer token' "
                        "https://tsecbench.zc.tencent.com/openapi/v1/challenges"
                    ),
                },
            },
        )

        self.assertFalse(decision.approved)
        self.assertIn("BENCHMARK_TOKEN", decision.reason)

    def test_benchmark_guard_blocks_x_benchmark_token_for_platform(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True

        handler = self.pipeline._make_subtask_approval_handler("TSec Benchmark")
        decision = handler(
            MagicMock(),
            {
                "name": "run_shell_command",
                "args": {
                    "command": (
                        "curl --interface enp0s20f0u3u4 "
                        "-H 'X-Benchmark-Token: token' "
                        "https://tsecbench.zc.tencent.com/openapi/v1/challenges"
                    ),
                },
            },
        )

        self.assertFalse(decision.approved)
        self.assertIn("X-Benchmark-Token", decision.reason)

    def test_benchmark_guard_blocks_challenge_id_submit_payload(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True

        handler = self.pipeline._make_subtask_approval_handler("TSec Benchmark")
        decision = handler(
            MagicMock(),
            {
                "name": "run_shell_command",
                "args": {
                    "command": (
                        "curl --interface enp0s20f0u3u4 "
                        "-H 'BENCHMARK_TOKEN: token' "
                        "-d '{\"challenge_id\":\"xben-001-24\","
                        "\"flag\":\"flag{x}\"}' "
                        "https://tsecbench.zc.tencent.com/"
                        "openapi/v1/challenges/submit"
                    ),
                },
            },
        )

        self.assertFalse(decision.approved)
        self.assertIn("unique_code", decision.reason)

    def test_benchmark_guard_blocks_hint_api_by_default(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True

        handler = self.pipeline._make_subtask_approval_handler("TSec Benchmark")
        decision = handler(
            MagicMock(),
            {
                "name": "run_shell_command",
                "args": {
                    "command": (
                        "curl --interface enp0s20f0u3u4 "
                        "-H 'BENCHMARK_TOKEN: token' "
                        "https://tsecbench.zc.tencent.com/openapi/v1/"
                        "challenges/hint?unique_code=xben-005-24"
                    ),
                },
            },
        )

        self.assertFalse(decision.approved)
        self.assertIn("hint 会扣分", decision.reason)

    def test_benchmark_guard_blocks_auto_solve_background_script(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True

        handler = self.pipeline._make_subtask_approval_handler("TSec Benchmark")
        decision = handler(
            MagicMock(),
            {
                "name": "run_shell_command",
                "args": {"command": "bash auto_solve.sh 2>&1 &"},
            },
        )

        self.assertFalse(decision.approved)
        self.assertIn("auto_solve.sh", decision.reason)

    def test_benchmark_guard_blocks_wrong_active_container_port(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["active_containers"] = {
                "xben-009-24": ["10.0.170.160:8000"],
            }

        handler = self.pipeline._make_subtask_approval_handler("TSec Benchmark")
        decision = handler(
            MagicMock(),
            {
                "name": "run_shell_command",
                "args": {
                    "command": "curl http://10.0.170.160:80/",
                },
            },
        )

        self.assertFalse(decision.approved)
        self.assertIn("10.0.170.160:8000", decision.reason)

    def test_benchmark_guard_blocks_home_study_probe(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True

        handler = self.pipeline._make_subtask_approval_handler("TSec Benchmark")
        decision = handler(
            MagicMock(),
            {
                "name": "run_shell_command",
                "args": {"command": "ls /home/study/pwn2own"},
            },
        )

        self.assertFalse(decision.approved)
        self.assertIn("/home/my/cyber/benchmark_test", decision.reason)

    def test_benchmark_guard_blocks_completed_challenge_restart(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["completed_challenges"] = {"xben-006-24"}

        handler = self.pipeline._make_subtask_approval_handler("TSec Benchmark")
        decision = handler(
            MagicMock(),
            {
                "name": "run_shell_command",
                "args": {
                    "command": (
                        "curl --interface enp0s20f0u3u4 -X POST "
                        "https://tsecbench.zc.tencent.com/openapi/v1/"
                        "challenges/start?unique_code=xben-006-24 "
                        "-H 'BENCHMARK_TOKEN: token'"
                    ),
                },
            },
        )

        self.assertFalse(decision.approved)
        self.assertIn("已确认通关", decision.reason)

    def test_benchmark_guard_blocks_closing_inactive_stopped_easy(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["last_challenges_snapshot"] = [
                {
                    "unique_code": "xben-013-24",
                    "difficulty": "easy",
                    "is_completed": False,
                    "container_status": "stopped",
                    "container_addr": [],
                }
            ]
            self.pipeline._benchmark_state["api_interface"] = "enp0s20f0u3u4"

        handler = self.pipeline._make_subtask_approval_handler("TSec Benchmark")
        decision = handler(
            MagicMock(),
            {
                "name": "run_shell_command",
                "args": {
                    "command": (
                        "curl --interface enp0s20f0u3u4 -X POST "
                        "https://tsecbench.zc.tencent.com/openapi/v1/"
                        "challenges/close?unique_code=xben-013-24 "
                        "-H 'BENCHMARK_TOKEN: token'"
                    ),
                },
            },
        )

        self.assertFalse(decision.approved)
        self.assertIn("未启动、未完成", decision.reason)

    def test_benchmark_guard_blocks_closing_inactive_stopped_policy_candidate(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "benchmark-profiles.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "selection_policy": {
                            "difficulty_order": ["medium", "easy", "hard"],
                            "fast_path_difficulties": ["medium"],
                        }
                    }
                ),
                encoding="utf-8",
            )
            self.pipeline._runtime_context["benchmark_profiles_path"] = str(profile_path)
            with self.pipeline._benchmark_state_lock:
                self.pipeline._benchmark_state["last_challenges_snapshot"] = [
                    {
                        "unique_code": "medium-1",
                        "difficulty": "medium",
                        "is_completed": False,
                        "container_status": "stopped",
                        "container_addr": [],
                    }
                ]
                self.pipeline._benchmark_state["api_interface"] = "enp0s20f0u3u4"

            handler = self.pipeline._make_subtask_approval_handler("TSec Benchmark")
            decision = handler(
                MagicMock(),
                {
                    "name": "run_shell_command",
                    "args": {
                        "command": (
                            "curl --interface enp0s20f0u3u4 -X POST "
                            "https://tsecbench.zc.tencent.com/openapi/v1/"
                            "challenges/close?unique_code=medium-1 "
                            "-H 'BENCHMARK_TOKEN: token'"
                        ),
                    },
                },
            )

        self.assertFalse(decision.approved)
        self.assertIn("stopped medium", decision.reason)

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


class CliBenchmarkApprovalGuardTestCase(unittest.TestCase):
    """测试 CLI 主 Agent 审批器的 Benchmark 硬性约束。"""

    def setUp(self):
        self.tool = MagicMock()
        self.tool.name = "run_shell_command"
        self.tool.metadata = {"risk": "execute"}
        self.runtime_context = {
            "approval_policy": ApprovalPolicy.AUTO,
            "benchmark_profile": "aggressive",
        }

    def test_cli_auto_approval_blocks_platform_api_on_tun(self):
        handler = create_approval_handler(self.runtime_context)

        decision = handler(
            self.tool,
            {
                "name": "run_shell_command",
                "args": {
                    "command": (
                        "curl -s --interface tun0 -H 'BENCHMARK_TOKEN: token' "
                        "https://tsecbench.zc.tencent.com/openapi/v1/challenges"
                    )
                },
            },
        )

        self.assertFalse(decision.approved)
        self.assertIn("平台 API 禁止走 tun0", decision.reason)

    def test_cli_auto_approval_blocks_platform_api_without_interface(self):
        handler = create_approval_handler(self.runtime_context)

        decision = handler(
            self.tool,
            {
                "name": "run_shell_command",
                "args": {
                    "command": (
                        "curl -s -H 'BENCHMARK_TOKEN: token' "
                        "https://tsecbench.zc.tencent.com/openapi/v1/challenges"
                    )
                },
            },
        )

        self.assertFalse(decision.approved)
        self.assertIn("必须显式绑定物理出口", decision.reason)

    def test_cli_auto_approval_allows_platform_api_on_physical_interface(self):
        handler = create_approval_handler(self.runtime_context)

        decision = handler(
            self.tool,
            {
                "name": "run_shell_command",
                "args": {
                    "command": (
                        "curl -s --interface enp0s20f0u3u4 "
                        "-H 'BENCHMARK_TOKEN: token' "
                        "https://tsecbench.zc.tencent.com/openapi/v1/challenges"
                    )
                },
            },
        )

        self.assertTrue(decision.approved)


if __name__ == "__main__":
    unittest.main()
