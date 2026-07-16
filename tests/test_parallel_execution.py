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
                "Benchmark dify handoff step 1：只深挖当前 active 题"
            )
        )

        self.assertFalse(pause)
        self.assertEqual(reason, "")

    def test_benchmark_handoff_selection_restores_close_step(self):
        subtasks = [
            {"task_description": "Benchmark dify handoff step 1：只深挖当前 active"},
            {"task_description": "Benchmark dify exploit step 2：尝试最高置信利用"},
            {"task_description": "Benchmark dify close step 3：无 flag 则 close"},
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

    def test_non_xben_benchmark_start_stays_sequential(self):
        self.pipeline._runtime_context["subtask_concurrency"] = "force"

        allowed, reason = self.pipeline._subtask_parallel_decision(
            {
                "task_description": "POST start c-03 并解析容器地址",
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

    def test_resource_keys_extract_current_benchmark_code_shapes(self):
        keys = self.pipeline._extract_subtask_resource_keys(
            {
                "task_description": (
                    "分析 c-03 和 f1-04，访问 10.0.180.232:3000，"
                    "然后 close 当前题"
                )
            }
        )

        self.assertIn("challenge:c-03", keys)
        self.assertIn("challenge:f1-04", keys)
        self.assertIn("api:tsecbench-close", keys)

    def test_max_subagents_one_disables_concurrency(self):
        self.pipeline._runtime_context["subtask_concurrency"] = "force"
        self.pipeline._runtime_context["max_subagents"] = 1

        self.assertEqual(self.pipeline._resolve_subtask_concurrency(), "off")

    def test_benchmark_stale_detector_forces_close_and_switch(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True

        directive = ""
        for _ in range(2):
            directive = self.pipeline._update_benchmark_stale_state(
                ["## [runner] 探索 xben-001-24\n测试 SQLi/SSTI，未发现有效结果。"]
            )

        self.assertIn("close?unique_code=xben-001-24", directive)
        self.assertIn("选择下一道未完成", directive)
        events = [item["event"] for item in self.pipeline._trace]
        self.assertIn("benchmark_stale_detected", events)

    def test_benchmark_stale_detector_switches_on_low_value_noise(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True

        directive = self.pipeline._update_benchmark_stale_state(
            [
                "## [runner] 探索 xben-006-24\n"
                + "\n".join(
                    ["输出: Internal Server Error"]
                    * BENCHMARK_LOW_VALUE_SIGNAL_LIMIT
                )
            ]
        )

        self.assertIn("close?unique_code=xben-006-24", directive)
        self.assertIn("低价值探测", directive)
        state = self.pipeline._benchmark_state_snapshot()
        self.assertIn("xben-006-24", state["abandoned_challenges"])

    def test_benchmark_stale_detector_handles_non_xben_code(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True

        directive = ""
        for _ in range(2):
            directive = self.pipeline._update_benchmark_stale_state(
                ["## [runner] 探索 c-03\n测试登录/API/路径，未发现有效结果。"]
            )

        self.assertIn("close?unique_code=c-03", directive)
        state = self.pipeline._benchmark_state_snapshot()
        self.assertIn("c-03", state["abandoned_challenges"])

    def test_benchmark_timeout_directive_marks_abandoned(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["current_challenge"] = "xben-006-24"
            self.pipeline._benchmark_current_challenge = "xben-006-24"

        directive = self.pipeline._benchmark_timeout_directive("timeout")

        self.assertIn("close?unique_code=xben-006-24", directive)
        state = self.pipeline._benchmark_state_snapshot()
        self.assertIn("xben-006-24", state["abandoned_challenges"])

    def test_benchmark_setup_timeout_stops_after_repeated_pre_challenge_timeouts(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True

        first = self.pipeline._benchmark_setup_timeout_stop_reason("timeout 1")
        second = self.pipeline._benchmark_setup_timeout_stop_reason("timeout 2")
        third = self.pipeline._benchmark_setup_timeout_stop_reason("timeout 3")

        self.assertEqual(first, "")
        self.assertEqual(second, "")
        self.assertIn("未锁定当前题前超时", third)
        state = self.pipeline._benchmark_state_snapshot()
        self.assertEqual(state["setup_timeout_streak"], 3)

    def test_benchmark_setup_timeout_does_not_stop_active_challenge(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["current_challenge"] = "xben-006-24"
            self.pipeline._benchmark_current_challenge = "xben-006-24"

        stop_reason = self.pipeline._benchmark_setup_timeout_stop_reason("timeout")
        directive = self.pipeline._benchmark_timeout_directive("timeout")

        self.assertEqual(stop_reason, "")
        self.assertIn("close?unique_code=xben-006-24", directive)

    def test_benchmark_setup_timeout_streak_resets_after_success(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["setup_timeout_streak"] = 2

        self.pipeline._benchmark_reset_setup_timeout_streak()

        state = self.pipeline._benchmark_state_snapshot()
        self.assertEqual(state["setup_timeout_streak"], 0)

    def test_benchmark_timeout_does_not_abandon_reasoning_challenge(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["current_challenge"] = "a-05"
            self.pipeline._benchmark_state["reasoning_challenges"] = {"a-05"}

        directive = self.pipeline._benchmark_timeout_directive("subtask timeout")

        state = self.pipeline._benchmark_state_snapshot()
        self.assertEqual(directive, "")
        self.assertNotIn("a-05", state["abandoned_challenges"])

    def test_benchmark_guard_blocks_abandoned_active_container(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["current_challenge"] = "xben-006-24"
            self.pipeline._benchmark_state["active_containers"] = {
                "xben-006-24": ["10.0.185.120:80"],
            }
            self.pipeline._benchmark_state["abandoned_challenges"] = {"xben-006-24"}

        handler = self.pipeline._make_subtask_approval_handler("runner\nprobe")
        decision = handler(
            MagicMock(),
            {
                "name": "run_shell_command",
                "args": {"command": "curl http://10.0.185.120:80/jobs"},
            },
        )

        self.assertFalse(decision.approved)
        self.assertIn("低收益题", decision.reason)

    def test_benchmark_guard_blocks_duplicate_close(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["closed_challenges"] = {"xben-006-24"}

        handler = self.pipeline._make_subtask_approval_handler("runner\nclose")
        decision = handler(
            MagicMock(),
            {
                "name": "run_shell_command",
                "args": {
                    "command": (
                        "curl --interface enp0s20f0u3u4 "
                        "https://tsecbench.zc.tencent.com/openapi/v1/challenges/"
                        "close?unique_code=xben-006-24 "
                        "-H 'BENCHMARK_TOKEN: token'"
                    )
                },
            },
        )

        self.assertFalse(decision.approved)
        self.assertIn("禁止重复 close", decision.reason)

    def test_benchmark_guard_blocks_close_of_reasoning_challenge(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["current_challenge"] = "c-03"
            self.pipeline._benchmark_state["active_containers"] = {
                "c-03": ["10.0.180.232:3000"],
            }
            self.pipeline._benchmark_state["reasoning_challenges"] = {"c-03"}

        handler = self.pipeline._make_subtask_approval_handler("runner\nclose")
        decision = handler(
            MagicMock(),
            {
                "name": "run_shell_command",
                "args": {
                    "command": (
                        "curl --interface enp0s20f0u3u4 -X POST "
                        "https://tsecbench.zc.tencent.com/openapi/v1/challenges/"
                        "close?unique_code=c-03 "
                        "-H 'BENCHMARK_TOKEN: token'"
                    )
                },
            },
        )

        self.assertFalse(decision.approved)
        self.assertIn("禁止作为无进展题 close", decision.reason)

    def test_benchmark_guard_blocks_start_close_json_body_unique_code(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True

        handler = self.pipeline._make_subtask_approval_handler("runner\napi")
        start_decision = handler(
            MagicMock(),
            {
                "name": "run_shell_command",
                "args": {
                    "command": (
                        "curl --interface enp0s20f0u3u4 -X POST "
                        "https://tsecbench.zc.tencent.com/openapi/v1/challenges/start "
                        "-H 'BENCHMARK_TOKEN: token' "
                        "-d '{\"unique_code\":\"xben-006-24\"}'"
                    )
                },
            },
        )
        close_decision = handler(
            MagicMock(),
            {
                "name": "run_shell_command",
                "args": {
                    "command": (
                        "curl --interface enp0s20f0u3u4 -X POST "
                        "https://tsecbench.zc.tencent.com/openapi/v1/challenges/close "
                        "-H 'BENCHMARK_TOKEN: token' "
                        "-d '{\"unique_code\":\"xben-006-24\"}'"
                    )
                },
            },
        )

        self.assertFalse(start_decision.approved)
        self.assertIn("start?unique_code=xben-006-24", start_decision.reason)
        self.assertFalse(close_decision.approved)
        self.assertIn("close?unique_code=xben-006-24", close_decision.reason)

    def test_benchmark_guard_blocks_form_body_unique_code(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True

        handler = self.pipeline._make_subtask_approval_handler("runner\napi")
        decision = handler(
            MagicMock(),
            {
                "name": "run_shell_command",
                "args": {
                    "command": (
                        "curl --interface enp0s20f0u3u4 -X POST "
                        "https://tsecbench.zc.tencent.com/openapi/v1/challenges/start "
                        "-H 'BENCHMARK_TOKEN: token' "
                        "-d unique_code=xben-020-24"
                    )
                },
            },
        )

        self.assertFalse(decision.approved)
        self.assertIn("start?unique_code=xben-020-24", decision.reason)

    def test_benchmark_guard_blocks_batch_start_commands(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True

        handler = self.pipeline._make_subtask_approval_handler("runner\nstart")
        decision = handler(
            MagicMock(),
            {
                "name": "run_shell_command",
                "args": {
                    "command": (
                        "curl --interface enp0s20f0u3u4 -X POST "
                        "https://tsecbench.zc.tencent.com/openapi/v1/challenges/"
                        "start?unique_code=xben-006-24 "
                        "-H 'BENCHMARK_TOKEN: token'; "
                        "curl --interface enp0s20f0u3u4 -X POST "
                        "https://tsecbench.zc.tencent.com/openapi/v1/challenges/"
                        "start?unique_code=xben-020-24 "
                        "-H 'BENCHMARK_TOKEN: token'"
                    )
                },
            },
        )

        self.assertFalse(decision.approved)
        self.assertIn("批量 start 多题", decision.reason)

    def test_benchmark_guard_blocks_start_while_active_open_exists(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["active_containers"] = {
                "xben-006-24": ["10.0.185.120:80"],
            }

        handler = self.pipeline._make_subtask_approval_handler("runner\nstart")
        new_challenge_decision = handler(
            MagicMock(),
            {
                "name": "run_shell_command",
                "args": {
                    "command": (
                        "curl --interface enp0s20f0u3u4 -X POST "
                        "https://tsecbench.zc.tencent.com/openapi/v1/challenges/"
                        "start?unique_code=xben-020-24 "
                        "-H 'BENCHMARK_TOKEN: token'"
                    )
                },
            },
        )
        same_challenge_decision = handler(
            MagicMock(),
            {
                "name": "run_shell_command",
                "args": {
                    "command": (
                        "curl --interface enp0s20f0u3u4 -X POST "
                        "https://tsecbench.zc.tencent.com/openapi/v1/challenges/"
                        "start?unique_code=xben-006-24 "
                        "-H 'BENCHMARK_TOKEN: token'"
                    )
                },
            },
        )

        self.assertFalse(new_challenge_decision.approved)
        self.assertIn("当前已有 active 容器 xben-006-24", new_challenge_decision.reason)
        self.assertFalse(same_challenge_decision.approved)
        self.assertIn("禁止继续 start", same_challenge_decision.reason)

    def test_benchmark_guard_blocks_medium_start_while_easy_remains(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["last_challenges_snapshot"] = [
                {
                    "unique_code": "xben-001-24",
                    "difficulty": "medium",
                    "is_completed": False,
                    "container_status": "stopped",
                },
                {
                    "unique_code": "xben-006-24",
                    "difficulty": "easy",
                    "is_completed": False,
                    "container_status": "stopped",
                },
            ]

        handler = self.pipeline._make_subtask_approval_handler("runner\nstart")
        decision = handler(
            MagicMock(),
            {
                "name": "run_shell_command",
                "args": {
                    "command": (
                        "curl --interface enp0s20f0u3u4 -X POST "
                        "https://tsecbench.zc.tencent.com/openapi/v1/challenges/"
                        "start?unique_code=xben-001-24 "
                        "-H 'BENCHMARK_TOKEN: token'"
                    )
                },
            },
        )

        self.assertFalse(decision.approved)
        self.assertIn("更高优先级未完成候选 xben-006-24", decision.reason)

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
        for _ in range(2):
            directive = self.pipeline._update_benchmark_stale_state(
                [
                    "继续探索 xben-001-24。API 文档说明字段 "
                    "`correct_flag_count`、`cumulative_score`，但没有提交结果。"
                ]
            )

        self.assertIn("close?unique_code=xben-001-24", directive)

    def test_benchmark_stale_detector_suppresses_reasoning_challenge(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        self.pipeline._benchmark_current_challenge = "d-01"
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["reasoning_challenges"] = {"d-01"}

        directive = self.pipeline._update_benchmark_stale_state(
            [
                "继续探索 d-01。404 404 404 404 404 404 404 404 "
                "但 d-01 已获取有效响应，保留 active 并切回四柱/runner 深挖。"
            ]
        )

        state = self.pipeline._benchmark_state_snapshot()
        self.assertEqual(directive, "")
        self.assertEqual(self.pipeline._benchmark_stale_rounds, 0)
        self.assertNotIn("d-01", state["abandoned_challenges"])

    def test_benchmark_runtime_state_tracks_submit_and_close(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["reasoning_challenges"] = {"xben-005-24"}
            self.pipeline._benchmark_state["probe_unreachable_streaks"] = {
                "xben-005-24": 1,
            }

        self.pipeline._update_benchmark_runtime_state(
            '命令: curl --interface enp0s20f0u3u4 '
            "-H 'BENCHMARK_TOKEN: token' "
            "-d '{\"unique_code\":\"xben-006-24\",\"flag\":\"flag{a}\"}' "
            "https://tsecbench.zc.tencent.com/openapi/v1/challenges/submit\n"
            "工作目录: /tmp\n退出码: 0\n输出:\n"
            '{"correct":true,"awarded":200,"cumulative_score":200,'
            '"correct_flag_count":1,"total_flag_count":1}'
        )
        self.pipeline._update_benchmark_runtime_state(
            "命令: curl --interface enp0s20f0u3u4 "
            "https://tsecbench.zc.tencent.com/openapi/v1/challenges/"
            "close?unique_code=xben-005-24\n"
            "工作目录: /tmp\n退出码: 0\n输出:\n"
            '{"unique_code":"xben-005-24","closed":true}'
        )

        state = self.pipeline._benchmark_state_snapshot()
        self.assertIn("xben-006-24", state["completed_challenges"])
        self.assertIn("xben-005-24", state["closed_challenges"])
        self.assertEqual(state["abandoned_challenges"], [])
        self.assertEqual(state["last_score"], 200)
        self.assertEqual(state["completed_scores"]["xben-006-24"], 200)
        self.assertNotIn("xben-005-24", state["reasoning_challenges"])
        self.assertNotIn("xben-005-24", state["probe_unreachable_streaks"])

    def test_benchmark_runtime_state_tracks_non_xben_codes(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True

        self.pipeline._update_benchmark_runtime_state(
            "命令: curl --interface enp0s20f0u3u4 "
            "https://tsecbench.zc.tencent.com/openapi/v1/challenges/"
            "start?unique_code=d-01\n"
            "工作目录: /tmp\n退出码: 0\n输出:\n"
            '{"unique_code":"d-01","container_addr":["10.0.180.232:8000"]}'
        )
        state = self.pipeline._benchmark_state_snapshot()
        self.assertEqual(state["current_challenge"], "d-01")
        self.assertEqual(state["active_containers"], {"d-01": ["10.0.180.232:8000"]})

        self.pipeline._update_benchmark_runtime_state(
            "命令: curl --interface enp0s20f0u3u4 "
            "-d '{\"unique_code\":\"d-01\",\"flag\":\"flag{a}\"}' "
            "https://tsecbench.zc.tencent.com/openapi/v1/challenges/submit\n"
            "工作目录: /tmp\n退出码: 0\n输出:\n"
            '{"correct":true,"awarded":200,"cumulative_score":200}'
        )
        self.pipeline._update_benchmark_runtime_state(
            "命令: curl --interface enp0s20f0u3u4 "
            "https://tsecbench.zc.tencent.com/openapi/v1/challenges/"
            "close?unique_code=d-01\n"
            "工作目录: /tmp\n退出码: 0\n输出:\n"
            '{"unique_code":"d-01","closed":true}'
        )

        state = self.pipeline._benchmark_state_snapshot()
        self.assertIn("d-01", state["completed_challenges"])
        self.assertIn("d-01", state["closed_challenges"])
        self.assertEqual(state["active_containers"], {})
        self.assertEqual(state["completed_scores"]["d-01"], 200)

    def test_benchmark_close_local_clears_active_state_immediately(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["current_challenge"] = "d-01"
            self.pipeline._benchmark_state["active_containers"] = {
                "d-01": ["10.0.180.232:8000"],
            }
            self.pipeline._benchmark_current_challenge = "d-01"

        with (
            patch.object(
                self.pipeline,
                "_benchmark_platform_request",
                return_value=(200, '{"unique_code":"d-01","closed":true}', ""),
            ),
            patch.object(self.pipeline, "_persist_benchmark_state") as persist,
        ):
            result = self.pipeline._benchmark_close_local("d-01")

        state = self.pipeline._benchmark_state_snapshot()
        self.assertIn('"closed":true', result)
        self.assertIsNone(state["current_challenge"])
        self.assertEqual(state["active_containers"], {})
        self.assertIn("d-01", state["closed_challenges"])
        persist.assert_called_once()

    def test_benchmark_runtime_state_does_not_record_tun_as_platform_interface(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True

        self.pipeline._update_benchmark_runtime_state(
            "命令: curl --interface tun1 "
            "-H 'BENCHMARK_TOKEN: token' "
            "https://tsecbench.zc.tencent.com/openapi/v1/challenges\n"
            "工作目录: /tmp\n退出码: 0\n输出:\n[]"
        )

        state = self.pipeline._benchmark_state_snapshot()
        self.assertIsNone(state["api_interface"])

    def test_benchmark_runtime_state_clears_current_after_close(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["current_challenge"] = "xben-006-24"
            self.pipeline._benchmark_current_challenge = "xben-006-24"
            self.pipeline._benchmark_state["active_containers"] = {
                "xben-006-24": ["10.0.1.2:80"],
            }

        self.pipeline._update_benchmark_runtime_state(
            "命令: curl --interface enp0s20f0u3u4 "
            "https://tsecbench.zc.tencent.com/openapi/v1/challenges/"
            "close?unique_code=xben-006-24\n"
            "工作目录: /tmp\n退出码: 0\n输出:\n"
            '{"unique_code":"xben-006-24","closed":true}'
        )

        state = self.pipeline._benchmark_state_snapshot()
        self.assertIsNone(state["current_challenge"])
        self.assertEqual(state["active_containers"], {})
        self.assertIn("xben-006-24", state["closed_challenges"])

    def test_benchmark_runtime_state_handles_close_without_unique_code(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["current_challenge"] = "xben-006-24"
            self.pipeline._benchmark_current_challenge = "xben-006-24"
            self.pipeline._benchmark_state["active_containers"] = {
                "xben-006-24": ["10.0.1.2:80"],
            }

        self.pipeline._update_benchmark_runtime_state(
            "命令: curl --interface enp0s20f0u3u4 "
            "https://tsecbench.zc.tencent.com/openapi/v1/challenges/"
            "close?unique_code=xben-006-24\n"
            "工作目录: /tmp\n退出码: 0\n输出:\n"
            '{"closed":true}'
        )

        state = self.pipeline._benchmark_state_snapshot()
        self.assertIsNone(state["current_challenge"])
        self.assertEqual(state["active_containers"], {})
        self.assertIn("xben-006-24", state["closed_challenges"])

    def test_benchmark_auto_submits_flag_from_container_output(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["api_interface"] = "enp0s20f0u3u4"
            self.pipeline._benchmark_state["current_challenge"] = "xben-006-24"
            self.pipeline._benchmark_state["active_containers"] = {
                "xben-006-24": ["10.0.1.2:80"],
            }

        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                '{"correct":true,"awarded":200,"cumulative_score":200,'
                '"correct_flag_count":1,"total_flag_count":1}'
            ),
            stderr="",
        )
        with (
            patch.object(
                self.pipeline,
                "_benchmark_api_config_from_workspace",
                return_value=("https://tsecbench.zc.tencent.com", "token"),
            ),
            patch(
                "cyber_agent.agent.pipeline.subprocess.run",
                return_value=completed,
            ) as run,
        ):
            self.pipeline._benchmark_auto_submit_flags_from_tool_result(
                "命令: curl http://10.0.1.2:80/flag\n"
                "工作目录: /tmp\n退出码: 0\n输出:\n"
                "flag{auto-submit-me}"
            )

        run.assert_called_once()
        state = self.pipeline._benchmark_state_snapshot()
        self.assertIn("xben-006-24", state["completed_challenges"])
        self.assertEqual(state["completed_scores"]["xben-006-24"], 200)
        self.assertIn("xben-006-24:flag{auto-submit-me}", state["auto_submitted_flags"])

    def test_benchmark_auto_submit_infers_challenge_from_container_address(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["api_interface"] = "enp0s20f0u3u4"
            self.pipeline._benchmark_state["active_containers"] = {
                "xben-006-24": ["10.0.1.2:80"],
                "xben-009-24": ["10.0.1.3:80"],
            }

        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"correct":true,"awarded":200,"cumulative_score":200}',
            stderr="",
        )
        with (
            patch.object(
                self.pipeline,
                "_benchmark_api_config_from_workspace",
                return_value=("https://tsecbench.zc.tencent.com", "token"),
            ),
            patch(
                "cyber_agent.agent.pipeline.subprocess.run",
                return_value=completed,
            ) as run,
        ):
            self.pipeline._benchmark_auto_submit_flags_from_tool_result(
                "命令: curl http://10.0.1.3:80/flag\n"
                "工作目录: /tmp\n退出码: 0\n输出:\n"
                "flag{address-inferred}"
            )

        run.assert_called_once()
        body = run.call_args.args[0][-1]
        self.assertIn('"unique_code":"xben-009-24"', body)
        state = self.pipeline._benchmark_state_snapshot()
        self.assertIn("xben-009-24", state["completed_challenges"])

    def test_benchmark_auto_submit_allows_abandoned_when_flag_is_found(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["api_interface"] = "enp0s20f0u3u4"
            self.pipeline._benchmark_state["current_challenge"] = "xben-006-24"
            self.pipeline._benchmark_state["active_containers"] = {
                "xben-006-24": ["10.0.1.2:80"],
            }
            self.pipeline._benchmark_state["abandoned_challenges"] = {"xben-006-24"}

        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"correct":true,"awarded":200,"cumulative_score":200}',
            stderr="",
        )
        with (
            patch.object(
                self.pipeline,
                "_benchmark_api_config_from_workspace",
                return_value=("https://tsecbench.zc.tencent.com", "token"),
            ),
            patch(
                "cyber_agent.agent.pipeline.subprocess.run",
                return_value=completed,
            ) as run,
        ):
            self.pipeline._benchmark_auto_submit_flags_from_tool_result(
                "命令: curl http://10.0.1.2:80/flag\n"
                "工作目录: /tmp\n退出码: 0\n输出:\n"
                "flag{late-but-valid}"
            )

        run.assert_called_once()
        state = self.pipeline._benchmark_state_snapshot()
        self.assertIn("xben-006-24", state["completed_challenges"])

    def test_benchmark_runtime_state_persists_submit_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
            self.pipeline._runtime_context["session_id"] = "bench-session"
            self.pipeline._runtime_context["session_base_dir"] = tmpdir
            self.pipeline._benchmark_profile_active = True

            self.pipeline._update_benchmark_runtime_state(
                '命令: curl --interface enp0s20f0u3u4 '
                "-H 'BENCHMARK_TOKEN: token' "
                "-d '{\"unique_code\":\"xben-006-24\",\"flag\":\"flag{a}\"}' "
                "https://tsecbench.zc.tencent.com/openapi/v1/challenges/submit\n"
                "工作目录: /tmp\n退出码: 0\n输出:\n"
                '{"correct":true,"awarded":200,"cumulative_score":200,'
                '"correct_flag_count":1,"total_flag_count":1}'
            )

            state_path = self.pipeline._benchmark_state_path()
            self.assertTrue(state_path.exists())
            saved = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["state"]["completed_scores"]["xben-006-24"], 200)
            self.assertIn("xben-006-24", saved["state"]["completed_challenges"])

    def test_benchmark_runtime_state_persists_active_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
            self.pipeline._runtime_context["session_id"] = "bench-session"
            self.pipeline._runtime_context["session_base_dir"] = tmpdir
            self.pipeline._benchmark_profile_active = True

            self.pipeline._update_benchmark_runtime_state(
                "命令: curl --interface enp0s20f0u3u4 -X POST "
                "https://tsecbench.zc.tencent.com/openapi/v1/challenges/"
                "start?unique_code=xben-006-24\n"
                "工作目录: /tmp\n退出码: 0\n输出:\n"
                '{"unique_code":"xben-006-24",'
                '"container_addr":["10.0.1.2:80"]}'
            )

            saved = json.loads(
                self.pipeline._benchmark_state_path().read_text(encoding="utf-8")
            )
            self.assertEqual(saved["state"]["current_challenge"], "xben-006-24")
            self.assertEqual(
                saved["state"]["active_containers"],
                {"xben-006-24": ["10.0.1.2:80"]},
            )

    def test_benchmark_runtime_state_ignores_api_doc_examples(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True

        self.pipeline._update_benchmark_runtime_state(
            "文件: CHALLENGES_API.md\n"
            "内容过长，已截断为前 8000 个字符：\n"
            '示例响应 [{"unique_code":"web_sql_injection_01",'
            '"is_completed":false,"container_status":"available",'
            '"container_addr":["10.0.1.5:8080"]}]\n'
            '示例 submit {"correct":true,"awarded":50,'
            '"cumulative_score":80,"correct_flag_count":2,'
            '"total_flag_count":3}'
        )

        state = self.pipeline._benchmark_state_snapshot()
        self.assertIsNone(state["last_challenges_snapshot"])
        self.assertEqual(state["active_containers"], {})
        self.assertEqual(state["completed_challenges"], [])
        self.assertEqual(state["completed_scores"], {})
        self.assertIsNone(state["last_score"])

    def test_benchmark_runtime_state_keeps_submit_score_over_snapshot_total(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True

        self.pipeline._update_benchmark_runtime_state(
            '命令: curl --interface enp0s20f0u3u4 '
            "-H 'BENCHMARK_TOKEN: token' "
            "-d '{\"unique_code\":\"xben-006-24\",\"flag\":\"flag{a}\"}' "
            "https://tsecbench.zc.tencent.com/openapi/v1/challenges/submit\n"
            "工作目录: /tmp\n退出码: 0\n输出:\n"
            '{"correct":true,"awarded":180,"cumulative_score":180,'
            '"correct_flag_count":1,"total_flag_count":1}'
        )
        self.pipeline._update_benchmark_runtime_state(
            "命令: curl --interface enp0s20f0u3u4 "
            "https://tsecbench.zc.tencent.com/openapi/v1/challenges\n"
            "工作目录: /tmp\n退出码: 0\n输出:\n"
            "["
            '{"unique_code":"xben-006-24","is_completed":true,'
            '"total_score":200,"container_status":"stopped",'
            '"container_addr":[]}'
            "]"
        )

        state = self.pipeline._benchmark_state_snapshot()
        self.assertEqual(state["completed_scores"]["xben-006-24"], 180)

    def test_benchmark_runtime_state_does_not_mark_stopped_list_items_active(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True

        self.pipeline._update_benchmark_runtime_state(
            "命令: curl --interface enp0s20f0u3u4 "
            "https://tsecbench.zc.tencent.com/openapi/v1/challenges\n"
            "工作目录: /tmp\n退出码: 0\n输出:\n"
            "["
            '{"unique_code":"xben-001-24","is_completed":false,'
            '"container_status":"stopped","container_addr":[]},'
            '{"unique_code":"xben-006-24","is_completed":true,'
            '"total_score":200,"container_status":"stopped",'
            '"container_addr":[]},'
            '{"unique_code":"xben-009-24","is_completed":false,'
            '"container_status":"available","container_addr":["10.0.1.2:80"]}'
            "]"
        )

        state = self.pipeline._benchmark_state_snapshot()
        self.assertEqual(state["active_containers"], {"xben-009-24": ["10.0.1.2:80"]})
        self.assertIn("xben-006-24", state["completed_challenges"])
        self.assertEqual(state["completed_scores"]["xben-006-24"], 200)
        self.assertNotIn("xben-001-24", state["active_containers"])

    def test_benchmark_platform_active_overrides_persisted_closed_state(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["closed_challenges"] = {"c-06"}

        self.pipeline._update_benchmark_runtime_state(
            "命令: curl --interface enp0s20f0u3u4 "
            "https://tsecbench.zc.tencent.com/openapi/v1/challenges\n"
            "工作目录: /tmp\n退出码: 0\n输出:\n"
            "["
            '{"unique_code":"c-06","difficulty":"easy","is_completed":false,'
            '"container_status":"available",'
            '"container_addr":["10.0.180.232:8080"]}'
            "]"
        )

        state = self.pipeline._benchmark_state_snapshot()
        self.assertEqual(state["current_challenge"], "c-06")
        self.assertEqual(
            state["active_containers"],
            {"c-06": ["10.0.180.232:8080"]},
        )
        self.assertNotIn("c-06", state["closed_challenges"])

    def test_benchmark_runtime_state_pretty_list_keeps_current_on_available_item(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True

        pretty = json.dumps(
            [
                {
                    "unique_code": "xben-009-24",
                    "is_completed": False,
                    "difficulty": "easy",
                    "container_status": "available",
                    "container_addr": ["10.0.170.160:8000"],
                },
                {
                    "unique_code": "xben-012-24",
                    "is_completed": False,
                    "difficulty": "easy",
                    "container_status": "stopped",
                    "container_addr": [],
                },
            ],
            indent=2,
        )

        self.pipeline._update_benchmark_runtime_state(
            "命令: curl --interface enp0s20f0u3u4 "
            "https://tsecbench.zc.tencent.com/openapi/v1/challenges | "
            "python3 -m json.tool\n"
            "工作目录: /home/my/cyber/benchmark_test\n退出码: 0\n输出:\n"
            f"{pretty}"
        )

        state = self.pipeline._benchmark_state_snapshot()
        self.assertEqual(state["current_challenge"], "xben-009-24")
        self.assertEqual(
            state["active_containers"],
            {"xben-009-24": ["10.0.170.160:8000"]},
        )

    def test_benchmark_runtime_state_does_not_replace_snapshot_from_filtered_pipe(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True

        self.pipeline._update_benchmark_runtime_state(
            "命令: curl --interface enp0s20f0u3u4 "
            "https://tsecbench.zc.tencent.com/openapi/v1/challenges\n"
            "工作目录: /tmp\n退出码: 0\n输出:\n"
            "["
            '{"unique_code":"xben-001-24","is_completed":false,'
            '"total_score":200,"container_status":"stopped","container_addr":[]},'
            '{"unique_code":"xben-005-24","is_completed":true,'
            '"total_score":200,"container_status":"stopped","container_addr":[]}'
            "]"
        )
        self.pipeline._update_benchmark_runtime_state(
            "命令: curl --interface enp0s20f0u3u4 "
            "https://tsecbench.zc.tencent.com/openapi/v1/challenges | "
            "python3 -c 'print filtered easy only'\n"
            "工作目录: /tmp\n退出码: 0\n输出:\n"
            "["
            '{"unique_code":"xben-009-24","is_completed":false,'
            '"total_score":200,"container_status":"stopped","container_addr":[]}'
            "]"
        )

        state = self.pipeline._benchmark_state_snapshot()
        self.assertEqual(state["last_challenges_snapshot"]["total"], 2)
        self.assertIn("xben-005-24", state["completed_challenges"])

    def test_benchmark_deterministic_setup_reuses_any_tun_for_vpn_only(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        calls: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            calls.append(list(cmd))
            if cmd[:4] == ["ip", "-o", "-4", "addr"]:
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout="32: tun1    inet 10.254.0.4/24 scope global tun1\n",
                    stderr="",
                )
            if cmd and cmd[0] == "curl":
                return subprocess.CompletedProcess(cmd, 0, stdout="[]", stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with (
            patch.object(
                self.pipeline,
                "_benchmark_api_config_from_workspace",
                return_value=("https://tsecbench.zc.tencent.com", "token"),
            ),
            patch("cyber_agent.agent.pipeline.subprocess.run", side_effect=fake_run),
        ):
            result = self.pipeline._benchmark_deterministic_fast_step(
                "Benchmark fast setup：只做前置校验。"
            )

        state = self.pipeline._benchmark_state_snapshot()
        self.assertIn("复用已连接 VPN：tun1", result)
        self.assertEqual(state["tun_interface"], "tun1")
        self.assertEqual(state["api_interface"], "enp0s20f0u3u4")
        self.assertFalse(any("openvpn" in part for cmd in calls for part in cmd))

    def test_benchmark_deterministic_step1_starts_next_easy(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["api_interface"] = "enp0s20f0u3u4"

        def fake_run(cmd, **kwargs):
            joined = " ".join(cmd)
            if "/openapi/v1/challenges/start?unique_code=xben-009-24" in joined:
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout=(
                        '{"unique_code":"xben-009-24",'
                        '"container_addr":["10.0.1.2:80"]}'
                    ),
                    stderr="",
                )
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=(
                    "["
                    '{"unique_code":"xben-006-24","difficulty":"easy",'
                    '"level":1,"total_score":200,"is_completed":true,'
                    '"container_status":"stopped","container_addr":[]},'
                    '{"unique_code":"xben-009-24","difficulty":"easy",'
                    '"level":1,"total_score":200,"is_completed":false,'
                    '"container_status":"stopped","container_addr":[]}'
                    "]"
                ),
                stderr="",
            )

        with (
            patch.object(
                self.pipeline,
                "_benchmark_api_config_from_workspace",
                return_value=("https://tsecbench.zc.tencent.com", "token"),
            ),
            patch("cyber_agent.agent.pipeline.subprocess.run", side_effect=fake_run),
        ):
            result = self.pipeline._benchmark_deterministic_fast_step(
                "Benchmark fast step 1：只做调度。"
            )

        state = self.pipeline._benchmark_state_snapshot()
        self.assertIn("启动下一道 easy xben-009-24", result)
        self.assertEqual(state["current_challenge"], "xben-009-24")
        self.assertEqual(
            state["active_containers"],
            {"xben-009-24": ["10.0.1.2:80"]},
        )

    def test_benchmark_deterministic_fast_step_uses_structured_action(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["api_interface"] = "enp0s20f0u3u4"

        def fake_run(cmd, **kwargs):
            joined = " ".join(cmd)
            if "/openapi/v1/challenges/start?unique_code=xben-010-24" in joined:
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout=(
                        '{"unique_code":"xben-010-24",'
                        '"container_addr":["10.0.1.3:80"]}'
                    ),
                    stderr="",
                )
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=(
                    "["
                    '{"unique_code":"xben-010-24","difficulty":"easy",'
                    '"level":1,"total_score":200,"is_completed":false,'
                    '"container_status":"stopped","container_addr":[]}'
                    "]"
                ),
                stderr="",
            )

        with (
            patch.object(
                self.pipeline,
                "_benchmark_api_config_from_workspace",
                return_value=("https://tsecbench.zc.tencent.com", "token"),
            ),
            patch("cyber_agent.agent.pipeline.subprocess.run", side_effect=fake_run),
        ):
            result = self.pipeline._benchmark_deterministic_fast_step(
                "结构化调度任务，不包含旧 fast step 文本。",
                action="schedule",
            )

        self.assertIn("启动下一道 easy xben-010-24", result)

    def test_benchmark_deterministic_start_clears_soft_closed_state(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["closed_challenges"] = {"c-03"}
            self.pipeline._benchmark_state["abandoned_challenges"] = {"c-03"}

        with patch.object(
            self.pipeline,
            "_benchmark_platform_request",
            return_value=(
                0,
                '{"unique_code":"c-03","container_addr":["10.0.1.2:3000"]}',
                "",
            ),
        ):
            result = self.pipeline._benchmark_start_local("c-03")

        state = self.pipeline._benchmark_state_snapshot()
        self.assertIn("10.0.1.2:3000", result)
        self.assertEqual(state["current_challenge"], "c-03")
        self.assertEqual(state["active_containers"], {"c-03": ["10.0.1.2:3000"]})
        self.assertNotIn("c-03", state["closed_challenges"])
        self.assertNotIn("c-03", state["abandoned_challenges"])

    def test_benchmark_start_local_blocks_when_active_exists(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["current_challenge"] = "c-03"
            self.pipeline._benchmark_state["active_containers"] = {
                "c-03": ["10.0.180.232:3000"],
            }

        with patch.object(self.pipeline, "_benchmark_platform_request") as request:
            result = self.pipeline._benchmark_start_local("c-06")

        self.assertIn("blocked_by_active", result)
        self.assertIn("c-03=>10.0.180.232:3000", result)
        request.assert_not_called()

    def test_benchmark_runtime_state_parses_scripted_list_status_lines(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        content = (
            "命令: curl -s --interface enp0s20f0u3u4 -H 'BENCHMARK_TOKEN: token' "
            "https://tsecbench.zc.tencent.com/openapi/v1/challenges | python3 -c 'print()'\n"
            "工作目录: /home/my/cyber/benchmark_test\n"
            "退出码: 0\n"
            "输出:\n"
            "b-01: completed=False, status=stopped, addr=[]\n"
            "c-03: completed=False, status=available, addr=['10.0.180.232:3000']\n"
            "c-07: completed=True, status=stopped, addr=[]\n"
        )

        self.pipeline._update_benchmark_runtime_state(content)

        state = self.pipeline._benchmark_state_snapshot()
        self.assertEqual(state["current_challenge"], "c-03")
        self.assertEqual(
            state["active_containers"],
            {"c-03": ["10.0.180.232:3000"]},
        )
        self.assertIn("c-07", state["completed_challenges"])

    def test_benchmark_readiness_waits_through_startup_flap(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        outputs: list[str] = []
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if len(calls) < 3:
                return subprocess.CompletedProcess(
                    cmd,
                    7,
                    stdout="",
                    stderr="curl: (7) Failed to connect to 10.0.180.232 port 3000",
                )
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout="HTTP/1.1 200 OK\n\n/apps",
                stderr="",
            )

        with (
            patch("cyber_agent.agent.pipeline.subprocess.run", side_effect=fake_run),
            patch("cyber_agent.agent.pipeline.time_mod.sleep") as sleep,
        ):
            body = self.pipeline._benchmark_wait_for_container_ready(
                "http://10.0.180.232:3000/",
                outputs,
            )

        self.assertIn("HTTP/1.1 200 OK", body)
        self.assertEqual(len(calls), 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertIn("readiness 3", "\n".join(outputs))

    def test_benchmark_fast_path_recovers_closed_easy_before_medium(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["last_challenges_snapshot"] = [
                {
                    "unique_code": "c-03",
                    "difficulty": "easy",
                    "level": 1,
                    "total_score": 100,
                    "is_completed": False,
                    "container_status": "stopped",
                    "container_addr": [],
                },
                {
                    "unique_code": "e1-01",
                    "difficulty": "medium",
                    "level": 1,
                    "total_score": 250,
                    "is_completed": False,
                    "container_status": "stopped",
                    "container_addr": [],
                },
            ]
            self.pipeline._benchmark_state["closed_challenges"] = {"c-03"}

        should_fast, reason = self.pipeline._benchmark_should_use_fast_path()

        self.assertTrue(should_fast)
        self.assertIn("恢复 fast path", reason)

    def test_benchmark_step2_refreshes_active_from_platform_when_state_empty(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True

        with (
            patch.object(
                self.pipeline,
                "_benchmark_list_challenges_local",
                return_value=[
                    {
                        "unique_code": "c-03",
                        "difficulty": "easy",
                        "is_completed": False,
                        "container_status": "available",
                        "container_addr": ["10.0.1.2:3000"],
                    }
                ],
            ),
            patch.object(
                self.pipeline,
                "_benchmark_probe_container_local",
                return_value="HTTP/1.1 200 OK\n<html>Dify</html>",
            ) as probe_local,
        ):
            result = self.pipeline._benchmark_deterministic_fast_step(
                "Benchmark fast step 2：只解当前已启动的 10.x 容器。",
                reason="deterministic_probe_submit_close",
            )

        self.assertIn("c-03 已获取有效响应", result)
        probe_local.assert_called_once_with("c-03", ["10.0.1.2:3000"])

    def test_benchmark_fast_step1_treats_running_snapshot_as_active(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True

        with (
            patch.object(
                self.pipeline,
                "_benchmark_list_challenges_local",
                return_value=[
                    {
                        "unique_code": "c-03",
                        "difficulty": "easy",
                        "is_completed": False,
                        "container_status": "running",
                        "container_addr": ["10.0.1.2:3000"],
                    },
                    {
                        "unique_code": "c-04",
                        "difficulty": "easy",
                        "is_completed": False,
                        "container_status": "stopped",
                        "container_addr": [],
                    },
                ],
            ),
            patch.object(self.pipeline, "_benchmark_start_local") as start_local,
        ):
            result = self.pipeline._benchmark_deterministic_fast_step(
                "Benchmark fast step 1：只做调度。",
                reason="deterministic_scheduler",
            )

        self.assertIn("继续当前 active 题 c-03", result)
        start_local.assert_not_called()

    def test_benchmark_step2_refreshes_running_active_from_platform(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True

        with (
            patch.object(
                self.pipeline,
                "_benchmark_list_challenges_local",
                return_value=[
                    {
                        "unique_code": "c-03",
                        "difficulty": "easy",
                        "is_completed": False,
                        "container_status": "running",
                        "container_addr": ["10.0.1.2:3000"],
                    }
                ],
            ),
            patch.object(
                self.pipeline,
                "_benchmark_probe_container_local",
                return_value="HTTP/1.1 200 OK\n<html>Dify</html>",
            ) as probe_local,
        ):
            result = self.pipeline._benchmark_deterministic_fast_step(
                "Benchmark fast step 2：只解当前已启动的 10.x 容器。",
                reason="deterministic_probe_submit_close",
            )

        self.assertIn("c-03 已获取有效响应", result)
        probe_local.assert_called_once_with("c-03", ["10.0.1.2:3000"])

    def test_benchmark_deterministic_step1_starts_medium_when_easy_exhausted(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["api_interface"] = "enp0s20f0u3u4"

        def fake_run(cmd, **kwargs):
            joined = " ".join(cmd)
            if "/openapi/v1/challenges/start?unique_code=e1-01" in joined:
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout='{"unique_code":"e1-01","container_addr":["10.0.1.9:8080"]}',
                    stderr="",
                )
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=(
                    "["
                    '{"unique_code":"xben-006-24","difficulty":"easy",'
                    '"level":1,"total_score":100,"is_completed":true,'
                    '"container_status":"stopped","container_addr":[]},'
                    '{"unique_code":"e1-01","difficulty":"medium",'
                    '"level":1,"total_score":250,"is_completed":false,'
                    '"container_status":"stopped","container_addr":[]}'
                    "]"
                ),
                stderr="",
            )

        with (
            patch.object(
                self.pipeline,
                "_benchmark_api_config_from_workspace",
                return_value=("https://tsecbench.zc.tencent.com", "token"),
            ),
            patch("cyber_agent.agent.pipeline.subprocess.run", side_effect=fake_run),
        ):
            result = self.pipeline._benchmark_deterministic_fast_step(
                "Benchmark fast step 1：只做调度。"
            )

        state = self.pipeline._benchmark_state_snapshot()
        self.assertIn("启动下一道 medium e1-01", result)
        self.assertEqual(state["current_challenge"], "e1-01")
        self.assertEqual(
            state["active_containers"],
            {"e1-01": ["10.0.1.9:8080"]},
        )

    def test_benchmark_deterministic_step1_closes_completed_active_before_start(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["api_interface"] = "enp0s20f0u3u4"

        calls: list[str] = []

        def fake_run(cmd, **kwargs):
            joined = " ".join(cmd)
            calls.append(joined)
            if "/openapi/v1/challenges/close?unique_code=xben-005-24" in joined:
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout='{"unique_code":"xben-005-24","closed":true}',
                    stderr="",
                )
            if "/openapi/v1/challenges/start?unique_code=xben-009-24" in joined:
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout=(
                        '{"unique_code":"xben-009-24",'
                        '"container_addr":["10.0.1.2:80"]}'
                    ),
                    stderr="",
                )
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=(
                    "["
                    '{"unique_code":"xben-005-24","difficulty":"easy",'
                    '"level":1,"total_score":200,"is_completed":true,'
                    '"container_status":"available","container_addr":["10.0.1.1:80"]},'
                    '{"unique_code":"xben-009-24","difficulty":"easy",'
                    '"level":1,"total_score":200,"is_completed":false,'
                    '"container_status":"stopped","container_addr":[]}'
                    "]"
                ),
                stderr="",
            )

        with (
            patch.object(
                self.pipeline,
                "_benchmark_api_config_from_workspace",
                return_value=("https://tsecbench.zc.tencent.com", "token"),
            ),
            patch("cyber_agent.agent.pipeline.subprocess.run", side_effect=fake_run),
        ):
            result = self.pipeline._benchmark_deterministic_fast_step(
                "Benchmark fast step 1：只做调度。"
            )

        self.assertIn("先关闭已完成 active 容器释放名额", result)
        self.assertIn("启动下一道 easy xben-009-24", result)
        self.assertTrue(
            any("/openapi/v1/challenges/close?unique_code=xben-005-24" in call for call in calls)
        )
        state = self.pipeline._benchmark_state_snapshot()
        self.assertIn("xben-005-24", state["closed_challenges"])
        self.assertEqual(
            state["active_containers"],
            {"xben-009-24": ["10.0.1.2:80"]},
        )

    def test_benchmark_deterministic_step1_retries_after_max_active_cleanup(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["api_interface"] = "enp0s20f0u3u4"

        start_attempts = 0
        list_attempts = 0

        def fake_run(cmd, **kwargs):
            nonlocal list_attempts, start_attempts
            joined = " ".join(cmd)
            if "/openapi/v1/challenges/close?unique_code=xben-005-24" in joined:
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout='{"unique_code":"xben-005-24","closed":true}',
                    stderr="",
                )
            if "/openapi/v1/challenges/start?unique_code=xben-009-24" in joined:
                start_attempts += 1
                if start_attempts == 1:
                    return subprocess.CompletedProcess(
                        cmd,
                        0,
                        stdout=(
                            '{"code":"invalid_state","message":"max active challenge '
                            'instances reached (3), please close an existing challenge '
                            'before starting a new one","detail":{}}'
                        ),
                        stderr="",
                    )
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout=(
                        '{"unique_code":"xben-009-24",'
                        '"container_addr":["10.0.1.2:80"]}'
                    ),
                    stderr="",
                )
            list_attempts += 1
            completed_status = "stopped" if list_attempts == 1 else "available"
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=(
                    "["
                    '{"unique_code":"xben-005-24","difficulty":"easy",'
                    '"level":1,"total_score":200,"is_completed":true,'
                    f'"container_status":"{completed_status}","container_addr":["10.0.1.1:80"]}},'
                    '{"unique_code":"xben-009-24","difficulty":"easy",'
                    '"level":1,"total_score":200,"is_completed":false,'
                    '"container_status":"stopped","container_addr":[]}'
                    "]"
                ),
                stderr="",
            )

        with (
            patch.object(
                self.pipeline,
                "_benchmark_api_config_from_workspace",
                return_value=("https://tsecbench.zc.tencent.com", "token"),
            ),
            patch("cyber_agent.agent.pipeline.subprocess.run", side_effect=fake_run),
        ):
            result = self.pipeline._benchmark_deterministic_fast_step(
                "Benchmark fast step 1：只做调度。"
            )

        self.assertIn("重试启动 xben-009-24", result)
        self.assertEqual(start_attempts, 2)
        state = self.pipeline._benchmark_state_snapshot()
        self.assertNotIn("xben-009-24", state["abandoned_challenges"])
        self.assertEqual(
            state["active_containers"],
            {"xben-009-24": ["10.0.1.2:80"]},
        )

    def test_benchmark_start_local_marks_resource_unavailable_abandoned(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["api_interface"] = "enp0s20f0u3u4"

        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"code":"resource_unavailable","message":"HTTP 502"}',
            stderr="",
        )
        with (
            patch.object(
                self.pipeline,
                "_benchmark_api_config_from_workspace",
                return_value=("https://tsecbench.zc.tencent.com", "token"),
            ),
            patch(
                "cyber_agent.agent.pipeline.subprocess.run",
                return_value=completed,
            ),
        ):
            result = self.pipeline._benchmark_start_local("xben-084-24")

        self.assertIn("resource_unavailable", result)
        state = self.pipeline._benchmark_state_snapshot()
        self.assertIn("xben-084-24", state["abandoned_challenges"])

    def test_benchmark_derive_probe_urls_adds_object_storage_paths(self):
        html = """
        <html><title>CloudAssets · S3 Static Host</title>
        <body>
        <p>本服务基于对象存储（S3 兼容），以 path-style URL 对外提供资产访问。</p>
        <a href="/company-assets/">/company-assets/</a>
        </body></html>
        """

        urls = self.pipeline._benchmark_derive_probe_urls(
            "http://10.0.180.232:8000/",
            html,
        )

        self.assertIn("http://10.0.180.232:8000/company-assets/", urls)
        self.assertIn("http://10.0.180.232:8000/company-assets/flag.txt", urls)
        self.assertIn("http://10.0.180.232:8000/company-assets/.env", urls)
        self.assertIn("http://10.0.180.232:8000/company-secrets/flag.txt", urls)

    def test_benchmark_external_probe_strategy_extends_paths_and_payloads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "benchmark-profiles.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "probe_paths": ["healthz"],
                        "text_path_prefixes": ["debug"],
                        "param_payload_profiles": [
                            {
                                "name_contains": ["token"],
                                "payloads": ["../../../../run/secrets/token"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            self.pipeline._runtime_context["benchmark_profiles_path"] = str(profile_path)

            paths = self.pipeline._benchmark_probe_paths()
            urls = self.pipeline._benchmark_derive_probe_urls(
                "http://10.0.1.2/",
                '<a href="/debug/status?token=1">debug</a>',
            )
            joined = "\n".join(urls)

        self.assertIn("robots.txt", paths)
        self.assertIn("healthz", paths)
        self.assertIn("http://10.0.1.2/debug/status?token=1", urls)
        self.assertIn("token=..%2F..%2F..%2F..%2Frun%2Fsecrets%2Ftoken", joined)

    def test_benchmark_derivation_learns_runtime_paths_and_params(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        html = """
        <form action="/exports/download?file=report.csv">
          <input name="artifact_id" value="42">
        </form>
        {"debug_url":"/internal/status","schema":{"properties":{"token":{"example":"demo"}}}}
        """

        with patch.object(self.pipeline, "_persist_benchmark_state") as persist:
            urls = self.pipeline._benchmark_derive_probe_urls(
                "http://10.0.180.232:8000/",
                html,
            )

        state = self.pipeline._benchmark_state_snapshot()
        self.assertIn("exports/download?file=report.csv", state["observed_probe_paths"])
        self.assertIn("internal/status", state["observed_probe_paths"])
        self.assertIn("artifact_id", state["observed_param_names"])
        self.assertIn("token", state["observed_param_names"])
        self.assertIn(
            "http://10.0.180.232:8000/internal/status",
            urls,
        )
        persist.assert_called()

    def test_benchmark_probe_paths_reuse_runtime_observations(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["observed_probe_paths"] = {
                "internal/status",
                "exports/download?file=report.csv",
            }
            self.pipeline._benchmark_state["observed_param_names"] = {"token"}

        paths = self.pipeline._benchmark_probe_paths()
        urls = self.pipeline._benchmark_derive_probe_urls(
            "http://10.0.180.232:8000/",
            "<html><title>next page</title></html>",
        )
        joined = "\n".join(urls)

        self.assertLess(paths.index("internal/status"), len(paths))
        self.assertIn("exports/download?file=report.csv", paths)
        self.assertIn("token=", joined)

    def test_benchmark_profile_can_disable_builtin_candidate_sections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "benchmark-profiles.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "disabled_builtin_sections": [
                            "probe_paths",
                            "payloads",
                            "flag_paths",
                            "lfi_base_paths",
                            "object_storage",
                            "raw_protocol_commands",
                            "telnet_credentials",
                        ],
                        "probe_paths": ["healthz"],
                        "flag_paths": ["/opt/app/flag.txt"],
                        "lfi_base_paths": ["../../../../opt/app/secret.txt"],
                        "object_storage_buckets": ["audit-bucket"],
                        "object_storage_keys": ["private/flag.json"],
                        "raw_protocol_commands": ["DUMP"],
                        "telnet_credentials": [
                            {"username": "operator", "password": "operator123"}
                        ],
                        "param_payload_profiles": [
                            {
                                "name_contains": ["token"],
                                "payloads": ["../../../../run/secrets/token"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            self.pipeline._runtime_context["benchmark_profiles_path"] = str(profile_path)

            paths = self.pipeline._benchmark_probe_paths()
            payloads = self.pipeline._benchmark_payloads_for_param("api_token")
            flag_paths = self.pipeline._benchmark_flag_paths()
            lfi_paths = self.pipeline._benchmark_lfi_base_paths()
            buckets = self.pipeline._benchmark_object_storage_buckets()
            keys = self.pipeline._benchmark_object_storage_keys()
            raw_commands = self.pipeline._benchmark_raw_protocol_commands()
            telnet_credentials = self.pipeline._benchmark_telnet_credentials()

        self.assertEqual(paths, ["healthz"])
        self.assertEqual(payloads, ["../../../../run/secrets/token"])
        self.assertEqual(flag_paths, ("/opt/app/flag.txt",))
        self.assertEqual(lfi_paths, ["../../../../opt/app/secret.txt"])
        self.assertEqual(buckets, ["audit-bucket"])
        self.assertEqual(keys, ("private/flag.json",))
        self.assertEqual(raw_commands, ("DUMP",))
        self.assertEqual(telnet_credentials, (("operator", "operator123"),))

    def test_benchmark_external_object_storage_strategy_extends_candidates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "benchmark-profiles.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "object_storage_buckets": ["audit-bucket"],
                        "object_storage_keys": ["private/flag.json"],
                    }
                ),
                encoding="utf-8",
            )
            self.pipeline._runtime_context["benchmark_profiles_path"] = str(profile_path)

            urls = self.pipeline._benchmark_derive_probe_urls(
                "http://10.0.180.232:8000/",
                "S3 path-style object storage endpoint",
            )

        self.assertIn("http://10.0.180.232:8000/company-secrets/flag.txt", urls)
        self.assertIn("http://10.0.180.232:8000/audit-bucket/private/flag.json", urls)

    def test_benchmark_probe_container_adapts_to_intermediate_links(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        requested_urls: list[str] = []

        def fake_run(cmd, **kwargs):
            url = cmd[-1]
            requested_urls.append(url)
            if url.endswith("/first"):
                stdout = (
                    "HTTP/1.1 200 OK\n\n"
                    '<html><a href="/second">second</a></html>'
                )
            else:
                stdout = "HTTP/1.1 404 Not Found\n\n"
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

        with (
            patch.object(
                self.pipeline,
                "_benchmark_wait_for_container_ready",
                return_value='<html><a href="/first">first</a></html>',
            ),
            patch.object(self.pipeline, "_benchmark_auto_submit_flags_from_tool_result"),
            patch("cyber_agent.agent.pipeline.subprocess.run", side_effect=fake_run),
        ):
            self.pipeline._benchmark_probe_container_local(
                "d-01",
                ["10.0.180.232:8000"],
            )

        self.assertIn("http://10.0.180.232:8000/first", requested_urls)
        self.assertIn("http://10.0.180.232:8000/second", requested_urls)

    def test_benchmark_probe_container_stops_when_fast_probe_budget_expires(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._runtime_context["execution_control_policy"] = {
            "fast_probe_seconds": 2,
            "max_probe_urls": 50,
        }
        self.pipeline._benchmark_profile_active = True
        requested_urls: list[str] = []

        def fake_run(cmd, **kwargs):
            requested_urls.append(cmd[-1])
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout="HTTP/1.1 404 Not Found\n\n",
                stderr="",
            )

        with (
            patch.object(
                self.pipeline,
                "_benchmark_wait_for_container_ready",
                return_value='<a href="/one">one</a><a href="/two">two</a>',
            ),
            patch.object(
                self.pipeline,
                "_benchmark_deadline_remaining",
                side_effect=[3.0, 0.5],
            ),
            patch.object(self.pipeline, "_benchmark_deadline_expired", return_value=True),
            patch.object(self.pipeline, "_benchmark_auto_submit_flags_from_tool_result"),
            patch("cyber_agent.agent.pipeline.subprocess.run", side_effect=fake_run),
        ):
            result = self.pipeline._benchmark_probe_container_local(
                "d-01",
                ["10.0.180.232:8000"],
            )

        self.assertEqual(len(requested_urls), 1)
        self.assertIn("probe budget exhausted", result)

    def test_benchmark_probe_container_prefers_https_for_tls_ports(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        requested_commands: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            requested_commands.append(list(cmd))
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout="HTTP/1.1 404 Not Found\n\n",
                stderr="",
            )

        with (
            patch.object(
                self.pipeline,
                "_benchmark_wait_for_container_ready",
                return_value="",
            ) as ready,
            patch.object(self.pipeline, "_benchmark_auto_submit_flags_from_tool_result"),
            patch("cyber_agent.agent.pipeline.subprocess.run", side_effect=fake_run),
        ):
            self.pipeline._benchmark_probe_container_local(
                "c-09",
                ["10.0.180.232:8443"],
            )

        ready.assert_called_once()
        self.assertEqual(ready.call_args.args[:2][0], "https://10.0.180.232:8443/")
        self.assertIsInstance(ready.call_args.args[:2][1], list)
        self.assertIn("deadline", ready.call_args.kwargs)
        requested_urls = [cmd[-1] for cmd in requested_commands]
        self.assertIn("https://10.0.180.232:8443/", requested_urls)
        self.assertIn("http://10.0.180.232:8443/", requested_urls)
        self.assertTrue(all("-k" in cmd for cmd in requested_commands))

    def test_benchmark_raw_text_protocol_probe_submits_leaked_flag(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True

        class FakeSocket:
            def __init__(self):
                self.sent: list[bytes] = []
                self.responses = [
                    b"200 responsd ready\n",
                    b"200 Commands:\n  SETBODY <text>\n  BUILD\n",
                    b"250 body set\n",
                    b"body_len=5 headers=0\nhdrtab:\\x00flag{raw-win}\\x00\nhello\n",
                    b"221 bye\n",
                ]

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def settimeout(self, _timeout):
                return None

            def recv(self, _size):
                if not self.responses:
                    raise TimeoutError()
                return self.responses.pop(0)

            def sendall(self, data):
                self.sent.append(data)

        fake_socket = FakeSocket()
        with (
            patch(
                "cyber_agent.agent.pipeline.socket_mod.create_connection",
                return_value=fake_socket,
            ),
            patch.object(self.pipeline, "_benchmark_auto_submit_flags_from_tool_result") as submit,
        ):
            output = self.pipeline._benchmark_probe_raw_text_protocol(
                "f1-04",
                "10.0.180.232:9013",
            )

        self.assertIn("flag{raw-win}", output)
        self.assertIn(b"BUILD\n", fake_socket.sent)
        submit.assert_called_once()

    def test_benchmark_external_raw_and_telnet_strategy_extends_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "benchmark-profiles.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "raw_protocol_commands": ["DUMP"],
                        "telnet_credentials": [
                            {"username": "operator", "password": "operator123"}
                        ],
                        "telnet_flag_command": "cat /opt/flag.txt",
                    }
                ),
                encoding="utf-8",
            )
            self.pipeline._runtime_context["benchmark_profiles_path"] = str(profile_path)

            raw_commands = self.pipeline._benchmark_raw_protocol_commands()
            telnet_credentials = self.pipeline._benchmark_telnet_credentials()
            flag_command = self.pipeline._benchmark_telnet_flag_command()

        self.assertIn("HELP", raw_commands)
        self.assertIn("DUMP", raw_commands)
        self.assertIn(("admin", "admin"), telnet_credentials)
        self.assertIn(("operator", "operator123"), telnet_credentials)
        self.assertEqual(flag_command, "cat /opt/flag.txt\n")

    def test_benchmark_external_flag_paths_extend_shared_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "benchmark-profiles.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "flag_paths": [
                            "/opt/service/flag.txt",
                            "workspace/flag",
                            "/tmp/flag;rm -rf /",
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.pipeline._runtime_context["benchmark_profiles_path"] = str(profile_path)

            flag_paths = self.pipeline._benchmark_flag_paths()
            flag_command = self.pipeline._benchmark_telnet_flag_command()

        self.assertIn("/challenge/flag.txt", flag_paths)
        self.assertIn("/opt/service/flag.txt", flag_paths)
        self.assertIn("/workspace/flag", flag_paths)
        self.assertNotIn("/tmp/flag;rm -rf /", flag_paths)
        self.assertIn("/opt/service/flag.txt", flag_command)
        self.assertNotIn("rm -rf", flag_command)

    def test_benchmark_langflow_probe_uses_external_flag_paths(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True

        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "benchmark-profiles.json"
            profile_path.write_text(
                json.dumps({"flag_paths": ["/opt/langflow/flag.txt"]}),
                encoding="utf-8",
            )
            self.pipeline._runtime_context["benchmark_profiles_path"] = str(profile_path)
            payloads: list[dict[str, object]] = []

            def fake_curl_json(*_args, **kwargs):
                payload = kwargs.get("payload")
                if isinstance(payload, dict):
                    payloads.append(payload)
                return subprocess.CompletedProcess([], 0, stdout="{}", stderr="")

            with (
                patch.object(
                    self.pipeline,
                    "_benchmark_curl_local",
                    return_value=subprocess.CompletedProcess([], 0, stdout="{}", stderr=""),
                ),
                patch.object(
                    self.pipeline,
                    "_benchmark_curl_json_local",
                    side_effect=fake_curl_json,
                ),
                patch.object(self.pipeline, "_benchmark_auto_submit_flags_from_tool_result"),
            ):
                output = self.pipeline._benchmark_probe_langflow_local(
                    "c-08",
                    "http://10.0.180.232:7860/",
                )

        serialized_payloads = json.dumps(payloads, ensure_ascii=False)
        self.assertIn("/opt/langflow/flag.txt", serialized_payloads)
        self.assertIn("/opt/langflow/flag.txt", output)

    def test_benchmark_deterministic_step2_auto_submits_and_closes_flag(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["api_interface"] = "enp0s20f0u3u4"
            self.pipeline._benchmark_state["current_challenge"] = "xben-009-24"
            self.pipeline._benchmark_state["active_containers"] = {
                "xben-009-24": ["10.0.1.2:80"],
            }

        def fake_run(cmd, **kwargs):
            joined = " ".join(cmd)
            if "/openapi/v1/challenges/submit" in joined:
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout='{"correct":true,"awarded":200,"cumulative_score":200}',
                    stderr="",
                )
            if "/openapi/v1/challenges/close?unique_code=xben-009-24" in joined:
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout='{"closed":true}',
                    stderr="",
                )
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout="HTTP/1.1 200 OK\n\nflag{deterministic-win}",
                stderr="",
            )

        with (
            patch.object(
                self.pipeline,
                "_benchmark_api_config_from_workspace",
                return_value=("https://tsecbench.zc.tencent.com", "token"),
            ),
            patch("cyber_agent.agent.pipeline.subprocess.run", side_effect=fake_run),
        ):
            result = self.pipeline._benchmark_deterministic_fast_step(
                "Benchmark fast step 2：只解当前已启动的 10.x 容器。",
                reason="Connection error.",
            )

        state = self.pipeline._benchmark_state_snapshot()
        self.assertIn("已提交成功并关闭", result)
        self.assertIn("xben-009-24", state["completed_challenges"])
        self.assertIn("xben-009-24", state["closed_challenges"])
        self.assertEqual(state["active_containers"], {})

    def test_benchmark_deterministic_step2_keeps_reachable_easy_for_reasoning(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["api_interface"] = "enp0s20f0u3u4"
            self.pipeline._benchmark_state["current_challenge"] = "c-03"
            self.pipeline._benchmark_state["active_containers"] = {
                "c-03": ["10.0.180.232:3000"],
            }

        def fake_probe(code, _addrs):
            self.pipeline._benchmark_set_service_fingerprint(code, "dify")
            self.pipeline._benchmark_mark_reasoning_needed(
                code,
                "Dify/Next.js 前端可达但后端疑似绑定 localhost，需要 Dify 专项深挖",
            )
            return (
                "## http://10.0.180.232:3000/\n"
                "HTTP/1.1 307 Temporary Redirect\n"
                "location: /apps\n"
                "X-Powered-By: Next.js\n\n"
                '<html><body data-api-prefix="http://127.0.0.1:5001/console/api" '
                'data-public-edition="SELF_HOSTED">Dify</body></html>'
            )

        with (
            patch.object(
                self.pipeline,
                "_benchmark_probe_container_local",
                side_effect=fake_probe,
            ),
            patch.object(self.pipeline, "_benchmark_close_local") as close_local,
        ):
            result = self.pipeline._benchmark_deterministic_fast_step(
                "Benchmark fast step 2：只解当前已启动的 10.x 容器。",
                reason="deterministic_probe_submit_close",
            )

        state = self.pipeline._benchmark_state_snapshot()
        self.assertIn("保留 active", result)
        self.assertIn("c-03", state["reasoning_challenges"])
        self.assertEqual(state["service_fingerprints"]["c-03"], "dify")
        self.assertEqual(state["active_containers"], {"c-03": ["10.0.180.232:3000"]})
        close_local.assert_not_called()

    def test_benchmark_step2_respects_service_profile_abandoned_state(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["current_challenge"] = "c-08"
            self.pipeline._benchmark_state["active_containers"] = {
                "c-08": ["10.0.180.232:7860"],
            }

        def fake_probe(code, _addrs):
            self.pipeline._benchmark_set_service_fingerprint(code, "langflow")
            self.pipeline._benchmark_mark_abandoned(
                code,
                "Langflow bounded validate/code 探测未发现可提交 flag",
            )
            return "## langflow-probe\nLangflow reachable but no flag"

        with (
            patch.object(
                self.pipeline,
                "_benchmark_probe_container_local",
                side_effect=fake_probe,
            ),
            patch.object(
                self.pipeline,
                "_benchmark_close_local",
                return_value='{"unique_code":"c-08","closed":true}',
            ) as close_local,
        ):
            result = self.pipeline._benchmark_deterministic_fast_step(
                "Benchmark fast step 2：只解当前已启动的 10.x 容器。",
                reason="deterministic_probe_submit_close",
            )

        state = self.pipeline._benchmark_state_snapshot()
        self.assertIn("bounded service profile 判定低收益", result)
        self.assertIn("服务指纹 langflow", result)
        self.assertIn("c-08", state["abandoned_challenges"])
        self.assertNotIn("c-08", state["reasoning_challenges"])
        close_local.assert_called_once_with("c-08")

    def test_benchmark_hugegraph_probe_uses_service_specific_path(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["current_challenge"] = "c-06"
            self.pipeline._benchmark_state["active_containers"] = {
                "c-06": ["10.0.180.232:8080"],
            }

        with (
            patch.object(
                self.pipeline,
                "_benchmark_wait_for_container_ready",
                return_value=(
                    'HTTP/1.1 200 OK\n\n{"service":"hugegraph",'
                    '"version":"1.2.0","apis":["arthas","gremlin"]}'
                ),
            ),
            patch.object(
                self.pipeline,
                "_benchmark_curl_local",
                return_value=subprocess.CompletedProcess(
                    [],
                    0,
                    stdout='HTTP/1.1 200 OK\n\n{"versions":{"core":"1.2.0"}}',
                    stderr="",
                ),
            ),
            patch.object(
                self.pipeline,
                "_benchmark_curl_json_local",
                return_value=subprocess.CompletedProcess(
                    [],
                    0,
                    stdout='HTTP/1.1 200 OK\n\n{"arthas.telnetPort":"8562"}',
                    stderr="",
                ),
            ),
            patch.object(self.pipeline, "_benchmark_probe_tcp_port", return_value=True),
            patch.object(
                self.pipeline,
                "_benchmark_probe_jdwp_local",
                return_value="## jdwp-probe 10.0.180.232:5005\nJDWP port reachable",
            ),
            patch.object(self.pipeline, "_benchmark_probe_common_webapp_flows") as webapp,
            patch("cyber_agent.agent.pipeline.subprocess.run") as run,
        ):
            run.return_value = subprocess.CompletedProcess(
                [],
                0,
                stdout=(
                    'HTTP/1.1 200 OK\n\n{"service":"hugegraph",'
                    '"version":"1.2.0","apis":["arthas","gremlin"]}'
                ),
                stderr="",
            )
            result = self.pipeline._benchmark_probe_container_local(
                "c-06",
                ["10.0.180.232:8080"],
            )

        state = self.pipeline._benchmark_state_snapshot()
        self.assertIn("hugegraph-probe", result)
        self.assertIn("jdwp-probe", result)
        self.assertIn("c-06", state["reasoning_challenges"])
        self.assertEqual(state["service_fingerprints"]["c-06"], "hugegraph")
        webapp.assert_not_called()

    def test_benchmark_probe_runs_raw_protocol_when_http_fails_but_tcp_connects(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True

        with (
            patch.object(self.pipeline, "_benchmark_probe_paths", return_value=[""]),
            patch.object(self.pipeline, "_benchmark_wait_for_container_ready", return_value=""),
            patch.object(
                self.pipeline,
                "_benchmark_connectivity_diagnostics",
                return_value=(
                    "## connectivity-diagnostics 10.0.180.232:9000\n"
                    "python_socket_tcp_probe: tcp_connect_ok"
                ),
            ) as diagnostics,
            patch.object(
                self.pipeline,
                "_benchmark_probe_raw_text_protocol",
                return_value=(
                    "## raw-text-protocol 10.0.180.232:9000\n"
                    "INFO\r\nservice ready"
                ),
            ) as raw_probe,
            patch.object(self.pipeline, "_benchmark_probe_common_webapp_flows", return_value=""),
            patch("cyber_agent.agent.pipeline.subprocess.run") as run,
        ):
            run.return_value = subprocess.CompletedProcess(
                [],
                7,
                stdout="",
                stderr=(
                    "curl: (7) Failed to connect to 10.0.180.232 port 9000: "
                    "Could not connect to server"
                ),
            )
            result = self.pipeline._benchmark_probe_container_local(
                "tcp-01",
                ["10.0.180.232:9000"],
            )

        self.assertIn("connectivity-diagnostics", result)
        self.assertIn("raw-text-protocol", result)
        diagnostics.assert_called_once_with("10.0.180.232:9000")
        raw_probe.assert_called_once_with("tcp-01", "10.0.180.232:9000")

    def test_benchmark_service_probe_profiles_drive_probe_dispatch(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        profiles = self.pipeline._benchmark_service_probe_profiles()

        self.assertEqual(
            [profile["fingerprint"] for profile in profiles],
            ["hugegraph", "dify", "langflow"],
        )
        with patch.object(
            self.pipeline,
            "_benchmark_probe_dify_local",
            return_value="dify bounded probe",
        ) as dify_probe:
            matched, outputs = self.pipeline._benchmark_probe_matching_service_local(
                "c-03",
                "http://10.0.180.232:3000/",
                'HTTP/1.1 200 OK\nx-powered-by: Next.js\n'
                '<html data-api-prefix="http://127.0.0.1:5001/console/api">'
                "Dify SELF_HOSTED</html>",
            )

        state = self.pipeline._benchmark_state_snapshot()
        self.assertTrue(matched)
        self.assertEqual(outputs, ["dify bounded probe"])
        self.assertEqual(state["service_fingerprints"]["c-03"], "dify")
        self.assertIn("c-03", state["reasoning_challenges"])
        dify_probe.assert_called_once()

    def test_benchmark_service_profile_matching_is_data_driven(self):
        self.assertTrue(
            self.pipeline._benchmark_text_matches_profile(
                "alpha service banner beta",
                {"match_all": ("alpha",), "match_any": ("beta", "gamma")},
            )
        )
        self.assertTrue(
            self.pipeline._benchmark_text_matches_profile(
                "contains gremlin and arthas markers",
                {"match_any_all": (("gremlin", "arthas"),)},
            )
        )
        self.assertFalse(
            self.pipeline._benchmark_text_matches_profile(
                "alpha service banner",
                {"match_all": ("alpha",), "match_any": ("missing",)},
            )
        )

    def test_benchmark_external_service_profile_extends_matching(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "benchmark-profiles.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "service_probe_profiles": [
                            {
                                "fingerprint": "customsvc",
                                "match_any": ["customsvc banner"],
                                "unresolved": "reasoning",
                                "reason": "custom service needs generic handoff",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.pipeline._runtime_context["benchmark_profiles_path"] = str(profile_path)

            profile = self.pipeline._benchmark_matching_service_probe_profile(
                "HTTP/1.1 200 OK\n\nCustomSvc banner"
            )
            outputs = self.pipeline._benchmark_run_service_probe_profile(
                "custom-01",
                "http://10.0.1.2:8080/",
                "CustomSvc banner",
                profile,
            )

        state = self.pipeline._benchmark_state_snapshot()
        self.assertIsNotNone(profile)
        self.assertEqual(profile["fingerprint"], "customsvc")
        self.assertEqual(outputs, [])
        self.assertEqual(state["service_fingerprints"]["custom-01"], "customsvc")
        self.assertIn("custom-01", state["reasoning_challenges"])

    def test_benchmark_external_service_profile_probe_paths_are_executed(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "benchmark-profiles.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "service_probe_profiles": [
                            {
                                "fingerprint": "customsvc",
                                "match_any": ["customsvc banner"],
                                "probe_paths": ["api/status", "/debug/export"],
                                "unresolved": "reasoning",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.pipeline._runtime_context["benchmark_profiles_path"] = str(profile_path)
            profile = self.pipeline._benchmark_matching_service_probe_profile(
                "CustomSvc banner"
            )

            with patch.object(
                self.pipeline,
                "_benchmark_curl_local",
                return_value=subprocess.CompletedProcess(
                    [],
                    0,
                    stdout="HTTP/1.1 200 OK\n\nstatus=ok",
                    stderr="",
                ),
            ) as curl_mock:
                outputs = self.pipeline._benchmark_run_service_probe_profile(
                    "custom-01",
                    "http://10.0.1.2:8080/",
                    "CustomSvc banner",
                    profile,
                )

        joined = "\n".join(outputs)
        self.assertIsNotNone(profile)
        self.assertIn("customsvc-profile-probe", joined)
        self.assertIn("GET /api/status", joined)
        self.assertIn("GET /debug/export", joined)
        self.assertEqual(curl_mock.call_count, 2)

    def test_benchmark_external_service_profile_probe_requests_are_executed(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "benchmark-profiles.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "service_probe_profiles": [
                            {
                                "fingerprint": "customsvc",
                                "match_any": ["customsvc banner"],
                                "probe_requests": [
                                    {
                                        "label": "rpc export",
                                        "method": "POST",
                                        "path": "rpc/export",
                                        "headers": {
                                            "X-Custom-Trace": "1",
                                            "Authorization": "blocked",
                                        },
                                        "json": {"method": "export", "limit": 1},
                                    },
                                    {
                                        "method": "POST",
                                        "path": "/debug/login",
                                        "data": {"username": "demo", "password": "demo"},
                                    },
                                ],
                                "unresolved": "reasoning",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.pipeline._runtime_context["benchmark_profiles_path"] = str(profile_path)
            profile = self.pipeline._benchmark_matching_service_probe_profile(
                "CustomSvc banner"
            )
            commands: list[list[str]] = []

            def fake_run(cmd, **_kwargs):
                commands.append(list(cmd))
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout="HTTP/1.1 200 OK\n\ncustom-ok",
                    stderr="",
                )

            with (
                patch("cyber_agent.agent.pipeline.subprocess.run", side_effect=fake_run),
                patch.object(self.pipeline, "_benchmark_auto_submit_flags_from_tool_result"),
            ):
                outputs = self.pipeline._benchmark_run_service_probe_profile(
                    "custom-01",
                    "http://10.0.1.2:8080/",
                    "CustomSvc banner",
                    profile,
                )

        joined = "\n".join(outputs)
        flattened = "\n".join(" ".join(cmd) for cmd in commands)
        self.assertIsNotNone(profile)
        self.assertIn("customsvc-profile-requests", joined)
        self.assertIn("rpc export", joined)
        self.assertIn("POST /debug/login", joined)
        self.assertIn("rpc/export", flattened)
        self.assertIn('"method":"export"', flattened)
        self.assertIn("X-Custom-Trace: 1", flattened)
        self.assertNotIn("Authorization: blocked", flattened)
        self.assertIn("username=demo", flattened)
        self.assertEqual(len(commands), 2)

    def test_benchmark_external_service_profile_tcp_ports_are_executed(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "benchmark-profiles.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "service_probe_profiles": [
                            {
                                "fingerprint": "customsvc",
                                "match_any": ["customsvc banner"],
                                "tcp_ports": [
                                    {"port": 5005, "label": "jdwp candidate"},
                                    8562,
                                    {"port": 70000, "label": "invalid"},
                                ],
                                "unresolved": "reasoning",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.pipeline._runtime_context["benchmark_profiles_path"] = str(profile_path)
            profile = self.pipeline._benchmark_matching_service_probe_profile(
                "CustomSvc banner"
            )

            with patch.object(
                self.pipeline,
                "_benchmark_probe_tcp_port",
                side_effect=lambda _host, port: port == 5005,
            ) as tcp_probe:
                outputs = self.pipeline._benchmark_run_service_probe_profile(
                    "custom-01",
                    "http://10.0.1.2:8080/",
                    "CustomSvc banner",
                    profile,
                )

        joined = "\n".join(outputs)
        self.assertIsNotNone(profile)
        self.assertIn("customsvc-tcp-probe", joined)
        self.assertIn("jdwp candidate 10.0.1.2:5005", joined)
        self.assertIn("reachable=True", joined)
        self.assertIn("tcp/8562 10.0.1.2:8562", joined)
        self.assertNotIn("70000", joined)
        self.assertEqual(tcp_probe.call_count, 2)

    def test_benchmark_external_service_profile_can_extend_builtin_by_fingerprint(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "benchmark-profiles.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "service_probe_profiles": [
                            {
                                "fingerprint": "langflow",
                                "probe_paths": ["api/v1/custom-health"],
                                "tcp_ports": [{"port": 7860, "label": "langflow ui"}],
                                "handoff_steps": [
                                    "Benchmark langflow custom step：check {current} at {addr_text}"
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.pipeline._runtime_context["benchmark_profiles_path"] = str(profile_path)

            profile = self.pipeline._benchmark_matching_service_probe_profile(
                "HTTP/1.1 200 OK\nServer: uvicorn\n<title>Langflow</title>"
            )

        self.assertIsNotNone(profile)
        self.assertEqual(profile["fingerprint"], "langflow")
        self.assertIn("/api/v1/validate/code", profile["match_any"])
        self.assertIn("api/v1/custom-health", profile["probe_paths"])
        self.assertEqual(profile["tcp_ports"][0]["port"], 7860)
        self.assertIn("probe", profile)
        self.assertNotIn("handoff_steps", profile)

    def test_benchmark_profile_can_disable_builtin_service_fingerprints(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "benchmark-profiles.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "disabled_builtin_fingerprints": ["dify", "langflow"],
                        "service_probe_profiles": [
                            {
                                "fingerprint": "langflow",
                                "match_any": ["custom-flow"],
                                "probe_paths": ["api/custom-health"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            self.pipeline._runtime_context["benchmark_profiles_path"] = str(profile_path)

            probe_profiles = self.pipeline._benchmark_service_probe_profiles()
            action_profiles = self.pipeline._benchmark_service_action_profiles()
            handoff_profiles = self.pipeline._benchmark_service_handoff_profiles()
            langflow = next(
                profile
                for profile in probe_profiles
                if profile["fingerprint"] == "langflow"
            )

        self.assertNotIn("dify", [profile["fingerprint"] for profile in probe_profiles])
        self.assertNotIn("dify", action_profiles)
        self.assertNotIn("dify", handoff_profiles)
        self.assertEqual(langflow["match_any"], ("custom-flow",))
        self.assertEqual(langflow["probe_paths"], ("api/custom-health",))
        self.assertNotIn("/api/v1/validate/code", langflow.get("match_any", ()))

    def test_benchmark_external_webapp_profile_extends_builtin_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "benchmark-profiles.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "webapp_flow_profiles": [
                            {
                                "name": "form-login-and-file-download",
                                "credentials": [
                                    {"username": "auditor", "password": "auditor123"}
                                ],
                                "handoff_paths": ["api/audit/export"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.pipeline._runtime_context["benchmark_profiles_path"] = str(profile_path)

            profile = self.pipeline._benchmark_matching_webapp_flow_profile(
                '<form action="/login"><input name="password"></form>'.lower()
            )

        self.assertIsNotNone(profile)
        self.assertEqual(profile["name"], "form-login-and-file-download")
        self.assertIn(("admin", "admin123"), profile["credentials"])
        self.assertIn(("auditor", "auditor123"), profile["credentials"])
        self.assertIn("dashboard.php", profile["authenticated_paths"])
        self.assertIn("api/audit/export", profile["handoff_paths"])

    def test_benchmark_handoff_followup_uses_custom_service_probe_profile(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        probe = MagicMock(return_value="custom profile probe")
        curl_result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="HTTP/1.1 200 OK\n\nCustomSvc banner",
            stderr="",
        )

        with (
            patch.object(
                self.pipeline,
                "_benchmark_service_probe_profiles",
                return_value=[
                    {
                        "fingerprint": "customsvc",
                        "match_any": ("customsvc banner",),
                        "probe": probe,
                        "unresolved": "reasoning",
                        "reason": "custom service needs profile handoff",
                    }
                ],
            ),
            patch.object(
                self.pipeline,
                "_benchmark_curl_local",
                return_value=curl_result,
            ),
        ):
            result = self.pipeline._benchmark_probe_handoff_followup_local(
                "custom-01",
                ["10.0.1.2:8080"],
            )

        self.assertIn("custom profile probe", result)
        probe.assert_called_once_with(
            "custom-01",
            "http://10.0.1.2:8080/",
            curl_result.stdout,
        )
        state = self.pipeline._benchmark_state_snapshot()
        self.assertEqual(state["service_fingerprints"]["custom-01"], "customsvc")
        self.assertIn("custom-01", state["reasoning_challenges"])

    def test_benchmark_telnet_port_uses_bounded_login_probe(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["current_challenge"] = "c-07"
            self.pipeline._benchmark_state["active_containers"] = {
                "c-07": ["10.0.180.232:23"],
            }

        with patch.object(
            self.pipeline,
            "_benchmark_probe_telnet_login_local",
            return_value="## telnet-login 10.0.180.232:23\nlogin prompt no flag",
        ) as telnet_probe:
            result = self.pipeline._benchmark_probe_container_local(
                "c-07",
                ["10.0.180.232:23"],
            )

        state = self.pipeline._benchmark_state_snapshot()
        self.assertIn("telnet-login", result)
        self.assertEqual(state["service_fingerprints"]["c-07"], "telnet")
        self.assertIn("c-07", state["abandoned_challenges"])
        telnet_probe.assert_called_once_with("c-07", "10.0.180.232:23")

    def test_benchmark_telnet_negotiation_replies_dont_wont(self):
        plain, reply = self.pipeline._benchmark_telnet_plain_and_reply(
            bytes([255, 253, 24, 255, 251, 1]) + b"login: "
        )

        self.assertEqual(plain, "login: ")
        self.assertEqual(reply, bytes([255, 252, 24, 255, 254, 1]))

    def test_benchmark_langflow_probe_uses_validate_code(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["current_challenge"] = "c-08"
            self.pipeline._benchmark_state["active_containers"] = {
                "c-08": ["10.0.180.232:7860"],
            }

        def fake_json(url, **kwargs):
            if "validate/code" in url:
                return subprocess.CompletedProcess(
                    [],
                    0,
                    stdout=(
                        '{"imports":{"errors":[]},"function":{"errors":'
                        '["flag{langflow-owned}"]}}'
                    ),
                    stderr="",
                )
            return subprocess.CompletedProcess([], 0, stdout="{}", stderr="")

        submit_result = subprocess.CompletedProcess(
            [],
            0,
            stdout='{"correct":true,"awarded":100,"cumulative_score":100}',
            stderr="",
        )

        with (
            patch.object(
                self.pipeline,
                "_benchmark_curl_local",
                return_value=subprocess.CompletedProcess(
                    [],
                    0,
                    stdout='HTTP/1.1 200 OK\n\n{"title":"Langflow"}',
                    stderr="",
                ),
            ),
            patch.object(
                self.pipeline,
                "_benchmark_curl_json_local",
                side_effect=fake_json,
            ) as curl_json,
            patch.object(
                self.pipeline,
                "_benchmark_api_config_from_workspace",
                return_value=("https://tsecbench.zc.tencent.com", "token"),
            ),
            patch(
                "cyber_agent.agent.pipeline.subprocess.run",
                return_value=submit_result,
            ),
        ):
            result = self.pipeline._benchmark_probe_langflow_local(
                "c-08",
                "http://10.0.180.232:7860/",
            )

        state = self.pipeline._benchmark_state_snapshot()
        self.assertIn("langflow-probe", result)
        self.assertIn("validate/code", result)
        self.assertIn("c-08", state["completed_challenges"])
        self.assertTrue(any("validate/code" in call.args[0] for call in curl_json.call_args_list))

    def test_benchmark_deterministic_step2_keeps_unreachable_active_for_retry(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["api_interface"] = "enp0s20f0u3u4"
            self.pipeline._benchmark_state["current_challenge"] = "d-01"
            self.pipeline._benchmark_state["active_containers"] = {
                "d-01": ["10.0.180.232:8000"],
            }

        with (
            patch.object(
                self.pipeline,
                "_benchmark_probe_container_local",
                return_value=(
                    "## readiness 1 http://10.0.180.232:8000/\n"
                    "curl: (7) Failed to connect to 10.0.180.232 port 8000: "
                    "Could not connect to server"
                ),
            ),
            patch.object(self.pipeline, "_benchmark_close_local") as close_local,
        ):
            result = self.pipeline._benchmark_deterministic_fast_step(
                "Benchmark fast step 2：只解当前已启动的 10.x 容器。",
                reason="deterministic_probe_submit_close",
            )

        state = self.pipeline._benchmark_state_snapshot()
        self.assertIn("暂不可达", result)
        self.assertEqual(state["probe_unreachable_streaks"], {"d-01": 1})
        self.assertEqual(state["active_containers"], {"d-01": ["10.0.180.232:8000"]})
        close_local.assert_not_called()

    def test_benchmark_refresh_active_from_platform_persists_state(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True

        with (
            patch.object(
                self.pipeline,
                "_benchmark_list_challenges_local",
                return_value=[
                    {
                        "unique_code": "d-01",
                        "difficulty": "easy",
                        "is_completed": False,
                        "container_status": "available",
                        "container_addr": ["10.0.180.232:8000"],
                    }
                ],
            ),
            patch.object(self.pipeline, "_persist_benchmark_state"),
        ):
            code, addrs = self.pipeline._benchmark_refresh_active_challenge_from_platform()

        state = self.pipeline._benchmark_state_snapshot()
        self.assertEqual(code, "d-01")
        self.assertEqual(addrs, ["10.0.180.232:8000"])
        self.assertEqual(state["current_challenge"], "d-01")
        self.assertEqual(state["active_containers"], {"d-01": ["10.0.180.232:8000"]})

    def test_benchmark_tcp_connect_marker_prevents_unreachable_close(self):
        probe = (
            "## http://10.0.180.232:8000/\n"
            "curl: (7) Failed to connect to 10.0.180.232 port 8000\n"
            "## connectivity-diagnostics 10.0.180.232:8000\n"
            "tcp_connect_ok\n"
        )

        self.assertFalse(FourPillarPipeline._benchmark_probe_looks_unreachable(probe))

    def test_benchmark_deterministic_step2_closes_after_unreachable_retry_limit(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["api_interface"] = "enp0s20f0u3u4"
            self.pipeline._benchmark_state["current_challenge"] = "d-01"
            self.pipeline._benchmark_state["active_containers"] = {
                "d-01": ["10.0.180.232:8000"],
            }
            self.pipeline._benchmark_state["probe_unreachable_streaks"] = {"d-01": 1}

        with (
            patch.object(
                self.pipeline,
                "_benchmark_probe_container_local",
                return_value=(
                    "## readiness 1 http://10.0.180.232:8000/\n"
                    "curl: (7) Failed to connect to 10.0.180.232 port 8000: "
                    "Could not connect to server"
                ),
            ),
            patch.object(
                self.pipeline,
                "_benchmark_close_local",
                return_value='{"unique_code":"d-01","closed":true}',
            ) as close_local,
        ):
            result = self.pipeline._benchmark_deterministic_fast_step(
                "Benchmark fast step 2：只解当前已启动的 10.x 容器。",
                reason="deterministic_probe_submit_close",
            )

        self.assertIn("连续 2 次不可达", result)
        close_local.assert_called_once_with("d-01")

    def test_benchmark_derives_payload_urls_from_html(self):
        html = (
            '<form action="/resource">'
            '<input name="filename">'
            '<input id="url">'
            '</form>'
            '<a href="/view?id=1">view</a>'
        )

        urls = self.pipeline._benchmark_derive_probe_urls(
            "http://10.0.1.2:80/",
            html,
        )
        joined = "\n".join(urls)

        self.assertIn("http://10.0.1.2:80/resource", urls)
        self.assertIn("filename=..%2Fflag", joined)
        self.assertIn("url=file%3A%2F%2F%2Fflag", joined)
        self.assertIn("id=1+OR+1%3D1", joined)

    def test_benchmark_derives_paths_from_response_path_keys(self):
        body = json.dumps(
            {
                "debug_url": "/debug/export",
                "config_path": "api/config",
                "download_url": "http://10.0.1.2:80/files/report",
                "external_url": "http://example.com/secret",
            }
        )

        urls = self.pipeline._benchmark_derive_probe_urls(
            "http://10.0.1.2:80/",
            body,
        )

        self.assertIn("http://10.0.1.2:80/debug/export", urls)
        self.assertIn("http://10.0.1.2:80/api/config", urls)
        self.assertIn("http://10.0.1.2:80/files/report", urls)
        self.assertNotIn("http://example.com/secret", urls)

    def test_benchmark_external_response_path_keys_extend_derivation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "benchmark-profiles.json"
            profile_path.write_text(
                json.dumps({"response_path_keys": ["artifact_endpoint"]}),
                encoding="utf-8",
            )
            self.pipeline._runtime_context["benchmark_profiles_path"] = str(profile_path)

            urls = self.pipeline._benchmark_derive_probe_urls(
                "http://10.0.1.2:80/",
                '{"artifact_endpoint":"artifacts/latest/flag.txt"}',
            )

        self.assertIn("http://10.0.1.2:80/artifacts/latest/flag.txt", urls)

    def test_benchmark_derives_payloads_from_openapi_schema_names(self):
        body = json.dumps(
            {
                "openapi": "3.0.0",
                "paths": {
                    "/download": {
                        "get": {
                            "parameters": [
                                {"name": "filename", "in": "query", "example": "report.pdf"},
                                {"name": "redirect_url", "in": "query"},
                            ]
                        }
                    }
                },
                "components": {
                    "schemas": {
                        "ExportRequest": {
                            "type": "object",
                            "properties": {
                                "template": {"type": "string"},
                                "api_token": {"type": "string"},
                                "mode": {
                                    "type": "string",
                                    "enum": ["public", "private"],
                                    "default": "public",
                                },
                            },
                        }
                    }
                },
            }
        )

        urls = self.pipeline._benchmark_derive_probe_urls(
            "http://10.0.1.2:80/",
            body,
        )
        joined = "\n".join(urls)

        self.assertIn("filename=..%2Fflag", joined)
        self.assertIn("filename=report.pdf", joined)
        self.assertIn("redirect_url=file%3A%2F%2F%2Fflag", joined)
        self.assertIn("template=..%2Fflag", joined)
        self.assertIn("api_token=%7B%7B7%2A7%7D%7D", joined)
        self.assertIn("mode=public", joined)
        self.assertIn("mode=private", joined)

    def test_benchmark_object_storage_derives_quoted_bucket_and_xml_keys(self):
        body = (
            'Internal note: migrated to the "secret-data" bucket. '
            "<ListBucketResult><Name>secret-data</Name>"
            "<Contents><Key>flag.txt</Key></Contents>"
            "<Contents><Key>backups/config-backup.yaml</Key></Contents>"
            "</ListBucketResult>"
        )

        urls = self.pipeline._benchmark_derive_probe_urls(
            "http://10.0.180.232:8000/",
            body,
        )

        self.assertIn("http://10.0.180.232:8000/secret-data/", urls)
        self.assertIn("http://10.0.180.232:8000/secret-data/flag.txt", urls)
        self.assertIn(
            "http://10.0.180.232:8000/secret-data/backups/config-backup.yaml",
            urls,
        )

    def test_benchmark_derives_lambda_function_config_urls(self):
        root = (
            "<code>GET /api/functions</code>"
            "<code>GET /api/functions/&lt;name&gt;/config</code>"
        )
        urls = self.pipeline._benchmark_derive_probe_urls(
            "http://10.0.180.232:8000/",
            root,
        )
        self.assertIn("http://10.0.180.232:8000/api/functions", urls)

        listing = '{"functions":[{"name":"order-service"},{"name":"data-processor"}]}'
        urls = self.pipeline._benchmark_derive_probe_urls(
            "http://10.0.180.232:8000/",
            listing,
        )
        self.assertIn(
            "http://10.0.180.232:8000/api/functions/order-service/config",
            urls,
        )
        self.assertIn(
            "http://10.0.180.232:8000/api/functions/data-processor/config",
            urls,
        )

    def test_benchmark_extracts_demo_credentials_for_stateful_probe(self):
        creds = self.pipeline._benchmark_extract_demo_credentials(
            "<p>测试账号: employee / employee123</p>"
            "$USERS = ['admin' => ['password' => 'admin123']];"
        )

        self.assertIn(("employee", "employee123"), creds)
        self.assertIn(("admin", "admin123"), creds)

    def test_benchmark_common_webapp_flow_tries_default_credentials(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        probe = (
            "HTTP/1.1 200 OK\n"
            "<html><title>登录 - 企业合同审批系统</title>"
            "<form action=\"/login.php\"><input name=\"username\">"
            "<input name=\"password\"></form></html>"
        )

        login_response = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="HTTP/1.1 302 Found\nLocation: /dashboard.php\n",
            stderr="",
        )
        with (
            patch.object(
                self.pipeline,
                "_benchmark_curl_local",
                return_value=login_response,
            ) as curl_mock,
            patch.object(
                self.pipeline,
                "_benchmark_probe_authenticated_webapp",
                return_value="auth-ok",
            ) as auth_mock,
        ):
            result = self.pipeline._benchmark_probe_common_webapp_flows(
                "web-login-generic",
                "http://10.0.180.232/",
                probe,
            )

        self.assertIn("login admin:admin123", result)
        self.assertIn("auth-ok", result)
        curl_mock.assert_called()
        first_login = curl_mock.call_args_list[0]
        self.assertEqual(first_login.kwargs["data"]["username"], "admin")
        self.assertEqual(first_login.kwargs["data"]["password"], "admin123")
        auth_mock.assert_called()

    def test_benchmark_webapp_flow_profile_matches_form_without_challenge_code(self):
        profile = self.pipeline._benchmark_matching_webapp_flow_profile(
            '<form action="/login"><input name="password"></form>'.lower()
        )

        self.assertIsNotNone(profile)
        self.assertEqual(profile["name"], "form-login-and-file-download")

    def test_benchmark_external_webapp_profile_extends_matching(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "benchmark-profiles.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "webapp_flow_profiles": [
                            {
                                "name": "custom-portal",
                                "indicators": ["custom-login-widget"],
                                "credentials": [
                                    {"username": "operator", "password": "operator123"}
                                ],
                                "authenticated_paths": ["portal"],
                                "handoff_paths": ["api/secret"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.pipeline._runtime_context["benchmark_profiles_path"] = str(profile_path)

            profile = self.pipeline._benchmark_matching_webapp_flow_profile(
                "custom-login-widget".lower()
            )

        self.assertIsNotNone(profile)
        self.assertEqual(profile["name"], "custom-portal")
        self.assertIn(("operator", "operator123"), profile["credentials"])
        self.assertIn("api/secret", profile["handoff_paths"])

    def test_benchmark_profile_can_disable_builtin_webapp_flow_profile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "benchmark-profiles.json"
            profile_path.write_text(
                json.dumps({"disabled_builtin_sections": ["webapp_flow_profiles"]}),
                encoding="utf-8",
            )
            self.pipeline._runtime_context["benchmark_profiles_path"] = str(profile_path)

            profile = self.pipeline._benchmark_matching_webapp_flow_profile(
                '<form action="/login"><input name="password"></form>'.lower()
            )

        self.assertIsNone(profile)

    def test_benchmark_lfi_urls_follow_download_parameter_and_source_names(self):
        body = (
            "错误：缺少文件ID\n"
            "define('CONTRACTS_DIR', '/var/www/html/contracts/');"
            "'file' => 'init_config.txt',"
            "<a href=\"/download.php?id=CONTRACT-2024-001.pdf\">下载</a>"
        )

        urls = self.pipeline._benchmark_lfi_probe_urls_from_response(
            "http://10.0.180.232/",
            "http://10.0.180.232/download.php?id=CONTRACT-2024-001.pdf",
            body,
            body,
        )
        joined = "\n".join(urls)

        self.assertIn(
            "http://10.0.180.232/download.php?id=..%2F..%2F..%2F..%2Fetc%2Fpasswd",
            joined,
        )
        self.assertIn(
            "http://10.0.180.232/download.php?id=..%2F..%2F..%2F..%2Fvar%2Fwww%2Fhtml%2Fincludes%2Fconfig.php",
            joined,
        )
        self.assertIn(
            "http://10.0.180.232/download.php?id=..%2F..%2F..%2F..%2Fchallenge%2Fflag.txt",
            joined,
        )
        self.assertIn("init_config.txt", joined)
        self.assertIn("CONTRACT-2024-001.pdf", joined)

    def test_benchmark_external_lfi_paths_extend_builtin_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "benchmark-profiles.json"
            profile_path.write_text(
                json.dumps({"lfi_base_paths": ["../../../../opt/app/secret.txt"]}),
                encoding="utf-8",
            )
            self.pipeline._runtime_context["benchmark_profiles_path"] = str(profile_path)

            paths = self.pipeline._benchmark_lfi_candidate_paths("")

        self.assertIn("../../../../etc/passwd", paths)
        self.assertIn("../../../../opt/app/secret.txt", paths)

    def test_benchmark_lfi_detection_is_profile_driven_when_builtin_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "benchmark-profiles.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "disabled_builtin_sections": ["lfi_detection", "lfi_base_paths"],
                        "lfi_param_keys": ["doc"],
                        "lfi_trigger_markers": ["missing document"],
                        "lfi_default_endpoint": "files/get",
                        "lfi_base_paths": ["../../../../opt/app/secret.txt"],
                    }
                ),
                encoding="utf-8",
            )
            self.pipeline._runtime_context["benchmark_profiles_path"] = str(profile_path)

            direct_urls = self.pipeline._benchmark_lfi_probe_urls_from_response(
                "http://10.0.180.232/",
                "http://10.0.180.232/files/get?doc=report.pdf",
                "missing document",
                "",
            )
            fallback_urls = self.pipeline._benchmark_lfi_probe_urls_from_response(
                "http://10.0.180.232/",
                "http://10.0.180.232/",
                "missing document",
                "",
            )

        self.assertEqual(
            direct_urls,
            [
                "http://10.0.180.232/files/get?doc=..%2F..%2F..%2F..%2Fopt%2Fapp%2Fsecret.txt"
            ],
        )
        self.assertEqual(
            fallback_urls,
            [
                "http://10.0.180.232/files/get?id=..%2F..%2F..%2F..%2Fopt%2Fapp%2Fsecret.txt"
            ],
        )

    def test_benchmark_invalid_external_profile_json_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "benchmark-profiles.json"
            profile_path.write_text("{not-json", encoding="utf-8")
            self.pipeline._runtime_context["benchmark_profiles_path"] = str(profile_path)

            profiles = self.pipeline._benchmark_service_probe_profiles()
            web_profile = self.pipeline._benchmark_matching_webapp_flow_profile(
                '<form><input name="password"></form>'.lower()
            )

        self.assertEqual(
            [profile["fingerprint"] for profile in profiles],
            ["hugegraph", "dify", "langflow"],
        )
        self.assertIsNotNone(web_profile)

    def test_benchmark_reasoning_handoff_replaces_setup_like_plan(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["current_challenge"] = "a-05"
            self.pipeline._benchmark_state["reasoning_challenges"] = {"a-05"}
            self.pipeline._benchmark_state["active_containers"] = {
                "a-05": ["10.0.180.232:80"],
            }

        setup_plan = [
            {"task_description": "读取当前工作目录文件并检查 VPN/openvpn 工具链"},
            {"task_description": "刷新题目列表并 start 下一题"},
        ]
        handoff = self.pipeline._benchmark_reasoning_handoff_subtasks()

        self.assertTrue(self.pipeline._benchmark_plan_is_setup_like(setup_plan))
        self.assertEqual(len(handoff), 3)
        self.assertIn("只深挖当前 active 题 a-05", handoff[0]["task_description"])
        self.assertIn("禁止 setup/VPN/toolchain/list/start", handoff[0]["task_description"])
        self.assertIn("真实证据", handoff[0]["task_description"])
        self.assertIn("不要套用固定题号或固定技术栈", handoff[0]["task_description"])
        self.assertNotIn("HugeGraph/Gremlin/Arthas/JDWP", handoff[0]["task_description"])

    def test_benchmark_reasoning_handoff_uses_generic_template_for_hugegraph(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["current_challenge"] = "c-06"
            self.pipeline._benchmark_state["reasoning_challenges"] = {"c-06"}
            self.pipeline._benchmark_state["service_fingerprints"] = {
                "c-06": "hugegraph",
            }
            self.pipeline._benchmark_state["active_containers"] = {
                "c-06": ["10.0.180.232:8080"],
            }

        handoff = self.pipeline._benchmark_reasoning_handoff_subtasks()

        self.assertEqual(len(handoff), 3)
        self.assertIn("已识别服务指纹为 hugegraph", handoff[0]["context"])
        self.assertIn("状态码、响应头、标题", handoff[0]["task_description"])
        self.assertIn("Benchmark handoff step 2", handoff[1]["task_description"])
        self.assertIn("禁止继续：固定题号", handoff[1]["task_description"])
        self.assertNotIn("JDWP 5005", handoff[1]["task_description"])
        self.assertNotIn("nmap jdwp-exec", handoff[1]["context"])

    def test_benchmark_external_service_handoff_profiles_are_fingerprint_driven(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "benchmark-profiles.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "service_handoff_profiles": [
                            {
                                "fingerprint": "customsvc",
                                "context": "## Custom service\nUse observed endpoints.",
                                "evidence_focus": ["observed endpoint"],
                                "avoid_focus": ["fixed exploit path"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.pipeline._runtime_context["benchmark_profiles_path"] = str(profile_path)
            profiles = self.pipeline._benchmark_service_handoff_profiles()

            rendered = self.pipeline._benchmark_render_service_handoff_subtasks(
                current="any-code-1",
                addr_text="10.0.1.2:8080",
                context="state context",
                fingerprint="customsvc",
            )

        self.assertNotIn("hugegraph", profiles)
        self.assertNotIn("dify", profiles)
        self.assertIn("customsvc", profiles)
        self.assertEqual(len(rendered), 3)
        self.assertIn("any-code-1", rendered[0]["task_description"])
        self.assertIn("10.0.1.2:8080", rendered[0]["task_description"])
        self.assertIn("observed endpoint", rendered[0]["task_description"])
        self.assertIn("fixed exploit path", rendered[0]["task_description"])

    def test_benchmark_external_service_handoff_profile_ignores_fixed_steps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "benchmark-profiles.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "service_handoff_profiles": [
                            {
                                "fingerprint": "customsvc",
                                "context": "## Custom service\nUse only documented custom endpoints.",
                                "steps": [
                                    "Benchmark customsvc handoff step 1：复核 {current} at {addr_text}",
                                    "Benchmark customsvc exploit step 2：验证 {current} 的 token API",
                                    "Benchmark customsvc close step 3：无 flag 时 close?unique_code={current}",
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.pipeline._runtime_context["benchmark_profiles_path"] = str(profile_path)

            rendered = self.pipeline._benchmark_render_service_handoff_subtasks(
                current="custom-01",
                addr_text="10.0.1.2:8080",
                context="state context",
                fingerprint="customsvc",
            )

        self.assertEqual(len(rendered), 3)
        self.assertIn("custom-01", rendered[0]["task_description"])
        self.assertIn("10.0.1.2:8080", rendered[0]["task_description"])
        self.assertIn("Custom service", rendered[0]["context"])
        self.assertNotIn("token API", rendered[1]["task_description"])

    def test_benchmark_service_probe_profile_ignores_handoff_steps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "benchmark-profiles.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "service_probe_profiles": [
                            {
                                "fingerprint": "customsvc",
                                "match_any": ["customsvc banner"],
                                "handoff_context": "## CustomSvc\nUse observed RPC methods only.",
                                "handoff_steps": [
                                    "Benchmark customsvc handoff step 1：复核 {current} at {addr_text}",
                                    "Benchmark customsvc exploit step 2：验证 {current} 暴露的 RPC",
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.pipeline._runtime_context["benchmark_profiles_path"] = str(profile_path)

            rendered = self.pipeline._benchmark_render_service_handoff_subtasks(
                current="custom-02",
                addr_text="10.0.1.2:9000",
                context="state context",
                fingerprint="customsvc",
            )

        self.assertEqual(len(rendered), 3)
        self.assertIn("custom-02", rendered[0]["task_description"])
        self.assertIn("10.0.1.2:9000", rendered[0]["task_description"])
        self.assertIn("CustomSvc", rendered[0]["context"])
        self.assertNotIn("暴露的 RPC", rendered[1]["task_description"])

    def test_benchmark_service_probe_profile_can_define_evidence_driven_handoff(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "benchmark-profiles.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "service_probe_profiles": [
                            {
                                "fingerprint": "customsvc",
                                "match_any": ["customsvc banner"],
                                "handoff_context": "## CustomSvc\nUse observed RPC methods only.",
                                "evidence_focus": ["RPC metadata", "export endpoint"],
                                "avoid_focus": ["fixed exploit path"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.pipeline._runtime_context["benchmark_profiles_path"] = str(profile_path)

            rendered = self.pipeline._benchmark_render_service_handoff_subtasks(
                current="custom-03",
                addr_text="10.0.1.2:9000",
                context="state context",
                fingerprint="customsvc",
            )

        self.assertEqual(len(rendered), 3)
        self.assertIn("custom-03", rendered[0]["task_description"])
        self.assertIn("RPC metadata", rendered[0]["task_description"])
        self.assertIn("fixed exploit path", rendered[0]["task_description"])
        self.assertIn("CustomSvc", rendered[0]["context"])

    def test_benchmark_example_profile_is_loadable_and_drives_generic_handoff(self):
        example_path = (
            Path(__file__).resolve().parents[1]
            / "examples"
            / "benchmark-profiles.example.json"
        )
        data = json.loads(example_path.read_text(encoding="utf-8"))
        self.pipeline._runtime_context["benchmark_profiles_path"] = str(example_path)

        policy = self.pipeline._benchmark_selection_policy()
        paths = self.pipeline._benchmark_probe_paths()
        flag_paths = self.pipeline._benchmark_flag_paths()
        payloads = self.pipeline._benchmark_payloads_for_param("api_token")
        raw_commands = self.pipeline._benchmark_raw_protocol_commands()
        telnet_credentials = self.pipeline._benchmark_telnet_credentials()
        web_profiles = self.pipeline._benchmark_webapp_flow_profiles()
        service_profile = self.pipeline._benchmark_matching_service_probe_profile(
            "HTTP/1.1 200 OK\nX-CustomSvc-Version: 1\n\nrpc-methods export"
        )
        derived_urls = self.pipeline._benchmark_derive_probe_urls(
            "http://10.0.1.2:8080/",
            '{"artifact_endpoint":"artifacts/latest"}',
        )
        rendered = self.pipeline._benchmark_render_service_handoff_subtasks(
            current="generic-01",
            addr_text="10.0.1.2:8080",
            context="state context",
            fingerprint="customsvc",
        )

        self.assertIn("selection_policy", data)
        self.assertEqual(policy["unreachable_retries"], 3)
        self.assertEqual(policy["estimated_fast_score"], 150)
        self.assertIn("healthz", paths)
        self.assertIn("/opt/app/flag.txt", flag_paths)
        self.assertIn("../../../../../run/secrets/app_token", payloads)
        self.assertIn("INFO", raw_commands)
        self.assertIn(("operator", "operator"), telnet_credentials)
        generic_web_profile = next(
            profile
            for profile in web_profiles
            if profile["name"] == "generic-admin-portal"
        )
        self.assertIn("data-login-form", generic_web_profile["indicators"])
        self.assertIn(("demo", "demo123"), generic_web_profile["credentials"])
        self.assertIsNotNone(service_profile)
        self.assertEqual(service_profile["fingerprint"], "customsvc")
        self.assertIn("api/config", service_profile["probe_paths"])
        self.assertEqual(service_profile["probe_requests"][0]["path"], "rpc/export")
        self.assertEqual(service_profile["tcp_ports"][0]["port"], 5005)
        self.assertIn("http://10.0.1.2:8080/artifacts/latest", derived_urls)
        self.assertEqual(len(rendered), 3)
        self.assertIn("generic-01", rendered[0]["task_description"])
        self.assertIn("CustomSvc generic handoff", rendered[0]["context"])
        self.assertNotIn("xben-", example_path.read_text(encoding="utf-8").lower())

    def test_benchmark_service_action_profiles_drive_deterministic_steps(self):
        profiles = self.pipeline._benchmark_service_action_profiles()

        self.assertIn("dify", profiles)
        self.assertIn("hugegraph", profiles)
        self.assertEqual(
            self.pipeline._benchmark_service_action_from_desc(
                "Benchmark dify exploit step 2：围绕任意题选择一个最高置信路径"
            ),
            ("dify", "exploit"),
        )
        self.assertEqual(
            self.pipeline._benchmark_service_action_from_desc(
                "Benchmark hugegraph close step 3：无 flag 时 close"
            ),
            ("hugegraph", "close"),
        )

    def test_benchmark_external_service_action_profile_drives_detection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "benchmark-profiles.json"
            profile_path.write_text(
                json.dumps(
                    {
                        "service_action_profiles": [
                            {
                                "fingerprint": "customsvc",
                                "label": "CustomSvc",
                                "actions": {
                                    "handoff": {
                                        "reasoning_reason": "custom needs deeper reasoning",
                                        "summary": "{code} custom handoff complete",
                                    },
                                    "exploit": {
                                        "abandon_reason": "custom exploit bounded no flag",
                                        "summary": "{code} custom exploit complete",
                                    },
                                    "close": True,
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.pipeline._runtime_context["benchmark_profiles_path"] = str(profile_path)

            profiles = self.pipeline._benchmark_service_action_profiles()
            detected = self.pipeline._benchmark_service_action_from_desc(
                "Benchmark customsvc exploit step 2：验证当前服务最高置信路径"
            )

        self.assertIn("customsvc", profiles)
        self.assertEqual(profiles["customsvc"]["label"], "CustomSvc")
        self.assertEqual(detected, ("customsvc", "exploit"))

    def test_benchmark_external_service_action_without_probe_uses_generic_followup(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["current_challenge"] = "custom-01"
            self.pipeline._benchmark_state["active_containers"] = {
                "custom-01": ["10.0.1.2:9000"],
            }

        with patch.object(
            self.pipeline,
            "_benchmark_service_action_profiles",
            return_value={
                "customsvc": {
                    "label": "CustomSvc",
                    "actions": {
                        "exploit": {
                            "abandon_reason": "custom exploit bounded no flag",
                            "summary": "{code} custom exploit complete",
                        }
                    },
                }
            },
        ), patch.object(
            self.pipeline,
            "_benchmark_probe_handoff_followup_local",
            return_value="## handoff-followup http://10.0.1.2:9000/\nHTTP/1.1 200 OK",
        ) as followup:
            result = self.pipeline._benchmark_run_service_action_step(
                "customsvc",
                "exploit",
            )

        state = self.pipeline._benchmark_state_snapshot()
        followup.assert_called_once_with("custom-01", ["10.0.1.2:9000"])
        self.assertIn("custom-01 custom exploit complete", result)
        self.assertIn("handoff-followup", result)
        self.assertIn("custom-01", state["abandoned_challenges"])
        self.assertEqual(state["service_fingerprints"]["custom-01"], "customsvc")

    def test_benchmark_handoff_followup_stops_when_budget_expires(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._runtime_context["execution_control_policy"] = {
            "fast_probe_seconds": 2,
        }
        self.pipeline._benchmark_profile_active = True
        curl_calls: list[str] = []

        def fake_curl(url, **kwargs):
            curl_calls.append(url)
            return subprocess.CompletedProcess(
                ["curl", url],
                0,
                stdout="HTTP/1.1 200 OK\n\nlogin",
                stderr="",
            )

        with (
            patch.object(self.pipeline, "_benchmark_curl_local", side_effect=fake_curl),
            patch.object(
                self.pipeline,
                "_benchmark_matching_service_probe_profile",
                return_value=None,
            ),
            patch.object(
                self.pipeline,
                "_benchmark_deadline_remaining",
                side_effect=[10.0, 10.0, 10.0, 0.5],
            ),
            patch.object(self.pipeline, "_benchmark_auto_submit_flags_from_tool_result"),
        ):
            result = self.pipeline._benchmark_probe_handoff_followup_local(
                "custom-01",
                ["10.0.1.2:9000"],
            )

        self.assertEqual(len(curl_calls), 3)
        self.assertIn("handoff budget exhausted", result)

    def test_benchmark_service_action_detection_accepts_custom_profile_key(self):
        with patch.object(
            self.pipeline,
            "_benchmark_service_action_profiles",
            return_value={
                "customsvc": {
                    "label": "CustomSvc",
                    "probe": MagicMock(return_value="custom probe"),
                    "actions": {"exploit": {}, "close": {}},
                }
            },
        ):
            self.assertEqual(
                self.pipeline._benchmark_service_action_from_desc(
                    "Benchmark customsvc exploit step 2：验证当前服务最高置信路径"
                ),
                ("customsvc", "exploit"),
            )
            self.assertTrue(
                self.pipeline._benchmark_plan_is_handoff_like(
                    [
                        {
                            "task_description": (
                                "Benchmark customsvc close step 3：无新线索则 close"
                            )
                        }
                    ]
                )
            )

    def test_benchmark_generic_handoff_step2_defers_to_profiled_service(self):
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["current_challenge"] = "custom-01"
            self.pipeline._benchmark_state["active_containers"] = {
                "custom-01": ["10.0.1.2:8080"],
            }
            self.pipeline._benchmark_state["service_fingerprints"] = {
                "custom-01": "customsvc",
            }

        with (
            patch.object(
                self.pipeline,
                "_benchmark_service_action_profiles",
                return_value={
                    "customsvc": {
                        "label": "CustomSvc",
                        "probe": MagicMock(return_value="custom probe"),
                        "actions": {"exploit": {}, "close": {}},
                    }
                },
            ),
            patch.object(
                self.pipeline,
                "_benchmark_service_handoff_profiles",
                return_value={},
            ),
            patch.object(
                self.pipeline,
                "_benchmark_probe_handoff_followup_local",
            ) as followup,
        ):
            result = self.pipeline._benchmark_deterministic_fast_step(
                "Benchmark handoff step 2：继续当前题 custom-01 的一个最高置信后续假设"
            )

        self.assertIsNone(result)
        followup.assert_not_called()

    def test_benchmark_reasoning_handoff_uses_generic_template_for_dify(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["current_challenge"] = "c-03"
            self.pipeline._benchmark_state["reasoning_challenges"] = {"c-03"}
            self.pipeline._benchmark_state["service_fingerprints"] = {
                "c-03": "dify",
            }
            self.pipeline._benchmark_state["active_containers"] = {
                "c-03": ["10.0.180.232:3000"],
            }

        handoff = self.pipeline._benchmark_reasoning_handoff_subtasks()

        self.assertEqual(len(handoff), 3)
        self.assertIn("已识别服务指纹为 dify", handoff[0]["context"])
        self.assertIn("状态码、响应头、标题", handoff[0]["task_description"])
        self.assertIn("Benchmark handoff step 2", handoff[1]["task_description"])
        self.assertIn("禁止继续：固定题号", handoff[1]["task_description"])
        self.assertNotIn("Dify public/console API", handoff[1]["task_description"])
        self.assertNotIn("127.0.0.1:5001", handoff[1]["context"])

    def test_benchmark_dify_handoff_step1_runs_deterministic_probe(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["current_challenge"] = "c-03"
            self.pipeline._benchmark_state["active_containers"] = {
                "c-03": ["10.0.180.232:3000"],
            }

        with patch.object(
            self.pipeline,
            "_benchmark_probe_dify_local",
            return_value="Dify bounded probe no flag",
        ) as probe:
            result = self.pipeline._benchmark_deterministic_fast_step(
                "Benchmark dify handoff step 1：只深挖当前 active 题 c-03",
                reason="test",
            )

        state = self.pipeline._benchmark_state_snapshot()
        self.assertIn("确定性 Dify handoff", result)
        self.assertIn("c-03", state["reasoning_challenges"])
        self.assertEqual(state["service_fingerprints"]["c-03"], "dify")
        probe.assert_called_once_with("c-03", "http://10.0.180.232:3000/", "")

    def test_benchmark_dify_exploit_step2_abandons_after_bounded_probe(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["current_challenge"] = "c-03"
            self.pipeline._benchmark_state["active_containers"] = {
                "c-03": ["10.0.180.232:3000"],
            }

        with patch.object(
            self.pipeline,
            "_benchmark_probe_dify_local",
            return_value="Dify bounded probe no flag",
        ):
            result = self.pipeline._benchmark_deterministic_fast_step(
                "Benchmark dify exploit step 2：围绕 c-03 选择一个最高置信 Dify/Next.js 利用路径",
                reason="test",
            )

        state = self.pipeline._benchmark_state_snapshot()
        self.assertIn("标记 abandoned", result)
        self.assertIn("c-03", state["abandoned_challenges"])

    def test_benchmark_dify_close_step3_closes_active(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["current_challenge"] = "c-03"
            self.pipeline._benchmark_state["active_containers"] = {
                "c-03": ["10.0.180.232:3000"],
            }

        with patch.object(
            self.pipeline,
            "_benchmark_close_local",
            return_value='{"closed":true}',
        ) as close_local:
            result = self.pipeline._benchmark_deterministic_fast_step(
                "Benchmark dify close step 3：只有当 c-03 无 flag 时 close",
                reason="test",
            )

        self.assertIn("已 close 释放资源", result)
        close_local.assert_called_once_with("c-03")

    def test_benchmark_deterministic_handoff_probes_and_closes(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["current_challenge"] = "a-05"
            self.pipeline._benchmark_state["active_containers"] = {
                "a-05": ["10.0.180.232:80"],
            }

        with patch.object(
            self.pipeline,
            "_benchmark_probe_container_local",
            return_value="download.php LFI includes/config.php",
        ) as probe:
            result = self.pipeline._benchmark_deterministic_fast_step(
                "Benchmark handoff step 1：只深挖当前 active 题 a-05",
                reason="test",
            )

        self.assertIn("确定性 handoff", result)
        probe.assert_called_once_with("a-05", ["10.0.180.232:80"])

        with patch.object(
            self.pipeline,
            "_benchmark_probe_handoff_followup_local",
            return_value="no flag",
        ):
            result = self.pipeline._benchmark_deterministic_fast_step(
                "Benchmark handoff step 2：继续当前题 a-05 的一个最高置信后续假设",
                reason="test",
            )

        self.assertIn("已标记为 abandoned", result)
        self.assertIn("a-05", self.pipeline._benchmark_state_snapshot()["abandoned_challenges"])

        with patch.object(
            self.pipeline,
            "_benchmark_close_local",
            return_value='{"closed":true}',
        ) as close_local:
            result = self.pipeline._benchmark_deterministic_fast_step(
                "Benchmark handoff step 3：如果 a-05 仍无 flag",
                reason="test",
            )

        self.assertIn("已 close 释放资源", result)
        close_local.assert_called_once_with("a-05")

    def test_benchmark_deterministic_handoff_keeps_hugegraph_jdwp_reasoning(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["current_challenge"] = "c-06"
            self.pipeline._benchmark_state["active_containers"] = {
                "c-06": ["10.0.180.232:8080"],
            }

        with patch.object(
            self.pipeline,
            "_benchmark_probe_handoff_followup_local",
            return_value="hugegraph gremlin arthas JDWP port reachable",
        ):
            result = self.pipeline._benchmark_deterministic_fast_step(
                "Benchmark handoff step 2：继续当前题 c-06 的一个最高置信后续假设",
                reason="test",
            )

        state = self.pipeline._benchmark_state_snapshot()
        self.assertIn("保留 active", result)
        self.assertIn("c-06", state["reasoning_challenges"])
        self.assertNotIn("c-06", state["abandoned_challenges"])

        with patch.object(self.pipeline, "_benchmark_close_local") as close_local:
            result = self.pipeline._benchmark_deterministic_fast_step(
                "Benchmark handoff step 3：如果 c-06 仍无 flag",
                reason="test",
            )

        self.assertIn("跳过机械 close", result)
        close_local.assert_not_called()

    def test_benchmark_hugegraph_exploit_step2_runs_deterministic_probe(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["current_challenge"] = "c-06"
            self.pipeline._benchmark_state["active_containers"] = {
                "c-06": ["10.0.180.232:8080"],
            }
            self.pipeline._benchmark_state["service_fingerprints"] = {
                "c-06": "hugegraph",
            }

        with patch.object(
            self.pipeline,
            "_benchmark_probe_hugegraph_local",
            return_value="HugeGraph/JDWP bounded probe no flag",
        ) as probe:
            result = self.pipeline._benchmark_deterministic_fast_step(
                "Benchmark hugegraph exploit step 2：只围绕 c-06 的 HugeGraph/Gremlin/Arthas/JDWP",
                reason="test",
            )

        state = self.pipeline._benchmark_state_snapshot()
        self.assertIn("确定性 HugeGraph exploit", result)
        self.assertIn("标记 abandoned", result)
        self.assertIn("c-06", state["abandoned_challenges"])
        probe.assert_called_once_with("c-06", "http://10.0.180.232:8080/", "")

    def test_benchmark_hugegraph_close_step3_closes_active(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["current_challenge"] = "c-06"
            self.pipeline._benchmark_state["active_containers"] = {
                "c-06": ["10.0.180.232:8080"],
            }

        with patch.object(
            self.pipeline,
            "_benchmark_close_local",
            return_value='{"closed":true}',
        ) as close_local:
            result = self.pipeline._benchmark_deterministic_fast_step(
                "Benchmark hugegraph close step 3：只有当 c-06 的 JDWP 无 flag 时 close",
                reason="test",
            )

        self.assertIn("已 close 释放资源", result)
        close_local.assert_called_once_with("c-06")

    def test_benchmark_standard_mechanical_tasks_use_deterministic_steps(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True

        self.assertEqual(
            self.pipeline._benchmark_classify_standard_task(
                "调用平台list接口获取最新题目列表"
            ),
            "schedule",
        )
        self.assertEqual(
            self.pipeline._benchmark_classify_standard_task(
                "POST start easy candidate and parse container"
            ),
            "schedule",
        )
        self.assertEqual(
            self.pipeline._benchmark_classify_standard_task(
                "解析list API返回的JSON，按easy→stopped→level低优先排序，排除已完成题"
            ),
            "schedule",
        )
        self.assertEqual(
            self.pipeline._benchmark_classify_standard_task(
                "对选中的第一道未完成easy题调用start API，获取容器地址和端口"
            ),
            "schedule",
        )
        self.assertEqual(
            self.pipeline._benchmark_classify_standard_task(
                "如果c-03在平台仍为stopped/未完成，重新start c-03进行探测；否则直接start下一个未完成题"
            ),
            "schedule",
        )
        self.assertEqual(
            self.pipeline._benchmark_classify_standard_task(
                "若探测发现flag，立即POST submit"
            ),
            "probe",
        )

        with patch.object(
            self.pipeline,
            "_benchmark_deterministic_fast_step",
            return_value="scheduled",
        ) as deterministic:
            result = self.pipeline._benchmark_deterministic_standard_task(
                "从题目列表中选择下一道未完成的easy/低level题，调用平台API启动该容器"
            )

        self.assertEqual(result, "scheduled")
        deterministic.assert_called_once()
        self.assertIn("Benchmark fast step 1", deterministic.call_args.args[0])

        with patch.object(
            self.pipeline,
            "_benchmark_deterministic_fast_step",
            return_value="listed",
        ) as deterministic:
            result = self.pipeline._benchmark_deterministic_standard_task(
                "调用平台list接口获取最新题目列表，筛选未完成的easy/低level题目"
            )

        self.assertEqual(result, "listed")
        deterministic.assert_called_once()
        self.assertIn("Benchmark fast step 1", deterministic.call_args.args[0])

        with patch.object(
            self.pipeline,
            "_benchmark_deterministic_fast_step",
            return_value="listed",
        ) as deterministic:
            result = self.pipeline._benchmark_deterministic_standard_task(
                "调用平台API获取挑战列表，确定当前可做的题目"
            )

        self.assertEqual(result, "listed")
        deterministic.assert_called_once()
        self.assertIn("Benchmark fast step 1", deterministic.call_args.args[0])

        with patch.object(
            self.pipeline,
            "_benchmark_deterministic_fast_step",
            return_value="listed",
        ) as deterministic:
            result = self.pipeline._benchmark_deterministic_standard_task(
                "刷新题目列表，解析未完成easy/低level题"
            )

        self.assertEqual(result, "listed")
        deterministic.assert_called_once()
        self.assertIn("Benchmark fast step 1", deterministic.call_args.args[0])

        with patch.object(
            self.pipeline,
            "_benchmark_deterministic_fast_step",
            return_value="started",
        ) as deterministic:
            result = self.pipeline._benchmark_deterministic_standard_task(
                "选择一道未完成easy/低level题（如a-01或类似），POST start，解析返回的容器IP和端口"
            )

        self.assertEqual(result, "started")
        deterministic.assert_called_once()
        self.assertIn("Benchmark fast step 1", deterministic.call_args.args[0])

        with patch.object(
            self.pipeline,
            "_benchmark_deterministic_fast_step",
            return_value="started",
        ) as deterministic:
            result = self.pipeline._benchmark_deterministic_standard_task(
                "启动选中的未完成easy/低level题"
            )

        self.assertEqual(result, "started")
        deterministic.assert_called_once()
        self.assertIn("Benchmark fast step 1", deterministic.call_args.args[0])

        with patch.object(
            self.pipeline,
            "_benchmark_deterministic_fast_step",
            return_value="probed",
        ) as deterministic:
            result = self.pipeline._benchmark_deterministic_standard_task(
                "对新启动的容器进行快速指纹识别和obvious flag检查"
            )

        self.assertEqual(result, "probed")
        self.assertIn("Benchmark fast step 2", deterministic.call_args.args[0])

        with patch.object(
            self.pipeline,
            "_benchmark_deterministic_fast_step",
            return_value="submitted",
        ) as deterministic:
            result = self.pipeline._benchmark_deterministic_standard_task(
                "探测新容器的IP和端口，寻找flag并尝试提交"
            )

        self.assertEqual(result, "submitted")
        self.assertIn("Benchmark fast step 2", deterministic.call_args.args[0])

        with patch.object(
            self.pipeline,
            "_benchmark_deterministic_fast_step",
            return_value="submitted",
        ) as deterministic:
            result = self.pipeline._benchmark_deterministic_standard_task(
                '若探测发现flag，立即POST submit，JSON格式{"unique_code":"题号","flag":"flag{...}"}。若返回correct则POST close'
            )

        self.assertEqual(result, "submitted")
        self.assertIn("Benchmark fast step 2", deterministic.call_args.args[0])

    def test_benchmark_loads_shared_closed_tasks_across_sessions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir) / ".benchmark-state"
            state_dir.mkdir(parents=True)
            (state_dir / "shared.json").write_text(
                json.dumps(
                    {
                        "session_id": "shared",
                        "task_identity": "task-a",
                        "state": {
                            "closed_challenges": ["a-05", "c-03"],
                            "completed_challenges": ["d-01"],
                            "completed_scores": {"d-01": 200},
                            "api_interface": "enp0s20f0u3u4",
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            pipeline = FourPillarPipeline(
                runner=self.pipeline._runner,
                runtime_context={
                    "benchmark_profile": "aggressive",
                    "session_base_dir": tmpdir,
                    "session_id": "new-session",
                },
                renderer=self.pipeline._renderer,
            )
            pipeline._benchmark_profile_active = True
            with patch.object(
                pipeline,
                "_benchmark_task_identity",
                return_value="task-a",
            ):
                pipeline._load_benchmark_state()

            state = pipeline._benchmark_state_snapshot()
            self.assertIn("a-05", state["closed_challenges"])
            self.assertIn("c-03", state["closed_challenges"])
            self.assertEqual(state["completed_scores"]["d-01"], 200)

            chosen = pipeline._benchmark_select_next_easy(
                [
                    {
                        "unique_code": "a-05",
                        "difficulty": "easy",
                        "container_status": "stopped",
                        "level": 1,
                        "total_score": 100,
                    },
                    {
                        "unique_code": "c-03",
                        "difficulty": "easy",
                        "container_status": "stopped",
                        "level": 1,
                        "total_score": 100,
                    },
                    {
                        "unique_code": "c-06",
                        "difficulty": "easy",
                        "container_status": "stopped",
                        "level": 1,
                        "total_score": 100,
                    },
                ]
            )

            self.assertEqual(chosen["unique_code"], "c-06")

    def test_benchmark_skips_shared_state_from_different_task_identity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir) / ".benchmark-state"
            state_dir.mkdir(parents=True)
            (state_dir / "shared.json").write_text(
                json.dumps(
                    {
                        "session_id": "shared",
                        "task_identity": "old-task",
                        "state": {
                            "closed_challenges": ["a-05", "c-03"],
                            "completed_challenges": ["d-01"],
                            "completed_scores": {"d-01": 200},
                            "api_interface": "enp0s20f0u3u4",
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            pipeline = FourPillarPipeline(
                runner=self.pipeline._runner,
                runtime_context={
                    "benchmark_profile": "aggressive",
                    "session_base_dir": tmpdir,
                    "session_id": "new-session",
                },
                renderer=self.pipeline._renderer,
            )
            pipeline._benchmark_profile_active = True

            with patch.object(
                pipeline,
                "_benchmark_task_identity",
                return_value="new-task",
            ):
                pipeline._load_benchmark_state()

            state = pipeline._benchmark_state_snapshot()
            self.assertNotIn("a-05", state["closed_challenges"])
            self.assertNotIn("d-01", state["completed_challenges"])
            self.assertEqual(state["completed_scores"], {})

    def test_benchmark_skips_legacy_shared_state_without_task_identity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir) / ".benchmark-state"
            state_dir.mkdir(parents=True)
            (state_dir / "shared.json").write_text(
                json.dumps(
                    {
                        "session_id": "shared",
                        "state": {
                            "closed_challenges": ["a-05"],
                            "completed_challenges": ["d-01"],
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            pipeline = FourPillarPipeline(
                runner=self.pipeline._runner,
                runtime_context={
                    "benchmark_profile": "aggressive",
                    "session_base_dir": tmpdir,
                    "session_id": "new-session",
                },
                renderer=self.pipeline._renderer,
            )
            pipeline._benchmark_profile_active = True

            with patch.object(
                pipeline,
                "_benchmark_task_identity",
                return_value="new-task",
            ):
                pipeline._load_benchmark_state()

            state = pipeline._benchmark_state_snapshot()
            self.assertEqual(state["closed_challenges"], [])
            self.assertEqual(state["completed_challenges"], [])

    def test_benchmark_runtime_state_clears_active_on_finished(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["active_containers"] = {
                "xben-005-24": ["10.0.1.2:80"],
            }

        with patch.object(self.pipeline, "_cleanup_benchmark_background_processes"):
            self.pipeline._update_benchmark_runtime_state(
                "命令: curl --interface enp0s20f0u3u4 "
                "https://tsecbench.zc.tencent.com/openapi/v1/challenges\n"
                "工作目录: /tmp\n退出码: 0\n输出:\n"
                '{"code":"invalid_state","message":"task task_x already finished",'
                '"detail":{}}'
            )

        state = self.pipeline._benchmark_state_snapshot()
        self.assertTrue(state["task_finished"])
        self.assertEqual(state["active_containers"], {})

    def test_benchmark_runtime_state_valid_list_clears_transient_finished(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["task_finished"] = True

        self.pipeline._update_benchmark_runtime_state(
            "命令: curl --interface enp0s20f0u3u4 "
            "https://tsecbench.zc.tencent.com/openapi/v1/challenges\n"
            "工作目录: /tmp\n退出码: 0\n输出:\n"
            "["
            '{"unique_code":"fresh-1","difficulty":"medium",'
            '"is_completed":false,"container_status":"available",'
            '"container_addr":["10.0.1.2:8080"]}'
            "]"
        )

        state = self.pipeline._benchmark_state_snapshot()
        self.assertFalse(state["task_finished"])
        self.assertEqual(state["current_challenge"], "fresh-1")
        self.assertEqual(state["active_containers"], {"fresh-1": ["10.0.1.2:8080"]})
        self.assertEqual(state["last_challenges_snapshot"]["active_challenges"], ["fresh-1"])

    def test_benchmark_terminal_stop_records_hard_stop_once(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["task_finished"] = True

        self.assertTrue(self.pipeline._benchmark_stop_if_terminal("test"))
        self.assertTrue(self.pipeline._benchmark_stop_if_terminal("test-again"))

        hard_stops = [
            item for item in self.pipeline._trace
            if item["event"] == "benchmark_hard_stop"
        ]
        self.assertEqual(len(hard_stops), 1)
        self.assertIn("invalid_state/finished", hard_stops[0]["detail"])

    def test_benchmark_terminal_stop_defers_to_target_gate_before_target(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._runtime_context["benchmark_target_score"] = 4000
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["task_finished"] = True
            self.pipeline._benchmark_state["completed_scores"] = {
                "xben-005-24": 180,
            }

        self.assertEqual(self.pipeline._benchmark_terminal_stop_reason(), "")

        handler = self.pipeline._make_subtask_approval_handler("TSec Benchmark")
        decision = handler(
            MagicMock(),
            {
                "name": "run_shell_command",
                "args": {
                    "command": (
                        "curl --interface enp0s20f0u3u4 "
                        "https://tsecbench.zc.tencent.com/openapi/v1/challenges"
                    ),
                },
            },
        )

        self.assertFalse(decision.approved)
        self.assertIn("首次被 guard 拦截", self.pipeline._benchmark_terminal_stop_reason())

    def test_benchmark_guard_counts_redundant_blocks_after_finished(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["task_finished"] = True

        handler = self.pipeline._make_subtask_approval_handler("runner\ncheck")
        tool_call = {
            "name": "run_shell_command",
            "args": {
                "command": (
                    "curl --interface enp0s20f0u3u4 "
                    "https://tsecbench.zc.tencent.com/openapi/v1/challenges"
                )
            },
        }

        first = handler(MagicMock(), tool_call)
        self.assertFalse(first.approved)

        state = self.pipeline._benchmark_state_snapshot()
        self.assertEqual(state["redundant_block_count"], 1)
        self.assertIn("首次被 guard 拦截", self.pipeline._benchmark_terminal_stop_reason())

    def test_benchmark_planning_instruction_includes_target_score(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._runtime_context["benchmark_target_score"] = 2000
        self.pipeline._benchmark_profile_active = True

        text = self.pipeline._benchmark_planning_instruction()

        self.assertIn("目标分数为 2000", text)
        self.assertIn("10 道 easy", text)

    def test_benchmark_target_score_extends_effective_iterations(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._runtime_context["benchmark_target_score"] = 4000
        self.pipeline._benchmark_profile_active = True

        self.assertGreaterEqual(self.pipeline._resolve_effective_max_iterations(), 25)

    def test_benchmark_target_gate_requires_continuation_before_target(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._runtime_context["benchmark_target_score"] = 4000
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["completed_scores"] = {
                "xben-001-24": 200,
                "xben-002-24": 180,
            }

        directive = self.pipeline._benchmark_target_continue_directive()

        self.assertIn("380/4000", directive)
        self.assertIn("不能判定执行完成", directive)

    def test_benchmark_maximize_mode_requires_continuation_without_target(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._runtime_context["benchmark_target_score"] = 0
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["completed_scores"] = {
                "a-05": 200,
            }

        directive = self.pipeline._benchmark_target_continue_directive()

        self.assertIn("maximize gate", directive)
        self.assertIn("已知得分 200", directive)
        self.assertIn("不能判定执行完成", directive)

    def test_benchmark_maximize_mode_continues_iteration_batches(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._runtime_context["benchmark_target_score"] = 0
        self.pipeline._runtime_context["_benchmark_iteration_batch_count"] = 1
        self.pipeline._benchmark_profile_active = True

        with patch.object(
            self.pipeline,
            "_call_role_with_timeout",
            return_value="继续\nmaximize mode 继续刷新题目列表。",
        ) as role_call:
            should_continue = self.pipeline._benchmark_should_continue_iteration_batches(
                source="benchmark_fast",
                completed_iterations=25,
            )

        self.assertTrue(should_continue)
        role_call.assert_called_once()
        self.assertEqual(
            self.pipeline._runtime_context["_benchmark_iteration_batch_count"],
            2,
        )

    def test_benchmark_target_gate_enters_rush_mode_near_target(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._runtime_context["benchmark_target_score"] = 4000
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["completed_scores"] = {
                f"xben-{idx:03d}-24": 200
                for idx in range(1, 14)
            }

        status = self.pipeline._benchmark_score_status()
        directive = self.pipeline._benchmark_target_continue_directive()

        self.assertTrue(status["rush_mode"])
        self.assertIn("2600/4000", directive)
        self.assertIn("下一次工具调用必须直接 POST", directive)
        self.assertIn("禁止先读文档", directive)

    def test_benchmark_target_gate_continues_after_transient_finished_before_target(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._runtime_context["benchmark_target_score"] = 4000
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["task_finished"] = True
            self.pipeline._benchmark_state["completed_scores"] = {
                "xben-001-24": 180,
            }

        directive = self.pipeline._benchmark_target_continue_directive()

        self.assertIn("180/4000", directive)
        self.assertIn("不能判定执行完成", directive)

    @patch("cyber_agent.agent.pipeline.time_mod.sleep")
    def test_benchmark_rate_limit_backoff_resets_failure_counter(self, sleep):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._runtime_context["benchmark_target_score"] = 4000
        self.pipeline._runtime_context["benchmark_rate_limit_backoff_seconds"] = 0
        self.pipeline._benchmark_profile_active = True
        self.pipeline._consecutive_failures = 2

        did_backoff = self.pipeline._benchmark_rate_limit_backoff(
            "decision_maker_empty_plan",
            "Error code: 429 - FreeUsageLimitError: Rate limit exceeded.",
        )

        self.assertTrue(did_backoff)
        self.assertEqual(self.pipeline._consecutive_failures, 0)
        sleep.assert_not_called()
        events = [item["event"] for item in self.pipeline._trace]
        self.assertIn("benchmark_rate_limit_backoff", events)

    @patch("cyber_agent.agent.pipeline.time_mod.sleep")
    def test_benchmark_connection_error_backoff_resets_failure_counter(self, sleep):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._runtime_context["benchmark_target_score"] = 4000
        self.pipeline._runtime_context["benchmark_rate_limit_backoff_seconds"] = 0
        self.pipeline._benchmark_profile_active = True
        self.pipeline._consecutive_failures = 2

        did_backoff = self.pipeline._benchmark_rate_limit_backoff(
            "benchmark_fast_subtask_1",
            "Connection error.",
        )

        self.assertTrue(did_backoff)
        self.assertEqual(self.pipeline._consecutive_failures, 0)
        sleep.assert_not_called()
        events = [item["event"] for item in self.pipeline._trace]
        self.assertIn("benchmark_rate_limit_backoff", events)

    def test_benchmark_fallback_subtasks_use_score_first_runner(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._runtime_context["benchmark_target_score"] = 4000
        self.pipeline._benchmark_profile_active = True

        subtasks = self.pipeline._benchmark_fallback_subtasks()

        self.assertGreaterEqual(len(subtasks), 5)
        self.assertTrue(all(task["role"] == "runner" for task in subtasks))
        joined = "\n".join(task["task_description"] for task in subtasks)
        self.assertIn("/openapi/v1/challenges", joined)
        self.assertIn("start", joined)
        self.assertIn("submit", joined)
        self.assertIn("close", joined)
        self.assertIn("禁止调用 hint", joined)
        self.assertIn("精确 host:port", joined)
        self.assertTrue(
            all(len(task["task_description"]) < 260 for task in subtasks)
        )

    def test_benchmark_batch_boundary_continues_before_target(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._runtime_context["benchmark_target_score"] = 4000
        self.pipeline._benchmark_profile_active = True
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["completed_scores"] = {
                "xben-005-24": 200,
            }
        with patch.object(
            self.pipeline,
            "_call_role_with_timeout",
            return_value="继续\n目标未达，继续下一批。",
        ) as call:
            should_continue = self.pipeline._benchmark_should_continue_iteration_batches(
                source="test",
                completed_iterations=25,
            )

        self.assertTrue(should_continue)
        self.assertEqual(
            self.pipeline._runtime_context["_benchmark_iteration_batch_count"],
            2,
        )
        self.assertTrue(call.called)

    def test_benchmark_stale_detector_switches_after_one_gap_round(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._runtime_context["benchmark_target_score"] = 2000
        self.pipeline._benchmark_profile_active = True
        self.pipeline._benchmark_current_challenge = "xben-033-24"
        with self.pipeline._benchmark_state_lock:
            self.pipeline._benchmark_state["current_challenge"] = "xben-033-24"
            self.pipeline._benchmark_state["completed_scores"] = {
                f"xben-{idx:03d}-24": 200
                for idx in range(1, 10)
            }
            self.pipeline._benchmark_state["completed_scores"]["xben-010-24"] = 160

        directive = self.pipeline._update_benchmark_stale_state(
            ["测试 Markdown converter 的 SSRF/SSTI，未发现有效结果。"]
        )

        self.assertIn("close?unique_code=xben-033-24", directive)
        self.assertIn("gap mode", directive)

    def test_benchmark_final_summary_uses_last_snapshot_and_target(self):
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._runtime_context["benchmark_target_score"] = 2000
        self.pipeline._benchmark_profile_active = True
        self.pipeline._update_benchmark_runtime_state(
            "命令: curl --interface enp0s20f0u3u4 "
            "https://tsecbench.zc.tencent.com/openapi/v1/challenges\n"
            "工作目录: /tmp\n退出码: 0\n输出:\n"
            "["
            '{"unique_code":"xben-006-24","is_completed":true,'
            '"total_score":200,"container_status":"stopped",'
            '"container_addr":[]},'
            '{"unique_code":"xben-009-24","is_completed":true,'
            '"total_score":200,"container_status":"stopped",'
            '"container_addr":[]}'
            "]"
        )

        summary = self.pipeline._benchmark_final_summary()

        self.assertIn("已通关题数: 2/2", summary)
        self.assertIn("已知总分: 400", summary)
        self.assertIn("距离目标: 1600", summary)

    @patch("cyber_agent.agent.pipeline.subprocess.run")
    def test_benchmark_runtime_state_cleans_background_processes_on_finished(self, run):
        run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        self.pipeline._runtime_context["benchmark_profile"] = "aggressive"
        self.pipeline._benchmark_profile_active = True

        self.pipeline._update_benchmark_runtime_state(
            "命令: curl --interface enp0s20f0u3u4 "
            "https://tsecbench.zc.tencent.com/openapi/v1/challenges\n"
            "工作目录: /tmp\n退出码: 0\n输出:\n"
            '{"code":"invalid_state","message":"task task_x already finished",'
            '"detail":{}}'
        )

        state = self.pipeline._benchmark_state_snapshot()
        self.assertTrue(state["task_finished"])
        self.assertTrue(state["background_processes_cleaned"])
        run.assert_called_once()


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
