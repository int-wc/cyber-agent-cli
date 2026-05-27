"""编排器修复项验证测试：空输出回退、摘要事件、token基线、复杂度检测。"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from cyber_agent.agent.orchestrator import (
    AgentResult,
    AgentRole,
    AgentTask,
    MultiAgentOrchestrator,
)
from cyber_agent.agent.runner import _estimate_tokens_from_text
from cyber_agent.cli.app import _detect_task_complexity
from cyber_agent.cli.render import CliRenderer


# ── Fake LLM for orchestrator tests ──

class _FakeLLM:
    """模拟 LLM，可编程控制返回的 AIMessage。"""

    def __init__(self, responses: list[AIMessage] | None = None) -> None:
        self.responses = responses or []
        self.call_count = 0
        self.bound_tools: list = []

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = list(tools)
        return self

    def invoke(self, messages):
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return resp
        # 默认：返回空文本无工具调用
        return AIMessage(content="", tool_calls=[])


# ── Tests ──


class RunRoleAgentEmptyOutputTestCase(unittest.TestCase):
    """测试 _run_role_agent 在模型返回空内容时的回退逻辑。"""

    def setUp(self):
        self.orchestrator = MultiAgentOrchestrator(tools=[])
        self.task = AgentTask(
            role=AgentRole.RUNNER,
            task_description="搜索 GitHub 上 Shiro 密钥泄露项目",
        )

    def test_returns_last_text_when_final_response_is_empty(self):
        """模型最后一轮返回空内容但无工具调用时，回退到上一轮文本。"""
        fake_llm = _FakeLLM([
            # 第 1 轮：模型调用工具
            AIMessage(content="Let me search", tool_calls=[{
                "name": "search_web",
                "args": {"query": "shiro key leak"},
                "id": "call_1",
                "type": "tool_call",
            }]),
            # 第 2 轮：模型返回空内容，无工具调用（BUG 场景）
            AIMessage(content="", tool_calls=[]),
        ])
        self.orchestrator._llm = fake_llm
        self.orchestrator._tool_registry = {
            "search_web": MagicMock(invoke=lambda args: "找到 3 个仓库"),
        }

        result = self.orchestrator._run_role_agent(self.task)

        self.assertTrue(result.success)
        self.assertIn("Let me search", result.output,
                      "应回退到上一轮有文本的响应")

    def test_builds_summary_from_tool_log_when_no_text(self):
        """模型从未生成文本时，从工具调用日志构建摘要。"""
        fake_llm = _FakeLLM([
            # 第 1 轮：工具调用，无文本
            AIMessage(content="", tool_calls=[{
                "name": "fetch_web_page",
                "args": {"url": "https://example.com"},
                "id": "call_1",
                "type": "tool_call",
            }]),
            # 第 2 轮：空内容，无工具调用
            AIMessage(content="", tool_calls=[]),
        ])
        self.orchestrator._llm = fake_llm
        self.orchestrator._tool_registry = {
            "fetch_web_page": MagicMock(invoke=lambda args: "页面内容..."),
        }

        result = self.orchestrator._run_role_agent(self.task)

        self.assertTrue(result.success)
        self.assertIn("已执行", result.output,
                      "应从工具日志构建摘要")
        self.assertIn("fetch_web_page", result.output,
                      "摘要应包含工具名称")

    def test_normal_text_response_works(self):
        """正常文本响应直接返回。"""
        fake_llm = _FakeLLM([
            AIMessage(content="找到 5 个 Shiro 密钥泄露仓库：...", tool_calls=[]),
        ])
        self.orchestrator._llm = fake_llm

        result = self.orchestrator._run_role_agent(self.task)

        self.assertTrue(result.success)
        self.assertIn("找到 5 个", result.output)

    def test_uses_last_text_across_multiple_tool_rounds(self):
        """多轮工具调用中，使用最后一轮有文本的响应。"""
        fake_llm = _FakeLLM([
            # 第 1 轮：文本 + 工具调用
            AIMessage(content="先搜索一下", tool_calls=[{
                "name": "search_web",
                "args": {"query": "round1"},
                "id": "call_1",
                "type": "tool_call",
            }]),
            # 第 2 轮：仅工具调用，无文本
            AIMessage(content="", tool_calls=[{
                "name": "search_web",
                "args": {"query": "round2"},
                "id": "call_2",
                "type": "tool_call",
            }]),
            # 第 3 轮：空内容，无工具调用
            AIMessage(content="", tool_calls=[]),
        ])
        self.orchestrator._llm = fake_llm
        self.orchestrator._tool_registry = {
            "search_web": MagicMock(invoke=lambda args: "搜索结果..."),
        }

        result = self.orchestrator._run_role_agent(self.task)

        self.assertTrue(result.success)
        self.assertIn("先搜索一下", result.output,
                      "应回退到第1轮有文本的响应")


class SubtaskCompleteEventTestCase(unittest.TestCase):
    """测试 subtask_complete 事件包含输出摘要字段。"""

    def test_event_payload_has_summary_fields(self):
        events: list[dict] = []

        def handler(event_type: str, payload: object) -> None:
            if event_type == "subtask_complete":
                events.append(payload)

        orch = MultiAgentOrchestrator(
            tools=[],
            event_handler=handler,
        )
        # 两个子任务各需一个响应
        orch._llm = _FakeLLM([
            AIMessage(content="[执行者] 搜索完成：找到 Shiro 相关仓库", tool_calls=[]),
            AIMessage(content="[分析者] 分析完成：共 3 个密钥泄露仓库", tool_calls=[]),
        ])

        orch._execute_plan(orch._default_plan("测试任务"))

        self.assertGreaterEqual(len(events), 2, "至少应有 2 个子任务完成事件")

        for evt in events:
            self.assertIn("output_summary", evt)
            self.assertIn("output_length", evt)
            self.assertGreater(evt["output_length"], 0,
                               f"{evt['role']} 输出长度应 > 0")


class TokenBaselineTestCase(unittest.TestCase):
    """测试 token 基线在工具回路中不会被重置。"""

    def setUp(self):
        self.renderer = CliRenderer()

    def test_baseline_not_reset_on_second_begin_stream(self):
        """多次 begin_response_stream 调用不会重置基线。"""
        self.renderer._cumulative_input_tokens = 1000
        self.renderer._cumulative_output_tokens = 500

        # 第一次调用 - 设置基线
        self.renderer.begin_response_stream()
        self.assertEqual(self.renderer._turn_baseline_input, 1000)
        self.assertTrue(self.renderer._turn_baseline_set)

        # 模拟工具执行期间累积的 token
        self.renderer._cumulative_input_tokens = 5000

        # 第二次调用 - 不应重置基线（工具回路中的第二次模型调用）
        self.renderer.begin_response_stream()
        self.assertEqual(self.renderer._turn_baseline_input, 1000,
                         "基线不应被第二次 begin_response_stream 重置")

        self.renderer.end_response_stream("done", False)

    def test_print_token_usage_resets_baseline_flag(self):
        """TURN_END 后 _turn_baseline_set 应重置为 False。"""
        self.renderer._cumulative_input_tokens = 1000
        self.renderer._cumulative_output_tokens = 500
        self.renderer.begin_response_stream()
        self.assertTrue(self.renderer._turn_baseline_set)

        self.renderer.print_token_usage({
            "input_tokens": 5000, "output_tokens": 800, "total_tokens": 5800,
        })
        self.assertFalse(self.renderer._turn_baseline_set,
                         "TURN_END 后基线标记应重置")

    def test_round_calculation_uses_original_baseline(self):
        """本轮用量 = 累计 - 原始基线（不受工具回路中重置影响）。"""
        self.renderer._cumulative_input_tokens = 3864
        self.renderer._cumulative_output_tokens = 196

        # 模拟完整一轮（含多次 begin/end）
        self.renderer.begin_response_stream()
        self.renderer._cumulative_input_tokens = 50000  # API 精确值前的估算
        self.renderer.end_response_stream("text", True)
        self.renderer.begin_response_stream()  # 工具回路第二次
        self.renderer.end_response_stream("final", False)

        self.renderer.print_token_usage({
            "input_tokens": 25000,
            "output_tokens": 800,
            "total_tokens": 25800,
        })

        # 累计 = 基线 + API 精确值
        self.assertEqual(self.renderer._cumulative_input_tokens, 3864 + 25000)
        self.assertEqual(self.renderer._cumulative_output_tokens, 196 + 800)

    def test_add_tool_result_tokens_updates_live_counter(self):
        """工具结果实时更新 live 计数器。"""
        self.renderer._cumulative_input_tokens = 1000
        initial_in = self.renderer._cumulative_input_tokens

        self.renderer.add_tool_result_tokens("x" * 3000)  # ~1000 tokens

        self.assertGreater(
            self.renderer._cumulative_input_tokens, initial_in,
            "工具结果应增加输入 token 估算",
        )


class TokenEstimationTestCase(unittest.TestCase):
    """测试 token 估算函数。"""

    def test_estimate_empty(self):
        self.assertEqual(_estimate_tokens_from_text(""), 0)

    def test_estimate_short(self):
        tokens = _estimate_tokens_from_text("hello")  # 5 chars
        self.assertEqual(tokens, max(1, 5 // 3))

    def test_estimate_long(self):
        """~3 字/token 的混合估算。"""
        text = "中" * 300  # 300 Chinese chars
        tokens = _estimate_tokens_from_text(text)
        self.assertEqual(tokens, 100)  # 300 // 3

    def test_no_negative(self):
        """即使输入为 1 字符，也应返回 >= 1。"""
        self.assertGreaterEqual(_estimate_tokens_from_text("a"), 1)


class ComplexityDetectionTestCase(unittest.TestCase):
    """测试复杂度检测为纯结构分析，不含领域关键词。"""

    def test_simple_greeting_is_not_complex(self):
        self.assertFalse(_detect_task_complexity("你好"))
        self.assertFalse(_detect_task_complexity("1+1等于几"))
        self.assertFalse(_detect_task_complexity("今天天气怎么样"))

    def test_multiple_questions_is_complex(self):
        self.assertTrue(_detect_task_complexity("A是什么？B呢？"))

    def test_multiple_sentences_is_complex(self):
        self.assertTrue(_detect_task_complexity(
            "先扫描端口。然后分析结果。最后写报告。"
        ))

    def test_coordination_markers_trigger_complex(self):
        """并列/时序连接词触发复杂检测。"""
        self.assertTrue(_detect_task_complexity(
            "分析本机漏洞并写一份报告"
        ))
        self.assertTrue(_detect_task_complexity(
            "先收集信息，然后扫描端口"
        ))
        self.assertTrue(_detect_task_complexity(
            "扫描端口并且识别服务版本"
        ))

    def test_numbered_items_trigger_complex(self):
        self.assertTrue(_detect_task_complexity(
            "第一步：信息收集。第二步：漏洞扫描。"
        ))
        self.assertTrue(_detect_task_complexity(
            "1. 扫描端口 2. 检测服务 3. 查找漏洞"
        ))

    def test_single_action_with_keyword_not_forced(self):
        """单个动作即使涉及安全领域，也不应仅凭关键词判定为复杂。"""
        # 这些是单句请求，无结构标记
        self.assertFalse(_detect_task_complexity(
            "分析这个CVE漏洞的影响范围"
        ))
        self.assertFalse(_detect_task_complexity(
            "Pwn2Own 2026发生了什么"
        ))

    def test_negation_not_trigger(self):
        """'并不' 不应被视为并列连接词 '并'。"""
        self.assertFalse(_detect_task_complexity("并不想要这个"))

    def test_long_dense_text_is_complex(self):
        self.assertTrue(_detect_task_complexity(
            "请详细说明这个问题的来龙去脉，包括背景、原因、"
            "影响范围以及可能的解决方案，并给出具体的实施步骤"
        ))


class ToolResultLiveCounterTestCase(unittest.TestCase):
    """测试工具结果实时更新 live token 计数器。"""

    def setUp(self):
        self.renderer = CliRenderer()

    def test_add_tool_result_tokens_with_live(self):
        self.renderer._cumulative_input_tokens = 1000
        self.renderer.begin_response_stream()

        before = self.renderer._cumulative_input_tokens
        self.renderer.add_tool_result_tokens("a" * 9000)  # ~3000 tokens
        after = self.renderer._cumulative_input_tokens

        expected = before + max(1, 9000 // 3)
        self.assertEqual(after, expected)

        self.renderer.end_response_stream("done", False)

    def test_multiple_tool_results_accumulate(self):
        self.renderer._cumulative_input_tokens = 0
        self.renderer.begin_response_stream()

        self.renderer.add_tool_result_tokens("x" * 3000)  # +1000
        self.renderer.add_tool_result_tokens("y" * 6000)  # +2000

        self.assertEqual(
            self.renderer._cumulative_input_tokens,
            3000,  # 0 + 1000 + 2000
        )

        self.renderer.end_response_stream("done", False)


if __name__ == "__main__":
    unittest.main()
