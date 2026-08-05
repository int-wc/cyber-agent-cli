"""原语工作流管线测试：管线选择、原语解析、链裁决与执行闭环。

用可编程 fake LLM 驱动全流程，验证：
- 管线模式选择（显式配置 / 语义自动判定）
- Phase1 原语解析 → 攻击面扩散 → 链跃迁 → 链裁决
- Phase2 链执行闭环（决策者→执行者→审计者→反思者）
- 聚合总结输出
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from cyber_agent.agent.primitive_pipeline import (
    PRIMITIVE_ROLE_PROMPTS,
    PrimitiveWorkflowPipeline,
)
from cyber_agent.agent.roles import AgentRole
from cyber_agent.cli.app_multi_agent import (
    _detect_primitive_workflow,
    _select_pipeline_mode,
)

# ═══ 可编程 fake LLM ═══

ANALYST_JSON = """{"target_summary": "目标A",
  "primitives": [
    {"endpoint": "/api/translateUrl", "method": "POST", "business_attr": "transfer",
     "attr_target": "remote_url", "attr_reason": "白名单URL拉取", "params": {"url": ""}, "risk": "SSRF"},
    {"endpoint": "/api/login", "method": "POST", "business_attr": "auth",
     "attr_target": "user_input", "attr_reason": "登录签发", "params": {"password": ""}, "risk": "弱口令"},
    {"endpoint": "/api/user/info", "method": "GET", "business_attr": "query_data",
     "attr_target": "db", "attr_reason": "用户详情", "params": {"id": ""}, "risk": "IDOR"}]}"""

DIFFUSER_JSON = """{"attack_plan": [
  {"endpoint": "/api/translateUrl", "business_attr": "transfer", "priority": "high",
   "attack_primitives": ["SSRF(回显)", "remote→local白名单绕过"],
   "test_vectors": ["curl -s -X POST -d 'url=http://169.254.169.254/latest/meta-data/' /api/translateUrl"]}]}"""

JUMPER_JSON = """{"chains": [
  {"chain_id": "ch_ssrf_to_auth", "validated": true, "priority": "high",
   "escalation": "SSRF打认证服务→token篡改→账户接管",
   "key_endpoints": {"transfer": ["/api/translateUrl"], "auth": ["/api/login"]}}],
 "novel_chains": []}"""

REFLECTOR_JSON = """{"verdict": "execute",
  "top_chains": [
    {"chain_id": "ch_ssrf_to_auth", "verdict": "execute", "priority": "high",
     "exploitation_plan": "先只读验证 transfer 端点 SSRF 可达性，再验证 auth 端点"}]}"""

DECISION_JSON = """{"reasoning": "先验证SSRF端点可达性",
  "subtasks": [
    {"role": "runner", "task_description": "curl 只读验证 /api/translateUrl 的 SSRF 可达性"}]}"""

THINKER_JSON = """{"reasoning": "选择关键子任务", "selected_indices": [0],
  "additional_context": "", "concerns": ""}"""


class _FakeLLM:
    """可编程返回 JSON 的 fake LLM。"""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.call_count = 0

    def invoke(self, messages):
        if self.call_count < len(self._responses):
            resp = self._responses[self.call_count]
            self.call_count += 1
            return MagicMock(content=resp)
        return MagicMock(content="{}")


class _FakeRunner:
    """满足管线主会话的最小 runner 替身。"""

    def __init__(self) -> None:
        self.history = []
        self.tools = []

    def get_history_snapshot(self) -> list:
        return list(self.history)

    def get_turn_count(self) -> int:
        return len(self.history)


class _FakeRenderer:
    console = MagicMock()
    console.print = MagicMock()

    def print_markdown(self, text: str) -> None:
        pass

    def add_token_usage(self, *args, **kwargs) -> None:
        pass


def _make_pipeline(**kwargs) -> PrimitiveWorkflowPipeline:
    runner = kwargs.get("runner") or _FakeRunner()
    runtime = kwargs.get("runtime_context") or {}
    return PrimitiveWorkflowPipeline(
        runner=runner,
        runtime_context=dict(runtime),
        renderer=kwargs.get("renderer") or _FakeRenderer(),
        event_handler=kwargs.get("event_handler"),
    )


# ═══ 管线模式选择 ═══


class PipelineModeTestCase(unittest.TestCase):
    """管线模式选择。"""

    def test_detect_primitive_keywords(self) -> None:
        self.assertTrue(_detect_primitive_workflow("挖掘目标 SSRF 漏洞"))
        self.assertTrue(_detect_primitive_workflow("分析这个 API 的业务原语和攻击面"))
        self.assertFalse(_detect_primitive_workflow("帮我整理周报"))

    def test_select_explicit_primitive(self) -> None:
        mode = _select_pipeline_mode({"pipeline_mode": "primitive"}, "随便什么")
        self.assertEqual(mode, "primitive")

    def test_select_explicit_four_pillar(self) -> None:
        mode = _select_pipeline_mode({"pipeline_mode": "four_pillar"}, "SSRF 挖洞")
        self.assertEqual(mode, "four_pillar")

    def test_select_auto_by_semantics(self) -> None:
        self.assertEqual(_select_pipeline_mode({}, "挖掘 SSRF"), "primitive")
        self.assertEqual(_select_pipeline_mode({}, "整理文档"), "four_pillar")


# ═══ 静态解析工具 ═══


class ParseHelpersTestCase(unittest.TestCase):
    """原语解析与 JSON 列表提取。"""

    def test_parse_primitive_endpoints(self) -> None:
        eps = PrimitiveWorkflowPipeline._parse_primitive_endpoints(ANALYST_JSON)
        self.assertEqual(len(eps), 3)
        attrs = {e.attr_key for e in eps}
        self.assertEqual(attrs, {"transfer", "auth", "query_data"})
        ep = next(e for e in eps if e.attr_key == "transfer")
        self.assertEqual(ep.endpoint, "/api/translateUrl")
        self.assertEqual(ep.attr_target.value, "remote_url")

    def test_parse_primitive_endpoints_empty(self) -> None:
        self.assertEqual(PrimitiveWorkflowPipeline._parse_primitive_endpoints("{}"), [])

    def test_extract_json_list(self) -> None:
        items = PrimitiveWorkflowPipeline._extract_json_list(REFLECTOR_JSON, "top_chains")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["chain_id"], "ch_ssrf_to_auth")

    def test_extract_json_list_missing_key(self) -> None:
        self.assertEqual(PrimitiveWorkflowPipeline._extract_json_list("{}", "nope"), [])

    def test_extract_subtasks_from_truncated_json(self) -> None:
        """回归：模型长输出截断时，_extract_subtasks 仍能恢复完整子任务。

        复现真实场景：决策者一次输出多个带 curl 的子任务，末尾 JSON 数组被截断，
        _parse_json 整体失败返回 {}，旧逻辑误判"未产出子任务"。
        """
        truncated = (
            '{"reasoning":"分解为2个子任务","subtasks":['
            '{"role":"runner","task_description":"curl -s https://a/b"},'
            '{"role":"runner","task_description":"curl -s https://a/c"}'
            # 末尾被截断，无 ] 闭合
        )
        raw = PrimitiveWorkflowPipeline._extract_subtasks(truncated)
        self.assertEqual(len(raw), 2)
        self.assertEqual(raw[0]["task_description"], "curl -s https://a/b")
        self.assertEqual(raw[1]["task_description"], "curl -s https://a/c")

    def test_extract_subtasks_full_json(self) -> None:
        raw = PrimitiveWorkflowPipeline._extract_subtasks(
            '{"subtasks":[{"role":"runner","task_description":"测A"}]}'
        )
        self.assertEqual(len(raw), 1)
        self.assertEqual(raw[0]["task_description"], "测A")


# ═══ 全流程（fake LLM 驱动）═══


class PrimitivePipelineFlowTestCase(unittest.TestCase):
    """原语工作流全流程。"""

    def _build(self, responses: list[str]):
        fake_llm = _FakeLLM(responses)
        runner = _FakeRunner()
        runtime = {"service_name": "deepseek", "auto_decision": True}
        pipe = _make_pipeline(runner=runner, runtime_context=runtime)
        # 替换 LLM 获取与超时调用，直接走 fake
        pipe._get_llm = MagicMock(return_value=fake_llm)
        return pipe, fake_llm

    def test_phase1_primitive_semantics(self) -> None:
        """Phase1 四柱按原语语义产出端点/攻击面/候选链/裁决。"""
        pipe, _ = self._build([ANALYST_JSON, DIFFUSER_JSON, JUMPER_JSON, REFLECTOR_JSON])
        # 直接调用 Phase1 相关：解析者输出 → 数据层程序化注入
        pipe._primitive_endpoints = pipe._parse_primitive_endpoints(ANALYST_JSON)
        from cyber_agent.agent.primitives import build_hint_report, build_link_report
        hint = build_hint_report(pipe._primitive_endpoints)
        chain = build_link_report(pipe._primitive_endpoints)
        self.assertGreater(hint["matched_count"], 0)
        # SSRF→认证绕过链应成为候选
        chain_ids = {c["chain_id"] for c in chain["candidates"]}
        self.assertIn("ch_ssrf_to_auth", chain_ids)

    def test_full_run_uses_primitive_pipeline(self) -> None:
        """完整 run：原语思考 → 链执行 → 聚合，不触碰四柱路径。"""
        responses = [
            ANALYST_JSON,   # 原语解析者
            DIFFUSER_JSON,  # 攻击面扩散者
            JUMPER_JSON,    # 链跃迁者
            REFLECTOR_JSON, # 链裁决者
            DECISION_JSON,  # 决策者
            THINKER_JSON,   # 思考者（auto_select）
            "结束协作",      # 反思者（第一轮即结束）
        ]
        pipe, fake_llm = self._build(responses)
        executed_prompt: list[str] = []

        def _fake_subtask(prompt: str, role_label: str, desc: str) -> str:
            executed_prompt.append(prompt)
            return "✅ /api/translateUrl 返回200，响应含内网元数据"

        with patch.object(pipe, "_run_subtask_with_escalating_timeout", side_effect=_fake_subtask):
            pipe.run("对目标A做SSRF漏洞挖掘", auto_decision=True)

        # 原语语义阶段确实走了
        self.assertGreater(len(pipe._primitive_endpoints), 0)
        self.assertGreater(len(pipe._chain_verdicts), 0)
        self.assertIn("ch_ssrf_to_auth", {v.get("chain_id") for v in pipe._chain_verdicts})
        # 链执行闭环真的执行了子任务（回归：task_description 字段必须透传给思考者/执行者）
        self.assertEqual(len(executed_prompt), 1, "决策者子任务应被选中并执行")
        self.assertIn("SSRF", executed_prompt[0])
        self.assertIn("curl", executed_prompt[0])
        # 最终总结写回
        self.assertIn("原语工作流", pipe._final_summary)

    def test_role_prompts_are_primitive_semantic(self) -> None:
        """原语角色提示词替换四柱提示词。"""
        self.assertIn("原语解析者", PRIMITIVE_ROLE_PROMPTS[AgentRole.ANALYST])
        self.assertIn("攻击面扩散者", PRIMITIVE_ROLE_PROMPTS[AgentRole.DIFFUSER])
        self.assertIn("链跃迁者", PRIMITIVE_ROLE_PROMPTS[AgentRole.JUMPER])
        self.assertIn("链裁决者", PRIMITIVE_ROLE_PROMPTS[AgentRole.REFLECTOR])


if __name__ == "__main__":
    unittest.main()
