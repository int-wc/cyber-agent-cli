"""trace 沉淀脚本核心逻辑测试。

覆盖：verdict JSON 解析（含围栏）、目标提取、占位目标过滤、
无执行过滤、已入库链实例沉淀、自创链候选识别。
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

# 以源码方式加载 scripts/sediment_primitive_instances.py
_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "sediment_primitive_instances.py"
_spec = importlib.util.spec_from_file_location("sediment_primitive_instances", _SCRIPT_PATH)
sediment = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(sediment)

VERDICT_JSON = """```json
{
  "verdict": "execute",
  "top_chains": [
    {"chain_id": "ch_ssrf_to_auth", "verdict": "execute", "priority": "high",
     "exploitation_plan": "只读验证 SSRF 可达性"},
    {"chain_id": "ch_model_self_made", "verdict": "execute", "priority": "medium",
     "exploitation_plan": "探测自创链"}
  ]
}
```"""


def _make_trace(*, has_tool_call: bool, target: str, verdict_detail: str = VERDICT_JSON) -> Path:
    """构造一份临时 trace 文件。"""
    events = [
        {"event": "pipeline_start", "detail": f"对补天收录厂商【{target}】做漏洞挖掘"},
        {"event": "primitive_verdict", "detail": verdict_detail},
    ]
    if has_tool_call:
        events.append({"event": "tool_call", "metadata": {"args": {"command": "curl -s https://x/api"}}})
    fd, tmp = tempfile.mkstemp(suffix=".trace.json", prefix="trace-test-")
    import os
    os.close(fd)
    Path(tmp).write_text(json.dumps(events, ensure_ascii=False), encoding="utf-8")
    return Path(tmp)


class VerdictParsingTestCase(unittest.TestCase):
    """verdict JSON 解析。"""

    def test_strip_fence(self) -> None:
        self.assertIn('"verdict"', sediment._strip_fence(VERDICT_JSON))
        self.assertEqual(sediment._strip_fence('{"a": 1}'), '{"a": 1}')

    def test_parse_verdict_top_chains(self) -> None:
        chains = sediment._parse_verdict(VERDICT_JSON)
        self.assertEqual(len(chains), 2)
        self.assertEqual(chains[0]["chain_id"], "ch_ssrf_to_auth")
        self.assertEqual(chains[1]["chain_id"], "ch_model_self_made")

    def test_parse_verdict_bare_json(self) -> None:
        chains = sediment._parse_verdict('{"verdict":"execute","top_chains":[{"chain_id":"a","verdict":"execute"}]}')
        self.assertEqual(len(chains), 1)
        self.assertEqual(chains[0]["chain_id"], "a")

    def test_parse_verdict_garbage(self) -> None:
        self.assertEqual(sediment._parse_verdict("no json here"), [])

    def test_extract_target_braced(self) -> None:
        self.assertEqual(sediment._extract_target("对【理想汽车】的 api 做挖掘"), "理想汽车")

    def test_extract_target_domain(self) -> None:
        self.assertEqual(
            sediment._extract_target("测试 api-app.lixiang.com 漏洞"),
            "api-app.lixiang.com",
        )


class SedimentFilterTestCase(unittest.TestCase):
    """沉淀过滤：无执行 / 占位目标不沉淀。"""

    def test_no_tool_call_skipped(self) -> None:
        """无 tool_call 的 trace（纯思考/测试）不沉淀。"""
        trace = _make_trace(has_tool_call=False, target="理想汽车")
        self.addCleanup(lambda: trace.unlink(missing_ok=True))
        added, sedimented, candidates = sediment.sediment_trace_file(
            trace, known_ids={"ch_ssrf_to_auth"}
        )
        self.assertEqual(added, 0)
        self.assertEqual(sedimented, [])
        self.assertEqual(candidates, [])

    def test_placeholder_target_skipped(self) -> None:
        """测试占位目标（目标A）不沉淀。"""
        trace = _make_trace(has_tool_call=True, target="目标A")
        self.addCleanup(lambda: trace.unlink(missing_ok=True))
        added, sedimented, candidates = sediment.sediment_trace_file(
            trace, known_ids={"ch_ssrf_to_auth"}
        )
        self.assertEqual(added, 0)
        self.assertEqual(candidates, [])


class SedimentWriteTestCase(unittest.TestCase):
    """沉淀写入：已入库链 → 实例；自创链 → 候选。"""

    def test_known_chain_records_instance(self) -> None:
        trace = _make_trace(has_tool_call=True, target="理想汽车")
        self.addCleanup(lambda: trace.unlink(missing_ok=True))
        added, sedimented, candidates = sediment.sediment_trace_file(
            trace,
            known_ids={"ch_ssrf_to_auth"},
            dry_run=True,
        )
        self.assertEqual(added, 1)
        self.assertEqual(sedimented[0]["chain_id"], "ch_ssrf_to_auth")
        self.assertEqual(sedimented[0]["target"], "理想汽车")
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["chain_id"], "ch_model_self_made")

    def test_only_chain_filter(self) -> None:
        trace = _make_trace(has_tool_call=True, target="理想汽车")
        self.addCleanup(lambda: trace.unlink(missing_ok=True))
        added, _, candidates = sediment.sediment_trace_file(
            trace,
            known_ids={"ch_ssrf_to_auth"},
            dry_run=True,
            only_chain="ch_ssrf_to_auth",
        )
        self.assertEqual(added, 1)
        self.assertEqual(candidates, [])


if __name__ == "__main__":
    unittest.main()
