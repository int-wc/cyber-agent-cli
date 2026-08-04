"""业务原语解析与原语链利用核心库测试。

覆盖：原语行解析、攻击面前匹配、原语链联动推理、数据模型枚举。
"""

from __future__ import annotations

import unittest

from cyber_agent.agent.primitives import (
    AttrTarget,
    BusinessPrimitive,
    PrimitiveEndpoint,
    build_hint_report,
    load_chains,
    load_surfaces,
    link,
    match_surfaces,
    parse_line,
    parse_text,
    serialize_endpoints,
)

# 样例：原语判定行（模拟 Phase2 深度分析输出）
SAMPLE_TEXT = """\
- /api/translateUrl POST business_attr=transfer attr_target=remote_url attr_reason=白名单URL拉取 params={url=} risk=SSRF
- /api/import_file POST business_attr=write_file attr_target=local_fs attr_reason=文件上传导入 params={file=} risk=上传
- /api/user/info GET business_attr=query_data attr_target=db attr_reason=用户详情 params={id=} risk=IDOR
- /api/login POST business_attr=auth attr_target=user_input attr_reason=登录签发 params={password=} risk=弱口令
- /api/render POST business_attr=exec_code attr_target=template attr_reason=模板渲染 params={tpl=} risk=SSTI
"""


class ParseLineTestCase(unittest.TestCase):
    """原语判定行解析。"""

    def test_parses_attr_target_params(self) -> None:
        ep = parse_line("- /api/translateUrl POST business_attr=transfer attr_target=remote_url")
        self.assertIsNotNone(ep)
        assert ep is not None
        self.assertEqual(ep.endpoint, "/api/translateUrl")
        self.assertEqual(ep.method, "POST")
        self.assertEqual(ep.business_attr, BusinessPrimitive.TRANSFER)
        self.assertEqual(ep.attr_target, AttrTarget.REMOTE_URL)

    def test_parses_unknown_attr_as_none(self) -> None:
        """未知原语名不抛错，business_attr 置 None，attr_key 回退为 unknown。"""
        ep = parse_line("- /x GET business_attr=whatever")
        self.assertIsNotNone(ep)
        assert ep is not None
        self.assertIsNone(ep.business_attr)
        self.assertEqual(ep.attr_key, "unknown")

    def test_ignores_non_attr_lines(self) -> None:
        """非原语判定行返回 None，不误解析。"""
        self.assertIsNone(parse_line("普通分析文本，没有原语标记"))
        self.assertIsNone(parse_line(""))

    def test_parse_text_counts(self) -> None:
        eps = parse_text(SAMPLE_TEXT)
        self.assertEqual(len(eps), 5)

    def test_serialize_roundtrip(self) -> None:
        """序列化后再解析，原语与端点保持一致。"""
        eps = parse_text(SAMPLE_TEXT)
        text = serialize_endpoints(eps)
        reparsed = parse_text(text)
        self.assertEqual(len(reparsed), len(eps))
        self.assertEqual(
            [(e.endpoint, e.attr_key) for e in reparsed],
            [(e.endpoint, e.attr_key) for e in eps],
        )


class SurfaceMatchTestCase(unittest.TestCase):
    """攻击面前匹配。"""

    def setUp(self) -> None:
        self.eps = parse_text(SAMPLE_TEXT)
        self.surfaces = load_surfaces()

    def test_all_surfaces_load(self) -> None:
        self.assertGreaterEqual(len(self.surfaces), 8)

    def test_transfer_endpoint_matches_url_fetch(self) -> None:
        ep = next(e for e in self.eps if e.attr_key == "transfer")
        matches = match_surfaces(ep, self.surfaces)
        self.assertGreater(len(matches), 0)
        top = matches[0]
        self.assertEqual(top.surface.surface_id, "sf_url_fetch_echo")
        self.assertIn("SSRF(回显)", top.surface.base_primitives)

    def test_primitive_hit_ranks_first(self) -> None:
        """原语命中（端点原语 ∈ 攻击面 primitives）应排在信号匹配前。"""
        ep = next(e for e in self.eps if e.attr_key == "exec_code")
        matches = match_surfaces(ep, self.surfaces)
        self.assertGreater(len(matches), 0)
        self.assertEqual(matches[0].surface.surface_id, "sf_template_render")

    def test_hint_report_shape(self) -> None:
        report = build_hint_report(self.eps, self.surfaces)
        self.assertEqual(report["total_endpoints"], 5)
        self.assertEqual(len(report["hints"]), 5)
        self.assertGreater(report["matched_count"], 0)


class ChainLinkTestCase(unittest.TestCase):
    """原语链联动推理。"""

    def setUp(self) -> None:
        self.eps = parse_text(SAMPLE_TEXT)
        self.chains = load_chains()

    def test_chains_load(self) -> None:
        self.assertGreaterEqual(len(self.chains), 8)

    def test_transfer_auth_chain_matches(self) -> None:
        """transfer + auth 齐备 → SSRF→认证绕过链成为候选。"""
        cands = link(self.eps, self.chains)
        by_id = {c.chain_id: c for c in cands}
        self.assertIn("ch_ssrf_to_auth", by_id)
        self.assertIn("/api/translateUrl", by_id["ch_ssrf_to_auth"].matched_endpoints["transfer"])
        self.assertIn("/api/login", by_id["ch_ssrf_to_auth"].matched_endpoints["auth"])

    def test_incomplete_chain_skipped(self) -> None:
        """read_file 端点缺失 → read→rce 链不成为候选。"""
        cands = link(self.eps, self.chains)
        by_id = {c.chain_id: c for c in cands}
        self.assertNotIn("ch_read_to_rce", by_id)

    def test_empty_endpoints_no_candidates(self) -> None:
        self.assertEqual(link([], self.chains), [])


class EnumTestCase(unittest.TestCase):
    """枚举宽松解析。"""

    def test_primitive_parse_case_insensitive(self) -> None:
        self.assertEqual(BusinessPrimitive.parse("Read_File"), BusinessPrimitive.READ_FILE)
        self.assertIsNone(BusinessPrimitive.parse(""))

    def test_target_parse(self) -> None:
        self.assertEqual(AttrTarget.parse("local_fs"), AttrTarget.LOCAL_FS)
        self.assertIsNone(AttrTarget.parse("nowhere"))

    def test_endpoint_to_dict(self) -> None:
        ep = PrimitiveEndpoint(
            endpoint="/x",
            method="POST",
            business_attr=BusinessPrimitive.WRITE_FILE,
            attr_target=AttrTarget.LOCAL_FS,
        )
        d = ep.to_dict()
        self.assertEqual(d["business_attr"], "write_file")
        self.assertEqual(d["attr_target"], "local_fs")


if __name__ == "__main__":
    unittest.main()
