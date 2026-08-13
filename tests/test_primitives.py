"""业务原语解析与原语链利用核心库测试。

覆盖：原语行解析、攻击面前匹配、原语链联动推理、数据模型枚举、库文件回写。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

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
    record_chain_instance,
    record_surface_instance,
    serialize_endpoints,
    upsert_chain,
    upsert_surface,
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


# ── 库文件回写（实例积累回路）──

def _make_temp_chains_file() -> Path:
    """创建一份临时的链库文件（复制真实数据，避免污染仓库数据）。"""
    raw = json.loads(
        Path("src/cyber_agent/agent/primitives/data/primitive-chains.json")
        .read_text(encoding="utf-8")
    )
    fd, tmp = tempfile.mkstemp(suffix=".json", prefix="chains-test-")
    import os
    os.close(fd)
    Path(tmp).write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    return Path(tmp)


def _make_temp_surfaces_file() -> Path:
    """创建一份临时的攻击面库文件（复制真实数据，避免污染仓库数据）。"""
    raw = json.loads(
        Path("src/cyber_agent/agent/primitives/data/attack_surfaces.json")
        .read_text(encoding="utf-8")
    )
    fd, tmp = tempfile.mkstemp(suffix=".json", prefix="surfaces-test-")
    import os
    os.close(fd)
    Path(tmp).write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    return Path(tmp)


class ChainWritebackTestCase(unittest.TestCase):
    """链库实例回写：追加、去重、限容、新增链。"""

    def setUp(self) -> None:
        self.path = _make_temp_chains_file()
        self.addCleanup(lambda: self.path.unlink(missing_ok=True))

    def _instance(self, target: str, endpoint: str) -> dict:
        return {
            "target": target,
            "endpoint": endpoint,
            "method": "GET",
            "verdict": "execute",
            "priority": "high",
            "finding": "未授权访问确认",
            "date": "2026-08-05",
        }

    def test_record_appends_instance(self) -> None:
        ok = record_chain_instance("ch_ssrf_to_auth", self._instance("厂商A", "/api/t"), path=self.path)
        self.assertTrue(ok)
        chains = load_chains(self.path)
        by_id = {c.chain_id: c for c in chains}
        self.assertEqual(len(by_id["ch_ssrf_to_auth"].instances), 1)

    def test_record_dedup_same_target_endpoint(self) -> None:
        record_chain_instance("ch_ssrf_to_auth", self._instance("厂商A", "/api/t"), path=self.path)
        record_chain_instance("ch_ssrf_to_auth", self._instance("厂商A", "/api/t"), path=self.path)
        chains = load_chains(self.path)
        by_id = {c.chain_id: c for c in chains}
        self.assertEqual(len(by_id["ch_ssrf_to_auth"].instances), 1)

    def test_record_unknown_chain_false(self) -> None:
        self.assertFalse(record_chain_instance("ch_does_not_exist", self._instance("A", "/x"), path=self.path))

    def test_record_cap_instances(self) -> None:
        from cyber_agent.agent.primitives import MAX_CHAIN_INSTANCES
        for i in range(MAX_CHAIN_INSTANCES + 10):
            inst = self._instance(f"厂商{i}", f"/api/{i}")
            self.assertTrue(record_chain_instance("ch_ssrf_to_auth", inst, path=self.path))
        chains = load_chains(self.path)
        by_id = {c.chain_id: c for c in chains}
        self.assertLessEqual(len(by_id["ch_ssrf_to_auth"].instances), MAX_CHAIN_INSTANCES)

    def test_upsert_new_chain(self) -> None:
        added, chain_id = upsert_chain(
            {
                "id": "ch_apisix_unauth",
                "name": "APISIX admin 未授权",
                "primitives": ["query_data"],
                "logic": "APISIX 控制面 admin API 未鉴权 → 读取路由/上游/消费者配置",
                "gain": "未授权读取 → 配置泄露/服务接管",
            },
            path=self.path,
        )
        self.assertTrue(added)
        self.assertEqual(chain_id, "ch_apisix_unauth")
        chains = load_chains(self.path)
        self.assertIn("ch_apisix_unauth", {c.chain_id for c in chains})

    def test_upsert_existing_keeps_instances(self) -> None:
        record_chain_instance("ch_ssrf_to_auth", self._instance("厂商A", "/api/t"), path=self.path)
        added, _ = upsert_chain(
            {"id": "ch_ssrf_to_auth", "name": "改名", "primitives": ["transfer", "auth"], "logic": "新逻辑", "gain": "新增益"},
            path=self.path,
        )
        self.assertFalse(added)
        chains = load_chains(self.path)
        by_id = {c.chain_id: c for c in chains}
        self.assertEqual(by_id["ch_ssrf_to_auth"].name, "改名")
        self.assertEqual(len(by_id["ch_ssrf_to_auth"].instances), 1)


class SurfaceWritebackTestCase(unittest.TestCase):
    """攻击面库实例回写：追加、去重、新增条目。"""

    def setUp(self) -> None:
        self.path = _make_temp_surfaces_file()
        self.addCleanup(lambda: self.path.unlink(missing_ok=True))

    def test_record_appends_instance(self) -> None:
        ok = record_surface_instance(
            "sf_url_fetch_echo",
            {"target": "厂商A", "endpoint": "/fetch", "method": "POST", "finding": "回显SSRF"},
            path=self.path,
        )
        self.assertTrue(ok)
        surfaces = load_surfaces(self.path)
        by_id = {s.surface_id: s for s in surfaces}
        self.assertEqual(len(by_id["sf_url_fetch_echo"].instances), 1)

    def test_record_dedup(self) -> None:
        inst = {"target": "厂商A", "endpoint": "/fetch", "method": "POST", "finding": "回显SSRF"}
        record_surface_instance("sf_url_fetch_echo", inst, path=self.path)
        record_surface_instance("sf_url_fetch_echo", inst, path=self.path)
        surfaces = load_surfaces(self.path)
        by_id = {s.surface_id: s for s in surfaces}
        self.assertEqual(len(by_id["sf_url_fetch_echo"].instances), 1)

    def test_record_unknown_surface_false(self) -> None:
        self.assertFalse(
            record_surface_instance("sf_missing", {"target": "A", "endpoint": "/x"}, path=self.path)
        )

    def test_upsert_new_surface(self) -> None:
        added, sid = upsert_surface(
            {
                "id": "sf_test_fake_component",
                "name": "测试用组件控制面",
                "signals": ["testfake"],
                "primitives": ["query_data"],
                "targets": ["db"],
                "base_primitives": ["未授权读取配置"],
                "risk": "",
            },
            path=self.path,
        )
        self.assertTrue(added)
        self.assertEqual(sid, "sf_test_fake_component")
        surfaces = load_surfaces(self.path)
        self.assertIn("sf_test_fake_component", {s.surface_id for s in surfaces})


class ExpandedLibraryTestCase(unittest.TestCase):
    """批次C扩充：组件控制面/认证绕过/API文档链与攻击面可加载、可匹配。"""

    def test_new_chains_loaded(self) -> None:
        ids = {c.chain_id for c in load_chains()}
        for expect in (
            "ch_component_admin_unauth",
            "ch_auth_bypass_matrix",
            "ch_api_docs_exposure",
            "ch_ssrf_to_internal_admin",
        ):
            self.assertIn(expect, ids)

    def test_new_surfaces_loaded(self) -> None:
        ids = {s.surface_id for s in load_surfaces()}
        self.assertIn("sf_component_admin", ids)
        self.assertIn("sf_auth_bypass", ids)

    def test_component_surface_matches_apisix_signal(self) -> None:
        """APISIX admin 端点应命中组件控制面攻击面。"""
        ep = parse_line(
            "- /apisix/admin/routes GET business_attr=query_data attr_target=db attr_reason=组件控制面风险=未授权"
        )
        assert ep is not None
        matches = match_surfaces(ep, load_surfaces())
        self.assertGreater(len(matches), 0)
        self.assertEqual(matches[0].surface.surface_id, "sf_component_admin")

    def test_component_surface_matches_extended_signals(self) -> None:
        """批次D：扩充的组件信号（hadoop/sangfor/ruoyi 等）能命中组件攻击面。"""
        for signal in ("hadoop", "sangfor", "webvpn", "ruoyi", "kkfileview", "casdoor"):
            ep = parse_line(
                f"- /{signal}/admin GET business_attr=query_data attr_target=db"
            )
            assert ep is not None
            matches = match_surfaces(ep, load_surfaces())
            ids = {m.surface.surface_id for m in matches}
            self.assertIn("sf_component_admin", ids, f"{signal} 应命中组件控制面")

    def test_error_enum_chain_loaded(self) -> None:
        """批次D：错误码枚举→越权链应可加载并成为候选。"""
        ids = {c.chain_id for c in load_chains()}
        self.assertIn("ch_error_enum_to_idor", ids)
        eps = parse_text("- /api/user/info GET business_attr=query_data attr_target=db")
        cands = link(eps, load_chains())
        self.assertIn("ch_error_enum_to_idor", {c.chain_id for c in cands})

    def test_component_chain_candidate_on_admin_endpoint(self) -> None:
        """组件控制面端点（query_data）应让组件控制面链成为候选。"""
        eps = parse_text(
            "- /apisix/admin/routes GET business_attr=query_data attr_target=db\n"
            "- /apisix/admin/upstreams PUT business_attr=modify_state attr_target=db"
        )
        cands = link(eps, load_chains())
        ids = {c.chain_id for c in cands}
        self.assertIn("ch_component_admin_unauth", ids)

    def test_auth_bypass_surface_matches_bearer_signal(self) -> None:
        ep = parse_line(
            "- /api/user/info GET business_attr=auth attr_target=user_input attr_reason=需要Bearer鉴权"
        )
        assert ep is not None
        matches = match_surfaces(ep, load_surfaces())
        ids = {m.surface.surface_id for m in matches}
        self.assertIn("sf_auth_bypass", ids)


if __name__ == "__main__":
    unittest.main()
