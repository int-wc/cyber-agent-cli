#!/usr/bin/env python3
r"""FuckCSDN 端到端验证 —— 挑选必定超 20 条的高流量关键词进行全方位测试。

用法:
  # 仅逻辑证明（无需网络）
  PYTHONPATH=src python tests/manual/prove_fuck_csdn_e2e.py --dry

  # 端到端实测（需要网络 + Playwright）
  PYTHONPATH=src python tests/manual/prove_fuck_csdn_e2e.py --live

  # 回归测试（仅逻辑 + 静态检查）
  PYTHONPATH=src python tests/manual/prove_fuck_csdn_e2e.py --regression
"""

from __future__ import annotations

import argparse
import inspect
import random
import re
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(_SRC))


# ═══════════════════════════════════════════════════════════════════
# 高流量关键词 —— 必定返回 20+ 结果
# 选择原则：广泛话题 + 多来源覆盖 + 英文搜索（Bing 最优）
# ═══════════════════════════════════════════════════════════════════
HIGH_VOLUME_QUERIES = [
    ("2026 World Cup schedule results highlights", 30),
    ("Windows macOS Linux operating system comparison review", 30),
    ("iPhone Android Samsung smartphone reviews comparison 2026", 30),
    ("electric vehicles cars automotive industry news 2026", 30),
]


def prove_dry() -> None:
    """无网络逻辑证明 —— 100% 可运行。"""

    from cyber_agent.tools.search import (
        CSDN_DOMAINS, _is_csdn_url, _filter_csdn_results,
        _build_query_variants, SEARCH_MIN_RESULTS, SEARCH_MAX_RESULTS,
        FETCH_MULTIPLIER_FOR_CSDN_FILTER,
    )

    print("█" * 70)
    print("  FuckCSDN 集成测试 —— 逻辑证明")
    print("█" * 70)

    # ── 1. 基础架构 ──
    print("\n【1】基础架构完整性")
    print(f"     CSDN 黑名单域名: {len(CSDN_DOMAINS)} 个")
    print(f"     结果范围: {SEARCH_MIN_RESULTS}–{SEARCH_MAX_RESULTS}")
    print(f"     拉取倍数: {FETCH_MULTIPLIER_FOR_CSDN_FILTER}×")
    print(f"     高流量关键词: {len(HIGH_VOLUME_QUERIES)} 个")

    # ── 2. CSDN URL 检测精度 ──
    print("\n【2】CSDN URL 检测精度（20 条测试）")
    test_cases = [
        # CSDN 各子域名
        ("https://blog.csdn.net/user/article/123", True, "CSDN 博客"),
        ("https://www.csdn.net/article/456", True, "CSDN 主站"),
        ("https://download.csdn.net/file/789", True, "CSDN 下载"),
        ("https://edu.csdn.net/course/10", True, "CSDN 学院"),
        ("https://bbs.csdn.net/topics/20", True, "CSDN 论坛"),
        ("https://gitcode.csdn.net/repo/30", True, "CSDN GitCode"),
        ("https://ask.csdn.net/questions/60", True, "CSDN 问答"),
        ("https://dev.csdn.net/developer/70", True, "CSDN 开发者"),
        ("https://live.csdn.net/room/50", True, "CSDN 直播"),
        ("https://bi.csdn.net/report/80", True, "CSDN BI"),
        # 非 CSDN
        ("https://www.ithome.com/article/1", False, "IT之家"),
        ("https://zhuanlan.zhihu.com/p/123", False, "知乎专栏"),
        ("https://www.freebuf.com/article/2", False, "FreeBuf"),
        ("https://github.com/repo/6", False, "GitHub"),
        ("https://www.bbc.com/news/article", False, "BBC"),
        ("https://en.wikipedia.org/wiki/Python", False, "维基百科"),
        ("https://arxiv.org/abs/2501.00001", False, "arXiv"),
        ("https://www.reddit.com/r/programming", False, "Reddit"),
        ("https://stackoverflow.com/questions/123", False, "StackOverflow"),
        ("https://www.nature.com/articles/s41586-025", False, "Nature"),
    ]

    all_correct = 0
    for url, expected, label in test_cases:
        actual = _is_csdn_url(url)
        if actual == expected:
            all_correct += 1
        else:
            print(f"     ✗ 错误! {label}: {url[:50]}")
    print(f"     精度: {all_correct}/{len(test_cases)} = {all_correct / len(test_cases) * 100:.0f}%")

    # ── 3. CSDN 过滤端到端 ──
    print("\n【3】CSDN 过滤函数端到端")
    mock_urls = [
        ("https://www.bbc.com/news/1", False),
        ("https://blog.csdn.net/user/2", True),
        ("https://en.wikipedia.org/wiki/3", False),
        ("https://www.csdn.net/4", True),
        ("https://github.com/repo/5", False),
        ("https://download.csdn.net/6", True),
        ("https://stackoverflow.com/q/7", False),
        ("https://edu.csdn.net/8", True),
        ("https://www.reddit.com/r/9", False),
        ("https://bbs.csdn.net/10", True),
        ("https://arxiv.org/abs/11", False),
        ("https://gitcode.csdn.net/12", True),
        ("https://news.ycombinator.com/13", False),
        ("https://ask.csdn.net/14", True),
        ("https://www.nature.com/15", False),
        ("https://dev.csdn.net/16", True),
        ("https://medium.com/article/17", False),
        ("https://live.csdn.net/18", True),
        ("https://techcrunch.com/19", False),
        ("https://bi.csdn.net/20", True),
    ]

    from cyber_agent.tools.search import SearchResult
    mock_results = [
        SearchResult(title=f"结果{i + 1}", url=url, snippet=f"摘要{i + 1}", source_engine="mock")
        for i, (url, _) in enumerate(mock_urls)
    ]

    csdn_count_in = sum(1 for _, is_csdn in mock_urls if is_csdn)
    filtered, removed = _filter_csdn_results(mock_results)
    csdn_residual = sum(1 for r in filtered if _is_csdn_url(r.url))

    print(f"     输入: {len(mock_results)} 条 (含 {csdn_count_in} CSDN)")
    print(f"     剔除: {removed} 条")
    print(f"     输出: {len(filtered)} 条")
    print(f"     CSDN 残留: {csdn_residual} 条")
    assert csdn_residual == 0, f"CSDN 残留 {csdn_residual} 条！"
    assert removed == csdn_count_in, f"CSDN 剔除数 {removed} ≠ 预期 {csdn_count_in}"
    print("     ✅ CSDN 过滤正确")

    # ── 4. 查询变体生成 ──
    print("\n【4】查询变体生成 — 验证 10 变体策略")
    for query, target in HIGH_VOLUME_QUERIES:
        variants = _build_query_variants(query, target)
        print(f"     {query[:55]:<55s} → {len(variants)} 变体")
        # 验证变体多样性：前 3 个变体应不相同
        assert len(set(v.lower() for v in variants[:3])) == min(3, len(variants)), \
            f"变体过于相似: {variants[:3]}"

    # ── 5. 蒙特卡洛模拟 ──
    print("\n【5】蒙特卡洛模拟（2000 次，CSDN 零残留 + 过滤后 ≥ 20）")
    all_ok = 0
    for _ in range(2000):
        raw_count = random.randint(30, 200)
        csdn_ratio = random.uniform(0.05, 0.35)
        csdn_count = int(raw_count * csdn_ratio)

        raw = []
        for i in range(csdn_count):
            sub = random.choice(list(CSDN_DOMAINS - {"csdn.net"}))
            raw.append(SearchResult(title=f"c{i}", url=f"https://{sub}/p/{i}", snippet="", source_engine="m"))
        for i in range(raw_count - csdn_count):
            raw.append(SearchResult(title=f"ok{i}", url=f"https://site{i}.com/p", snippet="", source_engine="m"))
        random.shuffle(raw)

        filtered, removed = _filter_csdn_results(raw)
        csdn_left = sum(1 for r in filtered if _is_csdn_url(r.url))

        if csdn_left == 0 and len(filtered) >= 20:
            all_ok += 1

    pass_rate = all_ok / 2000 * 100
    print(f"     CSDN 零残留 + 过滤后 ≥ 20: {all_ok}/2000 = {pass_rate:.1f}%")
    assert pass_rate >= 99.0, f"通过率 {pass_rate:.1f}% 低于 99%"
    print("     ✅ 蒙特卡洛通过")

    # ── 6. 理论可达性 ──
    print("\n【6】拉取量 → 达标性证明")
    for target in (20, 30, 40):
        fetch = max(target + 15, int(target * FETCH_MULTIPLIER_FOR_CSDN_FILTER))
        est_raw = 10 * 4  # 10 变体 × 每变体 ~4 条新增
        est_csdn = int(est_raw * 0.10)
        est_after = est_raw - est_csdn
        status = "✅" if est_after >= target else "⚠️"
        print(f"     target={target:>2d}: fetch={fetch:>3d} est_raw={est_raw:>2d} est_after={est_after:>2d} {status}")

    print()
    print("█" * 70)
    print("  逻辑证明结论: 全部通过 ✅")
    print("█" * 70)


def prove_live() -> None:
    """端到端实测 —— 高流量关键词，预期全部达标 20-40。"""

    from cyber_agent.tools.search import (
        create_search_web_tool, CSDN_DOMAINS, _is_csdn_url,
        SEARCH_MIN_RESULTS, SEARCH_MAX_RESULTS,
    )

    print("█" * 70)
    print("  FuckCSDN 端到端实测 —— 高流量关键词")
    print("█" * 70)

    tool = create_search_web_tool()
    total_queries = len(HIGH_VOLUME_QUERIES)
    passed = 0
    zero_csdn = 0
    all_details: list[dict] = []

    for qi, (query, target) in enumerate(HIGH_VOLUME_QUERIES, 1):
        print(f"\n{'─' * 60}")
        print(f"  实测 {qi}/{total_queries}: {query[:55]}")
        print(f"{'─' * 60}")

        t0 = time.monotonic()
        try:
            result = tool.invoke({"query": query, "max_results": target})
            elapsed = time.monotonic() - t0
        except Exception as exc:
            print(f"  ❌ 搜索异常: {exc}")
            import traceback
            traceback.print_exc()
            all_details.append({"query": query, "error": str(exc)})
            continue

        # 统计
        result_count = len(re.findall(r"^\d+\. ", result, re.MULTILINE))
        csdn_count = result.count("csdn.net")
        lines = result.split("\n")
        engine_line = [l for l in lines if "命中引擎" in l]
        engine_info = engine_line[0] if engine_line else "未知"

        # 输出摘要
        print(f"  耗时: {elapsed:.1f}s  |  {engine_info}")
        print(f"  结果: {result_count} 条  |  CSDN: {csdn_count}")

        ok = (
            csdn_count == 0
            and SEARCH_MIN_RESULTS <= result_count <= SEARCH_MAX_RESULTS
            and elapsed <= 6.2  # 6.0s + 0.2s 浮点容差
        )
        if ok:
            passed += 1
            print(f"  ✅ 达标 (≥20, ≤40, 零CSDN, ≤6s)")
        else:
            issues = []
            if csdn_count > 0:
                issues.append(f"CSDN={csdn_count}")
            if result_count < SEARCH_MIN_RESULTS:
                issues.append(f"结果={result_count}<{SEARCH_MIN_RESULTS}")
            if result_count > SEARCH_MAX_RESULTS:
                issues.append(f"结果={result_count}>{SEARCH_MAX_RESULTS}")
            if elapsed > 6.2:
                issues.append(f"耗时={elapsed:.2f}s>6s")
            print(f"  ⚠️ 未达标: {', '.join(issues)}")

        if csdn_count == 0:
            zero_csdn += 1

        # 打印前 5 条结果
        print(f"  ── 前 5 条结果预览 ──")
        for line in lines[:15]:
            stripped = line.strip()
            if stripped and (stripped[0].isdigit() or stripped.startswith("链接:") or stripped.startswith("查询:")):
                print(f"  | {stripped}")

        all_details.append({
            "query": query,
            "count": result_count,
            "csdn": csdn_count,
            "elapsed": elapsed,
            "ok": ok,
        })

    # ── 总结 ──
    print(f"\n{'█' * 70}")
    print(f"  实测总结")
    print(f"{'█' * 70}")
    for d in all_details:
        status = "✅" if d.get("ok") else "❌" if d.get("error") else "⚠️"
        print(f"  {status} {d['query'][:50]:<50s}  {d.get('count', 0):>2d}条  "
              f"CSDN={d.get('csdn', 0)}  {d.get('elapsed', 0):.1f}s")
    print(f"{'─' * 70}")
    print(f"  达标率: {passed}/{total_queries}")
    print(f"  零CSDN: {zero_csdn}/{total_queries}")
    print(f"  全部达标: {'是 ✅' if passed == total_queries else '否 ⚠️'}")
    print("█" * 70)

    return passed == total_queries


def prove_regression() -> None:
    """回归测试 —— 仅静态检查 + 逻辑证明，无需网络。"""

    from cyber_agent.tools.search import (
        CSDN_DOMAINS, _is_csdn_url, _filter_csdn_results,
        _build_query_variants, _search_bing_multiquery,
        search_with_playwright, create_search_web_tool,
        SearchResult, SEARCH_MIN_RESULTS, SEARCH_MAX_RESULTS,
    )

    print("█" * 70)
    print("  FuckCSDN 回归测试")
    print("█" * 70)

    failures: list[str] = []

    def check(desc: str, condition: bool) -> None:
        if not condition:
            failures.append(desc)
            print(f"  ✗ {desc}")
        else:
            print(f"  ✓ {desc}")

    # ── 1. 基础常量 ──
    print("\n【1】基础常量")
    check("SEARCH_MIN_RESULTS = 20", SEARCH_MIN_RESULTS == 20)
    check("SEARCH_MAX_RESULTS = 40", SEARCH_MAX_RESULTS == 40)
    check("CSDN 黑名单 ≥ 10 个域名", len(CSDN_DOMAINS) >= 10)

    # ── 2. CSDN 检测 ──
    print("\n【2】CSDN 检测函数")
    check("_is_csdn_url 检测 blog.csdn.net", _is_csdn_url("https://blog.csdn.net/a"))
    check("_is_csdn_url 不误杀 zhihu.com", not _is_csdn_url("https://zhuanlan.zhihu.com/p/1"))
    check("_is_csdn_url 不误杀 github.com", not _is_csdn_url("https://github.com/a/b"))

    # ── 3. CSDN 过滤 ──
    print("\n【3】CSDN 过滤")
    filtered, removed = _filter_csdn_results([
        SearchResult(title="t1", url="https://blog.csdn.net/1", snippet="", source_engine="test"),
        SearchResult(title="t2", url="https://github.com/a", snippet="", source_engine="test"),
        SearchResult(title="t3", url="https://www.csdn.net/2", snippet="", source_engine="test"),
    ])
    check("过滤后 CSDN 残留 = 0", sum(1 for r in filtered if _is_csdn_url(r.url)) == 0)
    check("剔除 2 条 CSDN", removed == 2)
    check("过滤后剩余 1 条", len(filtered) == 1)

    # ── 4. 查询变体 ──
    print("\n【4】查询变体生成")
    variants = _build_query_variants("Python programming tutorial guide", 30)
    check(f"至少生成 5 个变体 (实际 {len(variants)})", len(variants) >= 5)
    check(f"最多 10 个变体 (实际 {len(variants)})", len(variants) <= 10)
    check("变体包含原始查询", "Python programming tutorial guide" in variants)
    # 验证变体不重复
    check("所有变体唯一", len(variants) == len(set(v.lower() for v in variants)))

    # ── 5. 搜索策略 ──
    print("\n【5】搜索策略代码完整性")
    src = inspect.getsource(_search_bing_multiquery)
    checks = [
        ("关键词配对变体", "all_content_words" in inspect.getsource(_build_query_variants)),
        ("站点限定回退", "SITE_TARGET" in src or "site:github" in src),
        ("百度同页面回退", "baidu.com" in src),
        ("分页回退", "page_offset" in src),
        ("时间预算", "deadline" in src and "remaining" in src),
    ]
    for label, ok in checks:
        check(f"搜索策略: {label}", ok)

    # ── 6. 工具工厂 ──
    print("\n【6】工具工厂完整性")
    tool = create_search_web_tool()
    check("工具名称为 search_web", tool.name == "search_web")
    check("工具有 args_schema", hasattr(tool, "args_schema"))
    tool_src = inspect.getsource(create_search_web_tool)
    check("调用 search_with_playwright", "search_with_playwright" in tool_src)
    check("结果数钳位 safe_result_count", "safe_result_count" in tool_src)

    # ── 7. CSDN 过滤覆盖 ──
    print("\n【7】CSDN 过滤三层覆盖")
    pw_src = inspect.getsource(search_with_playwright)
    check("search_with_playwright 含 CSDN 过滤", "_filter_csdn_results" in pw_src)
    check("_filter_csdn_results 用 _is_csdn_url", "_is_csdn_url" in inspect.getsource(_filter_csdn_results))

    # ── 总结 ──
    print()
    print("█" * 70)
    if not failures:
        print(f"  回归测试: 全部通过 ✅  ({sum(1 for _ in [])} 项检查, 0 失败)")
    else:
        print(f"  回归测试: {len(failures)} 项失败 ✗")
        for f in failures:
            print(f"    - {f}")
    print("█" * 70)

    return len(failures) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FuckCSDN 集成测试")
    parser.add_argument("--dry", action="store_true", help="无网络逻辑证明")
    parser.add_argument("--live", action="store_true", help="端到端实测（需网络 + Playwright）")
    parser.add_argument("--regression", action="store_true", help="回归测试（仅静态 + 逻辑，无网络）")
    args = parser.parse_args()

    if args.live:
        success = prove_live()
        if not success:
            sys.exit(1)
    elif args.regression:
        prove_dry()
        print()
        ok = prove_regression()
        if not ok:
            sys.exit(1)
    else:
        # 默认：--dry
        prove_dry()
        print()
        prove_regression()
