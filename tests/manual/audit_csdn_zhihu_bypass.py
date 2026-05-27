#!/usr/bin/env python3
"""CSDN / 知乎绕过质量审计 & 冒烟测试。

测试维度：
  1. 绕过成功率（多 URL × 多轮次）
  2. 绕过前后内容质量对比（httpx 裸请求 vs Playwright 浏览器绕过）
  3. 内容真实性检测（是否真的拿到文章正文，而非验证页/空白页）
  4. 稳定性（同 URL 多次请求的一致性）

用法:
  PYTHONPATH=src python tests/manual/audit_csdn_zhihu_bypass.py
"""

from __future__ import annotations

import sys
import time
import statistics
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(_SRC))

# ── 测试用例：CSDN 和知乎的真实文章 URL ──
CSDN_URLS = [
    "https://blog.csdn.net/weixin_42376192/article/details/161153733",
    "https://blog.csdn.net/qq_34368655/article/details/148339576",
    "https://blog.csdn.net/csdnnews/article/details/148378995",
]

ZHIHU_URLS = [
    "https://zhuanlan.zhihu.com/p/665294318",
    "https://zhuanlan.zhihu.com/p/2039425125838235032",
]

ROUNDS_PER_URL = 3  # 每个 URL 测试轮数

# ═══════════════════════════════════════════════════════════════
print("=" * 80)
print("  CSDN / 知乎 绕过质量审计 & 冒烟测试")
print(f"  测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  CSDN URL 数: {len(CSDN_URLS)}")
print(f"  知乎 URL 数: {len(ZHIHU_URLS)}")
print(f"  每 URL 轮次: {ROUNDS_PER_URL}")
print(f"  总测试次数: {(len(CSDN_URLS) + len(ZHIHU_URLS)) * ROUNDS_PER_URL}")
print("=" * 80)

# ═══════════════════════════════════════════════════════════════
# PHASE 1: 绕过前基准测试 (httpx 裸请求 — 模拟不绕过的情况)
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  PHASE 1: 绕过前基准测试 (httpx 裸请求)")
print("  目的: 测量不使用浏览器绕过时，CSDN/知乎返回什么")
print("=" * 80)

from cyber_agent.tools.web_fetch import (
    _H2_AVAILABLE, _needs_browser_fetch,
    _FETCH_USER_AGENTS, _FETCH_HEADERS_CSDN, _FETCH_HEADERS_ZHIHU,
    _extract_text_from_html,
)

baseline_results: dict[str, list[dict]] = {}  # url -> list of {elapsed, len, preview}

for label, urls in [("CSDN", CSDN_URLS), ("知乎", ZHIHU_URLS)]:
    print(f"\n{'─' * 60}")
    print(f"  [{label}] 绕过前基准 — httpx 直接请求")
    print(f"{'─' * 60}")

    for url in urls:
        print(f"\n  📥 URL: {url}")
        print(f"     需要浏览器: {_needs_browser_fetch(url)}")
        print(f"     h2 可用: {_H2_AVAILABLE}")

        url_results = []
        for round_idx in range(ROUNDS_PER_URL):
            print(f"\n    --- 轮次 {round_idx + 1}/{ROUNDS_PER_URL} ---")

            import httpx, random
            headers = (_FETCH_HEADERS_ZHIHU if "zhihu.com" in url else _FETCH_HEADERS_CSDN).copy()
            headers["User-Agent"] = random.choice(_FETCH_USER_AGENTS)

            t0 = time.monotonic()
            try:
                with httpx.Client(
                    follow_redirects=True, timeout=8.0,
                    headers=headers, http2=_H2_AVAILABLE,
                ) as client:
                    resp = client.get(url)
                    resp.raise_for_status()
                    raw_html = resp.text
                text = _extract_text_from_html(raw_html)
                elapsed = time.monotonic() - t0
                success = True
                error = None
            except Exception as exc:
                text = ""
                elapsed = time.monotonic() - t0
                success = False
                error = str(exc)

            # 内容质量判定
            content_len = len(text)
            first_500 = text[:500] if text else ""

            # 真正的拦截页特征：页面主体全是验证/登录内容，没有实际文章
            _block_keywords = [
                "安全验证", "请输入验证码", "百度安全验证",
                "验证你是人类", "are you a robot", "unusual traffic",
            ]
            is_hard_blocked = any(kw in first_500 for kw in _block_keywords)

            # 软拦截：需要登录但没有文章内容 (知乎专用)
            # 如果页面包含登录提示 且 总长度 < 300 (基本只有登录框) → 真正被拦截
            is_login_only = (
                content_len < 300 and
                any(kw in first_500 for kw in ["登录后", "立即登录", "登录知乎", "请登录"])
            )

            is_empty = content_len < 100
            # 真实内容：长度足够 且 不是硬拦截 且 不是仅登录页 且 不是空页
            has_real_content = (
                content_len >= 500
                and not is_hard_blocked
                and not is_login_only
                and not is_empty
            )

            result = {
                "round": round_idx + 1,
                "elapsed": elapsed,
                "content_len": content_len,
                "success": success,
                "error": error,
                "is_blocked": is_hard_blocked,
                "is_login_only": is_login_only,
                "is_empty": is_empty,
                "has_real_content": has_real_content,
                "preview": text[:400] if text else "(空)",
            }
            url_results.append(result)

            print(f"      耗时: {elapsed:.2f}s")
            print(f"      成功: {success}")
            if error:
                print(f"      错误: {error[:120]}")
            print(f"      内容长度: {content_len} 字符")
            print(f"      被拦截(验证页): {is_hard_blocked}")
            print(f"      纯登录墙: {is_login_only}")
            print(f"      空内容: {is_empty}")
            print(f"      真实内容: {has_real_content}")
            print(f"      内容预览前 300 字:")
            print(f"      {'─' * 50}")
            for line in (text[:300] or "(空)").split("\n")[:10]:
                print(f"      │ {line[:120]}")
            print(f"      {'─' * 50}")

        baseline_results[url] = url_results

# ═══════════════════════════════════════════════════════════════
# PHASE 2: 绕过后测试 (Playwright 浏览器模式)
# ═══════════════════════════════════════════════════════════════

print("\n\n" + "=" * 80)
print("  PHASE 2: 绕过后测试 (Playwright 浏览器模式)")
print("  目的: 使用增强型反检测浏览器获取真实页面内容")
print("=" * 80)

bypass_results: dict[str, list[dict]] = {}

for label, urls in [("CSDN", CSDN_URLS), ("知乎", ZHIHU_URLS)]:
    print(f"\n{'─' * 60}")
    print(f"  [{label}] 绕过后测试 — Playwright 浏览器")
    print(f"{'─' * 60}")

    for url in urls:
        print(f"\n  📥 URL: {url}")
        print(f"     需要浏览器: {_needs_browser_fetch(url)}")

        url_results = []
        for round_idx in range(ROUNDS_PER_URL):
            print(f"\n    --- 轮次 {round_idx + 1}/{ROUNDS_PER_URL} ---")

            t0 = time.monotonic()
            try:
                from cyber_agent.tools.web_fetch import _fetch_with_playwright
                text = _fetch_with_playwright(url)
                elapsed = time.monotonic() - t0
                success = True
                error = None
            except Exception as exc:
                text = ""
                elapsed = time.monotonic() - t0
                success = False
                error = str(exc)

            # 内容质量判定
            content_len = len(text)
            first_500 = text[:500] if text else ""

            # 硬拦截：真正的反爬验证页
            _hard_block_kw = [
                "安全验证", "请输入验证码", "百度安全验证",
                "验证你是人类", "are you a robot", "unusual traffic",
            ]
            is_hard_blocked = any(kw in first_500 for kw in _hard_block_kw)

            # 软拦截：仅登录页无实质内容（长度 < 300，基本只有登录框）
            _login_kw = ["登录后", "立即登录", "登录知乎", "请登录", "开通机构号", "获取短信验证码"]
            is_login_only = (
                content_len < 300 and any(kw in first_500 for kw in _login_kw)
            )

            is_empty = content_len < 100
            # 真实内容：足够长 + 非硬拦截 + 非空 + 非纯登录页
            has_real_content = (
                content_len >= 500
                and not is_hard_blocked
                and not is_empty
                and not is_login_only
            )

            # 区分登录墙（页面有内容但被登录挡住，需人工判断）
            is_login_wall = (
                content_len >= 300 and content_len < 500
                and any(kw in first_500 for kw in _login_kw)
            )

            result = {
                "round": round_idx + 1,
                "elapsed": elapsed,
                "content_len": content_len,
                "success": success,
                "error": error,
                "is_blocked": is_hard_blocked,
                "is_login_wall": is_login_wall,
                "is_login_only": is_login_only,
                "is_empty": is_empty,
                "has_real_content": has_real_content,
                "preview": text[:400] if text else "(空)",
            }
            url_results.append(result)

            print(f"      耗时: {elapsed:.2f}s")
            print(f"      成功: {success}")
            if error:
                print(f"      错误: {error[:120]}")
            print(f"      内容长度: {content_len} 字符")
            print(f"      被拦截(验证页): {is_hard_blocked}")
            print(f"      纯登录墙: {is_login_only}")
            print(f"      登录墙(短): {is_login_wall}")
            print(f"      空内容: {is_empty}")
            print(f"      真实内容: {has_real_content}")
            print(f"      内容预览前 300 字:")
            print(f"      {'─' * 50}")
            for line in (text[:300] or "(空)").split("\n")[:10]:
                print(f"      │ {line[:120]}")
            print(f"      {'─' * 50}")

        bypass_results[url] = url_results

# ═══════════════════════════════════════════════════════════════
# PHASE 3: 完整页面内容展示 (绕过最后一轮的完整输出)
# ═══════════════════════════════════════════════════════════════

print("\n\n" + "=" * 80)
print("  PHASE 3: 绕过后 — 完整页面内容展示")
print("  目的: 让审计者肉眼验证内容真实性")
print("=" * 80)

for label, urls in [("CSDN", CSDN_URLS), ("知乎", ZHIHU_URLS)]:
    for url in urls:
        print(f"\n{'─' * 60}")
        print(f"  [{label}] 完整内容: {url}")
        print(f"{'─' * 60}")

        t0 = time.monotonic()
        try:
            from cyber_agent.tools.web_fetch import _fetch_with_playwright
            full_text = _fetch_with_playwright(url)
            elapsed = time.monotonic() - t0
        except Exception as exc:
            full_text = f"获取失败: {exc}"
            elapsed = time.monotonic() - t0

        print(f"  耗时: {elapsed:.2f}s")
        print(f"  总长度: {len(full_text)} 字符")
        print(f"  完整内容:")
        print(f"  {'═' * 70}")
        # 完整输出，不截断
        for i, line in enumerate(full_text.split("\n")):
            print(f"  {line}")
        print(f"  {'═' * 70}")

# ═══════════════════════════════════════════════════════════════
# PHASE 4: 统计汇总
# ═══════════════════════════════════════════════════════════════

print("\n\n" + "=" * 80)
print("  PHASE 4: 统计汇总")
print("=" * 80)

def compute_stats(results_dict: dict[str, list[dict]], label: str) -> dict:
    all_results = []
    for url, url_results in results_dict.items():
        all_results.extend(url_results)

    total = len(all_results)
    success_count = sum(1 for r in all_results if r["success"])
    real_content_count = sum(1 for r in all_results if r["has_real_content"])
    blocked_count = sum(1 for r in all_results if r.get("is_blocked", False))
    login_only_count = sum(1 for r in all_results if r.get("is_login_only", False))
    empty_count = sum(1 for r in all_results if r["is_empty"])
    elapsed_times = [r["elapsed"] for r in all_results if r["success"]]
    content_lens = [r["content_len"] for r in all_results if r["has_real_content"]]

    return {
        "label": label,
        "total": total,
        "success_rate": success_count / total * 100 if total else 0,
        "real_content_rate": real_content_count / total * 100 if total else 0,
        "blocked_rate": blocked_count / total * 100 if total else 0,
        "login_only_rate": login_only_count / total * 100 if total else 0,
        "empty_rate": empty_count / total * 100 if total else 0,
        "avg_elapsed": statistics.mean(elapsed_times) if elapsed_times else 0,
        "min_elapsed": min(elapsed_times) if elapsed_times else 0,
        "max_elapsed": max(elapsed_times) if elapsed_times else 0,
        "stdev_elapsed": statistics.stdev(elapsed_times) if len(elapsed_times) >= 2 else 0,
        "avg_content_len": statistics.mean(content_lens) if content_lens else 0,
        "min_content_len": min(content_lens) if content_lens else 0,
        "max_content_len": max(content_lens) if content_lens else 0,
    }

print("\n  ┌─ 绕过前 (httpx 裸请求) ─────────────────────────────")
baseline_stats = compute_stats(baseline_results, "baseline")
print(f"  │ 总测试数:     {baseline_stats['total']}")
print(f"  │ 请求成功率:   {baseline_stats['success_rate']:.1f}%")
print(f"  │ 真实内容率:   {baseline_stats['real_content_rate']:.1f}%")
print(f"  │ 被拦截率:     {baseline_stats['blocked_rate']:.1f}%")
print(f"  │ 空内容率:     {baseline_stats['empty_rate']:.1f}%")
print(f"  │ 平均耗时:     {baseline_stats['avg_elapsed']:.2f}s")
print(f"  │ 耗时范围:     {baseline_stats['min_elapsed']:.2f}s ~ {baseline_stats['max_elapsed']:.2f}s")
print(f"  │ 平均内容长度: {baseline_stats['avg_content_len']:.0f} 字符")

print("\n  ┌─ 绕过后 (Playwright 浏览器) ────────────────────────")
bypass_stats = compute_stats(bypass_results, "bypass")
print(f"  │ 总测试数:     {bypass_stats['total']}")
print(f"  │ 请求成功率:   {bypass_stats['success_rate']:.1f}%")
print(f"  │ 真实内容率:   {bypass_stats['real_content_rate']:.1f}%")
print(f"  │ 被拦截率:     {bypass_stats['blocked_rate']:.1f}%")
print(f"  │ 空内容率:     {bypass_stats['empty_rate']:.1f}%")
print(f"  │ 平均耗时:     {bypass_stats['avg_elapsed']:.2f}s")
print(f"  │ 耗时范围:     {bypass_stats['min_elapsed']:.2f}s ~ {bypass_stats['max_elapsed']:.2f}s")
print(f"  │ 耗时标准差:   {bypass_stats['stdev_elapsed']:.2f}s")
print(f"  │ 平均内容长度: {bypass_stats['avg_content_len']:.0f} 字符")

print("\n  ┌─ 绕过效果对比 ─────────────────────────────────────")
real_diff = bypass_stats['real_content_rate'] - baseline_stats['real_content_rate']
print(f"  │ 真实内容率提升: {real_diff:+.1f}%")
print(f"  │ 绕过前真实率:   {baseline_stats['real_content_rate']:.1f}%")
print(f"  │ 绕过后真实率:   {bypass_stats['real_content_rate']:.1f}%")
print(f"  │ 绕过倍率:       {bypass_stats['real_content_rate'] / max(baseline_stats['real_content_rate'], 0.1):.1f}x")

# 稳定性：检查同 URL 多轮次的结果一致性
print("\n  ┌─ 稳定性分析 (同 URL 多轮次) ───────────────────────")
for url in list(CSDN_URLS) + list(ZHIHU_URLS):
    br = bypass_results.get(url, [])
    if not br:
        continue
    lens = [r["content_len"] for r in br if r["success"]]
    reals = [r["has_real_content"] for r in br]
    real_rate = sum(reals) / len(reals) * 100 if reals else 0
    if len(lens) >= 2:
        cv = statistics.stdev(lens) / statistics.mean(lens) * 100 if statistics.mean(lens) > 0 else 0
        stability = "稳定" if cv < 20 else ("一般" if cv < 50 else "不稳定")
        print(f"  │ {Path(url).name[:50]}")
        print(f"  │   长度范围: {min(lens)}~{max(lens)}, 变异系数: {cv:.1f}% → {stability}")
        print(f"  │   真实率: {real_rate:.0f}% ({sum(reals)}/{len(reals)})")
    else:
        print(f"  │ {Path(url).name[:50]}: 成功次数不足 ({len(lens)})")

print()

# ── 逐个 URL 详细对比表 ──
print("  ┌─ 逐个 URL 详细对比 ─────────────────────────────────")
print(f"  │ {'URL':<55s} {'绕过前':>8s} {'绕过后':>8s} {'提升':>8s}")
print(f"  │ {'─'*55} {'─'*8} {'─'*8} {'─'*8}")
for url in list(CSDN_URLS) + list(ZHIHU_URLS):
    bl = baseline_results.get(url, [])
    br = bypass_results.get(url, [])
    bl_real = sum(1 for r in bl if r["has_real_content"]) / max(len(bl), 1) * 100
    br_real = sum(1 for r in br if r["has_real_content"]) / max(len(br), 1) * 100
    short_url = url.rsplit("/", 1)[-1][:45] if "/" in url else url[:45]
    domain = url.split("/")[2]
    print(f"  │ {domain+'/'+short_url:<55s} {bl_real:>7.0f}% {br_real:>7.0f}% {br_real-bl_real:>+7.0f}%")

# ── 最终结论 ──
print(f"\n  {'═' * 70}")
print(f"  审计结论")
print(f"  {'═' * 70}")

if bypass_stats['real_content_rate'] >= 80:
    print(f"  ✅ 绕过后真实内容率达到 {bypass_stats['real_content_rate']:.0f}%，绕过有效")
else:
    print(f"  ⚠️  绕过后真实内容率仅 {bypass_stats['real_content_rate']:.0f}%，需优化")

if real_diff >= 50:
    print(f"  ✅ 真实内容率提升 {real_diff:.0f}%，绕过效果显著")
elif real_diff >= 20:
    print(f"  ✅ 真实内容率提升 {real_diff:.0f}%，绕过有效果")
else:
    print(f"  ⚠️  真实内容率提升仅 {real_diff:.0f}%，效果有限")

if bypass_stats['stdev_elapsed'] < 2.0 and bypass_stats['success_rate'] >= 80:
    print(f"  ✅ 稳定性良好（耗时标准差 {bypass_stats['stdev_elapsed']:.2f}s，成功率 {bypass_stats['success_rate']:.0f}%）")
else:
    print(f"  ⚠️  稳定性需要关注（耗时标准差 {bypass_stats['stdev_elapsed']:.2f}s，成功率 {bypass_stats['success_rate']:.0f}%）")

print(f"\n  详细审计报告已完整输出在上方。")
print(f"  脚本位置: {__file__}")
print(f"  {'═' * 70}")
