#!/usr/bin/env python3
"""端到端实测脚本 —— 真正调用搜索/抓取，打印全部输入、过程、输出。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(_SRC))

# ═══════════════════════════════════════════════════════════════
print("=" * 80)
print("  实测 1: search_web 端到端测试")
print("=" * 80)

from cyber_agent.tools.search import create_search_web_tool

search_tool = create_search_web_tool()

query = "Pwn2Own 2026 Berlin results highlights"
max_results = 30

print(f"\n📥 输入参数:")
print(f"   query       = {query!r}")
print(f"   max_results = {max_results}")

print(f"\n⏱️  开始搜索...")
t0 = time.monotonic()
result = search_tool.invoke({"query": query, "max_results": max_results})
elapsed = time.monotonic() - t0

print(f"\n⏱️  耗时: {elapsed:.2f}s")
print(f"\n📤 原始输出 ({len(result)} 字符):")
print("-" * 80)
print(result)
print("-" * 80)

# 解析结果条数
import re
result_count = len(re.findall(r"^\d+\.", result, re.MULTILINE))
print(f"\n📊 实际返回结果条数: {result_count}")

if result_count < 20:
    print(f"   ⚠️  未达到最小 20 条要求!")
elif result_count > 40:
    print(f"   ⚠️  超过最大 40 条!")
else:
    print(f"   ✅ 在 20-40 范围内")

if elapsed <= 6.0:
    print(f"   ✅ 耗时 {elapsed:.2f}s ≤ 6s")
else:
    print(f"   ⚠️  耗时 {elapsed:.2f}s > 6s")

# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  实测 2: fetch_web_page — CSDN 绕过测试")
print("=" * 80)

from cyber_agent.tools.web_fetch import (
    create_web_fetch_tool, _H2_AVAILABLE, _needs_browser_fetch,
    BROWSER_FETCH_DOMAINS,
)

fetch_tool = create_web_fetch_tool()

# 找一个 CSDN 文章 URL
csdn_url = "https://blog.csdn.net/weixin_42376192/article/details/161153733"

print(f"\n📥 输入参数:")
print(f"   url        = {csdn_url!r}")
print(f"   use_browser = False (自动判断)")
print(f"\n🔍 预检测:")
print(f"   h2 可用     = {_H2_AVAILABLE}")
print(f"   需要浏览器  = {_needs_browser_fetch(csdn_url)}")
print(f"   命中域名    = {[d for d in BROWSER_FETCH_DOMAINS if d in csdn_url]}")

print(f"\n⏱️  开始抓取 CSDN...")
t0 = time.monotonic()
result = fetch_tool.invoke({"url": csdn_url})
elapsed = time.monotonic() - t0

print(f"\n⏱️  耗时: {elapsed:.2f}s")
print(f"\n📤 输出 ({len(result)} 字符):")
print("-" * 80)
print(result[:3000])
if len(result) > 3000:
    print(f"\n... (截断，完整 {len(result)} 字符)")
print("-" * 80)

# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  实测 3: fetch_web_page — 知乎绕过测试")
print("=" * 80)

zhihu_url = "https://zhuanlan.zhihu.com/p/665294318"

print(f"\n📥 输入参数:")
print(f"   url        = {zhihu_url!r}")
print(f"\n🔍 预检测:")
print(f"   需要浏览器  = {_needs_browser_fetch(zhihu_url)}")
print(f"   命中域名    = {[d for d in BROWSER_FETCH_DOMAINS if d in zhihu_url]}")

print(f"\n⏱️  开始抓取知乎...")
t0 = time.monotonic()
result = fetch_tool.invoke({"url": zhihu_url})
elapsed = time.monotonic() - t0

print(f"\n⏱️  耗时: {elapsed:.2f}s")
print(f"\n📤 输出 ({len(result)} 字符):")
print("-" * 80)
print(result[:3000])
if len(result) > 3000:
    print(f"\n... (截断，完整 {len(result)} 字符)")
print("-" * 80)

# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  实测 4: 多角色查询生成验证")
print("=" * 80)

from cyber_agent.tools.search import _generate_role_based_queries

for test_query, expected_min in [
    ("Pwn2Own 2026 Berlin", 6),
    ("Windows 11 zero-day vulnerability CVE", 5),
]:
    print(f"\n📥 输入: {test_query!r}")
    variants = _generate_role_based_queries(test_query, expected_min)
    print(f"📤 输出: {len(variants)} 个角色查询变体:")
    for role, q in variants:
        print(f"   [{role}]")
        print(f"     → {q}")
    print(f"   唯一性: {len({q.lower().strip() for _, q in variants})} == {len(variants)} ✅" if len({q.lower().strip() for _, q in variants}) == len(variants) else "   ⚠️ 存在重复")

# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  实测 5: /file 命令端到端 (模拟)")
print("=" * 80)

# 创建一个临时测试文件
import tempfile, os
tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8")
tmp.write('#!/usr/bin/env python3\n"""测试模块."""\n\ndef hello():\n    return "Hello World"\n\nif __name__ == "__main__":\n    print(hello())\n')
tmp.close()
test_path = tmp.name

from cyber_agent.cli.builtin_commands import _handle_file
from cyber_agent.cli.render import CliRenderer

class FakeRunner:
    pass

renderer = CliRenderer()
ctx: dict[str, object] = {}

print(f"\n📥 输入: /file {test_path}")
print(f"   文件存在: {Path(test_path).exists()}")
print(f"   文件内容:")
print(Path(test_path).read_text())

print(f"\n⏱️  执行 /file 命令...")
result = _handle_file(FakeRunner(), ctx, renderer, ["/file", test_path], f"/file {test_path}")

print(f"\n📤 返回值: {result}")
print(f"📤 runtime_context keys: {[k for k in ctx if k.startswith('__pending_file_')]}")
for k, v in ctx.items():
    if k.startswith("__pending_file_"):
        print(f"   {k}:")
        print(f"     path: {v['path']}")
        print(f"     lang: {v['lang']}")
        print(f"     content ({len(v['content'])} chars):")
        print(f"     {v['content'][:200]}...")

os.unlink(test_path)

# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  全部实测完成")
print("=" * 80)
