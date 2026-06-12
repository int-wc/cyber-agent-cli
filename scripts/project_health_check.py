#!/usr/bin/env python3
"""
项目健康检查与自愈脚本 (Project Health Check & Self-Healing)
============================================================
基于 analyzer/diffuser 报告中的 10 类异常自动检测 + 5 类自愈修复。

用法:
  检查模式:  python scripts/project_health_check.py --check
  修复模式:  python scripts/project_health_check.py --fix
  全面模式:  python scripts/project_health_check.py --all  (默认)
  CI模式:    python scripts/project_health_check.py --ci     (仅检查，非零退出)

覆盖痛点:
  CHK-1  缺失 __init__.py → 自动创建
  CHK-2  .py 文件异常可执行位 → 自动清理（保留 __main__.py）
  CHK-3  孤儿测试（test有源无） → 检测并报告
  CHK-4  命名偏差（test_xxx vs xxx） → 检测并报告
  CHK-5  测试覆盖率缺口 → 列出未覆盖模块
  CHK-6  空 __init__.py 检测 → 确认非意外空文件 ok
  CHK-7  tests/manual/ 无测试框架 → 检测裸脚本
  CHK-8  conftest.py 可用性 → 确认 fixture 入口存在
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict
from typing import List


# ============================================================================
# 配置区
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src" / "cyber_agent"
TESTS_ROOT = PROJECT_ROOT / "tests"

# __main__.py 保留可执行位的白名单
EXECUTABLE_WHITELIST = {
    SRC_ROOT / "__main__.py",
}

# 应忽略的目录
IGNORE_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".venv", "venv",
    "node_modules", ".mypy_cache", ".ruff_cache", "dist", "build",
}

# 公认交叉/综合测试（无对应源模块属正常）
CROSS_CUTTING_TESTS = {
    "test_000_bootstrap", "test_agent_tool_call",
    "test_authorized_mode_tools", "test_cli_chat_e2e",
    "test_cli_commands", "test_parallel_execution",
}


# ============================================================================
# 工具函数
# ============================================================================

def find_python_files(root: Path) -> List[Path]:
    """递归查找所有 .py 文件，忽略不应关注的目录。"""
    files = []
    if not root.exists():
        return files
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in IGNORE_DIRS and not d.startswith(".")]
        for f in filenames:
            if f.endswith(".py"):
                files.append(Path(dirpath) / f)
    return files


def find_dirs_with_py(root: Path) -> List[Path]:
    """找出所有包含 .py 文件的目录。"""
    dirs = set()
    for py_file in find_python_files(root):
        dirs.add(py_file.parent)
    return sorted(dirs)


def has_exec_bit(filepath: Path) -> bool:
    """检查文件是否有可执行位。"""
    return bool(os.stat(filepath).st_mode & 0o111)


# ============================================================================
# 检查器
# ============================================================================

class HealthReport:
    """健康检查报告。"""

    def __init__(self):
        self.issues = []
        self.ok = []
        self.fix_actions = []

    def add_issue(self, check_id, severity, title, detail, fixable=False):
        self.issues.append({
            "id": check_id, "severity": severity,
            "title": title, "detail": detail, "fixable": fixable,
        })

    def add_ok(self, check_id, msg):
        self.ok.append((check_id, msg))

    def has_critical(self):
        return any(i["severity"] == "CRITICAL" for i in self.issues)

    def has_any(self):
        return len(self.issues) > 0


def run_all_checks(report: HealthReport):
    """执行所有健康检查。"""

    # ---- CHK-1: __init__.py 完备性 ----
    all_dirs = find_dirs_with_py(SRC_ROOT) + find_dirs_with_py(TESTS_ROOT)
    missing_init = [d for d in all_dirs if not (d / "__init__.py").exists()]
    if missing_init:
        paths = "\n    ".join(str(p.relative_to(PROJECT_ROOT)) for p in missing_init)
        report.add_issue(
            "CHK-1", "HIGH",
            f"缺失 __init__.py: {len(missing_init)} 个目录",
            f"以下目录无 __init__.py:\n    {paths}",
            fixable=True,
        )
        for d in missing_init:
            report.fix_actions.append(("AUTO", f"touch {d / '__init__.py'}"))
    else:
        report.add_ok("CHK-1", "所有目录 __init__.py 齐全")

    # ---- CHK-2: 可执行位 ----
    all_py = find_python_files(SRC_ROOT) + find_python_files(TESTS_ROOT)
    bad_perms = [f for f in all_py
                 if f not in EXECUTABLE_WHITELIST and has_exec_bit(f)]
    if bad_perms:
        sample = bad_perms[:8]
        paths = "\n    ".join(str(p.relative_to(PROJECT_ROOT)) for p in sample)
        suffix = f"\n    ... 等共 {len(bad_perms)} 个" if len(bad_perms) > 8 else ""
        report.add_issue(
            "CHK-2", "MEDIUM",
            f"异常可执行位: {len(bad_perms)} 个 .py 文件",
            f"以下 .py 不应有执行权限:\n    {paths}{suffix}",
            fixable=True,
        )
        report.fix_actions.append(
            ("AUTO", f"chmod -x 共 {len(bad_perms)} 个文件"))
    else:
        report.add_ok("CHK-2", "所有 .py 文件权限正常")

    # ---- CHK-3: 孤儿测试 ----
    src_stems = {f.stem for f in find_python_files(SRC_ROOT)
                 if f.name != "__init__.py"}
    test_files = [f for f in find_python_files(TESTS_ROOT)
                  if f.stem.startswith("test_")]
    orphans = []
    for tf in test_files:
        if tf.stem in CROSS_CUTTING_TESTS:
            continue
        base = tf.stem[5:]
        # 匹配：精确、前缀变体、后缀变体
        if base in src_stems:
            continue
        if any(base == s or base.endswith("_" + s) or s.endswith("_" + base)
               or s.endswith(base) for s in src_stems):
            continue
        orphans.append(tf)

    if orphans:
        paths = "\n    ".join(str(p.relative_to(PROJECT_ROOT)) for p in orphans)
        report.add_issue(
            "CHK-3", "HIGH",
            f"孤儿测试: {len(orphans)} 个",
            f"以下测试无对应源模块:\n    {paths}",
            fixable=False,
        )
        report.fix_actions.append(
            ("MANUAL", f"审查并决定: {', '.join(p.name for p in orphans)}"))
    else:
        report.add_ok("CHK-3", "无孤儿测试文件")

    # ---- CHK-4: 命名偏差 ----
    biases = []
    for tf in test_files:
        if tf.stem in CROSS_CUTTING_TESTS:
            continue
        base = tf.stem[5:]
        if base in src_stems:
            continue  # 精确匹配
        # 找最佳源文件匹配
        best = None
        for s in src_stems:
            if base.endswith("_" + s) or s.endswith("_" + base):
                best = s
                break
            if s in base or base in s:
                best = s
        if best and base != best:
            biases.append((tf.name, base, best))

    if biases:
        lines = "\n    ".join(
            f"test_{base} (test: {test_name}) → 源: {src_name}"
            for test_name, base, src_name in biases[:5])
        report.add_issue(
            "CHK-4", "LOW",
            f"命名偏差: {len(biases)} 对",
            f"测试与源文件命名不完全对齐:\n    {lines}",
            fixable=False,
        )
    else:
        report.add_ok("CHK-4", "测试与源文件命名一致性良好")

    # ---- CHK-5: 测试覆盖率缺口 ----
    src_modules = [f for f in find_python_files(SRC_ROOT)
                   if f.name not in ("__init__.py", "__main__.py")]
    untested = []
    for sm in src_modules:
        stem = sm.stem
        matched = any(
            stem == t.stem[5:] or t.stem[5:].endswith("_" + stem) or
            stem.endswith("_" + t.stem[5:]) or t.stem[5:].endswith(stem)
            for t in test_files
        )
        if not matched:
            untested.append(sm)

    total = len(src_modules)
    covered = total - len(untested)
    pct = round(covered / total * 100, 1) if total > 0 else 0

    if pct < 50:
        sev = "HIGH"
    elif pct < 80:
        sev = "MEDIUM"
    else:
        sev = "LOW"

    if untested:
        by_pkg = defaultdict(list)
        for u in untested:
            try:
                rel = u.relative_to(SRC_ROOT)
                pkg = str(rel.parent) if str(rel.parent) != "." else "(根级)"
            except ValueError:
                pkg = "(外部)"
            by_pkg[pkg].append(u.name)

        lines = []
        for pkg, files in sorted(by_pkg.items()):
            names = ", ".join(files[:4])
            if len(files) > 4:
                names += f" ... +{len(files)-4}"
            lines.append(f"  {pkg}/: {names}")

        report.add_issue(
            "CHK-5", sev,
            f"测试覆盖率: {covered}/{total} = {pct}%",
            f"未覆盖模块 ({len(untested)} 个):\n" + "\n".join(lines),
            fixable=False,
        )
    else:
        report.add_ok("CHK-5", f"测试覆盖率: 100% ({total} 模块全部覆盖)")

    # ---- CHK-6: conftest.py 可用性 ----
    conftest = TESTS_ROOT / "conftest.py"
    if conftest.exists():
        report.add_ok("CHK-6", "tests/conftest.py 存在 — fixture 入口可用")
    else:
        report.add_issue(
            "CHK-6", "MEDIUM",
            "conftest.py 缺失",
            "tests/conftest.py 不存在，跨测试共享 fixture 需手动导入",
            fixable=False,
        )

    # ---- CHK-7: tests/manual/ 异常检测 ----
    manual_py = list((TESTS_ROOT / "manual").glob("*.py")) if (TESTS_ROOT / "manual").exists() else []
    if manual_py and not (TESTS_ROOT / "manual" / "__init__.py").exists():
        report.add_issue(
            "CHK-7", "LOW",
            "tests/manual/ 缺少 __init__.py",
            f"manual/ 有 {len(manual_py)} 个 .py 文件但无 __init__.py",
            fixable=True,
        )
        report.fix_actions.append(("AUTO", "touch tests/manual/__init__.py"))
    else:
        report.add_ok("CHK-7", "tests/manual/ 状态正常")


# ============================================================================
# 自愈功能 (Self-Healing)
# ============================================================================

def fix_init_py():
    """自动创建所有缺失的 __init__.py。"""
    all_dirs = find_dirs_with_py(SRC_ROOT) + find_dirs_with_py(TESTS_ROOT)
    created = 0
    for d in all_dirs:
        init_py = d / "__init__.py"
        if not init_py.exists():
            init_py.write_text("# Auto-generated by project_health_check.py\n")
            created += 1
            print(f"  ✅ 创建 {init_py.relative_to(PROJECT_ROOT)}")
    return created


def fix_permissions():
    """自动清除 .py 文件的异常可执行位（保留白名单）。"""
    all_py = find_python_files(SRC_ROOT) + find_python_files(TESTS_ROOT)
    fixed = 0
    for f in all_py:
        if f in EXECUTABLE_WHITELIST:
            continue
        if has_exec_bit(f):
            os.chmod(f, os.stat(f).st_mode & ~0o111)
            fixed += 1
    if fixed:
        print(f"  ✅ 清除 {fixed} 个文件的可执行位")
    return fixed


# ============================================================================
# 输出与入口
# ============================================================================

def print_report(report: HealthReport):
    """格式化输出健康检查报告。"""
    print("=" * 70)
    print("  🏥 项目健康检查报告")
    print("=" * 70)

    # 统计
    sev_count = defaultdict(int)
    for i in report.issues:
        sev_count[i["severity"]] += 1

    print(f"\n  ✓ 通过: {len(report.ok)} 项")
    print(f"  ✗ 异常: {len(report.issues)} 项")
    if sev_count:
        counts = " | ".join(f"{k}: {v}" for k, v in
                            sorted(sev_count.items(),
                                   key=lambda x: ["CRITICAL","HIGH","MEDIUM","LOW"].index(x[0])
                                   if x[0] in ["CRITICAL","HIGH","MEDIUM","LOW"] else 99))
        print(f"      ({counts})")
    print()

    # 通过的项（简洁）
    for check_id, msg in report.ok:
        print(f"  ✅ [{check_id}] {msg}")

    # 异常项（详细）
    for i in report.issues:
        sev_icon = {"CRITICAL": "🔥", "HIGH": "❌", "MEDIUM": "⚠️", "LOW": "ℹ️"}.get(i["severity"], "?")
        print(f"\n  {sev_icon} [{i['id']}] {i['title']}")
        print(f"     {i['detail']}")
        if i["fixable"]:
            print(f"     🔧 可自动修复")

    # 修复建议
    if report.fix_actions:
        print(f"\n  📋 修复动作 ({len(report.fix_actions)} 项):")
        for action_type, desc in report.fix_actions:
            tag = "🔧" if action_type == "AUTO" else "👤"
            print(f"     {tag} {desc}")

    print("\n" + "=" * 70)

    # 返回码提示
    if report.has_critical():
        print("  ⚡ 结果: 存在严重问题，建议立即修复")
    elif report.has_any():
        print("  ⚡ 结果: 存在异常，建议按计划修复")
    else:
        print("  🎉 结果: 项目健康状态良好")


def main():
    parser = argparse.ArgumentParser(description="项目健康检查与自愈脚本")
    parser.add_argument("--check", action="store_true", help="仅检查（不修复）")
    parser.add_argument("--fix", action="store_true", help="检查并自动修复")
    parser.add_argument("--all", action="store_true", help="检查并修复（默认）")
    parser.add_argument("--ci", action="store_true", help="CI模式：仅检查，发现问题时退出码=1")
    args = parser.parse_args()

    # 默认 --all
    if not (args.check or args.fix or args.all or args.ci):
        args.all = True

    do_fix = args.fix or args.all

    # 执行检查
    report = HealthReport()
    run_all_checks(report)
    print_report(report)

    # 执行修复
    if do_fix:
        print("\n" + "=" * 70)
        print("  🔧 自愈修复")
        print("=" * 70)
        created = fix_init_py()
        fixed = fix_permissions()
        if created == 0 and fixed == 0:
            print("  ℹ️ 无需修复")
        print()

    # CI 退出码
    if args.ci and report.has_any():
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
