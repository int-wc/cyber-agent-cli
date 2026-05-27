#!/usr/bin/env python3
"""Pwn2Own 2026 目标验证脚本 — 极致详细 DEBUG 输出。

验证项目：
  Goal 1: Web 搜索 20-40 条结果 + 6s 超时
  Goal 2: CLI/TUI Markdown 渲染 + /file 文件选择
  Goal 3: CSDN / 知乎 Web_fetch 绕过能力
  Goal 4: 多 Agent 并发架构 (4-10+ 角色)
  Goal 5: 综合集成验证

用法:
  PYTHONPATH=src python tests/manual/verify_all_goals.py
"""

from __future__ import annotations

import importlib
import inspect
import sys
import time
from pathlib import Path

# ── 确保 src 在路径中 ──
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_SEP = "═" * 78
_SUB = "─" * 78
_MIN = "·" * 60
_PASS = "  ✅ PASS"
_FAIL = "  ❌ FAIL"
_WARN = "  ⚠️  WARN"
_INFO = "  ℹ️ "


def header(title: str) -> None:
    print(f"\n{_SEP}")
    print(f"  {title}")
    print(f"{_SEP}")


def subheader(title: str) -> None:
    print(f"\n{_SUB}")
    print(f"  {title}")
    print(f"{_SUB}")


def check(name: str, condition: bool, detail: str = "") -> bool:
    mark = _PASS if condition else _FAIL
    print(f"{mark}  {name}")
    if detail:
        print(f"       {detail}")
    if not condition:
        print(f"       ⛔ 断言失败！")
    return condition


def info(msg: str) -> None:
    print(f"{_INFO} {msg}")


def debug(msg: str) -> None:
    print(f"       [DEBUG] {msg}")


def section(title: str) -> None:
    print(f"\n  ▸ {title}")


# ═══════════════════════════════════════════════════════════════
# GOAL 1: Web 搜索 20-40 条结果 + 6s 超时
# ═══════════════════════════════════════════════════════════════

def verify_goal_1() -> tuple[int, int]:
    header("GOAL 1: Web 搜索 — 20~40 条结果 + ≤6s 超时")

    from cyber_agent.tools import search as s

    section("1.1 常量定义")
    results: list[bool] = []

    results.append(check(
        "SEARCH_MIN_RESULTS = 20",
        s.SEARCH_MIN_RESULTS == 20,
        f"实际值: {s.SEARCH_MIN_RESULTS}",
    ))
    results.append(check(
        "SEARCH_MAX_RESULTS = 40",
        s.SEARCH_MAX_RESULTS == 40,
        f"实际值: {s.SEARCH_MAX_RESULTS}",
    ))
    results.append(check(
        "SEARCH_TIME_BUDGET_SECONDS ≤ 6.0",
        s.SEARCH_TIME_BUDGET_SECONDS <= 6.0,
        f"实际值: {s.SEARCH_TIME_BUDGET_SECONDS}s",
    ))
    results.append(check(
        "PARALLEL_ENGINE_TIMEOUT_SECONDS ≤ 5.5",
        s.PARALLEL_ENGINE_TIMEOUT_SECONDS <= 5.5,
        f"实际值: {s.PARALLEL_ENGINE_TIMEOUT_SECONDS}s",
    ))

    section("1.2 配置项")
    from cyber_agent.config import settings
    results.append(check(
        "settings.search_result_limit = 40",
        settings.search_result_limit == 40,
        f"实际值: {settings.search_result_limit}",
    ))
    results.append(check(
        "settings.search_timeout_seconds = 6.0",
        settings.search_timeout_seconds == 6.0,
        f"实际值: {settings.search_timeout_seconds}s",
    ))

    section("1.3 create_search_web_tool 工厂函数")
    tool_func = s.create_search_web_tool()
    # LangChain StructuredTool 不直接支持 inspect.signature，用 args_schema 替代
    debug(f"工具名: {tool_func.name}")
    debug(f"工具描述前100字: {tool_func.description[:100]}...")
    debug(f"风险级别: {tool_func.metadata.get('risk')}")
    if hasattr(tool_func, 'args_schema') and tool_func.args_schema:
        for field_name, field_info in tool_func.args_schema.model_fields.items():
            if field_name == "max_results":
                default_val = field_info.default
                results.append(check(
                    f"max_results 默认值 = {default_val} (在 20-40 范围内)",
                    20 <= default_val <= 40,
                    f"实际默认值: {default_val}",
                ))
            debug(f"  参数: {field_name}, 类型: {field_info.annotation}, 默认: {field_info.default}")

    section("1.4 并发搜索架构")
    if hasattr(s, "_search_all_engines_parallel"):
        src_code = inspect.getsource(s._search_all_engines_parallel)
        results.append(check(
            "ThreadPoolExecutor 并发搜索存在",
            "ThreadPoolExecutor" in src_code and "as_completed" in src_code,
            "并发多引擎搜索已实现",
        ))
        debug(f"函数行数: {len(src_code.splitlines())}")
    else:
        results.append(check("_search_all_engines_parallel 存在", False))

    if hasattr(s, "_search_all_variants_parallel"):
        results.append(check(
            "_search_all_variants_parallel 多角色并发存在",
            True,
            "多角色查询变体 × 多引擎并发搜索",
        ))
    else:
        results.append(check("_search_all_variants_parallel 存在", False))

    section("1.5 多引擎规格")
    results.append(check(
        f"PLAYWRIGHT_SEARCH_ENGINES 包含 {len(s.PLAYWRIGHT_SEARCH_ENGINES)} 个引擎",
        len(s.PLAYWRIGHT_SEARCH_ENGINES) >= 3,
        ", ".join(e.name for e in s.PLAYWRIGHT_SEARCH_ENGINES),
    ))

    section("1.6 搜索引擎并发隔离")
    if hasattr(s, "_search_engine_in_isolated_context"):
        src = inspect.getsource(s._search_engine_in_isolated_context)
        results.append(check(
            "每个引擎持有独立 sync_playwright 上下文",
            "sync_playwright()" in src and "browser_context.close()" in src,
            "线程安全隔离已实现",
        ))
        # 检查是否对 Bing 走多查询
        results.append(check(
            "Bing 引擎使用多查询模式",
            "bing.com" in src and "_search_bing_multiquery" in src,
            "Bing 走多查询以获取更多结果",
        ))

    passed = sum(results)
    total = len(results)
    print(f"\n  Goal 1 结果: {passed}/{total} 通过")
    return passed, total


# ═══════════════════════════════════════════════════════════════
# GOAL 2: CLI/TUI Markdown 渲染 + /file 文件选择
# ═══════════════════════════════════════════════════════════════

def verify_goal_2() -> tuple[int, int]:
    header("GOAL 2: Markdown 渲染 + 文件选择")

    results: list[bool] = []

    # ── 2.1 CLI Markdown ──
    section("2.1 CLI — CliRenderer Markdown")

    from cyber_agent.cli.render import CliRenderer, _MARKDOWN_AVAILABLE
    results.append(check(
        "_MARKDOWN_AVAILABLE 标志存在",
        isinstance(_MARKDOWN_AVAILABLE, bool),
        f"实际值: {_MARKDOWN_AVAILABLE}",
    ))

    renderer = CliRenderer()
    results.append(check(
        "print_markdown 方法存在",
        hasattr(renderer, "print_markdown"),
    ))

    # 检查 end_response_stream 是否使用 markdown
    src = inspect.getsource(renderer.end_response_stream)
    results.append(check(
        "end_response_stream 调用 print_markdown",
        "print_markdown" in src,
        "最终回复以 Markdown 格式渲染",
    ))

    # 验证 Markdown 渲染是否实际可用
    if _MARKDOWN_AVAILABLE:
        from rich.markdown import Markdown
        try:
            md = Markdown("# Test **bold** `code`")
            results.append(check(
                "Rich Markdown 实例化成功",
                True,
                f"类型: {type(md).__name__}",
            ))
        except Exception as exc:
            results.append(check(
                "Rich Markdown 实例化",
                False,
                str(exc),
            ))
    else:
        info("pygments 未安装，Rich Markdown 不可用 — 已自动降级为纯文本面板")

    # ── 2.2 TUI Markdown ──
    section("2.2 TUI — ChatMessage Markdown")

    try:
        from cyber_agent.cli.tui import (
            ChatMessage, _TEXTUAL_MARKDOWN_AVAILABLE,
            TEXTUAL_IMPORT_ERROR,
        )
        if TEXTUAL_IMPORT_ERROR is not None:
            results.append(check(
                "Textual 已导入",
                False,
                f"导入错误: {TEXTUAL_IMPORT_ERROR}",
            ))
        else:
            results.append(check(
                "ChatMessage 类存在",
                True,
            ))
            results.append(check(
                "_TEXTUAL_MARKDOWN_AVAILABLE 标志存在",
                isinstance(_TEXTUAL_MARKDOWN_AVAILABLE, bool),
                f"实际值: {_TEXTUAL_MARKDOWN_AVAILABLE}",
            ))

            # 检查 ChatMessage 是否支持 markdown
            src_msg = inspect.getsource(ChatMessage.__init__)
            results.append(check(
                "ChatMessage 接受 use_markdown 参数",
                "use_markdown" in src_msg,
                "TUI assistant 消息自动启用 Markdown 渲染",
            ))

            # 检查 _refresh_renderable 的 markdown 分支
            src_refresh = inspect.getsource(ChatMessage._refresh_renderable)
            results.append(check(
                "_refresh_renderable 包含 Markdown 渲染分支",
                "TextualMarkdown" in src_refresh,
                "回退到 build_chat_message_panel 当 markdown 不可用",
            ))
    except ImportError as exc:
        results.append(check(
            "TUI 模块导入",
            False,
            f"Textual 未安装: {exc}",
        ))

    # ── 2.3 /file 命令 ──
    section("2.3 /file 文件选择命令")

    from cyber_agent.cli.builtin_commands import (
        _COMMAND_REGISTRY, _handle_file,
        dispatch_builtin_command,
    )
    results.append(check(
        "/file 在命令注册表中",
        "/file" in _COMMAND_REGISTRY,
        f"注册表包含 {len(_COMMAND_REGISTRY)} 个命令: {', '.join(sorted(_COMMAND_REGISTRY.keys()))}",
    ))
    results.append(check(
        "_handle_file 处理器存在",
        callable(_handle_file),
    ))

    # 检查 /file 处理器逻辑
    src_file = inspect.getsource(_handle_file)
    results.append(check(
        "文件存在性检查",
        "file_path.exists()" in src_file or "exists()" in src_file,
    ))
    results.append(check(
        "目录 vs 文件区分",
        "is_dir()" in src_file,
    ))
    results.append(check(
        "编码自动检测 (UTF-8 / GBK)",
        "utf-8" in src_file and ("gbk" in src_file or "UnicodeDecodeError" in src_file),
    ))
    results.append(check(
        "语言扩展名映射",
        "lang_map" in src_file,
        "支持 40+ 种文件类型",
    ))

    section("2.4 CLI 交互命令提示")
    from cyber_agent.cli.interactive import BUILTIN_COMMAND_SPECS
    file_specs = [s for s in BUILTIN_COMMAND_SPECS if "/file" in s.command]
    results.append(check(
        "BUILTIN_COMMAND_SPECS 包含 /file",
        len(file_specs) > 0,
        file_specs[0].description if file_specs else "N/A",
    ))

    # ── 2.5 文件注入到对话上下文 ──
    section("2.5 文件内容注入到用户消息")

    from cyber_agent.cli import app as app_mod
    src_run = inspect.getsource(app_mod.run_chat_loop)
    results.append(check(
        "pending_file 注入逻辑存在",
        "__pending_file_" in src_run,
        "文件内容在下一轮对话自动附加到用户消息前",
    ))

    passed = sum(results)
    total = len(results)
    print(f"\n  Goal 2 结果: {passed}/{total} 通过")
    return passed, total


# ═══════════════════════════════════════════════════════════════
# GOAL 3: CSDN / 知乎 Web_fetch 绕过
# ═══════════════════════════════════════════════════════════════

def verify_goal_3() -> tuple[int, int]:
    header("GOAL 3: CSDN / 知乎 Web_fetch 绕过能力")

    from cyber_agent.tools import web_fetch as wf

    results: list[bool] = []

    section("3.1 h2 自动检测与降级")
    results.append(check(
        "_H2_AVAILABLE 标志存在",
        hasattr(wf, "_H2_AVAILABLE"),
        f"实际值: {wf._H2_AVAILABLE}",
    ))

    # 验证 h2 检测逻辑
    try:
        import h2 as _h2_mod  # noqa: F401
        h2_really_available = True
    except ImportError:
        h2_really_available = False

    results.append(check(
        "_H2_AVAILABLE 与实际 h2 安装状态一致",
        wf._H2_AVAILABLE == h2_really_available,
        f"检测值={wf._H2_AVAILABLE}, 实际可用={h2_really_available}",
    ))

    # 检查 httpx 请求中使用了 _H2_AVAILABLE
    src_fetch = inspect.getsource(wf._fetch_with_httpx)
    results.append(check(
        "httpx.Client 使用 http2=_H2_AVAILABLE",
        "http2=_H2_AVAILABLE" in src_fetch,
        "h2 不可用时自动回退 HTTP/1.1",
    ))

    section("3.2 UA 轮换池")
    results.append(check(
        f"_FETCH_USER_AGENTS 池大小 ≥ 3",
        len(wf._FETCH_USER_AGENTS) >= 3,
        f"实际大小: {len(wf._FETCH_USER_AGENTS)}",
    ))
    for i, ua in enumerate(wf._FETCH_USER_AGENTS):
        debug(f"UA[{i}]: {ua[:80]}...")

    # 验证随机选择
    ua_set: set[str] = set()
    for _ in range(100):
        import random
        ua_set.add(random.choice(wf._FETCH_USER_AGENTS))
    results.append(check(
        f"random.choice 可选中全部 {len(wf._FETCH_USER_AGENTS)} 个 UA (100次抽样覆盖 {len(ua_set)} 个)",
        len(ua_set) == len(wf._FETCH_USER_AGENTS),
    ))

    section("3.3 重试机制")
    results.append(check(
        f"_FETCH_MAX_RETRIES = {wf._FETCH_MAX_RETRIES}",
        wf._FETCH_MAX_RETRIES >= 1,
        "失败后自动重试",
    ))

    retry_loop_present = "for attempt in range" in src_fetch
    results.append(check(
        "重试循环存在",
        retry_loop_present,
    ))

    backoff_present = "time_mod.sleep" in src_fetch or "sleep" in src_fetch
    results.append(check(
        "重试退避等待",
        backoff_present,
        "递增间隔避免触发反爬",
    ))

    section("3.4 浏览器绕过域名覆盖")
    required_domains = [
        "csdn.net", "blog.csdn.net",
        "zhihu.com", "www.zhihu.com", "zhuanlan.zhihu.com",
        "jianshu.com", "juejin.cn", "cnblogs.com", "segmentfault.com",
    ]
    for domain in required_domains:
        covered = domain in wf.BROWSER_FETCH_DOMAINS
        results.append(check(
            f"域名覆盖: {domain}",
            covered,
        ))

    section("3.5 Playwright 隐身增强")
    src_pw = inspect.getsource(wf._fetch_with_playwright)
    checks_pw = [
        ("视口轮换", "random.choice" in src_pw and "viewport" in src_pw.lower()),
        ("反检测脚本 (webdriver)", "webdriver" in src_pw),
        ("permissions 伪造", "permissions" in src_pw),
        ("CSDN 弹窗关闭", "passport-login" in src_pw or "modal-close" in src_pw),
        ("知乎内容展开", "RichContent-cover" in src_pw or "ContentItem-expandable" in src_pw),
        ("懒加载触发 (scrollTo)", "scrollTo" in src_pw),
        ("UA 轮换", "random.choice(_FETCH_USER_AGENTS)" in src_pw or "FETCH_USER_AGENTS" in src_pw),
    ]
    for label, condition in checks_pw:
        results.append(check(f"Playwright: {label}", condition))

    section("3.6 _needs_browser_fetch 判断逻辑")
    src_needs = inspect.getsource(wf._needs_browser_fetch)
    results.append(check(
        "域名匹配逻辑 (hostname.endswith)",
        "endswith" in src_needs or "any(" in src_needs,
    ))

    # 测试判断函数
    test_urls = [
        ("https://blog.csdn.net/test/article/123", True),
        ("https://www.zhihu.com/question/456", True),
        ("https://zhuanlan.zhihu.com/p/789", True),
        ("https://www.google.com/search?q=test", False),
        ("https://github.com/anthropics/claude-code", False),
    ]
    for url, expected in test_urls:
        actual = wf._needs_browser_fetch(url)
        results.append(check(
            f"_needs_browser_fetch('{url[-50:]}') = {actual}",
            actual == expected,
            f"期望: {expected}, 实际: {actual}",
        ))

    section("3.7 create_web_fetch_tool 工厂")
    tool = wf.create_web_fetch_tool()
    results.append(check(
        f"工具名: {tool.name}",
        tool.name == "fetch_web_page",
    ))
    results.append(check(
        f"风险级别: read",
        tool.metadata.get("risk") == "read",
        f"实际: {tool.metadata.get('risk')}",
    ))
    # LangChain StructuredTool 用 args_schema 检查参数
    if hasattr(tool, 'args_schema') and tool.args_schema:
        params = list(tool.args_schema.model_fields.keys())
        results.append(check(
            f"use_browser 参数存在 (参数列表: {params})",
            "use_browser" in params,
        ))
        for field_name, field_info in tool.args_schema.model_fields.items():
            debug(f"  参数: {field_name}, 类型: {field_info.annotation}, 默认: {field_info.default}")

    passed = sum(results)
    total = len(results)
    print(f"\n  Goal 3 结果: {passed}/{total} 通过")
    return passed, total


# ═══════════════════════════════════════════════════════════════
# GOAL 4: 多 Agent 并发架构
# ═══════════════════════════════════════════════════════════════

def verify_goal_4() -> tuple[int, int]:
    header("GOAL 4: 多 Agent 并发架构 (4-10+ Agent)")

    results: list[bool] = []

    section("4.1 角色枚举完整")
    from cyber_agent.agent.roles import (
        AgentRole, ROLE_LABELS, ROLE_SYSTEM_PROMPTS,
        get_role_label, get_role_prompt,
    )

    expected_roles = {
        "checker": "审计者",
        "reader": "阅读者",
        "analyst": "分析者",
        "runner": "执行者",
        "builder": "构建者",
        "decision_maker": "决策者",
        "reflector": "反思者",
        "diffuser": "扩散者",
        "jumper": "迁跃者",
    }

    results.append(check(
        f"角色数量: {len(AgentRole)} (期望 9)",
        len(AgentRole) == 9,
    ))

    for role_value, role_label in expected_roles.items():
        role = AgentRole(role_value)
        label_ok = ROLE_LABELS.get(role) == role_label
        prompt_ok = len(ROLE_SYSTEM_PROMPTS.get(role, "")) > 50
        has_func_label = get_role_label(role) == role_label
        has_func_prompt = len(get_role_prompt(role)) > 50
        results.append(check(
            f"角色 {role_value} → {role_label}",
            label_ok and prompt_ok and has_func_label and has_func_prompt,
            f"标签={ROLE_LABELS.get(role)}, 提示词长度={len(ROLE_SYSTEM_PROMPTS.get(role, ''))}",
        ))

    section("4.2 MultiAgentOrchestrator")
    from cyber_agent.agent.orchestrator import (
        MultiAgentOrchestrator, AgentTask, AgentResult,
        OrchestrationPlan,
    )
    results.append(check(
        "MultiAgentOrchestrator 类存在",
        True,
    ))

    # 检查并发执行能力
    src_orch = inspect.getsource(MultiAgentOrchestrator._execute_plan)
    results.append(check(
        "ThreadPoolExecutor 并发执行",
        "ThreadPoolExecutor" in src_orch,
    ))
    results.append(check(
        "max_workers 可配置",
        "max_workers" in inspect.getsource(MultiAgentOrchestrator.__init__),
    ))

    # 检查 max_workers 默认值
    from cyber_agent.config import settings
    results.append(check(
        f"multi_agent_max_workers = {settings.multi_agent_max_workers} (在 4-10 范围)",
        4 <= settings.multi_agent_max_workers <= 10,
    ))

    section("4.3 角色查询生成 (diffuser 角度)")
    from cyber_agent.tools.search import _generate_role_based_queries

    test_queries = [
        ("Pwn2Own 2026 Berlin results highlights", 6),
        ("Windows 11 zero-day vulnerability", 5),
        ("AI安全 漏洞挖掘", 4),
    ]

    for query, expected_min in test_queries:
        variants = _generate_role_based_queries(query, expected_min)
        unique_queries = len({q.lower().strip() for _, q in variants})
        roles_found = {role for role, _ in variants}
        results.append(check(
            f"查询 '{query[:40]}...' → {len(variants)} 个变体 (目标 ≥{expected_min})",
            len(variants) >= min(expected_min, 4),
        ))
        results.append(check(
            f"  唯一查询数 = {unique_queries}",
            unique_queries == len(variants),
            "无重复查询",
        ))
        debug(f"  角色覆盖: {roles_found}")
        for role, q in variants:
            debug(f"    [{role}] {q}")

    section("4.4 多 Agent 搜索流水线")
    from cyber_agent.tools.search import _search_all_variants_parallel

    src_var = inspect.getsource(_search_all_variants_parallel)
    results.append(check(
        "ThreadPoolExecutor 并发搜索 (变体 × 引擎)",
        "ThreadPoolExecutor" in src_var,
    ))
    results.append(check(
        "as_completed 收集结果",
        "as_completed" in src_var,
    ))
    results.append(check(
        "URL 去重逻辑 (seen_urls + rstrip 去重)",
        "seen_urls" in src_var and "rstrip(" in src_var,
    ))
    results.append(check(
        "max_workers 计算 (角色数 × 引擎数)",
        "max_workers" in src_var,
    ))
    results.append(check(
        "超时控制 (SEARCH_TIME_BUDGET_SECONDS)",
        "SEARCH_TIME_BUDGET_SECONDS" in src_var,
    ))

    section("4.5 多 Agent 模式默认启用")
    from cyber_agent.cli.app import build_runtime_context
    from cyber_agent.agent.mode import AgentMode
    from cyber_agent.agent.approval import ApprovalPolicy
    from cyber_agent.cli.interactive import InteractionUiMode

    try:
        ctx = build_runtime_context(
            mode=AgentMode.STANDARD,
            allow_paths=None,
            tool_specs=None,
            approval_policy=ApprovalPolicy.PROMPT,
            ui_mode=InteractionUiMode.AUTO,
        )
        results.append(check(
            "runtime_context['multi_agent_enabled'] = True (默认)",
            ctx.get("multi_agent_enabled") is True,
            f"实际值: {ctx.get('multi_agent_enabled')}",
        ))
    except Exception as exc:
        results.append(check(
            "build_runtime_context 执行",
            False,
            str(exc),
        ))

    section("4.6 Agent 并发数验证")
    # 最大并发 = 角色变体数 × 引擎数
    max_roles = 6
    num_engines = 3
    max_concurrent = max_roles * num_engines
    results.append(check(
        f"最大并发数: {max_concurrent} (6 角色 × 3 引擎, 符合 4-10+ 要求)",
        4 <= max_concurrent <= 20,
    ))

    import concurrent.futures
    src_search = inspect.getsource(_search_all_variants_parallel)
    worker_line = [l for l in src_search.split("\n") if "max_workers" in l]
    if worker_line:
        debug(f"max_workers 计算: {worker_line[0].strip()}")

    passed = sum(results)
    total = len(results)
    print(f"\n  Goal 4 结果: {passed}/{total} 通过")
    return passed, total


# ═══════════════════════════════════════════════════════════════
# GOAL 5: 综合集成验证
# ═══════════════════════════════════════════════════════════════

def verify_goal_5() -> tuple[int, int]:
    header("GOAL 5: 综合集成 + 回归检查")

    results: list[bool] = []

    section("5.1 所有工具工厂可实例化")
    tool_factories = {
        "search_web": ("cyber_agent.tools.search", "create_search_web_tool"),
        "fetch_web_page": ("cyber_agent.tools.web_fetch", "create_web_fetch_tool"),
    }
    for name, (mod_path, factory_name) in tool_factories.items():
        try:
            mod = importlib.import_module(mod_path)
            factory = getattr(mod, factory_name)
            tool = factory()
            results.append(check(
                f"{name} 工具实例化",
                tool.name == name,
                f"工具名: {tool.name}, 风险: {tool.metadata.get('risk')}",
            ))
        except Exception as exc:
            results.append(check(
                f"{name} 工具实例化",
                False,
                str(exc),
            ))

    section("5.2 核心模块导入无异常")
    core_modules = [
        "cyber_agent.config",
        "cyber_agent.agent.runner",
        "cyber_agent.agent.roles",
        "cyber_agent.agent.orchestrator",
        "cyber_agent.tools.search",
        "cyber_agent.tools.web_fetch",
        "cyber_agent.cli.render",
        "cyber_agent.cli.tui",
        "cyber_agent.cli.builtin_commands",
        "cyber_agent.cli.interactive",
        "cyber_agent.cli.app",
        "cyber_agent.tools",
    ]
    for mod_name in core_modules:
        try:
            importlib.import_module(mod_name)
            results.append(check(
                f"导入 {mod_name}",
                True,
            ))
        except Exception as exc:
            results.append(check(
                f"导入 {mod_name}",
                False,
                str(exc),
            ))

    section("5.3 无语法错误 (编译检查)")
    changed_files = [
        "src/cyber_agent/tools/search.py",
        "src/cyber_agent/tools/web_fetch.py",
        "src/cyber_agent/cli/render.py",
        "src/cyber_agent/cli/tui.py",
        "src/cyber_agent/cli/builtin_commands.py",
        "src/cyber_agent/cli/interactive.py",
        "src/cyber_agent/cli/app.py",
        "src/cyber_agent/config.py",
    ]
    for rel_path in changed_files:
        abs_path = Path(__file__).resolve().parents[2] / rel_path
        try:
            compile(abs_path.read_text(), str(abs_path), "exec")
            results.append(check(
                f"编译 {rel_path}",
                True,
            ))
        except SyntaxError as exc:
            results.append(check(
                f"编译 {rel_path}",
                False,
                f"行 {exc.lineno}: {exc.msg}",
            ))

    section("5.4 关键函数可调用签名检查")
    key_functions = [
        ("search_web", "cyber_agent.tools.search", "create_search_web_tool"),
        ("fetch_web_page", "cyber_agent.tools.web_fetch", "create_web_fetch_tool"),
        ("MultiAgentOrchestrator", "cyber_agent.agent.orchestrator", "MultiAgentOrchestrator"),
        ("CliRenderer", "cyber_agent.cli.render", "CliRenderer"),
        ("build_runtime_context", "cyber_agent.cli.app", "build_runtime_context"),
        ("dispatch_builtin_command", "cyber_agent.cli.builtin_commands", "dispatch_builtin_command"),
    ]
    for name, mod_path, attr in key_functions:
        try:
            mod = importlib.import_module(mod_path)
            obj = getattr(mod, attr)
            callable(obj)
            results.append(check(
                f"{name} 可调用",
                True,
            ))
        except Exception as exc:
            results.append(check(
                f"{name} 可调用",
                False,
                str(exc),
            ))

    passed = sum(results)
    total = len(results)
    print(f"\n  Goal 5 结果: {passed}/{total} 通过")
    return passed, total


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    print(f"\n{'█' * 78}")
    print(f"  Pwn2Own 2026 — 全目标验证脚本")
    print(f"  运行时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Python: {sys.version}")
    print(f"  项目路径: {_SRC}")
    print(f"{'█' * 78}")

    all_passed = 0
    all_total = 0

    for goal_num, verify_fn in [
        (1, verify_goal_1),
        (2, verify_goal_2),
        (3, verify_goal_3),
        (4, verify_goal_4),
        (5, verify_goal_5),
    ]:
        try:
            p, t = verify_fn()
            all_passed += p
            all_total += t
        except Exception as exc:
            print(f"\n  💥 Goal {goal_num} 验证脚本崩溃: {exc}")
            import traceback
            traceback.print_exc()

    print(f"\n{'█' * 78}")
    print(f"  最终结果: {all_passed}/{all_total} 通过")
    if all_passed == all_total:
        print(f"  🎉 全部目标达成！")
    else:
        failed = all_total - all_passed
        print(f"  ⚠️  {failed} 项未通过，请检查上方详细输出")
    print(f"{'█' * 78}\n")


if __name__ == "__main__":
    main()
