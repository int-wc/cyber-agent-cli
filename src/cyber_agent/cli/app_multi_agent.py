"""多 Agent 编排相关逻辑：复杂度检测与四柱/原语管线执行。"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agent.runner import AgentRunner


# ── 管线选择 ──
# runtime_context 可用键：pipeline_mode = "primitive" | "four_pillar" | "auto"
PIPELINE_MODE_KEY = "pipeline_mode"

# SRC/渗透/挖洞类任务关键词 → 自动走原语工作流管线
_PRIMITIVE_WORKFLOW_KEYWORDS = (
    "SRC", "挖洞", "渗透", "漏洞挖掘", "渗透测试", "越权", "SSRF", "RCE",
    "IDOR", "未授权", "business_attr", "业务原语", "原语", "原语链",
    "攻击面", "API端点", "隐藏API", "路径遍历", "任意文件读取", "JWT",
    "鉴权绕过", "认证绕过", "webshell", "反序列化", "SSTI", "token签发",
    "测试端点", "漏洞复现", "POC", "exp", "猎洞", "资产分析",
)


def _detect_primitive_workflow(user_input: str) -> bool:
    """基于任务语义判断是否需要走原语工作流管线。

    当任务涉及漏洞挖掘/渗透/原语判定/攻击面分析时，使用 workflow 的
    业务原语解析 + 原语链利用管线；否则退回通用四柱管线。
    """
    text = user_input.strip()
    if not text:
        return False
    lowered = text.lower()
    for kw in _PRIMITIVE_WORKFLOW_KEYWORDS:
        if kw.lower() in lowered:
            return True
    return False


def _select_pipeline_mode(runtime_context: dict[str, object], user_input: str) -> str:
    """解析管线模式：显式配置优先，其次自动判定。"""
    configured = str(runtime_context.get(PIPELINE_MODE_KEY, "auto")).strip().lower()
    if configured in {"primitive", "four_pillar"}:
        return configured
    return "primitive" if _detect_primitive_workflow(user_input) else "four_pillar"


def _detect_task_complexity(user_input: str) -> bool:
    """基于结构特征判断任务是否需要多 Agent 协作。

    仅使用抽象结构指标，不依赖特定领域关键词：
    - 多语句/多问号 → 多个子问题
    - 并列连接词 → 多项独立操作
    - 序号/步骤标记 → 多阶段任务
    - 高信息密度 → 复杂需求
    - 指令 + 明确目标 → 单句但高密度的任务指令
    """
    text = user_input.strip()

    if len(text) <= 4:
        return False

    benchmark_markers = (
        "tsec benchmark",
        "tsecbench",
        "benchmark 跑分",
        "benchmark 正式测评",
        "benchmark 测评",
    )
    lowered = text.lower()
    if any(marker in lowered for marker in benchmark_markers):
        return True

    sentence_seps = len(re.findall(r"[。？！；\n]", text))
    if sentence_seps >= 3:
        return True

    question_marks = text.count("？") + text.count("?")
    if question_marks >= 2:
        return True

    coordination_markers = [
        r"并且", r"同时", r"以及", r"还有", r"另外", r"并(?!不)",
        r"然后", r"之后", r"接着", r"先.*再", r"首先.*然后",
        r"第一步", r"第二步", r"第三步",
        r"第\s*\d+\s*步", r"\d+\s*[\.、）\)]\s*[^\d\s]",
        r"一方面.*另一方面",
    ]
    for marker in coordination_markers:
        if re.search(marker, text):
            return True

    numbered_items = len(re.findall(
        r"(?:^|\n)\s*(?:\d+[\.\)、]|[一二三四五六七八九十]+[、．])", text
    ))
    if numbered_items >= 2:
        return True

    if len(text) > 150:
        return True

    comma_clauses = len(re.findall(r"[，,]", text))
    if len(text) > 60 and comma_clauses >= 3:
        return True

    # 指令 + 明确目标：单句但高密度的任务指令（如"对 X 做审计/挖掘/分析"）
    directive_patterns = [
        r"对[^\n]{1,60}?(做|进行|执行|实施|发起)",
        r"针对[^\n]{1,60}?(做|进行|执行|实施)",
        r"(?:请|帮我)\S{0,15}(分析|审计|检测|测试|扫描|检查|评估|排查|挖掘|审查)\S{0,8}",
        r"(审计|检测|测试|扫描|评估|排查|挖掘|分析)一下\S{0,30}",
    ]
    for pat in directive_patterns:
        if re.search(pat, text):
            return True

    return False


def _run_multi_agent_turn(
    user_input: str,
    runner: AgentRunner,
    runtime_context: dict[str, object],
    event_handler: Callable[[str, object], None] | None = None,
) -> None:
    """管线执行：原语工作流（业务原语解析 + 原语链利用）或四柱管线。

    - 原语工作流：ANALYST→原语解析、DIFFUSER→攻击面扩散、JUMPER→链跃迁、
      REFLECTOR→链裁决，再进入链执行闭环。
    - 四柱管线：分析→扩散→迁跃→反思→执行→审计→反思闭环。
    - 选择逻辑：runtime_context["pipeline_mode"] 显式指定，或按任务语义自动判定。
    """
    mode = _select_pipeline_mode(runtime_context, user_input)
    if mode == "primitive":
        from ..agent.primitive_pipeline import PrimitiveWorkflowPipeline
        pipeline_cls = PrimitiveWorkflowPipeline
        banner = "🧬 正在启动原语工作流管线（业务原语解析 → 原语链利用）..."
    else:
        from ..agent.pipeline import FourPillarPipeline
        pipeline_cls = FourPillarPipeline
        banner = "🚀 正在启动四柱 Agent 管线..."

    from .app import ensure_runtime_capabilities, renderer

    ensure_runtime_capabilities(runtime_context, runner)

    renderer.print_turn_start()
    renderer.console.print()
    renderer.console.print(f"[bold cyan]{banner}[/]")
    if mode == "primitive":
        renderer.console.print(
            "[dim]原语解析 → 攻击面扩散 → 链跃迁 → 链裁决[/]"
        )
    else:
        renderer.console.print(
            "[dim]分析为底 → 扩展为路 → 迁跃为辅 → 反思为主[/]"
        )

    auto_decision = bool(runtime_context.get("auto_decision", False))

    pipeline = pipeline_cls(
        runner=runner,
        runtime_context=runtime_context,
        renderer=renderer,
        event_handler=event_handler,
    )

    pipeline.run(user_input, auto_decision=auto_decision)
