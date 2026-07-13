"""多 Agent 编排相关逻辑：复杂度检测与四柱管线执行。"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agent.runner import AgentRunner


def _detect_task_complexity(user_input: str) -> bool:
    """基于结构特征判断任务是否需要多 Agent 协作。

    仅使用抽象结构指标，不依赖特定领域关键词：
    - 多语句/多问号 → 多个子问题
    - 并列连接词 → 多项独立操作
    - 序号/步骤标记 → 多阶段任务
    - 高信息密度 → 复杂需求
    """
    text = user_input.strip()

    if len(text) <= 4:
        return False

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

    return False


def _run_multi_agent_turn(
    user_input: str,
    runner: AgentRunner,
    runtime_context: dict[str, object],
    event_handler: Callable[[str, object], None] | None = None,
) -> None:
    """四柱管线：分析→扩散→迁跃→反思→执行→审计→反思闭环。

    所有 10 个角色各司其职：
    - 四柱核心：ANALYST(底) → DIFFUSER(路) → JUMPER(辅) → REFLECTOR(主)
    - 执行服务：DECISION_MAKER → THINKER/用户 → RUNNER/READER/BUILDER → CHECKER → REFLECTOR(闭环)
    """
    from ..agent.pipeline import FourPillarPipeline
    from .app import ensure_runtime_capabilities, renderer

    ensure_runtime_capabilities(runtime_context, runner)

    renderer.print_turn_start()
    renderer.console.print()
    renderer.console.print("[bold cyan]🚀 正在启动四柱 Agent 管线...[/]")
    renderer.console.print(
        "[dim]分析为底 → 扩展为路 → 迁跃为辅 → 反思为主[/]"
    )

    auto_decision = bool(runtime_context.get("auto_decision", False))

    pipeline = FourPillarPipeline(
        runner=runner,
        runtime_context=runtime_context,
        renderer=renderer,
        event_handler=event_handler,
    )

    pipeline.run(user_input, auto_decision=auto_decision)
