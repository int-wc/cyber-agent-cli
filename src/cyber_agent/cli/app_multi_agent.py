"""多 Agent 编排相关逻辑：复杂度检测与多 Agent 回合执行。从 app.py 拆分以便维护。"""
from __future__ import annotations

import re
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

    # 极短输入 → 简单
    if len(text) <= 4:
        return False

    # 多句子分隔符 → 多个独立表达
    sentence_seps = len(re.findall(r"[。？！；\n]", text))
    if sentence_seps >= 3:
        return True

    # 多个问号 → 多个并列问题
    question_marks = text.count("？") + text.count("?")
    if question_marks >= 2:
        return True

    # 并列/递进/时序连接词 → 多项操作组合
    coordination_markers = [
        r"并且", r"同时", r"以及", r"还有", r"另外", r"并(?!不)",
        r"然后", r"之后", r"接着", r"先.*再", r"首先.*然后",
        r"第一步", r"第二步", r"第三步",
        r"第\s*\d+\s*步", r"\d+\s*[\.、）\)]\s*\S",
        r"一方面.*另一方面",
    ]
    for marker in coordination_markers:
        if re.search(marker, text):
            return True

    # 列举序号（1. 2. 3. 或 1) 2) 或 一、二、）
    numbered_items = len(re.findall(
        r"(?:^|\n)\s*(?:\d+[\.\)、]|[一二三四五六七八九十]+[、．])", text
    ))
    if numbered_items >= 2:
        return True

    # 高信息密度：长度 > 150 字 → 很可能包含多个需求
    if len(text) > 150:
        return True

    # 中长度 + 多逗号分句 → 复合描述
    comma_clauses = len(re.findall(r"[，,]", text))
    if len(text) > 60 and comma_clauses >= 3:
        return True

    return False


def _run_multi_agent_turn(
    user_input: str,
    runner: AgentRunner,
    runtime_context: dict[str, object],
) -> None:
    """使用多 Agent 编排器执行一轮对话。"""
    from ..agent.orchestrator import MultiAgentOrchestrator
    from .app import ensure_runtime_capabilities, renderer

    # 确保工具等能力已延迟加载完成
    ensure_runtime_capabilities(runtime_context, runner)

    renderer.print_turn_start()
    renderer.print_info("[bold cyan]🚀 正在启动多 Agent 协作模式...[/]")

    # 编排器事件 → 渲染器进度
    def orchestration_event_handler(event_type: str, payload: object) -> None:
        if event_type == "orchestration_start":
            pass  # 已在上面打印
        elif event_type == "orchestration_planning":
            renderer.print_orchestration_planning(str(payload.get("input", "")))
        elif event_type == "orchestration_plan_done":
            renderer.print_orchestration_plan_done(
                subtask_count=int(payload.get("subtask_count", 0)),
                reasoning=str(payload.get("reasoning", "")),
            )
        elif event_type == "orchestration_executing":
            renderer.print_orchestration_executing(
                int(payload.get("subtask_count", 0))
            )
        elif event_type == "subtask_complete":
            renderer.print_subtask_complete(
                role=str(payload.get("role", "?")),
                success=bool(payload.get("success", False)),
                elapsed_ms=float(payload.get("elapsed_ms", 0)),
                output_summary=str(payload.get("output_summary", "")),
                output_length=int(payload.get("output_length", 0)),
            )
        elif event_type == "orchestration_checking":
            renderer.print_orchestration_checking(
                int(payload.get("result_count", 0))
            )
        elif event_type == "orchestration_reflecting":
            renderer.print_orchestration_reflecting(
                int(payload.get("failed_count", 0))
            )
        elif event_type == "orchestration_iteration":
            pass  # 静默处理
        elif event_type == "orchestration_synthesizing":
            renderer.print_orchestration_synthesize()
        elif event_type == "orchestration_end":
            renderer.print_orchestration_done(
                int(payload.get("total_results", 0))
            )

    orchestrator = MultiAgentOrchestrator(
        tools=list(getattr(runner, "tools", [])),
        execution_controller=runtime_context.get("execution_controller"),
        event_handler=orchestration_event_handler,
        service_name=str(runtime_context.get("service_name", "deepseek")),
        model_name=str(runtime_context.get("model_name", "")),
        api_key=str(runtime_context.get("api_key", "")),
        base_url=str(runtime_context.get("base_url", "")) if runtime_context.get("base_url") is not None else None,
    )

    try:
        result = orchestrator.run(user_input)
        renderer.print_markdown(result)
    except Exception as exc:
        renderer.print_error(f"多 Agent 协作失败：{exc}")


