"""多 Agent 编排相关逻辑：复杂度检测与多 Agent 回合执行。从 app.py 拆分以便维护。"""
from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..agent.runner import AgentRunner


def _build_auto_decision_handler(runtime_context: dict[str, object]):
    """构建自动决策处理器：用思考者(thinker)角色评估计划并自动选择子任务。

    当 --auto-decision 启用时，替代交互式菜单，
    由思考者分析决策者的计划，自动选出应执行的子任务并补充遗漏条件。
    """
    from ..config import settings

    def handler(stage: str, data: dict[str, Any]) -> dict[str, Any]:
        if stage != "plan_review":
            return {"action": "skip"}

        plan = data.get("plan", {})
        subtasks = plan.get("subtasks", [])
        reasoning = plan.get("reasoning", "")

        if not subtasks:
            return {"action": "selected", "selected_keys": []}

        from .app import renderer

        renderer.console.print()
        renderer.console.print(
            "  [bold magenta]🤔 思考者正在评估子任务计划...[/]"
        )

        # 构建子任务摘要供思考者分析
        tasks_text = "\n".join(
            f"  [{i}] 角色={t.get('role', '?')} | {t.get('task_description', '')[:200]}"
            for i, t in enumerate(subtasks)
        )

        # 加载思考者角色提示词
        from ..agent.roles import get_role_prompt, AgentRole
        thinker_prompt = get_role_prompt(AgentRole.THINKER)

        system_context = _build_system_context()

        system_prompt = f"""{thinker_prompt}

## 系统环境
{system_context}

## 决策者分析
{reasoning[:500]}

## 待评估的子任务
{tasks_text}

请分析以上子任务，输出 JSON 格式的执行决策。"""

        try:
            # 使用 orchestration 子模型（较轻量）做思考评估
            from .._lazy_imports import load_llm_for_api
            from langchain_core.messages import HumanMessage, SystemMessage

            service_name = str(runtime_context.get("service_name", "deepseek"))
            model_name = settings.subagent_model  # 用子模型，降低成本
            api_key = str(runtime_context.get("api_key", ""))
            base_url = str(runtime_context.get("base_url", "")) if runtime_context.get("base_url") is not None else None

            if not base_url:
                base_url = settings.resolve_base_url(service_name)

            llm_cls, is_anthropic = load_llm_for_api(base_url)
            import warnings
            kwargs = settings.get_chat_openai_kwargs(
                service_name,
                model_name=model_name,
                api_key=api_key,
                base_url=base_url,
            )
            if is_anthropic:
                kwargs["anthropic_api_key"] = kwargs.pop("api_key", "")
                kwargs.pop("openai_api_key", None)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*extra_body.*")
                llm = llm_cls(**kwargs)

            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content="请评估子任务计划并输出 JSON 决策。"),
            ])

            # 提取文本
            content = response.content
            if isinstance(content, list):
                content = "".join(
                    item if isinstance(item, str) else str(item.get("text", ""))
                    for item in content
                )
            content = str(content).strip()

            # 解析 JSON
            import json
            import re as _re
            # 去除 markdown 代码围栏
            if content.startswith("```"):
                lines = content.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines)
            decision = json.loads(content)

        except Exception as exc:
            from ..logging import log_error
            log_error("auto_decision", f"思考者评估失败，回退到执行全部子任务：{exc}")
            renderer.console.print(
                f"  [dim yellow]思考者评估失败({exc})，默认执行全部子任务。[/]"
            )
            return {
                "action": "selected",
                "selected_keys": [
                    f"{t.get('role', 'runner')}_{i}"
                    for i, t in enumerate(subtasks)
                ],
            }

        # 提取决策
        selected_indices = decision.get("selected_indices", list(range(len(subtasks))))
        additional_context = decision.get("additional_context", "")
        concerns = decision.get("concerns", "")
        think_reasoning = decision.get("reasoning", "")

        # 构建选中的 key 列表
        selected_keys = [
            f"{subtasks[i].get('role', 'runner')}_{i}"
            for i in selected_indices
            if 0 <= i < len(subtasks)
        ]

        # 展示思考者的决策
        renderer.console.print(
            f"  [dim]思考者决策: {think_reasoning[:200]}[/]"
        )
        if additional_context:
            renderer.console.print(
                f"  [dim yellow]补充条件: {additional_context[:200]}[/]"
            )
        if concerns:
            renderer.console.print(
                f"  [dim red]注意: {concerns[:200]}[/]"
            )
        renderer.console.print(
            f"  [dim]已选择 {len(selected_keys)}/{len(subtasks)} 个子任务[/]"
        )

        return {
            "action": "selected",
            "selected_keys": selected_keys,
            "custom_text": additional_context if additional_context else None,
        }

    return handler


def _build_system_context() -> str:
    """构建系统上下文信息，供角色提示词使用。"""
    from datetime import datetime, timezone
    import os
    now = datetime.now(timezone.utc).astimezone()
    return (
        f"当前日期时间: {now.strftime('%Y年%m月%d日 %H:%M')} "
        f"({now.strftime('%A')}, ISO {now.strftime('%Y-%m-%d')})\n"
        f"当前工作目录: {os.getcwd()}\n"
    )


def _build_user_interaction_handler(runtime_context: dict[str, object]):
    """构建用户交互处理器，用于多 Agent 协作中的方案选择和确认。"""

    def handler(stage: str, data: dict[str, Any]) -> dict[str, Any]:
        if stage != "plan_review":
            return {"action": "skip"}

        plan = data.get("plan", {})
        subtasks = plan.get("subtasks", [])

        # 如果没有子任务或非交互终端，跳过交互
        if not subtasks or not sys.stdin.isatty():
            return {"action": "skip"}

        from .selection_ui import present_multi_select_menu, SelectableOption
        from .app import renderer

        reasoning = plan.get("reasoning", "")

        # 展示决策者分析结果
        renderer.console.print()
        renderer.console.print(
            f"  [bold cyan]决策者分析:[/] [dim]{reasoning[:300]}[/]"
        )

        options = []
        for i, task in enumerate(subtasks):
            role = task.get("role", "?")
            desc = task.get("task_description", "")
            options.append(SelectableOption(
                key=f"{role}_{i}",
                label=f"[{role}] {desc}",
                metadata={"index": i, "role": role},
            ))

        # 展示交互式菜单
        result = present_multi_select_menu(
            options,
            title="选择要执行的子任务",
        )
        return result

    return handler


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
        r"第\s*\d+\s*步", r"\d+\s*[\.、）\)]\s*[^\d\s]",
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

    # 根据 --auto-decision 选择交互处理器
    auto_decision = runtime_context.get("auto_decision", False)
    if auto_decision:
        interaction_handler = _build_auto_decision_handler(runtime_context)
    else:
        interaction_handler = _build_user_interaction_handler(runtime_context)

    orchestrator = MultiAgentOrchestrator(
        tools=list(getattr(runner, "tools", [])),
        execution_controller=runtime_context.get("execution_controller"),
        event_handler=orchestration_event_handler,
        user_interaction_handler=interaction_handler,
        service_name=str(runtime_context.get("service_name", "deepseek")),
        model_name=str(runtime_context.get("model_name", "")),
        api_key=str(runtime_context.get("api_key", "")),
        base_url=str(runtime_context.get("base_url", "")) if runtime_context.get("base_url") is not None else None,
    )

    try:
        result = orchestrator.run(user_input)
        renderer.print_markdown(result)
        # 将编排器累计的 token 使用量同步到渲染器
        usage = orchestrator.get_usage_summary()
        if usage["total_tokens"] > 0:
            renderer.add_token_usage(usage["input_tokens"], usage["output_tokens"])
    except Exception as exc:
        renderer.print_error(f"多 Agent 协作失败：{exc}")


