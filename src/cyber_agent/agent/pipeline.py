"""四柱 Agent 管线：反思为主、迁跃为辅、分析为底、扩展为路。

管线流程:
  Phase 1 - 四柱思考（纯 LLM 调用，无工具，按序传递上下文）
    1. 分析者 ANALYST   → 深度分析 —— 为底
    2. 扩散者 DIFFUSER  → 路径探索 —— 为路
    3. 迁跃者 JUMPER    → 创造跨越 —— 为辅
    4. 反思者 REFLECTOR → 综合审视 + 制定执行计划 —— 为主

  Phase 2 - 执行循环（反思闭环，最多 3 轮）
    5. 决策者 DECISION_MAKER → 分解子任务
    6. 思考者 THINKER / 用户  → 选择子任务
    7. 执行者 RUNNER/READER/BUILDER → 顺序执行（runner.run()，已验证的工具链路）
    8. 审计者 CHECKER    → 验证结果
    9. 反思者 REFLECTOR  → 审视结果，决定循环继续或结束
"""

from __future__ import annotations

import concurrent.futures
import json
import re as _re_mod
import threading
import time as time_mod
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..execution_control import ExecutionInterruptedError
from .roles import AgentRole, get_role_label, get_role_prompt

if TYPE_CHECKING:
    from .runner import AgentRunner

# ── 超时与熔断常量 ──
SUBTASK_TIMEOUT_SECONDS = 180        # 单子任务超时（秒）
LLM_CALL_TIMEOUT_SECONDS = 120       # 单次角色 LLM 调用超时（秒）
CIRCUIT_BREAKER_CONSECUTIVE_FAILS = 2  # 连续失败 N 次触发熔断
PIPELINE_TOTAL_TIMEOUT_SECONDS = 600   # 管线总时限（秒）


class PipelineCircuitBreakerError(RuntimeError):
    """连续子任务失败触发的熔断异常。"""


class FourPillarPipeline:
    """四柱管线协调器。所有 10 个角色各司其职。"""

    def __init__(
        self,
        *,
        runner: AgentRunner,
        runtime_context: dict[str, object],
        renderer: Any,
    ) -> None:
        self._runner = runner
        self._runtime_context = runtime_context
        self._renderer = renderer
        self._llm: Any = None

        # 累计 token（供 renderer 读取）
        self.cumulative_input_tokens = 0
        self.cumulative_output_tokens = 0

        # 熔断器状态
        self._consecutive_failures = 0
        self._pipeline_start_time = 0.0  # 管线启动时间戳

    # ── LLM 管理 ──
    def _get_llm(self) -> Any:
        """懒加载用于角色思考的 LLM（无工具绑定，纯文本调用）。"""
        if self._llm is not None:
            return self._llm

        from .._lazy_imports import load_chat_openai
        from ..config import settings

        service_name = str(self._runtime_context.get("service_name", "deepseek"))
        api_key = str(self._runtime_context.get("api_key", ""))
        base_url = str(self._runtime_context.get("base_url", "")) if self._runtime_context.get("base_url") is not None else None
        if not base_url:
            base_url = settings.resolve_base_url(service_name)

        # 强制使用 OpenAI 格式端点
        if "api.deepseek.com/anthropic" in base_url:
            base_url = "https://api.deepseek.com/v1"

        # 角色思考用子模型降低成本
        model_name = settings.subagent_model.replace("[1m]", "").strip()

        kwargs = settings.get_chat_openai_kwargs(
            service_name,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
        )
        # 禁用 thinking，角色不需要深度推理
        if "extra_body" in kwargs and isinstance(kwargs["extra_body"], dict):
            kwargs["extra_body"]["thinking"] = {"type": "disabled"}

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*extra_body.*")
            self._llm = load_chat_openai()(**kwargs)
        return self._llm

    def _call_role(
        self,
        role: AgentRole,
        user_input: str,
        *,
        context: str = "",
        extra_instruction: str = "",
    ) -> str:
        """调用单个角色 LLM，返回纯文本输出。思考角色无工具绑定。"""
        label = get_role_label(role)
        system_prompt = get_role_prompt(role)
        system_context = self._build_system_context()

        full_system = f"""{system_prompt}

## 系统环境
{system_context}"""

        user_content = f"## 用户任务\n{user_input}"
        if context:
            user_content += f"\n\n## 前序角色输出（请基于此继续）\n{context}"
        if extra_instruction:
            user_content += f"\n\n{extra_instruction}"

        try:
            response = self._get_llm().invoke([
                SystemMessage(content=full_system),
                HumanMessage(content=user_content),
            ])
            self._track_llm_usage(response)
            return self._extract_text(response)
        except Exception as exc:
            from ..logging import log_error
            log_error("pipeline", f"{label} 调用失败：{exc}")
            return f"[{label} 调用失败: {exc}]"

    # ── 超时与熔断 ──
    def _check_pipeline_timeout(self) -> None:
        """检查管线总时限，超时则抛出异常。"""
        elapsed = time_mod.monotonic() - self._pipeline_start_time
        if elapsed > PIPELINE_TOTAL_TIMEOUT_SECONDS:
            raise TimeoutError(
                f"管线总时限已到（{PIPELINE_TOTAL_TIMEOUT_SECONDS}s），"
                f"已运行 {elapsed:.0f}s"
            )

    def _check_circuit_breaker(self) -> None:
        """检查熔断器：连续失败超过阈值则抛出异常。"""
        if self._consecutive_failures >= CIRCUIT_BREAKER_CONSECUTIVE_FAILS:
            raise PipelineCircuitBreakerError(
                f"连续 {self._consecutive_failures} 个子任务失败，触发熔断保护。"
                f"请检查任务是否合理或简化需求后重试。"
            )

    def _call_role_with_timeout(
        self,
        role: AgentRole,
        user_input: str,
        *,
        context: str = "",
        extra_instruction: str = "",
        timeout: float = LLM_CALL_TIMEOUT_SECONDS,
    ) -> str:
        """带超时的角色 LLM 调用。在线程池中执行，超时则取消。"""
        renderer = self._renderer

        def _invoke():
            return self._call_role(
                role, user_input,
                context=context,
                extra_instruction=extra_instruction,
            )

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_invoke)
                return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            label = get_role_label(role)
            renderer.console.print(
                f"  [red]✗ {label} 超时[/] [dim]({timeout}s 未响应)[/]"
            )
            return f"[{label} 调用超时: {timeout}s 内未返回]"
        except Exception as exc:
            label = get_role_label(role)
            renderer.console.print(
                f"  [red]✗ {label} 异常[/] [dim]({exc})[/]"
            )
            return f"[{label} 异常: {exc}]"

    def _run_subtask_with_timeout(
        self,
        subtask_prompt: str,
        role_label: str,
        timeout: float = SUBTASK_TIMEOUT_SECONDS,
    ) -> str:
        """带超时的子任务执行。通过 ExecutionController 安全中断。

        后台定时器在超时后调用 execution_controller.request_stop()，
        runner.run() 在关键边界检测 cancel 事件并抛出 ExecutionInterruptedError。
        """
        controller = getattr(self._runner, "execution_controller", None)
        renderer = self._renderer

        if controller is None:
            # 无控制器：直接调用，无超时保护
            return self._runner.run(subtask_prompt, verbose=False, event_handler=None)

        timer_fired = threading.Event()

        def _timeout_handler():
            timer_fired.set()
            controller.request_stop(f"子任务超时（{timeout}s）")

        timer = threading.Timer(timeout, _timeout_handler)
        timer.daemon = True
        timer.start()

        try:
            result = self._runner.run(
                subtask_prompt,
                verbose=False,
                event_handler=None,
            )
            return result
        except ExecutionInterruptedError:
            if timer_fired.is_set():
                raise TimeoutError(
                    f"子任务执行超时（{timeout}s），已自动中断。"
                )
            raise
        finally:
            timer.cancel()

    def _track_llm_usage(self, response: Any) -> None:
        """从 LLM 响应中提取并累计 token 使用量。"""
        from .runner import _extract_usage_from_chunk, _estimate_tokens_from_text
        usage = _extract_usage_from_chunk(response)
        if usage is None:
            usage = {
                "input_tokens": 0,
                "output_tokens": _estimate_tokens_from_text(self._extract_text(response)),
                "total_tokens": _estimate_tokens_from_text(self._extract_text(response)),
            }
        self.cumulative_input_tokens += usage["input_tokens"]
        self.cumulative_output_tokens += usage["output_tokens"]

    def get_usage_summary(self) -> dict[str, int]:
        """返回累计 token 使用量。"""
        return {
            "input_tokens": self.cumulative_input_tokens,
            "output_tokens": self.cumulative_output_tokens,
            "total_tokens": self.cumulative_input_tokens + self.cumulative_output_tokens,
        }

    @staticmethod
    def _build_system_context() -> str:
        from datetime import datetime, timezone
        import os
        now = datetime.now(timezone.utc).astimezone()
        return (
            f"当前日期时间: {now.strftime('%Y年%m月%d日 %H:%M')} "
            f"({now.strftime('%A')}, ISO {now.strftime('%Y-%m-%d')})\n"
            f"当前工作目录: {os.getcwd()}\n"
        )

    @staticmethod
    def _extract_text(response: Any) -> str:
        content = getattr(response, "content", "")
        if isinstance(content, list):
            return "".join(
                item if isinstance(item, str) else str(item.get("text", ""))
                for item in content
            )
        return str(content)

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = _re_mod.search(r"\{[\s\S]*\}", text)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return {}

    # ══════════════════════════════════════════════════════════════
    # 管线主入口
    # ══════════════════════════════════════════════════════════════
    def run(self, user_input: str, auto_decision: bool = False) -> None:
        """执行完整的四柱管线。"""
        renderer = self._renderer
        self._pipeline_start_time = time_mod.monotonic()
        self._consecutive_failures = 0

        try:
            self._run_phases(user_input, auto_decision)
        except (TimeoutError, PipelineCircuitBreakerError) as exc:
            renderer.console.print()
            renderer.console.print(
                f"  [bold red]⛔ 管线中止: {exc}[/]"
            )
        finally:
            # 同步 token 到 renderer
            self._renderer.add_token_usage(
                self.cumulative_input_tokens,
                self.cumulative_output_tokens,
            )

    def _run_phases(self, user_input: str, auto_decision: bool) -> None:
        """管线主逻辑，含超时保护和熔断机制。"""
        renderer = self._renderer

        # ── Phase 1: 四柱思考 ──
        renderer.console.print()
        renderer.console.print("[bold cyan]🧠 四柱思考阶段[/]")
        renderer.console.print("[dim]分析为底 → 扩展为路 → 迁跃为辅 → 反思为主[/]")

        # 1. 分析者（底）
        renderer.console.print("  [dim]⏳ 分析者 正在深度分析...[/]")
        self._check_pipeline_timeout()
        t0 = time_mod.monotonic()
        analysis = self._call_role_with_timeout(AgentRole.ANALYST, user_input)
        renderer.console.print(
            f"  [green]✓ 分析者 完成[/] [dim]({(time_mod.monotonic()-t0)*1000:.0f}ms)[/]"
        )
        renderer.console.print(
            f"  [dim]{analysis[:200].replace(chr(10), ' ')}...[/]"
        )

        # 2. 扩散者（路）
        self._check_pipeline_timeout()
        renderer.console.print("  [dim]⏳ 扩散者 正在探索路径...[/]")
        t0 = time_mod.monotonic()
        diffusion = self._call_role_with_timeout(
            AgentRole.DIFFUSER, user_input,
            context=f"## 分析结论\n{analysis}",
        )
        renderer.console.print(
            f"  [green]✓ 扩散者 完成[/] [dim]({(time_mod.monotonic()-t0)*1000:.0f}ms)[/]"
        )

        # 3. 迁跃者（辅）
        self._check_pipeline_timeout()
        renderer.console.print("  [dim]⏳ 迁跃者 正在创造性跨越...[/]")
        t0 = time_mod.monotonic()
        jump = self._call_role_with_timeout(
            AgentRole.JUMPER, user_input,
            context=f"## 分析者\n{analysis}\n\n## 扩散者\n{diffusion}",
        )
        renderer.console.print(
            f"  [green]✓ 迁跃者 完成[/] [dim]({(time_mod.monotonic()-t0)*1000:.0f}ms)[/]"
        )

        # 4. 反思者（主）—— 综合审视 + 制定执行计划
        self._check_pipeline_timeout()
        renderer.console.print("  [dim]⏳ 反思者 正在综合审视...[/]")
        t0 = time_mod.monotonic()
        reflection = self._call_role_with_timeout(
            AgentRole.REFLECTOR, user_input,
            context=(
                f"## 分析者（分析为底）\n{analysis}\n\n"
                f"## 扩散者（扩展为路）\n{diffusion}\n\n"
                f"## 迁跃者（迁跃为辅）\n{jump}"
            ),
            extra_instruction=(
                "请综合以上三个角色的输出，做出最终判断。"
                "输出执行计划时要具体、可操作，每个子任务分配明确的执行角色（runner/reader/builder）。"
            ),
        )
        elapsed = (time_mod.monotonic() - t0) * 1000
        renderer.console.print(
            f"  [green]✓ 反思者 完成[/] [dim]({elapsed:.0f}ms)[/]"
        )

        # 展示反思者输出
        renderer.console.print()
        renderer.console.print("[bold yellow]📋 反思者审视结论[/]")
        renderer.print_markdown(reflection[:1500])

        # ── Phase 2: 执行循环（反思闭环）──
        max_iterations = 3
        all_results: list[str] = []
        iteration = 0  # 在循环外声明，供 Phase 3 引用

        for iteration in range(1, max_iterations + 1):
            self._check_pipeline_timeout()

            renderer.console.print()
            renderer.console.print(
                f"[bold magenta]⚡ 执行循环 第 {iteration}/{max_iterations} 轮[/]"
            )

            # 5. 决策者 → 分解子任务
            renderer.console.print("  [dim]⏳ 决策者 正在分解子任务...[/]")
            iter_context = reflection
            if all_results:
                iter_context += f"\n\n## 上一轮执行结果\n" + "\n".join(
                    f"- {r[:300]}" for r in all_results[-3:]
                )
            plan_json = self._call_role_with_timeout(
                AgentRole.DECISION_MAKER, user_input,
                context=f"## 反思者执行计划\n{iter_context}",
            )
            plan = self._parse_json(plan_json)
            subtasks = plan.get("subtasks", [])
            reasoning = plan.get("reasoning", "")

            if not subtasks:
                renderer.console.print("  [dim]决策者未分解出子任务，结束执行。[/]")
                break

            renderer.console.print(
                f"  [green]✓ 决策者 分解出 {len(subtasks)} 个子任务[/]"
            )

            # 6. 选择子任务
            selected_indices = list(range(len(subtasks)))
            additional_context = ""

            if auto_decision:
                selected_indices, additional_context = self._auto_select(
                    subtasks, reasoning,
                )
            else:
                selected_indices, additional_context = self._user_select(
                    subtasks, reasoning, iteration,
                )

            if not selected_indices:
                renderer.console.print("  [dim]未选择任何子任务，结束执行。[/]")
                break

            renderer.console.print(
                f"  [dim]已选择 {len(selected_indices)}/{len(subtasks)} 个子任务[/]"
            )

            # 7. 顺序执行子任务（带超时和熔断）
            renderer.console.print()
            renderer.console.print(
                f"[bold yellow]🔧 执行 {len(selected_indices)} 个子任务[/]"
                f" [dim](超时={SUBTASK_TIMEOUT_SECONDS}s, 熔断阈值={CIRCUIT_BREAKER_CONSECUTIVE_FAILS})[/]"
            )

            round_results: list[str] = []
            circuit_broken = False

            for idx in selected_indices:
                if idx >= len(subtasks):
                    continue

                # 每轮子任务前检查总管线和熔断器
                self._check_pipeline_timeout()
                try:
                    self._check_circuit_breaker()
                except PipelineCircuitBreakerError as exc:
                    renderer.console.print(
                        f"  [bold red]⛔ {exc}[/]"
                    )
                    circuit_broken = True
                    break

                task = subtasks[idx]
                role_str = task.get("role", "runner")
                desc = task.get("task_description", str(task))
                ctx = task.get("context", "")
                if additional_context:
                    ctx = f"{ctx}\n补充: {additional_context}" if ctx else additional_context

                renderer.console.print(
                    f"  [dim]── [{role_str}] {desc[:80]}...[/]"
                )
                start = time_mod.monotonic()

                try:
                    subtask_prompt = (
                        f"你是{get_role_label(self._str_to_role(role_str))}。"
                        f"请完成以下子任务，只做这一件事，完成后给出结果摘要。\n\n"
                        f"子任务: {desc}\n"
                    )
                    if ctx:
                        subtask_prompt += f"\n上下文: {ctx}\n"
                    if reasoning:
                        subtask_prompt += f"\n整体背景: {reasoning[:300]}\n"
                    subtask_prompt += "\n请直接调用工具完成此子任务，给出核心结果。"

                    result = self._run_subtask_with_timeout(
                        subtask_prompt, get_role_label(self._str_to_role(role_str)),
                    )
                    elapsed = (time_mod.monotonic() - start) * 1000
                    renderer.console.print(
                        f"  [green]✓ 完成[/] [dim]({elapsed:.0f}ms, {len(result)}字)[/]"
                    )
                    round_results.append(
                        f"## [{role_str}] {desc}\n{result}"
                    )
                    # 成功 → 重置熔断计数器
                    self._consecutive_failures = 0

                except TimeoutError as exc:
                    elapsed = (time_mod.monotonic() - start) * 1000
                    self._consecutive_failures += 1
                    renderer.console.print(
                        f"  [red]⏰ 超时[/] [dim]({elapsed:.0f}ms)[/]: {exc}"
                    )
                    round_results.append(
                        f"## [{role_str}] {desc}\n❌ 超时: {exc}"
                    )

                except Exception as exc:
                    elapsed = (time_mod.monotonic() - start) * 1000
                    self._consecutive_failures += 1
                    renderer.console.print(
                        f"  [red]✗ 失败[/] [dim]({elapsed:.0f}ms)[/]: {exc}"
                    )
                    round_results.append(
                        f"## [{role_str}] {desc}\n❌ 失败: {exc}"
                    )

            all_results.extend(round_results)

            # 熔断退出
            if circuit_broken:
                break

            # 8. 审计者验证
            self._check_pipeline_timeout()
            renderer.console.print("  [dim]⏳ 审计者 正在验证结果...[/]")
            check = self._call_role_with_timeout(
                AgentRole.CHECKER, user_input,
                context=(
                    f"## 执行计划\n{plan_json[:1000]}\n\n"
                    f"## 执行结果\n" + "\n---\n".join(
                        r[:800] for r in round_results
                    )
                ),
            )
            renderer.console.print("  [green]✓ 审计者 完成[/]")

            # 9. 反思者审视 → 决定是否继续迭代
            if iteration < max_iterations:
                self._check_pipeline_timeout()
                renderer.console.print("  [dim]⏳ 反思者 正在审视是否需要迭代...[/]")
                reflection = self._call_role_with_timeout(
                    AgentRole.REFLECTOR, user_input,
                    context=(
                        f"## 本轮执行结果\n" + "\n---\n".join(
                            r[:600] for r in round_results
                        )
                        + f"\n\n## 审计者意见\n{check[:800]}"
                    ),
                    extra_instruction=(
                        "请判断当前结果是否已满足用户需求。"
                        "如果已满足，第一行写「执行完成」。"
                        "如果还需改进，第一行写「继续迭代」，并给出具体改进方向。"
                    ),
                )
                if "执行完成" in reflection or self._consecutive_failures > 0:
                    # 有失败时不再迭代，直接收尾
                    renderer.console.print(
                        "  [green]✓ 反思者判定：执行完成[/]"
                    )
                    break
                renderer.console.print(
                    "  [yellow]↻ 反思者判定：需继续迭代[/]"
                )
            else:
                renderer.console.print(
                    "  [dim]已达最大迭代次数，结束循环。[/]"
                )

        # ── Phase 3: 聚合输出 ──
        self._check_pipeline_timeout()
        renderer.console.print()
        renderer.console.print("[bold cyan]📊 四柱管线执行完成[/]")

        if all_results:
            aggregated = "\n\n---\n\n".join(all_results)
            summary = (
                f"## 四柱管线执行总结\n\n"
                f"共执行 {len(all_results)} 个子任务，"
                f"经过 {iteration} 轮迭代。\n\n"
                f"{aggregated}"
            )
            renderer.print_markdown(summary)

    # ── 子任务选择 ──
    def _auto_select(
        self, subtasks: list[dict], reasoning: str,
    ) -> tuple[list[int], str]:
        """思考者自动评估并选择子任务。"""
        renderer = self._renderer
        renderer.console.print(
            "  [bold magenta]🤔 思考者正在评估子任务...[/]"
        )

        tasks_text = "\n".join(
            f"  [{i}] 角色={t.get('role', '?')} | {t.get('task_description', '')[:200]}"
            for i, t in enumerate(subtasks)
        )

        self._check_pipeline_timeout()
        decision = self._call_role_with_timeout(
            AgentRole.THINKER, "",
            context=f"## 决策者分析\n{reasoning[:500]}\n\n## 子任务\n{tasks_text}",
            extra_instruction="请评估以上子任务并输出 JSON 决策。",
        )
        parsed = self._parse_json(decision)
        selected = parsed.get("selected_indices", list(range(len(subtasks))))
        additional = parsed.get("additional_context", "")
        concerns = parsed.get("concerns", "")
        think_reasoning = parsed.get("reasoning", "")

        renderer.console.print(
            f"  [dim]思考者: {think_reasoning[:200]}[/]"
        )
        if additional:
            renderer.console.print(
                f"  [dim yellow]补充: {additional[:200]}[/]"
            )
        if concerns:
            renderer.console.print(
                f"  [dim red]注意: {concerns[:200]}[/]"
            )

        return selected, additional

    def _user_select(
        self, subtasks: list[dict], reasoning: str, iteration: int,
    ) -> tuple[list[int], str]:
        """通过交互式菜单让用户选择子任务。"""
        import sys
        if not sys.stdin.isatty():
            return list(range(len(subtasks))), ""

        from ..cli.selection_ui import present_multi_select_menu, SelectableOption
        renderer = self._renderer

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

        result = present_multi_select_menu(
            options,
            title=f"第{iteration}轮 - 选择要执行的子任务",
        )

        if result.get("action") == "cancelled":
            return [], ""

        selected_keys = set(result.get("selected_keys", []))
        if not selected_keys:
            # 没选任何项，默认选全部
            return list(range(len(subtasks))), ""

        indices = []
        for key in selected_keys:
            for i, task in enumerate(subtasks):
                role = task.get("role", "?")
                if key == f"{role}_{i}" or key == str(i):
                    indices.append(i)
                    break

        custom_text = result.get("custom_text", "")
        return sorted(set(indices)) if indices else list(range(len(subtasks))), custom_text

    @staticmethod
    def _str_to_role(s: str) -> AgentRole:
        try:
            return AgentRole(s)
        except ValueError:
            return AgentRole.RUNNER
