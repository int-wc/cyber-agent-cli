"""多 Agent 编排器：将用户任务分解并分发给角色 Agent 并行执行。"""

from __future__ import annotations

import concurrent.futures
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from .._lazy_imports import load_chat_openai
from ..config import settings
from ..execution_control import ExecutionController, ExecutionInterruptedError
from ..logging import log_info, log_error
from .events import AgentEventType
from .roles import AgentRole, get_role_label, get_role_prompt


@dataclass
class AgentTask:
    """描述分配给单个角色 Agent 的子任务。"""

    role: AgentRole
    task_description: str
    context: str = ""
    expected_output: str = "text"


@dataclass
class AgentResult:
    """角色 Agent 的执行结果。"""

    role: AgentRole
    success: bool
    output: str
    error: str = ""
    elapsed_ms: float = 0.0


@dataclass
class OrchestrationPlan:
    """任务分解与执行计划。"""

    original_task: str
    subtasks: list[AgentTask] = field(default_factory=list)
    reasoning: str = ""
    iteration: int = 1


class MultiAgentOrchestrator:
    """多 Agent 编排器，管理角色 Agent 的创建、调度与结果聚合。

    工作流程：
    1. 决策者 (decision_maker) 分析任务并分解为子任务
    2. 子任务按角色分配后由执行线程池并发运行
    3. 审计者 (checker) 汇总验证各角色结果
    4. 反思者 (reflector) 评估是否需要迭代改进
    5. 如有需要，重新规划并迭代执行
    """

    def __init__(
        self,
        *,
        tools: list[BaseTool] | None = None,
        execution_controller: ExecutionController | None = None,
        event_handler: Callable[[str, Any], None] | None = None,
        service_name: str = "deepseek",
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_workers: int | None = None,
    ) -> None:
        self.tools = tools or []
        self.execution_controller = execution_controller
        self.event_handler = event_handler
        self.service_name = settings.normalize_service_name(service_name)
        self.model_name = settings.get_model_name(
            model_name,
            service_name=self.service_name,
        )
        self.api_key = settings.get_api_key(self.service_name, api_key=api_key)
        self.base_url = settings.resolve_base_url(self.service_name, base_url=base_url)
        self.max_workers = max_workers or settings.multi_agent_max_workers
        self._llm: Any | None = None

    def _get_llm(self) -> Any:
        """懒加载模型实例。"""
        if self._llm is None:
            chat_openai_cls = load_chat_openai()
            self._llm = chat_openai_cls(
                **settings.get_chat_openai_kwargs(
                    self.service_name,
                    model_name=self.model_name,
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
            )
        return self._llm

    def _emit(self, event_type: str, payload: Any) -> None:
        """发送编排事件。"""
        if self.event_handler is not None:
            self.event_handler(event_type, payload)

    def run(
        self,
        user_input: str,
        *,
        max_iterations: int = 3,
    ) -> str:
        """执行多 Agent 协作任务。

        Args:
            user_input: 用户输入的任务描述
            max_iterations: 最大迭代次数（审计-反思-重执行循环）
        """
        if not user_input.strip():
            return ""

        self._emit("orchestration_start", {"input": user_input})
        log_info("orchestrator", f"开始多 Agent 协作：{user_input[:100]}...")

        # 阶段 1: 任务规划（决策者角色）
        plan = self._plan_task(user_input)

        # 阶段 2: 并发执行子任务
        all_results: list[AgentResult] = []
        for iteration in range(1, max_iterations + 1):
            plan.iteration = iteration
            self._emit("orchestration_iteration", {"iteration": iteration})

            results = self._execute_plan(plan)
            all_results.extend(results)

            # 阶段 3: 审计验证
            checker_result = self._check_results(user_input, plan, results)
            all_results.append(checker_result)

            # 阶段 4: 反思与决策
            should_continue = self._reflect_and_decide(
                user_input, plan, results, checker_result
            )
            if not should_continue:
                break

            # 阶段 5: 重新规划（如果还需要迭代）
            if iteration < max_iterations:
                plan = self._replan(user_input, plan, results, checker_result)

        # 阶段 6: 聚合最终输出
        final_result = self._synthesize(user_input, all_results)
        self._emit("orchestration_end", {"output": final_result})
        log_info("orchestrator", f"多 Agent 协作完成，共 {len(all_results)} 个结果。")
        return final_result

    def _plan_task(self, user_input: str) -> OrchestrationPlan:
        """决策者分解任务为子任务。"""
        self._emit("orchestration_planning", {"input": user_input})

        system_prompt = f"""{get_role_prompt(AgentRole.DECISION_MAKER)}

请将以下用户任务分解为子任务，分配给最合适的角色。
可用角色：{', '.join(get_role_label(r) for r in AgentRole)}

输出必须是 JSON 对象：
{{
  "reasoning": "任务分解思路",
  "subtasks": [
    {{
      "role": "角色名（checker/reader/analyst/runner/builder/decision_maker/reflector/diffuser/jumper）",
      "task_description": "具体任务描述，确保清晰可执行",
      "context": "需要补充的上下文信息"
    }}
  ]
}}

原则：
1. 每个子任务只分配给一个最合适的角色
2. 子任务之间尽量独立，便于并行执行
3. 优先使用 runner（执行工具）、reader（阅读内容）、analyst（分析数据）
4. 任务数量控制在 2-6 个
5. 审计和反思无需在此分配，系统自动执行"""

        try:
            response = self._get_llm().invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_input),
            ])
            content = self._extract_text(response)
            plan_data = self._parse_json(content)
        except Exception as exc:
            log_error("orchestrator", f"任务规划失败：{exc}，使用默认计划。")
            return self._default_plan(user_input)

        subtasks = []
        for task_data in plan_data.get("subtasks", []):
            try:
                role = AgentRole(task_data.get("role", "runner"))
            except ValueError:
                role = AgentRole.RUNNER
            subtasks.append(AgentTask(
                role=role,
                task_description=str(task_data.get("task_description", "")),
                context=str(task_data.get("context", "")),
            ))

        if not subtasks:
            return self._default_plan(user_input)

        return OrchestrationPlan(
            original_task=user_input,
            subtasks=subtasks,
            reasoning=str(plan_data.get("reasoning", "")),
        )

    def _default_plan(self, user_input: str) -> OrchestrationPlan:
        """当模型规划失败时，使用默认任务分解。"""
        return OrchestrationPlan(
            original_task=user_input,
            subtasks=[
                AgentTask(role=AgentRole.RUNNER, task_description=user_input),
                AgentTask(
                    role=AgentRole.ANALYST,
                    task_description=f"分析任务 `{user_input[:100]}` 涉及的关键要素和潜在难点",
                ),
            ],
            reasoning="默认任务分解：执行者直接处理任务，分析者辅助分析。",
        )

    def _execute_plan(self, plan: OrchestrationPlan) -> list[AgentResult]:
        """并发执行所有子任务。"""
        if not plan.subtasks:
            return []

        self._emit("orchestration_executing", {"subtask_count": len(plan.subtasks)})
        results: list[AgentResult] = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(plan.subtasks), self.max_workers),
        ) as executor:
            future_to_task = {
                executor.submit(self._run_role_agent, task): task
                for task in plan.subtasks
            }
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                except ExecutionInterruptedError:
                    for f in future_to_task:
                        f.cancel()
                    raise
                except Exception as exc:
                    result = AgentResult(
                        role=task.role,
                        success=False,
                        output="",
                        error=str(exc),
                    )
                results.append(result)
                self._emit("subtask_complete", {
                    "role": result.role.value,
                    "success": result.success,
                    "elapsed_ms": result.elapsed_ms,
                })

        return results

    def _run_role_agent(self, task: AgentTask) -> AgentResult:
        """在独立线程中运行单个角色 Agent。"""
        import time as time_mod

        if self.execution_controller is not None:
            self.execution_controller.ensure_not_cancelled()

        role_label = get_role_label(task.role)
        log_info("orchestrator", f"启动 {role_label}({task.role.value})...")
        start = time_mod.monotonic()

        try:
            system_prompt = self._build_role_system_prompt(task)
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=self._build_role_user_message(task)),
            ]

            response = self._get_llm().invoke(messages)
            output = self._extract_text(response)
            elapsed = (time_mod.monotonic() - start) * 1000

            log_info("orchestrator", f"{role_label} 完成，耗时 {elapsed:.0f}ms")
            return AgentResult(
                role=task.role,
                success=True,
                output=output,
                elapsed_ms=elapsed,
            )
        except ExecutionInterruptedError:
            raise
        except Exception as exc:
            elapsed = (time_mod.monotonic() - start) * 1000
            log_error("orchestrator", f"{role_label} 执行失败：{exc}")
            return AgentResult(
                role=task.role,
                success=False,
                output="",
                error=str(exc),
                elapsed_ms=elapsed,
            )

    def _build_role_system_prompt(self, task: AgentTask) -> str:
        """构建角色专属系统提示词，注入任务上下文和可用工具信息。"""
        role_prompt = get_role_prompt(task.role)
        role_label = get_role_label(task.role)

        tool_descriptions = "\n".join(
            f"- {tool.name}: {tool.description[:120]}" for tool in self.tools[:20]
        ) if self.tools else "无额外工具"

        return f"""{role_prompt}

## 当前上下文
你正在参与一个多 Agent 协作任务。你的角色是 {role_label}。
当前子任务由决策者分配，需独立完成并返回结构化输出。

## 可用工具
{tool_descriptions}

## 输出要求
- 用中文回复
- 先给出核心结论或执行摘要
- 再展开详细内容
- 如有工具调用需求，明确说明需要什么
- 标注任何不确定的部分"""

    def _build_role_user_message(self, task: AgentTask) -> str:
        """构建角色 Agent 的用户消息。"""
        msg = f"## 子任务\n{task.task_description}"
        if task.context:
            msg += f"\n\n## 附加上下文\n{task.context}"
        return msg

    def _check_results(
        self,
        user_input: str,
        plan: OrchestrationPlan,
        results: list[AgentResult],
    ) -> AgentResult:
        """审计者验证所有结果。"""
        self._emit("orchestration_checking", {"result_count": len(results)})

        results_summary = "\n\n".join(
            f"[{get_role_label(r.role)}] {'成功' if r.success else '失败'}: "
            f"{r.output[:500] if r.output else r.error}"
            for r in results
        )

        checker_prompt = f"""{get_role_prompt(AgentRole.CHECKER)}

## 原始任务
{user_input}

## 子任务计划
{plan.reasoning}

## 各角色执行结果
{results_summary}

请审计以上结果：
1. 各角色的输出是否回答了分配的子任务
2. 是否存在明显错误或遗漏
3. 结果之间是否有矛盾
4. 整体上是否满足原始任务需求
5. 给出通过/需改进的总体评估"""

        try:
            response = self._get_llm().invoke([
                SystemMessage(content=checker_prompt),
                HumanMessage(content="请对上述多 Agent 协作结果进行审计。"),
            ])
            output = self._extract_text(response)
            return AgentResult(role=AgentRole.CHECKER, success=True, output=output)
        except Exception as exc:
            return AgentResult(
                role=AgentRole.CHECKER,
                success=False,
                output="",
                error=f"审计失败：{exc}",
            )

    def _reflect_and_decide(
        self,
        user_input: str,
        plan: OrchestrationPlan,
        results: list[AgentResult],
        checker_result: AgentResult,
    ) -> bool:
        """反思者评估是否需要继续迭代。"""
        if not checker_result.success:
            return False

        failed_count = sum(1 for r in results if not r.success)
        if failed_count == 0:
            return False  # 全部成功，无需迭代

        self._emit("orchestration_reflecting", {"failed_count": failed_count})

        reflector_prompt = f"""{get_role_prompt(AgentRole.REFLECTOR)}

## 原始任务
{user_input}

## 审计结果
{checker_result.output[:800]}

共 {len(results)} 个角色结果，{failed_count} 个失败。

请判断：
1. 当前结果是否已足够回答用户问题
2. 是否需要调整策略重新执行
3. 输出格式：第一行写 "继续迭代" 或 "结束协作"，然后给出理由"""

        try:
            response = self._get_llm().invoke([
                SystemMessage(content=reflector_prompt),
                HumanMessage(content="请决定是否继续迭代。"),
            ])
            output = self._extract_text(response)
            return "继续迭代" in output and failed_count > 0
        except Exception:
            return False

    def _replan(
        self,
        user_input: str,
        plan: OrchestrationPlan,
        results: list[AgentResult],
        checker_result: AgentResult,
    ) -> OrchestrationPlan:
        """基于审计和反思重新规划任务。"""
        failed_roles = [
            get_role_label(r.role) for r in results if not r.success
        ]

        replan_prompt = f"""{get_role_prompt(AgentRole.DECISION_MAKER)}

## 原始任务
{user_input}

## 上一轮计划
{plan.reasoning}

## 失败的角色
{', '.join(failed_roles) if failed_roles else '无'}

## 审计意见
{checker_result.output[:500]}

请重新规划，调整策略以弥补不足。输出 JSON 格式同上。"""

        try:
            response = self._get_llm().invoke([
                SystemMessage(content=replan_prompt),
                HumanMessage(content="请重新规划子任务。"),
            ])
            content = self._extract_text(response)
            plan_data = self._parse_json(content)
        except Exception:
            # 简化：只重试失败的角色
            retry_tasks = [
                AgentTask(
                    role=r.role,
                    task_description=f"重试失败的任务。原错误：{r.error[:200]}",
                )
                for r in results if not r.success
            ]
            return OrchestrationPlan(
                original_task=user_input,
                subtasks=retry_tasks or plan.subtasks,
                reasoning="重试失败的子任务。",
            )

        subtasks = []
        for task_data in plan_data.get("subtasks", []):
            try:
                role = AgentRole(task_data.get("role", "runner"))
            except ValueError:
                role = AgentRole.RUNNER
            subtasks.append(AgentTask(
                role=role,
                task_description=str(task_data.get("task_description", "")),
                context=str(task_data.get("context", "")),
            ))
        return OrchestrationPlan(
            original_task=user_input,
            subtasks=subtasks or plan.subtasks,
            reasoning=str(plan_data.get("reasoning", "")),
        )

    def _synthesize(
        self,
        user_input: str,
        all_results: list[AgentResult],
    ) -> str:
        """决策者综合所有结果生成最终输出。"""
        results_text = "\n\n---\n\n".join(
            f"## {get_role_label(r.role)}\n{r.output[:800]}"
            for r in all_results if r.success and r.output
        )

        if not results_text.strip():
            return "多 Agent 协作未产生有效输出。"

        synthesize_prompt = f"""{get_role_prompt(AgentRole.DECISION_MAKER)}

## 用户原始请求
{user_input}

## 各角色执行结果
{results_text}

请综合以上所有角色的输出，生成一个完整、清晰、结构化的最终回复。
要求：
1. 先给出核心结论（不超过3句话）
2. 再分章节展开细节
3. 优先采纳成功角色的输出
4. 标注信息的确定性级别"""

        try:
            response = self._get_llm().invoke([
                SystemMessage(content=synthesize_prompt),
                HumanMessage(content="请综合所有角色输出，生成最终回复。"),
            ])
            return self._extract_text(response)
        except Exception as exc:
            # 降级：直接拼接各角色输出
            parts = [f"## {get_role_label(r.role)}\n{r.output}"
                     for r in all_results if r.success and r.output]
            return f"多 Agent 协作结果（综合失败，以下为各角色输出）：\n\n" + "\n\n".join(parts)

    @staticmethod
    def _extract_text(response: Any) -> str:
        """从 LangChain 响应中提取文本。"""
        content = getattr(response, "content", "")
        if isinstance(content, list):
            return "".join(
                item if isinstance(item, str) else str(item.get("text", ""))
                for item in content
            )
        return str(content)

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """从模型输出中解析 JSON。"""
        text = text.strip()
        # 尝试去除 markdown 代码围栏
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
            # 尝试提取 JSON 块
            import re
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return {}
