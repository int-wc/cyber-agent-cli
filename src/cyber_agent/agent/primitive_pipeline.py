"""原语工作流管线：用 workflow 的业务原语解析 + 原语链利用替换四柱管线。

架构融合说明
============
继承 FourPillarPipeline（复用全部基础设施与执行机制）：
- 模型网关 / 轨迹追踪 / 会话持久化 / token 统计 / 熔断 / 超时叠加 /
  子任务 sub-runner / 审批 / 事件转发 —— 全部继承，零重复实现。

只重写 _run_phases 的语义：
- 四柱思考 → **原语思考**（同四柱骨架，语义换成原语工作流）：
    1. 分析者   → **原语解析者**：端点 → business_attr / attr_target / attr_reason（不看 API 名）
    2. 扩散者   → **攻击面扩散者**：原语 → 攻击基元枚举（攻击面库 + 自定义测试向量）
    3. 迁跃者   → **链跃迁者**：原语组合 → 候选链匹配（原语链库 + 跨域/remote→local 升级）
    4. 反思者   → **链裁决者**：候选链可串联性/可利用性裁决 → 链式执行计划
- 执行闭环   → **链执行闭环**：决策者按链分解子任务 → 执行者（sub-runner）→
  审计者验证链组合 → 反思者裁决是否达到危害阈值。

数据层复用 primitives/ 包：
- parser.py          原语判定行解析
- surface_matcher.py 攻击面前匹配（signals + 原语命中 → 攻击基元注入）
- chain_library.py   原语链联动推理（端点原语 → 链模板匹配）
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage, SystemMessage

from .pipeline import FourPillarPipeline
from .primitives import (
    BUSINESS_ATTR_GUIDE,
    append_chain_candidate,
    build_hint_report,
    build_link_report,
    load_chains,
    load_chain_ids,
    load_surfaces,
    parse_endpoint_dicts,
    record_chain_instance,
    serialize_endpoints,
    upsert_chain,
)
from .roles import AgentRole, get_role_label, get_role_prompt

if TYPE_CHECKING:
    from .runner import AgentRunner


# ═══ 原语角色系统提示词（替换四柱的通用角色提示）═══

PRIMITIVE_ROLE_PROMPTS: dict[AgentRole, str] = {
    # ── 原语解析者（原「分析者」——分析为底 → 原语为底）──
    AgentRole.ANALYST: """你是原语解析者。你不是做通用分析，而是对目标做**业务原语判定**。

你的唯一产出：把目标清单中的每个可测端点判定出核心业务原语 business_attr（它"到底对什么东西做什么操作"），
再给出 attr_target（作用对象）、attr_reason（推导依据）、params（关键参数）、risk（潜在风险）。

铁律：
1. **不看 API 名字或参数名**，从数据流/后端语义推断真实原语。名字含 load/parse/import/sync/render/convert/
   transform/download/upload 的端点最可能是双原语表面（read + exec），必须拆开判定。
2. 每个端点至少自问三连："如果我让它读本地文件 / 执行我给的代码 / 写入任意路径，业务上它会不会照做？"
3. business_attr ∈ {read_file, write_file, exec_code, modify_state, query_data, transfer, auth}
4. attr_target ∈ {local_fs, remote_url, db, template, user_input, worker}

输出必须是 JSON：
{"target_summary": "一句话目标描述", "primitives": [
  {"endpoint": "/api/xxx", "method": "POST", "business_attr": "transfer",
   "attr_target": "remote_url", "attr_reason": "白名单URL拉取",
   "params": {"url": ""}, "risk": "SSRF"}]}""",

    # ── 攻击面扩散者（原「扩散者」——扩展为路 → 攻击基元为路）──
    AgentRole.DIFFUSER: """你是攻击面扩散者。对已判定原语的每个端点，按原语扩散出**具体攻击基元与测试向量**。

你会收到：已解析的原语端点清单 + 程序化攻击面匹配结果（signals 命中 → base_primitives）。
你的职责是在程序化结果之上做**模型推理补充**：
- 对每个端点给出可直接执行的 attack_primitives（攻击基元）与 test_vectors（curl/参数测试向量）
- 对已有 base_primitives 做排序/裁剪，标注 priority
- 补程序化匹配漏掉的攻击面（信号不命中但语义符合的）
- 特别关注**原语切换**：如 transfer 原语被 URL 白名单拦截时，切到 local_fs 读本地路径（读本地不是 URL fetch）

输出必须是 JSON：
{"attack_plan": [
  {"endpoint": "/api/xxx", "business_attr": "transfer", "priority": "high",
   "attack_primitives": ["SSRF(回显)", "remote→local白名单绕过"],
   "test_vectors": ["curl -s -X POST ... -d 'url=http://169.254.169.254/...'"]}]}""",

    # ── 链跃迁者（原「迁跃者」——迁跃为辅 → 原语链跃迁为辅）──
    AgentRole.JUMPER: """你是链跃迁者。判断已解析的业务原语能否**串联成有效危害链**。

你会收到：已解析原语端点 + 程序化链匹配候选（primitive-chains.json 中组成原语在目标上齐备的链）。
你的职责：
- 对每条候选链，用模型推理验证**业务信任串联**是否真的成立（不只是原语齐备，还要看业务上下文是否允许串联）
- 标记每条链的 escalation（升级路径：低危原语 → 高危害）与 priority
- 提出程序化匹配没有的**新颖链**（跨域原语组合 / remote→local 切换 / 双原语表面）
- 检查组成原语端点是否"同域/同信任域"——跨域串联要标注

输出必须是 JSON：
{"chains": [
  {"chain_id": "ch_ssrf_to_auth", "validated": true, "priority": "high",
   "escalation": "SSRF打认证服务→token篡改→账户接管",
   "key_endpoints": {"transfer": ["/api/translateUrl"], "auth": ["/api/login"]}}],
 "novel_chains": [
  {"primitives": ["transfer", "auth"], "logic": "...", "gain": "..."}]}""",

    # ── 链裁决者（原「反思者」——反思为主 → 链裁决为主）──
    AgentRole.REFLECTOR: """你是链裁决者。对候选链做最终**可利用性裁决**，输出链式执行计划。

你会收到：原语解析结果、攻击面扩散结果、链跃迁候选（含优先级）。
你的职责：
- 逐链裁决 verdict ∈ {execute, needs_account, misreport}：
  - execute：可只读验证/可实际利用
  - needs_account：需要测试账号/登录态才能验证（红线：自备账号，越权证明≤5组）
  - misreport：误报 / 原语判定错误 / 无法串联
- 对 execute 链给出具体 exploitation_plan（怎么验证这条链）
- 最终决定：立即执行 top 链 / 需要补充分析迭代

输出必须是 JSON：
{"verdict": "execute|iterate",
 "top_chains": [
  {"chain_id": "ch_ssrf_to_auth", "verdict": "execute", "priority": "high",
   "exploitation_plan": "先只读验证 transfer 端点 SSRF 可达性，再验证 auth 端点是否能被内网调用..."}],
 "iterate_notes": "若不执行，说明缺什么"}""",

    # ── 决策者（执行闭环）——原语化：必须产出可执行 curl 子任务 ──
    AgentRole.DECISION_MAKER: """你是链执行决策者。把链裁决者选出的 top 链**分解为可直接执行的工具子任务**。

你会收到：链执行计划（含候选链的 exploitation_plan）与攻击面扩散的 test_vectors。
你的职责：
- 把每条 execute 链分解为**最多 4 个**子任务（高优先级链优先），每个子任务**一句话**，只含一条 curl 命令 + 观察点
- 严格避免长文本：若一个子任务描述超 200 字符就拆成更简单的命令或砍掉次要向量
- 红线：只读验证优先，越权尝试总计≤5组，写原语只记录不验证，禁止破坏性利用，禁止深入内网
- 每个子任务：{"role": "runner", "task_description": "一句 curl 命令+观察什么"}

输出必须是 JSON：
{"reasoning": "分解思路", "subtasks": [
  {"role": "runner", "task_description": "curl -s -H 'User-Agent: Mozilla/5.0' https://api-app.lixiang.com/api/user/info/ 观察是否从code:100012权限错误变为真实数据"}]}""",
}


class PrimitiveWorkflowPipeline(FourPillarPipeline):
    """原语工作流管线：四柱骨架 + 原语解析/链利用语义。"""

    def __init__(
        self,
        *,
        runner: AgentRunner,
        runtime_context: dict[str, object],
        renderer: Any,
        event_handler: Any = None,
    ) -> None:
        super().__init__(
            runner=runner,
            runtime_context=runtime_context,
            renderer=renderer,
            event_handler=event_handler,
        )
        # 原语工作流状态
        self._primitive_endpoints: list[Any] = []
        self._chain_candidates: list[Any] = []
        self._chain_verdicts: list[dict[str, Any]] = []
        self._attack_plan: list[dict[str, Any]] = []

    # ── 角色提示词覆写：原语语义替换四柱语义 ──
    def _call_role(
        self,
        role: AgentRole,
        user_input: str,
        *,
        context: str = "",
        extra_instruction: str = "",
        retries: int = 2,
    ) -> str:
        """调用单个角色 LLM。对瞬态网关错误（500/流式失败）自动重试。"""
        import time as _time

        label = get_role_label(role)
        system_prompt = PRIMITIVE_ROLE_PROMPTS.get(role, get_role_prompt(role))
        system_context = self._build_system_context()
        full_system = f"{system_prompt}\n\n## 系统环境\n{system_context}"

        user_content = f"## 用户任务\n{user_input}"
        if context:
            user_content += f"\n\n## 前序输出（请基于此继续）\n{context}"
        if extra_instruction:
            user_content += f"\n\n{extra_instruction}"

        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                response = self._get_llm().invoke([
                    SystemMessage(content=full_system),
                    HumanMessage(content=user_content),
                ])
                self._track_llm_usage(response)
                return self._extract_text(response)
            except Exception as exc:
                last_exc = exc
                err = str(exc)
                # 仅对瞬态错误重试：网关 5xx / 流式无 chunk / 连接类
                transient = (
                    "500" in err or "5 " in err and "Internal server" in err
                    or "No generation chunks" in err
                    or "Connection" in err or "Read timed out" in err
                    or "429" in err or "rate limit" in err.lower()
                )
                if attempt >= retries or not transient:
                    break
                from ..logging import log_warning
                log_warning("pipeline", f"{label} 调用瞬态失败({attempt+1}/{retries})：{err}，重试...")
                _time.sleep(3 + attempt * 3)

        from ..logging import log_error
        log_error("pipeline", f"{label} 调用失败：{last_exc}")
        return f"[{label} 调用失败: {last_exc}]"

    # ═══ 工具：原语阶段输出解析 ═══
    @staticmethod
    def _is_role_error(result: str) -> bool:
        return "调用失败" in result or "调用超时" in result

    @staticmethod
    def _parse_primitive_endpoints(analysis: str) -> list[Any]:
        """解析原语解析者的 JSON 输出，转为 PrimitiveEndpoint 列表。"""
        data = PrimitiveWorkflowPipeline._parse_json(analysis)
        raw = data.get("primitives", [])
        if not raw and "[" in analysis:
            # 兜底：尝试直接解析内嵌的 primitives 数组
            import re as _re
            m = _re.search(r'"primitives"\s*:\s*(\[.*?\])', analysis, _re.S)
            if m:
                try:
                    raw = json.loads(m.group(1))
                except json.JSONDecodeError:
                    raw = []
        return parse_endpoint_dicts(raw)

    @staticmethod
    def _extract_json_list(analysis: str, key: str) -> list[dict[str, Any]]:
        """从角色 JSON 输出中提取某个 key 下的列表。"""
        data = PrimitiveWorkflowPipeline._parse_json(analysis)
        items = data.get(key, [])
        if isinstance(items, list):
            return [it for it in items if isinstance(it, dict)]
        return []

    @staticmethod
    def _extract_subtasks(plan_json: str) -> list[dict[str, Any]]:
        """从决策者 JSON 输出中提取子任务（容错模型长输出截断）。

        模型一次输出较多 curl 子任务时常触发输出截断，导致 JSON 数组不完整、
        _parse_json 整体失败。本方法优先完整解析，失败则用正则逐个提取
        完整成对的 {"role":..,"task_description":..} 对象，最大程度保真。
        """
        import re as _re

        data = PrimitiveWorkflowPipeline._parse_json(plan_json)
        raw = data.get("subtasks", [])
        if raw and isinstance(raw, list):
            return [t for t in raw if isinstance(t, dict)]

        out: list[dict[str, Any]] = []
        # 宽松匹配每个子任务对象（容忍截断、无闭合括号、字段顺序任意）
        for m in _re.finditer(
            r'\{\s*"role"\s*:\s*"([^"]*)"\s*,\s*"task_description"\s*:\s*"'
            r'((?:[^"\\]|\\.)*)"',
            plan_json,
        ):
            desc = m.group(2).encode().decode("unicode_escape", errors="ignore")
            out.append({"role": m.group(1), "task_description": desc})
        if out:
            return out
        # 最后兜底：匹配 {"role":"x","task_description":"y"} 反序或通用结构
        for m in _re.finditer(r'"task_description"\s*:\s*"((?:[^"\\]|\\.)*)"', plan_json):
            out.append({"role": "runner", "task_description": m.group(1)})
        return out

    # ═══ 主流程：原语思考 → 链执行闭环 → 聚合 ═══
    def _run_phases(self, user_input: str, auto_decision: bool) -> None:
        renderer = self._renderer

        # ── Phase 1: 原语思考（四柱骨架 + 原语语义）──
        renderer.console.print()
        renderer.console.print("[dim bold]🧬 原语解析阶段[/]")
        renderer.console.print(
            "[dim]原语解析 → 攻击面扩散 → 链跃迁 → 链裁决[/]"
        )

        # 1. 原语解析者（分析为底 → 原语为底）
        renderer.console.print("  [dim]⏳ 原语解析者 正在判定业务原语...[/]")
        self._record_role_progress(
            "analyst", "原语解析者", "start",
            action="正在判定业务原语", phase="primitive",
        )
        t0 = self._monotonic()
        analysis = self._call_role_with_timeout(AgentRole.ANALYST, user_input)
        elapsed = (self._monotonic() - t0) * 1000
        self._record_role_progress(
            "analyst", "原语解析者", "done" if not self._is_role_error(analysis) else "error",
            detail=analysis[:500], elapsed_ms=elapsed, phase="primitive",
        )
        self._record_trace("primitive_parse", detail=analysis[:2000])
        self._primitive_endpoints = self._parse_primitive_endpoints(analysis)
        renderer.console.print(
            f"  [dim {'green' if self._primitive_endpoints else 'red'}]"
            f"✓ 原语解析完成：{len(self._primitive_endpoints)} 个端点原语判定[/]"
        )

        # 程序化注入：攻击面匹配 + 原语链联动（数据层）
        surfaces = load_surfaces()
        hint_report = build_hint_report(self._primitive_endpoints, surfaces)
        chain_report = build_link_report(self._primitive_endpoints)
        self._record_trace(
            "primitive_surface_match",
            detail=f"攻击面命中 {hint_report['matched_count']}/{hint_report['total_endpoints']}",
        )
        self._record_trace(
            "primitive_chain_link",
            detail=f"原语链候选 {len(chain_report['candidates'])}/{len(load_chains())}",
        )
        self._chain_candidates = chain_report["candidates"]

        # 2. 攻击面扩散者（扩展为路 → 攻击基元为路）
        renderer.console.print("  [dim]⏳ 攻击面扩散者 正在枚举攻击基元...[/]")
        self._record_role_progress(
            "diffuser", "攻击面扩散者", "start",
            action="正在枚举攻击基元", phase="primitive",
        )
        diffuser_context = (
            f"## 已解析原语端点\n{serialize_endpoints(self._primitive_endpoints) or '（无）'}\n\n"
            f"## 程序化攻击面匹配\n{json.dumps(hint_report, ensure_ascii=False, indent=2)[:4000]}"
        )
        t0 = self._monotonic()
        diffusion = self._call_role_with_timeout(
            AgentRole.DIFFUSER, user_input, context=diffuser_context,
        )
        elapsed = (self._monotonic() - t0) * 1000
        self._record_role_progress(
            "diffuser", "攻击面扩散者", "done" if not self._is_role_error(diffusion) else "error",
            detail=diffusion[:500], elapsed_ms=elapsed, phase="primitive",
        )
        self._record_trace("primitive_diffuse", detail=diffusion[:2000])
        self._attack_plan = self._extract_json_list(diffusion, "attack_plan")

        # 3. 链跃迁者（迁跃为辅 → 原语链跃迁为辅）
        renderer.console.print("  [dim]⏳ 链跃迁者 正在匹配原语链...[/]")
        self._record_role_progress(
            "jumper", "链跃迁者", "start",
            action="正在匹配原语链", phase="primitive",
        )
        jumper_context = (
            f"## 已解析原语端点\n{serialize_endpoints(self._primitive_endpoints) or '（无）'}\n\n"
            f"## 程序化链匹配候选\n{json.dumps(chain_report, ensure_ascii=False, indent=2)[:4000]}"
        )
        t0 = self._monotonic()
        jump = self._call_role_with_timeout(
            AgentRole.JUMPER, user_input, context=jumper_context,
        )
        elapsed = (self._monotonic() - t0) * 1000
        self._record_role_progress(
            "jumper", "链跃迁者", "done" if not self._is_role_error(jump) else "error",
            detail=jump[:500], elapsed_ms=elapsed, phase="primitive",
        )
        self._record_trace("primitive_chain_jump", detail=jump[:2000])

        # 4. 链裁决者（反思为主 → 链裁决为主）
        renderer.console.print("  [dim]⏳ 链裁决者 正在裁决可利用性...[/]")
        self._record_role_progress(
            "reflector", "链裁决者", "start",
            action="正在裁决可利用性", phase="primitive",
        )
        reflector_context = "\n\n".join(
            part for part in (
                f"## 原语解析\n{analysis}" if not self._is_role_error(analysis) else "",
                f"## 攻击面扩散\n{diffusion}" if not self._is_role_error(diffusion) else "",
                f"## 链跃迁\n{jump}" if not self._is_role_error(jump) else "",
            ) if part
        )
        t0 = self._monotonic()
        reflection = self._call_role_with_timeout(
            AgentRole.REFLECTOR, user_input,
            context=reflector_context or "所有前置阶段异常，请基于原始需求直接裁决。",
        )
        elapsed = (self._monotonic() - t0) * 1000
        self._record_role_progress(
            "reflector", "链裁决者", "done" if not self._is_role_error(reflection) else "error",
            detail=reflection[:500], elapsed_ms=elapsed, phase="primitive",
        )
        self._record_trace("primitive_verdict", detail=reflection[:2000])
        self._chain_verdicts = self._extract_json_list(reflection, "top_chains")

        if self._is_role_error(reflection):
            renderer.console.print(
                "  [bold yellow]⚠️ 链裁决阶段异常，管线将继续尝试但输出质量可能下降。[/]"
            )

        # ── Phase 2: 链执行闭环 ──
        max_iterations = self._resolve_effective_max_iterations()
        all_results: list[list[str]] = []
        iteration = 0

        chain_plan_hint = (
            self._format_chain_plan_hint()
            or json.dumps(self._chain_verdicts[:3], ensure_ascii=False, indent=2)[:2000]
        )

        for iteration in range(1, max_iterations + 1):
            renderer.console.print()
            renderer.console.print(
                f"[dim bold]⚡ 链执行闭环 第 {iteration}/{max_iterations} 轮[/]"
            )
            self._record_trace("iteration_start", detail=f"第 {iteration} 轮")

            # 5. 决策者 → 按链分解子任务
            renderer.console.print("  [dim]⏳ 决策者 正在按链分解子任务...[/]")
            iter_context = reflection if not self._is_role_error(reflection) else ""
            if all_results:
                prev_round = all_results[-1]
                iter_context += f"\n\n## 上一轮执行结果\n" + "\n".join(
                    f"- {r[:300]}" for r in prev_round[-5:]
                )
            plan_json = self._call_role_with_timeout(
                AgentRole.DECISION_MAKER, user_input,
                context=f"## 链执行计划\n{chain_plan_hint}\n\n{iter_context}",
                extra_instruction=(
                    "请把链验证/利用分解为可直接执行的工具子任务（最多4个，每个一句话）。"
                    "输出必须是 JSON："
                    '{"reasoning": "...", "subtasks": ['
                    '{"role": "runner", "task_description": "..."}]}'
                ),
                timeout=240,
            )
            self._record_role_progress(
                "decision_maker", "决策者", "done",
                detail=plan_json[:500], phase="execution",
            )
            plan = self._parse_json(plan_json)
            reasoning = plan.get("reasoning", "")
            # 容错提取子任务：模型长输出截断时也能恢复完整子任务（_extract_subtasks）
            raw_subtasks = self._extract_subtasks(plan_json)
            # 注意：继承的 _auto_select/_user_select 读 task_description 字段，
            # 不能改名成 desc，否则思考者看到空壳子任务
            subtasks = [
                {
                    "role": str(t.get("role", "runner")),
                    "task_description": str(t.get("task_description", str(t))),
                }
                for t in raw_subtasks if isinstance(t, dict)
            ]

            if not subtasks:
                renderer.console.print("  [dim yellow]决策者未产出子任务，结束执行。[/]")
                break

            # 6. 思考者 / 用户 → 选择子任务
            if auto_decision:
                selected, _ = self._auto_select(subtasks, reasoning)
            else:
                selected, _ = self._user_select(subtasks, reasoning, iteration)
            if not selected:
                renderer.console.print("  [dim yellow]未选择子任务，结束执行。[/]")
                break

            # 7. 执行者 → sub-runner 执行
            round_results: list[str] = []
            for idx in selected:
                task = subtasks[idx]
                task_desc = task.get("task_description", str(task))
                role_label = get_role_label(self._str_to_role(task["role"]))
                renderer.console.print(
                    f"  [dim]▶ 执行 [{role_label}] {task_desc[:120]}[/]"
                )
                prompt = (
                    f"你是{role_label}。请执行链验证/利用子任务并给出结果摘要。\n\n"
                    f"子任务: {task_desc}\n"
                    f"\n整体背景: {chain_plan_hint[:500]}\n"
                    f"\n红线: 只读验证优先，越权读取证明≤5组，写原语只记录不验证，禁止破坏性利用。"
                )
                try:
                    result = self._run_subtask_with_escalating_timeout(
                        prompt, role_label, task_desc,
                    )
                except TimeoutError as exc:
                    result = f"❌ 失败: {exc}"
                except Exception as exc:
                    # 子任务执行异常（如模型流式瞬断）不应中断整条管线
                    from ..logging import log_error
                    log_error("pipeline", f"子任务执行异常：{exc}")
                    result = f"❌ 失败: 子任务执行异常 {exc}"
                round_results.append(f"## {task_desc[:200]}\n{result}")
            all_results.append(round_results)

            # 8. 审计者 → 验证链组合
            renderer.console.print("  [dim]⏳ 审计者 正在验证链组合...[/]")
            results_text = "\n".join(round_results)
            check = self._call_role_with_timeout(
                AgentRole.CHECKER, user_input,
                context=f"## 链执行计划\n{chain_plan_hint}\n\n## 本轮执行结果\n{results_text[:4000]}",
            )
            self._record_role_progress(
                "checker", "审计者", "done",
                detail=check[:500], phase="execution",
            )

            # 9. 反思者 → 裁决继续/结束
            renderer.console.print("  [dim]⏳ 反思者 正在裁决是否继续...[/]")
            cont = self._call_role_with_timeout(
                AgentRole.REFLECTOR, user_input,
                context=f"## 链执行计划\n{chain_plan_hint}\n\n## 审计意见\n{check[:2000]}",
                extra_instruction=(
                    "判断当前链是否已构成有效危害或已无推进空间。"
                    "第一行写「继续迭代」或「结束协作」，然后给出理由。"
                ),
            )
            self._record_role_progress(
                "reflector", "反思者", "done",
                detail=cont[:500], phase="execution",
            )
            if "结束" in cont:
                break

        # ── Phase 3: 聚合输出 ──
        renderer.console.print()
        renderer.console.print("[dim bold]📊 原语工作流执行完成[/]")
        self._final_summary = self._build_primitive_summary(all_results, iteration)
        renderer.print_markdown(self._final_summary)

        # ── 积累回路：把本次运行的链裁决实例沉淀回数据层 ──
        self._feedback_sediment(all_results)

    # ═══ 积累回路（真实运行 → 链库/候选链回写）═══

    # 正向证据信号：执行结果命中这些关键词视为「验证有产出」
    _POSITIVE_EVIDENCE_MARKERS = (
        "泄露", "未授权", "越权", "确认", "存在", "可达", "暴露",
        "200", "401", "403", "302", "SSRF", "IDOR", "token", "内网",
        "路由", "版本", "指纹", "flag", "凭证", "源码",
    )

    def _feedback_sediment(self, all_results: list[list[str]]) -> None:
        """把本次原语工作流的链裁决实例沉淀回数据层（积累回路）。

        策略（避免污染正式链库）：
        - 仅处理裁决者判定 verdict=execute 的链；
        - 执行结果含正向证据关键词才写入实例（防止「候选未验证」污染）；
        - 链已在正式链库 → record_chain_instance 追加实例；
        - 链不在正式链库（模型自创）→ append_chain_candidate 落候选文件，
          由人工/沉淀脚本确认后再 upsert 进正式链库；
        - 开关：runtime_context["feedback_enabled"]=False 可关闭（测试/基准场景）。
        """
        if not self._chain_verdicts:
            return
        enabled = bool(self._runtime_context.get("feedback_enabled", True))
        if not enabled:
            return
        results_text = "\n".join(
            body for round_ in all_results for body in round_
        )
        has_evidence = any(m in results_text for m in self._POSITIVE_EVIDENCE_MARKERS)
        known_ids = load_chain_ids()
        target = str(self._runtime_context.get("task_target") or "")
        sediment_count = 0
        for verdict_item in self._chain_verdicts:
            if not isinstance(verdict_item, dict):
                continue
            if str(verdict_item.get("verdict", "")).strip().lower() != "execute":
                continue
            chain_id = str(verdict_item.get("chain_id", "")).strip()
            if not chain_id:
                continue
            instance = {
                "target": target,
                "endpoint": str(verdict_item.get("key_endpoints") or "")[:200],
                "method": "",
                "verdict": "execute",
                "priority": str(verdict_item.get("priority", "medium")),
                "finding": str(verdict_item.get("exploitation_plan", ""))[:300]
                if has_evidence else "候选链已裁决 execute，执行未确认正向证据",
                "date": datetime.now().strftime("%Y-%m-%d"),
            }
            if chain_id in known_ids:
                ok = record_chain_instance(chain_id, instance)
            else:
                ok = append_chain_candidate(
                    {
                        "id": chain_id,
                        "name": str(verdict_item.get("chain_id", chain_id)),
                        "primitives": [],
                        "logic": str(verdict_item.get("escalation", ""))[:300],
                        "gain": "候选链（模型自创，待确认）",
                        "source_plan": str(verdict_item.get("exploitation_plan", ""))[:500],
                    }
                )
            if ok:
                sediment_count += 1
        if sediment_count:
            self._record_trace(
                "feedback_sediment",
                detail=f"沉淀 {sediment_count} 条链实例（含候选）",
            )

    # ═══ 辅助 ═══
    def _format_chain_plan_hint(self) -> str:
        """把链裁决结果 + 程序化候选 + 攻击面测试向量汇总成可读的执行计划提示。"""
        lines = []
        if self._chain_verdicts:
            lines.append("## 链裁决（裁决者判定）")
            for v in self._chain_verdicts[:5]:
                cid = v.get("chain_id", "?")
                verdict = v.get("verdict", "?")
                plan = str(v.get("exploitation_plan", ""))[:400]
                lines.append(f"- {cid} [{verdict}]: {plan}")
        if self._chain_candidates:
            lines.append("## 程序化链候选（原语齐备）")
            for c in self._chain_candidates[:5]:
                lines.append(
                    f"- {c.get('name', c.get('chain_id', '?'))}: "
                    f"{'+'.join(c.get('primitives', []))} → {c.get('gain', '')}"
                )
        # 注入攻击面扩散的具体 curl 测试向量，供决策者直接复用
        if self._attack_plan:
            lines.append("## 攻击面测试向量（可直接复用为 curl 子任务）")
            for ap in self._attack_plan[:6]:
                ep = ap.get("endpoint", "?")
                attr = ap.get("business_attr", "?")
                vectors = ap.get("test_vectors", [])
                lines.append(f"- {ep} [{attr}] priority={ap.get('priority', '?')}")
                for vec in vectors[:4]:
                    lines.append(f"    `{vec}`")
        return "\n".join(lines)

    @staticmethod
    def _build_primitive_summary(all_results: list[list[str]], iteration: int) -> str:
        """构建原语工作流执行总结。"""
        total_tasks = sum(len(r) for r in all_results)
        fail_count = sum(
            1 for r in all_results for body in r if "❌ 失败:" in body[:80]
        )
        success = total_tasks - fail_count
        summary = [
            "## 🧬 原语工作流执行总结",
            "",
            f"共执行 {total_tasks} 个子任务（链验证/利用），经过 {iteration} 轮迭代。"
            f" ✅ 成功 {success} | ❌ 失败 {fail_count}",
        ]
        for idx, round_ in enumerate(all_results, 1):
            summary.append(f"\n### 第 {idx} 轮迭代 ({len(round_)} 个子任务)")
            for r in round_:
                lines = r.split("\n", 1)
                heading = lines[0].lstrip("# ")
                is_fail = "❌ 失败:" in lines[1][:80] if len(lines) > 1 else False
                summary.append(f"- {'❌' if is_fail else '✅'} {heading[:200]}")
        summary.append("")
        return "\n".join(summary)

    @staticmethod
    def _monotonic() -> float:
        import time as _time
        return _time.monotonic()