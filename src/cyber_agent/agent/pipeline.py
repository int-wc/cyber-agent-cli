"""四柱 Agent 管线：反思为主、迁跃为辅、分析为底、扩展为路。

管线流程:
  Phase 1 - 四柱思考（纯 LLM 调用，无工具，按序传递上下文）
    1. 分析者 ANALYST   → 深度分析 —— 为底
    2. 扩散者 DIFFUSER  → 路径探索 —— 为路
    3. 迁跃者 JUMPER    → 创造跨越 —— 为辅
    4. 反思者 REFLECTOR → 综合审视 + 制定执行计划 —— 为主

  Phase 2 - 执行循环（反思闭环，默认最多 20 轮，可通过 PIPELINE_MAX_ITERATIONS 配置）
    5. 决策者 DECISION_MAKER → 分解子任务
    6. 思考者 THINKER / 用户  → 选择子任务
    7. 执行者 RUNNER/READER/BUILDER → 使用隔离 runner 顺序/并行执行子任务
    8. 审计者 CHECKER    → 验证结果
    9. 反思者 REFLECTOR  → 审视结果，决定循环继续或结束
"""

from __future__ import annotations

import concurrent.futures
from collections.abc import Callable
from datetime import datetime
import json
from pathlib import Path
import re as _re_mod
import subprocess
import threading
import time as time_mod
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..execution_control import ExecutionController, ExecutionInterruptedError
from ..model_client import build_llm_with_proxy_fallback
from .events import AgentEventType
from .roles import AgentRole, get_role_label, get_role_prompt

if TYPE_CHECKING:
    from .runner import AgentRunner

# ── 超时与熔断常量 ──
BASE_SUBTASK_TIMEOUT = 300           # 子任务基础超时（秒），复杂分析需 5 分钟以上
TIMEOUT_ESCALATION_STEP = 60         # 每次超时叠加步长（秒）
MAX_TIMEOUT_ESCALATIONS = 3          # 最多叠加次数 → 最大 300+3×60=480s
LLM_CALL_TIMEOUT_SECONDS = 120       # 单次角色 LLM 调用超时（秒）
CIRCUIT_BREAKER_CONSECUTIVE_FAILS = 2  # 连续失败 N 次触发熔断


class PipelineCircuitBreakerError(RuntimeError):
    """连续子任务失败触发的熔断异常。"""


LOCAL_PROJECT_PROBE_PATTERNS = (
    "/cyber-agent-cli",
    "/home/my/cyber/claude.md",
    ".cyber-agent-cli",
    ".cyber-agent-cli-sessions",
    ".cyber-agent-cli-capabilities",
    ".cyber-agent-cli-history",
    ".cyber-agent-cli.json",
    ".cyber/sessions",
    ".cyber/",
    ".claude/",
    ".claude/settings",
    "claude.md",
    "webhook-routes.json",
    "/desktop/",
    "/src/cyber_agent/",
    "/tests/",
)
LOCAL_SECRET_PROBE_PATTERNS = (
    ".env",
    "opencode_api_key",
    "anthropic_auth_token",
    "gateway_api_key",
)
BROAD_LOCAL_SEARCH_PATTERNS = (
    "find /home",
    "find / ",
    "grep -r",
    "grep -rn",
    "cat /home",
)
LOCAL_ENV_PROBE_PATTERNS = (
    "env |",
    "env|",
    "printenv",
    "set |",
    "set|",
)
CYBER_AGENT_TASK_KEYWORDS = (
    "cyber-agent",
    "cyber_agent",
    "agent runner",
    "fourpillarpipeline",
    "四柱",
    "上下文压缩",
)
CYBER_AGENT_DEBUG_INTENT_KEYWORDS = (
    "调试",
    "排查",
    "修复",
    "修改",
    "实现",
    "开发",
    "测试",
    "debug",
    "fix",
    "bug",
    "review",
)


class FourPillarPipeline:
    """四柱管线协调器。所有 10 个角色各司其职。"""

    def __init__(
        self,
        *,
        runner: AgentRunner,
        runtime_context: dict[str, object],
        renderer: Any,
        event_handler: Callable[[str, object], None] | None = None,
    ) -> None:
        self._runner = runner
        self._runtime_context = runtime_context
        self._renderer = renderer
        self._event_handler = event_handler
        self._llm: Any = None

        # 累计 token（供 renderer 读取）
        self.cumulative_input_tokens = 0
        self.cumulative_output_tokens = 0

        # 熔断器状态
        self._consecutive_failures = 0

        # 执行轨迹
        self._trace: list[dict] = []
        self._session_id: str = ""
        self._trace_id: str = ""
        self._final_summary: str = ""
        self._trace_lock = threading.RLock()
        self._usage_lock = threading.RLock()
        self._benchmark_profile_active = False
        self._benchmark_current_challenge: str | None = None
        self._benchmark_stale_rounds = 0
        self._benchmark_forced_directive = ""
        self._benchmark_state_lock = threading.RLock()
        self._benchmark_state: dict[str, Any] = self._new_benchmark_state()

    @staticmethod
    def _new_benchmark_state() -> dict[str, Any]:
        return {
            "vpn_connected": False,
            "tun_interface": None,
            "tun_ip": None,
            "vpn_config": None,
            "api_interface": None,
            "task_finished": False,
            "current_challenge": None,
            "active_containers": {},
            "last_challenges_snapshot": None,
            "completed_challenges": set(),
            "completed_scores": {},
            "closed_challenges": set(),
            "last_score": None,
            "background_processes_cleaned": False,
        }

    # ── LLM 管理 ──
    def _get_llm(self) -> Any:
        """懒加载用于角色思考的 LLM（无工具绑定，纯文本调用）。
        自动检测 API 格式使用对应客户端。"""
        if self._llm is not None:
            return self._llm

        from .._lazy_imports import load_llm_for_api
        from ..config import settings

        service_name = str(self._runtime_context.get("service_name", "deepseek"))
        api_key = str(self._runtime_context.get("api_key", ""))
        base_url = str(self._runtime_context.get("base_url", "")) if self._runtime_context.get("base_url") is not None else None
        if not base_url:
            base_url = settings.resolve_base_url(service_name)

        # 角色思考用子模型降低成本
        model_name = settings.subagent_model.replace("[1m]", "").strip()

        llm_cls, is_anthropic = load_llm_for_api(base_url)

        kwargs = settings.get_chat_openai_kwargs(
            service_name,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
        )

        if is_anthropic:
            # ChatAnthropic 不支持 extra_body / openai_api_key
            kwargs.pop("extra_body", None)
            kwargs.pop("openai_api_key", None)
            kwargs.pop("openai_proxy", None)
            kwargs["anthropic_api_key"] = kwargs.pop("api_key", "")
            # 在 extra_body 被移除后不再设置 thinking
        else:
            # OpenAI 兼容端点：禁用 thinking，角色不需要深度推理
            if "extra_body" in kwargs and isinstance(kwargs["extra_body"], dict):
                kwargs["extra_body"]["thinking"] = {"type": "disabled"}

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*extra_body.*")
            self._llm = build_llm_with_proxy_fallback(llm_cls, kwargs)
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
        execution_guidance = self._build_execution_profile_guidance()

        full_system = f"""{system_prompt}

## 系统环境
{system_context}

## 执行姿态
{execution_guidance}"""

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

    # ── 执行轨迹 ──
    def _record_trace(
        self,
        event: str,
        *,
        detail: str = "",
        metadata: dict | None = None,
    ) -> None:
        """记录一条执行轨迹事件。"""
        trace_event = {
            "event": event,
            "timestamp": datetime.now().isoformat(),
            "detail": detail,
            "metadata": metadata or {},
        }
        with self._trace_lock:
            self._trace.append(trace_event)
        self._append_session_event(f"pipeline.{event}", trace_event)
        if self._event_handler is not None:
            try:
                self._event_handler(f"pipeline.{event}", trace_event)
            except Exception as exc:
                from ..logging import log_warning
                log_warning("pipeline", f"转发管线事件失败：{exc}")

    def _record_role_progress(
        self,
        role_key: str,
        label: str,
        status: str,
        *,
        action: str = "",
        detail: str = "",
        elapsed_ms: float | None = None,
        phase: str = "",
    ) -> None:
        metadata: dict[str, object] = {
            "role": role_key,
            "label": label,
            "status": status,
            "action": action,
            "phase": phase,
        }
        if elapsed_ms is not None:
            metadata["elapsed_ms"] = round(elapsed_ms)
        self._record_trace(
            "role_progress",
            detail=detail,
            metadata=metadata,
        )

    def _append_session_event(self, event: str, payload: object) -> None:
        """把管线事件同步写入当前会话事件流。"""
        session_id = str(self._runtime_context.get("session_id") or "").strip()
        if not session_id:
            return
        try:
            from ..session_store import append_session_event

            raw_base_dir = self._runtime_context.get("session_base_dir")
            base_dir = Path(str(raw_base_dir)).expanduser() if raw_base_dir else None
            event_path = append_session_event(
                session_id,
                event,
                payload=payload,
                base_dir=base_dir,
            )
            self._runtime_context["session_event_log"] = event_path
        except Exception as exc:
            from ..logging import log_error
            log_error("trace", f"保存管线事件失败：{exc}")

    def _persist_main_session(self) -> None:
        """管线内直接保存主 runner 历史，避免长任务结束前 session 不可见。"""
        session_id = str(self._runtime_context.get("session_id") or "").strip()
        if not session_id:
            return
        try:
            from ..session_store import save_session_history

            approval_policy = self._runtime_context.get("approval_policy", "prompt")
            approval_value = getattr(approval_policy, "value", str(approval_policy))
            raw_base_dir = self._runtime_context.get("session_base_dir")
            base_dir = Path(str(raw_base_dir)).expanduser() if raw_base_dir else None
            session_path = save_session_history(
                session_id,
                self._runner.get_history_snapshot(),
                mode=getattr(getattr(self._runner, "mode", None), "value", "standard"),
                approval_policy=approval_value,
                source_session_id=self._runtime_context.get("session_source_id"),
                recent_inputs=self._runtime_context.get("_recent_inputs"),
                base_dir=base_dir,
            )
            self._runtime_context["session_storage_dir"] = session_path.parent
        except Exception as exc:
            from ..logging import log_error
            log_error("trace", f"保存管线主会话失败：{exc}")

    def _append_pipeline_user_message(self, user_input: str) -> None:
        """四柱管线不走 AgentRunner.run，需要主动把用户输入纳入主 history。"""
        message = HumanMessage(
            content=user_input,
            additional_kwargs={"cyber_agent_pipeline": True},
        )
        self._runner.history.append(message)
        self._append_session_event(
            "history_updated",
            {
                "reason": "pipeline_human_message",
                "message_type": message.type,
                "message_count": len(self._runner.history),
                "turn_count": self._runner.get_turn_count(),
            },
        )
        self._persist_main_session()

    def _append_pipeline_summary_message(self) -> None:
        """把完整四柱总结写回主会话，供 /history 和后续对话使用。"""
        summary = self._final_summary.strip()
        if not summary:
            return
        message = AIMessage(
            content=summary,
            additional_kwargs={"cyber_agent_pipeline_summary": True},
        )
        self._runner.history.append(message)
        self._append_session_event(
            "history_updated",
            {
                "reason": "pipeline_summary_message",
                "message_type": message.type,
                "message_count": len(self._runner.history),
                "turn_count": self._runner.get_turn_count(),
            },
        )
        self._persist_main_session()

    @staticmethod
    def _build_execution_summary(
        all_results: list[list[str]],
        iteration: int,
    ) -> str:
        """构建完整的四柱执行总结，不截断子任务正文。"""
        total_tasks = sum(len(round_) for round_ in all_results)
        success_count = 0
        fail_count = 0
        for round_ in all_results:
            for r in round_:
                body = r.split("\n", 1)[1] if "\n" in r else r
                is_fail = body.startswith("❌ 失败:") or "❌ 全部超时叠加后重规划也失败" in body[:60]
                if is_fail:
                    fail_count += 1
                else:
                    success_count += 1

        summary_parts = [
            "## 📊 四柱管线执行总结\n",
            f"共执行 {total_tasks} 个子任务，经过 {iteration} 轮迭代。"
            f" ✅ 成功 {success_count} | ❌ 失败 {fail_count}",
        ]

        for iter_idx, round_ in enumerate(all_results, 1):
            summary_parts.append(
                f"\n### 第 {iter_idx} 轮迭代 ({len(round_)} 个子任务)"
            )
            for r in round_:
                lines = r.split("\n", 1)
                heading = lines[0].lstrip("# ")
                body = lines[1] if len(lines) > 1 else ""
                is_fail = body.startswith("❌ 失败:") or "❌ 全部超时叠加后重规划也失败" in body[:60]
                prefix = "❌" if is_fail else "✅"
                summary_parts.append(f"- {prefix} {heading}")
                if body.strip():
                    summary_parts.append("")
                    summary_parts.append(body.strip())
                    summary_parts.append("")
        summary_parts.append("")
        return "\n".join(summary_parts)

    def _save_trace(self) -> None:
        """将执行轨迹保存到会话目录。"""
        if not self._trace:
            return
        try:
            storage_dir = Path.home() / ".cyber-agent-cli-traces"
            storage_dir.mkdir(parents=True, exist_ok=True)
            sid = self._trace_id or self._session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
            trace_file = storage_dir / f"{sid}.trace.json"
            trace_file.write_text(
                json.dumps(self._trace, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self._append_session_event(
                "pipeline.trace_saved",
                {"trace_file": str(trace_file), "event_count": len(self._trace)},
            )
        except Exception as exc:
            from ..logging import log_error
            log_error("trace", f"保存执行轨迹失败：{exc}")

    def _load_trace(self, session_id: str) -> list[dict] | None:
        """加载指定会话的执行轨迹。"""
        try:
            trace_file = Path.home() / ".cyber-agent-cli-traces" / f"{session_id}.trace.json"
            if not trace_file.exists():
                return None
            return json.loads(trace_file.read_text(encoding="utf-8"))
        except Exception:
            return None

    # ── 超时与熔断 ──
    def _check_circuit_breaker(self) -> None:
        """检查熔断器：连续失败超过阈值则抛出异常。"""
        if self._consecutive_failures >= CIRCUIT_BREAKER_CONSECUTIVE_FAILS:
            raise PipelineCircuitBreakerError(
                f"连续 {self._consecutive_failures} 个子任务失败，触发熔断保护。"
                f"请检查任务是否合理或简化需求后重试。"
            )

    def _auto_approval_handler(self, tool: Any, tool_call: dict) -> "ApprovalDecision":
        """管线自动批准工具调用，由子任务审批器负责额外边界约束。"""
        from .approval import ApprovalDecision
        return ApprovalDecision(True, "管线自动批准所有工具调用。")

    @staticmethod
    def _extract_tool_text(tool_call: dict) -> str:
        args = tool_call.get("args", {})
        if not isinstance(args, dict):
            return ""
        parts: list[str] = []
        for key in ("command", "path", "working_directory", "url"):
            value = args.get(key)
            if isinstance(value, str):
                parts.append(value)
        return "\n".join(parts)

    @staticmethod
    def _command_uses_interface(command: str) -> str | None:
        match = _re_mod.search(r"--interface(?:=|\s+)([^\s'\"\\]+)", command)
        return match.group(1) if match else None

    @staticmethod
    def _extract_unique_code(text: str) -> str | None:
        match = _re_mod.search(r"\bunique_code[\"'=:/\s]+(xben-\d+-\d+)\b", text)
        if match:
            return match.group(1)
        match = _re_mod.search(r"\b(xben-\d+-\d+)\b", text)
        return match.group(1) if match else None

    def _benchmark_tool_guard(self, tool_call: dict) -> str | None:
        """Block Benchmark actions that are known to waste time or corrupt state."""
        if not self._is_benchmark_aggressive():
            return None

        tool_name = str(tool_call.get("name", ""))
        if tool_name != "run_shell_command":
            return None

        command = self._extract_tool_text(tool_call)
        lowered = command.lower()
        if not command:
            return None

        with self._benchmark_state_lock:
            state = dict(self._benchmark_state)
            completed = set(self._benchmark_state.get("completed_challenges", set()))
            closed = set(self._benchmark_state.get("closed_challenges", set()))
            active = dict(self._benchmark_state.get("active_containers", {}))

        if state.get("task_finished") and (
            "tsecbench.zc.tencent.com" in lowered
            or "/openapi/v1/challenges" in lowered
            or "10.0." in lowered
            or "auto_solve.sh" in lowered
        ):
            return "Benchmark 任务已返回 invalid_state/finished，禁止继续调用平台或靶场容器。"

        if "auto_solve.sh" in lowered and (
            _re_mod.search(r"\b(?:bash|sh)\s+\.?/?.*auto_solve\.sh\b", lowered)
            or "nohup" in lowered
            or "&" in command
        ):
            return (
                "Benchmark Hub 持有唯一 runner 和状态队列，禁止在子任务中启动 "
                "auto_solve.sh 后台跑分脚本，避免与 Hub 抢题、重复 start/submit。"
            )

        if "openvpn" in lowered and ("--config" in lowered or ".ovpn" in lowered):
            if state.get("vpn_connected"):
                tun = state.get("tun_interface") or "tun0"
                ip = state.get("tun_ip") or "已分配 IP"
                return (
                    f"VPN 已连接（{tun} {ip}），禁止重复启动 OpenVPN；"
                    "直接复用现有 VPN 状态继续解题。"
                )

        is_platform_api = (
            "tsecbench.zc.tencent.com" in lowered
            or "/openapi/v1/challenges" in lowered
        )
        if is_platform_api:
            used_interface = self._command_uses_interface(command)
            if used_interface == "tun0":
                api_interface = state.get("api_interface") or "物理网卡"
                return (
                    "平台 API 禁止走 tun0；tun0 只用于访问 10.x 容器。"
                    f"请改用 --interface {api_interface} 并携带 BENCHMARK_TOKEN。"
                )
            if state.get("api_interface") and used_interface is None:
                return (
                    "已确认平台 API 需要显式绑定出口，禁止不带 --interface 调用平台 API；"
                    f"请使用 --interface {state['api_interface']}。"
                )
            if "authorization:" in lowered and "benchmark_token:" not in lowered:
                return (
                    "TSec Benchmark 平台认证头必须是 BENCHMARK_TOKEN，"
                    "不要使用 Authorization: Bearer。"
                )
            if "x-benchmark-token:" in lowered:
                return (
                    "TSec Benchmark 平台认证头必须精确使用 BENCHMARK_TOKEN，"
                    "不要使用 X-Benchmark-Token。"
                )
            if "/challenges/submit" in lowered and "challenge_id" in lowered:
                return (
                    "提交接口请求体字段必须使用 unique_code，"
                    "不要使用 challenge_id。"
                )

            if "/challenges/start" in lowered:
                code = self._extract_unique_code(command)
                if code in completed:
                    return f"{code} 已确认通关，禁止重复 start。"
                if code in closed:
                    return f"{code} 已在本轮放弃/关闭，禁止回头重复 start；请选择下一道未完成题。"

        for code, addrs in active.items():
            if code in completed or code in closed:
                for addr in addrs or []:
                    if isinstance(addr, str) and addr and addr in command:
                        return (
                            f"{code} 的容器地址 {addr} 已不应继续探测；"
                            "请切换到当前活跃题或获取最新题目列表。"
                        )

        return None

    @staticmethod
    def _extract_command_from_tool_result(content: str) -> str:
        match = _re_mod.search(r"命令:\s*(.*?)\n工作目录:", content, _re_mod.DOTALL)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _extract_output_from_tool_result(content: str) -> str:
        match = _re_mod.search(r"\n输出:\n(.*)$", content, _re_mod.DOTALL)
        return match.group(1).strip() if match else content

    @staticmethod
    def _iter_json_fragments(text: str) -> list[Any]:
        decoder = json.JSONDecoder()
        results: list[Any] = []
        for idx, char in enumerate(text):
            if char not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(text[idx:])
            except Exception:
                continue
            results.append(value)
        return results

    def _update_benchmark_runtime_state(self, content: str) -> None:
        if not self._is_benchmark_aggressive() or not content:
            return

        command = self._extract_command_from_tool_result(content)
        output = self._extract_output_from_tool_result(content)
        combined = f"{command}\n{output}"
        lowered = combined.lower()
        command_lowered = command.lower()
        is_platform_challenges_command = "/openapi/v1/challenges" in command_lowered
        is_platform_submit_command = (
            is_platform_challenges_command and "/openapi/v1/challenges/submit" in command_lowered
        )
        updates: dict[str, Any] = {}

        tun_match = _re_mod.search(
            r"\b(tun\d+):[\s\S]{0,240}?\binet\s+([0-9.]+)/\d+",
            combined,
        )
        if tun_match:
            updates["vpn_connected"] = True
            updates["tun_interface"] = tun_match.group(1)
            updates["tun_ip"] = tun_match.group(2)

        vpn_config = _re_mod.search(r"openvpn\s+--config\s+([^\s]+\.ovpn)", command)
        if vpn_config:
            updates["vpn_config"] = vpn_config.group(1)

        if (
            "tsecbench.zc.tencent.com" in lowered
            and "/openapi/v1/challenges" in lowered
            and "退出码: 0" in content
            and "invalid_state" not in lowered
            and "task_not_found" not in lowered
        ):
            used_interface = self._command_uses_interface(command)
            if used_interface and used_interface != "tun0":
                updates["api_interface"] = used_interface

        if is_platform_challenges_command and "invalid_state" in lowered and "finished" in lowered:
            updates["task_finished"] = True

        completed: set[str] = set()
        completed_scores: dict[str, int] = {}
        closed: set[str] = set()
        active_updates: dict[str, list[str]] = {}
        current_challenge: str | None = None
        last_score: int | None = None
        challenges_snapshot: list[dict[str, Any]] | None = None

        json_fragments = self._iter_json_fragments(output) if is_platform_challenges_command else []
        for value in json_fragments:
            if isinstance(value, dict):
                code = value.get("unique_code")
                if isinstance(code, str) and _re_mod.fullmatch(r"xben-\d+-\d+", code):
                    current_challenge = code
                    addrs = value.get("container_addr")
                    status = value.get("container_status")
                    if isinstance(addrs, list) and addrs and (
                        status == "available"
                        or (
                            status is None
                            and "/challenges/start" in lowered
                        )
                    ):
                        active_updates[code] = [str(addr) for addr in addrs]
                    if value.get("closed") is True:
                        closed.add(code)

                if value.get("correct") is True:
                    submit_code = self._extract_unique_code(command)
                    if submit_code:
                        completed.add(submit_code)
                        current_challenge = submit_code
                    awarded = value.get("awarded")
                    score = value.get("cumulative_score")
                    if isinstance(score, int) and submit_code:
                        completed_scores[submit_code] = score
                    elif isinstance(awarded, int) and submit_code:
                        completed_scores[submit_code] = awarded
                    if isinstance(score, int):
                        last_score = score

                if value.get("is_completed") is True:
                    code = value.get("unique_code")
                    if isinstance(code, str):
                        completed.add(code)
                        total_score = value.get("total_score")
                        if isinstance(total_score, int):
                            completed_scores.setdefault(code, total_score)

            elif isinstance(value, list):
                if value and all(
                    isinstance(item, dict) and "unique_code" in item
                    for item in value
                ):
                    challenges_snapshot = [dict(item) for item in value]
                for item in value:
                    if not isinstance(item, dict):
                        continue
                    code = item.get("unique_code")
                    if not isinstance(code, str):
                        continue
                    if item.get("is_completed") is True:
                        completed.add(code)
                        total_score = item.get("total_score")
                        if isinstance(total_score, int):
                            completed_scores.setdefault(code, total_score)
                    if item.get("container_status") == "available":
                        addrs = item.get("container_addr")
                        if isinstance(addrs, list) and addrs:
                            active_updates[code] = [str(addr) for addr in addrs]
                            current_challenge = code

        state_changed = bool(
            updates
            or completed
            or completed_scores
            or closed
            or active_updates
            or challenges_snapshot is not None
            or current_challenge
            or last_score is not None
        )
        if not state_changed:
            return

        with self._benchmark_state_lock:
            self._benchmark_state.update(updates)
            active = dict(self._benchmark_state.get("active_containers", {}))
            active.update(active_updates)
            completed_set = set(self._benchmark_state.get("completed_challenges", set()))
            closed_set = set(self._benchmark_state.get("closed_challenges", set()))
            score_map = dict(self._benchmark_state.get("completed_scores", {}))
            completed_set.update(completed)
            closed_set.update(closed)
            for code, score in completed_scores.items():
                if is_platform_submit_command:
                    score_map[code] = score
                else:
                    score_map.setdefault(code, score)
            if challenges_snapshot is not None:
                self._benchmark_state["last_challenges_snapshot"] = challenges_snapshot
                snapshot_completed = {
                    str(item["unique_code"])
                    for item in challenges_snapshot
                    if item.get("is_completed") is True
                }
                completed_set.update(snapshot_completed)
                for item in challenges_snapshot:
                    code = item.get("unique_code")
                    total_score = item.get("total_score")
                    if (
                        isinstance(code, str)
                        and code in completed_set
                        and isinstance(total_score, int)
                    ):
                        score_map.setdefault(code, total_score)
                active = {}
                for item in challenges_snapshot:
                    code = item.get("unique_code")
                    addrs = item.get("container_addr")
                    if (
                        isinstance(code, str)
                        and item.get("container_status") == "available"
                        and isinstance(addrs, list)
                        and addrs
                        and code not in completed_set
                    ):
                        active[code] = [str(addr) for addr in addrs]
            for code in completed_set | closed_set:
                active.pop(code, None)
            if self._benchmark_state.get("task_finished"):
                active = {}
            self._benchmark_state["active_containers"] = active
            self._benchmark_state["completed_challenges"] = completed_set
            self._benchmark_state["completed_scores"] = score_map
            self._benchmark_state["closed_challenges"] = closed_set
            if current_challenge and current_challenge not in completed_set and current_challenge not in closed_set:
                self._benchmark_state["current_challenge"] = current_challenge
                self._benchmark_current_challenge = current_challenge
            if last_score is not None:
                self._benchmark_state["last_score"] = last_score
            should_cleanup = (
                bool(self._benchmark_state.get("task_finished"))
                and not bool(self._benchmark_state.get("background_processes_cleaned"))
            )
            snapshot = self._benchmark_state_snapshot_unlocked()

        self._record_trace(
            "benchmark_state_updated",
            detail=self._benchmark_state_summary(snapshot),
            metadata=snapshot,
        )
        if should_cleanup:
            self._cleanup_benchmark_background_processes()

    def _benchmark_state_snapshot_unlocked(self) -> dict[str, Any]:
        last_snapshot = self._benchmark_state.get("last_challenges_snapshot")
        snapshot_summary = None
        if isinstance(last_snapshot, list):
            completed_items = [
                item for item in last_snapshot
                if isinstance(item, dict) and item.get("is_completed") is True
            ]
            active_items = [
                item for item in last_snapshot
                if isinstance(item, dict) and item.get("container_status") == "available"
            ]
            snapshot_summary = {
                "total": len(last_snapshot),
                "completed_count": len(completed_items),
                "completed_total_score": sum(
                    item.get("total_score") or 0 for item in completed_items
                ),
                "completed_challenges": sorted(
                    str(item.get("unique_code")) for item in completed_items
                ),
                "active_challenges": sorted(
                    str(item.get("unique_code")) for item in active_items
                ),
            }
        return {
            "vpn_connected": bool(self._benchmark_state.get("vpn_connected")),
            "tun_interface": self._benchmark_state.get("tun_interface"),
            "tun_ip": self._benchmark_state.get("tun_ip"),
            "vpn_config": self._benchmark_state.get("vpn_config"),
            "api_interface": self._benchmark_state.get("api_interface"),
            "task_finished": bool(self._benchmark_state.get("task_finished")),
            "current_challenge": self._benchmark_state.get("current_challenge"),
            "active_containers": dict(self._benchmark_state.get("active_containers", {})),
            "last_challenges_snapshot": snapshot_summary,
            "completed_challenges": sorted(self._benchmark_state.get("completed_challenges", set())),
            "completed_scores": dict(self._benchmark_state.get("completed_scores", {})),
            "closed_challenges": sorted(self._benchmark_state.get("closed_challenges", set())),
            "last_score": self._benchmark_state.get("last_score"),
            "background_processes_cleaned": bool(
                self._benchmark_state.get("background_processes_cleaned")
            ),
        }

    def _benchmark_state_snapshot(self) -> dict[str, Any]:
        with self._benchmark_state_lock:
            return self._benchmark_state_snapshot_unlocked()

    @staticmethod
    def _benchmark_state_summary(state: dict[str, Any]) -> str:
        parts: list[str] = []
        if state.get("task_finished"):
            parts.append("任务状态：平台已返回 invalid_state/finished，必须停止。")
        snapshot = state.get("last_challenges_snapshot")
        if isinstance(snapshot, dict):
            parts.append(
                "最后平台快照："
                f"{snapshot.get('completed_count', 0)}/{snapshot.get('total', '?')} "
                f"题完成，已完成题总分 {snapshot.get('completed_total_score', 0)}。"
            )
        if state.get("vpn_connected"):
            tun = state.get("tun_interface") or "tun0"
            ip = state.get("tun_ip") or "unknown"
            parts.append(f"VPN：已连接 {tun} {ip}，不要重复启动 OpenVPN。")
        if state.get("api_interface"):
            parts.append(
                f"平台 API：使用 --interface {state['api_interface']}，"
                "认证头必须为 BENCHMARK_TOKEN；不要用 tun0 或 Authorization Bearer。"
            )
        active = state.get("active_containers") or {}
        if active:
            active_text = ", ".join(
                f"{code}=>{','.join(addrs)}"
                for code, addrs in sorted(active.items())
            )
            parts.append(f"当前活跃容器：{active_text}。")
        if state.get("current_challenge"):
            parts.append(f"当前题：{state['current_challenge']}。")
        if state.get("completed_challenges"):
            parts.append(
                "已通关题："
                + ", ".join(state["completed_challenges"])
                + "；禁止重复 start/探测。"
            )
            scores = state.get("completed_scores") or {}
            if scores:
                known_score = sum(
                    score for score in scores.values() if isinstance(score, int)
                )
                parts.append(
                    f"已知通关得分：{known_score}（{len(scores)} 道题有分值记录）。"
                )
        if state.get("closed_challenges"):
            parts.append(
                "已关闭/放弃题："
                + ", ".join(state["closed_challenges"])
                + "；本轮不要回头。"
            )
        if state.get("last_score") is not None:
            parts.append(f"最近题目累计得分：{state['last_score']}。")
        return "\n".join(parts)

    def _benchmark_state_context(self) -> str:
        if not self._is_benchmark_aggressive():
            return ""
        summary = self._benchmark_state_summary(self._benchmark_state_snapshot())
        target_directive = self._benchmark_target_continue_directive()
        if target_directive:
            summary = f"{summary}\n{target_directive}" if summary else target_directive
        if not summary:
            return ""
        return (
            "## Benchmark 已确认运行态（必须信任并复用）\n"
            f"{summary}\n"
            "除非上面的状态被平台 API 明确推翻，否则不要重复做 VPN 启动、"
            "不要回头探测已完成/已关闭题。"
        )

    def _benchmark_final_summary(self) -> str:
        if not self._is_benchmark_aggressive():
            return ""
        state = self._benchmark_state_snapshot()
        snapshot = state.get("last_challenges_snapshot")
        completed = list(state.get("completed_challenges") or [])
        scores = state.get("completed_scores") or {}
        total_score = sum(score for score in scores.values() if isinstance(score, int))
        total_count = len(completed)
        total_challenges: int | str = "?"

        if isinstance(snapshot, dict):
            total_count = int(snapshot.get("completed_count") or total_count)
            total_score = int(snapshot.get("completed_total_score") or total_score)
            total_challenges = int(snapshot.get("total") or 0) or "?"
            completed = list(snapshot.get("completed_challenges") or completed)

        target_score = self._resolve_benchmark_target_score()
        target_line = ""
        if target_score > 0:
            remain = max(0, target_score - total_score)
            target_line = (
                f"\n- 目标分数: {target_score}"
                f"\n- 距离目标: {remain}"
            )

        status = "finished" if state.get("task_finished") else "running/unknown"
        completed_text = ", ".join(completed) if completed else "无"
        return (
            "\n\n## Benchmark 最终状态\n"
            f"- 任务状态: {status}\n"
            f"- 已通关题数: {total_count}/{total_challenges}\n"
            f"- 已知总分: {total_score}"
            f"{target_line}\n"
            f"- 已通关题: {completed_text}"
        )

    def _cleanup_benchmark_background_processes(self) -> None:
        """Stop helper scripts after a Benchmark task is known to be finished."""
        with self._benchmark_state_lock:
            if self._benchmark_state.get("background_processes_cleaned"):
                return
            self._benchmark_state["background_processes_cleaned"] = True

        try:
            result = subprocess.run(
                ["pkill", "-TERM", "-f", r"bash .*auto_solve\.sh|auto_solve\.sh"],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
            )
            self._record_trace(
                "benchmark_background_cleanup",
                detail="已尝试终止 auto_solve.sh 后台进程。",
                metadata={
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                },
            )
        except Exception as exc:
            self._record_trace(
                "benchmark_background_cleanup_failed",
                detail=str(exc),
            )

    @staticmethod
    def _is_cyber_agent_debug_task(task_text: str) -> bool:
        lowered = task_text.lower().split("任务边界", 1)[0]
        mentions_project = any(
            keyword in lowered for keyword in CYBER_AGENT_TASK_KEYWORDS
        )
        has_debug_intent = any(
            keyword in lowered for keyword in CYBER_AGENT_DEBUG_INTENT_KEYWORDS
        )
        return mentions_project and has_debug_intent

    @staticmethod
    def _resolve_max_iterations() -> int:
        from ..config import settings

        raw_value = getattr(settings, "pipeline_max_iterations", 20)
        configured = int(raw_value if raw_value is not None else 20)
        return max(1, min(configured, 100))

    def _resolve_max_subagents(self) -> int:
        from ..config import settings

        raw_value = self._runtime_context.get(
            "max_subagents",
            getattr(settings, "pipeline_max_subagents", 4),
        )
        try:
            configured = int(raw_value)
        except (TypeError, ValueError):
            configured = 4
        return max(1, min(configured, 16))

    def _resolve_subtask_concurrency(self) -> str:
        from ..config import settings

        raw_value = str(
            self._runtime_context.get(
                "subtask_concurrency",
                getattr(settings, "pipeline_subtask_concurrency", "auto"),
            )
        ).strip().lower()
        if raw_value not in {"off", "auto", "force"}:
            return "auto"
        if self._resolve_max_subagents() <= 1:
            return "off"
        return raw_value

    @staticmethod
    def _extract_subtask_resource_keys(task: dict) -> set[str]:
        """提取子任务涉及的资源锁，避免并发时互相踩同一目标。"""
        text_parts = [
            str(task.get("task_description", "")),
            str(task.get("context", "")),
        ]
        text = "\n".join(text_parts)
        lowered = text.lower()
        keys: set[str] = set()

        for match in _re_mod.findall(r"\bxben-\d+-\d+\b", lowered):
            keys.add(f"challenge:{match}")

        for match in _re_mod.findall(
            r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d{1,5})?\b",
            lowered,
        ):
            keys.add(f"host:{match}")

        for match in _re_mod.findall(r"(?<!\w)/(?:[^\s'\"`<>|;&]+)", text):
            keys.add(f"file:{match.rstrip('.,:;')}")

        if any(word in lowered for word in ("session", "会话", "history", ".events.jsonl")):
            keys.add("session:current")

        if "tsecbench" in lowered or "/openapi/v1/challenges" in lowered:
            keys.add("api:tsecbench")

        if any(word in lowered for word in ("submit", "提交 flag", "提交flag")):
            keys.add("api:tsecbench-submit")
        if any(word in lowered for word in ("start", "启动")) and "xben-" in lowered:
            keys.add("api:tsecbench-start")
        if any(word in lowered for word in ("close", "关闭", "释放")) and "xben-" in lowered:
            keys.add("api:tsecbench-close")

        return keys

    @staticmethod
    def _is_subtask_sensitive(task: dict) -> bool:
        """敏感操作强制顺序，优先保证外部状态一致。"""
        text = (
            f"{task.get('task_description', '')}\n{task.get('context', '')}"
        ).lower()
        sensitive_markers = (
            "submit",
            "提交 flag",
            "提交flag",
            "close",
            "关闭容器",
            "释放资源",
            "/session",
            "切换会话",
            "save_session",
            ".events.jsonl",
        )
        if any(marker in text for marker in sensitive_markers):
            return True
        if "start" in text and "xben-" in text:
            return True
        if "启动" in text and "xben-" in text:
            return True
        return False

    def _subtask_parallel_decision(self, task: dict) -> tuple[bool, str]:
        strategy = self._resolve_subtask_concurrency()
        if strategy == "off":
            return False, "concurrency_off"
        if self._is_subtask_sensitive(task):
            return False, "sensitive_operation"
        if strategy == "force":
            return True, "force"
        if bool(task.get("parallel", False)):
            return True, "llm_parallel"
        return False, "not_marked_parallel"

    @staticmethod
    def _is_boundary_probe(tool_call: dict) -> str | None:
        tool_name = str(tool_call.get("name", ""))
        args = tool_call.get("args", {})
        if not isinstance(args, dict):
            return None

        haystack_parts = []
        for key in ("command", "path", "working_directory"):
            value = args.get(key)
            if isinstance(value, str):
                haystack_parts.append(value)
        haystack = "\n".join(haystack_parts).lower()
        if not haystack:
            return None

        if any(pattern in haystack for pattern in LOCAL_SECRET_PROBE_PATTERNS):
            return "禁止在内部子任务中探测本机密钥、.env 或凭证配置。"

        if any(pattern in haystack for pattern in LOCAL_PROJECT_PROBE_PATTERNS):
            return "禁止在无关内部子任务中探测 cyber-agent 本地源码、历史会话或桌面端目录。"

        if tool_name == "run_shell_command" and any(
            pattern in haystack for pattern in BROAD_LOCAL_SEARCH_PATTERNS
        ):
            return "禁止在内部子任务中对本机 /home 或根目录做大范围搜索。"

        if tool_name == "run_shell_command":
            probes_env = any(
                pattern in haystack for pattern in LOCAL_ENV_PROBE_PATTERNS
            )
            if probes_env:
                return "禁止在内部子任务中探测本机环境变量。"

        return None

    def _make_subtask_approval_handler(self, task_text: str) -> Any:
        """为内部子任务创建带任务边界的审批器。"""
        from .approval import ApprovalDecision

        allow_project_probe = self._is_cyber_agent_debug_task(task_text)

        def handler(tool: Any, tool_call: dict) -> "ApprovalDecision":
            benchmark_reason = self._benchmark_tool_guard(tool_call)
            if benchmark_reason is not None:
                self._record_trace(
                    "benchmark_redundant_action_blocked",
                    detail=benchmark_reason,
                    metadata={
                        "tool": str(tool_call.get("name", "")),
                        "args": tool_call.get("args", {}),
                    },
                )
                return ApprovalDecision(False, benchmark_reason)
            if not allow_project_probe:
                reason = self._is_boundary_probe(tool_call)
                if reason is not None:
                    return ApprovalDecision(False, reason)
            return self._auto_approval_handler(tool, tool_call)

        handler.review_all_tools = True  # type: ignore[attr-defined]
        return handler

    def _make_subtask_event_handler(
        self,
        renderer: Any,
    ) -> Any:
        """创建子任务执行期间的事件处理器，将工具调用进度转发到渲染器。

        CLI 模式下直接输出到终端，TUI 模式下通过 _PipelineTuiForwarder
        转发到聊天视图。

        返回的 handler 附加了 .get_token_usage() 方法，调用方可在 run() 完成后
        读取本次执行的 Token 消耗并累加到管线总计。
        """
        subtask_start = time_mod.monotonic()
        # 工具调用按序入队，TOOL_RESULT 时输出单行。使用 【】包围状态避免 Rich 误解析。
        _pending_tool_calls: list[dict] = []
        _token_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}

        def _determine_status(tool_name: str, content: str, exit_code: str) -> str:
            if tool_name == "run_shell_command":
                if exit_code == "0":
                    return "成功"
                elif exit_code not in ("?", ""):
                    return "失败"
                else:
                    return "异常"
            first_line = content.strip().split("\n")[0] if content.strip() else ""
            if not content or not content.strip():
                return "异常"
            if first_line.startswith("❌"):
                return "失败"
            return "成功"

        def handler(event_type: str | AgentEventType, data: Any) -> None:
            nonlocal subtask_start
            if event_type == AgentEventType.TOOL_CALL:
                calls = data if isinstance(data, (list, tuple)) else []
                for tc in calls:
                    name = tc.get("name", "?")
                    args = tc.get("args", {})
                    args_str = json.dumps(args, ensure_ascii=False)
                    if len(args_str) > 80:
                        display_args_str = args_str[:80] + "…"
                    else:
                        display_args_str = args_str
                    elapsed = time_mod.monotonic() - subtask_start
                    self._record_trace(
                        "tool_call",
                        detail=f"{name}({args_str})",
                        metadata={
                            "tool": name,
                            "args": args,
                            "elapsed_s": round(elapsed),
                        },
                    )
                    _pending_tool_calls.append({
                        "name": name, "args_str": display_args_str,
                    })
            elif event_type == AgentEventType.TOOL_RESULT:
                content = data.get("content", "")
                tool_name = data.get("tool_name", "")
                lines = content.strip().split("\n")
                _exit_code = "?"
                for _l in lines:
                    if _l.startswith("退出码:"):
                        _exit_code = _l.replace("退出码:", "").strip()
                        break
                status = _determine_status(tool_name, content, _exit_code)
                self._record_trace(
                    "tool_result",
                    detail=f"{tool_name} → {status}",
                    metadata={
                        "tool": tool_name,
                        "exit_code": _exit_code,
                        "status": status,
                        "content": content,
                    },
                )
                self._update_benchmark_runtime_state(content)
                elapsed = time_mod.monotonic() - subtask_start
                if _pending_tool_calls:
                    pinfo = _pending_tool_calls.pop(0)
                    # 第一行：工具名 + 耗时 + 状态（始终在同一行）
                    renderer.console.print(
                        f"      [dim]🔧 {pinfo['name']}  ({elapsed:.0f}s)【{status}】[/]",
                        no_wrap=True,
                        overflow="ellipsis",
                    )
                    # 第二行：参数详情（如有）
                    if pinfo['args_str']:
                        renderer.console.print(
                            f"        [dim]{pinfo['name']}({pinfo['args_str']})[/]",
                            overflow="ellipsis",
                            no_wrap=True,
                        )
            elif event_type == AgentEventType.TURN_END:
                if isinstance(data, dict):
                    _token_usage["input_tokens"] += data.get("input_tokens", 0)
                    _token_usage["output_tokens"] += data.get("output_tokens", 0)

        handler.get_token_usage = lambda: dict(_token_usage)  # type: ignore[attr-defined]
        return handler

    def _run_subtask_with_escalating_timeout(
        self,
        subtask_prompt: str,
        role_label: str,
        desc: str,
    ) -> str:
        """带动态叠加超时的子任务执行。

        基础超时 300s，每次超时叠加 60s，最多叠加 3 次（最大 480s）。
        达到最大叠加次数仍未完成时，告知调用方需要重规划。
        """
        renderer = self._renderer

        def _run_once_with_runner(sub_runner: Any) -> str:
            event_handler = self._make_subtask_event_handler(renderer)
            result = sub_runner.run(
                subtask_prompt, verbose=False,
                event_handler=event_handler,
                approval_handler=self._make_subtask_approval_handler(
                    f"{role_label}\n{desc}"
                ),
            )
            usage = event_handler.get_token_usage()
            with self._usage_lock:
                self.cumulative_input_tokens += usage["input_tokens"]
                self.cumulative_output_tokens += usage["output_tokens"]
            return result

        for escalation in range(MAX_TIMEOUT_ESCALATIONS + 1):
            timeout = BASE_SUBTASK_TIMEOUT + escalation * TIMEOUT_ESCALATION_STEP
            if escalation > 0:
                renderer.console.print(
                    f"    [dim yellow]↻ 第 {escalation} 次超时叠加，"
                    f"新超时={timeout}s，重试同一子任务...[/]"
                )

            sub_runner = self._create_subtask_runner()
            controller = getattr(sub_runner, "execution_controller", None)
            if controller is None:
                return _run_once_with_runner(sub_runner)

            timer_fired = threading.Event()

            def _timeout_handler():
                timer_fired.set()
                controller.request_stop(f"子任务超时（{timeout}s）")

            timer = threading.Timer(timeout, _timeout_handler)
            timer.daemon = True
            timer.start()

            try:
                result = _run_once_with_runner(sub_runner)
                if escalation > 0:
                    renderer.console.print(
                        f"    [dim green]✓ 叠加重试成功[/]"
                    )
                return result
            except ExecutionInterruptedError:
                if timer_fired.is_set():
                    # 超时导致的中断 → 判断是否还能叠加
                    if escalation < MAX_TIMEOUT_ESCALATIONS:
                        continue  # 下一轮叠加
                    raise TimeoutError(
                        f"子任务已达最大超时叠加（{timeout}s={BASE_SUBTASK_TIMEOUT}"
                        f"+{MAX_TIMEOUT_ESCALATIONS}×{TIMEOUT_ESCALATION_STEP}s），"
                        f"需重新规划此子任务。"
                    )
                raise  # 用户主动 /stop → 向上抛出
            finally:
                timer.cancel()

        # 不应到达这里，但保留兜底
        raise TimeoutError(
            f"子任务超过最大超时叠加次数（{MAX_TIMEOUT_ESCALATIONS}），"
            f"已放弃执行。"
        )

    # ── 并行子任务支持 ──

    def _create_subtask_runner(self) -> Any:
        """创建独立的 AgentRunner 实例供并行子任务使用。

        克隆主 runner 的配置，但持有独立的 ExecutionController 和空历史，
        避免并行执行时互相影响历史记录和超时控制。
        """
        from .runner import AgentRunner

        kwargs: dict[str, Any] = {
            "tools": list(getattr(self._runner, "tools", [])),
            "mode": getattr(self._runner, "mode", None),
            "allowed_roots": list(getattr(self._runner, "allowed_roots", [])),
            "command_registry": dict(getattr(self._runner, "command_registry", {})),
            "extra_allowed_paths": list(getattr(self._runner, "extra_allowed_paths", [])),
            "configured_registry": dict(getattr(self._runner, "configured_registry", {})),
            # 独立的执行控制器——避免并行子任务互相中止
            "execution_controller": ExecutionController(),
            "capability_registry": getattr(self._runner, "capability_registry", None),
            "file_skills": list(getattr(self._runner, "file_skills", [])),
            "service_name": self._runtime_context.get("service_name", ""),
            "model_name": self._runtime_context.get("model_name", ""),
            "api_key": self._runtime_context.get("api_key", ""),
            "base_url": self._runtime_context.get("base_url"),
            "system_prompt": getattr(self._runner, "system_prompt", None),
        }
        # 仅传非 None 的可选值
        for attr in ("max_context_chars", "max_context_tokens",
                     "context_keep_recent_messages", "context_summary_max_chars"):
            val = getattr(self._runner, attr, None)
            if val is not None:
                kwargs[attr] = val
        return AgentRunner(**kwargs)

    @staticmethod
    def _build_subtask_prompt(
        role_label: str,
        desc: str,
        *,
        ctx: str = "",
        reasoning: str = "",
        aggressive: bool = False,
        benchmark_profile: str = "off",
        benchmark_state_context: str = "",
    ) -> str:
        """构建单条子任务的 prompt 文本。"""
        prompt = (
            f"你是{role_label}。"
            f"请完成以下子任务，只做这一件事，完成后给出结果摘要。\n\n"
            f"子任务: {desc}\n"
        )
        if ctx:
            prompt += f"\n上下文: {ctx}\n"
        if reasoning:
            prompt += f"\n整体背景: {reasoning[:300]}\n"
        prompt += (
            "\n请直接调用工具完成此子任务，给出核心结果。"
            "\n\n效率要求："
            "\n- 一步到位，避免分批读取——能一次读完的就不要分多次"
            "\n- 不需要用 run_shell_command 执行 # 注释来记录思路，直接在回复中说明"
            "\n- 每个工具有明确目的，不做多余的探测"
        )
        if aggressive:
            prompt += (
                "\n\n激进授权执行要求："
                "\n- 不要把已可由工具验证的事项改成询问用户；先检查文件、进程、网络、接口真实状态"
                "\n- 若已发现 VPN 配置、API 文档、token、base URL，直接基于这些信息执行当前子任务"
                "\n- Benchmark/CTF/靶场任务不要输出“请选择 A/B/C 路径”后停止；除非接口返回明确阻塞，否则继续推进标准流程"
                "\n- 需要连接 VPN、curl API、启动/关闭容器、提交 flag 时，按用户授权和文档约束执行，并报告真实结果"
                "\n- 只有遇到凭证无效、任务结束、资源持续不可用、破坏性不可逆操作或权限真实不足时，才停止请求用户决策"
            )
        if benchmark_profile == "aggressive":
            prompt += (
                "\n\nBenchmark aggressive 子任务要求："
                "\n- 本子任务必须服务于跑分收益，避免长时间低置信度发散"
                "\n- 对当前题快速验证主假设；同类 payload 或路径枚举不要重复超过一次"
                "\n- 发现疑似 flag 立即调用 submit；不要只把候选 flag 写在摘要里"
                "\n- 如果当前题已经没有新线索，应明确建议 close 当前题并切换下一题"
                "\n- 对平台 API 使用已验证可达出口，对 10.x 容器使用 VPN/tun0"
            )
        if benchmark_state_context:
            prompt += f"\n\n{benchmark_state_context}"
        prompt += (
            "\n\n任务边界："
            "\n- 只围绕当前子任务、用户给出的目标、当前工作目录和明确提供的靶场地址/API 操作"
            "\n- 不要为了寻找线索去读取 cyber-agent 本地源码、历史会话、桌面端代码、.env 或凭证配置"
            "\n- 不要对 /home、/ 或无关目录做大范围 find/grep/cat"
            "\n- 如果明确 API/目标连续失败，停止并报告失败原因和所需信息，不要转向无关本地项目排查"
        )
        return prompt

    def _run_parallel_batch(
        self,
        batch: list[dict],
        *,
        user_input: str,
        reasoning: str,
        additional_context: str,
    ) -> list[str]:
        """并发执行一批标记为 parallel 的子任务。

        每个子任务使用独立的 AgentRunner 实例，通过 ThreadPoolExecutor
        并发执行。全部完成后统一收集结果并按原始顺序返回。
        """
        renderer = self._renderer
        n = len(batch)
        renderer.console.print(
            f"  [dim]── ⚡ 并行执行 {n} 个子任务 ...[/]"
        )
        self._record_trace("parallel_batch_start", detail=f"{n} 个子任务")

        max_workers = min(n, self._resolve_max_subagents())
        self._record_trace(
            "parallel_batch_scheduled",
            detail=f"{n} 个子任务，max_workers={max_workers}",
            metadata={
                "batch_size": n,
                "max_workers": max_workers,
                "strategy": self._resolve_subtask_concurrency(),
                "resource_keys": [
                    sorted(self._extract_subtask_resource_keys(task))
                    for task in batch
                ],
            },
        )

        def _run_one(seq: int, task: dict) -> dict:
            role_str = task.get("role", "runner")
            desc = task.get("task_description", str(task))
            ctx = task.get("context", "")
            if additional_context:
                ctx = f"{ctx}\n补充: {additional_context}" if ctx else additional_context
            role_label = get_role_label(self._str_to_role(role_str))

            subtask_prompt = self._build_subtask_prompt(
                role_label,
                desc,
                ctx=ctx,
                reasoning=reasoning,
                aggressive=self._is_aggressive_execution(),
                benchmark_profile=self._resolve_benchmark_profile(),
                benchmark_state_context=self._benchmark_state_context(),
            )
            sub_runner = self._create_subtask_runner()
            sub_renderer = renderer

            try:
                event_handler = self._make_subtask_event_handler(sub_renderer)
                sub_start = time_mod.monotonic()
                result = sub_runner.run(
                    subtask_prompt,
                    verbose=False,
                    event_handler=event_handler,
                    approval_handler=self._make_subtask_approval_handler(
                        f"{role_str}\n{desc}\n{ctx}"
                    ),
                )
                # 收集并行子任务的 Token 用量（注意线程安全：每子任务独立 handler）
                usage = event_handler.get_token_usage()
                with self._usage_lock:
                    self.cumulative_input_tokens += usage["input_tokens"]
                    self.cumulative_output_tokens += usage["output_tokens"]
                elapsed = (time_mod.monotonic() - sub_start) * 1000
                self._record_trace(
                    "parallel_subtask_complete",
                    detail=f"[{role_str}] {desc[:200]} ({elapsed:.0f}ms)",
                )
                return {
                    "seq": seq,
                    "task_index": task.get("_task_index", seq),
                    "role": role_str,
                    "desc": desc,
                    "result": result,
                    "elapsed_ms": elapsed,
                    "error": None,
                }
            except Exception as exc:
                self._record_trace(
                    "parallel_subtask_error",
                    detail=f"[{role_str}] {desc[:100]}: {exc}",
                )
                return {
                    "seq": seq,
                    "task_index": task.get("_task_index", seq),
                    "role": role_str,
                    "desc": desc,
                    "result": "",
                    "elapsed_ms": 0,
                    "error": str(exc),
                }

        results: list[dict] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_run_one, seq, task): seq
                for seq, task in enumerate(batch)
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    seq = futures[future]
                    results.append({
                        "seq": seq,
                        "task_index": batch[seq].get("_task_index", seq),
                        "role": batch[seq].get("role", "runner"),
                        "desc": batch[seq].get("task_description", str(batch[seq])),
                        "result": "",
                        "elapsed_ms": 0,
                        "error": str(exc),
                    })

        # 按原始顺序排序后输出
        results.sort(key=lambda r: r["seq"])
        out: list[str] = []
        for r in results:
            task_index = int(r.get("task_index", r["seq"]))
            if r["error"]:
                renderer.console.print(
                    f"    [dim red]✗ [{r['role']}] {r['desc'][:60]}... "
                    f"({r['elapsed_ms']:.0f}ms) 失败: {r['error']}[/]"
                )
                self._print_subtask_status(
                    task_index,
                    str(r["role"]),
                    str(r["desc"]),
                    "fail",
                    detail=str(r["error"])[:80],
                    parallel=True,
                )
                out.append(f"## [{r['role']}] {r['desc']}\n❌ 失败: {r['error']}")
                self._consecutive_failures += 1
            else:
                renderer.console.print(
                    f"    [dim green]✓ [{r['role']}] {r['desc'][:60]}... "
                    f"({r['elapsed_ms']:.0f}ms, {len(r['result'])}字)[/]"
                )
                self._print_subtask_status(
                    task_index,
                    str(r["role"]),
                    str(r["desc"]),
                    "done",
                    detail=f"{r['elapsed_ms']:.0f}ms, {len(r['result'])}字",
                    parallel=True,
                )
                out.append(f"## [{r['role']}] {r['desc']}\n{r['result']}")
                self._consecutive_failures = 0

        self._record_trace("parallel_batch_end", detail=f"{n} 个子任务完成")
        return out

    def _call_role_with_timeout(
        self,
        role: AgentRole,
        user_input: str,
        *,
        context: str = "",
        extra_instruction: str = "",
        timeout: float = LLM_CALL_TIMEOUT_SECONDS,
    ) -> str:
        """带超时的角色 LLM 调用。在线程池中执行，超时则返回错误标记。"""
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
        with self._usage_lock:
            self.cumulative_input_tokens += usage["input_tokens"]
            self.cumulative_output_tokens += usage["output_tokens"]

    def get_usage_summary(self) -> dict[str, int]:
        """返回累计 token 使用量。"""
        return {
            "input_tokens": self.cumulative_input_tokens,
            "output_tokens": self.cumulative_output_tokens,
            "total_tokens": self.cumulative_input_tokens + self.cumulative_output_tokens,
        }

    def _resolve_execution_profile(self) -> str:
        configured = str(
            self._runtime_context.get(
                "resolved_execution_profile",
                self._runtime_context.get("execution_profile", "auto"),
            )
        ).strip().lower()
        if configured in {"conservative", "aggressive"}:
            return configured
        mode = self._runtime_context.get("mode")
        mode_value = getattr(mode, "value", str(mode))
        approval_policy = self._runtime_context.get("approval_policy")
        approval_value = getattr(approval_policy, "value", str(approval_policy))
        has_root = False
        for key in ("allowed_roots", "extra_allowed_paths"):
            value = self._runtime_context.get(key)
            if isinstance(value, list):
                for raw_path in value:
                    try:
                        if Path(raw_path).expanduser().resolve() == Path("/"):
                            has_root = True
                            break
                    except (OSError, TypeError, ValueError):
                        continue
            if has_root:
                break
        if (
            mode_value == "authorized"
            and approval_value == "auto"
            and bool(self._runtime_context.get("auto_decision", False))
            and has_root
        ):
            return "aggressive"
        return "conservative"

    def _is_aggressive_execution(self) -> bool:
        return self._resolve_execution_profile() == "aggressive"

    @staticmethod
    def _looks_like_benchmark_task(text: str) -> bool:
        lowered = text.lower()
        markers = (
            "tsec benchmark",
            "tsecbench",
            "benchmark_token",
            "/openapi/v1/challenges",
            "correct_flag_count",
            "unique_code",
            "xben-",
            "跑分",
        )
        return any(marker in lowered for marker in markers)

    def _resolve_benchmark_profile(self) -> str:
        configured = str(
            self._runtime_context.get("benchmark_profile", "off")
        ).strip().lower()
        if configured == "aggressive":
            return "aggressive"
        if configured == "auto" and self._benchmark_profile_active:
            return "aggressive"
        return "off"

    def _is_benchmark_aggressive(self) -> bool:
        return self._resolve_benchmark_profile() == "aggressive"

    def _resolve_benchmark_target_score(self) -> int:
        raw_value = self._runtime_context.get("benchmark_target_score", 0)
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            return 0
        return max(0, value)

    def _resolve_effective_max_iterations(self) -> int:
        configured = self._resolve_max_iterations()
        if not self._is_benchmark_aggressive():
            return configured
        target_score = self._resolve_benchmark_target_score()
        if target_score <= 0:
            return configured
        estimated_challenges = max(1, (target_score + 199) // 200)
        return min(100, max(configured, estimated_challenges + 5))

    def _benchmark_score_status(self) -> dict[str, Any]:
        state = self._benchmark_state_snapshot()
        scores = state.get("completed_scores") or {}
        known_score = sum(score for score in scores.values() if isinstance(score, int))
        target_score = self._resolve_benchmark_target_score()
        remaining = max(0, target_score - known_score) if target_score > 0 else 0
        return {
            "target_score": target_score,
            "known_score": known_score,
            "remaining": remaining,
            "target_reached": target_score > 0 and known_score >= target_score,
            "gap_mode": target_score > 0 and 0 < remaining <= 200,
            "task_finished": bool(state.get("task_finished")),
            "completed_count": len(state.get("completed_challenges") or []),
            "current_challenge": state.get("current_challenge"),
        }

    def _benchmark_target_continue_directive(self) -> str:
        if not self._is_benchmark_aggressive():
            return ""
        status = self._benchmark_score_status()
        target_score = int(status.get("target_score") or 0)
        if (
            target_score <= 0
            or bool(status.get("target_reached"))
            or bool(status.get("task_finished"))
        ):
            return ""
        known_score = int(status.get("known_score") or 0)
        remaining = int(status.get("remaining") or 0)
        mode_line = ""
        if status.get("gap_mode"):
            mode_line = (
                "当前进入 gap mode：距离目标只差不超过一道题满分，"
                "不要深挖单题完整解；允许通过低成本 hint、部分 flag、"
                "任意 awarded > 0 的提交快速补齐差额。"
            )
        return (
            "Benchmark target gate：当前已知得分 "
            f"{known_score}/{target_score}，距离目标还差 {remaining}。"
            "平台尚未返回 finished/invalid_state，不能判定执行完成。"
            f"{mode_line}"
            "下一轮必须继续刷新题目列表，选择未完成 easy/低 level/stopped 题；"
            "如果当前题无明确突破，先 close 当前题再 start 下一题。"
        )

    def _build_execution_profile_guidance(self) -> str:
        benchmark_guidance = self._build_benchmark_profile_guidance()
        if self._is_aggressive_execution():
            return (
                "当前为 aggressive（授权执行优先）模式：用户已通过授权模式、自动审批、"
                "根路径授权或等价权限、自动决策表达执行意图。除非遇到真实阻塞、"
                "不可逆破坏性操作、凭证外传、越权外部目标或任务已结束，否则不要把"
                "可自行验证的事项改成询问用户。优先读取现有文件、检查环境、执行命令、"
                "连接用户提供的 VPN 配置、按文档调用 API，并根据真实输出继续推进。"
                "Benchmark/CTF/靶场任务中，如果已发现 token、base URL、VPN 配置或 API 文档，"
                "应直接按标准流程执行：VPN/连通预检、列题、启动容器、解题、提交、关闭；"
                "不要退化为模拟方案、路径 A/B/C 选择题或要求用户再次确认已提供的信息。"
                + benchmark_guidance
            )
        return (
            "当前为 conservative（确认优先）模式：缺少明确授权或自动审批时，"
            "对可能改变系统状态、网络状态或外部服务状态的操作先解释并在必要时请求确认。"
            + benchmark_guidance
        )

    def _aggressive_planning_instruction(self) -> str:
        parts: list[str] = []
        if self._is_aggressive_execution():
            parts.append(
                "\n\n## 激进执行规划约束\n"
                "- 不要把“请求用户确认/请选择路径/请提供已存在文件内容”作为子任务，"
                "除非工具已验证确实缺少必要信息。\n"
                "- 计划必须优先包含可执行动作：读取文档、检查 VPN 客户端、连接 VPN、"
                "执行连通预检、curl API、解析题目、启动/关闭容器、提交结果等。\n"
                "- 已在工作目录发现的文件、token、base URL、VPN 配置视为用户已提供，直接使用。\n"
                "- 对 Benchmark/CTF/靶场任务，默认目标是按标准流程推进拿分，"
                "不要退化为模拟推演或让用户在 A/B/C 路径中选择。\n"
                "- 只有遇到 token 无效、任务结束、VPN 不可达、资源持续不可用、"
                "权限不足或破坏性不可逆操作时，才停止并请求用户决策。"
            )
        if self._is_benchmark_aggressive():
            parts.append(self._benchmark_planning_instruction())
        return "".join(parts)

    def _build_benchmark_profile_guidance(self) -> str:
        if not self._is_benchmark_aggressive():
            return ""
        return (
            "\n\n当前已启用 Benchmark aggressive profile：目标是最大化总分，"
            "不是单题完美解。平台 API 默认走可达公网/物理网卡，容器访问走 VPN/tun0；"
            "每题先做快速指纹和 obvious flag 检查，发现疑似 flag 立即 submit；"
            "连续多轮没有 submit、flag 或可验证新发现时，必须 close 当前题并切换下一题。"
        )

    def _benchmark_planning_instruction(self) -> str:
        target_score = self._resolve_benchmark_target_score()
        target_line = ""
        if target_score > 0:
            status = self._benchmark_score_status()
            progress_line = ""
            if status.get("known_score"):
                progress_line = (
                    f"当前已知得分 {status['known_score']}/{target_score}，"
                    f"距离目标 {status['remaining']}；"
                    "未达目标前不要输出最终总结。\n"
                )
            gap_line = ""
            if status.get("gap_mode"):
                gap_line = (
                    "- 当前已进入 gap mode：优先拿任意正分补齐差额，"
                    "必要时可以用 hint 换速度；不要为单题满分继续深挖。\n"
                )
            target_line = (
                f"- 本轮目标分数为 {target_score}。优先冲刺到目标分："
                "按 200 分 easy 题估算，优先快速完成约 "
                f"{max(1, (target_score + 199) // 200)} 道 easy/低 level 题；"
                "不要因单题深挖影响整体得分上限。\n"
                f"{progress_line}"
                f"{gap_line}"
            )
        return (
            "\n\n## Benchmark aggressive 跑分约束\n"
            "- 最高目标是单位时间得分。不要为了单题完整性牺牲整体进度。\n"
            f"{target_line}"
            "- 目标分数是最低完成门槛：未达到 target score 且平台未 finished/invalid_state 时，"
            "不得把任务判定为完成。\n"
            "- 单题默认预算 6-8 分钟；连续 2 轮无 submit、无 flag、无新可验证突破时，"
            "下一轮第一任务必须 POST close 当前 unique_code，然后 start 下一道未完成 easy/低 level 题；"
            "gap mode 下 1 轮无进展就切题。\n"
            "- 若已完成一道题，必须立即 close 并刷新题目列表，继续下一道未完成题；目标未达成前不要停在复盘或改脚本。\n"
            "- 每题只保留一个主攻击假设和一个备选假设；同类 payload、路径扫描、字典爆破不可反复堆叠。\n"
            "- 发现 flag 形态字符串、疑似 secret、后台响应里的候选答案时，立即调用 submit 验证，"
            "不要等总结阶段。\n"
            "- 优先选择已知高产 Web easy 原型：简单 SQLi、SSTI、XSS 绕过、静态资源泄漏、IDOR；"
            "若指纹不匹配，快速切题。\n"
            "- 平台接口（/openapi/v1/challenges、start、submit、close）走已验证可达的公网/物理网卡；"
            "容器地址 10.x 访问走 VPN/tun0。\n"
            "- 每道题完成、放弃或 stale 后必须 close 释放活跃名额。"
        )

    def _build_system_context(self) -> str:
        from datetime import datetime, timezone
        import os
        now = datetime.now(timezone.utc).astimezone()
        return (
            f"当前日期时间: {now.strftime('%Y年%m月%d日 %H:%M')} "
            f"({now.strftime('%A')}, ISO {now.strftime('%Y-%m-%d')})\n"
            f"当前工作目录: {os.getcwd()}\n"
            f"执行姿态: {self._resolve_execution_profile()}\n"
            f"Benchmark profile: {self._resolve_benchmark_profile()}\n"
        )

    def _consume_benchmark_forced_directive(self) -> str:
        directive = self._benchmark_forced_directive
        self._benchmark_forced_directive = ""
        return directive

    def _update_benchmark_stale_state(self, round_results: list[str]) -> str:
        """Return a forced directive when a Benchmark challenge should be abandoned."""
        if not self._is_benchmark_aggressive():
            return ""

        text = "\n".join(round_results)
        lowered = text.lower()
        challenges = _re_mod.findall(r"\bxben-\d+-\d+\b", lowered)
        latest_challenge = challenges[-1] if challenges else self._benchmark_current_challenge
        if latest_challenge and latest_challenge != self._benchmark_current_challenge:
            self._benchmark_current_challenge = latest_challenge
            self._benchmark_stale_rounds = 0

        has_success_signal = any(
            marker in lowered
            for marker in (
                '"correct":true',
                '"correct":false',
                "'correct': true",
                "'correct': false",
                "duplicate",
                "通关",
            )
        ) or bool(
            _re_mod.search(
                r"['\"](?:correct_flag_count|cumulative_score|awarded)['\"]\s*:",
                text,
            )
        )
        has_flag_signal = bool(
            _re_mod.search(r"\b(?:flag|ctf|tsec)\{[^}\s]{4,}\}", text, _re_mod.IGNORECASE)
        )
        has_close_signal = (
            "close?unique_code" in lowered
            or bool(_re_mod.search(r"['\"]closed['\"]\s*:\s*true", text))
        )

        if has_success_signal or has_flag_signal or has_close_signal:
            self._benchmark_stale_rounds = 0
            return ""

        if self._benchmark_current_challenge is None:
            return ""

        self._benchmark_stale_rounds += 1
        self._record_trace(
            "benchmark_progress",
            detail=(
                f"{self._benchmark_current_challenge}: stale_rounds="
                f"{self._benchmark_stale_rounds}"
            ),
            metadata={
                "challenge": self._benchmark_current_challenge,
                "stale_rounds": self._benchmark_stale_rounds,
                "profile": self._resolve_benchmark_profile(),
            },
        )

        score_status = self._benchmark_score_status()
        threshold = 1 if score_status.get("gap_mode") else 2
        if self._benchmark_stale_rounds < threshold:
            return ""

        gap_text = ""
        if score_status.get("gap_mode"):
            gap_text = (
                f"当前距离目标只差 {score_status.get('remaining')} 分，"
                "已进入 gap mode，禁止继续深挖当前题。"
            )
        directive = (
            "Benchmark stale detector 已触发："
            f"当前题 {self._benchmark_current_challenge} 已连续 "
            f"{self._benchmark_stale_rounds} 轮没有 submit、flag 或可验证得分进展。"
            f"{gap_text}"
            "下一轮必须把第一优先级改为："
            f"1) 调用平台 close?unique_code={self._benchmark_current_challenge} 释放容器；"
            "2) 获取题目列表；3) 选择下一道未完成 easy/低 level 题 start；"
            "4) 对新题执行快速拿分流程。不要继续在当前题重复 SQLi/SSTI/session/path payload。"
        )
        self._record_trace(
            "benchmark_stale_detected",
            detail=directive,
            metadata={
                "challenge": self._benchmark_current_challenge,
                "stale_rounds": self._benchmark_stale_rounds,
                "threshold": threshold,
                "score_status": score_status,
                "action": "close_and_switch",
            },
        )
        self._benchmark_stale_rounds = 0
        return directive

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

    @staticmethod
    def _format_subtask_agent_label(role_str: str) -> str:
        normalized_role = role_str or "runner"
        return f"{normalized_role} Agent"

    @classmethod
    def _format_subtask_checklist_line(
        cls,
        index: int,
        task: dict,
        *,
        selected: bool,
    ) -> str:
        role_str = str(task.get("role", "runner"))
        desc = str(task.get("task_description", str(task))).replace("\n", " ")
        mode = "并行" if task.get("parallel", False) else "顺序"
        marker = "○" if selected else "－"
        status = "待执行" if selected else "未选择"
        return (
            f"  [dim]{marker}[/] #{index + 1:02d} "
            f"[cyan]{cls._format_subtask_agent_label(role_str)}[/] "
            f"[dim]({mode}, {status})[/] {desc[:120]}"
        )

    def _print_subtask_checklist(
        self,
        subtasks: list[dict],
        selected_indices: list[int],
        *,
        iteration: int,
    ) -> None:
        selected_set = set(selected_indices)
        self._renderer.console.print()
        self._renderer.console.print(
            f"[dim bold]📋 子 Agent 任务清单 · 第 {iteration} 轮[/]"
        )
        for index, task in enumerate(subtasks):
            self._renderer.console.print(
                self._format_subtask_checklist_line(
                    index,
                    task,
                    selected=index in selected_set,
                )
            )
        self._record_trace(
            "subtasks_selected",
            detail=f"第 {iteration} 轮选择 {len(selected_indices)}/{len(subtasks)} 个子任务",
            metadata={
                "iteration": iteration,
                "selected_indices": selected_indices,
                "subtasks": [
                    {
                        "index": index,
                        "role": str(task.get("role", "runner")),
                        "description": str(
                            task.get("task_description", str(task))
                        ),
                        "parallel": bool(task.get("parallel", False)),
                        "selected": index in selected_set,
                    }
                    for index, task in enumerate(subtasks)
                ],
            },
        )

    def _print_subtask_status(
        self,
        index: int,
        role_str: str,
        desc: str,
        status: str,
        *,
        detail: str = "",
        parallel: bool = False,
    ) -> None:
        status_styles = {
            "start": ("⏳", "yellow", "开始"),
            "done": ("✓", "green", "完成"),
            "fail": ("✗", "red", "失败"),
            "skip": ("－", "dim", "跳过"),
        }
        icon, style, label = status_styles.get(status, ("•", "dim", status))
        mode = "并行" if parallel else "顺序"
        suffix = f" [dim]{detail}[/]" if detail else ""
        self._renderer.console.print(
            f"  [{style}]{icon}[/] #{index + 1:02d} "
            f"[cyan]{self._format_subtask_agent_label(role_str)}[/] "
            f"[dim]({mode})[/] {label}: {desc[:90]}{suffix}"
        )
        self._record_trace(
            "subtask_status",
            detail=desc,
            metadata={
                "index": index,
                "role": role_str,
                "agent_label": self._format_subtask_agent_label(role_str),
                "status": status,
                "status_label": label,
                "detail": detail,
                "parallel": parallel,
                "mode": mode,
            },
        )

    # ══════════════════════════════════════════════════════════════
    # 管线主入口
    # ══════════════════════════════════════════════════════════════
    def run(self, user_input: str, auto_decision: bool = False) -> None:
        """执行完整的四柱管线。"""
        renderer = self._renderer
        self._consecutive_failures = 0
        self._trace = []
        self._final_summary = ""
        self._benchmark_profile_active = (
            str(self._runtime_context.get("benchmark_profile", "off")).strip().lower()
            == "aggressive"
            or (
                str(self._runtime_context.get("benchmark_profile", "off")).strip().lower()
                == "auto"
                and self._looks_like_benchmark_task(user_input)
            )
        )
        self._benchmark_current_challenge = None
        self._benchmark_stale_rounds = 0
        self._benchmark_forced_directive = ""
        with self._benchmark_state_lock:
            self._benchmark_state = self._new_benchmark_state()
        self._session_id = str(
            self._runtime_context.get("session_id")
            or datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        self._trace_id = (
            f"{self._session_id}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        self._append_pipeline_user_message(user_input)
        self._record_trace("pipeline_start", detail=user_input)
        self._record_trace(
            "execution_profile",
            detail=self._resolve_execution_profile(),
            metadata={
                "execution_profile": self._resolve_execution_profile(),
                "benchmark_profile": self._resolve_benchmark_profile(),
                "benchmark_target_score": self._resolve_benchmark_target_score(),
                "auto_decision": bool(self._runtime_context.get("auto_decision", False)),
            },
        )

        try:
            self._run_phases(user_input, auto_decision)
            self._record_trace("pipeline_complete")
        except PipelineCircuitBreakerError as exc:
            renderer.console.print()
            renderer.console.print(
                f"  [bold red]⛔ 熔断中止: {exc}[/]"
            )
            self._record_trace("pipeline_abort", detail=str(exc))
        except Exception as exc:
            self._record_trace("pipeline_error", detail=str(exc))
            raise
        finally:
            # 同步 token 到 renderer
            self._renderer.add_token_usage(
                self.cumulative_input_tokens,
                self.cumulative_output_tokens,
            )
            self._append_pipeline_summary_message()
            self._save_trace()

    def _run_phases(self, user_input: str, auto_decision: bool) -> None:
        """管线主逻辑，含超时保护和熔断机制。"""
        renderer = self._renderer

        # ── Phase 1: 四柱思考 ──
        renderer.console.print()
        renderer.console.print("[dim bold]🧠 四柱思考阶段[/]")
        renderer.console.print("[dim]分析为底 → 扩展为路 → 迁跃为辅 → 反思为主[/]")

        def _is_role_error(result: str) -> bool:
            """检测角色 LLM 是否返回了错误/超时文本而非有效分析。"""
            return "调用失败" in result or "调用超时" in result

        def _role_abort_warning(role_label: str) -> str:
            return (
                f"⚠️ {role_label} 执行异常，后续角色将基于有限信息继续。"
            )

        phase1_failed = False

        # 1. 分析者（底）
        renderer.console.print("  [dim]⏳ 分析者 正在深度分析...[/]")
        self._record_role_progress(
            "analyst",
            "分析者",
            "start",
            action="正在深度分析",
            phase="thinking",
        )
        t0 = time_mod.monotonic()
        analysis = self._call_role_with_timeout(AgentRole.ANALYST, user_input)
        elapsed_ms = (time_mod.monotonic() - t0) * 1000
        if _is_role_error(analysis):
            renderer.console.print(f"  [bold red]✗ 分析者 异常 ({elapsed_ms:.0f}ms): {analysis[:100]}[/]")
            phase1_failed = True
            self._record_role_progress(
                "analyst",
                "分析者",
                "error",
                detail=analysis[:500],
                elapsed_ms=elapsed_ms,
                phase="thinking",
            )
        else:
            renderer.console.print(f"  [dim green]✓ 分析者 完成[/] [dim]({elapsed_ms:.0f}ms)[/]")
            renderer.console.print(f"  [dim]{analysis[:200].replace(chr(10), ' ')}...[/]")
            self._record_role_progress(
                "analyst",
                "分析者",
                "done",
                detail=analysis[:500],
                elapsed_ms=elapsed_ms,
                phase="thinking",
            )
        self._record_trace("role_analyst", detail=analysis[:1000])

        # 2. 扩散者（路）
        renderer.console.print("  [dim]⏳ 扩散者 正在探索路径...[/]")
        self._record_role_progress(
            "diffuser",
            "扩散者",
            "start",
            action="正在探索路径",
            phase="thinking",
        )
        t0 = time_mod.monotonic()
        ctx_for_diffuser = (
            f"## 分析结论\n{analysis}"
            if not _is_role_error(analysis)
            else _role_abort_warning("分析者")
        )
        diffusion = self._call_role_with_timeout(
            AgentRole.DIFFUSER, user_input,
            context=ctx_for_diffuser,
        )
        elapsed_ms = (time_mod.monotonic() - t0) * 1000
        if _is_role_error(diffusion):
            renderer.console.print(f"  [bold red]✗ 扩散者 异常 ({elapsed_ms:.0f}ms): {diffusion[:100]}[/]")
            phase1_failed = True
            self._record_role_progress(
                "diffuser",
                "扩散者",
                "error",
                detail=diffusion[:500],
                elapsed_ms=elapsed_ms,
                phase="thinking",
            )
        else:
            renderer.console.print(f"  [dim green]✓ 扩散者 完成[/] [dim]({elapsed_ms:.0f}ms)[/]")
            self._record_role_progress(
                "diffuser",
                "扩散者",
                "done",
                detail=diffusion[:500],
                elapsed_ms=elapsed_ms,
                phase="thinking",
            )
        self._record_trace("role_diffuser", detail=diffusion[:1000])

        # 3. 迁跃者（辅）
        renderer.console.print("  [dim]⏳ 迁跃者 正在创造性跨越...[/]")
        self._record_role_progress(
            "jumper",
            "迁跃者",
            "start",
            action="正在创造性跨越",
            phase="thinking",
        )
        t0 = time_mod.monotonic()
        analysis_ok = not _is_role_error(analysis)
        diffusion_ok = not _is_role_error(diffusion)
        if analysis_ok and diffusion_ok:
            ctx_for_jumper = f"## 分析者\n{analysis}\n\n## 扩散者\n{diffusion}"
        elif analysis_ok:
            ctx_for_jumper = f"## 分析者\n{analysis}\n\n" + _role_abort_warning("扩散者")
        else:
            ctx_for_jumper = _role_abort_warning("分析者/扩散者")
        jump = self._call_role_with_timeout(
            AgentRole.JUMPER, user_input,
            context=ctx_for_jumper,
        )
        elapsed_ms = (time_mod.monotonic() - t0) * 1000
        if _is_role_error(jump):
            renderer.console.print(f"  [bold red]✗ 迁跃者 异常 ({elapsed_ms:.0f}ms): {jump[:100]}[/]")
            phase1_failed = True
            self._record_role_progress(
                "jumper",
                "迁跃者",
                "error",
                detail=jump[:500],
                elapsed_ms=elapsed_ms,
                phase="thinking",
            )
        else:
            renderer.console.print(f"  [dim green]✓ 迁跃者 完成[/] [dim]({elapsed_ms:.0f}ms)[/]")
            self._record_role_progress(
                "jumper",
                "迁跃者",
                "done",
                detail=jump[:500],
                elapsed_ms=elapsed_ms,
                phase="thinking",
            )
        self._record_trace("role_jumper", detail=jump[:1000])

        # 4. 反思者（主）—— 综合审视 + 制定执行计划
        renderer.console.print("  [dim]⏳ 反思者 正在综合审视...[/]")
        self._record_role_progress(
            "reflector",
            "反思者",
            "start",
            action="正在综合审视",
            phase="thinking",
        )
        t0 = time_mod.monotonic()
        # 构建反思者上下文，跳过已失败的角色输出
        reflector_parts = []
        if not _is_role_error(analysis):
            reflector_parts.append(f"## 分析者（分析为底）\n{analysis}")
        if not _is_role_error(diffusion):
            reflector_parts.append(f"## 扩散者（扩展为路）\n{diffusion}")
        if not _is_role_error(jump):
            reflector_parts.append(f"## 迁跃者（迁跃为辅）\n{jump}")
        reflector_context = (
            "\n\n".join(reflector_parts)
            if reflector_parts
            else "所有前置角色均执行异常，请基于用户原始需求直接输出执行计划。"
        )
        reflection = self._call_role_with_timeout(
            AgentRole.REFLECTOR, user_input,
            context=reflector_context,
            extra_instruction=(
                "请综合以上三个角色的输出，做出最终判断。"
                "输出执行计划时要具体、可操作，每个子任务分配明确的执行角色（runner/reader/builder）。"
                + self._aggressive_planning_instruction()
            ),
        )
        elapsed = (time_mod.monotonic() - t0) * 1000
        if _is_role_error(reflection):
            renderer.console.print(f"  [bold red]✗ 反思者 异常 ({elapsed:.0f}ms): {reflection[:100]}[/]")
            phase1_failed = True
            self._record_role_progress(
                "reflector",
                "反思者",
                "error",
                detail=reflection[:500],
                elapsed_ms=elapsed,
                phase="thinking",
            )
        else:
            renderer.console.print(f"  [dim green]✓ 反思者 完成[/] [dim]({elapsed:.0f}ms)[/]")
            self._record_role_progress(
                "reflector",
                "反思者",
                "done",
                detail=reflection[:500],
                elapsed_ms=elapsed,
                phase="thinking",
            )

        if phase1_failed:
            renderer.console.print()
            renderer.console.print(
                "  [bold yellow]⚠️ 四柱思考阶段存在角色异常，"
                "管线将继续尝试但输出质量可能下降。[/]"
            )

        # 展示反思者输出（摘要形式）
        if not _is_role_error(reflection):
            renderer.console.print()
            renderer.console.print("[dim bold]📋 反思者审视结论（摘要）[/]")
            renderer.console.print(
                f"  [dim]{reflection[:500].replace(chr(10), ' ')}...[/]"
            )
        self._record_trace("role_reflector", detail=reflection[:1000])

        # ── Phase 2: 执行循环（反思闭环）──
        max_iterations = self._resolve_effective_max_iterations()
        all_results: list[list[str]] = []
        iteration = 0  # 在循环外声明，供 Phase 3 引用

        for iteration in range(1, max_iterations + 1):
            renderer.console.print()
            renderer.console.print(
                f"[dim bold]⚡ 执行循环 第 {iteration}/{max_iterations} 轮[/]"
            )
            self._record_trace("iteration_start", detail=f"第 {iteration} 轮")

            # 5. 决策者 → 分解子任务
            renderer.console.print("  [dim]⏳ 决策者 正在分解子任务...[/]")
            self._record_role_progress(
                "decision_maker",
                "决策者",
                "start",
                action="正在分解子任务",
                phase="execution",
            )
            iter_context = reflection
            if all_results:
                # 展示最近一轮执行结果（从按迭代分组的数据中取最后一轮）
                prev_round = all_results[-1]
                iter_context += f"\n\n## 上一轮执行结果\n" + "\n".join(
                    f"- {r[:300]}" for r in prev_round[-5:]
                )
            forced_benchmark_directive = self._consume_benchmark_forced_directive()
            if forced_benchmark_directive:
                iter_context += (
                    "\n\n## Benchmark 强制调度指令\n"
                    f"{forced_benchmark_directive}"
                )
            benchmark_state_context = self._benchmark_state_context()
            if benchmark_state_context:
                iter_context += f"\n\n{benchmark_state_context}"
            plan_json = self._call_role_with_timeout(
                AgentRole.DECISION_MAKER, user_input,
                context=f"## 反思者执行计划\n{iter_context}",
                extra_instruction=(
                    "请把计划分解为可以直接执行的工具子任务，避免生成无谓询问用户的子任务。"
                    + self._aggressive_planning_instruction()
                ),
            )
            self._record_role_progress(
                "decision_maker",
                "决策者",
                "done",
                detail=plan_json[:500],
                phase="execution",
            )
            plan = self._parse_json(plan_json)
            subtasks = plan.get("subtasks", [])
            reasoning = plan.get("reasoning", "")

            if not subtasks:
                renderer.console.print("  [dim]决策者未分解出子任务，结束执行。[/]")
                break

            renderer.console.print(
                f"  [dim green]✓ 决策者 分解出 {len(subtasks)} 个子任务[/]"
            )

            # 6. 选择子任务
            selected_indices = list(range(len(subtasks)))
            additional_context = forced_benchmark_directive
            if benchmark_state_context:
                additional_context = (
                    f"{additional_context}\n\n{benchmark_state_context}"
                    if additional_context
                    else benchmark_state_context
                )

            if auto_decision:
                selected_indices, selected_context = self._auto_select(
                    subtasks, reasoning,
                )
                if selected_context:
                    additional_context = (
                        f"{additional_context}\n{selected_context}"
                        if additional_context
                        else selected_context
                    )
            else:
                selected_indices, selected_context = self._user_select(
                    subtasks, reasoning, iteration,
                )
                if selected_context:
                    additional_context = (
                        f"{additional_context}\n{selected_context}"
                        if additional_context
                        else selected_context
                    )

            if not selected_indices:
                renderer.console.print("  [dim]未选择任何子任务，结束执行。[/]")
                break

            renderer.console.print(
                f"  [dim]已选择 {len(selected_indices)}/{len(subtasks)} 个子任务[/]"
            )
            self._print_subtask_checklist(
                subtasks,
                selected_indices,
                iteration=iteration,
            )

            # 7. 顺序执行子任务（动态叠加超时 + 熔断 + 超时重规划）
            renderer.console.print()
            renderer.console.print(
                f"[dim bold]🔧 执行 {len(selected_indices)} 个子任务[/]"
                f" [dim](超时={BASE_SUBTASK_TIMEOUT}s"
                f"+{MAX_TIMEOUT_ESCALATIONS}×{TIMEOUT_ESCALATION_STEP}s,"
                f" 熔断={CIRCUIT_BREAKER_CONSECUTIVE_FAILS},"
                f" 并发={self._resolve_subtask_concurrency()},"
                f" max_subagents={self._resolve_max_subagents()})[/]"
            )
            self._record_trace(
                "subtask_scheduler_config",
                detail=(
                    f"strategy={self._resolve_subtask_concurrency()}, "
                    f"max_subagents={self._resolve_max_subagents()}"
                ),
                metadata={
                    "strategy": self._resolve_subtask_concurrency(),
                    "max_subagents": self._resolve_max_subagents(),
                },
            )

            round_results: list[str] = []
            circuit_broken = False

            # ── 批量执行：顺序 + 并行混合 ──
            # 连续标记 parallel=true 的子任务合并为一个并行批次；
            # 未标记或 parallel=false 的子任务仍顺序执行。
            batch_i = 0
            while batch_i < len(selected_indices):
                idx = selected_indices[batch_i]
                if idx >= len(subtasks):
                    batch_i += 1
                    continue

                # 每批次前检查熔断器
                try:
                    self._check_circuit_breaker()
                except PipelineCircuitBreakerError as exc:
                    renderer.console.print(
                        f"  [bold red]⛔ {exc}[/]"
                    )
                    circuit_broken = True
                    break

                task = subtasks[idx]
                is_parallel, parallel_reason = self._subtask_parallel_decision(task)

                if is_parallel:
                    # 收集连续且资源不冲突的候选子任务形成一个并行批次。
                    parallel_batch: list[dict] = []
                    batch_resource_keys: set[str] = set()
                    max_subagents = self._resolve_max_subagents()
                    while batch_i < len(selected_indices):
                        pidx = selected_indices[batch_i]
                        if pidx >= len(subtasks):
                            batch_i += 1
                            continue
                        ptask = subtasks[pidx]
                        can_parallel, reason = self._subtask_parallel_decision(ptask)
                        if not can_parallel:
                            self._record_trace(
                                "subtask_parallel_rejected",
                                detail=str(ptask.get("task_description", str(ptask)))[:200],
                                metadata={
                                    "index": pidx,
                                    "reason": reason,
                                },
                            )
                            break
                        resource_keys = self._extract_subtask_resource_keys(ptask)
                        conflicting_keys = sorted(batch_resource_keys & resource_keys)
                        if conflicting_keys:
                            self._record_trace(
                                "subtask_parallel_rejected",
                                detail=str(ptask.get("task_description", str(ptask)))[:200],
                                metadata={
                                    "index": pidx,
                                    "reason": "resource_conflict",
                                    "resource_keys": sorted(resource_keys),
                                    "conflicting_keys": conflicting_keys,
                                },
                            )
                            break
                        if len(parallel_batch) >= max_subagents:
                            self._record_trace(
                                "subtask_parallel_rejected",
                                detail=str(ptask.get("task_description", str(ptask)))[:200],
                                metadata={
                                    "index": pidx,
                                    "reason": "max_subagents_reached",
                                    "max_subagents": max_subagents,
                                },
                            )
                            break
                        ptask_with_index = dict(ptask)
                        ptask_with_index["_task_index"] = pidx
                        ptask_with_index["_resource_keys"] = sorted(resource_keys)
                        ptask_with_index["_parallel_reason"] = reason
                        parallel_batch.append(ptask_with_index)
                        batch_resource_keys.update(resource_keys)
                        self._print_subtask_status(
                            pidx,
                            str(ptask.get("role", "runner")),
                            str(ptask.get("task_description", str(ptask))),
                            "start",
                            parallel=True,
                        )
                        batch_i += 1

                    if not parallel_batch:
                        self._record_trace(
                            "subtask_parallel_rejected",
                            detail=str(task.get("task_description", str(task)))[:200],
                            metadata={
                                "index": idx,
                                "reason": parallel_reason,
                            },
                        )
                        continue

                    batch_results = self._run_parallel_batch(
                        parallel_batch,
                        user_input=user_input,
                        reasoning=reasoning,
                        additional_context=additional_context,
                    )
                    round_results.extend(batch_results)
                else:
                    # 顺序执行单条子任务（保持原有行为）
                    role_str = task.get("role", "runner")
                    desc = task.get("task_description", str(task))
                    ctx = task.get("context", "")
                    if additional_context:
                        ctx = f"{ctx}\n补充: {additional_context}" if ctx else additional_context

                    renderer.console.print(
                        f"  [dim]── [{role_str}] {desc[:80]}...[/]"
                    )
                    self._print_subtask_status(
                        idx,
                        role_str,
                        desc,
                        "start",
                    )
                    start = time_mod.monotonic()
                    self._record_trace(
                        "subtask_start",
                        detail=f"[{role_str}] {desc[:200]}",
                    )

                    subtask_prompt = self._build_subtask_prompt(
                        role_str,
                        desc,
                        ctx=ctx,
                        reasoning=reasoning,
                        aggressive=self._is_aggressive_execution(),
                        benchmark_profile=self._resolve_benchmark_profile(),
                        benchmark_state_context=self._benchmark_state_context(),
                    )

                    try:
                        result = self._run_subtask_with_escalating_timeout(
                            subtask_prompt, get_role_label(self._str_to_role(role_str)), desc,
                        )
                        elapsed = (time_mod.monotonic() - start) * 1000
                        renderer.console.print(
                            f"  [dim green]✓ 完成[/] [dim]({elapsed:.0f}ms, {len(result)}字)[/]"
                        )
                        self._print_subtask_status(
                            idx,
                            role_str,
                            desc,
                            "done",
                            detail=f"{elapsed:.0f}ms, {len(result)}字",
                        )
                        self._record_trace("subtask_complete", detail=result[:500])
                        round_results.append(
                            f"## [{role_str}] {desc}\n{result}"
                        )
                        self._consecutive_failures = 0

                    except TimeoutError as exc:
                        elapsed = (time_mod.monotonic() - start) * 1000
                        self._consecutive_failures += 1
                        self._record_trace("subtask_timeout", detail=str(exc))
                        renderer.console.print(
                            f"  [dim red]⏰ 全部叠加超时[/] [dim]({elapsed:.0f}ms)[/]"
                        )
                        self._print_subtask_status(
                            idx,
                            role_str,
                            desc,
                            "fail",
                            detail=f"超时 {elapsed:.0f}ms",
                        )
                        # 重规划：让决策者将此子任务拆分为更小粒度的子任务
                        replanned = self._replan_single_task(
                            desc, exc, user_input, reasoning,
                        )
                        if replanned:
                            renderer.console.print(
                                f"  [dim yellow]↻ 已重规划为 {len(replanned)} 个更小粒度的子任务，尝试执行...[/]"
                            )
                            for rt in replanned:
                                rstart = time_mod.monotonic()
                                try:
                                    rt_result = self._run_subtask_with_escalating_timeout(
                                        rt["prompt"], rt["label"], rt["desc"],
                                    )
                                    r_elapsed = (time_mod.monotonic() - rstart) * 1000
                                    renderer.console.print(
                                        f"    [dim green]✓ 重规划子任务完成[/] [dim]({r_elapsed:.0f}ms)[/]"
                                    )
                                    round_results.append(
                                        f"## [重规划] {rt['desc']}\n{rt_result}"
                                    )
                                    self._consecutive_failures = 0
                                except (TimeoutError, Exception) as r_exc:
                                    self._consecutive_failures += 1
                                    renderer.console.print(
                                        f"    [dim red]✗ 重规划子任务失败[/]: {r_exc}"
                                    )
                                    round_results.append(
                                        f"## [重规划] {rt['desc']}\n❌ 失败: {r_exc}"
                                    )
                        else:
                            renderer.console.print(
                                f"  [dim]重规划失败，记录原始错误。[/]"
                            )
                            round_results.append(
                                f"## [{role_str}] {desc}\n❌ 全部超时叠加后重规划也失败: {exc}"
                            )

                    except Exception as exc:
                        elapsed = (time_mod.monotonic() - start) * 1000
                        self._consecutive_failures += 1
                        self._record_trace("subtask_error", detail=f"{exc}")
                        renderer.console.print(
                            f"  [dim red]✗ 失败[/] [dim]({elapsed:.0f}ms)[/]: {exc}"
                        )
                        self._print_subtask_status(
                            idx,
                            role_str,
                            desc,
                            "fail",
                            detail=str(exc)[:80],
                        )
                        round_results.append(
                            f"## [{role_str}] {desc}\n❌ 失败: {exc}"
                        )

                    # ── 上下文压缩通知 ──
                    self._emit_compression_notice()

                    batch_i += 1

            all_results.append(round_results)
            benchmark_directive = self._update_benchmark_stale_state(round_results)
            if benchmark_directive:
                self._benchmark_forced_directive = benchmark_directive
                renderer.console.print(
                    "  [dim yellow]Benchmark stale detector 已触发，"
                    "下一轮将强制 close 当前题并切换下一题。[/]"
                )

            if circuit_broken:
                break

            # 8. 审计者验证
            renderer.console.print("  [dim]⏳ 审计者 正在验证结果...[/]")
            self._record_role_progress(
                "checker",
                "审计者",
                "start",
                action="正在验证结果",
                phase="execution",
            )
            check = self._call_role_with_timeout(
                AgentRole.CHECKER, user_input,
                context=(
                    f"## 执行计划\n{plan_json[:1000]}\n\n"
                    f"## 执行结果\n" + "\n---\n".join(
                        r[:800] for r in round_results
                    )
                ),
            )
            renderer.console.print("  [dim green]✓ 审计者 完成[/]")
            self._record_role_progress(
                "checker",
                "审计者",
                "done",
                detail=check[:500],
                phase="execution",
            )
            self._record_trace("role_checker", detail=check[:500])

            # 9. 反思者审视 → 决定是否继续迭代
            if iteration < max_iterations:
                renderer.console.print("  [dim]⏳ 反思者 正在审视是否需要迭代...[/]")
                self._record_role_progress(
                    "reflector",
                    "反思者",
                    "start",
                    action="正在审视是否需要迭代",
                    phase="execution",
                )
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
                        + self._aggressive_planning_instruction()
                    ),
                )
                self._record_role_progress(
                    "reflector",
                    "反思者",
                    "done",
                    detail=reflection[:500],
                    phase="execution",
                )
                if (
                    ("执行完成" in reflection and not self._benchmark_forced_directive)
                    or self._consecutive_failures > 0
                ):
                    benchmark_continue = self._benchmark_target_continue_directive()
                    if benchmark_continue:
                        self._benchmark_forced_directive = benchmark_continue
                        renderer.console.print(
                            "  [dim yellow]Benchmark target gate 未达标，"
                            "覆盖“执行完成”判定并继续迭代。[/]"
                        )
                        self._record_trace(
                            "benchmark_target_gate_continue",
                            detail=benchmark_continue,
                            metadata=self._benchmark_score_status(),
                        )
                        continue
                    # 有失败时不再迭代，直接收尾
                    renderer.console.print(
                        "  [dim green]✓ 反思者判定：执行完成[/]"
                    )
                    self._record_trace("iteration_done", detail=f"第 {iteration} 轮：执行完成")
                    break
                renderer.console.print(
                    "  [dim yellow]↻ 反思者判定：需继续迭代[/]"
                )
                self._record_trace("iteration_continue", detail=f"第 {iteration} 轮：继续迭代")
            else:
                renderer.console.print(
                    "  [dim]已达最大迭代次数，结束循环。[/]"
                )
                self._record_trace("iteration_done", detail=f"已达最大迭代次数")

        # ── Phase 3: 聚合输出 ──
        renderer.console.print()
        renderer.console.print("[dim bold]📊 四柱管线执行完成[/]")

        if all_results:
            self._final_summary = self._build_execution_summary(all_results, iteration)
            benchmark_summary = self._benchmark_final_summary()
            if benchmark_summary:
                self._final_summary += benchmark_summary
            renderer.print_markdown(self._final_summary)

    def _emit_compression_notice(self) -> None:
        """检查 runner 最近是否触发了上下文压缩，若有则打印通知。"""
        info = getattr(self._runner, "last_compression_info", None)
        if info:
            self._renderer.console.print(
                f"  [dim yellow]📦 上下文压缩: {info['count']} 条历史消息已压缩"
                f" ({info['method']})，按 Ctrl+B 查看详情[/]"
            )

    # ── 重规划超时子任务 ──
    def _replan_single_task(
        self,
        original_desc: str,
        timeout_exc: TimeoutError,
        user_input: str,
        reasoning: str,
    ) -> list[dict] | None:
        """对超时子任务进行重规划，拆分为更小粒度的子任务。

        让决策者分析失败原因并输出 JSON 格式的更小任务列表。
        """
        renderer = self._renderer
        replan_context = (
            f"## 原始用户任务\n{user_input}\n\n"
            f"## 整体计划\n{reasoning[:500]}\n\n"
            f"## 超时的子任务\n{original_desc}\n\n"
            f"## 超时信息\n{timeout_exc}\n"
        )

        decision = self._call_role_with_timeout(
            AgentRole.DECISION_MAKER, "",
            context=replan_context,
            extra_instruction=(
                "以上子任务因超时未能完成。请将其拆分为 2-3 个更小粒度的子任务，"
                "每个小任务应该更聚焦、更容易在短时间内完成。"
                "\n\n输出必须是 JSON："
                '{"reasoning": "...", "subtasks": ['
                '{"role": "runner", "task_description": "..."}, '
                '{"role": "reader", "task_description": "..."}]}'
            ),
        )
        parsed = self._parse_json(decision)
        raw_tasks = parsed.get("subtasks", [])
        if not raw_tasks:
            return None

        result: list[dict] = []
        for t in raw_tasks:
            desc = t.get("task_description", str(t))
            role = t.get("role", "runner")
            result.append({
                "desc": desc,
                "prompt": (
                    f"你是{get_role_label(self._str_to_role(role))}。"
                    f"这是拆分后的小任务，请只做这一件事，完成后给出结果摘要。\n\n"
                    f"子任务: {desc}\n"
                    f"\n整体背景: {reasoning[:300]}\n"
                    f"\n请直接调用工具完成此子任务，给出核心结果。"
                ),
                "label": get_role_label(self._str_to_role(role)),
            })
        return result

    # ── 子任务选择 ──
    def _auto_select(
        self, subtasks: list[dict], reasoning: str,
    ) -> tuple[list[int], str]:
        """思考者自动评估并选择子任务。"""
        renderer = self._renderer
        renderer.console.print(
            "  [dim bold]🤔 思考者正在评估子任务...[/]"
        )

        tasks_text = "\n".join(
            f"  [{i}] 角色={t.get('role', '?')} | {t.get('task_description', '')[:200]}"
            for i, t in enumerate(subtasks)
        )

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
