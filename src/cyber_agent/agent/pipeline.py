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
from collections.abc import Callable, Iterable
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
from . import benchmark_profiles as benchmark_profile_utils
from .benchmark_probe import BenchmarkProbeMixin
from .benchmark_runtime import BenchmarkRuntimeMixin
from .benchmark_strategy import BenchmarkStrategyMixin
from .events import AgentEventType
from .pipeline_constants import (
    BASE_SUBTASK_TIMEOUT,
    BENCHMARK_LOW_VALUE_SIGNAL_LIMIT,
    BENCHMARK_MAX_ITERATION_BATCHES,
    BENCHMARK_REDUNDANT_BLOCK_LIMIT,
    BENCHMARK_RUSH_REMAINING_THRESHOLD,
    BENCHMARK_SETUP_TIMEOUT_STREAK_LIMIT,
    BENCHMARK_SUBTASK_TIMEOUT,
    BENCHMARK_TIMEOUT_ESCALATIONS,
    CIRCUIT_BREAKER_CONSECUTIVE_FAILS,
    LLM_CALL_TIMEOUT_SECONDS,
    MAX_TIMEOUT_ESCALATIONS,
    TIMEOUT_ESCALATION_STEP,
)
from .roles import AgentRole, get_role_label, get_role_prompt

if TYPE_CHECKING:
    from .runner import AgentRunner

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


class FourPillarPipeline(
    BenchmarkStrategyMixin,
    BenchmarkRuntimeMixin,
    BenchmarkProbeMixin,
):
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
            "observed_probe_paths": set(),
            "observed_param_names": set(),
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
                "observed_probe_paths",
                "observed_param_names",
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
        return benchmark_profile_utils.workspace_path()

    def _benchmark_external_profiles_path(self) -> Path:
        return benchmark_profile_utils.external_profiles_path(self._runtime_context)

    def _benchmark_external_profiles(self) -> dict[str, Any]:
        return benchmark_profile_utils.load_external_profiles(
            self._benchmark_external_profiles_path()
        )

    def _benchmark_selection_policy(self) -> dict[str, Any]:
        return benchmark_profile_utils.selection_policy(self._benchmark_external_profiles())

    def _benchmark_execution_control_policy(self) -> dict[str, Any]:
        """Runtime breadth controls for benchmark scheduling and probing.

        Profiles may provide reusable numeric budgets, while runtime context can
        override them for a specific run. This keeps profiles as policy/data
        rather than fixed solve plans.
        """
        return benchmark_profile_utils.execution_control_policy(
            self._benchmark_external_profiles(),
            self._runtime_context,
        )

    def _benchmark_control_int(self, key: str) -> int:
        return int(self._benchmark_execution_control_policy()[key])

    def _benchmark_control_text(self, key: str) -> str:
        return str(self._benchmark_execution_control_policy()[key])

    @staticmethod
    def _benchmark_deadline_remaining(deadline: float | None) -> float:
        if deadline is None:
            return 999999.0
        return max(0.0, deadline - time_mod.monotonic())

    @staticmethod
    def _benchmark_deadline_expired(deadline: float | None) -> bool:
        return deadline is not None and time_mod.monotonic() >= deadline

    def _benchmark_disabled_builtin_fingerprints(self) -> set[str]:
        return benchmark_profile_utils.disabled_builtin_fingerprints(
            self._benchmark_external_profiles()
        )

    def _benchmark_builtin_section_enabled(self, section: str) -> bool:
        return benchmark_profile_utils.builtin_section_enabled(
            self._benchmark_external_profiles(),
            section,
        )

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
        return benchmark_profile_utils.string_tuple(value, limit=limit)

    @staticmethod
    def _benchmark_string_pair_tuple(
        value: Any,
        *,
        limit: int = 20,
    ) -> tuple[tuple[str, str], ...]:
        return benchmark_profile_utils.string_pair_tuple(value, limit=limit)

    @staticmethod
    def _benchmark_match_any_all_tuple(value: Any) -> tuple[tuple[str, ...], ...]:
        return benchmark_profile_utils.match_any_all_tuple(value)

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
            if code in closed and code not in abandoned and code not in recovered:
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
                            closed = set(self._benchmark_state.get("closed_challenges", set()))
                            recovered = set(
                                self._benchmark_state.get("recovery_attempted_challenges", set())
                            )
                            closed.discard(code)
                            recovered.add(code)
                            self._benchmark_state["closed_challenges"] = closed
                            self._benchmark_state["recovery_attempted_challenges"] = recovered
                    return best_recovery
            return best_candidate
        if priority_recovery:
            code = priority_recovery[0].get("unique_code")
            if isinstance(code, str):
                with self._benchmark_state_lock:
                    closed = set(self._benchmark_state.get("closed_challenges", set()))
                    recovered = set(
                        self._benchmark_state.get("recovery_attempted_challenges", set())
                    )
                    closed.discard(code)
                    recovered.add(code)
                    self._benchmark_state["closed_challenges"] = closed
                    self._benchmark_state["recovery_attempted_challenges"] = recovered
            return priority_recovery[0]
        if recovery_candidates:
            code = recovery_candidates[0].get("unique_code")
            if isinstance(code, str):
                with self._benchmark_state_lock:
                    closed = set(self._benchmark_state.get("closed_challenges", set()))
                    recovered = set(
                        self._benchmark_state.get("recovery_attempted_challenges", set())
                    )
                    closed.discard(code)
                    recovered.add(code)
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
            if code in closed and code not in abandoned and code not in recovered:
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
                    closed = set(self._benchmark_state.get("closed_challenges", set()))
                    recovered = set(
                        self._benchmark_state.get("recovery_attempted_challenges", set())
                    )
                    closed.discard(code)
                    recovered.add(code)
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
            "tcp_connect_ok",
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
        normalized_addrs = [str(addr) for addr in addrs or []]
        if normalized_addrs:
            self._benchmark_store_active_challenge(code, normalized_addrs)
        return code, normalized_addrs

    def _benchmark_store_active_challenge(self, code: str, addrs: list[str]) -> None:
        if not code or not addrs:
            return
        with self._benchmark_state_lock:
            active = dict(self._benchmark_state.get("active_containers", {}))
            closed = set(self._benchmark_state.get("closed_challenges", set()))
            abandoned = set(self._benchmark_state.get("abandoned_challenges", set()))
            recovered = set(self._benchmark_state.get("recovery_attempted_challenges", set()))
            active[code] = [str(addr) for addr in addrs]
            closed.discard(code)
            abandoned.discard(code)
            recovered.discard(code)
            self._benchmark_state["active_containers"] = active
            self._benchmark_state["current_challenge"] = code
            self._benchmark_state["closed_challenges"] = closed
            self._benchmark_state["abandoned_challenges"] = abandoned
            self._benchmark_state["recovery_attempted_challenges"] = recovered
            self._benchmark_current_challenge = code
        self._persist_benchmark_state()

    def _benchmark_close_local(self, code: str) -> str:
        _, stdout, _ = self._benchmark_platform_request(
            method="POST",
            path=f"/openapi/v1/challenges/close?unique_code={code}",
        )
        closed_ok = False
        try:
            value = json.loads(stdout)
        except Exception:
            value = None
        if isinstance(value, dict):
            closed_ok = value.get("closed") is True or value.get("status") in {
                "closed",
                "already_closed",
            }
        if closed_ok or "already" in stdout.lower():
            with self._benchmark_state_lock:
                closed = set(self._benchmark_state.get("closed_challenges", set()))
                closed.add(code)
                active = dict(self._benchmark_state.get("active_containers", {}))
                active.pop(code, None)
                self._benchmark_state["closed_challenges"] = closed
                self._benchmark_state["active_containers"] = active
                if self._benchmark_state.get("current_challenge") == code:
                    self._benchmark_state["current_challenge"] = None
                if self._benchmark_current_challenge == code:
                    self._benchmark_current_challenge = None
            self._persist_benchmark_state()
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
                self._benchmark_store_active_challenge(code, [str(addr) for addr in addrs])
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
        deadline = time_mod.monotonic() + self._benchmark_control_int("fast_probe_seconds")
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
        root_body = self._benchmark_wait_for_container_ready(
            base,
            outputs,
            deadline=deadline,
        )
        urls.extend(self._benchmark_derive_probe_urls(base, root_body))
        tun_interface = self._benchmark_tun_interface()
        seen_urls: set[str] = set()
        queue: list[str] = []
        for url in urls:
            if url in seen_urls:
                continue
            seen_urls.add(url)
            queue.append(url)
        max_probe_urls = self._benchmark_control_int("max_probe_urls")
        index = 0
        while index < len(queue) and index < max_probe_urls:
            remaining = self._benchmark_deadline_remaining(deadline)
            if remaining <= 1.0:
                outputs.append(
                    f"## probe budget exhausted {addr}\n"
                    f"fast_probe_seconds={self._benchmark_control_int('fast_probe_seconds')}"
                )
                break
            url = queue[index]
            index += 1
            curl_max_time = max(1, min(4, int(remaining)))
            run_timeout = max(2, min(6, int(remaining) + 1))
            cmd = [
                "curl",
                "-sS",
                "-k",
                "--interface",
                tun_interface,
                "--connect-timeout",
                "2",
                "--max-time",
                str(curl_max_time),
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
                    timeout=run_timeout,
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
        if self._benchmark_deadline_expired(deadline):
            return "\n".join(outputs)
        joined_outputs = "\n".join(outputs)
        service_matched, service_outputs = self._benchmark_probe_matching_service_local(
            code,
            base,
            joined_outputs,
        )
        if service_matched:
            outputs.extend(service_outputs)
            return "\n".join(outputs)
        if self._benchmark_deadline_expired(deadline):
            return "\n".join(outputs)
        webapp_output = self._benchmark_probe_common_webapp_flows(
            code,
            base,
            joined_outputs,
            deadline=deadline,
        )
        if webapp_output:
            outputs.append(webapp_output)
            joined_outputs = "\n".join(outputs)
        if self._benchmark_deadline_expired(deadline):
            return "\n".join(outputs)
        if self._benchmark_probe_suggests_raw_text_protocol(joined_outputs):
            raw_output = self._benchmark_probe_raw_text_protocol(code, addr)
            if raw_output:
                outputs.append(raw_output)
        joined_outputs = "\n".join(outputs)
        if self._benchmark_probe_looks_unreachable(joined_outputs):
            diagnostics = self._benchmark_connectivity_diagnostics(addr)
            outputs.append(diagnostics)
            if "tcp_connect_ok" in diagnostics.lower():
                raw_output = self._benchmark_probe_raw_text_protocol(code, addr)
                if raw_output:
                    outputs.append(raw_output)
        return "\n".join(outputs)

    def _benchmark_connectivity_diagnostics(self, addr: str) -> str:
        host, port_text = addr.rsplit(":", 1)
        tun_interface = self._benchmark_tun_interface()
        outputs: list[str] = [f"## connectivity-diagnostics {addr}"]
        try:
            with socket_mod.create_connection((host, int(port_text)), timeout=3):
                outputs.append("python_socket_tcp_probe: tcp_connect_ok")
        except Exception as exc:
            outputs.append(f"python_socket_tcp_probe: failed ({exc})")
        commands = [
            ["ip", "route", "get", host],
            ["ip", "addr", "show", tun_interface],
            [
                "curl",
                "-sS",
                "--interface",
                tun_interface,
                "--connect-timeout",
                "3",
                "--max-time",
                "5",
                "-v",
                f"telnet://{host}:{port_text}",
            ],
        ]
        for cmd in commands:
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
                outputs.append(f"$ {' '.join(cmd)}\nERROR: {exc}")
                continue
            body = (result.stdout or "")[:1500]
            err = (result.stderr or "")[:1500]
            marker = ""
            if cmd[0] == "curl" and (
                result.returncode == 0
                or "connected to " in err.lower()
            ):
                marker = "\ntcp_connect_ok"
            outputs.append(
                f"$ {' '.join(cmd)}\n"
                f"exit={result.returncode}{marker}\n"
                f"{body}\n{err}"
            )
        return "\n".join(outputs)

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

        if (
            self._benchmark_profile_active
            and "max_subagents" not in self._runtime_context
            and self._benchmark_control_int("max_subagents") > 0
        ):
            raw_value = self._benchmark_control_int("max_subagents")
        else:
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

        if self._benchmark_profile_active and "subtask_concurrency" not in self._runtime_context:
            policy_value = self._benchmark_control_text("subtask_concurrency")
        else:
            policy_value = ""
        raw_value = str(
            policy_value
            or self._runtime_context.get(
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
            "challenges_api.md",
            "benchmark_token",
            "/openapi/v1/challenges",
            "correct_flag_count",
            "unique_code",
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
                benchmark_action = self._benchmark_fast_action_from_task(task, desc)
                ctx = str(task.get("context", ""))
                if additional_context:
                    ctx = f"{ctx}\n补充: {additional_context}" if ctx else additional_context

                if benchmark_action == "probe":
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
                should_run_deterministic = benchmark_action in {"setup", "schedule", "probe"}
                if should_run_deterministic:
                    deterministic_step = benchmark_action
                    try:
                        deterministic_result = self._benchmark_deterministic_fast_step(
                            desc,
                            reason=(
                                "deterministic_probe_submit_close"
                                if deterministic_step == "probe"
                                else "deterministic_scheduler"
                            ),
                            action=benchmark_action,
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
                    if benchmark_action == "probe":
                        try:
                            fallback_result = self._benchmark_deterministic_fast_step(
                                desc,
                                reason=f"timeout:{str(exc)[:160]}",
                                action=benchmark_action,
                            )
                        except Exception as fallback_exc:
                            self._record_trace(
                                "benchmark_deterministic_fast_failed",
                                detail=str(fallback_exc),
                                metadata={
                                    "step": "probe_timeout",
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
                                action=benchmark_action or None,
                            )
                        except Exception as fallback_exc:
                            self._record_trace(
                                "benchmark_deterministic_fast_failed",
                                detail=str(fallback_exc),
                                metadata={
                                    "step": benchmark_action or "unknown",
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
                            # aggressive Benchmark 的 180s 超时是预期内止损信号，
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
