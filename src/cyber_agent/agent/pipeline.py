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
                role_label, desc, ctx=ctx, reasoning=reasoning,
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
        self._session_id = str(
            self._runtime_context.get("session_id")
            or datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        self._trace_id = (
            f"{self._session_id}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        self._append_pipeline_user_message(user_input)
        self._record_trace("pipeline_start", detail=user_input)

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
        max_iterations = self._resolve_max_iterations()
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
            plan_json = self._call_role_with_timeout(
                AgentRole.DECISION_MAKER, user_input,
                context=f"## 反思者执行计划\n{iter_context}",
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
                        role_str, desc, ctx=ctx, reasoning=reasoning,
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
                    ),
                )
                self._record_role_progress(
                    "reflector",
                    "反思者",
                    "done",
                    detail=reflection[:500],
                    phase="execution",
                )
                if "执行完成" in reflection or self._consecutive_failures > 0:
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
