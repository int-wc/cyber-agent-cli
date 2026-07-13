"""飞书（Feishu/Lark）专用模块：Trace 收集、进度推送、卡片构建、命令响应 Payload。

从 webhook.py 拆分以控制单文件行数，保持 Feishu 逻辑集中维护。
"""
from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from ..agent.approval import (
    ApprovalPolicy,
    get_approval_policy_label,
)
from ..agent.mode import get_mode_description, get_mode_label
from ..session_store import (
    get_session_storage_dir,
    list_stored_sessions,
    load_session_history,
    search_stored_sessions,
)
from ..tools import (
    describe_allowed_roots,
    describe_command_registry,
    describe_tool_instances,
)
# build_doctor_payload 延迟导入，见 _build_feishu_doctor_payload
from .interactive import BUILTIN_COMMAND_SPECS, get_interaction_ui_mode_label
from .webhook_models import *  # noqa: F403

class FeishuTraceCollector:
    """收集中间过程事件，供飞书消息卡片展示。"""

    def __init__(self) -> None:
        self.steps: list[FeishuTraceStep] = []

    def __call__(self, event_type: str, payload: object) -> None:
        self.steps.extend(self.build_steps(event_type, payload))

    @classmethod
    def build_steps(
        cls,
        event_type: str,
        payload: object,
    ) -> list[FeishuTraceStep]:
        """把运行器事件转换成适合飞书展示的步骤列表。"""
        if event_type.startswith("pipeline.") and isinstance(payload, Mapping):
            return cls._build_pipeline_steps(event_type, payload)

        if event_type == "tool_call" and isinstance(payload, list):
            steps: list[FeishuTraceStep] = []
            for tool_call in payload:
                if not isinstance(tool_call, Mapping):
                    continue
                tool_name = str(tool_call.get("name", "unknown")).strip() or "unknown"
                raw_args = tool_call.get("args", {})
                if isinstance(raw_args, str):
                    try:
                        raw_args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        pass
                args_text = cls._serialize_object(raw_args)
                detail = ""
                if args_text:
                    detail = cls._build_code_block(
                        args_text,
                        language="json",
                    )
                steps.append(
                    FeishuTraceStep(
                        kind="tool_call",
                        title=f"调用工具 `{tool_name}`",
                        detail=detail,
                    )
                )
            return steps

        if event_type == "approval_request" and isinstance(payload, Mapping):
            tool_name = str(payload.get("tool_name", "unknown")).strip() or "unknown"
            risk = str(payload.get("risk", "unknown")).strip() or "unknown"
            return [
                FeishuTraceStep(
                    kind="approval_request",
                    title=f"等待审批 `{tool_name}`",
                    detail=f"- 风险级别：`{risk}`",
                )
            ]

        if event_type == "approval_result" and isinstance(payload, Mapping):
            tool_name = str(payload.get("tool_name", "unknown")).strip() or "unknown"
            approved = bool(payload.get("approved", False))
            reason = str(payload.get("reason", "")).strip()
            detail_lines = [f"- 结果：`{'已批准' if approved else '已拒绝'}`"]
            if reason:
                detail_lines.append(f"- 说明：{reason}")
            return [
                FeishuTraceStep(
                    kind="approval_result",
                    title=f"审批结果 `{tool_name}`",
                    detail="\n".join(detail_lines),
                )
            ]

        if event_type == "tool_result" and isinstance(payload, Mapping):
            tool_name = str(payload.get("tool_name", "unknown")).strip() or "unknown"
            content = str(payload.get("content", "")).strip()
            normalized_content = _normalize_cli_output_for_feishu(content) or content or "（空结果）"
            return [
                FeishuTraceStep(
                    kind="tool_result",
                    title=f"采集结果 `{tool_name}`",
                    detail=cls._build_tool_result_detail(normalized_content),
                )
            ]

        return []

    @classmethod
    def _build_pipeline_steps(
        cls,
        event_type: str,
        payload: Mapping[str, object],
    ) -> list[FeishuTraceStep]:
        """把四柱管线事件转换成飞书进度步骤。"""
        event_name = event_type.removeprefix("pipeline.")
        detail = str(payload.get("detail", "")).strip()
        raw_metadata = payload.get("metadata", {})
        metadata = raw_metadata if isinstance(raw_metadata, Mapping) else {}

        if event_name == "pipeline_start":
            return [
                FeishuTraceStep(
                    kind="pipeline_start",
                    title="启动四柱 Agent 管线",
                    detail="- 阶段：`分析为底 → 扩展为路 → 迁跃为辅 → 反思为主`",
                )
            ]

        if event_name == "pipeline_complete":
            return [
                FeishuTraceStep(
                    kind="pipeline_done",
                    title="四柱管线执行完成",
                    detail="",
                )
            ]

        if event_name in {"pipeline_abort", "pipeline_error"}:
            return [
                FeishuTraceStep(
                    kind="pipeline_error",
                    title="四柱管线中止",
                    detail=f"- 原因：{cls._truncate_text(detail, max_chars=240)}",
                )
            ]

        if event_name == "role_progress":
            label = str(metadata.get("label", "子 Agent")).strip() or "子 Agent"
            status = str(metadata.get("status", "")).strip()
            action = str(metadata.get("action", "")).strip()
            phase = str(metadata.get("phase", "")).strip()
            status_label = {
                "start": "开始",
                "done": "完成",
                "error": "异常",
            }.get(status, status or "更新")
            detail_lines: list[str] = []
            if phase:
                detail_lines.append(f"- 阶段：`{phase}`")
            if action:
                detail_lines.append(f"- 动作：{action}")
            elapsed_ms = metadata.get("elapsed_ms")
            if isinstance(elapsed_ms, (int, float)):
                detail_lines.append(f"- 耗时：`{elapsed_ms:.0f}ms`")
            if detail:
                detail_lines.append(
                    "- 摘要："
                    + cls._truncate_text(
                        re.sub(r"\s+", " ", detail),
                        max_chars=260,
                    )
                )
            return [
                FeishuTraceStep(
                    kind=f"pipeline_role_{status or 'update'}",
                    title=f"{label} {status_label}",
                    detail="\n".join(detail_lines),
                )
            ]

        if event_name == "iteration_start":
            return [
                FeishuTraceStep(
                    kind="pipeline_iteration",
                    title=f"执行循环开始：{detail or '新一轮'}",
                    detail="",
                )
            ]

        if event_name in {"iteration_done", "iteration_continue"}:
            title = "执行循环完成" if event_name == "iteration_done" else "继续下一轮迭代"
            return [
                FeishuTraceStep(
                    kind="pipeline_iteration",
                    title=title,
                    detail=f"- 结论：{detail}" if detail else "",
                )
            ]

        if event_name == "subtasks_selected":
            return [cls._build_subtasks_selected_step(detail, metadata)]

        if event_name == "subtask_status":
            return [cls._build_subtask_status_step(detail, metadata)]

        if event_name == "subtask_scheduler_config":
            strategy = str(metadata.get("strategy", "")).strip() or "auto"
            max_subagents = metadata.get("max_subagents", "")
            return [
                FeishuTraceStep(
                    kind="pipeline_scheduler",
                    title="子任务调度策略",
                    detail=(
                        f"- 并发策略：`{strategy}`\n"
                        f"- 最大子 Agent：`{max_subagents}`"
                    ),
                )
            ]

        if event_name == "subtask_parallel_rejected":
            reason = str(metadata.get("reason", "")).strip() or "unknown"
            reason_label = {
                "concurrency_off": "并发已关闭",
                "sensitive_operation": "敏感操作需顺序执行",
                "not_marked_parallel": "未标记为可并发",
                "resource_conflict": "资源锁冲突",
                "max_subagents_reached": "达到最大并发数",
            }.get(reason, reason)
            detail_lines = [f"- 原因：`{reason_label}`"]
            resource_keys = metadata.get("conflicting_keys") or metadata.get("resource_keys")
            if isinstance(resource_keys, Sequence) and not isinstance(resource_keys, (str, bytes)):
                values = [str(item) for item in resource_keys[:6]]
                if values:
                    detail_lines.append("- 资源：" + "、".join(f"`{item}`" for item in values))
            if detail:
                detail_lines.append(
                    "- 子任务："
                    + cls._truncate_text(
                        re.sub(r"\s+", " ", detail),
                        max_chars=180,
                    )
                )
            return [
                FeishuTraceStep(
                    kind="pipeline_scheduler",
                    title="子任务改为顺序执行",
                    detail="\n".join(detail_lines),
                )
            ]

        if event_name in {"parallel_batch_start", "parallel_batch_end"}:
            title = "并行子任务批次开始" if event_name.endswith("start") else "并行子任务批次完成"
            return [
                FeishuTraceStep(
                    kind="pipeline_parallel",
                    title=title,
                    detail=f"- 范围：{detail}" if detail else "",
                )
            ]

        if event_name == "tool_call":
            tool_name = str(metadata.get("tool", "unknown")).strip() or "unknown"
            args_text = cls._serialize_object(metadata.get("args", {}))
            detail_lines = []
            elapsed_s = metadata.get("elapsed_s")
            if isinstance(elapsed_s, (int, float)):
                detail_lines.append(f"- 子任务耗时：`{elapsed_s:.0f}s`")
            if args_text:
                detail_lines.append(cls._build_code_block(args_text, language="json"))
            return [
                FeishuTraceStep(
                    kind="pipeline_tool_call",
                    title=f"子任务调用工具 `{tool_name}`",
                    detail="\n".join(detail_lines),
                )
            ]

        if event_name == "tool_result":
            tool_name = str(metadata.get("tool", "unknown")).strip() or "unknown"
            status = str(metadata.get("status", "")).strip()
            content = str(metadata.get("content", "")).strip()
            detail_lines = []
            if status:
                detail_lines.append(f"- 状态：`{status}`")
            exit_code = str(metadata.get("exit_code", "")).strip()
            if exit_code:
                detail_lines.append(f"- 退出码：`{exit_code}`")
            if content:
                normalized_content = _normalize_cli_output_for_feishu(content) or content
                detail_lines.append(cls._build_tool_result_detail(normalized_content))
            return [
                FeishuTraceStep(
                    kind="pipeline_tool_result",
                    title=f"子任务工具结果 `{tool_name}`",
                    detail="\n\n".join(detail_lines),
                )
            ]

        return []

    @classmethod
    def _build_subtasks_selected_step(
        cls,
        detail: str,
        metadata: Mapping[str, object],
    ) -> FeishuTraceStep:
        raw_subtasks = metadata.get("subtasks", [])
        subtasks = (
            raw_subtasks
            if isinstance(raw_subtasks, Sequence)
            and not isinstance(raw_subtasks, (str, bytes))
            else []
        )
        selected = [
            task for task in subtasks
            if isinstance(task, Mapping) and bool(task.get("selected", False))
        ]
        iteration = metadata.get("iteration")
        detail_lines = []
        if iteration:
            detail_lines.append(f"- 轮次：`第 {iteration} 轮`")
        if subtasks:
            detail_lines.append(f"- 已选择：`{len(selected)}/{len(subtasks)}`")
        for task in selected[:8]:
            index = cls._safe_int(task.get("index", 0)) + 1
            role = str(task.get("role", "runner")).strip() or "runner"
            mode = "并行" if bool(task.get("parallel", False)) else "顺序"
            desc = re.sub(r"\s+", " ", str(task.get("description", ""))).strip()
            detail_lines.append(
                f"- `#{index:02d}` `{role} Agent` `{mode}` "
                f"{cls._truncate_text(desc, max_chars=120)}"
            )
        if len(selected) > 8:
            detail_lines.append(f"- 其余 `{len(selected) - 8}` 个子任务已省略。")
        if not detail_lines and detail:
            detail_lines.append(f"- {detail}")
        return FeishuTraceStep(
            kind="pipeline_subtasks",
            title="子 Agent 任务清单",
            detail="\n".join(detail_lines),
        )

    @classmethod
    def _build_subtask_status_step(
        cls,
        detail: str,
        metadata: Mapping[str, object],
    ) -> FeishuTraceStep:
        index = cls._safe_int(metadata.get("index", 0)) + 1
        agent_label = str(metadata.get("agent_label", "")).strip()
        if not agent_label:
            role = str(metadata.get("role", "runner")).strip() or "runner"
            agent_label = f"{role} Agent"
        status = str(metadata.get("status", "")).strip()
        status_label = str(metadata.get("status_label", "")).strip() or status or "更新"
        mode = str(metadata.get("mode", "")).strip()
        extra_detail = str(metadata.get("detail", "")).strip()
        desc = re.sub(r"\s+", " ", detail).strip()
        detail_lines = []
        if mode:
            detail_lines.append(f"- 模式：`{mode}`")
        if desc:
            detail_lines.append(
                f"- 子任务：{cls._truncate_text(desc, max_chars=240)}"
            )
        if extra_detail:
            detail_lines.append(f"- 详情：{extra_detail}")
        return FeishuTraceStep(
            kind=f"pipeline_subtask_{status or 'update'}",
            title=f"#{index:02d} {agent_label} {status_label}",
            detail="\n".join(detail_lines),
        )

    @staticmethod
    def _safe_int(value: object, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _serialize_object(value: object) -> str:

        """把工具参数转成适合展示的文本。"""
        if value in (None, "", {}, []):
            return ""
        try:
            serialized = json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
            )
        except TypeError:
            serialized = str(value)
        return FeishuTraceCollector._truncate_text(serialized)

    @staticmethod
    def _truncate_text(text: str, *, max_chars: int = FEISHU_TRACE_DETAIL_MAX_CHARS) -> str:
        """限制单条中间步骤详情长度，避免卡片被超长输出撑爆。"""
        normalized_text = text.strip()
        if len(normalized_text) <= max_chars:
            return normalized_text
        return normalized_text[:max_chars].rstrip() + "\n... 内容较长，已截断。"

    @classmethod
    def _build_code_block(cls, text: str, *, language: str) -> str:
        """将中间过程详情包装成代码块，便于阅读命令和输出。"""
        normalized_text = cls._truncate_text(text)
        if not normalized_text:
            return ""
        return (
            f"```{language}\n"
            f"{_escape_feishu_code_block(normalized_text)}\n"
            "```"
        )

    @classmethod
    def _build_tool_result_detail(cls, text: str) -> str:
        """优先把普通工具输出里的键值结果渲染成飞书卡片键值列表。"""
        structured_detail = _build_feishu_tool_result_key_value_detail(text)
        if structured_detail:
            return structured_detail
        return cls._build_code_block(text, language="text")
class FeishuProgressMessageEmitter:
    """把飞书中间处理步骤作为独立消息即时发送。"""

    def __init__(
        self,
        send_step: Callable[[FeishuTraceStep, int], None],
    ) -> None:
        self._send_step = send_step
        self._step_index = 0
        self._lock = threading.Lock()
        self._heartbeat_stop_event = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._started_at = 0.0
        self._last_activity_at = 0.0
        self._latest_status_title = "等待开始"
        self._latest_status_detail = ""
        self._closed = False

    def start(self, user_input: str) -> None:

        """在任务正式进入运行器前先发出一条“已开始处理”的状态。"""
        input_preview = self._build_input_preview(user_input)
        with self._lock:
            if self._closed or self._started_at > 0.0:
                return
            now = time.monotonic()
            self._started_at = now
            self._last_activity_at = now
            self._latest_status_title = "等待模型开始分析"
            self._latest_status_detail = (
                f"- 用户请求：`{input_preview}`" if input_preview else ""
            )
        self._ensure_heartbeat_thread_started()
        detail_lines = ["- 状态：`已开始处理`"]
        if input_preview:
            detail_lines.append(f"- 用户请求：`{input_preview}`")
        self._emit_step(
            FeishuTraceStep(
                kind="start",
                title="已收到任务，开始处理",
                detail="\n".join(detail_lines),
            )
        )

    def close(self) -> None:
        """在任务结束后停止心跳线程，避免后台残留。"""
        heartbeat_thread: threading.Thread | None = None
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._heartbeat_stop_event.set()
            heartbeat_thread = self._heartbeat_thread
        if (
            heartbeat_thread is not None
            and heartbeat_thread.is_alive()
            and heartbeat_thread is not threading.current_thread()
        ):
            heartbeat_thread.join(timeout=0.2)

    def __call__(self, event_type: str, payload: object) -> None:
        if event_type == "turn_start":
            if isinstance(payload, Mapping):
                self.start(str(payload.get("input", "")))
            else:
                self.start("")
            return
        if event_type == "response_token":
            self._touch()
            return
        self._update_status(event_type, payload)
        for step in FeishuTraceCollector.build_steps(event_type, payload):
            self._emit_step(step)

    def _emit_step(self, step: FeishuTraceStep) -> None:
        """统一为各类进度消息分配序号，并刷新活跃时间。"""
        with self._lock:
            if self._closed:
                return
            self._step_index += 1
            step_index = self._step_index
            self._last_activity_at = time.monotonic()
        self._send_step(step, step_index)

    def _touch(self) -> None:
        """记录最近一次运行活动，避免在仍有输出时误发心跳。"""
        with self._lock:
            if self._closed:
                return
            if self._started_at <= 0.0:
                self._started_at = time.monotonic()
            self._last_activity_at = time.monotonic()

    def _update_status(self, event_type: str, payload: object) -> None:
        """根据运行器事件刷新“最近状态”，供心跳消息说明当前卡在哪一步。"""
        latest_status_title = ""
        latest_status_detail = ""
        if event_type == "response_begin":
            latest_status_title = "正在等待模型响应"
            latest_status_detail = "- 模型已开始分析当前问题。"
        elif event_type == "response_end" and isinstance(payload, Mapping):
            if bool(payload.get("has_tool_calls", False)):
                latest_status_title = "模型已生成工具计划"
                latest_status_detail = "- 即将开始执行工具步骤。"
            else:
                latest_status_title = "模型已生成最终结果"
                latest_status_detail = "- 正在发送最终回复。"
        elif event_type == "response_retry" and isinstance(payload, Mapping):
            latest_status_title = "模型空回复，正在重试"
            latest_status_detail = (
                f"- 原因：{str(payload.get('reason', '')).strip() or '模型未生成文本。'}"
            )
        elif event_type == "tool_call" and isinstance(payload, list):
            tool_names = [
                str(tool_call.get("name", "")).strip()
                for tool_call in payload
                if isinstance(tool_call, Mapping)
                and str(tool_call.get("name", "")).strip()
            ]
            if len(tool_names) == 1:
                latest_status_title = f"正在执行工具 `{tool_names[0]}`"
            elif tool_names:
                latest_status_title = f"正在执行 `{len(tool_names)}` 个工具"
                latest_status_detail = (
                    "- 工具列表："
                    + "、".join(f"`{tool_name}`" for tool_name in tool_names[:3])
                )
        elif event_type == "tool_result" and isinstance(payload, Mapping):
            tool_name = str(payload.get("tool_name", "")).strip() or "unknown"
            latest_status_title = f"已完成工具 `{tool_name}`"
            latest_status_detail = "- 正在继续整理结果。"
        elif event_type == "approval_request" and isinstance(payload, Mapping):
            tool_name = str(payload.get("tool_name", "")).strip() or "unknown"
            risk = str(payload.get("risk", "")).strip() or "unknown"
            latest_status_title = f"等待审批 `{tool_name}`"
            latest_status_detail = f"- 风险级别：`{risk}`"
        elif event_type == "approval_result" and isinstance(payload, Mapping):
            tool_name = str(payload.get("tool_name", "")).strip() or "unknown"
            approved = bool(payload.get("approved", False))
            latest_status_title = (
                f"审批已通过 `{tool_name}`"
                if approved
                else f"审批已拒绝 `{tool_name}`"
            )
            reason = str(payload.get("reason", "")).strip()
            if reason:
                latest_status_detail = f"- 说明：{reason}"
        elif event_type.startswith("pipeline.") and isinstance(payload, Mapping):
            steps = FeishuTraceCollector.build_steps(event_type, payload)
            if steps:
                latest_status_title = steps[-1].title
                latest_status_detail = steps[-1].detail

        with self._lock:
            if self._closed:
                return
            now = time.monotonic()
            if self._started_at <= 0.0:
                self._started_at = now
            self._last_activity_at = now
            if latest_status_title:
                self._latest_status_title = latest_status_title
                self._latest_status_detail = latest_status_detail

    def _ensure_heartbeat_thread_started(self) -> None:
        """只在需要时启动一个轻量心跳线程，用于长时间静默时补充状态。"""
        with self._lock:
            if self._closed:
                return
            if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
                return
            self._heartbeat_stop_event.clear()
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                name="feishu-progress-heartbeat",
                daemon=True,
            )
            heartbeat_thread = self._heartbeat_thread
        heartbeat_thread.start()

    def _heartbeat_loop(self) -> None:
        from .webhook import (
            FEISHU_PROGRESS_HEARTBEAT_IDLE_SECONDS,
            FEISHU_PROGRESS_HEARTBEAT_POLL_SECONDS,
        )
        while not self._heartbeat_stop_event.wait(FEISHU_PROGRESS_HEARTBEAT_POLL_SECONDS):
            heartbeat_step = self._build_heartbeat_step()
            if heartbeat_step is not None:
                self._emit_step(heartbeat_step)

    def _build_heartbeat_step(self) -> FeishuTraceStep | None:
        """仅当长时间没有新事件时才补一条心跳，避免飞书侧长时间静默。"""
        from .webhook import FEISHU_PROGRESS_HEARTBEAT_IDLE_SECONDS  # 延迟导入，支持测试 patch

        with self._lock:
            if self._closed or self._started_at <= 0.0:
                return None
            now = time.monotonic()
            if now - self._last_activity_at < FEISHU_PROGRESS_HEARTBEAT_IDLE_SECONDS:
                return None
            elapsed_seconds = now - self._started_at
            detail_lines = [f"- 已持续运行：`{self._format_elapsed_seconds(elapsed_seconds)}`"]
            if self._latest_status_title:
                detail_lines.append(f"- 最近状态：{self._latest_status_title}")
            if self._latest_status_detail:
                detail_lines.append(self._latest_status_detail)
        return FeishuTraceStep(
            kind="heartbeat",
            title="任务仍在执行中",
            detail="\n".join(detail_lines),
        )

    @staticmethod
    def _build_input_preview(user_input: str) -> str:
        normalized_text = re.sub(r"\s+", " ", user_input).strip()
        if len(normalized_text) <= FEISHU_PROGRESS_INPUT_PREVIEW_MAX_CHARS:
            return normalized_text
        return normalized_text[:FEISHU_PROGRESS_INPUT_PREVIEW_MAX_CHARS].rstrip() + "..."

    @staticmethod
    def _format_elapsed_seconds(elapsed_seconds: float) -> str:
        if elapsed_seconds < 60:
            return f"{elapsed_seconds:.1f}s"
        minutes, seconds = divmod(int(elapsed_seconds), 60)
        return f"{minutes}m{seconds:02d}s"

def _get_feishu_command_description(command: str) -> str:
    """返回飞书侧可见命令的说明，兼容 CLI 内建命令与飞书扩展命令。"""
    from .webhook import _get_builtin_command_description  # 延迟导入避免循环依赖

    normalized_command = command.strip().lower()
    return FEISHU_SESSION_COMMAND_DESCRIPTIONS.get(
        normalized_command,
        _get_builtin_command_description(command),
    )

def _build_feishu_chat_scope_id(chat_id: str) -> str:
    """为单个飞书聊天构造稳定的会话分组标识。"""
    return f"feishu-chat:{chat_id.strip()}"

def _build_feishu_session_state_path(base_dir: Path | None = None) -> Path:
    """返回飞书活动会话索引文件路径。"""
    return get_session_storage_dir(base_dir) / FEISHU_SESSION_STATE_FILENAME

def _load_feishu_session_state(base_dir: Path | None = None) -> dict[str, object]:
    """加载飞书活动会话索引；损坏时回退到空结构。"""
    state_path = _build_feishu_session_state_path(base_dir)
    if not state_path.exists():
        return {"version": 1, "chats": {}}
    try:
        raw_payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "chats": {}}
    if not isinstance(raw_payload, dict):
        return {"version": 1, "chats": {}}
    chat_payload = raw_payload.get("chats")
    if not isinstance(chat_payload, dict):
        raw_payload["chats"] = {}
    raw_payload.setdefault("version", 1)
    return raw_payload

def _save_feishu_session_state(
    payload: Mapping[str, object],
    base_dir: Path | None = None,
) -> None:
    """落盘飞书活动会话索引。"""
    state_path = _build_feishu_session_state_path(base_dir)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

def _build_feishu_session_entry(
    session_key: str,
    *,
    label: str = "",
    created_at: str | None = None,
) -> dict[str, str]:
    """构造单条飞书会话索引记录。"""
    from .webhook import build_webhook_session_id  # 延迟导入避免循环依赖

    return {
        "session_key": session_key,
        "session_id": build_webhook_session_id("feishu", session_key),
        "label": label.strip(),
        "created_at": created_at or datetime.now().astimezone().isoformat(),
    }

def _build_feishu_text_message_payload(text: str) -> dict[str, object]:
    """构造飞书文本消息体，复用于 reply_api 与 create_api。"""
    return {
        "msg_type": "text",
        "content": json.dumps(
            {"text": text},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }

def _truncate_feishu_markdown(text: str, *, max_chars: int = FEISHU_CARD_MARKDOWN_MAX_CHARS) -> str:
    """控制飞书卡片正文长度，避免超出单条消息体积限制。"""
    normalized_text = text.strip()
    if len(normalized_text) <= max_chars:
        return normalized_text
    return normalized_text[:max_chars].rstrip() + "\n\n... 内容较长，已截断。"

def _trim_feishu_list_items(
    items: Sequence[str],
    *,
    limit: int = FEISHU_CARD_LIST_LIMIT,
) -> list[str]:
    """限制飞书卡片中的列表长度，避免单条命令输出过长。"""
    normalized_items = [item.strip() for item in items if item.strip()]
    if len(normalized_items) <= limit:
        return normalized_items
    remaining_count = len(normalized_items) - limit
    return [
        *normalized_items[:limit],
        f"其余 {remaining_count} 项未展开，请在 CLI 中查看完整结果。",
    ]

def _build_feishu_markdown_section(title: str, lines: Sequence[str]) -> str:
    """按飞书 markdown 习惯构造一个简洁分节。"""
    normalized_lines = [line.rstrip() for line in lines if line.strip()]
    if not normalized_lines:
        return ""
    return f"**{title}**\n" + "\n".join(normalized_lines)

def _normalize_feishu_table_cell(text: str) -> str:
    """清洗飞书 markdown 表格单元格，避免换行和分隔符打乱布局。"""
    normalized_text = re.sub(r"\s+", " ", text).strip()
    if not normalized_text:
        return " "
    return normalized_text.replace("|", "\\|")

def _build_feishu_markdown_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> str:
    """把表格数据降级成飞书稳定支持的记录块样式。"""
    normalized_headers = [_normalize_feishu_table_cell(header) for header in headers]
    normalized_rows = [
        [_normalize_feishu_table_cell(cell) for cell in row]
        for row in rows
        if row
    ]
    if not normalized_headers or not normalized_rows:
        return ""
    if len(normalized_headers) == 2 and normalized_headers[0] == "工具":

        return "\n\n".join(
            f"**{cells[0]}**\n{cells[1]}"
            for cells in normalized_rows
            if len(cells) >= 2
        )
    if len(normalized_headers) == 2 and normalized_headers[0] == "依赖":
        return "\n".join(
            f"- **{cells[0]}**：{cells[1]}"
            for cells in normalized_rows
            if len(cells) >= 2
        )
    if "会话ID" in normalized_headers:
        record_blocks: list[str] = []
        for index, cells in enumerate(normalized_rows, start=1):
            row_mapping = {
                header: cells[position]
                for position, header in enumerate(normalized_headers)
                if position < len(cells)
            }
            title = row_mapping.get("标题") or row_mapping.get("会话ID") or f"记录 {index}"
            detail_lines = [f"**{index}. {title}**"]
            for header, cell in row_mapping.items():
                if header == "标题" and cell == title:
                    continue
                detail_lines.append(f"- {header}：{cell}")
            record_blocks.append("\n".join(detail_lines))
        return "\n\n".join(record_blocks)
    return "\n\n".join(
        "\n".join(
            [f"**记录 {index}**"]
            + [
                f"- {header}：{cells[position]}"
                for position, header in enumerate(normalized_headers)
                if position < len(cells)
            ]
        )
        for index, cells in enumerate(normalized_rows, start=1)
    )

def _build_feishu_key_value_table(
    rows: Sequence[tuple[str, str]],
    *,
    headers: tuple[str, str] = ("字段", "值"),
) -> str:
    """把常见键值对摘要统一渲染成飞书稳定支持的键值列表。"""
    _ = headers
    normalized_rows = [
        (_normalize_feishu_table_cell(key), _normalize_feishu_table_cell(value))
        for key, value in rows
        if key.strip() and value.strip()
    ]
    if not normalized_rows:
        return ""
    return "\n".join(
        f"- **{key}**：{value}"
        for key, value in normalized_rows
    )

def _build_feishu_tool_result_key_value_detail(text: str) -> str:
    """把工具返回中的 JSON 对象或多行键值结果转换成飞书可读键值块。"""
    rows = _extract_feishu_tool_result_json_rows(text)
    if not rows:
        rows = _extract_feishu_tool_result_line_rows(text)
    if not rows:
        return ""

    visible_rows = rows[:FEISHU_TOOL_RESULT_KEY_VALUE_MAX_ROWS]
    detail = _build_feishu_key_value_table(visible_rows)
    hidden_count = len(rows) - len(visible_rows)
    if hidden_count > 0:
        detail = f"{detail}\n- 其余 `{hidden_count}` 项未展开。"
    return detail

def _extract_feishu_tool_result_json_rows(text: str) -> list[tuple[str, str]]:
    """识别工具输出中的 JSON 对象，常见于结构化扫描类工具。"""
    normalized_text = text.strip()
    if not normalized_text:
        return []
    try:
        parsed = json.loads(normalized_text)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, Mapping):
        return []
    rows: list[tuple[str, str]] = []
    for raw_key, raw_value in parsed.items():
        key = str(raw_key).strip()
        if not _is_feishu_tool_result_key(key, raw_value):
            continue
        rows.append((key, _format_feishu_tool_result_value(raw_value)))
    return rows

def _extract_feishu_tool_result_line_rows(text: str) -> list[tuple[str, str]]:
    """识别普通文本里的多行 `键: 值` / `键=值` 结果，避免误判表格输出。"""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return []

    rows: list[tuple[str, str]] = []
    for raw_line in lines:
        line = raw_line.lstrip("-*•").strip()
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", line):
            continue
        match = re.match(
            rf"^(.{{1,{FEISHU_TOOL_RESULT_KEY_MAX_CHARS}}}?)\s*(?:[:：=])\s*(.*)$",
            line,
        )
        if not match:
            continue
        key = match.group(1).strip()
        value = match.group(2).strip()
        if not _is_feishu_tool_result_key(key, value):
            continue
        rows.append((key, _format_feishu_tool_result_value(value)))

    # 至少两行键值才认为是结构化结果，避免把单行日志误转成卡片字段。
    if len(rows) < 2:
        return []
    return rows

def _is_feishu_tool_result_key(key: str, value: object) -> bool:
    """过滤路径、URL、代码片段等容易被 `:` 误拆的文本。"""
    normalized_key = key.strip()
    if not normalized_key or len(normalized_key) > FEISHU_TOOL_RESULT_KEY_MAX_CHARS:
        return False
    if "|" in normalized_key or "```" in normalized_key:
        return False
    normalized_value = str(value).strip()
    if len(normalized_key) == 1 and normalized_value.startswith(("\\", "/")):
        return False
    if re.search(r"\s{3,}", normalized_key):
        return False
    return True

def _format_feishu_tool_result_value(value: object) -> str:
    """压缩工具结果值，避免单个字段把飞书卡片撑得过长。"""
    if value is None:
        normalized_value = "无"
    elif isinstance(value, (dict, list, tuple)):
        try:
            normalized_value = json.dumps(value, ensure_ascii=False)
        except TypeError:
            normalized_value = str(value)
    else:
        normalized_value = str(value)
    normalized_value = re.sub(r"\s+", " ", normalized_value).strip() or "（空）"
    if len(normalized_value) <= FEISHU_TOOL_RESULT_VALUE_MAX_CHARS:
        return normalized_value
    return normalized_value[:FEISHU_TOOL_RESULT_VALUE_MAX_CHARS].rstrip() + "..."

def _parse_feishu_tool_entries(tool_lines: Sequence[str]) -> list[tuple[str, str]]:
    """把 `工具名: 描述` 的工具摘要拆成两列，便于飞书中按表格展示。"""
    tool_entries: list[tuple[str, str]] = []
    for tool_line in tool_lines:
        normalized_line = re.sub(r"\s+", " ", tool_line).strip()
        if not normalized_line:
            continue
        raw_name, separator, raw_description = normalized_line.partition(":")
        tool_name = raw_name.strip() or "unknown"
        if separator:
            description = raw_description.strip() or "（暂无说明）"
        else:
            description = "（暂无说明）"
        tool_entries.append((tool_name, description))
    return tool_entries

def _trim_feishu_preview_lines(
    lines: Sequence[str],
    *,
    limit: int,
) -> list[str]:
    """限制预览区行数，避免单张卡片被超长上下文撑爆。"""
    normalized_lines = [line.strip() for line in lines if line.strip()]
    if len(normalized_lines) <= limit:
        return normalized_lines
    hidden_count = len(normalized_lines) - limit
    return [*normalized_lines[:limit], f"... 其余 {hidden_count} 行未展开。"]

def _resolve_feishu_command_button_spec(
    command_spec: FeishuCommandButtonSpec,
) -> tuple[str, str]:
    """将按钮显示文案与实际命令归一化，便于同一套按钮生成器复用。"""
    if isinstance(command_spec, tuple):
        label, command = command_spec
    else:
        label = command_spec
        command = command_spec
    return label.strip(), command.strip()

def _build_feishu_command_button(
    command_spec: FeishuCommandButtonSpec,
    *,
    primary: bool = False,
) -> dict[str, object]:
    """统一构造飞书命令按钮，便于 /start 与卡片菜单复用。"""
    label, command = _resolve_feishu_command_button_spec(command_spec)
    return {
        "tag": "button",
        "type": "primary" if primary else "default",
        "text": {
            "tag": "plain_text",
            "content": label,
        },
        "value": {
            "command": command,
        },
    }

def _build_feishu_command_action_rows(
    commands: Sequence[FeishuCommandButtonSpec],
    *,
    primary_commands: Sequence[str] = (),
    row_size: int = 4,
) -> list[dict[str, object]]:
    """将命令列表切分成飞书卡片按钮行。"""
    primary_command_set = {command.strip() for command in primary_commands}
    action_rows: list[dict[str, object]] = []
    normalized_command_specs = [
        (label, command)
        for label, command in (
            _resolve_feishu_command_button_spec(command_spec)
            for command_spec in commands
        )
        if label and command
    ]
    for start_index in range(0, len(normalized_command_specs), max(row_size, 1)):
        row_commands = normalized_command_specs[start_index : start_index + max(row_size, 1)]
        action_rows.append(
            {
                "tag": "action",
                "actions": [
                    _build_feishu_command_button(
                        (label, command),
                        primary=command in primary_command_set,
                    )
                    for label, command in row_commands
                ],
            }
        )
    return action_rows

def _build_feishu_interactive_card_payload(
    title: str,
    body_markdown: str,
    *,
    template: str = "blue",
    action_rows: Sequence[dict[str, object]] = (),
) -> dict[str, object]:
    """统一构造飞书交互卡片，减少各命令分支重复拼接 JSON。"""
    return _build_feishu_interactive_card_elements_payload(
        title,
        [
            {
                "tag": "markdown",
                "content": _truncate_feishu_markdown(body_markdown),
            }
        ],
        template=template,
        action_rows=action_rows,
    )

def _build_feishu_interactive_card_elements_payload(
    title: str,
    body_elements: Sequence[dict[str, object]],
    *,
    template: str = "blue",
    action_rows: Sequence[dict[str, object]] = (),
) -> dict[str, object]:
    """支持多段 markdown 元素的飞书交互卡片构造器。"""
    elements: list[dict[str, object]] = [
        dict(element)
        for element in body_elements
    ]
    elements.extend(dict(row) for row in action_rows)
    card_payload = {
        "config": {
            "wide_screen_mode": True,
        },
        "header": {
            "title": {
                "tag": "plain_text",
                "content": title,
            },
            "template": template,
        },
        "elements": elements,
    }
    return {
        "msg_type": "interactive",
        "content": json.dumps(
            card_payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }

def _should_use_feishu_rich_reply(reply_text: str) -> bool:
    """飞书中的普通 AI 回复统一走卡片，保证 markdown 一致渲染。"""
    return bool(reply_text.strip())

def _normalize_ai_reply_markdown_for_feishu(reply_text: str) -> str:
    """将普通 AI 回复整理为更适合飞书 markdown 卡片的文本。"""
    normalized_text = reply_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized_text:
        return "（空回复）"
    normalized_lines: list[str] = []
    in_code_block = False
    previous_line_blank = False
    for raw_line in normalized_text.splitlines():
        stripped_line = raw_line.rstrip()
        compact_line = stripped_line.strip()
        if compact_line.startswith("```"):
            fence_language = compact_line[3:].strip()
            if in_code_block:
                normalized_lines.append("```")
                in_code_block = False
            else:
                normalized_lines.append(f"```{fence_language}" if fence_language else "```")
                in_code_block = True
            previous_line_blank = False
            continue
        if in_code_block:
            normalized_lines.append(stripped_line)
            previous_line_blank = False
            continue
        if not compact_line:
            if normalized_lines and not previous_line_blank:
                normalized_lines.append("")
                previous_line_blank = True
            continue
        previous_line_blank = False
        if compact_line.startswith("#"):
            heading_text = compact_line.lstrip("#").strip()
            if heading_text:
                normalized_lines.append(f"**{heading_text}**")
                continue
        if re.match(r"^[-*+]\s+", compact_line):
            normalized_lines.append("- " + re.sub(r"^[-*+]\s+", "", compact_line))
            continue
        if re.match(r"^\d+[)]\s+", compact_line):
            normalized_lines.append(re.sub(r"^(\d+)[)]\s+", r"\1. ", compact_line))
            continue
        if re.fullmatch(r"[-=_]{3,}", compact_line):
            if normalized_lines and normalized_lines[-1] != "":
                normalized_lines.append("")
            continue
        normalized_lines.append(compact_line)
    if in_code_block:
        normalized_lines.append("```")
    return "\n".join(normalized_lines).strip()

def _extract_feishu_reply_title(reply_markdown: str) -> str:
    """尽量从回复正文里提取一个短标题，避免卡片永远都叫 AI 回复。"""
    def _normalize_title(candidate_title: str) -> str:
        return candidate_title.strip().rstrip("：:").strip()

    for raw_line in reply_markdown.splitlines():
        compact_line = raw_line.strip()
        if not compact_line or compact_line == "```":
            continue
        if compact_line.startswith("**") and compact_line.endswith("**") and len(compact_line) > 4:
            candidate_title = _normalize_title(compact_line[2:-2])
            if candidate_title:
                return candidate_title[:32]
        if compact_line.startswith(("- ", "> ", "```")):
            continue
        if re.match(r"^\d+\.\s+", compact_line):
            continue
        return _normalize_title(compact_line)[:32] or "AI 回复"
    return "AI 回复"

def _split_long_text_for_feishu(
    text: str,
    *,
    max_chars: int,
) -> list[str]:

    """按长度切分超长文本，优先在空格或换行处断开。"""
    normalized_text = text.strip()
    if len(normalized_text) <= max_chars:
        return [normalized_text]
    chunks: list[str] = []
    remaining_text = normalized_text
    while len(remaining_text) > max_chars:
        split_index = remaining_text.rfind("\n", 0, max_chars)
        if split_index < max_chars // 3:
            split_index = remaining_text.rfind(" ", 0, max_chars)
        if split_index < max_chars // 3:
            split_index = max_chars
        chunks.append(remaining_text[:split_index].rstrip())
        remaining_text = remaining_text[split_index:].lstrip()
    if remaining_text:
        chunks.append(remaining_text)
    return [chunk for chunk in chunks if chunk]

def _split_large_feishu_block(
    block: str,
    *,
    max_chars: int,
) -> list[str]:
    """拆分单个过长段落，避免飞书 markdown 元素过大。"""
    stripped_block = block.strip()
    if len(stripped_block) <= max_chars:
        return [stripped_block]
    if stripped_block.startswith("```") and stripped_block.endswith("```"):
        block_lines = stripped_block.splitlines()
        opening_fence = block_lines[0].strip() or "```"
        closing_fence = "```"
        code_lines = block_lines[1:-1]
        segments: list[str] = []
        current_code_lines: list[str] = []
        for code_line in code_lines:
            candidate_lines = [*current_code_lines, code_line]
            candidate_block = "\n".join([opening_fence, *candidate_lines, closing_fence]).strip()
            if len(candidate_block) > max_chars and current_code_lines:
                segments.append(
                    "\n".join([opening_fence, *current_code_lines, closing_fence]).strip()
                )
                current_code_lines = [code_line]
                continue
            current_code_lines = candidate_lines
        if current_code_lines:
            segments.append(
                "\n".join([opening_fence, *current_code_lines, closing_fence]).strip()
            )
        return segments
    split_blocks: list[str] = []
    current_lines: list[str] = []
    for line in stripped_block.splitlines():
        candidate_lines = [*current_lines, line]
        candidate_block = "\n".join(candidate_lines).strip()
        if len(candidate_block) > max_chars and current_lines:
            split_blocks.append("\n".join(current_lines).strip())
            current_lines = [line]
            continue
        current_lines = candidate_lines
    if current_lines:
        split_blocks.append("\n".join(current_lines).strip())
    final_blocks: list[str] = []
    for split_block in split_blocks:
        if len(split_block) <= max_chars:
            final_blocks.append(split_block)
            continue
        final_blocks.extend(_split_long_text_for_feishu(split_block, max_chars=max_chars))
    return final_blocks

def _split_feishu_markdown_blocks(
    reply_markdown: str,
    *,
    max_chars: int = FEISHU_RICH_REPLY_CHUNK_MAX_CHARS,
    max_chunks: int = FEISHU_RICH_REPLY_MAX_CHUNKS,
) -> list[str]:
    """按段落和代码块切分飞书回复内容，提升长回答可读性。"""
    source_blocks: list[str] = []
    current_lines: list[str] = []
    in_code_block = False
    for raw_line in reply_markdown.splitlines():
        compact_line = raw_line.rstrip()
        if compact_line.strip().startswith("```"):
            if not in_code_block and current_lines:
                source_blocks.append("\n".join(current_lines).strip())
                current_lines = []
            current_lines.append(compact_line)
            in_code_block = not in_code_block
            if not in_code_block:
                source_blocks.append("\n".join(current_lines).strip())
                current_lines = []
            continue
        if in_code_block:
            current_lines.append(compact_line)
            continue
        if not compact_line.strip():
            if current_lines:
                source_blocks.append("\n".join(current_lines).strip())
                current_lines = []
            continue
        current_lines.append(compact_line)
    if current_lines:
        source_blocks.append("\n".join(current_lines).strip())

    expanded_blocks: list[str] = []
    for block in source_blocks:
        expanded_blocks.extend(_split_large_feishu_block(block, max_chars=max_chars))

    packed_chunks: list[str] = []
    current_chunk = ""
    for block in expanded_blocks:
        candidate_chunk = block if not current_chunk else f"{current_chunk}\n\n{block}"
        if len(candidate_chunk) <= max_chars:
            current_chunk = candidate_chunk
            continue
        if current_chunk:
            packed_chunks.append(current_chunk)
        current_chunk = block
    if current_chunk:
        packed_chunks.append(current_chunk)

    if len(packed_chunks) <= max_chunks:
        return packed_chunks
    visible_chunks = packed_chunks[: max_chunks - 1]
    hidden_chunk_count = len(packed_chunks) - len(visible_chunks)
    visible_chunks.append(
        f"_内容较长，剩余 {hidden_chunk_count} 段未在飞书中展开。_"
    )
    return visible_chunks

def _looks_like_feishu_error_reply(reply_text: str) -> bool:
    """根据回复文案挑选更合适的卡片颜色。"""
    normalized_text = reply_text.strip()
    return normalized_text.startswith(("运行失败：", "处理失败："))

def _build_feishu_ai_reply_payload(
    reply_text: str,
    *,
    trace_steps: Sequence[FeishuTraceStep] = (),
) -> dict[str, object] | None:
    """为普通 AI 回复构造统一的飞书 markdown 卡片。"""
    if not _should_use_feishu_rich_reply(reply_text):
        return None
    reply_markdown = _normalize_ai_reply_markdown_for_feishu(reply_text)
    title = _extract_feishu_reply_title(reply_markdown)
    content_chunks = _split_feishu_markdown_blocks(reply_markdown)
    body_elements: list[dict[str, object]] = _build_feishu_trace_elements(trace_steps)
    if body_elements:
        body_elements.append(
            {
                "tag": "markdown",
                "content": "**最终结果**",
            }
        )
    if len(content_chunks) > 1:
        body_elements.append(
            {
                "tag": "markdown",
                "content": f"_内容较长，已分为 {len(content_chunks)} 段展示。_",
            }
        )
    body_elements.extend(
        {
            "tag": "markdown",
            "content": _truncate_feishu_markdown(chunk, max_chars=FEISHU_RICH_REPLY_CHUNK_MAX_CHARS),
        }
        for chunk in content_chunks
    )
    return _build_feishu_interactive_card_elements_payload(
        title,
        body_elements,
        template="red" if _looks_like_feishu_error_reply(reply_text) else "blue",
    )

def _normalize_cli_output_for_feishu(output: str) -> str:
    """去掉 Rich 面板边框与多余空白，避免把终端装饰原样发到飞书。"""
    normalized_lines: list[str] = []
    previous_line_blank = False
    for raw_line in output.splitlines():
        stripped_line = raw_line.rstrip()
        if not stripped_line.strip():
            if normalized_lines and not previous_line_blank:
                normalized_lines.append("")
                previous_line_blank = True
            continue
        previous_line_blank = False
        panel_match = FEISHU_RICH_PANEL_EDGE_RE.match(stripped_line)
        candidate_line = (
            panel_match.group(1).rstrip()
            if panel_match is not None
            else stripped_line.strip()
        )
        if (
            candidate_line
            and not FEISHU_BOX_DRAWING_LINE_RE.fullmatch(candidate_line)
            and re.search(r"[\u2500-\u257F\u2580-\u259F]", candidate_line)
        ):
            candidate_line = re.sub(
                r"^[\s\u2500-\u257F\u2580-\u259F]+",
                "",
                candidate_line,
            )
            candidate_line = re.sub(
                r"[\s\u2500-\u257F\u2580-\u259F]+$",
                "",
                candidate_line,
            )
        if not candidate_line:
            continue
        if FEISHU_BOX_DRAWING_LINE_RE.fullmatch(candidate_line):
            continue
        normalized_lines.append(candidate_line)
    return _truncate_feishu_markdown("\n".join(normalized_lines).strip())

def _escape_feishu_code_block(text: str) -> str:
    """避免兜底代码块中的围栏与飞书 markdown 语法冲突。"""
    return text.replace("```", "'''")

def _build_feishu_trace_elements(
    trace_steps: Sequence[FeishuTraceStep],
) -> list[dict[str, object]]:
    """把中间处理步骤转换成飞书卡片 markdown 元素。"""
    if not trace_steps:
        return []
    visible_steps = list(trace_steps[:FEISHU_TRACE_MAX_STEPS])
    tool_call_count = sum(step.kind == "tool_call" for step in trace_steps)
    summary_lines = [
        f"- 中间步骤：`{len(trace_steps)}`",
        f"- 工具调用：`{tool_call_count}`",
    ]
    hidden_step_count = len(trace_steps) - len(visible_steps)
    if hidden_step_count > 0:
        summary_lines.append(
            f"- 仅展示前 `{len(visible_steps)}` 条，剩余 `{hidden_step_count}` 条未展开。"
        )

    elements: list[dict[str, object]] = [
        {
            "tag": "markdown",
            "content": _build_feishu_markdown_section("处理过程", summary_lines),
        }
    ]
    for index, step in enumerate(visible_steps, start=1):
        step_markdown = f"**步骤 {index} · {step.title}**"
        if step.detail:
            step_markdown = f"{step_markdown}\n{step.detail}"
        elements.append(
            {
                "tag": "markdown",
                "content": _truncate_feishu_markdown(
                    step_markdown,
                    max_chars=FEISHU_RICH_REPLY_CHUNK_MAX_CHARS,
                ),
            }
        )
    return elements

def _resolve_feishu_progress_template(step: FeishuTraceStep) -> str:
    """按步骤类型选择更容易区分的飞书卡片配色。"""
    if step.kind == "start":
        return "blue"
    if step.kind == "heartbeat":
        return "orange"
    if step.kind == "pipeline_error":
        return "red"
    if step.kind in {"pipeline_done", "pipeline_role_done", "pipeline_subtask_done"}:
        return "green"
    if step.kind in {"pipeline_role_error", "pipeline_subtask_fail"}:
        return "red"
    if step.kind.startswith("pipeline_role_"):
        return "indigo"
    if step.kind.startswith("pipeline_subtask_"):
        return "indigo"
    if step.kind in {
        "pipeline_start",
        "pipeline_iteration",
        "pipeline_subtasks",
        "pipeline_scheduler",
    }:
        return "blue"
    if step.kind in {"pipeline_parallel", "pipeline_tool_call"}:
        return "indigo"
    if step.kind == "pipeline_tool_result":
        return "turquoise"
    if step.kind == "tool_call":
        return "indigo"
    if step.kind == "tool_result":
        return "turquoise"
    if step.kind == "approval_request":
        return "orange"
    if step.kind == "approval_result":
        return "green" if "已批准" in step.detail else "red"
    return "blue"

def _build_feishu_progress_payload(
    step: FeishuTraceStep,
    *,
    step_index: int,
) -> dict[str, object]:
    """为单条处理中间步骤构造独立飞书消息。"""
    if step.kind in {"start", "heartbeat"}:
        body_lines: list[str] = []
    else:
        body_lines = [f"- 步骤序号：`{step_index}`"]
    if step.detail:
        body_lines.append(step.detail)
    return _build_feishu_interactive_card_payload(
        f"处理中 · {step.title}",
        "\n\n".join(body_lines),
        template=_resolve_feishu_progress_template(step),
    )

def _looks_like_builtin_error(output: str) -> bool:
    """根据命令输出中的显式提示词判断是否需要把错误结果优先展示。"""
    normalized_output = output.strip()
    if not normalized_output:
        return False
    error_keywords = (
        "错误",
        "失败",
        "不支持",
        "请提供",
        "未找到",
        "缺少",

        "为空",
        "不存在",
        "无法",
    )
    return any(keyword in normalized_output for keyword in error_keywords)

def _build_feishu_notice_payload(
    title: str,
    message: str,
    *,
    template: str = "green",
    button_commands: Sequence[str] = (),
) -> dict[str, object]:
    """构造适合提示类命令结果的简洁卡片。"""
    action_rows = _build_feishu_command_action_rows(
        button_commands,
        primary_commands=button_commands[:1],
    )
    return _build_feishu_interactive_card_payload(
        title,
        message,
        template=template,
        action_rows=action_rows,
    )

def _build_feishu_help_payload() -> dict[str, object]:
    """构造飞书版帮助卡片，按主题分组展示所有内建命令。"""
    command_groups: tuple[tuple[str, tuple[str, ...]], ...] = (
        (
            "快捷入口",
            ("/help", "/tools", "/status", "/start"),
        ),
        (
            "会话与历史",
            (
                "/context",
                "/context clear",
                "/history",
                "/history show <会话ID>",
                "/history load <会话ID>",
                "/history search <关键词>",
                "/history export <会话ID> [路径]",
                "/clear",
                "/exit",
            ),
        ),
        (
            "模型与模式",
            (
                "/mode",
                "/mode standard",
                "/mode authorized",
                "/service",
                "/service <服务商>",
                "/model",
                "/model <模型名>",
            ),
        ),
        (
            "配置与权限",
            (
                "/config",
                "/config allow-path",
                "/config allow-path add <目录>",
                "/allow-path",
                "/allow-path add <目录>",
                "/approval",
                "/approval prompt",
                "/approval auto",
                "/approval never",
            ),
        ),
        (
            "诊断与控制",
            ("/doctor", "/version", "/stop"),
        ),
        (
            "飞书会话",
            (
                "/session",
                "/session new",
                "/session list",
                "/session default",
                "/session use <会话ID|序号>",

            ),
        ),
    )
    sections = [
        "发送 `/start` 可打开按钮菜单；也可以直接像 CLI 一样输入命令。",
    ]
    for title, commands in command_groups:
        section_lines = [
            f"- `{command}` {(_get_feishu_command_description(command) or '待补充说明').strip()}"
            for command in commands
        ]
        sections.append(_build_feishu_markdown_section(title, section_lines))
    return _build_feishu_interactive_card_payload(
        "内建命令",
        "\n\n".join(section for section in sections if section),
        template="blue",
        action_rows=[
            *_build_feishu_command_action_rows(
                FEISHU_START_MENU_COMMANDS,
                primary_commands=("/help",),
            ),
            *_build_feishu_command_action_rows(
                FEISHU_SESSION_SHORTCUT_COMMANDS,
                primary_commands=("/session current",),
                row_size=3,
            ),
        ],
    )

def _build_feishu_tools_payload(runner: "AgentRunner") -> dict[str, object]:
    """构造飞书版工具列表卡片。"""
    tool_entries = _parse_feishu_tool_entries(describe_tool_instances(runner.tools))
    visible_tool_entries = tool_entries[:FEISHU_CARD_LIST_LIMIT]
    hidden_tool_count = len(tool_entries) - len(visible_tool_entries)
    tool_table = _build_feishu_markdown_table(
        ("工具", "说明"),
        [
            (f"`{tool_name}`", description)
            for tool_name, description in visible_tool_entries
        ],
    )
    summary_lines = [f"- 工具总数：`{len(tool_entries)}`"]
    if hidden_tool_count > 0:
        summary_lines.append(
            f"- 当前仅展示前 `{len(visible_tool_entries)}` 个，其余 `{hidden_tool_count}` 个未展开。"
        )
    sections = [
        f"当前默认工具共 **{len(tool_entries)}** 个。",
        _build_feishu_markdown_section("概览", summary_lines),
        (
            _build_feishu_markdown_section("工具列表", [tool_table])
            if tool_table
            else _build_feishu_markdown_section("工具列表", ["- 当前没有默认工具。"])
        ),
    ]
    return _build_feishu_interactive_card_payload(
        "默认工具",
        "\n\n".join(section for section in sections if section),
        template="turquoise",
        action_rows=_build_feishu_command_action_rows(("/status", "/help")),
    )

def _build_feishu_context_payload(
    runner: "AgentRunner",
    runtime_context: Mapping[str, object],
) -> dict[str, object]:
    """把 `/context` 渲染成上下文摘要卡片。"""
    diagnostics = runner.get_context_diagnostics()
    overview_table = _build_feishu_key_value_table(
        (
            ("当前会话 ID", str(runtime_context.get("session_id", "") or "未分配")),
            ("消息数", str(diagnostics.get("history_message_count", 0))),
            ("用户轮数", str(runner.get_turn_count())),
            ("来源会话", str(runtime_context.get("session_source_id") or "无")),
            ("模型可见消息", str(diagnostics.get("model_message_count", 0))),
            ("已压缩历史消息", str(diagnostics.get("compressed_message_count", 0))),
        )
    )
    history_preview = _trim_feishu_preview_lines(
        [str(line) for line in diagnostics.get("history_preview", [])],
        limit=FEISHU_CONTEXT_PREVIEW_MAX_LINES,
    )
    model_preview = _trim_feishu_preview_lines(
        [str(line) for line in diagnostics.get("model_preview", [])],
        limit=FEISHU_CONTEXT_PREVIEW_MAX_LINES,
    )
    sections = [
        _build_feishu_markdown_section("概览", [overview_table] if overview_table else []),
        _build_feishu_markdown_section("当前会话预览", [f"- {line}" for line in history_preview]),
    ]
    compressed_summary = str(diagnostics.get("compressed_summary", "")).strip()
    if compressed_summary:
        sections.append(_build_feishu_markdown_section("压缩摘要", [compressed_summary]))
    sections.append(
        _build_feishu_markdown_section(
            "模型实际可见上下文",
            [f"- {line}" for line in model_preview],
        )
    )
    return _build_feishu_interactive_card_payload(
        "当前上下文",
        "\n\n".join(section for section in sections if section),
        template="carmine",
        action_rows=_build_feishu_command_action_rows(("/status", "/history", "/clear")),
    )

def _build_feishu_history_list_payload(base_dir: Path | None = None) -> dict[str, object]:
    """把 `/history` 渲染成历史会话总览表。"""
    stored_sessions = [
        summary
        for summary in list_stored_sessions(base_dir=base_dir)
        if summary.session_id != Path(FEISHU_SESSION_STATE_FILENAME).stem

    ]
    if not stored_sessions:
        return _build_feishu_notice_payload(
            "历史会话",
            "当前工作目录下还没有已保存的历史会话。",
            template="grey",
            button_commands=("/start", "/help"),
        )

    visible_sessions = stored_sessions[:FEISHU_CARD_LIST_LIMIT]
    hidden_count = len(stored_sessions) - len(visible_sessions)
    summary_lines = [f"- 会话总数：`{len(stored_sessions)}`"]
    if hidden_count > 0:
        summary_lines.append(
            f"- 当前仅展示前 `{len(visible_sessions)}` 个，其余 `{hidden_count}` 个未展开。"
        )
    history_table = _build_feishu_markdown_table(
        ("会话ID", "标题", "更新时间", "轮数"),
        [
            (
                f"`{summary.session_id}`",
                summary.title,
                summary.updated_at,
                str(summary.turn_count),
            )
            for summary in visible_sessions
        ],
    )
    sections = [
        _build_feishu_markdown_section("概览", summary_lines),
        _build_feishu_markdown_section("历史会话", [history_table] if history_table else []),
    ]
    return _build_feishu_interactive_card_payload(
        "历史会话",
        "\n\n".join(section for section in sections if section),
        template="wathet",
        action_rows=_build_feishu_command_action_rows(("/history search", "/status", "/help")),
    )

def _build_feishu_history_show_payload(
    session_id: str,
    *,
    base_dir: Path | None = None,
) -> dict[str, object]:
    """把 `/history show` 渲染成摘要加预览。"""
    stored_session = load_session_history(session_id, base_dir=base_dir)
    summary_table = _build_feishu_key_value_table(
        (
            ("会话 ID", stored_session.summary.session_id),
            ("创建时间", stored_session.summary.created_at),
            ("更新时间", stored_session.summary.updated_at),
            ("模式", stored_session.summary.mode),
            ("审批策略", stored_session.summary.approval_policy),
            ("消息数", str(stored_session.summary.message_count)),
            ("用户轮数", str(stored_session.summary.turn_count)),
            ("来源会话", str(stored_session.summary.source_session_id or "无")),
        )
    )
    try:
        from ..agent.runner import format_message_for_context_summary
    except Exception:  # noqa: BLE001 - 预览降级不应影响主流程
        preview_lines = ["消息预览暂不可用。"]
    else:
        preview_lines = [
            format_message_for_context_summary(message)
            for message in stored_session.messages
        ]
    preview_lines = _trim_feishu_preview_lines(
        preview_lines,
        limit=FEISHU_CONTEXT_PREVIEW_MAX_LINES,
    )
    sections = [
        _build_feishu_markdown_section("会话摘要", [summary_table] if summary_table else []),
        _build_feishu_markdown_section("最近消息", [f"- {line}" for line in preview_lines]),
    ]
    return _build_feishu_interactive_card_payload(
        "历史会话详情",
        "\n\n".join(section for section in sections if section),
        template="indigo",
        action_rows=_build_feishu_command_action_rows(("/history", "/context", "/help")),
    )

def _build_feishu_history_search_payload(
    query: str,
    *,
    base_dir: Path | None = None,
) -> dict[str, object]:
    """把 `/history search` 渲染成命中会话表和片段摘要。"""
    search_results = [
        result
        for result in search_stored_sessions(query, base_dir=base_dir)
        if result.session_id != Path(FEISHU_SESSION_STATE_FILENAME).stem
    ]
    if not search_results:
        return _build_feishu_notice_payload(
            "历史检索",
            f"未检索到包含关键词 `{query}` 的历史会话。",
            template="grey",
            button_commands=("/history", "/help"),
        )

    visible_results = search_results[:FEISHU_CARD_LIST_LIMIT]
    history_table = _build_feishu_markdown_table(
        ("会话ID", "标题", "命中消息", "更新时间"),
        [
            (
                f"`{result.session_id}`",
                result.title,
                str(result.matched_message_count),
                result.updated_at,
            )
            for result in visible_results
        ],
    )
    sections = [
        _build_feishu_markdown_section(
            "检索概览",
            [
                f"- 关键词：`{query}`",
                f"- 命中会话：`{len(search_results)}`",
            ],
        ),
        _build_feishu_markdown_section("命中会话", [history_table] if history_table else []),
    ]
    for result in search_results[:FEISHU_HISTORY_EXCERPT_RESULT_LIMIT]:
        excerpt_lines = _trim_feishu_preview_lines(
            [str(line) for line in result.excerpts],
            limit=FEISHU_HISTORY_EXCERPT_LINE_LIMIT,
        )
        if not excerpt_lines:
            continue
        sections.append(
            _build_feishu_markdown_section(
                f"命中片段 · {result.session_id}",
                [f"- {line}" for line in excerpt_lines],
            )
        )
    return _build_feishu_interactive_card_payload(
        f"历史检索：{query}",
        "\n\n".join(section for section in sections if section),
        template="purple",
        action_rows=_build_feishu_command_action_rows(("/history", "/history search", "/help")),
    )

def _build_feishu_history_load_payload(
    session_id: str,
    runner: "AgentRunner",
    runtime_context: Mapping[str, object],
    builtin_output: str,
) -> dict[str, object]:
    """把 `/history load` 渲染成加载成功卡片。"""
    normalized_output = _normalize_cli_output_for_feishu(builtin_output)
    approval_policy = runtime_context.get("approval_policy")
    approval_value = (
        approval_policy.value
        if isinstance(approval_policy, ApprovalPolicy)
        else str(approval_policy or "unknown")
    )
    body_lines = [
        f"- 已加载会话：`{session_id}`",
        f"- 当前模式：`{runner.mode.value}`",
        f"- 当前审批：`{approval_value}`",
        f"- 新会话 ID：`{runtime_context.get('session_id', '') or '未分配'}`",
    ]
    if normalized_output:
        body_lines.append(f"- 结果：{normalized_output}")
    return _build_feishu_notice_payload(
        "已加载历史会话",
        "\n".join(body_lines),
        template="green",
        button_commands=("/context", "/history", "/status"),
    )

def _build_feishu_history_export_payload(
    session_id: str,
    builtin_output: str,
) -> dict[str, object]:
    """把 `/history export` 渲染成导出结果卡片。"""
    normalized_output = _normalize_cli_output_for_feishu(builtin_output) or "历史会话已导出。"
    return _build_feishu_notice_payload(
        "已导出历史会话",
        f"- 会话 ID：`{session_id}`\n- 结果：{normalized_output}",
        template="blue",
        button_commands=("/history", "/status"),
    )

def _build_feishu_doctor_payload(
    runner: "AgentRunner",
    runtime_context: Mapping[str, object],
) -> dict[str, object]:
    """把 `/doctor` 渲染成分块诊断卡片。"""
    from .webhook import build_doctor_payload  # 延迟导入，通过 webhook 支持测试 patch

    payload = build_doctor_payload(runner, dict(runtime_context))
    summary_lines = [f"- 诊断结论：{payload['summary']['status_text']}"]
    reminder_lines = [
        f"- {item}" for item in payload["summary"]["reminders"] if str(item).strip()
    ] or ["- 当前没有额外提醒。"]
    runtime_table = _build_feishu_key_value_table(
        (
            ("项目版本", str(payload["project"]["version"])),
            ("Python", str(payload["project"]["python_version"])),
            ("模式", str(payload["runtime"]["mode_label"])),
            ("审批策略", str(payload["runtime"]["approval_policy_label"])),
            ("界面", str(payload["runtime"]["ui_mode_label"])),
            ("服务", str(payload["runtime"]["service"])),
            ("模型", str(payload["runtime"]["model"])),
            ("模型基址", str(payload["runtime"]["base_url"])),
            (
                "GATEWAY_API_KEY",
                "已配置" if payload["runtime"]["api_key_configured"] else "未配置",
            ),
        )
    )
    dependency_table = _build_feishu_markdown_table(
        ("依赖", "状态"),
        [
            ("`langchain_openai`", str(payload["dependencies"]["langchain_openai"]["status"])),
            ("`langgraph`", str(payload["dependencies"]["langgraph"]["status"])),
            ("`prompt_toolkit`", str(payload["dependencies"]["prompt_toolkit"]["status"])),
            ("`textual`", str(payload["dependencies"]["textual"]["status"])),
            ("`playwright`", str(payload["dependencies"]["playwright"]["status"])),
        ],
    )
    storage_table = _build_feishu_key_value_table(
        (
            ("浏览器搜索", str(payload["search"]["status"])),
            ("本地配置文件", str(payload["storage"]["local_config_path"])),
            ("本地配置状态", str(payload["storage"]["local_config_status"])),
            ("历史会话目录", str(payload["storage"]["session_storage_status"])),
            ("动态能力目录", str(payload["storage"]["capability_storage_status"])),
        )
    )
    sections = [
        _build_feishu_markdown_section("诊断概览", summary_lines),
        _build_feishu_markdown_section("诊断提醒", reminder_lines),
        _build_feishu_markdown_section("运行时", [runtime_table] if runtime_table else []),
        _build_feishu_markdown_section("依赖检查", [dependency_table] if dependency_table else []),
        _build_feishu_markdown_section("存储与能力", [storage_table] if storage_table else []),
        _build_feishu_markdown_section(
            "已保存允许目录",
            [f"- `{line}`" for line in payload["permissions"]["saved_allowed_paths"]]
            or ["- 无"],
        ),
        _build_feishu_markdown_section(
            "允许读取根路径",
            [f"- `{line}`" for line in payload["permissions"]["allowed_roots"]]
            or ["- 无"],
        ),
        _build_feishu_markdown_section(
            "已注册外部工具",
            [f"- `{line}`" for line in payload["permissions"]["registered_tools"]]
            or ["- 无"],
        ),
    ]
    return _build_feishu_interactive_card_payload(
        "运行诊断",
        "\n\n".join(section for section in sections if section),
        template="sunflower",
        action_rows=_build_feishu_command_action_rows(("/status", "/tools", "/help")),
    )

def _build_feishu_status_payload(
    runner: "AgentRunner",
    runtime_context: Mapping[str, object],
) -> dict[str, object]:
    """构造飞书版状态卡片，保留高频排障信息。"""
    approval_policy = runtime_context.get("approval_policy", ApprovalPolicy.NEVER)
    if not isinstance(approval_policy, ApprovalPolicy):
        approval_policy = ApprovalPolicy.NEVER
    overview_lines = [
        f"- 模式：{get_mode_label(runner.mode)} (`{runner.mode.value}`)",
        (
            f"- 审批策略：{get_approval_policy_label(approval_policy)} "
            f"(`{approval_policy.value}`)"
        ),
        f"- 服务：`{runner.service}`",
        f"- 模型：`{runner.model_name}`",
        f"- 模型基址：`{runner.base_url or '默认'}`",
        f"- 工作目录：`{Path.cwd()}`",
        f"- 会话轮数：`{runner.get_turn_count()}`",
        f"- 默认工具数：`{len(runner.tools)}`",
    ]
    ui_mode = runtime_context.get("ui_mode")
    if ui_mode is not None:
        try:
            overview_lines.append(
                f"- 界面：{get_interaction_ui_mode_label(ui_mode)} (`{ui_mode.value}`)"
            )
        except (AttributeError, KeyError):
            overview_lines.append(f"- 界面：`{ui_mode}`")
    session_id = str(runtime_context.get("session_id", "")).strip()
    if session_id:
        overview_lines.append(f"- 当前会话 ID：`{session_id}`")

    context_diagnostics = getattr(runner, "get_context_diagnostics", None)
    context_lines: list[str] = []
    if callable(context_diagnostics):
        diagnostic_payload = context_diagnostics()
        if isinstance(diagnostic_payload, Mapping):
            context_lines.append(
                "- 上下文消息："
                f"完整 `{diagnostic_payload.get('history_message_count', 0)}` / "
                f"模型可见 `{diagnostic_payload.get('model_message_count', 0)}`"
            )
            context_lines.append(
                "- 已压缩历史消息数："
                f"`{diagnostic_payload.get('compressed_message_count', 0)}`"
            )

    saved_allowed_paths = runtime_context.get("saved_allowed_paths", [])
    saved_allowed_lines = _trim_feishu_list_items(
        [str(path) for path in saved_allowed_paths if str(path).strip()]
    )
    allowed_root_lines = _trim_feishu_list_items(
        describe_allowed_roots(runner.allowed_roots)
    )
    registered_tool_lines = _trim_feishu_list_items(
        describe_command_registry(runner.command_registry)
    )
    sections = [
        _build_feishu_markdown_section("会话概览", overview_lines),
        _build_feishu_markdown_section("上下文", context_lines),
        _build_feishu_markdown_section(
            "当前允许访问目录",
            [f"- `{line}`" for line in allowed_root_lines] or ["- 暂无。"],
        ),
        _build_feishu_markdown_section(
            "本地已保存目录",
            [f"- `{line}`" for line in saved_allowed_lines] or ["- 暂无。"],
        ),
        _build_feishu_markdown_section(
            "已注册外部工具",
            [f"- `{line}`" for line in registered_tool_lines] or ["- 暂无。"],
        ),
    ]
    return _build_feishu_interactive_card_payload(
        "会话状态",
        "\n\n".join(section for section in sections if section),
        template="indigo",
        action_rows=_build_feishu_command_action_rows(
            ("/tools", "/allow-path", "/config", "/help")
        ),
    )

def _build_feishu_mode_payload(
    runner: "AgentRunner",
    builtin_output: str,
) -> dict[str, object]:
    """构造飞书版模式卡片。"""
    sections = []
    normalized_output = _normalize_cli_output_for_feishu(builtin_output)
    if _looks_like_builtin_error(normalized_output):
        sections.append(_build_feishu_markdown_section("执行结果", [normalized_output]))
    sections.append(
        _build_feishu_markdown_section(
            "当前模式",
            [
                f"- 名称：{get_mode_label(runner.mode)} (`{runner.mode.value}`)",
                f"- 说明：{get_mode_description(runner.mode)}",
            ],
        )
    )
    return _build_feishu_interactive_card_payload(
        "模式设置",
        "\n\n".join(section for section in sections if section),
        template="orange",
        action_rows=_build_feishu_command_action_rows(
            ("/mode", "/mode standard", "/mode authorized"),
            primary_commands=("/mode",),
        ),
    )

def _build_feishu_approval_payload(
    runtime_context: Mapping[str, object],
    builtin_output: str,
) -> dict[str, object]:
    """构造飞书版审批策略卡片。"""
    approval_policy = runtime_context.get("approval_policy", ApprovalPolicy.NEVER)
    if not isinstance(approval_policy, ApprovalPolicy):
        approval_policy = ApprovalPolicy.NEVER
    sections = []
    normalized_output = _normalize_cli_output_for_feishu(builtin_output)
    if _looks_like_builtin_error(normalized_output):
        sections.append(_build_feishu_markdown_section("执行结果", [normalized_output]))
    sections.append(
        _build_feishu_markdown_section(
            "当前审批策略",
            [
                f"- 名称：{get_approval_policy_label(approval_policy)} (`{approval_policy.value}`)",
                (
                    "- 说明："
                    + (
                        "需要时弹出审批确认。"
                        if approval_policy is ApprovalPolicy.PROMPT
                        else (
                            "自动批准工具执行。"
                            if approval_policy is ApprovalPolicy.AUTO
                            else "全部拒绝需要审批的动作。"
                        )
                    )
                ),
            ],
        )
    )
    return _build_feishu_interactive_card_payload(
        "审批策略",
        "\n\n".join(section for section in sections if section),
        template="sunflower",
        action_rows=_build_feishu_command_action_rows(
            ("/approval", "/approval prompt", "/approval auto", "/approval never"),
            primary_commands=("/approval",),
        ),
    )

def _build_feishu_config_payload(
    runtime_context: Mapping[str, object],
    builtin_output: str,
) -> dict[str, object]:
    """构造飞书版本地配置卡片。"""
    sections = []
    normalized_output = _normalize_cli_output_for_feishu(builtin_output)
    if normalized_output and (
        _looks_like_builtin_error(normalized_output) or "已" in normalized_output
    ):
        sections.append(_build_feishu_markdown_section("执行结果", [normalized_output]))
    local_config_path = str(runtime_context.get("local_config_path", "")).strip()
    config_lines = []
    if local_config_path:
        config_lines.append(f"- 本地配置文件：`{local_config_path}`")
    config_lines.extend(
        [
            f"- 当前服务：`{runtime_context.get('service_name', '未提供')}`",
            f"- 当前模型：`{runtime_context.get('model_name', '未提供')}`",
            f"- 当前模型基址：`{runtime_context.get('base_url') or '默认'}`",
        ]
    )
    saved_allowed_paths = runtime_context.get("saved_allowed_paths", [])
    saved_allowed_lines = _trim_feishu_list_items(
        [str(path) for path in saved_allowed_paths if str(path).strip()]
    )
    sections.extend(
        [
            _build_feishu_markdown_section("本地配置", config_lines),
            _build_feishu_markdown_section(
                "已保存允许目录",
                [f"- `{line}`" for line in saved_allowed_lines] or ["- 暂无。"],
            ),
        ]
    )
    return _build_feishu_interactive_card_payload(
        "本地配置",
        "\n\n".join(section for section in sections if section),
        template="purple",
        action_rows=_build_feishu_command_action_rows(
            ("/config", "/config allow-path", "/allow-path", "/status"),
            primary_commands=("/config",),
        ),
    )

def _build_feishu_allow_path_payload(
    runner: "AgentRunner",
    builtin_output: str,
) -> dict[str, object]:
    """构造飞书版允许目录卡片。"""
    sections = []
    normalized_output = _normalize_cli_output_for_feishu(builtin_output)
    if normalized_output and (
        _looks_like_builtin_error(normalized_output) or "已" in normalized_output
    ):
        sections.append(_build_feishu_markdown_section("执行结果", [normalized_output]))
    allowed_root_lines = _trim_feishu_list_items(
        describe_allowed_roots(runner.allowed_roots)
    )
    sections.append(
        _build_feishu_markdown_section(
            "当前允许访问目录",
            [f"- `{line}`" for line in allowed_root_lines] or ["- 暂无。"],
        )
    )
    return _build_feishu_interactive_card_payload(
        "允许访问目录",
        "\n\n".join(section for section in sections if section),
        template="cyan",
        action_rows=_build_feishu_command_action_rows(
            ("/allow-path", "/config allow-path", "/status"),
            primary_commands=("/allow-path",),
        ),
    )

def _build_hardcoded_feishu_model_config_payload(
    runner: "AgentRunner",
    builtin_output: str,
    *,
    title: str,
    model_services: list[dict] | None = None,
) -> dict[str, object]:
    """构造飞书版模型与服务配置卡片。"""
    sections = []
    normalized_output = _normalize_cli_output_for_feishu(builtin_output)
    if normalized_output and (
        _looks_like_builtin_error(normalized_output) or "已" in normalized_output
    ):
        sections.append(_build_feishu_markdown_section("执行结果", [normalized_output]))
    sections.append(
        _build_feishu_markdown_section(
            "当前配置",
            [
                f"- 服务：`{runner.service}`",
                f"- 模型：`{runner.model_name}`",
                f"- 模型基址：`{runner.base_url or '默认'}`",
            ],
        )
    )
    return _build_feishu_interactive_card_payload(
        title,
        "\n\n".join(section for section in sections if section),
        template="turquoise",
        action_rows=[
            *_build_feishu_command_action_rows(
                (
                    ("切到 OpenAI", "/service openai"),
                    ("切到 DeepSeek", "/service deepseek"),
                    ("OpenAI 默认模型", "/model gpt-5.4"),
                    ("DeepSeek 默认模型", "/model deepseek-v4-pro"),
                ),
                primary_commands=("/service openai", "/service deepseek"),
                row_size=2,
            ),
            *_build_feishu_command_action_rows(
                ("/service", "/model", "/status", "/help"),
                primary_commands=("/service",),
            ),
        ],
    )

def _build_feishu_model_config_payload(
    runner: "AgentRunner",
    builtin_output: str,
    *,
    title: str,
    model_services: list[dict] | None = None,
) -> dict[str, object]:
    """构造飞书版模型与服务配置卡片（动态按钮版）"""
    if not model_services:
        # 回退到原来的硬编码逻辑
        return _build_hardcoded_feishu_model_config_payload(runner, builtin_output, title=title)

    sections = []
    normalized_output = _normalize_cli_output_for_feishu(builtin_output)
    if normalized_output and (_looks_like_builtin_error(normalized_output) or "已" in normalized_output):
        sections.append(_build_feishu_markdown_section("执行结果", [normalized_output]))
    sections.append(
        _build_feishu_markdown_section(
            "当前配置",
            [
                f"- 服务：`{runner.service}`",
                f"- 模型：`{runner.model_name}`",
                f"- 模型基址：`{runner.base_url or '默认'}`",
            ],
        )
    )

    # 动态生成服务切换按钮（每行2个）
    service_buttons = [
        (svc["name"], svc["command"])
        for svc in model_services
    ]
    action_rows = [
        *_build_feishu_command_action_rows(
            service_buttons,
            primary_commands=[svc["command"] for svc in model_services[:1]],
            row_size=2,
        ),
    ]
    # 追加每个服务下的模型按钮
    for svc in model_services:
        if not svc.get("models"):
            continue
        model_buttons = [(m["name"], m["command"]) for m in svc["models"]]
        action_rows.extend(
            _build_feishu_command_action_rows(
                model_buttons,
                primary_commands=[model_buttons[0][1]] if model_buttons else [],
                row_size=3,
            )
        )
    # 保持原有的 /service, /model, /status, /help
    action_rows.extend(
        _build_feishu_command_action_rows(
            ("/service", "/model", "/status", "/help"),
            primary_commands=("/service",),
        )
    )

    return _build_feishu_interactive_card_payload(
        title,
        "\n\n".join(section for section in sections if section),
        template="turquoise",
        action_rows=action_rows,
    )

def _build_feishu_fallback_builtin_payload(
    command: str,
    builtin_output: str,
) -> dict[str, object]:
    """为暂未专门适配的命令提供飞书兜底卡片。"""
    normalized_output = _normalize_cli_output_for_feishu(builtin_output)
    if not normalized_output:
        normalized_output = "命令已执行完成。"
    body_markdown = (
        f"已执行 `{command}`。\n\n"
        f"```text\n{_escape_feishu_code_block(normalized_output)}\n```"
    )
    return _build_feishu_interactive_card_payload(
        "命令结果",
        body_markdown,
        template="grey",
        action_rows=_build_feishu_command_action_rows(("/help", "/status")),
    )

def _build_feishu_builtin_command_payload(
    command: str,
    runner: "AgentRunner",
    runtime_context: Mapping[str, object],
    builtin_output: str,
    *,
    base_dir: Path | None = None,
    model_services: list[dict] | None = None,     # ← 增加这个参数
) -> dict[str, object]:
    """将常用 CLI 内建命令映射为更适合飞书阅读的卡片。"""
    normalized_command = command.strip().lower()
    if normalized_command == "/help":
        return _build_feishu_help_payload()
    if normalized_command == "/tools":
        return _build_feishu_tools_payload(runner)
    if normalized_command == "/context":
        return _build_feishu_context_payload(runner, runtime_context)
    if normalized_command == "/history":
        return _build_feishu_history_list_payload(base_dir=base_dir)
    if normalized_command.startswith("/history show "):
        session_id = command.strip()[len("/history show "):].strip()
        normalized_output = _normalize_cli_output_for_feishu(builtin_output)
        if _looks_like_builtin_error(normalized_output):
            return _build_feishu_notice_payload(
                "历史会话详情",
                normalized_output,
                template="red",
                button_commands=("/history", "/help"),
            )
        return _build_feishu_history_show_payload(session_id, base_dir=base_dir)
    if normalized_command.startswith("/history search "):
        query = command.strip()[len("/history search "):].strip()
        normalized_output = _normalize_cli_output_for_feishu(builtin_output)
        if _looks_like_builtin_error(normalized_output):
            return _build_feishu_notice_payload(
                "历史检索",
                normalized_output,
                template="red",
                button_commands=("/history", "/help"),
            )
        return _build_feishu_history_search_payload(query, base_dir=base_dir)
    if normalized_command.startswith("/history load "):
        session_id = command.strip()[len("/history load "):].strip()
        normalized_output = _normalize_cli_output_for_feishu(builtin_output)
        if _looks_like_builtin_error(normalized_output):
            return _build_feishu_notice_payload(
                "加载历史会话失败",
                normalized_output,
                template="red",
                button_commands=("/history", "/help"),
            )
        return _build_feishu_history_load_payload(
            session_id,
            runner,
            runtime_context,
            builtin_output,
        )
    if normalized_command.startswith("/history export "):
        session_id = command.strip()[len("/history export "):].strip().split(maxsplit=1)[0]
        normalized_output = _normalize_cli_output_for_feishu(builtin_output)
        if _looks_like_builtin_error(normalized_output):
            return _build_feishu_notice_payload(
                "导出历史会话失败",
                normalized_output,
                template="red",
                button_commands=("/history", "/help"),
            )
        return _build_feishu_history_export_payload(session_id, builtin_output)
    if normalized_command == "/doctor":
        return _build_feishu_doctor_payload(runner, runtime_context)
    if normalized_command == "/status":
        return _build_feishu_status_payload(runner, runtime_context)
    if normalized_command == "/mode" or normalized_command.startswith("/mode "):
        return _build_feishu_mode_payload(runner, builtin_output)
    if normalized_command == "/approval" or normalized_command.startswith("/approval "):
        return _build_feishu_approval_payload(runtime_context, builtin_output)
    if normalized_command == "/config" or normalized_command.startswith("/config "):
        return _build_feishu_config_payload(runtime_context, builtin_output)
    if normalized_command == "/allow-path" or normalized_command.startswith("/allow-path "):
        return _build_feishu_allow_path_payload(runner, builtin_output)
    if normalized_command == "/service" or normalized_command.startswith("/service "):
        return _build_feishu_model_config_payload(
            runner,
            builtin_output,
            title="模型服务商",
            model_services=model_services,      # ← 传进去
        )
    if normalized_command == "/model" or normalized_command.startswith("/model "):
        return _build_feishu_model_config_payload(
            runner,
            builtin_output,
            title="模型配置",
            model_services=model_services,      # ← 传进去
        )
    if normalized_command in {"/clear", "/context clear"}:
        return _build_feishu_notice_payload(
            "会话已清空",
            "当前会话上下文已清空，后续消息会作为新会话继续处理。",
            template="green",
            button_commands=("/status", "/start"),
        )
    if normalized_command == "/version":
        normalized_output = _normalize_cli_output_for_feishu(builtin_output) or "版本信息为空。"
        return _build_feishu_notice_payload(
            "CLI 版本",
            normalized_output,
            template="blue",
            button_commands=("/status", "/help"),
        )
    if normalized_command == "/stop":
        normalized_output = _normalize_cli_output_for_feishu(builtin_output) or "已处理停止指令。"
        return _build_feishu_notice_payload(
            "停止任务",
            normalized_output,
            template="red",
            button_commands=("/status",),
        )
    return _build_feishu_fallback_builtin_payload(command, builtin_output)

def _truncate_feishu_button_label(label: str, *, max_chars: int = 12) -> str:
    """控制飞书按钮标题长度，避免最近会话标题过长影响排版。"""
    normalized_label = label.strip()
    if len(normalized_label) <= max_chars:
        return normalized_label
    return normalized_label[:max_chars].rstrip() + "..."

def _build_feishu_recent_session_button_specs(
    session_items: Sequence[Mapping[str, object]],
    *,
    limit: int = 3,
) -> list[tuple[str, str]]:
    """为 /start 菜单提取当前聊天最近会话的快捷切换按钮。"""
    button_specs: list[tuple[str, str]] = []
    for session_item in session_items[: max(limit, 0)]:
        session_id = str(session_item.get("session_id", "")).strip()
        title = str(session_item.get("title", "")).strip() or "未命名会话"
        if not session_id:
            continue
        if bool(session_item.get("is_default")):
            command = "/session default"
        else:
            command = f"/session use {session_id}"
        label_prefix = "当前" if bool(session_item.get("active")) else "最近"
        button_specs.append(
            (
                f"{label_prefix}·{_truncate_feishu_button_label(title)}",
                command,
            )
        )
    return button_specs

def _build_feishu_session_switch_button_specs(
    session_items: Sequence[Mapping[str, object]],
    *,
    limit: int = FEISHU_SESSION_SWITCH_BUTTON_LIMIT,
) -> list[tuple[str, str]]:
    """为 /session list 生成按序号快速切换的按钮。"""
    button_specs: list[tuple[str, str]] = []
    for session_item in session_items[: max(limit, 0)]:
        raw_index = session_item.get("index")
        index_text = str(raw_index).strip()
        if not index_text:
            continue
        title = str(session_item.get("title", "")).strip() or "未命名会话"
        label_prefix = "当前" if bool(session_item.get("active")) else "切换"
        button_specs.append(
            (
                f"{label_prefix} {index_text}·{_truncate_feishu_button_label(title, max_chars=8)}",
                f"/session use {index_text}",
            )
        )

    return button_specs

def _build_feishu_session_list_payload(
    session_items: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """构造更接近聊天软件会话列表的飞书卡片，并附带快速切换按钮。"""
    visible_session_items = list(session_items[:FEISHU_CARD_LIST_LIMIT])
    hidden_session_count = len(session_items) - len(visible_session_items)
    active_session = next(
        (session_item for session_item in session_items if bool(session_item.get("active"))),
        None,
    )
    summary_lines = [
        f"- 会话总数：`{len(session_items)}`",
        (
            f"- 当前会话：`{active_session.get('index')}` · "
            f"{active_session.get('title') or '未命名会话'}"
            if active_session is not None
            else "- 当前会话：未定位"
        ),
        "- 点击下方“切换 N”按钮可直接切到对应会话。",
    ]
    if hidden_session_count > 0:
        summary_lines.append(
            f"- 当前展示前 `{len(visible_session_items)}` 个，其余 `{hidden_session_count}` 个未展开。"
        )
    if len(session_items) > FEISHU_SESSION_SWITCH_BUTTON_LIMIT:
        summary_lines.append(
            f"- 快速切换按钮仅展示前 `{FEISHU_SESSION_SWITCH_BUTTON_LIMIT}` 个会话。"
        )

    body_elements: list[dict[str, object]] = [
        {
            "tag": "markdown",
            "content": _build_feishu_markdown_section("概览", summary_lines),
        }
    ]
    if not visible_session_items:
        body_elements.append(
            {
                "tag": "markdown",
                "content": _build_feishu_markdown_section(
                    "当前聊天会话",
                    ["- 当前聊天下还没有可切换的会话。"],
                ),
            }
        )
    for session_item in visible_session_items:
        index_text = str(session_item.get("index", "")).strip() or "?"
        title = str(session_item.get("title", "")).strip() or "未命名会话"
        status_label = (
            "当前会话"
            if bool(session_item.get("active"))
            else ("默认会话" if bool(session_item.get("is_default")) else "历史会话")
        )
        session_detail = _build_feishu_key_value_table(
            (
                ("状态", status_label),
                ("会话 ID", f"`{session_item.get('session_id', '')}`"),
                ("标题", title),
                ("轮数", f"`{session_item.get('turn_count', 0)}`"),
                ("消息数", f"`{session_item.get('message_count', 0)}`"),
                ("更新时间", f"`{session_item.get('updated_at', '未开始')}`"),
            )
        )
        body_elements.append(
            {
                "tag": "markdown",
                "content": _truncate_feishu_markdown(
                    f"**{status_label} · {index_text}. {title}**\n{session_detail}",
                    max_chars=FEISHU_RICH_REPLY_CHUNK_MAX_CHARS,
                ),
            }
        )

    switch_button_specs = _build_feishu_session_switch_button_specs(session_items)
    primary_switch_commands = tuple(
        f"/session use {session_item.get('index')}"
        for session_item in session_items[:FEISHU_SESSION_SWITCH_BUTTON_LIMIT]
        if bool(session_item.get("active")) and str(session_item.get("index", "")).strip()
    )
    action_rows = [
        *_build_feishu_command_action_rows(
            switch_button_specs,
            primary_commands=primary_switch_commands,
            row_size=3,
        ),
        *_build_feishu_command_action_rows(
            (
                ("新建会话", "/session new"),
                ("刷新列表", "/session list"),
                ("当前会话", "/session current"),
                ("回到默认", "/session default"),
            ),
            primary_commands=("/session list",),
        ),
    ]
    return _build_feishu_interactive_card_elements_payload(
        "飞书会话列表",
        body_elements,
        template="wathet",
        action_rows=action_rows,
    )

def _build_feishu_start_menu_payload(
    session_items: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """构造飞书 /start 命令返回的交互卡片菜单。"""
    command_descriptions = [
        f"- `{command}` {_get_feishu_command_description(command)}"
        for command in FEISHU_START_MENU_COMMANDS
    ]
    session_command_descriptions = [
        f"- `{command}` {_get_feishu_command_description(command)}"
        for command in FEISHU_SESSION_SHORTCUT_COMMANDS
    ]
    recent_session_lines = [
        (

            f"- {'**当前** ' if bool(session_item.get('active')) else ''}"
            f"{session_item.get('title', '未命名会话')} "
            f"(`{session_item.get('session_id', '')}`)"
        )
        for session_item in list(session_items or [])[:3]
    ]
    start_menu_sections = [
        "点击下方按钮即可直接操作；按钮文案更贴近聊天软件，但实际执行的仍是 CLI 命令。",
        _build_feishu_markdown_section(
            "聊天会话",
            [
                "- 新建会话：开始一条新的上下文",
                "- 最近会话：查看并切换当前聊天的历史会话",
                "- 回到默认会话：切回当前聊天的主线会话",
                "- 当前会话：查看当前上下文与压缩状态",
            ],
        ),
        _build_feishu_markdown_section("常用快捷命令", command_descriptions),
        _build_feishu_markdown_section("会话命令", session_command_descriptions),
        _build_feishu_markdown_section("最近会话", recent_session_lines),
    ]
    return _build_feishu_interactive_card_payload(
        "Cyber Agent 飞书快捷菜单",
        "\n\n".join(section for section in start_menu_sections if section),
        template="blue",
        action_rows=[
            *_build_feishu_command_action_rows(
                (
                    ("新建会话", "/session new"),
                    ("最近会话", "/session list"),
                    ("当前会话", "/session current"),
                    ("回到默认会话", "/session default"),
                ),
                primary_commands=("/session new",),
            ),
            *_build_feishu_command_action_rows(
                (
                    ("查看帮助", "/help"),
                    ("会话状态", "/status"),
                    ("可用工具", "/tools"),
                    ("结束会话", "/exit"),
                ),
                primary_commands=("/help",),
            ),
            *_build_feishu_command_action_rows(
                _build_feishu_recent_session_button_specs(session_items or []),
                row_size=3,
            ),
        ],
    )
