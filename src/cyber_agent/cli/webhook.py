from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import struct
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha1
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING, Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request, urlopen
from xml.etree import ElementTree
from xml.sax.saxutils import escape

from langchain_core.messages import AIMessage

from ..agent.approval import (
    ApprovalDecision,
    ApprovalPolicy,
    get_approval_policy_label,
)
from ..agent.mode import get_mode_description, get_mode_label
from ..session_store import (
    create_session_id,
    get_session_storage_dir,
    list_stored_sessions,
    load_session_history,
    save_session_history,
    search_stored_sessions,
)
from ..tools import (
    describe_allowed_roots,
    describe_command_registry,
    describe_tool_instances,
)
from .doctor import build_doctor_payload
from .interactive import BUILTIN_COMMAND_SPECS, get_interaction_ui_mode_label
from .render import CliRenderer

if TYPE_CHECKING:
    from ..agent.runner import AgentRunner

SUPPORTED_WEBHOOK_PROVIDERS = ("feishu", "dingtalk", "wecom", "email")
DEFAULT_WEBHOOK_HOST = "0.0.0.0"
DEFAULT_WEBHOOK_PORT = 8787
DEFAULT_WEBHOOK_REPLY_TIMEOUT_SECONDS = 10.0
DEFAULT_WEBHOOK_REPLY_RETRY_ATTEMPTS = 3
DEFAULT_WEBHOOK_REPLY_RETRY_BACKOFF_SECONDS = 1.0
DEFAULT_WEBHOOK_DEAD_LETTER_DIRNAME = ".cyber-agent-cli-webhook-dead-letters"
WEBHOOK_SESSION_ID_MAX_SLUG_LENGTH = 48
WEBHOOK_SECRET_HEADER = "x-cyber-agent-webhook-secret"
WEBHOOK_SECRET_QUERY_KEY = "secret"
WEBHOOK_CONTENT_TYPE_JSON = "application/json; charset=utf-8"
WEBHOOK_CONTENT_TYPE_TEXT = "text/plain; charset=utf-8"
WEBHOOK_CONTENT_TYPE_XML = "application/xml; charset=utf-8"
WEBHOOK_REPLY_SIGNATURE_HEADER = "x-cyber-agent-signature"
WEBHOOK_REPLY_TIMESTAMP_HEADER = "x-cyber-agent-timestamp"
WEBHOOK_REPLY_SIGNATURE_PREFIX = "sha256="
FEISHU_SIGNATURE_HEADER = "x-lark-signature"
FEISHU_TIMESTAMP_HEADER = "x-lark-request-timestamp"
FEISHU_NONCE_HEADER = "x-lark-request-nonce"
FEISHU_TENANT_ACCESS_TOKEN_URL = (
    "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
)
FEISHU_REPLY_MESSAGE_URL_TEMPLATE = (
    "https://open.feishu.cn/open-apis/im/v1/messages/{message_id}/reply"
)
FEISHU_CREATE_MESSAGE_URL = (
    "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id"
)
FEISHU_REPLY_API_MODE = "reply_api"
FEISHU_CREATE_API_MODE = "create_api"
FEISHU_TOKEN_CACHE_SAFETY_SECONDS = 60.0
FEISHU_CARD_MARKDOWN_MAX_CHARS = 3500
FEISHU_CARD_LIST_LIMIT = 20
FEISHU_RICH_REPLY_CHUNK_MAX_CHARS = 2200
FEISHU_RICH_REPLY_MAX_CHUNKS = 6
FEISHU_TRACE_MAX_STEPS = 10
FEISHU_TRACE_DETAIL_MAX_CHARS = 500
FEISHU_PROGRESS_HEARTBEAT_IDLE_SECONDS = 8.0
FEISHU_PROGRESS_HEARTBEAT_POLL_SECONDS = 1.0
FEISHU_PROGRESS_INPUT_PREVIEW_MAX_CHARS = 80
FEISHU_CONTEXT_PREVIEW_MAX_LINES = 8
FEISHU_HISTORY_EXCERPT_RESULT_LIMIT = 3
FEISHU_HISTORY_EXCERPT_LINE_LIMIT = 2
WECOM_MESSAGE_SIGNATURE_QUERY_KEY = "msg_signature"
WECOM_TIMESTAMP_QUERY_KEY = "timestamp"
WECOM_NONCE_QUERY_KEY = "nonce"
WECOM_ECHOSTR_QUERY_KEY = "echostr"

FEISHU_RICH_PANEL_EDGE_RE = re.compile(r"^\s*[│┃]\s?(.*?)\s*[│┃]\s*$")
FEISHU_BOX_DRAWING_LINE_RE = re.compile(r"^[\s\u2500-\u257F\u2580-\u259F]+$")

AgentRunnerFactory = Callable[[dict[str, object]], "AgentRunner"]
ReplySender = Callable[
    [str, dict[str, object], float, Mapping[str, str] | None],
    "WebhookDeliveryReceipt",
]
FeishuCommandButtonSpec = str | tuple[str, str]

FEISHU_START_MENU_COMMANDS: tuple[str, ...] = (
    "/help",
    "/tools",
    "/status",
    "/mode",
    "/config",
    "/allow-path",
    "/approval",
    "/exit",
)
FEISHU_SESSION_SHORTCUT_COMMANDS: tuple[str, ...] = (
    "/session current",
    "/session new",
    "/session list",
    "/session default",
)
FEISHU_SESSION_COMMAND_DESCRIPTIONS: dict[str, str] = {
    "/session": "查看当前飞书活动会话",
    "/session current": "查看当前飞书活动会话",
    "/session new": "新建并切换到新的飞书会话",
    "/session list": "列出当前飞书聊天下的会话",
    "/session default": "切回当前飞书聊天的默认会话",
    "/session use <会话ID|序号>": "切换到指定飞书会话",
}
FEISHU_SESSION_STATE_FILENAME = "feishu-chat-session-state.json"
FEISHU_DEFAULT_SESSION_LABEL = "默认会话"


@dataclass(slots=True, frozen=True)
class WebhookRouteConfig:
    """描述单条 webhook 路由的接入方式。"""

    provider: str
    path: str
    reply_webhook_url: str | None = None
    secret: str | None = None
    provider_options: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class WebhookEvent:
    """统一表示一条已归一化的第三方 webhook 消息。"""

    provider: str
    session_key: str
    sender_id: str
    sender_name: str
    message_id: str
    text: str
    reply_webhook_url: str | None = None
    subject: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class WebhookHttpResponse:
    """描述 HTTP 层要返回给第三方 webhook 调用方的响应。"""

    status_code: int
    body: bytes
    content_type: str = WEBHOOK_CONTENT_TYPE_JSON


@dataclass(slots=True)
class WebhookRequestOutcome:
    """描述 webhook 解析结果，要么进入消息处理，要么直接返回 HTTP 响应。"""

    event: WebhookEvent | None = None
    immediate_response: WebhookHttpResponse | None = None


@dataclass(slots=True, frozen=True)
class WebhookDeliveryReceipt:
    """描述一次回复投递的结果。"""

    status_code: int
    response_text: str


@dataclass(slots=True, frozen=True)
class WebhookAgentReply:
    """描述当前 webhook 请求经过智能体处理后的结果。"""

    session_id: str
    reply_text: str
    reply_payload_override: dict[str, object] | None = None


@dataclass(slots=True, frozen=True)
class FeishuTraceStep:
    """描述一条适合展示到飞书卡片中的中间处理步骤。"""

    kind: str
    title: str
    detail: str = ""


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
                    detail=cls._build_code_block(normalized_content, language="text"),
                )
            ]

        return []

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
        while not self._heartbeat_stop_event.wait(FEISHU_PROGRESS_HEARTBEAT_POLL_SECONDS):
            heartbeat_step = self._build_heartbeat_step()
            if heartbeat_step is not None:
                self._emit_step(heartbeat_step)

    def _build_heartbeat_step(self) -> FeishuTraceStep | None:
        """仅当长时间没有新事件时才补一条心跳，避免飞书侧长时间静默。"""
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


class WebhookAuthorizationError(Exception):
    """描述 webhook 请求在鉴权阶段被拒绝的错误。"""


class WebhookDeliveryError(Exception):
    """描述 reply webhook 投递失败的错误。"""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_text: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text


def _extract_webhook_response_reason(response: WebhookHttpResponse) -> str:
    """提取 webhook HTTP 响应中的主要错误信息，便于后台日志定位问题。"""
    decoded_body = response.body.decode("utf-8", errors="replace").strip()
    if not decoded_body:
        return ""
    try:
        payload = json.loads(decoded_body)
    except json.JSONDecodeError:
        return decoded_body
    if not isinstance(payload, dict):
        return decoded_body
    for key in ("reason", "status", "msg"):
        resolved_value = str(payload.get(key, "")).strip()
        if resolved_value:
            return resolved_value
    return decoded_body


def _capture_builtin_command_output_for_webhook(
    user_input: str,
    runner: "AgentRunner",
    runtime_context: dict[str, object],
) -> tuple[bool | None, str]:
    """延迟导入 CLI 内建命令捕获函数，避免 webhook 模块与 app 模块相互导入。"""
    from .app import capture_builtin_command_output

    return capture_builtin_command_output(
        user_input,
        runner,
        runtime_context,
        styled=False,
    )


def _get_builtin_command_description(command: str) -> str:
    """按统一命令清单返回说明，避免飞书菜单与 CLI 帮助脱节。"""
    normalized_command = command.strip().lower()
    for command_spec in BUILTIN_COMMAND_SPECS:
        if command_spec.command.lower() == normalized_command:
            return command_spec.description
    return ""


def _get_feishu_command_description(command: str) -> str:
    """返回飞书侧可见命令的说明，兼容 CLI 内建命令与飞书扩展命令。"""
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
                "/service <服务商> <基址>",
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
                "OPENAI_API_KEY",
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


def _build_feishu_model_config_payload(
    runner: "AgentRunner",
    builtin_output: str,
    *,
    title: str,
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
        action_rows=_build_feishu_command_action_rows(
            ("/service", "/model", "/status", "/help"),
            primary_commands=("/service",),
        ),
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
        )
    if normalized_command == "/model" or normalized_command.startswith("/model "):
        return _build_feishu_model_config_payload(
            runner,
            builtin_output,
            title="模型配置",
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


def normalize_webhook_provider(raw_provider: str) -> str:
    """规范化第三方 webhook 提供方名称。"""
    normalized_provider = raw_provider.strip().lower()
    if normalized_provider not in SUPPORTED_WEBHOOK_PROVIDERS:
        supported_providers = ", ".join(SUPPORTED_WEBHOOK_PROVIDERS)
        raise ValueError(
            f"不支持的 webhook 提供方：{raw_provider}。可选值：{supported_providers}"
        )
    return normalized_provider


def normalize_webhook_path(raw_path: str) -> str:
    """规范化 webhook 路由路径，避免配置中混入查询串和尾部空白。"""
    normalized_path = raw_path.strip()
    if not normalized_path:
        raise ValueError("webhook 路由路径不能为空。")
    if not normalized_path.startswith("/"):
        normalized_path = f"/{normalized_path}"
    parsed_path = urlsplit(normalized_path)
    if parsed_path.query:
        raise ValueError("webhook 路由路径不能包含查询参数。")
    return parsed_path.path.rstrip("/") or "/"


def build_default_webhook_routes(
    providers: list[str] | None = None,
) -> list[WebhookRouteConfig]:
    """根据给定平台列表构建默认 webhook 路由。"""
    normalized_providers: list[str] = []
    for raw_provider in providers or list(SUPPORTED_WEBHOOK_PROVIDERS):
        normalized_provider = normalize_webhook_provider(raw_provider)
        if normalized_provider not in normalized_providers:
            normalized_providers.append(normalized_provider)

    return [
        WebhookRouteConfig(
            provider=provider,
            path=f"/webhook/{provider}",
        )
        for provider in normalized_providers
    ]


def build_webhook_example_config() -> dict[str, object]:
    """生成适合 `webhook example-config` 输出的通用配置模板。"""
    return {
        "providers": {
            "feishu": {
                "path": "/webhook/feishu",
                "reply_webhook_url": "",
                "provider_options": {
                    "verification_token": "",
                    "encrypt_key": "",
                    "app_id": "",
                    "app_secret": "",
                    "reply_mode": "",
                    "reply_in_thread": "",
                    "reply_retry_attempts": "",
                    "reply_retry_backoff_seconds": "",
                    "reply_signing_secret": "",
                },
            },
            "dingtalk": {
                "path": "/webhook/dingtalk",
                "secret": "",
                "reply_webhook_url": "",
                "provider_options": {},
            },
            "wecom": {
                "path": "/webhook/wecom",
                "reply_webhook_url": "",
                "provider_options": {
                    "token": "",
                    "encoding_aes_key": "",
                    "receive_id": "",
                    "reply_mode": "",
                },
            },
            "email": {
                "path": "/webhook/email",
                "secret": "",
                "reply_webhook_url": "",
                "provider_options": {
                    "reply_retry_attempts": "",
                    "reply_retry_backoff_seconds": "",
                    "reply_signing_secret": "",
                    "reply_dead_letter_dir": "",
                },
            },
        }
    }


def _normalize_raw_provider_options(raw_provider_options: object) -> dict[str, str]:
    """规范化 provider_options，并自动丢弃空白占位字段。"""
    if not isinstance(raw_provider_options, dict):
        return {}
    return {
        str(key): str(value)
        for key, value in raw_provider_options.items()
        if str(value).strip()
    }


def _is_route_effectively_enabled(
    *,
    secret: str | None,
    reply_webhook_url: str | None,
    provider_options: Mapping[str, str],
) -> bool:
    """判断单条配置是否填写了足以启用 webhook 的关键信息。"""
    if secret:
        return True
    if reply_webhook_url:
        return True
    return bool(provider_options)


def _build_route_config_from_raw_route(
    raw_route: Mapping[str, object],
    *,
    route_label: str,
    allow_disabled_entry: bool,
) -> WebhookRouteConfig | None:
    """将单条原始配置归一化为路由对象，并在需要时自动跳过空配置。"""
    normalized_provider = normalize_webhook_provider(str(raw_route.get("provider", "")))
    normalized_path = normalize_webhook_path(
        str(raw_route.get("path", f"/webhook/{normalized_provider}"))
    )
    provider_options = _normalize_raw_provider_options(raw_route.get("provider_options", {}))
    reply_webhook_url = str(raw_route.get("reply_webhook_url", "")).strip() or None
    secret = str(raw_route.get("secret", "")).strip() or None

    if allow_disabled_entry and not _is_route_effectively_enabled(
        secret=secret,
        reply_webhook_url=reply_webhook_url,
        provider_options=provider_options,
    ):
        return None

    if not allow_disabled_entry and not _is_route_effectively_enabled(
        secret=secret,
        reply_webhook_url=reply_webhook_url,
        provider_options=provider_options,
    ):
        raise ValueError(f"{route_label} 未填写任何可启用 webhook 的关键字段。")

    return WebhookRouteConfig(
        provider=normalized_provider,
        path=normalized_path,
        reply_webhook_url=reply_webhook_url,
        secret=secret,
        provider_options=provider_options,
    )


def load_webhook_routes_from_file(config_path: Path | str) -> list[WebhookRouteConfig]:
    """从 JSON 配置文件中读取 webhook 路由。"""
    resolved_config_path = Path(config_path).expanduser().resolve()
    if not resolved_config_path.exists():
        raise ValueError(f"未找到 webhook 配置文件：{resolved_config_path}")

    try:
        raw_data = json.loads(resolved_config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"webhook 配置文件不是合法 JSON：{resolved_config_path}") from exc

    if not isinstance(raw_data, dict):
        raise ValueError("webhook 配置文件顶层必须是对象。")

    routes: list[WebhookRouteConfig] = []
    seen_paths: set[str] = set()
    raw_routes = raw_data.get("routes")
    raw_providers = raw_data.get("providers")
    if raw_routes is not None:
        if not isinstance(raw_routes, list) or not raw_routes:
            raise ValueError("webhook 配置文件中的 routes 必须是非空数组。")
        for index, raw_route in enumerate(raw_routes, start=1):
            if not isinstance(raw_route, dict):
                raise ValueError(f"第 {index} 条 webhook 路由必须是对象。")
            route = _build_route_config_from_raw_route(
                raw_route,
                route_label=f"第 {index} 条 webhook 路由",
                allow_disabled_entry=False,
            )
            if route.path in seen_paths:
                raise ValueError(f"发现重复的 webhook 路由路径：{route.path}")
            seen_paths.add(route.path)
            routes.append(route)
    elif raw_providers is not None:
        if not isinstance(raw_providers, dict) or not raw_providers:
            raise ValueError("webhook 配置文件中的 providers 必须是非空对象。")
        for provider_name in SUPPORTED_WEBHOOK_PROVIDERS:
            raw_provider_route = raw_providers.get(provider_name)
            if raw_provider_route is None:
                continue
            if not isinstance(raw_provider_route, dict):
                raise ValueError(f"providers.{provider_name} 必须是对象。")
            route = _build_route_config_from_raw_route(
                {
                    "provider": provider_name,
                    **raw_provider_route,
                },
                route_label=f"providers.{provider_name}",
                allow_disabled_entry=True,
            )
            if route is None:
                continue
            if route.path in seen_paths:
                raise ValueError(f"发现重复的 webhook 路由路径：{route.path}")
            seen_paths.add(route.path)
            routes.append(route)
    else:
        raise ValueError("webhook 配置文件必须包含 routes 或 providers 其中之一。")

    if not routes:
        raise ValueError("当前 webhook 配置中没有任何已启用的路由，请至少填写一个平台的关键字段。")

    return routes


def describe_webhook_routes(routes: list[WebhookRouteConfig]) -> list[str]:
    """返回适合启动日志输出的 webhook 路由摘要。"""
    descriptions: list[str] = []
    for route in routes:
        delivery_hint = _describe_webhook_delivery_hint(route)
        descriptions.append(
            f"{route.path} -> {route.provider} | 回复: {delivery_hint}"
        )
    return descriptions


def _get_route_option(
    route: WebhookRouteConfig,
    option_name: str,
) -> str | None:
    raw_value = route.provider_options.get(option_name)
    if raw_value is None:
        return None
    normalized_value = str(raw_value).strip()
    return normalized_value or None


def _get_route_float_option(
    route: WebhookRouteConfig,
    option_name: str,
    default_value: float,
    *,
    minimum: float = 0.0,
) -> float:
    raw_value = _get_route_option(route, option_name)
    if raw_value is None:
        return default_value
    try:
        parsed_value = float(raw_value)
    except ValueError:
        return default_value
    return max(parsed_value, minimum)


def _get_route_int_option(
    route: WebhookRouteConfig,
    option_name: str,
    default_value: int,
    *,
    minimum: int = 0,
) -> int:
    raw_value = _get_route_option(route, option_name)
    if raw_value is None:
        return default_value
    try:
        parsed_value = int(raw_value)
    except ValueError:
        return default_value
    return max(parsed_value, minimum)


def _get_route_bool_option(
    route: WebhookRouteConfig,
    option_name: str,
    default_value: bool = False,
) -> bool:
    raw_value = _get_route_option(route, option_name)
    if raw_value is None:
        return default_value
    normalized_value = raw_value.strip().lower()
    if normalized_value in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized_value in {"0", "false", "no", "n", "off"}:
        return False
    return default_value


def _describe_webhook_delivery_hint(route: WebhookRouteConfig) -> str:
    if route.provider == "feishu":
        configured_reply_mode = (_get_route_option(route, "reply_mode") or "").lower()
        if configured_reply_mode == FEISHU_CREATE_API_MODE:
            return "官方发送消息 API"
        if configured_reply_mode == FEISHU_REPLY_API_MODE:
            return "官方消息回复 API"
        if (
            configured_reply_mode != "reply_webhook"
            and not route.reply_webhook_url
            and _get_route_option(route, "app_id")
            and _get_route_option(route, "app_secret")
        ):
            return "官方消息回复 API"
    if route.provider == "wecom":
        configured_reply_mode = (_get_route_option(route, "reply_mode") or "").lower()
        if configured_reply_mode == "passive_xml":
            return "官方被动 XML 回包"
    return route.reply_webhook_url or "按请求内 reply_webhook_url 或 HTTP 响应回包"


def _resolve_feishu_reply_mode(
    route: WebhookRouteConfig,
    event: WebhookEvent,
) -> str:
    metadata_reply_mode = str(event.metadata.get("feishu_delivery_mode", "")).strip().lower()
    if metadata_reply_mode == FEISHU_CREATE_API_MODE:
        return FEISHU_CREATE_API_MODE
    configured_reply_mode = (_get_route_option(route, "reply_mode") or "").lower()
    if configured_reply_mode == FEISHU_CREATE_API_MODE:
        return FEISHU_CREATE_API_MODE
    if configured_reply_mode == FEISHU_REPLY_API_MODE:
        return FEISHU_REPLY_API_MODE
    if configured_reply_mode == "reply_webhook":
        return "reply_webhook"
    if event.reply_webhook_url:
        return "reply_webhook"
    if _get_route_option(route, "app_id") and _get_route_option(route, "app_secret"):
        return FEISHU_REPLY_API_MODE
    return "response_payload"


def _serialize_webhook_json_payload(payload: dict[str, object]) -> bytes:
    """统一 reply webhook 的 JSON 序列化方式，便于签名和重试复用。"""
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _build_reply_signature_headers(
    route: WebhookRouteConfig,
    payload_bytes: bytes,
) -> dict[str, str]:
    """按路由配置生成 reply webhook 出站签名头。"""
    signing_secret = _get_route_option(route, "reply_signing_secret")
    if signing_secret is None:
        return {}

    timestamp_value = str(int(time.time()))
    signature_header = (
        _get_route_option(route, "reply_signature_header")
        or WEBHOOK_REPLY_SIGNATURE_HEADER
    )
    timestamp_header = (
        _get_route_option(route, "reply_timestamp_header")
        or WEBHOOK_REPLY_TIMESTAMP_HEADER
    )
    signing_payload = timestamp_value.encode("utf-8") + b"." + payload_bytes
    digest = hmac.new(
        signing_secret.encode("utf-8"),
        signing_payload,
        hashlib.sha256,
    ).hexdigest()
    return {
        timestamp_header: timestamp_value,
        signature_header: f"{WEBHOOK_REPLY_SIGNATURE_PREFIX}{digest}",
    }


def _resolve_dead_letter_dir(
    route: WebhookRouteConfig,
    *,
    base_dir: Path | None,
) -> Path:
    configured_dir = (
        _get_route_option(route, "reply_dead_letter_dir")
        or DEFAULT_WEBHOOK_DEAD_LETTER_DIRNAME
    )
    resolved_base_dir = (base_dir or Path.cwd()).resolve()
    dead_letter_dir = Path(configured_dir).expanduser()
    if dead_letter_dir.is_absolute():
        return dead_letter_dir
    return resolved_base_dir / dead_letter_dir


def _redact_webhook_url(url: str) -> str:
    """对死信文件中的目标地址做最小脱敏，避免泄露查询串中的令牌。"""
    parsed_url = urlsplit(url)
    if not parsed_url.query:
        return url

    redacted_query_parts: list[str] = []
    for query_part in parsed_url.query.split("&"):
        if not query_part:
            continue
        if "=" not in query_part:
            redacted_query_parts.append(query_part)
            continue
        key, _value = query_part.split("=", 1)
        redacted_query_parts.append(f"{key}=***")

    redacted_query = "&".join(redacted_query_parts)
    return parsed_url._replace(query=redacted_query).geturl()


def _write_delivery_dead_letter(
    route: WebhookRouteConfig,
    event: WebhookEvent,
    agent_reply: WebhookAgentReply,
    target_url: str,
    reply_payload: dict[str, object],
    attempts: list[dict[str, object]],
    *,
    base_dir: Path | None,
) -> Path:
    """将重试耗尽后的 reply webhook 投递失败信息落盘，便于后续补偿。"""
    dead_letter_dir = _resolve_dead_letter_dir(route, base_dir=base_dir)
    dead_letter_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now().astimezone()
    digest = sha1(
        (
            f"{event.provider}:{agent_reply.session_id}:{event.message_id}:{target_url}"
        ).encode("utf-8")
    ).hexdigest()[:12]
    file_name = (
        f"{created_at.strftime('%Y%m%d-%H%M%S-%f')}"
        f"-{event.provider}-{digest}.json"
    )
    payload = {
        "created_at": created_at.isoformat(),
        "provider": event.provider,
        "route_path": route.path,
        "session_id": agent_reply.session_id,
        "message_id": event.message_id,
        "target_url": _redact_webhook_url(target_url),
        "reply_payload": reply_payload,
        "attempts": attempts,
    }
    dead_letter_path = dead_letter_dir / file_name
    dead_letter_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return dead_letter_path


def _pkcs7_unpad(payload: bytes, block_size: int) -> bytes:
    if not payload:
        raise ValueError("加密数据不能为空。")
    padding_size = payload[-1]
    if padding_size < 1 or padding_size > block_size:
        raise ValueError("加密数据的填充字节非法。")
    if payload[-padding_size:] != bytes([padding_size]) * padding_size:
        raise ValueError("加密数据的填充内容非法。")
    return payload[:-padding_size]


def _pkcs7_pad(payload: bytes, block_size: int) -> bytes:
    padding_size = block_size - (len(payload) % block_size)
    if padding_size == 0:
        padding_size = block_size
    return payload + bytes([padding_size]) * padding_size


def _load_optional_aes_cipher() -> tuple[Callable[[bytes, bytes, bytes], bytes], Callable[[bytes, bytes, bytes], bytes]]:
    """按可用性加载 AES-CBC 加解密实现，避免为 webhook 新增强制依赖。"""
    try:
        from Crypto.Cipher import AES  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        try:
            from cryptography.hazmat.backends import default_backend  # type: ignore[import-not-found]
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise ValueError(
                "当前运行环境缺少 AES 加解密依赖，请安装 pycryptodome 或 cryptography 后再启用官方加密回调。"
            ) from exc

        def decryptor(key: bytes, iv: bytes, payload: bytes) -> bytes:
            cipher = Cipher(
                algorithms.AES(key),
                modes.CBC(iv),
                backend=default_backend(),
            )
            decrypt_context = cipher.decryptor()
            return decrypt_context.update(payload) + decrypt_context.finalize()

        def encryptor(key: bytes, iv: bytes, payload: bytes) -> bytes:
            cipher = Cipher(
                algorithms.AES(key),
                modes.CBC(iv),
                backend=default_backend(),
            )
            encrypt_context = cipher.encryptor()
            return encrypt_context.update(payload) + encrypt_context.finalize()

        return decryptor, encryptor

    def decryptor(key: bytes, iv: bytes, payload: bytes) -> bytes:
        cipher = AES.new(key, AES.MODE_CBC, iv)
        return cipher.decrypt(payload)

    def encryptor(key: bytes, iv: bytes, payload: bytes) -> bytes:
        cipher = AES.new(key, AES.MODE_CBC, iv)
        return cipher.encrypt(payload)

    return decryptor, encryptor


def _aes_cbc_decrypt(key: bytes, iv: bytes, payload: bytes) -> bytes:
    decryptor, _ = _load_optional_aes_cipher()
    return decryptor(key, iv, payload)


def _aes_cbc_encrypt(key: bytes, iv: bytes, payload: bytes) -> bytes:
    _, encryptor = _load_optional_aes_cipher()
    return encryptor(key, iv, payload)


def create_webhook_approval_handler(
    policy: ApprovalPolicy,
) -> Callable[[object, dict[str, object]], ApprovalDecision]:
    """为 webhook 场景构建高风险工具审批处理器。"""

    def approval_handler(_tool: object, _tool_call: dict[str, object]) -> ApprovalDecision:
        if policy is ApprovalPolicy.AUTO:
            return ApprovalDecision(
                approved=True,
                reason="webhook 服务使用自动批准策略，已放行高风险工具调用。",
            )
        if policy is ApprovalPolicy.NEVER:
            return ApprovalDecision(
                approved=False,
                reason="webhook 服务当前使用全部拒绝策略，高风险工具已被拦截。",
            )
        return ApprovalDecision(
            approved=False,
            reason="webhook 服务不支持交互式审批，请改用 --approval-policy auto 或 never。",
        )

    return approval_handler


def send_webhook_json(
    url: str,
    payload: dict[str, object],
    timeout_seconds: float,
    headers: Mapping[str, str] | None = None,
) -> WebhookDeliveryReceipt:
    """以 JSON POST 的方式向第三方 reply webhook 发送回复。"""
    payload_bytes = _serialize_webhook_json_payload(payload)
    request_headers = {"Content-Type": WEBHOOK_CONTENT_TYPE_JSON}
    if headers is not None:
        request_headers.update(headers)
    request = Request(
        url,
        data=payload_bytes,
        headers=request_headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8", errors="replace")
            return WebhookDeliveryReceipt(
                status_code=response.status,
                response_text=response_body,
            )
    except HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        raise WebhookDeliveryError(
            f"reply webhook 返回 HTTP {exc.code}",
            status_code=exc.code,
            response_text=response_body,
        ) from exc
    except URLError as exc:
        raise WebhookDeliveryError(f"reply webhook 请求失败：{exc.reason}") from exc


def build_json_http_response(
    payload: dict[str, object],
    *,
    status_code: int = 200,
) -> WebhookHttpResponse:
    """构建标准 JSON HTTP 响应。"""
    return WebhookHttpResponse(
        status_code=status_code,
        body=(json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"),
    )


def build_text_http_response(
    content: str,
    *,
    status_code: int = 200,
) -> WebhookHttpResponse:
    """构建纯文本 HTTP 响应。"""
    return WebhookHttpResponse(
        status_code=status_code,
        body=content.encode("utf-8"),
        content_type=WEBHOOK_CONTENT_TYPE_TEXT,
    )


def build_xml_http_response(
    content: str,
    *,
    status_code: int = 200,
) -> WebhookHttpResponse:
    """构建 XML HTTP 响应。"""
    return WebhookHttpResponse(
        status_code=status_code,
        body=content.encode("utf-8"),
        content_type=WEBHOOK_CONTENT_TYPE_XML,
    )


def _extract_nested_value(payload: Mapping[str, object], field_path: tuple[str, ...]) -> object | None:
    current_value: object = payload
    for field_name in field_path:
        if not isinstance(current_value, Mapping):
            return None
        current_value = current_value.get(field_name)
    return current_value


def _extract_first_non_empty_string(
    payload: Mapping[str, object],
    *field_paths: tuple[str, ...],
) -> str:
    for field_path in field_paths:
        raw_value = _extract_nested_value(payload, field_path)
        if raw_value is None:
            continue
        if isinstance(raw_value, str):
            normalized_value = raw_value.strip()
            if normalized_value:
                return normalized_value
            continue
        if isinstance(raw_value, (dict, list)):
            continue
        normalized_value = str(raw_value).strip()
        if normalized_value:
            return normalized_value
    return ""


def _parse_json_payload(body: bytes) -> dict[str, object]:
    if not body:
        return {}
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("请求体不是合法 JSON。") from exc
    if not isinstance(payload, dict):
        raise ValueError("请求体顶层必须是 JSON 对象。")
    return payload


def _parse_form_payload(body: bytes) -> dict[str, object]:
    if not body:
        return {}
    payload = parse_qs(
        body.decode("utf-8", errors="replace"),
        keep_blank_values=True,
    )
    return {
        key: values[0] if len(values) == 1 else values
        for key, values in payload.items()
    }


def _parse_request_payload(headers: Mapping[str, str], body: bytes) -> dict[str, object]:
    content_type = headers.get("content-type", "").lower()
    if "application/json" in content_type or not content_type:
        return _parse_json_payload(body)
    if "application/x-www-form-urlencoded" in content_type:
        return _parse_form_payload(body)
    # TODO(联调补全): 邮件供应商若使用 multipart/form-data 回调，需按真实供应商字段补充解析。
    raise ValueError(f"当前暂不支持的 Content-Type：{content_type or 'unknown'}")


def _parse_xml_payload(xml_text: str) -> dict[str, str]:
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError as exc:
        raise ValueError("XML 请求体格式非法。") from exc

    payload: dict[str, str] = {}
    for child in root:
        payload[child.tag] = (child.text or "").strip()
    return payload


def _decrypt_feishu_payload(
    encrypt_value: str,
    encrypt_key: str,
) -> dict[str, object]:
    encrypted_bytes = base64.b64decode(encrypt_value)
    if len(encrypted_bytes) < 16:
        raise ValueError("飞书加密事件体长度不足。")
    iv = encrypted_bytes[:16]
    ciphertext = encrypted_bytes[16:]
    hashed_key = hashlib.sha256(encrypt_key.encode("utf-8")).digest()
    decrypted_bytes = _aes_cbc_decrypt(hashed_key, iv, ciphertext)
    unpadded_bytes = _pkcs7_unpad(decrypted_bytes, 16)
    return _parse_json_payload(unpadded_bytes)


def _verify_feishu_signature(
    headers: Mapping[str, str],
    body: bytes,
    encrypt_key: str,
) -> None:
    timestamp_value = headers.get(FEISHU_TIMESTAMP_HEADER, "").strip()
    nonce_value = headers.get(FEISHU_NONCE_HEADER, "").strip()
    provided_signature = headers.get(FEISHU_SIGNATURE_HEADER, "").strip().lower()
    if not timestamp_value or not nonce_value or not provided_signature:
        raise WebhookAuthorizationError("飞书签名头缺失，无法校验请求来源。")

    signature_payload = (
        timestamp_value.encode("utf-8")
        + nonce_value.encode("utf-8")
        + encrypt_key.encode("utf-8")
        + body
    )
    expected_signature = hashlib.sha256(signature_payload).hexdigest()
    if not hmac.compare_digest(provided_signature, expected_signature):
        raise WebhookAuthorizationError("飞书签名校验失败。")


def _require_feishu_token(payload: Mapping[str, object], route: WebhookRouteConfig) -> None:
    configured_token = _get_route_option(route, "verification_token")
    if configured_token is None:
        return
    payload_token = _extract_first_non_empty_string(
        payload,
        ("token",),
        ("header", "token"),
    )
    if not payload_token or not hmac.compare_digest(payload_token, configured_token):
        raise WebhookAuthorizationError("飞书 Verification Token 校验失败。")


def _normalize_wecom_aes_key(encoding_aes_key: str) -> bytes:
    try:
        aes_key = base64.b64decode(f"{encoding_aes_key}=")
    except Exception as exc:  # noqa: BLE001 - 兼容不同底层异常类型
        raise ValueError("企微 EncodingAESKey 不是合法的 Base64 编码。") from exc
    if len(aes_key) != 32:
        raise ValueError("企微 EncodingAESKey 解码后长度必须为 32 字节。")
    return aes_key


def _build_wecom_signature(
    token: str,
    timestamp_value: str,
    nonce_value: str,
    encrypted_payload: str,
) -> str:
    signature_text = "".join(
        sorted([token, timestamp_value, nonce_value, encrypted_payload])
    )
    return sha1(signature_text.encode("utf-8")).hexdigest()


def _verify_wecom_signature(
    token: str,
    timestamp_value: str,
    nonce_value: str,
    encrypted_payload: str,
    provided_signature: str,
) -> None:
    expected_signature = _build_wecom_signature(
        token,
        timestamp_value,
        nonce_value,
        encrypted_payload,
    )
    if not hmac.compare_digest(provided_signature, expected_signature):
        raise WebhookAuthorizationError("企微 msg_signature 校验失败。")


def _decrypt_wecom_ciphertext(
    encrypted_payload: str,
    encoding_aes_key: str,
    *,
    expected_receive_id: str | None,
) -> str:
    encrypted_bytes = base64.b64decode(encrypted_payload)
    aes_key = _normalize_wecom_aes_key(encoding_aes_key)
    decrypted_bytes = _aes_cbc_decrypt(aes_key, aes_key[:16], encrypted_bytes)
    unpadded_bytes = _pkcs7_unpad(decrypted_bytes, 32)
    if len(unpadded_bytes) < 20:
        raise ValueError("企微解密后的消息体长度不足。")

    message_length = struct.unpack(">I", unpadded_bytes[16:20])[0]
    message_start = 20
    message_end = message_start + message_length
    if message_end > len(unpadded_bytes):
        raise ValueError("企微解密后的消息体长度字段非法。")

    message_bytes = unpadded_bytes[message_start:message_end]
    receive_id = unpadded_bytes[message_end:].decode("utf-8", errors="replace")
    if expected_receive_id and receive_id != expected_receive_id:
        raise WebhookAuthorizationError("企微 ReceiveId 校验失败。")
    return message_bytes.decode("utf-8", errors="replace")


def _encrypt_wecom_plaintext(
    plaintext: str,
    encoding_aes_key: str,
    receive_id: str,
) -> str:
    aes_key = _normalize_wecom_aes_key(encoding_aes_key)
    plaintext_bytes = plaintext.encode("utf-8")
    raw_payload = (
        os.urandom(16)
        + struct.pack(">I", len(plaintext_bytes))
        + plaintext_bytes
        + receive_id.encode("utf-8")
    )
    padded_payload = _pkcs7_pad(raw_payload, 32)
    encrypted_bytes = _aes_cbc_encrypt(aes_key, aes_key[:16], padded_payload)
    return base64.b64encode(encrypted_bytes).decode("utf-8")


def _build_wecom_passive_text_xml(reply_text: str, event: WebhookEvent) -> str:
    to_user_name = escape(event.metadata.get("wecom_from_user_name", ""))
    from_user_name = escape(event.metadata.get("wecom_to_user_name", ""))
    agent_id = escape(event.metadata.get("wecom_agent_id", ""))
    content = escape(reply_text)
    create_time = str(int(time.time()))
    return (
        "<xml>"
        f"<ToUserName>{to_user_name}</ToUserName>"
        f"<FromUserName>{from_user_name}</FromUserName>"
        f"<CreateTime>{create_time}</CreateTime>"
        "<MsgType>text</MsgType>"
        f"<Content>{content}</Content>"
        f"<AgentID>{agent_id}</AgentID>"
        "</xml>"
    )


def _build_wecom_encrypted_reply(
    reply_text: str,
    event: WebhookEvent,
    route: WebhookRouteConfig,
) -> WebhookHttpResponse:
    token = _get_route_option(route, "token")
    encoding_aes_key = _get_route_option(route, "encoding_aes_key")
    if token is None or encoding_aes_key is None:
        raise ValueError("企微被动回包缺少 token 或 encoding_aes_key 配置。")

    receive_id = (
        _get_route_option(route, "receive_id")
        or event.metadata.get("wecom_receive_id")
        or event.metadata.get("wecom_to_user_name")
    )
    if not receive_id:
        raise ValueError("企微被动回包缺少 ReceiveId，无法生成加密响应。")

    plaintext_xml = _build_wecom_passive_text_xml(reply_text, event)
    encrypted_payload = _encrypt_wecom_plaintext(
        plaintext_xml,
        encoding_aes_key,
        receive_id,
    )
    timestamp_value = str(int(time.time()))
    nonce_value = os.urandom(8).hex()
    signature_value = _build_wecom_signature(
        token,
        timestamp_value,
        nonce_value,
        encrypted_payload,
    )
    response_xml = (
        "<xml>"
        f"<Encrypt>{escape(encrypted_payload)}</Encrypt>"
        f"<MsgSignature>{signature_value}</MsgSignature>"
        f"<TimeStamp>{timestamp_value}</TimeStamp>"
        f"<Nonce>{nonce_value}</Nonce>"
        "</xml>"
    )
    return build_xml_http_response(response_xml)


def _build_ignored_outcome(provider: str, reason: str) -> WebhookRequestOutcome:
    return WebhookRequestOutcome(
        immediate_response=build_json_http_response(
            {
                "status": "ignored",
                "provider": provider,
                "reason": reason,
            }
        )
    )


def _parse_feishu_text(content: object) -> str:
    if isinstance(content, str):
        stripped_content = content.strip()
        if not stripped_content:
            return ""
        try:
            parsed_content = json.loads(stripped_content)
        except json.JSONDecodeError:
            return stripped_content
        if isinstance(parsed_content, dict):
            return str(parsed_content.get("text", "")).strip()
        return str(parsed_content).strip()
    return ""


def parse_feishu_payload(
    payload: Mapping[str, object],
    route: WebhookRouteConfig,
    *,
    validate_token: bool = True,
) -> WebhookRequestOutcome:
    if validate_token:
        _require_feishu_token(payload, route)
    challenge = _extract_first_non_empty_string(payload, ("challenge",))
    request_type = _extract_first_non_empty_string(
        payload,
        ("type",),
        ("header", "event_type"),
    )
    if challenge and request_type == "url_verification":
        return WebhookRequestOutcome(
            immediate_response=build_json_http_response({"challenge": challenge})
        )

    event = payload.get("event")
    if not isinstance(event, dict):
        raise ValueError("飞书 webhook 缺少 event 对象。")
    message = event.get("message")
    if not isinstance(message, dict):
        raise ValueError("飞书 webhook 缺少 event.message 对象。")
    message_type = str(message.get("message_type", "text")).strip().lower()
    if message_type != "text":
        return _build_ignored_outcome("feishu", "当前仅支持飞书文本消息。")

    sender = event.get("sender")
    sender_payload = sender if isinstance(sender, dict) else {}
    sender_id_payload = sender_payload.get("sender_id")
    if isinstance(sender_id_payload, dict):
        sender_id = _extract_first_non_empty_string(
            sender_id_payload,
            ("open_id",),
            ("union_id",),
            ("user_id",),
        )
    else:
        sender_id = str(sender_id_payload or "").strip()

    text = _parse_feishu_text(message.get("content"))
    if not text:
        return _build_ignored_outcome("feishu", "飞书文本消息为空，已忽略。")

    session_key = str(
        message.get("chat_id") or sender_id or message.get("message_id") or "feishu-session"
    )
    reply_webhook_url = route.reply_webhook_url or _extract_first_non_empty_string(
        payload,
        ("reply_webhook_url",),
        ("event", "reply_webhook_url"),
    )
    return WebhookRequestOutcome(
        event=WebhookEvent(
            provider="feishu",
            session_key=session_key,
            sender_id=sender_id or session_key,
            sender_name=_extract_first_non_empty_string(
                sender_payload,
                ("sender_id", "open_id"),
            )
            or sender_id
            or "unknown",
            message_id=str(message.get("message_id") or session_key),
            text=text,
            reply_webhook_url=reply_webhook_url or None,
            metadata={
                "chat_id": str(message.get("chat_id", "")),
                "message_type": message_type,
                "schema": _extract_first_non_empty_string(payload, ("schema",)),
                "event_type": request_type,
            },
        )
    )


def parse_feishu_request(
    method: str,
    headers: Mapping[str, str],
    query: Mapping[str, list[str]],
    body: bytes,
    route: WebhookRouteConfig,
) -> WebhookRequestOutcome:
    _ = method, query
    payload = _parse_json_payload(body)
    encrypt_key = _get_route_option(route, "encrypt_key")
    if encrypt_key is not None and isinstance(payload.get("encrypt"), str):
        payload = _decrypt_feishu_payload(str(payload["encrypt"]), encrypt_key)

    challenge = str(payload.get("challenge", "")).strip()
    request_type = str(payload.get("type", "")).strip()
    if encrypt_key is not None and request_type != "url_verification":
        _verify_feishu_signature(headers, body, encrypt_key)
    return parse_feishu_payload(payload, route)


def parse_dingtalk_request(
    method: str,
    headers: Mapping[str, str],
    query: Mapping[str, list[str]],
    body: bytes,
    route: WebhookRouteConfig,
) -> WebhookRequestOutcome:
    payload = _parse_json_payload(body)
    challenge = str(payload.get("challenge", "")).strip()
    if challenge and not payload.get("text"):
        return WebhookRequestOutcome(
            immediate_response=build_json_http_response({"challenge": challenge})
        )

    message_type = str(payload.get("msgtype", "text")).strip().lower()
    if message_type != "text":
        return _build_ignored_outcome("dingtalk", "当前仅支持钉钉文本消息。")

    text = _extract_first_non_empty_string(
        payload,
        ("text", "content"),
        ("content",),
    )
    if not text:
        return _build_ignored_outcome("dingtalk", "钉钉文本消息为空，已忽略。")

    sender_id = _extract_first_non_empty_string(
        payload,
        ("senderStaffId",),
        ("senderId",),
        ("chatbotUserId",),
    )
    session_key = _extract_first_non_empty_string(
        payload,
        ("conversationId",),
        ("chatbotConversationId",),
    ) or sender_id
    reply_webhook_url = _extract_first_non_empty_string(
        payload,
        ("sessionWebhook",),
        ("reply_webhook_url",),
    ) or route.reply_webhook_url
    return WebhookRequestOutcome(
        event=WebhookEvent(
            provider="dingtalk",
            session_key=session_key or "dingtalk-session",
            sender_id=sender_id or session_key or "unknown",
            sender_name=_extract_first_non_empty_string(payload, ("senderNick",)) or sender_id or "unknown",
            message_id=_extract_first_non_empty_string(payload, ("msgId",)) or session_key or "unknown",
            text=text,
            reply_webhook_url=reply_webhook_url or None,
            metadata={
                "conversation_type": _extract_first_non_empty_string(payload, ("conversationType",)),
                "chatbot_corp_id": _extract_first_non_empty_string(payload, ("chatbotCorpId",)),
            },
        )
    )


def parse_wecom_request(
    method: str,
    headers: Mapping[str, str],
    query: Mapping[str, list[str]],
    body: bytes,
    route: WebhookRouteConfig,
) -> WebhookRequestOutcome:
    token = _get_route_option(route, "token")
    encoding_aes_key = _get_route_option(route, "encoding_aes_key")
    receive_id = _get_route_option(route, "receive_id")
    has_official_callback_config = token is not None and encoding_aes_key is not None

    if method.upper() == "GET":
        echo_string = query.get(WECOM_ECHOSTR_QUERY_KEY, [""])[0].strip()
        if echo_string:
            if has_official_callback_config:
                timestamp_value = query.get(WECOM_TIMESTAMP_QUERY_KEY, [""])[0].strip()
                nonce_value = query.get(WECOM_NONCE_QUERY_KEY, [""])[0].strip()
                signature_value = query.get(WECOM_MESSAGE_SIGNATURE_QUERY_KEY, [""])[0].strip()
                if not timestamp_value or not nonce_value or not signature_value:
                    raise WebhookAuthorizationError("企微 URL 校验缺少签名参数。")
                _verify_wecom_signature(
                    token,
                    timestamp_value,
                    nonce_value,
                    echo_string,
                    signature_value,
                )
                echo_string = _decrypt_wecom_ciphertext(
                    echo_string,
                    encoding_aes_key,
                    expected_receive_id=receive_id,
                )
            return WebhookRequestOutcome(
                immediate_response=build_text_http_response(echo_string)
            )

    content_type = headers.get("content-type", "").lower()
    request_text = body.decode("utf-8", errors="replace").strip()
    if request_text.startswith("<xml") or "xml" in content_type:
        if not has_official_callback_config:
            return _build_ignored_outcome(
                "wecom",
                "当前版本仅支持经 webhook 网关解密后的 JSON 企微回调，或在 provider_options 中补充 token 与 encoding_aes_key。",
            )

        timestamp_value = query.get(WECOM_TIMESTAMP_QUERY_KEY, [""])[0].strip()
        nonce_value = query.get(WECOM_NONCE_QUERY_KEY, [""])[0].strip()
        signature_value = query.get(WECOM_MESSAGE_SIGNATURE_QUERY_KEY, [""])[0].strip()
        if not timestamp_value or not nonce_value or not signature_value:
            raise WebhookAuthorizationError("企微 XML 回调缺少签名参数。")

        encrypted_payload = _parse_xml_payload(request_text).get("Encrypt", "").strip()
        if not encrypted_payload:
            raise ValueError("企微 XML 回调缺少 Encrypt 字段。")
        _verify_wecom_signature(
            token,
            timestamp_value,
            nonce_value,
            encrypted_payload,
            signature_value,
        )
        decrypted_xml = _decrypt_wecom_ciphertext(
            encrypted_payload,
            encoding_aes_key,
            expected_receive_id=receive_id,
        )
        payload = _parse_xml_payload(decrypted_xml)
        message_type = str(payload.get("MsgType", "text")).strip().lower()
        if message_type != "text":
            return _build_ignored_outcome("wecom", "当前仅支持企微文本消息。")

        text = str(payload.get("Content", "")).strip()
        if not text:
            return _build_ignored_outcome("wecom", "企微文本消息为空，已忽略。")

        sender_id = str(payload.get("FromUserName", "")).strip()
        target_id = str(payload.get("ToUserName", "")).strip()
        agent_id = str(payload.get("AgentID", "")).strip()
        response_mode = (
            _get_route_option(route, "reply_mode")
            or ("reply_webhook" if route.reply_webhook_url else "passive_xml")
        ).strip().lower()
        reply_webhook_url = (
            route.reply_webhook_url if response_mode == "reply_webhook" else None
        )
        session_key = (
            str(payload.get("ConversationId", "")).strip()
            or str(payload.get("ExternalUserID", "")).strip()
            or sender_id
        )
        return WebhookRequestOutcome(
            event=WebhookEvent(
                provider="wecom",
                session_key=session_key or "wecom-session",
                sender_id=sender_id or session_key or "unknown",
                sender_name=sender_id or "unknown",
                message_id=str(payload.get("MsgId", "")).strip() or session_key or "unknown",
                text=text,
                reply_webhook_url=reply_webhook_url,
                metadata={
                    "wecom_response_mode": response_mode,
                    "wecom_from_user_name": sender_id,
                    "wecom_to_user_name": target_id,
                    "wecom_agent_id": agent_id,
                    "wecom_receive_id": receive_id or target_id,
                },
            )
        )

    payload = _parse_request_payload(headers, body)
    message_type = _extract_first_non_empty_string(payload, ("msgtype",)) or "text"
    if message_type.lower() != "text":
        return _build_ignored_outcome("wecom", "当前仅支持企微文本消息。")

    text = _extract_first_non_empty_string(
        payload,
        ("text", "content"),
        ("content",),
        ("message", "text"),
    )
    if not text:
        return _build_ignored_outcome("wecom", "企微文本消息为空，已忽略。")

    sender_id = _extract_first_non_empty_string(
        payload,
        ("userid",),
        ("from_user",),
        ("sender", "userid"),
        ("sender", "id"),
    )
    session_key = _extract_first_non_empty_string(
        payload,
        ("conversation_id",),
        ("chatid",),
        ("session_id",),
    ) or sender_id
    reply_webhook_url = _extract_first_non_empty_string(
        payload,
        ("reply_webhook_url",),
    ) or route.reply_webhook_url
    return WebhookRequestOutcome(
        event=WebhookEvent(
            provider="wecom",
            session_key=session_key or "wecom-session",
            sender_id=sender_id or session_key or "unknown",
            sender_name=_extract_first_non_empty_string(
                payload,
                ("sender", "name"),
                ("name",),
            )
            or sender_id
            or "unknown",
            message_id=_extract_first_non_empty_string(
                payload,
                ("msgid",),
                ("message_id",),
            )
            or session_key
            or "unknown",
            text=text,
            reply_webhook_url=reply_webhook_url or None,
        )
    )


def parse_email_request(
    method: str,
    headers: Mapping[str, str],
    query: Mapping[str, list[str]],
    body: bytes,
    route: WebhookRouteConfig,
) -> WebhookRequestOutcome:
    payload = _parse_request_payload(headers, body)
    sender_id = _extract_first_non_empty_string(
        payload,
        ("from",),
        ("From",),
        ("sender",),
        ("sender_email",),
        ("envelope", "from"),
    )
    subject = _extract_first_non_empty_string(
        payload,
        ("subject",),
        ("Subject",),
    )
    text = _extract_first_non_empty_string(
        payload,
        ("text",),
        ("plain",),
        ("TextBody",),
        ("stripped-text",),
        ("body_plain",),
    )
    if not text:
        return _build_ignored_outcome("email", "邮件正文为空，已忽略。")

    reply_webhook_url = _extract_first_non_empty_string(
        payload,
        ("reply_webhook_url",),
    ) or route.reply_webhook_url
    return WebhookRequestOutcome(
        event=WebhookEvent(
            provider="email",
            session_key=sender_id or _extract_first_non_empty_string(payload, ("MessageID",)) or "email-session",
            sender_id=sender_id or "unknown@example.com",
            sender_name=sender_id or "unknown@example.com",
            message_id=_extract_first_non_empty_string(
                payload,
                ("message_id",),
                ("MessageID",),
            )
            or sender_id
            or "unknown",
            text=text,
            reply_webhook_url=reply_webhook_url or None,
            subject=subject or None,
        )
    )


def build_feishu_reply_payload(reply_text: str, event: WebhookEvent) -> dict[str, object]:
    return {
        "msg_type": "text",
        "content": {"text": reply_text},
    }


def build_dingtalk_reply_payload(reply_text: str, event: WebhookEvent) -> dict[str, object]:
    return {
        "msgtype": "text",
        "text": {"content": reply_text},
    }


def build_wecom_reply_payload(reply_text: str, event: WebhookEvent) -> dict[str, object]:
    return {
        "msgtype": "text",
        "text": {"content": reply_text},
    }


def build_email_reply_payload(reply_text: str, event: WebhookEvent) -> dict[str, object]:
    reply_subject = event.subject.strip() if event.subject else "Cyber Agent CLI 回复"
    if not reply_subject.lower().startswith("re:"):
        reply_subject = f"Re: {reply_subject}"
    return {
        "to": event.sender_id,
        "subject": reply_subject,
        "text": reply_text,
        "in_reply_to": event.message_id,
    }


@dataclass(slots=True, frozen=True)
class WebhookProviderAdapter:
    """定义单个平台的解析与回包规则。"""

    provider: str
    parse_request: Callable[
        [str, Mapping[str, str], Mapping[str, list[str]], bytes, WebhookRouteConfig],
        WebhookRequestOutcome,
    ]
    build_reply_payload: Callable[[str, WebhookEvent], dict[str, object]]
    supports_sync_response: bool = False


WEBHOOK_PROVIDER_ADAPTERS: dict[str, WebhookProviderAdapter] = {
    "feishu": WebhookProviderAdapter(
        provider="feishu",
        parse_request=parse_feishu_request,
        build_reply_payload=build_feishu_reply_payload,
    ),
    "dingtalk": WebhookProviderAdapter(
        provider="dingtalk",
        parse_request=parse_dingtalk_request,
        build_reply_payload=build_dingtalk_reply_payload,
        supports_sync_response=True,
    ),
    "wecom": WebhookProviderAdapter(
        provider="wecom",
        parse_request=parse_wecom_request,
        build_reply_payload=build_wecom_reply_payload,
    ),
    "email": WebhookProviderAdapter(
        provider="email",
        parse_request=parse_email_request,
        build_reply_payload=build_email_reply_payload,
    ),
}


def build_webhook_session_id(provider: str, session_key: str) -> str:
    """将第三方会话键转为适合本地文件存储的稳定会话 ID。"""
    normalized_provider = normalize_webhook_provider(provider)
    normalized_key = re.sub(r"[^a-zA-Z0-9._-]+", "-", session_key.strip().lower())
    normalized_key = normalized_key.strip("-._") or "session"
    normalized_key = normalized_key[:WEBHOOK_SESSION_ID_MAX_SLUG_LENGTH]
    digest = sha1(f"{normalized_provider}:{session_key}".encode("utf-8")).hexdigest()[:12]
    return f"webhook-{normalized_provider}-{normalized_key}-{digest}"


class WebhookGateway:
    """承载 webhook 解析、Agent 调用与回复投递的统一网关。"""

    def __init__(
        self,
        routes: list[WebhookRouteConfig],
        runtime_context: dict[str, object],
        runner_factory: AgentRunnerFactory,
        *,
        cli_renderer: CliRenderer | None = None,
        base_dir: Path | None = None,
        reply_timeout_seconds: float = DEFAULT_WEBHOOK_REPLY_TIMEOUT_SECONDS,
        reply_sender: ReplySender | None = None,
    ) -> None:
        if not routes:
            raise ValueError("webhook 路由列表不能为空。")
        self.routes = routes
        self.runtime_context = runtime_context
        self.runner_factory = runner_factory
        self.cli_renderer = cli_renderer or CliRenderer()
        self.base_dir = base_dir
        self.reply_timeout_seconds = max(1.0, reply_timeout_seconds)
        self.reply_sender = reply_sender or send_webhook_json
        self._processing_lock = threading.Lock()
        self._feishu_token_lock = threading.Lock()
        self._feishu_session_state_lock = threading.Lock()
        self._feishu_token_cache: dict[str, tuple[str, float]] = {}
        self._async_event_queue: Queue[tuple[WebhookRouteConfig, WebhookEvent]] = Queue()
        self._async_worker_thread: threading.Thread | None = None
        self._async_worker_start_lock = threading.Lock()
        self._routes_by_path = {route.path: route for route in routes}

    def describe_routes(self) -> list[str]:
        """返回当前网关已注册路由的摘要。"""
        return describe_webhook_routes(self.routes)

    def wait_until_async_idle(self, timeout_seconds: float = 5.0) -> bool:
        """等待后台异步 webhook 任务处理完成，主要用于测试和联调验证。"""
        deadline = time.time() + max(timeout_seconds, 0.0)
        while time.time() <= deadline:
            if self._async_event_queue.unfinished_tasks == 0:
                return True
            time.sleep(0.01)
        return self._async_event_queue.unfinished_tasks == 0

    def handle_request(
        self,
        method: str,
        raw_path: str,
        headers: Mapping[str, str],
        body: bytes,
    ) -> WebhookHttpResponse:
        parsed_url = urlsplit(raw_path)
        route = self._routes_by_path.get(parsed_url.path.rstrip("/") or "/")
        if route is None:
            return build_json_http_response(
                {
                    "status": "not_found",
                    "reason": f"未找到匹配的 webhook 路由：{parsed_url.path}",
                },
                status_code=404,
            )

        query = parse_qs(parsed_url.query, keep_blank_values=True)
        authorization_error = self._authorize_request(route, headers, query)
        if authorization_error is not None:
            return authorization_error

        adapter = WEBHOOK_PROVIDER_ADAPTERS[route.provider]
        try:
            outcome = adapter.parse_request(method, headers, query, body, route)
        except WebhookAuthorizationError as exc:
            return build_json_http_response(
                {
                    "status": "unauthorized",
                    "provider": route.provider,
                    "reason": str(exc),
                },
                status_code=401,
            )
        except ValueError as exc:
            return build_json_http_response(
                {
                    "status": "bad_request",
                    "provider": route.provider,
                    "reason": str(exc),
                },
                status_code=400,
            )

        if outcome.immediate_response is not None:
            return outcome.immediate_response
        if outcome.event is None:
            return build_json_http_response(
                {
                    "status": "ignored",
                    "provider": route.provider,
                    "reason": "当前请求未提取到可处理的消息事件。",
                }
            )

        if self._should_handle_event_async(route, outcome.event):
            self._enqueue_async_event(route, outcome.event)
            self.cli_renderer.print_info(
                "飞书 webhook 已快速确认，消息已转入后台处理："
                f"message_id={outcome.event.message_id} "
                f"chat_id={outcome.event.metadata.get('chat_id', '') or 'unknown'}"
            )
            return build_json_http_response({"msg": "success"})

        with self._processing_lock:
            agent_reply = self._run_agent_turn(route, outcome.event)
        return self._deliver_reply(route, adapter, outcome.event, agent_reply)

    def handle_event(
        self,
        route: WebhookRouteConfig,
        event: WebhookEvent,
    ) -> WebhookHttpResponse:
        adapter = WEBHOOK_PROVIDER_ADAPTERS[route.provider]
        with self._processing_lock:
            agent_reply = self._run_agent_turn(route, event)
        return self._deliver_reply(route, adapter, event, agent_reply)

    def _should_handle_event_async(
        self,
        route: WebhookRouteConfig,
        event: WebhookEvent,
    ) -> bool:
        """飞书官方消息接口模式需要先快速确认请求，避免平台侧回调超时。"""
        if route.provider != "feishu":
            return False
        return _resolve_feishu_reply_mode(route, event) in {
            FEISHU_REPLY_API_MODE,
            FEISHU_CREATE_API_MODE,
        }

    def _ensure_async_worker_started(self) -> None:
        if self._async_worker_thread is not None and self._async_worker_thread.is_alive():
            return
        with self._async_worker_start_lock:
            if self._async_worker_thread is not None and self._async_worker_thread.is_alive():
                return
            self._async_worker_thread = threading.Thread(
                target=self._async_worker_loop,
                name="webhook-async-worker",
                daemon=True,
            )
            self._async_worker_thread.start()

    def _enqueue_async_event(
        self,
        route: WebhookRouteConfig,
        event: WebhookEvent,
    ) -> None:
        self._ensure_async_worker_started()
        self._async_event_queue.put((route, event))

    def _async_worker_loop(self) -> None:
        while True:
            route, event = self._async_event_queue.get()
            try:
                response = self.handle_event(route, event)
                if response.status_code >= 400:
                    self.cli_renderer.print_error(
                        "飞书 webhook 后台回复失败："
                        f"message_id={event.message_id} "
                        f"reason={_extract_webhook_response_reason(response)}"
                    )
            except Exception as exc:  # noqa: BLE001 - 后台线程需保留真实错误便于排查
                self.cli_renderer.print_error(
                    "飞书 webhook 后台处理异常："
                    f"message_id={event.message_id} "
                    f"reason={exc}"
                )
            finally:
                self._async_event_queue.task_done()

    def _authorize_request(
        self,
        route: WebhookRouteConfig,
        headers: Mapping[str, str],
        query: Mapping[str, list[str]],
    ) -> WebhookHttpResponse | None:
        if not route.secret:
            return None
        provided_secret = headers.get(WEBHOOK_SECRET_HEADER, "").strip()
        if not provided_secret:
            provided_secret = query.get(WEBHOOK_SECRET_QUERY_KEY, [""])[0].strip()
        if provided_secret == route.secret:
            return None
        if route.provider == "feishu" and (
            _get_route_option(route, "verification_token")
            or _get_route_option(route, "encrypt_key")
        ):
            return None
        if route.provider == "wecom" and (
            _get_route_option(route, "token")
            and _get_route_option(route, "encoding_aes_key")
        ):
            return None
        return build_json_http_response(
            {
                "status": "unauthorized",
                "provider": route.provider,
                "reason": "webhook 共享密钥校验失败。",
            },
            status_code=401,
        )

    def _get_feishu_chat_id(self, event: WebhookEvent) -> str:
        """读取飞书事件中的 chat_id。"""
        return str(event.metadata.get("chat_id", "")).strip()

    def _resolve_source_session_id(self, event: WebhookEvent) -> str:
        """为当前 webhook 事件生成适合历史检索的来源会话分组。"""
        if event.provider == "feishu":
            chat_id = self._get_feishu_chat_id(event)
            if chat_id:
                return _build_feishu_chat_scope_id(chat_id)
        return f"{event.provider}:{event.sender_id}"

    def _get_or_create_feishu_chat_state(
        self,
        chat_id: str,
    ) -> tuple[dict[str, object], dict[str, object], bool]:
        """加载并标准化单个飞书聊天的活动会话状态。"""
        state_payload = _load_feishu_session_state(self.base_dir)
        chats_payload = state_payload.get("chats")
        if not isinstance(chats_payload, dict):
            chats_payload = {}
            state_payload["chats"] = chats_payload
        raw_chat_state = chats_payload.get(chat_id)
        chat_state = dict(raw_chat_state) if isinstance(raw_chat_state, dict) else {}
        state_changed = not isinstance(raw_chat_state, dict)

        raw_session_entries = chat_state.get("sessions")
        normalized_session_entries: list[dict[str, str]] = []
        seen_session_ids: set[str] = set()
        if isinstance(raw_session_entries, list):
            for raw_entry in raw_session_entries:
                if not isinstance(raw_entry, Mapping):
                    state_changed = True
                    continue
                session_key = str(raw_entry.get("session_key", "")).strip()
                if not session_key:
                    state_changed = True
                    continue
                session_id = (
                    str(raw_entry.get("session_id", "")).strip()
                    or build_webhook_session_id("feishu", session_key)
                )
                if session_id in seen_session_ids:
                    state_changed = True
                    continue
                normalized_session_entries.append(
                    {
                        "session_key": session_key,
                        "session_id": session_id,
                        "label": str(raw_entry.get("label", "")).strip(),
                        "created_at": str(raw_entry.get("created_at", "")).strip()
                        or datetime.now().astimezone().isoformat(),
                    }
                )
                seen_session_ids.add(session_id)
        elif raw_session_entries is not None:
            state_changed = True

        default_session_entry = _build_feishu_session_entry(
            chat_id,
            label=FEISHU_DEFAULT_SESSION_LABEL,
        )
        if default_session_entry["session_id"] not in seen_session_ids:
            normalized_session_entries.insert(0, default_session_entry)
            seen_session_ids.add(default_session_entry["session_id"])
            state_changed = True

        active_session_key = str(chat_state.get("active_session_key", "")).strip()
        if active_session_key and not any(
            entry["session_key"] == active_session_key
            for entry in normalized_session_entries
        ):
            active_session_key = ""
            state_changed = True
        if not active_session_key:
            active_session_key = chat_id
            state_changed = True

        chat_state["sessions"] = normalized_session_entries
        chat_state["active_session_key"] = active_session_key
        chats_payload[chat_id] = chat_state
        return state_payload, chat_state, state_changed

    def _resolve_feishu_active_session_key(self, event: WebhookEvent) -> str:
        """解析当前飞书聊天正在使用的活动会话键。"""
        chat_id = self._get_feishu_chat_id(event)
        if not chat_id:
            return event.session_key
        with self._feishu_session_state_lock:
            state_payload, chat_state, state_changed = self._get_or_create_feishu_chat_state(
                chat_id
            )
            if state_changed:
                _save_feishu_session_state(state_payload, self.base_dir)
            return str(chat_state.get("active_session_key", "")).strip() or chat_id

    def _set_feishu_active_session_key(
        self,
        chat_id: str,
        session_key: str,
    ) -> None:
        """切换飞书聊天当前使用的活动会话。"""
        with self._feishu_session_state_lock:
            state_payload, chat_state, _ = self._get_or_create_feishu_chat_state(chat_id)
            session_entries = list(chat_state.get("sessions", []))
            if not any(
                entry["session_key"] == session_key
                for entry in session_entries
                if isinstance(entry, Mapping)
            ):
                session_entries.insert(0, _build_feishu_session_entry(session_key))
                chat_state["sessions"] = session_entries
            chat_state["active_session_key"] = session_key
            state_payload.setdefault("chats", {})
            assert isinstance(state_payload["chats"], dict)
            state_payload["chats"][chat_id] = chat_state
            _save_feishu_session_state(state_payload, self.base_dir)

    def _create_feishu_chat_session(
        self,
        chat_id: str,
        *,
        label: str = "",
    ) -> dict[str, str]:
        """为当前飞书聊天创建新的可切换会话。"""
        new_session_key = f"{chat_id}::{create_session_id()}"
        new_session_entry = _build_feishu_session_entry(
            new_session_key,
            label=label,
        )
        with self._feishu_session_state_lock:
            state_payload, chat_state, _ = self._get_or_create_feishu_chat_state(chat_id)
            session_entries = [
                entry
                for entry in chat_state.get("sessions", [])
                if isinstance(entry, Mapping)
                and str(entry.get("session_id", "")).strip() != new_session_entry["session_id"]
            ]
            chat_state["sessions"] = [new_session_entry, *session_entries]
            chat_state["active_session_key"] = new_session_key
            state_payload.setdefault("chats", {})
            assert isinstance(state_payload["chats"], dict)
            state_payload["chats"][chat_id] = chat_state
            _save_feishu_session_state(state_payload, self.base_dir)
        return new_session_entry

    def _list_feishu_chat_sessions(
        self,
        event: WebhookEvent,
    ) -> list[dict[str, object]]:
        """列出当前飞书聊天下可切换的会话摘要。"""
        chat_id = self._get_feishu_chat_id(event)
        if not chat_id:
            return []
        default_session_id = build_webhook_session_id("feishu", chat_id)
        chat_scope_id = _build_feishu_chat_scope_id(chat_id)
        summary_by_session_id = {
            summary.session_id: summary
            for summary in list_stored_sessions(base_dir=self.base_dir)
            if summary.session_id == default_session_id
            or summary.source_session_id == chat_scope_id
        }
        with self._feishu_session_state_lock:
            state_payload, chat_state, state_changed = self._get_or_create_feishu_chat_state(
                chat_id
            )
            if state_changed:
                _save_feishu_session_state(state_payload, self.base_dir)
            active_session_key = str(chat_state.get("active_session_key", "")).strip() or chat_id
            raw_session_entries = list(chat_state.get("sessions", []))

        session_items: list[dict[str, object]] = []
        for raw_entry in raw_session_entries:
            if not isinstance(raw_entry, Mapping):
                continue
            session_key = str(raw_entry.get("session_key", "")).strip()
            session_id = str(raw_entry.get("session_id", "")).strip()
            if not session_key or not session_id:
                continue
            stored_summary = summary_by_session_id.get(session_id)
            session_title = (
                stored_summary.title.strip()
                if stored_summary is not None and stored_summary.title.strip()
                else str(raw_entry.get("label", "")).strip()
                or (
                    FEISHU_DEFAULT_SESSION_LABEL
                    if session_key == chat_id
                    else "未命名会话"
                )
            )
            session_items.append(
                {
                    "session_key": session_key,
                    "session_id": session_id,
                    "title": session_title,
                    "active": session_key == active_session_key,
                    "is_default": session_key == chat_id,
                    "updated_at": (
                        stored_summary.updated_at
                        if stored_summary is not None
                        else str(raw_entry.get("created_at", "")).strip() or "未开始"
                    ),
                    "turn_count": stored_summary.turn_count if stored_summary is not None else 0,
                    "message_count": (
                        stored_summary.message_count if stored_summary is not None else 0
                    ),
                }
            )

        session_items.sort(
            key=lambda item: (bool(item["active"]), str(item["updated_at"])),
            reverse=True,
        )
        for index, session_item in enumerate(session_items, start=1):
            session_item["index"] = index
        return session_items

    def _resolve_feishu_session_selection(
        self,
        event: WebhookEvent,
        raw_selector: str,
    ) -> dict[str, object]:
        """按序号或会话 ID 解析飞书会话切换目标。"""
        selector = raw_selector.strip()
        if not selector:
            raise ValueError("请提供要切换的会话序号或会话 ID。")
        session_items = self._list_feishu_chat_sessions(event)
        normalized_selector = selector.lower()
        if normalized_selector in {"default", "默认"}:
            default_session = next(
                (
                    session_item
                    for session_item in session_items
                    if bool(session_item.get("is_default"))
                ),
                None,
            )
            if default_session is not None:
                return default_session
            raise ValueError("当前聊天缺少默认会话，请先发送普通消息初始化。")
        if selector.isdigit():
            target_index = int(selector)
            for session_item in session_items:
                if int(session_item["index"]) == target_index:
                    return session_item
            raise ValueError("会话序号超出范围，请先发送 /session list 查看。")
        for session_item in session_items:
            if str(session_item["session_id"]) == selector:
                return session_item
        raise ValueError("未找到指定会话，请先发送 /session list 查看。")

    def _build_feishu_session_command_reply(
        self,
        event: WebhookEvent,
        session_id: str,
        runner: "AgentRunner",
        approval_policy: ApprovalPolicy,
    ) -> WebhookAgentReply | None:
        """处理飞书专属的会话切换命令。"""
        if event.provider != "feishu":
            return None
        stripped_text = event.text.strip()
        normalized_text = stripped_text.lower()
        if normalized_text == "/session":
            normalized_text = "/session current"
        if not normalized_text.startswith("/session"):
            return None

        chat_id = self._get_feishu_chat_id(event)
        if not chat_id:
            notice_payload = _build_feishu_notice_payload(
                "会话切换不可用",
                "当前飞书事件缺少 chat_id，无法在同一聊天里管理多会话。",
                template="red",
                button_commands=("/status",),
            )
            return WebhookAgentReply(
                session_id=session_id,
                reply_text="当前飞书事件缺少 chat_id，无法管理多会话。",
                reply_payload_override=notice_payload,
            )

        if normalized_text == "/session current":
            session_items = self._list_feishu_chat_sessions(event)
            current_session = next(
                (session_item for session_item in session_items if bool(session_item["active"])),
                None,
            )
            context_diagnostics = runner.get_context_diagnostics()
            current_lines = [
                f"- 当前会话 ID：`{current_session['session_id'] if current_session else session_id}`",
                f"- 标题：{current_session['title'] if current_session else '未命名会话'}",
                f"- 更新时间：`{current_session['updated_at'] if current_session else '未开始'}`",
                f"- 轮数：`{current_session['turn_count'] if current_session else 0}`",
                f"- 消息数：`{current_session['message_count'] if current_session else 0}`",
                f"- 历史消息：`{context_diagnostics.get('history_message_count', 0)}`",
                f"- 模型可见：`{context_diagnostics.get('model_message_count', 0)}`",
                f"- 已压缩历史消息：`{context_diagnostics.get('compressed_message_count', 0)}`",
            ]
            if context_diagnostics.get("compressed_summary"):
                current_lines.append("- 当前存在压缩摘要，说明上下文压缩已触发。")
            payload = _build_feishu_interactive_card_payload(
                "当前飞书会话",
                "\n\n".join(
                    section
                    for section in (
                        _build_feishu_markdown_section("当前会话", current_lines),
                        _build_feishu_markdown_section(
                            "可用命令",
                            [
                                "- `/session new` 新建并切换到新会话",
                                "- `/session list` 查看当前聊天下所有会话",
                                "- `/session default` 快速回到默认会话",
                                "- `/session use <序号或会话ID>` 切回指定会话",
                            ],
                        ),
                    )
                    if section
                ),
                template="wathet",
                action_rows=_build_feishu_command_action_rows(
                    FEISHU_SESSION_SHORTCUT_COMMANDS,
                    primary_commands=("/session current",),
                    row_size=3,
                ),
            )
            return WebhookAgentReply(
                session_id=current_session["session_id"] if current_session else session_id,
                reply_text="已显示当前飞书会话状态。",
                reply_payload_override=payload,
            )

        if normalized_text == "/session list":
            session_items = self._list_feishu_chat_sessions(event)
            session_lines = [
                (
                    f"- {'**当前** ' if bool(session_item['active']) else ''}"
                    f"`{session_item['index']}` `"
                    f"{session_item['session_id']}` | "
                    f"{session_item['title']} | "
                    f"轮数 `{session_item['turn_count']}` | "
                    f"更新时间 `{session_item['updated_at']}`"
                )
                for session_item in session_items
            ] or ["- 当前聊天下还没有可切换的会话。"]
            payload = _build_feishu_interactive_card_payload(
                "飞书会话列表",
                "\n\n".join(
                    section
                    for section in (
                        _build_feishu_markdown_section("当前聊天会话", session_lines),
                        _build_feishu_markdown_section(
                            "切换方法",
                            [
                                "- 发送 `/session use 1` 按序号切换",
                                "- 或发送 `/session use <会话ID>` 精确切换",
                                "- 发送 `/session default` 快速回到默认会话",
                            ],
                        ),
                    )
                    if section
                ),
                template="wathet",
                action_rows=_build_feishu_command_action_rows(
                    FEISHU_SESSION_SHORTCUT_COMMANDS,
                    primary_commands=("/session list",),
                    row_size=3,
                ),
            )
            return WebhookAgentReply(
                session_id=session_id,
                reply_text="已显示当前聊天下的飞书会话列表。",
                reply_payload_override=payload,
            )

        if normalized_text.startswith("/session new"):
            session_label = stripped_text[len("/session new") :].strip()
            new_session_entry = self._create_feishu_chat_session(
                chat_id,
                label=session_label,
            )
            runner.reset()
            save_session_history(
                new_session_entry["session_id"],
                runner.get_history_snapshot(),
                mode=runner.mode.value,
                approval_policy=approval_policy.value,
                source_session_id=self._resolve_source_session_id(event),
                base_dir=self.base_dir,
            )
            payload = _build_feishu_notice_payload(
                "已切换到新会话",
                (
                    f"当前活动会话已切换为 `{new_session_entry['session_id']}`。"
                    + (
                        f"\n\n会话备注：{session_label}"
                        if session_label
                        else ""
                    )
                ),
                template="green",
                button_commands=("/session current", "/session list"),
            )
            return WebhookAgentReply(
                session_id=new_session_entry["session_id"],
                reply_text="已创建并切换到新的飞书会话。",
                reply_payload_override=payload,
            )

        if normalized_text == "/session default":
            target_session = self._resolve_feishu_session_selection(event, "default")
            self._set_feishu_active_session_key(
                chat_id,
                str(target_session["session_key"]),
            )
            payload = _build_feishu_notice_payload(
                "已回到默认会话",
                (
                    f"当前活动会话已切换为 `{target_session['session_id']}`。\n\n"
                    f"标题：{target_session['title']}"
                ),
                template="green",
                button_commands=("/session current", "/session list"),
            )
            return WebhookAgentReply(
                session_id=str(target_session["session_id"]),
                reply_text="已切换回默认飞书会话。",
                reply_payload_override=payload,
            )

        if normalized_text.startswith("/session use "):
            raw_selector = stripped_text[len("/session use ") :].strip()
            try:
                target_session = self._resolve_feishu_session_selection(event, raw_selector)
            except ValueError as exc:
                error_payload = _build_feishu_notice_payload(
                    "会话切换失败",
                    str(exc),
                    template="red",
                    button_commands=("/session list",),
                )
                return WebhookAgentReply(
                    session_id=session_id,
                    reply_text=str(exc),
                    reply_payload_override=error_payload,
                )
            self._set_feishu_active_session_key(
                chat_id,
                str(target_session["session_key"]),
            )
            payload = _build_feishu_notice_payload(
                "会话已切换",
                (
                    f"当前活动会话已切换为 `{target_session['session_id']}`。\n\n"
                    f"标题：{target_session['title']}"
                ),
                template="green",
                button_commands=("/session current", "/session list"),
            )
            return WebhookAgentReply(
                session_id=str(target_session["session_id"]),
                reply_text="已切换到指定飞书会话。",
                reply_payload_override=payload,
            )

        error_payload = _build_feishu_notice_payload(
            "会话命令不支持",
            (
                "支持的命令有：`/session`、`/session current`、`/session new`、"
                "`/session list`、`/session default`、`/session use <会话ID|序号>`"
            ),
            template="red",
            button_commands=("/session current", "/session list"),
        )
        return WebhookAgentReply(
            session_id=session_id,
            reply_text="不支持的飞书会话命令。",
            reply_payload_override=error_payload,
        )

    def _build_webhook_builtin_reply(
        self,
        event: WebhookEvent,
        session_id: str,
        runner: "AgentRunner",
        approval_policy: ApprovalPolicy,
    ) -> WebhookAgentReply | None:
        """复用 CLI 内建命令处理链路，让 webhook 会话支持同一套快捷命令。"""
        stripped_text = event.text.strip()
        if not stripped_text.startswith("/"):
            return None

        normalized_text = stripped_text.lower()
        session_reply = self._build_feishu_session_command_reply(
            event,
            session_id,
            runner,
            approval_policy,
        )
        if session_reply is not None:
            return session_reply

        if normalized_text == "/start":
            if event.provider == "feishu":
                return WebhookAgentReply(
                    session_id=session_id,
                    reply_text="已发送飞书快捷菜单。",
                    reply_payload_override=_build_feishu_start_menu_payload(
                        self._list_feishu_chat_sessions(event)
                    ),
                )
            return WebhookAgentReply(
                session_id=session_id,
                reply_text=(
                    "可用快捷命令："
                    + " ".join(FEISHU_START_MENU_COMMANDS)
                ),
            )

        if normalized_text in {"/exit", "/quit", "exit", "quit", "q", ":q"}:
            runner.reset()
            save_session_history(
                session_id,
                runner.get_history_snapshot(),
                mode=runner.mode.value,
                approval_policy=approval_policy.value,
                source_session_id=self._resolve_source_session_id(event),
                base_dir=self.base_dir,
            )
            return WebhookAgentReply(
                session_id=session_id,
                reply_text="当前飞书会话已结束并清空上下文，后续消息将作为新会话重新开始。",
                reply_payload_override=(
                    _build_feishu_notice_payload(
                        "会话已结束",
                        "当前飞书会话上下文已清空，后续消息会作为新会话重新开始。",
                        template="green",
                        button_commands=("/start", "/status"),
                    )
                    if event.provider == "feishu"
                    else None
                ),
            )

        builtin_result, builtin_output = _capture_builtin_command_output_for_webhook(
            stripped_text,
            runner,
            self.runtime_context,
        )
        if builtin_result is None:
            return None

        # webhook 需要显式持久化命令后的会话状态，否则 /clear、/mode 等变更无法延续到下一条消息。
        resolved_runtime_policy = self.runtime_context.get("approval_policy", approval_policy)
        if not isinstance(resolved_runtime_policy, ApprovalPolicy):
            resolved_runtime_policy = approval_policy
        save_session_history(
            session_id,
            runner.get_history_snapshot(),
            mode=runner.mode.value,
            approval_policy=resolved_runtime_policy.value,
            source_session_id=self._resolve_source_session_id(event),
            base_dir=self.base_dir,
        )
        normalized_output = builtin_output.strip()
        reply_payload_override = (
            _build_feishu_builtin_command_payload(
                stripped_text,
                runner,
                self.runtime_context,
                normalized_output,
                base_dir=self.base_dir,
            )
            if event.provider == "feishu"
            else None
        )
        if builtin_result is False:
            return WebhookAgentReply(
                session_id=session_id,
                reply_text=(
                    normalized_output
                    or "当前飞书会话已结束并清空上下文，后续消息将作为新会话重新开始。"
                ),
                reply_payload_override=reply_payload_override,
            )
        return WebhookAgentReply(
            session_id=session_id,
            reply_text=normalized_output or "命令已执行完成。",
            reply_payload_override=reply_payload_override,
        )

    def _send_feishu_message_payload(
        self,
        route: WebhookRouteConfig,
        event: WebhookEvent,
        payload: dict[str, object],
        *,
        purpose: str,
    ) -> None:
        """按当前飞书回包模式发送一条独立消息，供过程步骤即时展示。"""
        feishu_reply_mode = _resolve_feishu_reply_mode(route, event)
        if feishu_reply_mode not in {FEISHU_CREATE_API_MODE, FEISHU_REPLY_API_MODE}:
            return

        tenant_access_token = self._get_feishu_tenant_access_token(route)
        progress_reply = WebhookAgentReply(
            session_id=f"feishu-progress:{event.message_id}",
            reply_text="",
            reply_payload_override=payload,
        )
        if feishu_reply_mode == FEISHU_CREATE_API_MODE:
            chat_id = str(event.metadata.get("chat_id", "")).strip()
            if not chat_id:
                raise ValueError("飞书过程消息缺少 chat_id，无法发送独立进度消息。")
            message_payload = self._build_feishu_api_message_payload(
                route,
                event,
                progress_reply,
                create_chat_message=True,
            )
            message_payload["receive_id"] = chat_id
            delivery_receipt = self._send_reply_webhook(
                FEISHU_CREATE_MESSAGE_URL,
                message_payload,
                {"Authorization": f"Bearer {tenant_access_token}"},
            )
            if delivery_receipt.status_code >= 400:
                raise WebhookDeliveryError(
                    f"飞书过程消息发送接口返回 HTTP {delivery_receipt.status_code}",
                    status_code=delivery_receipt.status_code,
                    response_text=delivery_receipt.response_text,
                )
            self._parse_feishu_message_delivery_receipt(
                delivery_receipt.response_text,
                delivery_receipt.status_code,
                error_prefix=purpose,
            )
            return

        reply_payload = self._build_feishu_api_message_payload(
            route,
            event,
            progress_reply,
            reply_in_thread=_get_route_bool_option(route, "reply_in_thread", False),
        )
        delivery_receipt = self._send_reply_webhook(
            FEISHU_REPLY_MESSAGE_URL_TEMPLATE.format(message_id=event.message_id),
            reply_payload,
            {"Authorization": f"Bearer {tenant_access_token}"},
        )
        if delivery_receipt.status_code >= 400:
            raise WebhookDeliveryError(
                f"飞书过程消息回复接口返回 HTTP {delivery_receipt.status_code}",
                status_code=delivery_receipt.status_code,
                response_text=delivery_receipt.response_text,
            )
        self._parse_feishu_message_delivery_receipt(
            delivery_receipt.response_text,
            delivery_receipt.status_code,
            error_prefix=purpose,
        )

    def _emit_feishu_progress_message(
        self,
        route: WebhookRouteConfig,
        event: WebhookEvent,
        step: FeishuTraceStep,
        step_index: int,
    ) -> None:
        """把单个处理中间步骤作为独立飞书消息发出去，失败时仅记日志。"""
        try:
            progress_payload = dict(
                _build_feishu_progress_payload(step, step_index=step_index)
            )
            progress_payload["uuid"] = sha1(
                f"feishu-progress:{event.message_id}:{step_index}".encode("utf-8")
            ).hexdigest()[:40]
            self._send_feishu_message_payload(
                route,
                event,
                progress_payload,
                purpose="飞书过程消息",
            )
        except Exception as exc:  # noqa: BLE001 - 进度消息失败不应中断最终回答
            self.cli_renderer.print_error(
                "飞书过程消息发送失败："
                f"message_id={event.message_id} "
                f"step={step_index} "
                f"reason={exc}"
            )

    def _run_agent_turn(
        self,
        route: WebhookRouteConfig,
        event: WebhookEvent,
    ) -> WebhookAgentReply:
        runner = self.runner_factory(self.runtime_context)
        resolved_session_key = (
            self._resolve_feishu_active_session_key(event)
            if event.provider == "feishu"
            else event.session_key
        )
        session_id = build_webhook_session_id(event.provider, resolved_session_key)
        try:
            stored_session = load_session_history(session_id, base_dir=self.base_dir)
        except ValueError:
            stored_session = None
        else:
            runner.restore_history(stored_session.messages)

        approval_policy = self.runtime_context.get("approval_policy", ApprovalPolicy.NEVER)
        if not isinstance(approval_policy, ApprovalPolicy):
            approval_policy = ApprovalPolicy.NEVER
        progress_emitter = (
            FeishuProgressMessageEmitter(
                lambda step, step_index: self._emit_feishu_progress_message(
                    route,
                    event,
                    step,
                    step_index,
                )
            )
            if event.provider == "feishu"
            else None
        )

        builtin_reply = self._build_webhook_builtin_reply(
            event,
            session_id,
            runner,
            approval_policy,
        )
        if builtin_reply is not None:
            return builtin_reply

        history_snapshot = runner.get_history_snapshot()
        reply_text = ""
        try:
            if progress_emitter is not None:
                progress_emitter.start(event.text)
            reply_text = runner.run(
                event.text,
                verbose=False,
                event_handler=progress_emitter,
                approval_handler=create_webhook_approval_handler(approval_policy),
            )
            history_snapshot = runner.get_history_snapshot()
        except ModuleNotFoundError as exc:
            reply_text = f"运行失败：{exc}"
            history_snapshot = [
                *history_snapshot,
                AIMessage(content=reply_text),
            ]
        except Exception as exc:  # noqa: BLE001 - webhook 网关需要把真实错误回给上游桥接层
            reply_text = f"处理失败：{exc}"
            history_snapshot = [
                *runner.get_history_snapshot(),
                AIMessage(content=reply_text),
            ]
        finally:
            if progress_emitter is not None:
                progress_emitter.close()

        save_session_history(
            session_id,
            history_snapshot,
            mode=runner.mode.value,
            approval_policy=approval_policy.value,
            source_session_id=self._resolve_source_session_id(event),
            base_dir=self.base_dir,
        )
        normalized_reply_text = reply_text.strip() or "（空回复）"
        return WebhookAgentReply(
            session_id=session_id,
            reply_text=normalized_reply_text,
            reply_payload_override=(
                _build_feishu_ai_reply_payload(normalized_reply_text)
                if event.provider == "feishu"
                else None
            ),
        )

    def _deliver_reply(
        self,
        route: WebhookRouteConfig,
        adapter: WebhookProviderAdapter,
        event: WebhookEvent,
        agent_reply: WebhookAgentReply,
    ) -> WebhookHttpResponse:
        if event.provider == "feishu":
            feishu_reply_mode = _resolve_feishu_reply_mode(route, event)
            if feishu_reply_mode == FEISHU_CREATE_API_MODE:
                try:
                    return self._deliver_feishu_create_api(route, event, agent_reply)
                except (ValueError, WebhookDeliveryError) as exc:
                    return build_json_http_response(
                        {
                            "status": "delivery_failed",
                            "provider": event.provider,
                            "session_id": agent_reply.session_id,
                            "reply_text": agent_reply.reply_text,
                            "reason": str(exc),
                        },
                        status_code=502,
                    )
            if feishu_reply_mode == FEISHU_REPLY_API_MODE:
                try:
                    return self._deliver_feishu_reply_api(route, event, agent_reply)
                except (ValueError, WebhookDeliveryError) as exc:
                    return build_json_http_response(
                        {
                            "status": "delivery_failed",
                            "provider": event.provider,
                            "session_id": agent_reply.session_id,
                            "reply_text": agent_reply.reply_text,
                            "reason": str(exc),
                        },
                        status_code=502,
                    )

        if (
            event.provider == "wecom"
            and event.metadata.get("wecom_response_mode") == "passive_xml"
        ):
            try:
                return _build_wecom_encrypted_reply(
                    agent_reply.reply_text,
                    event,
                    route,
                )
            except (ValueError, WebhookAuthorizationError) as exc:
                return build_json_http_response(
                    {
                        "status": "delivery_failed",
                        "provider": event.provider,
                        "session_id": agent_reply.session_id,
                        "reply_text": agent_reply.reply_text,
                        "reason": str(exc),
                    },
                    status_code=502,
                )

        reply_payload = agent_reply.reply_payload_override or adapter.build_reply_payload(
            agent_reply.reply_text,
            event,
        )
        if event.reply_webhook_url:
            return self._deliver_reply_webhook(
                route,
                event,
                agent_reply,
                reply_payload,
            )

        if adapter.supports_sync_response:
            return build_json_http_response(reply_payload)

        return build_json_http_response(
            {
                "status": "ok",
                "provider": event.provider,
                "session_id": agent_reply.session_id,
                "reply_text": agent_reply.reply_text,
                "reply_payload": reply_payload,
                "reason": "当前路由未配置 reply_webhook_url，已将建议回包返回给上游 webhook 网关。",
            }
        )

    def _send_reply_webhook(
        self,
        target_url: str,
        payload: dict[str, object],
        request_headers: Mapping[str, str] | None,
    ) -> WebhookDeliveryReceipt:
        try:
            return self.reply_sender(
                target_url,
                payload,
                self.reply_timeout_seconds,
                request_headers,
            )
        except TypeError:
            return self.reply_sender(  # type: ignore[misc]
                target_url,
                payload,
                self.reply_timeout_seconds,
            )

    def _deliver_reply_webhook(
        self,
        route: WebhookRouteConfig,
        event: WebhookEvent,
        agent_reply: WebhookAgentReply,
        reply_payload: dict[str, object],
    ) -> WebhookHttpResponse:
        retry_attempts = _get_route_int_option(
            route,
            "reply_retry_attempts",
            DEFAULT_WEBHOOK_REPLY_RETRY_ATTEMPTS,
            minimum=1,
        )
        retry_backoff_seconds = _get_route_float_option(
            route,
            "reply_retry_backoff_seconds",
            DEFAULT_WEBHOOK_REPLY_RETRY_BACKOFF_SECONDS,
            minimum=0.0,
        )
        payload_bytes = _serialize_webhook_json_payload(reply_payload)
        target_url = event.reply_webhook_url or ""
        attempt_records: list[dict[str, object]] = []

        for attempt_index in range(1, retry_attempts + 1):
            request_headers = _build_reply_signature_headers(route, payload_bytes)
            try:
                delivery_receipt = self._send_reply_webhook(
                    target_url,
                    reply_payload,
                    request_headers or None,
                )
                if delivery_receipt.status_code >= 400:
                    raise WebhookDeliveryError(
                        f"reply webhook 返回 HTTP {delivery_receipt.status_code}",
                        status_code=delivery_receipt.status_code,
                        response_text=delivery_receipt.response_text,
                    )
                attempt_records.append(
                    {
                        "attempt": attempt_index,
                        "status": "ok",
                        "status_code": delivery_receipt.status_code,
                        "response_text": delivery_receipt.response_text,
                        "signed": bool(request_headers),
                    }
                )
                return build_json_http_response(
                    {
                        "status": "ok",
                        "provider": event.provider,
                        "session_id": agent_reply.session_id,
                        "delivery": {
                            "method": "reply_webhook",
                            "target": _redact_webhook_url(target_url),
                            "attempt_count": attempt_index,
                            "signed": bool(request_headers),
                            "status_code": delivery_receipt.status_code,
                            "response_text": delivery_receipt.response_text,
                        },
                    }
                )
            except Exception as exc:  # noqa: BLE001 - 需要记录每次真实失败原因
                attempt_record = {
                    "attempt": attempt_index,
                    "status": "failed",
                    "reason": str(exc),
                    "signed": bool(request_headers),
                }
                if isinstance(exc, WebhookDeliveryError) and exc.status_code is not None:
                    attempt_record["status_code"] = exc.status_code
                    attempt_record["response_text"] = exc.response_text or ""
                attempt_records.append(attempt_record)
                if attempt_index < retry_attempts and retry_backoff_seconds > 0:
                    time.sleep(retry_backoff_seconds)

        dead_letter_path = _write_delivery_dead_letter(
            route,
            event,
            agent_reply,
            target_url,
            reply_payload,
            attempt_records,
            base_dir=self.base_dir,
        )
        return build_json_http_response(
            {
                "status": "delivery_failed",
                "provider": event.provider,
                "session_id": agent_reply.session_id,
                "reply_text": agent_reply.reply_text,
                "reason": attempt_records[-1]["reason"] if attempt_records else "未知错误",
                "dead_letter_path": str(dead_letter_path),
                "delivery": {
                    "method": "reply_webhook",
                    "target": _redact_webhook_url(target_url),
                    "attempt_count": len(attempt_records),
                    "attempts": attempt_records,
                },
            },
            status_code=502,
        )

    def _get_feishu_tenant_access_token(self, route: WebhookRouteConfig) -> str:
        app_id = _get_route_option(route, "app_id")
        app_secret = _get_route_option(route, "app_secret")
        if not app_id or not app_secret:
            raise ValueError("飞书官方回复模式缺少 provider_options.app_id 或 provider_options.app_secret。")

        cache_key = f"{app_id}:{app_secret}"
        current_time = time.time()
        with self._feishu_token_lock:
            cached_token = self._feishu_token_cache.get(cache_key)
            if cached_token is not None:
                token_value, expire_at = cached_token
                if expire_at - FEISHU_TOKEN_CACHE_SAFETY_SECONDS > current_time:
                    return token_value

        token_payload = {
            "app_id": app_id,
            "app_secret": app_secret,
        }
        delivery_receipt = self._send_reply_webhook(
            FEISHU_TENANT_ACCESS_TOKEN_URL,
            token_payload,
            None,
        )
        if delivery_receipt.status_code >= 400:
            raise WebhookDeliveryError(
                f"飞书 tenant_access_token 接口返回 HTTP {delivery_receipt.status_code}",
                status_code=delivery_receipt.status_code,
                response_text=delivery_receipt.response_text,
            )

        try:
            response_payload = json.loads(delivery_receipt.response_text or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError("飞书 tenant_access_token 接口返回的响应不是合法 JSON。") from exc

        if not isinstance(response_payload, dict):
            raise ValueError("飞书 tenant_access_token 接口返回了非对象 JSON。")

        if int(response_payload.get("code", -1)) != 0:
            raise WebhookDeliveryError(
                f"飞书 tenant_access_token 获取失败：{response_payload.get('msg', 'unknown error')}",
                status_code=delivery_receipt.status_code,
                response_text=delivery_receipt.response_text,
            )

        token_value = str(response_payload.get("tenant_access_token", "")).strip()
        if not token_value:
            raise ValueError("飞书 tenant_access_token 接口响应缺少 tenant_access_token。")

        expire_seconds = max(int(response_payload.get("expire", 7200) or 7200), 1)
        with self._feishu_token_lock:
            self._feishu_token_cache[cache_key] = (
                token_value,
                time.time() + expire_seconds,
            )
        return token_value

    def _build_feishu_api_message_payload(
        self,
        route: WebhookRouteConfig,
        event: WebhookEvent,
        agent_reply: WebhookAgentReply,
        *,
        reply_in_thread: bool = False,
        create_chat_message: bool = False,
    ) -> dict[str, object]:
        """统一构造飞书官方消息 API 的请求体，兼容文本与交互卡片。"""
        reply_payload = dict(
            agent_reply.reply_payload_override
            or _build_feishu_text_message_payload(agent_reply.reply_text)
        )
        if "msg_type" not in reply_payload:
            raise ValueError("飞书消息体缺少 msg_type，无法调用官方消息接口。")
        if "content" not in reply_payload:
            raise ValueError("飞书消息体缺少 content，无法调用官方消息接口。")

        if "uuid" not in reply_payload:
            uuid_source = (
                f"feishu-create:{event.message_id}"
                if create_chat_message
                else f"feishu-reply:{event.message_id}"
            )
            reply_payload["uuid"] = hashlib.sha1(uuid_source.encode("utf-8")).hexdigest()[:40]

        if reply_in_thread and not create_chat_message:
            reply_payload["reply_in_thread"] = True
        elif "reply_in_thread" in reply_payload and create_chat_message:
            reply_payload.pop("reply_in_thread", None)
        return reply_payload

    def _parse_feishu_message_delivery_receipt(
        self,
        response_text: str,
        status_code: int,
        *,
        error_prefix: str,
    ) -> str:
        """统一解析飞书消息接口响应，返回成功后的 message_id。"""
        try:
            response_payload = json.loads(response_text or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"{error_prefix}返回的响应不是合法 JSON。") from exc

        if not isinstance(response_payload, dict):
            raise ValueError(f"{error_prefix}返回了非对象 JSON。")

        if int(response_payload.get("code", -1)) != 0:
            raise WebhookDeliveryError(
                f"{error_prefix}失败：{response_payload.get('msg', 'unknown error')}",
                status_code=status_code,
                response_text=response_text,
            )

        response_data = response_payload.get("data")
        if not isinstance(response_data, dict):
            return ""
        return str(response_data.get("message_id", "")).strip()

    def _deliver_feishu_create_api(
        self,
        route: WebhookRouteConfig,
        event: WebhookEvent,
        agent_reply: WebhookAgentReply,
    ) -> WebhookHttpResponse:
        tenant_access_token = self._get_feishu_tenant_access_token(route)
        chat_id = str(event.metadata.get("chat_id", "")).strip()
        if not chat_id:
            raise ValueError("飞书发送消息模式缺少 chat_id，无法向会话发送新消息。")

        message_payload = self._build_feishu_api_message_payload(
            route,
            event,
            agent_reply,
            create_chat_message=True,
        )
        message_payload["receive_id"] = chat_id
        delivery_receipt = self._send_reply_webhook(
            FEISHU_CREATE_MESSAGE_URL,
            message_payload,
            {
                "Authorization": f"Bearer {tenant_access_token}",
            },
        )
        if delivery_receipt.status_code >= 400:
            raise WebhookDeliveryError(
                f"飞书发送消息接口返回 HTTP {delivery_receipt.status_code}",
                status_code=delivery_receipt.status_code,
                response_text=delivery_receipt.response_text,
            )

        reply_message_id = self._parse_feishu_message_delivery_receipt(
            delivery_receipt.response_text,
            delivery_receipt.status_code,
            error_prefix="飞书发送消息",
        )
        return build_json_http_response(
            {
                "status": "ok",
                "provider": event.provider,
                "session_id": agent_reply.session_id,
                "delivery": {
                    "method": "feishu_create_api",
                    "target": chat_id,
                    "status_code": delivery_receipt.status_code,
                    "message_id": reply_message_id,
                },
            }
        )

    def _deliver_feishu_reply_api(
        self,
        route: WebhookRouteConfig,
        event: WebhookEvent,
        agent_reply: WebhookAgentReply,
    ) -> WebhookHttpResponse:
        tenant_access_token = self._get_feishu_tenant_access_token(route)
        reply_payload = self._build_feishu_api_message_payload(
            route,
            event,
            agent_reply,
            reply_in_thread=_get_route_bool_option(route, "reply_in_thread", False),
        )

        delivery_receipt = self._send_reply_webhook(
            FEISHU_REPLY_MESSAGE_URL_TEMPLATE.format(message_id=event.message_id),
            reply_payload,
            {
                "Authorization": f"Bearer {tenant_access_token}",
            },
        )
        if delivery_receipt.status_code >= 400:
            raise WebhookDeliveryError(
                f"飞书消息回复接口返回 HTTP {delivery_receipt.status_code}",
                status_code=delivery_receipt.status_code,
                response_text=delivery_receipt.response_text,
            )

        reply_message_id = self._parse_feishu_message_delivery_receipt(
            delivery_receipt.response_text,
            delivery_receipt.status_code,
            error_prefix="飞书消息回复",
        )

        return build_json_http_response(
            {
                "status": "ok",
                "provider": event.provider,
                "session_id": agent_reply.session_id,
                "delivery": {
                    "method": "feishu_reply_api",
                    "target": event.message_id,
                    "status_code": delivery_receipt.status_code,
                    "message_id": reply_message_id,
                },
            }
        )


class WebhookGatewayHttpServer(ThreadingHTTPServer):
    """承载 Cyber Agent webhook 网关的简易 HTTP 服务器。"""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        gateway: WebhookGateway,
    ) -> None:
        self.gateway = gateway
        super().__init__(server_address, WebhookGatewayRequestHandler)


class WebhookGatewayRequestHandler(BaseHTTPRequestHandler):
    """将 HTTP 请求委托给 WebhookGateway 处理。"""

    server_version = "CyberAgentWebhook/0.1"

    @property
    def gateway(self) -> WebhookGateway:
        return self.server.gateway  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
        self._dispatch_request()

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
        self._dispatch_request()

    def log_message(self, format: str, *args: object) -> None:
        return None

    def _dispatch_request(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0") or 0)
        request_body = self.rfile.read(content_length) if content_length > 0 else b""
        headers = {
            key.lower(): value
            for key, value in self.headers.items()
        }
        response = self.gateway.handle_request(
            self.command,
            self.path,
            headers,
            request_body,
        )
        self.send_response(response.status_code)
        self.send_header("Content-Type", response.content_type)
        self.send_header("Content-Length", str(len(response.body)))
        self.end_headers()
        if response.body:
            self.wfile.write(response.body)


def create_webhook_http_server(
    host: str,
    port: int,
    gateway: WebhookGateway,
) -> WebhookGatewayHttpServer:
    """创建 webhook HTTP 服务实例，便于 CLI 与测试共用。"""
    return WebhookGatewayHttpServer((host, port), gateway)


def serve_webhook_gateway(
    host: str,
    port: int,
    routes: list[WebhookRouteConfig],
    runtime_context: dict[str, object],
    runner_factory: AgentRunnerFactory,
    *,
    cli_renderer: CliRenderer | None = None,
    base_dir: Path | None = None,
    reply_timeout_seconds: float = DEFAULT_WEBHOOK_REPLY_TIMEOUT_SECONDS,
    reply_sender: ReplySender | None = None,
) -> None:
    """启动 webhook HTTP 服务并阻塞当前进程。"""
    resolved_renderer = cli_renderer or CliRenderer()
    gateway = WebhookGateway(
        routes,
        runtime_context,
        runner_factory,
        cli_renderer=resolved_renderer,
        base_dir=base_dir,
        reply_timeout_seconds=reply_timeout_seconds,
        reply_sender=reply_sender,
    )
    server = create_webhook_http_server(host, port, gateway)
    actual_host, actual_port = server.server_address[:2]
    resolved_renderer.print_info(
        f"Webhook 服务已启动：{actual_host}:{actual_port}"
    )
    resolved_renderer.print_info(
        "已注册路由：\n" + "\n".join(f"- {item}" for item in gateway.describe_routes())
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        resolved_renderer.print_info("Webhook 服务已收到停止信号，正在关闭。")
    finally:
        server.server_close()
