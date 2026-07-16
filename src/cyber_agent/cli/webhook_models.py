"""Webhook 数据模型、常量和异常类。从 webhook.py 拆分以便维护。"""
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
from ..execution_control import ExecutionInterruptedError
from ..session_store import (
    create_session_id,
    get_session_storage_dir,
    list_stored_sessions,
    load_session_history,
    save_session_history,
    search_stored_sessions,
)
from .webhook_crypto import aes_cbc_decrypt, aes_cbc_encrypt, pkcs7_pad, pkcs7_unpad
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
FEISHU_SESSION_SWITCH_BUTTON_LIMIT = 12
FEISHU_RICH_REPLY_CHUNK_MAX_CHARS = 2200
FEISHU_RICH_REPLY_MAX_CHUNKS = 6
FEISHU_TRACE_MAX_STEPS = 10
FEISHU_TRACE_DETAIL_MAX_CHARS = 500
FEISHU_TOOL_RESULT_KEY_VALUE_MAX_ROWS = 16
FEISHU_TOOL_RESULT_KEY_MAX_CHARS = 48
FEISHU_TOOL_RESULT_VALUE_MAX_CHARS = 260
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

