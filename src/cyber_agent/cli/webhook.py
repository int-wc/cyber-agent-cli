from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from hashlib import sha1
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request, urlopen

from langchain_core.messages import AIMessage

from ..agent.approval import ApprovalDecision, ApprovalPolicy
from ..session_store import load_session_history, save_session_history
from .render import CliRenderer

if TYPE_CHECKING:
    from ..agent.runner import AgentRunner

SUPPORTED_WEBHOOK_PROVIDERS = ("feishu", "dingtalk", "wecom", "email")
DEFAULT_WEBHOOK_HOST = "0.0.0.0"
DEFAULT_WEBHOOK_PORT = 8787
DEFAULT_WEBHOOK_REPLY_TIMEOUT_SECONDS = 10.0
WEBHOOK_SESSION_ID_MAX_SLUG_LENGTH = 48
WEBHOOK_SECRET_HEADER = "x-cyber-agent-webhook-secret"
WEBHOOK_SECRET_QUERY_KEY = "secret"
WEBHOOK_CONTENT_TYPE_JSON = "application/json; charset=utf-8"
WEBHOOK_CONTENT_TYPE_TEXT = "text/plain; charset=utf-8"

AgentRunnerFactory = Callable[[dict[str, object]], "AgentRunner"]
ReplySender = Callable[[str, dict[str, object], float], "WebhookDeliveryReceipt"]


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
    """生成适合 `webhook example-config` 输出的示例配置。"""
    return {
        "routes": [
            {
                "provider": "feishu",
                "path": "/webhook/feishu",
                "secret": "replace-with-shared-secret",
                "reply_webhook_url": "https://open.feishu.cn/open-apis/bot/v2/hook/your-feishu-webhook",
            },
            {
                "provider": "dingtalk",
                "path": "/webhook/dingtalk",
                "secret": "replace-with-shared-secret",
                "reply_webhook_url": "",
            },
            {
                "provider": "wecom",
                "path": "/webhook/wecom",
                "secret": "replace-with-shared-secret",
                "reply_webhook_url": "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=your-wecom-webhook-key",
            },
            {
                "provider": "email",
                "path": "/webhook/email",
                "secret": "replace-with-shared-secret",
                "reply_webhook_url": "https://your-mail-bridge.example.com/outbound/reply",
            },
        ]
    }


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

    raw_routes = raw_data.get("routes")
    if not isinstance(raw_routes, list) or not raw_routes:
        raise ValueError("webhook 配置文件中的 routes 必须是非空数组。")

    routes: list[WebhookRouteConfig] = []
    seen_paths: set[str] = set()
    for index, raw_route in enumerate(raw_routes, start=1):
        if not isinstance(raw_route, dict):
            raise ValueError(f"第 {index} 条 webhook 路由必须是对象。")

        normalized_provider = normalize_webhook_provider(str(raw_route.get("provider", "")))
        normalized_path = normalize_webhook_path(str(raw_route.get("path", "")))
        if normalized_path in seen_paths:
            raise ValueError(f"发现重复的 webhook 路由路径：{normalized_path}")
        seen_paths.add(normalized_path)

        raw_provider_options = raw_route.get("provider_options", {})
        provider_options: dict[str, str] = {}
        if isinstance(raw_provider_options, dict):
            provider_options = {
                str(key): str(value)
                for key, value in raw_provider_options.items()
                if str(value).strip()
            }

        reply_webhook_url = str(raw_route.get("reply_webhook_url", "")).strip() or None
        secret = str(raw_route.get("secret", "")).strip() or None
        routes.append(
            WebhookRouteConfig(
                provider=normalized_provider,
                path=normalized_path,
                reply_webhook_url=reply_webhook_url,
                secret=secret,
                provider_options=provider_options,
            )
        )

    return routes


def describe_webhook_routes(routes: list[WebhookRouteConfig]) -> list[str]:
    """返回适合启动日志输出的 webhook 路由摘要。"""
    descriptions: list[str] = []
    for route in routes:
        delivery_hint = route.reply_webhook_url or "按请求内 reply_webhook_url 或 HTTP 响应回包"
        descriptions.append(
            f"{route.path} -> {route.provider} | 回复: {delivery_hint}"
        )
    return descriptions


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
) -> WebhookDeliveryReceipt:
    """以 JSON POST 的方式向第三方 reply webhook 发送回复。"""
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": WEBHOOK_CONTENT_TYPE_JSON},
        method="POST",
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        response_body = response.read().decode("utf-8", errors="replace")
        return WebhookDeliveryReceipt(
            status_code=response.status,
            response_text=response_body,
        )


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


def parse_feishu_request(
    method: str,
    headers: Mapping[str, str],
    query: Mapping[str, list[str]],
    body: bytes,
    route: WebhookRouteConfig,
) -> WebhookRequestOutcome:
    payload = _parse_json_payload(body)
    challenge = str(payload.get("challenge", "")).strip()
    if challenge and str(payload.get("type", "")).strip() == "url_verification":
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

    session_key = str(message.get("chat_id") or sender_id or message.get("message_id") or "feishu-session")
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
                "schema": str(payload.get("schema", "")),
            },
        )
    )


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
    if method.upper() == "GET":
        echo_string = query.get("echostr", [""])[0].strip()
        if echo_string:
            return WebhookRequestOutcome(
                immediate_response=build_text_http_response(echo_string)
            )

    content_type = headers.get("content-type", "").lower()
    request_text = body.decode("utf-8", errors="replace").strip()
    if request_text.startswith("<xml") or "xml" in content_type:
        # TODO(联调补全): 企微官方 XML 加密回调需补充验签与解密流程。
        return _build_ignored_outcome(
            "wecom",
            "当前版本仅支持经 webhook 网关解密后的 JSON 企微回调。",
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
        self._routes_by_path = {route.path: route for route in routes}

    def describe_routes(self) -> list[str]:
        """返回当前网关已注册路由的摘要。"""
        return describe_webhook_routes(self.routes)

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

        with self._processing_lock:
            agent_reply = self._run_agent_turn(outcome.event)
        return self._deliver_reply(adapter, outcome.event, agent_reply)

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
        return build_json_http_response(
            {
                "status": "unauthorized",
                "provider": route.provider,
                "reason": "webhook 共享密钥校验失败。",
            },
            status_code=401,
        )

    def _run_agent_turn(self, event: WebhookEvent) -> WebhookAgentReply:
        runner = self.runner_factory(self.runtime_context)
        session_id = build_webhook_session_id(event.provider, event.session_key)
        try:
            stored_session = load_session_history(session_id, base_dir=self.base_dir)
        except ValueError:
            stored_session = None
        else:
            runner.restore_history(stored_session.messages)

        approval_policy = self.runtime_context.get("approval_policy", ApprovalPolicy.NEVER)
        if not isinstance(approval_policy, ApprovalPolicy):
            approval_policy = ApprovalPolicy.NEVER

        history_snapshot = runner.get_history_snapshot()
        reply_text = ""
        try:
            reply_text = runner.run(
                event.text,
                verbose=False,
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

        save_session_history(
            session_id,
            history_snapshot,
            mode=runner.mode.value,
            approval_policy=approval_policy.value,
            source_session_id=f"{event.provider}:{event.sender_id}",
            base_dir=self.base_dir,
        )
        return WebhookAgentReply(
            session_id=session_id,
            reply_text=reply_text.strip() or "（空回复）",
        )

    def _deliver_reply(
        self,
        adapter: WebhookProviderAdapter,
        event: WebhookEvent,
        agent_reply: WebhookAgentReply,
    ) -> WebhookHttpResponse:
        reply_payload = adapter.build_reply_payload(agent_reply.reply_text, event)
        if event.reply_webhook_url:
            try:
                delivery_receipt = self.reply_sender(
                    event.reply_webhook_url,
                    reply_payload,
                    self.reply_timeout_seconds,
                )
            except Exception as exc:  # noqa: BLE001 - 需要让桥接层看见真实投递失败原因
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

            return build_json_http_response(
                {
                    "status": "ok",
                    "provider": event.provider,
                    "session_id": agent_reply.session_id,
                    "delivery": {
                        "method": "reply_webhook",
                        "target": event.reply_webhook_url,
                        "status_code": delivery_receipt.status_code,
                        "response_text": delivery_receipt.response_text,
                    },
                }
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
