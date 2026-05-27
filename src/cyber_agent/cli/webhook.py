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




from .webhook_models import *  # noqa: F403  # 向后兼容：模型已拆分至独立模块

from .webhook_feishu import (
    FeishuProgressMessageEmitter,
    FeishuTraceCollector,
    _build_feishu_ai_reply_payload,
    _build_feishu_allow_path_payload,
    _build_feishu_approval_payload,
    _build_feishu_builtin_command_payload,
    _build_feishu_chat_scope_id,
    _build_feishu_command_action_rows,
    _build_feishu_command_button,
    _build_feishu_config_payload,
    _build_feishu_context_payload,
    _build_feishu_doctor_payload,
    _build_feishu_fallback_builtin_payload,
    _build_feishu_help_payload,
    _build_feishu_history_export_payload,
    _build_feishu_history_list_payload,
    _build_feishu_history_load_payload,
    _build_feishu_history_search_payload,
    _build_feishu_history_show_payload,
    _build_feishu_interactive_card_elements_payload,
    _build_feishu_interactive_card_payload,
    _build_feishu_key_value_table,
    _build_feishu_markdown_section,
    _build_feishu_markdown_table,
    _build_feishu_mode_payload,
    _build_feishu_model_config_payload,
    _build_feishu_notice_payload,
    _build_feishu_progress_payload,
    _build_feishu_recent_session_button_specs,
    _build_feishu_session_entry,
    _build_feishu_session_list_payload,
    _build_feishu_session_state_path,
    _build_feishu_session_switch_button_specs,
    _build_feishu_start_menu_payload,
    _build_feishu_status_payload,
    _build_feishu_text_message_payload,
    _build_feishu_tool_result_key_value_detail,
    _build_feishu_tools_payload,
    _build_feishu_trace_elements,
    _build_hardcoded_feishu_model_config_payload,
    _escape_feishu_code_block,
    _extract_feishu_reply_title,
    _extract_feishu_tool_result_json_rows,
    _extract_feishu_tool_result_line_rows,
    _format_feishu_tool_result_value,
    _get_feishu_command_description,
    _is_feishu_tool_result_key,
    _load_feishu_session_state,
    _looks_like_builtin_error,
    _looks_like_feishu_error_reply,
    _normalize_ai_reply_markdown_for_feishu,
    _normalize_cli_output_for_feishu,
    _normalize_feishu_table_cell,
    _parse_feishu_tool_entries,
    _resolve_feishu_command_button_spec,
    _resolve_feishu_progress_template,
    _save_feishu_session_state,
    _should_use_feishu_rich_reply,
    _split_feishu_markdown_blocks,
    _split_large_feishu_block,
    _split_long_text_for_feishu,
    _trim_feishu_list_items,
    _trim_feishu_preview_lines,
    _truncate_feishu_button_label,
    _truncate_feishu_markdown,
)




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
    decrypted_bytes = aes_cbc_decrypt(hashed_key, iv, ciphertext)
    unpadded_bytes = pkcs7_unpad(decrypted_bytes, 16)
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


    decrypted_bytes = aes_cbc_decrypt(aes_key, aes_key[:16], encrypted_bytes)
    unpadded_bytes = pkcs7_unpad(decrypted_bytes, 32)
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
    padded_payload = pkcs7_pad(raw_payload, 32)
    encrypted_bytes = aes_cbc_encrypt(aes_key, aes_key[:16], padded_payload)
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


def _is_webhook_stop_command_event(event: WebhookEvent) -> bool:
    """判断事件是否为需要抢占处理的停止命令。"""
    return event.text.strip().lower() == "/stop"


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
        self.model_services = self._load_model_services(routes)

    @staticmethod
    def _load_model_services(routes: list[WebhookRouteConfig]) -> list[dict]:
        """从第一条 feishu 路由的 provider_options 中读取 model_services 配置。
           如果 provider_options 提供了 model_list_yaml 路径，则从 YAML 文件加载。
        """
        for route in routes:
            if route.provider == "feishu":
                # 优先从 YAML 文件动态加载
                yaml_path = route.provider_options.get("model_list_yaml")
                if yaml_path:
                    try:
                        import yaml
                    except ImportError:
                        raise ImportError(
                            "缺少 PyYAML 依赖，请执行 pip install pyyaml 后再使用 YAML 模型列表。"
                        )
                    with open(yaml_path, "r") as f:
                        raw = yaml.safe_load(f)
                    return WebhookGateway._parse_model_services_from_yaml(raw)

                # 否则尝试读取内嵌的 JSON 列表（字符串或直接是 list）
                raw_services = route.provider_options.get("model_services")
                if raw_services is not None:
                    if isinstance(raw_services, str):
                        try:
                            return json.loads(raw_services)
                        except json.JSONDecodeError:
                            pass  # 解析失败，回退到默认
                    elif isinstance(raw_services, list):
                        return raw_services  # 直接使用，要求每个元素符合约定的字典结构
        return []  # 无配置则回退默认硬编码

    @staticmethod
    def _parse_model_services_from_yaml(config: dict) -> list[dict]:
        """从 cli-proxy-api 风格的 config.yaml 中抽取 openai-compatibility 列表"""
        services = []
        openai_compat = config.get("openai-compatibility", [])
        for entry in openai_compat:
            svc_name = entry.get("name")
            if not svc_name:
                continue
            models = []
            for m in entry.get("models", []):
                m_name = m.get("name")
                if m_name:
                    models.append({"name": m_name, "command": f"/model {m_name}"})
            services.append({
                "name": svc_name,
                "command": f"/service {svc_name}",
                "models": models,
            })
        return services

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

        priority_response = self.handle_priority_event(route, outcome.event)
        if priority_response is not None:
            return priority_response

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
        priority_response = self.handle_priority_event(route, event)
        if priority_response is not None:
            return priority_response

        adapter = WEBHOOK_PROVIDER_ADAPTERS[route.provider]
        with self._processing_lock:
            agent_reply = self._run_agent_turn(route, event)
        return self._deliver_reply(route, adapter, event, agent_reply)

    def handle_priority_event(
        self,
        route: WebhookRouteConfig,
        event: WebhookEvent,
    ) -> WebhookHttpResponse | None:
        """处理必须抢占普通队列的控制命令。"""
        if _is_webhook_stop_command_event(event):
            return self._handle_stop_command_event(route, event)
        return None

    def _handle_stop_command_event(
        self,
        route: WebhookRouteConfig,
        event: WebhookEvent,
    ) -> WebhookHttpResponse:
        """立即处理 /stop，不等待当前长任务释放 webhook 队列。"""
        execution_controller = self.runtime_context.get("execution_controller")
        stop_message = "当前没有正在执行的任务。"
        accepted_stop = False

        is_running = False
        if execution_controller is not None:
            is_running_method = getattr(execution_controller, "is_running", None)
            if callable(is_running_method):
                is_running = bool(is_running_method())
            cancel_requested_method = getattr(
                execution_controller,
                "is_cancel_requested",
                None,
            )
            cancel_requested = (
                bool(cancel_requested_method())
                if callable(cancel_requested_method)
                else False
            )
            if is_running and cancel_requested:
                accepted_stop = True
                stop_message = "已请求停止当前任务，正在等待执行链路收尾。"
            elif is_running:
                request_stop_method = getattr(execution_controller, "request_stop", None)
                if callable(request_stop_method) and bool(
                    request_stop_method("用户通过 webhook /stop 请求停止当前任务")
                ):
                    accepted_stop = True
                    stop_message = "已收到 /stop，正在终止当前模型、Shell 与工具执行。"

        self.cli_renderer.print_info(stop_message)
        resolved_session_key = (
            self._resolve_feishu_active_session_key(event)
            if event.provider == "feishu"
            else event.session_key
        )
        session_id = build_webhook_session_id(event.provider, resolved_session_key)
        reply_payload_override = (
            _build_feishu_notice_payload(
                "停止任务",
                stop_message,
                template="red" if accepted_stop else "grey",
                button_commands=("/status", "/start"),
            )
            if event.provider == "feishu"
            else None
        )
        agent_reply = WebhookAgentReply(
            session_id=session_id,
            reply_text=stop_message,
            reply_payload_override=reply_payload_override,
        )
        return self._deliver_reply(
            route,
            WEBHOOK_PROVIDER_ADAPTERS[route.provider],
            event,
            agent_reply,
        )

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
            payload = _build_feishu_session_list_payload(session_items)
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

        if event.provider == "feishu" and normalized_text == "/status":
            return WebhookAgentReply(
                session_id=session_id,
                reply_text="已发送飞书会话状态卡片。",
                reply_payload_override=_build_feishu_status_payload(
                    runner,
                    self.runtime_context,
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
                model_services=self.model_services,
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
        except ExecutionInterruptedError as exc:
            reply_text = str(exc) or "当前任务已被 /stop 中断。"
            history_snapshot = [
                *runner.get_history_snapshot(),
                AIMessage(content=reply_text),
            ]
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
    from ..logging import log_info
    log_info("webhook", f"服务启动 {actual_host}:{actual_port}，{len(routes)} 条路由")
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


