from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlsplit

from .webhook_models import (
    FEISHU_CREATE_API_MODE,
    FEISHU_REPLY_API_MODE,
    SUPPORTED_WEBHOOK_PROVIDERS,
    WebhookRouteConfig,
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
    """描述路由回复投递方式，便于启动日志快速排查配置。"""
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
