from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING, Any

from .render import CliRenderer
from .webhook import (
    DEFAULT_WEBHOOK_REPLY_TIMEOUT_SECONDS,
    FEISHU_CREATE_API_MODE,
    WebhookEvent,
    WebhookGateway,
    WebhookHttpResponse,
    WebhookRouteConfig,
    parse_feishu_payload,
)

FEISHU_LONG_CONNECTION_DEDUP_WINDOW_SECONDS = 300.0

lark: Any | None = None
LARK_OAPI_IMPORT_ERROR: ModuleNotFoundError | None = None
ObservableLarkClient: Any | None = None

if TYPE_CHECKING:
    from ..agent.runner import AgentRunner


def _normalize_preview_text(text: str, max_length: int = 48) -> str:
    normalized_text = " ".join(text.split())
    if len(normalized_text) <= max_length:
        return normalized_text
    return normalized_text[: max_length - 3] + "..."


def select_feishu_long_connection_route(
    routes: list[WebhookRouteConfig],
    route_path: str | None = None,
) -> WebhookRouteConfig:
    candidate_routes = [route for route in routes if route.provider == "feishu"]
    if route_path is not None:
        normalized_path = route_path.rstrip("/") or "/"
        candidate_routes = [
            route for route in candidate_routes if route.path == normalized_path
        ]

    if not candidate_routes:
        if route_path is None:
            raise ValueError("配置中未找到可用于飞书长连接的 feishu 路由。")
        raise ValueError(f"配置中未找到路径为 {route_path} 的 feishu 路由。")
    if len(candidate_routes) > 1:
        route_descriptions = ", ".join(route.path for route in candidate_routes)
        raise ValueError(
            "检测到多个 feishu 路由，请通过 --path 指定其中一条："
            f"{route_descriptions}"
        )
    return candidate_routes[0]


def _ensure_lark_oapi_available() -> Any:
    """按需导入飞书 SDK，避免普通测试和 CLI 启动被 SDK 导入拖慢。"""
    global lark, LARK_OAPI_IMPORT_ERROR

    if lark is not None:
        return lark
    if LARK_OAPI_IMPORT_ERROR is not None:
        raise LARK_OAPI_IMPORT_ERROR

    try:
        import lark_oapi as loaded_lark
    except ModuleNotFoundError as exc:  # pragma: no cover - 通过 CLI 测试覆盖缺依赖分支
        LARK_OAPI_IMPORT_ERROR = exc
        raise ModuleNotFoundError(
            "缺少 `lark-oapi` 依赖，请先执行 `pip install -r requirements.txt`。"
        ) from exc

    lark = loaded_lark
    LARK_OAPI_IMPORT_ERROR = None
    return lark


def _serialize_lark_event(event: object) -> dict[str, object]:
    lark_module = _ensure_lark_oapi_available()
    try:
        payload = json.loads(lark_module.JSON.marshal(event))
    except json.JSONDecodeError as exc:
        raise ValueError("飞书长连接事件序列化后不是合法 JSON。") from exc
    if not isinstance(payload, dict):
        raise ValueError("飞书长连接事件序列化结果必须是 JSON 对象。")
    return payload


def _build_feishu_card_command_event(payload: Mapping[str, object]) -> WebhookEvent:
    """将飞书卡片按钮回调转换为统一的命令事件，复用现有 webhook 命令处理链路。"""
    event_payload = payload.get("event")
    if not isinstance(event_payload, Mapping):
        raise ValueError("飞书卡片回调缺少 event 对象。")

    action_payload = event_payload.get("action")
    if not isinstance(action_payload, Mapping):
        raise ValueError("飞书卡片回调缺少 action 对象。")
    action_value = action_payload.get("value")
    if not isinstance(action_value, Mapping):
        raise ValueError("飞书卡片按钮缺少 value 配置。")

    command_text = str(action_value.get("command", "")).strip()
    if not command_text.startswith("/"):
        raise ValueError("飞书卡片按钮缺少合法的命令文本。")

    context_payload = event_payload.get("context")
    if not isinstance(context_payload, Mapping):
        raise ValueError("飞书卡片回调缺少 context 对象。")
    chat_id = str(context_payload.get("open_chat_id", "")).strip()
    open_message_id = str(context_payload.get("open_message_id", "")).strip()
    if not chat_id:
        raise ValueError("飞书卡片回调缺少 open_chat_id，无法定位目标会话。")

    operator_payload = event_payload.get("operator")
    operator_mapping = operator_payload if isinstance(operator_payload, Mapping) else {}
    header_payload = payload.get("header")
    header_mapping = header_payload if isinstance(header_payload, Mapping) else {}
    sender_id = (
        str(operator_mapping.get("open_id", "")).strip()
        or str(operator_mapping.get("user_id", "")).strip()
        or str(operator_mapping.get("union_id", "")).strip()
        or chat_id
    )
    message_digest = hashlib.sha1(
        f"{open_message_id}:{command_text}:{time.time_ns()}".encode("utf-8")
    ).hexdigest()[:16]
    return WebhookEvent(
        provider="feishu",
        session_key=chat_id,
        sender_id=sender_id,
        sender_name=sender_id or "unknown",
        message_id=f"{open_message_id or 'card-action'}:{message_digest}",
        text=command_text,
        metadata={
            "chat_id": chat_id,
            "message_type": "card_action",
            "event_type": str(header_mapping.get("event_type", "")).strip()
            or "card.action.trigger",
            "feishu_delivery_mode": FEISHU_CREATE_API_MODE,
            "source_message_id": open_message_id,
        },
    )


def _extract_response_reason(response: WebhookHttpResponse) -> str:
    decoded_body = response.body.decode("utf-8", errors="replace").strip()
    try:
        payload = json.loads(decoded_body or "{}")
    except json.JSONDecodeError:
        return decoded_body
    if not isinstance(payload, dict):
        return decoded_body
    reason = str(payload.get("reason", "")).strip()
    if reason:
        return reason
    status = str(payload.get("status", "")).strip()
    if status:
        return status
    return decoded_body


def _extract_response_payload(response: WebhookHttpResponse) -> dict[str, object]:
    decoded_body = response.body.decode("utf-8", errors="replace").strip()
    try:
        payload = json.loads(decoded_body or "{}")
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _build_delivery_method_hint(payload: Mapping[str, object]) -> str:
    delivery_payload = payload.get("delivery")
    if isinstance(delivery_payload, Mapping):
        delivery_method = str(delivery_payload.get("method", "")).strip()
        if delivery_method:
            return delivery_method
    reply_payload = payload.get("reply_payload")
    if isinstance(reply_payload, Mapping):
        return "response_payload"
    return "unknown"


def _report_gateway_response(
    response: WebhookHttpResponse,
    cli_renderer: CliRenderer,
    *,
    event: WebhookEvent | None = None,
) -> None:
    if response.status_code >= 400:
        cli_renderer.print_error(
            f"飞书长连接消息处理失败：{_extract_response_reason(response)}"
        )
        return

    if event is None:
        return

    response_payload = _extract_response_payload(response)
    session_id = str(response_payload.get("session_id", "")).strip() or "unknown"
    delivery_method = _build_delivery_method_hint(response_payload)
    cli_renderer.print_info(
        "飞书消息已处理并完成回复："
        f"message_id={event.message_id} "
        f"session_id={session_id} "
        f"delivery={delivery_method}"
    )


class FeishuLongConnectionDispatcher:
    """将飞书长连接回调快速确认后，转到后台线程串行处理。"""

    def __init__(
        self,
        route: WebhookRouteConfig,
        gateway: WebhookGateway,
        cli_renderer: CliRenderer,
        *,
        dedup_window_seconds: float = FEISHU_LONG_CONNECTION_DEDUP_WINDOW_SECONDS,
    ) -> None:
        self.route = route
        self.gateway = gateway
        self.cli_renderer = cli_renderer
        self.dedup_window_seconds = max(1.0, dedup_window_seconds)
        self._queue: Queue[WebhookEvent] = Queue()
        self._message_seen_at: dict[str, float] = {}
        self._dedup_lock = threading.Lock()
        self._worker_thread: threading.Thread | None = None

    def start(self) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="feishu-long-connection-worker",
            daemon=True,
        )
        self._worker_thread.start()

    def submit_payload(self, payload: Mapping[str, object]) -> None:
        outcome = parse_feishu_payload(payload, self.route, validate_token=False)
        if outcome.immediate_response is not None:
            _report_gateway_response(outcome.immediate_response, self.cli_renderer)
            return
        if outcome.event is None:
            self.cli_renderer.print_info("飞书长连接收到未提取出可处理事件的消息，已忽略。")
            return

        if not self._mark_message_seen(outcome.event.message_id):
            self.cli_renderer.print_info(
                f"检测到飞书重复事件，已忽略：message_id={outcome.event.message_id}"
            )
            return

        priority_response = self._handle_priority_event(outcome.event)
        if priority_response is not None:
            self.cli_renderer.print_info(
                "收到飞书控制命令："
                f"chat_id={outcome.event.metadata.get('chat_id', '') or 'unknown'} "
                f"sender_id={outcome.event.sender_id} "
                f"message_id={outcome.event.message_id} "
                f"text={_normalize_preview_text(outcome.event.text)} "
                "status=handled"
            )
            _report_gateway_response(
                priority_response,
                self.cli_renderer,
                event=outcome.event,
            )
            return

        self.cli_renderer.print_info(
            "收到飞书文本消息："
            f"chat_id={outcome.event.metadata.get('chat_id', '') or 'unknown'} "
            f"sender_id={outcome.event.sender_id} "
            f"message_id={outcome.event.message_id} "
            f"text={_normalize_preview_text(outcome.event.text)} "
            "status=queued"
        )
        self.submit_event(outcome.event)

    def submit_event(self, event: WebhookEvent) -> None:
        """直接提交已归一化事件，供卡片动作等非文本入口复用后台处理链路。"""
        priority_response = self._handle_priority_event(event)
        if priority_response is not None:
            _report_gateway_response(
                priority_response,
                self.cli_renderer,
                event=event,
            )
            return
        self._queue.put(event)

    def _handle_priority_event(self, event: WebhookEvent) -> WebhookHttpResponse | None:
        """优先处理 /stop 等控制事件，避免排在长任务后面失效。"""
        priority_handler = getattr(self.gateway, "handle_priority_event", None)
        if not callable(priority_handler):
            return None
        return priority_handler(self.route, event)

    def wait_until_idle(self, timeout_seconds: float = 5.0) -> bool:
        deadline = time.time() + max(timeout_seconds, 0.0)
        while time.time() <= deadline:
            if self._queue.unfinished_tasks == 0:
                return True
            time.sleep(0.01)
        return self._queue.unfinished_tasks == 0

    def _mark_message_seen(self, message_id: str) -> bool:
        current_time = time.time()
        expire_before = current_time - self.dedup_window_seconds
        with self._dedup_lock:
            expired_message_ids = [
                cached_message_id
                for cached_message_id, seen_at in self._message_seen_at.items()
                if seen_at < expire_before
            ]
            for cached_message_id in expired_message_ids:
                self._message_seen_at.pop(cached_message_id, None)

            if message_id in self._message_seen_at:
                return False
            self._message_seen_at[message_id] = current_time
            return True

    def _worker_loop(self) -> None:
        while True:
            event = self._queue.get()
            try:
                response = self.gateway.handle_event(self.route, event)
                _report_gateway_response(
                    response,
                    self.cli_renderer,
                    event=event,
                )
            except Exception as exc:  # noqa: BLE001 - 后台处理需保留真实错误便于排障
                self.cli_renderer.print_error(f"飞书长连接后台处理失败：{exc}")
            finally:
                self._queue.task_done()


def _build_observable_lark_client_class(lark_module: Any) -> Any:
    """按需创建可观察连接状态的飞书客户端子类。"""

    class _ObservableLarkClient(lark_module.ws.Client):
        def __init__(
            self,
            *args,
            cli_renderer: CliRenderer,
            route: WebhookRouteConfig,
            **kwargs,
        ) -> None:
            super().__init__(*args, **kwargs)
            self._cli_renderer = cli_renderer
            self._route = route
            self._has_connected_once = False

        async def _connect(self) -> None:
            was_connected = self._has_connected_once
            await super()._connect()
            if self._conn is None:
                return

            if was_connected:
                self._cli_renderer.print_info(
                    "飞书长连接已重新连接成功，正在继续监听消息。"
                )
            else:
                self._cli_renderer.print_info(
                    "飞书长连接已连接成功，当前正在等待消息。"
                    f"路由={self._route.path}，向机器人发送文本即可触发处理，按 Ctrl+C 可退出。"
                )
                self._has_connected_once = True

    return _ObservableLarkClient


def serve_feishu_long_connection(
    route: WebhookRouteConfig,
    runtime_context: dict[str, object],
    runner_factory: Callable[[dict[str, object]], "AgentRunner"],
    *,
    cli_renderer: CliRenderer | None = None,
    base_dir: Path | None = None,
    reply_timeout_seconds: float = DEFAULT_WEBHOOK_REPLY_TIMEOUT_SECONDS,
) -> None:
    lark_module = _ensure_lark_oapi_available()
    if route.provider != "feishu":
        raise ValueError("飞书长连接仅支持 provider=feishu 的路由。")

    app_id = route.provider_options.get("app_id", "").strip()
    app_secret = route.provider_options.get("app_secret", "").strip()
    if not app_id or not app_secret:
        raise ValueError(
            "飞书长连接缺少 provider_options.app_id 或 provider_options.app_secret。"
        )

    cli_renderer = cli_renderer or CliRenderer()
    gateway = WebhookGateway(
        [route],
        runtime_context,
        runner_factory,
        cli_renderer=cli_renderer,
        base_dir=base_dir,
        reply_timeout_seconds=reply_timeout_seconds,
    )
    dispatcher = FeishuLongConnectionDispatcher(route, gateway, cli_renderer)
    dispatcher.start()

    def _on_message_receive(data: object) -> None:
        try:
            payload = _serialize_lark_event(data)
            dispatcher.submit_payload(payload)
        except Exception as exc:  # noqa: BLE001 - 回调内需要记录真实错误便于联调排查
            cli_renderer.print_error(f"飞书长连接事件处理失败：{exc}")

    def _on_card_action(data: object) -> dict[str, object]:
        try:
            payload = _serialize_lark_event(data)
            command_event = _build_feishu_card_command_event(payload)
            cli_renderer.print_info(
                "收到飞书卡片命令："
                f"chat_id={command_event.metadata.get('chat_id', '') or 'unknown'} "
                f"sender_id={command_event.sender_id} "
                f"command={command_event.text}"
            )
            dispatcher.submit_event(command_event)
            return {
                "toast": {
                    "type": "info",
                    "content": f"已执行 {command_event.text}",
                }
            }
        except Exception as exc:  # noqa: BLE001 - 需要把错误展示在终端并反馈给飞书客户端
            cli_renderer.print_error(f"飞书卡片命令处理失败：{exc}")
            return {
                "toast": {
                    "type": "error",
                    "content": f"处理失败：{exc}",
                }
            }

    def _on_message_read(data: object) -> None:
        """飞书已读回执不需要进入 Agent，只注册处理器以避免 SDK 打印误导性错误。"""
        _ = data

    def _on_bot_p2p_chat_entered(data: object) -> None:
        """机器人被用户打开单聊窗口时不触发 Agent，只显式吞掉系统事件。"""
        _ = data

    event_handler = (
        # 长连接模式由飞书官方链路保证事件来源，不复用 webhook 的 token / encrypt 校验。
        lark_module.EventDispatcherHandler.builder(
            "",
            "",
        )
        .register_p2_im_message_receive_v1(_on_message_receive)
        .register_p2_im_message_message_read_v1(_on_message_read)
        .register_p2_im_chat_access_event_bot_p2p_chat_entered_v1(_on_bot_p2p_chat_entered)
        .register_p2_card_action_trigger(_on_card_action)
        .build()
    )
    cli_renderer.print_info(
        "正在启动飞书长连接客户端，将复用现有 Agent 会话和飞书官方回复接口。"
    )
    client_class = ObservableLarkClient or _build_observable_lark_client_class(lark_module)
    client = client_class(
        app_id,
        app_secret,
        event_handler=event_handler,
        log_level=lark_module.LogLevel.INFO,
        cli_renderer=cli_renderer,
        route=route,
    )
    try:
        client.start()
    except KeyboardInterrupt:
        cli_renderer.print_info("已停止飞书长连接监听。")
