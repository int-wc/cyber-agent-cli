import unittest
from unittest.mock import patch

from cyber_agent.cli.feishu_long_connection import (
    FeishuLongConnectionDispatcher,
    _build_feishu_card_command_event,
    select_feishu_long_connection_route,
    serve_feishu_long_connection,
)
from cyber_agent.cli.webhook import (
    FEISHU_CREATE_API_MODE,
    WebhookRouteConfig,
    WebhookEvent,
    build_json_http_response,
)


class _FakeGateway:
    def __init__(self) -> None:
        self.events = []
        self.priority_events = []

    def handle_event(self, route, event):
        self.events.append((route, event))
        return build_json_http_response(
            {
                "status": "ok",
                "session_id": "webhook:feishu:oc_test_chat",
                "delivery": {"method": "feishu_reply_api"},
            }
        )

    def handle_priority_event(self, route, event):
        if event.text.strip().lower() != "/stop":
            return None
        self.priority_events.append((route, event))
        return build_json_http_response(
            {
                "status": "ok",
                "session_id": "webhook:feishu:oc_test_chat",
                "delivery": {"method": "feishu_reply_api"},
            }
        )


class _RecordingRenderer:
    def __init__(self) -> None:
        self.infos: list[str] = []
        self.errors: list[str] = []

    def print_info(self, content: str) -> None:
        self.infos.append(content)

    def print_error(self, content: str) -> None:
        self.errors.append(content)


def _build_message_payload() -> dict[str, object]:
    return {
        "schema": "2.0",
        "header": {
            "event_type": "im.message.receive_v1",
        },
        "event": {
            "sender": {
                "sender_id": {
                    "open_id": "ou_test_user",
                }
            },
            "message": {
                "chat_id": "oc_test_chat",
                "message_id": "om_test_message",
                "message_type": "text",
                "content": "{\"text\":\"你好，飞书长连接\"}",
            },
        },
    }


class FeishuLongConnectionTestCase(unittest.TestCase):
    def test_select_feishu_long_connection_route_requires_explicit_path(self) -> None:
        """测试：配置里存在多条 feishu 路由时，必须显式指定路径。"""
        routes = [
            WebhookRouteConfig(provider="feishu", path="/webhook/feishu/a"),
            WebhookRouteConfig(provider="feishu", path="/webhook/feishu/b"),
        ]

        with self.assertRaisesRegex(ValueError, "--path"):
            select_feishu_long_connection_route(routes)

    def test_dispatcher_can_process_message_in_background(self) -> None:
        """测试：长连接消息会快速入队，并由后台线程复用现有网关处理。"""
        route = WebhookRouteConfig(
            provider="feishu",
            path="/webhook/feishu",
            provider_options={"verification_token": "feishu-token"},
        )
        gateway = _FakeGateway()
        renderer = _RecordingRenderer()
        dispatcher = FeishuLongConnectionDispatcher(route, gateway, renderer)
        dispatcher.start()

        dispatcher.submit_payload(_build_message_payload())

        self.assertTrue(dispatcher.wait_until_idle())
        self.assertEqual(len(gateway.events), 1)
        _, event = gateway.events[0]
        self.assertEqual(event.provider, "feishu")
        self.assertEqual(event.text, "你好，飞书长连接")
        self.assertEqual(renderer.errors, [])
        self.assertEqual(len(renderer.infos), 2)
        self.assertIn("收到飞书文本消息", renderer.infos[0])
        self.assertIn("status=queued", renderer.infos[0])
        self.assertIn("飞书消息已处理并完成回复", renderer.infos[1])
        self.assertIn("feishu_reply_api", renderer.infos[1])

    def test_dispatcher_can_ignore_duplicate_message_id(self) -> None:
        """测试：同一条飞书消息重复投递时，只会进入后台处理一次。"""
        route = WebhookRouteConfig(
            provider="feishu",
            path="/webhook/feishu",
        )
        gateway = _FakeGateway()
        renderer = _RecordingRenderer()
        dispatcher = FeishuLongConnectionDispatcher(route, gateway, renderer)
        dispatcher.start()

        payload = _build_message_payload()
        dispatcher.submit_payload(payload)
        dispatcher.submit_payload(payload)

        self.assertTrue(dispatcher.wait_until_idle())
        self.assertEqual(len(gateway.events), 1)
        self.assertEqual(renderer.errors, [])
        self.assertTrue(
            any("重复事件" in info for info in renderer.infos),
            msg=f"未找到重复事件提示：{renderer.infos}",
        )

    def test_build_feishu_card_command_event_can_convert_button_action(self) -> None:
        """测试：飞书卡片按钮回调可转换为统一命令事件，并改用会话级发消息模式。"""
        event = _build_feishu_card_command_event(
            {
                "schema": "2.0",
                "header": {
                    "event_type": "card.action.trigger",
                },
                "event": {
                    "operator": {
                        "open_id": "ou_test_user",
                    },
                    "action": {
                        "value": {
                            "command": "/help",
                        }
                    },
                    "context": {
                        "open_chat_id": "oc_test_chat",
                        "open_message_id": "om_card_message",
                    },
                },
            }
        )

        self.assertIsInstance(event, WebhookEvent)
        self.assertEqual(event.provider, "feishu")
        self.assertEqual(event.session_key, "oc_test_chat")
        self.assertEqual(event.sender_id, "ou_test_user")
        self.assertEqual(event.text, "/help")
        self.assertEqual(event.metadata["chat_id"], "oc_test_chat")
        self.assertEqual(event.metadata["feishu_delivery_mode"], FEISHU_CREATE_API_MODE)
        self.assertIn("om_card_message", event.message_id)

    def test_dispatcher_can_accept_direct_command_event(self) -> None:
        """测试：长连接调度器支持直接提交命令事件，供卡片按钮回调用。"""
        route = WebhookRouteConfig(
            provider="feishu",
            path="/webhook/feishu",
        )
        gateway = _FakeGateway()
        renderer = _RecordingRenderer()
        dispatcher = FeishuLongConnectionDispatcher(route, gateway, renderer)
        dispatcher.start()

        dispatcher.submit_event(
            WebhookEvent(
                provider="feishu",
                session_key="oc_test_chat",
                sender_id="ou_test_user",
                sender_name="ou_test_user",
                message_id="card-action-001",
                text="/help",
                metadata={
                    "chat_id": "oc_test_chat",
                    "feishu_delivery_mode": FEISHU_CREATE_API_MODE,
                },
            )
        )

        self.assertTrue(dispatcher.wait_until_idle())
        self.assertEqual(len(gateway.events), 1)
        _, event = gateway.events[0]
        self.assertEqual(event.text, "/help")
        self.assertEqual(event.metadata["feishu_delivery_mode"], FEISHU_CREATE_API_MODE)

    def test_dispatcher_handles_stop_before_queueing(self) -> None:
        """测试：长连接 /stop 会优先处理，不会排在普通后台队列后面。"""
        route = WebhookRouteConfig(
            provider="feishu",
            path="/webhook/feishu",
        )
        gateway = _FakeGateway()
        renderer = _RecordingRenderer()
        dispatcher = FeishuLongConnectionDispatcher(route, gateway, renderer)
        dispatcher.start()

        dispatcher.submit_event(
            WebhookEvent(
                provider="feishu",
                session_key="oc_test_chat",
                sender_id="ou_test_user",
                sender_name="ou_test_user",
                message_id="stop-001",
                text="/stop",
                metadata={
                    "chat_id": "oc_test_chat",
                    "feishu_delivery_mode": FEISHU_CREATE_API_MODE,
                },
            )
        )

        self.assertTrue(dispatcher.wait_until_idle())
        self.assertEqual(gateway.events, [])
        self.assertEqual(len(gateway.priority_events), 1)
        _, event = gateway.priority_events[0]
        self.assertEqual(event.text, "/stop")
        self.assertTrue(
            any("飞书消息已处理并完成回复" in info for info in renderer.infos),
            msg=f"未找到处理完成提示：{renderer.infos}",
        )

    def test_serve_registers_bot_p2p_chat_entered_handler(self) -> None:
        """测试：注册机器人进入单聊事件处理器，避免 SDK 报 processor not found。"""
        registered_handlers: list[str] = []

        class FakeBuilder:
            def register_p2_im_message_receive_v1(self, handler):
                registered_handlers.append("im.message.receive_v1")
                return self

            def register_p2_im_message_message_read_v1(self, handler):
                registered_handlers.append("im.message.message_read_v1")
                return self

            def register_p2_im_chat_access_event_bot_p2p_chat_entered_v1(self, handler):
                registered_handlers.append("im.chat.access_event.bot_p2p_chat_entered_v1")
                return self

            def register_p2_card_action_trigger(self, handler):
                registered_handlers.append("card.action.trigger")
                return self

            def build(self):
                return object()

        class FakeEventDispatcherHandler:
            @staticmethod
            def builder(encrypt_key: str, verification_token: str):
                return FakeBuilder()

        class FakeLogLevel:
            INFO = "info"

        class FakeLark:
            EventDispatcherHandler = FakeEventDispatcherHandler
            LogLevel = FakeLogLevel

        class FakeClient:
            def __init__(self, *args, **kwargs) -> None:
                self.args = args
                self.kwargs = kwargs

            def start(self) -> None:
                return None

        route = WebhookRouteConfig(
            provider="feishu",
            path="/webhook/feishu",
            provider_options={
                "app_id": "cli_test_app",
                "app_secret": "test-secret",
            },
        )
        renderer = _RecordingRenderer()

        with (
            patch("cyber_agent.cli.feishu_long_connection.lark", FakeLark),
            patch(
                "cyber_agent.cli.feishu_long_connection.ObservableLarkClient",
                FakeClient,
            ),
        ):
            serve_feishu_long_connection(
                route,
                {},
                lambda runtime_context: None,
                cli_renderer=renderer,
            )

        self.assertIn(
            "im.chat.access_event.bot_p2p_chat_entered_v1",
            registered_handlers,
        )


if __name__ == "__main__":
    unittest.main()
