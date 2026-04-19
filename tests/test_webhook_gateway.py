import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlencode

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from cyber_agent.agent.approval import ApprovalPolicy
from cyber_agent.agent.mode import AgentMode
from cyber_agent.cli.webhook import (
    WebhookDeliveryReceipt,
    WebhookGateway,
    WebhookRouteConfig,
)


class FakeWebhookRunner:
    """用于 webhook 网关测试的简化运行器。"""

    def __init__(self) -> None:
        self.mode = AgentMode.STANDARD
        self.history = [SystemMessage(content="webhook system prompt")]

    def restore_history(self, messages) -> None:
        self.history = list(messages)

    def get_history_snapshot(self):
        return list(self.history)

    def get_turn_count(self) -> int:
        return sum(isinstance(message, HumanMessage) for message in self.history)

    def run(self, user_input: str, verbose: bool = False, approval_handler=None) -> str:
        self.history.append(HumanMessage(content=user_input))
        current_turn = self.get_turn_count()
        reply_text = f"第{current_turn}轮回复: {user_input}"
        self.history.append(AIMessage(content=reply_text))
        return reply_text


class WebhookGatewayTestCase(unittest.TestCase):
    def test_feishu_url_verification_returns_challenge(self) -> None:
        """测试：飞书 challenge 校验请求会被原样返回。"""
        gateway = WebhookGateway(
            [WebhookRouteConfig(provider="feishu", path="/webhook/feishu")],
            {"approval_policy": ApprovalPolicy.NEVER},
            lambda runtime_context: FakeWebhookRunner(),
        )

        response = gateway.handle_request(
            "POST",
            "/webhook/feishu",
            {"content-type": "application/json"},
            json.dumps(
                {"type": "url_verification", "challenge": "challenge-token"},
                ensure_ascii=False,
            ).encode("utf-8"),
        )

        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.body)
        self.assertEqual(payload["challenge"], "challenge-token")

    def test_dingtalk_request_can_use_session_webhook_and_restore_history(self) -> None:
        """测试：钉钉 webhook 会优先用 sessionWebhook 回复，并复用同一会话历史。"""
        sent_payloads: list[tuple[str, dict[str, object], float]] = []

        def fake_reply_sender(
            url: str,
            payload: dict[str, object],
            timeout_seconds: float,
        ) -> WebhookDeliveryReceipt:
            sent_payloads.append((url, payload, timeout_seconds))
            return WebhookDeliveryReceipt(status_code=200, response_text="ok")

        with TemporaryDirectory() as temp_dir:
            gateway = WebhookGateway(
                [WebhookRouteConfig(provider="dingtalk", path="/webhook/dingtalk")],
                {"approval_policy": ApprovalPolicy.NEVER},
                lambda runtime_context: FakeWebhookRunner(),
                base_dir=Path(temp_dir),
                reply_sender=fake_reply_sender,
            )
            first_response = gateway.handle_request(
                "POST",
                "/webhook/dingtalk",
                {"content-type": "application/json"},
                json.dumps(
                    {
                        "msgtype": "text",
                        "msgId": "msg-001",
                        "conversationId": "cid-test-001",
                        "senderNick": "张三",
                        "senderStaffId": "zhangsan",
                        "sessionWebhook": "https://oapi.dingtalk.com/robot/sendBySession?token=test",
                        "text": {"content": "第一条消息"},
                    },
                    ensure_ascii=False,
                ).encode("utf-8"),
            )
            second_response = gateway.handle_request(
                "POST",
                "/webhook/dingtalk",
                {"content-type": "application/json"},
                json.dumps(
                    {
                        "msgtype": "text",
                        "msgId": "msg-002",
                        "conversationId": "cid-test-001",
                        "senderNick": "张三",
                        "senderStaffId": "zhangsan",
                        "sessionWebhook": "https://oapi.dingtalk.com/robot/sendBySession?token=test",
                        "text": {"content": "第二条消息"},
                    },
                    ensure_ascii=False,
                ).encode("utf-8"),
            )

        self.assertEqual(first_response.status_code, 200)
        self.assertEqual(second_response.status_code, 200)
        self.assertEqual(len(sent_payloads), 2)
        self.assertIn("第1轮回复", sent_payloads[0][1]["text"]["content"])
        self.assertIn("第一条消息", sent_payloads[0][1]["text"]["content"])
        self.assertIn("第2轮回复", sent_payloads[1][1]["text"]["content"])
        self.assertIn("第二条消息", sent_payloads[1][1]["text"]["content"])

    def test_email_request_requires_shared_secret_and_uses_configured_reply_webhook(self) -> None:
        """测试：邮件 webhook 可校验共享密钥，并将回复投递到配置中的回包地址。"""
        sent_payloads: list[tuple[str, dict[str, object], float]] = []

        def fake_reply_sender(
            url: str,
            payload: dict[str, object],
            timeout_seconds: float,
        ) -> WebhookDeliveryReceipt:
            sent_payloads.append((url, payload, timeout_seconds))
            return WebhookDeliveryReceipt(status_code=202, response_text="accepted")

        with TemporaryDirectory() as temp_dir:
            gateway = WebhookGateway(
                [
                    WebhookRouteConfig(
                        provider="email",
                        path="/webhook/email",
                        reply_webhook_url="https://mail-bridge.example.com/reply",
                        secret="test-secret",
                    )
                ],
                {"approval_policy": ApprovalPolicy.NEVER},
                lambda runtime_context: FakeWebhookRunner(),
                base_dir=Path(temp_dir),
                reply_sender=fake_reply_sender,
            )
            unauthorized_response = gateway.handle_request(
                "POST",
                "/webhook/email",
                {"content-type": "application/x-www-form-urlencoded"},
                urlencode(
                    {
                        "from": "alice@example.com",
                        "subject": "Webhook 邮件",
                        "text": "请总结一下今天的告警",
                    }
                ).encode("utf-8"),
            )
            authorized_response = gateway.handle_request(
                "POST",
                "/webhook/email",
                {
                    "content-type": "application/x-www-form-urlencoded",
                    "x-cyber-agent-webhook-secret": "test-secret",
                },
                urlencode(
                    {
                        "from": "alice@example.com",
                        "subject": "Webhook 邮件",
                        "text": "请总结一下今天的告警",
                    }
                ).encode("utf-8"),
            )

        self.assertEqual(unauthorized_response.status_code, 401)
        self.assertEqual(authorized_response.status_code, 200)
        self.assertEqual(len(sent_payloads), 1)
        self.assertEqual(sent_payloads[0][0], "https://mail-bridge.example.com/reply")
        self.assertEqual(sent_payloads[0][1]["to"], "alice@example.com")
        self.assertIn("Webhook 邮件", sent_payloads[0][1]["subject"])
        self.assertIn("今天的告警", sent_payloads[0][1]["text"])

    def test_wecom_xml_request_returns_bridge_notice(self) -> None:
        """测试：企微官方 XML 回调未联调前，会明确返回网关桥接说明。"""
        gateway = WebhookGateway(
            [WebhookRouteConfig(provider="wecom", path="/webhook/wecom")],
            {"approval_policy": ApprovalPolicy.NEVER},
            lambda runtime_context: FakeWebhookRunner(),
        )

        response = gateway.handle_request(
            "POST",
            "/webhook/wecom",
            {"content-type": "text/xml"},
            b"<xml><ToUserName>test</ToUserName></xml>",
        )

        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.body)
        self.assertEqual(payload["status"], "ignored")
        self.assertIn("JSON 企微回调", payload["reason"])


if __name__ == "__main__":
    unittest.main()
