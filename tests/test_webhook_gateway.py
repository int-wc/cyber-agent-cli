import base64
import hashlib
import json
import os
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote_plus, urlencode
from unittest.mock import patch
from xml.etree import ElementTree

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from cyber_agent.agent.approval import ApprovalPolicy
from cyber_agent.agent.mode import AgentMode
from cyber_agent.cli.webhook import (
    FEISHU_NONCE_HEADER,
    FEISHU_SIGNATURE_HEADER,
    FEISHU_TIMESTAMP_HEADER,
    WECOM_ECHOSTR_QUERY_KEY,
    WECOM_MESSAGE_SIGNATURE_QUERY_KEY,
    WECOM_NONCE_QUERY_KEY,
    WECOM_TIMESTAMP_QUERY_KEY,
    WebhookDeliveryReceipt,
    WebhookEvent,
    WebhookGateway,
    WebhookRouteConfig,
    _aes_cbc_encrypt,
    _build_wecom_signature,
    _decrypt_wecom_ciphertext,
    _encrypt_wecom_plaintext,
    build_webhook_session_id,
    load_webhook_routes_from_file,
    parse_feishu_payload,
)
from cyber_agent.session_store import load_session_history


class FakeWebhookRunner:
    """用于 webhook 网关测试的简化运行器。"""

    def __init__(self) -> None:
        self.mode = AgentMode.STANDARD
        self.tools = []
        self.allowed_roots = []
        self.extra_allowed_paths = []
        self.command_registry = {}
        self.configured_registry = {}
        self.service = "openai"
        self.model_name = "gpt-5.4"
        self.base_url = None
        self.api_key = "test-key"
        self.history = [SystemMessage(content="webhook system prompt")]

    def restore_history(self, messages) -> None:
        self.history = list(messages)

    def get_history_snapshot(self):
        return list(self.history)

    def get_turn_count(self) -> int:
        return sum(isinstance(message, HumanMessage) for message in self.history)

    def get_context_diagnostics(self) -> dict[str, int]:
        return {
            "history_message_count": len(self.history),
            "model_message_count": len(self.history),
            "compressed_message_count": 0,
        }

    def reset(self) -> None:
        self.history = [SystemMessage(content="webhook system prompt")]

    def switch_mode(self, mode: AgentMode) -> None:
        self.mode = mode

    def run(
        self,
        user_input: str,
        verbose: bool = False,
        event_handler=None,
        approval_handler=None,
    ) -> str:
        _ = verbose, event_handler, approval_handler
        self.history.append(HumanMessage(content=user_input))
        current_turn = self.get_turn_count()
        reply_text = f"第{current_turn}轮回复: {user_input}"
        self.history.append(AIMessage(content=reply_text))
        return reply_text


class BlockingWebhookRunner(FakeWebhookRunner):
    """用于验证异步回包链路是否会先快速确认请求。"""

    def __init__(self, release_event: threading.Event) -> None:
        super().__init__()
        self.release_event = release_event

    def run(
        self,
        user_input: str,
        verbose: bool = False,
        event_handler=None,
        approval_handler=None,
    ) -> str:
        _ = verbose, event_handler, approval_handler
        if not self.release_event.wait(timeout=2.0):
            raise TimeoutError("测试中的阻塞运行器未被释放。")
        return super().run(
            user_input,
            verbose=verbose,
            event_handler=event_handler,
            approval_handler=approval_handler,
        )


class RichReplyWebhookRunner(FakeWebhookRunner):
    """用于验证普通 AI 长回答在飞书里会被格式化成更易读的卡片。"""

    def run(
        self,
        user_input: str,
        verbose: bool = False,
        event_handler=None,
        approval_handler=None,
    ) -> str:
        _ = verbose, event_handler, approval_handler
        self.history.append(HumanMessage(content=user_input))
        reply_text = (
            "# 巡检结论\n\n"
            "本次分析已完成，下面给出结构化结果。\n\n"
            "1. 入口链路已经打通。\n"
            "2. 飞书消息投递成功。\n"
            "3. 建议继续观察日志中的重试情况。\n\n"
            "```python\n"
            "def summarize() -> str:\n"
            "    return 'ok'\n"
            "```\n\n"
            "后续如果仍有异常，请继续补充现场日志。"
        )
        self.history.append(AIMessage(content=reply_text))
        return reply_text


class ShortMarkdownReplyWebhookRunner(FakeWebhookRunner):
    """用于验证普通 AI 短回答在飞书里也会统一走 markdown 卡片。"""

    def run(
        self,
        user_input: str,
        verbose: bool = False,
        event_handler=None,
        approval_handler=None,
    ) -> str:
        _ = user_input, verbose, event_handler, approval_handler
        self.history.append(HumanMessage(content="查看机器配置"))
        reply_text = (
            "当前机器配置如下（基于实际命令输出）：\n\n"
            "**系统环境**\n"
            "- Debian GNU/Linux 13 (trixie)\n"
            "- 内核：`6.6.87.2-microsoft-standard-WSL2`\n\n"
            "**简要结论**\n"
            "- 这是一台 **Windows + WSL2** 上的 Debian 13 环境"
        )
        self.history.append(AIMessage(content=reply_text))
        return reply_text


class ProcessTraceWebhookRunner(FakeWebhookRunner):
    """用于验证飞书会把工具调用和采集结果一起展示出来。"""

    def run(
        self,
        user_input: str,
        verbose: bool = False,
        event_handler=None,
        approval_handler=None,
    ) -> str:
        _ = verbose, approval_handler
        self.history.append(HumanMessage(content=user_input))
        if callable(event_handler):
            event_handler(
                "tool_call",
                [
                    {
                        "id": "call_001",
                        "name": "run_shell_command",
                        "args": {
                            "command": "uname -a",
                        },
                    }
                ],
            )
            event_handler(
                "tool_result",
                {
                    "tool_name": "run_shell_command",
                    "content": "Linux demo-host 6.6.87.2-microsoft-standard-WSL2 x86_64",
                },
            )
            event_handler(
                "tool_call",
                [
                    {
                        "id": "call_002",
                        "name": "run_shell_command",
                        "args": {
                            "command": "df -h /",
                        },
                    }
                ],
            )
            event_handler(
                "tool_result",
                {
                    "tool_name": "run_shell_command",
                    "content": "Filesystem Size Used Avail Use% Mounted on\n/dev/sdb 1007G 149G 808G 16% /",
                },
            )
        reply_text = (
            "当前系统空间充足，根分区 `/` 还有约 `808G` 可用。\n\n"
            "建议继续关注 `D:` 盘占用情况。"
        )
        self.history.append(AIMessage(content=reply_text))
        return reply_text


class ToolListWebhookRunner(FakeWebhookRunner):
    """用于验证飞书 `/tools` 会把工具说明排版成更易读的表格。"""

    def __init__(self) -> None:
        super().__init__()
        self.tools = [
            type(
                "FakeTool",
                (),
                {
                    "name": "scan_port",
                    "description": "扫描指定目标IP或域名的指定端口是否开放。\n仅保留首行说明。",
                },
            )(),
            type(
                "FakeTool",
                (),
                {
                    "name": "search_web",
                    "description": "执行关键词网络搜索，优先使用浏览器首页交互搜索。",
                },
            )(),
            type(
                "FakeTool",
                (),
                {
                    "name": "run_shell_command",
                    "description": "在受限工作目录内执行 shell 命令。",
                },
            )(),
        ]


class ContextPreviewWebhookRunner(FakeWebhookRunner):
    """用于验证飞书 `/context` 会以结构化卡片展示上下文摘要。"""

    def get_model_context_snapshot(self):
        return list(self.history)

    def get_context_diagnostics(self) -> dict[str, object]:
        return {
            "history_message_count": 6,
            "model_message_count": 4,
            "compressed_message_count": 2,
            "compressed_summary": "用户此前已确认需要飞书端结构化展示命令结果。",
            "history_preview": [
                "system: webhook system prompt",
                "user: 查看当前状态",
                "assistant: 已开始处理",
                "tool:run_shell_command: df -h /",
                "assistant: 当前磁盘空间充足",
            ],
            "model_preview": [
                "system: webhook system prompt",
                "user: 查看当前状态",
                "assistant: 当前磁盘空间充足",
            ],
        }


def _pad_pkcs7(payload: bytes, block_size: int) -> bytes:
    padding_size = block_size - (len(payload) % block_size)
    if padding_size == 0:
        padding_size = block_size
    return payload + bytes([padding_size]) * padding_size


def _build_feishu_encrypted_body(
    payload: dict[str, object],
    encrypt_key: str,
) -> bytes:
    plaintext = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    iv = bytes(range(16))
    aes_key = hashlib.sha256(encrypt_key.encode("utf-8")).digest()
    ciphertext = _aes_cbc_encrypt(aes_key, iv, _pad_pkcs7(plaintext, 16))
    encrypted_payload = base64.b64encode(iv + ciphertext).decode("utf-8")
    return json.dumps({"encrypt": encrypted_payload}, ensure_ascii=False).encode("utf-8")


def _build_feishu_signature_headers(
    body: bytes,
    encrypt_key: str,
    *,
    timestamp_value: str = "1713600000",
    nonce_value: str = "nonce-001",
) -> dict[str, str]:
    signature = hashlib.sha256(
        timestamp_value.encode("utf-8")
        + nonce_value.encode("utf-8")
        + encrypt_key.encode("utf-8")
        + body
    ).hexdigest()
    return {
        FEISHU_TIMESTAMP_HEADER: timestamp_value,
        FEISHU_NONCE_HEADER: nonce_value,
        FEISHU_SIGNATURE_HEADER: signature,
    }


def _build_wecom_request_xml(
    plaintext_xml: str,
    *,
    token: str,
    encoding_aes_key: str,
    receive_id: str,
    timestamp_value: str = "1713600001",
    nonce_value: str = "nonce-002",
) -> tuple[str, str]:
    encrypted_payload = _encrypt_wecom_plaintext(
        plaintext_xml,
        encoding_aes_key,
        receive_id,
    )
    signature = _build_wecom_signature(
        token,
        timestamp_value,
        nonce_value,
        encrypted_payload,
    )
    request_xml = (
        "<xml>"
        f"<ToUserName>{receive_id}</ToUserName>"
        f"<Encrypt>{encrypted_payload}</Encrypt>"
        "</xml>"
    )
    return request_xml, signature


def _build_feishu_event(
    text: str,
    *,
    message_id: str,
    chat_id: str = "oc_test_chat",
    sender_id: str = "ou_feishu_user",
) -> WebhookEvent:
    """构造飞书事件对象，便于直接复用网关的 handle_event 测试分支。"""
    return WebhookEvent(
        provider="feishu",
        session_key=chat_id,
        sender_id=sender_id,
        sender_name=sender_id,
        message_id=message_id,
        text=text,
        metadata={"chat_id": chat_id},
    )


class WebhookGatewayTestCase(unittest.TestCase):
    def test_load_webhook_routes_from_provider_map_skips_blank_entries(self) -> None:
        """测试：providers 通用配置中未填写关键字段的平台会被自动跳过。"""
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "webhook-routes.json"
            config_path.write_text(
                json.dumps(
                    {
                        "providers": {
                            "feishu": {
                                "path": "/webhook/feishu",
                                "provider_options": {
                                    "verification_token": "feishu-token",
                                    "encrypt_key": "feishu-encrypt-key",
                                },
                            },
                            "dingtalk": {
                                "path": "/webhook/dingtalk",
                                "secret": "",
                            },
                            "wecom": {
                                "path": "/webhook/wecom",
                                "provider_options": {
                                    "token": "",
                                    "encoding_aes_key": "",
                                },
                            },
                            "email": {
                                "path": "/webhook/email",
                                "secret": "mail-secret",
                            },
                        }
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            routes = load_webhook_routes_from_file(config_path)

        self.assertEqual(
            {(route.provider, route.path) for route in routes},
            {
                ("feishu", "/webhook/feishu"),
                ("email", "/webhook/email"),
            },
        )

    def test_load_webhook_routes_from_provider_map_rejects_all_disabled_entries(self) -> None:
        """测试：providers 通用配置若全部留空，应明确报错提示没有启用任何 webhook。"""
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "webhook-routes.json"
            config_path.write_text(
                json.dumps(
                    {
                        "providers": {
                            "feishu": {"path": "/webhook/feishu"},
                            "dingtalk": {"path": "/webhook/dingtalk"},
                        }
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "没有任何已启用的路由"):
                load_webhook_routes_from_file(config_path)

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
        sent_payloads: list[tuple[str, dict[str, object], float, dict[str, str] | None]] = []

        def fake_reply_sender(
            url: str,
            payload: dict[str, object],
            timeout_seconds: float,
            headers: dict[str, str] | None = None,
        ) -> WebhookDeliveryReceipt:
            sent_payloads.append((url, payload, timeout_seconds, headers))
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
        sent_payloads: list[tuple[str, dict[str, object], float, dict[str, str] | None]] = []

        def fake_reply_sender(
            url: str,
            payload: dict[str, object],
            timeout_seconds: float,
            headers: dict[str, str] | None = None,
        ) -> WebhookDeliveryReceipt:
            sent_payloads.append((url, payload, timeout_seconds, headers))
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

    def test_feishu_encrypted_url_verification_can_use_official_encrypt_key(self) -> None:
        """测试：飞书配置 Encrypt Key 后，仍可完成官方 challenge 校验。"""
        encrypt_key = "feishu-encrypt-key"
        gateway = WebhookGateway(
            [
                WebhookRouteConfig(
                    provider="feishu",
                    path="/webhook/feishu",
                    provider_options={
                        "verification_token": "feishu-token",
                        "encrypt_key": encrypt_key,
                    },
                )
            ],
            {"approval_policy": ApprovalPolicy.NEVER},
            lambda runtime_context: FakeWebhookRunner(),
        )

        response = gateway.handle_request(
            "POST",
            "/webhook/feishu",
            {"content-type": "application/json"},
            _build_feishu_encrypted_body(
                {
                    "type": "url_verification",
                    "token": "feishu-token",
                    "challenge": "encrypted-challenge",
                },
                encrypt_key,
            ),
        )

        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.body)
        self.assertEqual(payload["challenge"], "encrypted-challenge")

    def test_feishu_encrypted_event_requires_valid_signature(self) -> None:
        """测试：飞书加密事件会校验官方签名，失败时返回 401。"""
        encrypt_key = "feishu-event-encrypt-key"
        gateway = WebhookGateway(
            [
                WebhookRouteConfig(
                    provider="feishu",
                    path="/webhook/feishu",
                    provider_options={
                        "verification_token": "feishu-token",
                        "encrypt_key": encrypt_key,
                    },
                )
            ],
            {"approval_policy": ApprovalPolicy.NEVER},
            lambda runtime_context: FakeWebhookRunner(),
        )
        encrypted_body = _build_feishu_encrypted_body(
            {
                "schema": "2.0",
                "token": "feishu-token",
                "type": "event_callback",
                "event": {
                    "sender": {
                        "sender_id": {
                            "open_id": "ou_feishu_user",
                        }
                    },
                    "message": {
                        "message_id": "om_001",
                        "chat_id": "oc_test_chat",
                        "message_type": "text",
                        "content": json.dumps({"text": "加密飞书消息"}, ensure_ascii=False),
                    },
                },
            },
            encrypt_key,
        )

        unauthorized_response = gateway.handle_request(
            "POST",
            "/webhook/feishu",
            {"content-type": "application/json"},
            encrypted_body,
        )
        authorized_response = gateway.handle_request(
            "POST",
            "/webhook/feishu",
            {"content-type": "application/json", **_build_feishu_signature_headers(encrypted_body, encrypt_key)},
            encrypted_body,
        )

        self.assertEqual(unauthorized_response.status_code, 401)
        self.assertEqual(authorized_response.status_code, 200)
        payload = json.loads(authorized_response.body)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["provider"], "feishu")
        self.assertIn("reply_payload", payload)

    def test_feishu_long_connection_payload_can_reuse_existing_parser(self) -> None:
        """测试：飞书长连接事件体可直接复用现有 feishu 事件解析逻辑。"""
        route = WebhookRouteConfig(
            provider="feishu",
            path="/webhook/feishu",
            provider_options={"verification_token": "feishu-token"},
        )

        outcome = parse_feishu_payload(
            {
                "schema": "2.0",
                "header": {
                    "event_type": "im.message.receive_v1",
                    "token": "feishu-token",
                },
                "event": {
                    "sender": {
                        "sender_id": {
                            "open_id": "ou_feishu_user",
                        }
                    },
                    "message": {
                        "chat_id": "oc_test_chat",
                        "message_id": "om_test_message",
                        "message_type": "text",
                        "content": "{\"text\":\"你好，长连接\"}",
                    },
                },
            },
            route,
        )

        self.assertIsNone(outcome.immediate_response)
        self.assertIsNotNone(outcome.event)
        assert outcome.event is not None
        self.assertEqual(outcome.event.provider, "feishu")
        self.assertEqual(outcome.event.sender_id, "ou_feishu_user")
        self.assertEqual(outcome.event.message_id, "om_test_message")
        self.assertEqual(outcome.event.text, "你好，长连接")
        self.assertEqual(outcome.event.metadata["event_type"], "im.message.receive_v1")

    def test_feishu_payload_still_requires_verification_token_by_default(self) -> None:
        """测试：普通 feishu 解析默认仍要求 Verification Token 校验。"""
        route = WebhookRouteConfig(
            provider="feishu",
            path="/webhook/feishu",
            provider_options={"verification_token": "feishu-token"},
        )

        with self.assertRaisesRegex(Exception, "Verification Token"):
            parse_feishu_payload(
                {
                    "schema": "2.0",
                    "header": {
                        "event_type": "im.message.receive_v1",
                    },
                    "event": {
                        "sender": {
                            "sender_id": {
                                "open_id": "ou_feishu_user",
                            }
                        },
                        "message": {
                            "chat_id": "oc_test_chat",
                            "message_id": "om_test_message",
                            "message_type": "text",
                            "content": "{\"text\":\"你好，长连接\"}",
                        },
                    },
                },
                route,
            )

    def test_feishu_event_can_reply_via_official_reply_api(self) -> None:
        """测试：飞书事件在配置应用凭证后，可通过官方回复接口回到原消息。"""
        sent_requests: list[tuple[str, dict[str, object], dict[str, str] | None]] = []

        def fake_reply_sender(
            url: str,
            payload: dict[str, object],
            timeout_seconds: float,
            headers: dict[str, str] | None = None,
        ) -> WebhookDeliveryReceipt:
            _ = timeout_seconds
            sent_requests.append((url, payload, headers))
            if url.endswith("/auth/v3/tenant_access_token/internal"):
                return WebhookDeliveryReceipt(
                    status_code=200,
                    response_text=json.dumps(
                        {
                            "code": 0,
                            "tenant_access_token": "tenant-token",
                            "expire": 7200,
                        },
                        ensure_ascii=False,
                    ),
                )
            if url.endswith("/messages/om_001/reply"):
                return WebhookDeliveryReceipt(
                    status_code=200,
                    response_text=json.dumps(
                        {
                            "code": 0,
                            "data": {
                                "message_id": "om_reply_001",
                            },
                        },
                        ensure_ascii=False,
                    ),
                )
            raise AssertionError(f"未预期的请求地址: {url}")

        gateway = WebhookGateway(
            [
                WebhookRouteConfig(
                    provider="feishu",
                    path="/webhook/feishu",
                    provider_options={
                        "verification_token": "feishu-token",
                        "app_id": "cli_test_app",
                        "app_secret": "test-secret",
                        "reply_mode": "reply_api",
                        "reply_in_thread": "true",
                    },
                )
            ],
            {"approval_policy": ApprovalPolicy.NEVER},
            lambda runtime_context: FakeWebhookRunner(),
            reply_sender=fake_reply_sender,
        )

        response = gateway.handle_request(
            "POST",
            "/webhook/feishu",
            {"content-type": "application/json"},
            json.dumps(
                {
                    "schema": "2.0",
                    "token": "feishu-token",
                    "type": "event_callback",
                    "event": {
                        "sender": {
                            "sender_id": {
                                "open_id": "ou_feishu_user",
                            }
                        },
                        "message": {
                            "message_id": "om_001",
                            "chat_id": "oc_test_chat",
                            "message_type": "text",
                            "content": json.dumps({"text": "你好"}, ensure_ascii=False),
                        },
                    },
                },
                ensure_ascii=False,
            ).encode("utf-8"),
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(gateway.wait_until_async_idle())
        response_payload = json.loads(response.body)
        self.assertEqual(response_payload["msg"], "success")
        self.assertEqual(len(sent_requests), 3)
        self.assertTrue(sent_requests[0][0].endswith("/auth/v3/tenant_access_token/internal"))
        self.assertTrue(sent_requests[1][0].endswith("/messages/om_001/reply"))
        self.assertTrue(sent_requests[2][0].endswith("/messages/om_001/reply"))
        self.assertEqual(
            sent_requests[1][2],
            {"Authorization": "Bearer tenant-token"},
        )
        self.assertEqual(sent_requests[1][1]["msg_type"], "interactive")
        self.assertEqual(sent_requests[1][1]["reply_in_thread"], True)
        start_card_payload = json.loads(str(sent_requests[1][1]["content"]))
        start_card_text = json.dumps(start_card_payload, ensure_ascii=False)
        self.assertIn("处理中 · 已收到任务，开始处理", start_card_text)

        card_payload = json.loads(str(sent_requests[2][1]["content"]))
        serialized_card = json.dumps(card_payload, ensure_ascii=False)
        self.assertIn("你好", serialized_card)
        self.assertTrue(str(sent_requests[2][1]["uuid"]))

    def test_feishu_long_ai_reply_can_use_interactive_markdown_card(self) -> None:
        """测试：普通 AI 长回答会自动转成更适合飞书阅读的交互卡片。"""
        sent_requests: list[tuple[str, dict[str, object], dict[str, str] | None]] = []

        def fake_reply_sender(
            url: str,
            payload: dict[str, object],
            timeout_seconds: float,
            headers: dict[str, str] | None = None,
        ) -> WebhookDeliveryReceipt:
            _ = timeout_seconds
            sent_requests.append((url, payload, headers))
            if url.endswith("/auth/v3/tenant_access_token/internal"):
                return WebhookDeliveryReceipt(
                    status_code=200,
                    response_text=json.dumps(
                        {
                            "code": 0,
                            "tenant_access_token": "tenant-token",
                            "expire": 7200,
                        },
                        ensure_ascii=False,
                    ),
                )
            if url.endswith("/messages/om_rich/reply"):
                return WebhookDeliveryReceipt(
                    status_code=200,
                    response_text=json.dumps(
                        {
                            "code": 0,
                            "data": {
                                "message_id": "om_rich_reply",
                            },
                        },
                        ensure_ascii=False,
                    ),
                )
            raise AssertionError(f"未预期的请求地址: {url}")

        gateway = WebhookGateway(
            [
                WebhookRouteConfig(
                    provider="feishu",
                    path="/webhook/feishu",
                    provider_options={
                        "verification_token": "feishu-token",
                        "app_id": "cli_test_app",
                        "app_secret": "test-secret",
                        "reply_mode": "reply_api",
                    },
                )
            ],
            {"approval_policy": ApprovalPolicy.NEVER},
            lambda runtime_context: RichReplyWebhookRunner(),
            reply_sender=fake_reply_sender,
        )

        response = gateway.handle_request(
            "POST",
            "/webhook/feishu",
            {"content-type": "application/json"},
            json.dumps(
                {
                    "schema": "2.0",
                    "token": "feishu-token",
                    "type": "event_callback",
                    "event": {
                        "sender": {
                            "sender_id": {
                                "open_id": "ou_feishu_user",
                            }
                        },
                        "message": {
                            "message_id": "om_rich",
                            "chat_id": "oc_test_chat",
                            "message_type": "text",
                            "content": json.dumps({"text": "请总结当前状态"}, ensure_ascii=False),
                        },
                    },
                },
                ensure_ascii=False,
            ).encode("utf-8"),
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(gateway.wait_until_async_idle())
        self.assertEqual(json.loads(response.body)["msg"], "success")
        self.assertEqual(len(sent_requests), 3)
        start_card = json.loads(str(sent_requests[1][1]["content"]))
        start_text = json.dumps(start_card, ensure_ascii=False)
        self.assertIn("处理中 · 已收到任务，开始处理", start_text)
        self.assertIn("请总结当前状态", start_text)

        self.assertEqual(sent_requests[2][1]["msg_type"], "interactive")
        card_payload = json.loads(str(sent_requests[2][1]["content"]))
        serialized_card = json.dumps(card_payload, ensure_ascii=False)
        self.assertEqual(card_payload["header"]["title"]["content"], "巡检结论")
        self.assertIn("本次分析已完成", serialized_card)
        self.assertIn("1. 入口链路已经打通。", serialized_card)
        self.assertIn("```python", serialized_card)
        self.assertNotIn("# 巡检结论", serialized_card)

    def test_feishu_short_ai_reply_also_uses_interactive_markdown_card(self) -> None:
        """测试：普通 AI 短回答在飞书里也统一使用 markdown 卡片。"""
        sent_requests: list[tuple[str, dict[str, object], dict[str, str] | None]] = []

        def fake_reply_sender(
            url: str,
            payload: dict[str, object],
            timeout_seconds: float,
            headers: dict[str, str] | None = None,
        ) -> WebhookDeliveryReceipt:
            _ = timeout_seconds
            sent_requests.append((url, payload, headers))
            if url.endswith("/auth/v3/tenant_access_token/internal"):
                return WebhookDeliveryReceipt(
                    status_code=200,
                    response_text=json.dumps(
                        {
                            "code": 0,
                            "tenant_access_token": "tenant-token",
                            "expire": 7200,
                        },
                        ensure_ascii=False,
                    ),
                )
            if url.endswith("/messages/om_short/reply"):
                return WebhookDeliveryReceipt(
                    status_code=200,
                    response_text=json.dumps(
                        {
                            "code": 0,
                            "data": {
                                "message_id": "om_short_reply",
                            },
                        },
                        ensure_ascii=False,
                    ),
                )
            raise AssertionError(f"未预期的请求地址: {url}")

        gateway = WebhookGateway(
            [
                WebhookRouteConfig(
                    provider="feishu",
                    path="/webhook/feishu",
                    provider_options={
                        "verification_token": "feishu-token",
                        "app_id": "cli_test_app",
                        "app_secret": "test-secret",
                        "reply_mode": "reply_api",
                    },
                )
            ],
            {"approval_policy": ApprovalPolicy.NEVER},
            lambda runtime_context: ShortMarkdownReplyWebhookRunner(),
            reply_sender=fake_reply_sender,
        )

        response = gateway.handle_request(
            "POST",
            "/webhook/feishu",
            {"content-type": "application/json"},
            json.dumps(
                {
                    "schema": "2.0",
                    "token": "feishu-token",
                    "type": "event_callback",
                    "event": {
                        "sender": {
                            "sender_id": {
                                "open_id": "ou_feishu_user",
                            }
                        },
                        "message": {
                            "message_id": "om_short",
                            "chat_id": "oc_test_chat",
                            "message_type": "text",
                            "content": json.dumps({"text": "查看机器配置"}, ensure_ascii=False),
                        },
                    },
                },
                ensure_ascii=False,
            ).encode("utf-8"),
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(gateway.wait_until_async_idle())
        self.assertEqual(json.loads(response.body)["msg"], "success")
        self.assertEqual(len(sent_requests), 3)
        start_card = json.loads(str(sent_requests[1][1]["content"]))
        start_text = json.dumps(start_card, ensure_ascii=False)
        self.assertIn("处理中 · 已收到任务，开始处理", start_text)
        self.assertIn("查看机器配置", start_text)

        self.assertEqual(sent_requests[2][1]["msg_type"], "interactive")
        card_payload = json.loads(str(sent_requests[2][1]["content"]))
        serialized_card = json.dumps(card_payload, ensure_ascii=False)
        self.assertEqual(
            card_payload["header"]["title"]["content"],
            "当前机器配置如下（基于实际命令输出）",
        )
        self.assertIn("**系统环境**", serialized_card)
        self.assertIn("`6.6.87.2-microsoft-standard-WSL2`", serialized_card)
        self.assertIn("**Windows + WSL2**", serialized_card)

    def test_feishu_ai_reply_can_send_tool_trace_as_separate_messages(self) -> None:
        """测试：飞书普通回复会把工具调用和采集结果拆成多条独立消息发送。"""
        sent_requests: list[tuple[str, dict[str, object], dict[str, str] | None]] = []

        def fake_reply_sender(
            url: str,
            payload: dict[str, object],
            timeout_seconds: float,
            headers: dict[str, str] | None = None,
        ) -> WebhookDeliveryReceipt:
            _ = timeout_seconds
            sent_requests.append((url, payload, headers))
            if url.endswith("/auth/v3/tenant_access_token/internal"):
                return WebhookDeliveryReceipt(
                    status_code=200,
                    response_text=json.dumps(
                        {
                            "code": 0,
                            "tenant_access_token": "tenant-token",
                            "expire": 7200,
                        },
                        ensure_ascii=False,
                    ),
                )
            if url.endswith("/open-apis/im/v1/messages?receive_id_type=chat_id"):
                return WebhookDeliveryReceipt(
                    status_code=200,
                    response_text=json.dumps(
                        {
                            "code": 0,
                            "data": {
                                "message_id": "om_trace_reply",
                            },
                        },
                        ensure_ascii=False,
                    ),
                )
            raise AssertionError(f"未预期的请求地址: {url}")

        gateway = WebhookGateway(
            [
                WebhookRouteConfig(
                    provider="feishu",
                    path="/webhook/feishu",
                    provider_options={
                        "verification_token": "feishu-token",
                        "app_id": "cli_test_app",
                        "app_secret": "test-secret",
                        "reply_mode": "create_api",
                    },
                )
            ],
            {"approval_policy": ApprovalPolicy.NEVER},
            lambda runtime_context: ProcessTraceWebhookRunner(),
            reply_sender=fake_reply_sender,
        )

        response = gateway.handle_request(
            "POST",
            "/webhook/feishu",
            {"content-type": "application/json"},
            json.dumps(
                {
                    "schema": "2.0",
                    "token": "feishu-token",
                    "type": "event_callback",
                    "event": {
                        "sender": {
                            "sender_id": {
                                "open_id": "ou_feishu_user",
                            }
                        },
                        "message": {
                            "message_id": "om_trace",
                            "chat_id": "oc_test_chat",
                            "message_type": "text",
                            "content": json.dumps({"text": "查看当前系统空间"}, ensure_ascii=False),
                        },
                    },
                },
                ensure_ascii=False,
            ).encode("utf-8"),
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(gateway.wait_until_async_idle())
        self.assertEqual(json.loads(response.body)["msg"], "success")
        self.assertEqual(len(sent_requests), 7)
        self.assertTrue(
            sent_requests[0][0].endswith("/auth/v3/tenant_access_token/internal")
        )
        progress_payloads = [payload for _url, payload, _headers in sent_requests[1:]]
        self.assertTrue(all(payload["msg_type"] == "interactive" for payload in progress_payloads))

        first_progress_card = json.loads(str(progress_payloads[0]["content"]))
        first_progress_text = json.dumps(first_progress_card, ensure_ascii=False)
        self.assertIn("处理中 · 已收到任务，开始处理", first_progress_text)
        self.assertIn("查看当前系统空间", first_progress_text)

        tool_call_card = json.loads(str(progress_payloads[1]["content"]))
        tool_call_text = json.dumps(tool_call_card, ensure_ascii=False)
        self.assertIn("处理中 · 调用工具 `run_shell_command`", tool_call_text)
        self.assertIn("uname -a", tool_call_text)

        result_progress_card = json.loads(str(progress_payloads[2]["content"]))
        result_progress_text = json.dumps(result_progress_card, ensure_ascii=False)
        self.assertIn("处理中 · 采集结果 `run_shell_command`", result_progress_text)
        self.assertIn("Linux demo-host", result_progress_text)

        second_tool_call_card = json.loads(str(progress_payloads[3]["content"]))
        second_tool_call_text = json.dumps(second_tool_call_card, ensure_ascii=False)
        self.assertIn("处理中 · 调用工具 `run_shell_command`", second_tool_call_text)
        self.assertIn("df -h /", second_tool_call_text)

        second_result_card = json.loads(str(progress_payloads[4]["content"]))
        second_result_text = json.dumps(second_result_card, ensure_ascii=False)
        self.assertIn("处理中 · 采集结果 `run_shell_command`", second_result_text)
        self.assertIn("/dev/sdb 1007G 149G 808G 16% /", second_result_text)

        final_card = json.loads(str(progress_payloads[-1]["content"]))
        final_text = json.dumps(final_card, ensure_ascii=False)
        self.assertNotIn("处理过程", final_text)
        self.assertNotIn("处理中 ·", final_text)
        self.assertIn("当前系统空间充足", final_text)

    def test_feishu_ai_reply_can_send_heartbeat_when_runner_stays_silent(self) -> None:
        """测试：长时间没有新事件时，飞书会补发一条“仍在执行中”的心跳消息。"""
        sent_requests: list[tuple[str, dict[str, object], dict[str, str] | None]] = []
        release_event = threading.Event()

        def fake_reply_sender(
            url: str,
            payload: dict[str, object],
            timeout_seconds: float,
            headers: dict[str, str] | None = None,
        ) -> WebhookDeliveryReceipt:
            _ = timeout_seconds
            sent_requests.append((url, payload, headers))
            if url.endswith("/auth/v3/tenant_access_token/internal"):
                return WebhookDeliveryReceipt(
                    status_code=200,
                    response_text=json.dumps(
                        {
                            "code": 0,
                            "tenant_access_token": "tenant-token",
                            "expire": 7200,
                        },
                        ensure_ascii=False,
                    ),
                )
            if url.endswith("/open-apis/im/v1/messages?receive_id_type=chat_id"):
                return WebhookDeliveryReceipt(
                    status_code=200,
                    response_text=json.dumps(
                        {
                            "code": 0,
                            "data": {
                                "message_id": "om_heartbeat_reply",
                            },
                        },
                        ensure_ascii=False,
                    ),
                )
            raise AssertionError(f"未预期的请求地址: {url}")

        gateway = WebhookGateway(
            [
                WebhookRouteConfig(
                    provider="feishu",
                    path="/webhook/feishu",
                    provider_options={
                        "verification_token": "feishu-token",
                        "app_id": "cli_test_app",
                        "app_secret": "test-secret",
                        "reply_mode": "create_api",
                    },
                )
            ],
            {"approval_policy": ApprovalPolicy.NEVER},
            lambda runtime_context: BlockingWebhookRunner(release_event),
            reply_sender=fake_reply_sender,
        )

        with patch(
            "cyber_agent.cli.webhook.FEISHU_PROGRESS_HEARTBEAT_IDLE_SECONDS",
            0.05,
        ), patch(
            "cyber_agent.cli.webhook.FEISHU_PROGRESS_HEARTBEAT_POLL_SECONDS",
            0.01,
        ):
            response = gateway.handle_request(
                "POST",
                "/webhook/feishu",
                {"content-type": "application/json"},
                json.dumps(
                    {
                        "schema": "2.0",
                        "token": "feishu-token",
                        "type": "event_callback",
                        "event": {
                            "sender": {
                                "sender_id": {
                                    "open_id": "ou_feishu_user",
                                }
                            },
                            "message": {
                                "message_id": "om_heartbeat",
                                "chat_id": "oc_test_chat",
                                "message_type": "text",
                                "content": json.dumps(
                                    {"text": "长时间采集"},
                                    ensure_ascii=False,
                                ),
                            },
                        },
                    },
                    ensure_ascii=False,
                ).encode("utf-8"),
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(json.loads(response.body)["msg"], "success")

            heartbeat_seen = False
            deadline = time.monotonic() + 0.5
            while time.monotonic() <= deadline:
                heartbeat_seen = any(
                    "处理中 · 任务仍在执行中"
                    in json.dumps(
                        json.loads(str(payload["content"])),
                        ensure_ascii=False,
                    )
                    for _url, payload, _headers in sent_requests[1:]
                    if payload.get("msg_type") == "interactive"
                )
                if heartbeat_seen:
                    break
                time.sleep(0.01)
            self.assertTrue(heartbeat_seen)

            release_event.set()
            self.assertTrue(gateway.wait_until_async_idle())

        final_card = json.loads(str(sent_requests[-1][1]["content"]))
        final_text = json.dumps(final_card, ensure_ascii=False)
        self.assertIn("长时间采集", final_text)

    def test_feishu_start_command_can_reply_with_interactive_menu_card(self) -> None:
        """测试：/start 会返回带按钮的飞书交互卡片菜单。"""
        sent_requests: list[tuple[str, dict[str, object], dict[str, str] | None]] = []

        def fake_reply_sender(
            url: str,
            payload: dict[str, object],
            timeout_seconds: float,
            headers: dict[str, str] | None = None,
        ) -> WebhookDeliveryReceipt:
            _ = timeout_seconds
            sent_requests.append((url, payload, headers))
            if url.endswith("/auth/v3/tenant_access_token/internal"):
                return WebhookDeliveryReceipt(
                    status_code=200,
                    response_text=json.dumps(
                        {
                            "code": 0,
                            "tenant_access_token": "tenant-token",
                            "expire": 7200,
                        },
                        ensure_ascii=False,
                    ),
                )
            if url.endswith("/messages/om_start/reply"):
                return WebhookDeliveryReceipt(
                    status_code=200,
                    response_text=json.dumps(
                        {
                            "code": 0,
                            "data": {
                                "message_id": "om_start_reply",
                            },
                        },
                        ensure_ascii=False,
                    ),
                )
            raise AssertionError(f"未预期的请求地址: {url}")

        gateway = WebhookGateway(
            [
                WebhookRouteConfig(
                    provider="feishu",
                    path="/webhook/feishu",
                    provider_options={
                        "verification_token": "feishu-token",
                        "app_id": "cli_test_app",
                        "app_secret": "test-secret",
                        "reply_mode": "reply_api",
                    },
                )
            ],
            {"approval_policy": ApprovalPolicy.NEVER},
            lambda runtime_context: FakeWebhookRunner(),
            reply_sender=fake_reply_sender,
        )

        response = gateway.handle_request(
            "POST",
            "/webhook/feishu",
            {"content-type": "application/json"},
            json.dumps(
                {
                    "schema": "2.0",
                    "token": "feishu-token",
                    "type": "event_callback",
                    "event": {
                        "sender": {
                            "sender_id": {
                                "open_id": "ou_feishu_user",
                            }
                        },
                        "message": {
                            "message_id": "om_start",
                            "chat_id": "oc_test_chat",
                            "message_type": "text",
                            "content": json.dumps({"text": "/start"}, ensure_ascii=False),
                        },
                    },
                },
                ensure_ascii=False,
            ).encode("utf-8"),
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(gateway.wait_until_async_idle())
        self.assertEqual(json.loads(response.body)["msg"], "success")
        self.assertEqual(len(sent_requests), 2)
        self.assertEqual(sent_requests[1][1]["msg_type"], "interactive")
        card_payload = json.loads(str(sent_requests[1][1]["content"]))
        self.assertEqual(
            card_payload["header"]["title"]["content"],
            "Cyber Agent 飞书快捷菜单",
        )
        serialized_card = json.dumps(card_payload, ensure_ascii=False)
        self.assertIn("新建会话", serialized_card)
        self.assertIn("最近会话", serialized_card)
        self.assertIn("回到默认会话", serialized_card)
        self.assertIn("/session default", serialized_card)
        self.assertIn("/help", serialized_card)
        self.assertIn("/exit", serialized_card)

    def test_feishu_start_command_can_show_recent_session_shortcuts(self) -> None:
        """测试：飞书 /start 会按当前聊天动态展示最近会话的快捷切换入口。"""
        with TemporaryDirectory() as temp_dir:
            gateway = WebhookGateway(
                [WebhookRouteConfig(provider="feishu", path="/webhook/feishu")],
                {"approval_policy": ApprovalPolicy.NEVER},
                lambda runtime_context: FakeWebhookRunner(),
                base_dir=Path(temp_dir),
            )
            route = WebhookRouteConfig(provider="feishu", path="/webhook/feishu")
            default_session_id = build_webhook_session_id("feishu", "oc_test_chat")

            gateway.handle_event(
                route,
                _build_feishu_event("默认会话问题", message_id="om_default_001"),
            )
            create_response = gateway.handle_event(
                route,
                _build_feishu_event("/session new 新专题", message_id="om_session_new"),
            )
            new_session_id = str(json.loads(create_response.body)["session_id"])
            start_response = gateway.handle_event(
                route,
                _build_feishu_event("/start", message_id="om_start_local"),
            )
            start_payload = json.loads(start_response.body)
            reply_payload = dict(start_payload["reply_payload"])
            card_payload = json.loads(str(reply_payload["content"]))
            serialized_card = json.dumps(card_payload, ensure_ascii=False)

        self.assertEqual(start_response.status_code, 200)
        self.assertIn("最近会话", serialized_card)
        self.assertIn(default_session_id, serialized_card)
        self.assertIn(new_session_id, serialized_card)
        self.assertIn("/session default", serialized_card)
        self.assertIn(f"/session use {new_session_id}", serialized_card)

    def test_feishu_tools_command_can_reply_with_markdown_table(self) -> None:
        """测试：飞书 `/tools` 会把工具说明拆成表格，而不是整行项目符号堆叠。"""
        sent_requests: list[tuple[str, dict[str, object], dict[str, str] | None]] = []

        def fake_reply_sender(
            url: str,
            payload: dict[str, object],
            timeout_seconds: float,
            headers: dict[str, str] | None = None,
        ) -> WebhookDeliveryReceipt:
            _ = timeout_seconds
            sent_requests.append((url, payload, headers))
            if url.endswith("/auth/v3/tenant_access_token/internal"):
                return WebhookDeliveryReceipt(
                    status_code=200,
                    response_text=json.dumps(
                        {
                            "code": 0,
                            "tenant_access_token": "tenant-token",
                            "expire": 7200,
                        },
                        ensure_ascii=False,
                    ),
                )
            if url.endswith("/messages/om_tools/reply"):
                return WebhookDeliveryReceipt(
                    status_code=200,
                    response_text=json.dumps(
                        {
                            "code": 0,
                            "data": {
                                "message_id": "om_tools_reply",
                            },
                        },
                        ensure_ascii=False,
                    ),
                )
            raise AssertionError(f"未预期的请求地址: {url}")

        gateway = WebhookGateway(
            [
                WebhookRouteConfig(
                    provider="feishu",
                    path="/webhook/feishu",
                    provider_options={
                        "verification_token": "feishu-token",
                        "app_id": "cli_test_app",
                        "app_secret": "test-secret",
                        "reply_mode": "reply_api",
                    },
                )
            ],
            {"approval_policy": ApprovalPolicy.NEVER},
            lambda runtime_context: ToolListWebhookRunner(),
            reply_sender=fake_reply_sender,
        )

        response = gateway.handle_request(
            "POST",
            "/webhook/feishu",
            {"content-type": "application/json"},
            json.dumps(
                {
                    "schema": "2.0",
                    "token": "feishu-token",
                    "type": "event_callback",
                    "event": {
                        "sender": {
                            "sender_id": {
                                "open_id": "ou_feishu_user",
                            }
                        },
                        "message": {
                            "message_id": "om_tools",
                            "chat_id": "oc_test_chat",
                            "message_type": "text",
                            "content": json.dumps({"text": "/tools"}, ensure_ascii=False),
                        },
                    },
                },
                ensure_ascii=False,
            ).encode("utf-8"),
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(gateway.wait_until_async_idle())
        self.assertEqual(json.loads(response.body)["msg"], "success")
        self.assertEqual(len(sent_requests), 2)
        self.assertEqual(sent_requests[1][1]["msg_type"], "interactive")
        card_payload = json.loads(str(sent_requests[1][1]["content"]))
        serialized_card = json.dumps(card_payload, ensure_ascii=False)
        self.assertEqual(card_payload["header"]["title"]["content"], "默认工具")
        self.assertIn("**`scan_port`**", serialized_card)
        self.assertIn("扫描指定目标IP或域名的指定端口是否开放。", serialized_card)
        self.assertIn("**`search_web`**", serialized_card)
        self.assertIn("执行关键词网络搜索，优先使用浏览器首页交互搜索。", serialized_card)
        self.assertIn("**`run_shell_command`**", serialized_card)
        self.assertNotIn("| 工具 | 说明 |", serialized_card)

    def test_feishu_context_command_can_reply_with_structured_context_card(self) -> None:
        """测试：飞书 `/context` 会展示概览、压缩摘要和上下文预览。"""
        route = WebhookRouteConfig(provider="feishu", path="/webhook/feishu")
        gateway = WebhookGateway(
            [route],
            {"approval_policy": ApprovalPolicy.NEVER, "session_id": "session-demo"},
            lambda runtime_context: ContextPreviewWebhookRunner(),
        )

        response = gateway.handle_event(
            route,
            _build_feishu_event("/context", message_id="om_context"),
        )

        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.body)
        reply_payload = dict(payload["reply_payload"])
        self.assertEqual(reply_payload["msg_type"], "interactive")
        card_payload = json.loads(str(reply_payload["content"]))
        serialized_card = json.dumps(card_payload, ensure_ascii=False)
        self.assertEqual(card_payload["header"]["title"]["content"], "当前上下文")
        self.assertIn("**概览**", serialized_card)
        self.assertIn("**当前会话 ID**", serialized_card)
        self.assertIn("session-demo", serialized_card)
        self.assertIn("**压缩摘要**", serialized_card)
        self.assertIn("当前会话预览", serialized_card)
        self.assertIn("模型实际可见上下文", serialized_card)

    def test_feishu_history_command_can_reply_with_session_table(self) -> None:
        """测试：飞书 `/history` 会把历史会话渲染成结构化表格。"""
        with TemporaryDirectory() as temp_dir:
            route = WebhookRouteConfig(provider="feishu", path="/webhook/feishu")
            gateway = WebhookGateway(
                [route],
                {"approval_policy": ApprovalPolicy.NEVER},
                lambda runtime_context: FakeWebhookRunner(),
                base_dir=Path(temp_dir),
            )
            gateway.handle_event(
                route,
                _build_feishu_event("排查磁盘空间", message_id="om_hist_seed_1"),
            )
            gateway.handle_event(
                route,
                _build_feishu_event("继续查看端口", message_id="om_hist_seed_2"),
            )

            response = gateway.handle_event(
                route,
                _build_feishu_event("/history", message_id="om_history_list"),
            )

        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.body)
        reply_payload = dict(payload["reply_payload"])
        self.assertEqual(reply_payload["msg_type"], "interactive")
        card_payload = json.loads(str(reply_payload["content"]))
        serialized_card = json.dumps(card_payload, ensure_ascii=False)
        self.assertEqual(card_payload["header"]["title"]["content"], "历史会话")
        self.assertIn("**1. 排查磁盘空间**", serialized_card)
        self.assertIn("会话ID", serialized_card)
        self.assertIn("轮数", serialized_card)
        self.assertIn("排查磁盘空间", serialized_card)
        self.assertNotIn("feishu-chat-session-state", serialized_card)

    def test_feishu_doctor_command_can_reply_with_structured_card(self) -> None:
        """测试：飞书 `/doctor` 会按诊断概览、依赖与存储分块展示。"""
        route = WebhookRouteConfig(provider="feishu", path="/webhook/feishu")
        gateway = WebhookGateway(
            [route],
            {"approval_policy": ApprovalPolicy.NEVER},
            lambda runtime_context: FakeWebhookRunner(),
        )

        with patch(
            "cyber_agent.cli.webhook._capture_builtin_command_output_for_webhook",
            return_value=(True, "运行诊断已生成"),
        ), patch(
            "cyber_agent.cli.webhook.build_doctor_payload",
            return_value={
                "summary": {
                    "status_text": "环境基本正常",
                    "reminders": ["建议补装 playwright 浏览器依赖。"],
                },
                "project": {
                    "version": "1.2.3",
                    "python_version": "3.12.2",
                },
                "runtime": {
                    "mode_label": "授权模式",
                    "approval_policy_label": "自动批准",
                    "ui_mode_label": "CLI",
                    "service": "openai",
                    "model": "gpt-5.4",
                    "base_url": "默认",
                    "api_key_configured": True,
                },
                "dependencies": {
                    "langchain_openai": {"status": "已安装"},
                    "langgraph": {"status": "已安装"},
                    "prompt_toolkit": {"status": "已安装"},
                    "textual": {"status": "未安装"},
                    "playwright": {"status": "未安装"},
                },
                "search": {"status": "可用"},
                "storage": {
                    "local_config_path": "D:/demo/config.json",
                    "local_config_status": "存在",
                    "session_storage_status": "正常",
                    "capability_storage_status": "正常",
                },
                "permissions": {
                    "saved_allowed_paths": ["D:/project"],
                    "allowed_roots": ["D:/project"],
                    "registered_tools": ["scan_port", "run_shell_command"],
                },
            },
        ):
            response = gateway.handle_event(
                route,
                _build_feishu_event("/doctor", message_id="om_doctor"),
            )

        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.body)
        reply_payload = dict(payload["reply_payload"])
        self.assertEqual(reply_payload["msg_type"], "interactive")
        card_payload = json.loads(str(reply_payload["content"]))
        serialized_card = json.dumps(card_payload, ensure_ascii=False)
        self.assertEqual(card_payload["header"]["title"]["content"], "运行诊断")
        self.assertIn("**诊断概览**", serialized_card)
        self.assertIn("**依赖检查**", serialized_card)
        self.assertIn("**`playwright`**", serialized_card)
        self.assertIn("playwright", serialized_card)
        self.assertIn("D:/project", serialized_card)

    def test_feishu_help_command_can_reply_with_pretty_card(self) -> None:
        """测试：飞书 /help 会返回结构化卡片，而不是把终端边框原样发出去。"""
        sent_requests: list[tuple[str, dict[str, object], dict[str, str] | None]] = []

        def fake_reply_sender(
            url: str,
            payload: dict[str, object],
            timeout_seconds: float,
            headers: dict[str, str] | None = None,
        ) -> WebhookDeliveryReceipt:
            _ = timeout_seconds
            sent_requests.append((url, payload, headers))
            if url.endswith("/auth/v3/tenant_access_token/internal"):
                return WebhookDeliveryReceipt(
                    status_code=200,
                    response_text=json.dumps(
                        {
                            "code": 0,
                            "tenant_access_token": "tenant-token",
                            "expire": 7200,
                        },
                        ensure_ascii=False,
                    ),
                )
            if url.endswith("/messages/om_help/reply"):
                return WebhookDeliveryReceipt(
                    status_code=200,
                    response_text=json.dumps(
                        {
                            "code": 0,
                            "data": {
                                "message_id": "om_help_reply",
                            },
                        },
                        ensure_ascii=False,
                    ),
                )
            raise AssertionError(f"未预期的请求地址: {url}")

        gateway = WebhookGateway(
            [
                WebhookRouteConfig(
                    provider="feishu",
                    path="/webhook/feishu",
                    provider_options={
                        "verification_token": "feishu-token",
                        "app_id": "cli_test_app",
                        "app_secret": "test-secret",
                        "reply_mode": "reply_api",
                    },
                )
            ],
            {"approval_policy": ApprovalPolicy.NEVER},
            lambda runtime_context: FakeWebhookRunner(),
            reply_sender=fake_reply_sender,
        )

        with patch(
            "cyber_agent.cli.webhook._capture_builtin_command_output_for_webhook",
            return_value=(
                True,
                (
                    "╭──────────── 内建命令 ────────────╮\n"
                    "│ /help  查看帮助                  │\n"
                    "│ /status 查看当前会话与配置状态   │\n"
                    "╰──────────────────────────────────╯"
                ),
            ),
        ) as mock_capture:
            response = gateway.handle_request(
                "POST",
                "/webhook/feishu",
                {"content-type": "application/json"},
                json.dumps(
                    {
                        "schema": "2.0",
                        "token": "feishu-token",
                        "type": "event_callback",
                        "event": {
                            "sender": {
                                "sender_id": {
                                    "open_id": "ou_feishu_user",
                                }
                            },
                            "message": {
                                "message_id": "om_help",
                                "chat_id": "oc_test_chat",
                                "message_type": "text",
                                "content": json.dumps({"text": "/help"}, ensure_ascii=False),
                            },
                        },
                    },
                    ensure_ascii=False,
                ).encode("utf-8"),
            )
            self.assertTrue(gateway.wait_until_async_idle())
            mock_capture.assert_called_once()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(sent_requests), 2)
        self.assertEqual(sent_requests[1][1]["msg_type"], "interactive")
        card_payload = json.loads(str(sent_requests[1][1]["content"]))
        serialized_card = json.dumps(card_payload, ensure_ascii=False)
        self.assertEqual(card_payload["header"]["title"]["content"], "内建命令")
        self.assertIn("/help", serialized_card)
        self.assertIn("/history", serialized_card)
        self.assertIn("/approval", serialized_card)
        self.assertNotIn("╭", serialized_card)
        self.assertNotIn("╰", serialized_card)
        self.assertNotIn("────────", serialized_card)

    def test_feishu_session_new_can_create_and_switch_active_session(self) -> None:
        """测试：飞书 /session new 会创建新会话、切换活动会话，并按聊天维度持久化分组。"""
        with TemporaryDirectory() as temp_dir:
            gateway = WebhookGateway(
                [WebhookRouteConfig(provider="feishu", path="/webhook/feishu")],
                {"approval_policy": ApprovalPolicy.NEVER},
                lambda runtime_context: FakeWebhookRunner(),
                base_dir=Path(temp_dir),
            )
            route = WebhookRouteConfig(provider="feishu", path="/webhook/feishu")
            default_session_id = build_webhook_session_id("feishu", "oc_test_chat")

            gateway.handle_event(
                route,
                _build_feishu_event("默认会话问题", message_id="om_default_001"),
            )
            create_response = gateway.handle_event(
                route,
                _build_feishu_event("/session new 排查磁盘", message_id="om_session_new"),
            )
            create_payload = json.loads(create_response.body)
            new_session_id = str(create_payload["session_id"])
            follow_response = gateway.handle_event(
                route,
                _build_feishu_event("新会话问题", message_id="om_new_001"),
            )
            follow_payload = json.loads(follow_response.body)

            default_session = load_session_history(
                default_session_id,
                base_dir=Path(temp_dir),
            )
            new_session = load_session_history(
                new_session_id,
                base_dir=Path(temp_dir),
            )

        self.assertEqual(create_response.status_code, 200)
        self.assertNotEqual(new_session_id, default_session_id)
        self.assertEqual(follow_payload["session_id"], new_session_id)
        self.assertEqual(default_session.summary.source_session_id, "feishu-chat:oc_test_chat")
        self.assertEqual(new_session.summary.source_session_id, "feishu-chat:oc_test_chat")
        self.assertEqual(default_session.summary.turn_count, 1)
        self.assertEqual(new_session.summary.turn_count, 1)
        self.assertEqual(
            [message.content for message in default_session.messages if isinstance(message, HumanMessage)],
            ["默认会话问题"],
        )
        self.assertEqual(
            [message.content for message in new_session.messages if isinstance(message, HumanMessage)],
            ["新会话问题"],
        )

    def test_feishu_session_use_can_switch_back_to_previous_session(self) -> None:
        """测试：飞书 /session use 可切回之前的会话，后续消息会落到目标历史文件。"""
        with TemporaryDirectory() as temp_dir:
            gateway = WebhookGateway(
                [WebhookRouteConfig(provider="feishu", path="/webhook/feishu")],
                {"approval_policy": ApprovalPolicy.NEVER},
                lambda runtime_context: FakeWebhookRunner(),
                base_dir=Path(temp_dir),
            )
            route = WebhookRouteConfig(provider="feishu", path="/webhook/feishu")
            default_session_id = build_webhook_session_id("feishu", "oc_test_chat")

            gateway.handle_event(
                route,
                _build_feishu_event("默认会话问题", message_id="om_default_001"),
            )
            create_response = gateway.handle_event(
                route,
                _build_feishu_event("/session new 第二会话", message_id="om_session_new"),
            )
            new_session_id = str(json.loads(create_response.body)["session_id"])
            gateway.handle_event(
                route,
                _build_feishu_event("第二会话问题", message_id="om_new_001"),
            )
            switch_response = gateway.handle_event(
                route,
                _build_feishu_event(
                    f"/session use {default_session_id}",
                    message_id="om_session_use",
                ),
            )
            follow_response = gateway.handle_event(
                route,
                _build_feishu_event("回到默认会话", message_id="om_default_002"),
            )
            default_session = load_session_history(
                default_session_id,
                base_dir=Path(temp_dir),
            )
            new_session = load_session_history(
                new_session_id,
                base_dir=Path(temp_dir),
            )

        self.assertEqual(switch_response.status_code, 200)
        self.assertEqual(json.loads(follow_response.body)["session_id"], default_session_id)
        self.assertEqual(default_session.summary.turn_count, 2)
        self.assertEqual(new_session.summary.turn_count, 1)
        self.assertEqual(
            [message.content for message in default_session.messages if isinstance(message, HumanMessage)],
            ["默认会话问题", "回到默认会话"],
        )
        self.assertEqual(
            [message.content for message in new_session.messages if isinstance(message, HumanMessage)],
            ["第二会话问题"],
        )

    def test_feishu_session_default_can_switch_back_to_default_session(self) -> None:
        """测试：飞书 /session default 可一键回到默认会话。"""
        with TemporaryDirectory() as temp_dir:
            gateway = WebhookGateway(
                [WebhookRouteConfig(provider="feishu", path="/webhook/feishu")],
                {"approval_policy": ApprovalPolicy.NEVER},
                lambda runtime_context: FakeWebhookRunner(),
                base_dir=Path(temp_dir),
            )
            route = WebhookRouteConfig(provider="feishu", path="/webhook/feishu")
            default_session_id = build_webhook_session_id("feishu", "oc_test_chat")

            gateway.handle_event(
                route,
                _build_feishu_event("默认会话问题", message_id="om_default_001"),
            )
            gateway.handle_event(
                route,
                _build_feishu_event("/session new 第二会话", message_id="om_session_new"),
            )
            gateway.handle_event(
                route,
                _build_feishu_event("第二会话问题", message_id="om_new_001"),
            )
            switch_response = gateway.handle_event(
                route,
                _build_feishu_event("/session default", message_id="om_session_default"),
            )
            follow_response = gateway.handle_event(
                route,
                _build_feishu_event("默认会话继续", message_id="om_default_002"),
            )
            default_session = load_session_history(
                default_session_id,
                base_dir=Path(temp_dir),
            )

        self.assertEqual(switch_response.status_code, 200)
        self.assertEqual(json.loads(follow_response.body)["session_id"], default_session_id)
        self.assertEqual(default_session.summary.turn_count, 2)
        self.assertEqual(
            [message.content for message in default_session.messages if isinstance(message, HumanMessage)],
            ["默认会话问题", "默认会话继续"],
        )

    def test_feishu_session_list_can_render_current_chat_sessions(self) -> None:
        """测试：飞书 /session list 会列出当前聊天下的可切换会话，而不是混入其他来源会话。"""
        with TemporaryDirectory() as temp_dir:
            gateway = WebhookGateway(
                [WebhookRouteConfig(provider="feishu", path="/webhook/feishu")],
                {"approval_policy": ApprovalPolicy.NEVER},
                lambda runtime_context: FakeWebhookRunner(),
                base_dir=Path(temp_dir),
            )
            route = WebhookRouteConfig(provider="feishu", path="/webhook/feishu")
            default_session_id = build_webhook_session_id("feishu", "oc_test_chat")

            gateway.handle_event(
                route,
                _build_feishu_event("默认会话问题", message_id="om_default_001"),
            )
            create_response = gateway.handle_event(
                route,
                _build_feishu_event("/session new 研发排查", message_id="om_session_new"),
            )
            new_session_id = str(json.loads(create_response.body)["session_id"])
            gateway.handle_event(
                route,
                _build_feishu_event("第二会话问题", message_id="om_new_001"),
            )
            list_response = gateway.handle_event(
                route,
                _build_feishu_event("/session list", message_id="om_session_list"),
            )
            list_payload = json.loads(list_response.body)
            reply_payload = dict(list_payload["reply_payload"])
            card_payload = json.loads(str(reply_payload["content"]))
            serialized_card = json.dumps(card_payload, ensure_ascii=False)

        self.assertEqual(list_response.status_code, 200)
        self.assertEqual(reply_payload["msg_type"], "interactive")
        self.assertIn("飞书会话列表", serialized_card)
        self.assertIn(default_session_id, serialized_card)
        self.assertIn(new_session_id, serialized_card)
        self.assertIn("/session use 1", serialized_card)
        self.assertIn("/session use <会话ID>", serialized_card)

    def test_feishu_fallback_builtin_command_can_strip_terminal_borders(self) -> None:
        """测试：暂未专门适配的飞书命令会清洗 Rich 边框后再放进兜底卡片。"""
        sent_requests: list[tuple[str, dict[str, object], dict[str, str] | None]] = []

        def fake_reply_sender(
            url: str,
            payload: dict[str, object],
            timeout_seconds: float,
            headers: dict[str, str] | None = None,
        ) -> WebhookDeliveryReceipt:
            _ = timeout_seconds
            sent_requests.append((url, payload, headers))
            if url.endswith("/auth/v3/tenant_access_token/internal"):
                return WebhookDeliveryReceipt(
                    status_code=200,
                    response_text=json.dumps(
                        {
                            "code": 0,
                            "tenant_access_token": "tenant-token",
                            "expire": 7200,
                        },
                        ensure_ascii=False,
                    ),
                )
            if url.endswith("/messages/om_custom/reply"):
                return WebhookDeliveryReceipt(
                    status_code=200,
                    response_text=json.dumps(
                        {
                            "code": 0,
                            "data": {
                                "message_id": "om_custom_reply",
                            },
                        },
                        ensure_ascii=False,
                    ),
                )
            raise AssertionError(f"未预期的请求地址: {url}")

        gateway = WebhookGateway(
            [
                WebhookRouteConfig(
                    provider="feishu",
                    path="/webhook/feishu",
                    provider_options={
                        "verification_token": "feishu-token",
                        "app_id": "cli_test_app",
                        "app_secret": "test-secret",
                        "reply_mode": "reply_api",
                    },
                )
            ],
            {"approval_policy": ApprovalPolicy.NEVER},
            lambda runtime_context: FakeWebhookRunner(),
            reply_sender=fake_reply_sender,
        )

        with patch(
            "cyber_agent.cli.webhook._capture_builtin_command_output_for_webhook",
            return_value=(
                True,
                (
                    "╭──────────── 历史会话 ────────────╮\n"
                    "│ session-001                     │\n"
                    "│ session-002                     │\n"
                    "╰──────────────────────────────────╯"
                ),
            ),
        ):
            response = gateway.handle_request(
                "POST",
                "/webhook/feishu",
                {"content-type": "application/json"},
                json.dumps(
                    {
                        "schema": "2.0",
                        "token": "feishu-token",
                        "type": "event_callback",
                        "event": {
                            "sender": {
                                "sender_id": {
                                    "open_id": "ou_feishu_user",
                                }
                            },
                            "message": {
                                "message_id": "om_custom",
                                "chat_id": "oc_test_chat",
                                "message_type": "text",
                                "content": json.dumps({"text": "/custom"}, ensure_ascii=False),
                            },
                        },
                    },
                    ensure_ascii=False,
                ).encode("utf-8"),
            )
            self.assertTrue(gateway.wait_until_async_idle())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(sent_requests), 2)
        self.assertEqual(sent_requests[1][1]["msg_type"], "interactive")
        card_payload = json.loads(str(sent_requests[1][1]["content"]))
        serialized_card = json.dumps(card_payload, ensure_ascii=False)
        self.assertEqual(card_payload["header"]["title"]["content"], "命令结果")
        self.assertIn("/custom", serialized_card)
        self.assertIn("session-001", serialized_card)
        self.assertIn("session-002", serialized_card)
        self.assertNotIn("╭", serialized_card)
        self.assertNotIn("╰", serialized_card)
        self.assertNotIn("────────", serialized_card)

    def test_feishu_reply_api_can_ack_before_background_processing(self) -> None:
        """测试：飞书 reply_api 模式会先快速确认，并在后台先推送开始状态。"""
        sent_requests: list[tuple[str, dict[str, object], dict[str, str] | None]] = []
        release_event = threading.Event()

        def fake_reply_sender(
            url: str,
            payload: dict[str, object],
            timeout_seconds: float,
            headers: dict[str, str] | None = None,
        ) -> WebhookDeliveryReceipt:
            _ = timeout_seconds
            sent_requests.append((url, payload, headers))
            if url.endswith("/auth/v3/tenant_access_token/internal"):
                return WebhookDeliveryReceipt(
                    status_code=200,
                    response_text=json.dumps(
                        {
                            "code": 0,
                            "tenant_access_token": "tenant-token",
                            "expire": 7200,
                        },
                        ensure_ascii=False,
                    ),
                )
            if url.endswith("/messages/om_async/reply"):
                return WebhookDeliveryReceipt(
                    status_code=200,
                    response_text=json.dumps(
                        {
                            "code": 0,
                            "data": {
                                "message_id": "om_async_reply",
                            },
                        },
                        ensure_ascii=False,
                    ),
                )
            raise AssertionError(f"未预期的请求地址: {url}")

        gateway = WebhookGateway(
            [
                WebhookRouteConfig(
                    provider="feishu",
                    path="/webhook/feishu",
                    provider_options={
                        "verification_token": "feishu-token",
                        "app_id": "cli_test_app",
                        "app_secret": "test-secret",
                        "reply_mode": "reply_api",
                    },
                )
            ],
            {"approval_policy": ApprovalPolicy.NEVER},
            lambda runtime_context: BlockingWebhookRunner(release_event),
            reply_sender=fake_reply_sender,
        )

        started_at = time.monotonic()
        response = gateway.handle_request(
            "POST",
            "/webhook/feishu",
            {"content-type": "application/json"},
            json.dumps(
                {
                    "schema": "2.0",
                    "token": "feishu-token",
                    "type": "event_callback",
                    "event": {
                        "sender": {
                            "sender_id": {
                                "open_id": "ou_feishu_user",
                            }
                        },
                        "message": {
                            "message_id": "om_async",
                            "chat_id": "oc_test_chat",
                            "message_type": "text",
                            "content": json.dumps({"text": "异步确认"}, ensure_ascii=False),
                        },
                    },
                },
                ensure_ascii=False,
            ).encode("utf-8"),
        )
        elapsed_seconds = time.monotonic() - started_at

        self.assertLess(elapsed_seconds, 0.5)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.body)["msg"], "success")

        deadline = time.monotonic() + 0.5
        while time.monotonic() <= deadline and len(sent_requests) < 2:
            time.sleep(0.01)
        self.assertGreaterEqual(len(sent_requests), 2)
        start_card = json.loads(str(sent_requests[1][1]["content"]))
        start_text = json.dumps(start_card, ensure_ascii=False)
        self.assertIn("处理中 · 已收到任务，开始处理", start_text)

        release_event.set()
        self.assertTrue(gateway.wait_until_async_idle())
        self.assertEqual(len(sent_requests), 3)
        self.assertTrue(sent_requests[2][0].endswith("/messages/om_async/reply"))

    def test_feishu_reply_api_reuses_cached_tenant_access_token(self) -> None:
        """测试：飞书官方回复模式会缓存 tenant_access_token，避免重复请求。"""
        sent_requests: list[tuple[str, dict[str, object], dict[str, str] | None]] = []

        def fake_reply_sender(
            url: str,
            payload: dict[str, object],
            timeout_seconds: float,
            headers: dict[str, str] | None = None,
        ) -> WebhookDeliveryReceipt:
            _ = timeout_seconds
            sent_requests.append((url, payload, headers))
            if url.endswith("/auth/v3/tenant_access_token/internal"):
                return WebhookDeliveryReceipt(
                    status_code=200,
                    response_text=json.dumps(
                        {
                            "code": 0,
                            "tenant_access_token": "tenant-token",
                            "expire": 7200,
                        },
                        ensure_ascii=False,
                    ),
                )
            if "/messages/" in url and url.endswith("/reply"):
                message_id = url.split("/messages/", 1)[1].split("/reply", 1)[0]
                return WebhookDeliveryReceipt(
                    status_code=200,
                    response_text=json.dumps(
                        {
                            "code": 0,
                            "data": {
                                "message_id": f"reply-for-{message_id}",
                            },
                        },
                        ensure_ascii=False,
                    ),
                )
            raise AssertionError(f"未预期的请求地址: {url}")

        with TemporaryDirectory() as temp_dir:
            gateway = WebhookGateway(
                [
                    WebhookRouteConfig(
                        provider="feishu",
                        path="/webhook/feishu",
                        provider_options={
                            "verification_token": "feishu-token",
                            "app_id": "cli_test_app",
                            "app_secret": "test-secret",
                            "reply_mode": "reply_api",
                        },
                    )
                ],
                {"approval_policy": ApprovalPolicy.NEVER},
                lambda runtime_context: FakeWebhookRunner(),
                base_dir=Path(temp_dir),
                reply_sender=fake_reply_sender,
            )

            for message_id, content in (("om_101", "第一条"), ("om_102", "第二条")):
                response = gateway.handle_request(
                    "POST",
                    "/webhook/feishu",
                    {"content-type": "application/json"},
                    json.dumps(
                        {
                            "schema": "2.0",
                            "token": "feishu-token",
                            "type": "event_callback",
                            "event": {
                                "sender": {
                                    "sender_id": {
                                        "open_id": "ou_feishu_user",
                                    }
                                },
                                "message": {
                                    "message_id": message_id,
                                    "chat_id": "oc_test_chat",
                                    "message_type": "text",
                                    "content": json.dumps({"text": content}, ensure_ascii=False),
                                },
                            },
                        },
                        ensure_ascii=False,
                    ).encode("utf-8"),
                )
                self.assertEqual(response.status_code, 200)
            self.assertTrue(gateway.wait_until_async_idle())

        auth_request_count = sum(
            1 for url, _payload, _headers in sent_requests if url.endswith("/auth/v3/tenant_access_token/internal")
        )
        reply_request_count = sum(
            1 for url, _payload, _headers in sent_requests if url.endswith("/reply")
        )
        self.assertEqual(auth_request_count, 1)
        self.assertEqual(reply_request_count, 4)

    def test_webhook_exit_command_can_clear_persisted_context(self) -> None:
        """测试：webhook 内的 /exit 会重置当前会话历史，便于移动端重新开始。"""
        with TemporaryDirectory() as temp_dir:
            gateway = WebhookGateway(
                [WebhookRouteConfig(provider="email", path="/webhook/email")],
                {"approval_policy": ApprovalPolicy.NEVER},
                lambda runtime_context: FakeWebhookRunner(),
                base_dir=Path(temp_dir),
            )
            route = WebhookRouteConfig(provider="email", path="/webhook/email")
            normal_event = WebhookEvent(
                provider="email",
                session_key="mail-session",
                sender_id="alice@example.com",
                sender_name="Alice",
                message_id="mail-001",
                text="普通消息",
            )
            exit_event = WebhookEvent(
                provider="email",
                session_key="mail-session",
                sender_id="alice@example.com",
                sender_name="Alice",
                message_id="mail-002",
                text="/exit",
            )

            gateway.handle_event(route, normal_event)
            exit_response = gateway.handle_event(route, exit_event)
            stored_session = load_session_history(
                build_webhook_session_id("email", "mail-session"),
                base_dir=Path(temp_dir),
            )

        self.assertEqual(exit_response.status_code, 200)
        self.assertEqual(len(stored_session.messages), 1)
        self.assertIsInstance(stored_session.messages[0], SystemMessage)

    def test_wecom_callback_get_can_verify_and_decrypt_echostr(self) -> None:
        """测试：企微官方 GET 校验会验签并返回解密后的 echostr。"""
        token = "wecom-token"
        encoding_aes_key = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFG"
        receive_id = "ww1234567890"
        gateway = WebhookGateway(
            [
                WebhookRouteConfig(
                    provider="wecom",
                    path="/webhook/wecom",
                    provider_options={
                        "token": token,
                        "encoding_aes_key": encoding_aes_key,
                        "receive_id": receive_id,
                    },
                )
            ],
            {"approval_policy": ApprovalPolicy.NEVER},
            lambda runtime_context: FakeWebhookRunner(),
        )
        echo_plaintext = "verify-ok"
        encrypted_echostr = _encrypt_wecom_plaintext(
            echo_plaintext,
            encoding_aes_key,
            receive_id,
        )
        signature = _build_wecom_signature(
            token,
            "1713600003",
            "nonce-003",
            encrypted_echostr,
        )

        response = gateway.handle_request(
            "GET",
            (
                "/webhook/wecom?"
                f"{WECOM_MESSAGE_SIGNATURE_QUERY_KEY}={signature}&"
                f"{WECOM_TIMESTAMP_QUERY_KEY}=1713600003&"
                f"{WECOM_NONCE_QUERY_KEY}=nonce-003&"
                f"{WECOM_ECHOSTR_QUERY_KEY}={quote_plus(encrypted_echostr)}"
            ),
            {},
            b"",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.body.decode("utf-8"), echo_plaintext)

    def test_wecom_xml_request_can_decrypt_and_passively_reply(self) -> None:
        """测试：企微官方 XML 回调可验签解密，并返回加密后的被动文本回复。"""
        token = "wecom-token"
        encoding_aes_key = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFG"
        receive_id = "ww1234567890"
        with TemporaryDirectory() as temp_dir:
            gateway = WebhookGateway(
                [
                    WebhookRouteConfig(
                        provider="wecom",
                        path="/webhook/wecom",
                        provider_options={
                            "token": token,
                            "encoding_aes_key": encoding_aes_key,
                            "receive_id": receive_id,
                            "reply_mode": "passive_xml",
                        },
                    )
                ],
                {"approval_policy": ApprovalPolicy.NEVER},
                lambda runtime_context: FakeWebhookRunner(),
                base_dir=Path(temp_dir),
            )
            request_xml, signature = _build_wecom_request_xml(
                (
                    "<xml>"
                    "<ToUserName>ww1234567890</ToUserName>"
                    "<FromUserName>zhangsan</FromUserName>"
                    "<CreateTime>1713600004</CreateTime>"
                    "<MsgType>text</MsgType>"
                    "<Content>企微官方消息</Content>"
                    "<MsgId>msg-001</MsgId>"
                    "<AgentID>1000002</AgentID>"
                    "</xml>"
                ),
                token=token,
                encoding_aes_key=encoding_aes_key,
                receive_id=receive_id,
                timestamp_value="1713600004",
                nonce_value="nonce-004",
            )

            response = gateway.handle_request(
                "POST",
                (
                    "/webhook/wecom?"
                    f"{WECOM_MESSAGE_SIGNATURE_QUERY_KEY}={signature}&"
                    f"{WECOM_TIMESTAMP_QUERY_KEY}=1713600004&"
                    f"{WECOM_NONCE_QUERY_KEY}=nonce-004"
                ),
                {"content-type": "text/xml"},
                request_xml.encode("utf-8"),
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "application/xml; charset=utf-8")
        response_xml = ElementTree.fromstring(response.body.decode("utf-8"))
        encrypted_reply = response_xml.findtext("Encrypt", default="")
        decrypted_reply = _decrypt_wecom_ciphertext(
            encrypted_reply,
            encoding_aes_key,
            expected_receive_id=receive_id,
        )
        self.assertIn("第1轮回复", decrypted_reply)
        self.assertIn("企微官方消息", decrypted_reply)

    def test_reply_webhook_can_sign_requests_and_write_dead_letter_after_retries(self) -> None:
        """测试：reply webhook 支持出站签名、失败重试与死信落盘。"""
        sent_attempts: list[tuple[str, dict[str, object], float, dict[str, str] | None]] = []

        def fake_reply_sender(
            url: str,
            payload: dict[str, object],
            timeout_seconds: float,
            headers: dict[str, str] | None = None,
        ) -> WebhookDeliveryReceipt:
            sent_attempts.append((url, payload, timeout_seconds, headers))
            raise RuntimeError("temporary downstream failure")

        with TemporaryDirectory() as temp_dir:
            gateway = WebhookGateway(
                [
                    WebhookRouteConfig(
                        provider="email",
                        path="/webhook/email",
                        reply_webhook_url="https://mail-bridge.example.com/reply?token=test",
                        secret="test-secret",
                        provider_options={
                            "reply_retry_attempts": "2",
                            "reply_retry_backoff_seconds": "0",
                            "reply_signing_secret": "signing-secret",
                            "reply_dead_letter_dir": "custom-dead-letters",
                        },
                    )
                ],
                {"approval_policy": ApprovalPolicy.NEVER},
                lambda runtime_context: FakeWebhookRunner(),
                base_dir=Path(temp_dir),
                reply_sender=fake_reply_sender,
            )
            response = gateway.handle_request(
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

            dead_letter_dir = Path(temp_dir) / "custom-dead-letters"
            dead_letter_files = list(dead_letter_dir.glob("*.json"))
            dead_letter_payload = json.loads(dead_letter_files[0].read_text(encoding="utf-8"))

        self.assertEqual(response.status_code, 502)
        self.assertEqual(len(sent_attempts), 2)
        self.assertIsNotNone(sent_attempts[0][3])
        self.assertIn("x-cyber-agent-timestamp", sent_attempts[0][3])
        self.assertIn("x-cyber-agent-signature", sent_attempts[0][3])
        self.assertEqual(len(dead_letter_files), 1)
        self.assertEqual(dead_letter_payload["provider"], "email")
        self.assertEqual(len(dead_letter_payload["attempts"]), 2)
        self.assertIn("token=***", dead_letter_payload["target_url"])


if __name__ == "__main__":
    unittest.main()
