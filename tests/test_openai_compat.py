import unittest

from langchain_core.messages import AIMessage, AIMessageChunk

from cyber_agent.openai_compat import (
    ensure_deepseek_reasoning_content_compat,
    prepare_messages_for_openai_compatible_service,
)


class OpenAICompatTestCase(unittest.TestCase):
    def test_deepseek_reasoning_content_is_preserved_in_langchain_converters(self) -> None:
        """测试：DeepSeek thinking 模式返回的 reasoning_content 会被保留并回传。"""
        ensure_deepseek_reasoning_content_compat()
        from langchain_openai.chat_models.base import (
            _convert_delta_to_message_chunk,
            _convert_message_to_dict,
        )

        chunk = _convert_delta_to_message_chunk(
            {
                "role": "assistant",
                "reasoning_content": "需要先调用工具。",
            },
            AIMessageChunk,
        )
        self.assertIsInstance(chunk, AIMessageChunk)
        self.assertEqual(
            chunk.additional_kwargs["reasoning_content"],
            "需要先调用工具。",
        )

        message_payload = _convert_message_to_dict(
            AIMessage(
                content="",
                additional_kwargs={"reasoning_content": "需要先调用工具。"},
                tool_calls=[
                    {
                        "id": "call_lookup",
                        "name": "lookup",
                        "args": {},
                    }
                ],
            )
        )

        self.assertEqual(message_payload["reasoning_content"], "需要先调用工具。")
        self.assertIn("tool_calls", message_payload)

    def test_prepare_messages_adds_missing_reasoning_content_for_deepseek_tool_calls(self) -> None:
        """测试：旧历史中的 DeepSeek 工具调用消息会补齐 reasoning_content 字段。"""
        message = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_lookup",
                    "name": "lookup",
                    "args": {},
                }
            ],
        )

        prepared_messages = prepare_messages_for_openai_compatible_service(
            [message],
            "deepseek",
            deepseek_thinking_enabled=True,
        )

        self.assertEqual(
            prepared_messages[0].additional_kwargs["reasoning_content"],
            "",
        )

    def test_prepare_messages_strips_reasoning_content_when_deepseek_thinking_disabled(self) -> None:
        """测试：DeepSeek thinking 关闭时不回传 reasoning_content，避免工具链触发 400。"""
        message = AIMessage(
            content="",
            additional_kwargs={"reasoning_content": "需要先调用工具。"},
            tool_calls=[
                {
                    "id": "call_lookup",
                    "name": "lookup",
                    "args": {},
                }
            ],
        )

        prepared_messages = prepare_messages_for_openai_compatible_service(
            [message],
            "deepseek",
            deepseek_thinking_enabled=False,
        )

        self.assertNotIn("reasoning_content", prepared_messages[0].additional_kwargs)

    def test_prepare_messages_strips_reasoning_content_for_openai(self) -> None:
        """测试：切回 OpenAI 时移除 DeepSeek 专属字段，避免非 DeepSeek 接口拒绝。"""
        message = AIMessage(
            content="回答",
            additional_kwargs={"reasoning_content": "内部思考"},
        )

        prepared_messages = prepare_messages_for_openai_compatible_service(
            [message],
            "openai",
        )

        self.assertNotIn("reasoning_content", prepared_messages[0].additional_kwargs)


if __name__ == "__main__":
    unittest.main()
