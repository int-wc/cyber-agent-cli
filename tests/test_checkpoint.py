"""中断续传快照功能的完整测试。"""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from cyber_agent.session_store import (
    clear_interrupt_checkpoint,
    has_interrupt_checkpoint,
    load_interrupt_checkpoint,
    save_interrupt_checkpoint,
)


class CheckpointSaveLoadTestCase(unittest.TestCase):
    """测试中断快照的保存、加载、清除。"""

    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.base_dir = Path(self.tmp.name)
        self.messages = [
            SystemMessage(content="系统提示词"),
            HumanMessage(content="用户问题：测试"),
            AIMessage(content="助手回答：已处理"),
        ]

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_save_and_load_checkpoint(self) -> None:
        """保存快照后能成功加载。"""
        save_interrupt_checkpoint(
            "test-session-001",
            self.messages,
            mode="authorized",
            approval_policy="auto",
            base_dir=self.base_dir,
        )
        self.assertTrue(has_interrupt_checkpoint(self.base_dir))

        checkpoint = load_interrupt_checkpoint(self.base_dir)
        self.assertIsNotNone(checkpoint)
        assert checkpoint is not None
        self.assertEqual(checkpoint["session_id"], "test-session-001")
        self.assertEqual(checkpoint["mode"], "authorized")
        self.assertEqual(checkpoint["approval_policy"], "auto")
        self.assertEqual(checkpoint["turn_count"], 1)
        self.assertEqual(checkpoint["message_count"], 3)
        self.assertIn("interrupted_at", checkpoint)

    def test_load_no_checkpoint_returns_none(self) -> None:
        """无快照时返回 None。"""
        self.assertIsNone(load_interrupt_checkpoint(self.base_dir))

    def test_has_checkpoint_detects_presence(self) -> None:
        """正确检测快照存在性。"""
        self.assertFalse(has_interrupt_checkpoint(self.base_dir))
        save_interrupt_checkpoint("s1", self.messages, mode="standard", approval_policy="prompt", base_dir=self.base_dir)
        self.assertTrue(has_interrupt_checkpoint(self.base_dir))

    def test_clear_removes_checkpoint(self) -> None:
        """清除后快照不再存在。"""
        save_interrupt_checkpoint("s1", self.messages, mode="standard", approval_policy="prompt", base_dir=self.base_dir)
        clear_interrupt_checkpoint(self.base_dir)
        self.assertFalse(has_interrupt_checkpoint(self.base_dir))

    def test_clear_nonexistent_does_not_error(self) -> None:
        """清除不存在的快照不抛出异常。"""
        clear_interrupt_checkpoint(self.base_dir)  # no-op, no error

    def test_load_corrupt_checkpoint_returns_none(self) -> None:
        """损坏的快照文件返回 None。"""
        from cyber_agent.session_store import get_session_storage_dir
        storage_dir = get_session_storage_dir(self.base_dir)
        storage_dir.mkdir(parents=True, exist_ok=True)
        (storage_dir / "_checkpoint.json").write_text("not json", encoding="utf-8")
        self.assertIsNone(load_interrupt_checkpoint(self.base_dir))

    def test_checkpoint_messages_are_deserializable(self) -> None:
        """快照中的消息可以反序列化为 LangChain 消息。"""
        save_interrupt_checkpoint("s1", self.messages, mode="standard", approval_policy="prompt", base_dir=self.base_dir)
        checkpoint = load_interrupt_checkpoint(self.base_dir)
        assert checkpoint is not None
        from langchain_core.messages import messages_from_dict
        msgs = messages_from_dict(checkpoint["messages"])
        self.assertEqual(len(msgs), 3)
        self.assertIsInstance(msgs[0], SystemMessage)
        self.assertIsInstance(msgs[1], HumanMessage)
        self.assertIsInstance(msgs[2], AIMessage)


if __name__ == "__main__":
    unittest.main()
