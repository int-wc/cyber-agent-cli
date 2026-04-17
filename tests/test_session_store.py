import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from cyber_agent.session_store import (
    create_session_id,
    list_stored_sessions,
    load_session_history,
    save_session_history,
)


class SessionStoreTestCase(unittest.TestCase):
    def test_session_store_can_save_list_and_load_history(self) -> None:
        """
        测试：会话存储支持保存、枚举和读取历史消息。
        """
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            session_id = create_session_id()
            messages = [
                SystemMessage(content="system-prompt"),
                HumanMessage(content="hello history"),
                AIMessage(content="history response"),
            ]

            saved_path = save_session_history(
                session_id,
                messages,
                mode="standard",
                approval_policy="prompt",
                base_dir=base_dir,
            )
            summaries = list_stored_sessions(base_dir)
            loaded_session = load_session_history(session_id, base_dir=base_dir)

            self.assertTrue(saved_path.exists())
            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0].session_id, session_id)
            self.assertEqual(summaries[0].title, "hello history")
            self.assertEqual(summaries[0].turn_count, 1)
            self.assertEqual(loaded_session.summary.session_id, session_id)
            self.assertEqual(len(loaded_session.messages), 3)
            self.assertEqual(loaded_session.messages[1].content, "hello history")

    def test_session_store_preserves_source_session_id(self) -> None:
        """
        测试：由历史会话派生的新会话会保留来源会话标识。
        """
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            session_id = create_session_id()
            save_session_history(
                session_id,
                [
                    SystemMessage(content="system"),
                    HumanMessage(content="derived session"),
                ],
                mode="authorized",
                approval_policy="auto",
                source_session_id="source-001",
                base_dir=base_dir,
            )
            loaded_session = load_session_history(session_id, base_dir=base_dir)

        self.assertEqual(loaded_session.summary.source_session_id, "source-001")
        self.assertEqual(loaded_session.summary.mode, "authorized")
        self.assertEqual(loaded_session.summary.approval_policy, "auto")


if __name__ == "__main__":
    unittest.main()
