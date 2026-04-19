import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from cyber_agent.session_store import (
    create_session_id,
    export_session_history,
    list_stored_sessions,
    load_session_history,
    save_session_history,
    search_stored_sessions,
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

    def test_session_store_can_search_keyword_across_saved_sessions(self) -> None:
        """
        测试：历史会话检索可以返回命中的会话摘要与片段。
        """
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            save_session_history(
                "session-001",
                [
                    SystemMessage(content="system"),
                    HumanMessage(content="请检查 openai agent 的搜索链路"),
                    AIMessage(content="已记录搜索链路。"),
                ],
                mode="standard",
                approval_policy="prompt",
                base_dir=base_dir,
            )
            save_session_history(
                "session-002",
                [
                    SystemMessage(content="system"),
                    HumanMessage(content="这个会话不会命中"),
                ],
                mode="authorized",
                approval_policy="auto",
                base_dir=base_dir,
            )

            search_results = search_stored_sessions("openai agent", base_dir=base_dir)

        self.assertEqual(len(search_results), 1)
        self.assertEqual(search_results[0].session_id, "session-001")
        self.assertEqual(search_results[0].matched_message_count, 1)
        self.assertTrue(any("openai agent" in excerpt for excerpt in search_results[0].excerpts))

    def test_session_store_can_export_markdown_and_json(self) -> None:
        """
        测试：历史会话支持导出为 Markdown 和 JSON，便于人工排查与脚本处理。
        """
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            save_session_history(
                "session-export",
                [
                    SystemMessage(content="system"),
                    HumanMessage(content="导出这段历史"),
                    AIMessage(content="已准备导出内容。"),
                ],
                mode="standard",
                approval_policy="prompt",
                base_dir=base_dir,
            )

            markdown_path = export_session_history("session-export", base_dir=base_dir)
            json_path = export_session_history(
                "session-export",
                output_path=base_dir / "exports" / "session-export.json",
                base_dir=base_dir,
            )

            self.assertTrue(markdown_path.exists())
            self.assertTrue(json_path.exists())
            markdown_text = markdown_path.read_text(encoding="utf-8")
            json_text = json_path.read_text(encoding="utf-8")

        self.assertIn("历史会话导出", markdown_text)
        self.assertIn("导出这段历史", markdown_text)
        self.assertIn('"session_id": "session-export"', json_text)

    def test_session_store_search_skips_invalid_session_files(self) -> None:
        """
        测试：历史检索遇到损坏或缺少消息数组的会话文件时，应跳过而不是中断全部结果。
        """
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            save_session_history(
                "session-valid",
                [
                    SystemMessage(content="system"),
                    HumanMessage(content="需要检索的有效历史"),
                ],
                mode="standard",
                approval_policy="prompt",
                base_dir=base_dir,
            )
            storage_dir = base_dir / ".cyber-agent-cli-sessions"
            (storage_dir / "session-broken.json").write_text("{ not-json", encoding="utf-8")
            (storage_dir / "session-missing-messages.json").write_text(
                '{"session_id":"session-missing-messages","title":"损坏记录"}',
                encoding="utf-8",
            )

            search_results = search_stored_sessions("有效历史", base_dir=base_dir)

        self.assertEqual(len(search_results), 1)
        self.assertEqual(search_results[0].session_id, "session-valid")


if __name__ == "__main__":
    unittest.main()
