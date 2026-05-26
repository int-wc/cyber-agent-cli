"""跨会话记忆系统的完整测试。"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cyber_agent.memory import (
    MemoryEntry,
    MemorySearchResult,
    _build_memory_excerpt,
    _parse_frontmatter,
    _update_memory_index,
    build_memory_system_prompt,
    delete_memory,
    get_memory_dir,
    load_all_memories,
    save_memory,
    search_memories,
)


class MemoryFrontmatterTestCase(unittest.TestCase):
    """测试 frontmatter 解析和构建。"""

    def test_parse_frontmatter_extracts_fields(self) -> None:
        """正确提取 frontmatter 字段和正文。"""
        content = (
            "---\n"
            "name: test-entry\n"
            "description: 测试条目\n"
            "metadata:\n"
            "  type: user_preference\n"
            "---\n"
            "\n"
            "这是正文内容。\n"
        )
        metadata, body = _parse_frontmatter(content)
        self.assertEqual(metadata.get("name"), "test-entry")
        self.assertEqual(metadata.get("description"), "测试条目")
        self.assertIn("这是正文内容", body)

    def test_parse_no_frontmatter_returns_empty(self) -> None:
        """无 frontmatter 时返回空字典且 body 为原文。"""
        content = "纯正文，无元数据。"
        metadata, body = _parse_frontmatter(content)
        self.assertEqual(metadata, {})
        self.assertEqual(body, content)

    def test_build_excerpt_centers_on_query(self) -> None:
        """摘要围绕查询关键词生成。"""
        text = "这是一段很长的文本，" + "中间部分" + "包含关键词，然后继续延伸。"
        query = "中间部分"
        excerpt = _build_memory_excerpt(text * 5, query, max_chars=50)
        self.assertIn(query, excerpt)

    def test_build_excerpt_no_match_returns_truncated(self) -> None:
        """不匹配时返回截断文本。"""
        text = "长文本" * 50
        excerpt = _build_memory_excerpt(text, "不存在")
        self.assertTrue(excerpt.endswith("...") or len(excerpt) <= 200)


class MemorySaveLoadTestCase(unittest.TestCase):
    """测试记忆保存、读取、删除。"""

    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.base_dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_save_and_load_single_memory(self) -> None:
        """保存一条记忆后能读取到。"""
        save_memory(
            "test-mem",
            "测试记忆描述",
            "这是正文。\n多行内容。",
            memory_type="project_context",
            base_dir=self.base_dir,
        )
        entries = load_all_memories(self.base_dir)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].name, "test-mem")
        self.assertEqual(entries[0].memory_type, "project_context")
        self.assertIn("这是正文", entries[0].body)

    def test_save_multiple_types(self) -> None:
        """不同类型记忆正确分组。"""
        save_memory("pref-1", "偏好1", "body1", memory_type="user_preference", base_dir=self.base_dir)
        save_memory("proj-1", "项目1", "body2", memory_type="project_context", base_dir=self.base_dir)
        save_memory("dec-1", "决策1", "body3", memory_type="decision", base_dir=self.base_dir)
        entries = load_all_memories(self.base_dir)
        types = {e.memory_type for e in entries}
        self.assertEqual(len(entries), 3)
        self.assertIn("user_preference", types)
        self.assertIn("project_context", types)
        self.assertIn("decision", types)

    def test_index_file_is_generated(self) -> None:
        """保存后自动生成 MEMORY.md 索引。"""
        save_memory("idx-test", "索引测试", "body", base_dir=self.base_dir)
        index_path = get_memory_dir(self.base_dir) / "MEMORY.md"
        self.assertTrue(index_path.exists())
        content = index_path.read_text(encoding="utf-8")
        self.assertIn("idx-test", content)
        self.assertIn("索引测试", content)

    def test_delete_memory_removes_file_and_updates_index(self) -> None:
        """删除记忆后文件和索引均更新。"""
        save_memory("del-me", "待删除", "body", base_dir=self.base_dir)
        self.assertTrue(delete_memory("del-me", base_dir=self.base_dir))
        entries = load_all_memories(self.base_dir)
        self.assertEqual(len(entries), 0)

    def test_delete_nonexistent_returns_false(self) -> None:
        """删除不存在的记忆返回 False。"""
        self.assertFalse(delete_memory("no-such", base_dir=self.base_dir))

    def test_load_empty_dir_returns_empty(self) -> None:
        """无记忆的目录返回空列表。"""
        entries = load_all_memories(self.base_dir)
        self.assertEqual(entries, [])


class MemorySearchTestCase(unittest.TestCase):
    """测试记忆检索。"""

    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.base_dir = Path(self.tmp.name)
        save_memory("mem-a", "AAA项目", "讨论了 AAA 项目的架构设计", base_dir=self.base_dir)
        save_memory("mem-b", "BBB配置", "配置了 BBB 环境变量", base_dir=self.base_dir)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_search_finds_relevant_entries(self) -> None:
        """按关键词检索返回匹配结果。"""
        results = search_memories("AAA", base_dir=self.base_dir)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].entry.name, "mem-a")

    def test_search_no_match_returns_empty(self) -> None:
        """无匹配返回空列表。"""
        results = search_memories("ZZZ", base_dir=self.base_dir)
        self.assertEqual(results, [])

    def test_search_empty_query_returns_empty(self) -> None:
        """空查询返回空。"""
        results = search_memories("  ", base_dir=self.base_dir)
        self.assertEqual(results, [])


class MemorySystemPromptTestCase(unittest.TestCase):
    """测试记忆系统提示词构建。"""

    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.base_dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_no_memories_returns_empty_prompt(self) -> None:
        """无记忆时返回空字符串。"""
        prompt = build_memory_system_prompt(self.base_dir)
        self.assertEqual(prompt, "")

    def test_with_memories_includes_entries(self) -> None:
        """有记忆时提示词包含条目信息。"""
        save_memory("test", "测试条目", "这条记忆的正文。", base_dir=self.base_dir)
        prompt = build_memory_system_prompt(self.base_dir)
        self.assertIn("测试条目", prompt)
        self.assertIn("持久化记忆", prompt)

    def test_prompt_groups_by_type(self) -> None:
        """不同类型分组显示。"""
        save_memory("p1", "偏好", "body", memory_type="user_preference", base_dir=self.base_dir)
        save_memory("d1", "决策", "body", memory_type="decision", base_dir=self.base_dir)
        prompt = build_memory_system_prompt(self.base_dir)
        self.assertIn("用户偏好", prompt)
        self.assertIn("历史决策", prompt)


if __name__ == "__main__":
    unittest.main()
