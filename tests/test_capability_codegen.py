"""capability 代码生成模块的单元测试。"""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cyber_agent.capability_codegen import (
    CapabilityArtifacts,
    _strip_markdown_code_fence,
    build_capability_source,
    default_skill_python_code,
    default_tool_python_code,
    validate_capability_source,
)


class StripMarkdownFenceTestCase(unittest.TestCase):
    """测试 Markdown 代码栅栏剥离。"""

    def test_strip_json_fence(self) -> None:
        """剥离 ```json ... ``` 包裹。"""
        result = _strip_markdown_code_fence('```json\n{"key": "value"}\n```')
        self.assertEqual(result, '{"key": "value"}')

    def test_strip_python_fence(self) -> None:
        """剥离 ```python ... ``` 包裹。"""
        result = _strip_markdown_code_fence('```python\nprint("hello")\n```')
        self.assertEqual(result, 'print("hello")')

    def test_no_fence_passes_through(self) -> None:
        """无栅栏的文本原样返回。"""
        result = _strip_markdown_code_fence("plain text")
        self.assertEqual(result, "plain text")


class DefaultCodeTestCase(unittest.TestCase):
    """测试默认骨架代码生成。"""

    def test_default_tool_code_defines_handle_request(self) -> None:
        """默认工具骨架包含 handle_request 函数。"""
        code = default_tool_python_code()
        self.assertIn("def handle_request", code)
        self.assertIn("TODO(人工实现)", code)

    def test_default_skill_code_defines_build_skill_prompt(self) -> None:
        """默认技能骨架包含 build_skill_prompt 函数。"""
        code = default_skill_python_code()
        self.assertIn("def build_skill_prompt", code)
        self.assertIn("TODO(人工实现)", code)


class BuildSourceTestCase(unittest.TestCase):
    """测试源码构建。"""

    def test_build_tool_source(self) -> None:
        """构建工具型 capability 源码。"""
        source = build_capability_source(
            name="test_tool",
            kind="tool",
            description="测试工具",
            register_as_tool=True,
            tool_python_code="def handle_request(request, context):\n    return 'ok'\n",
            skill_python_code="",
        )
        self.assertIn("CAPABILITY_NAME", source)
        self.assertIn("test_tool", source)
        self.assertIn("def handle_request", source)
        self.assertIn("def build_skill_prompt", source)  # TODO(人工实现)骨架
        self.assertIn("def _main()", source)

    def test_build_skill_source(self) -> None:
        """构建技能型 capability 源码。"""
        source = build_capability_source(
            name="test_skill",
            kind="skill",
            description="测试技能",
            register_as_tool=False,
            tool_python_code="",
            skill_python_code="def build_skill_prompt():\n    return '提示词'\n",
        )
        self.assertIn("test_skill", source)
        self.assertIn("def build_skill_prompt", source)
        self.assertIn("CAPABILITY_KIND: Final[str] = 'skill'", source)

    def test_empty_tool_code_gets_default_skeleton(self) -> None:
        """无工具代码时回退到默认骨架。"""
        source = build_capability_source(
            name="empty_tool",
            kind="tool",
            description="空工具",
            register_as_tool=True,
            tool_python_code="",
            skill_python_code="",
        )
        self.assertIn("TODO(人工实现)", source)


class ValidateSourceTestCase(unittest.TestCase):
    """测试源码校验。"""

    def test_valid_tool_source_passes(self) -> None:
        """包含正确函数的工具源码通过校验。"""
        source = build_capability_source(
            name="v",
            kind="tool",
            description="d",
            register_as_tool=True,
            tool_python_code="def handle_request(r, c):\n    return 'ok'\n",
            skill_python_code="",
        )
        issues = validate_capability_source(source, requires_tool=True, requires_skill=False)
        self.assertEqual(issues, [])

    def test_missing_handle_request_is_reported(self) -> None:
        """缺少 handle_request 被检测到。"""
        source = build_capability_source(
            name="v",
            kind="tool",
            description="d",
            register_as_tool=True,
            tool_python_code="x = 1\n",
            skill_python_code="",
        )
        issues = validate_capability_source(source, requires_tool=True, requires_skill=False)
        self.assertTrue(any("handle_request" in i for i in issues))

    def test_syntax_error_is_reported(self) -> None:
        """语法错误被检测到。"""
        issues = validate_capability_source(
            "def broken(:\n    pass\n",  # 语法错误
            requires_tool=False,
            requires_skill=False,
        )
        self.assertTrue(any("语法错误" in i for i in issues))


if __name__ == "__main__":
    unittest.main()
