"""默认工具注册表测试：注册/替换/禁用/构建。"""

from __future__ import annotations

import unittest

from langchain_core.tools import tool as lc_tool

from cyber_agent.tools.registry import ToolRegistry


class ToolRegistryTestCase(unittest.TestCase):
    """注册表核心行为。"""

    def setUp(self) -> None:
        self.registry = ToolRegistry()
        self.built: list[str] = []

        @lc_tool("tool_a")
        def tool_a() -> str:
            """工具A。"""
            return "a"

        @lc_tool("tool_b")
        def tool_b() -> str:
            """工具B。"""
            return "b"

        self.tool_a = tool_a
        self.tool_b = tool_b

    def test_register_and_build(self) -> None:
        self.registry.register("tool_a", lambda **ctx: self.tool_a)
        self.registry.register("tool_b", lambda **ctx: self.tool_b)
        tools = self.registry.build()
        self.assertEqual({t.name for t in tools}, {"tool_a", "tool_b"})

    def test_duplicate_register_rejected(self) -> None:
        self.registry.register("tool_a", lambda **ctx: self.tool_a)
        with self.assertRaises(ValueError):
            self.registry.register("tool_a", lambda **ctx: self.tool_a)

    def test_replace_overwrites(self) -> None:
        self.registry.register("tool_a", lambda **ctx: self.tool_a)
        self.registry.replace("tool_a", lambda **ctx: self.tool_b)
        tools = self.registry.build()
        self.assertEqual(len(tools), 1)
        self.assertIs(tools[0], self.tool_b)

    def test_disable_and_enable(self) -> None:
        self.registry.register("tool_a", lambda **ctx: self.tool_a)
        self.registry.register("tool_b", lambda **ctx: self.tool_b)
        self.registry.disable("tool_a")
        self.assertEqual({t.name for t in self.registry.build()}, {"tool_b"})
        self.assertFalse(self.registry.is_enabled("tool_a"))
        self.registry.enable("tool_a")
        self.assertEqual({t.name for t in self.registry.build()}, {"tool_a", "tool_b"})

    def test_unregister_removes(self) -> None:
        self.registry.register("tool_a", lambda **ctx: self.tool_a)
        self.registry.unregister("tool_a")
        self.assertFalse(self.registry.is_registered("tool_a"))
        self.assertEqual(self.registry.build(), [])

    def test_factory_error_skips_tool(self) -> None:
        """单个 factory 抛错不中断整体构建。"""

        def bad_factory(**ctx):
            raise RuntimeError("构建失败")

        self.registry.register("bad_tool", bad_factory)
        self.registry.register("tool_a", lambda **ctx: self.tool_a)
        tools = self.registry.build()
        self.assertEqual({t.name for t in tools}, {"tool_a"})

    def test_context_passed_to_factory(self) -> None:
        seen: dict = {}

        def factory(**ctx):
            seen.update(ctx)
            return self.tool_a

        self.registry.register("tool_a", factory)
        self.registry.build(allowed_roots=["/tmp"], mode="standard")
        self.assertEqual(seen.get("allowed_roots"), ["/tmp"])
        self.assertEqual(seen.get("mode"), "standard")


class DefaultToolsRegistryTestCase(unittest.TestCase):
    """get_default_tools 从注册表构建 + 可替换。"""

    def test_get_default_tools_uses_registry(self) -> None:
        from cyber_agent.agent.mode import AgentMode
        from cyber_agent.tools import get_default_tools

        tools = get_default_tools(mode=AgentMode.STANDARD)
        names = {t.name for t in tools}
        for expect in (
            "scan_port",
            "search_web",
            "list_directory",
            "read_text_file",
            "write_text_file",
            "run_shell_command",
        ):
            self.assertIn(expect, names)

    def test_get_default_tools_accepts_custom_registry(self) -> None:
        from cyber_agent.agent.mode import AgentMode
        from cyber_agent.tools import get_default_tools

        registry = ToolRegistry()

        @lc_tool("custom_only")
        def custom_only() -> str:
            """仅自定义工具。"""
            return "custom"

        registry.register("custom_only", lambda **ctx: custom_only)
        tools = get_default_tools(mode=AgentMode.STANDARD, registry=registry)
        self.assertEqual({t.name for t in tools}, {"custom_only"})


if __name__ == "__main__":
    unittest.main()
