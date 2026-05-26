"""共享工具调用执行器的单元测试。"""
from __future__ import annotations

import unittest

from cyber_agent.agent.tool_executor import invoke_tool_simple
from cyber_agent.tools.security import scan_port


class ToolExecutorTestCase(unittest.TestCase):
    """验证共享工具调用：参数规范化、异常处理、未知工具。"""

    def test_invoke_known_tool_succeeds(self) -> None:
        """已知工具调用应返回 ToolMessage 且内容包含正常结果。"""
        from langchain_core.messages import ToolMessage

        registry = {scan_port.name: scan_port}
        tool_call = {"name": "scan_port", "args": {"host": "127.0.0.1", "port": 22, "timeout": 1}, "id": "call_1"}

        result = invoke_tool_simple(tool_call, registry)
        self.assertIsInstance(result, ToolMessage)
        self.assertEqual(result.name, "scan_port")
        self.assertEqual(result.tool_call_id, "call_1")

    def test_invoke_unknown_tool_returns_error(self) -> None:
        """未知工具返回错误标记的 ToolMessage。"""
        from langchain_core.messages import ToolMessage

        registry = {"scan_port": scan_port}
        tool_call = {"name": "nonexistent_tool", "args": {}, "id": "call_2"}

        result = invoke_tool_simple(tool_call, registry)
        self.assertIsInstance(result, ToolMessage)
        self.assertIn("未知工具", result.content)

    def test_invoke_with_bad_args_returns_error(self) -> None:
        """参数无效时返回错误标记而非崩溃。"""
        registry = {scan_port.name: scan_port}
        tool_call = {"name": "scan_port", "args": "not-json", "id": "call_3"}

        result = invoke_tool_simple(tool_call, registry)
        self.assertIn("工具参数", result.content)

    def test_invoke_with_string_args_is_normalized(self) -> None:
        """JSON 字符串参数应被规范化为字典。"""
        import json
        from langchain_core.messages import ToolMessage

        registry = {scan_port.name: scan_port}
        tool_call = {
            "name": "scan_port",
            "args": json.dumps({"host": "127.0.0.1", "port": 22, "timeout": 1}),
            "id": "call_4",
        }

        result = invoke_tool_simple(tool_call, registry)
        self.assertIsInstance(result, ToolMessage)


if __name__ == "__main__":
    unittest.main()
