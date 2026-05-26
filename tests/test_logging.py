"""结构化日志模块的基础验证。"""
from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from cyber_agent.logging import (
    _resolve_log_enabled,
    _resolve_log_level,
    log_context_compression,
    log_error,
    log_info,
    log_model_call,
    log_tool_execution,
    log_warning,
)


class LoggingModuleTestCase(unittest.TestCase):
    """测试日志模块核心功能 — 级别解析、便捷函数、JSON 格式。"""

    def test_resolve_log_level_defaults_to_info(self) -> None:
        """无环境变量时默认 INFO 级别。"""
        self.assertEqual(_resolve_log_level(), 20)  # INFO

    def test_resolve_log_level_from_env(self) -> None:
        """从环境变量正确解析日志级别。"""
        with patch.dict("os.environ", {"CYBER_LOG_LEVEL": "DEBUG"}):
            self.assertEqual(_resolve_log_level(), 10)  # DEBUG

    def test_resolve_log_level_invalid_falls_back_to_info(self) -> None:
        """无效级别回落 INFO。"""
        with patch.dict("os.environ", {"CYBER_LOG_LEVEL": "TRACE"}):
            self.assertEqual(_resolve_log_level(), 20)  # INFO

    def test_resolve_log_enabled_defaults_to_true(self) -> None:
        """默认启用日志。"""
        self.assertTrue(_resolve_log_enabled())

    def test_resolve_log_enabled_from_env(self) -> None:
        """环境变量控制开关。"""
        with patch.dict("os.environ", {"CYBER_LOG_ENABLED": "false"}):
            self.assertFalse(_resolve_log_enabled())

    def test_log_model_call_does_not_raise(self) -> None:
        """便捷函数调用不抛出异常。"""
        log_model_call("deepseek", "deepseek-v4-pro", char_count=100, success=True)

    def test_log_tool_execution_does_not_raise(self) -> None:
        """工具执行日志调用不抛出异常。"""
        log_tool_execution("scan_port", success=True, result_len=42)
        log_tool_execution("bad_tool", success=False, error="connection refused")

    def test_log_context_compression_does_not_raise(self) -> None:
        """上下文压缩日志不抛出异常。"""
        log_context_compression(before_chars=20000, after_chars=5000, compressed_count=3)

    def test_log_error_and_warning(self) -> None:
        """通用错误/警告日志不抛出异常。"""
        log_error("test_module", "测试错误")
        log_warning("test_module", "测试警告")
        log_info("test_module", "测试信息")


class LoggingFormatterTestCase(unittest.TestCase):
    """测试 JSON 行格式输出。"""

    def test_structured_formatter_outputs_valid_json(self) -> None:
        """格式化输出为合法 JSON 对象。"""
        from cyber_agent.logging import _StructuredFormatter
        import logging

        formatter = _StructuredFormatter()
        record = logging.LogRecord(
            name="cyber-agent",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="测试消息",
            args=None,
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        self.assertEqual(parsed["msg"], "测试消息")
        self.assertEqual(parsed["level"], "INFO")


if __name__ == "__main__":
    unittest.main()
