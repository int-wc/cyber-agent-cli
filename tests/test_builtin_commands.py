"""命令注册表的单元测试。"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from cyber_agent.agent.mode import AgentMode
from cyber_agent.cli.builtin_commands import (
    _COMMAND_REGISTRY,
    dispatch_builtin_command,
)


class BuiltinCommandRegistryTestCase(unittest.TestCase):
    """验证命令注册表分发和退出命令。"""

    def setUp(self) -> None:
        self.runner = MagicMock()
        self.runner.mode = AgentMode.STANDARD
        self.ctx: dict[str, object] = {"approval_policy": "prompt"}

    def test_registry_has_all_expected_commands(self) -> None:
        """确保所有交互模式命令都已注册。"""
        expected_commands = {
            "/stop", "/help", "/tools", "/context",
            "/history", "/doctor", "/status", "/version",
            "/config", "/service", "/model", "/allow-path",
            "/clear", "/memory", "/mode", "/approval", "/multi",
            "/auto-decision", "/file", "/session", "/capabilities",
            "/new", "/trace", "/pipeline",
        }
        self.assertTrue(expected_commands.issubset(set(_COMMAND_REGISTRY.keys())))

    def test_exit_commands_return_false(self) -> None:
        """退出命令返回 False（停止交互循环）。"""
        for cmd in ("quit", "exit", "/quit", "/exit"):
            result = dispatch_builtin_command(cmd, self.runner, self.ctx)
            self.assertFalse(result, f"{cmd} 应停止循环")

    def test_unknown_input_returns_none(self) -> None:
        """非内建命令返回 None。"""
        result = dispatch_builtin_command("普通用户消息", self.runner, self.ctx)
        self.assertIsNone(result)

    def test_empty_input_returns_none(self) -> None:
        """空输入返回 None。"""
        result = dispatch_builtin_command("", self.runner, self.ctx)
        self.assertIsNone(result)

    def test_help_returns_true(self) -> None:
        """/help 命令返回 True（继续循环）。"""
        with patch("cyber_agent.cli.app.print_help"):
            result = dispatch_builtin_command("/help", self.runner, self.ctx)
            self.assertTrue(result)

    def test_clear_starts_fresh_visible_session(self) -> None:
        """/clear 同时清上下文、最近输入和可见会话窗口。"""
        renderer = MagicMock()
        self.ctx["_recent_inputs"] = ["旧输入"]
        clear_checkpoint = MagicMock()

        with patch(
            "cyber_agent.cli.app._load_session_store_support",
            return_value={"clear_interrupt_checkpoint": clear_checkpoint},
        ):
            result = dispatch_builtin_command(
                "/clear",
                self.runner,
                self.ctx,
                renderer,
            )

        self.assertTrue(result)
        self.runner.reset.assert_called_once()
        renderer.clear_screen.assert_called_once()
        renderer.print_info.assert_called_once()
        self.assertEqual(self.ctx["_recent_inputs"], [])
        self.assertTrue(self.ctx["__clear_visible_session"])
        self.assertIn("session_id", self.ctx)
        clear_checkpoint.assert_called_once_with()

    def test_session_new_starts_fresh_visible_session(self) -> None:
        """/session new 的可见聊天窗口也应像新会话一样清空。"""
        renderer = MagicMock()
        self.ctx["_recent_inputs"] = ["旧输入"]
        clear_checkpoint = MagicMock()

        with patch(
            "cyber_agent.cli.app._load_session_store_support",
            return_value={"clear_interrupt_checkpoint": clear_checkpoint},
        ):
            result = dispatch_builtin_command(
                "/session new",
                self.runner,
                self.ctx,
                renderer,
            )

        self.assertTrue(result)
        self.runner.reset.assert_called_once()
        renderer.clear_screen.assert_called_once()
        renderer.print_info.assert_called_once()
        self.assertEqual(self.ctx["_recent_inputs"], [])
        self.assertTrue(self.ctx["__clear_visible_session"])
        self.assertIn("session_id", self.ctx)
        clear_checkpoint.assert_called_once_with()

    def test_new_alias_starts_fresh_visible_session(self) -> None:
        """/new 是 /session new 的短别名，不应进入模型。"""
        renderer = MagicMock()
        self.ctx["_recent_inputs"] = ["旧输入"]
        clear_checkpoint = MagicMock()

        with patch(
            "cyber_agent.cli.app._load_session_store_support",
            return_value={"clear_interrupt_checkpoint": clear_checkpoint},
        ):
            result = dispatch_builtin_command(
                "/new",
                self.runner,
                self.ctx,
                renderer,
            )

        self.assertTrue(result)
        self.runner.reset.assert_called_once()
        renderer.clear_screen.assert_called_once()
        renderer.print_info.assert_called_once()
        self.assertEqual(self.ctx["_recent_inputs"], [])
        self.assertTrue(self.ctx["__clear_visible_session"])
        self.assertIn("session_id", self.ctx)
        clear_checkpoint.assert_called_once_with()

    def test_unknown_command_prefix_returns_none(self) -> None:
        """未知 / 前缀命令返回 None（非内建命令）。"""
        result = dispatch_builtin_command("/unknown_command", self.runner, self.ctx)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
