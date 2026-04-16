import unittest

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from cyber_agent.cli.interactive import (
    InteractionUiMode,
    build_command_hint_lines,
    get_auto_completion,
    get_banner_command_summary,
    match_builtin_commands,
    parse_interaction_ui_mode,
)
from cyber_agent.cli.prompting import BuiltinCommandCompleter


class CliInteractiveHelperTestCase(unittest.TestCase):
    def test_parse_interaction_ui_mode_can_recognize_tui_and_cli(self) -> None:
        """
        测试：界面模式字符串可解析为统一枚举，供 CLI 和 TUI 入口共用。
        """

        self.assertIs(parse_interaction_ui_mode("tui"), InteractionUiMode.TUI)
        self.assertIs(parse_interaction_ui_mode("cli"), InteractionUiMode.CLI)

    def test_prompt_completer_uses_same_builtin_command_registry(self) -> None:
        """
        测试：CLI 补全器与共享命令表保持一致，不额外维护第二份补全列表。
        """

        completer = BuiltinCommandCompleter()
        completions = [
            completion.text
            for completion in completer.get_completions(
                Document("/approval"),
                CompleteEvent(completion_requested=True),
            )
        ]

        self.assertIn("/approval", completions)
        self.assertIn("/approval prompt", completions)
        self.assertIn("/approval auto", completions)
        self.assertIn("/approval never", completions)

    def test_get_auto_completion_can_complete_short_builtin_command(self) -> None:
        """
        测试：输入内建命令前缀时，可返回首个可接受的自动补全结果。
        """

        self.assertEqual(get_auto_completion("/sta"), "/status")
        self.assertEqual(get_auto_completion("/mode a"), "/mode authorized")

    def test_match_builtin_commands_can_return_related_command_group(self) -> None:
        """
        测试：命令匹配会返回同一前缀下的相关命令，便于提醒和补全共用。
        """

        matches = match_builtin_commands("/approval")
        commands = [item.command for item in matches]

        self.assertIn("/approval", commands)
        self.assertIn("/approval prompt", commands)
        self.assertIn("/approval auto", commands)
        self.assertIn("/approval never", commands)

    def test_build_command_hint_lines_can_fallback_to_default_command_list(self) -> None:
        """
        测试：普通输入场景下，会回退展示默认命令提醒，而不是空白区域。
        """

        hint_lines = build_command_hint_lines("扫描一下端口", limit=4)

        self.assertEqual(len(hint_lines), 4)
        self.assertTrue(any("/help" in line for line in hint_lines))
        self.assertTrue(any("/tools" in line for line in hint_lines))

    def test_banner_command_summary_is_based_on_shared_command_registry(self) -> None:
        """
        测试：欢迎区快捷命令摘要来自统一命令配置，避免与帮助和补全脱节。
        """

        summary = get_banner_command_summary()

        self.assertIn("/help", summary)
        self.assertIn("/status", summary)
        self.assertIn("/approval", summary)


if __name__ == "__main__":
    unittest.main()
