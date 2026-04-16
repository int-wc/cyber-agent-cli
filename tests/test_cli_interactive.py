import unittest
from pathlib import Path

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from rich.cells import cell_len
from rich.console import Console

from cyber_agent.agent.approval import ApprovalPolicy
from cyber_agent.agent.mode import AgentMode
from cyber_agent.cli.branding import (
    STARTUP_PANEL_TITLE,
    STARTUP_SUBTITLE,
    STARTUP_TITLE,
    build_startup_renderable,
)
from cyber_agent.cli.interactive import (
    InteractionUiMode,
    build_command_hint_lines,
    get_auto_completion,
    get_banner_command_summary,
    match_builtin_commands,
    parse_interaction_ui_mode,
)
from cyber_agent.cli.prompting import (
    PROMPT_STYLE,
    PROMPT_TOOLKIT_IMPORT_ERROR,
    BuiltinCommandCompleter,
)
from cyber_agent.cli.render import CliRenderer
from cyber_agent.cli.theme import USER_TEXT_COLOR


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

    def test_startup_renderable_contains_ascii_title_and_subtitle(self) -> None:
        """
        测试：启动页区块会输出统一的项目标题与副标题，供 CLI 与 TUI 共用。
        """

        console = Console(record=True, width=120)
        console.print(build_startup_renderable())

        output = console.export_text()
        self.assertIn(STARTUP_PANEL_TITLE, output)
        self.assertIn(STARTUP_TITLE, output)
        self.assertIn(STARTUP_SUBTITLE, output)
        self.assertNotIn("启动完成", output)

    def test_startup_renderable_centers_title_lines_by_display_width(self) -> None:
        """
        测试：启动页标题区按终端显示宽度居中，避免中文双宽字符导致视觉偏移。
        """

        console = Console(record=True, width=120)
        console.print(build_startup_renderable())

        output_lines = console.export_text().splitlines()
        for target in (STARTUP_TITLE, STARTUP_SUBTITLE):
            target_line = next(line for line in output_lines if target in line)
            left_border = target_line.index("│")
            right_border = target_line.rindex("│")
            inner = target_line[left_border + 1 : right_border]
            left_padding = len(inner) - len(inner.lstrip(" "))
            right_padding = len(inner) - len(inner.rstrip(" "))
            content = inner.strip(" ")
            available_width = cell_len(inner)
            content_width = cell_len(content)

            self.assertEqual(content, target)
            self.assertLessEqual(abs(left_padding - right_padding), 1)
            self.assertEqual(left_padding + right_padding + content_width, available_width)

    def test_prompt_style_sets_distinct_color_for_user_typed_text(self) -> None:
        """
        测试：CLI 输入提示器会为用户正在输入的正文设置独立颜色，避免与提示文案混淆。
        """

        self.assertIsNone(PROMPT_TOOLKIT_IMPORT_ERROR)
        self.assertIn(("", f"bold {USER_TEXT_COLOR}"), PROMPT_STYLE.style_rules)

    def test_cli_renderer_prints_startup_splash_before_system_banner(self) -> None:
        """
        测试：CLI 启动时先打印独立启动页，再继续输出系统提示面板。
        """

        console = Console(record=True, width=120)
        renderer = CliRenderer(console=console)
        renderer.print_startup_splash()
        renderer.print_banner(
            mode=AgentMode.STANDARD,
            service="openai",
            model="gpt-5.4",
            cwd=Path("D:/cyber-agent-cli"),
            approval_policy=ApprovalPolicy.PROMPT,
        )

        output = console.export_text()
        self.assertLess(output.index(STARTUP_PANEL_TITLE), output.index("系统提示"))
        self.assertIn(STARTUP_TITLE, output)
        self.assertIn("Cyber Agent CLI 交互界面", output)

    def test_cli_banner_adds_visual_gap_between_title_and_overview(self) -> None:
        """
        测试：欢迎面板标题下方保留一行留白，避免与后续会话概览信息过于贴近。
        """

        console = Console(record=True, width=100)
        CliRenderer(console=console).print_banner(
            mode=AgentMode.STANDARD,
            service="openai",
            model="gpt-5.4",
            cwd=Path("D:/cyber-agent-cli"),
            approval_policy=ApprovalPolicy.PROMPT,
        )

        output_lines = console.export_text().splitlines()
        title_index = next(
            index
            for index, line in enumerate(output_lines)
            if "Cyber Agent CLI 交互界面" in line
        )

        self.assertEqual(output_lines[title_index + 1].strip(" │"), "")
        self.assertIn("当前模式", output_lines[title_index + 2])


if __name__ == "__main__":
    unittest.main()
