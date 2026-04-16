import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from typer.testing import CliRunner

from cyber_agent.cli.app import app


class FakeChatOpenAI:
    """
    用于 CLI 命令测试的假模型。
    这些用例不会真正触发模型调用，只需要保证初始化可完成。
    """

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def bind_tools(self, tools, **kwargs):
        return self

    def stream(self, messages):
        raise AssertionError("当前用例不应触发真实对话调用。")


class CliBuiltinCommandTestCase(unittest.TestCase):
    def test_chat_loop_supports_builtin_commands(self) -> None:
        """
        测试：交互模式支持 /tools、/status、/mode、/allow-path、/approval 等基础命令。
        """
        cli_runner = CliRunner()
        with TemporaryDirectory() as temp_dir:
            allowed_dir = Path(temp_dir) / "allowed root"
            allowed_dir.mkdir()

            with patch("cyber_agent.agent.runner.ChatOpenAI", FakeChatOpenAI):
                result = cli_runner.invoke(
                    app,
                    [],
                    input=(
                        "/tools\n"
                        "/status\n"
                        "/config\n"
                        "/mode authorized\n"
                        f"/allow-path add {allowed_dir}\n"
                        "/allow-path\n"
                        "/approval auto\n"
                        "/approval\n"
                        "/clear\n"
                        "quit\n"
                    ),
                )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("默认工具", result.output)
        self.assertIn("write_text_file", result.output)
        self.assertIn("replace_in_file", result.output)
        self.assertIn("apply_unified_patch", result.output)
        self.assertIn("run_shell_command", result.output)
        self.assertIn("当前状态", result.output)
        self.assertIn("本地配置文件", result.output)
        self.assertIn("已切换到 授权模式", result.output)
        self.assertIn("已添加允许访问目录", result.output)
        self.assertIn(str(allowed_dir), result.output)
        self.assertIn("允许访问目录", result.output)
        self.assertIn("已切换到 自动批准", result.output)
        self.assertIn("当前审批策略：自动批准", result.output)
        self.assertIn("会话上下文已清空", result.output)
        self.assertIn("再见", result.output)

    def test_root_options_can_start_in_authorized_mode_with_auto_approval(self) -> None:
        """
        测试：根命令支持通过 --mode 和 --approval-policy 指定运行策略。
        """
        cli_runner = CliRunner()

        with patch("cyber_agent.agent.runner.ChatOpenAI", FakeChatOpenAI):
            result = cli_runner.invoke(
                app,
                ["--mode", "authorized", "--approval-policy", "auto"],
                input="quit\n",
            )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("当前模式：authorized", result.output)
        self.assertIn("审批策略：auto", result.output)

    def test_ui_cli_mode_will_not_launch_tui(self) -> None:
        """
        测试：显式指定 --ui cli 时，即使存在 TUI 入口也应保持命令行交互。
        """
        cli_runner = CliRunner()

        with patch("cyber_agent.agent.runner.ChatOpenAI", FakeChatOpenAI):
            with patch(
                "cyber_agent.cli.tui.launch_textual_chat",
                side_effect=AssertionError("不应启动 TUI"),
            ):
                result = cli_runner.invoke(
                    app,
                    ["--ui", "cli"],
                    input="quit\n",
                )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("启动页面", result.output)
        self.assertIn("Cyber Agent CLI", result.output)
        self.assertIn("再见", result.output)

    def test_ui_tui_mode_can_launch_tui_entry(self) -> None:
        """
        测试：显式指定 --ui tui 时，应优先进入 TUI 入口。
        """
        cli_runner = CliRunner()

        with patch("cyber_agent.agent.runner.ChatOpenAI", FakeChatOpenAI):
            with patch("cyber_agent.cli.tui.launch_textual_chat") as mock_launch:
                result = cli_runner.invoke(app, ["--ui", "tui"])

        self.assertEqual(result.exit_code, 0)
        mock_launch.assert_called_once()

    def test_ui_gui_mode_is_rejected(self) -> None:
        """
        测试：GUI 已移除后，旧的 --ui gui 参数应直接报错。
        """
        cli_runner = CliRunner()

        result = cli_runner.invoke(app, ["--ui", "gui"], input="quit\n")

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("不支持的界面模式", result.output)
        self.assertIn("auto, tui, cli", result.output)

    def test_doctor_can_show_extra_allowed_path_and_registered_tool(self) -> None:
        """
        测试：授权模式下可在状态里看到额外允许路径和已注册外部工具。
        """
        cli_runner = CliRunner()
        with TemporaryDirectory() as temp_dir:
            tool_spec = f"python={Path(sys.executable).resolve()}"
            with patch("cyber_agent.agent.runner.ChatOpenAI", FakeChatOpenAI):
                result = cli_runner.invoke(
                    app,
                    [
                        "--mode",
                        "authorized",
                        "--approval-policy",
                        "auto",
                        "--allow-path",
                        temp_dir,
                        "--tool",
                        tool_spec,
                        "doctor",
                    ],
                )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("允许读取根路径", result.output)
        self.assertIn(temp_dir, result.output)
        self.assertIn("已注册外部工具", result.output)
        self.assertIn("python=", result.output)
        self.assertIn("审批策略", result.output)

    def test_config_command_can_persist_allow_path_for_future_runs(self) -> None:
        """
        测试：/config allow-path add 可将目录写入本地配置，并在后续运行中自动加载。
        """
        cli_runner = CliRunner()
        with cli_runner.isolated_filesystem():
            working_directory = Path.cwd()
            allowed_dir = working_directory / "persisted root"
            allowed_dir.mkdir()
            local_config_path = working_directory / ".cyber-agent-cli.json"

            with patch("cyber_agent.agent.runner.ChatOpenAI", FakeChatOpenAI):
                first_result = cli_runner.invoke(
                    app,
                    [],
                    input=(
                        f"/config allow-path add {allowed_dir}\n"
                        "/config\n"
                        "quit\n"
                    ),
                )
                second_result = cli_runner.invoke(
                    app,
                    ["--mode", "authorized", "doctor"],
                )

            self.assertEqual(first_result.exit_code, 0)
            self.assertIn("已写入本地配置", first_result.output)
            self.assertIn("切换到授权模式后会自动生效", first_result.output)
            self.assertIn("本地配置文件", first_result.output)
            self.assertIn(str(allowed_dir), first_result.output)
            self.assertTrue(local_config_path.exists())
            saved_config = json.loads(local_config_path.read_text(encoding="utf-8"))
            self.assertIn(str(allowed_dir), saved_config["allow_paths"])

            self.assertEqual(second_result.exit_code, 0)
            self.assertIn("本地配置文件", second_result.output)
            self.assertIn("已保存允许目录", second_result.output)
            self.assertIn(str(allowed_dir), second_result.output)
            self.assertIn("允许读取根路径", second_result.output)


if __name__ == "__main__":
    unittest.main()
