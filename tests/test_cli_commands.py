import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typer.testing import CliRunner

from cyber_agent.cli.app import app
from cyber_agent.config import settings
from cyber_agent.session_store import save_session_history


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
        测试：交互模式支持 /tools、/context、/history、/mode、/allow-path、/approval 等基础命令。
        """
        cli_runner = CliRunner()
        with cli_runner.isolated_filesystem():
            allowed_dir = Path.cwd() / "allowed root"
            allowed_dir.mkdir()

            with patch("cyber_agent.agent.runner.ChatOpenAI", FakeChatOpenAI):
                result = cli_runner.invoke(
                    app,
                    [],
                    input=(
                        "/tools\n"
                        "/context\n"
                        "/history\n"
                        "/stop\n"
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
        self.assertIn("search_web", result.output)
        self.assertIn("create_generated_capability", result.output)
        self.assertIn("revise_generated_capability", result.output)
        self.assertIn("list_generated_capabilities", result.output)
        self.assertIn("show_generated_capability", result.output)
        self.assertIn("mark_generated_capability_satisfied", result.output)
        self.assertIn("当前会话 ID", result.output)
        self.assertIn("当前工作目录下还没有已保存的历史会话", result.output)
        self.assertIn("当前没有正在执行的任务", result.output)
        self.assertIn("当前状态", result.output)
        self.assertIn("本地配置文件", result.output)
        self.assertIn("已切换到 授权模式", result.output)
        self.assertIn("已添加允许访问目录", result.output)
        self.assertIn(str(allowed_dir), result.output)
        self.assertIn("允许访问目录", result.output)
        self.assertIn("已切换到 自动批准", result.output)
        self.assertIn("当前审批策略：自动批准", result.output)
        self.assertIn("会话上下文已清空，并已开始新的会话", result.output)
        self.assertIn("再见", result.output)

    def test_history_commands_can_show_and_load_stored_session(self) -> None:
        """
        测试：/history show 和 /history load 可以访问并恢复已保存历史会话。
        """
        cli_runner = CliRunner()
        with cli_runner.isolated_filesystem():
            session_id = "history-001"
            save_session_history(
                session_id,
                [
                    SystemMessage(content="system prompt"),
                    HumanMessage(content="saved human message"),
                    AIMessage(content="saved ai response"),
                ],
                mode="authorized",
                approval_policy="auto",
                source_session_id="source-root",
            )

            with patch("cyber_agent.agent.runner.ChatOpenAI", FakeChatOpenAI):
                result = cli_runner.invoke(
                    app,
                    [],
                    input=(
                        "/history\n"
                        f"/history show {session_id}\n"
                        f"/history load {session_id}\n"
                        "/context\n"
                        "quit\n"
                    ),
                )

        self.assertEqual(result.exit_code, 0)
        self.assertIn(session_id, result.output)
        self.assertIn("saved human message", result.output)
        self.assertIn("已加载历史会话", result.output)
        self.assertIn("后续继续对话时会保存为新的会话副本", result.output)
        self.assertIn("来源会话", result.output)
        self.assertIn("source-root", result.output)

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

    def test_builtin_commands_can_switch_service_and_model_in_current_session(self) -> None:
        """
        测试：/service 与 /model 可在当前会话中切换服务商和模型，并体现在状态输出中。
        """
        cli_runner = CliRunner()

        with patch("cyber_agent.agent.runner.ChatOpenAI", FakeChatOpenAI):
            result = cli_runner.invoke(
                app,
                [],
                input=(
                    "/service deepseek\n"
                    "/model deepseek-chat\n"
                    "/service\n"
                    "/status\n"
                    "quit\n"
                ),
            )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("已切换当前会话服务商：deepseek", result.output)
        self.assertIn("已切换当前会话模型：deepseek / deepseek-chat", result.output)
        self.assertIn("当前服务", result.output)
        self.assertIn("当前模型", result.output)
        self.assertIn("deepseek-chat", result.output)
        self.assertIn("https://api.deepseek.com/v1", result.output)

    def test_runtime_context_uses_configured_service_and_model_at_startup(self) -> None:
        """
        测试：启动时应优先使用当前配置中的服务商与模型，而不是固定展示 openai。
        """
        cli_runner = CliRunner()

        with (
            patch.object(settings, "service_name", "deepseek"),
            patch.object(settings, "openai_model", "deepseek-chat"),
            patch.object(settings, "openai_base_url", None),
            patch("cyber_agent.agent.runner.ChatOpenAI", FakeChatOpenAI),
        ):
            result = cli_runner.invoke(
                app,
                [],
                input="quit\n",
            )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("模型服务：deepseek", result.output)
        self.assertIn("模型名称：deepseek-chat", result.output)

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
            self.assertIn(str(working_directory), second_result.output)
            self.assertIn("persisted", second_result.output)
            self.assertIn("root", second_result.output)
            self.assertIn("允许读取根路径", second_result.output)


if __name__ == "__main__":
    unittest.main()
