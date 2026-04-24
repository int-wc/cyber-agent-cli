import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typer.testing import CliRunner

from cyber_agent import __version__
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
                        "/doctor\n"
                        "/version\n"
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
        self.assertIn("运行诊断", result.output)
        self.assertIn("项目版本", result.output)
        self.assertIn(f"cyber-agent-cli {__version__}", result.output)
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

    def test_history_search_and_export_commands_can_help_debug_long_sessions(self) -> None:
        """
        测试：/history search 和 /history export 可用于定位历史内容并导出排查材料。
        """
        cli_runner = CliRunner()
        with cli_runner.isolated_filesystem():
            session_id = "history-search-001"
            export_path = Path.cwd() / "exports" / "history search result.md"
            save_session_history(
                session_id,
                [
                    SystemMessage(content="system prompt"),
                    HumanMessage(content="请排查 search_web 的回退路径"),
                    AIMessage(content="已定位到 HTTP fallback 行为。"),
                ],
                mode="authorized",
                approval_policy="auto",
            )

            with patch("cyber_agent.agent.runner.ChatOpenAI", FakeChatOpenAI):
                result = cli_runner.invoke(
                    app,
                    [],
                    input=(
                        "/history search fallback\n"
                        f"/history export {session_id} {export_path}\n"
                        "quit\n"
                    ),
                )

            self.assertTrue(export_path.exists())
            exported_markdown = export_path.read_text(encoding="utf-8")

        self.assertEqual(result.exit_code, 0)
        self.assertIn("历史检索: fallback", result.output)
        self.assertIn(session_id, result.output)
        self.assertIn("HTTP fallback", result.output)
        self.assertIn("已导出历史会话", result.output)
        self.assertIn("历史会话导出", exported_markdown)
        self.assertIn("search_web", exported_markdown)

    def test_top_level_history_commands_can_list_show_search_and_export(self) -> None:
        """
        测试：顶层 history 命令组支持脚本化列出、查看、检索和导出历史会话。
        """
        cli_runner = CliRunner()
        with cli_runner.isolated_filesystem():
            session_id = "history-cli-001"
            export_path = Path.cwd() / "exports" / "history-cli-001.json"
            save_session_history(
                session_id,
                [
                    SystemMessage(content="system prompt"),
                    HumanMessage(content="请排查 doctor json 输出"),
                    AIMessage(content="history cli export ready"),
                ],
                mode="standard",
                approval_policy="prompt",
            )

            list_result = cli_runner.invoke(app, ["history"])
            show_result = cli_runner.invoke(app, ["history", "show", session_id])
            search_result = cli_runner.invoke(app, ["history", "search", "json"])
            export_result = cli_runner.invoke(
                app,
                ["history", "export", session_id, str(export_path)],
            )

            self.assertTrue(export_path.exists())
            exported_json = json.loads(export_path.read_text(encoding="utf-8"))

        self.assertEqual(list_result.exit_code, 0)
        self.assertIn("历史会话", list_result.output)
        self.assertIn(session_id, list_result.output)

        self.assertEqual(show_result.exit_code, 0)
        self.assertIn("会话 ID", show_result.output)
        self.assertIn("doctor json 输出", show_result.output)

        self.assertEqual(search_result.exit_code, 0)
        self.assertIn("历史检索: json", search_result.output)
        self.assertIn(session_id, search_result.output)

        self.assertEqual(export_result.exit_code, 0)
        self.assertIn("已导出历史会话", export_result.output)
        self.assertEqual(exported_json["session_id"], session_id)
        self.assertEqual(exported_json["mode"], "standard")

    def test_context_command_can_render_ai_tool_calls_without_plain_text(self) -> None:
        """
        测试：/context 在遇到仅包含 tool_calls 的 AI 消息时，不应因上下文预览渲染而崩溃。
        """
        cli_runner = CliRunner()
        with cli_runner.isolated_filesystem():
            session_id = "history-tool-call"
            save_session_history(
                session_id,
                [
                    SystemMessage(content="system prompt"),
                    HumanMessage(content="saved human message"),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "echo_tool",
                                "args": {"text": "saved human message"},
                                "id": "call_echo_tool",
                                "type": "tool_call",
                            }
                        ],
                    ),
                ],
                mode="authorized",
                approval_policy="auto",
            )

            with patch("cyber_agent.agent.runner.ChatOpenAI", FakeChatOpenAI):
                result = cli_runner.invoke(
                    app,
                    [],
                    input=(
                        f"/history load {session_id}\n"
                        "/context\n"
                        "quit\n"
                    ),
                )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("echo_tool", result.output)
        self.assertIn("tool_call", result.output)

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
                    "/service deepseek https://example.test/v1\n"
                    "/model deepseek-chat\n"
                    "/service\n"
                    "/status\n"
                    "quit\n"
                ),
            )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("已忽略 /service 中的基址参数", result.output)
        self.assertIn("已切换当前会话服务商：deepseek", result.output)
        self.assertIn("已切换当前会话模型：deepseek / deepseek-chat", result.output)
        self.assertIn("当前服务", result.output)
        self.assertIn("当前模型", result.output)
        self.assertIn("deepseek-chat", result.output)
        self.assertIn("http://localhost:8317/", result.output)

    def test_runtime_context_uses_configured_service_and_model_at_startup(self) -> None:
        """
        测试：启动时应优先使用当前配置中的服务商与模型，而不是固定展示 openai。
        """
        cli_runner = CliRunner()

        with (
            patch.object(settings, "service_name", "deepseek"),
            patch.object(settings, "deepseek_model", "deepseek-chat"),
            patch.object(settings, "openai_base_url", None),
            patch.object(settings, "deepseek_base_url", None),
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
        测试：doctor 会展示依赖、版本、额外允许路径和已注册外部工具。
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
        self.assertIn("运行诊断", result.output)
        self.assertIn("项目版本", result.output)
        self.assertIn("Python", result.output)
        self.assertIn("prompt_toolkit", result.output)
        self.assertIn("textual", result.output)
        self.assertIn("playwright", result.output)
        self.assertIn("浏览器搜索", result.output)
        self.assertIn("允许读取根路径", result.output)
        self.assertIn(temp_dir, result.output)
        self.assertIn("已注册外部工具", result.output)
        self.assertIn("python=", result.output)
        self.assertIn("审批策略", result.output)

    def test_doctor_can_output_json_for_scripts_and_ci(self) -> None:
        """
        测试：doctor --json 会输出稳定的结构化诊断结果。
        """
        cli_runner = CliRunner()

        with patch("cyber_agent.agent.runner.ChatOpenAI", FakeChatOpenAI):
            result = cli_runner.invoke(app, ["doctor", "--json"])

        self.assertEqual(result.exit_code, 0)
        payload = json.loads(result.output)
        self.assertEqual(payload["project"]["version"], __version__)
        self.assertIn(payload["summary"]["status"], {"ok", "warning"})
        self.assertIn("dependencies", payload)
        self.assertIn("runtime", payload)
        self.assertIn("permissions", payload)
        self.assertIn("capabilities", payload)

    def test_doctor_still_works_without_langchain_openai_dependency(self) -> None:
        """
        测试：缺少 langchain_openai 时，doctor 仍可输出诊断结果而不是在初始化阶段崩溃。
        """
        cli_runner = CliRunner()
        import_error = ModuleNotFoundError("No module named 'langchain_openai'")

        with (
            patch("cyber_agent.agent.runner.ChatOpenAI", None),
            patch("cyber_agent.agent.runner.LANGCHAIN_OPENAI_IMPORT_ERROR", import_error),
            patch("cyber_agent.capability_registry.ChatOpenAI", None),
            patch("cyber_agent.capability_registry.LANGCHAIN_OPENAI_IMPORT_ERROR", import_error),
        ):
            result = cli_runner.invoke(app, ["doctor"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("运行诊断", result.output)
        self.assertIn("langchain_openai", result.output)

    def test_webhook_example_config_command_can_output_sample_routes(self) -> None:
        """
        测试：webhook example-config 会输出可直接复制修改的 JSON 示例。
        """
        cli_runner = CliRunner()

        result = cli_runner.invoke(app, ["webhook", "example-config"])

        self.assertEqual(result.exit_code, 0)
        payload = json.loads(result.output)
        self.assertIn("providers", payload)
        self.assertEqual(len(payload["providers"]), 4)
        self.assertEqual(
            set(payload["providers"]),
            {"feishu", "dingtalk", "wecom", "email"},
        )

    def test_webhook_feishu_long_connection_command_can_start_with_config(self) -> None:
        """
        测试：webhook serve-feishu-long-connection 会加载配置并调用飞书长连接入口。
        """
        cli_runner = CliRunner()

        with cli_runner.isolated_filesystem():
            config_path = Path("webhook-routes.json")
            config_path.write_text(
                json.dumps(
                    {
                        "providers": {
                            "feishu": {
                                "path": "/webhook/feishu",
                                "provider_options": {
                                    "app_id": "cli_test_app",
                                    "app_secret": "test-secret",
                                },
                            }
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with patch("cyber_agent.cli.app._load_feishu_long_connection_support") as mock_load:
                mock_serve = Mock()
                mock_load.return_value = (
                    lambda routes, route_path=None: routes[0],
                    mock_serve,
                )
                result = cli_runner.invoke(
                    app,
                    [
                        "webhook",
                        "serve-feishu-long-connection",
                        "--config",
                        str(config_path),
                    ],
                )

        self.assertEqual(result.exit_code, 0)
        mock_serve.assert_called_once()
        resolved_route = mock_serve.call_args.args[0]
        self.assertEqual(resolved_route.provider, "feishu")
        self.assertEqual(resolved_route.path, "/webhook/feishu")

    def test_run_command_reports_missing_langchain_openai_dependency_cleanly(self) -> None:
        """
        测试：单次 run 命令在缺少 langchain_openai 时返回明确错误，而不是直接抛出堆栈。
        """
        cli_runner = CliRunner()
        import_error = ModuleNotFoundError("No module named 'langchain_openai'")

        with (
            patch("cyber_agent.agent.runner.ChatOpenAI", None),
            patch("cyber_agent.agent.runner.LANGCHAIN_OPENAI_IMPORT_ERROR", import_error),
            patch("cyber_agent.capability_registry.ChatOpenAI", None),
            patch("cyber_agent.capability_registry.LANGCHAIN_OPENAI_IMPORT_ERROR", import_error),
        ):
            result = cli_runner.invoke(app, ["run", "hello"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("运行失败", result.output)
        self.assertIn("langchain_openai", result.output)

    def test_version_command_and_root_option_can_show_current_version(self) -> None:
        """
        测试：顶层 version 子命令和 --version 选项都能输出当前 CLI 版本。
        """
        cli_runner = CliRunner()

        version_command_result = cli_runner.invoke(app, ["version"])
        version_option_result = cli_runner.invoke(app, ["--version"])

        self.assertEqual(version_command_result.exit_code, 0)
        self.assertEqual(version_option_result.exit_code, 0)
        self.assertEqual(version_command_result.output.strip(), f"cyber-agent-cli {__version__}")
        self.assertEqual(version_option_result.output.strip(), f"cyber-agent-cli {__version__}")

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
