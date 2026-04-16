import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cyber_agent.tools.filesystem import (
    create_read_text_file_tool,
    create_replace_in_file_tool,
    create_write_text_file_tool,
)
from cyber_agent.tools.patching import create_apply_unified_patch_tool
from cyber_agent.tools.system import (
    create_run_registered_tool_tool,
    create_run_shell_command_tool,
)


class AuthorizedModeToolTestCase(unittest.TestCase):
    def test_read_text_file_can_access_extra_allowed_path(self) -> None:
        """
        测试：额外声明的允许路径可以被读取。
        """
        with TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir) / "sample.txt"
            temp_file.write_text("authorized-content", encoding="utf-8")
            read_text_file = create_read_text_file_tool([Path.cwd(), temp_dir])

            result = read_text_file.invoke({"path": str(temp_file)})

        self.assertIn("authorized-content", result)
        self.assertIn(str(temp_file), result)

    def test_read_text_file_rejects_path_outside_allowed_roots(self) -> None:
        """
        测试：未声明的路径不允许读取。
        """
        with TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir) / "blocked.txt"
            temp_file.write_text("blocked-content", encoding="utf-8")
            read_text_file = create_read_text_file_tool([Path.cwd()])

            result = read_text_file.invoke({"path": str(temp_file)})

        self.assertIn("超出允许访问范围", result)

    def test_run_registered_tool_can_invoke_registered_executable(self) -> None:
        """
        测试：已注册外部工具可以被调用，并继承当前进程权限执行。
        """
        run_registered_tool = create_run_registered_tool_tool(
            {"python": Path(sys.executable).resolve()}
        )

        result = run_registered_tool.invoke(
            {
                "tool_name": "python",
                "arguments": ["-c", "print('registered-tool-ok')"],
            }
        )

        self.assertIn("registered-tool-ok", result)
        self.assertIn("继承当前 CLI 进程权限", result)

    def test_write_and_replace_tools_can_modify_allowed_file(self) -> None:
        """
        测试：写文件和文本替换工具可以在允许范围内修改文件。
        """
        with TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir) / "editable.txt"
            write_text_file = create_write_text_file_tool([Path.cwd(), temp_dir])
            replace_in_file = create_replace_in_file_tool([Path.cwd(), temp_dir])

            write_result = write_text_file.invoke(
                {"path": str(temp_file), "content": "hello world"}
            )
            replace_result = replace_in_file.invoke(
                {
                    "path": str(temp_file),
                    "old_text": "world",
                    "new_text": "agent",
                }
            )

            final_content = temp_file.read_text(encoding="utf-8")

        self.assertIn("已写入文件", write_result)
        self.assertIn("已更新文件", replace_result)
        self.assertEqual(final_content, "hello agent")

    def test_apply_unified_patch_can_update_allowed_file(self) -> None:
        """
        测试：统一补丁工具可以修改允许范围内的文件。
        """
        with TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir) / "patch.txt"
            temp_file.write_text("alpha\nbeta\n", encoding="utf-8")
            apply_unified_patch = create_apply_unified_patch_tool([Path.cwd(), temp_dir])
            patch_text = (
                "--- a/patch.txt\n"
                "+++ b/patch.txt\n"
                "@@ -1,2 +1,2 @@\n"
                " alpha\n"
                "-beta\n"
                "+gamma\n"
            )

            result = apply_unified_patch.invoke({"patch_text": patch_text})
            final_content = temp_file.read_text(encoding="utf-8")

        self.assertIn("已更新文件", result)
        self.assertEqual(final_content, "alpha\ngamma\n")

    def test_run_shell_command_can_execute_shell_command(self) -> None:
        """
        测试：通用命令执行工具可以在允许工作目录内运行命令。
        """
        run_shell_command = create_run_shell_command_tool([Path.cwd()])
        command = (
            "Write-Output 'shell-tool-ok'"
            if sys.platform.startswith("win")
            else "printf 'shell-tool-ok'"
        )

        result = run_shell_command.invoke(
            {
                "command": command,
                "working_directory": str(Path.cwd()),
            }
        )

        self.assertIn("shell-tool-ok", result)
        self.assertIn("执行权限", result)


if __name__ == "__main__":
    unittest.main()
