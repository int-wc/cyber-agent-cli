import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from langchain_core.tools import tool

from .filesystem import normalize_allowed_roots, resolve_permitted_path

MAX_COMMAND_OUTPUT_CHARS = 4000
MAX_TOOL_TIMEOUT_SECONDS = 120


def normalize_command_registry(
    command_registry: Mapping[str, Path | str],
) -> dict[str, Path]:
    """规范化外部工具注册表。"""
    normalized_registry: dict[str, Path] = {}
    for tool_name, executable_path in command_registry.items():
        normalized_registry[tool_name] = Path(executable_path).expanduser().resolve()
    return normalized_registry


def describe_command_registry(command_registry: Mapping[str, Path]) -> list[str]:
    """返回适合状态展示的外部工具列表。"""
    return [
        f"{tool_name}={executable_path}"
        for tool_name, executable_path in command_registry.items()
    ]


def _truncate_output(output: str) -> str:
    truncated_output = output[:MAX_COMMAND_OUTPUT_CHARS]
    if len(output) > MAX_COMMAND_OUTPUT_CHARS:
        truncated_output += "\n... 输出过长，已截断。"
    return truncated_output


def _format_completed_process_output(
    *,
    command_description: str,
    working_directory: Path,
    completed_process: subprocess.CompletedProcess[str],
) -> str:
    stdout = completed_process.stdout.strip()
    stderr = completed_process.stderr.strip()
    combined_output = "\n".join(
        part for part in [stdout, stderr] if part
    ).strip() or "无输出。"

    return (
        f"{command_description}\n"
        f"工作目录: {working_directory}\n"
        f"退出码: {completed_process.returncode}\n"
        f"执行权限: 继承当前 CLI 进程权限，不会自动提权。\n"
        f"输出:\n{_truncate_output(combined_output)}"
    )


def _build_shell_command(command: str) -> list[str]:
    if os.name == "nt":
        powershell_executable = shutil.which("pwsh") or shutil.which("powershell")
        if powershell_executable is None:
            raise FileNotFoundError("未找到可用的 PowerShell 可执行文件。")
        return [
            powershell_executable,
            "-NoLogo",
            "-NoProfile",
            "-Command",
            command,
        ]

    return ["/bin/sh", "-lc", command]


def create_run_shell_command_tool(allowed_roots: Sequence[Path]):
    """创建受工作目录范围约束的 Shell 命令执行工具。"""
    normalized_roots = normalize_allowed_roots(allowed_roots)

    @tool("run_shell_command", extras={"risk": "execute"})
    def run_shell_command(
        command: str,
        working_directory: str = ".",
        timeout_seconds: int = 60,
    ) -> str:
        """
        在受限工作目录内执行 shell 命令。
        适合运行测试、构建命令、代码格式化或外部程序。
        """
        try:
            resolved_working_directory = resolve_permitted_path(
                working_directory,
                normalized_roots,
            )
        except ValueError as exc:
            return f"❌ {exc}"

        if not resolved_working_directory.exists():
            return f"❌ 工作目录不存在：{working_directory}"
        if not resolved_working_directory.is_dir():
            return f"❌ 工作目录不是目录：{working_directory}"

        safe_timeout_seconds = max(1, min(timeout_seconds, MAX_TOOL_TIMEOUT_SECONDS))
        try:
            completed_process = subprocess.run(
                _build_shell_command(command),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=safe_timeout_seconds,
                cwd=resolved_working_directory,
                shell=False,
                check=False,
            )
        except FileNotFoundError as exc:
            return f"❌ 命令执行环境不可用：{exc}"
        except subprocess.TimeoutExpired:
            return f"❌ 命令执行超时：{command}"
        except Exception as exc:
            return f"❌ 执行命令时发生错误：{exc}"

        return _format_completed_process_output(
            command_description=f"命令: {command}",
            working_directory=resolved_working_directory,
            completed_process=completed_process,
        )

    return run_shell_command


def create_run_registered_tool_tool(command_registry: Mapping[str, Path | str]):
    """创建只允许执行已注册外部工具的命令工具。"""
    normalized_registry = normalize_command_registry(command_registry)
    registered_tool_names = ", ".join(sorted(normalized_registry)) or "无"

    @tool("run_registered_tool", extras={"risk": "execute"})
    def run_registered_tool(
        tool_name: str,
        arguments: list[str] | None = None,
        timeout_seconds: int = 30,
    ) -> str:
        """
        执行已注册的外部工具。
        该工具只允许调用显式注册过的绝对路径程序，
        命令会继承当前 CLI 进程权限，不会自动使用 sudo 或其他提权方式。
        """
        command_path = normalized_registry.get(tool_name)
        if command_path is None:
            return (
                f"❌ 未注册的外部工具：{tool_name}。"
                f"当前已注册工具有：{registered_tool_names}"
            )

        safe_timeout_seconds = max(1, min(timeout_seconds, MAX_TOOL_TIMEOUT_SECONDS))
        command_arguments = [str(argument) for argument in (arguments or [])]
        command = [str(command_path), *command_arguments]

        try:
            completed_process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=safe_timeout_seconds,
                cwd=Path.cwd(),
                shell=False,
                check=False,
            )
        except FileNotFoundError:
            return f"❌ 工具路径不存在：{command_path}"
        except subprocess.TimeoutExpired:
            return f"❌ 工具执行超时：{tool_name}"
        except Exception as exc:
            return f"❌ 调用外部工具时发生错误：{exc}"

        return _format_completed_process_output(
            command_description=f"工具: {tool_name}\n路径: {command_path}",
            working_directory=Path.cwd(),
            completed_process=completed_process,
        )

    return run_registered_tool
