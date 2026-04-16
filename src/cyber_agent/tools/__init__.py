from collections.abc import Iterable, Mapping
from pathlib import Path

from ..agent.mode import AgentMode
from .filesystem import (
    create_list_directory_tool,
    create_read_text_file_tool,
    create_replace_in_file_tool,
    create_write_text_file_tool,
    describe_allowed_roots,
    normalize_allowed_roots,
)
from .patching import create_apply_unified_patch_tool
from .security import scan_port
from .system import (
    create_run_registered_tool_tool,
    create_run_shell_command_tool,
    describe_command_registry,
    normalize_command_registry,
)


def resolve_allowed_roots(
    mode: AgentMode,
    extra_allowed_paths: Iterable[Path | str] | None = None,
) -> list[Path]:
    """计算当前模式下生效的允许路径根目录。"""
    roots: list[Path | str] = [Path.cwd().resolve()]
    if mode is AgentMode.AUTHORIZED and extra_allowed_paths:
        roots.extend(extra_allowed_paths)
    return normalize_allowed_roots(roots)


def resolve_command_registry(
    mode: AgentMode,
    command_registry: Mapping[str, Path | str] | None = None,
) -> dict[str, Path]:
    """计算当前模式下生效的外部工具注册表。"""
    if mode is not AgentMode.AUTHORIZED or not command_registry:
        return {}
    return normalize_command_registry(command_registry)


def get_default_tools(
    mode: AgentMode = AgentMode.STANDARD,
    extra_allowed_paths: Iterable[Path | str] | None = None,
    command_registry: Mapping[str, Path | str] | None = None,
):
    """返回默认启用的工具列表。"""
    allowed_roots = resolve_allowed_roots(mode, extra_allowed_paths)
    normalized_registry = resolve_command_registry(mode, command_registry)

    tools = [
        scan_port,
        create_list_directory_tool(allowed_roots),
        create_read_text_file_tool(allowed_roots),
        create_write_text_file_tool(allowed_roots),
        create_replace_in_file_tool(allowed_roots),
        create_apply_unified_patch_tool(allowed_roots),
        create_run_shell_command_tool(allowed_roots),
    ]
    if normalized_registry:
        tools.append(create_run_registered_tool_tool(normalized_registry))
    return tools


def describe_tools(
    mode: AgentMode = AgentMode.STANDARD,
    extra_allowed_paths: Iterable[Path | str] | None = None,
    command_registry: Mapping[str, Path | str] | None = None,
) -> list[str]:
    """生成适合 CLI 展示的工具说明。"""
    descriptions: list[str] = []
    for tool in get_default_tools(mode, extra_allowed_paths, command_registry):
        first_line = tool.description.strip().splitlines()[0]
        descriptions.append(f"{tool.name}: {first_line}")
    return descriptions


__all__ = [
    "describe_allowed_roots",
    "describe_command_registry",
    "describe_tools",
    "get_default_tools",
    "resolve_allowed_roots",
    "resolve_command_registry",
    "scan_port",
]
