from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from ..agent.mode import AgentMode
from ..execution_control import ExecutionController
from .filesystem import (
    create_list_directory_tool,
    create_read_text_file_tool,
    create_replace_in_file_tool,
    create_write_text_file_tool,
    describe_allowed_roots,
    normalize_allowed_roots,
)
from .patching import create_apply_unified_patch_tool
from .search import create_search_web_tool
from .security import scan_port
from .system import (
    create_run_registered_tool_tool,
    create_run_shell_command_tool,
    describe_command_registry,
    normalize_command_registry,
)

if TYPE_CHECKING:
    from ..capability_registry import CapabilityRegistry


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
    execution_controller: ExecutionController | None = None,
    capability_registry: CapabilityRegistry | None = None,
):
    """返回默认启用的工具列表。"""
    allowed_roots = resolve_allowed_roots(mode, extra_allowed_paths)
    normalized_registry = resolve_command_registry(mode, command_registry)

    tools = [
        scan_port,
        create_search_web_tool(
            execution_controller,
            capability_registry=capability_registry,
        ),
        create_list_directory_tool(allowed_roots),
        create_read_text_file_tool(allowed_roots),
        create_write_text_file_tool(allowed_roots),
        create_replace_in_file_tool(allowed_roots),
        create_apply_unified_patch_tool(allowed_roots),
        create_run_shell_command_tool(allowed_roots, execution_controller),
    ]
    if normalized_registry:
        tools.append(
            create_run_registered_tool_tool(
                normalized_registry,
                execution_controller,
            )
        )
    if capability_registry is not None:
        tools.extend(capability_registry.get_dynamic_tools())
    return tools


def describe_tool_instances(tools) -> list[str]:
    """基于实际工具实例生成适合 CLI 展示的工具说明。"""
    descriptions: list[str] = []
    for tool in tools:
        first_line = tool.description.strip().splitlines()[0]
        descriptions.append(f"{tool.name}: {first_line}")
    return descriptions


def describe_tools(
    mode: AgentMode = AgentMode.STANDARD,
    extra_allowed_paths: Iterable[Path | str] | None = None,
    command_registry: Mapping[str, Path | str] | None = None,
    capability_registry: CapabilityRegistry | None = None,
) -> list[str]:
    """生成适合 CLI 展示的工具说明。"""
    return describe_tool_instances(
        get_default_tools(
            mode,
            extra_allowed_paths,
            command_registry,
            capability_registry=capability_registry,
        )
    )


__all__ = [
    "describe_allowed_roots",
    "describe_command_registry",
    "describe_tool_instances",
    "describe_tools",
    "get_default_tools",
    "resolve_allowed_roots",
    "resolve_command_registry",
    "scan_port",
]
