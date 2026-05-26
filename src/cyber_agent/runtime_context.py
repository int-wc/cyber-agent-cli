"""运行时上下文类型定义，替代散落在各模块中的裸 dict。"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from .agent.approval import ApprovalPolicy
    from .agent.mode import AgentMode
    from .capability_registry import CapabilityRegistry
    from .cli.interactive import InteractionUiMode
    from .execution_control import ExecutionController
    from .mcp_client import MCPClient


class RuntimeContext(TypedDict, total=False):
    """CLI 会话运行时上下文，贯穿命令解析、交互循环与 webhook 网关。"""

    mode: "AgentMode"
    approval_policy: "ApprovalPolicy"
    ui_mode: "InteractionUiMode"
    service_name: str
    model_name: str
    api_key: str
    base_url: str | None
    execution_controller: "ExecutionController"
    capability_registry: "CapabilityRegistry | None"
    mcp_client: "MCPClient | None"
    tools: list["BaseTool"]
    file_skills: list
    allowed_roots: list[Path]
    command_registry: dict[str, Path]
    extra_allowed_paths: list[Path]
    configured_registry: dict[str, Path]
    saved_allowed_paths: list[Path]
    local_config_path: Path
    session_id: str
    session_source_id: str | None
    session_storage_dir: Path
    runtime_capabilities_loaded: bool
    _stop_input_buffer: str
