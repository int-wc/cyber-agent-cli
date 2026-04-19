"""Cyber Agent CLI 包的轻量级导出入口。"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.1.0"

_MODULE_EXPORTS = {
    "agent": "cyber_agent.agent",
    "capability_registry": "cyber_agent.capability_registry",
    "cli": "cyber_agent.cli",
    "config": "cyber_agent.config",
    "execution_control": "cyber_agent.execution_control",
    "local_config": "cyber_agent.local_config",
    "session_store": "cyber_agent.session_store",
    "tools": "cyber_agent.tools",
}

_ATTRIBUTE_EXPORTS = {
    "AgentMode": ("cyber_agent.agent.mode", "AgentMode"),
    "CapabilityRegistry": ("cyber_agent.capability_registry", "CapabilityRegistry"),
    "ExecutionController": ("cyber_agent.execution_control", "ExecutionController"),
    "describe_allowed_roots": ("cyber_agent.tools", "describe_allowed_roots"),
    "describe_command_registry": ("cyber_agent.tools", "describe_command_registry"),
    "describe_tools": ("cyber_agent.tools", "describe_tools"),
    "get_default_tools": ("cyber_agent.tools", "get_default_tools"),
    "scan_port": ("cyber_agent.tools", "scan_port"),
}

__all__ = [
    "AgentMode",
    "CapabilityRegistry",
    "ExecutionController",
    "agent",
    "capability_registry",
    "cli",
    "config",
    "describe_allowed_roots",
    "describe_command_registry",
    "describe_tools",
    "execution_control",
    "get_default_tools",
    "local_config",
    "scan_port",
    "session_store",
    "tools",
]


def __getattr__(name: str) -> Any:
    """按需导出子模块和常用对象，避免包导入时触发整棵依赖树。"""
    module_path = _MODULE_EXPORTS.get(name)
    if module_path is not None:
        module = import_module(module_path)
        globals()[name] = module
        return module

    export_target = _ATTRIBUTE_EXPORTS.get(name)
    if export_target is not None:
        module = import_module(export_target[0])
        value = getattr(module, export_target[1])
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """补齐交互式环境下的自动补全候选。"""
    return sorted({*globals().keys(), *__all__})
