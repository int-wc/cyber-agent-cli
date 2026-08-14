"""默认工具注册表：让核心工具可注册、可替换、可禁用。

对标 dsh「一切皆插件、无特权内核」的最小落地：
- 核心工具（scan_port/list_directory/run_shell_command 等）不再硬编码，
  而是以 factory 注册到注册表（延迟构建，支持上下文参数）；
- 外部代码可 register() 新增工具、replace() 覆盖默认实现、
  disable()/enable() 禁用或恢复；
- get_default_tools() 保持原签名从注册表构建，全部现有调用点兼容。

设计约束：
- factory 签名：factory(**context) -> BaseTool，context 携带
  allowed_roots/execution_controller/mode 等构建参数；
- 注册表是模块级单例（default_tool_registry），测试可重建隔离实例；
- 禁用不删除注册，enable 可恢复。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool

# 工具 factory：接收上下文 kwargs，返回工具实例
ToolFactory = Callable[..., BaseTool]


class ToolRegistry:
    """可注册/可替换/可禁用的默认工具注册表。"""

    def __init__(self) -> None:
        self._factories: dict[str, ToolFactory] = {}
        self._disabled: set[str] = set()

    def register(
        self,
        name: str,
        factory: ToolFactory,
        *,
        replace: bool = False,
    ) -> None:
        """注册一个工具 factory。

        默认禁止覆盖已存在的工具名（避免误替换核心工具）；
        replace=True 显式允许覆盖（对标 dsh 的 patch 覆盖语义）。
        """
        if not name or not callable(factory):
            raise ValueError("工具注册需要非空名称与可调用 factory。")
        if name in self._factories and not replace:
            raise ValueError(
                f"工具 {name} 已注册；如需覆盖请用 replace=True 或 registry.replace()。"
            )
        self._factories[name] = factory

    def replace(self, name: str, factory: ToolFactory) -> None:
        """覆盖已注册工具的 factory（不存在时等同 register）。"""
        self.register(name, factory, replace=True)

    def unregister(self, name: str) -> None:
        """移除工具注册。"""
        self._factories.pop(name, None)
        self._disabled.discard(name)

    def disable(self, name: str) -> None:
        """禁用工具（保留注册，构建时跳过）。"""
        if name in self._factories:
            self._disabled.add(name)

    def enable(self, name: str) -> None:
        """恢复被禁用的工具。"""
        self._disabled.discard(name)

    def is_registered(self, name: str) -> bool:
        """工具是否已注册。"""
        return name in self._factories

    def is_enabled(self, name: str) -> bool:
        """工具是否已注册且未被禁用。"""
        return name in self._factories and name not in self._disabled

    def registered_names(self) -> list[str]:
        """返回全部已注册工具名（含被禁用的）。"""
        return sorted(self._factories.keys())

    def build(self, **context: Any) -> list[BaseTool]:
        """按注册顺序构建全部启用的工具实例。

        factory 抛错时跳过该工具（返回错误信息占位），不中断整体构建，
        与旧版 get_default_tools 的容错语义保持一致。
        """
        tools: list[BaseTool] = []
        for name, factory in self._factories.items():
            if name in self._disabled:
                continue
            try:
                tools.append(factory(**context))
            except Exception as exc:  # noqa: BLE001 - 单工具构建失败不中断
                from ..logging import log_error
                log_error("tool_registry", f"构建工具 {name} 失败：{exc}")
        return tools


# 模块级单例：get_default_tools 从它构建，外部可 register/replace/disable
default_tool_registry = ToolRegistry()
