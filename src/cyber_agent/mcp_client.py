from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, tool

from .tools.metadata import attach_tool_risk

MCP_AVAILABLE = False
MCP_IMPORT_ERROR: ModuleNotFoundError | None = None

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    MCP_AVAILABLE = True
except ModuleNotFoundError as exc:
    MCP_IMPORT_ERROR = exc

PROJECT_MCP_CONFIG = ".mcp.json"
MCP_CONNECTION_TIMEOUT_SECONDS = 30
MCP_TOOL_CALL_TIMEOUT_SECONDS = 60


@dataclass(slots=True)
class MCPServerConfig:
    """描述一个 MCP 服务器的连接配置。"""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None


@dataclass(slots=True)
class MCPToolInfo:
    """描述从 MCP 服务器发现的工具。"""

    server_name: str
    tool_name: str
    description: str
    input_schema: dict[str, Any]


def discover_mcp_configs() -> list[MCPServerConfig]:
    """扫描项目级 MCP 配置文件，返回服务器配置列表。"""
    config_paths = [
        Path(PROJECT_MCP_CONFIG).resolve(),
    ]

    servers: dict[str, MCPServerConfig] = {}

    for config_path in config_paths:
        if not config_path.is_file():
            continue
        try:
            raw_data = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(raw_data, dict):
            continue

        mcp_servers = raw_data.get("mcpServers")
        if not isinstance(mcp_servers, dict):
            continue

        for server_name, server_config in mcp_servers.items():
            if not isinstance(server_config, dict):
                continue
            if server_name in servers:
                continue
            command = server_config.get("command")
            if not command or not isinstance(command, str):
                continue
            servers[server_name] = MCPServerConfig(
                name=server_name,
                command=command,
                args=[
                    str(arg) for arg in server_config.get("args", [])
                    if isinstance(arg, str)
                ],
                env={
                    str(k): str(v)
                    for k, v in server_config.get("env", {}).items()
                    if isinstance(k, str) and v is not None
                },
                cwd=(
                    str(server_config.get("cwd"))
                    if server_config.get("cwd")
                    else None
                ),
            )

    return list(servers.values())


class MCPClient:
    """管理 MCP 服务器连接、工具发现与跨线程调用。"""

    def __init__(self) -> None:
        self._configs: list[MCPServerConfig] = []
        self._tools: list[MCPToolInfo] = []
        self._server_sessions: dict[str, ClientSession] = {}
        self._server_streams: dict[str, tuple[Any, Any]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._connected = False

    @property
    def tools(self) -> list[MCPToolInfo]:
        return list(self._tools)

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self, configs: list[MCPServerConfig] | None = None) -> None:
        """在后台事件循环中连接所有 MCP 服务器并发现工具。"""
        if not MCP_AVAILABLE:
            return

        if configs is not None:
            self._configs = list(configs)

        if not self._configs:
            return

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._run_event_loop,
            daemon=True,
        )
        self._loop_thread.start()

        future = asyncio.run_coroutine_threadsafe(
            self._connect_all_servers(),
            self._loop,
        )
        try:
            future.result(timeout=MCP_CONNECTION_TIMEOUT_SECONDS)
        except (TimeoutError, Exception):
            pass

    def disconnect(self) -> None:
        """断开所有 MCP 服务器连接并停止事件循环。"""
        if self._loop is not None and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._disconnect_all_servers(),
                self._loop,
            )
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._loop_thread is not None and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=5)

        self._connected = False

    def get_langchain_tools(self) -> list[BaseTool]:
        """将发现的 MCP 工具包装为 LangChain BaseTool 列表。"""
        wrapped_tools: list[BaseTool] = []
        for mcp_tool in self._tools:
            wrapped_tools.append(self._wrap_mcp_tool(mcp_tool))
        return wrapped_tools

    def _run_event_loop(self) -> None:
        """后台线程的事件循环。"""
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _connect_all_servers(self) -> None:
        """连接所有已配置的 MCP 服务器。"""
        for config in self._configs:
            try:
                await self._connect_server(config)
            except Exception:
                continue
        self._connected = True

    async def _connect_server(self, config: MCPServerConfig) -> None:
        """连接单个 MCP 服务器并发现其工具。"""
        env = None
        if config.env:
            import os
            env = {**os.environ, **config.env}

        server_params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=env,
        )

        transport = await asyncio.wait_for(
            stdio_client(server_params).__aenter__(),
            timeout=MCP_CONNECTION_TIMEOUT_SECONDS,
        )
        read_stream, write_stream = transport

        session = await asyncio.wait_for(
            ClientSession(read_stream, write_stream).__aenter__(),
            timeout=MCP_CONNECTION_TIMEOUT_SECONDS,
        )
        await asyncio.wait_for(
            session.initialize(),
            timeout=MCP_CONNECTION_TIMEOUT_SECONDS,
        )

        tools_result = await asyncio.wait_for(
            session.list_tools(),
            timeout=MCP_CONNECTION_TIMEOUT_SECONDS,
        )

        self._server_sessions[config.name] = session
        self._server_streams[config.name] = (read_stream, write_stream)

        for mcp_tool in tools_result.tools:
            self._tools.append(
                MCPToolInfo(
                    server_name=config.name,
                    tool_name=mcp_tool.name,
                    description=mcp_tool.description or "",
                    input_schema=mcp_tool.inputSchema,
                )
            )

    async def _disconnect_all_servers(self) -> None:
        """断开所有服务器连接。"""
        for session in self._server_sessions.values():
            try:
                await session.__aexit__(None, None, None)
            except Exception:
                pass
        self._server_sessions.clear()
        self._server_streams.clear()

    def _call_mcp_tool_sync(
        self, server_name: str, tool_name: str, arguments: dict[str, Any]
    ) -> str:
        """跨线程同步调用 MCP 工具。"""
        if self._loop is None or not self._loop.is_running():
            return "❌ MCP 事件循环未运行。"

        session = self._server_sessions.get(server_name)
        if session is None:
            return f"❌ MCP 服务器未连接：{server_name}"

        async def call_tool() -> str:
            try:
                result = await asyncio.wait_for(
                    session.call_tool(tool_name, arguments=arguments),
                    timeout=MCP_TOOL_CALL_TIMEOUT_SECONDS,
                )
                if result.isError:
                    error_text = ""
                    for block in result.content:
                        if hasattr(block, "text"):
                            error_text += str(block.text)
                    return f"❌ MCP 工具返回错误：{error_text}"

                output_parts: list[str] = []
                for block in result.content:
                    if hasattr(block, "text"):
                        output_parts.append(str(block.text))
                    elif hasattr(block, "data"):
                        output_parts.append(f"[二进制数据: {len(block.data)} bytes]")
                return "\n".join(output_parts) if output_parts else "无输出。"
            except asyncio.TimeoutError:
                return "❌ MCP 工具调用超时。"
            except Exception as exc:
                return f"❌ MCP 工具调用异常：{exc}"

        future = asyncio.run_coroutine_threadsafe(call_tool(), self._loop)
        try:
            return future.result(timeout=MCP_TOOL_CALL_TIMEOUT_SECONDS + 5)
        except TimeoutError:
            return "❌ MCP 工具调用超时。"
        except Exception as exc:
            return f"❌ MCP 工具调用失败：{exc}"

    def _wrap_mcp_tool(self, mcp_tool: MCPToolInfo) -> BaseTool:
        """将单个 MCP 工具包装为同步 LangChain tool。"""
        mcp_client = self

        @tool(mcp_tool.tool_name)
        def mcp_wrapper(**kwargs: Any) -> str:
            """由 MCP 客户端动态包装的工具。"""
            return mcp_client._call_mcp_tool_sync(
                mcp_tool.server_name,
                mcp_tool.tool_name,
                kwargs,
            )

        mcp_wrapper.description = (
            f"[MCP:{mcp_tool.server_name}] {mcp_tool.description}"
        )
        return attach_tool_risk(mcp_wrapper, "execute")


def load_mcp_client() -> MCPClient | None:
    """加载 MCP 配置并连接所有服务器，返回已连接的客户端。MCP 不可用时返回 None。"""
    if not MCP_AVAILABLE:
        return None

    configs = discover_mcp_configs()
    if not configs:
        return None

    client = MCPClient()
    client.connect(configs)
    return client


def describe_mcp_tools(client: MCPClient) -> list[str]:
    """生成适合 CLI 展示的 MCP 工具描述。"""
    lines: list[str] = []
    for tool_info in client.tools:
        lines.append(
            f"[MCP:{tool_info.server_name}] {tool_info.tool_name}: "
            f"{tool_info.description}"
        )
    return lines
