from __future__ import annotations

import json
import logging
import re
import threading
import time
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any

import typer
from click.exceptions import Abort
from rich.console import Console, RenderableType

from ..agent.approval import (
    ApprovalDecision,
    ApprovalPolicy,
    get_approval_policy_label,
    parse_approval_policy,
)
from ..agent.events import AgentEventType
from ..agent.mode import AgentMode, get_mode_label, parse_agent_mode
from ..execution_control import ExecutionController, ExecutionInterruptedError
from ..local_config import (
    add_allow_path_to_local_config,
    get_local_config_path,
    load_config_with_fallback,
    load_local_cli_config,
    merge_allow_paths,
)
from ..version import get_version_display
from .interactive import (
    EXIT_COMMANDS,
    InteractionUiMode,
    get_interaction_ui_mode_label,
    parse_interaction_ui_mode,
)
from .render import CliRenderer
if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
    from langchain_core.tools import BaseTool
    from ..agent.runner import AgentRunner
    from ..capability_registry import CapabilityRegistry

SUPPORTED_WEBHOOK_PROVIDERS = ("feishu", "dingtalk", "wecom", "email")
DEFAULT_WEBHOOK_HOST = "0.0.0.0"
DEFAULT_WEBHOOK_PORT = 8787
DEFAULT_WEBHOOK_REPLY_TIMEOUT_SECONDS = 10.0
logger = logging.getLogger(__name__)
SESSION_STORAGE_DIRNAME = ".cyber-agent-cli-sessions"
RUNTIME_CAPABILITY_REQUIRED_KEYS = (
    "execution_controller",
    "service_name",
    "model_name",
    "api_key",
    "mode",
    "extra_allowed_paths",
    "configured_registry",
)

app = typer.Typer(
    add_completion=False,
    help="一个支持工具调用的命令行智能体原型。",
)
history_app = typer.Typer(
    add_completion=False,
    help="查看、检索与导出当前工作目录下的历史会话。",
)
webhook_app = typer.Typer(
    add_completion=False,
    help="通过 webhook 接入飞书、钉钉、企微、邮件等移动端消息桥接。",
)
hub_app = typer.Typer(
    add_completion=False,
    help="启动 CLI/飞书共享同一 AgentRunner 的 Cyber Agent Hub。",
)
app.add_typer(history_app, name="history")
app.add_typer(webhook_app, name="webhook")
app.add_typer(hub_app, name="hub")

TOOL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
renderer = CliRenderer()
_cli_prompt_session = None
_prompt_toolkit_disabled = False


def _load_feishu_long_connection_support():
    """按需加载飞书长连接支持，避免普通 CLI 启动被 SDK 导入拖慢。"""
    from .feishu_long_connection import (
        select_feishu_long_connection_route,
        serve_feishu_long_connection,
    )

    return select_feishu_long_connection_route, serve_feishu_long_connection


def _get_settings():
    """按需读取配置对象，避免版本和帮助命令导入 pydantic-settings。"""
    from ..config import settings

    return settings


def _load_agent_runner_support():
    """按需加载运行器支持，避免命令解析阶段提前初始化模型依赖。"""
    from ..agent.runner import AgentRunner, extract_text_content

    return AgentRunner, extract_text_content


def _load_message_type_support():
    """按需加载 LangChain 消息类型，避免普通 CLI 启动提前导入 LangChain。"""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    return AIMessage, HumanMessage, SystemMessage, ToolMessage


def _load_capability_registry_support():
    """按需加载动态 capability 注册表。"""
    from ..capability_registry import CapabilityRegistry

    return CapabilityRegistry


def _load_tool_support():
    """按需加载工具集合，避免 `--help` 和 `version` 命令承担工具导入成本。"""
    from ..tools import (
        describe_allowed_roots,
        describe_command_registry,
        describe_tool_instances,
        get_default_tools,
        resolve_allowed_roots,
        resolve_command_registry,
    )

    return {
        "describe_allowed_roots": describe_allowed_roots,
        "describe_command_registry": describe_command_registry,
        "describe_tool_instances": describe_tool_instances,
        "get_default_tools": get_default_tools,
        "resolve_allowed_roots": resolve_allowed_roots,
        "resolve_command_registry": resolve_command_registry,
    }


def _load_session_store_support():
    """按需加载历史会话存储，避免帮助和版本命令导入 LangChain 消息类型。"""
    from ..session_store import (
        append_session_event,
        clear_interrupt_checkpoint,
        create_session_id,
        export_session_history,
        get_session_storage_dir,
        has_interrupt_checkpoint,
        list_stored_sessions,
        load_interrupt_checkpoint,
        load_session_history,
        save_interrupt_checkpoint,
        save_session_history,
        search_stored_sessions,
    )

    return {
        "append_session_event": append_session_event,
        "clear_interrupt_checkpoint": clear_interrupt_checkpoint,
        "create_session_id": create_session_id,
        "export_session_history": export_session_history,
        "get_session_storage_dir": get_session_storage_dir,
        "has_interrupt_checkpoint": has_interrupt_checkpoint,
        "list_stored_sessions": list_stored_sessions,
        "load_interrupt_checkpoint": load_interrupt_checkpoint,
        "load_session_history": load_session_history,
        "save_interrupt_checkpoint": save_interrupt_checkpoint,
        "save_session_history": save_session_history,
        "search_stored_sessions": search_stored_sessions,
    }


def create_runtime_session_id(now: datetime | None = None) -> str:
    """轻量生成会话 ID，避免启动阶段导入完整历史存储模块。"""
    resolved_now = now or datetime.now().astimezone()
    return resolved_now.strftime("%Y%m%d-%H%M%S-%f")


def get_runtime_session_storage_dir(base_dir: Path | None = None) -> Path:
    """轻量计算历史目录路径，支持任意目录启动时回溯查找。"""
    from ..local_config import find_data_dir
    return find_data_dir(SESSION_STORAGE_DIRNAME, base_dir)


def _load_doctor_support():
    """按需加载 doctor 诊断模块，避免普通启动导入 prompt_toolkit/Textual。"""
    from .doctor import build_doctor_payload, build_doctor_rows

    return build_doctor_payload, build_doctor_rows


def _load_webhook_support():
    """按需加载 webhook 网关，避免普通 CLI 启动提前导入移动端桥接链路。"""
    from .webhook import (
        build_default_webhook_routes,
        build_webhook_example_config,
        load_webhook_routes_from_file,
        serve_webhook_gateway,
    )

    return {
        "build_default_webhook_routes": build_default_webhook_routes,
        "build_webhook_example_config": build_webhook_example_config,
        "load_webhook_routes_from_file": load_webhook_routes_from_file,
        "serve_webhook_gateway": serve_webhook_gateway,
    }


def _print_version_and_exit(value: bool) -> None:
    """处理全局 --version 选项，便于脚本和排障快速查看版本。"""
    if not value:
        return
    typer.echo(f"cyber-agent-cli {get_version_display()}")
    raise typer.Exit()


class BuiltinCommandCaptureRenderer(CliRenderer):
    """收集内建命令的 Rich 输出，供 TUI 直接挂载原始面板。"""

    def __init__(self) -> None:
        super().__init__(console=Console(record=True, width=100))
        self.renderables: list[RenderableType] = []

    def print_renderable(self, renderable: RenderableType) -> None:
        """捕获渲染对象而不是直接写到终端。"""
        self.ensure_response_stream_closed()
        self.renderables.append(renderable)


def parse_registered_tool_specs(tool_specs: list[str] | None) -> dict[str, Path]:
    """解析 `name=absolute_path` 格式的外部工具注册参数。"""
    registry: dict[str, Path] = {}

    for tool_spec in tool_specs or []:
        tool_name, separator, raw_path = tool_spec.partition("=")
        if separator != "=" or not tool_name or not raw_path:
            raise typer.BadParameter(
                f"无效的 --tool 参数：{tool_spec}。正确格式应为 name=absolute_path"
            )

        if not TOOL_NAME_PATTERN.fullmatch(tool_name):
            raise typer.BadParameter(
                f"无效的工具名：{tool_name}。仅允许字母、数字、下划线和短横线。"
            )

        executable_path = Path(raw_path).expanduser()
        if not executable_path.is_absolute():
            raise typer.BadParameter(
                f"工具路径必须是绝对路径：{raw_path}"
            )
        if not executable_path.exists():
            raise typer.BadParameter(
                f"工具路径不存在：{raw_path}"
            )
        if executable_path.is_dir():
            raise typer.BadParameter(
                f"工具路径不能是目录：{raw_path}"
            )

        registry[tool_name] = executable_path.resolve()

    return registry


def build_runtime_context(
    mode: AgentMode,
    allow_paths: list[str] | None,
    tool_specs: list[str] | None,
    approval_policy: ApprovalPolicy,
    ui_mode: InteractionUiMode,
    skill_dirs: list[str] | None = None,
    auto_decision: bool = False,
) -> dict[str, object]:
    """统一构建 CLI 运行上下文，避免多处分散解析。"""
    local_config = load_config_with_fallback()
    persisted_allowed_paths = list(local_config.allow_paths)
    cli_allowed_paths = [Path(path).expanduser() for path in (allow_paths or [])]
    extra_allowed_paths = merge_allow_paths(
        persisted_allowed_paths,
        cli_allowed_paths,
    )
    settings = _get_settings()
    service_name = settings.get_service()
    model_name = settings.get_model_name(service_name=service_name)
    api_key = settings.get_api_key(service_name)
    base_url = settings.resolve_base_url(service_name)
    configured_registry = parse_registered_tool_specs(tool_specs)
    allowed_roots = [Path.cwd().resolve()]
    if mode is AgentMode.AUTHORIZED:
        allowed_roots = merge_allow_paths(allowed_roots, extra_allowed_paths)
    command_registry = configured_registry if mode is AgentMode.AUTHORIZED else {}
    execution_controller = ExecutionController()
    from ..skill_loader import load_all_skills as _load_skills

    file_skills = _load_skills(extra_dirs=skill_dirs)
    return {
        "mode": mode,
        "extra_allowed_paths": extra_allowed_paths,
        "saved_allowed_paths": persisted_allowed_paths,
        "local_config_path": get_local_config_path(),
        "allowed_roots": allowed_roots,
        "configured_registry": configured_registry,
        "command_registry": command_registry,
        "tools": [],
        "approval_policy": approval_policy,
        "ui_mode": ui_mode,
        "execution_controller": execution_controller,
        "capability_registry": None,
        "runtime_capabilities_loaded": False,
        "file_skills": file_skills,
        "service_name": service_name,
        "model_name": model_name,
        "base_url": base_url,
        "api_key": api_key,
        "session_id": create_runtime_session_id(),
        "session_source_id": None,
        "session_storage_dir": get_runtime_session_storage_dir(),
        "_stop_input_buffer": "",
        "multi_agent_enabled": "auto",
        "subtask_concurrency": settings.pipeline_subtask_concurrency,
        "max_subagents": settings.pipeline_max_subagents,
        "execution_profile": settings.pipeline_execution_profile,
        "auto_decision": auto_decision,
        "_recent_inputs": [],  # 最近 50 条用户输入
    }


def _path_list_contains_root(paths: object) -> bool:
    if not isinstance(paths, list):
        return False
    for raw_path in paths:
        try:
            if Path(raw_path).expanduser().resolve() == Path("/"):
                return True
        except (OSError, TypeError, ValueError):
            continue
    return False


def infer_execution_profile(runtime_context: dict[str, object]) -> str:
    """根据运行授权推导四柱执行姿态。"""
    configured = str(runtime_context.get("execution_profile", "auto")).strip().lower()
    if configured in {"conservative", "aggressive"}:
        return configured
    if configured not in {"", "auto"}:
        return "auto"

    mode = runtime_context.get("mode")
    mode_value = getattr(mode, "value", str(mode))
    approval_policy = runtime_context.get("approval_policy")
    approval_value = getattr(approval_policy, "value", str(approval_policy))
    has_root = _path_list_contains_root(runtime_context.get("allowed_roots")) or _path_list_contains_root(
        runtime_context.get("extra_allowed_paths")
    )
    if (
        mode_value == AgentMode.AUTHORIZED.value
        and approval_value == ApprovalPolicy.AUTO.value
        and bool(runtime_context.get("auto_decision", False))
        and has_root
    ):
        return "aggressive"
    return "conservative"


def get_or_build_runtime_context(ctx: typer.Context) -> dict[str, object]:
    """按需构建运行上下文，让不需要 Agent 的命令快速启动。"""
    ctx.ensure_object(dict)
    runtime_context = ctx.obj.get("runtime_context")
    if isinstance(runtime_context, dict):
        return runtime_context

    runtime_options = ctx.obj.get("runtime_options")
    if not isinstance(runtime_options, dict):
        raise RuntimeError("运行上下文尚未初始化。")

    runtime_context = build_runtime_context(
        runtime_options["mode"],
        runtime_options["allow_paths"],
        runtime_options["tool_specs"],
        runtime_options["approval_policy"],
        runtime_options["ui_mode"],
        skill_dirs=runtime_options.get("skill_dirs"),
        auto_decision=runtime_options.get("auto_decision", False),
    )
    ctx.obj["runtime_context"] = runtime_context
    return runtime_context


def _sync_runner_capabilities_from_context(
    runtime_context: dict[str, object],
    runner: AgentRunner,
) -> None:
    """把上下文中已有的工具作用域同步给运行器，兼容 webhook 最小上下文。"""
    runner.allowed_roots = list(
        runtime_context.get("allowed_roots", getattr(runner, "allowed_roots", []))
    )
    runner.command_registry = dict(
        runtime_context.get(
            "command_registry",
            getattr(runner, "command_registry", {}),
        )
    )
    runner.tools = list(runtime_context.get("tools", getattr(runner, "tools", [])))
    runner.capability_registry = runtime_context.get(
        "capability_registry",
        getattr(runner, "capability_registry", None),
    )


def ensure_runtime_capabilities(
    runtime_context: dict[str, object],
    runner: AgentRunner | None = None,
) -> None:
    """在真正需要工具或 capability 时再加载重依赖。"""
    if runtime_context.get("runtime_capabilities_loaded") is True:
        if runner is not None:
            _sync_runner_capabilities_from_context(runtime_context, runner)
        return

    missing_runtime_keys = [
        key for key in RUNTIME_CAPABILITY_REQUIRED_KEYS if key not in runtime_context
    ]
    if missing_runtime_keys:
        if runner is None:
            raise KeyError(missing_runtime_keys[0])
        runtime_context["allowed_roots"] = list(getattr(runner, "allowed_roots", []))
        runtime_context["command_registry"] = dict(
            getattr(runner, "command_registry", {})
        )
        runtime_context["tools"] = list(getattr(runner, "tools", []))
        runtime_context["capability_registry"] = getattr(
            runner,
            "capability_registry",
            None,
        )
        runtime_context["runtime_capabilities_loaded"] = True
        _sync_runner_capabilities_from_context(runtime_context, runner)
        return

    # 确保 .cyber/ 项目目录存在（类似 .claude/）
    try:
        from ..local_config import find_cyber_dir
        find_cyber_dir()
    except Exception:
        pass

    CapabilityRegistry = _load_capability_registry_support()
    tool_support = _load_tool_support()
    capability_registry = CapabilityRegistry(
        execution_controller=runtime_context["execution_controller"],
        service_name=str(runtime_context["service_name"]),
        model_name=str(runtime_context["model_name"]),
        api_key=str(runtime_context["api_key"]),
        base_url=(
            str(runtime_context["base_url"])
            if runtime_context.get("base_url") is not None
            else None
        ),
    )
    allowed_roots = tool_support["resolve_allowed_roots"](
        runtime_context["mode"],
        runtime_context["extra_allowed_paths"],
    )
    command_registry = tool_support["resolve_command_registry"](
        runtime_context["mode"],
        runtime_context["configured_registry"],
    )
    mcp_client = None
    try:
        from ..mcp_client import load_mcp_client as _load_mcp

        mcp_client = _load_mcp()
    except Exception as exc:
        from ..logging import log_warning
        log_warning("cli.app", f"MCP 客户端加载失败，MCP 工具不可用：{exc}")

    tools = tool_support["get_default_tools"](
        runtime_context["mode"],
        runtime_context["extra_allowed_paths"],
        runtime_context["configured_registry"],
        runtime_context["execution_controller"],
        capability_registry,
        mcp_client=mcp_client,
    )

    runtime_context["allowed_roots"] = allowed_roots
    runtime_context["command_registry"] = command_registry
    runtime_context["tools"] = tools
    runtime_context["capability_registry"] = capability_registry
    runtime_context["mcp_client"] = mcp_client
    runtime_context["runtime_capabilities_loaded"] = True

    if runner is not None:
        runner.allowed_roots = list(allowed_roots)
        runner.command_registry = dict(command_registry)
        runner.tools = list(tools)
        runner.capability_registry = capability_registry
        capability_registry.register_refresh_callback(
            lambda: _refresh_runner_capabilities(runtime_context, runner)
        )


from .app_multi_agent import _detect_task_complexity, _run_multi_agent_turn  # noqa: E402


def create_runner(runtime_context: dict[str, object]) -> AgentRunner:
    """按运行上下文创建会话运行器。"""
    AgentRunner, _ = _load_agent_runner_support()
    runner = AgentRunner(
        runtime_context["tools"],
        mode=runtime_context["mode"],
        allowed_roots=runtime_context["allowed_roots"],
        command_registry=runtime_context["command_registry"],
        extra_allowed_paths=runtime_context["extra_allowed_paths"],
        configured_registry=runtime_context["configured_registry"],
        execution_controller=runtime_context["execution_controller"],
        capability_registry=runtime_context["capability_registry"],
        file_skills=runtime_context.get("file_skills", []),
        service_name=runtime_context["service_name"],
        model_name=runtime_context["model_name"],
        api_key=runtime_context["api_key"],
        base_url=runtime_context["base_url"],
    )
    capability_registry = runtime_context.get("capability_registry")
    register_refresh_callback = getattr(
        capability_registry,
        "register_refresh_callback",
        None,
    )
    if callable(register_refresh_callback):
        register_refresh_callback(
            lambda: _refresh_runner_capabilities(runtime_context, runner)
        )
    runner.runtime_capability_loader = lambda: ensure_runtime_capabilities(
        runtime_context,
        runner,
    )
    # 同步模型名称以正确计算 token 花费
    renderer._model_name = str(runtime_context.get("model_name", ""))
    return runner


def _refresh_runner_capabilities(
    runtime_context: dict[str, object],
    runner: AgentRunner,
) -> None:
    """在动态 capability 变更后刷新运行器和 CLI 运行时上下文。"""
    runner._refresh_runtime_scope()
    sync_runtime_context_from_runner(runtime_context, runner)


def sync_runtime_context_from_runner(
    runtime_context: dict[str, object],
    runner: AgentRunner,
) -> None:
    """将运行器中的动态状态回写到 CLI 运行上下文。"""
    runtime_context["mode"] = runner.mode
    runtime_context["extra_allowed_paths"] = list(runner.extra_allowed_paths)
    runtime_context["allowed_roots"] = list(runner.allowed_roots)
    runtime_context["command_registry"] = dict(runner.command_registry)
    runtime_context["tools"] = list(runner.tools)
    runtime_context["service_name"] = runner.service
    runtime_context["model_name"] = runner.model_name
    runtime_context["base_url"] = runner.base_url
    runtime_context["api_key"] = runner.api_key


def print_banner(
    runner: AgentRunner,
    runtime_context: dict[str, object],
    cli_renderer: CliRenderer = renderer,
) -> None:
    """输出交互模式欢迎信息。"""
    cli_renderer.print_banner(
        mode=runner.mode,
        service=runner.service,
        model=runner.model_name,
        cwd=Path.cwd(),
        approval_policy=runtime_context["approval_policy"],
        version=get_version_display(),
    )


def print_runtime_banner(
    runtime_context: dict[str, object],
    cli_renderer: CliRenderer = renderer,
) -> None:
    """在运行器尚未创建时输出轻量欢迎信息。"""
    cli_renderer.print_banner(
        mode=runtime_context["mode"],
        service=str(runtime_context["service_name"]),
        model=str(runtime_context["model_name"]),
        cwd=Path.cwd(),
        approval_policy=runtime_context["approval_policy"],
        version=get_version_display(),
    )


def print_help(cli_renderer: CliRenderer = renderer) -> None:
    """输出交互模式内建命令。"""
    cli_renderer.print_help()


def print_tools(
    runner: AgentRunner,
    runtime_context: dict[str, object] | None = None,
    cli_renderer: CliRenderer = renderer,
) -> None:
    """输出默认工具清单。"""
    if runtime_context is not None:
        ensure_runtime_capabilities(runtime_context, runner)
    tool_support = _load_tool_support()
    cli_renderer.print_tools(tool_support["describe_tool_instances"](runner.tools))


def describe_capability_lines(capability_registry: CapabilityRegistry) -> list[str]:
    """将动态 capability 摘要压缩为适合状态面板展示的文本行。"""
    capabilities = capability_registry.list_capabilities()
    if not capabilities:
        return ["无"]

    lines: list[str] = []
    for capability in capabilities:
        lines.append(
            f"{capability.name} | kind={capability.kind} | "
            f"tool={str(capability.register_as_tool).lower()} | "
            f"status={capability.status} | rev={capability.revision}"
        )
    return lines


def print_status(
    runner: AgentRunner,
    runtime_context: dict[str, object],
    cli_renderer: CliRenderer = renderer,
) -> None:
    """输出便于试用排障的运行状态。"""
    ensure_runtime_capabilities(runtime_context, runner)
    tool_support = _load_tool_support()
    capability_registry = runtime_context["capability_registry"]
    mcp_client = runtime_context.get("mcp_client")
    context_diagnostics = runner.get_context_diagnostics()
    api_key_configured = (
        "已配置"
        if runner.api_key and runner.api_key != "sk-default"
        else "未配置或仍为默认占位值"
    )
    saved_allowed_path_lines = "\n".join(
        tool_support["describe_allowed_roots"](runtime_context["saved_allowed_paths"])
    ) or "无"
    allowed_root_lines = "\n".join(tool_support["describe_allowed_roots"](runner.allowed_roots))
    registered_tool_lines = "\n".join(
        tool_support["describe_command_registry"](runner.command_registry)
    ) or "无"
    capability_lines = "\n".join(describe_capability_lines(capability_registry))
    cli_renderer.print_status(
        [
            ("版本", get_version_display()),
            ("模式", f"{get_mode_label(runner.mode)} ({runner.mode.value})"),
            (
                "审批策略",
                f"{get_approval_policy_label(runtime_context['approval_policy'])}"
                f" ({runtime_context['approval_policy'].value})",
            ),
            ("服务", runner.service),
            ("模型", runner.model_name),
            ("模型基址", str(runner.base_url or "默认")),
            ("工作目录", str(Path.cwd())),
            ("会话轮数", str(runner.get_turn_count())),
            ("默认工具数", str(len(runner.tools))),
            (
                "界面",
                f"{get_interaction_ui_mode_label(runtime_context['ui_mode'])}"
                f" ({runtime_context['ui_mode'].value})",
            ),
            ("当前会话 ID", str(runtime_context["session_id"])),
            ("动态能力数", str(len(capability_registry.list_capabilities()))),
            (
                "上下文消息",
                f"完整 {context_diagnostics['history_message_count']} / "
                f"模型可见 {context_diagnostics['model_message_count']}",
            ),
            ("已压缩历史消息数", str(context_diagnostics["compressed_message_count"])),
            ("GATEWAY_API_KEY", api_key_configured),
            ("本地配置文件", str(runtime_context["local_config_path"])),
            ("历史会话目录", str(runtime_context["session_storage_dir"])),
            ("已保存允许目录", saved_allowed_path_lines),
            ("允许读取根路径", allowed_root_lines),
            ("已注册外部工具", registered_tool_lines),
            ("动态能力", capability_lines),
            (
                "MCP 服务器",
                str(len(mcp_client._configs) if mcp_client is not None else 0),
            ),
            (
                "MCP 工具",
                str(len(mcp_client.tools) if mcp_client is not None else 0),
            ),
        ]
    )


def print_doctor_report(
    runner: AgentRunner,
    runtime_context: dict[str, object],
    cli_renderer: CliRenderer = renderer,
) -> None:
    """输出更接近真实环境检查的 doctor 诊断结果。"""
    ensure_runtime_capabilities(runtime_context, runner)
    _, build_doctor_rows = _load_doctor_support()
    cli_renderer.print_status(
        build_doctor_rows(runner, runtime_context),
        title="运行诊断",
    )


def print_allowed_roots(
    runner: AgentRunner,
    cli_renderer: CliRenderer = renderer,
) -> None:
    """输出当前会话允许访问的目录根路径。"""
    tool_support = _load_tool_support()
    cli_renderer.print_allowed_roots(
        tool_support["describe_allowed_roots"](runner.allowed_roots)
    )


def print_local_config(
    runtime_context: dict[str, object],
    cli_renderer: CliRenderer = renderer,
) -> None:
    """输出当前工作目录下的本地配置内容。"""
    tool_support = _load_tool_support()
    saved_allowed_path_lines = "\n".join(
        tool_support["describe_allowed_roots"](runtime_context["saved_allowed_paths"])
    ) or "无"
    cli_renderer.print_status(
        [
            ("本地配置文件", str(runtime_context["local_config_path"])),
            ("已保存允许目录", saved_allowed_path_lines),
            ("当前服务", str(runtime_context["service_name"])),
            ("当前模型", str(runtime_context["model_name"])),
            ("当前模型基址", str(runtime_context["base_url"] or "默认")),
        ]
    )


def add_allowed_path(
    raw_path: str,
    runner: AgentRunner,
    runtime_context: dict[str, object],
    cli_renderer: CliRenderer = renderer,
) -> None:
    """为当前会话动态增加允许访问目录，并同步刷新工具范围。"""
    if not raw_path.strip():
        raise ValueError("请提供要添加的目录路径。")

    added_path, was_added = runner.add_allowed_path(raw_path.strip())
    sync_runtime_context_from_runner(runtime_context, runner)

    if was_added:
        cli_renderer.print_info(f"已添加允许访问目录：{added_path}")
        return
    cli_renderer.print_info(f"目录已在允许访问范围内：{added_path}")


def add_persisted_allowed_path(
    raw_path: str,
    runner: AgentRunner,
    runtime_context: dict[str, object],
    cli_renderer: CliRenderer = renderer,
) -> None:
    """将目录持久化到本地配置，并同步更新当前会话。"""
    if not raw_path.strip():
        raise ValueError("请提供要保存的目录路径。")

    persisted_path, was_persisted, config_path = add_allow_path_to_local_config(
        raw_path.strip()
    )
    runtime_context["local_config_path"] = config_path
    runtime_context["saved_allowed_paths"] = list(
        load_local_cli_config().allow_paths
    )

    runner.register_allowed_path(persisted_path)
    sync_runtime_context_from_runner(runtime_context, runner)

    if was_persisted:
        if runner.mode is AgentMode.AUTHORIZED:
            cli_renderer.print_info(f"已写入本地配置并加入当前会话：{persisted_path}")
            return
        cli_renderer.print_info(
            f"已写入本地配置：{persisted_path}。切换到授权模式后会自动生效。"
        )
        return

    if runner.mode is AgentMode.AUTHORIZED:
        cli_renderer.print_info(f"目录已存在于本地配置和当前会话中：{persisted_path}")
        return
    cli_renderer.print_info(f"目录已存在于本地配置中：{persisted_path}")


def print_model_config(
    runner: AgentRunner,
    cli_renderer: CliRenderer = renderer,
) -> None:
    """输出当前会话正在使用的模型配置。"""
    cli_renderer.print_status(
        [
            ("当前服务", runner.service),
            ("当前模型", runner.model_name),
            ("当前模型基址", str(runner.base_url or "默认")),
        ]
    )


def switch_runtime_model(
    raw_model_name: str,
    runner: AgentRunner,
    runtime_context: dict[str, object],
    cli_renderer: CliRenderer = renderer,
) -> None:
    """在当前会话中切换模型名称。"""
    settings = _get_settings()
    normalized_model_name = settings.get_model_name(raw_model_name)
    runner.update_llm_config(model_name=normalized_model_name)
    sync_runtime_context_from_runner(runtime_context, runner)
    cli_renderer.print_info(
        f"已切换当前会话模型：{runner.service} / {runner.model_name}"
    )


def switch_runtime_service(
    raw_service_name: str,
    runner: AgentRunner,
    runtime_context: dict[str, object],
    cli_renderer: CliRenderer = renderer,
    *,
    base_url: str | None = None,
) -> None:
    """在当前会话中切换服务商，模型网关入口固定读取 GATEWAY_BASE_URL。"""
    settings = _get_settings()
    normalized_service_name = settings.normalize_service_name(raw_service_name)
    if base_url is not None and base_url.strip():
        cli_renderer.print_info(
            "模型基址固定使用 GATEWAY_BASE_URL，已忽略 /service 中的基址参数。"
        )
    runner.update_llm_config(
        service_name=normalized_service_name,
    )
    sync_runtime_context_from_runner(runtime_context, runner)
    cli_renderer.print_info(
        "已切换当前会话服务商："
        f"{runner.service}，模型：{runner.model_name}，基址：{runner.base_url or '默认'}"
    )


def start_new_runtime_session(
    runtime_context: dict[str, object],
    *,
    source_session_id: str | None = None,
) -> str:
    """为当前运行上下文分配新的会话标识，避免覆盖既有历史。"""
    session_id = create_runtime_session_id()
    runtime_context["session_id"] = session_id
    runtime_context["session_source_id"] = source_session_id
    runtime_context["_stop_input_buffer"] = ""
    return session_id


def start_fresh_visible_runtime_session(
    runtime_context: dict[str, object],
    *,
    source_session_id: str | None = None,
) -> str:
    """开始一个对用户可见也全新的会话窗口。"""
    session_id = start_new_runtime_session(
        runtime_context,
        source_session_id=source_session_id,
    )
    runtime_context["_recent_inputs"] = []
    runtime_context["__clear_visible_session"] = True
    try:
        session_store = _load_session_store_support()
        session_store["clear_interrupt_checkpoint"]()
    except Exception as exc:
        logger.warning("清理中断快照失败: %s", exc)
    return session_id


def _try_persist(
    runner: AgentRunner,
    runtime_context: dict[str, object],
    *,
    force: bool = False,
) -> None:
    """安全保存会话，失败时仅记录日志不影响主流程。"""
    try:
        persist_runtime_session(runner, runtime_context, force=force)
    except Exception as exc:
        from ..logging import log_warning
        log_warning("app", f"会话持久化失败：{exc}")


def _get_runtime_session_base_dir(
    runtime_context: dict[str, object],
) -> Path | None:
    """读取运行期指定的会话存储基准目录；CLI 默认使用当前目录发现规则。"""
    raw_base_dir = runtime_context.get("session_base_dir")
    if raw_base_dir is None:
        return None
    return Path(str(raw_base_dir)).expanduser()


def persist_runtime_session(
    runner: AgentRunner,
    runtime_context: dict[str, object],
    *,
    force: bool = False,
) -> Path | None:
    """按当前工作目录自动保存会话历史，供后续 /history 访问。"""
    history = runner.get_history_snapshot()
    if not force and len(history) <= 1 and runner.get_turn_count() == 0:
        return None

    session_store = _load_session_store_support()
    session_path = session_store["save_session_history"](
        str(runtime_context["session_id"]),
        history,
        mode=runner.mode.value,
        approval_policy=runtime_context["approval_policy"].value,
        source_session_id=runtime_context.get("session_source_id"),
        recent_inputs=runtime_context.get("_recent_inputs"),
        base_dir=_get_runtime_session_base_dir(runtime_context),
    )
    runtime_context["session_storage_dir"] = session_path.parent
    return session_path


def append_runtime_session_event(
    runtime_context: dict[str, object],
    event_type: str | AgentEventType,
    payload: object = None,
) -> Path | None:
    """把运行期事件追加到当前会话的 JSONL 事件流。"""
    session_id = runtime_context.get("session_id")
    if not session_id:
        return None
    try:
        session_store = _load_session_store_support()
        event_path = session_store["append_session_event"](
            str(session_id),
            str(event_type),
            payload=payload,
            base_dir=_get_runtime_session_base_dir(runtime_context),
        )
        runtime_context["session_event_log"] = event_path
        return event_path
    except Exception as exc:
        from ..logging import log_warning
        log_warning("app", f"会话事件落盘失败：{exc}")
        return None


def create_persisting_event_handler(
    runner: AgentRunner,
    runtime_context: dict[str, object],
    inner_handler: Any | None,
) -> Any:
    """包装运行器事件处理器：保留原展示逻辑，同时实时写事件和会话快照。"""
    semantic_events = {
        AgentEventType.TURN_START,
        AgentEventType.RESPONSE_END,
        AgentEventType.RESPONSE_RETRY,
        AgentEventType.TOOL_CALL,
        AgentEventType.TOOL_RESULT,
        AgentEventType.APPROVAL_REQUEST,
        AgentEventType.APPROVAL_RESULT,
        AgentEventType.TURN_END,
        AgentEventType.HISTORY_UPDATED,
    }

    def handler(event_type: str | AgentEventType, payload: object) -> None:
        if inner_handler is not None:
            inner_handler(event_type, payload)
        try:
            normalized_event: str | AgentEventType = AgentEventType(event_type)
        except ValueError:
            normalized_event = str(event_type)
        if normalized_event in semantic_events:
            append_runtime_session_event(runtime_context, normalized_event, payload)
        if normalized_event == AgentEventType.HISTORY_UPDATED:
            _try_persist(runner, runtime_context, force=True)

    return handler


def _save_interrupt_checkpoint(
    runner: AgentRunner,
    runtime_context: dict[str, object],
) -> None:
    """会话异常中断时保存续传快照，下次启动可恢复。"""
    try:
        session_store = _load_session_store_support()
        session_store["save_interrupt_checkpoint"](
            str(runtime_context["session_id"]),
            runner.get_history_snapshot(),
            mode=runner.mode.value,
            approval_policy=runtime_context["approval_policy"].value,
        )
    except Exception as exc:
        from ..logging import log_warning
        log_warning("app", f"中断快照保存失败：{exc}")


def _resolve_resume_session(
    runtime_context: dict[str, object],
) -> tuple[str, list[BaseMessage], str, str] | None:
    """检查是否存在可续传的中断快照，返回 (session_id, messages, mode, approval_policy) 或 None。"""
    session_store = _load_session_store_support()
    checkpoint = session_store["load_interrupt_checkpoint"]()
    if checkpoint is None:
        return None

    try:
        raw_messages = checkpoint.get("messages", [])
        if not isinstance(raw_messages, list) or not raw_messages:
            return None
        from langchain_core.messages import messages_from_dict
        messages = messages_from_dict(raw_messages)
    except Exception as exc:
        from ..logging import log_warning
        log_warning("app", f"中断快照消息反序列化失败：{exc}")
        return None

    session_id = str(checkpoint.get("session_id", ""))
    mode = str(checkpoint.get("mode", "standard"))
    approval_policy = str(checkpoint.get("approval_policy", "prompt"))
    return session_id, messages, mode, approval_policy


def _has_pending_checkpoint() -> bool:
    """是否存在待恢复的中断快照。"""
    session_store = _load_session_store_support()
    return session_store["has_interrupt_checkpoint"]()


def _try_auto_resume(runtime_context: dict[str, object]) -> None:
    """尝试从最近的中断快照恢复会话并进入交互循环。"""
    result = _resolve_resume_session(runtime_context)
    if result is None:
        renderer.print_info("未检测到可恢复的中断会话。")
        renderer.print_info("正在初始化新会话...")
        run_chat_loop(None, runtime_context)
        return

    session_id, messages, saved_mode, saved_policy = result
    from ..agent.approval import parse_approval_policy
    from ..agent.mode import parse_agent_mode

    try:
        target_mode = parse_agent_mode(saved_mode)
        target_policy = parse_approval_policy(saved_policy)
    except ValueError:
        renderer.print_error("快照中的模式或审批策略无效，无法恢复。")
        return

    runner = create_runner(runtime_context)
    runner.switch_mode(target_mode)
    try:
        runner.restore_history(messages)
    except ValueError as exc:
        renderer.print_error(f"快照恢复失败：{exc}")
        return

    runtime_context["approval_policy"] = target_policy
    sync_runtime_context_from_runner(runtime_context, runner)
    start_new_runtime_session(runtime_context, source_session_id=session_id)
    session_store = _load_session_store_support()
    session_store["clear_interrupt_checkpoint"]()

    renderer.print_info(
        f"已恢复中断会话 {session_id}，"
        f"模式={saved_mode}，消息数={len(messages)}。"
    )
    run_chat_loop(runner, runtime_context, show_banner=True)


def _format_context_message(message: BaseMessage, index: int) -> str:
    """将消息压缩为适合终端浏览的一行上下文摘要。"""
    AIMessage, HumanMessage, SystemMessage, ToolMessage = _load_message_type_support()
    role_label = "系统"
    if isinstance(message, HumanMessage):
        role_label = "用户"
    elif isinstance(message, AIMessage):
        role_label = "助手"
    elif isinstance(message, ToolMessage):
        role_label = f"工具({message.name or 'unknown'})"
    elif isinstance(message, SystemMessage):
        role_label = "系统"

    _, extract_text_content = _load_agent_runner_support()
    content = extract_text_content(message.content).strip()
    if isinstance(message, AIMessage) and message.tool_calls and not content:
        content = f"工具调用：{json.dumps(message.tool_calls, ensure_ascii=False)}"
    if not content:
        content = "（空内容）"

    single_line_content = " ".join(
        part.strip() for part in content.splitlines() if part.strip()
    )
    if len(single_line_content) > 180:
        single_line_content = f"{single_line_content[:180]}..."
    return f"{index}. {role_label}: {single_line_content}"


def build_context_preview(
    messages: list[BaseMessage],
    *,
    limit: int | None = 12,
) -> str:
    """构建当前上下文或历史会话的文本预览。"""
    if not messages:
        return "当前上下文为空。"

    preview_messages = messages if limit is None else messages[-limit:]
    lines: list[str] = []
    if limit is not None and len(messages) > len(preview_messages):
        lines.append(f"... 已省略更早的 {len(messages) - len(preview_messages)} 条消息")

    start_index = len(messages) - len(preview_messages) + 1
    for offset, message in enumerate(preview_messages, start=start_index):
        lines.append(_format_context_message(message, offset))
    return "\n".join(lines)


def print_context(
    runner: AgentRunner,
    runtime_context: dict[str, object],
    cli_renderer: CliRenderer = renderer,
) -> None:
    """显示当前内存上下文，便于确认模型本轮能读取到的历史。"""
    history_messages = runner.get_history_snapshot()
    model_messages = runner.get_model_context_snapshot()
    diagnostics = runner.get_context_diagnostics()
    messages = history_messages
    cli_renderer.print_status(
        [
            ("当前会话 ID", str(runtime_context["session_id"])),
            ("消息数", str(len(messages))),
            ("用户轮数", str(runner.get_turn_count())),
            (
                "来源会话",
                str(runtime_context.get("session_source_id") or "无"),
            ),
        ]
    )
    cli_renderer.print_chat_message("system", build_context_preview(messages))
    if diagnostics["compressed_summary"]:
        cli_renderer.print_chat_message(
            "system",
            "压缩摘要\n" + str(diagnostics["compressed_summary"]),
        )
    cli_renderer.print_chat_message(
        "system",
        "模型实际可见上下文预览\n"
        + build_context_preview(model_messages, limit=None),
    )


def print_history_list(
    runtime_context: dict[str, object],
    cli_renderer: CliRenderer = renderer,
) -> None:
    """列出当前工作目录下可访问的历史会话摘要。"""
    session_store = _load_session_store_support()
    stored_sessions = session_store["list_stored_sessions"]()
    if not stored_sessions:
        cli_renderer.print_info("当前工作目录下还没有已保存的历史会话。")
        return

    rows = []
    for summary in stored_sessions:
        detail_lines = [
            f"更新时间: {summary.updated_at}",
            (
                f"模式: {summary.mode} | 审批: {summary.approval_policy}"
                f" | 轮数: {summary.turn_count} | 消息: {summary.message_count}"
            ),
            f"标题: {summary.title}",
        ]
        if summary.source_session_id:
            detail_lines.append(f"来源: {summary.source_session_id}")
        rows.append((summary.session_id, "\n".join(detail_lines)))

    cli_renderer.print_status(rows, title="历史会话")


def print_history_search_results(
    query: str,
    cli_renderer: CliRenderer = renderer,
) -> None:
    """按关键词检索历史会话，便于从长会话中快速定位线索。"""
    session_store = _load_session_store_support()
    search_results = session_store["search_stored_sessions"](query)
    if not search_results:
        cli_renderer.print_info(f"未检索到包含关键词 `{query}` 的历史会话。")
        return

    rows: list[tuple[str, str]] = []
    for result in search_results:
        detail_lines = [
            f"更新时间: {result.updated_at}",
            (
                f"模式: {result.mode} | 审批: {result.approval_policy}"
                f" | 命中消息: {result.matched_message_count}"
            ),
            f"标题: {result.title}",
        ]
        if result.source_session_id:
            detail_lines.append(f"来源: {result.source_session_id}")
        if result.excerpts:
            detail_lines.append("命中片段:")
            detail_lines.extend(result.excerpts)
        rows.append((result.session_id, "\n".join(detail_lines)))

    cli_renderer.print_status(rows, title=f"历史检索: {query}")


def show_history_session(
    session_id: str,
    cli_renderer: CliRenderer = renderer,
) -> None:
    """显示指定历史会话的完整内容。"""
    session_store = _load_session_store_support()
    stored_session = session_store["load_session_history"](session_id)
    cli_renderer.print_status(
        [
            ("会话 ID", stored_session.summary.session_id),
            ("创建时间", stored_session.summary.created_at),
            ("更新时间", stored_session.summary.updated_at),
            ("模式", stored_session.summary.mode),
            ("审批策略", stored_session.summary.approval_policy),
            ("消息数", str(stored_session.summary.message_count)),
            ("用户轮数", str(stored_session.summary.turn_count)),
            ("来源会话", str(stored_session.summary.source_session_id or "无")),
        ]
    )
    cli_renderer.print_chat_message(
        "system",
        build_context_preview(stored_session.messages, limit=None),
    )


def load_history_session_into_runner(
    session_id: str,
    runner: AgentRunner,
    runtime_context: dict[str, object],
    cli_renderer: CliRenderer = renderer,
) -> None:
    """将历史会话恢复进当前上下文，并作为新会话继续演进。"""
    session_store = _load_session_store_support()
    stored_session = session_store["load_session_history"](session_id)
    target_mode = parse_agent_mode(stored_session.summary.mode)
    target_approval_policy = parse_approval_policy(
        stored_session.summary.approval_policy
    )

    runner.switch_mode(target_mode)
    runner.restore_history(stored_session.messages)
    runtime_context["approval_policy"] = target_approval_policy
    sync_runtime_context_from_runner(runtime_context, runner)
    # 恢复该历史会话的用户输入记录（用于上下键导航）
    if stored_session.recent_inputs:
        runtime_context["_recent_inputs"] = list(stored_session.recent_inputs)
    start_new_runtime_session(
        runtime_context,
        source_session_id=stored_session.summary.session_id,
    )
    cli_renderer.print_info(
        f"已加载历史会话：{stored_session.summary.session_id}。"
        "后续继续对话时会保存为新的会话副本。"
    )


def export_history_session(
    session_id: str,
    raw_output_path: str | None,
    cli_renderer: CliRenderer = renderer,
) -> None:
    """导出指定历史会话为更适合排查的 Markdown 或 JSON 文件。"""
    target_path = Path(raw_output_path).expanduser() if raw_output_path else None
    session_store = _load_session_store_support()
    exported_path = session_store["export_session_history"](
        session_id,
        output_path=target_path,
    )
    cli_renderer.print_info(f"已导出历史会话：{session_id} -> {exported_path}")


from .agent_executor import (
    _consume_stop_input_nonblocking,
    _reset_stop_input_buffer,
    create_approval_handler,
    create_cli_background_approval_handler,
    handle_pending_cli_approval_request,
    request_running_task_stop,
    run_agent_turn_with_stop_support,
)


def render_agent_event(event_type: str, payload: object) -> None:
    """将运行器事件映射为富文本展示。"""
    if event_type == AgentEventType.TURN_START:
        renderer.print_turn_start()
        return
    if event_type == AgentEventType.RESPONSE_BEGIN:
        renderer.begin_response_stream()
        return
    if event_type == AgentEventType.REASONING_TOKEN:
        renderer.append_reasoning_token(str(payload))
        return
    if event_type == AgentEventType.RESPONSE_TOKEN:
        renderer.append_response_token(str(payload))
        return
    if event_type == AgentEventType.RESPONSE_END:
        if isinstance(payload, dict):
            renderer.end_response_stream(
                str(payload.get("content", "")),
                bool(payload.get("has_tool_calls", False)),
            )
        return
    if event_type == AgentEventType.TOOL_CALL:
        renderer.print_tool_call(payload if isinstance(payload, list) else [])
        return
    if event_type == AgentEventType.APPROVAL_REQUEST:
        if isinstance(payload, dict):
            renderer.print_approval_request(payload)
        return
    if event_type == AgentEventType.APPROVAL_RESULT:
        if isinstance(payload, dict):
            renderer.print_approval_result(payload)
        return
    if event_type == AgentEventType.TURN_END:
        if isinstance(payload, dict):
            renderer.print_token_usage(payload)
        return
    if event_type == AgentEventType.TOOL_RESULT:
        if isinstance(payload, dict):
            renderer.print_tool_result(str(payload.get("content", "")))
        else:
            renderer.print_tool_result(str(payload))
        return


def handle_builtin_command(
    user_input: str,
    runner: AgentRunner,
    runtime_context: dict[str, object],
    cli_renderer: CliRenderer = renderer,
) -> bool | None:
    """处理交互模式下的内建命令，委托到命令注册表分发。"""
    from .builtin_commands import dispatch_builtin_command

    return dispatch_builtin_command(user_input, runner, runtime_context, cli_renderer)


def capture_builtin_command_output(
    user_input: str,
    runner: AgentRunner,
    runtime_context: dict[str, object],
    *,
    styled: bool = True,
) -> tuple[bool | None, str]:
    """执行内建命令并捕获文本结果，供其他界面复用。"""

    capture_console = Console(record=True, width=100)
    capture_renderer = CliRenderer(console=capture_console)
    result = handle_builtin_command(
        user_input,
        runner,
        runtime_context,
        capture_renderer,
    )
    output = capture_console.export_text(styles=styled).strip()
    return result, output


def capture_builtin_command_renderables(
    user_input: str,
    runner: AgentRunner,
    runtime_context: dict[str, object],
) -> tuple[bool | None, list[RenderableType]]:
    """执行内建命令并保留原始 Rich 渲染对象，供 TUI 复用 CLI 面板。"""

    capture_renderer = BuiltinCommandCaptureRenderer()
    result = handle_builtin_command(
        user_input,
        runner,
        runtime_context,
        capture_renderer,
    )
    return result, capture_renderer.renderables


def run_chat_loop(
    runner: AgentRunner | None,
    runtime_context: dict[str, object],
    show_banner: bool = True,
) -> None:
    """运行类似 Claude Code 的交互式命令行循环。"""
    ui_mode = runtime_context.get("ui_mode", InteractionUiMode.AUTO)
    if show_banner and sys.stdout.isatty():
        renderer.clear_screen()

    if ui_mode is InteractionUiMode.TUI:
        if runner is None:
            runner = create_runner(runtime_context)
        try:
            from .tui import launch_textual_chat
        except (ModuleNotFoundError, RuntimeError) as exc:
            renderer.print_error(f"TUI 启动失败，已回退到 CLI：{exc}")
        else:
            launch_textual_chat(runner, runtime_context, show_banner=show_banner)
            return

    if (
        ui_mode is InteractionUiMode.AUTO
        and sys.stdin.isatty()
        and sys.stdout.isatty()
    ):
        if runner is None:
            runner = create_runner(runtime_context)
        try:
            from .tui import launch_textual_chat
        except (ModuleNotFoundError, RuntimeError):
            pass
        else:
            launch_textual_chat(runner, runtime_context, show_banner=show_banner)
            return

    if show_banner:
        renderer.print_startup_splash()
        if runner is None:
            print_runtime_banner(runtime_context)
        else:
            print_banner(runner, runtime_context)

    while True:
        try:
            # 状态栏已由 prompt_toolkit bottom_toolbar 承载，不再单独 print
            user_input = prompt_chat_input().strip()
        except (Abort, EOFError, KeyboardInterrupt):
            renderer.print_info("\n👋 再见！")
            break

        if not user_input:
            continue

        renderer.print_user_message(user_input)
        if user_input.strip().lower() in EXIT_COMMANDS:
            renderer.print_info("\n👋 再见！")
            break
        if runner is None:
            runner = create_runner(runtime_context)
            _try_persist(runner, runtime_context, force=True)
        builtin_result = handle_builtin_command(user_input, runner, runtime_context)
        if builtin_result is False:
            break
        if builtin_result is True:
            runtime_context.pop("__clear_visible_session", None)
            continue

        # 注入待处理的文件内容到用户消息
        pending_files = {
            k: v for k, v in runtime_context.items()
            if k.startswith("__pending_file_")
        }
        if pending_files:
            file_context_parts = ["以下是用户通过 /file 命令加载的文件内容：\n"]
            for key, file_info in pending_files.items():
                fmt = file_info.get("lang", "")
                file_context_parts.append(
                    f"### 文件：{file_info['path']}\n"
                    f"```{fmt}\n{file_info['content']}\n```\n"
                )
                del runtime_context[key]
            file_context_parts.append(f"---\n用户问题：{user_input}")
            user_input = "\n".join(file_context_parts)

        try:
            recent = runtime_context.setdefault("_recent_inputs", [])
            recent.append(user_input)
            if len(recent) > 50:
                recent.pop(0)
            append_runtime_session_event(
                runtime_context,
                "user_input_received",
                {"input": user_input},
            )
            _try_persist(runner, runtime_context, force=True)

            # 判断是否使用多 Agent 编排
            multi_setting = runtime_context.get("multi_agent_enabled", "auto")
            if multi_setting is True or (
                multi_setting == "auto" and _detect_task_complexity(user_input)
            ):
                _run_multi_agent_turn(user_input, runner, runtime_context)
            elif sys.stdin.isatty() and sys.stdout.isatty():
                run_agent_turn_with_stop_support(
                    runner,
                    user_input,
                    runtime_context,
                    event_handler=create_persisting_event_handler(
                        runner,
                        runtime_context,
                        render_agent_event,
                    ),
                )
            else:
                runner.run(
                    user_input,
                    verbose=False,
                    event_handler=create_persisting_event_handler(
                        runner,
                        runtime_context,
                        render_agent_event,
                    ),
                    approval_handler=create_approval_handler(runtime_context),
                )

            _try_persist(runner, runtime_context)
        except KeyboardInterrupt:
            # Ctrl+C 中断：保存现场后退出循环
            _try_persist(runner, runtime_context)
            _save_interrupt_checkpoint(runner, runtime_context)
            renderer.print_info("\n任务已中断。使用 --resume 恢复。")
            break
        except ExecutionInterruptedError as exc:
            _try_persist(runner, runtime_context)
            _save_interrupt_checkpoint(runner, runtime_context)
            renderer.print_info(str(exc))
        except Exception as exc:
            _try_persist(runner, runtime_context)
            _save_interrupt_checkpoint(runner, runtime_context)
            renderer.print_error(f"运行失败：{exc}")


def prompt_chat_input() -> str:
    """为纯 CLI 模式读取一行输入，优先使用支持补全的终端提示器。"""

    global _cli_prompt_session, _prompt_toolkit_disabled

    if (
        not _prompt_toolkit_disabled
        and sys.stdin.isatty()
        and sys.stdout.isatty()
    ):
        try:
            from .prompting import CliPromptSession, PROMPT_TOOLKIT_IMPORT_ERROR
        except ModuleNotFoundError:
            _prompt_toolkit_disabled = True
        else:
            if PROMPT_TOOLKIT_IMPORT_ERROR is None:
                try:
                    if _cli_prompt_session is None:
                        # 状态栏回调：每次刷新时读取 renderer 的累计 token
                        def status_provider():
                            t = renderer._cumulative_input_tokens
                            o = renderer._cumulative_output_tokens
                            c = renderer._cumulative_cost
                            return (
                                f"累计 ↑{t} ↓{o} Σ{t+o} │ ¥{c:.4f}"
                            )
                        _cli_prompt_session = CliPromptSession(
                            status_provider=status_provider,
                        )
                    return _cli_prompt_session.prompt()
                except Exception as exc:  # noqa: BLE001 - 终端兼容失败时需要自动降级
                    _prompt_toolkit_disabled = True
                    renderer.print_error(f"CLI 补全已降级为基础输入：{exc}")

    return typer.prompt("›")


def _load_hub_support():
    """按需加载 Hub 支持，避免普通命令导入额外桥接代码。"""
    from ..hub import CyberAgentHub, HubEvent, HubTaskSource

    return CyberAgentHub, HubEvent, HubTaskSource


def build_hub(
    runtime_context: dict[str, object],
    *,
    base_dir: Path | None = None,
):
    """创建持有唯一 AgentRunner 的本地 Hub。"""
    CyberAgentHub, _, _ = _load_hub_support()
    if base_dir is not None:
        runtime_context["session_base_dir"] = base_dir
        runtime_context["session_storage_dir"] = get_runtime_session_storage_dir(base_dir)
    ensure_runtime_capabilities(runtime_context)
    runner = create_runner(runtime_context)
    _try_persist(runner, runtime_context, force=True)
    return CyberAgentHub(
        runner=runner,
        runtime_context=runtime_context,
        approval_handler_factory=create_approval_handler,
        detect_task_complexity=_detect_task_complexity,
        run_multi_agent_turn=_run_multi_agent_turn,
        renderless_event_handler_factory=create_persisting_event_handler,
        base_dir=base_dir,
    )


def _normalize_hub_agent_event(event_type: str) -> AgentEventType | None:
    try:
        return AgentEventType(event_type)
    except ValueError:
        return None


def subscribe_cli_to_hub(
    hub,
    *,
    show_status: bool = False,
    render_remote_events: bool = False,
) -> Any:
    """把 Hub 事件渲染到当前 CLI。"""

    def _subscriber(event) -> None:
        source_kind = getattr(event.source, "kind", "") if event.source is not None else ""
        normalized_event = _normalize_hub_agent_event(event.type)
        if normalized_event is not None:
            if source_kind and source_kind != "cli" and not render_remote_events:
                return
            render_agent_event(normalized_event, event.payload)
            return

        source_label = ""
        if event.source is not None:
            source_label = f"[{event.source.kind}] "
        payload = event.payload if isinstance(event.payload, dict) else {}
        if event.type == "hub_started":
            if show_status:
                renderer.print_info(f"Hub 会话：{payload.get('session_id', 'unknown')}")
        elif event.type == "hub_stopped":
            if show_status:
                renderer.print_info(f"Hub 已停止，会话：{payload.get('session_id', 'unknown')}")
        elif event.type == "task_queued":
            if not show_status:
                return
            renderer.print_info(
                f"{source_label}任务已入队，队列长度：{payload.get('queue_size', 0)}"
            )
        elif event.type == "task_started":
            if not show_status:
                return
            renderer.print_info(f"{source_label}开始执行：{payload.get('text', '')}")
        elif event.type == "task_finished":
            if not show_status:
                return
            renderer.print_info(f"{source_label}任务完成，会话：{payload.get('session_id', '')}")
        elif event.type == "task_interrupted":
            renderer.print_info(str(payload.get("message", "当前任务已被停止。")))
        elif event.type == "task_error":
            renderer.print_error(str(payload.get("message", "Hub 任务执行失败。")))
        elif event.type == "task_stop_requested":
            if show_status:
                renderer.print_info(str(payload.get("reason", "已请求停止当前任务。")))
        elif event.type == "session_switched":
            if show_status:
                renderer.print_info(
                    f"{source_label}已切换会话：{payload.get('session_id', '')}"
                )
        elif event.type == "session_switch_failed":
            renderer.print_error(str(payload.get("reason", "会话切换失败。")))

    return hub.subscribe(_subscriber)


class HubFeishuBridge:
    """把 Hub 事件桥接到飞书，并把飞书输入提交给 Hub。"""

    def __init__(
        self,
        *,
        hub: Any,
        route: Any,
        runtime_context: dict[str, object],
        base_dir: Path | None,
        reply_timeout_seconds: float,
        broadcast_chat_ids: list[str] | None = None,
    ) -> None:
        from .webhook import WebhookGateway

        self.hub = hub
        self.route = route
        self.gateway = WebhookGateway(
            [route],
            runtime_context,
            create_runner,
            cli_renderer=renderer,
            base_dir=base_dir,
            reply_timeout_seconds=reply_timeout_seconds,
        )
        self._known_events_by_chat: dict[str, Any] = {}
        self._task_targets: dict[int, list[Any]] = {}
        self._progress_emitters: dict[tuple[int, str], Any] = {}
        self._synthetic_counter = 0
        self._lock = threading.RLock()
        self._delivery_executor = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="cyber-agent-hub-feishu",
        )
        for chat_id in broadcast_chat_ids or []:
            self._register_known_chat_event(chat_id)

    def consume_event(self, event) -> Any:
        from .webhook import FEISHU_CREATE_API_MODE, build_json_http_response
        _, _, HubTaskSource = _load_hub_support()

        event.metadata.setdefault("feishu_delivery_mode", FEISHU_CREATE_API_MODE)
        chat_id = str(event.metadata.get("chat_id", "")).strip()
        if chat_id:
            self._remember_chat_event(chat_id, event)
            self._persist_known_chat_id(chat_id)
        self.hub.submit(
            event.text,
            source=HubTaskSource(
                "feishu",
                event.sender_name or event.sender_id,
                {
                    "event": event,
                    "chat_id": chat_id,
                    "message_id": event.message_id,
                    "route_path": self.route.path,
                },
            ),
        )
        return build_json_http_response(
            {
                "status": "queued",
                "provider": event.provider,
                "session_id": self.hub.session_id,
            }
        )

    def __call__(self, event) -> None:
        normalized_event = _normalize_hub_agent_event(event.type)
        if event.type == "task_started":
            self._start_task(event)
            return
        if normalized_event is not None:
            self._broadcast_progress(event, normalized_event)
            return
        if event.type.startswith("pipeline."):
            self._broadcast_progress(event, event.type)
            return
        if event.type == "task_finished":
            self._finish_task(event, str(self._payload(event).get("reply_text", "")).strip())
            return
        if event.type == "task_interrupted":
            message = str(
                self._payload(event).get("reply_text")
                or self._payload(event).get("message")
                or "当前任务已被停止。"
            )
            self._finish_task(event, message)
            return
        if event.type == "task_error":
            message = str(
                self._payload(event).get("reply_text")
                or self._payload(event).get("message")
                or "Hub 任务执行失败。"
            )
            self._finish_task(event, message)
            return
        if event.type == "task_stop_requested":
            self._send_control_notice(event, "已收到 /stop，正在请求停止当前任务。")
            return
        if event.type == "session_switched":
            session_id = str(self._payload(event).get("session_id", ""))
            self._send_control_notice(event, f"已切换会话：{session_id}")
            return
        if event.type == "session_switch_failed":
            reason = str(self._payload(event).get("reason", "会话切换失败。"))
            self._send_control_notice(event, reason)

    @staticmethod
    def _payload(event) -> dict[str, object]:
        return event.payload if isinstance(event.payload, dict) else {}

    def _task_key(self, event) -> int:
        return id(event.source) if event.source is not None else 0

    def _source_feishu_event(self, event) -> Any | None:
        metadata = getattr(event.source, "metadata", {}) if event.source is not None else {}
        candidate = metadata.get("event") if isinstance(metadata, dict) else None
        if getattr(candidate, "provider", None) == "feishu":
            return candidate
        return None

    def _start_task(self, event) -> None:
        targets = self._resolve_targets(event, create_for_broadcast=True)
        if not targets:
            return
        key = self._task_key(event)
        with self._lock:
            self._task_targets[key] = targets
        user_input = str(self._payload(event).get("text", ""))
        for target in targets:
            emitter = self._get_progress_emitter(key, target)
            emitter.start(user_input)

    def _broadcast_progress(
        self,
        event,
        event_type: str | AgentEventType,
    ) -> None:
        key = self._task_key(event)
        with self._lock:
            targets = list(self._task_targets.get(key, []))
        if not targets:
            return
        for target in targets:
            self._get_progress_emitter(key, target)(event_type, event.payload)

    def _finish_task(self, event, reply_text: str) -> None:
        key = self._task_key(event)
        with self._lock:
            targets = self._task_targets.pop(key, [])
        if not targets:
            targets = self._resolve_targets(event, create_for_broadcast=False)
        if not targets:
            return
        reply_text = reply_text.strip() or "（空回复）"
        for target in targets:
            self._close_progress_emitter(key, target)
            self._deliver_reply(target, reply_text)

    def _send_control_notice(self, event, message: str) -> None:
        for target in self._resolve_targets(event, create_for_broadcast=True):
            self._deliver_reply(target, message, rich=False)

    def _resolve_targets(self, event, *, create_for_broadcast: bool) -> list[Any]:
        source_event = self._source_feishu_event(event)
        if source_event is not None:
            return [source_event]
        with self._lock:
            known_events = list(self._known_events_by_chat.values())
        if not create_for_broadcast:
            return known_events
        return [self._synthetic_event(base_event, event) for base_event in known_events]

    def _synthetic_event(self, base_event, event):
        from .webhook import FEISHU_CREATE_API_MODE, WebhookEvent

        with self._lock:
            self._synthetic_counter += 1
            counter = self._synthetic_counter
        metadata = dict(base_event.metadata)
        metadata["feishu_delivery_mode"] = FEISHU_CREATE_API_MODE
        chat_id = str(metadata.get("chat_id", "")).strip()
        return WebhookEvent(
            provider="feishu",
            session_key=base_event.session_key,
            sender_id=base_event.sender_id,
            sender_name=base_event.sender_name,
            message_id=f"hub-{self.hub.session_id}-{counter}",
            text=str(self._payload(event).get("text", "")),
            reply_webhook_url=base_event.reply_webhook_url,
            metadata={**metadata, "chat_id": chat_id},
        )

    def _get_progress_emitter(self, task_key: int, event):
        from .webhook_feishu import FeishuProgressMessageEmitter

        emitter_key = (task_key, event.message_id)
        with self._lock:
            emitter = self._progress_emitters.get(emitter_key)
            if emitter is not None:
                return emitter
            emitter = FeishuProgressMessageEmitter(
                lambda step, step_index, target=event: self._delivery_executor.submit(
                    self.gateway._emit_feishu_progress_message,
                    self.route,
                    target,
                    step,
                    step_index,
                )
            )
            self._progress_emitters[emitter_key] = emitter
            return emitter

    def _close_progress_emitter(self, task_key: int, event) -> None:
        emitter_key = (task_key, event.message_id)
        with self._lock:
            emitter = self._progress_emitters.pop(emitter_key, None)
        if emitter is not None:
            emitter.close()

    def _deliver_reply(self, event, reply_text: str, *, rich: bool = True) -> None:
        from .webhook import WEBHOOK_PROVIDER_ADAPTERS
        from .webhook_models import WebhookAgentReply
        from .webhook_feishu import _build_feishu_ai_reply_payload

        def _send() -> None:
            try:
                payload_override = (
                    _build_feishu_ai_reply_payload(reply_text)
                    if rich
                    else None
                )
                self.gateway._deliver_reply(
                    self.route,
                    WEBHOOK_PROVIDER_ADAPTERS[self.route.provider],
                    event,
                    WebhookAgentReply(
                        session_id=self.hub.session_id,
                        reply_text=reply_text,
                        reply_payload_override=payload_override,
                    ),
                )
            except Exception as exc:  # noqa: BLE001 - 飞书投递失败不应影响 Hub 主循环
                renderer.print_error(
                    "Hub 飞书消息发送失败："
                    f"chat_id={event.metadata.get('chat_id', '') or 'unknown'} "
                    f"reason={exc}"
                )

        self._delivery_executor.submit(_send)

    def close(self) -> None:
        self._delivery_executor.shutdown(wait=False, cancel_futures=True)

    def has_broadcast_targets(self) -> bool:
        with self._lock:
            return bool(self._known_events_by_chat)

    def _remember_chat_event(self, chat_id: str, event: Any) -> None:
        with self._lock:
            self._known_events_by_chat.setdefault(chat_id, event)

    def _register_known_chat_event(self, chat_id: str) -> None:
        from .webhook import FEISHU_CREATE_API_MODE, WebhookEvent

        normalized_chat_id = chat_id.strip()
        if not normalized_chat_id:
            return
        event = WebhookEvent(
            provider="feishu",
            session_key=normalized_chat_id,
            sender_id="hub",
            sender_name="Cyber Agent Hub",
            message_id=f"hub-default-{normalized_chat_id}",
            text="",
            metadata={
                "chat_id": normalized_chat_id,
                "message_type": "hub_default_target",
                "feishu_delivery_mode": FEISHU_CREATE_API_MODE,
            },
        )
        self._remember_chat_event(normalized_chat_id, event)

    def _persist_known_chat_id(self, chat_id: str) -> None:
        from .webhook_feishu import (
            _build_feishu_session_entry,
            _load_feishu_session_state,
            _save_feishu_session_state,
        )

        normalized_chat_id = chat_id.strip()
        if not normalized_chat_id:
            return
        try:
            state_payload = _load_feishu_session_state(self.gateway.base_dir)
            chats_payload = state_payload.get("chats")
            if not isinstance(chats_payload, dict):
                chats_payload = {}
                state_payload["chats"] = chats_payload
            chat_state = chats_payload.get(normalized_chat_id)
            if not isinstance(chat_state, dict):
                chat_state = {}
            chat_state.setdefault("active_session_key", normalized_chat_id)
            sessions = chat_state.get("sessions")
            if not isinstance(sessions, list):
                sessions = []
            if not any(
                isinstance(entry, dict)
                and str(entry.get("session_key", "")).strip() == normalized_chat_id
                for entry in sessions
            ):
                sessions.insert(0, _build_feishu_session_entry(normalized_chat_id))
            chat_state["sessions"] = sessions
            chats_payload[normalized_chat_id] = chat_state
            _save_feishu_session_state(state_payload, self.gateway.base_dir)
        except Exception as exc:  # noqa: BLE001 - 记录 chat_id 失败不应影响消息处理
            logger.warning("保存飞书 chat_id 失败: %s", exc)


def _parse_feishu_broadcast_chat_ids(
    route: Any,
    option_values: list[str] | None,
    *,
    base_dir: Path | None,
) -> list[str]:
    raw_values: list[str] = []
    raw_values.extend(option_values or [])
    provider_options = getattr(route, "provider_options", {}) or {}
    for key in ("hub_broadcast_chat_id", "hub_broadcast_chat_ids"):
        value = str(provider_options.get(key, "")).strip()
        if value:
            raw_values.append(value)
    raw_values.extend(_load_persisted_feishu_chat_ids(base_dir))

    chat_ids: list[str] = []
    for raw_value in raw_values:
        for item in str(raw_value).replace("\n", ",").split(","):
            chat_id = item.strip()
            if chat_id and chat_id not in chat_ids:
                chat_ids.append(chat_id)
    return chat_ids


def _load_persisted_feishu_chat_ids(base_dir: Path | None) -> list[str]:
    from .webhook_feishu import _load_feishu_session_state

    try:
        state_payload = _load_feishu_session_state(base_dir)
    except Exception as exc:  # noqa: BLE001 - 自动发现失败不影响 Hub 启动
        logger.warning("加载已知飞书 chat_id 失败: %s", exc)
        return []
    chats_payload = state_payload.get("chats")
    if not isinstance(chats_payload, dict):
        return []
    chat_ids = [
        str(chat_id).strip()
        for chat_id in chats_payload.keys()
        if str(chat_id).strip()
    ]
    return chat_ids


class QuietInfoRenderer:
    """用于后台前端：吞掉 info，只把错误转给主 CLI。"""

    def __init__(self, inner: CliRenderer) -> None:
        self._inner = inner

    def print_info(self, content: str) -> None:
        _ = content

    def print_error(self, content: str) -> None:
        self._inner.print_error(content)


def _is_hub_control_command(user_input: str) -> bool:
    normalized = user_input.strip().lower()
    return (
        normalized == "/stop"
        or normalized in {"/new", "/clear", "/session new"}
        or normalized.startswith("/session use ")
        or normalized.startswith("/session load ")
    )


def run_hub_cli_loop(
    hub,
    *,
    subscribe: bool = True,
    show_status: bool = False,
    render_remote_events: bool = False,
) -> None:
    """运行 Hub 模式下的 CLI 前端。"""
    _, _, HubTaskSource = _load_hub_support()
    unsubscribe = (
        subscribe_cli_to_hub(
            hub,
            show_status=show_status,
            render_remote_events=render_remote_events,
        )
        if subscribe
        else None
    )
    try:
        while True:
            try:
                user_input = prompt_chat_input().strip()
            except (Abort, EOFError, KeyboardInterrupt):
                renderer.print_info("\n👋 再见！")
                break
            if not user_input:
                continue
            renderer.print_user_message(user_input)
            if user_input.strip().lower() in EXIT_COMMANDS:
                renderer.print_info("\n👋 再见！")
                break

            if _is_hub_control_command(user_input):
                hub.submit(user_input, source=HubTaskSource("cli", "cli"))
                while not hub.wait_until_idle(0.2):
                    pass
                continue

            builtin_result = handle_builtin_command(
                user_input,
                hub.runner,
                hub.runtime_context,
            )
            if builtin_result is False:
                break
            if builtin_result is True:
                hub.runtime_context.pop("__clear_visible_session", None)
                continue

            hub.submit(user_input, source=HubTaskSource("cli", "cli"))
            while not hub.wait_until_idle(0.2):
                pass
    finally:
        if unsubscribe is not None:
            unsubscribe()


@hub_app.command("serve")
def hub_serve(
    ctx: typer.Context,
    feishu_config_path: str | None = typer.Option(
        None,
        "--feishu-config",
        help="可选：飞书长连接复用的 webhook JSON 配置文件路径。",
    ),
    feishu_route_path: str | None = typer.Option(
        None,
        "--feishu-path",
        help="当配置中存在多条 feishu 路由时，用于指定要复用的路由路径。",
    ),
    storage_dir: str | None = typer.Option(
        None,
        "--storage-dir",
        help="Hub 会话历史落盘使用的工作根目录；省略时默认使用当前工作目录。",
    ),
    reply_timeout_seconds: float = typer.Option(
        DEFAULT_WEBHOOK_REPLY_TIMEOUT_SECONDS,
        "--reply-timeout-seconds",
        help="调用飞书官方回复接口时的超时时间（秒）。",
    ),
    no_cli: bool = typer.Option(
        False,
        "--no-cli",
        help="只启动 Hub/飞书前端，不进入本地 CLI 输入循环。",
    ),
    multi_agent: str = typer.Option(
        "off",
        "--multi-agent",
        help="Hub 多 Agent 编排策略，可选 off、auto、on；默认 off 以保持唯一 runner。",
    ),
    subtask_concurrency: str = typer.Option(
        "auto",
        "--subtask-concurrency",
        help="四柱子任务并发策略，可选 off、auto、force；默认 auto。",
    ),
    max_subagents: int = typer.Option(
        4,
        "--max-subagents",
        help="四柱子任务最多并发的子 Agent 数量；1 表示实际顺序执行。",
    ),
    execution_profile: str = typer.Option(
        "auto",
        "--execution-profile",
        help="四柱执行姿态，可选 auto、conservative、aggressive；auto 会根据授权参数推导。",
    ),
    benchmark_profile: str = typer.Option(
        "off",
        "--benchmark-profile",
        help="Benchmark 跑分策略，可选 off、auto、aggressive；aggressive 会启用限时止损和切题约束。",
    ),
    benchmark_target_score: int = typer.Option(
        0,
        "--benchmark-target-score",
        help="Benchmark aggressive 目标分数；例如 2000 会让规划优先按约 10 道 easy 题冲分。",
    ),
    feishu_broadcast_chat_ids: list[str] | None = typer.Option(
        None,
        "--feishu-broadcast-chat-id",
        help="CLI 任务默认同步通知的飞书 chat_id，可重复传入；也可写入 provider_options.hub_broadcast_chat_ids。",
    ),
    feishu_connect_timeout_seconds: float = typer.Option(
        15.0,
        "--feishu-connect-timeout-seconds",
        help="进入 CLI 交互前等待飞书长连接建立的最长秒数。",
    ),
    hub_verbose: bool = typer.Option(
        False,
        "--hub-verbose",
        help="显示 Hub/飞书连接和队列状态；默认静默以避免打断 CLI 交互。",
    ),
    render_remote_events: bool = typer.Option(
        False,
        "--render-remote-events",
        help="在本地 CLI 中渲染飞书端触发的 Agent 输出；默认不渲染以避免打断输入。",
    ),
) -> None:
    """
    启动 Cyber Agent Hub，让 CLI 和飞书共享同一个 runner、队列和会话。
    """
    runtime_context = get_or_build_runtime_context(ctx)
    normalized_multi_agent = multi_agent.strip().lower()
    if normalized_multi_agent not in {"off", "auto", "on"}:
        renderer.print_error("--multi-agent 仅支持 off、auto、on。")
        raise typer.Exit(code=1)
    normalized_subtask_concurrency = subtask_concurrency.strip().lower()
    if normalized_subtask_concurrency not in {"off", "auto", "force"}:
        renderer.print_error("--subtask-concurrency 仅支持 off、auto、force。")
        raise typer.Exit(code=1)
    if max_subagents < 1:
        renderer.print_error("--max-subagents 必须大于等于 1。")
        raise typer.Exit(code=1)
    normalized_execution_profile = execution_profile.strip().lower()
    if normalized_execution_profile not in {"auto", "conservative", "aggressive"}:
        renderer.print_error("--execution-profile 仅支持 auto、conservative、aggressive。")
        raise typer.Exit(code=1)
    normalized_benchmark_profile = benchmark_profile.strip().lower()
    if normalized_benchmark_profile not in {"off", "auto", "aggressive"}:
        renderer.print_error("--benchmark-profile 仅支持 off、auto、aggressive。")
        raise typer.Exit(code=1)
    if benchmark_target_score < 0:
        renderer.print_error("--benchmark-target-score 必须大于等于 0。")
        raise typer.Exit(code=1)
    runtime_context["multi_agent_enabled"] = {
        "off": False,
        "auto": "auto",
        "on": True,
    }[normalized_multi_agent]
    runtime_context["subtask_concurrency"] = normalized_subtask_concurrency
    runtime_context["max_subagents"] = max_subagents
    runtime_context["execution_profile"] = normalized_execution_profile
    runtime_context["resolved_execution_profile"] = infer_execution_profile(runtime_context)
    runtime_context["benchmark_profile"] = normalized_benchmark_profile
    runtime_context["benchmark_target_score"] = benchmark_target_score
    resolved_storage_dir = (
        Path(storage_dir).expanduser().resolve()
        if storage_dir is not None
        else None
    )
    hub = build_hub(runtime_context, base_dir=resolved_storage_dir)
    unsubscribe_cli = subscribe_cli_to_hub(
        hub,
        show_status=hub_verbose,
        render_remote_events=render_remote_events,
    )
    feishu_bridge: HubFeishuBridge | None = None
    renderer.print_startup_splash()
    print_banner(hub.runner, runtime_context)
    hub.start()
    feishu_thread: threading.Thread | None = None
    try:
        if feishu_config_path is not None:
            select_feishu_long_connection_route, serve_feishu_long_connection = (
                _load_feishu_long_connection_support()
            )
            webhook_support = _load_webhook_support()
            routes = webhook_support["load_webhook_routes_from_file"](feishu_config_path)
            resolved_route = select_feishu_long_connection_route(
                routes,
                feishu_route_path,
            )
            broadcast_chat_ids = _parse_feishu_broadcast_chat_ids(
                resolved_route,
                feishu_broadcast_chat_ids,
                base_dir=resolved_storage_dir,
            )
            feishu_bridge = HubFeishuBridge(
                hub=hub,
                route=resolved_route,
                runtime_context=runtime_context,
                base_dir=resolved_storage_dir,
                reply_timeout_seconds=reply_timeout_seconds,
                broadcast_chat_ids=broadcast_chat_ids,
            )
            hub.subscribe(feishu_bridge)
            feishu_connected_event = threading.Event()

            def _serve_feishu() -> None:
                try:
                    feishu_renderer = renderer if hub_verbose else QuietInfoRenderer(renderer)
                    serve_feishu_long_connection(
                        resolved_route,
                        runtime_context,
                        create_runner,
                        cli_renderer=feishu_renderer,
                        base_dir=resolved_storage_dir,
                        reply_timeout_seconds=reply_timeout_seconds,
                        event_consumer=feishu_bridge.consume_event,
                        connected_event=feishu_connected_event,
                    )
                except Exception as exc:  # noqa: BLE001 - 后台前端失败需要显式提示
                    renderer.print_error(f"Hub 飞书前端退出：{exc}")

            feishu_thread = threading.Thread(
                target=_serve_feishu,
                name="cyber-agent-hub-feishu",
                daemon=True,
            )
            feishu_thread.start()
            if hub_verbose:
                renderer.print_info("Hub 飞书前端已启动，正在等待长连接建立。")
            if feishu_connected_event.wait(max(0.0, feishu_connect_timeout_seconds)):
                if hub_verbose:
                    if feishu_bridge.has_broadcast_targets():
                        renderer.print_info(
                            f"飞书同步通知已启用，目标会话数：{len(broadcast_chat_ids)}。"
                        )
                    else:
                        renderer.print_info(
                            "飞书长连接已就绪。首次请先在飞书里给机器人发一条消息，"
                            "Hub 会自动记住 chat_id；之后 CLI 任务会同步到飞书。"
                        )
            else:
                renderer.print_error(
                    "飞书长连接尚未确认建立，仍将进入 CLI；连接完成后飞书端会继续可用。"
                )

        if no_cli:
            if hub_verbose:
                renderer.print_info("Hub 正在后台运行，按 Ctrl+C 停止。")
            while True:
                time.sleep(1.0)
        else:
            run_hub_cli_loop(
                hub,
                subscribe=False,
                show_status=hub_verbose,
                render_remote_events=render_remote_events,
            )
    except (ModuleNotFoundError, ValueError) as exc:
        renderer.print_error(f"运行失败：{exc}")
        raise typer.Exit(code=1) from exc
    except KeyboardInterrupt:
        if hub_verbose:
            renderer.print_info("\nHub 已收到退出请求。")
    finally:
        unsubscribe_cli()
        if feishu_bridge is not None:
            feishu_bridge.close()
        hub.stop()


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        callback=_print_version_and_exit,
        is_eager=True,
        help="显示当前 CLI 版本并退出。",
    ),
    mode: str = typer.Option(
        AgentMode.STANDARD.value,
        "--mode",
        help="运行模式，可选 standard 或 authorized。",
    ),
    allow_paths: list[str] | None = typer.Option(
        None,
        "--allow-path",
        help="授权模式下额外允许读取的路径根目录，可重复传入。",
    ),
    tool_specs: list[str] | None = typer.Option(
        None,
        "--tool",
        help="授权模式下注册外部工具，格式为 name=absolute_path，可重复传入。",
    ),
    approval_policy: str = typer.Option(
        ApprovalPolicy.PROMPT.value,
        "--approval-policy",
        help="高风险工具调用的审批策略，可选 prompt、auto、never。",
    ),
    ui: str = typer.Option(
        InteractionUiMode.AUTO.value,
        "--ui",
        help="界面模式，可选 auto、tui、cli。",
    ),
    skill_dir: list[str] | None = typer.Option(
        None,
        "--skill-dir",
        help="额外 skill 目录，可重复传入以加载多个目录下的 SKILL.md。",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="检测并恢复上次中断的会话。",
    ),
    auto_decision: bool = typer.Option(
        False,
        "--auto-decision",
        help="多 Agent 模式下自动评估并选择子任务，无需手动交互确认。",
    ),
) -> None:
    """默认无子命令时直接进入交互式对话。"""
    ctx.ensure_object(dict)
    try:
        parsed_mode = parse_agent_mode(mode)
        parsed_approval_policy = parse_approval_policy(approval_policy)
        parsed_ui_mode = parse_interaction_ui_mode(ui)
        ctx.obj["runtime_options"] = {
            "mode": parsed_mode,
            "allow_paths": allow_paths,
            "tool_specs": tool_specs,
            "approval_policy": parsed_approval_policy,
            "ui_mode": parsed_ui_mode,
            "skill_dirs": skill_dir,
            "auto_decision": auto_decision,
        }
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if ctx.invoked_subcommand is None:
        runtime_context = get_or_build_runtime_context(ctx)
        if resume:
            _try_auto_resume(runtime_context)
            return
        # 检测到中断快照时自动提示
        if _has_pending_checkpoint():
            renderer.print_info(
                "检测到上次会话异常中断的快照。使用 --resume 恢复，"
                "或 /history load 选择其他会话。"
            )
        renderer.print_info("正在初始化会话，首次模型调用时会按需加载工具。")
        run_chat_loop(
            None,
            runtime_context,
        )


@app.command()
def chat(
    ctx: typer.Context,
    message: str | None = typer.Option(
        None,
        "--message",
        "-m",
        help="直接执行一轮对话，不进入交互模式。",
    ),
) -> None:
    """
    进入交互式聊天模式，或执行单轮对话。
    """
    runtime_context = get_or_build_runtime_context(ctx)
    if message is not None:
        renderer.print_info("正在初始化运行器，首次启动可能需要数秒。")
        runner = create_runner(runtime_context)
        try:
            if runner.llm is None:
                renderer.print_info("正在初始化模型客户端，首次请求可能需要数十秒。")
            runtime_context.setdefault("_recent_inputs", []).append(message)
            append_runtime_session_event(
                runtime_context,
                "user_input_received",
                {"input": message},
            )
            persist_runtime_session(runner, runtime_context, force=True)
            runner.run(
                message,
                verbose=False,
                event_handler=create_persisting_event_handler(
                    runner,
                    runtime_context,
                    render_agent_event,
                ),
                approval_handler=create_approval_handler(runtime_context),
            )
        except ModuleNotFoundError as exc:
            renderer.print_error(f"运行失败：{exc}")
            raise typer.Exit(code=1) from exc
        finally:
            persist_runtime_session(runner, runtime_context)
        return
    run_chat_loop(None, runtime_context)


@app.command()
def run(
    ctx: typer.Context,
    message: str = typer.Argument(..., help="要发送给智能体的一轮消息。"),
) -> None:
    """
    执行单轮对话，适合脚本或快速试验。
    """
    runtime_context = get_or_build_runtime_context(ctx)
    runner = create_runner(runtime_context)
    try:
        if runner.llm is None:
            renderer.print_info("正在初始化模型客户端，首次请求可能需要数十秒。")
        runtime_context.setdefault("_recent_inputs", []).append(message)
        append_runtime_session_event(
            runtime_context,
            "user_input_received",
            {"input": message},
        )
        persist_runtime_session(runner, runtime_context, force=True)
        runner.run(
            message,
            verbose=False,
            event_handler=create_persisting_event_handler(
                runner,
                runtime_context,
                render_agent_event,
            ),
            approval_handler=create_approval_handler(runtime_context),
        )
    except ModuleNotFoundError as exc:
        renderer.print_error(f"运行失败：{exc}")
        raise typer.Exit(code=1) from exc
    finally:
        persist_runtime_session(runner, runtime_context)


@app.command()
def tools(ctx: typer.Context) -> None:
    """
    查看当前默认启用的工具列表。
    """
    runtime_context = get_or_build_runtime_context(ctx)
    runner = create_runner(runtime_context)
    print_tools(runner, runtime_context)


@app.command()
def skills(ctx: typer.Context) -> None:
    """
    列出当前已加载的 SKILLS.md 技能。
    """
    runtime_context = get_or_build_runtime_context(ctx)
    file_skills = runtime_context.get("file_skills", [])
    if not file_skills:
        renderer.print_info("当前没有加载任何 SKILLS.md 技能。")
        renderer.print_info(
            "将 SKILL.md 放入 .skills/<skill-name>/ 或 "
            "~/.skills/<skill-name>/ 目录即可自动加载。"
        )
        return

    rows = []
    for skill in file_skills:
        detail_lines = [
            f"名称: {skill.name}",
            f"描述: {skill.description}",
            f"路径: {skill.source_path}",
        ]
        if skill.version:
            detail_lines.append(f"版本: {skill.version}")
        if skill.model:
            detail_lines.append(f"模型: {skill.model}")
        if skill.allowed_tools:
            detail_lines.append(f"允许工具: {', '.join(skill.allowed_tools)}")
        rows.append((skill.name, "\n".join(detail_lines)))

    renderer.print_status(rows, title="已加载技能")


@app.command()
def mcp(ctx: typer.Context) -> None:
    """
    查看当前已连接的 MCP 服务器和工具。
    """
    runtime_context = get_or_build_runtime_context(ctx)
    mcp_client = runtime_context.get("mcp_client")
    if mcp_client is None:
        renderer.print_info("当前未连接任何 MCP 服务器。")
        renderer.print_info(
            "在项目根目录创建 .mcp.json，"
            '配置 mcpServers 字段即可自动连接。'
        )
        return

    configs = getattr(mcp_client, "_configs", [])
    if not configs:
        renderer.print_info("当前没有配置任何 MCP 服务器。")
        return

    rows: list[tuple[str, str]] = []

    for config in configs:
        detail_lines = [
            f"命令: {config.command} {' '.join(config.args)}",
        ]
        if config.env:
            detail_lines.append(
                f"环境变量: {', '.join(config.env.keys())}"
            )
        rows.append((config.name, "\n".join(detail_lines)))

    renderer.print_status(rows, title="MCP 服务器")

    tools = mcp_client.tools
    if tools:
        tool_rows = []
        for t in tools:
            tool_rows.append(
                (t.tool_name, f"[{t.server_name}] {t.description}")
            )
        renderer.print_status(tool_rows, title="MCP 工具")


@history_app.callback(invoke_without_command=True)
def history_callback(ctx: typer.Context) -> None:
    """默认列出当前工作目录下的历史会话。"""
    if ctx.invoked_subcommand is not None:
        return
    print_history_list(get_or_build_runtime_context(ctx))


@history_app.command("show")
def history_show(
    ctx: typer.Context,
    session_id: str = typer.Argument(..., help="要查看的历史会话 ID。"),
) -> None:
    """
    显示指定历史会话的完整内容。
    """
    show_history_session(session_id)


@history_app.command("search")
def history_search(
    ctx: typer.Context,
    query: str = typer.Argument(..., help="用于检索历史会话的关键词。"),
) -> None:
    """
    按关键词检索当前工作目录下的历史会话。
    """
    print_history_search_results(query)


@history_app.command("export")
def history_export(
    ctx: typer.Context,
    session_id: str = typer.Argument(..., help="要导出的历史会话 ID。"),
    output_path: str | None = typer.Argument(
        None,
        help="导出目标路径；省略时默认导出到历史会话目录。",
    ),
) -> None:
    """
    将指定历史会话导出为 Markdown 或 JSON 文件。
    """
    export_history_session(session_id, output_path)


@webhook_app.command("example-config")
def webhook_example_config(
    output_path: str | None = typer.Argument(
        None,
        help="可选的输出文件路径；省略时直接打印到标准输出。",
    ),
) -> None:
    """
    输出 webhook 路由示例配置，便于对接第三方平台或中继网关。
    """
    webhook_support = _load_webhook_support()
    serialized_config = json.dumps(
        webhook_support["build_webhook_example_config"](),
        ensure_ascii=False,
        indent=2,
    )
    if output_path is None:
        typer.echo(serialized_config)
        return

    resolved_output_path = Path(output_path).expanduser().resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(serialized_config + "\n", encoding="utf-8")
    renderer.print_info(f"已写出 webhook 示例配置：{resolved_output_path}")


@webhook_app.command("serve")
def webhook_serve(
    ctx: typer.Context,
    host: str = typer.Option(
        DEFAULT_WEBHOOK_HOST,
        "--host",
        help="Webhook HTTP 服务监听地址。",
    ),
    port: int = typer.Option(
        DEFAULT_WEBHOOK_PORT,
        "--port",
        help="Webhook HTTP 服务监听端口。",
    ),
    providers: list[str] | None = typer.Option(
        None,
        "--provider",
        help=(
            "使用默认路由时启用的第三方平台，可重复传入；"
            f"可选值：{', '.join(SUPPORTED_WEBHOOK_PROVIDERS)}。"
        ),
    ),
    config_path: str | None = typer.Option(
        None,
        "--config",
        help="Webhook JSON 配置文件路径；提供后优先使用配置文件中的 routes。",
    ),
    storage_dir: str | None = typer.Option(
        None,
        "--storage-dir",
        help="Webhook 会话历史落盘使用的工作根目录；省略时默认使用当前工作目录。",
    ),
    reply_timeout_seconds: float = typer.Option(
        DEFAULT_WEBHOOK_REPLY_TIMEOUT_SECONDS,
        "--reply-timeout-seconds",
        help="向第三方 reply webhook 回包时的超时时间（秒）。",
    ),
) -> None:
    """
    启动 webhook HTTP 服务，将第三方消息桥接到当前智能体会话。
    """
    runtime_context = get_or_build_runtime_context(ctx)
    webhook_support = _load_webhook_support()
    if config_path is not None:
        routes = webhook_support["load_webhook_routes_from_file"](config_path)
    else:
        routes = webhook_support["build_default_webhook_routes"](providers)

    resolved_storage_dir = (
        Path(storage_dir).expanduser().resolve()
        if storage_dir is not None
        else None
    )
    webhook_support["serve_webhook_gateway"](
        host,
        port,
        routes,
        runtime_context,
        create_runner,
        cli_renderer=renderer,
        base_dir=resolved_storage_dir,
        reply_timeout_seconds=reply_timeout_seconds,
    )


@webhook_app.command("serve-feishu-long-connection")
def webhook_serve_feishu_long_connection(
    ctx: typer.Context,
    config_path: str = typer.Option(
        ...,
        "--config",
        help="飞书长连接使用的 webhook JSON 配置文件路径。",
    ),
    route_path: str | None = typer.Option(
        None,
        "--path",
        help="当配置中存在多条 feishu 路由时，用于指定要复用的路由路径。",
    ),
    storage_dir: str | None = typer.Option(
        None,
        "--storage-dir",
        help="长连接会话历史落盘使用的工作根目录；省略时默认使用当前工作目录。",
    ),
    reply_timeout_seconds: float = typer.Option(
        DEFAULT_WEBHOOK_REPLY_TIMEOUT_SECONDS,
        "--reply-timeout-seconds",
        help="调用飞书官方回复接口时的超时时间（秒）。",
    ),
) -> None:
    """
    启动飞书官方 SDK 长连接客户端，无需公网回调地址即可接收消息并回复。
    """
    runtime_context = get_or_build_runtime_context(ctx)
    try:
        select_feishu_long_connection_route, serve_feishu_long_connection = (
            _load_feishu_long_connection_support()
        )
        webhook_support = _load_webhook_support()
        routes = webhook_support["load_webhook_routes_from_file"](config_path)
        resolved_route = select_feishu_long_connection_route(routes, route_path)
        resolved_storage_dir = (
            Path(storage_dir).expanduser().resolve()
            if storage_dir is not None
            else None
        )
        serve_feishu_long_connection(
            resolved_route,
            runtime_context,
            create_runner,
            cli_renderer=renderer,
            base_dir=resolved_storage_dir,
            reply_timeout_seconds=reply_timeout_seconds,
        )
    except (ModuleNotFoundError, ValueError) as exc:
        renderer.print_error(f"运行失败：{exc}")
        raise typer.Exit(code=1) from exc


@app.command(name="doctor")
def doctor(
    ctx: typer.Context,
    json_output: bool = typer.Option(
        False,
        "--json",
        help="以 JSON 输出诊断结果，便于脚本和 CI 使用。",
    ),
) -> None:
    """
    检查当前 CLI 运行所依赖的关键配置。
    """
    runtime_context = get_or_build_runtime_context(ctx)
    runner = create_runner(runtime_context)
    if json_output:
        build_doctor_payload, _ = _load_doctor_support()
        typer.echo(
            json.dumps(
                build_doctor_payload(runner, runtime_context),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    print_doctor_report(runner, runtime_context)


@app.command()
def ide(
    mode: str = typer.Option(
        AgentMode.STANDARD.value,
        "--mode",
        help="运行模式，可选 standard 或 authorized。",
    ),
    allow_paths: list[str] | None = typer.Option(
        None,
        "--allow-path",
        help="授权模式下额外允许读取的路径根目录，可重复传入。",
    ),
    approval_policy: str = typer.Option(
        ApprovalPolicy.PROMPT.value,
        "--approval-policy",
        help="高风险工具调用的审批策略，可选 prompt、auto、never。",
    ),
    service: str | None = typer.Option(
        None,
        "--service",
        help="模型服务商，可选 opencode、openai、deepseek、claude、mimo、baisub。",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="模型名称。",
    ),
    dev: bool = typer.Option(
        False,
        "--dev",
        help="开发模式，跳过前端构建检查。",
    ),
) -> None:
    """
    启动桌面 IDE（Electron + FastAPI 后端）。
    """
    from .ide_launcher import launch_ide

    try:
        parse_agent_mode(mode)
        parse_approval_policy(approval_policy)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    exit_code = launch_ide(
        mode=mode,
        allow_paths=allow_paths,
        approval_policy=approval_policy,
        service_name=service,
        model_name=model,
        dev=dev,
    )
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


@app.command()
def version() -> None:
    """
    输出当前 CLI 版本。
    """
    typer.echo(f"cyber-agent-cli {get_version_display()}")


def main() -> None:
    """提供给 python -m cyber_agent 的统一入口。"""
    app()
