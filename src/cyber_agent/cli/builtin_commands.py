"""内建命令注册表，替代 app.py 中的巨型 if/elif 链。

每个命令处理器接收:
- tokens: list[str] — 分词后的用户输入
- runner: AgentRunner — 当前运行器实例
- runtime_context: dict — 运行时上下文
- cli_renderer: CliRenderer — 终端渲染器

返回: True=继续会话, False=退出, None=非内建命令
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..agent.approval import (
    ApprovalPolicy,
    get_approval_policy_label,
    parse_approval_policy,
)
from ..agent.mode import AgentMode, get_mode_label, parse_agent_mode
from ..local_config import (
    add_allow_path_to_local_config,
    load_local_cli_config,
)
from .interactive import (
    EXIT_COMMANDS,
    get_interaction_ui_mode_label,
)

if TYPE_CHECKING:
    from ..agent.runner import AgentRunner
    from ..capability_registry import CapabilityRegistry
    from .render import CliRenderer

# ── 命令处理器类型 ──

CommandHandler = Callable[
    ["AgentRunner", dict[str, object], "CliRenderer", list[str], str],
    bool | None,
]

# ── 独立子命令的模块目标 ──

_COMMAND_ROUTES: dict[str, str] = {
    "/history": "history",
    "/config": "config",
}


def _dispatch_to_module(tokens: list[str]) -> str | None:
    """解析是否需要将子命令路由到独立处理模块。"""
    if not tokens:
        return None
    return _COMMAND_ROUTES.get(tokens[0])


# ── 单级命令处理器 ──

def _handle_stop(
    runner: AgentRunner,
    runtime_context: dict[str, object],
    cli_renderer: CliRenderer,
    tokens: list[str],
    raw_input: str,
) -> bool | None:
    from .app import request_running_task_stop

    request_running_task_stop(runtime_context, cli_renderer)
    return True


def _handle_help(
    runner: AgentRunner,
    runtime_context: dict[str, object],
    cli_renderer: CliRenderer,
    tokens: list[str],
    raw_input: str,
) -> bool | None:
    from .app import print_help
    print_help(cli_renderer)
    return True


def _handle_tools(
    runner: AgentRunner,
    runtime_context: dict[str, object],
    cli_renderer: CliRenderer,
    tokens: list[str],
    raw_input: str,
) -> bool | None:
    from .app import print_tools
    print_tools(runner, runtime_context, cli_renderer)
    return True


def _handle_context(
    runner: AgentRunner,
    runtime_context: dict[str, object],
    cli_renderer: CliRenderer,
    tokens: list[str],
    raw_input: str,
) -> bool | None:
    if len(tokens) >= 2 and tokens[1] == "clear":
        from .app import start_new_runtime_session
        runner.reset()
        start_new_runtime_session(runtime_context)
        cli_renderer.print_info("会话上下文已清空，并已开始新的会话。")
        return True
    from .app import print_context
    print_context(runner, runtime_context, cli_renderer)
    return True


def _handle_history(
    runner: AgentRunner,
    runtime_context: dict[str, object],
    cli_renderer: CliRenderer,
    tokens: list[str],
    raw_input: str,
) -> bool | None:
    from .app import (
        export_history_session,
        load_history_session_into_runner,
        print_history_list,
        print_history_search_results,
        show_history_session,
    )

    if len(tokens) < 2:
        print_history_list(runtime_context, cli_renderer)
        return True

    sub_cmd = tokens[1]
    # 从原始输入重建剩余参数
    args = tokens[2:] if len(tokens) > 2 else []

    if sub_cmd == "show":
        if not args:
            cli_renderer.print_error("请提供要查看的会话 ID。")
            return True
        try:
            show_history_session(args[0], cli_renderer)
        except ValueError as exc:
            cli_renderer.print_error(str(exc))
        return True

    if sub_cmd == "load":
        if not args:
            cli_renderer.print_error("请提供要加载的会话 ID。")
            return True
        try:
            load_history_session_into_runner(args[0], runner, runtime_context, cli_renderer)
        except ValueError as exc:
            cli_renderer.print_error(str(exc))
        return True

    if sub_cmd == "search":
        # 检索关键词可能包含空格，需要从原始输入提取
        search_prefix = "/history search "
        search_query = raw_input[len(search_prefix):].strip()
        if not search_query:
            cli_renderer.print_error("请提供要检索的关键词。")
            return True
        try:
            print_history_search_results(search_query, cli_renderer)
        except ValueError as exc:
            cli_renderer.print_error(str(exc))
        return True

    if sub_cmd == "export":
        # 用原始输入解析参数，避免空格路径被 token 拆分
        export_prefix = "/history export "
        export_args = raw_input[len(export_prefix):].strip()
        session_parts = export_args.split(maxsplit=1)
        session_id = session_parts[0].strip() if session_parts else ""
        output_path = session_parts[1].strip() if len(session_parts) == 2 else None
        if not session_id:
            cli_renderer.print_error("请提供要导出的会话 ID。")
            return True
        try:
            export_history_session(session_id, output_path, cli_renderer)
        except ValueError as exc:
            cli_renderer.print_error(str(exc))
        return True

    cli_renderer.print_error("不支持的 /history 子命令。")
    return True


def _handle_doctor(
    runner: AgentRunner,
    runtime_context: dict[str, object],
    cli_renderer: CliRenderer,
    tokens: list[str],
    raw_input: str,
) -> bool | None:
    from .app import print_doctor_report
    print_doctor_report(runner, runtime_context, cli_renderer)
    return True


def _handle_status(
    runner: AgentRunner,
    runtime_context: dict[str, object],
    cli_renderer: CliRenderer,
    tokens: list[str],
    raw_input: str,
) -> bool | None:
    from .app import print_status
    print_status(runner, runtime_context, cli_renderer)
    return True


def _handle_version(
    runner: AgentRunner,
    runtime_context: dict[str, object],
    cli_renderer: CliRenderer,
    tokens: list[str],
    raw_input: str,
) -> bool | None:
    from .. import __version__
    cli_renderer.print_info(f"cyber-agent-cli {__version__}")
    return True


def _handle_config(
    runner: AgentRunner,
    runtime_context: dict[str, object],
    cli_renderer: CliRenderer,
    tokens: list[str],
    raw_input: str,
) -> bool | None:
    from .app import (
        add_persisted_allowed_path,
        print_local_config,
    )

    if len(tokens) < 2 or (len(tokens) == 2 and tokens[1] == "allow-path"):
        print_local_config(runtime_context, cli_renderer)
        return True

    if tokens[1] == "allow-path":
        if len(tokens) >= 3 and tokens[2] == "add":
            # 路径可能包含空格，从原始输入提取
            path_prefix = "/config allow-path add "
            raw_path = raw_input[len(path_prefix):].strip()
            if not raw_path:
                cli_renderer.print_error("请提供要保存的目录路径。")
                return True
            try:
                add_persisted_allowed_path(raw_path, runner, runtime_context, cli_renderer)
            except ValueError as exc:
                cli_renderer.print_error(str(exc))
            return True
        cli_renderer.print_error("不支持的 /config allow-path 子命令。")
        return True

    cli_renderer.print_error("不支持的 /config 子命令。")
    return True


def _handle_service(
    runner: AgentRunner,
    runtime_context: dict[str, object],
    cli_renderer: CliRenderer,
    tokens: list[str],
    raw_input: str,
) -> bool | None:
    from .app import print_model_config, switch_runtime_service

    if len(tokens) < 2:
        print_model_config(runner, cli_renderer)
        return True

    base_url = tokens[2] if len(tokens) >= 3 else None
    try:
        switch_runtime_service(tokens[1], runner, runtime_context, cli_renderer, base_url=base_url)
    except ValueError as exc:
        cli_renderer.print_error(str(exc))
    return True


def _handle_model(
    runner: AgentRunner,
    runtime_context: dict[str, object],
    cli_renderer: CliRenderer,
    tokens: list[str],
    raw_input: str,
) -> bool | None:
    from .app import print_model_config, switch_runtime_model

    if len(tokens) < 2:
        print_model_config(runner, cli_renderer)
        return True

    try:
        switch_runtime_model(tokens[1], runner, runtime_context, cli_renderer)
    except ValueError as exc:
        cli_renderer.print_error(str(exc))
    return True


def _handle_allow_path(
    runner: AgentRunner,
    runtime_context: dict[str, object],
    cli_renderer: CliRenderer,
    tokens: list[str],
    raw_input: str,
) -> bool | None:
    from .app import add_allowed_path, print_allowed_roots

    if len(tokens) < 2:
        print_allowed_roots(runner, cli_renderer)
        return True

    if tokens[1] == "add":
        # 路径可能包含空格，从原始输入提取
        path_prefix = "/allow-path add "
        raw_path = raw_input[len(path_prefix):].strip()
        if not raw_path:
            cli_renderer.print_error("请提供要添加的目录路径。")
            return True
        try:
            add_allowed_path(raw_path, runner, runtime_context, cli_renderer)
        except ValueError as exc:
            cli_renderer.print_error(str(exc))
        return True

    # 兼容直接 /allow-path <路径> 的用法
    raw_path_prefix = "/allow-path "
    raw_path = raw_input[len(raw_path_prefix):].strip()
    if not raw_path:
        cli_renderer.print_error("请提供要添加的目录路径。")
        return True
    try:
        add_allowed_path(raw_path, runner, runtime_context, cli_renderer)
    except ValueError as exc:
        cli_renderer.print_error(str(exc))
    return True


def _handle_clear(
    runner: AgentRunner,
    runtime_context: dict[str, object],
    cli_renderer: CliRenderer,
    tokens: list[str],
    raw_input: str,
) -> bool | None:
    from .app import start_new_runtime_session
    runner.reset()
    start_new_runtime_session(runtime_context)
    cli_renderer.print_info("会话上下文已清空，并已开始新的会话。")
    return True


def _handle_mode(
    runner: AgentRunner,
    runtime_context: dict[str, object],
    cli_renderer: CliRenderer,
    tokens: list[str],
    raw_input: str,
) -> bool | None:
    from .app import start_new_runtime_session, sync_runtime_context_from_runner

    if len(tokens) < 2:
        cli_renderer.print_mode_notice(runner.mode, switched=False)
        return True

    try:
        target_mode = parse_agent_mode(tokens[1])
    except ValueError as exc:
        cli_renderer.print_error(str(exc))
        return True
    runner.switch_mode(target_mode)
    sync_runtime_context_from_runner(runtime_context, runner)
    start_new_runtime_session(runtime_context)
    cli_renderer.print_mode_notice(target_mode, switched=True)
    return True


def _handle_approval(
    runner: AgentRunner,
    runtime_context: dict[str, object],
    cli_renderer: CliRenderer,
    tokens: list[str],
    raw_input: str,
) -> bool | None:
    if len(tokens) < 2:
        cli_renderer.print_approval_policy_notice(
            runtime_context["approval_policy"],
            switched=False,
        )
        return True

    try:
        target_policy = parse_approval_policy(tokens[1])
    except ValueError as exc:
        cli_renderer.print_error(str(exc))
        return True
    runtime_context["approval_policy"] = target_policy
    cli_renderer.print_approval_policy_notice(target_policy, switched=True)
    return True


# ── 命令注册表 ──

_COMMAND_REGISTRY: dict[str, CommandHandler] = {
    "/stop": _handle_stop,
    "/help": _handle_help,
    "/tools": _handle_tools,
    "/context": _handle_context,
    "/history": _handle_history,
    "/doctor": _handle_doctor,
    "/status": _handle_status,
    "/version": _handle_version,
    "/config": _handle_config,
    "/service": _handle_service,
    "/model": _handle_model,
    "/allow-path": _handle_allow_path,
    "/clear": _handle_clear,
    "/mode": _handle_mode,
    "/approval": _handle_approval,
}


def dispatch_builtin_command(
    user_input: str,
    runner: AgentRunner,
    runtime_context: dict[str, object],
    cli_renderer: CliRenderer | None = None,
) -> bool | None:
    """按命令注册表分发内建命令，返回是否继续会话。"""
    from .render import CliRenderer
    if cli_renderer is None:
        from .app import renderer
        cli_renderer = renderer

    stripped = user_input.strip()
    normalized = stripped.lower()
    tokens = normalized.split()

    # 退出命令
    if normalized in EXIT_COMMANDS:
        cli_renderer.print_info("👋 再见！")
        return False

    if not tokens:
        return None

    # 按命令名查找处理器
    handler = _COMMAND_REGISTRY.get(tokens[0])
    if handler is None:
        return None

    return handler(runner, runtime_context, cli_renderer, tokens, stripped)
