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


# ── 公共辅助 ──


def _extract_arg(raw_input: str, prefix: str) -> str:
    """从原始输入中去掉命令前缀，返回剩余参数部分。"""
    return raw_input[len(prefix):].strip()


def _safe(fn, cli_renderer: CliRenderer) -> None:
    """执行可能出错的操作，自动捕获并渲染错误。"""
    try:
        fn()
    except (ValueError, OSError) as exc:
        cli_renderer.print_error(str(exc))


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
        _safe(lambda: show_history_session(args[0], cli_renderer), cli_renderer)
        return True

    if sub_cmd == "load":
        if not args:
            cli_renderer.print_error("请提供要加载的会话 ID。")
            return True
        _safe(lambda: load_history_session_into_runner(args[0], runner, runtime_context, cli_renderer), cli_renderer)
        return True

    if sub_cmd == "search":
        search_query = _extract_arg(raw_input, "/history search ")
        if not search_query:
            cli_renderer.print_error("请提供要检索的关键词。")
            return True
        _safe(lambda: print_history_search_results(search_query, cli_renderer), cli_renderer)
        return True

    if sub_cmd == "export":
        export_args = _extract_arg(raw_input, "/history export ")
        session_parts = export_args.split(maxsplit=1)
        session_id = session_parts[0].strip() if session_parts else ""
        output_path = session_parts[1].strip() if len(session_parts) == 2 else None
        if not session_id:
            cli_renderer.print_error("请提供要导出的会话 ID。")
            return True
        _safe(lambda: export_history_session(session_id, output_path, cli_renderer), cli_renderer)
        return True

    if sub_cmd == "recent":
        recent = runtime_context.get("_recent_inputs", [])
        if not recent:
            cli_renderer.print_info("暂无最近输入记录。")
            return True
        lines = [f"  {i+1:2d}. {inp[:100]}" for i, inp in enumerate(recent)]
        cli_renderer.print_info(f"最近 {len(recent)} 条输入：\n" + "\n".join(lines))
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

    if len(tokens) < 2:
        print_local_config(runtime_context, cli_renderer)
        return True

    if tokens[1] == "allow-path":
        # /config allow-path → 显示配置
        if len(tokens) == 2:
            print_local_config(runtime_context, cli_renderer)
            return True

        # /config allow-path add <路径> → 显式添加
        if tokens[2] == "add":
            raw_path = _extract_arg(raw_input, "/config allow-path add ")
            if not raw_path:
                cli_renderer.print_error("请提供要保存的目录路径。")
                return True
            _safe(lambda: add_persisted_allowed_path(raw_path, runner, runtime_context, cli_renderer), cli_renderer)
            return True

        # /config allow-path <路径> → 简写形式，等同 add
        # 从原始输入提取 "allow-path " 之后的内容作为路径
        raw_path = _extract_arg(raw_input, "/config allow-path ")
        if raw_path:
            _safe(lambda: add_persisted_allowed_path(raw_path, runner, runtime_context, cli_renderer), cli_renderer)
            return True
        cli_renderer.print_error("请提供要保存的目录路径，如 /config allow-path add /path 或 /config allow-path /path。")
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
        raw_path = _extract_arg(raw_input, "/allow-path add ")
        if not raw_path:
            cli_renderer.print_error("请提供要添加的目录路径。")
            return True
        _safe(lambda: add_allowed_path(raw_path, runner, runtime_context, cli_renderer), cli_renderer)
        return True

    # 兼容直接 /allow-path <路径> 的用法
    raw_path = _extract_arg(raw_input, "/allow-path ")
    if not raw_path:
        cli_renderer.print_error("请提供要添加的目录路径。")
        return True
    _safe(lambda: add_allowed_path(raw_path, runner, runtime_context, cli_renderer), cli_renderer)
    return True


def _handle_memory(
    runner: AgentRunner,
    runtime_context: dict[str, object],
    cli_renderer: CliRenderer,
    tokens: list[str],
    raw_input: str,
) -> bool | None:
    """管理跨会话持久化记忆。"""
    from ..memory import (
        delete_memory,
        load_all_memories,
        save_memory,
        search_memories,
    )

    if len(tokens) < 2:
        entries = load_all_memories()
        if not entries:
            cli_renderer.print_info("当前没有已保存的记忆。使用 /memory add 添加。")
            return True
        rows = [(e.name, f"[{e.memory_type}] {e.description}") for e in entries]
        cli_renderer.print_status(rows, title="持久化记忆")
        return True

    sub_cmd = tokens[1]
    if sub_cmd == "add" and len(tokens) >= 4:
        mem_name = tokens[2]
        mem_type = tokens[3]
        body_parts = _extract_arg(raw_input, "/memory add").split(maxsplit=2)
        body = body_parts[2] if len(body_parts) >= 3 else ""
        description = body[:120] if body else mem_name
        _safe(lambda: save_memory(mem_name, description, body, memory_type=mem_type)
              and cli_renderer.print_info(f"已保存记忆：{mem_name}"), cli_renderer)
        return True

    if sub_cmd == "search" and len(tokens) >= 3:
        query = _extract_arg(raw_input, "/memory search ")
        results = search_memories(query)
        if not results:
            cli_renderer.print_info(f"未找到与 `{query}` 相关的记忆。")
            return True
        rows = [
            (r.entry.name,
             f"[{r.entry.memory_type}] {r.entry.description}\n{r.excerpt}")
            for r in results
        ]
        cli_renderer.print_status(rows, title=f"记忆检索: {query}")
        return True

    if sub_cmd == "delete" and len(tokens) >= 3:
        deleted = delete_memory(tokens[2])
        cli_renderer.print_info(
            f"已删除记忆：{tokens[2]}" if deleted else f"未找到记忆：{tokens[2]}"
        )
        return True

    cli_renderer.print_error("用法: /memory [add <name> <type> <text> | search <query> | delete <name>]")
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


def _handle_auto_decision(
    runner: AgentRunner,
    runtime_context: dict[str, object],
    cli_renderer: CliRenderer,
    tokens: list[str],
    raw_input: str,
) -> bool | None:
    """切换多 Agent 自动决策模式。"""
    current = runtime_context.get("auto_decision", False)
    state_label = "已启用（思考者自动评估并选择子任务）" if current else "已禁用（展示菜单等待用户选择）"

    if len(tokens) == 1:
        cli_renderer.print_info(
            f"自动决策：{state_label}。使用 /auto-decision on|off 切换。"
        )
        return True

    toggle = tokens[1].lower()
    if toggle in ("on", "enable", "yes", "true"):
        runtime_context["auto_decision"] = True
        cli_renderer.print_info(
            "已启用自动决策。多 Agent 模式下将由思考者自动评估并选择子任务，跳过交互菜单。"
        )
    elif toggle in ("off", "disable", "no", "false"):
        runtime_context["auto_decision"] = False
        cli_renderer.print_info(
            "已禁用自动决策。多 Agent 模式下将展示菜单等待用户选择子任务。"
        )
    else:
        cli_renderer.print_error(f"无效参数：{tokens[1]}。支持 on/off。")
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


def _handle_multi(
    runner: AgentRunner,
    runtime_context: dict[str, object],
    cli_renderer: CliRenderer,
    tokens: list[str],
    raw_input: str,
) -> bool | None:
    """切换多 Agent 并发模式。"""
    current = runtime_context.get("multi_agent_enabled", "auto")
    state_labels = {
        True: "强制启用",
        False: "强制禁用",
        "auto": "自动判断",
    }
    if len(tokens) == 1:
        label = state_labels.get(current, str(current))
        cli_renderer.print_info(
            f"多 Agent 模式：{label}。使用 /multi on|off|auto 切换。"
        )
        return True

    toggle = tokens[1].lower()
    if toggle in ("on", "enable", "yes", "true"):
        runtime_context["multi_agent_enabled"] = True
        cli_renderer.print_info(
            "已启用多 Agent 并发模式（强制）。任务将始终分解并分配给 "
            "checker/reader/analyst/runner/builder/decision-maker/"
            "reflector/diffuser/jumper 角色并行执行。"
        )
    elif toggle in ("off", "disable", "no", "false"):
        runtime_context["multi_agent_enabled"] = False
        cli_renderer.print_info("已禁用多 Agent 模式，始终使用单 Agent 执行。")
    elif toggle in ("auto", "smart"):
        runtime_context["multi_agent_enabled"] = "auto"
        cli_renderer.print_info(
            "多 Agent 模式设为自动判断。简单任务单 Agent，"
            "复杂任务自动启用多 Agent 协作。"
        )
    else:
        cli_renderer.print_error(f"无效参数：{tokens[1]}。支持 on/off/auto。")
    return True


def _handle_file(
    runner: AgentRunner,
    runtime_context: dict[str, object],
    cli_renderer: CliRenderer,
    tokens: list[str],
    raw_input: str,
) -> bool | None:
    """加载文件内容到当前会话上下文。"""
    if len(tokens) < 2:
        cli_renderer.print_error("/file 需要指定文件路径。用法: /file <文件路径>")
        return True

    file_path_str = _extract_arg(raw_input, "/file")
    file_path = Path(file_path_str).expanduser().resolve()

    if not file_path.exists():
        cli_renderer.print_error(f"文件不存在：{file_path}")
        return True

    if file_path.is_dir():
        cli_renderer.print_error(f"路径是目录而非文件：{file_path}。请指定具体文件。")
        return True

    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            content = file_path.read_text(encoding="gbk")
        except Exception as exc:
            cli_renderer.print_error(f"无法读取文件（编码错误）：{exc}")
            return True
    except Exception as exc:
        cli_renderer.print_error(f"读取文件失败：{exc}")
        return True

    ext = file_path.suffix.lstrip(".").lower() or "text"
    lang_map = {
        "py": "python", "js": "javascript", "ts": "typescript", "tsx": "tsx",
        "jsx": "jsx", "json": "json", "yaml": "yaml", "yml": "yaml",
        "toml": "toml", "md": "markdown", "sh": "bash", "bash": "bash",
        "sql": "sql", "html": "html", "css": "css", "rs": "rust",
        "go": "go", "java": "java", "cpp": "cpp", "c": "c", "h": "c",
        "rb": "ruby", "php": "php", "swift": "swift", "kt": "kotlin",
    }
    lang = lang_map.get(ext, "")

    max_chars = 10000
    if len(content) > max_chars:
        truncated = content[:max_chars]
        note = f"\n\n... (文件已截断，共 {len(content)} 字符，显示前 {max_chars} 字符)"
        content = truncated + note

    # 将文件内容作为上下文注入到下一轮对话
    ctx_key = f"__pending_file_{file_path.name}"
    runtime_context[ctx_key] = {
        "path": str(file_path),
        "content": content,
        "lang": lang,
    }

    cli_renderer.print_markdown(
        f"已加载文件 **{file_path.name}**（{len(content)} 字符），"
        f"内容将在下一轮对话中自动附加。\n\n"
        f"```{lang}\n{content[:2000]}\n```\n"
        + (f"\n\n... 预览已截断，共 {len(content)} 字符。" if len(content) > 2000 else "")
    )
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
    "/memory": _handle_memory,
    "/mode": _handle_mode,
    "/approval": _handle_approval,
    "/multi": _handle_multi,
    "/auto-decision": _handle_auto_decision,
    "/file": _handle_file,
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
