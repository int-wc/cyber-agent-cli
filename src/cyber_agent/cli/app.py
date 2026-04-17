import re
import threading
import time
import sys
from pathlib import Path
from queue import Empty, Queue

import typer
from click.exceptions import Abort
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from rich.console import Console

from ..agent.approval import (
    ApprovalDecision,
    ApprovalPolicy,
    get_approval_policy_label,
    parse_approval_policy,
)
from ..agent.mode import AgentMode, get_mode_label, parse_agent_mode
from ..agent.runner import AgentRunner, extract_text_content
from ..config import settings
from ..execution_control import ExecutionController, ExecutionInterruptedError
from ..local_config import (
    add_allow_path_to_local_config,
    get_local_config_path,
    load_local_cli_config,
    merge_allow_paths,
)
from ..session_store import (
    create_session_id,
    get_session_storage_dir,
    list_stored_sessions,
    load_session_history,
    save_session_history,
)
from ..tools import (
    describe_allowed_roots,
    describe_command_registry,
    describe_tools,
    get_default_tools,
    resolve_allowed_roots,
    resolve_command_registry,
)
from .interactive import (
    EXIT_COMMANDS,
    InteractionUiMode,
    get_interaction_ui_mode_label,
    parse_interaction_ui_mode,
)
from .render import CliRenderer

app = typer.Typer(
    add_completion=False,
    help="一个支持工具调用的命令行智能体原型。",
)

TOOL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
renderer = CliRenderer()
_cli_prompt_session = None
_prompt_toolkit_disabled = False


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
) -> dict[str, object]:
    """统一构建 CLI 运行上下文，避免多处分散解析。"""
    local_config = load_local_cli_config()
    persisted_allowed_paths = list(local_config.allow_paths)
    cli_allowed_paths = [Path(path).expanduser() for path in (allow_paths or [])]
    extra_allowed_paths = merge_allow_paths(
        persisted_allowed_paths,
        cli_allowed_paths,
    )
    configured_registry = parse_registered_tool_specs(tool_specs)
    allowed_roots = resolve_allowed_roots(mode, extra_allowed_paths)
    command_registry = resolve_command_registry(mode, configured_registry)
    execution_controller = ExecutionController()
    tools = get_default_tools(
        mode,
        extra_allowed_paths,
        configured_registry,
        execution_controller,
    )

    return {
        "mode": mode,
        "extra_allowed_paths": extra_allowed_paths,
        "saved_allowed_paths": persisted_allowed_paths,
        "local_config_path": get_local_config_path(),
        "allowed_roots": allowed_roots,
        "configured_registry": configured_registry,
        "command_registry": command_registry,
        "tools": tools,
        "approval_policy": approval_policy,
        "ui_mode": ui_mode,
        "execution_controller": execution_controller,
        "session_id": create_session_id(),
        "session_source_id": None,
        "session_storage_dir": get_session_storage_dir(),
        "_stop_input_buffer": "",
    }


def create_runner(runtime_context: dict[str, object]) -> AgentRunner:
    """按运行上下文创建会话运行器。"""
    return AgentRunner(
        runtime_context["tools"],
        mode=runtime_context["mode"],
        allowed_roots=runtime_context["allowed_roots"],
        command_registry=runtime_context["command_registry"],
        extra_allowed_paths=runtime_context["extra_allowed_paths"],
        configured_registry=runtime_context["configured_registry"],
        execution_controller=runtime_context["execution_controller"],
    )


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


def print_banner(
    runner: AgentRunner,
    runtime_context: dict[str, object],
    cli_renderer: CliRenderer = renderer,
) -> None:
    """输出交互模式欢迎信息。"""
    cli_renderer.print_banner(
        mode=runner.mode,
        service=settings.get_service(),
        model=settings.openai_model,
        cwd=Path.cwd(),
        approval_policy=runtime_context["approval_policy"],
    )


def print_help(cli_renderer: CliRenderer = renderer) -> None:
    """输出交互模式内建命令。"""
    cli_renderer.print_help()


def print_tools(
    runner: AgentRunner,
    cli_renderer: CliRenderer = renderer,
) -> None:
    """输出默认工具清单。"""
    cli_renderer.print_tools(
        describe_tools(
            runner.mode,
            runner.extra_allowed_paths,
            runner.configured_registry,
        )
    )


def print_status(
    runner: AgentRunner,
    runtime_context: dict[str, object],
    cli_renderer: CliRenderer = renderer,
) -> None:
    """输出便于试用排障的运行状态。"""
    api_key_configured = (
        "已配置"
        if settings.openai_api_key and settings.openai_api_key != "sk-default"
        else "未配置或仍为默认占位值"
    )
    saved_allowed_path_lines = "\n".join(
        describe_allowed_roots(runtime_context["saved_allowed_paths"])
    ) or "无"
    allowed_root_lines = "\n".join(describe_allowed_roots(runner.allowed_roots))
    registered_tool_lines = "\n".join(
        describe_command_registry(runner.command_registry)
    ) or "无"
    cli_renderer.print_status(
        [
            ("模式", f"{get_mode_label(runner.mode)} ({runner.mode.value})"),
            (
                "审批策略",
                f"{get_approval_policy_label(runtime_context['approval_policy'])}"
                f" ({runtime_context['approval_policy'].value})",
            ),
            ("服务", runner.service),
            ("模型", settings.openai_model),
            ("工作目录", str(Path.cwd())),
            ("会话轮数", str(runner.get_turn_count())),
            ("默认工具数", str(len(runner.tools))),
            (
                "界面",
                f"{get_interaction_ui_mode_label(runtime_context['ui_mode'])}"
                f" ({runtime_context['ui_mode'].value})",
            ),
            ("当前会话 ID", str(runtime_context["session_id"])),
            ("OPENAI_API_KEY", api_key_configured),
            ("本地配置文件", str(runtime_context["local_config_path"])),
            ("历史会话目录", str(runtime_context["session_storage_dir"])),
            ("已保存允许目录", saved_allowed_path_lines),
            ("允许读取根路径", allowed_root_lines),
            ("已注册外部工具", registered_tool_lines),
        ]
    )


def print_allowed_roots(
    runner: AgentRunner,
    cli_renderer: CliRenderer = renderer,
) -> None:
    """输出当前会话允许访问的目录根路径。"""
    cli_renderer.print_allowed_roots(describe_allowed_roots(runner.allowed_roots))


def print_local_config(
    runtime_context: dict[str, object],
    cli_renderer: CliRenderer = renderer,
) -> None:
    """输出当前工作目录下的本地配置内容。"""
    saved_allowed_path_lines = "\n".join(
        describe_allowed_roots(runtime_context["saved_allowed_paths"])
    ) or "无"
    cli_renderer.print_status(
        [
            ("本地配置文件", str(runtime_context["local_config_path"])),
            ("已保存允许目录", saved_allowed_path_lines),
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


def start_new_runtime_session(
    runtime_context: dict[str, object],
    *,
    source_session_id: str | None = None,
) -> str:
    """为当前运行上下文分配新的会话标识，避免覆盖既有历史。"""
    session_id = create_session_id()
    runtime_context["session_id"] = session_id
    runtime_context["session_source_id"] = source_session_id
    runtime_context["_stop_input_buffer"] = ""
    return session_id


def persist_runtime_session(
    runner: AgentRunner,
    runtime_context: dict[str, object],
) -> Path | None:
    """按当前工作目录自动保存会话历史，供后续 /history 访问。"""
    history = runner.get_history_snapshot()
    if len(history) <= 1 and runner.get_turn_count() == 0:
        return None

    session_path = save_session_history(
        str(runtime_context["session_id"]),
        history,
        mode=runner.mode.value,
        approval_policy=runtime_context["approval_policy"].value,
        source_session_id=runtime_context.get("session_source_id"),
    )
    runtime_context["session_storage_dir"] = session_path.parent
    return session_path


def _format_context_message(message: BaseMessage, index: int) -> str:
    """将消息压缩为适合终端浏览的一行上下文摘要。"""
    role_label = "系统"
    if isinstance(message, HumanMessage):
        role_label = "用户"
    elif isinstance(message, AIMessage):
        role_label = "助手"
    elif isinstance(message, ToolMessage):
        role_label = f"工具({message.name or 'unknown'})"
    elif isinstance(message, SystemMessage):
        role_label = "系统"

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
    messages = runner.get_history_snapshot()
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


def print_history_list(
    runtime_context: dict[str, object],
    cli_renderer: CliRenderer = renderer,
) -> None:
    """列出当前工作目录下可访问的历史会话摘要。"""
    stored_sessions = list_stored_sessions()
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

    cli_renderer.print_status(rows)


def show_history_session(
    session_id: str,
    cli_renderer: CliRenderer = renderer,
) -> None:
    """显示指定历史会话的完整内容。"""
    stored_session = load_session_history(session_id)
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
    stored_session = load_session_history(session_id)
    target_mode = parse_agent_mode(stored_session.summary.mode)
    target_approval_policy = parse_approval_policy(
        stored_session.summary.approval_policy
    )

    runner.switch_mode(target_mode)
    runner.restore_history(stored_session.messages)
    runtime_context["approval_policy"] = target_approval_policy
    sync_runtime_context_from_runner(runtime_context, runner)
    start_new_runtime_session(
        runtime_context,
        source_session_id=stored_session.summary.session_id,
    )
    cli_renderer.print_info(
        f"已加载历史会话：{stored_session.summary.session_id}。"
        "后续继续对话时会保存为新的会话副本。"
    )


def request_running_task_stop(
    runtime_context: dict[str, object],
    cli_renderer: CliRenderer = renderer,
    *,
    reason: str = "用户输入 /stop",
) -> bool:
    """请求中断当前任务，并给出统一提示。"""
    execution_controller: ExecutionController = runtime_context["execution_controller"]
    if execution_controller.is_cancel_requested():
        cli_renderer.print_info("已请求停止当前任务，正在等待执行链路收尾。")
        return True
    if not execution_controller.request_stop(reason):
        cli_renderer.print_info("当前没有正在执行的任务。")
        return False

    cli_renderer.print_info("已收到 /stop，正在终止当前模型、Shell 与工具执行...")
    return True


def _reset_stop_input_buffer(runtime_context: dict[str, object]) -> None:
    """清理忙碌态下的临时输入缓冲，避免污染下一轮正常提示。"""
    runtime_context["_stop_input_buffer"] = ""


def _consume_stop_input_nonblocking(
    runtime_context: dict[str, object],
) -> str | None:
    """在任务执行期间非阻塞轮询用户是否输入了 /stop。"""
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return None

    if sys.platform.startswith("win"):
        import msvcrt

        buffered_input = str(runtime_context.get("_stop_input_buffer", ""))
        while msvcrt.kbhit():
            input_character = msvcrt.getwch()
            if input_character in ("\r", "\n"):
                runtime_context["_stop_input_buffer"] = ""
                return buffered_input.strip()
            if input_character == "\003":
                raise KeyboardInterrupt
            if input_character == "\b":
                buffered_input = buffered_input[:-1]
                continue
            if input_character in ("\x00", "\xe0"):
                if msvcrt.kbhit():
                    msvcrt.getwch()
                continue
            buffered_input += input_character

        runtime_context["_stop_input_buffer"] = buffered_input
        return None

    try:
        import select

        ready_inputs, _, _ = select.select([sys.stdin], [], [], 0)
    except (OSError, ValueError):
        return None
    if not ready_inputs:
        return None
    return sys.stdin.readline().strip()


def render_agent_event(event_type: str, payload: object) -> None:
    """将运行器事件映射为富文本展示。"""
    if event_type == "turn_start":
        renderer.print_turn_start()
        return
    if event_type == "response_begin":
        renderer.begin_response_stream()
        return
    if event_type == "response_token":
        renderer.append_response_token(str(payload))
        return
    if event_type == "response_end":
        if isinstance(payload, dict):
            renderer.end_response_stream(
                str(payload.get("content", "")),
                bool(payload.get("has_tool_calls", False)),
            )
        return
    if event_type == "tool_call":
        renderer.print_tool_call(payload if isinstance(payload, list) else [])
        return
    if event_type == "approval_request":
        if isinstance(payload, dict):
            renderer.print_approval_request(payload)
        return
    if event_type == "approval_result":
        if isinstance(payload, dict):
            renderer.print_approval_result(payload)
        return
    if event_type == "tool_result":
        if isinstance(payload, dict):
            renderer.print_tool_result(str(payload.get("content", "")))
        else:
            renderer.print_tool_result(str(payload))
        return


def create_approval_handler(runtime_context: dict[str, object]):
    """按当前审批策略生成工具调用审批处理器。"""

    def approval_handler(tool: BaseTool, tool_call: dict) -> ApprovalDecision:
        policy = runtime_context["approval_policy"]
        tool_name = str(tool_call.get("name", tool.name))
        risk = str((tool.metadata or {}).get("risk", "unknown"))

        if policy is ApprovalPolicy.AUTO:
            return ApprovalDecision(True, "当前审批策略为自动批准。")
        if policy is ApprovalPolicy.NEVER:
            return ApprovalDecision(False, "当前审批策略拒绝所有高风险工具调用。")

        try:
            approved = typer.confirm(
                f"是否批准高风险工具调用？工具={tool_name}，风险={risk}",
                default=False,
            )
        except (Abort, EOFError, KeyboardInterrupt):
            approved = False

        if approved:
            return ApprovalDecision(True, "用户已在交互审批中明确批准。")
        return ApprovalDecision(False, "用户在交互审批中拒绝执行。")

    return approval_handler


def create_cli_background_approval_handler(
    runtime_context: dict[str, object],
    approval_requests: Queue[dict[str, object]],
):
    """为后台执行线程生成审批处理器，由主线程统一收集用户确认。"""
    execution_controller: ExecutionController = runtime_context["execution_controller"]

    def approval_handler(tool: BaseTool, tool_call: dict) -> ApprovalDecision:
        policy = runtime_context["approval_policy"]
        tool_name = str(tool_call.get("name", tool.name))
        risk = str((tool.metadata or {}).get("risk", "unknown"))

        if policy is ApprovalPolicy.AUTO:
            return ApprovalDecision(True, "当前审批策略为自动批准。")
        if policy is ApprovalPolicy.NEVER:
            return ApprovalDecision(False, "当前审批策略拒绝所有高风险工具调用。")

        approval_request = {
            "tool": tool,
            "tool_call": tool_call,
            "tool_name": tool_name,
            "risk": risk,
            "decision": None,
            "event": threading.Event(),
        }
        approval_requests.put(approval_request)

        while not approval_request["event"].wait(timeout=0.05):
            execution_controller.ensure_not_cancelled()

        decision = approval_request["decision"]
        if isinstance(decision, ApprovalDecision):
            return decision
        return ApprovalDecision(False, "审批结果缺失，已拒绝执行。")

    return approval_handler


def handle_pending_cli_approval_request(
    approval_requests: Queue[dict[str, object]],
) -> bool:
    """处理后台线程提交到主线程的审批请求。"""
    try:
        approval_request = approval_requests.get_nowait()
    except Empty:
        return False

    tool_name = str(approval_request["tool_name"])
    risk = str(approval_request["risk"])

    try:
        approved = typer.confirm(
            f"是否批准高风险工具调用？工具={tool_name}，风险={risk}",
            default=False,
        )
    except (Abort, EOFError, KeyboardInterrupt):
        approved = False

    if approved:
        decision = ApprovalDecision(True, "用户已在交互审批中明确批准。")
    else:
        decision = ApprovalDecision(False, "用户在交互审批中拒绝执行。")

    approval_request["decision"] = decision
    approval_request["event"].set()
    return True


def run_agent_turn_with_stop_support(
    runner: AgentRunner,
    user_input: str,
    runtime_context: dict[str, object],
) -> None:
    """在纯 CLI 交互中以后台线程运行任务，并轮询 /stop 与审批输入。"""
    worker_errors: list[BaseException] = []
    approval_requests: Queue[dict[str, object]] = Queue()
    approval_handler = create_cli_background_approval_handler(
        runtime_context,
        approval_requests,
    )

    def run_agent() -> None:
        try:
            runner.run(
                user_input,
                verbose=False,
                event_handler=render_agent_event,
                approval_handler=approval_handler,
            )
        except BaseException as exc:  # noqa: BLE001 - 需跨线程回传真实异常
            worker_errors.append(exc)

    worker_thread = threading.Thread(target=run_agent, daemon=True)
    worker_thread.start()

    if sys.stdin.isatty() and sys.stdout.isatty():
        renderer.print_info("任务执行中，可随时输入 /stop 并回车中断。")

    while worker_thread.is_alive():
        if handle_pending_cli_approval_request(approval_requests):
            continue

        try:
            stop_command = _consume_stop_input_nonblocking(runtime_context)
        except KeyboardInterrupt:
            request_running_task_stop(
                runtime_context,
                reason="用户通过键盘中断请求停止当前任务",
            )
            stop_command = None

        if stop_command is not None:
            if not stop_command:
                pass
            elif stop_command.lower() == "/stop":
                request_running_task_stop(runtime_context)
            else:
                renderer.print_info("当前任务执行中，仅支持输入 /stop。")

        worker_thread.join(timeout=0.05)
        time.sleep(0.02)

    worker_thread.join()
    _reset_stop_input_buffer(runtime_context)

    if worker_errors:
        raise worker_errors[0]


def handle_builtin_command(
    user_input: str,
    runner: AgentRunner,
    runtime_context: dict[str, object],
    cli_renderer: CliRenderer = renderer,
) -> bool | None:
    """处理交互模式下的内建命令，返回是否继续会话。"""
    stripped_input = user_input.strip()
    normalized_input = stripped_input.lower()
    tokens = normalized_input.split()

    if normalized_input in EXIT_COMMANDS:
        cli_renderer.print_info("👋 再见！")
        return False
    if normalized_input == "/stop":
        request_running_task_stop(runtime_context, cli_renderer)
        return True
    if normalized_input == "/help":
        print_help(cli_renderer)
        return True
    if normalized_input == "/tools":
        print_tools(runner, cli_renderer)
        return True
    if normalized_input == "/context":
        print_context(runner, runtime_context, cli_renderer)
        return True
    if normalized_input == "/context clear":
        runner.reset()
        start_new_runtime_session(runtime_context)
        cli_renderer.print_info("会话上下文已清空，并已开始新的会话。")
        return True
    if normalized_input == "/history":
        print_history_list(runtime_context, cli_renderer)
        return True
    if normalized_input.startswith("/history "):
        history_remainder = stripped_input[len("/history"):].strip()
        normalized_history_remainder = history_remainder.lower()
        if normalized_history_remainder.startswith("show "):
            session_id = history_remainder[5:].strip()
            if not session_id:
                cli_renderer.print_error("请提供要查看的会话 ID。")
                return True
            try:
                show_history_session(session_id, cli_renderer)
            except ValueError as exc:
                cli_renderer.print_error(str(exc))
            return True
        if normalized_history_remainder.startswith("load "):
            session_id = history_remainder[5:].strip()
            if not session_id:
                cli_renderer.print_error("请提供要加载的会话 ID。")
                return True
            try:
                load_history_session_into_runner(
                    session_id,
                    runner,
                    runtime_context,
                    cli_renderer,
                )
            except ValueError as exc:
                cli_renderer.print_error(str(exc))
            return True
        cli_renderer.print_error("不支持的 /history 子命令。")
        return True
    if normalized_input == "/status":
        print_status(runner, runtime_context, cli_renderer)
        return True
    if normalized_input == "/config":
        print_local_config(runtime_context, cli_renderer)
        return True
    if normalized_input.startswith("/config "):
        config_remainder = stripped_input[len("/config"):].strip()
        if config_remainder.lower() == "allow-path":
            print_local_config(runtime_context, cli_renderer)
            return True
        if config_remainder.lower().startswith("allow-path"):
            allow_path_remainder = config_remainder[len("allow-path"):].strip()
            if allow_path_remainder.lower().startswith("add "):
                persisted_path = allow_path_remainder[4:].strip()
                try:
                    add_persisted_allowed_path(
                        persisted_path,
                        runner,
                        runtime_context,
                        cli_renderer,
                    )
                except ValueError as exc:
                    cli_renderer.print_error(str(exc))
                return True
            cli_renderer.print_error("不支持的 /config allow-path 子命令。")
            return True
        cli_renderer.print_error("不支持的 /config 子命令。")
        return True
    if normalized_input == "/allow-path":
        print_allowed_roots(runner, cli_renderer)
        return True
    if normalized_input.startswith("/allow-path "):
        remainder = stripped_input[len("/allow-path"):].strip()
        if not remainder:
            cli_renderer.print_error("请提供要添加的目录路径。")
            return True

        if remainder.lower().startswith("add "):
            remainder = remainder[4:].strip()

        try:
            add_allowed_path(remainder, runner, runtime_context, cli_renderer)
        except ValueError as exc:
            cli_renderer.print_error(str(exc))
        return True
    if normalized_input == "/clear":
        runner.reset()
        start_new_runtime_session(runtime_context)
        cli_renderer.print_info("会话上下文已清空，并已开始新的会话。")
        return True
    if normalized_input == "/mode":
        cli_renderer.print_mode_notice(runner.mode, switched=False)
        return True
    if len(tokens) == 2 and tokens[0] == "/mode":
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
    if normalized_input == "/approval":
        cli_renderer.print_approval_policy_notice(
            runtime_context["approval_policy"],
            switched=False,
        )
        return True
    if len(tokens) == 2 and tokens[0] == "/approval":
        try:
            target_policy = parse_approval_policy(tokens[1])
        except ValueError as exc:
            cli_renderer.print_error(str(exc))
            return True
        runtime_context["approval_policy"] = target_policy
        cli_renderer.print_approval_policy_notice(target_policy, switched=True)
        return True

    return None


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


def run_chat_loop(
    runner: AgentRunner,
    runtime_context: dict[str, object],
    show_banner: bool = True,
) -> None:
    """运行类似 Claude Code 的交互式命令行循环。"""
    ui_mode = runtime_context.get("ui_mode", InteractionUiMode.AUTO)
    if ui_mode is InteractionUiMode.TUI:
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
        try:
            from .tui import launch_textual_chat
        except (ModuleNotFoundError, RuntimeError):
            pass
        else:
            launch_textual_chat(runner, runtime_context, show_banner=show_banner)
            return

    if show_banner:
        renderer.print_startup_splash()
        print_banner(runner, runtime_context)

    while True:
        try:
            user_input = prompt_chat_input().strip()
        except (Abort, EOFError, KeyboardInterrupt):
            renderer.print_info("\n👋 再见！")
            break

        if not user_input:
            continue

        renderer.print_user_message(user_input)
        builtin_result = handle_builtin_command(user_input, runner, runtime_context)
        if builtin_result is False:
            break
        if builtin_result is True:
            continue

        try:
            if sys.stdin.isatty() and sys.stdout.isatty():
                run_agent_turn_with_stop_support(
                    runner,
                    user_input,
                    runtime_context,
                )
            else:
                runner.run(
                    user_input,
                    verbose=False,
                    event_handler=render_agent_event,
                    approval_handler=create_approval_handler(runtime_context),
                )
            persist_runtime_session(runner, runtime_context)
        except ExecutionInterruptedError as exc:
            persist_runtime_session(runner, runtime_context)
            renderer.print_info(str(exc))
        except Exception as exc:
            persist_runtime_session(runner, runtime_context)
            # TODO(联调补全): 后续可按网络、工具、模型错误分别渲染更具体的提示。
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
                        _cli_prompt_session = CliPromptSession()
                    return _cli_prompt_session.prompt()
                except Exception as exc:  # noqa: BLE001 - 终端兼容失败时需要自动降级
                    _prompt_toolkit_disabled = True
                    renderer.print_error(f"CLI 补全已降级为基础输入：{exc}")

    return typer.prompt("›")


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
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
) -> None:
    """默认无子命令时直接进入交互式对话。"""
    ctx.ensure_object(dict)
    try:
        parsed_mode = parse_agent_mode(mode)
        parsed_approval_policy = parse_approval_policy(approval_policy)
        parsed_ui_mode = parse_interaction_ui_mode(ui)
        ctx.obj["runtime_context"] = build_runtime_context(
            parsed_mode,
            allow_paths,
            tool_specs,
            parsed_approval_policy,
            parsed_ui_mode,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if ctx.invoked_subcommand is None:
        run_chat_loop(
            create_runner(ctx.obj["runtime_context"]),
            ctx.obj["runtime_context"],
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
    runtime_context = ctx.obj["runtime_context"]
    runner = create_runner(runtime_context)
    if message is not None:
        try:
            runner.run(
                message,
                verbose=False,
                event_handler=render_agent_event,
                approval_handler=create_approval_handler(runtime_context),
            )
        finally:
            persist_runtime_session(runner, runtime_context)
        return
    run_chat_loop(runner, runtime_context)


@app.command()
def run(
    ctx: typer.Context,
    message: str = typer.Argument(..., help="要发送给智能体的一轮消息。"),
) -> None:
    """
    执行单轮对话，适合脚本或快速试验。
    """
    runtime_context = ctx.obj["runtime_context"]
    runner = create_runner(runtime_context)
    try:
        runner.run(
            message,
            verbose=False,
            event_handler=render_agent_event,
            approval_handler=create_approval_handler(runtime_context),
        )
    finally:
        persist_runtime_session(runner, runtime_context)


@app.command()
def tools(ctx: typer.Context) -> None:
    """
    查看当前默认启用的工具列表。
    """
    runner = create_runner(ctx.obj["runtime_context"])
    print_tools(runner)


@app.command()
def doctor(ctx: typer.Context) -> None:
    """
    检查当前 CLI 运行所依赖的关键配置。
    """
    runtime_context = ctx.obj["runtime_context"]
    runner = create_runner(runtime_context)
    print_status(runner, runtime_context)


def main() -> None:
    """提供给 python -m cyber_agent 的统一入口。"""
    app()
