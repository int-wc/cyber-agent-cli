import re
from pathlib import Path

import typer
from click.exceptions import Abort
from langchain_core.tools import BaseTool

from ..agent.approval import (
    ApprovalDecision,
    ApprovalPolicy,
    get_approval_policy_label,
    parse_approval_policy,
)
from ..agent.mode import AgentMode, get_mode_label, parse_agent_mode
from ..agent.runner import AgentRunner
from ..config import settings
from ..local_config import (
    add_allow_path_to_local_config,
    get_local_config_path,
    load_local_cli_config,
    merge_allow_paths,
)
from ..tools import (
    describe_allowed_roots,
    describe_command_registry,
    describe_tools,
    get_default_tools,
    resolve_allowed_roots,
    resolve_command_registry,
)
from .render import CliRenderer

app = typer.Typer(
    add_completion=False,
    help="一个支持工具调用的命令行智能体原型。",
)

EXIT_COMMANDS = {"quit", "exit", "q", "/quit", "/exit", ":q"}
TOOL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
renderer = CliRenderer()


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
    tools = get_default_tools(mode, extra_allowed_paths, configured_registry)

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


def print_banner(runner: AgentRunner, runtime_context: dict[str, object]) -> None:
    """输出交互模式欢迎信息。"""
    renderer.print_banner(
        mode=runner.mode,
        service=settings.get_service(),
        model=settings.openai_model,
        cwd=Path.cwd(),
        approval_policy=runtime_context["approval_policy"],
    )


def print_help() -> None:
    """输出交互模式内建命令。"""
    renderer.print_help()


def print_tools(runner: AgentRunner) -> None:
    """输出默认工具清单。"""
    renderer.print_tools(
        describe_tools(
            runner.mode,
            runner.extra_allowed_paths,
            runner.configured_registry,
        )
    )


def print_status(runner: AgentRunner, runtime_context: dict[str, object]) -> None:
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
    renderer.print_status(
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
            ("OPENAI_API_KEY", api_key_configured),
            ("本地配置文件", str(runtime_context["local_config_path"])),
            ("已保存允许目录", saved_allowed_path_lines),
            ("允许读取根路径", allowed_root_lines),
            ("已注册外部工具", registered_tool_lines),
        ]
    )


def print_allowed_roots(runner: AgentRunner) -> None:
    """输出当前会话允许访问的目录根路径。"""
    renderer.print_allowed_roots(describe_allowed_roots(runner.allowed_roots))


def print_local_config(runtime_context: dict[str, object]) -> None:
    """输出当前工作目录下的本地配置内容。"""
    saved_allowed_path_lines = "\n".join(
        describe_allowed_roots(runtime_context["saved_allowed_paths"])
    ) or "无"
    renderer.print_status(
        [
            ("本地配置文件", str(runtime_context["local_config_path"])),
            ("已保存允许目录", saved_allowed_path_lines),
        ]
    )


def add_allowed_path(
    raw_path: str,
    runner: AgentRunner,
    runtime_context: dict[str, object],
) -> None:
    """为当前会话动态增加允许访问目录，并同步刷新工具范围。"""
    if not raw_path.strip():
        raise ValueError("请提供要添加的目录路径。")

    added_path, was_added = runner.add_allowed_path(raw_path.strip())
    sync_runtime_context_from_runner(runtime_context, runner)

    if was_added:
        renderer.print_info(f"已添加允许访问目录：{added_path}")
        return
    renderer.print_info(f"目录已在允许访问范围内：{added_path}")


def add_persisted_allowed_path(
    raw_path: str,
    runner: AgentRunner,
    runtime_context: dict[str, object],
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
            renderer.print_info(f"已写入本地配置并加入当前会话：{persisted_path}")
            return
        renderer.print_info(
            f"已写入本地配置：{persisted_path}。切换到授权模式后会自动生效。"
        )
        return

    if runner.mode is AgentMode.AUTHORIZED:
        renderer.print_info(f"目录已存在于本地配置和当前会话中：{persisted_path}")
        return
    renderer.print_info(f"目录已存在于本地配置中：{persisted_path}")


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
        risk = str((tool.extras or {}).get("risk", "unknown"))

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


def handle_builtin_command(
    user_input: str,
    runner: AgentRunner,
    runtime_context: dict[str, object],
) -> bool | None:
    """处理交互模式下的内建命令，返回是否继续会话。"""
    stripped_input = user_input.strip()
    normalized_input = stripped_input.lower()
    tokens = normalized_input.split()

    if normalized_input in EXIT_COMMANDS:
        renderer.print_info("👋 再见！")
        return False
    if normalized_input == "/help":
        print_help()
        return True
    if normalized_input == "/tools":
        print_tools(runner)
        return True
    if normalized_input == "/status":
        print_status(runner, runtime_context)
        return True
    if normalized_input == "/config":
        print_local_config(runtime_context)
        return True
    if normalized_input.startswith("/config "):
        config_remainder = stripped_input[len("/config"):].strip()
        if config_remainder.lower() == "allow-path":
            print_local_config(runtime_context)
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
                    )
                except ValueError as exc:
                    renderer.print_error(str(exc))
                return True
            renderer.print_error("不支持的 /config allow-path 子命令。")
            return True
        renderer.print_error("不支持的 /config 子命令。")
        return True
    if normalized_input == "/allow-path":
        print_allowed_roots(runner)
        return True
    if normalized_input.startswith("/allow-path "):
        remainder = stripped_input[len("/allow-path"):].strip()
        if not remainder:
            renderer.print_error("请提供要添加的目录路径。")
            return True

        if remainder.lower().startswith("add "):
            remainder = remainder[4:].strip()

        try:
            add_allowed_path(remainder, runner, runtime_context)
        except ValueError as exc:
            renderer.print_error(str(exc))
        return True
    if normalized_input == "/clear":
        runner.reset()
        renderer.print_info("会话上下文已清空。")
        return True
    if normalized_input == "/mode":
        renderer.print_mode_notice(runner.mode, switched=False)
        return True
    if len(tokens) == 2 and tokens[0] == "/mode":
        try:
            target_mode = parse_agent_mode(tokens[1])
        except ValueError as exc:
            renderer.print_error(str(exc))
            return True
        runner.switch_mode(target_mode)
        sync_runtime_context_from_runner(runtime_context, runner)
        renderer.print_mode_notice(target_mode, switched=True)
        return True
    if normalized_input == "/approval":
        renderer.print_approval_policy_notice(
            runtime_context["approval_policy"],
            switched=False,
        )
        return True
    if len(tokens) == 2 and tokens[0] == "/approval":
        try:
            target_policy = parse_approval_policy(tokens[1])
        except ValueError as exc:
            renderer.print_error(str(exc))
            return True
        runtime_context["approval_policy"] = target_policy
        renderer.print_approval_policy_notice(target_policy, switched=True)
        return True

    return None


def run_chat_loop(
    runner: AgentRunner,
    runtime_context: dict[str, object],
    show_banner: bool = True,
) -> None:
    """运行类似 Claude Code 的交互式命令行循环。"""
    if show_banner:
        print_banner(runner, runtime_context)

    while True:
        try:
            user_input = typer.prompt("›").strip()
        except (Abort, EOFError, KeyboardInterrupt):
            renderer.print_info("\n👋 再见！")
            break

        if not user_input:
            continue

        builtin_result = handle_builtin_command(user_input, runner, runtime_context)
        if builtin_result is False:
            break
        if builtin_result is True:
            continue

        try:
            runner.run(
                user_input,
                verbose=False,
                event_handler=render_agent_event,
                approval_handler=create_approval_handler(runtime_context),
            )
        except Exception as exc:
            # TODO(联调补全): 后续可按网络、工具、模型错误分别渲染更具体的提示。
            renderer.print_error(f"运行失败：{exc}")


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
) -> None:
    """默认无子命令时直接进入交互式对话。"""
    parsed_mode = parse_agent_mode(mode)
    parsed_approval_policy = parse_approval_policy(approval_policy)
    ctx.ensure_object(dict)
    try:
        ctx.obj["runtime_context"] = build_runtime_context(
            parsed_mode,
            allow_paths,
            tool_specs,
            parsed_approval_policy,
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
        runner.run(
            message,
            verbose=False,
            event_handler=render_agent_event,
            approval_handler=create_approval_handler(runtime_context),
        )
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
    runner.run(
        message,
        verbose=False,
        event_handler=render_agent_event,
        approval_handler=create_approval_handler(runtime_context),
    )


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
