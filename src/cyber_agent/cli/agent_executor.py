"""CLI 交互模式下的 Agent 执行、审批、/stop 支持。

从 app.py 中提取的独立模块，不依赖 Typer 命令定义。
"""
from __future__ import annotations

import sys
import threading
import time
from queue import Empty, Queue
from typing import TYPE_CHECKING

import typer
from click.exceptions import Abort

from ..agent.approval import ApprovalDecision, ApprovalPolicy

if TYPE_CHECKING:
    from ..agent.runner import AgentRunner
    from ..execution_control import ExecutionController
    from .render import CliRenderer


def request_running_task_stop(
    runtime_context: dict[str, object],
    cli_renderer: CliRenderer | None = None,
    *,
    reason: str = "用户输入 /stop",
) -> bool:
    """请求中断当前任务，并给出统一提示。"""
    if cli_renderer is None:
        from .app import renderer as _renderer
        cli_renderer = _renderer
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


def create_approval_handler(runtime_context: dict[str, object]):
    """按当前审批策略生成工具调用审批处理器（同步交互模式）。"""

    def approval_handler(tool, tool_call: dict) -> ApprovalDecision:
        from langchain_core.tools import BaseTool
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

    def approval_handler(tool, tool_call: dict) -> ApprovalDecision:
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
    *,
    cli_renderer: CliRenderer | None = None,
    event_handler=None,
) -> None:
    """在纯 CLI 交互中以后台线程运行任务，并轮询 /stop 与审批输入。"""
    if cli_renderer is None:
        from .app import renderer as _renderer
        cli_renderer = _renderer
    if event_handler is None:
        from .app import render_agent_event
        event_handler = render_agent_event

    worker_errors: list[Exception] = []
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
                event_handler=event_handler,
                approval_handler=approval_handler,
            )
        except Exception as exc:
            # 跨线程回传异常，不捕获 KeyboardInterrupt/SystemExit
            worker_errors.append(exc)

    worker_thread = threading.Thread(target=run_agent, daemon=True)
    if runner.llm is None:
        cli_renderer.print_info("正在初始化模型客户端，首次请求可能需要数十秒。")
    worker_thread.start()

    if sys.stdin.isatty() and sys.stdout.isatty():
        cli_renderer.print_info("任务执行中，可随时输入 /stop 并回车中断。")

    while worker_thread.is_alive():
        if handle_pending_cli_approval_request(approval_requests):
            continue

        try:
            stop_command = _consume_stop_input_nonblocking(runtime_context)
        except KeyboardInterrupt:
            request_running_task_stop(
                runtime_context,
                cli_renderer,
                reason="用户通过键盘中断请求停止当前任务",
            )
            stop_command = None

        if stop_command is not None:
            if not stop_command:
                pass
            elif stop_command.lower() == "/stop":
                request_running_task_stop(runtime_context, cli_renderer)
            else:
                cli_renderer.print_info("当前任务执行中，仅支持输入 /stop。")

        worker_thread.join(timeout=0.05)
        time.sleep(0.02)

    worker_thread.join()
    _reset_stop_input_buffer(runtime_context)

    if worker_errors:
        raise worker_errors[0]
