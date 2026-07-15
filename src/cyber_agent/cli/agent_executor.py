"""CLI 交互模式下的 Agent 执行、审批、/stop 支持。

从 app.py 中提取的独立模块，不依赖 Typer 命令定义。
"""
from __future__ import annotations

import re
import subprocess
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


def _is_benchmark_aggressive_context(runtime_context: dict[str, object]) -> bool:
    profile = str(runtime_context.get("benchmark_profile") or "").lower()
    return profile == "aggressive"


def _extract_cli_tool_text(tool_call: dict[str, object]) -> str:
    args = tool_call.get("args")
    if isinstance(args, dict):
        parts: list[str] = []
        for key in ("command", "url", "path"):
            value = args.get(key)
            if isinstance(value, str):
                parts.append(value)
        if parts:
            return "\n".join(parts)
    return str(args or tool_call)


def _command_uses_interface(command: str) -> str | None:
    match = re.search(r"(?:--interface|-interface)\s+([A-Za-z0-9_.:-]+)", command)
    return match.group(1) if match else None


def _detect_existing_tun_interface() -> str | None:
    try:
        result = subprocess.run(
            ["ip", "-o", "-4", "addr", "show"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=3,
        )
    except Exception:
        return None
    for line in (result.stdout or "").splitlines():
        match = re.search(r"\b(tun\d+)\b.*\binet\s+[0-9.]+/\d+", line)
        if match:
            return match.group(1)
    return None


def _benchmark_cli_tool_guard(
    runtime_context: dict[str, object],
    tool_call: dict[str, object],
) -> str | None:
    if not _is_benchmark_aggressive_context(runtime_context):
        return None
    tool_name = str(tool_call.get("name", ""))
    if tool_name not in {"run_shell_command", "fetch_web_page"}:
        return None

    command = _extract_cli_tool_text(tool_call)
    lowered = command.lower()
    used_interface = _command_uses_interface(command)
    api_interface = str(runtime_context.get("benchmark_api_interface") or "enp0s20f0u3u4")
    is_platform_api = (
        "tsecbench.zc.tencent.com" in lowered
        or "/openapi/v1/challenges" in lowered
    )
    has_container_url = bool(
        re.search(r"https?://10\.\d{1,3}\.\d{1,3}\.\d{1,3}", command)
    )

    if "openvpn" in lowered and ("--config" in lowered or ".ovpn" in lowered):
        existing_tun = _detect_existing_tun_interface()
        if existing_tun:
            return (
                f"Benchmark VPN 已连接（{existing_tun}），禁止重复启动 OpenVPN；"
                "请复用现有 tun 接口继续。"
            )

    if is_platform_api:
        if "/challenges/hint" in lowered:
            return "Benchmark 正式测评禁止调用 hint API，除非用户明确要求。"
        if used_interface and re.fullmatch(r"tun\d+", used_interface):
            return (
                f"平台 API 禁止走 {used_interface}；VPN/tun 只用于访问 10.x 容器。"
                f"请改用 --interface {api_interface} 并携带 BENCHMARK_TOKEN。"
            )
        if used_interface is None:
            return (
                "Benchmark 平台 API 必须显式绑定物理出口，禁止不带 --interface 调用；"
                f"请使用 --interface {api_interface}。"
            )
        if "benchmark_token:" not in lowered:
            return "TSec Benchmark 平台认证头必须是 BENCHMARK_TOKEN。"
        if "authorization:" in lowered or "x-benchmark-token:" in lowered:
            return "TSec Benchmark 平台禁止 Authorization/X-Benchmark-Token 认证头。"

    if has_container_url and not is_platform_api:
        tun_interface = _detect_existing_tun_interface() or "tun0"
        if tool_name == "fetch_web_page":
            return (
                "Benchmark 10.x 容器必须用 curl 绑定 VPN 接口访问；"
                f"请改用 curl --interface {tun_interface}。"
            )
        if used_interface is None:
            return (
                "Benchmark 10.x 容器访问必须显式绑定 VPN 接口；"
                f"请使用 --interface {tun_interface}。"
            )
        if not re.fullmatch(r"tun\d+", used_interface):
            return (
                "Benchmark 10.x 容器禁止走物理网卡；"
                f"请改用 --interface {tun_interface}。"
            )
    return None


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
        benchmark_reason = _benchmark_cli_tool_guard(runtime_context, tool_call)
        if benchmark_reason:
            return ApprovalDecision(False, benchmark_reason)

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
        benchmark_reason = _benchmark_cli_tool_guard(runtime_context, tool_call)
        if benchmark_reason:
            return ApprovalDecision(False, benchmark_reason)

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
