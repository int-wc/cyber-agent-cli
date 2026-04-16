import json
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from ..agent.approval import ApprovalPolicy, get_approval_policy_label
from ..agent.mode import AgentMode, get_mode_description, get_mode_label


class CliRenderer:
    """负责将 CLI 运行信息渲染成更适合终端展示的富文本输出。"""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self._streaming_response_started = False
        self._streaming_prefix_printed = False
        self._streamed_response_chunks: list[str] = []

    def print_banner(
        self,
        *,
        mode: AgentMode,
        service: str,
        model: str,
        cwd: Path,
        approval_policy: ApprovalPolicy,
    ) -> None:
        """打印欢迎面板，强化 CLI 的会话感。"""
        title = Text("Cyber Agent CLI", style="bold")
        subtitle = Text(f"{get_mode_label(mode)} · {service} · {model}", style="cyan")
        body = Table.grid(padding=(0, 1))
        body.add_row("工作目录", str(cwd))
        body.add_row("模式说明", get_mode_description(mode))
        body.add_row("审批策略", get_approval_policy_label(approval_policy))
        body.add_row(
            "快捷命令",
            "/help  /tools  /status  /mode  /config  /allow-path  /approval  /exit",
        )
        self.console.print(
            Panel(
                body,
                box=box.ROUNDED,
                title=title,
                subtitle=subtitle,
                border_style="bright_blue",
                padding=(1, 2),
            )
        )

    def print_help(self) -> None:
        """打印内建命令帮助。"""
        command_table = Table(box=box.SIMPLE_HEAVY, show_header=True)
        command_table.add_column("命令", style="bold cyan", no_wrap=True)
        command_table.add_column("说明", style="white")
        command_table.add_row("/help", "查看帮助")
        command_table.add_row("/tools", "查看默认工具")
        command_table.add_row("/status", "查看当前会话与配置状态")
        command_table.add_row("/mode", "查看当前模式")
        command_table.add_row("/mode standard", "切回标准模式")
        command_table.add_row("/mode authorized", "切到授权模式")
        command_table.add_row("/config", "查看当前工作目录下的本地配置")
        command_table.add_row("/config allow-path", "查看本地配置中已保存的目录")
        command_table.add_row("/config allow-path add <目录>", "将目录持久化写入本地配置")
        command_table.add_row("/allow-path", "查看当前允许访问目录")
        command_table.add_row("/allow-path add <目录>", "为当前会话添加允许访问目录")
        command_table.add_row("/approval", "查看当前审批策略")
        command_table.add_row("/approval prompt", "切到交互审批")
        command_table.add_row("/approval auto", "切到自动批准")
        command_table.add_row("/approval never", "切到全部拒绝")
        command_table.add_row("/clear", "清空当前会话上下文")
        command_table.add_row("/exit", "退出交互模式")
        self.console.print(Panel(command_table, title="内建命令", border_style="blue"))

    def print_allowed_roots(self, allowed_roots: list[str]) -> None:
        """打印当前允许访问的目录根路径。"""
        root_table = Table(box=box.SIMPLE_HEAVY, show_header=True)
        root_table.add_column("序号", style="bold cyan", no_wrap=True)
        root_table.add_column("目录", style="white")
        if not allowed_roots:
            root_table.add_row("-", "无")
        else:
            for index, allowed_root in enumerate(allowed_roots, start=1):
                root_table.add_row(str(index), allowed_root)
        self.console.print(
            Panel(root_table, title="允许访问目录", border_style="cyan")
        )

    def print_tools(self, descriptions: list[str]) -> None:
        """打印工具列表。"""
        tool_table = Table(box=box.SIMPLE_HEAVY, show_header=True)
        tool_table.add_column("工具名", style="bold green", no_wrap=True)
        tool_table.add_column("说明", style="white")
        for description in descriptions:
            tool_name, _, summary = description.partition(":")
            tool_table.add_row(tool_name, summary.strip())
        self.console.print(Panel(tool_table, title="默认工具", border_style="green"))

    def print_status(self, rows: list[tuple[str, str]]) -> None:
        """打印状态概览。"""
        status_table = Table.grid(padding=(0, 2))
        for label, value in rows:
            status_table.add_row(f"[bold cyan]{label}[/bold cyan]", value)
        self.console.print(Panel(status_table, title="当前状态", border_style="cyan"))

    def print_mode_notice(self, mode: AgentMode, switched: bool = True) -> None:
        """打印模式结果与安全边界提示。"""
        title = (
            f"已切换到 {get_mode_label(mode)}"
            if switched
            else f"当前模式：{get_mode_label(mode)}"
        )
        self.console.print(
            Panel(
                get_mode_description(mode),
                title=title,
                border_style="yellow" if mode is AgentMode.AUTHORIZED else "cyan",
            )
        )

    def print_approval_policy_notice(
        self,
        policy: ApprovalPolicy,
        switched: bool = True,
    ) -> None:
        """打印审批策略查看或切换结果。"""
        title = (
            f"已切换到 {get_approval_policy_label(policy)}"
            if switched
            else f"当前审批策略：{get_approval_policy_label(policy)}"
        )
        self.console.print(
            Panel(
                (
                    "高风险工具包括命令执行、文件写入、补丁应用等。"
                    if policy is not ApprovalPolicy.NEVER
                    else "当前策略会拒绝所有高风险工具调用。"
                ),
                title=title,
                border_style="yellow" if policy is ApprovalPolicy.PROMPT else "cyan",
            )
        )

    def print_turn_start(self) -> None:
        """打印一条分隔线，用于区分每轮会话。"""
        self.ensure_response_stream_closed()
        self.console.print(Rule(style="grey50"))

    def print_tool_call(self, tool_calls: list[dict]) -> None:
        """打印工具调用事件。"""
        self.ensure_response_stream_closed()
        pretty_calls = json.dumps(tool_calls, ensure_ascii=False, indent=2)
        self.console.print(
            Panel(
                pretty_calls,
                title="工具调用",
                border_style="magenta",
            )
        )

    def print_tool_result(self, content: str) -> None:
        """打印工具执行结果。"""
        self.ensure_response_stream_closed()
        self.console.print(
            Panel(
                content,
                title="工具结果",
                border_style="green",
            )
        )

    def begin_response_stream(self) -> None:
        """开始一轮模型 token 流。"""
        self.ensure_response_stream_closed()
        self._streaming_response_started = True
        self._streaming_prefix_printed = False
        self._streamed_response_chunks = []

    def append_response_token(self, token_text: str) -> None:
        """将 token 追加到终端。"""
        if not self._streaming_response_started:
            self.begin_response_stream()
        if not self._streaming_prefix_printed:
            self.console.print("[bold bright_blue]Agent[/bold bright_blue] › ", end="")
            self._streaming_prefix_printed = True
        self._streamed_response_chunks.append(token_text)
        self.console.print(token_text, end="", soft_wrap=True, highlight=False)

    def end_response_stream(self, content: str, has_tool_calls: bool) -> None:
        """结束一轮模型 token 流。"""
        if self._streaming_response_started:
            if self._streaming_prefix_printed:
                if not self._streamed_response_chunks and content:
                    self.console.print(content, end="", soft_wrap=True, highlight=False)
                self.console.print("")
            elif content and not has_tool_calls:
                self.console.print(
                    Panel(
                        content,
                        title="Agent",
                        border_style="bright_blue",
                        padding=(1, 2),
                    )
                )
        elif content and not has_tool_calls:
            self.console.print(
                Panel(
                    content,
                    title="Agent",
                    border_style="bright_blue",
                    padding=(1, 2),
                )
            )

        self._streaming_response_started = False
        self._streaming_prefix_printed = False
        self._streamed_response_chunks = []

    def ensure_response_stream_closed(self) -> None:
        """在打印其他区块前确保流式输出已换行结束。"""
        if self._streaming_response_started:
            if self._streaming_prefix_printed:
                self.console.print("")
            self._streaming_response_started = False
            self._streaming_prefix_printed = False
            self._streamed_response_chunks = []

    def print_approval_request(self, payload: dict) -> None:
        """打印审批请求。"""
        self.ensure_response_stream_closed()
        tool_name = str(payload.get("tool_name", "unknown"))
        risk = str(payload.get("risk", "unknown"))
        tool_call = payload.get("tool_call", {})
        pretty_call = json.dumps(tool_call, ensure_ascii=False, indent=2)
        self.console.print(
            Panel(
                f"风险级别: {risk}\n\n{pretty_call}",
                title=f"审批请求：{tool_name}",
                border_style="yellow",
            )
        )

    def print_approval_result(self, payload: dict) -> None:
        """打印审批结果。"""
        self.ensure_response_stream_closed()
        approved = bool(payload.get("approved", False))
        tool_name = str(payload.get("tool_name", "unknown"))
        reason = str(payload.get("reason", ""))
        self.console.print(
            Panel(
                reason,
                title=f"{'已批准' if approved else '已拒绝'}：{tool_name}",
                border_style="green" if approved else "red",
            )
        )

    def print_info(self, content: str) -> None:
        """打印普通提示。"""
        self.ensure_response_stream_closed()
        self.console.print(content)

    def print_error(self, content: str) -> None:
        """打印错误信息。"""
        self.ensure_response_stream_closed()
        self.console.print(Panel(content, title="错误", border_style="red"))
