import json
import time
from pathlib import Path

from rich import box
from rich.console import Console, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from ..agent.approval import ApprovalPolicy, get_approval_policy_label
from ..agent.mode import AgentMode, get_mode_description, get_mode_label
from .branding import (
    STARTUP_ANIMATION_DELAY_SECONDS,
    STARTUP_ANIMATION_FRAMES,
    build_startup_frame,
)
from .interactive import (
    BUILTIN_COMMAND_SPECS,
    build_session_overview,
    get_banner_command_summary,
)
from .theme import (
    ASSISTANT_BORDER_COLOR,
    ASSISTANT_TEXT_COLOR,
    COMMAND_DESC_STYLE,
    COMMAND_NAME_STYLE,
    KEYCAP_STYLE,
    ROLE_STYLES,
    SYSTEM_LABEL_STYLE,
    SYSTEM_VALUE_STYLE,
    SYSTEM_VALUE_STYLES,
)


def append_system_kv_line(
    text: Text,
    label: str,
    value: str,
    value_style: str,
) -> None:
    """向欢迎面板文本追加一行键值信息。"""
    text.append(label, style=SYSTEM_LABEL_STYLE)
    text.append("：", style=SYSTEM_LABEL_STYLE)
    text.append(value, style=value_style)
    text.append("\n")


def build_banner_body(
    *,
    mode: AgentMode,
    service: str,
    model: str,
    cwd: Path,
    approval_policy: ApprovalPolicy,
) -> Text:
    """构建 CLI 与 TUI 共用的欢迎面板正文。"""
    body = Text()
    body.append("Cyber Agent CLI 交互界面\n", style="bold #f8fafc")
    body.append("\n")
    for item in build_session_overview(
        mode_value=mode.value,
        approval_policy_value=approval_policy.value,
        service=service,
        model=model,
        cwd=str(cwd),
    ):
        append_system_kv_line(
            body,
            item.label,
            item.value,
            SYSTEM_VALUE_STYLES.get(item.value_style_key, SYSTEM_VALUE_STYLE),
        )

    body.append("快捷命令", style=SYSTEM_LABEL_STYLE)
    body.append("：", style=SYSTEM_LABEL_STYLE)
    for index, command in enumerate(get_banner_command_summary().split("  ")):
        if index > 0:
            body.append("  ", style=SYSTEM_LABEL_STYLE)
        body.append(command, style=COMMAND_NAME_STYLE)
    body.append("\n")

    body.append("命令补全", style=SYSTEM_LABEL_STYLE)
    body.append("：", style=SYSTEM_LABEL_STYLE)
    body.append("输入 ", style=COMMAND_DESC_STYLE)
    body.append("/", style=COMMAND_NAME_STYLE)
    body.append(" 后按 ", style=COMMAND_DESC_STYLE)
    body.append("Tab", style=KEYCAP_STYLE)
    body.append(" 可自动补全。", style=COMMAND_DESC_STYLE)
    return body


def build_banner_panel(
    *,
    mode: AgentMode,
    service: str,
    model: str,
    cwd: Path,
    approval_policy: ApprovalPolicy,
) -> Panel:
    """构建 CLI 与 TUI 共用的欢迎面板。"""
    return Panel(
        build_banner_body(
            mode=mode,
            service=service,
            model=model,
            cwd=cwd,
            approval_policy=approval_policy,
        ),
        box=box.ROUNDED,
        title=ROLE_STYLES["system"]["title"],
        border_style=ROLE_STYLES["system"]["border_style"],
        padding=(0, 1),
    )


def build_chat_message_panel(role: str, content: str | Text) -> Panel:
    """构建 CLI 与 TUI 共用的消息面板。"""
    style = ROLE_STYLES.get(role, ROLE_STYLES["system"])
    if isinstance(content, Text):
        if content.plain.strip():
            message_text = content.copy()
        else:
            message_text = Text("正在处理...", style=style["text_style"])
    else:
        message_text = Text(
            content.strip() or "正在处理...",
            style=style["text_style"],
        )
    return Panel(
        message_text,
        title=style["title"],
        border_style=style["border_style"],
        box=box.ROUNDED,
        padding=(0, 1),
    )


def build_tool_call_panel(tool_calls: list[dict]) -> Panel:
    """构建工具调用面板。"""
    return Panel(
        json.dumps(tool_calls, ensure_ascii=False, indent=2),
        title="工具调用",
        border_style="magenta",
    )


def build_tool_result_panel(content: str) -> Panel:
    """构建工具结果面板。"""
    return Panel(content, title="工具结果", border_style="green")


def build_approval_request_panel(payload: dict) -> Panel:
    """构建审批请求面板。"""
    tool_name = str(payload.get("tool_name", "unknown"))
    risk = str(payload.get("risk", "unknown"))
    tool_call = payload.get("tool_call", {})
    pretty_call = json.dumps(tool_call, ensure_ascii=False, indent=2)
    return Panel(
        f"风险级别: {risk}\n\n{pretty_call}",
        title=f"审批请求：{tool_name}",
        border_style="yellow",
    )


def build_approval_result_panel(payload: dict) -> Panel:
    """构建审批结果面板。"""
    approved = bool(payload.get("approved", False))
    tool_name = str(payload.get("tool_name", "unknown"))
    reason = str(payload.get("reason", ""))
    return Panel(
        reason,
        title=f"{'已批准' if approved else '已拒绝'}：{tool_name}",
        border_style="green" if approved else "red",
    )


def build_help_panel() -> Panel:
    """构建内建命令帮助面板，供 CLI 与 TUI 统一复用。"""
    command_table = Table(box=box.SIMPLE_HEAVY, show_header=True)
    command_table.add_column("命令", style="bold cyan", no_wrap=True)
    command_table.add_column("说明", style="white")
    for command in BUILTIN_COMMAND_SPECS:
        command_table.add_row(command.command, command.description)
    return Panel(command_table, title="内建命令", border_style="blue")


def build_allowed_roots_panel(allowed_roots: list[str]) -> Panel:
    """构建允许访问目录面板，避免 CLI 与 TUI 各维护一套表格样式。"""
    root_table = Table(box=box.SIMPLE_HEAVY, show_header=True)
    root_table.add_column("序号", style="bold cyan", no_wrap=True)
    root_table.add_column("目录", style="white")
    if not allowed_roots:
        root_table.add_row("-", "无")
    else:
        for index, allowed_root in enumerate(allowed_roots, start=1):
            root_table.add_row(str(index), allowed_root)
    return Panel(root_table, title="允许访问目录", border_style="cyan")


def build_tools_panel(descriptions: list[str]) -> Panel:
    """构建工具列表面板，保证两种界面的表头与配色一致。"""
    tool_table = Table(box=box.SIMPLE_HEAVY, show_header=True)
    tool_table.add_column("工具名", style="bold green", no_wrap=True)
    tool_table.add_column("说明", style="white")
    for description in descriptions:
        tool_name, _, summary = description.partition(":")
        tool_table.add_row(tool_name, summary.strip())
    return Panel(tool_table, title="默认工具", border_style="green")


def build_status_panel(rows: list[tuple[str, str]]) -> Panel:
    """构建状态概览面板，供状态查看和历史信息展示共用。"""
    status_table = Table.grid(padding=(0, 2))
    for label, value in rows:
        status_table.add_row(f"[bold cyan]{label}[/bold cyan]", value)
    return Panel(status_table, title="当前状态", border_style="cyan")


def build_mode_notice_panel(mode: AgentMode, switched: bool = True) -> Panel:
    """构建模式提示面板，统一切换结果与当前模式查看样式。"""
    title = (
        f"已切换到 {get_mode_label(mode)}"
        if switched
        else f"当前模式：{get_mode_label(mode)}"
    )
    return Panel(
        get_mode_description(mode),
        title=title,
        border_style="yellow" if mode is AgentMode.AUTHORIZED else "cyan",
    )


def build_approval_policy_notice_panel(
    policy: ApprovalPolicy,
    switched: bool = True,
) -> Panel:
    """构建审批策略提示面板，供 CLI 与 TUI 统一展示。"""
    title = (
        f"已切换到 {get_approval_policy_label(policy)}"
        if switched
        else f"当前审批策略：{get_approval_policy_label(policy)}"
    )
    return Panel(
        (
            "高风险工具包括命令执行、文件写入、补丁应用等。"
            if policy is not ApprovalPolicy.NEVER
            else "当前策略会拒绝所有高风险工具调用。"
        ),
        title=title,
        border_style="yellow" if policy is ApprovalPolicy.PROMPT else "cyan",
    )


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
        """打印欢迎面板，保持与 TUI 欢迎区一致。"""
        body = Text()
        body.append("Cyber Agent CLI 交互界面\n", style="bold #f8fafc")
        body.append("\n")
        for item in build_session_overview(
            mode_value=mode.value,
            approval_policy_value=approval_policy.value,
            service=service,
            model=model,
            cwd=str(cwd),
        ):
            self._append_system_kv_line(
                body,
                item.label,
                item.value,
                SYSTEM_VALUE_STYLES.get(item.value_style_key, SYSTEM_VALUE_STYLE),
            )

        body.append("快捷命令", style=SYSTEM_LABEL_STYLE)
        body.append("：", style=SYSTEM_LABEL_STYLE)
        for index, command in enumerate(get_banner_command_summary().split("  ")):
            if index > 0:
                body.append("  ", style=SYSTEM_LABEL_STYLE)
            body.append(command, style=COMMAND_NAME_STYLE)
        body.append("\n")

        body.append("命令补全", style=SYSTEM_LABEL_STYLE)
        body.append("：", style=SYSTEM_LABEL_STYLE)
        body.append("输入 ", style=COMMAND_DESC_STYLE)
        body.append("/", style=COMMAND_NAME_STYLE)
        body.append(" 后按 ", style=COMMAND_DESC_STYLE)
        body.append("Tab", style=KEYCAP_STYLE)
        body.append(" 可自动补全。", style=COMMAND_DESC_STYLE)
        self.console.print(
            Panel(
                body,
                box=box.ROUNDED,
                title=ROLE_STYLES["system"]["title"],
                border_style=ROLE_STYLES["system"]["border_style"],
                padding=(0, 1),
            )
        )

    def print_startup_splash(self) -> None:
        """打印启动页；真实终端播放动画，其余场景回退为静态区块。"""

        self.ensure_response_stream_closed()
        if not self.console.is_terminal or self.console.is_dumb_terminal:
            self.console.print(build_startup_frame(STARTUP_ANIMATION_FRAMES - 1))
            self.console.print()
            return

        with Live(
            build_startup_frame(0),
            console=self.console,
            refresh_per_second=max(24, STARTUP_ANIMATION_FRAMES),
            transient=False,
        ) as live:
            for frame_index in range(1, STARTUP_ANIMATION_FRAMES):
                time.sleep(STARTUP_ANIMATION_DELAY_SECONDS)
                live.update(build_startup_frame(frame_index))
        self.console.print()

    def clear_screen(self) -> None:
        """在真实终端启动前清屏，避免旧内容干扰启动页展示。"""
        self.ensure_response_stream_closed()
        if self.console.is_terminal and not self.console.is_dumb_terminal:
            self.console.clear(home=True)

    def print_help(self) -> None:
        """打印内建命令帮助。"""
        command_table = Table(box=box.SIMPLE_HEAVY, show_header=True)
        command_table.add_column("命令", style="bold cyan", no_wrap=True)
        command_table.add_column("说明", style="white")
        for command in BUILTIN_COMMAND_SPECS:
            command_table.add_row(command.command, command.description)
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

    def print_chat_message(self, role: str, content: str) -> None:
        """以统一气泡样式打印一条会话消息。"""
        self.ensure_response_stream_closed()
        style = ROLE_STYLES.get(role, ROLE_STYLES["system"])
        self.console.print(
            Panel(
                Text(content, style=style["text_style"]),
                title=style["title"],
                border_style=style["border_style"],
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )

    def print_user_message(self, content: str) -> None:
        """打印用户输入，便于与 TUI 的消息气泡对齐。"""
        self.print_chat_message("user", content)

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
            self.console.print(
                f"[bold {ASSISTANT_BORDER_COLOR}]智能体输出[/bold {ASSISTANT_BORDER_COLOR}]"
                " › ",
                end="",
            )
            self._streaming_prefix_printed = True
        self._streamed_response_chunks.append(token_text)
        self.console.print(
            Text(token_text, style=ASSISTANT_TEXT_COLOR),
            end="",
            soft_wrap=True,
            highlight=False,
        )

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
                        Text(content, style=ROLE_STYLES["assistant"]["text_style"]),
                        title=ROLE_STYLES["assistant"]["title"],
                        border_style=ROLE_STYLES["assistant"]["border_style"],
                        box=box.ROUNDED,
                        padding=(0, 1),
                    )
                )
        elif content and not has_tool_calls:
            self.console.print(
                Panel(
                    Text(content, style=ROLE_STYLES["assistant"]["text_style"]),
                    title=ROLE_STYLES["assistant"]["title"],
                    border_style=ROLE_STYLES["assistant"]["border_style"],
                    box=box.ROUNDED,
                    padding=(0, 1),
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
        self.print_chat_message("error", content)

    def print_renderable(self, renderable: RenderableType) -> None:
        """统一输出 Rich 渲染对象，便于 TUI 直接复用同一份面板。"""
        self.ensure_response_stream_closed()
        self.console.print(renderable)

    def print_banner(
        self,
        *,
        mode: AgentMode,
        service: str,
        model: str,
        cwd: Path,
        approval_policy: ApprovalPolicy,
    ) -> None:
        """打印共享欢迎面板，确保 CLI 与 TUI 使用同一份欢迎区样式。"""
        self.print_renderable(
            build_banner_panel(
                mode=mode,
                service=service,
                model=model,
                cwd=cwd,
                approval_policy=approval_policy,
            )
        )

    def print_help(self) -> None:
        """打印统一的内建命令帮助面板。"""
        self.print_renderable(build_help_panel())

    def print_allowed_roots(self, allowed_roots: list[str]) -> None:
        """打印统一的允许访问目录面板。"""
        self.print_renderable(build_allowed_roots_panel(allowed_roots))

    def print_tools(self, descriptions: list[str]) -> None:
        """打印统一的工具列表面板。"""
        self.print_renderable(build_tools_panel(descriptions))

    def print_status(self, rows: list[tuple[str, str]]) -> None:
        """打印统一的状态概览面板。"""
        self.print_renderable(build_status_panel(rows))

    def print_mode_notice(self, mode: AgentMode, switched: bool = True) -> None:
        """打印模式查看或切换结果。"""
        self.print_renderable(build_mode_notice_panel(mode, switched=switched))

    def print_approval_policy_notice(
        self,
        policy: ApprovalPolicy,
        switched: bool = True,
    ) -> None:
        """打印审批策略查看或切换结果。"""
        self.print_renderable(
            build_approval_policy_notice_panel(policy, switched=switched)
        )

    def print_chat_message(self, role: str, content: str | Text) -> None:
        """打印统一聊天面板，供 CLI 与 TUI 共享。"""
        self.print_renderable(build_chat_message_panel(role, content))

    def print_tool_call(self, tool_calls: list[dict]) -> None:
        """打印工具调用事件面板。"""
        self.print_renderable(build_tool_call_panel(tool_calls))

    def print_tool_result(self, content: str) -> None:
        """打印工具执行结果面板。"""
        self.print_renderable(build_tool_result_panel(content))

    def end_response_stream(self, content: str, has_tool_calls: bool) -> None:
        """结束一轮流式输出，并在需要时回退到统一助手面板。"""
        if self._streaming_response_started:
            if self._streaming_prefix_printed:
                if not self._streamed_response_chunks and content:
                    self.console.print(content, end="", soft_wrap=True, highlight=False)
                self.console.print("")
            elif content and not has_tool_calls:
                self.console.print(build_chat_message_panel("assistant", content))
        elif content and not has_tool_calls:
            self.console.print(build_chat_message_panel("assistant", content))

        self._streaming_response_started = False
        self._streaming_prefix_printed = False
        self._streamed_response_chunks = []

    def print_approval_request(self, payload: dict) -> None:
        """打印审批请求面板。"""
        self.print_renderable(build_approval_request_panel(payload))

    def print_approval_result(self, payload: dict) -> None:
        """打印审批结果面板。"""
        self.print_renderable(build_approval_result_panel(payload))

    def print_info(self, content: str) -> None:
        """打印普通提示文本；捕获模式下也会按原样保留。"""
        self.print_renderable(content)

    def print_error(self, content: str) -> None:
        """打印错误消息面板。"""
        self.print_chat_message("error", content)

    def _append_system_kv_line(
        self,
        text: Text,
        label: str,
        value: str,
        value_style: str,
    ) -> None:
        text.append(label, style=SYSTEM_LABEL_STYLE)
        text.append("：", style=SYSTEM_LABEL_STYLE)
        text.append(value, style=value_style)
        text.append("\n")
