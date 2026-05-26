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


def build_status_panel(
    rows: list[tuple[str, str]],
    *,
    title: str = "当前状态",
) -> Panel:
    """构建状态概览面板，供状态查看和历史信息展示共用。"""
    status_table = Table.grid(padding=(0, 2))
    for label, value in rows:
        status_table.add_row(f"[bold cyan]{label}[/bold cyan]", value)
    return Panel(status_table, title=title, border_style="cyan")


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


