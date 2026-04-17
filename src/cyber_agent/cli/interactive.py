from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


@dataclass(frozen=True)
class BuiltinCommandSpec:
    """统一描述交互模式内建命令，供帮助、补全与提示共用。"""

    command: str
    description: str
    show_in_banner: bool = False


@dataclass(frozen=True)
class SessionOverviewItem:
    """统一描述欢迎区中的状态项，供 CLI 与 TUI 复用。"""

    label: str
    value: str
    value_style_key: str


BUILTIN_COMMAND_SPECS: tuple[BuiltinCommandSpec, ...] = (
    BuiltinCommandSpec("/help", "查看帮助", show_in_banner=True),
    BuiltinCommandSpec("/tools", "查看默认工具", show_in_banner=True),
    BuiltinCommandSpec("/context", "查看当前内存上下文"),
    BuiltinCommandSpec("/context clear", "清空当前上下文并开始新会话"),
    BuiltinCommandSpec("/history", "查看当前工作目录下的历史会话"),
    BuiltinCommandSpec("/history show <会话ID>", "查看指定历史会话内容"),
    BuiltinCommandSpec("/history load <会话ID>", "将历史会话加载进当前上下文"),
    BuiltinCommandSpec("/status", "查看当前会话与配置状态", show_in_banner=True),
    BuiltinCommandSpec("/mode", "查看当前模式", show_in_banner=True),
    BuiltinCommandSpec("/mode standard", "切回标准模式"),
    BuiltinCommandSpec("/mode authorized", "切到授权模式"),
    BuiltinCommandSpec("/config", "查看当前工作目录下的本地配置", show_in_banner=True),
    BuiltinCommandSpec("/config allow-path", "查看本地配置中已保存的目录"),
    BuiltinCommandSpec("/config allow-path add <目录>", "将目录持久化写入本地配置"),
    BuiltinCommandSpec("/service", "查看当前模型服务商"),
    BuiltinCommandSpec("/service <服务商>", "切换当前会话的模型服务商"),
    BuiltinCommandSpec("/service <服务商> <基址>", "切换服务商并显式指定兼容接口基址"),
    BuiltinCommandSpec("/model", "查看当前模型名称"),
    BuiltinCommandSpec("/model <模型名>", "切换当前会话的模型名称"),
    BuiltinCommandSpec("/allow-path", "查看当前允许访问目录", show_in_banner=True),
    BuiltinCommandSpec("/allow-path add <目录>", "为当前会话添加允许访问目录"),
    BuiltinCommandSpec("/approval", "查看当前审批策略", show_in_banner=True),
    BuiltinCommandSpec("/approval prompt", "切到交互审批"),
    BuiltinCommandSpec("/approval auto", "切到自动批准"),
    BuiltinCommandSpec("/approval never", "切到全部拒绝"),
    BuiltinCommandSpec("/stop", "停止当前正在执行的任务"),
    BuiltinCommandSpec("/clear", "清空当前会话上下文"),
    BuiltinCommandSpec("/exit", "退出交互模式", show_in_banner=True),
)

EXIT_COMMANDS = {"quit", "exit", "q", "/quit", "/exit", ":q"}


class InteractionUiMode(StrEnum):
    """定义交互入口可使用的界面模式。"""

    AUTO = "auto"
    TUI = "tui"
    CLI = "cli"


INTERACTION_UI_MODE_LABELS: dict[InteractionUiMode, str] = {
    InteractionUiMode.AUTO: "自动选择（TUI 优先）",
    InteractionUiMode.TUI: "终端 TUI",
    InteractionUiMode.CLI: "命令行界面",
}


def parse_interaction_ui_mode(raw_value: str) -> InteractionUiMode:
    """将外部传入的界面模式字符串解析为内部枚举。"""

    normalized_value = raw_value.strip().lower()
    try:
        return InteractionUiMode(normalized_value)
    except ValueError as exc:
        supported_modes = ", ".join(mode.value for mode in InteractionUiMode)
        raise ValueError(
            f"不支持的界面模式：{raw_value}。可选值：{supported_modes}"
        ) from exc


def get_interaction_ui_mode_label(ui_mode: InteractionUiMode) -> str:
    """返回适合终端展示的界面模式名称。"""

    return INTERACTION_UI_MODE_LABELS[ui_mode]


def build_session_overview(
    *,
    mode_value: str,
    approval_policy_value: str,
    service: str,
    model: str,
    cwd: str,
) -> tuple[SessionOverviewItem, ...]:
    """构建欢迎区和状态面板共用的会话概览数据。"""

    return (
        SessionOverviewItem("当前模式", mode_value, mode_value),
        SessionOverviewItem("审批策略", approval_policy_value, approval_policy_value),
        SessionOverviewItem("模型服务", service, "service"),
        SessionOverviewItem("模型名称", model, "model"),
        SessionOverviewItem("工作目录", cwd, "cwd"),
    )


def get_input_composer_summary() -> str:
    """返回输入区快捷说明，供 CLI 与 TUI 共用。"""

    return "输入消息后按 Enter 发送，按 Tab 接受补全，输入 / 可查看命令。"


def get_autocomplete_summary() -> str:
    """返回自动补全说明，供 CLI 与 TUI 共用。"""

    return "输入 / 后按 Tab 可自动补全。"


def list_builtin_command_names() -> list[str]:
    """返回可用于补全的命令模板列表。"""

    return [item.command for item in BUILTIN_COMMAND_SPECS]


def get_banner_command_summary() -> str:
    """返回欢迎区展示的快捷命令摘要。"""

    return "  ".join(
        item.command for item in BUILTIN_COMMAND_SPECS if item.show_in_banner
    )


def match_builtin_commands(
    user_input: str,
    *,
    limit: int | None = None,
) -> list[BuiltinCommandSpec]:
    """根据当前输入匹配内建命令，供命令提醒与自动补全使用。"""

    normalized_input = user_input.strip().lower()

    if not normalized_input:
        matches = list(BUILTIN_COMMAND_SPECS)
    else:
        matches = [
            item
            for item in BUILTIN_COMMAND_SPECS
            if item.command.lower().startswith(normalized_input)
        ]

    if limit is None:
        return matches
    return matches[:limit]


def get_auto_completion(user_input: str) -> str | None:
    """返回当前输入可接受的首个补全结果。"""

    stripped_input = user_input.strip()
    if not stripped_input.startswith("/"):
        return None

    matches = match_builtin_commands(stripped_input, limit=1)
    if not matches:
        return None

    candidate = matches[0].command
    if candidate.lower() == stripped_input.lower():
        return None
    return candidate


def build_command_hint_lines(user_input: str, *, limit: int = 6) -> list[str]:
    """构建输入框下方展示的命令提醒文案。"""

    stripped_input = user_input.strip()
    matches = match_builtin_commands(stripped_input, limit=limit)

    if stripped_input.startswith("/") and matches:
        return [f"{item.command}  {item.description}" for item in matches]

    if stripped_input.startswith("/"):
        return ["未匹配到内建命令，可输入 /help 查看完整命令列表。"]

    defaults = match_builtin_commands("", limit=limit)
    return [f"{item.command}  {item.description}" for item in defaults]
