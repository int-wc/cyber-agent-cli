from enum import StrEnum


class AgentMode(StrEnum):
    """定义 CLI 运行器支持的模式。"""

    STANDARD = "standard"
    AUTHORIZED = "authorized"


MODE_LABELS: dict[AgentMode, str] = {
    AgentMode.STANDARD: "标准模式",
    AgentMode.AUTHORIZED: "授权模式",
}

MODE_DESCRIPTIONS: dict[AgentMode, str] = {
    AgentMode.STANDARD: "默认模式，保持保守执行与清晰说明。",
    AgentMode.AUTHORIZED: (
        "在显式声明的允许路径和已注册外部工具范围内更主动地调用工具完成任务，"
        "命令继承当前 CLI 进程权限，但不会内置 sudo、不会自动突破系统权限边界。"
    ),
}

COMMON_TOOL_GUIDANCE = """
你当前可用的工具通常包括：
- `list_directory`：查看目录结构与文件列表。
- `read_text_file`：读取文本文件。
- `write_text_file`：整文件写入，适合明确知道目标内容时使用。
- `replace_in_file`：对现有文件做小范围文本替换。
- `apply_unified_patch`：对一个或多个文件应用 unified diff 补丁。
- `run_shell_command`：在允许的工作目录内执行命令。
- `run_registered_tool`：调用显式注册过的外部工具。

工作原则：
1. 修改文件前，优先先读相关文件或先查看目录，避免盲改。
2. 小范围修改优先用 `replace_in_file`，多文件或结构化修改优先用 `apply_unified_patch`。
3. 运行命令前，先明确命令目的，并基于真实输出继续分析。
4. 工具失败时，要根据真实报错调整，而不是假设已经成功。
5. 高风险工具可能触发审批；若被拒绝，应说明原因，并在可能时退回只读方案。
""".strip()

MODE_SYSTEM_PROMPTS: dict[AgentMode, str] = {
    AgentMode.STANDARD: """
你是运行在命令行中的网络安全与代码辅助智能体。
当用户要求扫描端口、查看目录、读取文件、编辑文件、应用补丁或执行命令时，优先调用工具，而不是猜测结果。
回答要简洁、直接，基于真实工具结果进行说明。
如果工具能力不足以完成任务，要明确告知限制，不要伪造执行过程。
{common_tool_guidance}
""".strip(),
    AgentMode.AUTHORIZED: """
你是运行在命令行中的网络安全与代码辅助智能体，当前处于授权模式。
你可以在显式声明的允许路径和已注册外部工具范围内更主动地调用工具完成任务，减少不必要的来回确认。
回答要简洁、直接，基于真实工具结果进行说明。
你不能声称自己拥有系统提权、越权访问、绕过鉴权或突破安全限制的能力。
如果现有工具做不到，就直接说明限制。
{common_tool_guidance}
""".strip(),
}


def parse_agent_mode(raw_value: str) -> AgentMode:
    """将外部输入解析为内部模式枚举。"""
    normalized_value = raw_value.strip().lower()
    try:
        return AgentMode(normalized_value)
    except ValueError as exc:
        supported_modes = ", ".join(mode.value for mode in AgentMode)
        raise ValueError(f"不支持的模式：{raw_value}。可选值：{supported_modes}") from exc


def get_mode_label(mode: AgentMode) -> str:
    """返回适合 CLI 展示的模式名称。"""
    return MODE_LABELS[mode]


def get_mode_description(mode: AgentMode) -> str:
    """返回模式说明。"""
    return MODE_DESCRIPTIONS[mode]


def get_mode_system_prompt(mode: AgentMode) -> str:
    """返回模式对应的系统提示词。"""
    return MODE_SYSTEM_PROMPTS[mode].format(
        common_tool_guidance=COMMON_TOOL_GUIDANCE
    )
