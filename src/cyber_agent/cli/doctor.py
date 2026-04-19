from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version as read_distribution_version
from pathlib import Path
from typing import TYPE_CHECKING

from .. import __version__
from ..agent.approval import get_approval_policy_label
from ..agent.mode import get_mode_label
from ..config import settings
from ..tools import describe_allowed_roots, describe_command_registry
from ..tools.search import PLAYWRIGHT_AVAILABLE
from .interactive import InteractionUiMode, get_interaction_ui_mode_label
from .prompting import PROMPT_TOOLKIT_IMPORT_ERROR
from .tui import TEXTUAL_IMPORT_ERROR

if TYPE_CHECKING:
    from ..agent.runner import AgentRunner


def _get_distribution_version(distribution_name: str) -> str | None:
    """读取已安装分发包版本；未安装时返回 None。"""
    try:
        return read_distribution_version(distribution_name)
    except PackageNotFoundError:
        return None


def _format_dependency_status(
    distribution_name: str,
    *,
    available: bool,
    missing_message: str,
) -> str:
    """格式化依赖检查结果，便于统一展示。"""
    installed_version = _get_distribution_version(distribution_name)
    if available and installed_version:
        return f"已安装 {installed_version}"
    if available:
        return "已安装"
    return f"未安装，{missing_message}"


def _build_dependency_payload(
    distribution_name: str,
    *,
    missing_message: str,
) -> dict[str, object]:
    """构建单个依赖项的结构化诊断结果。"""
    installed_version = _get_distribution_version(distribution_name)
    available = installed_version is not None
    return {
        "available": available,
        "version": installed_version,
        "status": _format_dependency_status(
            distribution_name,
            available=available,
            missing_message=missing_message,
        ),
    }


def _format_directory_state(directory_path: Path) -> str:
    """描述诊断目录当前是否存在，避免 doctor 主动写文件。"""
    resolved_directory = directory_path.resolve()
    if resolved_directory.exists():
        if resolved_directory.is_dir():
            return f"已存在：{resolved_directory}"
        return f"路径冲突：{resolved_directory} 已存在但不是目录。"
    return f"尚未创建，首次使用时会自动创建：{resolved_directory}"


def _describe_capability_lines(capability_registry: object) -> str:
    """将动态能力摘要压缩成适合诊断面板展示的文本。"""
    capabilities = _list_capability_payloads(capability_registry)
    if not capabilities:
        return "无"

    lines: list[str] = []
    for capability in capabilities:
        lines.append(
            f"{capability['name']} | kind={capability['kind']} | "
            f"tool={str(capability['register_as_tool']).lower()} | "
            f"status={capability['status']} | rev={capability['revision']}"
        )
    return "\n".join(lines)


def _list_capability_payloads(capability_registry: object) -> list[dict[str, object]]:
    """提取动态能力的结构化摘要，供文本与 JSON 输出共用。"""
    list_capabilities = getattr(capability_registry, "list_capabilities", None)
    if not callable(list_capabilities):
        return []

    capabilities = list_capabilities()
    payloads: list[dict[str, object]] = []
    for capability in capabilities:
        payloads.append(
            {
                "name": capability.name,
                "kind": capability.kind,
                "register_as_tool": capability.register_as_tool,
                "status": capability.status,
                "revision": capability.revision,
                "description": capability.description,
                "usage_hint": capability.usage_hint,
            }
        )
    return payloads


def _build_doctor_reminders(
    *,
    api_key_configured: bool,
    langgraph_available: bool,
    prompt_toolkit_available: bool,
    textual_available: bool,
    playwright_available: bool,
    ui_mode: InteractionUiMode,
) -> list[str]:
    """整理值得用户关注的诊断提醒。"""
    reminders: list[str] = []
    if not api_key_configured:
        reminders.append("OPENAI_API_KEY 未配置或仍为默认占位值，真实模型调用会失败。")
    if not langgraph_available:
        reminders.append("langgraph 未安装，create_agent_graph 将使用内置降级执行器。")
    if not prompt_toolkit_available:
        reminders.append("prompt_toolkit 未安装，CLI 补全会自动降级为基础输入。")
    if not textual_available:
        reminders.append("textual 未安装，TUI 模式当前不可用。")
    if ui_mode is InteractionUiMode.TUI and not textual_available:
        reminders.append("当前界面模式指定为 TUI，但运行环境缺少 textual 依赖。")
    if not playwright_available:
        reminders.append("Playwright 未安装，search_web 只能回退到 HTTP HTML 搜索。")
    return reminders


def build_doctor_rows(
    runner: AgentRunner,
    runtime_context: dict[str, object],
) -> list[tuple[str, str]]:
    """构建 doctor 命令要展示的完整诊断信息。"""
    payload = build_doctor_payload(runner, runtime_context)
    reminder_text = (
        "\n".join(f"- {item}" for item in payload["summary"]["reminders"])
        if payload["summary"]["reminders"]
        else "无"
    )
    capability_lines = _describe_capability_lines(runtime_context["capability_registry"])
    saved_allowed_path_lines = "\n".join(payload["permissions"]["saved_allowed_paths"]) or "无"
    allowed_root_lines = "\n".join(payload["permissions"]["allowed_roots"]) or "无"
    registered_tool_lines = (
        "\n".join(payload["permissions"]["registered_tools"])
        if payload["permissions"]["registered_tools"]
        else "无"
    )

    return [
        ("诊断结论", str(payload["summary"]["status_text"])),
        ("诊断提醒", reminder_text),
        ("项目版本", str(payload["project"]["version"])),
        ("Python", str(payload["project"]["python_version"])),
        ("模式", str(payload["runtime"]["mode_label"])),
        ("审批策略", str(payload["runtime"]["approval_policy_label"])),
        ("界面", str(payload["runtime"]["ui_mode_label"])),
        ("服务", str(payload["runtime"]["service"])),
        ("模型", str(payload["runtime"]["model"])),
        ("模型基址", str(payload["runtime"]["base_url"])),
        (
            "OPENAI_API_KEY",
            (
                "已配置"
                if payload["runtime"]["api_key_configured"]
                else "未配置或仍为默认占位值"
            ),
        ),
        ("langchain_openai", str(payload["dependencies"]["langchain_openai"]["status"])),
        ("langgraph", str(payload["dependencies"]["langgraph"]["status"])),
        ("prompt_toolkit", str(payload["dependencies"]["prompt_toolkit"]["status"])),
        ("textual", str(payload["dependencies"]["textual"]["status"])),
        ("playwright", str(payload["dependencies"]["playwright"]["status"])),
        ("浏览器搜索", str(payload["search"]["status"])),
        ("本地配置文件", str(payload["storage"]["local_config_path"])),
        ("本地配置状态", str(payload["storage"]["local_config_status"])),
        ("历史会话目录", str(payload["storage"]["session_storage_status"])),
        ("动态能力目录", str(payload["storage"]["capability_storage_status"])),
        ("已保存允许目录", saved_allowed_path_lines),
        ("允许读取根路径", allowed_root_lines),
        ("已注册外部工具", registered_tool_lines),
        ("动态能力", capability_lines),
    ]


def build_doctor_payload(
    runner: AgentRunner,
    runtime_context: dict[str, object],
) -> dict[str, object]:
    """构建供终端展示与 JSON 输出共用的结构化诊断结果。"""
    capability_registry = runtime_context["capability_registry"]
    ui_mode = runtime_context["ui_mode"]
    if not isinstance(ui_mode, InteractionUiMode):
        ui_mode = InteractionUiMode.AUTO

    api_key_configured = bool(runner.api_key and runner.api_key != "sk-default")
    dependencies = {
        "langchain_openai": _build_dependency_payload(
            "langchain-openai",
            missing_message="模型客户端不可用，CLI 将无法创建 LLM。",
        ),
        "langgraph": _build_dependency_payload(
            "langgraph",
            missing_message="图执行链路会改用内置降级实现。",
        ),
        "prompt_toolkit": {
            "available": PROMPT_TOOLKIT_IMPORT_ERROR is None,
            "version": _get_distribution_version("prompt_toolkit"),
            "status": _format_dependency_status(
                "prompt_toolkit",
                available=PROMPT_TOOLKIT_IMPORT_ERROR is None,
                missing_message="CLI 补全与底部命令提醒不可用。",
            ),
        },
        "textual": {
            "available": TEXTUAL_IMPORT_ERROR is None,
            "version": _get_distribution_version("textual"),
            "status": _format_dependency_status(
                "textual",
                available=TEXTUAL_IMPORT_ERROR is None,
                missing_message="TUI 入口会自动回退到 CLI。",
            ),
        },
        "playwright": {
            "available": PLAYWRIGHT_AVAILABLE,
            "version": _get_distribution_version("playwright"),
            "status": _format_dependency_status(
                "playwright",
                available=PLAYWRIGHT_AVAILABLE,
                missing_message="网页搜索会回退到 HTTP HTML 模式。",
            ),
        },
    }
    reminders = _build_doctor_reminders(
        api_key_configured=api_key_configured,
        langgraph_available=bool(dependencies["langgraph"]["available"]),
        prompt_toolkit_available=bool(dependencies["prompt_toolkit"]["available"]),
        textual_available=bool(dependencies["textual"]["available"]),
        playwright_available=bool(dependencies["playwright"]["available"]),
        ui_mode=ui_mode,
    )
    local_config_path = Path(runtime_context["local_config_path"]).resolve()
    session_storage_dir = Path(runtime_context["session_storage_dir"]).resolve()
    capability_storage_dir = Path(
        getattr(capability_registry, "storage_dir", Path.cwd() / ".cyber-agent-cli-capabilities")
    ).resolve()

    saved_allowed_paths = [
        str(Path(path).expanduser().resolve())
        for path in runtime_context["saved_allowed_paths"]
    ]
    allowed_roots = [str(path) for path in describe_allowed_roots(runner.allowed_roots)]
    registered_tools = describe_command_registry(runner.command_registry)

    search_status = (
        f"Playwright 可用，当前为 {'可见窗口' if settings.search_show_browser else '无头窗口'}"
        if PLAYWRIGHT_AVAILABLE
        else "Playwright 不可用，将回退到 HTTP HTML 搜索"
    )

    return {
        "summary": {
            "status": "ok" if not reminders else "warning",
            "status_text": "通过" if not reminders else f"存在 {len(reminders)} 项提醒",
            "reminder_count": len(reminders),
            "reminders": reminders,
        },
        "project": {
            "version": __version__,
            "python_version": sys.version.split()[0],
            "cwd": str(Path.cwd()),
        },
        "runtime": {
            "mode": runner.mode.value,
            "mode_label": f"{get_mode_label(runner.mode)} ({runner.mode.value})",
            "approval_policy": runtime_context["approval_policy"].value,
            "approval_policy_label": (
                f"{get_approval_policy_label(runtime_context['approval_policy'])} "
                f"({runtime_context['approval_policy'].value})"
            ),
            "ui_mode": ui_mode.value,
            "ui_mode_label": f"{get_interaction_ui_mode_label(ui_mode)} ({ui_mode.value})",
            "service": runner.service,
            "model": runner.model_name,
            "base_url": runner.base_url,
            "api_key_configured": api_key_configured,
        },
        "dependencies": dependencies,
        "search": {
            "playwright_available": PLAYWRIGHT_AVAILABLE,
            "show_browser": settings.search_show_browser,
            "status": search_status,
        },
        "storage": {
            "local_config_path": str(local_config_path),
            "local_config_exists": local_config_path.exists(),
            "local_config_status": (
                "已存在" if local_config_path.exists() else "未创建，将在首次持久化配置后生成"
            ),
            "session_storage_dir": str(session_storage_dir),
            "session_storage_status": _format_directory_state(session_storage_dir),
            "capability_storage_dir": str(capability_storage_dir),
            "capability_storage_status": _format_directory_state(capability_storage_dir),
        },
        "permissions": {
            "saved_allowed_paths": saved_allowed_paths,
            "allowed_roots": allowed_roots,
            "registered_tools": registered_tools,
        },
        "capabilities": _list_capability_payloads(capability_registry),
    }
