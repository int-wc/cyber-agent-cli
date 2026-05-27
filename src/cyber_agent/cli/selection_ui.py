"""交互式选项菜单，供决策者 Agent 展示多方案供用户选择。

基于 prompt_toolkit 实现，支持上下方向键选择、Enter 确认、
可选的"其他"输入以及自定义调整。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SelectableOption:
    """供用户交互选择的单个选项。"""

    key: str
    label: str
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def _try_radiolist(options: list[SelectableOption], title: str) -> str | None:
    """尝试使用 prompt_toolkit 的 radiolist_dialog 展示选项。

    返回用户选择的 option.key，或 None 表示回退。
    """
    try:
        from prompt_toolkit.shortcuts import radiolist_dialog
        from prompt_toolkit.styles import Style

        values = [
            (opt.key, opt.label)
            for opt in options
        ]

        style = Style.from_dict({
            "dialog": "bg:#1a1a2e",
            "dialog frame.label": "bg:#16213e #00d4ff",
            "dialog.body": "bg:#1a1a2e #e0e0e0",
            "button": "bg:#0f3460",
            "button.focused": "bg:#e94560",
            "label": "#a0a0b0",
            "radio-checked": "#00d4ff",
            "radio-selected": "#e94560",
        })

        result = radiolist_dialog(
            title=title,
            text="使用 ↑↓ 方向键选择，Enter 确认：",
            values=values,
            style=style,
        ).run()
        return result
    except Exception:
        return None


def _try_checkboxlist(options: list[SelectableOption], title: str) -> list[str] | None:
    """尝试使用 prompt_toolkit 的 checkboxlist_dialog 展示多选。

    返回用户选择的 option.key 列表，或 None 表示回退。
    """
    try:
        from prompt_toolkit.shortcuts import checkboxlist_dialog
        from prompt_toolkit.styles import Style

        values = [
            (opt.key, opt.label)
            for opt in options
        ]

        style = Style.from_dict({
            "dialog": "bg:#1a1a2e",
            "dialog frame.label": "bg:#16213e #00d4ff",
            "dialog.body": "bg:#1a1a2e #e0e0e0",
            "button": "bg:#0f3460",
            "button.focused": "bg:#e94560",
            "label": "#a0a0b0",
        })

        result = checkboxlist_dialog(
            title=title,
            text="使用 ↑↓ 移动，Space 勾选/取消，Enter 确认：",
            values=values,
            style=style,
        ).run()
        return result
    except Exception:
        return None


def present_selection_menu(
    options: list[SelectableOption],
    *,
    title: str = "请选择方案",
    allow_other: bool = True,
    allow_custom: bool = True,
) -> dict[str, Any]:
    """展示交互式选项菜单，返回用户决策。

    返回格式：
    {
        "action": "selected" | "other" | "custom" | "cancelled" | "fallback",
        "selected_keys": [...],  # action=selected 时
        "custom_text": "...",     # action=custom 时
    }

    优先使用 prompt_toolkit 对话框；失败时回退到文本序号输入。
    """
    # 构建包含预设操作和自定义入口的完整选项列表
    all_options = list(options)

    if allow_other and all_options:
        all_options.append(SelectableOption(
            key="__other__",
            label="↳ 其他方案（输入自定义需求）",
            description="不选择上述方案，输入自定义调整",
        ))

    if allow_custom and all_options:
        all_options.append(SelectableOption(
            key="__custom__",
            label="↳ 修改方案细节（输入调整方法）",
            description="在现有方案基础上输入补充或修改",
        ))

    if not all_options:
        all_options.append(SelectableOption(
            key="__cancel__",
            label="↳ 跳过（直接执行）",
            description="不选择，使用默认方案执行",
        ))

    # 尝试 prompt_toolkit 图形菜单
    result = _try_radiolist(all_options, title)
    if result is not None:
        if result == "__other__":
            custom_text = _read_custom_input("请输入你的需求：")
            if custom_text is None:
                return {"action": "cancelled"}
            return {"action": "other", "custom_text": custom_text}
        if result == "__custom__":
            custom_text = _read_custom_input("请输入调整方法：")
            if custom_text is None:
                return {"action": "cancelled"}
            return {"action": "custom", "custom_text": custom_text}
        return {"action": "selected", "selected_keys": [result]}

    # 回退：纯文本序号菜单
    return _fallback_text_menu(options, all_options, title, allow_other, allow_custom)


def _read_custom_input(prompt: str) -> str | None:
    """读取用户自定义文本输入。"""
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.styles import Style as PTStyle

        style = PTStyle.from_dict({
            "prompt": "#00d4ff bold",
        })
        session = PromptSession(style=style)
        return session.prompt(f"{prompt}\n> ")
    except Exception:
        try:
            return input(f"{prompt}\n> ")
        except (EOFError, KeyboardInterrupt):
            return None


def _fallback_text_menu(
    options: list[SelectableOption],
    all_options: list[SelectableOption],
    title: str,
    allow_other: bool,
    allow_custom: bool,
) -> dict[str, Any]:
    """纯文本序号菜单，在 prompt_toolkit 不可用时回退。"""
    from .render import CliRenderer
    renderer = CliRenderer()
    renderer.print_info(f"\n{title}")
    renderer.print_info("─" * 40)

    for i, opt in enumerate(all_options, 1):
        desc = f" — {opt.description}" if opt.description else ""
        renderer.print_info(f"  [{i}] {opt.label}{desc}")

    renderer.print_info("─" * 40)

    try:
        choice = input("请输入序号 (直接回车=默认执行): ").strip()
    except (EOFError, KeyboardInterrupt):
        return {"action": "cancelled"}

    if not choice:
        return {"action": "selected", "selected_keys": [o.key for o in options]}

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(all_options):
            selected = all_options[idx]
            if selected.key == "__other__":
                custom_text = _read_custom_input("请输入你的需求：")
                if custom_text is None:
                    return {"action": "cancelled"}
                return {"action": "other", "custom_text": custom_text}
            if selected.key == "__custom__":
                custom_text = _read_custom_input("请输入调整方法：")
                if custom_text is None:
                    return {"action": "cancelled"}
                return {"action": "custom", "custom_text": custom_text}
            return {"action": "selected", "selected_keys": [selected.key]}
    except ValueError:
        pass

    return {"action": "selected", "selected_keys": [o.key for o in options]}


def present_multi_select_menu(
    options: list[SelectableOption],
    *,
    title: str = "请选择要执行的子任务",
) -> dict[str, Any]:
    """展示多选菜单，让用户勾选要执行的子任务。

    返回格式同 present_selection_menu。
    """
    all_options = list(options)
    all_options.append(SelectableOption(
        key="__all__",
        label="↳ 全部执行（默认）",
        description="执行所有子任务",
    ))
    all_options.append(SelectableOption(
        key="__cancel__",
        label="↳ 取消（返回对话）",
        description="不执行任何子任务",
    ))

    # 尝试 prompt_toolkit 多选
    result = _try_checkboxlist(all_options, title)
    if result is not None:
        if not result or "__cancel__" in result:
            return {"action": "cancelled"}
        if "__all__" in result:
            return {"action": "selected", "selected_keys": [o.key for o in options]}
        return {"action": "selected", "selected_keys": result}

    # 回退文本菜单
    return _fallback_text_menu(options, all_options, title, True, False)
