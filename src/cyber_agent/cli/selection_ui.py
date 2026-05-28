"""交互式选项菜单，供决策者 Agent 展示多方案供用户选择。

使用纯文本序号菜单，不弹全屏对话框，用户直接输入数字选择。
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


def _print_option_line(console, key: str, color: str, label: str) -> None:
    """打印带颜色标记的菜单选项行，避免 Rich markup 解析问题。"""
    from rich.text import Text
    t = Text("  [", style=f"bold {color}")
    t.append(key, style=f"bold {color}")
    t.append("] ", style=f"bold {color}")
    t.append(label, style="dim")
    console.print(t)


def _read_line(prompt: str) -> str | None:
    """读取一行用户输入。"""
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return None


def present_multi_select_menu(
    options: list[SelectableOption],
    *,
    title: str = "请选择要执行的子任务",
) -> dict[str, Any]:
    """展示多选菜单，返回用户决策。

    菜单格式：
      [1] 选项1
      [2] 选项2
      ...
      [a] 全部执行（默认）
      [0] 自定义输入（附加条件）
      [c] 取消

    返回格式：
      {"action": "selected", "selected_keys": [...]}
      {"action": "custom", "custom_text": "..."}
      {"action": "cancelled"}
    """
    from .render import CliRenderer
    from rich.text import Text

    renderer = CliRenderer()
    console = renderer.console

    # ── 渲染菜单 ──
    console.print()
    console.print(f"  [bold cyan]{title}[/]")
    console.print("  " + "─" * 72)

    for i, opt in enumerate(options, 1):
        t = Text("  [", style="bold white")
        t.append(str(i), style="bold white")
        t.append("] ", style="bold white")
        t.append(opt.label, style="dim")
        console.print(t)

    _print_option_line(console, "a", "green", "全部执行（默认）")
    _print_option_line(console, "0", "yellow", "自定义输入（附加条件、调整参数等）")
    _print_option_line(console, "c", "red", "取消")
    console.print("  " + "─" * 72)

    # ── 读取输入 ──
    choice = _read_line("  输入序号/字母 (直接回车=全部执行): ")

    if choice is None:
        return {"action": "cancelled"}

    # 直接回车 = 全部执行
    if not choice:
        return {"action": "selected", "selected_keys": [o.key for o in options]}

    # [0] 自定义输入
    if choice == "0":
        console.print("  [bold yellow]请输入附加条件或调整说明:[/]")
        custom = _read_line("  > ")
        if custom is None or not custom.strip():
            return {"action": "cancelled"}
        return {"action": "custom", "custom_text": custom.strip()}

    # [a] 或 [A] 全部执行
    if choice.lower() == "a":
        return {"action": "selected", "selected_keys": [o.key for o in options]}

    # [c] 或 [C] 取消
    if choice.lower() == "c":
        return {"action": "cancelled"}

    # 尝试解析逗号分隔的序号列表，如 "1,3" 或 "1"
    try:
        indices = _parse_indices(choice, len(options))
        if indices:
            keys = [options[i].key for i in indices]
            return {"action": "selected", "selected_keys": keys}
    except ValueError:
        pass

    # 解析失败，当作自定义文本
    console.print("  [dim]未识别为序号，作为自定义输入处理。[/]")
    return {"action": "custom", "custom_text": choice}


def _parse_indices(raw: str, max_n: int) -> list[int]:
    """解析用户输入的序号，支持逗号分隔和范围标记。

    示例: "1" → [0], "1,3" → [0,2], "1-3" → [0,1,2]
    """
    indices: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if "-" in part:
            start_s, _, end_s = part.partition("-")
            try:
                start = int(start_s.strip()) - 1
                end = int(end_s.strip()) - 1
                for n in range(max(0, start), min(max_n - 1, end) + 1):
                    if n not in indices:
                        indices.append(n)
            except ValueError:
                raise
        else:
            try:
                n = int(part) - 1
                if 0 <= n < max_n and n not in indices:
                    indices.append(n)
            except ValueError:
                raise
    return indices
