from __future__ import annotations

from rich import box
from rich.align import Align
from rich.cells import cell_len
from rich.console import RenderableType
from rich.panel import Panel
from rich.text import Text

from .theme import (
    ASSISTANT_BORDER_COLOR,
    SERVICE_VALUE_COLOR,
    TEXT_MUTED,
    TEXT_PRIMARY,
)

STARTUP_PANEL_TITLE = "启动页面"
STARTUP_TITLE = "Cyber Agent CLI"
STARTUP_SUBTITLE = "网络安全智能体终端"
STARTUP_ANIMATION_FRAMES = 16
STARTUP_ANIMATION_DELAY_SECONDS = 0.045

STARTUP_ART_LINES: tuple[str, ...] = (
    "   ______      __                ___                    __",
    "  / ____/_  __/ /_  ___  _____  /   | ____ ____  ____  / /_",
    " / /   / / / / __ \\/ _ \\/ ___/ / /| |/ __ `/ _ \\/ __ \\/ __/",
    "/ /___/ /_/ / /_/ /  __/ /    / ___ / /_/ /  __/ / / / /_",
    "\\____/\\__, /_.___/\\___/_/    /_/  |_\\__, /\\___/_/ /_/\\__/",
    "     /____/                        /____/",
)


def build_startup_renderable(progress: float = 1.0) -> RenderableType:
    """构建启动页字符画区块，供 CLI 与 TUI 共用。"""

    normalized_progress = _clamp_progress(progress)
    block_width = max(
        max(cell_len(line) for line in STARTUP_ART_LINES),
        cell_len(STARTUP_TITLE),
        cell_len(STARTUP_SUBTITLE),
    )

    body = Text(no_wrap=True)
    for index, line in enumerate(STARTUP_ART_LINES):
        line_progress = _phase_progress(normalized_progress, index * 0.05, 0.48)
        body.append(
            _fit_block_line(_reveal_ascii_line(line, line_progress), block_width),
            style=f"bold {SERVICE_VALUE_COLOR}",
        )
        body.append("\n")

    body.append("\n")
    body.append(
        _center_line(
            _reveal_dynamic_text(STARTUP_TITLE, _phase_progress(normalized_progress, 0.48, 0.24)),
            block_width,
        ),
        style=f"bold {TEXT_PRIMARY}",
    )
    body.append("\n")
    body.append(
        _center_line(
            _reveal_dynamic_text(
                STARTUP_SUBTITLE,
                _phase_progress(normalized_progress, 0.62, 0.22),
            ),
            block_width,
        ),
        style=TEXT_MUTED,
    )

    return Align.center(
        Panel.fit(
            body,
            box=box.ROUNDED,
            title=STARTUP_PANEL_TITLE,
            border_style=ASSISTANT_BORDER_COLOR,
            padding=(1, 2),
        )
    )


def build_startup_frame(frame_index: int) -> RenderableType:
    """根据帧序号构建启动页动画帧。"""

    if STARTUP_ANIMATION_FRAMES <= 1:
        return build_startup_renderable(1.0)
    normalized_index = max(0, min(frame_index, STARTUP_ANIMATION_FRAMES - 1))
    progress = normalized_index / (STARTUP_ANIMATION_FRAMES - 1)
    return build_startup_renderable(progress)


def _reveal_ascii_line(line: str, progress: float) -> str:
    if not line:
        return line
    if progress >= 1:
        return line
    leading_spaces = len(line) - len(line.lstrip(" "))
    prefix = line[:leading_spaces]
    body = line[leading_spaces:]
    return prefix + _reveal_fixed_width_text(body, progress)


def _reveal_fixed_width_text(content: str, progress: float) -> str:
    if not content:
        return content
    if progress <= 0:
        return " " * len(content)
    if progress >= 1:
        return content

    visible_length = min(len(content), int(len(content) * progress))
    if visible_length >= len(content):
        return content
    return (
        content[:visible_length]
        + "█"
        + " " * max(0, len(content) - visible_length - 1)
    )


def _reveal_dynamic_text(content: str, progress: float) -> str:
    if not content:
        return content
    if progress <= 0:
        return ""
    if progress >= 1:
        return content

    visible_length = max(1, min(len(content), int(len(content) * progress)))
    return content[:visible_length]


def _center_line(line: str, width: int) -> str:
    line_width = cell_len(line)
    if line_width >= width:
        return line

    left_padding = (width - line_width) // 2
    right_padding = width - line_width - left_padding
    return f"{' ' * left_padding}{line}{' ' * right_padding}"


def _fit_block_line(line: str, width: int) -> str:
    line_width = cell_len(line)
    if line_width >= width:
        return line
    return f"{line}{' ' * (width - line_width)}"


def _phase_progress(progress: float, start: float, span: float) -> float:
    if span <= 0:
        return 1.0 if progress >= start else 0.0
    return _clamp_progress((progress - start) / span)


def _clamp_progress(progress: float) -> float:
    return max(0.0, min(progress, 1.0))
