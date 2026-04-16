from __future__ import annotations

from html import escape

try:
    from prompt_toolkit import HTML, PromptSession
    from prompt_toolkit.completion import CompleteEvent, Completer, Completion
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.shortcuts.prompt import CompleteStyle
    from prompt_toolkit.styles import Style

    PROMPT_TOOLKIT_IMPORT_ERROR: ModuleNotFoundError | None = None
except ModuleNotFoundError as exc:  # pragma: no cover - 缺依赖时走 CLI 降级
    PROMPT_TOOLKIT_IMPORT_ERROR = exc

from .interactive import build_command_hint_lines, match_builtin_commands
from .theme import (
    COMMAND_DESC_COLOR,
    COMMAND_NAME_COLOR,
    SURFACE_BG,
    TEXT_PRIMARY,
    TEXT_MUTED,
    USER_BORDER_COLOR,
    USER_TEXT_COLOR,
)


if PROMPT_TOOLKIT_IMPORT_ERROR is None:
    PROMPT_STYLE = Style.from_dict(
        {
            "": f"bold {USER_TEXT_COLOR}",
            "prompt-label": f"bold {USER_BORDER_COLOR}",
            "prompt-separator": TEXT_MUTED,
            "bottom-toolbar": f"bg:{SURFACE_BG} {TEXT_PRIMARY}",
        }
    )

    class BuiltinCommandCompleter(Completer):
        """为纯 CLI 模式提供与 TUI 一致的内建命令补全。"""

        def get_completions(self, document, complete_event: CompleteEvent):
            current_input = document.text_before_cursor.strip()
            if not current_input.startswith("/"):
                return

            for item in match_builtin_commands(current_input, limit=None):
                yield Completion(
                    item.command,
                    start_position=-len(document.text_before_cursor),
                    display=item.command,
                    display_meta=item.description,
                )


    class CliPromptSession:
        """封装 prompt_toolkit 交互，统一 CLI 的补全和命令提醒。"""

        def __init__(self) -> None:
            self._session = PromptSession(history=InMemoryHistory())
            self._completer = BuiltinCommandCompleter()

        def prompt(self) -> str:
            return self._session.prompt(
                message=HTML(
                    "<prompt-label>用户输入</prompt-label>"
                    "<prompt-separator> › </prompt-separator>"
                ),
                completer=self._completer,
                complete_style=CompleteStyle.COLUMN,
                complete_while_typing=True,
                reserve_space_for_menu=6,
                style=PROMPT_STYLE,
                bottom_toolbar=self._build_bottom_toolbar,
            )

        def _build_bottom_toolbar(self):
            user_input = self._session.default_buffer.text
            return HTML(build_prompt_toolbar_markup(user_input))


def build_prompt_toolbar_markup(user_input: str) -> str:
    """将命令提醒渲染为 prompt_toolkit 底部工具栏。"""

    lines = build_command_hint_lines(user_input, limit=6)
    markup_lines = [
        (
            f'<style fg="{TEXT_PRIMARY}" bg="{SURFACE_BG}">'
            "命令提醒"
            "</style>"
        )
    ]

    for line in lines:
        command, separator, description = line.partition("  ")
        if separator:
            markup_lines.append(
                (
                    f'<style fg="{COMMAND_NAME_COLOR}" bg="{SURFACE_BG}">'
                    f"{escape(command)}"
                    "</style>"
                    f'<style fg="{COMMAND_DESC_COLOR}" bg="{SURFACE_BG}">'
                    f"{escape(separator + description)}"
                    "</style>"
                )
            )
            continue

        markup_lines.append(
            f'<style fg="{TEXT_MUTED}" bg="{SURFACE_BG}">{escape(line)}</style>'
        )

    return "\n".join(markup_lines)
