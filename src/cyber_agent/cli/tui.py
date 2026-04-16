from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich import box
from rich.panel import Panel
from rich.text import Text

from ..config import settings
from .interactive import (
    build_session_overview,
    build_command_hint_lines,
    get_auto_completion,
    get_banner_command_summary,
    list_builtin_command_names,
    match_builtin_commands,
)
from .theme import (
    COMMAND_DESC_STYLE,
    COMMAND_NAME_STYLE,
    HINT_TITLE_STYLE,
    KEYCAP_STYLE,
    PANEL_BORDER,
    ROLE_STYLES,
    SURFACE_BG,
    SYSTEM_LABEL_STYLE,
    SYSTEM_VALUE_STYLE,
    SYSTEM_VALUE_STYLES,
    TEXT_MUTED,
    WINDOW_BG,
)

try:
    from textual import work
    from textual.app import App, ComposeResult
    from textual.containers import Container, ScrollableContainer
    from textual.widgets import Footer, Header, Input, Static

    try:
        from textual.suggester import SuggestFromList
    except ModuleNotFoundError:
        SuggestFromList = None

    TEXTUAL_IMPORT_ERROR: ModuleNotFoundError | None = None
except ModuleNotFoundError as exc:  # pragma: no cover - 运行环境缺依赖时走降级
    TEXTUAL_IMPORT_ERROR = exc


if TEXTUAL_IMPORT_ERROR is None:

    class ChatMessage(Static):
        """用于显示聊天消息的富文本气泡。"""

        def __init__(self, role: str, content: str | Text) -> None:
            super().__init__()
            self.role = role
            self.content = content
            self._refresh_renderable()

        def set_content(self, content: str | Text) -> None:
            self.content = content
            self._refresh_renderable()

        def append_content(self, content: str) -> None:
            if isinstance(self.content, Text):
                self.content.append(content, style=ROLE_STYLES[self.role]["text_style"])
            else:
                self.content += content
            self._refresh_renderable()

        def has_content(self) -> bool:
            if isinstance(self.content, Text):
                return bool(self.content.plain.strip())
            return bool(self.content.strip())

        def _refresh_renderable(self) -> None:
            style = ROLE_STYLES.get(self.role, ROLE_STYLES["system"])
            if isinstance(self.content, Text):
                if self.content.plain.strip():
                    message_text = self.content.copy()
                else:
                    message_text = Text("正在处理...", style=style["text_style"])
            else:
                message_text = Text(
                    self.content.strip() or "正在处理...",
                    style=style["text_style"],
                )
            self.update(
                Panel(
                    message_text,
                    title=style["title"],
                    border_style=style["border_style"],
                    box=box.ROUNDED,
                    padding=(0, 1),
                )
            )


    class CyberAgentTUI(App):
        """基于 Textual 的交互式聊天界面。"""

        CSS = f"""
        Screen {
            background: {WINDOW_BG};
            color: #e2e8f0;
        }

        #chat-view {
            border: round #334155;
            background: {SURFACE_BG};
            height: 1fr;
            margin: 1 1 0 1;
            padding: 1;
        }

        #composer {
            border: round {PANEL_BORDER};
            background: {SURFACE_BG};
            margin: 0 1 1 1;
            padding: 0 1 1 1;
        }

        #composer-title {
            color: {TEXT_MUTED};
            padding: 0 0 1 0;
        }

        #chat-input {
            border: round #f59e0b;
            background: {WINDOW_BG};
            color: #f8fafc;
        }

        #chat-input:focus {
            border: round #14b8a6;
        }

        #command-hint {
            color: {TEXT_MUTED};
            padding: 1 0 0 0;
        }

        ChatMessage {
            margin: 0 0 1 0;
        }
        """

        BINDINGS = [
            ("tab", "accept_completion", "接受补全"),
        ]

        def __init__(
            self,
            runner: Any,
            runtime_context: dict[str, object],
            *,
            show_banner: bool = True,
        ) -> None:
            super().__init__()
            self.runner = runner
            self.runtime_context = runtime_context
            self.show_banner = show_banner
            self._is_busy = False
            self._active_assistant_message: ChatMessage | None = None

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            yield ScrollableContainer(id="chat-view")
            with Container(id="composer"):
                yield Static(self._build_composer_title(), id="composer-title")
                yield self._build_input_widget()
                yield Static(id="command-hint")
            yield Footer()

        def on_mount(self) -> None:
            self._update_command_hint("")
            self.query_one("#chat-input", Input).focus()
            if self.show_banner:
                self._add_message("system", self._build_welcome_message())

        def action_accept_completion(self) -> None:
            if self._is_busy:
                return

            input_widget = self.query_one("#chat-input", Input)
            suggestion = get_auto_completion(input_widget.value)
            if suggestion is None:
                return

            input_widget.value = suggestion
            if hasattr(input_widget, "cursor_position"):
                input_widget.cursor_position = len(suggestion)
            self._update_command_hint(suggestion)

        def on_input_changed(self, event: Input.Changed) -> None:
            self._update_command_hint(event.value)

        def on_input_submitted(self, event: Input.Submitted) -> None:
            user_input = event.value.strip()
            if not user_input or self._is_busy:
                return

            event.input.value = ""
            self._update_command_hint("")
            self._add_message("user", user_input)

            from .app import capture_builtin_command_output

            builtin_result, output = capture_builtin_command_output(
                user_input,
                self.runner,
                self.runtime_context,
            )
            if builtin_result is False:
                if output:
                    self._add_message("system", Text.from_ansi(output))
                self.exit()
                return
            if builtin_result is True:
                if output:
                    self._add_message("system", Text.from_ansi(output))
                return

            self._set_busy(True)
            self._active_assistant_message = self._add_message("assistant", "正在思考...")
            self._run_agent(user_input)

        @work(thread=True)
        def _run_agent(self, user_input: str) -> None:
            from .app import create_approval_handler

            def event_handler(event_type: str, payload: object) -> None:
                if event_type == "response_begin":
                    self.call_from_thread(self._set_assistant_content, "")
                    return
                if event_type == "response_token":
                    self.call_from_thread(self._append_assistant_content, str(payload))
                    return
                if event_type == "response_end" and isinstance(payload, dict):
                    content = str(payload.get("content", ""))
                    has_tool_calls = bool(payload.get("has_tool_calls", False))
                    if content and not has_tool_calls:
                        self.call_from_thread(self._set_assistant_content, content)
                        return
                    if has_tool_calls:
                        self.call_from_thread(self._set_assistant_content, "正在调用工具...")
                    return
                if event_type == "tool_call":
                    formatted = json.dumps(payload, ensure_ascii=False, indent=2)
                    self.call_from_thread(
                        self._add_message,
                        "system",
                        f"工具调用\n{formatted}",
                    )
                    return
                if event_type == "tool_result":
                    content = ""
                    if isinstance(payload, dict):
                        content = str(payload.get("content", ""))
                    else:
                        content = str(payload)
                    self.call_from_thread(
                        self._add_message,
                        "system",
                        f"工具结果\n{content}",
                    )
                    return
                if event_type == "approval_request":
                    formatted = json.dumps(payload, ensure_ascii=False, indent=2)
                    self.call_from_thread(
                        self._add_message,
                        "system",
                        f"审批请求\n{formatted}",
                    )
                    return
                if event_type == "approval_result":
                    formatted = json.dumps(payload, ensure_ascii=False, indent=2)
                    self.call_from_thread(
                        self._add_message,
                        "system",
                        f"审批结果\n{formatted}",
                    )
                    return

            try:
                final_response = self.runner.run(
                    user_input,
                    verbose=False,
                    event_handler=event_handler,
                    approval_handler=create_approval_handler(self.runtime_context),
                )
                if final_response:
                    self.call_from_thread(
                        self._ensure_final_assistant_content,
                        final_response,
                    )
            except Exception as exc:  # noqa: BLE001 - 终端界面需要直接反馈真实异常
                self.call_from_thread(
                    self._replace_assistant_with_error,
                    f"运行失败：{exc}",
                )
            finally:
                self.call_from_thread(self._finish_request)

        def _build_input_widget(self) -> Input:
            input_kwargs: dict[str, object] = {
                "placeholder": "输入消息，或输入 /help 查看命令",
                "id": "chat-input",
            }
            if SuggestFromList is not None:
                try:
                    input_kwargs["suggester"] = SuggestFromList(
                        list_builtin_command_names(),
                        case_sensitive=False,
                    )
                except TypeError:
                    input_kwargs["suggester"] = SuggestFromList(
                        list_builtin_command_names(),
                    )
            try:
                return Input(**input_kwargs)
            except TypeError:
                input_kwargs.pop("suggester", None)
                return Input(**input_kwargs)

        def _build_composer_title(self) -> Text:
            composer_title = Text(style=COMMAND_DESC_STYLE)
            composer_title.append("输入消息后按 ")
            composer_title.append("Enter", style=KEYCAP_STYLE)
            composer_title.append(" 发送，按 ")
            composer_title.append("Tab", style=KEYCAP_STYLE)
            composer_title.append(" 接受补全，输入 ")
            composer_title.append("/", style=COMMAND_NAME_STYLE)
            composer_title.append(" 可查看命令。")
            return composer_title

        def _build_welcome_message(self) -> Text:
            welcome = Text()
            welcome.append("Cyber Agent CLI 交互界面\n", style="bold #f8fafc")
            for item in build_session_overview(
                mode_value=self.runner.mode.value,
                approval_policy_value=self.runtime_context["approval_policy"].value,
                service=self.runner.service,
                model=settings.openai_model,
                cwd=str(Path.cwd()),
            ):
                self._append_system_kv_line(
                    welcome,
                    item.label,
                    item.value,
                    SYSTEM_VALUE_STYLES.get(item.value_style_key, SYSTEM_VALUE_STYLE),
                )

            welcome.append("快捷命令", style=SYSTEM_LABEL_STYLE)
            welcome.append("：", style=SYSTEM_LABEL_STYLE)
            command_summary = get_banner_command_summary().split("  ")
            for index, command in enumerate(command_summary):
                if index > 0:
                    welcome.append("  ", style=SYSTEM_LABEL_STYLE)
                welcome.append(command, style=COMMAND_NAME_STYLE)
            welcome.append("\n")

            welcome.append("命令补全", style=SYSTEM_LABEL_STYLE)
            welcome.append("：", style=SYSTEM_LABEL_STYLE)
            welcome.append("输入 ", style=COMMAND_DESC_STYLE)
            welcome.append("/", style=COMMAND_NAME_STYLE)
            welcome.append(" 后按 ", style=COMMAND_DESC_STYLE)
            welcome.append("Tab", style=KEYCAP_STYLE)
            welcome.append(" 可自动补全。", style=COMMAND_DESC_STYLE)
            return welcome

        def _update_command_hint(self, user_input: str) -> None:
            self.query_one("#command-hint", Static).update(
                self._build_command_hint(user_input)
            )

        def _build_command_hint(self, user_input: str) -> Text:
            hint = Text()
            hint.append("命令提醒\n", style=HINT_TITLE_STYLE)

            matches = match_builtin_commands(user_input.strip(), limit=6)
            if user_input.strip().startswith("/") and not matches:
                hint.append(
                    "未匹配到内建命令，可输入 ",
                    style=COMMAND_DESC_STYLE,
                )
                hint.append("/help", style=COMMAND_NAME_STYLE)
                hint.append(" 查看完整命令列表。", style=COMMAND_DESC_STYLE)
                return hint

            if not matches:
                for line in build_command_hint_lines(user_input, limit=6):
                    hint.append(line, style=COMMAND_DESC_STYLE)
                    hint.append("\n")
                return hint

            for index, item in enumerate(matches):
                hint.append(item.command, style=COMMAND_NAME_STYLE)
                hint.append("  ", style=COMMAND_DESC_STYLE)
                hint.append(item.description, style=COMMAND_DESC_STYLE)
                if index < len(matches) - 1:
                    hint.append("\n")
            return hint

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

        def _add_message(self, role: str, content: str | Text) -> ChatMessage:
            message = ChatMessage(role, content)
            chat_view = self.query_one("#chat-view", ScrollableContainer)
            chat_view.mount(message)
            chat_view.scroll_end(animate=False)
            return message

        def _set_assistant_content(self, content: str) -> None:
            if self._active_assistant_message is None:
                self._active_assistant_message = self._add_message("assistant", content)
                return
            self._active_assistant_message.set_content(content)
            self.query_one("#chat-view", ScrollableContainer).scroll_end(animate=False)

        def _append_assistant_content(self, content: str) -> None:
            if self._active_assistant_message is None:
                self._active_assistant_message = self._add_message("assistant", content)
                return
            self._active_assistant_message.append_content(content)
            self.query_one("#chat-view", ScrollableContainer).scroll_end(animate=False)

        def _ensure_final_assistant_content(self, content: str) -> None:
            if self._active_assistant_message is None:
                self._active_assistant_message = self._add_message("assistant", content)
                return
            if not self._active_assistant_message.has_content():
                self._active_assistant_message.set_content(content)

        def _replace_assistant_with_error(self, content: str) -> None:
            if self._active_assistant_message is None:
                self._active_assistant_message = self._add_message("error", content)
                return
            self._active_assistant_message.role = "error"
            self._active_assistant_message.set_content(content)

        def _finish_request(self) -> None:
            self._set_busy(False)
            self.query_one("#chat-input", Input).focus()
            self._active_assistant_message = None

        def _set_busy(self, value: bool) -> None:
            self._is_busy = value
            self.query_one("#chat-input", Input).disabled = value


def launch_textual_chat(
    runner: Any,
    runtime_context: dict[str, object],
    *,
    show_banner: bool = True,
) -> None:
    """在真实终端中启动 Textual 聊天界面。"""

    if TEXTUAL_IMPORT_ERROR is not None:  # pragma: no cover - 降级分支
        raise ModuleNotFoundError(str(TEXTUAL_IMPORT_ERROR)) from TEXTUAL_IMPORT_ERROR

    CyberAgentTUI(
        runner,
        runtime_context,
        show_banner=show_banner,
    ).run()
