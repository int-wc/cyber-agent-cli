from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import RenderableType
from rich.text import Text

from ..execution_control import ExecutionInterruptedError
from .branding import (
    STARTUP_ANIMATION_DELAY_SECONDS,
    STARTUP_ANIMATION_FRAMES,
    build_startup_frame,
)
from .interactive import (
    build_command_hint_lines,
    get_auto_completion,
    list_builtin_command_names,
    match_builtin_commands,
)
from .render import (
    build_approval_request_panel,
    build_approval_result_panel,
    build_banner_panel,
    build_chat_message_panel,
    build_tool_call_panel,
    build_tool_result_panel,
)
from .theme import (
    COMMAND_DESC_STYLE,
    COMMAND_NAME_STYLE,
    HINT_TITLE_STYLE,
    KEYCAP_STYLE,
    PANEL_BORDER,
    ROLE_STYLES,
    SURFACE_BG,
    TEXT_MUTED,
    WINDOW_BG,
)

try:
    from textual import work
    from textual.app import App, ComposeResult
    from textual.containers import Container, ScrollableContainer
    from textual.widgets import Input, Static

    try:
        from textual.suggester import SuggestFromList
    except ModuleNotFoundError:
        SuggestFromList = None

    TEXTUAL_IMPORT_ERROR: ModuleNotFoundError | None = None
except ModuleNotFoundError as exc:  # pragma: no cover - 运行环境缺依赖时走降级
    TEXTUAL_IMPORT_ERROR = exc


if TEXTUAL_IMPORT_ERROR is None:

    class RenderableBlock(Static):
        """用于在聊天区挂载共享 Rich 面板，避免 TUI 重新实现一套样式。"""

        def __init__(self, renderable: RenderableType) -> None:
            super().__init__()
            self.renderable = renderable
            self.update(renderable)


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
            self.update(build_chat_message_panel(self.role, self.content))


    class CyberAgentTUI(App):
        """基于 Textual 的交互式聊天界面。"""

        CSS = f"""
        Screen {{
            layers: base overlay;
            background: {WINDOW_BG};
            color: #e2e8f0;
        }}

        #chat-view {{
            border: round #334155;
            background: {SURFACE_BG};
            height: 1fr;
            margin: 1 1 0 1;
            padding: 1;
        }}

        #composer {{
            border: round {PANEL_BORDER};
            background: {SURFACE_BG};
            margin: 0 1 1 1;
            padding: 0 1 1 1;
        }}

        #composer-title {{
            color: {TEXT_MUTED};
            padding: 0 0 1 0;
        }}

        #chat-input {{
            border: round #f59e0b;
            background: {WINDOW_BG};
            color: #f8fafc;
        }}

        #chat-input:focus {{
            border: round #14b8a6;
        }}

        #command-hint {{
            color: {TEXT_MUTED};
            padding: 1 0 0 0;
        }}

        ChatMessage {{
            margin: 0 0 1 0;
        }}

        RenderableBlock {{
            margin: 0 0 1 0;
        }}

        #startup-view {{
            display: none;
            layer: overlay;
            width: 100%;
            height: 100%;
            content-align: center middle;
            background: {WINDOW_BG};
        }}

        #startup-panel {{
            width: auto;
            height: auto;
        }}
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
            self._startup_frame_index = 0
            self._startup_timer = None

        def compose(self) -> ComposeResult:
            yield ScrollableContainer(id="chat-view")
            with Container(id="composer"):
                yield Static(self._build_composer_title(), id="composer-title")
                yield self._build_input_widget()
                yield Static(id="command-hint")
            with Container(id="startup-view"):
                yield Static(id="startup-panel")

        def on_mount(self) -> None:
            self._update_command_hint("")
            if self.show_banner:
                self._start_startup_animation()
                return
            self._finish_startup(show_welcome=False)

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
            if not user_input:
                return

            event.input.value = ""
            self._update_command_hint("")

            if self._is_busy:
                if user_input.lower() == "/stop":
                    from .app import request_running_task_stop

                    request_running_task_stop(self.runtime_context)
                    self._add_message("system", "已收到 /stop，正在终止当前任务...")
                else:
                    self._add_message("system", "当前任务执行中，仅支持输入 /stop。")
                return

            self._add_message("user", user_input)

            from .app import capture_builtin_command_renderables

            builtin_result, renderables = capture_builtin_command_renderables(
                user_input,
                self.runner,
                self.runtime_context,
            )
            if builtin_result is False:
                for renderable in renderables:
                    self._add_renderable(renderable)
                self.exit()
                return
            if builtin_result is True:
                for renderable in renderables:
                    self._add_renderable(renderable)
                return

            self._set_busy(True)
            self._active_assistant_message = self._add_message("assistant", "正在思考...")
            self._run_agent(user_input)

        @work(thread=True)
        def _run_agent(self, user_input: str) -> None:
            from .app import create_approval_handler, persist_runtime_session

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
                    self.call_from_thread(
                        self._add_renderable,
                        build_tool_call_panel(payload if isinstance(payload, list) else []),
                    )
                    return
                if event_type == "tool_result":
                    content = ""
                    if isinstance(payload, dict):
                        content = str(payload.get("content", ""))
                    else:
                        content = str(payload)
                    self.call_from_thread(
                        self._add_renderable,
                        build_tool_result_panel(content),
                    )
                    return
                if event_type == "approval_request" and isinstance(payload, dict):
                    self.call_from_thread(
                        self._add_renderable,
                        build_approval_request_panel(payload),
                    )
                    return
                if event_type == "approval_result" and isinstance(payload, dict):
                    self.call_from_thread(
                        self._add_renderable,
                        build_approval_result_panel(payload),
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
            except ExecutionInterruptedError as exc:
                self.call_from_thread(
                    self._set_assistant_content,
                    str(exc),
                )
            except Exception as exc:  # noqa: BLE001 - 终端界面需要直接反馈真实异常
                self.call_from_thread(
                    self._replace_assistant_with_error,
                    f"运行失败：{exc}",
                )
            finally:
                persist_runtime_session(self.runner, self.runtime_context)
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

        def _build_welcome_panel(self) -> RenderableType:
            return build_banner_panel(
                mode=self.runner.mode,
                service=self.runner.service,
                model=self.runner.model_name,
                cwd=Path.cwd(),
                approval_policy=self.runtime_context["approval_policy"],
            )

        def _update_command_hint(self, user_input: str) -> None:
            self.query_one("#command-hint", Static).update(
                self._build_command_hint(user_input)
            )

        def _build_command_hint(self, user_input: str) -> Text:
            hint = Text()
            hint.append("命令提醒\n", style=HINT_TITLE_STYLE)

            if self._is_busy:
                hint.append("/stop", style=COMMAND_NAME_STYLE)
                hint.append("  停止当前正在执行的任务", style=COMMAND_DESC_STYLE)
                return hint

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

        def _add_message(self, role: str, content: str | Text) -> ChatMessage:
            message = ChatMessage(role, content)
            chat_view = self.query_one("#chat-view", ScrollableContainer)
            chat_view.mount(message)
            chat_view.scroll_end(animate=False)
            return message

        def _add_renderable(self, renderable: RenderableType) -> RenderableBlock:
            block = RenderableBlock(renderable)
            chat_view = self.query_one("#chat-view", ScrollableContainer)
            chat_view.mount(block)
            chat_view.scroll_end(animate=False)
            return block

        def _start_startup_animation(self) -> None:
            startup_view = self.query_one("#startup-view", Container)
            startup_view.display = True
            self._startup_frame_index = 0
            self.query_one("#startup-panel", Static).update(build_startup_frame(0))
            self._startup_timer = self.set_interval(
                STARTUP_ANIMATION_DELAY_SECONDS,
                self._advance_startup_animation,
            )

        def _advance_startup_animation(self) -> None:
            self._startup_frame_index += 1
            if self._startup_frame_index >= STARTUP_ANIMATION_FRAMES:
                if self._startup_timer is not None:
                    self._startup_timer.stop()
                    self._startup_timer = None
                self._finish_startup(show_welcome=True)
                return
            self.query_one("#startup-panel", Static).update(
                build_startup_frame(self._startup_frame_index)
            )

        def _finish_startup(self, *, show_welcome: bool) -> None:
            self.query_one("#startup-view", Container).display = False
            self.query_one("#chat-input", Input).focus()
            if show_welcome:
                self._add_renderable(self._build_welcome_panel())

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
            input_widget = self.query_one("#chat-input", Input)
            input_widget.disabled = False
            input_widget.placeholder = (
                "任务执行中，输入 /stop 立即中断当前任务"
                if value
                else "输入消息，或输入 /help 查看命令"
            )
            self._update_command_hint(input_widget.value)


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
