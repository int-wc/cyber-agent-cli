from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import RenderableType
from rich.text import Text

from ..agent.events import AgentEventType
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
    _estimate_cost,
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
    from textual.screen import Screen
    from textual.widgets import Button, Input, Static, TextArea

    try:
        from textual.suggester import SuggestFromList
    except ModuleNotFoundError:
        SuggestFromList = None

    try:
        from textual.widgets import Markdown as TextualMarkdown
        _TEXTUAL_MARKDOWN_AVAILABLE = True
    except ImportError:
        TextualMarkdown = None
        _TEXTUAL_MARKDOWN_AVAILABLE = False

    TEXTUAL_IMPORT_ERROR: ModuleNotFoundError | None = None
except ModuleNotFoundError as exc:  # pragma: no cover - 运行环境缺依赖时走降级
    TEXTUAL_IMPORT_ERROR = exc


if TEXTUAL_IMPORT_ERROR is None:

    class RenderableBlock(Static):
        """用于在聊天区挂载共享 Rich 面板，支持右键菜单。"""

        def __init__(
            self,
            renderable: RenderableType,
            *,
            role: str = "system",
            text_content: str = "",
        ) -> None:
            super().__init__()
            self.renderable = renderable
            self.role = role
            self._text_content = text_content or self._extract_text(renderable)
            self.update(renderable)

        @staticmethod
        def _extract_text(renderable: RenderableType) -> str:
            """从 Rich renderable 中递归提取纯文本，支持 Panel、Table、Text 等嵌套结构。"""
            from rich.panel import Panel as _RichPanel
            if isinstance(renderable, str):
                return renderable
            if isinstance(renderable, Text):
                return renderable.plain
            # 对 Rich Panel 递归提取内部 renderable。
            if isinstance(renderable, _RichPanel):
                inner = renderable.renderable
                if inner is not None and inner is not renderable:
                    return RenderableBlock._extract_text(inner)
            # 对 Rich Table 遍历所有列的单元格（col.cells 是 col._cells 的生成器）。
            if hasattr(renderable, "columns"):
                parts: list[str] = []
                for col in renderable.columns:
                    if hasattr(col, "cells"):
                        for cell in col.cells:
                            txt = RenderableBlock._extract_text(cell)
                            if txt.strip():
                                parts.append(txt)
                return " ".join(parts)
            return str(renderable)

        def on_click(self, event) -> None:
            """右键点击弹出上下文菜单。"""
            if getattr(event, "button", 1) == 3:
                event.stop()
                self.app._show_context_menu(self)


    class ChatMessage(Static):
        """用于显示聊天消息的富文本气泡，支持可选 Markdown 渲染。"""

        ALLOW_SELECT = True
        FOCUS_ON_CLICK = True
        can_focus = True

        def __init__(
            self,
            role: str,
            content: str | Text,
            *,
            use_markdown: bool = False,
        ) -> None:
            super().__init__()
            self.role = role
            self._text: str | Text = content
            self._use_markdown = use_markdown and _TEXTUAL_MARKDOWN_AVAILABLE and role == "assistant"
            self._refresh_renderable()

        # ── 焦点静态指示条 ──────────────────────────────────────

        def on_focus(self) -> None:
            """获得焦点：背景加深 + 左侧 rose 指示条。"""
            from textual.color import Color as _Color
            self.styles.background = _Color(26, 35, 50)  # #1a2332
            self.styles.border_left = ("heavy", _Color(244, 61, 94))

        def on_blur(self) -> None:
            """失去焦点：清除样式。"""
            self.styles.border_left = None
            self.styles.background = None

        def set_content(self, content: str | Text) -> None:
            self._text = content
            self._refresh_renderable()

        def append_content(self, content: str) -> None:
            if isinstance(self._text, Text):
                self._text.append(content, style=ROLE_STYLES[self.role]["text_style"])
            else:
                self._text += content
            self._refresh_renderable()

        def has_content(self) -> bool:
            if isinstance(self._text, Text):
                return bool(self._text.plain.strip())
            return bool(self._text.strip())

        @staticmethod
        def _render_markdown_to_rich(content: str) -> RenderableType:
            """预渲染 Markdown 到 Rich Text（保留样式），避免 Rich Markdown 在 Textual 中渲染丢失样式。"""
            try:
                from rich.console import Console as _RichConsole
                from rich.markdown import Markdown as _RichMd
                from rich.text import Text as _RichText
                _buf = _RichConsole(width=240, color_system="truecolor", force_terminal=True)
                with _buf.capture() as _cap:
                    _buf.print(_RichMd(content, code_theme="monokai"))
                _styled = _cap.get()
                if _styled.strip():
                    return _RichText.from_ansi(_styled)
            except Exception:
                pass
            return content

        def _refresh_renderable(self) -> None:
            if self._use_markdown and isinstance(self._text, str) and self._text.strip():
                rendered = self._render_markdown_to_rich(self._text)
                self.update(build_chat_message_panel(self.role, rendered))
                return
            self.update(build_chat_message_panel(self.role, self._text))

        def on_click(self, event) -> None:
            """右键点击弹出上下文菜单。"""
            if getattr(event, "button", 1) == 3:
                event.stop()
                self.app._show_context_menu(self)


    class ContextMenuScreen(Screen[None]):
        """右键上下文菜单——支持 ChatMessage 与 RenderableBlock。"""

        ALLOW_SELECT = True

        def __init__(
            self, widget: ChatMessage | RenderableBlock, selected_text: str,
        ) -> None:
            super().__init__()
            self._widget = widget
            self._selected_text = selected_text

        def compose(self) -> ComposeResult:
            with Container(id="ctx-panel"):
                yield Static("选择操作", id="ctx-title")
                yield Button("📋 复制全文", id="copy-all")
                if self._selected_text:
                    yield Button("✂️ 复制选中", id="copy-selected")
                yield Button("🪟 查看详情", id="view-detail")
                yield Button("✕ 关闭", id="close")

        CSS = f"""
        ContextMenuScreen {{
            align: center middle;
            background: rgba(0,0,0,0.35);
        }}
        #ctx-panel {{
            width: auto;
            min-width: 28;
            height: auto;
            background: {SURFACE_BG};
            border: round #14b8a6;
            padding: 1 2;
        }}
        #ctx-title {{
            text-align: center;
            color: {TEXT_MUTED};
            margin: 0 0 1 0;
        }}
        #ctx-panel > Button {{
            width: 100%;
            margin: 0 0 1 0;
        }}
        """

        BINDINGS = [("escape", "close_menu", "关闭")]

        def action_close_menu(self) -> None:
            self.app.pop_screen()

        def on_button_pressed(self, event: Button.Pressed) -> None:
            bid = event.button.id
            if bid == "view-detail":
                # 先弹菜单，再推详情屏，防止 pop_screen 误弹
                self.app.pop_screen()
                self.app.push_screen(MessageDetailScreen(self._widget))
                return
            if bid == "copy-all":
                self._copy_all()
            elif bid == "copy-selected":
                self._copy_selected()
            self.app.pop_screen()

        def _copy_all(self) -> None:
            content = self._get_widget_text(self._widget)
            self.app.copy_to_clipboard(content)
            self.app.notify("✅ 已复制全文", severity="information")

        @staticmethod
        def _get_widget_text(widget: object) -> str:
            """从 ChatMessage 或 RenderableBlock 中提取纯文本。"""
            if hasattr(widget, "_text"):
                t = getattr(widget, "_text")
                return t.plain if isinstance(t, Text) else str(t)
            if hasattr(widget, "_text_content"):
                return str(getattr(widget, "_text_content"))
            return str(widget)

        def _copy_selected(self) -> None:
            self.app.copy_to_clipboard(self._selected_text)
            self.app.notify("✅ 已复制选中文字", severity="information")


    class MessageDetailScreen(Screen[None]):
        """消息详情全屏查看——支持 ChatMessage 与 RenderableBlock。"""

        ALLOW_SELECT = True

        def __init__(self, widget: ChatMessage | RenderableBlock) -> None:
            super().__init__()
            self._widget = widget

        @staticmethod
        def _get_widget_text(widget: object) -> str:
            if hasattr(widget, "_text"):
                t = getattr(widget, "_text")
                return t.plain if isinstance(t, Text) else str(t)
            if hasattr(widget, "_text_content"):
                return str(getattr(widget, "_text_content"))
            return str(widget)

        @staticmethod
        def _get_widget_role(widget: object) -> str:
            return str(getattr(widget, "role", "system"))

        def compose(self) -> ComposeResult:
            content = self._get_widget_text(self._widget)
            role = self._get_widget_role(self._widget)
            with Container(id="detail-container"):
                with Container(id="title-bar"):
                    yield Static(
                        f"📄 消息详情 — {role}",
                        id="detail-title",
                    )
                    yield Static("Ctrl+Shift+G 复制选中 · Ctrl+Shift+Y 全部复制", id="detail-hint")
                    yield Button("✕ 关闭", id="title-close", variant="warning")
                text_area = TextArea(content, id="detail-content")
                text_area.read_only = True
                yield text_area
                yield Static(
                    "Esc 关闭 · 鼠标拖拽选中",
                    id="detail-footer",
                )

        CSS = f"""
        MessageDetailScreen {{
            align: center middle;
            background: rgba(0,0,0,0.7);
        }}
        #detail-container {{
            width: 86%;
            height: 82%;
            background: {WINDOW_BG};
            border: round #14b8a6;
            padding: 1 1 0 1;
            layout: vertical;
        }}
        #title-bar {{
            layout: horizontal;
            height: 3;
            width: 100%;
        }}
        #detail-title {{
            width: auto;
            height: 100%;
            padding: 0 1;
            color: #14b8a6;
            text-style: bold;
            content-align: left middle;
        }}
        #detail-hint {{
            width: 1fr;
            height: 100%;
            padding: 0 1;
            color: $secondary;
            text-style: italic;
            content-align: center middle;
        }}
        #title-close {{
            width: auto;
            min-width: 10;
            height: 100%;
            margin: 0;
        }}
        #detail-content {{
            height: 1fr;
            margin: 0 0 1 0;
            border: none;
        }}
        #detail-footer {{
            text-align: center;
            color: $secondary;
            height: 1;
            padding-bottom: 1;
        }}
        """

        BINDINGS = [
            ("escape", "close_detail", "关闭"),
            ("ctrl+shift+g", "copy_selected", "复制选中"),
            ("ctrl+shift+y", "copy_all", "全部复制"),
            ("ctrl+c", "ignore_detail", ""),
        ]

        def action_close_detail(self) -> None:
            self.app.pop_screen()

        def action_ignore_detail(self) -> None:
            pass

        def action_copy_selected(self) -> None:
            """复制选中文字。"""
            text_area = self.query_one("#detail-content", TextArea)
            selected = text_area.selected_text or ""
            if selected:
                self.app.copy_to_clipboard(selected)
                self.app.notify("✅ 已复制选中文字", severity="information")
            else:
                self.app.notify("⚠️ 请先拖拽选中文字", severity="warning")

        def action_copy_all(self) -> None:
            """复制全部内容。"""
            content = self._get_widget_text(self._widget)
            if content.strip():
                self.app.copy_to_clipboard(content)
                self.app.notify("✅ 已复制全部内容", severity="information")

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "title-close":
                self.app.pop_screen()


    class CyberAgentTUI(App):
        """基于 Textual 的交互式聊天界面。"""

        CSS = f"""
        Screen {{
            layers: base overlay;
            background: {WINDOW_BG};
            color: #e2e8f0;
        }}

        #chat-view {{
            border: none;
            background: {SURFACE_BG};
            margin: 0;
            padding: 1 2;
        }}

        #composer {{
            dock: bottom;
            background: {SURFACE_BG};
            border-top: solid #334155;
            height: auto;
            margin: 0;
            padding: 0 1;
        }}

        #composer-top-bar {{
            layout: horizontal;
            height: auto;
            width: 100%;
            margin: 0 0 1 0;
        }}

        #composer-title {{
            color: {TEXT_MUTED};
            height: auto;
            width: 1fr;
        }}

        #token-status {{
            color: {TEXT_MUTED};
            height: auto;
            width: auto;
            text-align: right;
            padding: 0 1;
        }}

        #chat-input {{
            border: round #f59e0b;
            background: {WINDOW_BG};
            color: #f8fafc;
            height: auto;
        }}

        #chat-input:focus {{
            border: round #14b8a6;
        }}

        #command-hint {{
            color: {TEXT_MUTED};
            height: auto;
            max-height: 4;
            overflow-y: auto;
            margin: 1 0 0 0;
        }}

        ChatMessage {{
            margin: 0 0 1 0;
        }}

        ChatMessage:focus {{
            /* 动画由 on_focus / on_blur 程序化控制 */
        }}

        ChatMessage:focus > .rich-text {{
            background: #1a2332;
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
            ("ctrl+y", "copy_last_response", "复制最后回复"),
            ("ctrl+b", "toggle_compressed_history", "压缩历史"),
            ("ctrl+q", "quit_app", "退出程序"),
            ("ctrl+c", "cancel_task", "暂停/取消"),
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
            self._reasoning_parts: list[str] = []
            self._reasoning_message: ChatMessage | None = None
            self._startup_frame_index = 0
            self._startup_timer = None
            # 累计 token / 花费统计
            self._cumulative_input_tokens = 0
            self._cumulative_output_tokens = 0
            self._cumulative_cost = 0.0
            # 流式 token 估算（用于实时刷新状态栏）
            self._current_response_chars = 0   # 当前响应段已收到的字符数
            self._estimated_output_tokens = 0   # 本轮估算的 output token 总和
            # 压缩历史视图状态
            self._compression_visible = False
            self._compression_widget: Static | None = None
            # 最近一次助手回复原文（用于复制等操作）
            self._last_assistant_raw_text: str = ""
            # 输入历史导航（上下键）
            self._history_index = -1
            self._history_draft: str = ""

        def compose(self) -> ComposeResult:
            yield ScrollableContainer(id="chat-view")
            with Container(id="composer"):
                with Container(id="composer-top-bar"):
                    yield Static(self._build_composer_title(), id="composer-title")
                    yield Static(id="token-status")
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

        def on_key(self, event) -> None:
            """全局键盘事件：输入框聚焦时上下键切换输入历史。

            注意：不能用 on_input_key——Textual 没有 Input.Key 事件，
            键盘事件统一通过 App/Screen 层的 on_key 分发。
            """
            focused = self.focused
            if focused is None:
                return
            # 只拦截 chat-input 获得焦点时的 up/down
            if getattr(focused, "id", None) == "chat-input":
                if event.key == "up":
                    event.stop()
                    self._navigate_history(-1)
                elif event.key == "down":
                    event.stop()
                    self._navigate_history(1)

        def _navigate_history(self, direction: int) -> None:
            """沿输入历史向前（-1=UP）或向后（1=DOWN）导航。"""
            input_widget = self.query_one("#chat-input", Input)
            recent = self.runtime_context.get("_recent_inputs", [])
            if not recent:
                return

            if direction == -1:  # UP → 更旧
                if self._history_index == -1:
                    # 首次离开当前输入，保存草稿
                    self._history_draft = input_widget.value
                new_index = min(self._history_index + 1, len(recent) - 1)
                if new_index == self._history_index:
                    return
                self._history_index = new_index
                input_widget.value = recent[-(self._history_index + 1)]

            elif direction == 1:  # DOWN → 更新
                if self._history_index == -1:
                    return  # 已经在最新位置
                self._history_index -= 1
                if self._history_index == -1:
                    input_widget.value = self._history_draft
                else:
                    input_widget.value = recent[-(self._history_index + 1)]

            if hasattr(input_widget, "cursor_position"):
                input_widget.cursor_position = len(input_widget.value)

        def on_input_submitted(self, event: Input.Submitted) -> None:
            user_input = event.value.strip()
            if not user_input:
                return

            # 提交后重置历史导航
            self._history_index = -1
            self._history_draft = ""

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

            # 记录最近输入历史（上限 50 条）
            recent = self.runtime_context.setdefault("_recent_inputs", [])
            recent.append(user_input)
            if len(recent) > 50:
                recent.pop(0)

            from .app import capture_builtin_command_renderables
            from .session_runtime import (
                append_runtime_session_event,
                persist_runtime_session,
            )

            append_runtime_session_event(
                self.runtime_context,
                "user_input_received",
                {"input": user_input},
            )
            persist_runtime_session(self.runner, self.runtime_context, force=True)

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
                if self.runtime_context.pop("__clear_visible_session", False):
                    self._clear_chat_view()
                for renderable in renderables:
                    self._add_renderable(renderable)
                # session 切换后刷新标题栏
                self._refresh_composer_title()
                return

            self._set_busy(True)
            self._run_agent(user_input)

        @work(thread=True)
        def _run_agent(self, user_input: str) -> None:
            from .app import create_approval_handler
            from .session_runtime import (
                create_persisting_event_handler,
                persist_runtime_session,
            )

            def event_handler(event_type: str, payload: object) -> None:
                if event_type == AgentEventType.REASONING_TOKEN:
                    self.call_from_thread(self._append_reasoning, str(payload))
                    return
                if event_type == AgentEventType.RESPONSE_BEGIN:
                    self._current_response_chars = 0
                    self.call_from_thread(self._flush_reasoning)
                    self.call_from_thread(self._set_assistant_content, "")
                    return
                if event_type == AgentEventType.RESPONSE_TOKEN:
                    self._current_response_chars += 1
                    self.call_from_thread(self._append_assistant_content, str(payload))
                    return
                if event_type == AgentEventType.RESPONSE_END and isinstance(payload, dict):
                    self.call_from_thread(self._on_response_segment_end)
                    content = str(payload.get("content", ""))
                    has_tool_calls = bool(payload.get("has_tool_calls", False))
                    if content and not has_tool_calls:
                        self.call_from_thread(
                            self._ensure_final_assistant_content, content,
                        )
                        return
                    if has_tool_calls:
                        self.call_from_thread(self._set_assistant_content, "正在调用工具...")
                    return
                if event_type == AgentEventType.TOOL_CALL:
                    self._active_assistant_message = None  # 关闭当前消息，后续回复创建新消息
                    self.call_from_thread(
                        self._add_renderable,
                        build_tool_call_panel(payload if isinstance(payload, list) else []),
                    )
                    return
                if event_type == AgentEventType.TOOL_RESULT:
                    content = ""
                    if isinstance(payload, dict):
                        content = str(payload.get("content", ""))
                    else:
                        content = str(payload)
                    self.call_from_thread(self._add_tool_result, content)
                    return
                if event_type == AgentEventType.APPROVAL_REQUEST and isinstance(payload, dict):
                    self.call_from_thread(
                        self._add_renderable,
                        build_approval_request_panel(payload),
                    )
                    return
                if event_type == AgentEventType.APPROVAL_RESULT and isinstance(payload, dict):
                    self.call_from_thread(
                        self._add_renderable,
                        build_approval_result_panel(payload),
                    )
                    return
                if event_type == AgentEventType.TURN_END and isinstance(payload, dict):
                    self.call_from_thread(self._show_token_usage, payload)
                    return

            # 判断是否使用多 Agent 编排
            from .app import _detect_task_complexity
            multi_setting = self.runtime_context.get("multi_agent_enabled", "auto")
            if multi_setting is True or (
                multi_setting == "auto" and _detect_task_complexity(user_input)
            ):
                self._run_multi_agent_turn(user_input)
                persist_runtime_session(self.runner, self.runtime_context)
                self.call_from_thread(self._finish_request)
                return

            try:
                final_response = self.runner.run(
                    user_input,
                    verbose=False,
                    event_handler=create_persisting_event_handler(
                        self.runner,
                        self.runtime_context,
                        event_handler,
                    ),
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

        def _run_multi_agent_turn(self, user_input: str) -> None:
            """使用四柱管线（FourPillarPipeline）替代旧的多 Agent 编排器。"""
            import re as _re
            from ..agent.pipeline import FourPillarPipeline
            from .app import ensure_runtime_capabilities
            from .render import CliRenderer
            from rich.console import Console

            ensure_runtime_capabilities(self.runtime_context, self.runner)

            self.call_from_thread(
                self._set_assistant_content,
                "🚀 正在启动四柱 Agent 管线...",
            )

            auto_decision = bool(self.runtime_context.get("auto_decision", False))
            real_console = Console()

            # ── 控制台转发器：将管线输出同时发往 stdout 和 TUI 聊天视图 ──
            class _PipelineTuiForwarder:
                """将 Rich Console.print() 的输出同时转发到 TUI 聊天视图。"""

                def __init__(self, tui_app: "CyberAgentTUI") -> None:
                    self._console = real_console
                    self._tui = tui_app
                    self._last_text = ""

                def print(self, *args: object, **kwargs: object) -> None:  # noqa: A003
                    self._console.print(*args, **kwargs)
                    text = self._extract_plain(*args)
                    if text:
                        self._last_text = text
                        self._tui.call_from_thread(
                            self._tui._append_assistant_content,
                            text + "\n",
                        )

                @staticmethod
                def _extract_plain(*args: object) -> str:
                    """将 Rich print 参数转为纯文本，去除 Rich 标记样式。"""
                    parts: list[str] = []
                    for arg in args:
                        if arg is None:
                            continue
                        if isinstance(arg, str):
                            parts.append(_re.sub(r'\[/?\w+(?: [^\]]+)?\]', '', arg))
                        elif hasattr(arg, 'plain'):
                            parts.append(str(getattr(arg, 'plain', '')))
                        else:
                            parts.append(str(arg))
                    return " ".join(parts).strip()

            # ── 创建 TUI 兼容的 renderer ──
            tui_console = _PipelineTuiForwarder(self)
            tui_renderer = CliRenderer(console=tui_console)  # type: ignore[arg-type]

            # 接管 print_markdown：最终摘要以 Markdown 渲染到聊天视图
            _original_print_md = tui_renderer.print_markdown
            def _tui_print_markdown(content: str) -> None:
                _original_print_md(content)
                if content.strip():
                    self.call_from_thread(
                        self._ensure_final_assistant_content,
                        content,
                    )
            tui_renderer.print_markdown = _tui_print_markdown  # type: ignore[method-assign]

            # 接管 add_token_usage：同步 token 到状态栏
            _original_add_token = tui_renderer.add_token_usage
            def _tui_add_token_usage(in_tokens: int, out_tokens: int) -> None:
                _original_add_token(in_tokens, out_tokens)
                self.call_from_thread(self._update_token_status)
            tui_renderer.add_token_usage = _tui_add_token_usage  # type: ignore[method-assign]

            pipeline = FourPillarPipeline(
                runner=self.runner,
                runtime_context=self.runtime_context,
                renderer=tui_renderer,
            )

            try:
                pipeline.run(user_input, auto_decision=auto_decision)
            except Exception as exc:
                self.call_from_thread(
                    self._replace_assistant_with_error,
                    f"四柱管线执行失败：{exc}",
                )
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
            composer_title.append("Enter", style=KEYCAP_STYLE)
            composer_title.append(" 发送，")
            composer_title.append("Tab", style=KEYCAP_STYLE)
            composer_title.append(" 补全，")
            composer_title.append("/", style=COMMAND_NAME_STYLE)
            composer_title.append(" 命令，")
            composer_title.append("右键", style=KEYCAP_STYLE)
            composer_title.append(" 菜单，")
            composer_title.append("Ctrl+Y", style=KEYCAP_STYLE)
            composer_title.append(" 复制，")
            composer_title.append("Ctrl+Q", style=KEYCAP_STYLE)
            composer_title.append(" 退出，")
            composer_title.append("Ctrl+C", style=KEYCAP_STYLE)
            composer_title.append(" 取消")
            # 会话标识
            session_id = str(self.runtime_context.get("session_id", ""))
            if session_id:
                short_id = session_id[-12:] if len(session_id) > 12 else session_id
                composer_title.append(f" │ {short_id}", style="dim #64748b")
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

            matches = match_builtin_commands(user_input.strip(), limit=3)
            if user_input.strip().startswith("/") and not matches:
                hint.append(
                    "未匹配到内建命令，可输入 ",
                    style=COMMAND_DESC_STYLE,
                )
                hint.append("/help", style=COMMAND_NAME_STYLE)
                hint.append(" 查看完整命令列表。", style=COMMAND_DESC_STYLE)
                return hint

            if not matches:
                for line in build_command_hint_lines(user_input, limit=3):
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

        def _add_message(
            self, role: str, content: str | Text, *, use_markdown: bool = False,
        ) -> ChatMessage:
            message = ChatMessage(role, content, use_markdown=use_markdown)
            chat_view = self.query_one("#chat-view", ScrollableContainer)
            chat_view.mount(message)
            chat_view.scroll_end(animate=False)
            return message

        def _append_reasoning(self, content: str) -> None:
            """追加思考过程文本。"""
            self._reasoning_parts.append(content)
            if self._reasoning_message is None:
                self._reasoning_message = self._add_message("system", "")
                self._reasoning_message.role = "reasoning"
            self._reasoning_message.set_content("".join(self._reasoning_parts))

        def _flush_reasoning(self) -> None:
            """思考阶段结束，重置累积状态。"""
            self._reasoning_parts = []
            self._reasoning_message = None

        def _on_response_segment_end(self) -> None:
            """一段响应结束，用字符数估算 output token 并实时刷新状态栏。"""
            if self._current_response_chars > 0:
                # 中英文平均约 4 字符 = 1 token
                estimated = max(1, self._current_response_chars // 4)
                self._estimated_output_tokens += estimated
                self._cumulative_output_tokens += estimated
                self._current_response_chars = 0
                self._update_token_status()

        def _show_token_usage(self, usage: dict[str, int]) -> None:
            """累计 token 消耗并更新底部统计栏——用实际值纠正 stream 期间的估算。"""
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            # 如果有尚未估算的残留字符，先估算
            if self._current_response_chars > 0:
                estimated = max(1, self._current_response_chars // 4)
                self._estimated_output_tokens += estimated
                self._cumulative_output_tokens += estimated
                self._current_response_chars = 0
            # 移除本轮估算值，替换为实际 API 返回值
            self._cumulative_output_tokens -= self._estimated_output_tokens
            self._cumulative_output_tokens += output_tokens
            self._estimated_output_tokens = 0
            # 输入 token 仅在 TURN_END 可知。
            self._cumulative_input_tokens += input_tokens
            cost = _estimate_cost(
                input_tokens, output_tokens,
                str(self.runtime_context.get("model_name", "")),
            )
            self._cumulative_cost += cost
            self._update_token_status()

        def _update_token_status(self) -> None:
            """更新底部持久状态栏。"""
            try:
                status = self.query_one("#token-status", Static)
                t = self._cumulative_input_tokens
                o = self._cumulative_output_tokens
                status.update(
                    f" 累计 ↑{t} ↓{o} Σ{t+o} │ ¥{self._cumulative_cost:.4f}"
                )
            except Exception:
                pass

        def _add_renderable(self, renderable: RenderableType) -> RenderableBlock:
            block = RenderableBlock(renderable)
            chat_view = self.query_one("#chat-view", ScrollableContainer)
            chat_view.mount(block)
            chat_view.scroll_end(animate=False)
            return block

        def _clear_chat_view(self) -> None:
            """清空当前可见聊天窗口，保持输入区和状态栏不变。"""
            chat_view = self.query_one("#chat-view", ScrollableContainer)
            for child in list(chat_view.children):
                try:
                    chat_view.remove(child)
                except Exception:
                    try:
                        child.remove()
                    except Exception:
                        pass
            self._active_assistant_message = None
            self._reasoning_parts = []
            self._reasoning_message = None
            self._compression_visible = False
            self._compression_widget = None
            self._last_assistant_raw_text = ""
            chat_view.scroll_home(animate=False)

        def _add_tool_result(self, content: str) -> None:
            """添加工具结果（纯 Panel，不做 markdown 解析）。"""
            chat_view = self.query_one("#chat-view", ScrollableContainer)
            chat_view.mount(RenderableBlock(
                build_tool_result_panel(content),
                role="tool",
                text_content=content,
            ))
            chat_view.scroll_end(animate=False)

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
            self._update_token_status()

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
            """确保最终输出以 Markdown 渲染。

            先预渲染 Markdown → Rich Text（保留样式），再放入 Panel，
            避免替换 widget 带来的滚动问题。
            """
            chat_view = self.query_one("#chat-view", ScrollableContainer)
            self._last_assistant_raw_text = content

            if self._active_assistant_message is None:
                self._active_assistant_message = self._add_message(
                    "assistant", content, use_markdown=True,
                )
                return
            # 已存在消息（如流式输出），强制启用 Markdown 并替换内容
            if _TEXTUAL_MARKDOWN_AVAILABLE:
                self._active_assistant_message._use_markdown = True
            self._active_assistant_message.set_content(content)
            chat_view.scroll_end(animate=False)

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

        def action_copy_last_response(self) -> None:
            """复制文本：优先选中文字 → 焦点消息全文 → 最后助手回复原文。"""
            # 1. 优先复制用户用鼠标选中的文字片段
            selected = self.screen.get_selected_text()
            if selected:
                self.copy_to_clipboard(selected)
                return

            # 2. 复制当前焦点消息的全文（ChatMessage 模式）
            focused = self.focused
            if isinstance(focused, ChatMessage):
                text = focused._text
                content = text.plain if isinstance(text, Text) else str(text)
                if content.strip():
                    self.copy_to_clipboard(content)
                    return

            # 3. 兜底：复制最后一条助手回复原文（兼容 Markdown 容器模式）
            if self._last_assistant_raw_text.strip():
                self.copy_to_clipboard(self._last_assistant_raw_text)
                return

            # 4. 更兜底：扫描 ChatMessage
            chat_view = self.query_one("#chat-view", ScrollableContainer)
            for child in reversed(list(chat_view.children)):
                if isinstance(child, ChatMessage) and child.role == "assistant":
                    text = child._text
                    content = text.plain if isinstance(text, Text) else str(text)
                    if content.strip():
                        self.copy_to_clipboard(content)
                    break

        def action_toggle_compressed_history(self) -> None:
            """切换显示/隐藏上下文压缩摘要。"""
            summary = getattr(self.runner, "compressed_summary", "") or ""
            if not summary.strip():
                return
            chat_view = self.query_one("#chat-view", ScrollableContainer)

            if self._compression_visible:
                # 隐藏压缩摘要
                if self._compression_widget is not None:
                    try:
                        chat_view.remove(self._compression_widget)
                    except Exception:
                        pass
                    self._compression_widget = None
                self._compression_visible = False
                return

            # 首次展开：创建并挂载压缩摘要面板
            if self._compression_widget is None:
                import re as _re
                from rich.text import Text as RichText
                from rich.panel import Panel
                # 去除 Rich 标记，展示为纯文本
                plain = _re.sub(r'\[/?\w+(?: [^\]]+)?\]', '', summary)
                panel = Panel(
                    RichText(f"📦 上下文压缩摘要\n\n{plain}", style="dim white"),
                    title="压缩历史",
                    border_style="dim yellow",
                )
                self._compression_widget = Static(panel)
                # 挂载到聊天视图末尾
                chat_view.mount(self._compression_widget)
                chat_view.scroll_end(animate=False)
            else:
                # 已创建则直接显示
                self._compression_widget.display = True

            self._compression_visible = True

        def _set_busy(self, value: bool) -> None:
            self._is_busy = value
            input_widget = self.query_one("#chat-input", Input)
            input_widget.disabled = False
            input_widget.placeholder = (
                "任务执行中，输入 /stop 或按 Ctrl+C 中断当前任务"
                if value
                else "输入消息，或输入 /help 查看命令"
            )
            self._update_command_hint(input_widget.value)

        def _refresh_composer_title(self) -> None:
            """刷新标题栏（会话切换后更新 session ID 显示）。"""
            try:
                self.query_one("#composer-title", Static).update(
                    self._build_composer_title()
                )
            except Exception:
                pass

        def _show_context_menu(
            self, widget: ChatMessage | RenderableBlock,
        ) -> None:
            """弹出右键上下文菜单——支持 ChatMessage 与 RenderableBlock。"""
            selected = self.screen.get_selected_text() or ""
            self.push_screen(ContextMenuScreen(widget, selected))

        def action_quit_app(self) -> None:
            """退出整个程序（Ctrl+Q），保存当前会话。"""
            from .session_runtime import persist_runtime_session
            try:
                persist_runtime_session(self.runner, self.runtime_context)
            except Exception:
                pass
            self.exit()

        def action_cancel_task(self) -> None:
            """取消当前运行中的任务（Ctrl+C）。"""
            if self._is_busy:
                from .app import request_running_task_stop
                request_running_task_stop(self.runtime_context, reason="用户按 Ctrl+C")
                self._add_message("system", "已按 Ctrl+C，正在终止当前任务...")
            else:
                self._add_message("system", "当前没有正在执行的任务。")


def launch_textual_chat(
    runner: Any,
    runtime_context: dict[str, object],
    *,
    show_banner: bool = True,
) -> None:
    """在真实终端中启动 Textual 聊天界面。"""

    if TEXTUAL_IMPORT_ERROR is not None:  # pragma: no cover - 降级分支
        raise ModuleNotFoundError(str(TEXTUAL_IMPORT_ERROR)) from TEXTUAL_IMPORT_ERROR

    app = CyberAgentTUI(
        runner,
        runtime_context,
        show_banner=show_banner,
    )
    try:
        app.run()
    finally:
        # 异常退出（终端关闭、SIGINT 等）时兜底保存当前会话
        from .session_runtime import persist_runtime_session
        try:
            persist_runtime_session(runner, runtime_context)
        except Exception:
            pass
