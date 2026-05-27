import json
import time
from pathlib import Path

from rich import box
from rich.console import Console, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

try:
    from rich.markdown import Markdown
    _MARKDOWN_AVAILABLE = True
except ImportError:
    Markdown = None  # type: ignore[assignment,misc]
    _MARKDOWN_AVAILABLE = False

from ..agent.approval import ApprovalPolicy, get_approval_policy_label
from ..agent.mode import AgentMode, get_mode_description, get_mode_label
from .branding import (
    STARTUP_ANIMATION_DELAY_SECONDS,
    STARTUP_ANIMATION_FRAMES,
    build_startup_frame,
)
from .interactive import (
    BUILTIN_COMMAND_SPECS,
    build_session_overview,
    get_banner_command_summary,
)
from .theme import (
    ASSISTANT_BORDER_COLOR,
    ASSISTANT_TEXT_COLOR,
    COMMAND_DESC_STYLE,
    COMMAND_NAME_STYLE,
    KEYCAP_STYLE,
    ROLE_STYLES,
    SYSTEM_LABEL_STYLE,
    SYSTEM_VALUE_STYLE,
    SYSTEM_VALUE_STYLES,
)
from .render_panels import (
    append_system_kv_line,
    build_allowed_roots_panel,
    build_approval_policy_notice_panel,
    build_approval_request_panel,
    build_approval_result_panel,
    build_banner_body,
    build_banner_panel,
    build_chat_message_panel,
    build_help_panel,
    build_mode_notice_panel,
    build_status_panel,
    build_tool_call_panel,
    build_tool_result_panel,
    build_tools_panel,
)

class CliRenderer:
    """负责将 CLI 运行信息渲染成更适合终端展示的富文本输出。"""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self._streaming_response_started = False
        self._streaming_prefix_printed = False
        self._streamed_response_chunks: list[str] = []
        self._reasoning_parts: list[str] = []
        self._reasoning_printed = False

    def print_startup_splash(self) -> None:
        """打印启动页；真实终端播放动画，其余场景回退为静态区块。"""

        self.ensure_response_stream_closed()
        if not self.console.is_terminal or self.console.is_dumb_terminal:
            self.console.print(build_startup_frame(STARTUP_ANIMATION_FRAMES - 1))
            self.console.print()
            return

        with Live(
            build_startup_frame(0),
            console=self.console,
            refresh_per_second=max(24, STARTUP_ANIMATION_FRAMES),
            transient=False,
        ) as live:
            for frame_index in range(1, STARTUP_ANIMATION_FRAMES):
                time.sleep(STARTUP_ANIMATION_DELAY_SECONDS)
                live.update(build_startup_frame(frame_index))
        self.console.print()

    def clear_screen(self) -> None:
        """在真实终端启动前清屏，避免旧内容干扰启动页展示。"""
        self.ensure_response_stream_closed()
        if self.console.is_terminal and not self.console.is_dumb_terminal:
            self.console.clear(home=True)

    def print_turn_start(self) -> None:
        """打印一条分隔线，用于区分每轮会话。"""
        self.ensure_response_stream_closed()
        self.console.print(Rule(style="grey50"))

    def print_token_usage(self, usage: dict[str, int]) -> None:
        """打印本轮 token 消耗统计。"""
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        total_tokens = usage.get("total_tokens", input_tokens + output_tokens)
        self.console.print(
            f"  [dim]↑ {input_tokens} tokens  ↓ {output_tokens} tokens  ∑ {total_tokens} tokens[/dim]"
        )

    def begin_response_stream(self) -> None:
        """开始一轮模型 token 流。"""
        self.ensure_response_stream_closed()
        self._streaming_response_started = True
        self._streaming_prefix_printed = False
        self._streamed_response_chunks = []
        self._reasoning_parts = []
        self._reasoning_printed = False

    def append_reasoning_token(self, token_text: str) -> None:
        """追加思考过程文本，在首次响应正文前展示为折叠思考块。"""
        self._reasoning_parts.append(token_text)
        if not self._reasoning_printed:
            self.console.print(
                f"[dim italic {ASSISTANT_BORDER_COLOR}]思考过程[/dim italic {ASSISTANT_BORDER_COLOR}]"
                " › ",
                end="",
            )
            self._reasoning_printed = True
        self.console.print(
            Text(token_text, style=f"dim italic {ASSISTANT_TEXT_COLOR}"),
            end="",
            soft_wrap=True,
            highlight=False,
        )

    def _flush_reasoning_if_needed(self) -> None:
        """思考过程结束后换行，与正文分隔。"""
        if self._reasoning_printed:
            self.console.print("")
            self._reasoning_printed = False
            self._reasoning_parts = []

    def append_response_token(self, token_text: str) -> None:
        """将 token 追加到终端。"""
        if not self._streaming_response_started:
            self.begin_response_stream()
        self._flush_reasoning_if_needed()
        if not self._streaming_prefix_printed:
            self.console.print(
                f"[bold {ASSISTANT_BORDER_COLOR}]智能体输出[/bold {ASSISTANT_BORDER_COLOR}]"
                " › ",
                end="",
            )
            self._streaming_prefix_printed = True
        self._streamed_response_chunks.append(token_text)
        self.console.print(
            Text(token_text, style=ASSISTANT_TEXT_COLOR),
            end="",
            soft_wrap=True,
            highlight=False,
        )

    def ensure_response_stream_closed(self) -> None:
        """在打印其他区块前确保流式输出已换行结束。"""
        if self._streaming_response_started:
            if self._streaming_prefix_printed:
                self.console.print("")
            self._streaming_response_started = False
            self._streaming_prefix_printed = False
            self._streamed_response_chunks = []

    def print_renderable(self, renderable: RenderableType) -> None:
        """统一输出 Rich 渲染对象，便于 TUI 直接复用同一份面板。"""
        self.ensure_response_stream_closed()
        self.console.print(renderable)

    def print_banner(
        self,
        *,
        mode: AgentMode,
        service: str,
        model: str,
        cwd: Path,
        approval_policy: ApprovalPolicy,
    ) -> None:
        """打印共享欢迎面板，确保 CLI 与 TUI 使用同一份欢迎区样式。"""
        self.print_renderable(
            build_banner_panel(
                mode=mode,
                service=service,
                model=model,
                cwd=cwd,
                approval_policy=approval_policy,
            )
        )

    def print_help(self) -> None:
        """打印统一的内建命令帮助面板。"""
        self.print_renderable(build_help_panel())

    def print_allowed_roots(self, allowed_roots: list[str]) -> None:
        """打印统一的允许访问目录面板。"""
        self.print_renderable(build_allowed_roots_panel(allowed_roots))

    def print_tools(self, descriptions: list[str]) -> None:
        """打印统一的工具列表面板。"""
        self.print_renderable(build_tools_panel(descriptions))

    def print_status(
        self,
        rows: list[tuple[str, str]],
        *,
        title: str = "当前状态",
    ) -> None:
        """打印统一的状态概览面板。"""
        self.print_renderable(build_status_panel(rows, title=title))

    def print_mode_notice(self, mode: AgentMode, switched: bool = True) -> None:
        """打印模式查看或切换结果。"""
        self.print_renderable(build_mode_notice_panel(mode, switched=switched))

    def print_approval_policy_notice(
        self,
        policy: ApprovalPolicy,
        switched: bool = True,
    ) -> None:
        """打印审批策略查看或切换结果。"""
        self.print_renderable(
            build_approval_policy_notice_panel(policy, switched=switched)
        )

    def print_chat_message(self, role: str, content: str | Text) -> None:
        """打印统一聊天面板，供 CLI 与 TUI 共享。"""
        self.print_renderable(build_chat_message_panel(role, content))

    def print_user_message(self, content: str) -> None:
        """打印用户输入，便于与 TUI 的消息气泡对齐。"""
        self.print_chat_message("user", content)

    def print_tool_call(self, tool_calls: list[dict]) -> None:
        """打印工具调用事件面板。"""
        self.print_renderable(build_tool_call_panel(tool_calls))

    def print_tool_result(self, content: str) -> None:
        """打印工具执行结果面板。"""
        self.print_renderable(build_tool_result_panel(content))

    def end_response_stream(self, content: str, has_tool_calls: bool) -> None:
        """结束一轮流式输出，对最终回复使用 Markdown 渲染。"""
        if self._streaming_response_started:
            if self._streaming_prefix_printed:
                if not self._streamed_response_chunks and content:
                    self.console.print(content, end="", soft_wrap=True, highlight=False)
                self.console.print("")
            elif content and not has_tool_calls:
                self.print_markdown(content)
        elif content and not has_tool_calls:
            self.print_markdown(content)

        self._streaming_response_started = False
        self._streaming_prefix_printed = False
        self._streamed_response_chunks = []

    def print_approval_request(self, payload: dict) -> None:
        """打印审批请求面板。"""
        self.print_renderable(build_approval_request_panel(payload))

    def print_approval_result(self, payload: dict) -> None:
        """打印审批结果面板。"""
        self.print_renderable(build_approval_result_panel(payload))

    def print_markdown(self, content: str) -> None:
        """以 Markdown 格式渲染文本内容，不可用时回退到纯文本面板。"""
        self.ensure_response_stream_closed()
        if not content.strip():
            return
        if not _MARKDOWN_AVAILABLE:
            self.print_chat_message("assistant", content)
            return
        try:
            md = Markdown(
                content,
                code_theme="monokai",
                inline_code_theme="monokai",
                justify="left",
            )
            self.console.print(md)
        except Exception:
            self.print_chat_message("assistant", content)

    def print_info(self, content: str) -> None:
        """打印普通提示文本；捕获模式下也会按原样保留。"""
        self.print_renderable(content)

    def print_error(self, content: str) -> None:
        """打印错误消息面板。"""
        self.print_chat_message("error", content)

    # ── 多 Agent 编排器进度渲染 ──

    def print_orchestration_planning(self, user_input: str) -> None:
        """任务规划阶段 —— 决策者分析并分解任务。"""
        self.console.print()
        self.console.print(
            Panel(
                f"[bold cyan]🎯 决策者正在分析任务...[/]\n"
                f"[dim]{user_input[:120]}[/]",
                title="多 Agent 协作 · 阶段 1/6",
                border_style="cyan",
            )
        )

    def print_orchestration_plan_done(
        self, subtask_count: int, reasoning: str
    ) -> None:
        """任务规划完成 —— 显示分解结果。"""
        self.console.print(
            Panel(
                f"[green]✓ 任务已分解为 {subtask_count} 个子任务[/]\n"
                f"[dim]{reasoning[:200]}[/]",
                border_style="green",
            )
        )

    def print_orchestration_executing(self, subtask_count: int) -> None:
        """并发执行阶段 —— 子任务分发给角色 Agent。"""
        role_icons = {
            "决策者": "🎯", "审计者": "🔍", "阅读者": "📖",
            "分析者": "📊", "执行者": "⚡", "构建者": "🔧",
            "反思者": "🪞", "扩散者": "🌐", "迁跃者": "🚀",
        }
        self.console.print()
        self.console.print(
            Panel(
                f"[bold yellow]⚡ 正在并发执行 {subtask_count} 个子任务...[/]",
                title="多 Agent 协作 · 阶段 2/6",
                border_style="yellow",
            )
        )

    def print_subtask_complete(
        self, role: str, success: bool, elapsed_ms: float
    ) -> None:
        """单个子任务完成。"""
        icon = "✓" if success else "✗"
        style = "green" if success else "red"
        self.console.print(
            f"  [{style}]{icon} {role}[/] "
            f"[dim]({elapsed_ms:.0f}ms)[/]"
        )

    def print_orchestration_checking(self, result_count: int) -> None:
        """审计验证阶段。"""
        self.console.print()
        self.console.print(
            Panel(
                f"[bold magenta]🔍 审计者正在验证 {result_count} 个结果...[/]",
                title="多 Agent 协作 · 阶段 3/6",
                border_style="magenta",
            )
        )

    def print_orchestration_reflecting(self, failed_count: int) -> None:
        """反思评估阶段。"""
        self.console.print()
        self.console.print(
            Panel(
                f"[bold blue]🪞 反思者评估中（{failed_count} 个任务需关注）...[/]",
                title="多 Agent 协作 · 阶段 4/6",
                border_style="blue",
            )
        )

    def print_orchestration_synthesize(self) -> None:
        """综合产出阶段。"""
        self.console.print()
        self.console.print(
            Panel(
                "[bold green]📝 决策者正在综合各角色输出...[/]",
                title="多 Agent 协作 · 阶段 5/6",
                border_style="green",
            )
        )

    def print_orchestration_done(self, total_results: int) -> None:
        """多 Agent 协作完成。"""
        self.console.print(
            Panel(
                f"[bold green]✅ 多 Agent 协作完成[/] "
                f"[dim](共 {total_results} 个角色结果)[/]",
                border_style="green",
            )
        )
