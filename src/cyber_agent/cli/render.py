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

# DeepSeek 定价（人民币 元/百万 token，缓存未命中）
# 参考: https://api-docs.deepseek.com/zh-cn/quick_start/pricing
_DEEPSEEK_PRICING: dict[str, dict[str, float]] = {
    "deepseek-v4-pro":   {"input": 1.0, "output": 2.0},
    "deepseek-v4-flash": {"input": 0.5, "output": 1.0},
    # 兼容旧模型名
    "deepseek-chat":     {"input": 1.0, "output": 2.0},
    "deepseek-reasoner": {"input": 1.0, "output": 2.0},
}
_DEFAULT_PRICING = {"input": 1.0, "output": 2.0}


def _estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model_name: str = "",
) -> float:
    """根据 DeepSeek 定价估算本轮花费（人民币 元）。"""
    pricing = _DEFAULT_PRICING
    for key, rates in _DEEPSEEK_PRICING.items():
        if key in model_name.lower():
            pricing = rates
            break
    return (input_tokens / 1_000_000) * pricing["input"] + (
        output_tokens / 1_000_000
    ) * pricing["output"]


class CliRenderer:
    """负责将 CLI 运行信息渲染成更适合终端展示的富文本输出。"""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self._streaming_response_started = False
        self._streaming_prefix_printed = False
        self._streamed_response_chunks: list[str] = []
        self._reasoning_parts: list[str] = []
        self._reasoning_printed = False
        # 累计 token 统计（以 API 返回精确值为准）
        self._cumulative_input_tokens = 0
        self._cumulative_output_tokens = 0
        self._cumulative_cost = 0.0
        # 上轮开始前的基线（用于避免 live 估算与 API 精确值重复计数）
        self._turn_baseline_input = 0
        self._turn_baseline_output = 0
        self._turn_baseline_set = False  # 每轮只设一次基线
        self._model_name = ""
        # 实时 token 计数器
        self._live: Live | None = None

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

    def add_token_usage(self, input_tokens: int, output_tokens: int) -> None:
        """从外部直接累加 token 使用量（用于多 Agent 编排器等非标准路径）。"""
        if input_tokens <= 0 and output_tokens <= 0:
            return
        self._cumulative_input_tokens += input_tokens
        self._cumulative_output_tokens += output_tokens
        self._cumulative_cost = _estimate_cost(
            self._cumulative_input_tokens,
            self._cumulative_output_tokens,
            self._model_name,
        )

    def print_token_usage(self, usage: dict[str, int]) -> None:
        """打印本轮及累计 token 消耗统计（含花费估算）。

        API 返回的 usage 已包含所有工具调用轮次的输入/输出。
        本轮开始前 live 计数器可能已估算部分 token，此处重置到基线再累加 API 精确值。
        """
        api_input = usage.get("input_tokens", 0)
        api_output = usage.get("output_tokens", 0)
        api_total = usage.get("total_tokens", api_input + api_output)

        # 以 API 精确值为准：有数据的维度用 API，缺失的保留 live 估算
        if api_input > 0:
            self._cumulative_input_tokens = (
                self._turn_baseline_input + api_input
            )
        if api_output > 0:
            self._cumulative_output_tokens = (
                self._turn_baseline_output + api_output
            )
        # 重置基线标记，下轮重新记录
        self._turn_baseline_set = False

        # 重新计算累计花费
        self._cumulative_cost = _estimate_cost(
            self._cumulative_input_tokens,
            self._cumulative_output_tokens,
            self._model_name,
        )

        # 本轮花费
        round_input = self._cumulative_input_tokens - self._turn_baseline_input
        round_output = self._cumulative_output_tokens - self._turn_baseline_output
        round_cost = _estimate_cost(round_input, round_output, self._model_name)

        self.console.print(
            f"  [dim]本轮 ↑{round_input} ↓{round_output}"
            f" ∑{round_input + round_output}"
            f"  │  ¥{round_cost:.4f}"
            f"  │  累计 ↑{self._cumulative_input_tokens}"
            f" ↓{self._cumulative_output_tokens}"
            f" ∑{self._cumulative_input_tokens + self._cumulative_output_tokens}"
            f"  │  ¥{self._cumulative_cost:.4f}[/dim]"
        )

    def _build_token_status_line(self) -> Text:
        """构建实时 token 状态行。"""
        # 估算当前输出 token 数（中文 ~1.5 字/token, 英文 ~4 字/token）
        chars = sum(len(c) for c in self._streamed_response_chunks)
        est_output = max(1, chars // 2)
        total_in = self._cumulative_input_tokens
        total_out = self._cumulative_output_tokens + est_output
        total = total_in + total_out
        cost = _estimate_cost(total_in, total_out, self._model_name)
        return Text(
            f" 累计 ↑{total_in} ↓{total_out} Σ{total} │ ¥{cost:.4f}",
            style="dim",
        )

    def begin_response_stream(self) -> None:
        """开始一轮模型 token 流，启动实时计数器。"""
        self.ensure_response_stream_closed()
        self._streaming_response_started = True
        self._streaming_prefix_printed = False
        self._streamed_response_chunks = []
        self._reasoning_parts = []
        self._reasoning_printed = False
        # 每轮只设一次基线（多次 _stream_model_response 调用共享同一基线）
        if not self._turn_baseline_set:
            self._turn_baseline_input = self._cumulative_input_tokens
            self._turn_baseline_output = self._cumulative_output_tokens
            self._turn_baseline_set = True
        # 启动实时 token 计数
        self._live = Live(
            self._build_token_status_line(),
            console=self.console,
            auto_refresh=False,
            transient=True,
        )
        self._live.start()

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
        """累积响应 token，刷新实时计数器。"""
        if not self._streaming_response_started:
            self.begin_response_stream()
        self._flush_reasoning_if_needed()
        if not self._streaming_prefix_printed:
            self.console.print(
                f"[bold {ASSISTANT_BORDER_COLOR}]智能体输出[/bold {ASSISTANT_BORDER_COLOR}]"
                " › [dim]接收中...[/dim]",
            )
            self._streaming_prefix_printed = True
        self._streamed_response_chunks.append(token_text)
        # 刷新实时 token 计数
        if self._live is not None:
            self._live.update(self._build_token_status_line(), refresh=True)

    def _stop_live(self) -> None:
        """停止实时 token 计数器。"""
        if self._live is not None:
            self._live.stop()
            self._live = None

    def ensure_response_stream_closed(self) -> None:
        """在打印其他区块前确保流式输出已换行结束。"""
        if self._streaming_response_started:
            self._stop_live()
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

    def add_tool_result_tokens(self, content: str) -> None:
        """将工具结果内容估算为输入 token 并计入累计（供 live 状态栏实时更新）。"""
        est_input = max(1, len(content) // 3)
        self._cumulative_input_tokens += est_input
        if self._live is not None:
            self._live.update(self._build_token_status_line(), refresh=True)

    def print_tool_result(self, content: str) -> None:
        """打印工具执行结果，使用 Markdown 渲染。"""
        self.ensure_response_stream_closed()
        # 工具结果计入输入 token 估算
        self.add_tool_result_tokens(content)
        # 截取前 4000 字符以防工具结果过长导致渲染卡顿
        truncated = content[:4000]
        if len(content) > 4000:
            truncated += f"\n\n*... 已截断，共 {len(content)} 字符*"
        if _MARKDOWN_AVAILABLE:
            try:
                md = Markdown(truncated, code_theme="monokai", justify="left")
                self.console.print(
                    Panel(md, title="工具结果", border_style="green")
                )
                return
            except Exception:
                pass
        self.print_renderable(build_tool_result_panel(truncated))

    def end_response_stream(self, content: str, has_tool_calls: bool) -> None:
        """结束一轮流式输出，以 Markdown 渲染最终回复。"""
        if self._streaming_response_started:
            if self._streaming_prefix_printed:
                # 替换"接收中..."为换行，然后渲染完整 Markdown
                self.console.print("")
            if content and not has_tool_calls:
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

    def print_status_line(self, recent_inputs: list[str] | None = None) -> None:
        """打印持续显示的 token/花费状态行 + 最近输入历史。"""
        t = self._cumulative_input_tokens
        o = self._cumulative_output_tokens
        self.console.print(
            f"  [dim]累计 ↑{t} ↓{o} Σ{t+o} │ ¥{self._cumulative_cost:.4f}[/dim]"
        )
        if recent_inputs:
            # 显示最近 5 条输入（截断到 40 字）
            previews = []
            for inp in recent_inputs[-5:]:
                short = inp[:40].replace("\n", " ")
                if len(inp) > 40:
                    short += "…"
                previews.append(short)
            self.console.print(
                f"  [dim]最近: {' → '.join(previews)}[/dim]"
            )

    def print_error(self, content: str) -> None:
        """打印错误消息，支持 Markdown 渲染。"""
        self.ensure_response_stream_closed()
        if _MARKDOWN_AVAILABLE and content.strip():
            try:
                md = Markdown(content, code_theme="monokai", justify="left")
                self.console.print(
                    Panel(md, title="错误", border_style="red")
                )
                return
            except Exception:
                pass
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
        self,
        role: str,
        success: bool,
        elapsed_ms: float,
        output_summary: str = "",
        output_length: int = 0,
    ) -> None:
        """单个子任务完成，显示耗时和输出摘要。"""
        icon = "✓" if success else "✗"
        style = "green" if success else "red"
        self.console.print(
            f"  [{style}]{icon} {role}[/] "
            f"[dim]({elapsed_ms:.0f}ms, {output_length} 字符)[/]"
        )
        if output_summary.strip():
            summary_line = output_summary.strip()[:200].replace("\n", " ")
            self.console.print(f"    [dim]{summary_line}[/]")

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
