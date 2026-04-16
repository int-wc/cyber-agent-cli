# src/cyber_agent/cli/tui.py
from textual.app import App, ComposeResult
from textual.containers import Container, ScrollableContainer
from textual.widgets import Header, Footer, Input, Static
from textual.reactive import reactive
from ..agent.runner import AgentRunner
from ..tools.security import scan_port


class ChatMessage(Static):
    """用于显示聊天消息的组件"""
    pass


class CyberAgentTUI(App):
    """网络安全智能体 TUI"""
    CSS = """
    Screen {
        background: $surface;
    }
    #chat-view {
        border: solid $primary;
        height: 1fr;
        margin: 1;
        padding: 1;
    }
    #input-area {
        dock: bottom;
        margin: 1;
    }
    ChatMessage {
        padding: 1;
    }
    """

    def __init__(self):
        super().__init__()
        self.runner = AgentRunner([scan_port])

    def compose(self) -> ComposeResult:
        yield Header()
        yield ScrollableContainer(Static(id="chat-view"))
        yield Input(placeholder="输入指令...", id="input-area")
        yield Footer()

    async def on_input_submitted(self, event: Input.Submitted):
        user_input = event.value
        event.input.value = ""  # 清空输入框
        chat_view = self.query_one("#chat-view", Static)

        # 显示用户消息
        await chat_view.mount(ChatMessage(f"🧑 You: {user_input}"))

        # 运行智能体（这里需要异步处理）
        response = self.runner.run(user_input)  # 需要改成异步

        # 显示助手回复
        await chat_view.mount(ChatMessage(f"🤖 Agent: {response}"))
        chat_view.scroll_end(animate=False)


def main():
    app = CyberAgentTUI()
    app.run()