# src/cyber_agent/cli/app.py
import typer
from ..agent.runner import AgentRunner
from ..tools.security import scan_port  # 导入你的工具

app = typer.Typer()


@app.command()
def chat():
    """
    进入交互式聊天模式
    """
    # 1. 准备工具
    tools = [scan_port]  # 将你的所有工具添加到这里

    # 2. 初始化智能体运行器
    runner = AgentRunner(tools)

    # 3. 进入对话循环
    typer.echo("🛡️  网络安全智能体已启动。输入 'quit' 退出。")
    while True:
        user_input = typer.prompt("You")
        if user_input.lower() in ["quit", "exit", "q"]:
            typer.echo("👋 再见！")
            break
        runner.run(user_input)