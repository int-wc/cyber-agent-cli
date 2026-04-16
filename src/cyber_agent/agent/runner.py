# src/cyber_agent/agent/runner.py
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from .core import AgentState, create_agent_graph
from ..config import settings


class AgentRunner:
    def __init__(self, tools):
        self.service = settings.get_service()
        self.llm = ChatOpenAI(**settings.get_chat_openai_kwargs(self.service))
        self.tools = tools
        self.app = create_agent_graph(self.llm, self.tools)

    def run(self, user_input: str):
        """运行一次智能体对话，并输出执行过程。"""
        print("开始处理用户输入...")

        # 将用户输入封装为 LangChain 消息，作为图执行的初始状态。
        inputs: AgentState = {"messages": [HumanMessage(content=user_input)]}

        # 先流式输出中间步骤，便于在命令行中观察模型决策和工具调用。
        for output in self.app.stream(inputs):
            for key, value in output.items():
                # `agent` 节点负责产出模型回复，也可能包含工具调用请求。
                if key == "agent":
                    response = value["messages"][-1]
                    if hasattr(response, "tool_calls") and response.tool_calls:
                        print(f"智能体决定调用工具: {response.tool_calls}")
                    else:
                        print(f"智能体回复: {response.content}")
                elif key == "tools":
                    print("工具执行完成。")

        # 再执行一次完整调用，拿到最终状态中的最后一条消息作为返回值。
        final_state = self.app.invoke(inputs)
        final_response = final_state["messages"][-1].content
        print(f"最终回复: {final_response}")
        return final_response
