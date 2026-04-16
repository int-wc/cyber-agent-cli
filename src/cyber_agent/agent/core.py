# src/cyber_agent/agent/core.py
import operator
from typing import List, Annotated, TypedDict
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, AIMessage

# 1. 定义状态 (State)：用于在图的节点间传递数据
class AgentState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], operator.add]

# 2. 定义节点 (Nodes)：
#    - `agent`节点：调用LLM，让它决定下一步做什么（调用工具或直接回答）
def agent(state: AgentState, llm, tools):
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

#    - `tools`节点：执行LLM选择的具体工具
# LangGraph提供了一个预制的`ToolNode`，可以自动处理工具调用

# 3. 构建图 (Graph)
def create_agent_graph(llm, tools):
    # 初始化图
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("agent", lambda state: agent(state, llm, tools))
    workflow.add_node("tools", ToolNode(tools))

    # 添加边 (Edges)：定义节点间的流转逻辑
    workflow.set_entry_point("agent")  # 从`agent`节点开始
    # 添加条件边：`agent`节点执行后，是去`tools`节点还是直接结束(END)
    workflow.add_conditional_edges("agent", tools_condition)
    # `tools`节点执行完后，再回到`agent`节点，形成循环
    workflow.add_edge("tools", "agent")

    # 编译图，生成可执行的App
    return workflow.compile()