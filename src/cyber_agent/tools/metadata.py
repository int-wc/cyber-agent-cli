from langchain_core.tools import BaseTool


def attach_tool_risk(tool: BaseTool, risk: str) -> BaseTool:
    """为工具补充风险级别元数据，兼容当前 LangChain 工具模型。"""

    original_metadata = tool.metadata or {}
    tool.metadata = {
        **original_metadata,
        "risk": risk,
    }
    return tool
