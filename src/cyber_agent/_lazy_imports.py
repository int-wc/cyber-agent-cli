"""共享的按需导入工具，避免多处重复定义相同的懒加载逻辑。"""
from __future__ import annotations

from typing import Any

ChatOpenAI: Any | None = None
LANGCHAIN_OPENAI_IMPORT_ERROR: ModuleNotFoundError | None = None


def load_chat_openai() -> Any:
    """按需导入 ChatOpenAI，避免 CLI 启动阶段加载 OpenAI 全量依赖树。"""
    global ChatOpenAI, LANGCHAIN_OPENAI_IMPORT_ERROR

    if ChatOpenAI is not None:
        return ChatOpenAI
    if LANGCHAIN_OPENAI_IMPORT_ERROR is not None:
        raise LANGCHAIN_OPENAI_IMPORT_ERROR

    try:
        from langchain_openai import ChatOpenAI as LoadedChatOpenAI
    except ModuleNotFoundError as exc:  # pragma: no cover - 是否安装依赖由运行环境决定
        LANGCHAIN_OPENAI_IMPORT_ERROR = exc
        raise

    ChatOpenAI = LoadedChatOpenAI
    LANGCHAIN_OPENAI_IMPORT_ERROR = None
    return ChatOpenAI
