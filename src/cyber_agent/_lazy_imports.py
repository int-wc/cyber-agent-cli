"""共享的按需导入工具，避免多处重复定义相同的懒加载逻辑。"""
from __future__ import annotations

from typing import Any

ChatOpenAI: Any | None = None
LANGCHAIN_OPENAI_IMPORT_ERROR: ModuleNotFoundError | None = None

ChatAnthropic: Any | None = None
LANGCHAIN_ANTHROPIC_IMPORT_ERROR: ModuleNotFoundError | None = None


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


def load_chat_anthropic() -> Any:
    """按需导入 ChatAnthropic，用于 Anthropic 兼容 API（DeepSeek/MiMo 等）。"""
    global ChatAnthropic, LANGCHAIN_ANTHROPIC_IMPORT_ERROR

    if ChatAnthropic is not None:
        return ChatAnthropic
    if LANGCHAIN_ANTHROPIC_IMPORT_ERROR is not None:
        raise LANGCHAIN_ANTHROPIC_IMPORT_ERROR

    try:
        from langchain_anthropic import ChatAnthropic as LoadedChatAnthropic
    except ModuleNotFoundError as exc:
        LANGCHAIN_ANTHROPIC_IMPORT_ERROR = exc
        raise

    ChatAnthropic = LoadedChatAnthropic
    LANGCHAIN_ANTHROPIC_IMPORT_ERROR = None
    return ChatAnthropic


def is_anthropic_api(base_url: str | None) -> bool:
    """判断 base_url 是否指向 Anthropic 兼容 API。"""
    if not base_url:
        return False
    return "/anthropic" in base_url or base_url.endswith("/anthropic")


def load_llm_for_api(
    base_url: str | None,
) -> tuple[Any, bool]:
    """根据 API 格式自动选择正确的 LangChain 客户端。
    返回 (LLMClass, is_anthropic)。"""
    if is_anthropic_api(base_url):
        return load_chat_anthropic(), True
    return load_chat_openai(), False
