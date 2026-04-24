from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

MODEL_GATEWAY_BASE_URL = "http://localhost:8317/"
DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-5.4",
    "deepseek": "deepseek-v4-pro",
}


class Settings(BaseSettings):
    openai_api_key: str = Field(
        default="sk-default",
        validation_alias="OPENAI_API_KEY",
    )
    openai_model: str = Field(
        default="gpt-5.4",
        validation_alias="OPENAI_MODEL",
    )
    openai_base_url: str | None = Field(
        default=None,
        validation_alias="OPENAI_BASE_URL",
    )
    deepseek_api_key: str | None = Field(
        default=None,
        validation_alias="DEEPSEEK_API_KEY",
    )
    deepseek_model: str = Field(
        default=DEFAULT_MODELS["deepseek"],
        validation_alias="DEEPSEEK_MODEL",
    )
    deepseek_base_url: str | None = Field(
        default=None,
        validation_alias="DEEPSEEK_BASE_URL",
    )
    deepseek_thinking_mode: str = Field(
        default="disabled",
        validation_alias="DEEPSEEK_THINKING_MODE",
    )
    service_name: str = Field(
        default="openai",
        validation_alias="SERVICE_NAME",
    )
    search_endpoint: str = Field(
        default="https://html.duckduckgo.com/html/",
        validation_alias="SEARCH_ENDPOINT",
    )
    search_timeout_seconds: float = Field(
        default=10.0,
        validation_alias="SEARCH_TIMEOUT_SECONDS",
    )
    search_result_limit: int = Field(
        default=5,
        validation_alias="SEARCH_RESULT_LIMIT",
    )
    search_show_browser: bool = Field(
        default=True,
        validation_alias="SEARCH_SHOW_BROWSER",
    )
    max_context_chars: int = Field(
        default=14000,
        validation_alias="MAX_CONTEXT_CHARS",
    )
    context_keep_recent_messages: int = Field(
        default=8,
        validation_alias="CONTEXT_KEEP_RECENT_MESSAGES",
    )
    context_summary_max_chars: int = Field(
        default=2000,
        validation_alias="CONTEXT_SUMMARY_MAX_CHARS",
    )
    capability_audit_min_score: int = Field(
        default=75,
        validation_alias="CAPABILITY_AUDIT_MIN_SCORE",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    def normalize_service_name(self, service_name: str | None = None) -> str:
        """规范化服务商名称，避免展示和运行时出现大小写不一致。"""
        normalized_service_name = (service_name or self.service_name).strip().lower()
        return normalized_service_name or "openai"

    def get_service(self) -> str:
        """返回当前默认服务商名称。"""
        return self.normalize_service_name()

    def get_model_name(
        self,
        model_name: str | None = None,
        service_name: str | None = None,
    ) -> str:
        """返回当前默认模型名称。"""
        if model_name is not None:
            normalized_model_name = model_name.strip()
        else:
            normalized_service_name = self.normalize_service_name(service_name)
            normalized_model_name = (
                self.deepseek_model
                if normalized_service_name == "deepseek"
                else self.openai_model
            ).strip()
        if not normalized_model_name:
            raise ValueError("模型名称不能为空。")
        return normalized_model_name

    def get_api_key(
        self,
        service_name: str | None = None,
        api_key: str | None = None,
    ) -> str:
        """按服务商解析 API Key，兼容旧版仅配置 OPENAI_API_KEY 的写法。"""
        if api_key is not None:
            resolved_api_key = api_key.strip()
        elif self.normalize_service_name(service_name) == "deepseek":
            resolved_api_key = (self.deepseek_api_key or self.openai_api_key).strip()
        else:
            resolved_api_key = self.openai_api_key.strip()
        if not resolved_api_key:
            raise ValueError("模型 API Key 不能为空。")
        return resolved_api_key

    def get_default_base_url_for_service(self, service_name: str | None = None) -> str | None:
        """返回统一模型网关基址，切换服务商时不改变请求入口。"""
        _ = service_name
        return MODEL_GATEWAY_BASE_URL

    def get_deepseek_thinking_mode(self) -> str:
        """返回 DeepSeek thinking 模式开关，默认关闭以兼容工具调用长链路。"""
        normalized_mode = self.deepseek_thinking_mode.strip().lower()
        if normalized_mode in {"", "disabled", "disable", "off", "false", "0", "no"}:
            return "disabled"
        if normalized_mode in {"enabled", "enable", "on", "true", "1", "yes"}:
            return "enabled"
        raise ValueError("DEEPSEEK_THINKING_MODE 仅支持 enabled 或 disabled。")

    def is_deepseek_thinking_enabled(self) -> bool:
        """判断 DeepSeek 是否启用 thinking 模式。"""
        return self.get_deepseek_thinking_mode() == "enabled"

    def resolve_base_url(
        self,
        service_name: str | None = None,
        base_url: str | None = None,
    ) -> str | None:
        """解析运行时应使用的模型服务基址。"""
        _ = base_url
        return self.get_default_base_url_for_service(service_name)

    def get_chat_openai_kwargs(
        self,
        service_name: str | None = None,
        *,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> dict:
        """构建 OpenAI 兼容接口的模型初始化参数。"""
        resolved_service_name = self.normalize_service_name(service_name)
        resolved_model_name = self.get_model_name(
            model_name,
            service_name=resolved_service_name,
        )
        resolved_api_key = self.get_api_key(
            resolved_service_name,
            api_key=api_key,
        )
        resolved_base_url = self.resolve_base_url(
            resolved_service_name,
            base_url=base_url,
        )

        extra_body: dict[str, object] = {
            "provider": resolved_service_name,
        }
        if resolved_service_name == "deepseek":
            extra_body["thinking"] = {"type": self.get_deepseek_thinking_mode()}

        kwargs = {
            "model": resolved_model_name,
            "api_key": resolved_api_key,
            "base_url": resolved_base_url,
            "temperature": 0.7,
            "max_tokens": 1024,
            "extra_body": extra_body,
        }
        return {key: value for key, value in kwargs.items() if value is not None}


settings = Settings()
