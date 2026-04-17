from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_BASE_URLS: dict[str, str] = {
    "deepseek": "https://api.deepseek.com/v1",
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

    def get_model_name(self, model_name: str | None = None) -> str:
        """返回当前默认模型名称。"""
        normalized_model_name = (model_name or self.openai_model).strip()
        if not normalized_model_name:
            raise ValueError("模型名称不能为空。")
        return normalized_model_name

    def get_default_base_url_for_service(self, service_name: str | None = None) -> str | None:
        """返回已知服务商的默认 OpenAI 兼容基址。"""
        return DEFAULT_BASE_URLS.get(self.normalize_service_name(service_name))

    def resolve_base_url(
        self,
        service_name: str | None = None,
        base_url: str | None = None,
    ) -> str | None:
        """解析运行时应使用的模型服务基址。"""
        if base_url is not None:
            normalized_base_url = base_url.strip()
            return normalized_base_url or None

        default_base_url = self.get_default_base_url_for_service(service_name)
        if default_base_url:
            return default_base_url

        if self.openai_base_url:
            normalized_base_url = self.openai_base_url.strip()
            if normalized_base_url:
                return normalized_base_url

        return None

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
        resolved_model_name = self.get_model_name(model_name)
        resolved_api_key = (api_key if api_key is not None else self.openai_api_key).strip()
        resolved_base_url = self.resolve_base_url(
            resolved_service_name,
            base_url=base_url,
        )

        if not resolved_api_key:
            raise ValueError("模型 API Key 不能为空。")

        kwargs = {
            "model": resolved_model_name,
            "api_key": resolved_api_key,
            "base_url": resolved_base_url,
            "temperature": 0.7,
            "max_tokens": 1024,
        }
        return {key: value for key, value in kwargs.items() if value is not None}


settings = Settings()
