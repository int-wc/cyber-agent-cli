from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_MODEL_GATEWAY_BASE_URL = "http://127.0.0.1:8317/v1"
# 服务商默认模型。可通过 GATEWAY_DEFAULT_MODEL_<SERVICE> 环境变量覆盖。
DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-5.4",
    "deepseek": "deepseek-v4-pro[1m]",
    "mimo": "mimo-v2.5-pro",
    "claude": "claude-opus-4-6"
}
# DeepSeek V4 Pro [1m] 上下文窗口参数
DEEPSEEK_MAX_CONTEXT_TOKENS = 1_000_000
DEEPSEEK_AUTO_COMPACT_WINDOW = 400_000


class Settings(BaseSettings):
    # ── 统一模型网关 ──
    gateway_api_key: str = Field(
        default="sk-default",
        validation_alias="GATEWAY_API_KEY",
    )
    gateway_default_model: str = Field(
        default="deepseek-v4-pro[1m]",
        validation_alias="GATEWAY_DEFAULT_MODEL",
    )
    gateway_base_url: str | None = Field(
        default=None,
        validation_alias="GATEWAY_BASE_URL",
    )
    gateway_default_service: str = Field(
        default="deepseek",
        validation_alias="GATEWAY_DEFAULT_SERVICE",
    )

    # ── Anthropic 兼容 API（DeepSeek / MiMo 等支持 Anthropic 格式的服务商）──
    anthropic_base_url: str | None = Field(
        default=None,
        validation_alias="ANTHROPIC_BASE_URL",
    )
    anthropic_auth_token: str | None = Field(
        default=None,
        validation_alias="ANTHROPIC_AUTH_TOKEN",
    )

    # ── DeepSeek 专属 ──
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
        default="enabled",
        validation_alias="DEEPSEEK_THINKING_MODE",
    )

    # ── MiMo 专属 ──
    mimo_api_key: str | None = Field(
        default=None,
        validation_alias="MIMO_API_KEY",
    )
    mimo_model: str = Field(
        default=DEFAULT_MODELS["mimo"],
        validation_alias="MIMO_MODEL",
    )
    mimo_base_url: str | None = Field(
        default=None,
        validation_alias="MIMO_BASE_URL",
    )

    # ── 搜索工具 ──
    search_endpoint: str = Field(
        default="https://html.duckduckgo.com/html/",
        validation_alias="SEARCH_ENDPOINT",
    )
    search_timeout_seconds: float = Field(
        default=6.0,
        validation_alias="SEARCH_TIMEOUT_SECONDS",
    )
    search_result_limit: int = Field(
        default=40,
        validation_alias="SEARCH_RESULT_LIMIT",
    )
    search_show_browser: bool = Field(
        default=True,
        validation_alias="SEARCH_SHOW_BROWSER",
    )

    # ── 上下文压缩 ──
    max_context_chars: int = Field(
        default=30000,
        validation_alias="MAX_CONTEXT_CHARS",
    )
    max_context_tokens: int = Field(
        default=DEEPSEEK_AUTO_COMPACT_WINDOW,
        validation_alias="MAX_CONTEXT_TOKENS",
    )
    auto_compact_window: int = Field(
        default=DEEPSEEK_AUTO_COMPACT_WINDOW,
        validation_alias="AUTO_COMPACT_WINDOW",
    )
    context_keep_recent_messages: int = Field(
        default=12,
        validation_alias="CONTEXT_KEEP_RECENT_MESSAGES",
    )
    context_summary_max_chars: int = Field(
        default=4000,
        validation_alias="CONTEXT_SUMMARY_MAX_CHARS",
    )

    # ── 多 Agent 架构 ──
    subagent_model: str = Field(
        default="deepseek-v4-flash",
        validation_alias="SUBAGENT_MODEL",
    )
    multi_agent_max_workers: int = Field(
        default=8,
        validation_alias="MULTI_AGENT_MAX_WORKERS",
    )
    agent_effort_level: str = Field(
        default="max",
        validation_alias="AGENT_EFFORT_LEVEL",
    )

    # ── 动态 capability ──
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
        normalized_service_name = (service_name or self.gateway_default_service).strip().lower()
        return normalized_service_name or "openai"

    def get_service(self) -> str:
        """返回当前默认服务商名称。"""
        return self.normalize_service_name()

    def get_model_name(
        self,
        model_name: str | None = None,
        service_name: str | None = None,
    ) -> str:
        """返回当前默认模型名称。

        优先级：显式传入 > GATEWAY_DEFAULT_MODEL_<SERVICE> > GATEWAY_DEFAULT_MODEL > 服务商默认。
        """
        import os

        if model_name is not None:
            return model_name.strip()
        normalized_service_name = self.normalize_service_name(service_name)
        # 服务商专属模型环境变量
        env_key = f"GATEWAY_DEFAULT_MODEL_{normalized_service_name.upper()}"
        env_model = os.getenv(env_key, "").strip()
        if env_model:
            return env_model
        # 全局默认模型
        global_default = self.gateway_default_model.strip()
        if global_default:
            return global_default
        # 服务商默认模型
        resolved = DEFAULT_MODELS.get(normalized_service_name, DEFAULT_MODELS.get("openai", ""))
        if not resolved:
            raise ValueError("模型名称不能为空。")
        return resolved

    def get_api_key(
        self,
        service_name: str | None = None,
        api_key: str | None = None,
    ) -> str:
        """按服务商解析 API Key。
        优先级：显式传入 > 服务商专属 key > ANTHROPIC_AUTH_TOKEN > gateway_api_key。"""
        if api_key is not None:
            return api_key.strip()
        normalized = self.normalize_service_name(service_name)
        # 服务商专属 key
        service_keys: dict[str, str | None] = {
            "deepseek": self.deepseek_api_key,
            "mimo": self.mimo_api_key,
        }
        specific_key = service_keys.get(normalized)
        if specific_key and specific_key.strip():
            return specific_key.strip()
        # Anthropic 兼容 API token（DeepSeek/MiMo 都支持）
        if self.anthropic_auth_token and self.anthropic_auth_token.strip():
            return self.anthropic_auth_token.strip()
        # 默认 gateway key
        resolved = (self.gateway_api_key or "").strip()
        if not resolved:
            raise ValueError(
                f"服务商 {normalized} 的 API Key 未配置。"
                f" 请设置 ANTHROPIC_AUTH_TOKEN 或 DEEPSEEK_API_KEY 或 GATEWAY_API_KEY。"
            )
        return resolved

    def get_default_base_url_for_service(self, service_name: str | None = None) -> str | None:
        """返回统一模型网关基址，切换服务商时只改变 provider 与模型名称。"""
        _ = service_name
        configured_base_url = (self.gateway_base_url or "").strip()
        return configured_base_url or DEFAULT_MODEL_GATEWAY_BASE_URL

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
        """解析运行时应使用的模型服务基址。

        优先级：显式传入 > 服务商专属 URL > GATEWAY_BASE_URL > ANTHROPIC_BASE_URL > 默认网关。
        """
        if base_url is not None and base_url.strip():
            return base_url.strip()
        normalized = self.normalize_service_name(service_name)
        service_urls: dict[str, str | None] = {
            "deepseek": self.deepseek_base_url,
            "mimo": self.mimo_base_url,
        }
        specific_url = service_urls.get(normalized)
        if specific_url and specific_url.strip():
            return specific_url.strip()
        # 显式配置的网关基址优先
        gateway_url = (self.gateway_base_url or "").strip()
        if gateway_url:
            return gateway_url
        # Anthropic 兼容 API 兜底
        if self.anthropic_base_url and self.anthropic_base_url.strip():
            return self.anthropic_base_url.strip()
        return DEFAULT_MODEL_GATEWAY_BASE_URL

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
