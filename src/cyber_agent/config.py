from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .cc_switch_sync import load_cc_switch_openai_compatible_providers
from .local_config import get_application_env_file

DEFAULT_MODEL_GATEWAY_BASE_URL = "http://127.0.0.1:8317/v1"
# 服务商默认模型。可通过 GATEWAY_DEFAULT_MODEL_<SERVICE> 环境变量覆盖。
DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-5.4",
    "deepseek": "deepseek-v4-flash",
    "opencode": "deepseek-v4-flash-free",
    "mimo": "mimo-v2.5-pro",
    "claude": "claude-opus-4-6",
    "baisub": "",
    "ccswitch": "",
}
DEFAULT_OPENCODE_BASE_URL = "https://opencode.ai/zen/v1"
DEFAULT_BAISUB_BASE_URL = "https://baisub.bai.edu.kg/v1"
DEFAULT_OPENCODE_PROXY_URL = "http://192.168.31.47:7892"
ANTHROPIC_TOKEN_SERVICES = frozenset({"deepseek", "mimo", "claude"})
# DeepSeek V4 Pro [1m] 上下文窗口参数
DEEPSEEK_MAX_CONTEXT_TOKENS = 4_000_000
DEEPSEEK_AUTO_COMPACT_WINDOW = 4_000_000
DEFAULT_MAX_CONTEXT_CHARS = 4_000_000


class Settings(BaseSettings):
    # ── 统一模型网关 ──
    gateway_api_key: str = Field(
        default="sk-default",
        validation_alias="GATEWAY_API_KEY",
    )
    gateway_default_model: str = Field(
        default=DEFAULT_MODELS["opencode"],
        validation_alias="GATEWAY_DEFAULT_MODEL",
    )
    gateway_base_url: str | None = Field(
        default=None,
        validation_alias="GATEWAY_BASE_URL",
    )
    model_proxy_url: str | None = Field(
        default=None,
        validation_alias="MODEL_PROXY_URL",
    )
    gateway_default_service: str = Field(
        default="opencode",
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

    # ── OpenCode Zen 专属 ──
    opencode_api_key: str | None = Field(
        default=None,
        validation_alias="OPENCODE_API_KEY",
    )
    opencode_model: str = Field(
        default=DEFAULT_MODELS["opencode"],
        validation_alias="OPENCODE_MODEL",
    )
    opencode_base_url: str | None = Field(
        default=DEFAULT_OPENCODE_BASE_URL,
        validation_alias="OPENCODE_BASE_URL",
    )
    opencode_proxy_url: str | None = Field(
        default=DEFAULT_OPENCODE_PROXY_URL,
        validation_alias="OPENCODE_PROXY_URL",
    )

    # ── BaiSub OpenAI 兼容 API ──
    baisub_api_key: str | None = Field(
        default=None,
        validation_alias="BAISUB_API_KEY",
    )
    baisub_api_keys: str | None = Field(
        default=None,
        validation_alias="BAISUB_API_KEYS",
    )
    baisub_model: str | None = Field(
        default=None,
        validation_alias="BAISUB_MODEL",
    )
    baisub_models: str | None = Field(
        default=None,
        validation_alias="BAISUB_MODELS",
    )
    baisub_base_url: str | None = Field(
        default=DEFAULT_BAISUB_BASE_URL,
        validation_alias="BAISUB_BASE_URL",
    )
    cc_switch_sync_enabled: bool = Field(
        default=True,
        validation_alias="CC_SWITCH_SYNC_ENABLED",
    )
    cc_switch_db_path: str | None = Field(
        default=None,
        validation_alias="CC_SWITCH_DB_PATH",
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
        default=DEFAULT_MAX_CONTEXT_CHARS,
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
        default=DEFAULT_MODELS["opencode"],
        validation_alias="SUBAGENT_MODEL",
    )
    multi_agent_max_workers: int = Field(
        default=8,
        validation_alias="MULTI_AGENT_MAX_WORKERS",
    )
    pipeline_max_subagents: int = Field(
        default=4,
        validation_alias="PIPELINE_MAX_SUBAGENTS",
    )
    pipeline_subtask_concurrency: str = Field(
        default="auto",
        validation_alias="PIPELINE_SUBTASK_CONCURRENCY",
    )
    pipeline_execution_profile: str = Field(
        default="auto",
        validation_alias="PIPELINE_EXECUTION_PROFILE",
    )
    pipeline_max_iterations: int = Field(
        default=20,
        validation_alias="PIPELINE_MAX_ITERATIONS",
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
        env_file=str(get_application_env_file()),
        extra="ignore",
    )

    def normalize_service_name(self, service_name: str | None = None) -> str:
        """规范化服务商名称，避免展示和运行时出现大小写不一致。"""
        normalized_service_name = (service_name or self.gateway_default_service).strip().lower()
        return normalized_service_name or "openai"

    def get_service(self) -> str:
        """返回当前默认服务商名称。"""
        return self.normalize_service_name()

    @staticmethod
    def _split_config_list(raw_value: str | None) -> list[str]:
        if not raw_value:
            return []
        return [
            item.strip()
            for item in raw_value.replace("\n", ",").split(",")
            if item.strip()
        ]

    def _get_baisub_models(self) -> list[str]:
        models = self._split_config_list(self.baisub_models)
        if models:
            return models
        model = (self.baisub_model or "").strip()
        return [model] if model else []

    def _get_cc_switch_provider_pool(self):
        if not self.cc_switch_sync_enabled:
            return []
        return load_cc_switch_openai_compatible_providers(self.cc_switch_db_path)

    def _get_primary_cc_switch_provider(self):
        providers = self._get_cc_switch_provider_pool()
        return providers[0] if providers else None

    def get_model_name(
        self,
        model_name: str | None = None,
        service_name: str | None = None,
    ) -> str:
        """返回当前默认模型名称。

        优先级：显式传入 > GATEWAY_DEFAULT_MODEL_<SERVICE> > 服务商专属模型 >
        GATEWAY_DEFAULT_MODEL > 服务商默认。
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
        primary_cc_switch_provider = self._get_primary_cc_switch_provider()
        service_models: dict[str, str | None] = {
            "deepseek": self.deepseek_model,
            "mimo": self.mimo_model,
            "opencode": self.opencode_model,
            "baisub": (self.baisub_model or (self._get_baisub_models() or [""])[0]),
            "ccswitch": primary_cc_switch_provider.model if primary_cc_switch_provider else None,
        }
        service_model = (service_models.get(normalized_service_name) or "").strip()
        if service_model:
            return service_model
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
        primary_cc_switch_provider = self._get_primary_cc_switch_provider()
        service_keys: dict[str, str | None] = {
            "deepseek": self.deepseek_api_key,
            "mimo": self.mimo_api_key,
            "opencode": self.opencode_api_key,
            "baisub": self.baisub_api_key or (self._split_config_list(self.baisub_api_keys) or [None])[0],
            "ccswitch": primary_cc_switch_provider.api_key if primary_cc_switch_provider else None,
        }
        specific_key = service_keys.get(normalized)
        if specific_key and specific_key.strip():
            return specific_key.strip()
        # Anthropic 兼容 API token（DeepSeek/MiMo/Claude 等支持）
        if (
            normalized in ANTHROPIC_TOKEN_SERVICES
            and self.anthropic_auth_token
            and self.anthropic_auth_token.strip()
        ):
            return self.anthropic_auth_token.strip()
        # 默认 gateway key
        resolved = (self.gateway_api_key or "").strip()
        if not resolved:
            raise ValueError(
                f"服务商 {normalized} 的 API Key 未配置。"
                f" 请设置 {normalized.upper()}_API_KEY 或 GATEWAY_API_KEY。"
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
        primary_cc_switch_provider = self._get_primary_cc_switch_provider()
        service_urls: dict[str, str | None] = {
            "deepseek": self.deepseek_base_url,
            "mimo": self.mimo_base_url,
            "opencode": self.opencode_base_url,
            "baisub": self.baisub_base_url,
            "ccswitch": primary_cc_switch_provider.base_url if primary_cc_switch_provider else None,
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

    def normalize_proxy_url(self, proxy_url: str | None = None) -> str | None:
        """规范化代理地址；socks:// 作为 socks5:// 的便捷别名。"""
        normalized = (proxy_url or "").strip()
        if not normalized:
            return None
        if normalized.startswith("socks://"):
            return "socks5://" + normalized[len("socks://"):]
        return normalized

    def resolve_proxy_url(self, service_name: str | None = None) -> str | None:
        """解析模型客户端代理地址。

        优先级：服务商专属代理 > MODEL_PROXY_URL。当前仅为 opencode 设置默认代理。
        """
        normalized_service = self.normalize_service_name(service_name)
        service_proxies: dict[str, str | None] = {
            "opencode": self.opencode_proxy_url,
        }
        specific_proxy = self.normalize_proxy_url(service_proxies.get(normalized_service))
        if specific_proxy:
            return specific_proxy
        return self.normalize_proxy_url(self.model_proxy_url)

    def get_default_headers(
        self,
        service_name: str | None = None,
        base_url: str | None = None,
    ) -> dict[str, str] | None:
        """返回 OpenAI 兼容客户端的默认请求头。

        某些供应商在流式调用时会对默认 User-Agent 有限制，这里集中处理。
        """
        normalized = self.normalize_service_name(service_name)
        if normalized in {"ai952048", "ccswitch"} or "ai.952048.xyz" in (base_url or ""):
            return {"User-Agent": "curl/8.0"}
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
            "extra_body": extra_body,
        }
        resolved_proxy_url = self.resolve_proxy_url(resolved_service_name)
        if resolved_proxy_url:
            kwargs["openai_proxy"] = resolved_proxy_url
        default_headers = self.get_default_headers(
            resolved_service_name,
            base_url=resolved_base_url,
        )
        if default_headers:
            kwargs["default_headers"] = default_headers
        fallback_kwargs = self.get_chat_openai_fallback_kwargs(
            resolved_service_name,
            model_name=resolved_model_name,
            api_key=resolved_api_key,
            base_url=resolved_base_url,
        )
        if fallback_kwargs:
            kwargs["_fallback_kwargs"] = fallback_kwargs
        return {key: value for key, value in kwargs.items() if value is not None}

    def get_chat_openai_fallback_kwargs(
        self,
        service_name: str | None = None,
        *,
        model_name: str,
        api_key: str,
        base_url: str | None,
    ) -> list[dict]:
        """返回 OpenAI 兼容客户端的备用 key/model 初始化参数。"""
        resolved_service_name = self.normalize_service_name(service_name)
        if resolved_service_name == "ccswitch":
            fallback_kwargs: list[dict] = []
            if self.cc_switch_sync_enabled:
                primary_provider = self._get_primary_cc_switch_provider()
                for provider in self._get_cc_switch_provider_pool():
                    if primary_provider is not None and provider.id == primary_provider.id:
                        continue
                    candidate = {
                        "model": provider.model,
                        "api_key": provider.api_key,
                        "base_url": provider.base_url,
                        "temperature": 0.7,
                        "extra_body": {"provider": provider.service_name},
                    }
                    default_headers = self.get_default_headers(
                        provider.service_name,
                        base_url=provider.base_url,
                    )
                    if default_headers:
                        candidate["default_headers"] = default_headers
                    fallback_kwargs.append(
                        {item_key: value for item_key, value in candidate.items() if value is not None}
                    )

            baisub_models = self._get_baisub_models()
            for index, key in enumerate(self._split_config_list(self.baisub_api_keys)):
                model = (
                    baisub_models[index]
                    if len(baisub_models) > index
                    else (baisub_models[0] if baisub_models else model_name)
                )
                fallback_kwargs.append(
                    {
                        "model": model,
                        "api_key": key,
                        "base_url": self.baisub_base_url,
                        "temperature": 0.7,
                        "extra_body": {"provider": "baisub"},
                    }
                )
            if self.opencode_api_key and self.opencode_api_key.strip():
                candidate = {
                    "model": self.opencode_model,
                    "api_key": self.opencode_api_key.strip(),
                    "base_url": self.opencode_base_url,
                    "temperature": 0.7,
                    "extra_body": {"provider": "opencode"},
                }
                resolved_proxy_url = self.resolve_proxy_url("opencode")
                if resolved_proxy_url:
                    candidate["openai_proxy"] = resolved_proxy_url
                fallback_kwargs.append(
                    {item_key: value for item_key, value in candidate.items() if value is not None}
                )
            return fallback_kwargs
        if resolved_service_name == "ai952048":
            fallback_kwargs: list[dict] = []
            baisub_keys = self._split_config_list(self.baisub_api_keys)
            baisub_models = self._get_baisub_models()
            if baisub_keys:
                for index, key in enumerate(baisub_keys):
                    candidate_model = (
                        baisub_models[index]
                        if len(baisub_models) > index
                        else (baisub_models[0] if baisub_models else model_name)
                    )
                    candidate = {
                        "model": candidate_model,
                        "api_key": key,
                        "base_url": self.baisub_base_url,
                        "temperature": 0.7,
                        "extra_body": {"provider": "baisub"},
                    }
                    resolved_proxy_url = self.resolve_proxy_url("baisub")
                    if resolved_proxy_url:
                        candidate["openai_proxy"] = resolved_proxy_url
                    fallback_kwargs.append(
                        {item_key: value for item_key, value in candidate.items() if value is not None}
                    )
            if self.opencode_api_key and self.opencode_api_key.strip():
                candidate = {
                    "model": self.opencode_model,
                    "api_key": self.opencode_api_key.strip(),
                    "base_url": self.opencode_base_url,
                    "temperature": 0.7,
                    "extra_body": {"provider": "opencode"},
                }
                resolved_proxy_url = self.resolve_proxy_url("opencode")
                if resolved_proxy_url:
                    candidate["openai_proxy"] = resolved_proxy_url
                fallback_kwargs.append(
                    {item_key: value for item_key, value in candidate.items() if value is not None}
                )
            return fallback_kwargs
        if resolved_service_name != "baisub":
            return []

        keys: list[str] = []
        for key in [api_key, self.baisub_api_key or "", *self._split_config_list(self.baisub_api_keys)]:
            cleaned = key.strip()
            if cleaned and cleaned not in keys:
                keys.append(cleaned)
        if len(keys) <= 1:
            return []

        configured_models = self._get_baisub_models()
        fallback_kwargs: list[dict] = []
        for index, key in enumerate(keys[1:], start=1):
            if len(configured_models) > index:
                candidate_model = configured_models[index]
            elif len(configured_models) == 1:
                candidate_model = configured_models[0]
            else:
                candidate_model = model_name
            candidate = {
                "model": candidate_model,
                "api_key": key,
                "base_url": base_url,
                "temperature": 0.7,
                "extra_body": {"provider": resolved_service_name},
            }
            resolved_proxy_url = self.resolve_proxy_url(resolved_service_name)
            if resolved_proxy_url:
                candidate["openai_proxy"] = resolved_proxy_url
            fallback_kwargs.append(
                {item_key: value for item_key, value in candidate.items() if value is not None}
            )
        return fallback_kwargs


settings = Settings()
