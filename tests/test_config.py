import importlib
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

EnvKeys = tuple[str, ...]

CONFIG_ENV_KEYS: EnvKeys = (
    "GATEWAY_API_KEY",
    "GATEWAY_DEFAULT_MODEL",
    "GATEWAY_DEFAULT_MODEL_OPENCODE",
    "GATEWAY_BASE_URL",
    "GATEWAY_DEFAULT_SERVICE",
    "MODEL_PROXY_URL",
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_MODEL",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_THINKING_MODE",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_AUTH_TOKEN",
    "MIMO_API_KEY",
    "MIMO_MODEL",
    "MIMO_BASE_URL",
    "OPENCODE_API_KEY",
    "OPENCODE_MODEL",
    "OPENCODE_BASE_URL",
    "OPENCODE_PROXY_URL",
    "SUBAGENT_MODEL",
    "CYBER_AGENT_HOME",
)


@contextmanager
def temporary_config_env(**updates: Optional[str]) -> Generator[None, None, None]:
    """临时修改环境变量，上下文结束后自动恢复。"""
    original_values: Dict[str, Optional[str]] = {
        key: os.environ.get(key) for key in CONFIG_ENV_KEYS
    }
    try:
        for key in CONFIG_ENV_KEYS:
            os.environ.pop(key, None)
        for key, value in updates.items():
            if value is not None:
                os.environ[key] = value
        yield
    finally:
        for key in CONFIG_ENV_KEYS:
            os.environ.pop(key, None)
        for key, value in original_values.items():
            if value is not None:
                os.environ[key] = value


def import_config_module() -> Any:
    """重新导入 cyber_agent.config 模块，确保每次都是全新加载。"""
    sys.modules.pop("cyber_agent.config", None)
    return importlib.import_module("cyber_agent.config")


class SettingsTestCase(unittest.TestCase):
    def test_settings_can_load_required_env_and_code_defaults(self) -> None:
        """
        测试：仅提供 GATEWAY_API_KEY 时，其他字段应使用代码中的默认值。
        """
        with temporary_config_env(GATEWAY_API_KEY="test-key"):
            config_module = import_config_module()
            settings = config_module.Settings(_env_file=None)

        self.assertEqual(settings.gateway_api_key, "test-key")
        self.assertEqual(settings.gateway_default_model, "deepseek-v4-flash-free")
        self.assertIsNone(settings.gateway_base_url)
        self.assertEqual(settings.resolve_base_url(), "https://opencode.ai/zen/v1")
        self.assertEqual(settings.resolve_proxy_url(), "http://192.168.31.47:7892")
        self.assertEqual(settings.subagent_model, "deepseek-v4-flash-free")
        self.assertEqual(settings.max_context_tokens, 400_000)
        self.assertEqual(settings.get_service(), "opencode")

    def test_module_level_settings_can_be_used_by_callers(self) -> None:
        """
        测试：模块级 settings 单例能正确加载所有环境变量。
        """
        with temporary_config_env(
            GATEWAY_API_KEY="runtime-key",
            GATEWAY_DEFAULT_MODEL="gpt-5.4-mini",
            GATEWAY_BASE_URL="https://example.test/v1",
        ):
            config_module = import_config_module()

        self.assertIsInstance(config_module.settings, config_module.Settings)
        self.assertEqual(config_module.settings.gateway_api_key, "runtime-key")
        self.assertEqual(config_module.settings.gateway_default_model, "gpt-5.4-mini")
        self.assertEqual(
            config_module.settings.gateway_base_url,
            "https://example.test/v1",
        )

    def test_settings_load_env_from_application_home(self) -> None:
        """
        测试：默认配置文件来自 CYBER_AGENT_HOME/.env，而不是当前工作目录。
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            app_home = os.path.join(temp_dir, "app-home")
            cwd = os.path.join(temp_dir, "launch-cwd")
            os.makedirs(app_home)
            os.makedirs(cwd)
            with open(os.path.join(app_home, ".env"), "w", encoding="utf-8") as handle:
                handle.write(
                    "GATEWAY_DEFAULT_SERVICE=opencode\n"
                    "OPENCODE_API_KEY=app-home-key\n"
                    "OPENCODE_MODEL=deepseek-v4-flash-free\n"
                    "OPENCODE_BASE_URL=https://opencode.ai/zen/v1\n"
                )
            with open(os.path.join(cwd, ".env"), "w", encoding="utf-8") as handle:
                handle.write(
                    "GATEWAY_DEFAULT_SERVICE=opencode\n"
                    "OPENCODE_API_KEY=cwd-key-should-not-win\n"
                )

            old_cwd = os.getcwd()
            try:
                os.chdir(cwd)
                with temporary_config_env(CYBER_AGENT_HOME=app_home):
                    config_module = import_config_module()
                    settings = config_module.Settings()
            finally:
                os.chdir(old_cwd)

        self.assertEqual(settings.get_api_key("opencode"), "app-home-key")

    def test_settings_can_build_deepseek_compatible_kwargs(self) -> None:
        """
        测试：切换到 deepseek 时，GATEWAY_DEFAULT_MODEL 优先于 DEEPSEEK_MODEL。
        """
        with temporary_config_env(
            GATEWAY_API_KEY="openai-key",
            GATEWAY_DEFAULT_MODEL="deepseek-v4-pro",
            DEEPSEEK_API_KEY="deepseek-key",
            DEEPSEEK_MODEL="deepseek-v4-pro",
            GATEWAY_BASE_URL="http://127.0.0.1:8317/v1",
            GATEWAY_DEFAULT_SERVICE="deepseek",
        ):
            config_module = import_config_module()
            settings = config_module.Settings(_env_file=None)

        kwargs = settings.get_chat_openai_kwargs(settings.get_service())

        self.assertEqual(settings.get_service(), "deepseek")
        self.assertEqual(kwargs["model"], "deepseek-v4-pro")
        self.assertEqual(kwargs["api_key"], "deepseek-key")
        self.assertEqual(kwargs["base_url"], "http://127.0.0.1:8317/v1")
        self.assertEqual(kwargs["extra_body"]["provider"], "deepseek")
        self.assertEqual(kwargs["extra_body"]["thinking"], {"type": "enabled"})

    def test_service_base_url_uses_service_specific_first(self) -> None:
        """
        测试：服务商专属基址优先于通用网关基址。
        """
        with temporary_config_env(
            DEEPSEEK_API_KEY="deepseek-key",
            DEEPSEEK_MODEL="deepseek-v4-pro",
            GATEWAY_BASE_URL="https://example.test/v1",
            DEEPSEEK_BASE_URL="https://deepseek.example/v1",
            GATEWAY_DEFAULT_SERVICE="deepseek",
        ):
            config_module = import_config_module()
            settings = config_module.Settings(_env_file=None)

        kwargs = settings.get_chat_openai_kwargs(settings.get_service())
        # DEEPSEEK_BASE_URL 优先于 GATEWAY_BASE_URL
        self.assertEqual(kwargs["base_url"], "https://deepseek.example/v1")

    def test_opencode_can_use_service_specific_key_model_and_base_url(self) -> None:
        """
        测试：OpenCode Zen 供应商可使用专属 key、模型和基址。
        """
        with temporary_config_env(
            GATEWAY_API_KEY="gateway-key",
            GATEWAY_DEFAULT_MODEL="global-model",
            GATEWAY_BASE_URL="https://gateway.example/v1",
            GATEWAY_DEFAULT_SERVICE="opencode",
            OPENCODE_API_KEY="opencode-key",
            OPENCODE_MODEL="deepseek-v4-flash-free",
            OPENCODE_BASE_URL="https://opencode.ai/zen/v1",
            OPENCODE_PROXY_URL="http://proxy.example:7892",
        ):
            config_module = import_config_module()
            settings = config_module.Settings(_env_file=None)

        kwargs = settings.get_chat_openai_kwargs(settings.get_service())

        self.assertEqual(settings.get_service(), "opencode")
        self.assertEqual(kwargs["model"], "deepseek-v4-flash-free")
        self.assertEqual(kwargs["api_key"], "opencode-key")
        self.assertEqual(kwargs["base_url"], "https://opencode.ai/zen/v1")
        self.assertEqual(kwargs["openai_proxy"], "http://proxy.example:7892")
        self.assertEqual(kwargs["extra_body"], {"provider": "opencode"})

    def test_opencode_proxy_can_use_socks_alias(self) -> None:
        """
        测试：socks:// 作为 socks5:// 的便捷写法。
        """
        with temporary_config_env(
            GATEWAY_DEFAULT_SERVICE="opencode",
            OPENCODE_API_KEY="opencode-key",
            OPENCODE_PROXY_URL="socks://192.168.31.47:7892",
        ):
            config_module = import_config_module()
            settings = config_module.Settings(_env_file=None)

        kwargs = settings.get_chat_openai_kwargs(settings.get_service())

        self.assertEqual(kwargs["openai_proxy"], "socks5://192.168.31.47:7892")

    def test_opencode_proxy_does_not_leak_to_other_services(self) -> None:
        """
        测试：OpenCode 专属代理不会影响其他服务商。
        """
        with temporary_config_env(
            GATEWAY_DEFAULT_SERVICE="deepseek",
            DEEPSEEK_API_KEY="deepseek-key",
            OPENCODE_PROXY_URL="http://proxy.example:7892",
        ):
            config_module = import_config_module()
            settings = config_module.Settings(_env_file=None)

        kwargs = settings.get_chat_openai_kwargs(settings.get_service())

        self.assertNotIn("openai_proxy", kwargs)

    def test_opencode_model_can_be_overridden_by_gateway_service_model(self) -> None:
        """
        测试：GATEWAY_DEFAULT_MODEL_<SERVICE> 仍可覆盖服务商专属模型。
        """
        with temporary_config_env(
            GATEWAY_DEFAULT_SERVICE="opencode",
            GATEWAY_DEFAULT_MODEL_OPENCODE="custom-opencode-model",
            OPENCODE_API_KEY="opencode-key",
            OPENCODE_MODEL="deepseek-v4-flash-free",
        ):
            config_module = import_config_module()
            settings = config_module.Settings(_env_file=None)

            self.assertEqual(
                settings.get_chat_openai_kwargs(settings.get_service())["model"],
                "custom-opencode-model",
            )

    def test_deepseek_thinking_mode_can_be_enabled_explicitly(self) -> None:
        """
        测试：只有显式配置时才为 DeepSeek 启用 thinking 模式。
        """
        with temporary_config_env(
            DEEPSEEK_API_KEY="deepseek-key",
            DEEPSEEK_THINKING_MODE="enabled",
            GATEWAY_DEFAULT_SERVICE="deepseek",
        ):
            config_module = import_config_module()
            settings = config_module.Settings(_env_file=None)

        kwargs = settings.get_chat_openai_kwargs(settings.get_service())

        self.assertTrue(settings.is_deepseek_thinking_enabled())
        self.assertEqual(
            kwargs["extra_body"],
            {"provider": "deepseek", "thinking": {"type": "enabled"}},
        )

    def test_deepseek_thinking_mode_rejects_unknown_value(self) -> None:
        """
        测试：DeepSeek thinking 模式只接受 enabled 或 disabled，避免静默错配。
        """
        with temporary_config_env(
            DEEPSEEK_API_KEY="deepseek-key",
            DEEPSEEK_THINKING_MODE="maybe",
            GATEWAY_DEFAULT_SERVICE="deepseek",
        ):
            config_module = import_config_module()
            settings = config_module.Settings(_env_file=None)

        with self.assertRaisesRegex(ValueError, "DEEPSEEK_THINKING_MODE"):
            settings.get_chat_openai_kwargs(settings.get_service())

    def test_deepseek_api_key_can_fallback_to_gateway_key(self) -> None:
        """
        测试：未配置 DEEPSEEK_API_KEY 时，回退到 GATEWAY_API_KEY。
        """
        with temporary_config_env(
            GATEWAY_API_KEY="legacy-deepseek-key",
            GATEWAY_DEFAULT_SERVICE="deepseek",
        ):
            config_module = import_config_module()
            settings = config_module.Settings(_env_file=None)

        kwargs = settings.get_chat_openai_kwargs(settings.get_service())

        self.assertEqual(kwargs["api_key"], "legacy-deepseek-key")

    def test_openai_kwargs_include_provider_for_local_gateway(self) -> None:
        """
        测试：OpenAI 服务也会向本地网关传递 provider 字段。
        """
        with temporary_config_env(
            GATEWAY_API_KEY="openai-key",
            GATEWAY_DEFAULT_MODEL="gpt-5.4-mini",
            GATEWAY_BASE_URL="http://127.0.0.1:8317/v1",
            GATEWAY_DEFAULT_SERVICE="openai",
        ):
            config_module = import_config_module()
            settings = config_module.Settings(_env_file=None)

        kwargs = settings.get_chat_openai_kwargs(settings.get_service())

        self.assertEqual(kwargs["base_url"], "http://127.0.0.1:8317/v1")
        self.assertEqual(kwargs["extra_body"], {"provider": "openai"})

    def test_package_root_import_should_not_eagerly_import_heavy_submodules(self) -> None:
        """
        测试：导入 cyber_agent 包根模块时，不应立刻加载 CLI 和搜索等重模块。
        """
        module_names = (
            "cyber_agent",
            "cyber_agent.agent",
            "cyber_agent.agent.mode",
            "cyber_agent.cli.app",
            "cyber_agent.tools",
            "cyber_agent.tools.search",
            "cyber_agent.capability_registry",
        )
        original_modules = {
            module_name: sys.modules.get(module_name)
            for module_name in module_names
        }

        try:
            with temporary_config_env(GATEWAY_API_KEY="lazy-import-key"):
                for module_name in module_names:
                    sys.modules.pop(module_name, None)

                package_module = importlib.import_module("cyber_agent")

            self.assertEqual(package_module.__version__, "0.1.0")
            self.assertNotIn("cyber_agent.cli.app", sys.modules)
            self.assertNotIn("cyber_agent.tools.search", sys.modules)
            self.assertNotIn("cyber_agent.capability_registry", sys.modules)

            self.assertEqual(package_module.AgentMode.__name__, "AgentMode")
            self.assertIn("cyber_agent.agent.mode", sys.modules)
        finally:
            for module_name in module_names:
                sys.modules.pop(module_name, None)
            for module_name, module in original_modules.items():
                if module is not None:
                    sys.modules[module_name] = module

if __name__ == "__main__":
    unittest.main()
