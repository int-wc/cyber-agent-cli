import importlib
import os
import sys
import unittest
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

# 定义环境变量键名的类型别名
EnvKeys = tuple[str, ...]

CONFIG_ENV_KEYS: EnvKeys = (
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_BASE_URL",
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_MODEL",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_THINKING_MODE",
    "SERVICE_NAME",
)


@contextmanager
def temporary_config_env(**updates: Optional[str]) -> Generator[None, None, None]:
    """
    临时修改环境变量，上下文结束后自动恢复。
    :param updates: 要设置的环境变量键值对，值为 None 时表示删除该变量。
    """
    # 保存原始值
    original_values: Dict[str, Optional[str]] = {
        key: os.environ.get(key) for key in CONFIG_ENV_KEYS
    }
    try:
        # 删除所有相关环境变量
        for key in CONFIG_ENV_KEYS:
            os.environ.pop(key, None)
        # 设置传入的新值
        for key, value in updates.items():
            if value is not None:
                os.environ[key] = value
        yield
    finally:
        # 恢复原始环境变量
        for key in CONFIG_ENV_KEYS:
            os.environ.pop(key, None)
        for key, value in original_values.items():
            if value is not None:
                os.environ[key] = value


def import_config_module() -> Any:
    """
    重新导入 cyber_agent.config 模块，确保每次都是全新加载。
    :return: 重新加载后的 config 模块对象
    """
    sys.modules.pop("cyber_agent.config", None)
    return importlib.import_module("cyber_agent.config")


class SettingsTestCase(unittest.TestCase):
    def test_settings_can_load_required_env_and_code_defaults(self) -> None:
        """
        测试：仅提供 OPENAI_API_KEY 时，其他字段应使用代码中的默认值。
        """
        with temporary_config_env(OPENAI_API_KEY="test-key"):
            config_module = import_config_module()
            # 关闭 .env 文件读取，确保只依赖环境变量和代码默认值
            settings = config_module.Settings(_env_file=None)

        self.assertEqual(settings.openai_api_key, "test-key")
        self.assertEqual(settings.openai_model, "gpt-5.4")
        self.assertIsNone(settings.openai_base_url)
        self.assertEqual(settings.get_service(), "openai")

    def test_module_level_settings_can_be_used_by_callers(self) -> None:
        """
        测试：模块级 settings 单例能正确加载所有环境变量。
        """
        with temporary_config_env(
            OPENAI_API_KEY="runtime-key",
            OPENAI_MODEL="gpt-5.4-mini",
            OPENAI_BASE_URL="https://example.test/v1",
        ):
            config_module = import_config_module()

        self.assertIsInstance(config_module.settings, config_module.Settings)
        self.assertEqual(config_module.settings.openai_api_key, "runtime-key")
        self.assertEqual(config_module.settings.openai_model, "gpt-5.4-mini")
        self.assertEqual(
            config_module.settings.openai_base_url,
            "https://example.test/v1",
        )

    def test_settings_can_build_deepseek_compatible_kwargs(self) -> None:
        """
        测试：切换到 deepseek 时，应保留模型名并继续使用统一模型网关。
        """
        with temporary_config_env(
            OPENAI_API_KEY="openai-key",
            DEEPSEEK_API_KEY="deepseek-key",
            DEEPSEEK_MODEL="deepseek-v4-pro",
            SERVICE_NAME="deepseek",
        ):
            config_module = import_config_module()
            settings = config_module.Settings(_env_file=None)

        kwargs = settings.get_chat_openai_kwargs(settings.get_service())

        self.assertEqual(settings.get_service(), "deepseek")
        self.assertEqual(kwargs["model"], "deepseek-v4-pro")
        self.assertEqual(kwargs["api_key"], "deepseek-key")
        self.assertEqual(kwargs["base_url"], "http://localhost:8317/")
        self.assertEqual(
            kwargs["extra_body"],
            {"provider": "deepseek", "thinking": {"type": "disabled"}},
        )

    def test_service_base_url_always_uses_local_model_gateway(self) -> None:
        """
        测试：切换服务商时不使用服务商专属基址，只走本地模型网关。
        """
        with temporary_config_env(
            DEEPSEEK_API_KEY="deepseek-key",
            DEEPSEEK_MODEL="deepseek-v4-pro",
            OPENAI_BASE_URL="https://example.test/v1",
            DEEPSEEK_BASE_URL="https://deepseek.example/v1",
            SERVICE_NAME="deepseek",
        ):
            config_module = import_config_module()
            settings = config_module.Settings(_env_file=None)

        kwargs = settings.get_chat_openai_kwargs(settings.get_service())

        self.assertEqual(kwargs["base_url"], "http://localhost:8317/")

    def test_deepseek_thinking_mode_can_be_enabled_explicitly(self) -> None:
        """
        测试：只有显式配置时才为 DeepSeek 启用 thinking 模式。
        """
        with temporary_config_env(
            DEEPSEEK_API_KEY="deepseek-key",
            DEEPSEEK_THINKING_MODE="enabled",
            SERVICE_NAME="deepseek",
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
            SERVICE_NAME="deepseek",
        ):
            config_module = import_config_module()
            settings = config_module.Settings(_env_file=None)

        with self.assertRaisesRegex(ValueError, "DEEPSEEK_THINKING_MODE"):
            settings.get_chat_openai_kwargs(settings.get_service())

    def test_deepseek_api_key_can_fallback_to_legacy_openai_key(self) -> None:
        """
        测试：未配置 DEEPSEEK_API_KEY 时，仍兼容旧版只写 OPENAI_API_KEY 的配置。
        """
        with temporary_config_env(
            OPENAI_API_KEY="legacy-deepseek-key",
            SERVICE_NAME="deepseek",
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
            OPENAI_API_KEY="openai-key",
            OPENAI_MODEL="gpt-5.4-mini",
            SERVICE_NAME="openai",
        ):
            config_module = import_config_module()
            settings = config_module.Settings(_env_file=None)

        kwargs = settings.get_chat_openai_kwargs(settings.get_service())

        self.assertEqual(kwargs["base_url"], "http://localhost:8317/")
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
            with temporary_config_env(OPENAI_API_KEY="lazy-import-key"):
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
