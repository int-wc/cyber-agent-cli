import importlib
import os
import sys
import unittest
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional, TypeVar

# 定义环境变量键名的类型别名
EnvKeys = tuple[str, str, str]

CONFIG_ENV_KEYS: EnvKeys = (
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_BASE_URL",
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
        测试：切换到 deepseek 时，应保留模型名并自动补出默认兼容基址。
        """
        with temporary_config_env(
            OPENAI_API_KEY="deepseek-key",
            OPENAI_MODEL="deepseek-chat",
            SERVICE_NAME="deepseek",
        ):
            config_module = import_config_module()
            settings = config_module.Settings(_env_file=None)

        kwargs = settings.get_chat_openai_kwargs(settings.get_service())

        self.assertEqual(settings.get_service(), "deepseek")
        self.assertEqual(kwargs["model"], "deepseek-chat")
        self.assertEqual(kwargs["api_key"], "deepseek-key")
        self.assertEqual(kwargs["base_url"], "https://api.deepseek.com/v1")

    def test_deepseek_default_base_url_should_override_generic_proxy_base_url(self) -> None:
        """
        测试：当服务商是 deepseek 时，不应继续沿用 OPENAI_BASE_URL 中的通用代理地址。
        """
        with temporary_config_env(
            OPENAI_API_KEY="deepseek-key",
            OPENAI_MODEL="deepseek-chat",
            OPENAI_BASE_URL="https://example.test/v1",
            SERVICE_NAME="deepseek",
        ):
            config_module = import_config_module()
            settings = config_module.Settings(_env_file=None)

        kwargs = settings.get_chat_openai_kwargs(settings.get_service())

        self.assertEqual(kwargs["base_url"], "https://api.deepseek.com/v1")

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
