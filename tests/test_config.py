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


if __name__ == "__main__":
    unittest.main()
