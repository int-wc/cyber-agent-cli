"""pytest 配置：测试环境隔离，确保测试不使用生产 Anthropic API 配置。"""

from __future__ import annotations

import os
from contextlib import contextmanager
from tempfile import TemporaryDirectory

from typer.testing import CliRunner


if not hasattr(CliRunner, "isolated_filesystem"):
    @contextmanager
    def _isolated_filesystem(self: CliRunner):  # type: ignore[no-untyped-def]
        """兼容旧版测试写法：在临时目录中执行 CliRunner 用例。"""
        old_cwd = os.getcwd()
        old_home = os.environ.get("CYBER_AGENT_HOME")
        with TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            os.environ["CYBER_AGENT_HOME"] = temp_dir
            try:
                yield temp_dir
            finally:
                os.chdir(old_cwd)
                if old_home is None:
                    os.environ.pop("CYBER_AGENT_HOME", None)
                else:
                    os.environ["CYBER_AGENT_HOME"] = old_home

    CliRunner.isolated_filesystem = _isolated_filesystem  # type: ignore[attr-defined]


def pytest_configure(config: object) -> None:
    """在 pytest 启动时（任何模块导入之前）清除 Anthropic API 环境变量。

    这样 cyber_agent.config.Settings 单例在首次实例化时就不会读取到
    ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN，从而回退到 OpenAI 兼容路径。
    """
    for key in (
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_AUTH_TOKEN",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_API_KEY",
        "MIMO_BASE_URL",
        "MIMO_API_KEY",
        "OPENCODE_BASE_URL",
        "OPENCODE_API_KEY",
    ):
        os.environ.pop(key, None)
    # 设置测试用网关基址，确保 resolve_base_url 走 OpenAI 路径
    os.environ.setdefault("GATEWAY_BASE_URL", "http://127.0.0.1:8317/v1")
    os.environ.setdefault("GATEWAY_API_KEY", "test-gateway-key")
