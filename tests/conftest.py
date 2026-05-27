"""pytest 配置：测试环境隔离，确保测试不使用生产 Anthropic API 配置。"""

from __future__ import annotations

import os


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
    ):
        os.environ.pop(key, None)
    # 设置测试用网关基址，确保 resolve_base_url 走 OpenAI 路径
    os.environ.setdefault("GATEWAY_BASE_URL", "http://127.0.0.1:8317/v1")
    os.environ.setdefault("GATEWAY_API_KEY", "test-gateway-key")
