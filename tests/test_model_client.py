import unittest
from unittest.mock import patch

import cyber_agent.model_client as model_client
from cyber_agent.model_client import (
    build_llm_with_proxy_fallback,
    get_http_proxy_fallback_url,
)


class ModelClientProxyFallbackTestCase(unittest.TestCase):
    def setUp(self) -> None:
        with model_client._SOCKSIO_FALLBACK_WARNING_LOCK:
            model_client._SOCKSIO_FALLBACK_WARNING_EMITTED = False

    def test_socks_proxy_can_fallback_to_http_proxy(self) -> None:
        self.assertEqual(
            get_http_proxy_fallback_url("socks5://192.168.31.47:7892"),
            "http://192.168.31.47:7892",
        )

    def test_missing_socksio_retries_with_http_proxy(self) -> None:
        calls: list[str] = []

        class FakeLlm:
            def __init__(self, **kwargs):
                proxy = kwargs.get("openai_proxy")
                calls.append(proxy)
                if proxy == "socks5://192.168.31.47:7892":
                    raise ImportError(
                        "Using SOCKS proxy, but the 'socksio' package is not installed."
                    )
                self.proxy = proxy

        llm = build_llm_with_proxy_fallback(
            FakeLlm,
            {"openai_proxy": "socks5://192.168.31.47:7892"},
        )

        self.assertEqual(llm.proxy, "http://192.168.31.47:7892")
        self.assertEqual(
            calls,
            ["socks5://192.168.31.47:7892", "http://192.168.31.47:7892"],
        )

    def test_missing_socksio_warning_is_emitted_once(self) -> None:
        class FakeLlm:
            def __init__(self, **kwargs):
                proxy = kwargs.get("openai_proxy")
                if str(proxy).startswith("socks"):
                    raise ImportError(
                        "Using SOCKS proxy, but the 'socksio' package is not installed."
                    )
                self.proxy = proxy

        with patch("cyber_agent.logging.log_warning") as log_warning:
            first = build_llm_with_proxy_fallback(
                FakeLlm,
                {"openai_proxy": "socks5://192.168.31.47:7892"},
            )
            second = build_llm_with_proxy_fallback(
                FakeLlm,
                {"openai_proxy": "socks5://192.168.31.47:7892"},
            )

        self.assertEqual(first.proxy, "http://192.168.31.47:7892")
        self.assertEqual(second.proxy, "http://192.168.31.47:7892")
        log_warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()
