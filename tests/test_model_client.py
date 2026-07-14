import unittest
from unittest.mock import patch

import cyber_agent.model_client as model_client
from cyber_agent.model_client import (
    RotatingLlmClient,
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

    def test_provider_fallback_retries_with_next_key_and_model(self) -> None:
        calls: list[tuple[str, str]] = []

        class FakeLlm:
            def __init__(self, **kwargs):
                self.api_key = kwargs["api_key"]
                self.model = kwargs["model"]

            def invoke(self, messages):
                calls.append((self.api_key, self.model))
                if self.api_key == "bad-key":
                    raise RuntimeError("401 invalid api key")
                return f"ok:{self.api_key}:{self.model}"

        llm = build_llm_with_proxy_fallback(
            FakeLlm,
            {
                "api_key": "bad-key",
                "model": "model-a",
                "_fallback_kwargs": [
                    {
                        "api_key": "good-key",
                        "model": "model-b",
                    }
                ],
            },
        )

        self.assertIsInstance(llm, RotatingLlmClient)
        self.assertEqual(llm.invoke(["hi"]), "ok:good-key:model-b")
        self.assertEqual(calls, [("bad-key", "model-a"), ("good-key", "model-b")])

    def test_provider_fallback_works_after_bind_tools(self) -> None:
        calls: list[tuple[str, str, str]] = []

        class FakeBoundLlm:
            def __init__(self, api_key: str, model: str):
                self.api_key = api_key
                self.model = model

            def invoke(self, messages):
                calls.append(("invoke", self.api_key, self.model))
                if self.api_key == "bad-key":
                    raise RuntimeError("403 permission denied")
                return "bound-ok"

        class FakeLlm:
            def __init__(self, **kwargs):
                self.api_key = kwargs["api_key"]
                self.model = kwargs["model"]

            def bind_tools(self, *args, **kwargs):
                calls.append(("bind", self.api_key, self.model))
                return FakeBoundLlm(self.api_key, self.model)

        llm = build_llm_with_proxy_fallback(
            FakeLlm,
            {
                "api_key": "bad-key",
                "model": "model-a",
                "_fallback_kwargs": [
                    {
                        "api_key": "good-key",
                        "model": "model-b",
                    }
                ],
            },
        )

        bound = llm.bind_tools([])

        self.assertEqual(bound.invoke(["hi"]), "bound-ok")
        self.assertEqual(
            calls,
            [
                ("bind", "bad-key", "model-a"),
                ("invoke", "bad-key", "model-a"),
                ("bind", "good-key", "model-b"),
                ("invoke", "good-key", "model-b"),
            ],
        )

    def test_provider_fallback_treats_503_as_retryable(self) -> None:
        calls: list[str] = []

        class FakeLlm:
            def __init__(self, **kwargs):
                self.api_key = kwargs["api_key"]

            def invoke(self, messages):
                calls.append(self.api_key)
                if self.api_key == "bad-key":
                    raise RuntimeError("503 Service temporarily unavailable")
                return "ok"

        llm = build_llm_with_proxy_fallback(
            FakeLlm,
            {
                "api_key": "bad-key",
                "model": "model-a",
                "_fallback_kwargs": [
                    {
                        "api_key": "good-key",
                        "model": "model-b",
                    }
                ],
            },
        )

        self.assertEqual(llm.invoke(["hi"]), "ok")
        self.assertEqual(calls, ["bad-key", "good-key"])


if __name__ == "__main__":
    unittest.main()
