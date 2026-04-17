import unittest
from unittest.mock import patch

import httpx

from cyber_agent.tools.search import (
    SearchResult,
    create_search_web_tool,
    parse_duckduckgo_html_results,
)


SAMPLE_SEARCH_HTML = """
<html>
  <body>
    <a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdoc">
      Example Doc
    </a>
    <a class="result__snippet">Example summary line.</a>
    <a class="result__a" href="https://example.org/news">Example News</a>
    <div class="result__snippet">Latest example news.</div>
  </body>
</html>
"""


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


class FakeHttpxClient:
    def __init__(self, *args, **kwargs) -> None:
        self.calls: list[tuple[str, dict[str, str]]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def get(self, url: str, params: dict[str, str]):
        self.calls.append((url, params))
        return FakeResponse(SAMPLE_SEARCH_HTML)


class FallbackHttpxClient:
    def __init__(self, *args, **kwargs) -> None:
        self.calls: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def get(self, url: str, params: dict[str, str]):
        self.calls.append(url)
        if url == "https://html.duckduckgo.com/html/":
            raise httpx.ConnectError(
                "[Errno 101] Network is unreachable",
                request=httpx.Request("GET", url),
            )
        return FakeResponse(SAMPLE_SEARCH_HTML)


class AlwaysFailHttpxClient:
    def __init__(self, *args, **kwargs) -> None:
        self.calls: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def get(self, url: str, params: dict[str, str]):
        self.calls.append(url)
        raise httpx.ConnectError(
            "[Errno 101] Network is unreachable",
            request=httpx.Request("GET", url),
        )


class SearchToolTestCase(unittest.TestCase):
    def test_parse_duckduckgo_html_results_extracts_title_url_and_snippet(self) -> None:
        """
        测试：DuckDuckGo HTML 结果页会被解析为标题、链接和摘要。
        """
        results = parse_duckduckgo_html_results(SAMPLE_SEARCH_HTML)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].title, "Example Doc")
        self.assertEqual(results[0].url, "https://example.com/doc")
        self.assertEqual(results[0].snippet, "Example summary line.")
        self.assertEqual(results[1].title, "Example News")
        self.assertEqual(results[1].url, "https://example.org/news")

    def test_search_web_tool_formats_ranked_results(self) -> None:
        """
        测试：搜索工具会返回类似通用搜索引擎的排序结果文本。
        """
        with patch("cyber_agent.tools.search.PLAYWRIGHT_AVAILABLE", False), patch(
            "cyber_agent.tools.search.httpx.Client",
            FakeHttpxClient,
        ):
            search_tool = create_search_web_tool()
            result = search_tool.invoke({"query": "example", "max_results": 2})

        self.assertIn("查询: example", result)
        self.assertIn("1. Example Doc", result)
        self.assertIn("2. Example News", result)
        self.assertIn("链接: https://example.com/doc", result)
        self.assertIn("摘要: Latest example news.", result)

    def test_search_web_tool_can_fallback_to_secondary_endpoint(self) -> None:
        """
        测试：首个搜索端点连接失败时，会自动尝试内置兜底端点。
        """
        with patch("cyber_agent.tools.search.PLAYWRIGHT_AVAILABLE", False), patch(
            "cyber_agent.tools.search.httpx.Client",
            FallbackHttpxClient,
        ):
            search_tool = create_search_web_tool()
            result = search_tool.invoke({"query": "example", "max_results": 2})

        self.assertIn("查询: example", result)
        self.assertIn("1. Example Doc", result)

    def test_search_web_tool_reports_network_unavailable_clearly(self) -> None:
        """
        测试：当所有端点都无法访问时，应明确提示外网不可用而不是只返回模糊异常。
        """
        with patch("cyber_agent.tools.search.PLAYWRIGHT_AVAILABLE", False), patch(
            "cyber_agent.tools.search.httpx.Client",
            AlwaysFailHttpxClient,
        ):
            search_tool = create_search_web_tool()
            result = search_tool.invoke({"query": "example"})

        self.assertIn("当前运行环境可能无法访问外部搜索服务", result)
        self.assertIn("Playwright unavailable", result)
        self.assertIn("https://html.duckduckgo.com/html/", result)
        self.assertIn("https://duckduckgo.com/html/", result)

    def test_search_web_tool_prefers_playwright_results_when_available(self) -> None:
        """
        测试：若 Playwright 搜索成功，应优先返回浏览器搜索和访问后的结果。
        """
        browser_results = [
            SearchResult(
                title="Example Browser Result",
                url="https://example.com/browser",
                snippet="Visited summary from real page.",
                source_engine="bing",
                visited=True,
            )
        ]

        with patch("cyber_agent.tools.search.PLAYWRIGHT_AVAILABLE", True), patch(
            "cyber_agent.tools.search.search_with_playwright",
            return_value=(browser_results, ["bing 返回 3 条候选结果。"]),
        ) as mock_browser_search, patch(
            "cyber_agent.tools.search.search_with_httpx",
        ) as mock_http_search:
            search_tool = create_search_web_tool()
            result = search_tool.invoke({"query": "example", "max_results": 2})

        self.assertEqual(mock_browser_search.call_count, 1)
        self.assertEqual(mock_http_search.call_count, 0)
        self.assertIn("Example Browser Result", result)
        self.assertIn("来源: bing", result)
        self.assertIn("已访问: 是", result)

    def test_search_web_tool_falls_back_to_http_when_playwright_returns_no_results(self) -> None:
        """
        测试：若浏览器搜索没有拿到可用结果，会自动回退到 HTTP 搜索。
        """
        http_results = [
            SearchResult(
                title="Fallback Result",
                url="https://example.com/fallback",
                snippet="Fallback summary.",
            )
        ]

        with patch("cyber_agent.tools.search.PLAYWRIGHT_AVAILABLE", True), patch(
            "cyber_agent.tools.search.search_with_playwright",
            return_value=([], ["google 未解析到可用结果。"]),
        ), patch(
            "cyber_agent.tools.search.search_with_httpx",
            return_value=(http_results, ["已回退到 HTTP 搜索。"], []),
        ):
            search_tool = create_search_web_tool()
            result = search_tool.invoke({"query": "example", "max_results": 1})

        self.assertIn("Fallback Result", result)
        self.assertIn("google 未解析到可用结果。", result)
        self.assertIn("已回退到 HTTP 搜索。", result)


if __name__ == "__main__":
    unittest.main()
