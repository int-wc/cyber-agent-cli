import unittest
from unittest.mock import patch

import httpx

from cyber_agent.config import settings
from cyber_agent.tools.search import (
    PLAYWRIGHT_SEARCH_ENGINES,
    SearchResult,
    _annotate_result_relevance,
    _page_looks_blocked,
    _search_with_single_engine,
    create_search_web_tool,
    enrich_results_with_page_visits,
    parse_duckduckgo_html_results,
    search_with_playwright,
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


class MissingLocator:
    def wait_for(self, **kwargs) -> None:
        raise RuntimeError("missing")

    def click(self, **kwargs) -> None:
        raise RuntimeError("missing")

    def fill(self, *args, **kwargs) -> None:
        raise RuntimeError("missing")

    def type(self, *args, **kwargs) -> None:
        raise RuntimeError("missing")

    def press(self, *args, **kwargs) -> None:
        raise RuntimeError("missing")

    def text_content(self, **kwargs):
        raise RuntimeError("missing")

    def inner_text(self, **kwargs):
        raise RuntimeError("missing")

    def get_attribute(self, *args, **kwargs):
        raise RuntimeError("missing")


class FakeLocator:
    def __init__(
        self,
        *,
        text: str = "",
        attrs: dict[str, str] | None = None,
        actions: list | None = None,
        children: dict[str, "FakeLocatorCollection"] | None = None,
    ):
        self.text = text
        self.attrs = attrs or {}
        self.actions = actions if actions is not None else []
        self.children = children or {}

    def wait_for(self, **kwargs) -> None:
        return None

    def click(self, **kwargs) -> None:
        self.actions.append(("click",))

    def fill(self, value: str, **kwargs) -> None:
        self.actions.append(("fill", value))

    def type(self, value: str, delay: int | None = None, **kwargs) -> None:
        self.actions.append(("type", value, delay))

    def press(self, key: str, **kwargs) -> None:
        self.actions.append(("press", key))

    def text_content(self, **kwargs):
        return self.text

    def inner_text(self, **kwargs):
        return self.text

    def get_attribute(self, name: str, **kwargs):
        return self.attrs.get(name)

    def locator(self, selector: str):
        return self.children.get(selector, FakeLocatorCollection([]))


class FakeLocatorCollection:
    def __init__(self, locators: list[FakeLocator] | None = None) -> None:
        self.locators = locators or []

    @property
    def first(self):
        return self.locators[0] if self.locators else MissingLocator()

    def nth(self, index: int):
        return self.locators[index] if index < len(self.locators) else MissingLocator()

    def count(self) -> int:
        return len(self.locators)


class FakeMouse:
    def __init__(self, actions: list[tuple]) -> None:
        self.actions = actions

    def wheel(self, delta_x: int, delta_y: int) -> None:
        self.actions.append(("wheel", delta_x, delta_y))


class FakeSearchPage:
    def __init__(self, *, engine_name: str = "bing", body_text: str = "search body") -> None:
        self.engine_name = engine_name
        self.goto_urls: list[str] = []
        self.waits: list[int] = []
        self.input_actions: list[tuple] = []
        self.load_state_waits: list[str] = []
        self.load_state_calls: list[tuple[str, int | None]] = []
        self.scroll_actions: list[tuple] = []
        self.evaluate_calls: list[str] = []
        self.url = ""
        self._title = engine_name.title()
        self.mouse = FakeMouse(self.scroll_actions)
        self._locator_map = self._build_locator_map(engine_name, body_text)

    def _build_locator_map(self, engine_name: str, body_text: str) -> dict[str, FakeLocatorCollection]:
        input_locator = FakeLocator(actions=self.input_actions)
        locator_map = {
            "body": FakeLocatorCollection([FakeLocator(text=body_text)]),
        }
        if engine_name == "baidu":
            baidu_result_link = FakeLocator(
                text="Example Baidu Result",
                attrs={"href": "https://example.com/baidu"},
            )
            baidu_result_card = FakeLocator(
                text="result card",
                children={
                    "h3 a": FakeLocatorCollection([baidu_result_link]),
                    ".c-abstract": FakeLocatorCollection([FakeLocator(text="Example baidu summary.")]),
                },
            )
            locator_map.update(
                {
                    "textarea[name='wd']": FakeLocatorCollection([input_locator]),
                    "input[name='wd']": FakeLocatorCollection([input_locator]),
                    "#content_left": FakeLocatorCollection([FakeLocator(text="results ready")]),
                    "#content_left > div.result, #content_left > div.result-op, div.result": FakeLocatorCollection(
                        [baidu_result_card]
                    ),
                    "#content_left > div.result h3 a, #content_left > div.result-op h3 a, div.result h3 a": FakeLocatorCollection(
                        [baidu_result_link]
                    ),
                    "#content_left > div.result .c-abstract, #content_left > div.result-op .c-abstract": (
                        FakeLocatorCollection([FakeLocator(text="Example baidu summary.")])
                    ),
                    "div.result .c-abstract": (
                        FakeLocatorCollection([FakeLocator(text="Example baidu summary.")])
                    ),
                }
            )
            return locator_map

        bing_result_link = FakeLocator(
            text="Example Result",
            attrs={"href": "https://example.com/article"},
        )
        bing_result_card = FakeLocator(
            text="result card",
            children={
                "h2 a": FakeLocatorCollection([bing_result_link]),
                ".b_caption p": FakeLocatorCollection([FakeLocator(text="Example summary.")]),
            },
        )
        locator_map.update(
            {
                "textarea[name='q']": FakeLocatorCollection([input_locator]),
                "input[name='q']": FakeLocatorCollection([input_locator]),
                "#b_results": FakeLocatorCollection([FakeLocator(text="results ready")]),
                "li.b_algo h2 a": FakeLocatorCollection([bing_result_link]),
                "li.b_algo .b_caption p": FakeLocatorCollection(
                    [
                        FakeLocator(text="Example summary."),
                    ]
                ),
                "li.b_algo": FakeLocatorCollection([bing_result_card]),
            }
        )
        return locator_map

    def route(self, pattern: str, handler) -> None:
        return None  # no-op for speed optimization tests

    def goto(self, url: str, **kwargs) -> None:
        self.goto_urls.append(url)
        self.url = url

    def wait_for_timeout(self, milliseconds: int) -> None:
        self.waits.append(milliseconds)

    def wait_for_load_state(self, state: str, **kwargs) -> None:
        self.load_state_waits.append(state)
        self.load_state_calls.append((state, kwargs.get("timeout")))

    def evaluate(self, script: str):
        self.evaluate_calls.append(script)
        # 返回模拟的 JS 提取结果
        if "li.b_algo" in script and "querySelectorAll" in script:
            results = []
            link = self._locator_map.get(
                "li.b_algo h2 a",
                FakeLocatorCollection([FakeLocator(text="Example Result", attrs={"href": "https://example.com/article"})])
            ).first
            if hasattr(link, 'text') and hasattr(link, 'attrs'):
                results.append({"title": link.text, "url": link.attrs.get("href", ""), "snippet": "Example summary."})
            return results
        return None

    def title(self) -> str:
        return self._title

    def locator(self, selector: str):
        return self._locator_map.get(selector, FakeLocatorCollection([]))

    def close(self) -> None:
        return None


class FakeBrowserPage:
    def __init__(
        self,
        *,
        title: str = "OpenAI Agent Guide",
        description: str = "OpenAI agent usage guide.",
        body_text: str = "This page explains how the OpenAI agent works in detail.",
    ) -> None:
        self.goto_urls: list[str] = []
        self.waits: list[int] = []
        self.load_state_waits: list[str] = []
        self.load_state_calls: list[tuple[str, int | None]] = []
        self.scroll_actions: list[tuple] = []
        self.url = ""
        self._title = title
        self.mouse = FakeMouse(self.scroll_actions)
        self._locator_map = {
            "meta[name='description']": FakeLocatorCollection([FakeLocator(attrs={"content": description})]),
            "meta[property='og:description']": FakeLocatorCollection([]),
            "main": FakeLocatorCollection([FakeLocator(text=body_text)]),
            "article": FakeLocatorCollection([]),
            "[role='main']": FakeLocatorCollection([]),
            "main p": FakeLocatorCollection([FakeLocator(text=body_text)]),
            "article p": FakeLocatorCollection([]),
            "p": FakeLocatorCollection([FakeLocator(text=body_text)]),
            "body": FakeLocatorCollection([FakeLocator(text=body_text)]),
        }

    def goto(self, url: str, **kwargs) -> None:
        self.goto_urls.append(url)
        self.url = url

    def wait_for_timeout(self, milliseconds: int) -> None:
        self.waits.append(milliseconds)

    def wait_for_load_state(self, state: str, **kwargs) -> None:
        self.load_state_waits.append(state)
        self.load_state_calls.append((state, kwargs.get("timeout")))

    def evaluate(self, script: str):
        return None

    def title(self) -> str:
        return self._title

    def locator(self, selector: str):
        return self._locator_map.get(selector, FakeLocatorCollection([]))

    def close(self) -> None:
        return None


class FakeBrowserContext:
    def __init__(self, page_factory=None) -> None:
        self.new_page_count = 0
        self.page_factory = page_factory or (lambda: FakeBrowserPage())
        self.created_pages: list[FakeBrowserPage] = []
        self.init_scripts: list[str] = []

    def new_page(self):
        self.new_page_count += 1
        page = self.page_factory()
        self.created_pages.append(page)
        return page

    def add_init_script(self, script: str) -> None:
        self.init_scripts.append(script)

    def close(self) -> None:
        return None


class FakeBrowser:
    def __init__(self, context: FakeBrowserContext | None = None) -> None:
        self.context_kwargs: dict | None = None
        self.context = context or FakeBrowserContext()

    def new_context(self, **kwargs):
        self.context_kwargs = kwargs
        return self.context

    def close(self) -> None:
        return None


class FakeChromium:
    def __init__(self) -> None:
        self.launch_kwargs: dict | None = None
        self.browser = FakeBrowser()

    def launch(self, **kwargs):
        self.launch_kwargs = {k: v for k, v in kwargs.items() if k != "args"}
        return self.browser


class FakePlaywrightManager:
    def __init__(self, chromium: FakeChromium) -> None:
        self.chromium = chromium

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class FakeCapabilityRegistry:
    def __init__(self, response: dict | Exception) -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def invoke_json_prompt(self, system_prompt: str, user_prompt: str) -> dict:
        self.calls.append((system_prompt, user_prompt))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


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

    def test_single_engine_search_uses_homepage_and_types_query(self) -> None:
        """
        测试：当搜索引擎未配置 search_url_template 时，应回退到首页模拟输入流程。
        """
        from cyber_agent.tools.search import SearchEngineSpec

        page = FakeSearchPage()
        # 构造无快速通道的引擎规格，确保走首页输入流程
        engine_spec = SearchEngineSpec(
            name="bing",
            homepage_url="https://www.bing.com/",
            search_input_selectors=("textarea[name='q']", "input[name='q']"),
            result_ready_selectors=("#b_results", "li.b_algo"),
            result_selector="li.b_algo",
            link_selector="li.b_algo h2 a",
            title_selector="li.b_algo h2 a",
            snippet_selectors=("li.b_algo .b_caption p", "li.b_algo .b_snippet"),
            card_link_selectors=("h2 a",),
            card_title_selectors=("h2 a",),
            card_snippet_selectors=(".b_caption p", ".b_snippet"),
            search_button_selectors=("button#search_icon", "input#sb_form_go"),
        )

        results, note = _search_with_single_engine(
            page,
            engine_spec,
            "openai agent",
            3,
            None,
        )

        self.assertIsNone(note)
        self.assertEqual(page.goto_urls, ["https://www.bing.com/"])
        self.assertIn(("fill", "openai agent"), page.input_actions)
        self.assertIn(("press", "Enter"), page.input_actions)
        self.assertGreaterEqual(len(page.scroll_actions), 1)
        self.assertEqual(results[0].title, "Example Result")
        self.assertEqual(results[0].url, "https://example.com/article")
        self.assertTrue(results[0].relevance_summary)

    def test_page_looks_blocked_detects_body_verification_text(self) -> None:
        """
        测试：即使标题未命中，只要正文出现人机验证提示，也会识别为拦截页。
        """
        page = FakeSearchPage(body_text="Verify you are human to continue")
        engine_spec = PLAYWRIGHT_SEARCH_ENGINES[0]

        self.assertTrue(_page_looks_blocked(page, engine_spec))

    def test_single_engine_search_prefers_card_local_extraction(self) -> None:
        """
        测试：当全局选择器缺失时，仍可从结果卡片内部抽取标题、链接和摘要。
        """
        page = FakeSearchPage()
        page._locator_map["li.b_algo h2 a"] = FakeLocatorCollection([])
        page._locator_map["li.b_algo .b_caption p"] = FakeLocatorCollection([])
        engine_spec = PLAYWRIGHT_SEARCH_ENGINES[0]

        results, note = _search_with_single_engine(
            page,
            engine_spec,
            "openai agent",
            3,
            None,
        )

        self.assertIsNone(note)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Example Result")
        self.assertEqual(results[0].url, "https://example.com/article")
        self.assertEqual(results[0].snippet, "Example summary.")

    def test_baidu_search_waits_for_full_load_before_extracting_results(self) -> None:
        """
        测试：Baidu 搜索会在结果页执行自动滚动并提取结果。
        """
        page = FakeSearchPage(engine_name="baidu")
        engine_spec = next(engine for engine in PLAYWRIGHT_SEARCH_ENGINES if engine.name == "baidu")

        results, note = _search_with_single_engine(
            page,
            engine_spec,
            "openai agent",
            3,
            None,
        )

        self.assertIsNone(note)
        self.assertGreaterEqual(len(page.scroll_actions), 1)
        self.assertEqual(results[0].url, "https://example.com/baidu")

    def test_annotate_result_relevance_handles_chinese_query_terms(self) -> None:
        """
        测试：中文查询即使没有空格，也能识别标题中的相关词片段。
        """
        result = SearchResult(
            title="网络与安全实践",
            url="https://example.com/security",
            snippet="介绍攻防和加固方法。",
        )

        _annotate_result_relevance("网络安全", result)

        self.assertGreater(result.relevance_score, 0)
        self.assertIn(result.relevance_summary, {"高度相关", "相关", "弱相关，建议人工复核"})

    def test_search_with_playwright_respects_visible_browser_switch(self) -> None:
        """
        测试：配置打开显示浏览器时，应以非无头模式启动 Playwright。
        """
        chromium = FakeChromium()
        manager = FakePlaywrightManager(chromium)

        with patch("cyber_agent.tools.search.PLAYWRIGHT_AVAILABLE", True), patch(
            "cyber_agent.tools.search.sync_playwright",
            return_value=manager,
        ), patch(
            "cyber_agent.tools.search.PLAYWRIGHT_SEARCH_ENGINES",
            (PLAYWRIGHT_SEARCH_ENGINES[0],),
        ), patch(
            "cyber_agent.tools.search.enrich_results_with_page_visits",
            return_value=[],
        ), patch.object(settings, "search_show_browser", True):
            results, notes = search_with_playwright("example", 3)

        self.assertIn("浏览器模式：可见窗口", notes)
        self.assertEqual(chromium.launch_kwargs, {"headless": False})

    def test_search_with_playwright_ranks_by_rule_relevance_when_no_registry(self) -> None:
        """
        测试：不提供模型能力注册器时，仅使用规则评分排序搜索结果。
        """
        chromium = FakeChromium()
        chromium.browser = FakeBrowser(
            context=FakeBrowserContext(page_factory=FakeSearchPage)
        )
        manager = FakePlaywrightManager(chromium)

        with patch("cyber_agent.tools.search.PLAYWRIGHT_AVAILABLE", True), patch(
            "cyber_agent.tools.search.sync_playwright",
            return_value=manager,
        ), patch(
            "cyber_agent.tools.search.PLAYWRIGHT_SEARCH_ENGINES",
            (PLAYWRIGHT_SEARCH_ENGINES[0],),
        ):
            results, notes = search_with_playwright("openai agent", 1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].relevance_source, "rule")
        self.assertTrue(results[0].relevance_summary)

    def test_search_with_playwright_returns_rule_ranked_results(self) -> None:
        """
        测试：搜索返回基于规则相关性评分的排序结果。
        """
        chromium = FakeChromium()
        chromium.browser = FakeBrowser(
            context=FakeBrowserContext(page_factory=FakeSearchPage)
        )
        manager = FakePlaywrightManager(chromium)

        with patch("cyber_agent.tools.search.PLAYWRIGHT_AVAILABLE", True), patch(
            "cyber_agent.tools.search.sync_playwright",
            return_value=manager,
        ), patch(
            "cyber_agent.tools.search.PLAYWRIGHT_SEARCH_ENGINES",
            (PLAYWRIGHT_SEARCH_ENGINES[0],),
        ):
            results, notes = search_with_playwright(
                "openai agent",
                3,
            )

        self.assertGreaterEqual(len(results), 1)
        self.assertTrue(all(r.relevance_source == "rule" for r in results))
        self.assertTrue(all(r.relevance_summary for r in results))

    def test_enrich_results_with_page_visits_marks_page_relevance(self) -> None:
        """
        测试：访问真实页面后，会补充标题摘要并给出页面相关性判断。
        """
        result = SearchResult(
            title="Old Title",
            url="https://example.com/agent",
            snippet="Old snippet",
            source_engine="bing",
        )
        visit_page = FakeBrowserPage(
            title="OpenAI Agent Guide",
            description="OpenAI agent usage guide.",
            body_text="This page explains how the OpenAI agent works in detail.",
        )
        browser_context = FakeBrowserContext(page_factory=lambda: visit_page)

        notes = enrich_results_with_page_visits(
            browser_context,
            "openai agent",
            [result],
            None,
            1,
        )

        self.assertEqual(notes, [])
        self.assertTrue(result.visited)
        self.assertEqual(result.title, "OpenAI Agent Guide")
        self.assertEqual(result.snippet, "OpenAI agent usage guide.")
        self.assertIn(result.relevance_summary, {"高度相关", "相关"})
        self.assertGreaterEqual(len(visit_page.scroll_actions), 1)

    def test_search_web_tool_formats_ranked_results(self) -> None:
        """
        测试：搜索工具会返回排序后的文本结果。
        """
        with patch("cyber_agent.tools.search.httpx.Client", FakeHttpxClient), patch(
            "cyber_agent.tools.search.search_with_playwright",
            return_value=([], ["当前环境未安装 Playwright，已跳过浏览器搜索。"]),
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
        测试：首个 HTTP 端点失败时，会继续尝试备用端点。
        """
        with patch("cyber_agent.tools.search.httpx.Client", FallbackHttpxClient), patch(
            "cyber_agent.tools.search.search_with_playwright",
            return_value=([], ["当前环境未安装 Playwright，已跳过浏览器搜索。"]),
        ):
            search_tool = create_search_web_tool()
            result = search_tool.invoke({"query": "example", "max_results": 2})

        self.assertIn("查询: example", result)
        self.assertIn("1. Example Doc", result)

    def test_search_web_tool_reports_network_unavailable_clearly(self) -> None:
        """
        测试：所有 HTTP 端点都不可用时，应明确提示外网不可用与浏览器搜索回退原因。
        """
        with patch("cyber_agent.tools.search.httpx.Client", AlwaysFailHttpxClient), patch(
            "cyber_agent.tools.search.search_with_playwright",
            return_value=([], ["当前环境未安装 Playwright，已跳过浏览器搜索。"]),
        ):
            search_tool = create_search_web_tool()
            result = search_tool.invoke({"query": "example"})

        self.assertIn("当前运行环境可能无法访问外部搜索服务", result)
        self.assertIn("当前环境未安装 Playwright", result)
        self.assertIn("https://html.duckduckgo.com/html/", result)
        self.assertIn("https://duckduckgo.com/html/", result)

    def test_search_web_tool_prefers_playwright_results_when_available(self) -> None:
        """
        测试：若浏览器搜索成功，应优先返回浏览器搜索与访问后的结果。
        """
        browser_results = [
            SearchResult(
                title="Example Browser Result",
                url="https://example.com/browser",
                snippet="Visited summary from real page.",
                source_engine="bing",
                visited=True,
                relevance_summary="高度相关",
                relevance_source="model",
                relevance_reason="页面内容直接回答了查询问题。",
            )
        ]

        with patch(
            "cyber_agent.tools.search.search_with_playwright",
            return_value=(browser_results, ["浏览器模式：可见窗口"]),
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
        self.assertIn("页面判断: 高度相关（模型）", result)
        self.assertIn("判定依据: 页面内容直接回答了查询问题。", result)


class FuckCSDNRegressionTestCase(unittest.TestCase):
    """FuckCSDN 回归测试 —— 验证 CSDN 剔除逻辑和搜索策略完整性（无需网络）。"""

    def test_csdn_domains_blacklist_has_minimum_coverage(self) -> None:
        """测试：CSDN 黑名单至少覆盖 10 个子域名。"""
        from cyber_agent.tools.search import CSDN_DOMAINS
        self.assertGreaterEqual(len(CSDN_DOMAINS), 10)

    def test_is_csdn_url_detects_all_subdomains(self) -> None:
        """测试：_is_csdn_url 正确检测所有 CSDN 子域名变体。"""
        from cyber_agent.tools.search import _is_csdn_url

        csdn_urls = [
            "https://blog.csdn.net/user/article/123",
            "https://www.csdn.net/article/456",
            "https://download.csdn.net/file/789",
            "https://edu.csdn.net/course/10",
            "https://bbs.csdn.net/topics/20",
            "https://gitcode.csdn.net/repo/30",
            "https://ask.csdn.net/questions/60",
            "https://dev.csdn.net/developer/70",
            "https://live.csdn.net/room/50",
            "https://bi.csdn.net/report/80",
            "https://inscode.csdn.net/project/40",
            "https://spider.csdn.net/data/90",
        ]
        for url in csdn_urls:
            self.assertTrue(_is_csdn_url(url), f"应检测为 CSDN: {url}")

    def test_is_csdn_url_does_not_false_positive(self) -> None:
        """测试：_is_csdn_url 不会误杀正常网站。"""
        from cyber_agent.tools.search import _is_csdn_url

        non_csdn_urls = [
            "https://github.com/user/repo",
            "https://zhuanlan.zhihu.com/p/123",
            "https://www.freebuf.com/article/1",
            "https://en.wikipedia.org/wiki/Main_Page",
            "https://stackoverflow.com/questions/1",
            "https://www.ithome.com/news/1",
            "https://www.secrss.com/article/1",
            "https://cloud.tencent.com/article/1",
            "https://news.qq.com/article/1",
            "https://tech.ifeng.com/article/1",
            "https://www.reddit.com/r/programming",
            "https://arxiv.org/abs/2501.00001",
            "https://medium.com/article/1",
            "https://www.nature.com/article/1",
            "https://www.bbc.com/news/article",
        ]
        for url in non_csdn_urls:
            self.assertFalse(_is_csdn_url(url), f"不应检测为 CSDN: {url}")

    def test_filter_csdn_results_removes_all_csdn(self) -> None:
        """测试：_filter_csdn_results 100% 剔除 CSDN 结果。"""
        from cyber_agent.tools.search import _filter_csdn_results, _is_csdn_url, SearchResult

        results = [
            SearchResult(title="t1", url="https://blog.csdn.net/1", snippet="", source_engine="test"),
            SearchResult(title="t2", url="https://github.com/a", snippet="", source_engine="test"),
            SearchResult(title="t3", url="https://www.csdn.net/2", snippet="", source_engine="test"),
            SearchResult(title="t4", url="https://stackoverflow.com/q/1", snippet="", source_engine="test"),
            SearchResult(title="t5", url="https://download.csdn.net/3", snippet="", source_engine="test"),
        ]
        filtered, removed = _filter_csdn_results(results)
        self.assertEqual(removed, 3)
        self.assertEqual(len(filtered), 2)
        csdn_residual = sum(1 for r in filtered if _is_csdn_url(r.url))
        self.assertEqual(csdn_residual, 0, f"CSDN 残留 {csdn_residual} 条")

    def test_filter_csdn_results_preserves_non_csdn(self) -> None:
        """测试：_filter_csdn_results 保留所有非 CSDN 结果的原始数据。"""
        from cyber_agent.tools.search import _filter_csdn_results, SearchResult

        original = SearchResult(
            title="My Result", url="https://example.com/page",
            snippet="A great page", source_engine="bing",
        )
        filtered, _ = _filter_csdn_results([original])
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].title, "My Result")
        self.assertEqual(filtered[0].url, "https://example.com/page")
        self.assertEqual(filtered[0].snippet, "A great page")

    def test_build_query_variants_generates_diverse_results(self) -> None:
        """测试：查询变体生成至少 5 个不重复变体。"""
        from cyber_agent.tools.search import _build_query_variants

        variants = _build_query_variants("Python programming tutorial guide beginner 2026", 30)
        self.assertGreaterEqual(len(variants), 5)
        self.assertLessEqual(len(variants), 10)
        # 验证所有变体唯一（忽略大小写）
        lower_variants = [v.lower() for v in variants]
        self.assertEqual(len(lower_variants), len(set(lower_variants)),
                         f"变体不唯一: {variants}")

    def test_build_query_variants_includes_keyword_pairs(self) -> None:
        """测试：变体生成包含关键词配对（不同 2-3 词组合）。"""
        from cyber_agent.tools.search import _build_query_variants

        variants = _build_query_variants("Python programming tutorial guide beginner", 30)
        # 应有 "Python programming", "Python tutorial" 等核心词配对
        has_pair = any("Python" in v and len(v.split()) <= 3 and v != "Python" for v in variants)
        self.assertTrue(has_pair, f"变体中没有核心词配对: {variants}")

    def test_search_constants_in_range(self) -> None:
        """测试：搜索常量在合理范围内。"""
        from cyber_agent.tools.search import (
            SEARCH_MIN_RESULTS, SEARCH_MAX_RESULTS, SEARCH_TIME_BUDGET_SECONDS,
            FETCH_MULTIPLIER_FOR_CSDN_FILTER,
        )
        self.assertEqual(SEARCH_MIN_RESULTS, 20)
        self.assertEqual(SEARCH_MAX_RESULTS, 40)
        self.assertGreaterEqual(SEARCH_TIME_BUDGET_SECONDS, 5.5)
        self.assertLessEqual(SEARCH_TIME_BUDGET_SECONDS, 7.0)
        self.assertGreaterEqual(FETCH_MULTIPLIER_FOR_CSDN_FILTER, 2.0)

    def test_search_strategy_has_all_fallbacks(self) -> None:
        """测试：搜索策略包含所有回退层。"""
        import inspect
        from cyber_agent.tools.search import _search_bing_multiquery
        src = inspect.getsource(_search_bing_multiquery)
        self.assertIn("baidu.com", src, "缺少百度回退")
        self.assertIn("site:github", src, "缺少站点限定回退")
        self.assertIn("page_offset", src, "缺少分页回退")

    def test_tool_factory_creates_search_web(self) -> None:
        """测试：工具工厂创建正确的 search_web 工具。"""
        from cyber_agent.tools.search import create_search_web_tool
        tool = create_search_web_tool()
        self.assertEqual(tool.name, "search_web")
        self.assertTrue(hasattr(tool, "args_schema"))

    def test_csdn_filter_in_search_pipeline(self) -> None:
        """测试：search_with_playwright 包含 CSDN 过滤。"""
        import inspect
        from cyber_agent.tools.search import search_with_playwright
        src = inspect.getsource(search_with_playwright)
        self.assertIn("_filter_csdn_results", src, "search_with_playwright 未调用 CSDN 过滤")

    def test_monte_carlo_csdn_free_simulation(self) -> None:
        """蒙特卡洛测试：随机 500 次，验证 CSDN 过滤 100% 有效。"""
        import random
        from cyber_agent.tools.search import CSDN_DOMAINS, _filter_csdn_results, _is_csdn_url, SearchResult

        csdn_subs = list(CSDN_DOMAINS - {"csdn.net"})
        for _ in range(500):
            n_csdn = random.randint(0, 15)
            n_other = random.randint(10, 50)
            results = []
            for i in range(n_csdn):
                sub = random.choice(csdn_subs)
                results.append(SearchResult(
                    title=f"c{i}", url=f"https://{sub}/p/{i}", snippet="", source_engine="m",
                ))
            for i in range(n_other):
                results.append(SearchResult(
                    title=f"ok{i}", url=f"https://site{i}.com/p", snippet="", source_engine="m",
                ))
            random.shuffle(results)
            filtered, _ = _filter_csdn_results(results)
            csdn_left = sum(1 for r in filtered if _is_csdn_url(r.url))
            self.assertEqual(csdn_left, 0, f"CSDN 残留 {csdn_left} 条")


if __name__ == "__main__":
    unittest.main()
