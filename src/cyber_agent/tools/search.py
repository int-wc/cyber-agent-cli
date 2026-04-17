from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, quote_plus, urljoin, urlparse

import httpx
from langchain_core.tools import tool

from ..config import settings
from ..execution_control import ExecutionController, ExecutionInterruptedError
from .metadata import attach_tool_risk

try:
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover - 依赖缺失时走 HTTP 回退路径
    PlaywrightError = RuntimeError
    sync_playwright = None
    PLAYWRIGHT_AVAILABLE = False

DEFAULT_SEARCH_ENDPOINT = "https://html.duckduckgo.com/html/"
FALLBACK_SEARCH_ENDPOINTS = (
    DEFAULT_SEARCH_ENDPOINT,
    "https://duckduckgo.com/html/",
)
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)
MAX_SEARCH_QUERY_LENGTH = 300
PLAYWRIGHT_SEARCH_RESULT_MULTIPLIER = 3
PLAYWRIGHT_VISIT_RESULT_LIMIT = 2
PLAYWRIGHT_WAIT_MILLISECONDS = 800
PLAYWRIGHT_VISIT_WAIT_MILLISECONDS = 200
PLAYWRIGHT_SEARCH_TIMEOUT_MILLISECONDS = 4000
PLAYWRIGHT_VISIT_TIMEOUT_MILLISECONDS = 4000


@dataclass(slots=True)
class SearchResult:
    """描述单条搜索结果。"""

    title: str
    url: str
    snippet: str
    source_engine: str = ""
    visited: bool = False
    visit_summary: str = ""


@dataclass(frozen=True, slots=True)
class SearchEngineSpec:
    """描述单个搜索引擎的入口地址与解析规则。"""

    name: str
    search_url_template: str
    result_selector: str
    link_selector: str
    title_selector: str
    snippet_selectors: tuple[str, ...]
    blocked_title_markers: tuple[str, ...] = ()
    blocked_url_markers: tuple[str, ...] = ()

    def build_search_url(self, query: str) -> str:
        """根据查询词拼接实际搜索 URL。"""
        return self.search_url_template.format(query=quote_plus(query))


PLAYWRIGHT_SEARCH_ENGINES = (
    SearchEngineSpec(
        name="bing",
        search_url_template="https://www.bing.com/search?q={query}",
        result_selector="li.b_algo",
        link_selector="li.b_algo h2 a",
        title_selector="li.b_algo h2 a",
        snippet_selectors=("li.b_algo .b_caption p", "li.b_algo .b_snippet"),
    ),
    SearchEngineSpec(
        name="google",
        search_url_template="https://www.google.com/search?q={query}",
        result_selector="div.g",
        link_selector="div.g a[href]:has(h3)",
        title_selector="div.g a[href]:has(h3) h3",
        snippet_selectors=("div.g div.VwiC3b", "div.g span.aCOpRe", "div.g div[data-sncf='1']"),
        blocked_title_markers=("https://www.google.com/search?", "unusual traffic"),
        blocked_url_markers=("/sorry/",),
    ),
    SearchEngineSpec(
        name="baidu",
        search_url_template="https://www.baidu.com/s?wd={query}",
        result_selector="#content_left > div.result, #content_left > div.result-op",
        link_selector="#content_left > div.result h3 a, #content_left > div.result-op h3 a",
        title_selector="#content_left > div.result h3 a, #content_left > div.result-op h3 a",
        snippet_selectors=(
            "#content_left > div.result .c-abstract, #content_left > div.result-op .c-abstract",
            "#content_left > div.result .content-right_8Zs40, #content_left > div.result-op .content-right_8Zs40",
            "#content_left > div.result .c-span-last, #content_left > div.result-op .c-span-last",
        ),
        blocked_title_markers=("百度安全验证", "安全验证"),
    ),
)


def _normalize_whitespace(text: str) -> str:
    """压缩 HTML 解析后的多余空白字符。"""
    return re.sub(r"\s+", " ", unescape(text)).strip()


def _unwrap_duckduckgo_redirect(raw_url: str) -> str:
    """将 DuckDuckGo 的跳转链接还原为真实目标地址。"""
    parsed_url = urlparse(raw_url)
    if "duckduckgo.com" not in parsed_url.netloc and not raw_url.startswith("/l/"):
        return raw_url

    query_parameters = parse_qs(parsed_url.query)
    unwrapped_urls = query_parameters.get("uddg")
    if not unwrapped_urls:
        return raw_url
    return unwrapped_urls[0]


def _unwrap_search_result_url(raw_url: str, base_url: str) -> str:
    """统一还原不同搜索引擎的真实结果链接。"""
    if not raw_url:
        return ""

    absolute_url = urljoin(base_url, raw_url)
    parsed_url = urlparse(absolute_url)
    if absolute_url.startswith("https://www.google.com/url?") or raw_url.startswith("/url?"):
        query_parameters = parse_qs(parsed_url.query)
        google_urls = query_parameters.get("q")
        if google_urls:
            return google_urls[0]
    return _unwrap_duckduckgo_redirect(absolute_url)


class DuckDuckGoHtmlParser(HTMLParser):
    """解析 DuckDuckGo HTML 搜索页面中的标题、链接与摘要。"""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[SearchResult] = []
        self._current_title_href = ""
        self._current_title_chunks: list[str] = []
        self._collecting_title = False
        self._collecting_snippet = False
        self._current_snippet_chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attribute_map = {key: (value or "") for key, value in attrs}
        class_names = set(attribute_map.get("class", "").split())
        href = attribute_map.get("href", "")

        if tag == "a" and {"result__a", "result-link"} & class_names and href:
            self._collecting_title = True
            self._current_title_href = _unwrap_duckduckgo_redirect(href)
            self._current_title_chunks = []
            return

        if (
            tag in {"a", "div", "span"}
            and {"result__snippet", "result-snippet"} & class_names
            and self.results
        ):
            self._collecting_snippet = True
            self._current_snippet_chunks = []

    def handle_data(self, data: str) -> None:
        if self._collecting_title:
            self._current_title_chunks.append(data)
        if self._collecting_snippet:
            self._current_snippet_chunks.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._collecting_title:
            title = _normalize_whitespace("".join(self._current_title_chunks))
            if title and self._current_title_href:
                self.results.append(
                    SearchResult(
                        title=title,
                        url=self._current_title_href,
                        snippet="",
                    )
                )
            self._collecting_title = False
            self._current_title_chunks = []
            self._current_title_href = ""
            return

        if tag in {"a", "div", "span"} and self._collecting_snippet:
            snippet = _normalize_whitespace("".join(self._current_snippet_chunks))
            if snippet and self.results:
                latest_result = self.results[-1]
                latest_result.snippet = snippet
            self._collecting_snippet = False
            self._current_snippet_chunks = []


def parse_duckduckgo_html_results(html_text: str) -> list[SearchResult]:
    """将 DuckDuckGo HTML 页面解析为结果列表。"""
    parser = DuckDuckGoHtmlParser()
    parser.feed(html_text)
    return [result for result in parser.results if result.title and result.url]


def _extract_locator_text(locator: Any) -> str:
    """从 Playwright 定位器中安全提取文本。"""
    try:
        text_content = locator.text_content(timeout=2000)
    except Exception:
        text_content = None
    normalized_text = _normalize_whitespace(str(text_content or ""))
    if normalized_text:
        return normalized_text
    try:
        return _normalize_whitespace(locator.inner_text(timeout=2000))
    except Exception:
        return ""


def _extract_first_non_empty_locator_text(scope: Any, selectors: tuple[str, ...]) -> str:
    """依次尝试多个选择器，返回首个非空文本。"""
    for selector in selectors:
        try:
            locator = scope.locator(selector).first
            text = _extract_locator_text(locator)
        except Exception:
            continue
        if text:
            return text
    return ""


def _extract_locator_attribute(locator: Any, attribute_name: str) -> str:
    """从 Playwright 定位器中安全提取属性，避免默认长超时卡住整轮搜索。"""
    try:
        value = locator.get_attribute(attribute_name, timeout=1500)
    except TypeError:
        try:
            value = locator.get_attribute(attribute_name)
        except Exception:
            return ""
    except Exception:
        return ""
    return str(value or "").strip()


def _extract_page_snippet_by_index(page: Any, selectors: tuple[str, ...], index: int) -> str:
    """在无法稳定定位结果卡片时，按结果序号提取同序号摘要。"""
    for selector in selectors:
        try:
            locator = page.locator(selector).nth(index)
            text = _extract_locator_text(locator)
        except Exception:
            continue
        if text:
            return text
    return ""


def _page_looks_blocked(page: Any, engine_spec: SearchEngineSpec) -> bool:
    """识别安全验证或反爬拦截页面，避免把校验页当搜索结果。"""
    try:
        page_title = _normalize_whitespace(page.title()).lower()
    except Exception:
        page_title = ""
    current_url = str(getattr(page, "url", "")).lower()
    return any(marker.lower() in page_title for marker in engine_spec.blocked_title_markers) or any(
        marker.lower() in current_url for marker in engine_spec.blocked_url_markers
    )


def _search_with_single_engine(
    page: Any,
    engine_spec: SearchEngineSpec,
    query: str,
    result_limit: int,
    execution_controller: ExecutionController | None,
) -> tuple[list[SearchResult], str | None]:
    """使用单个搜索引擎采集候选结果。"""
    if execution_controller is not None:
        execution_controller.ensure_not_cancelled()

    page.goto(
        engine_spec.build_search_url(query),
        wait_until="domcontentloaded",
        timeout=PLAYWRIGHT_SEARCH_TIMEOUT_MILLISECONDS,
    )
    page.wait_for_timeout(PLAYWRIGHT_WAIT_MILLISECONDS)

    if _page_looks_blocked(page, engine_spec):
        blocked_title = _normalize_whitespace(page.title()) or str(getattr(page, "url", ""))
        return [], f"{engine_spec.name} 返回验证或拦截页面：{blocked_title}"

    results: list[SearchResult] = []
    inspect_count = max(result_limit * PLAYWRIGHT_SEARCH_RESULT_MULTIPLIER, result_limit)
    title_locators = page.locator(engine_spec.title_selector)
    title_count = int(title_locators.count())
    if title_count <= 0:
        return [], f"{engine_spec.name} 未解析到可用结果。"

    link_locators = page.locator(engine_spec.link_selector)
    for index in range(min(title_count, inspect_count)):
        if execution_controller is not None:
            execution_controller.ensure_not_cancelled()

        title_locator = title_locators.nth(index)
        link_locator = link_locators.nth(index)
        title = _extract_locator_text(title_locator)
        if not title:
            continue
        raw_url = _extract_locator_attribute(link_locator, "href")
        url = _unwrap_search_result_url(raw_url, str(getattr(page, "url", "")))
        snippet = _extract_page_snippet_by_index(page, engine_spec.snippet_selectors, index)

        if not url or url.startswith("javascript:"):
            continue
        results.append(
            SearchResult(
                title=title,
                url=url,
                snippet=snippet,
                source_engine=engine_spec.name,
            )
        )

    if not results:
        return [], f"{engine_spec.name} 结果页存在，但没有抽取到标题和链接。"
    return results, None


def _build_query_terms(query: str) -> list[str]:
    """构建用于简单关联度打分的查询词片段。"""
    collapsed_query = _normalize_whitespace(query).lower()
    query_terms = [term for term in re.split(r"\s+", collapsed_query) if term]
    if collapsed_query and collapsed_query not in query_terms:
        query_terms.append(collapsed_query)
    return query_terms


def rank_search_results(query: str, results: list[SearchResult]) -> list[SearchResult]:
    """对来自多搜索引擎的结果做去重和简单关联度排序。"""
    engine_bonus = {"bing": 3, "google": 2, "baidu": 1}
    query_terms = _build_query_terms(query)
    deduplicated: dict[str, tuple[int, int, SearchResult]] = {}
    total_count = len(results)

    for index, result in enumerate(results):
        normalized_url = result.url.rstrip("/")
        haystack = f"{result.title} {result.snippet}".lower()
        match_score = sum(1 for term in query_terms if term and term in haystack)
        exact_bonus = 10 if query.lower() in haystack else 0
        rank_bonus = max(total_count - index, 1)
        score = exact_bonus + match_score * 4 + engine_bonus.get(result.source_engine, 0) + rank_bonus

        existing = deduplicated.get(normalized_url)
        if existing is None or score > existing[0]:
            deduplicated[normalized_url] = (score, index, result)

    return [
        item[2]
        for item in sorted(
            deduplicated.values(),
            key=lambda item: (-item[0], item[1]),
        )
    ]


def _extract_page_description(page: Any) -> str:
    """从已访问页面提取更贴近正文的摘要。"""
    for selector in (
        "meta[name='description']",
        "meta[property='og:description']",
        "main p",
        "article p",
        "p",
    ):
        try:
            locator = page.locator(selector).first
            if selector.startswith("meta"):
                content = str(locator.get_attribute("content") or "").strip()
                text = _normalize_whitespace(content)
            else:
                text = _extract_locator_text(locator)
        except Exception:
            continue
        if text:
            return text
    return ""


def enrich_results_with_page_visits(
    browser_context: Any,
    results: list[SearchResult],
    execution_controller: ExecutionController | None,
    visit_limit: int,
) -> list[str]:
    """访问前几条高相关结果，尽量补全真实页面标题和摘要。"""
    if visit_limit <= 0 or not results:
        return []

    visit_notes: list[str] = []
    visit_page = browser_context.new_page()
    try:
        for result in results[:visit_limit]:
            if execution_controller is not None:
                execution_controller.ensure_not_cancelled()

            try:
                visit_page.goto(
                    result.url,
                    wait_until="domcontentloaded",
                    timeout=PLAYWRIGHT_VISIT_TIMEOUT_MILLISECONDS,
                )
                visit_page.wait_for_timeout(PLAYWRIGHT_VISIT_WAIT_MILLISECONDS)
                page_title = _normalize_whitespace(visit_page.title())
                page_description = _extract_page_description(visit_page)
                if page_title and not page_title.startswith("http"):
                    result.title = page_title
                if page_description:
                    result.snippet = page_description
                result.visited = True
            except ExecutionInterruptedError:
                raise
            except Exception as exc:
                result.visit_summary = f"访问失败：{exc}"
                visit_notes.append(f"访问 {result.url} 失败：{exc}")
    finally:
        visit_page.close()
    return visit_notes


def search_with_playwright(
    query: str,
    result_limit: int,
    execution_controller: ExecutionController | None = None,
) -> tuple[list[SearchResult], list[str]]:
    """使用无头浏览器模拟真实搜索行为，并访问前几条高相关结果。"""
    if not PLAYWRIGHT_AVAILABLE or sync_playwright is None:
        return [], ["当前环境未安装 Playwright，已跳过浏览器搜索。"]

    notes: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        browser_context = browser.new_context()
        search_page = browser_context.new_page()
        try:
            raw_results: list[SearchResult] = []
            for engine_spec in PLAYWRIGHT_SEARCH_ENGINES:
                try:
                    engine_results, engine_note = _search_with_single_engine(
                        search_page,
                        engine_spec,
                        query,
                        result_limit,
                        execution_controller,
                    )
                except ExecutionInterruptedError:
                    raise
                except PlaywrightError as exc:
                    notes.append(f"{engine_spec.name} 搜索失败：{exc}")
                    continue

                if engine_note:
                    notes.append(engine_note)
                raw_results.extend(engine_results)
                if len(raw_results) >= result_limit * PLAYWRIGHT_SEARCH_RESULT_MULTIPLIER:
                    break

            ranked_results = rank_search_results(query, raw_results)[:result_limit]
            notes.extend(
                enrich_results_with_page_visits(
                    browser_context,
                    ranked_results,
                    execution_controller,
                    min(PLAYWRIGHT_VISIT_RESULT_LIMIT, len(ranked_results)),
                )
            )
            return ranked_results, notes
        finally:
            search_page.close()
            browser_context.close()
            browser.close()


def render_search_results(
    query: str,
    results: list[SearchResult],
    notes: list[str] | None = None,
) -> str:
    """将搜索结果格式化为适合工具返回的文本。"""
    if not results:
        return f"查询 `{query}` 未返回可解析的搜索结果。"

    lines = [f"查询: {query}"]
    if notes:
        lines.append("搜索说明:")
        lines.extend(f"- {note}" for note in notes if note.strip())
    for index, result in enumerate(results, start=1):
        lines.append(f"{index}. {result.title}")
        lines.append(f"   链接: {result.url}")
        lines.append(f"   摘要: {result.snippet or '无摘要'}")
        if result.source_engine:
            lines.append(f"   来源: {result.source_engine}")
        if result.visited:
            lines.append("   已访问: 是")
        elif result.visit_summary:
            lines.append(f"   页面访问: {result.visit_summary}")
    return "\n".join(lines)


def iter_search_endpoints(configured_endpoint: str | None) -> list[str]:
    """合并配置端点与内置兜底端点，避免重复尝试同一地址。"""
    endpoints: list[str] = []
    for candidate in [configured_endpoint or "", *FALLBACK_SEARCH_ENDPOINTS]:
        normalized_endpoint = candidate.strip()
        if normalized_endpoint and normalized_endpoint not in endpoints:
            endpoints.append(normalized_endpoint)
    return endpoints


def render_search_failure(
    request_errors: list[tuple[str, Exception]],
    notes: list[str] | None = None,
) -> str:
    """将搜索失败原因整理成便于模型继续决策的错误文本。"""
    lines: list[str] = []
    if notes:
        lines.extend(note for note in notes if note.strip())

    if not request_errors:
        lines.append("❌ 网络搜索请求失败：未获取到可用的搜索响应。")
        return "\n".join(lines)

    network_error_types = (
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.ProxyError,
    )
    likely_network_unavailable = any(
        isinstance(error, network_error_types)
        for _, error in request_errors
    )
    lines.append(
        "❌ 网络搜索暂时不可用，当前运行环境可能无法访问外部搜索服务。"
        if likely_network_unavailable
        else "❌ 网络搜索请求失败。"
    )
    lines.append("已尝试的搜索端点：")
    for endpoint, error in request_errors:
        lines.append(f"- {endpoint} -> {error}")
    return "\n".join(lines)


def search_with_httpx(
    query: str,
    result_limit: int,
    execution_controller: ExecutionController | None = None,
) -> tuple[list[SearchResult], list[str], list[tuple[str, Exception]]]:
    """使用原有的 HTTP HTML 搜索路径做回退。"""
    request_errors: list[tuple[str, Exception]] = []
    notes: list[str] = []

    for endpoint in iter_search_endpoints(settings.search_endpoint):
        try:
            if execution_controller is not None:
                execution_controller.ensure_not_cancelled()
            with httpx.Client(
                follow_redirects=True,
                timeout=settings.search_timeout_seconds,
                headers={"User-Agent": DEFAULT_USER_AGENT},
            ) as client:
                response = client.get(
                    endpoint,
                    params={"q": query},
                )
                response.raise_for_status()
        except ExecutionInterruptedError:
            raise
        except httpx.HTTPError as exc:
            request_errors.append((endpoint, exc))
            continue

        if execution_controller is not None:
            execution_controller.ensure_not_cancelled()

        results = parse_duckduckgo_html_results(response.text)[:result_limit]
        if results:
            return results, notes, request_errors
        notes.append(f"{endpoint} 请求成功，但未解析到可用结果。")

    return [], notes, request_errors


def create_search_web_tool(
    execution_controller: ExecutionController | None = None,
):
    """创建对外可用的 Web 搜索工具。"""

    @tool("search_web")
    def search_web(
        query: str,
        max_results: int = 5,
    ) -> str:
        """
        执行关键词网络搜索，优先使用无头浏览器模拟真实搜索行为，
        需要时回退到 HTTP HTML 搜索，返回标题、链接和摘要结果。
        """
        if execution_controller is not None:
            execution_controller.ensure_not_cancelled()

        normalized_query = query.strip()
        if not normalized_query:
            return "❌ 搜索关键词不能为空。"
        if len(normalized_query) > MAX_SEARCH_QUERY_LENGTH:
            return (
                "❌ 搜索关键词过长。"
                f"当前长度为 {len(normalized_query)}，最大允许 {MAX_SEARCH_QUERY_LENGTH}。"
            )

        safe_result_count = max(1, min(max_results, settings.search_result_limit))
        browser_notes: list[str] = []
        if not PLAYWRIGHT_AVAILABLE:
            browser_notes = [
                "Playwright unavailable; skipped browser search and fell back to HTTP search."
            ]

        if PLAYWRIGHT_AVAILABLE:
            try:
                browser_results, browser_notes = search_with_playwright(
                    normalized_query,
                    safe_result_count,
                    execution_controller,
                )
            except ExecutionInterruptedError:
                raise
            except Exception as exc:
                browser_results = []
                browser_notes = [f"Playwright 搜索失败，已回退到 HTTP 搜索：{exc}"]

            if browser_results:
                return render_search_results(
                    normalized_query,
                    browser_results,
                    browser_notes,
                )

        httpx_results, httpx_notes, request_errors = search_with_httpx(
            normalized_query,
            safe_result_count,
            execution_controller,
        )
        if httpx_results:
            notes = [*browser_notes, *httpx_notes]
            return render_search_results(normalized_query, httpx_results, notes)

        return render_search_failure(request_errors, [*browser_notes, *httpx_notes])

    return attach_tool_risk(search_web, "read")
