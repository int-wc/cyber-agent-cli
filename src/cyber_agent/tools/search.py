from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, urljoin, urlparse

import httpx
from langchain_core.tools import tool

from ..config import settings
from ..execution_control import ExecutionController, ExecutionInterruptedError
from .metadata import attach_tool_risk

try:
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover - 缺少依赖时回退到 HTTP 搜索
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
PLAYWRIGHT_VISIT_RESULT_LIMIT = 3
PLAYWRIGHT_WAIT_MILLISECONDS = 800
PLAYWRIGHT_VISIT_WAIT_MILLISECONDS = 200
PLAYWRIGHT_SEARCH_TIMEOUT_MILLISECONDS = 6000
PLAYWRIGHT_VISIT_TIMEOUT_MILLISECONDS = 4000
PLAYWRIGHT_TYPE_DELAY_MILLISECONDS = 80
PLAYWRIGHT_PAGE_LOAD_TIMEOUT_MILLISECONDS = 4000
PLAYWRIGHT_SCROLL_STEP_PIXELS = 960
PLAYWRIGHT_PAGE_TEXT_MAX_CHARS = 2400
PLAYWRIGHT_RELEVANCE_HIGH_SCORE = 12
PLAYWRIGHT_RELEVANCE_MEDIUM_SCORE = 6
PLAYWRIGHT_RELEVANCE_LOW_SCORE = 3


@dataclass(slots=True)
class SearchResult:
    """描述单条搜索结果。"""

    title: str
    url: str
    snippet: str
    source_engine: str = ""
    visited: bool = False
    visit_summary: str = ""
    relevance_score: int = 0
    relevance_summary: str = ""


@dataclass(frozen=True, slots=True)
class SearchEngineSpec:
    """描述单个搜索引擎的首页交互方式与结果解析规则。"""

    name: str
    homepage_url: str
    search_input_selectors: tuple[str, ...]
    result_ready_selectors: tuple[str, ...]
    result_selector: str
    link_selector: str
    title_selector: str
    snippet_selectors: tuple[str, ...]
    consent_button_selectors: tuple[str, ...] = ()
    search_button_selectors: tuple[str, ...] = ()
    blocked_title_markers: tuple[str, ...] = ()
    blocked_url_markers: tuple[str, ...] = ()
    blocked_text_markers: tuple[str, ...] = ()
    result_ready_timeout_milliseconds: int = PLAYWRIGHT_SEARCH_TIMEOUT_MILLISECONDS
    post_submit_wait_milliseconds: int = PLAYWRIGHT_WAIT_MILLISECONDS
    settle_wait_milliseconds: int = 500
    auto_scroll_rounds: int = 3
    wait_for_full_page_load: bool = False


PLAYWRIGHT_SEARCH_ENGINES = (
    SearchEngineSpec(
        name="bing",
        homepage_url="https://www.bing.com/",
        search_input_selectors=("textarea[name='q']", "input[name='q']"),
        result_ready_selectors=("#b_results", "li.b_algo"),
        result_selector="li.b_algo",
        link_selector="li.b_algo h2 a",
        title_selector="li.b_algo h2 a",
        snippet_selectors=("li.b_algo .b_caption p", "li.b_algo .b_snippet"),
        search_button_selectors=("button#search_icon", "input#sb_form_go"),
        blocked_text_markers=(
            "verify you are human",
            "one last step",
            "enter the characters you see",
            "detected unusual traffic",
            "security check",
            "请输入验证码",
            "请完成验证",
        ),
    ),
    SearchEngineSpec(
        name="google",
        homepage_url="https://www.google.com/",
        search_input_selectors=("textarea[name='q']", "input[name='q']"),
        result_ready_selectors=("#search", "div.g"),
        result_selector="div.g",
        link_selector="div.g a[href]:has(h3)",
        title_selector="div.g a[href]:has(h3) h3",
        snippet_selectors=(
            "div.g div.VwiC3b",
            "div.g span.aCOpRe",
            "div.g div[data-sncf='1']",
        ),
        consent_button_selectors=(
            "button:has-text('Accept all')",
            "button:has-text('I agree')",
            "button:has-text('全部接受')",
            "button:has-text('接受全部')",
        ),
        blocked_title_markers=("unusual traffic", "before you continue"),
        blocked_url_markers=("/sorry/",),
        blocked_text_markers=("unusual traffic", "before you continue", "our systems have detected"),
    ),
    SearchEngineSpec(
        name="baidu",
        homepage_url="https://www.baidu.com/",
        search_input_selectors=("textarea[name='wd']", "input[name='wd']"),
        result_ready_selectors=(
            "#content_left",
            "#content_left > div.result",
            "#content_left > div.result-op",
        ),
        result_selector="#content_left > div.result, #content_left > div.result-op",
        link_selector="#content_left > div.result h3 a, #content_left > div.result-op h3 a",
        title_selector="#content_left > div.result h3 a, #content_left > div.result-op h3 a",
        snippet_selectors=(
            "#content_left > div.result .c-abstract, #content_left > div.result-op .c-abstract",
            "#content_left > div.result .content-right_8Zs40, #content_left > div.result-op .content-right_8Zs40",
            "#content_left > div.result .c-span-last, #content_left > div.result-op .c-span-last",
        ),
        search_button_selectors=("input#su", "button#su"),
        blocked_text_markers=("百度安全验证", "安全验证", "请输入验证码"),
        result_ready_timeout_milliseconds=9000,
        post_submit_wait_milliseconds=1600,
        settle_wait_milliseconds=800,
        auto_scroll_rounds=4,
        wait_for_full_page_load=True,
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
    """统一还原搜索结果页中的真实目标地址。"""
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
                self.results[-1].snippet = snippet
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


def _extract_locator_attribute(locator: Any, attribute_name: str) -> str:
    """从 Playwright 定位器中安全提取属性。"""
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


def _wait_for_first_visible_locator(page: Any, selectors: tuple[str, ...]) -> Any | None:
    """按顺序等待第一个可见元素。"""
    for selector in selectors:
        try:
            locator = page.locator(selector).first
            locator.wait_for(state="visible", timeout=2000)
            return locator
        except Exception:
            continue
    return None


def _wait_for_any_selector(
    page: Any,
    selectors: tuple[str, ...],
    timeout_milliseconds: int = PLAYWRIGHT_SEARCH_TIMEOUT_MILLISECONDS,
) -> bool:
    """等待任一结果选择器出现。"""
    for selector in selectors:
        try:
            page.locator(selector).first.wait_for(
                state="visible",
                timeout=timeout_milliseconds,
            )
            return True
        except Exception:
            continue
    return False


def _click_first_available(page: Any, selectors: tuple[str, ...]) -> bool:
    """尝试点击第一个可交互元素。"""
    for selector in selectors:
        try:
            locator = page.locator(selector).first
            locator.click(timeout=1500)
            return True
        except Exception:
            continue
    return False


def _type_query_like_human(input_locator: Any, query: str) -> None:
    """模拟真人逐字输入关键词。"""
    input_locator.click(timeout=2000)
    try:
        input_locator.fill("", timeout=1000)
    except Exception:
        pass
    try:
        input_locator.type(query, delay=PLAYWRIGHT_TYPE_DELAY_MILLISECONDS)
        return
    except Exception:
        pass
    input_locator.fill(query)


def _submit_search(page: Any, input_locator: Any, engine_spec: SearchEngineSpec) -> None:
    """提交搜索表单，优先点击搜索按钮，否则回车。"""
    if engine_spec.search_button_selectors and _click_first_available(
        page,
        engine_spec.search_button_selectors,
    ):
        return
    input_locator.press("Enter", timeout=1500)


def _extract_page_snippet_by_index(page: Any, selectors: tuple[str, ...], index: int) -> str:
    """按结果序号提取摘要文本。"""
    for selector in selectors:
        try:
            locator = page.locator(selector).nth(index)
            text = _extract_locator_text(locator)
        except Exception:
            continue
        if text:
            return text
    return ""


def _wait_for_load_state(page: Any, state: str, timeout_milliseconds: int) -> bool:
    """尝试等待页面达到指定加载状态。"""
    try:
        page.wait_for_load_state(state, timeout=timeout_milliseconds)
        return True
    except Exception:
        return False


def _count_locators(locator_collection: Any) -> int:
    """安全统计定位器集合中的元素数量。"""
    try:
        return int(locator_collection.count())
    except Exception:
        return 0


def _scroll_page_once(page: Any, wait_milliseconds: int) -> bool:
    """向下滚动一屏，兼容可见窗口与无头窗口。"""
    scrolled = False
    try:
        page.mouse.wheel(0, PLAYWRIGHT_SCROLL_STEP_PIXELS)
        scrolled = True
    except Exception:
        pass
    if not scrolled:
        try:
            page.evaluate(
                f"() => window.scrollBy(0, Math.max(window.innerHeight || 0, {PLAYWRIGHT_SCROLL_STEP_PIXELS}))"
            )
            scrolled = True
        except Exception:
            return False
    page.wait_for_timeout(wait_milliseconds)
    return True


def _wait_for_results_to_settle(
    page: Any,
    engine_spec: SearchEngineSpec,
    execution_controller: ExecutionController | None,
) -> bool:
    """等待结果区域稳定，并通过滚动触发可能的懒加载。"""
    if not _wait_for_any_selector(
        page,
        engine_spec.result_ready_selectors,
        timeout_milliseconds=engine_spec.result_ready_timeout_milliseconds,
    ):
        return False

    if engine_spec.wait_for_full_page_load:
        _wait_for_load_state(
            page,
            "load",
            min(engine_spec.result_ready_timeout_milliseconds, PLAYWRIGHT_PAGE_LOAD_TIMEOUT_MILLISECONDS),
        )
        _wait_for_load_state(
            page,
            "networkidle",
            min(engine_spec.result_ready_timeout_milliseconds, PLAYWRIGHT_PAGE_LOAD_TIMEOUT_MILLISECONDS),
        )

    page.wait_for_timeout(engine_spec.post_submit_wait_milliseconds)

    previous_count = -1
    stable_rounds = 0
    best_count = 0
    total_rounds = max(1, engine_spec.auto_scroll_rounds + 2)

    for round_index in range(total_rounds):
        if execution_controller is not None:
            execution_controller.ensure_not_cancelled()

        current_count = _count_locators(page.locator(engine_spec.result_selector))
        best_count = max(best_count, current_count)
        if current_count > 0 and current_count == previous_count:
            stable_rounds += 1
            if stable_rounds >= 1:
                break
        else:
            stable_rounds = 0

        if round_index < engine_spec.auto_scroll_rounds:
            _scroll_page_once(page, engine_spec.settle_wait_milliseconds)
        else:
            page.wait_for_timeout(engine_spec.settle_wait_milliseconds)

        previous_count = current_count

    return best_count > 0


def _extract_page_text(page: Any) -> str:
    """提取用于相关性判断的页面核心文本。"""
    for selector in ("main", "article", "[role='main']", "body"):
        try:
            text = _extract_locator_text(page.locator(selector).first)
        except Exception:
            continue
        if text:
            return text[:PLAYWRIGHT_PAGE_TEXT_MAX_CHARS]
    return ""


def _annotate_result_relevance(
    query: str,
    result: SearchResult,
    *,
    page_text: str = "",
) -> None:
    """根据查询与页面内容给出简单相关性判断。"""
    normalized_query = _normalize_whitespace(query).lower()
    query_terms = _build_query_terms(query)
    title_text = result.title.lower()
    snippet_text = result.snippet.lower()
    body_text = page_text.lower()
    url_text = result.url.lower()
    combined_text = " ".join(part for part in (title_text, snippet_text, body_text) if part)

    score = 0
    if normalized_query and normalized_query in combined_text:
        score += 12
    score += sum(4 for term in query_terms if term and term in title_text)
    score += sum(3 for term in query_terms if term and term in snippet_text)
    score += sum(2 for term in query_terms if term and term in body_text)
    score += sum(1 for term in query_terms if term and term in url_text)

    if score >= PLAYWRIGHT_RELEVANCE_HIGH_SCORE:
        summary = "高度相关"
    elif score >= PLAYWRIGHT_RELEVANCE_MEDIUM_SCORE:
        summary = "相关"
    elif score >= PLAYWRIGHT_RELEVANCE_LOW_SCORE:
        summary = "弱相关，建议人工复核"
    else:
        summary = "疑似不相关"

    result.relevance_score = score
    result.relevance_summary = summary


def _page_looks_blocked(page: Any, engine_spec: SearchEngineSpec) -> bool:
    """识别验证页或反爬拦截页。"""
    try:
        page_title = _normalize_whitespace(page.title()).lower()
    except Exception:
        page_title = ""
    current_url = str(getattr(page, "url", "")).lower()
    try:
        body_text = _extract_locator_text(page.locator("body").first).lower()
    except Exception:
        body_text = ""
    return (
        any(marker.lower() in page_title for marker in engine_spec.blocked_title_markers)
        or any(marker.lower() in current_url for marker in engine_spec.blocked_url_markers)
        or any(marker.lower() in body_text for marker in engine_spec.blocked_text_markers)
    )


def _search_with_single_engine(
    page: Any,
    engine_spec: SearchEngineSpec,
    query: str,
    result_limit: int,
    execution_controller: ExecutionController | None,
) -> tuple[list[SearchResult], str | None]:
    """使用单个搜索引擎执行真人式浏览器搜索。"""
    if execution_controller is not None:
        execution_controller.ensure_not_cancelled()

    page.goto(
        engine_spec.homepage_url,
        wait_until="domcontentloaded",
        timeout=PLAYWRIGHT_SEARCH_TIMEOUT_MILLISECONDS,
    )
    page.wait_for_timeout(PLAYWRIGHT_WAIT_MILLISECONDS)
    if engine_spec.consent_button_selectors:
        _click_first_available(page, engine_spec.consent_button_selectors)
        page.wait_for_timeout(200)

    search_input = _wait_for_first_visible_locator(page, engine_spec.search_input_selectors)
    if search_input is None:
        return [], f"{engine_spec.name} 未找到搜索输入框。"

    _type_query_like_human(search_input, query)
    page.wait_for_timeout(200)
    _submit_search(page, search_input, engine_spec)
    if not _wait_for_results_to_settle(page, engine_spec, execution_controller):
        if _page_looks_blocked(page, engine_spec):
            blocked_title = _normalize_whitespace(page.title()) or str(getattr(page, "url", ""))
            return [], f"{engine_spec.name} 返回验证或拦截页面：{blocked_title}"
        return [], f"{engine_spec.name} 未等待到稳定的搜索结果区域。"

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
        _annotate_result_relevance(query, results[-1])

    if not results:
        return [], f"{engine_spec.name} 结果页存在，但没有抽取到标题和链接。"
    return results, None


def _build_query_terms(query: str) -> list[str]:
    """构建用于简单相关度打分的查询词片段。"""
    collapsed_query = _normalize_whitespace(query).lower()
    query_terms = [term for term in re.split(r"\s+", collapsed_query) if term]
    if collapsed_query and collapsed_query not in query_terms:
        query_terms.append(collapsed_query)
    return query_terms


def rank_search_results(query: str, results: list[SearchResult]) -> list[SearchResult]:
    """对来自多搜索引擎的结果做去重和简单相关度排序。"""
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


def rerank_results_by_relevance(results: list[SearchResult]) -> list[SearchResult]:
    """在原有排序基础上加入页面相关性和访问状态。"""
    indexed_results = list(enumerate(results))
    return [
        result
        for _, result in sorted(
            indexed_results,
            key=lambda item: (
                -item[1].relevance_score,
                0 if item[1].visited else 1,
                item[0],
            ),
        )
    ]


def _extract_page_description(page: Any) -> str:
    """从已访问页面提取更接近正文的摘要。"""
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
    query: str,
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
                _wait_for_load_state(
                    visit_page,
                    "load",
                    PLAYWRIGHT_PAGE_LOAD_TIMEOUT_MILLISECONDS,
                )
                visit_page.wait_for_timeout(PLAYWRIGHT_VISIT_WAIT_MILLISECONDS)
                _scroll_page_once(visit_page, PLAYWRIGHT_VISIT_WAIT_MILLISECONDS)
                page_title = _normalize_whitespace(visit_page.title())
                page_description = _extract_page_description(visit_page)
                page_text = _extract_page_text(visit_page)
                if page_title and not page_title.startswith("http"):
                    result.title = page_title
                if page_description:
                    result.snippet = page_description
                result.visited = True
                _annotate_result_relevance(query, result, page_text=page_text)
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
    """使用 Playwright 按真人搜索流程执行浏览器搜索。"""
    if not PLAYWRIGHT_AVAILABLE or sync_playwright is None:
        return [], ["当前环境未安装 Playwright，已跳过浏览器搜索。"]

    notes = [
        (
            "浏览器模式：可见窗口"
            if settings.search_show_browser
            else "浏览器模式：无头窗口"
        )
    ]
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=not settings.search_show_browser)
        browser_context = browser.new_context(
            user_agent=DEFAULT_USER_AGENT,
            viewport={"width": 1366, "height": 900},
        )
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

            ranked_results = rank_search_results(
                query,
                raw_results,
            )[: min(len(raw_results), max(result_limit, PLAYWRIGHT_VISIT_RESULT_LIMIT))]
            notes.extend(
                enrich_results_with_page_visits(
                    browser_context,
                    query,
                    ranked_results,
                    execution_controller,
                    min(PLAYWRIGHT_VISIT_RESULT_LIMIT, len(ranked_results)),
                )
            )
            return rerank_results_by_relevance(ranked_results)[:result_limit], notes
        finally:
            search_page.close()
            browser_context.close()
            browser.close()


def render_search_results(
    query: str,
    results: list[SearchResult],
    notes: list[str] | None = None,
) -> str:
    """将搜索结果格式化为文本。"""
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
            if result.relevance_summary:
                lines.append(f"   页面判断: {result.relevance_summary}")
        elif result.visit_summary:
            lines.append(f"   页面访问: {result.visit_summary}")
        elif result.relevance_summary:
            lines.append(f"   结果判断: {result.relevance_summary}")
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
    """整理搜索失败原因。"""
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
        执行关键词网络搜索，优先使用浏览器首页交互搜索，
        必要时回退到 HTTP HTML 搜索，返回标题、链接和摘要结果。
        """
        if execution_controller is not None:
            execution_controller.ensure_not_cancelled()

        normalized_query = query.strip()
        if not normalized_query:
            return "❌ 搜索关键词不能为空。"
        if len(normalized_query) > MAX_SEARCH_QUERY_LENGTH:
            return (
                "❌ 搜索关键词过长。"
                f" 当前长度为 {len(normalized_query)}，最大允许 {MAX_SEARCH_QUERY_LENGTH}。"
            )

        safe_result_count = max(1, min(max_results, settings.search_result_limit))
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
