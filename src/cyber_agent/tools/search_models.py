"""搜索工具数据模型与常量。

从 search.py 拆分以控制单文件行数。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_SEARCH_ENDPOINT = "https://html.duckduckgo.com/html/"
FALLBACK_SEARCH_ENDPOINTS = (
    DEFAULT_SEARCH_ENDPOINT,
    "https://duckduckgo.com/html/",
)
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
# 多 UA 池：降低单一指纹被识别概率


# ── 反检测配置 ──
_ROTATING_USER_AGENTS = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0",
)
# 常见视口尺寸轮播，避免固定指纹
_ROTATING_VIEWPORTS = (
    {"width": 1366, "height": 768},
    {"width": 1920, "height": 1080},
    {"width": 1440, "height": 900},
    {"width": 1536, "height": 864},
    {"width": 1280, "height": 720},
)
# 浏览器启动参数：关闭 Chrome 自动化检测标记。
_STEALTH_LAUNCH_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--disable-features=IsolateOrigins,site-per-process",
    "--no-sandbox",
    "--disable-setuid-sandbox",
    "--disable-infobars",
    "--disable-dev-shm-usage",
    "--disable-web-security",
    "--disable-features=VizDisplayCompositor",
]
# 注入页面以隐藏 WebDriver 特征的脚本
_STEALTH_SCRIPT = """
// 移除 navigator.webdriver 标记
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
// 伪造 plugins 数量
Object.defineProperty(navigator, 'plugins', { get: () => [1,2,3,4,5] });
// 伪造 languages
Object.defineProperty(navigator, 'languages', { get: () => ['zh-CN','zh','en-US','en'] });
// 伪造 chrome 对象
window.chrome = { runtime: {} };
// 伪造权限查询
const originalQuery = window.navigator.permissions.query;
window.navigator.permissions.query = (parameters) => (
    parameters.name === 'notifications' ?
    Promise.resolve({ state: Notification.permission }) :
    originalQuery(parameters)
);
"""
SEARCH_TIME_BUDGET_SECONDS = 6.0
MAX_SEARCH_QUERY_LENGTH = 300
SEARCH_MIN_RESULTS = 20
SEARCH_MAX_RESULTS = 40
PLAYWRIGHT_SEARCH_RESULT_MULTIPLIER = 6
PLAYWRIGHT_VISIT_RESULT_LIMIT = 3
PLAYWRIGHT_WAIT_MILLISECONDS = 200
PLAYWRIGHT_VISIT_WAIT_MILLISECONDS = 100
PLAYWRIGHT_SEARCH_TIMEOUT_MILLISECONDS = 5000
PLAYWRIGHT_VISIT_TIMEOUT_MILLISECONDS = 2500
PLAYWRIGHT_TYPE_DELAY_MILLISECONDS = 0
PLAYWRIGHT_PAGE_LOAD_TIMEOUT_MILLISECONDS = 3000
PLAYWRIGHT_SCROLL_STEP_PIXELS = 960
PLAYWRIGHT_PAGE_TEXT_MAX_CHARS = 2400
PARALLEL_ENGINE_TIMEOUT_SECONDS = 6.0

# 搜索黑名单域名：CSDN 全家桶 —— 结果中自动剔除
CSDN_DOMAINS = frozenset({
    "csdn.net", "blog.csdn.net", "www.csdn.net",
    "bbs.csdn.net", "download.csdn.net", "edu.csdn.net",
    "live.csdn.net", "ask.csdn.net", "bi.csdn.net",
    "dev.csdn.net", "gitcode.csdn.net", "inscode.csdn.net",
    "spider.csdn.net",
})

# 拉取倍数：原始拉取量为目标量的 N 倍，弥补 CSDN 剔除后的缺口
# 3.5x + 分页 = 确保剔除 CSDN 后仍 ≥ 20 条
FETCH_MULTIPLIER_FOR_CSDN_FILTER = 3.5
PLAYWRIGHT_RELEVANCE_HIGH_SCORE = 12
PLAYWRIGHT_RELEVANCE_MEDIUM_SCORE = 6
PLAYWRIGHT_RELEVANCE_LOW_SCORE = 3
MODEL_RELEVANCE_EVALUATION_LIMIT = 5
MODEL_RELEVANCE_REASON_MAX_CHARS = 120
MODEL_RELEVANCE_PAGE_EXCERPT_MAX_CHARS = 1200


@dataclass(slots=True)


# ── 数据模型 ──
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
    relevance_reason: str = ""
    relevance_source: str = ""
    page_excerpt: str = ""


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
    card_link_selectors: tuple[str, ...] = ()
    card_title_selectors: tuple[str, ...] = ()
    card_snippet_selectors: tuple[str, ...] = ()
    consent_button_selectors: tuple[str, ...] = ()
    search_button_selectors: tuple[str, ...] = ()
    blocked_title_markers: tuple[str, ...] = ()
    blocked_url_markers: tuple[str, ...] = ()
    blocked_text_markers: tuple[str, ...] = ()
    search_url_template: str = ""
    homepage_goto_timeout_milliseconds: int = PLAYWRIGHT_SEARCH_TIMEOUT_MILLISECONDS
    result_ready_timeout_milliseconds: int = PLAYWRIGHT_SEARCH_TIMEOUT_MILLISECONDS
    load_state_timeout_milliseconds: int = PLAYWRIGHT_PAGE_LOAD_TIMEOUT_MILLISECONDS
    post_submit_wait_milliseconds: int = PLAYWRIGHT_WAIT_MILLISECONDS
    settle_wait_milliseconds: int = 500
    auto_scroll_rounds: int = 3
    wait_for_full_page_load: bool = False
