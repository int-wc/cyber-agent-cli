"""Web 页面获取工具，对 CSDN、知乎等站点启用浏览器绕过。"""

from __future__ import annotations

import re
import time as time_mod
from typing import Any
from urllib.parse import urlparse

from langchain_core.tools import tool

from ..config import settings
from ..execution_control import ExecutionController, ExecutionInterruptedError
from .metadata import attach_tool_risk

# 检测 h2 是否可用，不可用时回退到 HTTP/1.1
try:
    import h2  # noqa: F401
    _H2_AVAILABLE = True
except ImportError:
    _H2_AVAILABLE = False

# CSDN、知乎等需要浏览器绕过的站点
BROWSER_FETCH_DOMAINS = {
    "csdn.net", "blog.csdn.net", "www.csdn.net", "bbs.csdn.net",
    "zhihu.com", "www.zhihu.com", "zhuanlan.zhihu.com",
    "jianshu.com", "www.jianshu.com",
    "juejin.cn", "juejin.im",
    "cnblogs.com", "www.cnblogs.com",
    "segmentfault.com",
}

# 多 UA 池用于轮换，降低反爬识别
_FETCH_USER_AGENTS = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
)

# 普通 HTTP 抓取使用的请求头
_FETCH_HEADERS_CSDN = {
    "User-Agent": _FETCH_USER_AGENTS[0],
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Sec-Ch-Ua": '"Chromium";v="131", "Not_A Brand";v="24"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}

_FETCH_HEADERS_ZHIHU = {
    **_FETCH_HEADERS_CSDN,
    "Referer": "https://www.zhihu.com/",
    "Cookie": "",  # 留空，大部分公开内容不需要登录
}

MAX_FETCH_CHARS = 8000
FETCH_TIMEOUT_SECONDS = 8.0
_FETCH_MAX_RETRIES = 2


def _needs_browser_fetch(url: str) -> bool:
    """判断 URL 是否需要浏览器级别的抓取。"""
    try:
        hostname = urlparse(url).hostname or ""
    except Exception:
        return False
    return any(hostname == d or hostname.endswith("." + d) for d in BROWSER_FETCH_DOMAINS)


def _fetch_with_httpx(url: str) -> str:
    """使用 HTTP 请求获取页面内容，支持重试与 UA 轮换。"""
    import httpx
    import random

    last_error: Exception | None = None

    for attempt in range(_FETCH_MAX_RETRIES + 1):
        headers = _FETCH_HEADERS_CSDN.copy()
        if "zhihu.com" in url:
            headers = _FETCH_HEADERS_ZHIHU.copy()
        # 轮换 UA
        headers["User-Agent"] = random.choice(_FETCH_USER_AGENTS)

        try:
            with httpx.Client(
                follow_redirects=True,
                timeout=FETCH_TIMEOUT_SECONDS,
                headers=headers,
                http2=_H2_AVAILABLE,
            ) as client:
                response = client.get(url)
                response.raise_for_status()
                html = response.text

            # CSDN 可能返回空内容或验证页面，此时重试
            if "csdn.net" in url and len(html) < 500:
                if attempt < _FETCH_MAX_RETRIES:
                    time_mod.sleep(0.5 * (attempt + 1))
                    continue
            return _extract_text_from_html(html)

        except Exception as exc:
            last_error = exc
            if attempt < _FETCH_MAX_RETRIES:
                time_mod.sleep(0.5 * (attempt + 1))
                continue

    raise last_error  # type: ignore[misc]


def _fetch_with_playwright(url: str) -> str:
    """使用 Playwright 浏览器获取页面内容（绕过 JS 验证）。"""
    import random

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return _fetch_with_httpx(url)

    # 增强型反检测脚本：针对知乎等强反爬站点
    _ADVANCED_STEALTH_SCRIPT = """
// 移除 webdriver 标记
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
// 伪造 plugins — 知乎会检查
Object.defineProperty(navigator, 'plugins', {
    get: () => {
        const plugins = [
            { name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer', description: 'Portable Document Format' },
            { name: 'Chrome PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai', description: '' },
            { name: 'Native Client', filename: 'internal-nacl-plugin', description: '' },
        ];
        plugins.item = (i) => plugins[i];
        plugins.namedItem = (n) => plugins.find(p => p.name === n);
        plugins.refresh = () => {};
        Object.setPrototypeOf(plugins, PluginArray.prototype);
        return plugins;
    }
});
// 伪造 languages
Object.defineProperty(navigator, 'languages', { get: () => ['zh-CN', 'zh', 'en-US', 'en'] });
Object.defineProperty(navigator, 'language', { get: () => 'zh-CN' });
// 伪造硬件信息
Object.defineProperty(navigator, 'hardwareConcurrency', { get: () => 8 });
Object.defineProperty(navigator, 'deviceMemory', { get: () => 8 });
// 伪造 platform
Object.defineProperty(navigator, 'platform', { get: () => 'Win32' });
// chrome 对象
window.chrome = {
    runtime: {},
    loadTimes: () => {},
    csi: () => {},
    app: {},
};
// 伪造权限查询
const origQuery = window.navigator.permissions.query;
window.navigator.permissions.query = (params) => (
    params.name === 'notifications'
        ? Promise.resolve({ state: Notification.permission })
        : origQuery(params)
);
// 覆盖 headless 检测相关属性
Object.defineProperty(document, 'hidden', { get: () => false });
Object.defineProperty(document, 'visibilityState', { get: () => 'visible' });
// 伪造 WebGL vendor/renderer — 知乎可能用 canvas 指纹
const getParameterProto = WebGLRenderingContext.prototype.getParameter;
WebGLRenderingContext.prototype.getParameter = function(param) {
    if (param === 37445) return 'Intel Inc.';
    if (param === 37446) return 'Intel Iris OpenGL Engine';
    return getParameterProto.call(this, param);
};
"""

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-features=IsolateOrigins,site-per-process",
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-infobars",
                    "--disable-dev-shm-usage",
                    "--disable-component-extensions-with-background-pages",
                    "--disable-client-side-phishing-detection",
                    "--disable-sync",
                    "--disable-default-apps",
                    "--hide-scrollbars",
                    "--mute-audio",
                    "--no-first-run",
                    "--no-default-browser-check",
                ],
            )
            viewport = random.choice([
                {"width": 1366, "height": 768},
                {"width": 1920, "height": 1080},
                {"width": 1440, "height": 900},
            ])
            context = browser.new_context(
                user_agent=random.choice(_FETCH_USER_AGENTS),
                viewport=viewport,
                locale="zh-CN",
                timezone_id="Asia/Shanghai",
                # 额外 HTTP 头模拟真实浏览器
                extra_http_headers={
                    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Sec-Ch-Ua": '"Chromium";v="131", "Not_A Brand";v="24"',
                    "Sec-Ch-Ua-Mobile": "?0",
                    "Sec-Ch-Ua-Platform": '"Windows"',
                },
            )
            context.add_init_script(_ADVANCED_STEALTH_SCRIPT)
            page = context.new_page()
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=10000)
                page.wait_for_timeout(1500)

                # CSDN 特殊处理：等待文章内容加载并尝试关闭弹窗
                if "csdn.net" in url:
                    try:
                        # 尝试关闭可能的登录弹窗
                        page.evaluate("""
const closeBtns = document.querySelectorAll('.passport-login-tip .close, .modal-close, .login-mark .close');
closeBtns.forEach(btn => btn.click());
""")
                    except Exception:
                        pass
                    try:
                        page.wait_for_selector(
                            "article, #content_views, .article_content, .markdown_views, #article_content",
                            timeout=3000,
                        )
                    except Exception:
                        pass

                # 知乎特殊处理：等待文章内容
                if "zhihu.com" in url:
                    try:
                        page.wait_for_selector(
                            ".Post-RichText, .RichText, .ArticleItem-content, .RichContent-inner",
                            timeout=3000,
                        )
                    except Exception:
                        pass
                    # 尝试展开折叠内容
                    try:
                        page.evaluate("""
document.querySelectorAll('.RichContent-cover, .ContentItem-expandable').forEach(el => {
    el.click();
});
""")
                    except Exception:
                        pass

                # 滚动以触发懒加载
                page.evaluate("window.scrollTo(0, document.body.scrollHeight * 0.3)")
                page.wait_for_timeout(500)
                page.evaluate("window.scrollTo(0, document.body.scrollHeight * 0.6)")
                page.wait_for_timeout(500)

                html = page.content()
            finally:
                page.close()
                context.close()
                browser.close()
    except Exception:
        return _fetch_with_httpx(url)

    return _extract_text_from_html(html)


def _extract_text_from_html(html: str) -> str:
    """从 HTML 中提取主要文本内容。"""
    # 移除 script/style 标签
    html = re.sub(r"<script[^>]*>[\s\S]*?</script>", "", html)
    html = re.sub(r"<style[^>]*>[\s\S]*?</style>", "", html)

    # 移除 HTML 标签
    text = re.sub(r"<[^>]+>", " ", html)

    # 清理空白
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"&#?\w+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) > MAX_FETCH_CHARS:
        text = text[:MAX_FETCH_CHARS] + "\n\n... 内容过长，已截断。"

    return text


def create_web_fetch_tool(
    execution_controller: ExecutionController | None = None,
):
    """创建 Web 页面获取工具，对 CSDN/知乎自动切换浏览器模式。"""

    @tool("fetch_web_page")
    def fetch_web_page(
        url: str,
        use_browser: bool = False,
    ) -> str:
        """
        获取网页内容。必须提供 url 参数（字符串，如 "https://example.com"）。
        可选 use_browser 参数（默认False）。
        示例: fetch_web_page(url="https://downloads.openwrt.org/")
        """
        if execution_controller is not None:
            execution_controller.ensure_not_cancelled()

        normalized_url = url.strip()
        if not normalized_url:
            return "❌ URL 不能为空。"
        if not normalized_url.startswith(("http://", "https://")):
            normalized_url = "https://" + normalized_url

        needs_browser = use_browser or _needs_browser_fetch(normalized_url)
        start = time_mod.monotonic()

        try:
            if needs_browser:
                text = _fetch_with_playwright(normalized_url)
                method = "浏览器"
            else:
                text = _fetch_with_httpx(normalized_url)
                method = "HTTP"
        except ExecutionInterruptedError:
            raise
        except Exception as exc:
            return f"❌ 获取页面失败：{exc}"

        elapsed = time_mod.monotonic() - start
        return (
            f"页面内容（{method}获取，耗时 {elapsed:.1f}s）：\n\n{text}"
            if text else
            f"❌ 未能从 {normalized_url} 提取到有效文本内容。"
        )

    return attach_tool_risk(fetch_web_page, "read")
