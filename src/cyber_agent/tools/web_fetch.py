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

# CSDN、知乎等需要浏览器绕过的站点
BROWSER_FETCH_DOMAINS = {
    "csdn.net", "blog.csdn.net", "www.csdn.net",
    "zhihu.com", "www.zhihu.com", "zhuanlan.zhihu.com",
    "jianshu.com", "www.jianshu.com",
    "juejin.cn", "juejin.im",
    "cnblogs.com", "www.cnblogs.com",
    "segmentfault.com",
}

# 普通 HTTP 抓取使用的请求头
_FETCH_HEADERS_CSDN = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
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


def _needs_browser_fetch(url: str) -> bool:
    """判断 URL 是否需要浏览器级别的抓取。"""
    try:
        hostname = urlparse(url).hostname or ""
    except Exception:
        return False
    return any(hostname == d or hostname.endswith("." + d) for d in BROWSER_FETCH_DOMAINS)


def _fetch_with_httpx(url: str) -> str:
    """使用 HTTP 请求获取页面内容。"""
    import httpx

    headers = _FETCH_HEADERS_CSDN.copy()
    if "zhihu.com" in url:
        headers = _FETCH_HEADERS_ZHIHU.copy()

    with httpx.Client(
        follow_redirects=True,
        timeout=FETCH_TIMEOUT_SECONDS,
        headers=headers,
        http2=True,
    ) as client:
        response = client.get(url)
        response.raise_for_status()
        html = response.text

    return _extract_text_from_html(html)


def _fetch_with_playwright(url: str) -> str:
    """使用 Playwright 浏览器获取页面内容（绕过 JS 验证）。"""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return _fetch_with_httpx(url)

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=True,
                args=["--disable-blink-features=AutomationControlled"],
            )
            context = browser.new_context(
                user_agent=_FETCH_HEADERS_CSDN["User-Agent"],
                viewport={"width": 1366, "height": 768},
                locale="zh-CN",
                timezone_id="Asia/Shanghai",
            )
            context.add_init_script("""
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
window.chrome = { runtime: {} };
""")
            page = context.new_page()
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=10000)
                page.wait_for_timeout(1500)

                # CSDN 特殊处理：等待文章内容加载
                if "csdn.net" in url:
                    try:
                        page.wait_for_selector(
                            "article, #content_views, .article_content, .markdown_views",
                            timeout=3000,
                        )
                    except Exception:
                        pass

                # 知乎特殊处理：等待文章内容
                if "zhihu.com" in url:
                    try:
                        page.wait_for_selector(
                            ".Post-RichText, .RichText, .ArticleItem-content",
                            timeout=3000,
                        )
                    except Exception:
                        pass

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
        获取指定 URL 的网页文本内容。
        对 CSDN、知乎等需要 JS 渲染的站点自动使用浏览器获取。
        url 是需要获取的网页地址。
        use_browser 为 True 时强制使用浏览器模式。
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
