from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit, urlunsplit


def _is_missing_socksio_error(exc: ImportError) -> bool:
    message = str(exc)
    return "SOCKS proxy" in message and "socksio" in message


def get_http_proxy_fallback_url(proxy_url: object) -> str | None:
    """Return an HTTP proxy fallback for socks proxies using the same host/port."""
    if not isinstance(proxy_url, str) or not proxy_url.strip():
        return None

    parsed = urlsplit(proxy_url.strip())
    if parsed.scheme not in {"socks", "socks4", "socks5"} or not parsed.netloc:
        return None
    return urlunsplit(("http", parsed.netloc, "", "", ""))


def build_llm_with_proxy_fallback(llm_cls: type[Any], kwargs: dict[str, Any]) -> Any:
    """Build an LLM client, falling back from SOCKS to HTTP if socksio is absent."""
    try:
        return llm_cls(**kwargs)
    except ImportError as exc:
        if not _is_missing_socksio_error(exc):
            raise

        fallback_proxy = get_http_proxy_fallback_url(kwargs.get("openai_proxy"))
        if not fallback_proxy:
            raise

        from .logging import log_warning

        retry_kwargs = dict(kwargs)
        retry_kwargs["openai_proxy"] = fallback_proxy
        log_warning(
            "model_client",
            "当前 Python 环境缺少 socksio，已自动改用 HTTP 代理："
            f"{fallback_proxy}",
        )
        return llm_cls(**retry_kwargs)
