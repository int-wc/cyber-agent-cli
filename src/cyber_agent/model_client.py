from __future__ import annotations

from threading import Lock
from typing import Any
from urllib.parse import urlsplit, urlunsplit

_SOCKSIO_FALLBACK_WARNING_LOCK = Lock()
_SOCKSIO_FALLBACK_WARNING_EMITTED = False


def _is_missing_socksio_error(exc: ImportError) -> bool:
    message = str(exc)
    return "SOCKS proxy" in message and "socksio" in message


def _warn_missing_socksio_once(fallback_proxy: str) -> None:
    global _SOCKSIO_FALLBACK_WARNING_EMITTED

    with _SOCKSIO_FALLBACK_WARNING_LOCK:
        if _SOCKSIO_FALLBACK_WARNING_EMITTED:
            return
        _SOCKSIO_FALLBACK_WARNING_EMITTED = True

    from .logging import log_warning

    log_warning(
        "model_client",
        "当前 Python 环境缺少 socksio，已自动改用 HTTP 代理："
        f"{fallback_proxy}",
    )


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

        retry_kwargs = dict(kwargs)
        retry_kwargs["openai_proxy"] = fallback_proxy
        _warn_missing_socksio_once(fallback_proxy)
        return llm_cls(**retry_kwargs)
