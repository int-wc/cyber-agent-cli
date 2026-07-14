from __future__ import annotations

from threading import Lock
from typing import Any, Callable, Iterable
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


def _is_provider_fallback_error(exc: Exception) -> bool:
    message = str(exc).lower()
    markers = (
        "401",
        "403",
        "429",
        "503",
        "unauthorized",
        "forbidden",
        "invalid api key",
        "incorrect api key",
        "api key",
        "quota",
        "rate limit",
        "insufficient",
        "permission",
        "model not found",
        "model_not_found",
        "does not exist",
        "not support",
        "unsupported model",
        "freeusagelimiterror",
        "service temporarily unavailable",
    )
    return any(marker in message for marker in markers)


def _safe_model_label(kwargs: dict[str, Any]) -> str:
    model = str(kwargs.get("model") or "unknown")
    base_url = str(kwargs.get("base_url") or "default")
    return f"{model} @ {base_url}"


def _build_llm_once_with_proxy_fallback(
    llm_cls: type[Any],
    kwargs: dict[str, Any],
) -> Any:
    """Build an LLM client, falling back from SOCKS to HTTP if socksio is absent."""
    clean_kwargs = dict(kwargs)
    clean_kwargs.pop("_fallback_kwargs", None)
    try:
        return llm_cls(**clean_kwargs)
    except ImportError as exc:
        if not _is_missing_socksio_error(exc):
            raise

        fallback_proxy = get_http_proxy_fallback_url(clean_kwargs.get("openai_proxy"))
        if not fallback_proxy:
            raise

        retry_kwargs = dict(clean_kwargs)
        retry_kwargs["openai_proxy"] = fallback_proxy
        _warn_missing_socksio_once(fallback_proxy)
        return llm_cls(**retry_kwargs)


class RotatingLlmClient:
    """Minimal wrapper that retries LLM calls with fallback key/model configs."""

    def __init__(
        self,
        llm_cls: type[Any] | None,
        kwargs_list: list[dict[str, Any]],
        *,
        client_factory: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        if not kwargs_list:
            raise ValueError("kwargs_list must not be empty")
        self._llm_cls = llm_cls
        self._kwargs_list = [dict(kwargs) for kwargs in kwargs_list]
        self._client_factory = client_factory
        self._clients: dict[int, Any] = {}
        self._active_index = 0
        self._lock = Lock()

    def _build_client(self, index: int) -> Any:
        if self._client_factory is not None:
            return self._client_factory(dict(self._kwargs_list[index]))
        if self._llm_cls is None:
            raise ValueError("llm_cls is required when client_factory is not provided")
        return _build_llm_once_with_proxy_fallback(self._llm_cls, self._kwargs_list[index])

    def _get_client(self, index: int | None = None) -> Any:
        target_index = self._active_index if index is None else index
        with self._lock:
            if target_index not in self._clients:
                self._clients[target_index] = self._build_client(target_index)
            return self._clients[target_index]

    def _ordered_indices(self) -> list[int]:
        count = len(self._kwargs_list)
        return [(self._active_index + offset) % count for offset in range(count)]

    def _promote(self, index: int) -> None:
        with self._lock:
            self._active_index = index

    def _call_with_fallback(self, caller: Callable[[Any], Any]) -> Any:
        last_exc: Exception | None = None
        for index in self._ordered_indices():
            client = self._get_client(index)
            try:
                result = caller(client)
            except Exception as exc:
                last_exc = exc
                if len(self._kwargs_list) <= 1 or not _is_provider_fallback_error(exc):
                    raise
                from .logging import log_warning

                log_warning(
                    "model_client",
                    "模型调用失败，准备切换备用 key/model："
                    f"{_safe_model_label(self._kwargs_list[index])}；原因：{exc}",
                )
                continue
            self._promote(index)
            return result
        assert last_exc is not None
        raise last_exc

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        return self._call_with_fallback(lambda client: client.invoke(*args, **kwargs))

    def stream(self, *args: Any, **kwargs: Any) -> Iterable[Any]:
        last_exc: Exception | None = None
        for index in self._ordered_indices():
            client = self._get_client(index)
            yielded = False
            try:
                for chunk in client.stream(*args, **kwargs):
                    yielded = True
                    yield chunk
                self._promote(index)
                return
            except Exception as exc:
                last_exc = exc
                if yielded or len(self._kwargs_list) <= 1 or not _is_provider_fallback_error(exc):
                    raise
                from .logging import log_warning

                log_warning(
                    "model_client",
                    "模型流式调用启动失败，准备切换备用 key/model："
                    f"{_safe_model_label(self._kwargs_list[index])}；原因：{exc}",
                )
                continue
        assert last_exc is not None
        raise last_exc

    def bind_tools(self, *args: Any, **kwargs: Any) -> "RotatingLlmClient":
        def _factory(candidate_kwargs: dict[str, Any]) -> Any:
            if self._client_factory is not None:
                base_client = self._client_factory(candidate_kwargs)
            else:
                if self._llm_cls is None:
                    raise ValueError("llm_cls is required when client_factory is not provided")
                base_client = _build_llm_once_with_proxy_fallback(
                    self._llm_cls,
                    candidate_kwargs,
                )
            return base_client.bind_tools(*args, **kwargs)

        return RotatingLlmClient(None, self._kwargs_list, client_factory=_factory)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_client(), name)


def build_llm_with_proxy_fallback(llm_cls: type[Any], kwargs: dict[str, Any]) -> Any:
    fallback_kwargs = kwargs.get("_fallback_kwargs") or []
    primary_kwargs = dict(kwargs)
    primary_kwargs.pop("_fallback_kwargs", None)
    if fallback_kwargs:
        return RotatingLlmClient(llm_cls, [primary_kwargs, *list(fallback_kwargs)])
    return _build_llm_once_with_proxy_fallback(llm_cls, primary_kwargs)
