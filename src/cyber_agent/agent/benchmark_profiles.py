"""Benchmark 配置档和候选归一化工具。

这里仅放无状态、可复用的数据解析逻辑。实际调度、探测、提交和关闭仍由
``FourPillarPipeline`` 负责，避免配置解析与运行流程继续耦合在同一个大文件里。
"""

from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Any
from urllib.parse import urljoin, urlparse, urlunparse


DEFAULT_WORKSPACE = Path("/home/my/cyber/benchmark_test")


def workspace_path() -> Path:
    return DEFAULT_WORKSPACE


def external_profiles_path(runtime_context: dict[str, object]) -> Path:
    raw_path = runtime_context.get("benchmark_profiles_path")
    if raw_path:
        return Path(str(raw_path)).expanduser()
    return workspace_path() / "benchmark-profiles.json"


def load_external_profiles(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def string_tuple(value: Any, *, limit: int = 80) -> tuple[str, ...]:
    if isinstance(value, str):
        items: list[Any] = [value]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        return ()
    result: list[str] = []
    for item in items[:limit]:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        if cleaned:
            result.append(cleaned)
    return tuple(dict.fromkeys(result))


def string_pair_tuple(
    value: Any,
    *,
    limit: int = 20,
) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    pairs: list[tuple[str, str]] = []
    for item in list(value)[:limit]:
        username = password = ""
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            username, password = str(item[0]).strip(), str(item[1]).strip()
        elif isinstance(item, dict):
            username = str(item.get("username") or item.get("user") or "").strip()
            password = str(item.get("password") or item.get("pass") or "").strip()
        if username and password:
            pairs.append((username, password))
    return tuple(dict.fromkeys(pairs))


def match_any_all_tuple(value: Any) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    groups: list[tuple[str, ...]] = []
    for group in list(value)[:40]:
        normalized = string_tuple(group, limit=20)
        if normalized:
            groups.append(normalized)
    return tuple(groups)


def selection_policy(data: dict[str, Any]) -> dict[str, Any]:
    raw_policy = data.get("selection_policy", data.get("scheduling_policy", {}))
    if not isinstance(raw_policy, dict):
        raw_policy = {}

    def difficulty_list(key: str, default: tuple[str, ...]) -> tuple[str, ...]:
        configured = string_tuple(raw_policy.get(key), limit=10)
        normalized = tuple(
            item.lower()
            for item in configured
            if item.lower() in {"easy", "medium", "hard"}
        )
        return normalized or default

    def positive_int(key: str, default: int, *, low: int = 1, high: int = 20) -> int:
        try:
            value = int(raw_policy.get(key))
        except (TypeError, ValueError):
            return default
        return min(high, max(low, value))

    return {
        "difficulty_order": difficulty_list(
            "difficulty_order",
            ("easy", "medium", "hard"),
        ),
        "fast_path_difficulties": difficulty_list(
            "fast_path_difficulties",
            ("easy",),
        ),
        "handoff_difficulties": difficulty_list(
            "handoff_difficulties",
            ("medium", "hard"),
        ),
        "recovery_difficulties": difficulty_list(
            "recovery_difficulties",
            ("easy",),
        ),
        "unreachable_retries": positive_int("unreachable_retries", 2, high=10),
        "estimated_fast_score": positive_int(
            "estimated_fast_score",
            200,
            high=1000,
        ),
    }


def execution_control_policy(
    data: dict[str, Any],
    runtime_context: dict[str, object],
) -> dict[str, Any]:
    raw_policy = data.get(
        "execution_control_policy",
        data.get("tool_scheduler_policy", {}),
    )
    if not isinstance(raw_policy, dict):
        raw_policy = {}
    runtime_policy = runtime_context.get("execution_control_policy")
    if not isinstance(runtime_policy, dict):
        runtime_policy = runtime_context.get("tool_scheduler_policy")
    if isinstance(runtime_policy, dict):
        raw_policy = {**raw_policy, **runtime_policy}

    def bounded_int(key: str, default: int, *, low: int, high: int) -> int:
        try:
            value = int(raw_policy.get(key))
        except (TypeError, ValueError):
            return default
        return min(high, max(low, value))

    def enum_value(key: str, default: str, allowed: set[str]) -> str:
        value = str(raw_policy.get(key, default)).strip().lower()
        return value if value in allowed else default

    return {
        "max_probe_paths": bounded_int(
            "max_probe_paths",
            180,
            low=1,
            high=500,
        ),
        "max_probe_urls": bounded_int(
            "max_probe_urls",
            60,
            low=5,
            high=500,
        ),
        "max_authenticated_urls": bounded_int(
            "max_authenticated_urls",
            100,
            low=5,
            high=500,
        ),
        "max_payloads_per_param": bounded_int(
            "max_payloads_per_param",
            40,
            low=1,
            high=200,
        ),
        "max_flag_paths": bounded_int(
            "max_flag_paths",
            40,
            low=1,
            high=100,
        ),
        "fast_probe_seconds": bounded_int(
            "fast_probe_seconds",
            45,
            low=10,
            high=180,
        ),
        "max_subagents": bounded_int(
            "max_subagents",
            0,
            low=0,
            high=16,
        ),
        "subtask_concurrency": enum_value(
            "subtask_concurrency",
            "",
            {"", "off", "auto", "force"},
        ),
    }


def disabled_builtin_fingerprints(data: dict[str, Any]) -> set[str]:
    raw = data.get(
        "disabled_builtin_fingerprints",
        data.get(
            "disable_builtin_fingerprints",
            data.get("disabled_service_fingerprints", ()),
        ),
    )
    return {
        value.lower()
        for value in string_tuple(raw, limit=80)
        if re.fullmatch(r"[a-z0-9_.-]{1,80}", value.lower())
    }


def builtin_section_enabled(data: dict[str, Any], section: str) -> bool:
    raw = data.get(
        "disabled_builtin_sections",
        data.get("disable_builtin_sections", ()),
    )
    disabled = {
        value.lower()
        for value in string_tuple(raw, limit=80)
        if re.fullmatch(r"[a-z0-9_.-]{1,80}", value.lower())
    }
    normalized = section.lower()
    aliases = {
        "object_storage_buckets": "object_storage",
        "object_storage_keys": "object_storage",
        "lfi_paths": "lfi_base_paths",
        "lfi_keys": "lfi_detection",
        "lfi_param_keys": "lfi_detection",
        "lfi_trigger_markers": "lfi_detection",
        "payload_profiles": "payloads",
        "param_payload_profiles": "payloads",
        "telnet": "telnet_credentials",
        "webapp_profiles": "webapp_flow_profiles",
    }
    canonical = aliases.get(normalized, normalized)
    return "all" not in disabled and canonical not in disabled


def builtin_probe_paths() -> list[str]:
    return [
        "",
        "robots.txt",
        "sitemap.xml",
        "openapi.json",
        "docs",
        "login.php",
        "dashboard.php",
        "download.php",
        "upload.php",
        "flag",
        "flag.txt",
        "admin",
        "api/flag",
        ".git/config",
        "backup.zip",
        "page?name=x%22;alert(%22XSS%22)%3Bfoo=%22",
        "?name=x%22;alert(%22XSS%22)%3Bfoo=%22",
        "?name={{{{7*7}}}}",
        "page?name={{{{7*7}}}}",
        "search?q=%27%20OR%20%271%27%3D%271",
        "?id=1%20OR%201=1",
        "user?id=1",
        "api/user?id=1",
    ]


def url_is_same_container(base: str, url: str) -> bool:
    base_parsed = urlparse(base)
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and parsed.netloc == base_parsed.netloc


def safe_observed_probe_path(base: str, url: str) -> str | None:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        parsed = urlparse(urljoin(base, url))
    normalized_url = urlunparse(parsed)
    if not url_is_same_container(base, normalized_url):
        return None
    path = parsed.path.lstrip("/")
    if parsed.query:
        path = f"{path}?{parsed.query}" if path else f"?{parsed.query}"
    if not path or len(path) > 220:
        return None
    if path.startswith(("../", "./")):
        return None
    if not re.fullmatch(r"[A-Za-z0-9_./?=&:%+\-,]+", path):
        return None
    return path


def builtin_flag_paths() -> tuple[str, ...]:
    return (
        "/challenge/flag.txt",
        "/challenge/flag",
        "/flag",
        "/flag.txt",
        "/root/flag",
        "/root/flag.txt",
        "/tmp/flag",
        "/tmp/flag.txt",
        "/home/flag",
        "/home/flag.txt",
        "/run/secrets/flag",
        "/hugegraph-server/flag",
        "/hugegraph-server/flag.txt",
    )


def is_safe_flag_path(path: str) -> bool:
    return bool(re.fullmatch(r"/[A-Za-z0-9._/\-{}]{1,220}", path))


def normalize_probe_headers(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    headers: dict[str, str] = {}
    for key, value in list(raw.items())[:8]:
        header = str(key or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9-]{1,60}", header):
            continue
        if header.lower() in {"host", "content-length", "authorization", "benchmark_token"}:
            continue
        headers[header] = str(value or "").strip()[:300]
    return headers


def normalize_probe_body(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    body: dict[str, Any] = {}
    for key, value in list(raw.items())[:20]:
        body_key = str(key or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,80}", body_key):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            body[body_key] = value if not isinstance(value, str) else value[:1000]
        elif isinstance(value, (list, tuple)):
            items: list[Any] = []
            for item in list(value)[:12]:
                if isinstance(item, (str, int, float, bool)) or item is None:
                    items.append(item if not isinstance(item, str) else item[:500])
            body[body_key] = items
        elif isinstance(value, dict):
            nested: dict[str, Any] = {}
            for nested_key, nested_value in list(value.items())[:12]:
                nested_name = str(nested_key or "").strip()
                if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,80}", nested_name):
                    continue
                if isinstance(nested_value, (str, int, float, bool)) or nested_value is None:
                    nested[nested_name] = (
                        nested_value
                        if not isinstance(nested_value, str)
                        else nested_value[:500]
                    )
            if nested:
                body[body_key] = nested
    return body or None


def normalize_probe_requests(raw: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(raw, list):
        return ()
    requests: list[dict[str, Any]] = []
    for item in raw[:20]:
        if not isinstance(item, dict):
            continue
        raw_path = str(item.get("path") or item.get("url_path") or "").strip()
        if not raw_path or raw_path.startswith(("http://", "https://", "//")):
            continue
        path = raw_path.lstrip("/")
        if not re.fullmatch(r"[A-Za-z0-9_./?=&:%+\-]{1,220}", path):
            continue
        method = str(item.get("method") or "GET").strip().upper()
        if method not in {"GET", "POST", "PUT"}:
            continue
        request: dict[str, Any] = {"method": method, "path": path}
        label = str(item.get("label") or "").strip()
        if label:
            request["label"] = label[:100]
        headers = normalize_probe_headers(item.get("headers"))
        if headers:
            request["headers"] = headers
        json_body = normalize_probe_body(item.get("json"))
        data_body = normalize_probe_body(item.get("data", item.get("form")))
        if json_body is not None:
            request["json"] = json_body
        elif data_body is not None:
            request["data"] = {
                key: str(value)[:1000]
                for key, value in data_body.items()
                if isinstance(value, (str, int, float, bool)) or value is None
            }
        requests.append(request)
    return tuple(requests)


def normalize_tcp_ports(raw: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(raw, list):
        return ()
    ports: list[dict[str, Any]] = []
    for item in raw[:20]:
        label = ""
        raw_port: Any = item
        if isinstance(item, dict):
            raw_port = item.get("port")
            label = str(item.get("label") or "").strip()[:80]
        try:
            port = int(raw_port)
        except (TypeError, ValueError):
            continue
        if port < 1 or port > 65535:
            continue
        entry: dict[str, Any] = {"port": port}
        if label:
            entry["label"] = label
        ports.append(entry)
    seen: set[int] = set()
    unique: list[dict[str, Any]] = []
    for entry in ports:
        port = int(entry["port"])
        if port in seen:
            continue
        seen.add(port)
        unique.append(entry)
    return tuple(unique)
