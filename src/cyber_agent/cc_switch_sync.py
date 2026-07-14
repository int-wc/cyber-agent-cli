from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sqlite3
import tomllib
from typing import Any


DEFAULT_CC_SWITCH_DB_PATH = Path.home() / ".cc-switch" / "cc-switch.db"
OPENAI_COMPATIBLE_PROTOCOLS = frozenset({"openai_chat", "openai_responses"})


@dataclass(frozen=True)
class CcSwitchProvider:
    id: str
    app_type: str
    name: str
    model: str
    api_key: str
    base_url: str
    protocol: str
    raw_base_url: str = ""
    wire_api: str = ""
    is_current: bool = False
    in_failover_queue: bool = False

    @property
    def service_name(self) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", self.name.strip().lower())
        return f"ccswitch_{self.app_type}_{slug.strip('_') or 'provider'}_{self.id[:8]}"

    @property
    def is_openai_compatible(self) -> bool:
        return self.protocol in OPENAI_COMPATIBLE_PROTOCOLS


def normalize_openai_base_url(raw_base_url: str) -> str:
    """Convert CC-SWITCH OpenAI/Responses-style URLs into /v1 base URLs."""
    base_url = raw_base_url.strip().rstrip("/")
    if not base_url:
        return ""
    if base_url.endswith("/v1/responses"):
        base_url = base_url[: -len("/responses")]
    elif base_url.endswith("/responses"):
        base_url = base_url[: -len("/responses")]
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url


def _normalize_protocol(raw_protocol: object, wire_api: object = "") -> str:
    protocol = str(raw_protocol or wire_api or "").strip().lower()
    if protocol in {"responses", "openai-responses"}:
        return "openai_responses"
    if protocol in {"chat", "openai", "openai-chat"}:
        return "openai_chat"
    if protocol in {"anthropic", "claude"}:
        return "anthropic"
    return protocol or "unknown"


def _first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = mapping.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _load_codex_provider_from_row(
    row: sqlite3.Row,
    settings_config: dict[str, Any],
    meta: dict[str, Any],
) -> CcSwitchProvider | None:
    auth = settings_config.get("auth") or {}
    api_key = str(auth.get("OPENAI_API_KEY") or "").strip()
    if not api_key or api_key == "sk-placeholder":
        return None

    config = tomllib.loads(str(settings_config.get("config") or ""))
    model = str(config.get("model") or "").strip()
    custom_provider = (config.get("model_providers") or {}).get("custom") or {}
    raw_base_url = str(custom_provider.get("base_url") or "").strip()
    wire_api = str(custom_provider.get("wire_api") or "").strip()
    protocol = _normalize_protocol(meta.get("apiFormat"), wire_api)
    base_url = (
        normalize_openai_base_url(raw_base_url)
        if protocol in OPENAI_COMPATIBLE_PROTOCOLS
        else raw_base_url.rstrip("/")
    )
    if not model or not api_key or not base_url:
        return None
    return CcSwitchProvider(
        id=str(row["id"]),
        app_type=str(row["app_type"]),
        name=str(row["name"] or row["id"]),
        model=model,
        api_key=api_key,
        base_url=base_url,
        protocol=protocol,
        raw_base_url=raw_base_url,
        wire_api=wire_api,
        is_current=bool(row["is_current"]),
        in_failover_queue=bool(row["in_failover_queue"]),
    )


def _load_claude_provider_from_row(
    row: sqlite3.Row,
    settings_config: dict[str, Any],
    meta: dict[str, Any],
) -> CcSwitchProvider | None:
    env = settings_config.get("env") or {}
    api_key = _first_present(env, ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"))
    if not api_key or api_key == "sk-placeholder":
        return None

    raw_base_url = _first_present(env, ("ANTHROPIC_BASE_URL", "OPENAI_BASE_URL", "BASE_URL"))
    protocol = _normalize_protocol(meta.get("apiFormat"))
    model = _first_present(
        env,
        (
            "ANTHROPIC_MODEL",
            "ANTHROPIC_DEFAULT_SONNET_MODEL_NAME",
            "ANTHROPIC_DEFAULT_SONNET_MODEL",
            "ANTHROPIC_DEFAULT_OPUS_MODEL_NAME",
            "ANTHROPIC_DEFAULT_OPUS_MODEL",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL_NAME",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        ),
    )
    base_url = (
        normalize_openai_base_url(raw_base_url)
        if protocol in OPENAI_COMPATIBLE_PROTOCOLS
        else raw_base_url.rstrip("/")
    )
    if not model or not api_key or not base_url:
        return None
    return CcSwitchProvider(
        id=str(row["id"]),
        app_type=str(row["app_type"]),
        name=str(row["name"] or row["id"]),
        model=model.replace("[1M]", "").replace("[1m]", "").strip(),
        api_key=api_key,
        base_url=base_url,
        protocol=protocol,
        raw_base_url=raw_base_url,
        wire_api="",
        is_current=bool(row["is_current"]),
        in_failover_queue=bool(row["in_failover_queue"]),
    )


def _load_provider_from_row(row: sqlite3.Row) -> CcSwitchProvider | None:
    try:
        settings_config = json.loads(row["settings_config"] or "{}")
        meta = json.loads(row["meta"] or "{}")
        app_type = str(row["app_type"] or "").strip().lower()
        if app_type == "codex":
            return _load_codex_provider_from_row(row, settings_config, meta)
        if app_type == "claude":
            return _load_claude_provider_from_row(row, settings_config, meta)
        return None
    except (json.JSONDecodeError, tomllib.TOMLDecodeError, KeyError, TypeError):
        return None


def load_cc_switch_providers(
    db_path: str | Path | None = None,
    *,
    protocols: set[str] | frozenset[str] | None = None,
    app_types: set[str] | frozenset[str] | None = None,
) -> list[CcSwitchProvider]:
    """Load all usable CC-SWITCH providers, de-duplicated and classified."""
    resolved_path = Path(db_path).expanduser() if db_path else DEFAULT_CC_SWITCH_DB_PATH
    if not resolved_path.exists():
        return []

    try:
        connection = sqlite3.connect(str(resolved_path))
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                """
                select id, app_type, name, settings_config, meta, is_current,
                       in_failover_queue, sort_index, created_at
                from providers
                order by is_current desc,
                         case app_type when 'codex' then 0 when 'claude' then 1 else 2 end,
                         in_failover_queue desc, sort_index, created_at
                """
            ).fetchall()
        finally:
            connection.close()
    except sqlite3.Error:
        return []

    normalized_protocols = {item.lower() for item in protocols} if protocols else None
    normalized_app_types = {item.lower() for item in app_types} if app_types else None
    providers: list[CcSwitchProvider] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows:
        provider = _load_provider_from_row(row)
        if provider is None:
            continue
        if normalized_protocols and provider.protocol not in normalized_protocols:
            continue
        if normalized_app_types and provider.app_type not in normalized_app_types:
            continue
        key = (provider.protocol, provider.base_url, provider.model, provider.api_key)
        if key in seen:
            continue
        seen.add(key)
        providers.append(provider)
    return providers


def load_cc_switch_openai_compatible_providers(
    db_path: str | Path | None = None,
) -> list[CcSwitchProvider]:
    return load_cc_switch_providers(
        db_path,
        protocols=OPENAI_COMPATIBLE_PROTOCOLS,
    )


def group_cc_switch_providers_by_protocol(
    db_path: str | Path | None = None,
) -> dict[str, list[CcSwitchProvider]]:
    grouped: dict[str, list[CcSwitchProvider]] = defaultdict(list)
    for provider in load_cc_switch_providers(db_path):
        grouped[provider.protocol].append(provider)
    return dict(grouped)


def load_cc_switch_codex_providers(
    db_path: str | Path | None = None,
    *,
    current_only: bool = False,
) -> list[CcSwitchProvider]:
    """Backward-compatible wrapper for existing callers."""
    providers = load_cc_switch_providers(db_path, app_types={"codex"})
    if current_only:
        providers = [provider for provider in providers if provider.is_current]
    return providers


def get_current_cc_switch_codex_provider(
    db_path: str | Path | None = None,
) -> CcSwitchProvider | None:
    providers = load_cc_switch_codex_providers(db_path, current_only=True)
    return providers[0] if providers else None
