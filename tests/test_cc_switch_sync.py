import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from cyber_agent.cc_switch_sync import (
    get_current_cc_switch_codex_provider,
    group_cc_switch_providers_by_protocol,
    load_cc_switch_codex_providers,
    load_cc_switch_openai_compatible_providers,
    normalize_openai_base_url,
)


class CcSwitchSyncTestCase(unittest.TestCase):
    def test_normalize_openai_base_url(self) -> None:
        self.assertEqual(
            normalize_openai_base_url("https://example.test"),
            "https://example.test/v1",
        )
        self.assertEqual(
            normalize_openai_base_url("https://example.test/v1/responses"),
            "https://example.test/v1",
        )

    def test_load_current_codex_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "cc-switch.db"
            connection = sqlite3.connect(db_path)
            try:
                connection.execute(
                    """
                    create table providers (
                        id text not null,
                        app_type text not null,
                        name text not null,
                        settings_config text not null,
                        meta text not null default '{}',
                        is_current integer not null default 0,
                        in_failover_queue integer not null default 0,
                        sort_index integer,
                        created_at integer,
                        primary key (id, app_type)
                    )
                    """
                )
                connection.execute(
                    """
                    insert into providers (
                        id, app_type, name, settings_config, meta,
                        is_current, in_failover_queue, sort_index, created_at
                    ) values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "current-id",
                        "codex",
                        "Current Provider",
                        json.dumps(
                            {
                                "auth": {"OPENAI_API_KEY": "sk-current"},
                                "config": (
                                    'model = "gpt-5.5"\n'
                                    '[model_providers.custom]\n'
                                    'base_url = "https://example.test/v1/responses"\n'
                                ),
                            }
                        ),
                        json.dumps({"apiFormat": "openai_responses"}),
                        1,
                        0,
                        0,
                        1,
                    ),
                )
                connection.commit()
            finally:
                connection.close()

            provider = get_current_cc_switch_codex_provider(db_path)
            providers = load_cc_switch_codex_providers(db_path)
            compatible_providers = load_cc_switch_openai_compatible_providers(db_path)
            grouped = group_cc_switch_providers_by_protocol(db_path)

        self.assertIsNotNone(provider)
        assert provider is not None
        self.assertEqual(provider.model, "gpt-5.5")
        self.assertEqual(provider.api_key, "sk-current")
        self.assertEqual(provider.base_url, "https://example.test/v1")
        self.assertEqual(provider.app_type, "codex")
        self.assertEqual(provider.protocol, "openai_responses")
        self.assertEqual([item.id for item in providers], ["current-id"])
        self.assertEqual([item.id for item in compatible_providers], ["current-id"])
        self.assertEqual(list(grouped), ["openai_responses"])


if __name__ == "__main__":
    unittest.main()
