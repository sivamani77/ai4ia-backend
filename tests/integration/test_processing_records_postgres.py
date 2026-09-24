"""prompt_resolutions / llm_call_payloads against a real Postgres, in a throwaway schema.

Skips unless the local infra Postgres (docker-compose.infra.yml, :15432) is up.
"""
from __future__ import annotations

import threading
import uuid

import pytest

psycopg2 = pytest.importorskip("psycopg2")

BASE_URL = "postgresql://ai4ia:ai4ia_local_dev@localhost:15432/ai4ia"


def _reachable() -> bool:
    try:
        psycopg2.connect(BASE_URL, connect_timeout=2).close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _reachable(), reason="local Postgres on :15432 not running")


@pytest.fixture
def scratch_db(monkeypatch):
    """Point the app at a fresh schema; drop it afterwards."""
    from app.db import postgres
    from app.db import schema as schema_module

    schema = f"test_pr_{uuid.uuid4().hex[:8]}"
    admin = psycopg2.connect(BASE_URL)
    admin.autocommit = True
    admin.cursor().execute(f"create schema {schema}")
    url = f"{BASE_URL}?options=-csearch_path%3D{schema}"
    # Patch the settings objects these modules hold; other tests may have swapped the global one.
    monkeypatch.setattr(postgres.settings, "database_url", url)
    monkeypatch.setattr(schema_module.settings, "database_url", url)
    postgres.reset_pool()
    try:
        yield schema
    finally:
        postgres.reset_pool()
        admin.cursor().execute(f"drop schema {schema} cascade")
        admin.close()


def _pipeline_ddl() -> list[str]:
    from app.db import schema
    return [s for s in schema._STATEMENTS if "prompt_resolutions" in s or "llm_call_payloads" in s]


def test_concurrent_startups_create_the_tables_once(scratch_db):
    from app.db import schema

    errors: list[Exception] = []
    barrier = threading.Barrier(8)

    def startup():
        barrier.wait()
        try:
            schema._run_locked(_pipeline_ddl())
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=startup) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    assert errors == []


def test_records_round_trip(scratch_db):
    from app.db import schema
    from app.db.postgres import pg_fetch_one
    from app.db.repositories.llm_payload_repository import LLMPayloadRepository
    from app.db.repositories.prompt_resolution_repository import PromptResolutionRepository

    schema._run_locked(_pipeline_ddl())
    resolutions, payloads = PromptResolutionRepository(), LLMPayloadRepository()

    resolutions.create({"id": "pr_1", "job_id": "vjob_1", "surface": "video", "user_id": "u1",
                        "raw_prompt": "What did Cordel Green say?", "layer_outcome": "in_progress",
                        "expires_at": "2026-12-31T23:59:59+00:00"})
    resolutions.update_by_job("vjob_1", fields={"layer_outcome": "l1_rejected", "reason_code": "off_topic"},
                              merge={"timings_ms": {"l1": 900}})
    resolutions.update_by_job("vjob_1", merge={"timings_ms": {"l2": 40}, "flags": {"x": True}})
    resolutions.update_by_job("vjob_1", fields={"layer_outcome": "failed"}, only_if_outcome="in_progress")

    row = resolutions.get_by_job("vjob_1")
    assert row["layer_outcome"] == "l1_rejected"
    assert row["timings_ms"] == {"l1": 900, "l2": 40} and row["flags"] == {"x": True}

    invalid_json = '{"allowed": tru'
    payloads.create({"id": "lx_1", "job_id": "vjob_1", "user_id": "u1", "stage": "07b",
                     "request_text": '{"messages": ["ümlaut"]}', "response_text": invalid_json,
                     "expires_at": "2026-12-31T23:59:59+00:00"})
    (stored,) = payloads.list_by_job("vjob_1")
    assert stored["response_text"] == invalid_json
    assert stored["size_bytes"] == len('{"messages": ["ümlaut"]}'.encode()) + len(invalid_json)

    # Expiry sweep removes only expired rows.
    payloads.create({"id": "lx_old", "job_id": "vjob_0", "request_text": "{}", "expires_at": "2020-01-01T00:00:00+00:00"})
    assert payloads.purge_expired() == 1
    assert pg_fetch_one("select count(*) as n from llm_call_payloads")["n"] == 1

    # Account deletion: disconnect keeps the record, delete removes it.
    resolutions.disconnect_by_user("u1")
    assert resolutions.get_by_job("vjob_1")["user_id"] is None
    payloads.delete_by_user("u1")
    assert payloads.list_by_job("vjob_1") == []
