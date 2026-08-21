"""Unit tests for DB repository normalize/marshal helpers — Postgres-backed,
no real database connection required. Pure _to_row/_from_row/_normalize_*
helpers are exercised directly; a handful of tests that touch a query path
mock pg_fetch_one/pg_fetch_all/pg_execute at the repository module level."""
from __future__ import annotations

from decimal import Decimal
from datetime import date
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# EventRepository._normalize_input / _normalize_output
# ---------------------------------------------------------------------------

class TestEventRepositoryNormalize:
    def _repo(self):
        from app.db.repositories.event_repository import EventRepository
        return EventRepository()

    def test_normalize_input_folds_stage_into_payload(self):
        repo = self._repo()
        result = repo._normalize_input({"job_id": "j1", "stage": "transcribe", "payload": {"x": 1}})
        assert "stage" not in result
        assert result["payload"]["stage"] == "transcribe"

    def test_normalize_input_creates_payload_when_missing(self):
        repo = self._repo()
        result = repo._normalize_input({"job_id": "j1", "stage": "transcribe"})
        assert result["payload"] == {"stage": "transcribe"}

    def test_normalize_output_returns_event_record(self):
        repo = self._repo()
        row = {"job_id": "j1", "event_type": "stage_update", "payload": {"x": 2}}
        result = repo._normalize_output(row)
        assert result["event_type"] == "stage_update"

    def test_normalize_value_decimal(self):
        repo = self._repo()
        assert repo._normalize_value(Decimal("3.14")) == pytest.approx(3.14)

    def test_normalize_value_date(self):
        repo = self._repo()
        d = date(2025, 1, 15)
        result = repo._normalize_value(d)
        assert result == "2025-01-15"


# ---------------------------------------------------------------------------
# InsightRepository._to_row / _from_row
# ---------------------------------------------------------------------------

class TestInsightRepositoryMarshal:
    def _repo(self):
        from app.db.repositories.insight_repository import InsightRepository
        return InsightRepository()

    def test_to_row_defaults_lists_to_empty(self):
        repo = self._repo()
        row = repo._to_row({"id": "i1", "query": "test"})
        assert row["patterns"] == []
        assert row["contrasts"] == []
        assert row["recommended_clips"] == []

    def test_to_row_evidence_coverage_float(self):
        repo = self._repo()
        row = repo._to_row({"id": "i1", "query": "q", "evidence_coverage": "0.75"})
        assert row["evidence_coverage"] == pytest.approx(0.75)

    def test_from_row_fills_defaults(self):
        repo = self._repo()
        record = repo._from_row({"id": "i1", "query": "q"})
        assert record["patterns"] == []
        assert record["evidence_pool"] == []

    def test_from_row_preserves_id(self):
        repo = self._repo()
        record = repo._from_row({"id": "insight_xyz", "query": "q"})
        assert record["id"] == "insight_xyz"

    def test_to_row_uses_insight_id_key(self):
        repo = self._repo()
        row = repo._to_row({"insight_id": "id_from_insight_key", "query": "q"})
        assert row["id"] == "id_from_insight_key"


# ---------------------------------------------------------------------------
# StorylineRepository._to_row / _from_row
# ---------------------------------------------------------------------------

class TestStorylineRepositoryMarshal:
    def _repo(self):
        from app.db.repositories.storyline_repository import StorylineRepository
        return StorylineRepository()

    def test_to_row_defaults(self):
        repo = self._repo()
        row = repo._to_row({"id": "s1"})
        assert row["sections"] == []
        assert row["is_validated"] is False

    def test_from_row_is_validated_false_by_default(self):
        repo = self._repo()
        record = repo._from_row({"id": "s1"})
        assert record["is_validated"] is False

    def test_from_row_preserves_sections(self):
        repo = self._repo()
        record = repo._from_row({"id": "s1", "sections": [{"title": "Intro"}]})
        assert record["sections"] == [{"title": "Intro"}]


# ---------------------------------------------------------------------------
# JobRepository._normalize_input
# ---------------------------------------------------------------------------

class TestJobRepositoryNormalize:
    def _repo(self):
        from app.db.repositories.job_repository import JobRepository
        return JobRepository()

    def test_filters_out_extra_columns(self):
        repo = self._repo()
        data = {"id": "j1", "status": "queued", "nonexistent_column": "should_drop"}
        payload = repo._normalize_input(data)
        assert "nonexistent_column" not in payload

    def test_keeps_schema_columns(self):
        repo = self._repo()
        data = {"id": "j1", "status": "queued", "progress": 0}
        payload = repo._normalize_input(data)
        assert payload["id"] == "j1"
        assert payload["status"] == "queued"

    def test_normalize_input_preserves_user_id_column(self):
        repo = self._repo()
        payload = repo._normalize_input({"id": "j1", "user_id": "user_abc"})
        assert payload["user_id"] == "user_abc"

    def test_normalize_output_preserves_user_id_column(self):
        repo = self._repo()
        output = repo._normalize_output({"id": "j1", "user_id": "user_abc"})
        assert output["user_id"] == "user_abc"

    def test_normalizes_decimal_values(self):
        repo = self._repo()
        payload = repo._normalize_input({"id": "j1", "progress": Decimal("42")})
        assert payload["progress"] == 42.0 or payload["progress"] == 42

    def test_get_job_uses_id(self):
        from app.db.repositories.job_repository import JobRepository
        repo = JobRepository()
        with patch("app.db.repositories.job_repository.pg_fetch_one", return_value={"id": "j1", "status": "queued"}):
            result = repo.get_job("j1")
        assert result["id"] == "j1"


# ---------------------------------------------------------------------------
# FragmentRepository — Postgres primary, legacy Supabase fallback disabled
# ---------------------------------------------------------------------------

class TestFragmentRepository:
    def _repo(self):
        from app.db.repositories.fragment_repository import FragmentRepository
        return FragmentRepository()

    def test_get_returns_none_on_empty(self):
        repo = self._repo()
        with patch("app.db.repositories.fragment_repository.pg_fetch_one", return_value=None), \
             patch.object(repo, "_legacy_table", return_value=None):
            result = repo.get("frag_xyz")
        assert result is None

    def test_list_all_returns_list(self):
        repo = self._repo()
        with patch("app.db.repositories.fragment_repository.pg_fetch_one", return_value={"has_data": True}), \
             patch("app.db.repositories.fragment_repository.pg_fetch_all",
                   return_value=[{"fragment_id": "f1", "text": "hello"}]):
            result = repo.list_all()
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["fragment_id"] == "f1"

    def test_list_all_falls_back_to_legacy_when_postgres_empty(self):
        repo = self._repo()
        legacy_rows = [{"fragment_id": "legacy1", "text": "old row"}]
        with patch("app.db.repositories.fragment_repository.pg_fetch_one", return_value={"has_data": False}), \
             patch.object(repo, "_legacy_list_all", return_value=[repo._from_row(r) for r in legacy_rows]) as legacy:
            result = repo.list_all()
        legacy.assert_called_once()
        assert result[0]["fragment_id"] == "legacy1"


# ---------------------------------------------------------------------------
# JobRepository create / upsert / update
# ---------------------------------------------------------------------------

class TestJobRepositoryCRUD:
    def _repo(self):
        from app.db.repositories.job_repository import JobRepository
        return JobRepository()

    def test_upsert_job_calls_execute(self):
        repo = self._repo()
        with patch("app.db.repositories.job_repository.pg_execute") as mock_execute:
            repo.upsert_job({"id": "j1", "status": "queued"})
        mock_execute.assert_called_once()

    def test_list_jobs_returns_list(self):
        repo = self._repo()
        with patch("app.db.repositories.job_repository.pg_fetch_all",
                   return_value=[{"id": "j1"}, {"id": "j2"}]):
            result = repo.list_jobs()
        assert len(result) == 2
