"""Unit tests for app.services.ingest_job_runner."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest


class TestIngestStageWeights:
    def test_weights_sum_to_100(self):
        from app.services.ingest_job_runner import INGEST_STAGE_WEIGHTS
        assert sum(INGEST_STAGE_WEIGHTS.values()) == 100

    def test_all_weights_positive(self):
        from app.services.ingest_job_runner import INGEST_STAGE_WEIGHTS
        for stage, weight in INGEST_STAGE_WEIGHTS.items():
            assert weight > 0, f"Stage {stage!r} has non-positive weight {weight}"

    def test_has_expected_stages(self):
        from app.services.ingest_job_runner import INGEST_STAGE_WEIGHTS
        expected = {
            "discover", "download", "extract_audio",
            "transcribe", "create_fragments", "classify", "embed",
        }
        assert set(INGEST_STAGE_WEIGHTS.keys()) == expected

    def test_sum_exactly_100(self):
        from app.services.ingest_job_runner import INGEST_STAGE_WEIGHTS
        # Verify no floating point tricks
        assert sum(INGEST_STAGE_WEIGHTS.values()) == 100
        assert isinstance(sum(INGEST_STAGE_WEIGHTS.values()), int)


class TestCreateIngestJob:
    def _patch_ingest_store(self):
        """Return context manager patching ingest store to in-memory."""
        jobs = {}
        events = []
        import contextlib

        @contextlib.contextmanager
        def ctx():
            with patch("app.services.ingest_job_runner.save_ingest_job",
                       side_effect=lambda jid, d: jobs.update({jid: dict(d)})), \
                 patch("app.services.ingest_job_runner.load_ingest_job",
                       side_effect=lambda jid: dict(jobs[jid])), \
                 patch("app.services.ingest_job_runner.append_ingest_event",
                       side_effect=lambda jid, evt, payload: events.append((jid, evt))), \
                 patch("app.services.ingest_job_runner.settings") as mock_settings:
                mock_settings.youtube_channel_url = "https://youtube.com/@test"
                yield jobs, events

        return ctx()

    def test_returns_string_job_id(self):
        with self._patch_ingest_store() as (jobs, events):
            from app.services.ingest_job_runner import create_ingest_job
            result = create_ingest_job()
        assert isinstance(result, str)

    def test_job_id_starts_with_ijob(self):
        with self._patch_ingest_store() as (jobs, events):
            from app.services.ingest_job_runner import create_ingest_job
            result = create_ingest_job()
        assert result.startswith("ijob_")

    def test_job_saved_with_queued_status(self):
        with self._patch_ingest_store() as (jobs, events):
            from app.services.ingest_job_runner import create_ingest_job
            job_id = create_ingest_job()
        assert jobs[job_id]["status"] == "queued"

    def test_custom_channel_url_stored(self):
        with self._patch_ingest_store() as (jobs, events):
            from app.services.ingest_job_runner import create_ingest_job
            job_id = create_ingest_job(channel_url="https://custom.channel")
        assert jobs[job_id]["channel_url"] == "https://custom.channel"

    def test_created_event_emitted(self):
        with self._patch_ingest_store() as (jobs, events):
            from app.services.ingest_job_runner import create_ingest_job
            create_ingest_job()
        assert any(evt == "created" for _, evt in events)

    def test_falls_back_to_settings_channel(self):
        with self._patch_ingest_store() as (jobs, events):
            from app.services.ingest_job_runner import create_ingest_job
            job_id = create_ingest_job(channel_url=None)
        assert jobs[job_id]["channel_url"] == "https://youtube.com/@test"
