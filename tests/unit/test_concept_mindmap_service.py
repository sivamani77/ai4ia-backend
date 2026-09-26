"""Unit tests for concept mindmap storyline resolution."""
from __future__ import annotations

import json
from unittest.mock import patch


class TestResolveStoryline:
    def test_resolves_video_job_id_to_validated_storyline_path(self, tmp_path, monkeypatch):
        from app.services import concept_mindmap_service as service
        import app.services.job_store as job_store

        storyline_path = tmp_path / "storyline_test_validated.json"
        payload = {
            "user_prompt": "create a video on deep fake",
            "storyline": {"title": "Deep Fake", "sections": []},
        }
        storyline_path.write_text(json.dumps(payload), encoding="utf-8")

        monkeypatch.setattr(service, "STORYLINES_DIR", tmp_path)
        monkeypatch.setattr(job_store, "load_job", lambda job_id: {
            "job_id": job_id,
            "paths": {"validated_storyline_path": str(storyline_path)},
        })
        monkeypatch.setattr("app.pipeline.s3_helpers.use_s3", lambda: False)

        resolved, source_path = service._resolve_storyline(
            "create a video on deep fake",
            "vjob_20260626_091247_dc4c1f69",
        )

        assert resolved == payload
        assert source_path == storyline_path

    def test_resolves_video_job_id_to_storylines_dir_basename(self, tmp_path, monkeypatch):
        from app.services import concept_mindmap_service as service
        import app.services.job_store as job_store

        local_storyline = tmp_path / "storyline_from_job_validated.json"
        payload = {
            "user_prompt": "create a video on deep fake",
            "storyline": {"title": "Deep Fake", "sections": []},
        }
        local_storyline.write_text(json.dumps(payload), encoding="utf-8")

        monkeypatch.setattr(service, "STORYLINES_DIR", tmp_path)
        monkeypatch.setattr(job_store, "load_job", lambda job_id: {
            "job_id": job_id,
            "paths": {
                "validated_storyline_path": f"/old/container/path/{local_storyline.name}",
            },
        })
        monkeypatch.setattr("app.pipeline.s3_helpers.use_s3", lambda: False)

        resolved, source_path = service._resolve_storyline(
            "create a video on deep fake",
            "vjob_20260626_091247_dc4c1f69",
        )

        assert resolved == payload
        assert source_path == local_storyline


class TestOwnedStoryline:
    """Non-admins reach a storyline only through their own jobs (C-02, C-03)."""

    def _setup(self, tmp_path, monkeypatch):
        from app.services import concept_mindmap_service as service
        import app.services.job_store as job_store

        mine = tmp_path / "storyline_mine_validated.json"
        mine.write_text(json.dumps({"user_prompt": "deep fakes", "storyline": {"title": "Deep Fakes"}}))
        theirs = tmp_path / "storyline_theirs_validated.json"
        theirs.write_text(json.dumps({"user_prompt": "deep fakes", "storyline": {"title": "Theirs"}}))
        jobs = {
            "vjob_mine": {"job_id": "vjob_mine", "user_id": "u1", "paths": {"validated_storyline_path": str(mine)}},
            "vjob_theirs": {"job_id": "vjob_theirs", "user_id": "u2", "paths": {"validated_storyline_path": str(theirs)}},
        }
        monkeypatch.setattr(service, "STORYLINES_DIR", tmp_path)
        monkeypatch.setattr(job_store, "load_job", lambda job_id: jobs[job_id])
        monkeypatch.setattr(job_store, "list_by_user", lambda uid: [j for j in jobs.values() if j["user_id"] == uid])
        monkeypatch.setattr("app.pipeline.s3_helpers.use_s3", lambda: False)
        return service, mine

    def test_own_job_id_resolves(self, tmp_path, monkeypatch):
        service, mine = self._setup(tmp_path, monkeypatch)
        _, path = service._resolve_owned_storyline("deep fakes", "vjob_mine", "u1")
        assert path == mine

    def test_own_storyline_file_stem_resolves(self, tmp_path, monkeypatch):
        service, mine = self._setup(tmp_path, monkeypatch)
        _, path = service._resolve_owned_storyline("deep fakes", "storyline_mine_validated", "u1")
        assert path == mine

    def test_other_users_job_or_file_is_not_found(self, tmp_path, monkeypatch):
        import pytest

        service, _ = self._setup(tmp_path, monkeypatch)
        for storyline_id in ("vjob_theirs", "storyline_theirs_validated", "theirs"):
            with pytest.raises(FileNotFoundError):
                service._resolve_owned_storyline("deep fakes", storyline_id, "u1")

    def test_without_an_id_only_own_storylines_are_ranked(self, tmp_path, monkeypatch):
        service, mine = self._setup(tmp_path, monkeypatch)
        _, path = service._resolve_owned_storyline("deep fakes", None, "u1")
        assert path == mine

    def test_user_with_no_videos_gets_not_found(self, tmp_path, monkeypatch):
        import pytest

        service, _ = self._setup(tmp_path, monkeypatch)
        with pytest.raises(FileNotFoundError):
            service._resolve_owned_storyline("deep fakes", None, "u3")

    def test_path_shaped_ids_are_rejected(self, tmp_path, monkeypatch):
        import pytest

        service, _ = self._setup(tmp_path, monkeypatch)
        for storyline_id in ("../../etc/passwd", "a/b", "x.json", "..", "%2e%2e"):
            with pytest.raises(FileNotFoundError):
                service.generate_concept_mindmap("deep fakes", storyline_id=storyline_id)


class TestResolveSectionSources:
    def test_stream_url_computed_from_source_video(self):
        """GAP-014 regression: fragment rows have no stream_url column, so it
        must be derived from source_video, not read from a field that's
        never set on the raw fragment."""
        from app.services import concept_mindmap_service as service

        section = {"evidence_fragment_ids": ["f1"]}
        frag = {"fragment_id": "f1", "source_video": "0001_abc123_speaker", "text": "hello"}
        with patch("app.services.concept_mindmap_service.fragment_store.get", return_value=frag):
            sources = service._resolve_section_sources(section, {})

        assert len(sources) == 1
        assert sources[0]["stream_url"] == "/api/archive/videos/0001_abc123_speaker/stream"

    def test_stream_url_none_when_source_video_missing(self):
        from app.services import concept_mindmap_service as service

        section = {"evidence_fragment_ids": ["f1"]}
        frag = {"fragment_id": "f1", "text": "hello"}
        with patch("app.services.concept_mindmap_service.fragment_store.get", return_value=frag):
            sources = service._resolve_section_sources(section, {})

        assert sources[0]["stream_url"] is None
