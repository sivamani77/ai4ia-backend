"""Unit tests for concept mindmap storyline resolution."""
from __future__ import annotations

import json


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
