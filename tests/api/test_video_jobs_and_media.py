from __future__ import annotations

import copy
import json
import uuid
from pathlib import Path

import pytest


OWNER_ID = "550e8400-e29b-41d4-a716-446655440000"
OTHER_ID = "550e8400-e29b-41d4-a716-446655440001"


@pytest.fixture(autouse=True)
def _jwt_signing_key(monkeypatch):
    """/api/video-jobs/* now identifies callers via a JWT bearer token
    (Depends(get_current_user)), replacing the old X-Session-ID header.
    create_access_token needs a non-empty HMAC key; settings.jwt_signing_key
    defaults to "" with no .env present in this environment — see the
    identical fixture in test_core_security.py."""
    from app.core import security as security_module

    monkeypatch.setattr(
        security_module.settings, "jwt_signing_key",
        "test-signing-key-that-is-at-least-32-bytes-long", raising=False,
    )


def _auth_headers(user_id: str) -> dict:
    from app.core.security import create_access_token

    token = create_access_token(user_id=user_id, persona="general")
    return {"Authorization": f"Bearer {token}"}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _seed_video_job(mock_settings, mock_job_store, sample_job):
    # Unique per call: some app-side module globals (e.g. storage_service's
    # _LOCAL_JOBS_DIR) are resolved once at import time from whichever test's
    # settings happened to be active first, so the on-disk media registry
    # path for a fixed job_id can be shared — and polluted — across tests in
    # the same pytest process. A unique job_id keeps each test's registry
    # file separate regardless.
    job_id = f"vjob_20260630_000001_{uuid.uuid4().hex[:8]}"
    job = copy.deepcopy(sample_job)
    job.update(
        {
            "job_id": job_id,
            "status": "completed",
            "user_id": OWNER_ID,
            "prompt": "AI access documentary",
            "progress_percent": 100,
            "current_stage": "completed",
            "message": "Done",
            "paths": {
                "storyline_path": str(mock_settings.project_root / "data" / "storylines" / f"{job_id}.json"),
                "validated_storyline_path": str(mock_settings.project_root / "data" / "storylines" / f"{job_id}_validated.json"),
                "asset_manifest_path": str(mock_settings.ai4ia_local_assets_dir / f"{job_id}_asset_manifest.json"),
                "export_manifest_path": str(mock_settings.ai4ia_local_exports_dir / f"{job_id}_export_manifest.json"),
                "timeline_metadata_path": None,
                "video_path": str(mock_settings.ai4ia_local_exports_dir / f"{job_id}.mp4"),
                "srt_path": str(mock_settings.ai4ia_local_exports_dir / f"{job_id}.srt"),
                "vtt_path": str(mock_settings.ai4ia_local_exports_dir / f"{job_id}.vtt"),
            },
        }
    )
    mock_job_store["jobs"][job_id] = job

    storyline_payload = {
        "storyline": {
            "title": "AI access documentary",
            "logline": "AI can improve access to information",
        },
        "prompt_profile": {
            "raw_prompt": "AI access documentary",
            "intent_type": "documentary",
            "topic_terms": ["AI", "access"],
            "requested_speaker": "Alice",
            "speaker_required": True,
            "theme_hint": "access",
            "chapter_hint": "public access",
            "evidence_requirement": "high",
            "strictness": "strict",
        },
        "retrieval_diagnostics": {
            "semantic_count": 1,
            "keyword_count": 2,
            "chapter_count": 0,
            "speaker_count": 1,
        },
        "corpus_diagnostics": {
            "total_embedding_records": 10,
            "records_after_filtering": 5,
            "embedding_dimension": 768,
        },
        "retrieval_summary": {"primary_mode": "semantic"},
        "scoring_summary": {"exact_clip_count": 2, "usable_speaker_clip_count": 1},
        "planning_decisions": {"mode": "speaker_led_documentary"},
        "timeline_construction": {"selected_sections": 2},
        "editorial_gates": {"approved": True},
        "selected_candidates": [{"fragment_id": "frag_1", "score": 0.99}],
        "supporting_candidates": [{"fragment_id": "frag_2", "score": 0.5}],
        "final_timeline": [
            {
                "section_id": "sec1",
                "type": "speaker_clip",
                "speaker": "Alice",
                "quote_text": "AI helps people find information.",
                "render_start": 0,
                "render_end": 10,
                "fragment_id": "frag_1",
            },
            {
                "section_id": "sec2",
                "type": "narration_broll",
                "voiceover_text": "Narration about access.",
                "visual_title": "Information access",
                "visual_asset_type": "image",
                "visual_asset_prompt": "A public library",
                "visual_keywords": ["library"],
                "render_visual_path": "visual.png",
                "audio_path": "voiceover.mp3",
            },
        ],
        "editorial_quality_gates": {
            "all_sections_covered": True,
            "all_speaker_labels_present": True,
            "all_speaker_quotes_complete": True,
            "no_instruction_leakage_in_voiceover": True,
            "all_voiceovers_topic_specific": True,
            "final_timeline_approved": True,
            "all_narration_broll_have_visual_asset_prompt": True,
            "speaker_clip_count": 1,
            "narration_broll_count": 1,
            "sections_covered": ["sec1", "sec2"],
            "offending_items": {},
        },
        "repair_iterations": 1,
        "warnings": ["Review narration wording"],
    }
    validated_payload = dict(storyline_payload)
    validated_payload["final_timeline"] = [
        storyline_payload["final_timeline"][0],
        {
            **storyline_payload["final_timeline"][1],
            "demotion_reason": "Reduced for pacing",
        },
    ]

    _write_json(Path(job["paths"]["storyline_path"]), storyline_payload)
    _write_json(Path(job["paths"]["validated_storyline_path"]), validated_payload)

    asset_manifest = {
        "visual_strategy": "hybrid",
        "visual_quality_gates": {"all_visuals_present": True},
        "export_timeline": [
            {
                "type": "narration_broll",
                "section_id": "sec2",
                "voiceover_text": "Narration about access.",
                "subtitle_text": "Narration about access.",
                "visual_title": "Information access",
                "visual_asset_type": "image",
                "visual_keywords": ["library"],
                "audio_path": str(mock_settings.ai4ia_local_assets_dir / "voiceover.mp3"),
                "render_visual_path": str(mock_settings.ai4ia_local_assets_dir / "visual.png"),
            }
        ],
        "cards": {
            "title_card": True,
            "closing_card": True,
            "section_cards": [{"section_id": "sec2"}],
        },
    }
    _write_json(Path(job["paths"]["asset_manifest_path"]), asset_manifest)

    export_manifest = {
        "quality_checks": {
            "speaker_clip_count": 1,
            "ai_voiceover_segments": 1,
        },
        "generated_visual_segments": 1,
        "muted_archive_narration_segments": 0,
        "subtitles": {
            "entry_count": 2,
            "coverage": {"percent": 100},
        },
        "visual_strategy": "hybrid",
    }
    _write_json(Path(job["paths"]["export_manifest_path"]), export_manifest)

    for path, content in {
        Path(job["paths"]["video_path"]): b"video-bytes",
        Path(job["paths"]["srt_path"]): b"1\n00:00:00,000 --> 00:00:01,000\nHello\n",
        Path(job["paths"]["vtt_path"]): b"WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello\n",
        mock_settings.ai4ia_local_assets_dir / "voiceover.mp3": b"audio-bytes",
        mock_settings.ai4ia_local_assets_dir / "visual.png": b"image-bytes",
        mock_settings.ai4ia_local_assets_dir / "title_card.png": b"title-card",
        mock_settings.ai4ia_local_assets_dir / "thumbnail.png": b"thumbnail",
    }.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    registry = {
        "storage_backend": "local",
        "objects": {
            "video": {
                "kind": "video",
                "filename": Path(job["paths"]["video_path"]).name,
                "content_type": "video/mp4",
                "local_path": job["paths"]["video_path"],
                "size_bytes": len(b"video-bytes"),
            },
            "srt": {
                "kind": "subtitle",
                "subtitle_format": "srt",
                "filename": Path(job["paths"]["srt_path"]).name,
                "content_type": "application/x-subrip",
                "local_path": job["paths"]["srt_path"],
                "size_bytes": len(b"1\n00:00:00,000 --> 00:00:01,000\nHello\n"),
            },
            "vtt": {
                "kind": "subtitle",
                "subtitle_format": "vtt",
                "filename": Path(job["paths"]["vtt_path"]).name,
                "content_type": "text/vtt",
                "local_path": job["paths"]["vtt_path"],
                "size_bytes": len(b"WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello\n"),
            },
            "manifest": {
                "kind": "manifest",
                "filename": Path(job["paths"]["export_manifest_path"]).name,
                "content_type": "application/json",
                "local_path": job["paths"]["export_manifest_path"],
                "size_bytes": Path(job["paths"]["export_manifest_path"]).stat().st_size,
            },
            "voiceover_sec2": {
                "kind": "voiceover_audio",
                "filename": "voiceover.mp3",
                "content_type": "audio/mpeg",
                "local_path": str(mock_settings.ai4ia_local_assets_dir / "voiceover.mp3"),
                "size_bytes": len(b"audio-bytes"),
                "section_id": "sec2",
            },
            "visual_sec2": {
                "kind": "visual_asset",
                "filename": "visual.png",
                "content_type": "image/png",
                "local_path": str(mock_settings.ai4ia_local_assets_dir / "visual.png"),
                "size_bytes": len(b"image-bytes"),
                "section_id": "sec2",
                "visual_title": "Information access",
                "visual_asset_type": "image",
            },
            "title_card": {
                "kind": "card",
                "card_type": "title_card",
                "filename": "title_card.png",
                "content_type": "image/png",
                "local_path": str(mock_settings.ai4ia_local_assets_dir / "title_card.png"),
                "size_bytes": len(b"title-card"),
            },
            "thumbnail": {
                "kind": "thumbnail",
                "filename": "thumbnail.png",
                "content_type": "image/png",
                "local_path": str(mock_settings.ai4ia_local_assets_dir / "thumbnail.png"),
                "size_bytes": len(b"thumbnail"),
            },
        },
    }
    _write_json(mock_settings.ai4ia_local_jobs_dir / job_id / "media_registry.json", registry)

    return job_id, job


def test_video_job_lifecycle_and_traffic_limits(client, mock_settings, mock_job_store, sample_job, monkeypatch):
    import app.api.video_jobs as video_jobs_module

    monkeypatch.setattr(video_jobs_module, "start_video_job_background", lambda *args, **kwargs: None)

    response = client.post(
        "/api/video-jobs",
        headers=_auth_headers(OWNER_ID),
        json={
            "prompt": "Create a documentary about AI access",
            "target_duration": 150,
            "user_id": OWNER_ID,
        },
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    assert response.json()["status"] == "queued"

    response = client.get("/api/video-jobs", headers=_auth_headers(OWNER_ID))
    assert response.status_code == 200
    assert response.json()["total"] == 1

    response = client.get(f"/api/video-jobs/{job_id}", headers=_auth_headers(OWNER_ID))
    assert response.status_code == 200
    assert response.json()["job_id"] == job_id

    response = client.delete(f"/api/video-jobs/{job_id}", headers=_auth_headers(OWNER_ID))
    assert response.status_code == 202
    assert response.json()["status"] == "cancelled"

    response = client.delete(f"/api/video-jobs/{job_id}", headers=_auth_headers(OWNER_ID))
    assert response.status_code == 409

    response = client.get(f"/api/video-jobs/{job_id}", headers=_auth_headers(OTHER_ID))
    assert response.status_code == 404


def test_video_job_rate_limit_and_capacity(client, mock_job_store, monkeypatch):
    import app.api.video_jobs as video_jobs_module

    monkeypatch.setattr(video_jobs_module, "start_video_job_background", lambda *args, **kwargs: None)
    monkeypatch.setattr(video_jobs_module, "list_jobs", lambda: [])

    for idx in range(5):
        response = client.post(
            "/api/video-jobs",
            headers=_auth_headers(OWNER_ID),
            json={
                "prompt": f"Create documentary {idx}",
                "target_duration": 150,
                "user_id": OWNER_ID,
            },
        )
        assert response.status_code == 202

    response = client.post(
        "/api/video-jobs",
        headers=_auth_headers(OWNER_ID),
        json={
            "prompt": "Create documentary 6",
            "target_duration": 150,
            "user_id": OWNER_ID,
        },
    )
    assert response.status_code == 429

    monkeypatch.setattr(video_jobs_module, "_create_limiter", video_jobs_module.TokenBucket(rate=5, per=60.0))
    monkeypatch.setattr(
        video_jobs_module,
        "list_jobs",
        # _MAX_GLOBAL_QUEUED is 50 — must meet or exceed it to trip the cap.
        lambda: [{"job_id": f"j{i}", "status": "running"} for i in range(50)],
    )

    response = client.post(
        "/api/video-jobs",
        headers=_auth_headers(OTHER_ID),
        json={
            "prompt": "Create documentary at capacity",
            "target_duration": 150,
            "user_id": OTHER_ID,
        },
    )
    assert response.status_code == 503


def test_video_job_detail_events_and_reports(client, mock_settings, mock_job_store, sample_job):
    job_id, job = _seed_video_job(mock_settings, mock_job_store, sample_job)

    mock_job_store["events"][job_id] = [
        {
            "id": 1,
            "job_id": job_id,
            "event_type": "created",
            "stage": "queued",
            "payload": {"prompt": job["prompt"]},
        },
        {
            "id": 2,
            "job_id": job_id,
            "event_type": "completed",
            "stage": "completed",
            "payload": {"message": "done"},
        },
    ]

    response = client.get(f"/api/video-jobs/{job_id}", headers=_auth_headers(OWNER_ID))
    assert response.status_code == 200
    assert response.json()["result_available"] is True

    response = client.get(f"/api/video-jobs/{job_id}/events", headers=_auth_headers(OWNER_ID))
    assert response.status_code == 200
    assert "event: done" in response.text

    response = client.get(f"/api/video-jobs/{job_id}/storyline", headers=_auth_headers(OWNER_ID))
    assert response.status_code == 200
    assert response.json()["status"] == "available"
    assert len(response.json()["sections"]) == 2

    # /explainability was folded into /provenance (see get_job_provenance's
    # docstring); the response is now flat — no "status" wrapper key.
    response = client.get(f"/api/video-jobs/{job_id}/provenance", headers=_auth_headers(OWNER_ID))
    assert response.status_code == 200
    assert response.json()["job_id"] == job_id

    response = client.get(f"/api/video-jobs/{job_id}/validation-report", headers=_auth_headers(OWNER_ID))
    assert response.status_code == 200
    assert response.json()["approved_fragments"] == 1
    assert response.json()["changed_decisions"][0]["reason"] == "Reduced for pacing"

    response = client.get(f"/api/video-jobs/{job_id}/assets", headers=_auth_headers(OWNER_ID))
    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert response.json()["voiceovers"][0]["asset_id"] == "voiceover_sec2"

    response = client.get(f"/api/video-jobs/{job_id}/result", headers=_auth_headers(OWNER_ID))
    assert response.status_code == 200
    assert response.json()["job_id"] == job_id

    response = client.get(f"/api/video-jobs/{job_id}/result", headers=_auth_headers(OTHER_ID))
    assert response.status_code == 404


def test_video_media_endpoints(client, mock_settings, mock_job_store, sample_job):
    job_id, _job = _seed_video_job(mock_settings, mock_job_store, sample_job)

    response = client.get(f"/api/video-jobs/{job_id}/video/stream", headers=_auth_headers(OWNER_ID))
    assert response.status_code == 200
    assert response.content == b"video-bytes"

    response = client.get(
        f"/api/video-jobs/{job_id}/video/stream",
        headers={**_auth_headers(OWNER_ID), "Range": "bytes=0-3"},
    )
    assert response.status_code == 206
    assert response.headers["content-range"].startswith("bytes 0-3/")

    response = client.get(
        f"/api/video-jobs/{job_id}/video/stream",
        headers={**_auth_headers(OWNER_ID), "Range": "bytes=99999999999-"},
    )
    assert response.status_code == 416
    assert response.headers["content-range"] == f"bytes */{len(b'video-bytes')}"
    assert response.json() == {"detail": "Range Not Satisfiable"}

    response = client.get(f"/api/video-jobs/{job_id}/video/download", headers=_auth_headers(OWNER_ID))
    assert response.status_code == 200
    assert response.headers["content-disposition"].startswith("attachment")

    response = client.get(f"/api/video-jobs/{job_id}/video/signed-url", headers=_auth_headers(OWNER_ID))
    assert response.status_code == 200
    assert response.json()["storage_backend"] == "local"

    response = client.get(f"/api/video-jobs/{job_id}/subtitles/srt", headers=_auth_headers(OWNER_ID))
    assert response.status_code == 200

    response = client.get(f"/api/video-jobs/{job_id}/subtitles/ass", headers=_auth_headers(OWNER_ID))
    assert response.status_code == 400

    response = client.get(f"/api/video-jobs/{job_id}/manifest", headers=_auth_headers(OWNER_ID))
    assert response.status_code == 200
    assert response.json()["quality_checks"]["speaker_clip_count"] == 1

    response = client.get(f"/api/video-jobs/{job_id}/media", headers=_auth_headers(OWNER_ID))
    assert response.status_code == 200
    assert response.json()["job_id"] == job_id
    assert any(item["asset_id"] == "video" for item in response.json()["objects"])
    assert all(item["asset_id"] != "thumbnail" for item in response.json()["objects"])

    response = client.get(f"/api/video-jobs/{job_id}/thumbnail", headers=_auth_headers(OWNER_ID))
    assert response.status_code == 200

    response = client.get(
        f"/api/video-jobs/{job_id}/assets/voiceover_sec2/stream",
        headers={**_auth_headers(OWNER_ID), "Range": "bytes=0-4"},
    )
    assert response.status_code == 206
    assert response.content == b"audio"

    response = client.get(f"/api/video-jobs/{job_id}/assets/voiceover_sec2/download", headers=_auth_headers(OWNER_ID))
    assert response.status_code == 200

    response = client.get(f"/api/video-jobs/{job_id}/assets/voiceover_sec2/metadata", headers=_auth_headers(OWNER_ID))
    assert response.status_code == 200
    assert response.json()["asset_id"] == "voiceover_sec2"

    response = client.get(f"/api/video-jobs/{job_id}/assets/voiceover_sec2/signed-url", headers=_auth_headers(OWNER_ID))
    assert response.status_code == 200
    assert response.json()["storage_backend"] == "local"

    response = client.get(f"/api/video-jobs/{job_id}/assets/visual_sec2/stream", headers=_auth_headers(OWNER_ID))
    assert response.status_code == 200

    response = client.get(f"/api/video-jobs/{job_id}/assets/visual_sec2/download", headers=_auth_headers(OWNER_ID))
    assert response.status_code == 200
