from __future__ import annotations

import json
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_storylines_and_assets_routes(client, mock_settings, monkeypatch):
    import app.api.storylines as storylines_module
    import app.api.assets as assets_module
    import app.services.insight_service as insight_service

    class FakeStorylineRepo:
        def __init__(self) -> None:
            pass

        def list(self, limit: int = 100):
            raise RuntimeError("db offline")

        def get_latest(self):
            raise RuntimeError("db offline")

        def get(self, storyline_id: str):
            raise RuntimeError("db offline")

    monkeypatch.setattr(storylines_module, "StorylineRepository", FakeStorylineRepo)
    monkeypatch.setattr(insight_service, "storyline_evidence_map", lambda storyline: {
        "storyline_title": storyline.get("storyline", {}).get("title", ""),
        "evidence_count": 1,
    })

    storyline_dir = mock_settings.project_root / "data" / "storylines"
    storyline_path = storyline_dir / "storyline_20260630_000001.json"
    storyline_payload = {
        "storyline": {
            "title": "AI for public access",
            "logline": "How AI helps people find information",
            "sections": [{"id": "sec1"}, {"id": "sec2"}],
        },
        "created_at": "2026-06-30T00:00:00Z",
        "target_duration_seconds": 180,
    }
    _write_json(storyline_path, storyline_payload)

    latest_manifest = mock_settings.ai4ia_local_assets_dir / "latest_asset_manifest.json"
    _write_json(latest_manifest, {"status": "ok", "items": 3})

    response = client.get("/api/storylines")
    assert response.status_code == 200
    assert response.json()[0]["id"] == "storyline_20260630_000001"

    response = client.get("/api/storylines/latest")
    assert response.status_code == 200
    assert response.json()["storyline_id"] == "storyline_20260630_000001"

    response = client.get("/api/storylines/storyline_20260630_000001")
    assert response.status_code == 200
    assert response.json()["storyline_id"] == "storyline_20260630_000001"

    response = client.get("/api/storylines/storyline_20260630_000001/evidence-map")
    assert response.status_code == 200
    assert response.json()["evidence_count"] == 1

    created_jobs: list[dict] = []

    def fake_create_job(kind: str, payload: dict):
        job = {"job_id": f"{kind}_job_001", "kind": kind, "payload": payload}
        created_jobs.append(job)
        return job

    monkeypatch.setattr(storylines_module, "create_job", fake_create_job)
    monkeypatch.setattr(storylines_module, "run_script_background", lambda **kwargs: None)
    monkeypatch.setattr(assets_module, "create_job", fake_create_job)
    monkeypatch.setattr(assets_module, "run_script_background", lambda **kwargs: None)

    response = client.post(
        "/api/storylines/generate",
        json={"prompt": "A longer prompt about AI access", "duration": 120},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "running"

    response = client.get("/api/assets/latest")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

    response = client.post(
        "/api/assets/generate",
        json={"storyline_path": str(storyline_path), "skip_tts": True},
    )
    assert response.status_code == 200
    assert response.json()["job_id"] == "generate_assets_job_001"

    response = client.post("/api/assets/generate-latest", json={"skip_tts": True})
    assert response.status_code == 200
    assert response.json()["storyline_path"].endswith("storyline_20260630_000001.json")

    assert len(created_jobs) == 2


def test_export_routes(client, mock_settings, monkeypatch):
    import app.api.exports as exports_module

    export_dir = mock_settings.ai4ia_local_exports_dir
    export_name = "demo_export.mp4"
    export_path = export_dir / export_name
    export_path.write_bytes(b"mp4-bytes")
    (export_dir / "demo_export.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
    (export_dir / "demo_export.vtt").write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello\n", encoding="utf-8")
    _write_json(export_dir / "demo_export_export_manifest.json", {"ok": True})
    _write_json(export_dir / "demo_export_qa_report.json", {"quality": "good"})
    _write_json(export_dir / "demo_export_timeline_metadata.json", {"timeline": []})
    _write_json(mock_settings.ai4ia_local_assets_dir / "latest_asset_manifest.json", {"ready": True})

    monkeypatch.setattr(exports_module, "create_job", lambda kind, payload: {"job_id": f"{kind}_job"})
    monkeypatch.setattr(exports_module, "run_script_background", lambda **kwargs: None)

    response = client.get("/api/exports")
    assert response.status_code == 200
    assert response.json()[0]["filename"] == export_name

    response = client.get(f"/api/exports/{export_name}/stream")
    assert response.status_code == 200
    assert response.content == b"mp4-bytes"

    response = client.get(f"/api/exports/{export_name}/artifact/download")
    assert response.status_code == 200

    response = client.get(f"/api/exports/{export_name}/artifact/vtt")
    assert response.status_code == 200

    response = client.get(f"/api/exports/{export_name}/artifact/manifest")
    assert response.status_code == 200
    assert response.json()["ok"] is True

    response = client.post("/api/exports/generate", json={"output": "custom.mp4"})
    assert response.status_code == 200
    assert response.json()["status"] == "running"

    response = client.post("/api/exports/generate-latest", json={"output": "custom.mp4"})
    assert response.status_code == 200
    assert response.json()["manifest_path"].endswith("latest_asset_manifest.json")


def test_archive_routes(client, mock_settings, monkeypatch):
    import app.api.archive as archive_module
    import app.services.fragment_store as fragment_store
    import app.services.embedding_index as embedding_index

    raw_video_dir = mock_settings.project_root / "data" / "raw" / "youtube" / "ai4ia_channel"
    raw_video_dir.mkdir(parents=True, exist_ok=True)
    (raw_video_dir / "video123.mp4").write_bytes(b"video-bytes")

    fragment = {
        "fragment_id": "frag_123",
        "speaker": "Alice",
        "speaker_attribution": "Alice — Researcher",
        "text": "AI helps people access information.",
        "primary_theme": "access",
        "primary_chapter": "chapter-one",
        "start": 1.0,
        "end": 11.0,
        "duration": 10.0,
        "source_video": "video123",
        "year": 2026,
        "edition": "AI4IA 2026",
    }

    monkeypatch.setattr(archive_module, "list_videos", lambda: [{
        "video_id": "video123",
        "title": "AI access talk",
        "speaker_or_topic": "Alice",
        "duration": 10,
        "webpage_url": "https://example.com/video123",
    }])
    monkeypatch.setattr(archive_module, "get_video", lambda video_id: {
        "video_id": video_id,
        "title": "AI access talk",
        "speaker_or_topic": "Alice",
    })
    monkeypatch.setattr(archive_module, "search_fragments", lambda **kwargs: [fragment])
    monkeypatch.setattr(archive_module, "get_fragment", lambda fragment_id: None)
    monkeypatch.setattr(archive_module, "get_themes_and_chapters", lambda: {"themes": ["access"]})
    monkeypatch.setattr(fragment_store, "speakers_summary", lambda: [{"speaker": "Alice", "count": 1}])
    monkeypatch.setattr(fragment_store, "get", lambda fragment_id: fragment if fragment_id == "frag_123" else None)
    monkeypatch.setattr(embedding_index, "available", lambda: True)
    monkeypatch.setattr(embedding_index, "search", lambda query, top_k=10, filters=None: [fragment])

    response = client.get("/api/archive/speakers")
    assert response.status_code == 200
    assert response.json()["count"] == 1

    response = client.get("/api/archive/videos?query=access")
    assert response.status_code == 200
    assert response.json()[0]["video_id"] == "video123"

    response = client.get("/api/archive/videos/video123")
    assert response.status_code == 200

    response = client.get("/api/archive/videos/video123/stream")
    assert response.status_code == 200
    assert response.content == b"video-bytes"

    response = client.get("/api/archive/themes")
    assert response.status_code == 200

    response = client.get("/api/archive/fragments?query=ai")
    assert response.status_code == 200
    assert response.json()[0]["fragment_id"] == "frag_123"

    response = client.get("/api/archive/search?q=ai")
    assert response.status_code == 200

    response = client.post("/api/archive/semantic-search", json={"query": "ai access", "top_k": 5})
    assert response.status_code == 200
    assert response.json()["mode"] == "semantic"

    response = client.get("/api/archive/fragments/frag_123/preview")
    assert response.status_code == 200
    assert response.json()["video_stream_url"].endswith("/api/archive/videos/video123/stream")

    response = client.get("/api/archive/fragments/frag_123")
    assert response.status_code == 200
    assert response.json()["fragment_id"] == "frag_123"


def test_insights_compare_and_mindmap_routes(client, monkeypatch):
    import app.services.insight_service as insight_service
    import app.services.comparison_service as comparison_service
    import app.services.concept_mindmap_service as mindmap_service

    monkeypatch.setattr(insight_service, "generate_insight", lambda query, filters, limit: {
        "query": query,
        "limit": limit,
        "count": 2,
    })
    monkeypatch.setattr(insight_service, "list_insights", lambda: [{"id": "insight_1"}])
    monkeypatch.setattr(insight_service, "get_insight", lambda insight_id: {"id": insight_id, "title": "Insight"})
    monkeypatch.setattr(insight_service, "build_sources_map", lambda insight: {"sources": [insight["id"]]})

    monkeypatch.setattr(comparison_service, "compare_speakers", lambda speakers, topic, limit: {
        "speakers": speakers,
        "topic": topic,
        "limit": limit,
    })
    monkeypatch.setattr(comparison_service, "synthesize_theme", lambda theme, limit: {
        "theme": theme,
        "limit": limit,
    })
    monkeypatch.setattr(comparison_service, "build_relay", lambda topic, limit: {
        "topic": topic,
        "limit": limit,
    })
    monkeypatch.setattr(mindmap_service, "generate_concept_mindmap", lambda query, storyline_id=None: {
        "query": query,
        "storyline_id": storyline_id,
    })

    response = client.post("/api/insights/generate", json={"query": "AI access"})
    assert response.status_code == 200
    assert response.json()["count"] == 2

    response = client.get("/api/insights")
    assert response.status_code == 200
    assert response.json()["count"] == 1

    response = client.get("/api/insights/insight_1")
    assert response.status_code == 200

    response = client.get("/api/insights/insight_1/sources")
    assert response.status_code == 200
    assert response.json()["sources"] == ["insight_1"]

    response = client.post("/api/compare/speakers", json={"speakers": ["Alice", "Bob"]})
    assert response.status_code == 200
    assert response.json()["speakers"] == ["Alice", "Bob"]

    response = client.post("/api/synthesis/theme", json={"theme": "access"})
    assert response.status_code == 200
    assert response.json()["theme"] == "access"

    response = client.post("/api/relay", json={"topic": "AI access"})
    assert response.status_code == 200

    response = client.post("/api/v1/concept-mindmap", json={"query": "How AI helps access"})
    assert response.status_code == 200
    assert response.json()["query"] == "How AI helps access"
