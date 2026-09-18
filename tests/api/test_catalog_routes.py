from __future__ import annotations

import json
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


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
    import app.main as main
    from app.core.auth import CurrentUser, get_current_user
    import app.services.insight_service as insight_service
    import app.services.comparison_service as comparison_service
    import app.services.concept_mindmap_service as mindmap_service

    # api_generate_insight now requires a verified user (Depends(get_current_user))
    # so it can attribute the insight to current_user.id — override the dependency
    # rather than crafting a real JWT.
    main.app.dependency_overrides[get_current_user] = lambda: CurrentUser(id="test-user", persona="general")
    try:
        _run_insights_compare_and_mindmap(client, monkeypatch, insight_service, comparison_service, mindmap_service)
    finally:
        main.app.dependency_overrides.pop(get_current_user, None)


def _run_insights_compare_and_mindmap(client, monkeypatch, insight_service, comparison_service, mindmap_service):
    import app.api.concept_mindmap as concept_mindmap_module
    monkeypatch.setattr(insight_service, "generate_insight", lambda query, filters, limit, user_id=None: {
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
    monkeypatch.setattr(concept_mindmap_module, "generate_concept_mindmap", lambda query, storyline_id=None, mode="legacy": {
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
