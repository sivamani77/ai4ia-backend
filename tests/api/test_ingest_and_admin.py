from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path


def test_ingest_sync_jobs_and_video_inventory(client, mock_settings, monkeypatch):
    import app.main as main
    from app.core.auth import CurrentUser, get_current_user
    import app.api.ingest as ingest_module

    # /api/ingest/* is registered with _admin_dep (API key + admin role) —
    # override the auth dependency instead of crafting a real admin JWT.
    monkeypatch.setitem(
        main.app.dependency_overrides, get_current_user,
        lambda: CurrentUser(id="admin-user", persona="admin"),
    )
    # _trigger_limiter (rate=3/300s, keyed by user id) is a process-wide
    # singleton shared with every other test using the same "admin-user" id
    # — reset it so this test's own calls don't inherit exhausted tokens
    # from (or leak them into) another test in the same session.
    monkeypatch.setattr(ingest_module, "_trigger_limiter", ingest_module.TokenBucket(rate=3, per=300.0))

    jobs: list[dict] = []
    ingest_jobs: dict[str, dict] = {}

    def fake_create_ingest_job(channel_url=None, video_limit=None):
        job_id = f"ingest_{len(jobs) + 1}"
        job = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "message": "queued",
            "channel_url": channel_url,
            "video_limit": video_limit,
            "current_stage": "discover",
            "created_at": "2026-06-30T00:00:00+00:00",
            "started_at": None,
            "completed_at": None,
            "summary": {},
            "error": None,
            "stage_status": {},
        }
        jobs.append(job)
        ingest_jobs[job_id] = job
        return job_id

    monkeypatch.setattr(ingest_module, "create_ingest_job", fake_create_ingest_job)
    monkeypatch.setattr(ingest_module, "start_ingest_job_background", lambda job_id: None)
    monkeypatch.setattr(ingest_module, "list_ingest_jobs", lambda limit=20: list(reversed(jobs))[:limit])
    monkeypatch.setattr(ingest_module, "load_ingest_job", lambda job_id: ingest_jobs[job_id])
    monkeypatch.setattr(ingest_module, "load_ingest_events_after", lambda job_id, cursor: [
        {"id": 1, "type": "progress", "data": {"stage": "discover"}},
        {"id": 2, "type": "completed", "data": {"status": "done"}},
    ])
    monkeypatch.setattr(ingest_module, "list_archive_videos", lambda: [
        {"video_id": "video123", "title": "Talk", "duration": 10, "webpage_url": "https://example.com/1"}
    ])
    monkeypatch.setattr(ingest_module, "get_videos_stages_bulk", lambda ids: [
        {
            "video_id": "video123",
            "stage": "embed",
            "status": "completed",
            "error": None,
            "metadata": {},
            "updated_at": "2026-06-30T00:00:00+00:00",
        }
    ])
    monkeypatch.setattr(ingest_module, "get_video_stages", lambda video_id: [
        {
            "video_id": "video123",
            "stage": "embed",
            "status": "completed",
            "error": None,
            "metadata": {},
            "updated_at": "2026-06-30T00:00:00+00:00",
        }
    ])

    response = client.post("/api/ingest/sync", json={"channel_url": "https://youtube.com/channel/test", "video_limit": 3})
    assert response.status_code == 202
    ingest_job_id = response.json()["job_id"]

    response = client.post("/api/ingest/videos/video123/sync")
    assert response.status_code == 202
    assert response.json()["video_id"] == "video123"

    response = client.get("/api/ingest/jobs")
    assert response.status_code == 200
    assert response.json()["count"] >= 1

    response = client.get(f"/api/ingest/jobs/{ingest_job_id}")
    assert response.status_code == 200
    assert response.json()["job_id"] == ingest_job_id

    response = client.get(f"/api/ingest/jobs/{ingest_job_id}/events")
    assert response.status_code == 200
    assert "event: completed" in response.text

    # /explainability was folded into /provenance (see ingest.py's
    # get_ingest_provenance docstring); the response is now flat rather than
    # wrapped in an "overview" key.
    response = client.get(f"/api/ingest/jobs/{ingest_job_id}/provenance")
    assert response.status_code == 200
    assert response.json()["job_id"] == ingest_job_id

    response = client.get(f"/api/ingest/jobs/{ingest_job_id}/logs")
    assert response.status_code == 200

    response = client.get(f"/api/ingest/jobs/{ingest_job_id}/logs/discover")
    assert response.status_code == 404

    response = client.get("/api/ingest/videos")
    assert response.status_code == 200
    assert response.json()["count"] == 1

    response = client.get("/api/ingest/videos?status=completed")
    assert response.status_code == 200
    assert response.json()["count"] == 1

    response = client.get("/api/ingest/videos/video123")
    assert response.status_code == 200
    assert response.json()["overall_status"] == "completed"


def test_ingest_conflict_and_admin_validation(client, mock_settings, monkeypatch):
    import app.main as main
    from app.core.auth import CurrentUser, get_current_user
    import app.api.ingest as ingest_module
    import app.api.admin as admin_module
    import app.services.fragment_store as fragment_store
    import app.services.embedding_index as embedding_index

    # /api/admin/* now requires require_role("admin") on top of the API key —
    # override the auth dependency instead of crafting a real admin JWT.
    # monkeypatch.setitem reverts this (deleting the key) at test teardown.
    monkeypatch.setitem(
        main.app.dependency_overrides, get_current_user,
        lambda: CurrentUser(id="admin-user", persona="admin"),
    )
    # See test_ingest_sync_jobs_and_video_inventory — _trigger_limiter is a
    # process-wide singleton shared across tests using the same user id.
    monkeypatch.setattr(ingest_module, "_trigger_limiter", ingest_module.TokenBucket(rate=3, per=300.0))

    monkeypatch.setattr(ingest_module, "list_ingest_jobs", lambda limit=10: [
        {"job_id": "active_1", "status": "running"}
    ])

    response = client.post("/api/ingest/sync")
    assert response.status_code == 409

    monkeypatch.setattr(ingest_module, "list_ingest_jobs", lambda limit=10: [])
    monkeypatch.setattr(ingest_module, "create_ingest_job", lambda channel_url=None, video_limit=None: "ingest_2")
    monkeypatch.setattr(ingest_module, "start_ingest_job_background", lambda job_id: None)
    monkeypatch.setattr(ingest_module, "load_ingest_job", lambda job_id: {"job_id": job_id, "stage_status": {}, "summary": {}})
    monkeypatch.setattr(ingest_module, "list_archive_videos", lambda: [])
    monkeypatch.setattr(ingest_module, "get_videos_stages_bulk", lambda ids: [])
    monkeypatch.setattr(ingest_module, "get_video_stages", lambda video_id: [])

    response = client.post("/api/ingest/videos/video999/sync")
    assert response.status_code == 202

    monkeypatch.setattr(fragment_store, "reload", lambda: {"reloaded": True})
    monkeypatch.setattr(embedding_index, "reload", lambda: {"reloaded": True})

    response = client.post("/api/admin/reindex")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

    response = client.get("/api/admin/storage/space")
    assert response.status_code == 400

    response = client.delete("/api/admin/storage/directories/videos")
    assert response.status_code == 400

    # httpx's TestClient.delete() no longer takes a `json=` body kwarg (this
    # endpoint's DELETE genuinely expects one) — use .request() instead.
    response = client.request("DELETE", "/api/admin/storage/object", json={"key": ""})
    assert response.status_code == 400

    response = client.post("/api/admin/cleanup/local-exports")
    assert response.status_code == 200
    assert response.json()["dry_run"] is True
