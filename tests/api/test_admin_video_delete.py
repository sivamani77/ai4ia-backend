"""DELETE /api/admin/storage/videos/{video_stem} — confirms deletion now
propagates to ChromaDB and Postgres, not just S3, and that a Chroma failure
aborts before Postgres/S3 are touched (fail-closed).

See local/reports/AI4IA_Production_Readiness_Audit.xlsx, Checklist 4.2.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.auth import CurrentUser, get_current_user


@pytest.fixture(autouse=True)
def _clear_dependency_overrides(api_app):
    """api_app returns the process-wide main.app singleton — clear overrides
    so they don't leak into other test modules in the same session."""
    yield
    api_app.dependency_overrides.clear()


@pytest.fixture
def admin_client(api_app):
    api_app.dependency_overrides[get_current_user] = lambda: CurrentUser(id="admin-1", persona="admin")
    with TestClient(api_app) as c:
        yield c


def test_delete_video_artifacts_deletes_chroma_then_postgres_then_s3(admin_client, mock_settings, monkeypatch):
    import app.api.admin as admin_module

    mock_settings.ai4ia_storage_backend = "s3"
    stem = "0001_abc123_speaker"
    fragment_ids = [f"{stem}_0001", f"{stem}_0002"]
    call_order: list[str] = []

    monkeypatch.setattr(
        admin_module.fragment_store, "list_fragment_ids_by_source_video",
        lambda source_video: fragment_ids if source_video == stem else [],
    )

    def fake_embedding_delete(ids):
        call_order.append("chroma")
        assert ids == fragment_ids
        return len(ids)

    def fake_fragment_delete_many(ids):
        call_order.append("postgres")
        assert ids == fragment_ids
        return list(ids)

    monkeypatch.setattr(admin_module.embedding_index, "delete", fake_embedding_delete)
    monkeypatch.setattr(admin_module.fragment_store, "delete_many", fake_fragment_delete_many)

    monkeypatch.setattr(admin_module, "_list_keys", lambda prefix: [f"{prefix}{stem}.mp4"])

    def fake_delete_keys(keys):
        call_order.append("s3")
        return len(keys)

    monkeypatch.setattr(admin_module, "_delete_keys", fake_delete_keys)

    response = admin_client.delete(f"/api/admin/storage/videos/{stem}", params={"confirm": stem})

    assert response.status_code == 200
    body = response.json()
    assert body["fragments_deleted"] == 2
    assert body["embeddings_deleted"] == 2
    assert body["video_stem"] == stem
    assert call_order == ["chroma", "postgres", "s3"]


def test_delete_video_artifacts_chroma_failure_aborts_before_postgres_and_s3(admin_client, mock_settings, monkeypatch):
    import app.api.admin as admin_module

    mock_settings.ai4ia_storage_backend = "s3"
    stem = "0001_abc123_speaker"
    fragment_ids = [f"{stem}_0001"]

    monkeypatch.setattr(
        admin_module.fragment_store, "list_fragment_ids_by_source_video", lambda source_video: fragment_ids
    )

    def fake_embedding_delete(ids):
        raise RuntimeError("ChromaDB unreachable")

    monkeypatch.setattr(admin_module.embedding_index, "delete", fake_embedding_delete)

    postgres_called = []
    s3_called = []
    monkeypatch.setattr(admin_module.fragment_store, "delete_many", lambda ids: postgres_called.append(ids))
    monkeypatch.setattr(admin_module, "_list_keys", lambda prefix: [f"{prefix}{stem}.mp4"])
    monkeypatch.setattr(admin_module, "_delete_keys", lambda keys: s3_called.append(keys))

    response = admin_client.delete(f"/api/admin/storage/videos/{stem}", params={"confirm": stem})

    assert response.status_code == 502
    assert postgres_called == []
    assert s3_called == []


def test_delete_video_artifacts_404_when_nothing_found(admin_client, mock_settings, monkeypatch):
    import app.api.admin as admin_module

    mock_settings.ai4ia_storage_backend = "s3"
    stem = "nonexistent_video"

    monkeypatch.setattr(admin_module.fragment_store, "list_fragment_ids_by_source_video", lambda source_video: [])
    monkeypatch.setattr(admin_module, "_list_keys", lambda prefix: [])

    response = admin_client.delete(f"/api/admin/storage/videos/{stem}", params={"confirm": stem})

    assert response.status_code == 404


def test_delete_video_artifacts_requires_confirm_match(admin_client, mock_settings, monkeypatch):
    import app.api.admin as admin_module

    mock_settings.ai4ia_storage_backend = "s3"
    monkeypatch.setattr(admin_module.fragment_store, "list_fragment_ids_by_source_video", lambda source_video: [])
    monkeypatch.setattr(admin_module, "_list_keys", lambda prefix: [f"{prefix}some_video.mp4"])

    response = admin_client.delete("/api/admin/storage/videos/some_video", params={"confirm": "wrong_stem"})

    assert response.status_code == 400


def test_delete_video_artifacts_requires_admin(api_app, mock_settings):
    """Non-admin callers must not reach this endpoint at all — regression
    guard for the router-level admin gate this fix relies on."""
    api_app.dependency_overrides[get_current_user] = lambda: CurrentUser(id="user-1", persona="general")
    with TestClient(api_app) as client:
        response = client.delete("/api/admin/storage/videos/some_video", params={"confirm": "some_video"})
    assert response.status_code == 403
