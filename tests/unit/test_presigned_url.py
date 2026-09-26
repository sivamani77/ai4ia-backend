"""Presigned S3 URLs must target a host the bucket's region accepts."""
from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

import pytest

import app.api.video_media as video_media


@pytest.fixture
def s3_settings(monkeypatch):
    monkeypatch.setattr(video_media, "_OBJECT_STORAGE_BUCKET", "ai4ia-data")
    monkeypatch.setattr(video_media, "_OBJECT_STORAGE_REGION", "ap-south-2")
    monkeypatch.setattr(video_media, "_OBJECT_STORAGE_ACCESS_KEY", "AKIATESTTESTTESTTEST")
    monkeypatch.setattr(video_media, "_OBJECT_STORAGE_SECRET_KEY", "test-secret")
    monkeypatch.setattr(video_media, "_OBJECT_STORAGE_ENDPOINT", "")
    return monkeypatch


def test_opt_in_region_signs_for_regional_host(s3_settings):
    url = urlsplit(video_media._make_presigned_url("jobs/j/v.mp4"))
    assert url.netloc == "ai4ia-data.s3.ap-south-2.amazonaws.com"
    assert url.path == "/jobs/j/v.mp4"
    query = parse_qs(url.query)
    assert query["X-Amz-Algorithm"] == ["AWS4-HMAC-SHA256"]
    assert query["X-Amz-Expires"] == [str(video_media._PRESIGNED_EXPIRY)]
    assert "/ap-south-2/s3/" in query["X-Amz-Credential"][0]


def test_custom_endpoint_is_kept(s3_settings):
    s3_settings.setattr(video_media, "_OBJECT_STORAGE_ENDPOINT", "http://minio:9000")
    url = urlsplit(video_media._make_presigned_url("jobs/j/v.mp4"))
    assert url.netloc == "minio:9000"
    assert url.path == "/ai4ia-data/jobs/j/v.mp4"
