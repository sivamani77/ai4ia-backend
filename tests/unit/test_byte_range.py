"""HTTP Range parsing and the S3 range response for video streaming."""
from __future__ import annotations

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from app.api.video_media import _s3_range_response, parse_byte_range
from app.services.storage_service import StorageObject

SIZE = 1000


class TestParseByteRange:
    @pytest.mark.parametrize("header, expected", [
        ("bytes=0-99", (0, 99)),
        ("bytes=500-", (500, 999)),
        ("bytes=990-5000", (990, 999)),   # end past EOF is trimmed
        ("bytes=-100", (900, 999)),       # suffix: last 100 bytes
        ("bytes=-5000", (0, 999)),        # suffix larger than the file
        ("bytes=999-999", (999, 999)),
        ("BYTES=0-0", (0, 0)),
    ])
    def test_satisfiable(self, header, expected):
        assert parse_byte_range(header, SIZE) == expected

    @pytest.mark.parametrize("header", [
        "bytes=99999999999-",
        "bytes=1000-",
        "bytes=1000-2000",
        "bytes=500-100",
        "bytes=-0",
    ])
    def test_unsatisfiable_is_416_with_content_range(self, header):
        with pytest.raises(HTTPException) as exc:
            parse_byte_range(header, SIZE)
        assert exc.value.status_code == 416
        assert exc.value.headers == {"Content-Range": f"bytes */{SIZE}"}

    @pytest.mark.parametrize("header", ["bytes=abc-", "items=0-1", "bytes", "bytes=0-1,5-6"])
    def test_malformed_is_416(self, header):
        with pytest.raises(HTTPException) as exc:
            parse_byte_range(header, SIZE)
        assert exc.value.status_code == 416


class _FakeBody:
    def __init__(self, data: bytes):
        self._data = data

    def read(self, n: int) -> bytes:
        chunk, self._data = self._data[:n], self._data[n:]
        return chunk


class _FakeS3:
    def __init__(self, data: bytes):
        self.data = data
        self.get_calls: list[str] = []

    def head_object(self, Bucket, Key):
        return {"ContentLength": len(self.data)}

    def get_object(self, Bucket, Key, Range=None):
        self.get_calls.append(Range)
        start, end = (int(x) for x in Range.removeprefix("bytes=").split("-"))
        return {"Body": _FakeBody(self.data[start:end + 1])}


def _request(range_header: str | None) -> Request:
    headers = [(b"range", range_header.encode())] if range_header else []
    return Request({"type": "http", "method": "GET", "path": "/", "headers": headers})


@pytest.fixture
def fake_s3(monkeypatch):
    import boto3

    s3 = _FakeS3(bytes(range(256)) * 4)
    monkeypatch.setattr(boto3, "client", lambda *a, **k: s3)
    return s3


def _video_obj() -> StorageObject:
    return StorageObject(
        object_id="video", job_id="j", kind="video", storage_backend="s3",
        content_type="video/mp4", filename="v.mp4", object_key="jobs/j/v.mp4",
    )


class TestS3RangeResponse:
    def test_start_past_eof_is_416_before_any_s3_read(self, fake_s3):
        with pytest.raises(HTTPException) as exc:
            _s3_range_response(_video_obj(), _request("bytes=99999999999-"))
        assert exc.value.status_code == 416
        assert exc.value.headers["Content-Range"] == "bytes */1024"
        assert fake_s3.get_calls == []

    async def test_partial_range_headers_match_body(self, fake_s3):
        resp = _s3_range_response(_video_obj(), _request("bytes=1000-"))
        assert resp.status_code == 206
        assert resp.headers["content-range"] == "bytes 1000-1023/1024"
        assert resp.headers["content-length"] == "24"
        body = b"".join([chunk async for chunk in resp.body_iterator])
        assert body == fake_s3.data[1000:]
