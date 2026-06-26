"""
Storage abstraction layer.

LocalStorageService resolves files from disk.
S3StorageService resolves from AWS S3 (or S3-compatible endpoints).
Switch via AI4IA_STORAGE_BACKEND=s3 in .env.
"""
from __future__ import annotations

import io
import json
import mimetypes
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from app.core.settings import settings

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------

STORAGE_BACKEND = settings.ai4ia_storage_backend

# Local paths (overridable via env)
_LOCAL_EXPORTS_DIR = settings.ai4ia_local_exports_dir
_LOCAL_ASSETS_DIR = settings.ai4ia_local_assets_dir
_LOCAL_JOBS_DIR = settings.ai4ia_local_jobs_dir

# Future object storage config (placeholders — not implemented)
_OBJECT_STORAGE_BUCKET = settings.ai4ia_object_storage_bucket
_OBJECT_STORAGE_ENDPOINT = settings.ai4ia_object_storage_endpoint
_OBJECT_STORAGE_REGION = settings.ai4ia_object_storage_region
_OBJECT_STORAGE_ACCESS_KEY = settings.ai4ia_object_storage_access_key
_OBJECT_STORAGE_SECRET_KEY = settings.ai4ia_object_storage_secret_key

# Allowed roots for path-traversal protection
ALLOWED_MEDIA_ROOTS: list[Path] = [
    _LOCAL_EXPORTS_DIR,
    _LOCAL_ASSETS_DIR,
    _LOCAL_JOBS_DIR,
]


# ---------------------------------------------------------------------------
# StorageObject
# ---------------------------------------------------------------------------

@dataclass
class StorageObject:
    object_id: str
    job_id: str | None
    kind: str                      # video, subtitle, manifest, visual_asset, voiceover_audio, card, thumbnail
    storage_backend: str           # local | s3 | gcs | azure
    content_type: str
    filename: str
    local_path: str | None = None
    object_key: str | None = None  # for object storage
    size_bytes: int | None = None
    created_at: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "job_id": self.job_id,
            "kind": self.kind,
            "storage_backend": self.storage_backend,
            "content_type": self.content_type,
            "filename": self.filename,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
            **self.metadata,
        }


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class StorageService(ABC):

    @abstractmethod
    def resolve(self, job_id: str, object_id: str) -> StorageObject:
        ...

    @abstractmethod
    def exists(self, obj: StorageObject) -> bool:
        ...

    @abstractmethod
    def open_stream(self, obj: StorageObject) -> Iterator[bytes]:
        ...

    @abstractmethod
    def get_local_path(self, obj: StorageObject) -> Path | None:
        """Return a local Path for FileResponse use, or None if not locally available."""
        ...

    def build_api_urls(self, job_id: str, object_id: str) -> dict[str, str]:
        kind = self.resolve(job_id, object_id).kind
        base = f"/api/video-jobs/{job_id}"
        if kind == "video":
            return {
                "stream_url": f"{base}/video/stream",
                "download_url": f"{base}/video/download",
            }
        if kind == "subtitle":
            fmt = object_id  # "srt" or "vtt"
            return {"stream_url": f"{base}/subtitles/{fmt}"}
        if kind == "manifest":
            return {"stream_url": f"{base}/manifest"}
        # generic assets
        return {
            "stream_url": f"{base}/assets/{object_id}/stream",
            "download_url": f"{base}/assets/{object_id}/download",
        }


# ---------------------------------------------------------------------------
# Local storage implementation
# ---------------------------------------------------------------------------

def _safe_path(raw_path: str) -> Path | None:
    """Resolve path and verify it falls under an allowed root."""
    try:
        p = Path(raw_path).resolve()
    except Exception:
        return None
    for root in ALLOWED_MEDIA_ROOTS:
        try:
            p.relative_to(root)
            return p
        except ValueError:
            continue
    return None


def _guess_content_type(path: Path) -> str:
    ct, _ = mimetypes.guess_type(str(path))
    if ct:
        return ct
    ext = path.suffix.lower()
    return {
        ".mp4": "video/mp4",
        ".srt": "application/x-subrip",
        ".vtt": "text/vtt",
        ".json": "application/json",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".webm": "video/webm",
    }.get(ext, "application/octet-stream")


def _registry_path(job_id: str) -> Path:
    return _LOCAL_JOBS_DIR / job_id / "media_registry.json"


def load_registry(job_id: str) -> dict[str, Any]:
    p = _registry_path(job_id)
    if not p.exists():
        raise FileNotFoundError(f"Media registry not found for job {job_id}")
    return json.loads(p.read_text(encoding="utf-8"))


def save_registry(job_id: str, registry: dict[str, Any]) -> None:
    p = _registry_path(job_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


class LocalStorageService(StorageService):

    def resolve(self, job_id: str, object_id: str) -> StorageObject:
        reg = load_registry(job_id)
        objects = reg.get("objects", {})
        entry = objects.get(object_id)
        if not entry:
            raise KeyError(f"Asset '{object_id}' not in media registry for job {job_id}")

        raw_path = entry.get("local_path", "")
        safe = _safe_path(raw_path) if raw_path else None
        ct = entry.get("content_type") or (
            _guess_content_type(Path(raw_path)) if raw_path else "application/octet-stream"
        )
        size = None
        if safe and safe.exists():
            size = safe.stat().st_size

        meta = {k: v for k, v in entry.items()
                if k not in ("object_id", "kind", "local_path", "content_type", "filename",
                             "storage_backend", "job_id", "size_bytes", "created_at")}

        return StorageObject(
            object_id=object_id,
            job_id=job_id,
            kind=entry.get("kind", "unknown"),
            storage_backend="local",
            content_type=ct,
            filename=entry.get("filename", Path(raw_path).name if raw_path else object_id),
            local_path=str(safe) if safe else raw_path,
            size_bytes=size,
            metadata=meta,
        )

    def exists(self, obj: StorageObject) -> bool:
        if not obj.local_path:
            return False
        safe = _safe_path(obj.local_path)
        return bool(safe and safe.exists())

    def open_stream(self, obj: StorageObject) -> Iterator[bytes]:
        if not obj.local_path:
            raise FileNotFoundError("No local path for object")
        safe = _safe_path(obj.local_path)
        if not safe or not safe.exists():
            raise FileNotFoundError(f"File not found: {obj.local_path}")
        with safe.open("rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                yield chunk

    def get_local_path(self, obj: StorageObject) -> Path | None:
        if not obj.local_path:
            return None
        return _safe_path(obj.local_path)

    def list_objects(self, job_id: str) -> list[StorageObject]:
        reg = load_registry(job_id)
        objects = []
        for oid in reg.get("objects", {}):
            try:
                objects.append(self.resolve(job_id, oid))
            except Exception:
                pass
        return objects


# ---------------------------------------------------------------------------
# S3 storage implementation
# ---------------------------------------------------------------------------

class S3StorageService(StorageService):
    """Resolves StorageObjects from AWS S3 (or any S3-compatible endpoint)."""

    def __init__(self) -> None:
        import boto3

        kwargs: dict[str, Any] = {
            "aws_access_key_id": _OBJECT_STORAGE_ACCESS_KEY or None,
            "aws_secret_access_key": _OBJECT_STORAGE_SECRET_KEY or None,
            "region_name": _OBJECT_STORAGE_REGION or None,
        }
        if _OBJECT_STORAGE_ENDPOINT:
            kwargs["endpoint_url"] = _OBJECT_STORAGE_ENDPOINT

        self._s3 = boto3.client("s3", **kwargs)
        self._bucket = _OBJECT_STORAGE_BUCKET

    # ------------------------------------------------------------------
    # Registry helpers (same structure as local — stored as JSON in S3)
    # ------------------------------------------------------------------

    def _registry_key(self, job_id: str) -> str:
        return f"jobs/{job_id}/media_registry.json"

    def _load_registry(self, job_id: str) -> dict[str, Any]:
        key = self._registry_key(job_id)
        try:
            resp = self._s3.get_object(Bucket=self._bucket, Key=key)
            return json.loads(resp["Body"].read().decode("utf-8"))
        except self._s3.exceptions.NoSuchKey:
            raise FileNotFoundError(f"Media registry not found for job {job_id} in S3")

    # ------------------------------------------------------------------
    # StorageService interface
    # ------------------------------------------------------------------

    def resolve(self, job_id: str, object_id: str) -> StorageObject:
        reg = self._load_registry(job_id)
        objects = reg.get("objects", {})
        entry = objects.get(object_id)
        if not entry:
            raise KeyError(f"Asset '{object_id}' not in media registry for job {job_id}")

        object_key = entry.get("object_key") or entry.get("local_path", "")
        ct = entry.get("content_type") or (
            _guess_content_type(Path(object_key)) if object_key else "application/octet-stream"
        )

        size = entry.get("size_bytes")
        if size is None and object_key:
            try:
                head = self._s3.head_object(Bucket=self._bucket, Key=object_key)
                size = head.get("ContentLength")
            except Exception:
                pass

        meta = {k: v for k, v in entry.items()
                if k not in ("object_id", "kind", "local_path", "object_key",
                             "content_type", "filename", "storage_backend",
                             "job_id", "size_bytes", "created_at")}

        return StorageObject(
            object_id=object_id,
            job_id=job_id,
            kind=entry.get("kind", "unknown"),
            storage_backend="s3",
            content_type=ct,
            filename=entry.get("filename", Path(object_key).name if object_key else object_id),
            object_key=object_key,
            size_bytes=size,
            created_at=entry.get("created_at"),
            metadata=meta,
        )

    def exists(self, obj: StorageObject) -> bool:
        if not obj.object_key:
            return False
        try:
            self._s3.head_object(Bucket=self._bucket, Key=obj.object_key)
            return True
        except Exception:
            return False

    def open_stream(self, obj: StorageObject) -> Iterator[bytes]:
        if not obj.object_key:
            raise FileNotFoundError("No object key for S3 object")
        resp = self._s3.get_object(Bucket=self._bucket, Key=obj.object_key)
        body = resp["Body"]
        while True:
            chunk = body.read(65536)
            if not chunk:
                break
            yield chunk

    def get_local_path(self, obj: StorageObject) -> Path | None:
        # S3 objects have no local path — callers must use open_stream
        return None

    def upload_fileobj(self, fileobj: io.IOBase, key: str,
                       content_type: str = "application/octet-stream") -> None:
        self._s3.upload_fileobj(
            fileobj, self._bucket, key,
            ExtraArgs={"ContentType": content_type},
        )

    def download_fileobj(self, key: str, fileobj: io.IOBase) -> None:
        self._s3.download_fileobj(self._bucket, key, fileobj)

    def delete_object(self, key: str) -> None:
        self._s3.delete_object(Bucket=self._bucket, Key=key)

    def list_objects(self, prefix: str = "", max_keys: int = 1000) -> list[dict[str, Any]]:
        resp = self._s3.list_objects_v2(
            Bucket=self._bucket, Prefix=prefix, MaxKeys=max_keys
        )
        return resp.get("Contents", [])


# ---------------------------------------------------------------------------
# S3 upload helper — call after a job completes to push all outputs to S3
# and rewrite the media registry with object_key entries.
# ---------------------------------------------------------------------------

def upload_job_outputs_to_s3(job_id: str) -> dict[str, Any]:
    """Upload all local job files to S3 and rewrite the media registry.

    Reads the local media_registry.json, uploads every file that has a
    local_path, replaces local_path with object_key in the registry, and
    writes the updated registry back to S3 (and locally so the API can
    serve it immediately without re-fetching from S3).

    Returns the updated registry dict.
    """
    import logging
    log = logging.getLogger("ai4ia.storage.s3_upload")

    svc = get_storage_service()
    if not isinstance(svc, S3StorageService):
        raise RuntimeError("upload_job_outputs_to_s3 called but storage backend is not s3")

    reg = load_registry(job_id)
    objects = reg.get("objects", {})
    updated: dict[str, Any] = {}

    # Build a map of known S3 keys from job paths (set by stage_09 before cleanup)
    from app.services.job_store import load_job as _load_job
    try:
        _job = _load_job(job_id)
        _jpaths = _job.get("paths", {})
    except Exception:
        _jpaths = {}
    _known_s3_keys: dict[str, str] = {
        "video":    _jpaths.get("video_s3_key", ""),
        "manifest": _jpaths.get("export_manifest_s3_key", ""),
        "srt":      _jpaths.get("srt_s3_key", ""),
        "vtt":      _jpaths.get("vtt_s3_key", ""),
    }

    for oid, entry in objects.items():
        local_path = entry.get("local_path", "")
        p = Path(local_path) if local_path else None

        if not p or not p.exists():
            # File already uploaded by stage — use the known S3 key if available
            known_key = _known_s3_keys.get(oid, "")
            if known_key:
                try:
                    svc._s3.head_object(Bucket=svc._bucket, Key=known_key)
                    new_entry = {k: v for k, v in entry.items() if k != "local_path"}
                    new_entry["object_key"] = known_key
                    new_entry["storage_backend"] = "s3"
                    updated[oid] = new_entry
                    log.info("[%s] %s already in S3 at %s", job_id, oid, known_key)
                    continue
                except Exception:
                    pass
            log.warning("[%s] skipping %s — local file missing: %s", job_id, oid, local_path)
            updated[oid] = entry
            continue

        object_key = f"jobs/{job_id}/{p.name}"
        ct = entry.get("content_type") or _guess_content_type(p)

        try:
            with p.open("rb") as f:
                svc.upload_fileobj(f, object_key, content_type=ct)
            log.info("[%s] uploaded %s → s3://%s/%s", job_id, oid, svc._bucket, object_key)
        except Exception as exc:
            log.error("[%s] failed to upload %s: %s", job_id, oid, exc)
            updated[oid] = entry
            continue

        new_entry = {k: v for k, v in entry.items() if k != "local_path"}
        new_entry["object_key"] = object_key
        new_entry["storage_backend"] = "s3"
        updated[oid] = new_entry

    # Add any known S3 entries that build_media_registry missed
    # (happens when local files were cleaned up before registry was built)
    _kind_map = {"video": "video", "manifest": "manifest", "srt": "subtitle", "vtt": "subtitle"}
    _ct_map = {
        "video": "video/mp4",
        "manifest": "application/json",
        "srt": "application/x-subrip",
        "vtt": "text/vtt",
    }
    for oid, s3_key in _known_s3_keys.items():
        if not s3_key or oid in updated:
            continue
        try:
            svc._s3.head_object(Bucket=svc._bucket, Key=s3_key)
            updated[oid] = {
                "object_id": oid,
                "kind": _kind_map.get(oid, oid),
                "object_key": s3_key,
                "storage_backend": "s3",
                "content_type": _ct_map.get(oid, "application/octet-stream"),
                "filename": Path(s3_key).name,
                "job_id": job_id,
            }
            log.info("[%s] recovered missing registry entry %s from S3 key %s", job_id, oid, s3_key)
        except Exception:
            pass

    reg["objects"] = updated
    reg["storage_backend"] = "s3"

    # Write registry to S3
    registry_key = f"jobs/{job_id}/media_registry.json"
    registry_bytes = json.dumps(reg, indent=2, ensure_ascii=False).encode("utf-8")
    svc.upload_fileobj(io.BytesIO(registry_bytes), registry_key, content_type="application/json")

    # Also keep a local copy so the API can serve immediately
    save_registry(job_id, reg)

    log.info("[%s] media registry uploaded to s3://%s/%s", job_id, svc._bucket, registry_key)
    return reg


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_service_instance: StorageService | None = None


def get_storage_service() -> StorageService:
    global _service_instance
    if _service_instance is None:
        backend = STORAGE_BACKEND
        if backend == "local":
            _service_instance = LocalStorageService()
        elif backend == "s3":
            _service_instance = S3StorageService()
        else:
            raise NotImplementedError(f"Storage backend '{backend}' not implemented")
    return _service_instance
