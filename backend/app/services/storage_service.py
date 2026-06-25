"""
Storage abstraction layer.

LocalStorageService resolves files from disk.
Later, ObjectStorageService can resolve from S3/MinIO/GCS without changing API contracts.
"""
from __future__ import annotations

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
# Singleton factory
# ---------------------------------------------------------------------------

_service_instance: StorageService | None = None


def get_storage_service() -> StorageService:
    global _service_instance
    if _service_instance is None:
        backend = STORAGE_BACKEND
        if backend == "local":
            _service_instance = LocalStorageService()
        else:
            raise NotImplementedError(f"Storage backend '{backend}' not implemented")
    return _service_instance
