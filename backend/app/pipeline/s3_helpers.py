"""S3 helpers for pipeline stage runners.

Each stage runner uses these to download inputs from S3 to /tmp
and upload outputs from /tmp back to S3.
"""
from __future__ import annotations

from pathlib import Path


def use_s3() -> bool:
    from app.core.settings import settings
    return settings.ai4ia_storage_backend == "s3"


def _s3():
    from app.services.storage_service import (
        _OBJECT_STORAGE_BUCKET,
        _OBJECT_STORAGE_REGION,
        _OBJECT_STORAGE_ACCESS_KEY,
        _OBJECT_STORAGE_SECRET_KEY,
        _OBJECT_STORAGE_ENDPOINT,
    )
    import boto3
    kwargs = {
        "aws_access_key_id": _OBJECT_STORAGE_ACCESS_KEY or None,
        "aws_secret_access_key": _OBJECT_STORAGE_SECRET_KEY or None,
        "region_name": _OBJECT_STORAGE_REGION or None,
    }
    if _OBJECT_STORAGE_ENDPOINT:
        kwargs["endpoint_url"] = _OBJECT_STORAGE_ENDPOINT
    return boto3.client("s3", **kwargs)


def _bucket() -> str:
    from app.services.storage_service import _OBJECT_STORAGE_BUCKET
    return _OBJECT_STORAGE_BUCKET


def download(s3_key: str, local_path: Path) -> Path:
    """Download s3_key to local_path. Creates parent dirs."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    _s3().download_file(_bucket(), s3_key, str(local_path))
    return local_path


def upload(local_path: Path, s3_key: str) -> None:
    """Upload local_path to s3_key."""
    _s3().upload_file(str(local_path), _bucket(), s3_key)


def list_keys(prefix: str) -> list[str]:
    """List all S3 keys under prefix."""
    s3 = _s3()
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=_bucket(), Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def key_exists(s3_key: str) -> bool:
    try:
        _s3().head_object(Bucket=_bucket(), Key=s3_key)
        return True
    except Exception:
        return False


def download_prefix(s3_prefix: str, local_dir: Path) -> list[Path]:
    """Download all objects under s3_prefix into local_dir, preserving relative structure."""
    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for key in list_keys(s3_prefix):
        rel = key[len(s3_prefix):].lstrip("/")
        if not rel:
            continue
        local_path = local_dir / rel
        download(key, local_path)
        downloaded.append(local_path)
    return downloaded


def upload_dir(local_dir: Path, s3_prefix: str, suffixes: tuple[str, ...] | None = None) -> list[str]:
    """Upload all files from local_dir to s3_prefix. Returns list of uploaded keys."""
    uploaded = []
    for f in local_dir.rglob("*"):
        if not f.is_file():
            continue
        if suffixes and not any(f.name.endswith(s) for s in suffixes):
            continue
        rel = f.relative_to(local_dir)
        key = f"{s3_prefix.rstrip('/')}/{rel}"
        upload(f, key)
        uploaded.append(key)
    return uploaded


def tmp_dir(name: str) -> Path:
    """Return a /tmp subdirectory for pipeline use."""
    p = Path(f"/tmp/ai4ia_pipeline/{name}")
    p.mkdir(parents=True, exist_ok=True)
    return p


def cleanup_tmp_dir(name: str) -> None:
    """Delete a tmp_dir after its contents have been uploaded to S3."""
    import shutil
    p = Path(f"/tmp/ai4ia_pipeline/{name}")
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)
