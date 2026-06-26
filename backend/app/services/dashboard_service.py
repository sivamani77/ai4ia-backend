from pathlib import Path

from app.core.config import (
    EMBEDDING_MODEL,
    STRONG_LLM_MODEL,
    FAST_LLM_MODEL,
    TTS_PROVIDER,
    EXPORT_ENGINE,
)
from app.core.paths import (
    RAW_VIDEO_DIR,
    AUDIO_DIR,
    TRANSCRIPTS_DIR,
    FRAGMENTS_DIR,
    CLASSIFIED_FRAGMENTS_DIR,
    EMBEDDINGS_DIR,
    STORYLINES_DIR,
    GENERATED_ASSETS_DIR,
    EXPORTS_DIR,
    LATEST_ASSET_MANIFEST,
)
from app.core.settings import settings


def count_files(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    return len(list(path.glob(pattern)))


def _s3_count(prefix: str, suffix: str) -> int:
    try:
        from app.services.storage_service import (
            _OBJECT_STORAGE_BUCKET, _OBJECT_STORAGE_REGION,
            _OBJECT_STORAGE_ACCESS_KEY, _OBJECT_STORAGE_SECRET_KEY, _OBJECT_STORAGE_ENDPOINT,
        )
        import boto3
        kwargs = {
            "aws_access_key_id": _OBJECT_STORAGE_ACCESS_KEY or None,
            "aws_secret_access_key": _OBJECT_STORAGE_SECRET_KEY or None,
            "region_name": _OBJECT_STORAGE_REGION or None,
        }
        if _OBJECT_STORAGE_ENDPOINT:
            kwargs["endpoint_url"] = _OBJECT_STORAGE_ENDPOINT
        s3 = boto3.client("s3", **kwargs)
        paginator = s3.get_paginator("list_objects_v2")
        count = 0
        for page in paginator.paginate(Bucket=_OBJECT_STORAGE_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(suffix):
                    count += 1
        return count
    except Exception:
        return 0


def _s3_key_exists(key: str) -> bool:
    try:
        from app.services.storage_service import (
            _OBJECT_STORAGE_BUCKET, _OBJECT_STORAGE_REGION,
            _OBJECT_STORAGE_ACCESS_KEY, _OBJECT_STORAGE_SECRET_KEY, _OBJECT_STORAGE_ENDPOINT,
        )
        import boto3
        kwargs = {
            "aws_access_key_id": _OBJECT_STORAGE_ACCESS_KEY or None,
            "aws_secret_access_key": _OBJECT_STORAGE_SECRET_KEY or None,
            "region_name": _OBJECT_STORAGE_REGION or None,
        }
        if _OBJECT_STORAGE_ENDPOINT:
            kwargs["endpoint_url"] = _OBJECT_STORAGE_ENDPOINT
        s3 = boto3.client("s3", **kwargs)
        s3.head_object(Bucket=_OBJECT_STORAGE_BUCKET, Key=key)
        return True
    except Exception:
        return False


def _s3_latest_file(prefix: str, suffix: str) -> str | None:
    try:
        from app.services.storage_service import (
            _OBJECT_STORAGE_BUCKET, _OBJECT_STORAGE_REGION,
            _OBJECT_STORAGE_ACCESS_KEY, _OBJECT_STORAGE_SECRET_KEY, _OBJECT_STORAGE_ENDPOINT,
        )
        import boto3
        kwargs = {
            "aws_access_key_id": _OBJECT_STORAGE_ACCESS_KEY or None,
            "aws_secret_access_key": _OBJECT_STORAGE_SECRET_KEY or None,
            "region_name": _OBJECT_STORAGE_REGION or None,
        }
        if _OBJECT_STORAGE_ENDPOINT:
            kwargs["endpoint_url"] = _OBJECT_STORAGE_ENDPOINT
        s3 = boto3.client("s3", **kwargs)
        paginator = s3.get_paginator("list_objects_v2")
        matches = []
        for page in paginator.paginate(Bucket=_OBJECT_STORAGE_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(suffix):
                    matches.append((obj["LastModified"], Path(obj["Key"]).name))
        if not matches:
            return None
        matches.sort(reverse=True)
        return matches[0][1]
    except Exception:
        return None


def count_classified_fragments() -> int:
    try:
        from app.db.repositories.fragment_repository import FragmentRepository
        return FragmentRepository().count()
    except Exception:
        return 0


def _latest_storyline_id() -> str | None:
    try:
        from app.db.repositories.storyline_repository import StorylineRepository
        row = StorylineRepository().get_latest()
        return row.get("id") if row else None
    except Exception:
        return latest_file(STORYLINES_DIR, "storyline_*.json")


def latest_file(path: Path, pattern: str) -> str | None:
    if not path.exists():
        return None

    files = sorted(path.glob(pattern))
    if not files:
        return None

    return files[-1].name


def get_dashboard() -> dict:
    use_s3 = settings.ai4ia_storage_backend == "s3"

    if use_s3:
        raw_videos = _s3_count("raw/youtube/ai4ia_channel/", ".mp4")
        audio_files = _s3_count("raw/audio/", ".wav")
        transcripts_json = _s3_count("raw/transcripts/", ".json")
        transcripts_txt = _s3_count("raw/transcripts/", ".txt")
        fragment_files = _s3_count("raw/fragments/", ".fragments.json")
        classified_fragment_files = _s3_count("raw/fragments_classified/", ".classified.json")
        from app.services.storage_service import _OBJECT_STORAGE_BUCKET
        embeddings_ready = _s3_key_exists("raw/embeddings/classified_fragments_embeddings.jsonl")
        embedding_matrix_val = f"s3://{_OBJECT_STORAGE_BUCKET}/raw/embeddings/classified_fragments_embeddings.jsonl" if embeddings_ready else None
        latest_export = _s3_latest_file("exports/", ".mp4")
        latest_manifest = _s3_latest_file("exports/", "_export_manifest.json")
    else:
        embedding_matrix = EMBEDDINGS_DIR / "fragment_embedding_matrix.npy"
        embedding_jsonl = EMBEDDINGS_DIR / "classified_fragments_embeddings.jsonl"
        raw_videos = count_files(RAW_VIDEO_DIR, "*.mp4")
        audio_files = count_files(AUDIO_DIR, "*.wav")
        transcripts_json = count_files(TRANSCRIPTS_DIR, "*.json")
        transcripts_txt = count_files(TRANSCRIPTS_DIR, "*.txt")
        fragment_files = count_files(FRAGMENTS_DIR, "*.fragments.json")
        classified_fragment_files = count_files(CLASSIFIED_FRAGMENTS_DIR, "*.classified.json")
        embeddings_ready = embedding_matrix.exists() and embedding_jsonl.exists()
        embedding_matrix_val = str(embedding_matrix) if embedding_matrix.exists() else None
        latest_export = latest_file(EXPORTS_DIR, "*.mp4")
        latest_manifest = str(LATEST_ASSET_MANIFEST) if LATEST_ASSET_MANIFEST.exists() else None

    return {
        "archive": {
            "raw_videos": raw_videos,
            "audio_files": audio_files,
            "transcripts_json": transcripts_json,
            "transcripts_txt": transcripts_txt,
            "fragment_files": fragment_files,
            "classified_fragment_files": classified_fragment_files,
            "classified_fragments": count_classified_fragments(),
            "embeddings_ready": embeddings_ready,
            "embedding_matrix": embedding_matrix_val,
        },
        "latest": {
            "storyline": _latest_storyline_id(),
            "asset_manifest": latest_manifest,
            "export": latest_export,
        },
        "models": {
            "embedding": EMBEDDING_MODEL,
            "strong_llm": STRONG_LLM_MODEL,
            "fast_llm": FAST_LLM_MODEL,
            "tts": TTS_PROVIDER,
            "export": EXPORT_ENGINE,
        },
        "paths": {
            "storylines_dir": str(STORYLINES_DIR),
            "generated_assets_dir": str(GENERATED_ASSETS_DIR),
            "exports_dir": str(EXPORTS_DIR),
        }
    }
