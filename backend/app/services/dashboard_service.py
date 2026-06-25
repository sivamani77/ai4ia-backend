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


def count_files(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    return len(list(path.glob(pattern)))


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
    embedding_matrix = EMBEDDINGS_DIR / "fragment_embedding_matrix.npy"
    embedding_jsonl = EMBEDDINGS_DIR / "classified_fragments_embeddings.jsonl"

    return {
        "archive": {
            "raw_videos": count_files(RAW_VIDEO_DIR, "*.mp4"),
            "audio_files": count_files(AUDIO_DIR, "*.wav"),
            "transcripts_json": count_files(TRANSCRIPTS_DIR, "*.json"),
            "transcripts_txt": count_files(TRANSCRIPTS_DIR, "*.txt"),
            "fragment_files": count_files(FRAGMENTS_DIR, "*.fragments.json"),
            "classified_fragment_files": count_files(CLASSIFIED_FRAGMENTS_DIR, "*.classified.json"),
            "classified_fragments": count_classified_fragments(),
            "embeddings_ready": embedding_matrix.exists() and embedding_jsonl.exists(),
            "embedding_matrix": str(embedding_matrix) if embedding_matrix.exists() else None,
        },
        "latest": {
            "storyline": _latest_storyline_id(),
            "asset_manifest": str(LATEST_ASSET_MANIFEST) if LATEST_ASSET_MANIFEST.exists() else None,
            "export": latest_file(EXPORTS_DIR, "*.mp4"),
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
