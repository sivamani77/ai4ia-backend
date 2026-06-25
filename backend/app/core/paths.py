from pathlib import Path

from app.core.settings import settings

PROJECT_ROOT = settings.project_root
BACKEND_DIR = PROJECT_ROOT / "backend"

SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
TEMP_DIR = PROJECT_ROOT / "temp"

RUNTIME_DIR = BACKEND_DIR / "runtime"

RAW_VIDEO_DIR = DATA_DIR / "raw" / "youtube" / "ai4ia_channel"
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
FRAGMENTS_DIR = DATA_DIR / "fragments"
CLASSIFIED_FRAGMENTS_DIR = DATA_DIR / "fragments_classified"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
CHROMA_DIR = DATA_DIR / "chroma"
STORYLINES_DIR = DATA_DIR / "storylines"
INSIGHTS_DIR = DATA_DIR / "insights"
GENERATED_ASSETS_DIR = settings.ai4ia_local_assets_dir
EXPORTS_DIR = settings.ai4ia_local_exports_dir
METADATA_DIR = DATA_DIR / "metadata" / "youtube"
SPEAKERS_DIR = DATA_DIR / "metadata" / "speakers"

LATEST_ASSET_MANIFEST = GENERATED_ASSETS_DIR / "latest_asset_manifest.json"


def ensure_runtime_dirs() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    (LOGS_DIR / "jobs").mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
