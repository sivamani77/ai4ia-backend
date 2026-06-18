from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent

SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
TEMP_DIR = PROJECT_ROOT / "temp"

RUNTIME_DIR = BACKEND_DIR / "runtime"
JOBS_DIR = RUNTIME_DIR / "jobs"

RAW_VIDEO_DIR = DATA_DIR / "raw" / "youtube" / "ai4ia_channel"
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
FRAGMENTS_DIR = DATA_DIR / "fragments"
CLASSIFIED_FRAGMENTS_DIR = DATA_DIR / "fragments_classified"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
EMBEDDING_MATRIX_PATH = EMBEDDINGS_DIR / "fragment_embedding_matrix.npy"
EMBEDDING_JSONL_PATH = EMBEDDINGS_DIR / "classified_fragments_embeddings.jsonl"
STORYLINES_DIR = DATA_DIR / "storylines"
INSIGHTS_DIR = DATA_DIR / "insights"
GENERATED_ASSETS_DIR = DATA_DIR / "generated_assets"
EXPORTS_DIR = DATA_DIR / "exports"
METADATA_DIR = DATA_DIR / "metadata" / "youtube"
SPEAKERS_DIR = DATA_DIR / "metadata" / "speakers"

LATEST_ASSET_MANIFEST = GENERATED_ASSETS_DIR / "latest_asset_manifest.json"


def ensure_runtime_dirs() -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    (LOGS_DIR / "jobs").mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
