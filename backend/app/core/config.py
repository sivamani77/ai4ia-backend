import os

APP_NAME = "AI4IA Demo Video Studio Backend"
API_PREFIX = "/api"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama-mobius-sales.mobiusdtaas.ai")

EMBEDDING_MODEL = "nomic-embed-text:v1.5"
STRONG_LLM_MODEL = "llama3.1:70b"
FAST_LLM_MODEL = "llama3.1:8b"
SPEAKER_EXTRACTION_MODEL = os.getenv("SPEAKER_EXTRACTION_MODEL", STRONG_LLM_MODEL)
TTS_PROVIDER = "edge-tts"
EXPORT_ENGINE = "ffmpeg"

SEMANTIC_TOP_K_DEFAULT = 20

# Synthesis is reasoning-heavy and runs once per insight, so use the strong model.
INSIGHT_MODEL = os.getenv("INSIGHT_MODEL", STRONG_LLM_MODEL)
INSIGHT_DEFAULT_LIMIT = 15

