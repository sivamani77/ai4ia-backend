from app.core.settings import settings

APP_NAME = "AI4IA Demo Video Studio Backend"
API_PREFIX = "/api"

OLLAMA_BASE_URL = settings.ollama_base_url
REDIS_URL = settings.redis_url
SUPABASE_URL = settings.supabase_url
SUPABASE_SERVICE_KEY = settings.supabase_service_key

EMBEDDING_MODEL = "nomic-embed-text:v1.5"
STRONG_LLM_MODEL = "llama3.1:70b"
FAST_LLM_MODEL = "llama3.1:8b"
SPEAKER_EXTRACTION_MODEL = settings.speaker_extraction_model
TTS_PROVIDER = "edge-tts"
EXPORT_ENGINE = "ffmpeg"

SEMANTIC_TOP_K_DEFAULT = 20

INSIGHT_MODEL = settings.insight_model
INSIGHT_DEFAULT_LIMIT = 15
