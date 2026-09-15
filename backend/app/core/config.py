from app.core.settings import settings

APP_NAME = "AI4IA Demo Video Studio Backend"
API_PREFIX = "/api"

OLLAMA_BASE_URL = settings.ollama_base_url
REDIS_URL = settings.redis_url

EMBEDDING_MODEL = settings.embedding_model
FAST_LLM_MODEL = settings.fast_llm_model
SPEAKER_EXTRACTION_MODEL = settings.speaker_extraction_model

VLLM_BASE_URL = settings.vllm_base_url
VLLM_MODEL = settings.vllm_model
VLLM_CHAT_ENDPOINT = settings.vllm_chat_endpoint
STRONG_LLM_MODEL = VLLM_MODEL   # alias — strong model is the vLLM model

# Separate, larger vLLM deployment (gpt-oss-120b) — used by the video pipeline
# (scripts/07_generate_storyline.py) and, via INSIGHT_MODEL below, by
# Insights/Relay of Ideas/Theme Synthesis/Speaker Comparison. Distinct host
# from VLLM_BASE_URL above, which still serves the smaller 20b model for any
# other feature (translation, concept mindmap) that doesn't opt into 120b.
STORYLINE_VLLM_BASE_URL = settings.storyline_vllm_base_url
STORYLINE_VLLM_MODEL = settings.storyline_vllm_model
TTS_PROVIDER = "edge-tts"
EXPORT_ENGINE = "ffmpeg"

SEMANTIC_TOP_K_DEFAULT = 20

INSIGHT_MODEL = settings.insight_model
INSIGHT_DEFAULT_LIMIT = 15

PROMPT_GUARDRAIL_ENABLED = settings.prompt_guardrail_enabled
PROMPT_GUARDRAIL_MODEL = settings.prompt_guardrail_model
PROMPT_GUARDRAIL_FAIL_OPEN = settings.prompt_guardrail_fail_open
