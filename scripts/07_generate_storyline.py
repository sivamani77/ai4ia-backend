import argparse
import json
import os
import re
import time
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Allow the stage runner to override the embeddings directory via env var.
# In S3 deployment mode the stage runner downloads the embeddings from S3
# to a /tmp path and passes AI4IA_EMBEDDINGS_DIR so we read the correctly
# classified data instead of whatever is on the worker VM's local disk.
_env_embeddings_dir = os.environ.get("AI4IA_EMBEDDINGS_DIR")
EMBEDDINGS_DIR = Path(_env_embeddings_dir) if _env_embeddings_dir else PROJECT_ROOT / "data" / "embeddings"
STORYLINES_DIR = PROJECT_ROOT / "data" / "storylines"

MATRIX_PATH = EMBEDDINGS_DIR / "fragment_embedding_matrix.npy"
JSONL_PATH = EMBEDDINGS_DIR / "classified_fragments_embeddings.jsonl"

STORYLINES_DIR.mkdir(parents=True, exist_ok=True)

# Runtime-configurable corpus state (set once in main() before any retrieval)
_ACTIVE_EMBEDDINGS_DIR: Path = EMBEDDINGS_DIR
_ACTIVE_MATRIX_PATH: Path = MATRIX_PATH
_ACTIVE_JSONL_PATH: Path = JSONL_PATH
_ACTIVE_CORPUS_VERSION: str = "original_v1"
_FILTERED_RECORDS: list | None = None
_FILTERED_MATRIX: Any = None
_CORPUS_DIAGNOSTICS: dict[str, Any] = {}

# ChromaDB-loaded corpus (populated by _load_corpus_from_chroma when CHROMA_MODE is set)
_CHROMA_MATRIX: Any = None
_CHROMA_RECORDS: list | None = None


def _chroma_mode() -> bool:
    return os.environ.get("CHROMA_MODE", "").strip().lower() == "http"


def _load_corpus_from_chroma() -> tuple[Any, list[dict[str, Any]]]:
    """Load embedding matrix and metadata records directly from ChromaDB."""
    global _CHROMA_MATRIX, _CHROMA_RECORDS
    if _CHROMA_MATRIX is not None and _CHROMA_RECORDS is not None:
        return _CHROMA_MATRIX, _CHROMA_RECORDS

    import chromadb

    host = os.environ.get("CHROMA_HOST", "chromadb")
    port = int(os.environ.get("CHROMA_PORT", "8000"))
    collection_name = os.environ.get("CHROMA_COLLECTION", "fragments")

    client = chromadb.HttpClient(host=host, port=port)
    collection = client.get_collection(collection_name)
    total = collection.count()
    if total == 0:
        raise RuntimeError("ChromaDB collection is empty — run ingestion first")

    log(f"Loading {total} fragments from ChromaDB {host}:{port} collection={collection_name}")

    batch_size = 500
    all_embeddings = []
    all_records = []
    offset = 0
    while offset < total:
        result = collection.get(
            limit=batch_size,
            offset=offset,
            include=["embeddings", "metadatas"],
        )
        for fid, emb, meta in zip(result["ids"], result["embeddings"], result["metadatas"]):
            all_embeddings.append(emb)
            row = dict(meta or {})
            row["fragment_id"] = fid
            all_records.append(row)
        offset += batch_size

    _CHROMA_MATRIX = np.array(all_embeddings, dtype="float32")
    _CHROMA_RECORDS = all_records
    log(f"ChromaDB corpus loaded: {len(all_records)} records, matrix shape={_CHROMA_MATRIX.shape}")
    return _CHROMA_MATRIX, _CHROMA_RECORDS


STRICT_NO_GENERIC_FALLBACK = True

OLLAMA_BASE_URL = "http://ollama-mobius-sales.mobiusdtaas.ai"

EMBED_MODEL = "nomic-embed-text:v1.5"
FAST_LLM_MODEL = "llama3.1:8b"
STRONG_LLM_MODEL = "llama3.1:70b"

EMBED_ENDPOINT = "/api/embed"
GENERATE_ENDPOINT = "/api/generate"

REQUEST_TIMEOUT_SECONDS = 300
EMBED_TIMEOUT_SECONDS = 300       # nomic can be slow to cold-load on a busy shared server
MAX_RETRIES = 4
RETRY_SLEEP_SECONDS = 5           # base for exponential backoff (5, 10, 20, ...)


DEFAULT_TOP_K = 90
DEFAULT_FINAL_CANDIDATES = 35
MAX_SOURCE_VIDEO_OCCURRENCES = 2
MAX_SPEAKER_OCCURRENCES = 2


def log(message: str) -> None:
    print(message, flush=True)


def post_with_retries(url: str, payload: dict[str, Any], timeout: int = REQUEST_TIMEOUT_SECONDS) -> dict[str, Any]:
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            log(f"Request failed attempt {attempt}/{MAX_RETRIES}: {exc}")
            if attempt < MAX_RETRIES:
                # Exponential backoff: gives a cold/overloaded Ollama time to recover.
                time.sleep(RETRY_SLEEP_SECONDS * (2 ** (attempt - 1)))

    raise RuntimeError(f"Request failed after retries: {last_error}")


def warm_up_ollama() -> None:
    """Ping Ollama and pre-load the embedding model so the first real call
    doesn't hit a cold-load timeout. Best-effort: failures here are non-fatal."""
    try:
        requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=15)
    except Exception as exc:  # noqa: BLE001
        log(f"Warm-up ping failed (continuing): {exc}")
        return
    try:
        log("Warming up embedding model ...")
        requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": "warm up"},
            timeout=EMBED_TIMEOUT_SECONDS,
        )
        log("Warm-up complete.")
    except Exception as exc:  # noqa: BLE001
        log(f"Warm-up embed failed (continuing, will retry on real call): {exc}")


def clean_text(text: str) -> str:
    return " ".join((text or "").split()).strip()


def truncate_text(text: str, max_chars: int = 650) -> str:
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def tokenize_prompt_keywords(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]{3,}", (text or "").lower())
    stopwords = {
        "about", "into", "with", "from", "that", "this", "there", "their", "would",
        "could", "should", "have", "what", "when", "where", "while", "video", "make",
        "need", "using", "used", "than", "then", "also", "your", "more", "less",
        "over", "under", "across", "around", "prompt", "clips", "clip", "theme",
        "chapter", "related", "exact", "evidence", "explain", "explainer",
    }
    ordered = []
    seen = set()
    for token in tokens:
        if token in stopwords or token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


STOPWORDS = {
    "create", "make", "generate", "video", "documentary", "minute", "minutes",
    "one", "two", "three", "on", "about", "what", "how", "why", "who", "all",
    "the", "a", "an", "and", "or", "of", "for", "to", "in", "with", "from",
    "by", "as", "is", "are", "was", "were", "be", "been", "being", "ai",
    "artificial", "intelligence", "ai4ia", "conference", "theme", "themes",
    "speaker", "spoke", "that", "this", "there", "their", "would", "could",
    "should", "have", "when", "where", "while", "need", "using", "used",
    "than", "then", "also", "your", "more", "less", "over", "under", "across",
    "around", "prompt", "clips", "clip", "chapter", "related", "exact",
    "evidence", "explain", "explainer", "into", "show", "please", "can",
    "give", "want", "get", "its", "has", "had", "will", "just", "only",
    "but", "not", "so", "if", "do", "did", "does", "may", "might", "some",
    "any", "new", "use", "based", "first", "last", "high", "good", "best",
    "way", "see", "per", "each", "key", "top", "tell", "discuss", "talk",
    "said", "say", "says", "discussing", "talking", "covering", "covered",
    "content", "information", "data", "system", "systems", "technology",
    "technologies", "tool", "tools",
}

SPEAKER_PROMPT_STOPWORDS = {
    "create", "make", "generate", "video", "documentary", "minute", "minutes",
    "what", "who", "said", "spoke", "about", "all", "theme", "themes",
    "chapter", "chapters", "the", "a", "an", "and", "or", "of", "for", "to",
    "in", "on", "with", "from", "by", "ai", "artificial", "intelligence",
    "ai4ia", "conference",
}

SYNONYM_MAP = {
    "women": {"women", "woman", "female", "gender", "girls"},
    "woman": {"women", "woman", "female", "gender", "girls"},
    "empowerment": {"empowerment", "empower", "agency", "participation", "leadership"},
    "empower": {"empowerment", "empower", "agency", "participation", "leadership"},
    "healthcare": {"healthcare", "health", "medicine", "medical", "patient", "hospital", "doctor", "clinical", "public health"},
    "health": {"healthcare", "health", "medicine", "medical", "patient", "hospital", "doctor", "clinical", "public health"},
    "governance": {"governance", "policy", "regulation", "oversight", "accountability", "transparency", "responsible"},
    "ethics": {"ethics", "ethical", "fairness", "bias", "accountability", "transparency", "responsibility"},
    "ethical": {"ethics", "ethical", "fairness", "bias", "accountability", "transparency", "responsibility"},
    "youth": {"youth", "young", "students", "children", "generational"},
    "young": {"youth", "young", "students", "children", "generational"},
    "access": {"access", "accessibility"},
    "crypto": {"crypto", "cryptocurrency", "blockchain", "web3", "digital currency", "token", "defi", "crypto scam"},
    "cryptocurrency": {"crypto", "cryptocurrency", "blockchain", "web3", "digital currency", "token"},
    "blockchain": {"crypto", "cryptocurrency", "blockchain", "web3"},
    "deepfake": {"deepfake", "deepfakes", "deep fake", "synthetic media", "AI-generated media", "counterfeit", "disinformation", "misinformation", "information integrity", "media trust", "identity misuse", "malicious AI"},
    "fake": {"deepfake", "deepfakes", "synthetic media", "disinformation", "misinformation", "fabricated", "forged", "manipulated media"},
    "deep": {"deepfake", "deepfakes", "deep fake", "synthetic media"},
}

PROMPT_TOPIC_CONFIG: dict[str, dict[str, Any]] = {
    "crypto": {
        "expanded_terms": ["cryptocurrency", "blockchain", "web3", "digital currency", "crypto scam", "token"],
        "negative_terms": ["cryptography", "encryption", "quantum cryptography", "post-quantum", "quantum random number"],
        "strictness": "high",
    },
    "women": {
        "expanded_terms": ["woman", "female", "girls", "gender", "agency", "participation", "leadership"],
        "generic_terms_removed": ["inclusion", "equity", "access", "diversity"],
        "strictness": "high",
    },
    "empowerment": {
        "expanded_terms": ["woman", "female", "girls", "gender", "agency", "participation", "leadership"],
        "generic_terms_removed": ["inclusion", "equity", "access", "diversity"],
        "strictness": "high",
    },
    "healthcare": {
        "expanded_terms": ["health", "medicine", "medical", "patient", "doctor", "hospital", "clinical", "public health"],
        "theme_hint": "AI in Health, Science & Environment",
        "chapter_hint": "Ch 13: AI in Healthcare & Medicine",
        "strictness": "medium",
    },
    "health": {
        "expanded_terms": ["healthcare", "medicine", "medical", "patient", "doctor", "hospital", "clinical", "public health"],
        "theme_hint": "AI in Health, Science & Environment",
        "chapter_hint": "Ch 13: AI in Healthcare & Medicine",
        "strictness": "medium",
    },
    "deepfake": {
        "expanded_terms": ["deepfakes", "synthetic media", "AI-generated media", "disinformation", "misinformation", "information integrity", "media trust", "identity misuse", "malicious AI", "fabricated video", "manipulated media"],
        "strictness": "medium",
    },
    "fake": {
        "expanded_terms": ["deepfake", "deepfakes", "synthetic media", "disinformation", "misinformation", "fabricated", "manipulated media"],
        "strictness": "medium",
    },
    "deep": {
        "expanded_terms": ["deepfake", "deepfakes", "deep fake", "synthetic media"],
        "strictness": "medium",
    },
}

GENERIC_ONLY_TERMS = {"inclusion", "equity", "access", "diversity", "accessibility"}

GENERIC_FALLBACK_PHRASES = [
    "artificial intelligence is no longer a distant idea",
    "shaping how people learn, communicate, govern and access knowledge",
    "across the ai4ia archive",
    "across the ai for ia archive",
    "the promise of ai is powerful",
    "the risks are equally real",
    "inclusive design, local data, ethical governance",
    "technology around real needs",
    "ai can become a bridge to knowledge",
    "how ai can expand access to information without deepening existing divides",
    "exclusion, weak infrastructure, language gaps, bias and loss of trust",
    "governance, rights, media literacy, accessibility, local knowledge",
    "this video explores",
    "through evidence from the ai4ia conference archive",
    "the selected clips address",
    "these archive fragments provide evidence-based context",
    "this documentary draws exclusively on confirmed archive clips",
    "the archive evidence shows:",
    "archive fragments provide",
    "draws exclusively on confirmed archive",
]

DISALLOWED_FALLBACK_TYPES = {
    "generic_ai4ia_documentary",
    "generic_ai_transformation_intro",
    "generic_governance_rights_accessibility_outro",
    "generic_archive_summary",
}

PROMPT_TERM_REQUIRED_WORDS: dict[str, set[str]] = {
    "healthcare": {"health", "healthcare", "medicine", "medical", "patient", "doctor", "clinical", "hospital"},
    "health": {"health", "healthcare", "medicine", "medical", "patient", "doctor", "clinical"},
    "women": {"women", "woman", "female", "gender", "girls", "empowerment", "agency", "representation"},
    "empowerment": {"women", "woman", "female", "gender", "empowerment", "agency", "leadership"},
    "crypto": {"crypto", "cryptocurrency", "blockchain", "web3", "digital currency"},
    "cryptocurrency": {"crypto", "cryptocurrency", "blockchain", "web3"},
}

INTERNAL_PIPELINE_TERMS = [
    "archive fragment",
    "archive fragments",
    "selected clips",
    "selected evidence",
    "confirmed archive clips",
    "evidence package",
    "evidence packages",
    "retrieval",
    "fallback",
    "llm",
    "json",
    "manifest",
    "qa report",
    "pipeline",
    "background task",
    "candidate bucket",
    "validation issue",
]

SIDE_TOPIC_TERMS = [
    "banking",
    "financial services",
    "customer service",
    "education and carbon emission",
    "finance",
    "marketing",
    "gaming",
    "musical art",
    "restroom",
    "housekeeping",
    "welcome remarks",
    "background task",
    "chip",
    "let me introduce",
    "thank you for inviting me",
    "next slide",
]

NATIONAL_GEOGRAPHIC_STYLE_GUIDE = {
    "voice": "observational, human, cinematic, clear",
    "sentence_length": "medium",
    "avoid": [
        "technical backend terms",
        "generic AI hype",
        "claims not supported by clips",
        "bullet-like narration",
        "conference admin language",
    ],
    "prefer": [
        "human stakes",
        "place/context",
        "tension",
        "careful synthesis",
        "specific evidence from speakers",
        "clear closing insight",
    ],
}

CINEMATIC_HOOKS: dict[str, str] = {
    "healthcare": "In hospitals, clinics, and homes, artificial intelligence is beginning to reshape the way care is delivered.",
    "health": "In hospitals, clinics, and homes, artificial intelligence is beginning to reshape the way care is delivered.",
    "women": "Across the world, millions of women are navigating systems that were not designed with them in mind.",
    "empowerment": "Across the world, millions of women are navigating systems that were not designed with them in mind.",
    "education": "In classrooms, libraries, and remote communities, the question is not whether AI will change learning, but whether it will do so fairly.",
    "governance": "As artificial intelligence moves into public systems, the question of who controls it — and who answers for it — grows more urgent.",
    "youth": "A generation that has grown up online is now asking a different question: not what technology can do, but who it serves.",
    "crypto": "Digital currencies and blockchain technology have promised a new financial order. But for many, the risks have arrived before the benefits.",
    "accessibility": "For millions of people, access to information is not a given. It is a daily negotiation with systems that were built without them in mind.",
}

CINEMATIC_TENSIONS: dict[str, str] = {
    "healthcare": "But the promise of AI in healthcare depends on more than speed. Speakers warn that trust, patient safety, ethical use, and regulation must shape how these systems enter clinical life.",
    "health": "But the promise of AI in healthcare depends on more than speed. Speakers warn that trust, patient safety, ethical use, and regulation must shape how these systems enter clinical life.",
    "women": "But access to tools is not the same as access to power. For many women, technology arrives within systems that still constrain their choices and reduce their voices.",
    "empowerment": "But access to tools is not the same as access to power. For many women, technology arrives within systems that still constrain their choices and reduce their voices.",
    "governance": "The question is not only what AI can do. It is who writes the rules, who bears the risk, and who can challenge a decision made by a machine.",
    "youth": "But younger generations do not simply inherit the future. They are being shaped by algorithmic systems whose decisions they cannot see and rarely control.",
    "education": "But digital tools also replicate the inequalities they were meant to solve, reinforcing divides of language, access, and the quality of what people find.",
    "crypto": "The risks have been real. Volatility, fraud, and the absence of consumer protection have left many people worse off than before.",
    "accessibility": "Yet the technology that could expand access is often designed for those who need it least, leaving the most vulnerable populations without tools that match their realities.",
}

CINEMATIC_CLOSINGS: dict[str, str] = {
    "healthcare": "The future of AI in healthcare will not be measured only by what machines can predict, but by whether they help people receive safer, fairer, and more human care.",
    "health": "The future of AI in healthcare will not be measured only by what machines can predict, but by whether they help people receive safer, fairer, and more human care.",
    "women": "The measure of progress will not be the number of tools deployed, but whether women gain real agency, real voice, and real power in the systems that shape their lives.",
    "empowerment": "The measure of progress will not be the number of tools deployed, but whether women gain real agency, real voice, and real power in the systems that shape their lives.",
    "governance": "The future of AI governance will not be written by technology alone. It will be shaped by institutions, civil society, and people who insist that accountability must be built in from the start.",
    "youth": "The generation growing up with AI will not simply use the systems they inherit. They will decide whether to change them.",
    "education": "The future of learning is not defined by the speed of information, but by whether every learner — regardless of where they were born — can access knowledge that is true, useful, and theirs.",
    "crypto": "The promise of a more open financial system remains. But it will only arrive through transparency, safety, and rules that protect the people it was meant to serve.",
    "accessibility": "The future of information access will not be determined by what technology can do, but by whether it is built to serve the people who have been left out the longest.",
}

SPEAKER_ALIAS_MAP: dict[str, str] = {
    "codrel green": "Cordel Green",
    "cordel green": "Cordel Green",
    "cordell green": "Cordel Green",
    "codel green": "Cordel Green",
    "kodel green": "Cordel Green",
    "kodel codel green": "Cordel Green",
    "cordell kodel green": "Cordel Green",
    "cordell greene": "Cordel Green",
}


def normalize_plural(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def normalize_name(value: str) -> str:
    lowered = (value or "").lower().replace("_", " ").replace("-", " ")
    lowered = re.sub(r"[^a-z0-9\s]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def extract_meaningful_prompt_terms(prompt: str) -> set[str]:
    raw_tokens = re.findall(r"[a-z0-9]+", (prompt or "").lower())
    terms = set()
    for token in raw_tokens:
        if len(token) < 3:
            continue
        if token in STOPWORDS:
            continue
        normalized = normalize_plural(token)
        if normalized in STOPWORDS:
            continue
        terms.add(normalized)
    return terms


def expand_prompt_terms(terms: set[str]) -> set[str]:
    expanded = set(terms)
    for term in terms:
        if term in SYNONYM_MAP:
            expanded.update(SYNONYM_MAP[term])
    return expanded


def load_known_speakers(metadata_dir: Path | None = None) -> list[str]:
    speakers_dir = metadata_dir or (PROJECT_ROOT / "data" / "metadata" / "speakers")
    known: list[str] = []
    seen: set[str] = set()
    if not speakers_dir.exists():
        return known
    for fpath in speakers_dir.glob("*.json"):
        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
            values: list[str] = []
            primary = data.get("primary_speaker") or data.get("speaker")
            if isinstance(primary, str) and primary.strip():
                values.append(primary)
            for sp in data.get("speakers", []):
                if isinstance(sp, dict):
                    name = sp.get("name")
                    if isinstance(name, str) and name.strip():
                        values.append(name)
                elif isinstance(sp, str) and sp.strip():
                    values.append(sp)
            stem_guess = fpath.stem.split("_", 1)[-1].replace("_", " ").strip()
            if stem_guess:
                values.append(stem_guess)
            for name in values:
                norm = normalize_name(name)
                if norm and norm not in seen:
                    seen.add(norm)
                    known.append(clean_text(str(name)))
        except Exception:
            pass
    return known


def detect_requested_speaker(prompt: str, known_speakers: list[str]) -> str | None:
    normalized_prompt = normalize_name(prompt)

    # Check alias map first (handles typos like "Codrel Green")
    for alias, canonical in SPEAKER_ALIAS_MAP.items():
        alias_norm = normalize_name(alias)
        if re.search(rf"\b{re.escape(alias_norm)}\b", normalized_prompt):
            # Verify canonical exists in known_speakers or close match
            for known in known_speakers:
                if normalize_name(canonical) in normalize_name(known) or normalize_name(known) in normalize_name(canonical):
                    return known
            # Return canonical even if not in known_speakers (low confidence)
            return canonical

    best_match = None
    best_len = 0
    for speaker in known_speakers:
        normalized_speaker = normalize_name(speaker)
        # Require at least 5 chars and a space (at least first + last name) to avoid false positives
        if len(normalized_speaker) < 5 or " " not in normalized_speaker:
            continue
        if re.search(rf"\b{re.escape(normalized_speaker)}\b", normalized_prompt) and len(normalized_speaker) > best_len:
            best_match = speaker
            best_len = len(normalized_speaker)
    return best_match


def extract_topic_terms(prompt: str, requested_speaker: str | None = None) -> set[str]:
    text = normalize_name(prompt)
    if requested_speaker:
        text = re.sub(rf"\b{re.escape(normalize_name(requested_speaker))}\b", " ", text)
    raw_tokens = re.findall(r"[a-z0-9]+", text)
    terms = set()
    for token in raw_tokens:
        if len(token) < 3:
            continue
        normalized = normalize_plural(token)
        if normalized in STOPWORDS or normalized in SPEAKER_PROMPT_STOPWORDS:
            continue
        terms.add(normalized)
    return terms


def infer_theme_and_chapter_hints(topic_terms: list[str]) -> tuple[str | None, str | None]:
    theme_hint = None
    chapter_hint = None
    for term in topic_terms:
        config = PROMPT_TOPIC_CONFIG.get(term) or {}
        theme_hint = theme_hint or config.get("theme_hint")
        chapter_hint = chapter_hint or config.get("chapter_hint")
    return theme_hint, chapter_hint


def build_prompt_profile(
    prompt: str,
    known_speakers: list[str],
    target_duration_seconds: int | None = None,
) -> dict[str, Any]:
    requested_speaker = detect_requested_speaker(prompt, known_speakers)
    normalized_prompt = normalize_name(prompt)
    all_themes_requested = bool(re.search(r"\ball(?:\s+the)?\s+themes?\b", normalized_prompt))
    topic_terms = sorted(extract_topic_terms(prompt, requested_speaker=requested_speaker))
    if requested_speaker:
        speaker_name_tokens = {normalize_plural(tok) for tok in re.findall(r"[a-z0-9]+", normalize_name(requested_speaker))}
        topic_terms = [term for term in topic_terms if term not in speaker_name_tokens]
    if requested_speaker and all_themes_requested:
        topic_terms = []

    expanded_terms: list[str] = []
    negative_terms: list[str] = []
    generic_terms_removed: list[str] = []
    theme_hint, chapter_hint = infer_theme_and_chapter_hints(topic_terms)
    strictness = "medium"

    for term in topic_terms:
        config = PROMPT_TOPIC_CONFIG.get(term) or {}
        for value in config.get("expanded_terms", []):
            if value not in expanded_terms:
                expanded_terms.append(value)
        for value in config.get("negative_terms", []):
            if value not in negative_terms:
                negative_terms.append(value)
        for value in config.get("generic_terms_removed", []):
            if value not in generic_terms_removed:
                generic_terms_removed.append(value)
        config_strictness = config.get("strictness")
        if config_strictness == "high":
            strictness = "high"
        elif config_strictness == "low" and strictness != "high":
            strictness = "low"
        theme_hint = theme_hint or config.get("theme_hint")
        chapter_hint = chapter_hint or config.get("chapter_hint")

    scope = "general"
    if requested_speaker and all_themes_requested:
        scope = "all_themes"
    elif topic_terms or theme_hint or chapter_hint:
        scope = "specific_topic"

    speaker_required = requested_speaker is not None
    if requested_speaker and scope == "all_themes":
        intent_type = "speaker_summary"
        evidence_requirement = "speaker_source_evidence"
        strictness = "high"
    elif requested_speaker:
        intent_type = "speaker_topic_summary"
        evidence_requirement = "speaker_source_evidence"
        strictness = "high"
    elif chapter_hint:
        intent_type = "chapter_video"
        evidence_requirement = "direct_topic_evidence"
    elif theme_hint and not topic_terms:
        intent_type = "theme_video"
        evidence_requirement = "broad_theme_evidence"
    else:
        intent_type = "topic_video"
        evidence_requirement = "direct_topic_evidence"

    return {
        "raw_prompt": prompt,
        "intent_type": intent_type,
        "requested_speaker": requested_speaker,
        "speaker_required": speaker_required,
        "topic_terms": topic_terms,
        "expanded_terms": expanded_terms,
        "negative_terms": negative_terms,
        "generic_terms_removed": generic_terms_removed,
        "theme_hint": theme_hint,
        "chapter_hint": chapter_hint,
        "scope": scope,
        "evidence_requirement": evidence_requirement,
        "strictness": strictness,
        "target_duration_seconds": target_duration_seconds,
    }


def lookup_requested_chapter(prompt: str, records: list[dict[str, Any]]) -> str | None:
    normalized_prompt = normalize_name(prompt)
    chapter_candidates = sorted(
        {
            str(record.get("primary_chapter")).strip()
            for record in records
            if record.get("primary_chapter")
        },
        key=len,
        reverse=True,
    )
    chapter_number_match = re.search(r"\bch(?:apter)?\s*(\d+)\b", normalized_prompt)
    chapter_number = chapter_number_match.group(1) if chapter_number_match else None

    for chapter in chapter_candidates:
        normalized_chapter = normalize_name(chapter)
        if normalized_chapter and normalized_chapter in normalized_prompt:
            return chapter
        if chapter_number and re.search(rf"\bch(?:apter)?\s*0*{re.escape(chapter_number)}\b", normalized_chapter):
            return chapter
    return None


def lookup_requested_theme(prompt_terms: set[str], records: list[dict[str, Any]]) -> str | None:
    theme_candidates = {
        str(record.get("primary_theme")).strip()
        for record in records
        if record.get("primary_theme")
    }
    best_theme = None
    best_score = 0
    for theme in theme_candidates:
        theme_tokens = extract_topic_terms(theme)
        score = len(prompt_terms & theme_tokens)
        if score > best_score:
            best_theme = theme
            best_score = score
    return best_theme if best_score > 0 else None


def detect_prompt_intent(
    prompt: str,
    known_speakers: list[str],
    records: list[dict[str, Any]],
    target_duration_seconds: int | None = None,
) -> dict[str, Any]:
    profile = build_prompt_profile(prompt, known_speakers, target_duration_seconds=target_duration_seconds)
    topic_terms = set(profile.get("topic_terms") or [])
    chapter = lookup_requested_chapter(prompt, records)
    theme = lookup_requested_theme(topic_terms, records)
    if chapter:
        profile["chapter_hint"] = chapter
        if profile["intent_type"] == "topic_video":
            profile["intent_type"] = "chapter_video"
    if theme:
        profile["theme_hint"] = theme
        if profile["intent_type"] == "topic_video" and not topic_terms:
            profile["intent_type"] = "theme_video"
    return profile


def candidate_speaker_matches(record: dict[str, Any], requested_speaker: str) -> tuple[bool, str]:
    requested = normalize_name(requested_speaker)
    fields = [
        ("speaker", record.get("speaker")),
        ("primary_speaker", record.get("primary_speaker")),
        ("source_video", record.get("source_video")),
        ("video_title", record.get("video_title")),
    ]
    for key, value in fields:
        normalized_value = normalize_name(str(value or ""))
        if normalized_value and requested in normalized_value:
            return True, f"{key} contains {requested_speaker}"

    speaker_candidates = record.get("speaker_candidates") or []
    for item in speaker_candidates:
        candidate_name = item.get("name") if isinstance(item, dict) else item
        normalized_value = normalize_name(str(candidate_name or ""))
        if normalized_value and requested in normalized_value:
            return True, f"speaker_candidates contains {requested_speaker}"

    return False, "speaker mismatch"


CRYPTO_TERMS = {"crypto", "cryptocurrency", "blockchain", "web3", "digital currency", "token", "defi", "nft", "exchange", "wallet", "satoshi", "bitcoin", "ethereum"}
CRYPTOGRAPHY_TERMS = {"cryptography", "encryption", "decryption", "cipher", "quantum cryptography", "post quantum", "homomorphic", "zero knowledge", "cryptographic", "hashing", "hash function", "public key", "private key"}

GENERIC_EXPANSION_TERMS = {"inclusion", "equity", "access", "diversity", "transparency", "accountability", "responsibility", "knowledge", "information", "accessibility"}


def classify_candidate_relevance(
    candidate: dict[str, Any],
    prompt_terms: set[str],
    expanded_terms: set[str],
    intent: dict[str, Any],
) -> dict[str, Any]:
    enriched = dict(candidate)
    blob = candidate_search_blob(enriched)
    blob_tokens = set(re.findall(r"[a-z0-9]+", blob))
    requested_speaker = intent.get("requested_speaker")
    requested_theme = intent.get("theme")
    requested_chapter = intent.get("chapter")
    scope = intent.get("scope")

    # Detect if this is a crypto-specific prompt (not cryptography)
    is_crypto_prompt = "crypto" in prompt_terms and "cryptography" not in prompt_terms and "encryption" not in prompt_terms
    is_cryptography_prompt = any(t in prompt_terms for t in {"cryptography", "encryption", "quantum"})

    # Compute meaningful overlap: only prompt_terms (not stopwords), not expanded synonyms
    meaningful_overlap = [t for t in prompt_terms if t in blob_tokens]

    # Expanded overlap: only non-generic expansion terms count toward exactness
    expanded_overlap_raw = [t for t in expanded_terms if t in blob_tokens and t not in prompt_terms]
    expanded_overlap = [t for t in expanded_overlap_raw if t not in GENERIC_EXPANSION_TERMS]
    generic_expanded_overlap = [t for t in expanded_overlap_raw if t in GENERIC_EXPANSION_TERMS]

    semantic_score = float(enriched.get("semantic_score") or 0.0)
    quote_strength = float(enriched.get("quote_strength") or 0.0)
    clarity_score = float(enriched.get("clarity_score") or 0.0)
    _raw_usage = enriched.get("clip_usage_candidates") or []
    if isinstance(_raw_usage, str):
        usage_candidates = set(c.strip() for c in _raw_usage.split(",") if c.strip())
    else:
        usage_candidates = set(_raw_usage)

    # Crypto/cryptography disambiguation
    if is_crypto_prompt:
        candidate_blob_has_crypto = any(t in blob_tokens for t in CRYPTO_TERMS)
        candidate_blob_has_cryptography = any(t in blob_tokens for t in CRYPTOGRAPHY_TERMS)
        if candidate_blob_has_cryptography and not candidate_blob_has_crypto:
            # This is a cryptography clip for a crypto (cryptocurrency) prompt — demote
            expanded_overlap = []
            meaningful_overlap = []
            semantic_score = min(semantic_score, 0.58)  # cap it so it can't reach exact

    # Theme/chapter match
    theme = (enriched.get("primary_theme") or "").lower()
    chapter = (enriched.get("primary_chapter") or "").lower()
    theme_chapter_text = theme + " " + chapter
    theme_tokens = set(re.findall(r"[a-z0-9]+", theme_chapter_text))
    theme_overlap = [t for t in prompt_terms if t in theme_tokens]
    expanded_theme_overlap = [t for t in expanded_terms if t in theme_tokens and t not in GENERIC_EXPANSION_TERMS]

    # Speaker constraint check
    speaker_clip_allowed = False
    broll_allowed = True
    speaker_match = requested_speaker is None
    speaker_match_reason = "no speaker requested"

    if requested_speaker is not None:
        speaker_match, speaker_match_reason = candidate_speaker_matches(enriched, requested_speaker)
        if not speaker_match:
            enriched["meaningful_prompt_overlap"] = meaningful_overlap
            enriched["requested_speaker"] = requested_speaker
            enriched["speaker_match"] = False
            enriched["speaker_match_reason"] = speaker_match_reason
            enriched["prompt_relevance_score"] = round(float(enriched.get("semantic_score") or 0.0), 6)
            enriched["relevance_role"] = "reject"
            enriched["speaker_soundbite_allowed"] = False
            enriched["speaker_visual_allowed"] = False
            enriched["speaker_clip_allowed"] = False
            enriched["broll_allowed"] = False
            enriched["final_video_allowed"] = False
            enriched["use_as"] = "reject"
            enriched["relevance_reason"] = f"Speaker constraint: {speaker_match_reason}"
            return enriched

    # Compute total evidence signals (non-generic only)
    total_meaningful = len(meaningful_overlap) + len(expanded_overlap)
    total_theme = len(theme_overlap) + len(expanded_theme_overlap)
    chapter_match = bool(requested_chapter and normalize_name(requested_chapter) == normalize_name(str(enriched.get("primary_chapter") or "")))
    theme_match = bool(requested_theme and normalize_name(requested_theme) == normalize_name(str(enriched.get("primary_theme") or "")))
    prompt_relevance_score = float(semantic_score)
    if chapter_match:
        prompt_relevance_score += 0.14
    if theme_match:
        prompt_relevance_score += 0.1
    prompt_relevance_score += min(0.16, len(meaningful_overlap) * 0.06)
    prompt_relevance_score += min(0.06, len(expanded_overlap) * 0.03)
    prompt_relevance_score += min(0.06, quote_strength * 0.06)

    # Side-topic pollution penalty
    candidate_blob_lower = blob.lower()
    side_topic_count = sum(1 for term in SIDE_TOPIC_TERMS if term.lower() in candidate_blob_lower)
    if side_topic_count >= 2:
        prompt_relevance_score -= 0.25
    elif side_topic_count == 1 and len(meaningful_overlap) == 0:
        prompt_relevance_score -= 0.15

    # Classification logic
    if requested_speaker and scope == "all_themes":
        if chapter_match or theme_match or len(meaningful_overlap) >= 1 or semantic_score >= 0.58:
            role = "exact"
            reason = "Speaker matched; selected as one of the strongest clips across available themes."
            speaker_clip_allowed = (
                "speaker_soundbite" in usage_candidates
                and quote_strength >= 0.50
                and clarity_score >= 0.75
            )
        elif semantic_score >= 0.52:
            role = "strong_related"
            reason = f"Speaker matched with limited topic evidence; semantic {semantic_score:.3f}"
            speaker_clip_allowed = (
                "speaker_soundbite" in usage_candidates
                and quote_strength >= 0.55
                and clarity_score >= 0.75
            )
        else:
            role = "weak_related"
            reason = f"Speaker matched but topic evidence is weak; semantic {semantic_score:.3f}"
            speaker_clip_allowed = False
    elif chapter_match and (theme_match or len(meaningful_overlap) >= 1 or semantic_score >= 0.56):
        role = "exact"
        reason = "Speaker and requested chapter matched."
        speaker_clip_allowed = (
            "speaker_soundbite" in usage_candidates
            and quote_strength >= 0.50
            and clarity_score >= 0.75
        )
    elif theme_match and (len(meaningful_overlap) >= 1 or semantic_score >= 0.58):
        role = "exact"
        reason = "Speaker and requested theme matched."
        speaker_clip_allowed = (
            "speaker_soundbite" in usage_candidates
            and quote_strength >= 0.50
            and clarity_score >= 0.75
        )
    elif len(meaningful_overlap) >= 2 and semantic_score >= 0.62:
        role = "exact"
        reason = f"Strong meaningful overlap ({meaningful_overlap}) + semantic {semantic_score:.3f}"
        speaker_clip_allowed = (
            "speaker_soundbite" in usage_candidates
            and quote_strength >= 0.50
            and clarity_score >= 0.75
        )
    elif len(meaningful_overlap) >= 1 and semantic_score >= 0.66:
        role = "exact"
        reason = f"Meaningful overlap ({meaningful_overlap}) + high semantic {semantic_score:.3f}"
        speaker_clip_allowed = (
            "speaker_soundbite" in usage_candidates
            and quote_strength >= 0.50
            and clarity_score >= 0.75
        )
    elif len(expanded_overlap) >= 2 and semantic_score >= 0.65:
        role = "exact"
        reason = f"Non-generic expanded overlap ({expanded_overlap}) + semantic {semantic_score:.3f}"
        speaker_clip_allowed = (
            "speaker_soundbite" in usage_candidates
            and quote_strength >= 0.50
            and clarity_score >= 0.75
        )
    elif len(generic_expanded_overlap) >= 2 and semantic_score >= 0.67:
        # Generic terms (inclusion/equity/diversity) only → strong_related at best
        role = "strong_related"
        reason = f"Generic expanded terms only ({generic_expanded_overlap}) + semantic {semantic_score:.3f}; not exact"
        speaker_clip_allowed = False
    elif semantic_score >= 0.68 and (total_theme >= 1 or total_meaningful >= 1):
        role = "strong_related"
        reason = f"High semantic {semantic_score:.3f} + theme/expanded match"
        speaker_clip_allowed = (
            "speaker_soundbite" in usage_candidates
            and quote_strength >= 0.55
            and clarity_score >= 0.75
        )
    elif (total_theme >= 1 or total_meaningful >= 1 or len(expanded_overlap) >= 1) and semantic_score >= 0.62:
        role = "strong_related"
        reason = f"Topic/theme overlap at moderate semantic {semantic_score:.3f}: meaningful={meaningful_overlap}, theme={theme_overlap}, expanded={expanded_overlap[:3]}"
        speaker_clip_allowed = (
            "speaker_soundbite" in usage_candidates
            and quote_strength >= 0.55
            and clarity_score >= 0.75
        )
    elif total_meaningful >= 1 or total_theme >= 1 or len(expanded_overlap) >= 1:
        role = "weak_related"
        reason = f"Partial overlap below semantic threshold: meaningful={meaningful_overlap}, theme={theme_overlap}, expanded={expanded_overlap[:3]}"
        speaker_clip_allowed = False
    elif len(generic_expanded_overlap) >= 1 and semantic_score >= 0.60:
        role = "contextual"
        reason = f"Only generic synonym match ({generic_expanded_overlap}) + semantic {semantic_score:.3f}"
        speaker_clip_allowed = False
    elif semantic_score >= 0.60:
        role = "contextual"
        reason = f"Broad semantic similarity {semantic_score:.3f} only, no topic overlap"
        speaker_clip_allowed = False
    else:
        role = "reject"
        reason = f"Low semantic {semantic_score:.3f}, no topic overlap"
        speaker_clip_allowed = False
        broll_allowed = False

    # Hard rule: never allow speaker soundbite for weak/contextual/reject
    if role in ("weak_related", "contextual", "reject"):
        speaker_clip_allowed = False

    # speaker_soundbite_allowed: prompt-relative gate — role must be exact/strong_related
    # so the same clip can be a soundbite for one prompt and b-roll for another.
    # Thresholds lowered from 0.65/0.85 since quote_strength is now a better signal.
    if role in ("exact", "strong_related"):
        speaker_soundbite_allowed = (
            quote_strength >= 0.50
            and clarity_score >= 0.80
        )
    else:
        speaker_soundbite_allowed = False

    # speaker_visual_allowed: true for speaker-prompt clips that belong to the requested speaker,
    # even if not good enough for a soundbite. The clip can run muted under narration.
    speaker_visual_allowed = False
    if requested_speaker and speaker_match and role not in ("reject",):
        duration = float(enriched.get("duration") or 0.0)
        start = enriched.get("start")
        end = enriched.get("end")
        if duration > 8 and start is not None and end is not None:
            speaker_visual_allowed = True

    # Determine use_as and final_video_allowed
    if role == "reject":
        use_as = "reject"
        final_video_allowed = False
    elif role == "contextual":
        use_as = "context_only"
        final_video_allowed = False
    elif role == "weak_related":
        if speaker_visual_allowed:
            # Speaker-matched weak clip — usable as visual under narration
            use_as = "speaker_visual"
            final_video_allowed = True
        else:
            use_as = "narration_support"
            final_video_allowed = False  # will be promoted to True only for low-evidence fallback
    elif speaker_soundbite_allowed:
        use_as = "speaker_soundbite"
        final_video_allowed = True
    elif speaker_visual_allowed:
        use_as = "speaker_visual"
        final_video_allowed = True
    elif "archive_visual" in usage_candidates or "evidence_clip" in usage_candidates:
        use_as = "archive_visual"
        final_video_allowed = True
    else:
        use_as = "narration_support"
        final_video_allowed = True

    # archive_visual clips must never be speaker_soundbite_allowed
    if use_as == "archive_visual":
        speaker_soundbite_allowed = False
        speaker_clip_allowed = False

    enriched["meaningful_prompt_overlap"] = meaningful_overlap
    enriched["requested_speaker"] = requested_speaker
    enriched["speaker_match"] = speaker_match
    enriched["speaker_match_reason"] = speaker_match_reason
    enriched["prompt_relevance_score"] = round(prompt_relevance_score, 6)
    enriched["relevance_role"] = role
    enriched["speaker_soundbite_allowed"] = speaker_soundbite_allowed
    enriched["speaker_visual_allowed"] = speaker_visual_allowed
    enriched["speaker_clip_allowed"] = speaker_soundbite_allowed  # backward compat alias
    enriched["broll_allowed"] = broll_allowed
    enriched["final_video_allowed"] = final_video_allowed
    enriched["use_as"] = use_as
    enriched["relevance_reason"] = reason
    return enriched


def candidate_search_blob(record: dict[str, Any]) -> str:
    parts = [
        record.get("text", ""),
        record.get("summary", ""),
        " ".join(record.get("keywords", []) or []),
        record.get("source_video", ""),
        record.get("speaker", ""),
        record.get("primary_speaker", ""),
        record.get("primary_theme", ""),
        record.get("primary_chapter", ""),
    ]
    return clean_text(" ".join(str(part) for part in parts if part)).lower()


def match_terms_in_text(text: str, terms: list[str]) -> list[str]:
    normalized = normalize_name(text)
    matches: list[str] = []
    for term in terms:
        normalized_term = normalize_name(term)
        if normalized_term and re.search(rf"\b{re.escape(normalized_term)}\b", normalized):
            matches.append(term)
    return matches


def score_candidate_for_prompt(candidate: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(candidate)
    blob = candidate_search_blob(enriched)
    direct_terms = match_terms_in_text(blob, profile.get("topic_terms") or [])
    expanded_terms = [
        term for term in match_terms_in_text(blob, profile.get("expanded_terms") or [])
        if term not in direct_terms
    ]
    negative_terms = match_terms_in_text(blob, profile.get("negative_terms") or [])
    generic_removed = set(profile.get("generic_terms_removed") or [])
    generic_overlap = [term for term in expanded_terms if normalize_name(term) in generic_removed or normalize_name(term) in GENERIC_ONLY_TERMS]
    non_generic_expanded = [term for term in expanded_terms if term not in generic_overlap]

    semantic_score = float(enriched.get("semantic_score") or 0.0)
    quote_strength = float(enriched.get("quote_strength") or 0.0)
    clarity_score = float(enriched.get("clarity_score") or 0.0)
    theme_match = bool(
        profile.get("theme_hint")
        and normalize_name(str(enriched.get("primary_theme") or "")) == normalize_name(str(profile.get("theme_hint")))
    )
    chapter_match = bool(
        profile.get("chapter_hint")
        and normalize_name(str(enriched.get("primary_chapter") or "")) == normalize_name(str(profile.get("chapter_hint")))
    )
    requested_speaker = profile.get("requested_speaker")
    if requested_speaker:
        speaker_match, speaker_match_reason = candidate_speaker_matches(enriched, requested_speaker)
    else:
        speaker_match, speaker_match_reason = True, "no speaker requested"

    keyword_score = min(1.0, len(direct_terms) * 0.5)
    expanded_keyword_score = min(1.0, len(non_generic_expanded) * 0.35)
    theme_chapter_score = 1.0 if chapter_match else (0.8 if theme_match else 0.0)
    speaker_score = 1.0 if speaker_match and requested_speaker else 0.0
    quote_quality_score = min(1.0, (quote_strength * 0.6) + (clarity_score * 0.4))
    generic_penalty = min(0.25, len(generic_overlap) * 0.08)
    negative_term_penalty = min(0.45, len(negative_terms) * 0.18)
    final_prompt_score = (
        0.30 * semantic_score
        + 0.25 * keyword_score
        + 0.15 * expanded_keyword_score
        + 0.15 * theme_chapter_score
        + 0.10 * speaker_score
        + 0.05 * quote_quality_score
        - generic_penalty
        - negative_term_penalty
    )

    reasons = []
    if direct_terms:
        reasons.append(f"direct terms {direct_terms[:3]}")
    if non_generic_expanded:
        reasons.append(f"expanded terms {non_generic_expanded[:3]}")
    if chapter_match:
        reasons.append(f"chapter match {enriched.get('primary_chapter')}")
    elif theme_match:
        reasons.append(f"theme match {enriched.get('primary_theme')}")
    if requested_speaker and speaker_match:
        reasons.append(f"speaker match {requested_speaker}")
    if not reasons:
        reasons.append(f"semantic similarity {semantic_score:.3f}")
    if negative_terms:
        reasons.append(f"negative conflict {negative_terms[:3]}")

    enriched.update(
        {
            "keyword_score": round(keyword_score, 6),
            "expanded_keyword_score": round(expanded_keyword_score, 6),
            "theme_chapter_score": round(theme_chapter_score, 6),
            "speaker_score": round(speaker_score, 6),
            "quote_quality_score": round(quote_quality_score, 6),
            "generic_penalty": round(generic_penalty, 6),
            "negative_term_penalty": round(negative_term_penalty, 6),
            "final_prompt_score": round(final_prompt_score, 6),
            "direct_evidence_terms": direct_terms,
            "expanded_evidence_terms": non_generic_expanded,
            "negative_evidence_terms": negative_terms,
            "generic_evidence_terms": generic_overlap,
            "selection_reason": "Matched " + ", ".join(reasons) + ".",
            "requested_speaker": requested_speaker,
            "speaker_match": speaker_match,
            "speaker_match_reason": speaker_match_reason,
            "chapter_match": chapter_match,
            "theme_match": theme_match,
            "meaningful_prompt_overlap": direct_terms,
            "prompt_keyword_overlap": direct_terms,
            "prompt_keyword_overlap_count": len(direct_terms),
        }
    )
    return enriched


def apply_hard_rejection_rules(candidate: dict[str, Any], profile: dict[str, Any]) -> tuple[bool, str]:
    if profile.get("requested_speaker") and not candidate.get("speaker_match"):
        return True, "Rejected: speaker prompt requires requested speaker match"

    topic_terms = set(profile.get("topic_terms") or [])
    direct_terms = candidate.get("direct_evidence_terms") or []
    expanded_terms = candidate.get("expanded_evidence_terms") or []
    negative_terms = candidate.get("negative_evidence_terms") or []
    generic_terms = candidate.get("generic_evidence_terms") or []

    if "crypto" in topic_terms:
        crypto_vocab = {normalize_name(v) for v in CRYPTO_TERMS}
        crypto_positive = any(normalize_name(term) in crypto_vocab for term in direct_terms + expanded_terms)
        if negative_terms and not crypto_positive:
            return True, "Rejected: cryptography is not cryptocurrency evidence"

    if topic_terms & {"women", "woman", "empowerment", "empower"}:
        if not direct_terms and generic_terms and not expanded_terms:
            return True, "Rejected: generic inclusion/access terms are not direct women empowerment evidence"

    if not direct_terms and not expanded_terms and negative_terms and float(candidate.get("semantic_score") or 0.0) < 0.7:
        return True, "Rejected: negative-term conflict without direct evidence"

    return False, ""


def classify_candidate_relevance_v2(candidate: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    enriched = score_candidate_for_prompt(candidate, profile)
    reject, reject_reason = apply_hard_rejection_rules(enriched, profile)

    requested_speaker = profile.get("requested_speaker")
    scope = profile.get("scope")
    direct_terms = enriched.get("direct_evidence_terms") or []
    expanded_hits = enriched.get("expanded_evidence_terms") or []
    semantic_score = float(enriched.get("semantic_score") or 0.0)
    final_prompt_score = float(enriched.get("final_prompt_score") or 0.0)
    quote_strength = float(enriched.get("quote_strength") or 0.0)
    clarity_score = float(enriched.get("clarity_score") or 0.0)
    _raw_usage2 = enriched.get("clip_usage_candidates") or []
    if isinstance(_raw_usage2, str):
        usage_candidates = set(c.strip() for c in _raw_usage2.split(",") if c.strip())
    else:
        usage_candidates = set(_raw_usage2)
    chapter_match = bool(enriched.get("chapter_match"))
    theme_match = bool(enriched.get("theme_match"))
    generic_terms = enriched.get("generic_evidence_terms") or []

    if reject:
        role = "reject"
        reason = reject_reason
    elif requested_speaker and scope == "all_themes":
        if enriched.get("speaker_match") and (semantic_score >= 0.45 or quote_strength >= 0.55 or clarity_score >= 0.9):
            role = "exact"
            reason = f"Requested speaker match with strong archive evidence across themes. {enriched.get('selection_reason')}"
        elif enriched.get("speaker_match") and (semantic_score >= 0.30 or final_prompt_score >= 0.20):
            role = "strong_related"
            reason = f"Requested speaker match with limited evidence. {enriched.get('selection_reason')}"
        else:
            role = "weak_related"
            reason = f"Requested speaker match but weak evidence. {enriched.get('selection_reason')}"
    elif direct_terms and final_prompt_score >= 0.48:
        role = "exact"
        reason = enriched.get("selection_reason")
    elif (chapter_match or theme_match) and final_prompt_score >= 0.46 and (direct_terms or expanded_hits or requested_speaker):
        role = "exact"
        reason = enriched.get("selection_reason")
    elif (theme_match or chapter_match or expanded_hits) and final_prompt_score >= 0.34:
        role = "strong_related"
        reason = enriched.get("selection_reason")
    elif direct_terms or expanded_hits:
        role = "weak_related"
        reason = enriched.get("selection_reason")
    elif generic_terms and semantic_score >= 0.60:
        role = "contextual"
        reason = f"Only generic support terms matched. {enriched.get('selection_reason')}"
    elif semantic_score >= 0.60:
        role = "contextual"
        reason = f"Broad semantic similarity only. {enriched.get('selection_reason')}"
    else:
        role = "reject"
        reason = f"Low topic evidence. {enriched.get('selection_reason')}"

    if generic_terms and not direct_terms and role == "exact":
        role = "strong_related"
        reason = f"Generic-only evidence cannot be exact. {enriched.get('selection_reason')}"

    speaker_soundbite_allowed = False
    if role in ("exact", "strong_related"):
        speaker_soundbite_allowed = (
            quote_strength >= 0.50
            and clarity_score >= 0.80
        )

    speaker_visual_allowed = False
    if requested_speaker and enriched.get("speaker_match") and role != "reject":
        duration = float(enriched.get("duration") or 0.0)
        if duration > 8 and enriched.get("start") is not None and enriched.get("end") is not None:
            speaker_visual_allowed = True

    broll_allowed = role != "reject"
    if role == "reject":
        use_as = "reject"
        final_video_allowed = False
    elif role == "contextual":
        use_as = "context_only"
        final_video_allowed = False
    elif role == "weak_related":
        if speaker_visual_allowed:
            use_as = "speaker_visual"
            final_video_allowed = True
        else:
            use_as = "narration_support"
            final_video_allowed = False
    elif speaker_soundbite_allowed:
        use_as = "speaker_soundbite"
        final_video_allowed = True
    elif speaker_visual_allowed:
        use_as = "speaker_visual"
        final_video_allowed = True
    elif "archive_visual" in usage_candidates or "evidence_clip" in usage_candidates:
        use_as = "archive_visual"
        final_video_allowed = True
    else:
        use_as = "narration_support"
        final_video_allowed = True

    enriched.update(
        {
            "prompt_relevance_score": round(final_prompt_score, 6),
            "relevance_role": role,
            "speaker_soundbite_allowed": speaker_soundbite_allowed,
            "speaker_visual_allowed": speaker_visual_allowed,
            "speaker_clip_allowed": speaker_soundbite_allowed,
            "broll_allowed": broll_allowed,
            "final_video_allowed": final_video_allowed,
            "use_as": use_as,
            "relevance_reason": reason,
        }
    )
    return enriched


def annotate_candidate_relevance(
    records: list[dict[str, Any]],
    prompt: str,
    intent: dict[str, Any],
    known_speakers: list[str] | None = None,
) -> list[dict[str, Any]]:
    if not records:
        return []

    if known_speakers is None:
        known_speakers = load_known_speakers()

    requested_speaker = intent.get("requested_speaker")

    if requested_speaker:
        log(f"Speaker constraint detected: '{requested_speaker}'")
    if intent.get("topic_terms"):
        log(f"Meaningful prompt terms: {sorted(intent.get('topic_terms') or [])}")
    if intent.get("expanded_terms"):
        log(f"Expanded synonym terms: {sorted(intent.get('expanded_terms') or [])}")
    if intent.get("negative_terms"):
        log(f"Negative terms: {sorted(intent.get('negative_terms') or [])}")

    annotated = []
    for record in records:
        enriched = classify_candidate_relevance_v2(record, intent)
        annotated.append(enriched)

    return annotated


def theme_coverage(records: list[dict[str, Any]]) -> list[str]:
    return sorted({str(item.get("primary_theme")) for item in records if item.get("primary_theme")})


def chapter_coverage(records: list[dict[str, Any]]) -> list[str]:
    return sorted({str(item.get("primary_chapter")) for item in records if item.get("primary_chapter")})


def choose_video_mode(metrics: dict[str, Any], intent: dict[str, Any] | None = None) -> str:
    requested_speaker = (intent or {}).get("requested_speaker")
    exact_clip_count = int(metrics.get("exact_clip_count") or 0)
    related_clip_count = int(metrics.get("related_clip_count") or 0)
    usable_speaker_clip_count = int(metrics.get("usable_speaker_clip_count") or 0)
    unique_speaker_count = int(metrics.get("unique_speaker_count") or 0)
    unique_source_video_count = int(metrics.get("unique_source_video_count") or 0)
    theme_count = len(metrics.get("theme_coverage") or [])
    total_usable = exact_clip_count + related_clip_count

    distinct_non_null_years = len(set(
        y for y in (metrics.get("video_years") or []) if y is not None
    ))
    # Fall back to unique_source_video_count when speaker field is absent from records
    effective_voice_count = unique_speaker_count if unique_speaker_count > 0 else unique_source_video_count
    if (
        not requested_speaker
        and distinct_non_null_years >= 2
        and exact_clip_count >= 3
        and effective_voice_count >= 2
    ):
        return "chronological_relay"

    if requested_speaker:
        usable_speaker_visual_count = int(metrics.get("usable_speaker_visual_count") or 0)
        total_speaker_usable = usable_speaker_clip_count + usable_speaker_visual_count
        if total_speaker_usable == 0:
            return "no_evidence"
        if total_speaker_usable <= 2:
            return "low_evidence_speaker_brief"
        return "speaker_led_theme_summary"

    if exact_clip_count == 0:
        return "low_evidence_brief"
    if exact_clip_count >= 4 and usable_speaker_clip_count >= 3:
        return "speaker_led_documentary"
    if exact_clip_count >= 3 and unique_speaker_count >= 3 and theme_count <= 4:
        return "relay_of_ideas"
    if exact_clip_count >= 2 and related_clip_count >= 4:
        return "theme_summary_explainer"
    if exact_clip_count >= 1 and related_clip_count >= 3:
        return "narration_led_explainer"
    if total_usable < 3 or unique_source_video_count < 2:
        return "low_evidence_brief"
    return "narration_led_explainer"


def evidence_strength_from_metrics(metrics: dict[str, Any], intent: dict[str, Any] | None = None) -> str:
    requested_speaker = (intent or {}).get("requested_speaker")
    exact_clip_count = int(metrics.get("exact_clip_count") or 0)
    strong_related_count = int(metrics.get("strong_related_count") or 0)
    usable_speaker_clip_count = int(metrics.get("usable_speaker_clip_count") or 0)

    if requested_speaker:
        usable_speaker_visual_count = int(metrics.get("usable_speaker_visual_count") or 0)
        total_speaker_usable = usable_speaker_clip_count + usable_speaker_visual_count
        if total_speaker_usable >= 4:
            return "high"
        if total_speaker_usable >= 2:
            return "medium"
        if total_speaker_usable == 1:
            return "low"
        return "none"

    if exact_clip_count >= 4 and usable_speaker_clip_count >= 3:
        return "high"
    if exact_clip_count >= 2 or strong_related_count >= 4:
        return "medium"
    return "low"


def get_selected_candidate_cap(target_duration_seconds: int, video_mode: str, evidence_strength: str) -> int:
    if evidence_strength == "low" or video_mode in ("low_evidence_brief", "low_evidence_speaker_brief", "no_evidence"):
        return 5
    if target_duration_seconds <= 60:
        return 10
    if target_duration_seconds <= 120:
        return 16
    if target_duration_seconds <= 180:
        return 24
    return 30


def sort_final_video_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(c: dict[str, Any]) -> tuple:
        role = c.get("relevance_role", "")
        role_priority = 0
        if role == "exact":
            role_priority = 3
        elif role == "strong_related":
            role_priority = 2
        elif role == "weak_related":
            role_priority = 1
        return (
            role_priority,
            float(c.get("final_prompt_score") or c.get("prompt_relevance_score") or 0),
            1 if c.get("speaker_soundbite_allowed") or c.get("speaker_clip_allowed") else 0,
            1 if c.get("speaker_visual_allowed") else 0,
            1 if c.get("use_as") in ("speaker_soundbite", "speaker_visual") else 0,
            float(c.get("selection_score") or 0),
            float(c.get("semantic_score") or 0),
            float(c.get("quote_strength") or 0),
            float(c.get("clarity_score") or 0),
        )
    return sorted(candidates, key=sort_key, reverse=True)


def apply_diversity_cap(
    candidates: list[dict[str, Any]],
    max_per_source: int = 2,
    max_per_speaker: int = 2,
    requested_speaker: str | None = None,
) -> list[dict[str, Any]]:
    source_counts: dict[str, int] = {}
    speaker_counts: dict[str, int] = {}
    result = []
    for c in candidates:
        source = str(c.get("source_video") or "unknown")
        speaker = str(c.get("speaker") or "unknown")
        if source_counts.get(source, 0) >= max_per_source:
            continue
        # Speaker cap only applies when no speaker is specifically requested
        if not requested_speaker and speaker != "unknown" and speaker_counts.get(speaker, 0) >= max_per_speaker:
            continue
        result.append(c)
        source_counts[source] = source_counts.get(source, 0) + 1
        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
    # Ensure at least 1 candidate survives when input is non-empty
    if not result and candidates:
        result = [candidates[0]]
    return result


def dedupe_by_fragment_id(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        fragment_id = str(candidate.get("fragment_id") or "").strip()
        if not fragment_id or fragment_id in seen:
            continue
        seen.add(fragment_id)
        deduped.append(candidate)
    return deduped


def timeline_segment_count_for_duration(target_duration_seconds: int) -> int:
    if target_duration_seconds <= 60:
        return 4
    if target_duration_seconds <= 120:
        return 6
    if target_duration_seconds <= 300:
        return 8
    if target_duration_seconds <= 600:
        return 12
    return 15


def timeline_card_duration(target_duration_seconds: int) -> int:
    return 4 if target_duration_seconds <= 60 else 5


def classify_timeline_segment_type(candidate: dict[str, Any]) -> str:
    if candidate.get("speaker_clip_allowed") or candidate.get("speaker_soundbite_allowed"):
        return "speaker_clip"
    if candidate.get("speaker_visual_allowed"):
        return "speaker_visual"
    if candidate.get("use_as") == "archive_visual":
        return "archive_visual"
    if candidate.get("use_as") == "speaker_soundbite":
        return "speaker_clip"
    return "narration_broll"


def normalize_candidate_time_bounds(candidate: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    start = candidate.get("start")
    end = candidate.get("end")
    duration = candidate.get("duration")

    if start is None:
        start = candidate.get("start_seconds")
    if end is None:
        end = candidate.get("end_seconds")

    try:
        start_value = float(start) if start is not None else None
    except (TypeError, ValueError):
        start_value = None
    try:
        end_value = float(end) if end is not None else None
    except (TypeError, ValueError):
        end_value = None
    try:
        duration_value = float(duration) if duration is not None else None
    except (TypeError, ValueError):
        duration_value = None

    if duration_value is None and start_value is not None and end_value is not None:
        duration_value = max(0.0, end_value - start_value)
    if end_value is None and start_value is not None and duration_value is not None:
        end_value = start_value + duration_value
    if start_value is None and end_value is not None and duration_value is not None:
        start_value = max(0.0, end_value - duration_value)

    return start_value, end_value, duration_value


def build_export_timeline_from_candidates(
    selected_candidates: list[dict],
    prompt_profile: dict,
    target_duration_seconds: int,
    title: str | None = None,
    video_mode: str | None = None,
) -> list[dict]:
    safe_candidates = [
        dict(candidate)
        for candidate in selected_candidates
        if candidate.get("final_video_allowed") is True and candidate.get("fragment_id")
    ]
    safe_candidates = dedupe_by_fragment_id(safe_candidates)
    if not safe_candidates:
        return []

    if video_mode == "chronological_relay":
        ranked_candidates = sorted(
            safe_candidates,
            key=lambda c: (
                c.get("video_year") is None,
                c.get("video_year") or 9999,
                -float(c.get("final_prompt_score") or c.get("prompt_relevance_score") or 0.0),
            ),
        )
    else:
        def sort_key(candidate: dict[str, Any]) -> tuple:
            segment_type = classify_timeline_segment_type(candidate)
            type_priority = {
                "speaker_clip": 4,
                "speaker_visual": 3,
                "archive_visual": 2,
                "narration_broll": 1,
            }.get(segment_type, 0)
            role_priority = {
                "exact": 3,
                "strong_related": 2,
                "weak_related": 1,
            }.get(str(candidate.get("relevance_role") or ""), 0)
            return (
                type_priority,
                role_priority,
                float(candidate.get("final_prompt_score") or candidate.get("prompt_relevance_score") or 0.0),
                float(candidate.get("selection_score") or 0.0),
                float(candidate.get("semantic_score") or 0.0),
            )

        ranked_candidates = sorted(safe_candidates, key=sort_key, reverse=True)

    desired_segment_count = min(timeline_segment_count_for_duration(target_duration_seconds), len(ranked_candidates))
    chosen: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}

    if video_mode == "chronological_relay":
        # Per-year cap: prevent one over-represented year from filling all slots.
        # Each distinct year gets at most ceil(desired / distinct_year_count) slots
        # in the first pass, then backfill relaxes the year cap (source cap stays).
        distinct_years = sorted(set(
            c.get("video_year") for c in ranked_candidates
            if c.get("video_year") is not None
        ))
        n_years = max(1, len(distinct_years))
        # ceiling division — every year gets at least 1 guaranteed slot
        year_cap = max(2, -(-desired_segment_count // n_years))
        year_counts: dict[int, int] = {}

        for candidate in ranked_candidates:
            if len(chosen) >= desired_segment_count:
                break
            source_video = str(candidate.get("source_video") or "unknown")
            year = candidate.get("video_year")
            if source_counts.get(source_video, 0) >= 2:
                continue
            if year is not None and year_counts.get(year, 0) >= year_cap:
                continue
            chosen.append(candidate)
            source_counts[source_video] = source_counts.get(source_video, 0) + 1
            if year is not None:
                year_counts[year] = year_counts.get(year, 0) + 1

        # Backfill: relax year cap, keep source cap
        if len(chosen) < desired_segment_count:
            chosen_ids = {str(item.get("fragment_id")) for item in chosen}
            for candidate in ranked_candidates:
                if len(chosen) >= desired_segment_count:
                    break
                fragment_id = str(candidate.get("fragment_id") or "")
                if fragment_id in chosen_ids:
                    continue
                source_video = str(candidate.get("source_video") or "unknown")
                if source_counts.get(source_video, 0) >= 2:
                    continue
                chosen.append(candidate)
                chosen_ids.add(fragment_id)
                source_counts[source_video] = source_counts.get(source_video, 0) + 1
    else:
        for candidate in ranked_candidates:
            if len(chosen) >= desired_segment_count:
                break
            source_video = str(candidate.get("source_video") or "unknown")
            if source_counts.get(source_video, 0) >= 2:
                continue
            chosen.append(candidate)
            source_counts[source_video] = source_counts.get(source_video, 0) + 1

    if len(chosen) < desired_segment_count:
        chosen_ids = {str(item.get("fragment_id")) for item in chosen}
        for candidate in ranked_candidates:
            if len(chosen) >= desired_segment_count:
                break
            fragment_id = str(candidate.get("fragment_id") or "")
            if fragment_id in chosen_ids:
                continue
            chosen.append(candidate)
            chosen_ids.add(fragment_id)

    if not chosen:
        return []

    card_duration = timeline_card_duration(target_duration_seconds)
    content_budget = max(1, target_duration_seconds - (card_duration * 2))
    per_segment_duration = max(6.0, round(content_budget / max(1, len(chosen)), 2))
    timeline: list[dict[str, Any]] = [
        {
            "type": "title_card",
            "title": title or prompt_profile.get("raw_prompt") or "AI4IA Storyline",
            "duration": float(card_duration),
        }
    ]

    used_fragment_ids: set[str] = set()
    section_roles = [
        "hook", "human_context", "context", "tension",
        "problem_or_tension", "expert_evidence", "solution_direction", "final_synthesis",
    ]
    _last_year_stamp: int | None = object()  # sentinel — never equals any real year

    for idx, candidate in enumerate(chosen):
        fragment_id = str(candidate.get("fragment_id") or "").strip()
        if not fragment_id:
            continue

        if video_mode == "chronological_relay":
            candidate_year = candidate.get("video_year")
            if candidate_year is not None and candidate_year != _last_year_stamp:
                timeline.append({
                    "type": "year_stamp",
                    "year": candidate_year,
                    "duration": 2.5,
                    "audio_mode": "silent",
                    "muted": True,
                    "section_role": "temporal_transition",
                })
                _last_year_stamp = candidate_year

        section_role = section_roles[idx % len(section_roles)]
        dummy_section = {"section_type": section_role, "role": section_role}

        moment = select_best_moment_for_section(
            section=dummy_section,
            candidates=[candidate],
            prompt_profile=prompt_profile,
            used_fragment_ids=used_fragment_ids,
            section_role=section_role,
            window_seconds=per_segment_duration,
        )

        if moment is None:
            # Fallback: use legacy subclip window logic
            start_value, end_value, duration_value = normalize_candidate_time_bounds(candidate)
            clip_duration = duration_value if duration_value is not None else per_segment_duration
            clip_duration = max(1.0, min(per_segment_duration, clip_duration))
            if start_value is not None:
                end_value = start_value + clip_duration if end_value is None else min(end_value, start_value + clip_duration)
                clip_duration = max(0.1, end_value - start_value)
            window_candidate = dict(candidate)
            if start_value is not None:
                window_candidate["start"] = start_value
            if end_value is not None:
                window_candidate["end"] = end_value
            legacy_window = find_best_subclip_window(window_candidate, prompt_profile, desired_duration=clip_duration)
            if legacy_window.get("adjusted"):
                start_value = legacy_window["start"]
                end_value = legacy_window["end"]
                clip_duration = max(0.1, end_value - start_value)
            legacy_seg_type = classify_timeline_segment_type(candidate)
            segment = {
                "type": legacy_seg_type,
                "segment_type": legacy_seg_type,
                "audio_mode": "source_audio" if legacy_seg_type == "speaker_clip" else "silent",
                "fragment_id": fragment_id,
                "source_video": candidate.get("source_video"),
                "video_year": candidate.get("video_year"),
                "video_upload_date": candidate.get("video_upload_date"),
                "start": round(float(start_value), 3) if start_value is not None else None,
                "end": round(float(end_value), 3) if end_value is not None else None,
                "duration": round(float(clip_duration), 3),
                "use_as": candidate.get("use_as"),
                "relevance_role": candidate.get("relevance_role"),
                "selection_reason": candidate.get("selection_reason"),
                "speaker_clip_allowed": bool(candidate.get("speaker_clip_allowed") or candidate.get("speaker_soundbite_allowed")),
                "speaker_visual_allowed": bool(candidate.get("speaker_visual_allowed")),
                "subclip_adjusted": legacy_window.get("adjusted", False),
                "moment_text": "",
                "moment_summary": "",
                "moment_selection_reason": "legacy_fallback_no_moment_selected",
                "section_role": section_role,
                "muted": not bool(candidate.get("speaker_clip_allowed") or candidate.get("speaker_soundbite_allowed")),
                "subtitle_text": "",
                "subtitle_kind": "none",
                "speech_quality": {},
                "moment_scores": {},
            }
        else:
            used_fragment_ids.add(fragment_id)
            seg_type = moment.get("segment_type") or moment["type"]
            audio_mode = moment.get("audio_mode") or (
                "source_audio" if seg_type == "speaker_clip" else "silent"
            )
            segment = {
                "type": seg_type,
                "segment_type": seg_type,
                "audio_mode": audio_mode,
                "fragment_id": fragment_id,
                "source_video": moment.get("source_video") or candidate.get("source_video"),
                "video_year": candidate.get("video_year"),
                "video_upload_date": candidate.get("video_upload_date"),
                "start": moment["start"],
                "end": moment["end"],
                "duration": moment["duration"],
                "use_as": moment.get("use_as") or candidate.get("use_as"),
                "relevance_role": moment.get("relevance_role") or candidate.get("relevance_role"),
                "selection_reason": moment.get("selection_reason") or candidate.get("selection_reason"),
                "speaker_clip_allowed": moment.get("speaker_clip_allowed", False),
                "speaker_visual_allowed": moment.get("speaker_visual_allowed", False),
                "subclip_adjusted": moment.get("subclip_adjusted", False),
                "moment_text": moment.get("moment_text", ""),
                "moment_summary": moment.get("moment_summary", ""),
                "moment_selection_reason": moment.get("moment_selection_reason", ""),
                "section_role": moment.get("section_role", section_role),
                "muted": moment.get("muted", True),
                "subtitle_text": moment.get("subtitle_text", ""),
                "subtitle_kind": moment.get("subtitle_kind", "none"),
                "speech_quality": moment.get("speech_quality", {}),
                "moment_scores": moment.get("moment_scores", {}),
            }
        timeline.append(segment)

    timeline.append(
        {
            "type": "closing_card",
            "title": title or prompt_profile.get("raw_prompt") or "AI4IA Storyline",
            "duration": float(card_duration),
        }
    )
    return timeline


def build_evidence_gap_notes(
    prompt_intent: dict[str, Any],
    prompt_terms: set[str],
    evidence_metrics: dict[str, Any],
) -> list[str]:
    notes = []
    exact = int(evidence_metrics.get("exact_clip_count") or 0)
    usable_speaker = int(evidence_metrics.get("usable_speaker_clip_count") or 0)
    requested_speaker = prompt_intent.get("requested_speaker")

    WOMEN_TERMS = {"women", "woman", "female", "girls", "empowerment", "empower"}
    CRYPTO_TERMS_SET = {"crypto", "cryptocurrency", "blockchain", "web3"}

    if exact == 0:
        notes.append(
            "No direct evidence clips were found for this prompt. "
            "The video should be narration-led and should not present adjacent material as direct evidence."
        )

    if exact == 0 and prompt_terms & WOMEN_TERMS:
        notes.append(
            "No direct women/empowerment clips were found. "
            "Adjacent gender, diversity, or inclusion clips may be used only as contextual support."
        )

    if exact == 0 and prompt_terms & CRYPTO_TERMS_SET:
        notes.append(
            "No direct cryptocurrency/blockchain evidence was found. "
            "Do not use cryptography or general AI governance clips as direct crypto evidence."
        )

    usable_speaker_visual = int(evidence_metrics.get("usable_speaker_visual_count") or 0)
    total_speaker_usable = usable_speaker + usable_speaker_visual
    if requested_speaker and total_speaker_usable <= 2:
        count_label = "No" if total_speaker_usable == 0 else f"Only {total_speaker_usable}"
        notes.append(
            f"{count_label} usable speaker clip(s) or visual(s) found for {requested_speaker}. "
            f"Do not use unrelated speakers as filler."
        )

    return notes


def build_storyline_evidence_package(profile: dict, buckets: dict) -> dict:
    selected = buckets.get("selected_candidates", [])
    supporting = buckets.get("supporting_candidates", [])
    exact_selected = [item for item in selected if item.get("relevance_role") == "exact"]

    evidence_gaps: list[str] = []
    must_not_claim: list[str] = []

    topic_terms = set(profile.get("topic_terms") or [])
    if profile.get("requested_speaker") and not exact_selected:
        evidence_gaps.append(f"Only limited archive evidence was found for {profile.get('requested_speaker')}.")
    elif not exact_selected:
        evidence_gaps.append("No direct evidence clips were found for this prompt.")
    if topic_terms & {"women", "woman", "empowerment", "empower"} and not exact_selected:
        evidence_gaps.append("No direct women/empowerment clips were found.")
        must_not_claim.append("Do not present generic inclusion/accessibility clips as direct women empowerment evidence.")
    if "crypto" in topic_terms and not exact_selected:
        evidence_gaps.append("No direct cryptocurrency/blockchain evidence was found.")
        must_not_claim.append("Do not treat cryptography or quantum encryption as cryptocurrency evidence.")
    elif "crypto" in topic_terms and not any(
        any(normalize_name(term) in {normalize_name(v) for v in CRYPTO_TERMS} for term in (item.get("direct_evidence_terms") or []) + (item.get("expanded_evidence_terms") or []))
        for item in selected
    ):
        must_not_claim.append("Do not treat cryptography or quantum encryption as cryptocurrency evidence.")
    if profile.get("requested_speaker"):
        must_not_claim.append("Do not use unrelated speakers as filler.")

    return {
        "prompt_profile": profile,
        "selected_evidence": build_candidate_pack(selected),
        "supporting_context": build_candidate_pack(supporting[:10]),
        "evidence_gaps": evidence_gaps,
        "must_not_claim": must_not_claim,
    }


def fallback_risk_from_metrics(metrics: dict[str, Any]) -> str:
    total_usable = int(metrics.get("exact_clip_count") or 0) + int(metrics.get("related_clip_count") or 0)
    unique_source_video_count = int(metrics.get("unique_source_video_count") or 0)

    if total_usable >= 6 and unique_source_video_count >= 4:
        return "low"
    if total_usable >= 3:
        return "medium"
    return "high"


def build_director_plan(metrics: dict[str, Any], intent: dict[str, Any] | None = None) -> dict[str, Any]:
    requested_speaker = (intent or {}).get("requested_speaker")
    video_mode = choose_video_mode(metrics, intent=intent)
    evidence_strength = evidence_strength_from_metrics(metrics, intent=intent)

    structure_map = {
        "chronological_relay": [
            {
                "role": "temporal_context",
                "section_type": "narration",
                "narration_guidance": "Establish the historical or multi-year arc — describe how the topic has evolved across conference years.",
                "evidence_expectation": "clips or narration grounded in the earliest year represented in the candidate pool",
            },
            {
                "role": "human_problem",
                "section_type": "speaker_soundbite",
                "narration_guidance": "A speaker articulates the core human challenge or gap that drives the relay of ideas across time.",
                "evidence_expectation": "exact-match soundbite from any year that names the problem clearly",
            },
            {
                "role": "risk_escalation",
                "section_type": "speaker_soundbite",
                "narration_guidance": "A voice from a later year escalates the stakes — what has worsened, spread, or gone unaddressed.",
                "evidence_expectation": "strong-related or exact clip from a more recent year than the human_problem clip",
            },
            {
                "role": "solution_direction",
                "section_type": "speaker_soundbite",
                "narration_guidance": "A speaker proposes or describes a concrete direction, policy, or intervention pointing toward resolution.",
                "evidence_expectation": "exact or strong-related clip with forward-looking or prescriptive language",
            },
            {
                "role": "synthesis",
                "section_type": "narration",
                "narration_guidance": "Narrator closes the relay — summarising what the sequence of voices reveals about progress, persistence of the problem, or urgency for action.",
                "evidence_expectation": "narration only; may reference multiple clips from the relay by year",
            },
        ],
        "speaker_led_documentary": [
            "opening_hook",
            "exact_evidence_clip",
            "speaker_voice_two",
            "bridge_narration",
            "speaker_voice_three",
            "closing_takeaway",
        ],
        "relay_of_ideas": [
            "opening_context",
            "idea_one",
            "idea_two",
            "tension_point",
            "synthesis_narration",
            "closing_takeaway",
        ],
        "theme_summary_explainer": [
            "opening_context",
            "related_theme_clip",
            "explanation_bridge",
            "exact_evidence_clip",
            "related_governance_clip",
            "closing_takeaway",
        ],
        "narration_led_explainer": [
            "opening_context",
            "exact_evidence_clip",
            "related_theme_clip",
            "synthesis_narration",
            "related_supporting_clip",
            "closing_takeaway",
        ],
        "low_evidence_brief": [
            "opening_context",
            "available_evidence_clip",
            "evidence_gap_narration",
            "related_context",
            "closing_takeaway",
        ],
        "speaker_led_theme_summary": [
            "opening_context",
            "exact_evidence_clip",
            "exact_evidence_clip",
            "bridge_narration",
            "exact_evidence_clip",
            "closing_takeaway",
        ],
        "low_evidence_speaker_brief": [
            "opening_context",
            "available_evidence_clip",
            "evidence_gap_narration",
            "closing_takeaway",
        ],
        "no_evidence": [
            "opening_context",
            "evidence_gap_narration",
            "closing_takeaway",
        ],
    }

    notes = []
    exact_clip_count = metrics.get("exact_clip_count", 0)
    usable_soundbite_count = metrics.get("usable_speaker_clip_count", 0)
    usable_visual_count = metrics.get("usable_speaker_visual_count", 0)
    total_speaker_usable = usable_soundbite_count + usable_visual_count
    if requested_speaker and total_speaker_usable == 0:
        notes.append(f"No archive clips were found for {requested_speaker}. No unrelated speakers were used as filler.")
    elif requested_speaker and total_speaker_usable <= 2:
        notes.append(f"Limited archive evidence was found for {requested_speaker}. The video uses only available {requested_speaker} clips and does not include unrelated speakers.")
    elif exact_clip_count == 0:
        notes.append("No direct evidence clips found for this prompt. Direct topic evidence is limited or absent.")
    elif exact_clip_count <= 2:
        notes.append("Exact prompt evidence was limited, so the plan leans on related theme and chapter evidence with narration bridges.")
    if usable_soundbite_count == 0 and usable_visual_count == 0:
        notes.append("No speaker clips are approved for soundbite or visual use. Narration and b-roll will carry the video.")
    elif usable_soundbite_count == 0 and usable_visual_count > 0:
        notes.append(f"{usable_visual_count} speaker visual(s) are approved for silent/muted use under narration. No high-quality soundbites available.")
    elif metrics.get("unique_speaker_count", 0) < 3:
        notes.append("Speaker diversity is limited, so narration carries more of the connective explanation.")
    if not notes:
        notes.append("The candidate pool is diverse enough to support a multi-voice documentary structure.")

    return {
        "enabled": True,
        "video_mode": video_mode,
        "evidence_strength": evidence_strength,
        "exact_clip_count": metrics.get("exact_clip_count", 0),
        "related_clip_count": metrics.get("related_clip_count", 0),
        "strong_related_count": metrics.get("strong_related_count", 0),
        "usable_speaker_clip_count": usable_soundbite_count,
        "usable_speaker_visual_count": usable_visual_count,
        "rejected_clip_count": metrics.get("rejected_clip_count", 0),
        "unique_speaker_count": metrics.get("unique_speaker_count", 0),
        "unique_source_video_count": metrics.get("unique_source_video_count", 0),
        "theme_coverage": metrics.get("theme_coverage", []),
        "chapter_coverage": metrics.get("chapter_coverage", []),
        "average_similarity": metrics.get("average_similarity"),
        "average_quote_strength": metrics.get("average_quote_strength"),
        "recommended_structure": structure_map[video_mode],
        "fallback_risk": fallback_risk_from_metrics(metrics),
        "planning_notes": notes,
        "requested_speaker": requested_speaker,
    }


def compute_evidence_metrics(
    selected: list[dict[str, Any]],
    supporting: list[dict[str, Any]] | None = None,
    context: list[dict[str, Any]] | None = None,
    rejected: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    supporting = supporting or []
    context = context or []
    rejected = rejected or []

    exact_records = [r for r in selected if r.get("relevance_role") == "exact"]
    strong_related = [r for r in selected if r.get("relevance_role") == "strong_related"]
    weak_in_selected = [r for r in selected if r.get("relevance_role") == "weak_related"]
    usable_speaker_clips = [r for r in selected if r.get("speaker_soundbite_allowed") is True or r.get("speaker_clip_allowed") is True]
    usable_speaker_visuals = [r for r in selected if r.get("speaker_visual_allowed") is True]

    usable_records = selected + supporting

    speakers = {str(r.get("speaker")) for r in selected if r.get("speaker")}
    sources = {str(r.get("source_video")) for r in selected if r.get("source_video")}
    video_years = [r.get("video_year") for r in selected]
    similarities = [float(r.get("semantic_score") or 0.0) for r in usable_records]
    quote_strengths = [float(r.get("quote_strength") or 0.0) for r in usable_records if r.get("quote_strength") is not None]

    return {
        "exact_clip_count": len(exact_records),
        "related_clip_count": len(strong_related) + len(weak_in_selected),
        "strong_related_count": len(strong_related),
        "weak_related_count": len(weak_in_selected),
        "supporting_candidate_count": len(supporting),
        "contextual_count": len(context),
        "rejected_clip_count": len(rejected),
        "usable_speaker_clip_count": len(usable_speaker_clips),
        "usable_speaker_visual_count": len(usable_speaker_visuals),
        "unique_speaker_count": len(speakers),
        "unique_source_video_count": len(sources),
        "video_years": video_years,
        "theme_coverage": theme_coverage(selected),
        "chapter_coverage": chapter_coverage(selected),
        "average_similarity": round(float(np.mean(similarities)), 6) if similarities else None,
        "average_quote_strength": round(float(np.mean(quote_strengths)), 6) if quote_strengths else None,
    }


def bucket_candidates(
    classified: list[dict[str, Any]],
    intent: dict[str, Any],
    max_low_evidence_weak: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    requested_speaker = intent.get("requested_speaker")

    selected: list[dict[str, Any]] = []
    supporting: list[dict[str, Any]] = []
    context: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for c in classified:
        role = c.get("relevance_role")
        fva = c.get("final_video_allowed", False)
        use_as = c.get("use_as", "")
        if role in ("exact", "strong_related") and fva:
            selected.append(c)
        elif role == "weak_related" and use_as == "speaker_visual" and fva:
            # Speaker-matched weak clip: allowed as a visual under narration
            selected.append(c)
        elif role == "weak_related":
            supporting.append(c)
        elif role == "contextual":
            context.append(c)
        else:
            rejected.append(c)

    # If no selected candidates, promote top weak_related as narration_support (low-evidence mode)
    if not selected and supporting:
        promoted = supporting[:max_low_evidence_weak]
        for c in promoted:
            c = dict(c)
            c["final_video_allowed"] = True
            c["use_as"] = "narration_support"
            c["speaker_soundbite_allowed"] = False
            c["speaker_visual_allowed"] = False
            c["speaker_clip_allowed"] = False
            selected.append(c)
        supporting = supporting[max_low_evidence_weak:]

    # Speaker prompt hard enforcement: remove non-matching from selected
    if requested_speaker:
        bad = [c for c in selected if not c.get("speaker_match")]
        if bad:
            for c in bad:
                c2 = dict(c)
                c2["final_video_allowed"] = False
                c2["use_as"] = "reject"
                rejected.append(c2)
            selected = [c for c in selected if c.get("speaker_match")]

    return {
        "selected_candidates": selected,
        "supporting_candidates": supporting,
        "context_candidates": context,
        "rejected_candidates": rejected,
    }


def score_all_records(prompt: str) -> list[dict[str, Any]]:
    global _FILTERED_RECORDS, _FILTERED_MATRIX

    if _FILTERED_RECORDS is not None and _FILTERED_MATRIX is not None:
        records = _FILTERED_RECORDS
        matrix = _FILTERED_MATRIX
    elif _chroma_mode():
        matrix, records = _load_corpus_from_chroma()
    else:
        if not _ACTIVE_MATRIX_PATH.exists():
            raise FileNotFoundError(f"Missing embedding matrix: {_ACTIVE_MATRIX_PATH}")
        if not _ACTIVE_JSONL_PATH.exists():
            raise FileNotFoundError(f"Missing embedding JSONL: {_ACTIVE_JSONL_PATH}")
        matrix = np.load(_ACTIVE_MATRIX_PATH)
        records = load_embedding_records()

    if matrix.shape[0] != len(records):
        raise RuntimeError(
            f"Matrix/metadata mismatch. Matrix rows={matrix.shape[0]}, records={len(records)}"
        )

    query_vector = embed_query(prompt)
    similarities = matrix @ query_vector
    top_indices = np.argsort(similarities)[::-1]

    scored = []
    for rank, idx in enumerate(top_indices, start=1):
        record = dict(records[int(idx)])
        record["retrieval_rank"] = rank
        record["semantic_score"] = round(float(similarities[int(idx)]), 6)
        scored.append(record)

    return scored


def embed_query(query: str) -> np.ndarray:
    """
    Embed the user prompt for retrieval.

    Supports both Ollama embedding APIs:
    - /api/embed       -> {"model": ..., "input": ...}
    - /api/embeddings -> {"model": ..., "prompt": ...}

    Some Ollama deployments expose only one of these endpoints.
    """
    normalized_query = query.strip()

    if not normalized_query.startswith("search_query:"):
        normalized_query = "search_query: " + normalized_query

    errors = []

    # Newer Ollama endpoint
    embed_url = f"{OLLAMA_BASE_URL}/api/embed"
    embed_payload = {
        "model": EMBED_MODEL,
        "input": normalized_query,
    }

    try:
        data = post_with_retries(embed_url, embed_payload, timeout=EMBED_TIMEOUT_SECONDS)

        if "embedding" in data:
            return np.array(data["embedding"], dtype=np.float32)

        if "embeddings" in data and data["embeddings"]:
            return np.array(data["embeddings"][0], dtype=np.float32)

        errors.append(f"/api/embed returned unexpected response keys: {list(data.keys())}")
    except Exception as exc:
        errors.append(f"/api/embed failed: {exc}")

    # Older Ollama endpoint
    embeddings_url = f"{OLLAMA_BASE_URL}/api/embeddings"
    embeddings_payload = {
        "model": EMBED_MODEL,
        "prompt": normalized_query,
    }

    try:
        data = post_with_retries(embeddings_url, embeddings_payload, timeout=EMBED_TIMEOUT_SECONDS)

        if "embedding" in data:
            return np.array(data["embedding"], dtype=np.float32)

        if "embeddings" in data and data["embeddings"]:
            return np.array(data["embeddings"][0], dtype=np.float32)

        errors.append(f"/api/embeddings returned unexpected response keys: {list(data.keys())}")
    except Exception as exc:
        errors.append(f"/api/embeddings failed: {exc}")

    raise RuntimeError("Embedding request failed. " + " | ".join(errors))


def load_embedding_records() -> list[dict[str, Any]]:
    if _chroma_mode():
        _, records = _load_corpus_from_chroma()
        return records

    records = []
    with _ACTIVE_JSONL_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                record.pop("embedding", None)
                record.pop("embedding_text", None)
                records.append(record)
    return records


def score_and_retrieve(prompt: str, top_k: int) -> list[dict[str, Any]]:
    return score_all_records(prompt)[:top_k]


def merge_candidate_sets(*candidate_sets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    ordered: list[dict[str, Any]] = []
    for candidate_set in candidate_sets:
        for item in candidate_set:
            fragment_id = str(item.get("fragment_id") or "")
            if not fragment_id:
                continue
            channels = list(dict.fromkeys(item.get("retrieval_channels") or []))
            if fragment_id in merged:
                existing = merged[fragment_id]
                existing["retrieval_channels"] = list(
                    dict.fromkeys((existing.get("retrieval_channels") or []) + channels)
                )
                continue
            candidate = dict(item)
            candidate["retrieval_channels"] = channels
            merged[fragment_id] = candidate
            ordered.append(candidate)
    return ordered


def hybrid_retrieve_candidates(
    profile: dict,
    records: list[dict],
    embeddings_index,
    top_k: int = 100,
) -> list[dict]:
    requested_speaker = profile.get("requested_speaker")
    topic_terms = profile.get("topic_terms") or []
    expanded_terms = profile.get("expanded_terms") or []
    search_terms = list(dict.fromkeys(topic_terms + expanded_terms))
    theme_hint = profile.get("theme_hint")
    chapter_hint = profile.get("chapter_hint")

    def with_channel(items: list[dict[str, Any]], channel: str) -> list[dict[str, Any]]:
        result = []
        for item in items:
            candidate = dict(item)
            candidate["retrieval_channels"] = list(dict.fromkeys((candidate.get("retrieval_channels") or []) + [channel]))
            result.append(candidate)
        return result

    semantic_pool = with_channel(list(embeddings_index[:top_k]), "semantic")

    keyword_pool: list[dict[str, Any]] = []
    theme_pool: list[dict[str, Any]] = []
    speaker_pool: list[dict[str, Any]] = []

    for record in records:
        blob = candidate_search_blob(record)
        if search_terms:
            direct_hits = match_terms_in_text(blob, topic_terms)
            expanded_hits = match_terms_in_text(blob, expanded_terms)
            if direct_hits or expanded_hits:
                keyword_pool.append(dict(record))
        if theme_hint and normalize_name(str(record.get("primary_theme") or "")) == normalize_name(str(theme_hint)):
            theme_pool.append(dict(record))
        if chapter_hint and normalize_name(str(record.get("primary_chapter") or "")) == normalize_name(str(chapter_hint)):
            theme_pool.append(dict(record))
        if requested_speaker:
            speaker_match, speaker_reason = candidate_speaker_matches(record, requested_speaker)
            if speaker_match:
                candidate = dict(record)
                candidate["speaker_match"] = True
                candidate["speaker_match_reason"] = speaker_reason
                speaker_pool.append(candidate)

    keyword_pool = sorted(
        with_channel(keyword_pool, "keyword"),
        key=lambda item: (
            len(match_terms_in_text(candidate_search_blob(item), topic_terms)),
            len(match_terms_in_text(candidate_search_blob(item), expanded_terms)),
            float(item.get("semantic_score") or 0.0),
        ),
        reverse=True,
    )[: max(10, top_k // 3)]
    theme_pool = sorted(
        with_channel(theme_pool, "chapter" if chapter_hint else "theme"),
        key=lambda item: float(item.get("semantic_score") or 0.0),
        reverse=True,
    )[: max(10, top_k // 3)]
    speaker_pool = sorted(
        with_channel(speaker_pool, "speaker"),
        key=lambda item: (
            float(item.get("quote_strength") or 0.0),
            float(item.get("clarity_score") or 0.0),
            float(item.get("semantic_score") or 0.0),
        ),
        reverse=True,
    )[:top_k]

    if requested_speaker:
        return merge_candidate_sets(speaker_pool, keyword_pool, theme_pool)
    return merge_candidate_sets(semantic_pool, keyword_pool, theme_pool)


def compute_retrieval_diagnostics(profile: dict[str, Any], retrieved: list[dict[str, Any]], buckets: dict[str, list[dict[str, Any]]], top_k: int) -> dict[str, Any]:
    semantic_count = sum(1 for item in retrieved if "semantic" in (item.get("retrieval_channels") or []))
    keyword_count = sum(1 for item in retrieved if "keyword" in (item.get("retrieval_channels") or []))
    chapter_count = sum(1 for item in retrieved if any(ch in (item.get("retrieval_channels") or []) for ch in ("chapter", "theme")))
    speaker_count = sum(1 for item in retrieved if "speaker" in (item.get("retrieval_channels") or []))
    return {
        "semantic_count": min(top_k, semantic_count),
        "keyword_count": keyword_count,
        "chapter_count": chapter_count,
        "speaker_count": speaker_count,
        "merged_count": len(retrieved),
        "selected_count": len(buckets.get("selected_candidates") or []),
        "supporting_count": len(buckets.get("supporting_candidates") or []),
        "context_count": len(buckets.get("context_candidates") or []),
        "rejected_count": len(buckets.get("rejected_candidates") or []),
        "speaker_required": bool(profile.get("requested_speaker")),
    }


def retrieve_candidates(
    prompt: str,
    intent: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    scored = score_all_records(prompt)
    hybrid = hybrid_retrieve_candidates(intent, scored, scored, top_k=top_k)
    return hybrid[: max(top_k, len(hybrid))]


def utility_boost(record: dict[str, Any]) -> float:
    usage = set(record.get("clip_usage_candidates", []))

    boost = 0.0

    if "speaker_soundbite" in usage:
        boost += 0.08
    if "narration_support" in usage:
        boost += 0.06
    if "evidence_clip" in usage:
        boost += 0.05
    if "context_visual" in usage:
        boost += 0.03

    quote_strength = float(record.get("quote_strength") or 0)
    clarity_score = float(record.get("clarity_score") or 0)
    narration_support_score = float(record.get("narration_support_score") or 0)
    classification_confidence = float(record.get("classification_confidence") or 0)

    boost += quote_strength * 0.06
    boost += clarity_score * 0.04
    boost += narration_support_score * 0.04
    boost += classification_confidence * 0.03

    duration = float(record.get("duration") or 0)

    if 20 <= duration <= 100:
        boost += 0.04
    elif duration > 160:
        boost -= 0.05

    return boost


def bucket_candidates_by_year(candidates: list[dict[str, Any]]) -> dict[int | None, list[dict[str, Any]]]:
    buckets: dict[int | None, list[dict[str, Any]]] = {}
    for candidate in candidates:
        year = candidate.get("video_year")  # int or None
        buckets.setdefault(year, []).append(candidate)
    for year in buckets:
        buckets[year] = sorted(
            buckets[year],
            key=lambda c: float(c.get("final_prompt_score") or c.get("prompt_relevance_score") or c.get("semantic_score") or 0),
            reverse=True,
        )
    return buckets


def _assign_relay_sections_by_year(
    recommended_structure: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    For chronological_relay mode: pre-assign real fragments to each recommended
    section grouped by year so the LLM receives concrete year anchors.

    Each content section gets:
      - year_slot: the representative year for that section
      - speaker_soundbites / archive_visual_fragments: 1-2 best candidates from
        that year (for the LLM to reference and narrate around)

    Years are distributed across sections in chronological order so older years
    appear earlier and newer years appear later.
    """
    if not candidates:
        return recommended_structure

    # Bucket by year; discard year=None for slot assignment
    by_year: dict[int, list[dict[str, Any]]] = {}
    for c in candidates:
        y = c.get("video_year")
        if y is not None:
            by_year.setdefault(int(y), []).append(c)

    distinct_years = sorted(by_year.keys())
    if not distinct_years:
        return recommended_structure

    # Content sections = everything except the final synthesis/narration-only section
    # which may span multiple years and shouldn't be pinned to one year.
    content_roles = [
        s for s in recommended_structure
        if s.get("section_type") != "narration" or s.get("role") not in ("temporal_context", "synthesis")
    ]
    narration_only_roles = [
        s for s in recommended_structure
        if s not in content_roles
    ]

    # Distribute years across content sections so ALL distinct years are represented.
    # When there are more years than sections, pick evenly-spaced years so the
    # earliest and latest years are always included (guaranteeing the full temporal arc).
    # When there are more sections than years, repeat years cyclically.
    year_for_section: list[int] = []
    n_sections = len(content_roles)
    n_years = len(distinct_years)
    if n_sections >= n_years:
        # More sections than years — assign years cyclically
        for i in range(n_sections):
            year_for_section.append(distinct_years[i % n_years])
    else:
        # Fewer sections than years — pick evenly-spaced years so the arc spans
        # from the earliest to the latest year (always include first and last).
        if n_sections == 1:
            year_for_section = [distinct_years[0]]
        else:
            step = (n_years - 1) / (n_sections - 1)
            year_for_section = [distinct_years[round(i * step)] for i in range(n_sections)]

    # Build enriched section copies with real fragments attached
    used_frag_ids: set[str] = set()
    enriched: list[dict[str, Any]] = []

    content_idx = 0
    for section in recommended_structure:
        sec = dict(section)
        if section in narration_only_roles:
            # Narration-only sections: assign median year as context hint but no clips
            all_years = sorted(by_year.keys())
            sec["year_slot"] = all_years[len(all_years) // 2] if all_years else None
            sec["speaker_soundbites"] = []
            sec["archive_visual_fragments"] = []
            enriched.append(sec)
            continue

        year = year_for_section[content_idx] if content_idx < len(year_for_section) else distinct_years[-1]
        content_idx += 1
        sec["year_slot"] = year

        pool = [c for c in by_year.get(year, []) if c.get("fragment_id") not in used_frag_ids]
        soundbites = []
        visuals = []
        for c in pool[:3]:
            fid = c.get("fragment_id")
            seg_type = c.get("segment_type") or c.get("type") or ""
            entry = {
                "fragment_id": fid,
                "start": c.get("start"),
                "end": c.get("end"),
                "video_year": year,
                "speaker_name": c.get("speaker_name") or c.get("speaker") or "",
                "text": (c.get("text") or "")[:200],
            }
            if seg_type in ("speaker_clip", "speaker_visual") or c.get("speaker_name"):
                soundbites.append(entry)
            else:
                visuals.append({**entry, "usage": "evidence_visual"})
            if fid:
                used_frag_ids.add(fid)
        sec["speaker_soundbites"] = soundbites[:2]
        sec["archive_visual_fragments"] = visuals[:2]
        enriched.append(sec)

    return enriched


def select_balanced_candidates(retrieved: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    for record in retrieved:
        record["selection_score"] = round(
            float(record.get("final_prompt_score") or record.get("prompt_relevance_score") or record["semantic_score"]) + utility_boost(record),
            6,
        )

    def sort_key(record: dict[str, Any]) -> tuple[int, float]:
        relevance_role = record.get("relevance_role")
        priority = 0
        if relevance_role == "exact":
            priority = 4
        elif relevance_role == "strong_related":
            priority = 3
        elif relevance_role == "weak_related":
            priority = 2
        elif relevance_role == "contextual":
            priority = 1
        # "reject" stays at 0
        return (priority, float(record["selection_score"]))

    # Filter out rejected candidates before selection
    retrieved = [r for r in retrieved if r.get("relevance_role") != "reject"]

    sorted_records = sorted(retrieved, key=sort_key, reverse=True)

    selected = []
    requested_speaker = next((r.get("requested_speaker") for r in retrieved if r.get("requested_speaker")), None)
    used_videos = {}
    used_speakers = {}
    theme_counts = {}
    year_counts: dict[int | None, int] = {}

    MAX_PER_YEAR_BUCKET = 3

    # First pass: prefer diversity by video and theme.
    for record in sorted_records:
        if len(selected) >= limit:
            break

        source_video = record.get("source_video") or "unknown"
        theme = record.get("primary_theme") or "unknown"

        used_count = used_videos.get(source_video, 0)
        theme_count = theme_counts.get(theme, 0)
        speaker = record.get("speaker") or "unknown"
        speaker_count = used_speakers.get(speaker, 0)
        year_key = record.get("video_year")  # int or None
        year_count = year_counts.get(year_key, 0)

        if used_count >= MAX_SOURCE_VIDEO_OCCURRENCES:
            continue

        if requested_speaker and source_video == "unknown" and record.get("speaker_match") is not True:
            continue

        if theme_count >= (2 if requested_speaker else 10):
            continue

        if speaker != "unknown" and speaker_count >= MAX_SPEAKER_OCCURRENCES:
            continue

        if year_count >= MAX_PER_YEAR_BUCKET:
            continue

        selected.append(record)
        used_videos[source_video] = used_count + 1
        theme_counts[theme] = theme_count + 1
        used_speakers[speaker] = speaker_count + 1
        year_counts[year_key] = year_count + 1

    # Second pass: fill remaining slots if diversity filters were too strict.
    if len(selected) < limit:
        selected_ids = {item.get("fragment_id") for item in selected}

        for record in sorted_records:
            if len(selected) >= limit:
                break

            if record.get("fragment_id") not in selected_ids:
                selected.append(record)
                selected_ids.add(record.get("fragment_id"))

    return selected[:limit]


def build_candidate_pack(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a compact candidate pack for the LLM prompt.
    Only include fields the LLM needs — trimmed to keep prompt under token budget."""
    packed = []
    for item in candidates:
        # Resolve clip_usage_candidates — may be comma-string from ChromaDB
        cuc = item.get("clip_usage_candidates", [])
        if isinstance(cuc, str):
            cuc = [c.strip() for c in cuc.split(",") if c.strip()]

        packed.append({
            "fragment_id": item.get("fragment_id"),
            "source_video": item.get("source_video"),
            "video_id": item.get("video_id"),
            "start": item.get("start"),
            "end": item.get("end"),
            "duration": item.get("duration"),
            "primary_theme": item.get("primary_theme"),
            "primary_chapter": item.get("primary_chapter"),
            "summary": truncate_text(item.get("summary", ""), 150),
            "keywords": item.get("keywords", [])[:6],
            "clip_usage_candidates": cuc,
            "quote_strength": item.get("quote_strength"),
            "clarity_score": item.get("clarity_score"),
            "relevance_role": item.get("relevance_role"),
            "speaker_clip_allowed": item.get("speaker_clip_allowed", False),
            "speaker_soundbite_allowed": item.get("speaker_soundbite_allowed", False),
            "relevance_reason": item.get("relevance_reason", ""),
            "text": truncate_text(item.get("text", ""), 200),
            "video_year": item.get("video_year"),
            "audio_quality_status": item.get("audio_quality_status"),
            "final_video_allowed": item.get("final_video_allowed", False),
        })
    return packed


def generate_llm_storyline(
    user_prompt: str,
    duration: int,
    candidates: list[dict[str, Any]],
    style: str,
    director_plan: dict[str, Any],
    prompt_intent: dict[str, Any],
    evidence_package: dict[str, Any],
) -> dict[str, Any]:
    candidate_pack = build_candidate_pack(candidates)

    _num_sections_hint = timeline_segment_count_for_duration(duration)
    _per_section_s_hint = round(max(1, duration - 10) / _num_sections_hint)
    _words_per_section_hint = max(30, round(_per_section_s_hint * 2.5))

    system_instruction = f"""
You are a senior documentary producer and AI research editor.

Create a production-ready storyline blueprint for a documentary-style video.
The source archive contains conference footage, speaker clips, panel discussions, and event visuals.
The final video should feel like a National Geographic / premium explainer documentary:
- narrator-led
- evidence-backed
- visually planned
- speaker soundbites used selectively
- generated or internet B-roll used where conference footage is insufficient

=== NARRATION LENGTH — CRITICAL REQUIREMENT ===
The video is {duration} seconds long with {_num_sections_hint} sections.
Each section narration MUST be at least {_words_per_section_hint} words long.
At 2.5 words/second TTS pace, {_words_per_section_hint} words = {_per_section_s_hint} seconds of voiceover per section.
NEVER write a narration shorter than {_words_per_section_hint} words. One or two sentences is NOT acceptable.
Write full, substantive paragraphs. Each narration should be a complete passage of {_words_per_section_hint}+ words.
===============================================

Important rules:
1. Do not invent speaker names unless they are visible in source_video text.
2. Use only fragment_id values from the provided candidate fragments.
3. Do not assign a permanent storyline_role to fragments.
4. Decide how each fragment is used only for this specific user prompt.
5. Separate real archive clips from generated/internet assets.
6. Keep narration cinematic but factual.
7. Return valid JSON only. No markdown. No backticks.
8. Respect evidence_gaps and must_not_claim. Never upgrade supporting/context clips into direct claims.
"""

    video_mode = director_plan.get("video_mode", "")
    if video_mode == "chronological_relay":
        system_instruction = system_instruction.rstrip() + """

TEMPORAL ARC RULES (chronological_relay mode):
- Each section corresponds to a distinct time period. The year_slot for each section is provided in the director_plan.
- Every section narration MUST open with a temporal anchor sentence: "In [year]..." or "By [year], ..." or "As early as [year]...".
- Do NOT invent years — use only the year_slot value provided for that section.
- The narration must reflect the intellectual/historical progression: context → problem → risk → solution → synthesis.
- Voice-over bridges between sections must reference the time gap: "Three years later...", "By the following decade...", etc.
- If year_slot is null for a section, omit the temporal anchor and write a thematic bridge instead.
"""
        relay_plan = dict(director_plan)
        relay_structure = relay_plan.get("recommended_structure") or []
        relay_plan["recommended_structure"] = relay_structure
        director_plan = relay_plan

    # Compute word budget so the LLM writes narration proportional to the target duration.
    # Cards take ~10s; remaining content split equally across sections. ~2.5 words/sec for TTS.
    _num_sections = timeline_segment_count_for_duration(duration)
    _content_budget_s = max(1, duration - 10)
    _per_section_s = round(_content_budget_s / _num_sections)
    _words_per_section = max(30, round(_per_section_s * 2.5))

    user_instruction = {
        "user_prompt": user_prompt,
        "prompt_intent": prompt_intent,
        "target_duration_seconds": duration,
        "narration_guidance": (
            f"The video is {duration} seconds long with {_num_sections} sections. "
            f"Each section has roughly {_per_section_s} seconds of content. "
            f"Write each section's narration/voiceover as approximately {_words_per_section} words "
            f"(at ~2.5 words/second TTS pace). "
            f"IMPORTANT: Do NOT write short narrations. You MUST write at least {_words_per_section} words per section. "
            f"Fill every section with substantive, specific, evidence-grounded content that the audience can follow."
        ),
        "style": style,
        "director_plan": director_plan,
        "evidence_package": evidence_package,
        "available_candidate_fragments": candidate_pack,
        "required_output_schema": {
            "title": "string",
            "logline": "string",
            "style": "string",
            "target_duration_seconds": "number",
            "estimated_duration_seconds": "number",
            "director_plan": "copy the provided director plan and refine planning_notes only if useful",
            "narrative_arc": [
                "opening_hook",
                "context",
                "problem_or_tension",
                "human_perspective",
                "expert_evidence",
                "solution_direction",
                "final_synthesis"
            ],
            "sections": [
                {
                    "section_id": "S1",
                    "section_type": "opening_hook | context | problem_or_tension | human_perspective | expert_evidence | solution_direction | final_synthesis",
                    "target_duration_seconds": "number",
                    "director_role": "opening_context | exact_evidence_clip | related_theme_clip | related_governance_clip | synthesis_narration | closing_takeaway | evidence_gap_narration | available_evidence_clip",
                    "evidence_role": "exact | related | synthesis | low_evidence",
                    "evidence_confidence": "high | medium | low",
                    "fallback_allowed": "boolean",
                    "planning_note": "string",
                    "narration": "voice-over narration for this section",
                    "archive_visual_fragments": [
                        {
                            "fragment_id": "must come from candidates",
                            "usage": "background_visual | evidence_visual | context_visual",
                            "start": "number",
                            "end": "number",
                            "reason": "why this visual supports the narration"
                        }
                    ],
                    "speaker_soundbites": [
                        {
                            "fragment_id": "must come from candidates",
                            "start": "number",
                            "end": "number",
                            "quote_summary": "short summary of what the speaker says",
                            "reason": "why this is useful"
                        }
                    ],
                    "generated_assets_needed": [
                        {
                            "asset_type": "title_card | transition_card | map_card | data_card | abstract_ai_visual",
                            "description": "what to generate",
                            "on_screen_text": "short text if needed"
                        }
                    ],
                    "internet_assets_needed": [
                        {
                            "asset_type": "b_roll | image | map | ambient_visual",
                            "description": "what to source",
                            "search_keywords": ["keyword1", "keyword2"]
                        }
                    ]
                }
            ],
            "voiceover_script": [
                {
                    "section_id": "S1",
                    "text": "narrator line",
                    "tone": "calm documentary narrator"
                }
            ],
            "selected_speaker_soundbites": [
                {
                    "fragment_id": "must come from candidates",
                    "source_video": "string",
                    "start": "number",
                    "end": "number",
                    "quote_summary": "string",
                    "use_in_section": "section_id"
                }
            ],
            "asset_plan": {
                "real_archive_assets": ["fragment_id list"],
                "generated_assets": ["descriptions"],
                "internet_assets": ["descriptions"],
                "voiceover_required": True,
                "subtitles_required": True,
                "background_music_required": True
            },
            "export_timeline": [
                {
                    "order": "number",
                    "timeline_type": "voiceover | speaker_clip | archive_visual | generated_card | internet_broll",
                    "section_id": "string",
                    "fragment_id": "optional string",
                    "duration_seconds": "number",
                    "description": "string"
                }
            ],
            "editor_notes": [
                "string"
            ]
        }
    }

    prompt = system_instruction.strip() + "\n\nINPUT:\n" + json.dumps(user_instruction, ensure_ascii=False, indent=2)

    url = OLLAMA_BASE_URL.rstrip("/") + GENERATE_ENDPOINT

    payload = {
        "model": STRONG_LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.4,
            "top_p": 0.9,
            "num_ctx": 32000
        }
    }

    log(f"Calling storyline model: {STRONG_LLM_MODEL}")
    data = post_with_retries(url, payload, timeout=600)

    response_text = data.get("response", "")

    parsed = extract_json_object(response_text)

    if parsed is None:
        raise RuntimeError("Could not parse JSON from strong LLM response")

    # Strip empty fragment_id values from export_timeline (LLMs often emit "" for narration/card entries)
    for item in parsed.get("export_timeline") or []:
        if isinstance(item, dict) and str(item.get("fragment_id") or "").strip() == "":
            item.pop("fragment_id", None)

    return parsed


def extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None

    cleaned = text.strip()

    # Remove markdown fences/backticks if model adds them.
    cleaned = re.sub(r"^```(?:json)?", "", cleaned.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"```$", "", cleaned.strip())
    cleaned = cleaned.strip("`").strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return None

    candidate = cleaned[start:end + 1]

    try:
        return json.loads(candidate)
    except Exception:
        return None


def repair_storyline_json(raw_text: str) -> dict[str, Any] | None:
    repair_prompt = f"""
Repair the following text into valid JSON only.
Do not add markdown or explanation.

TEXT:
{raw_text}
"""

    url = OLLAMA_BASE_URL.rstrip("/") + GENERATE_ENDPOINT

    payload = {
        "model": FAST_LLM_MODEL,
        "prompt": repair_prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": 12000
        }
    }

    data = post_with_retries(url, payload, timeout=300)
    return extract_json_object(data.get("response", ""))


def expand_short_narrations(
    storyline: dict[str, Any],
    target_duration_seconds: int,
) -> dict[str, Any]:
    """Expand any section narrations that are shorter than the word budget."""
    sections = storyline.get("sections") or []
    if not sections:
        return storyline

    _n = timeline_segment_count_for_duration(target_duration_seconds)
    _per_s = round(max(1, target_duration_seconds - 10) / _n)
    min_words = max(30, round(_per_s * 2.5))

    url = OLLAMA_BASE_URL.rstrip("/") + GENERATE_ENDPOINT
    expanded = 0

    for sec in sections:
        narr = sec.get("narration", "")
        word_count = len(narr.split())
        if word_count >= min_words:
            continue

        context = {
            "section_type": sec.get("section_type", ""),
            "planning_note": sec.get("planning_note", ""),
            "keywords": sec.get("evidence_terms", [])[:6],
        }

        expand_prompt = (
            f"Expand the following documentary narration to at least {min_words} words. "
            f"Keep the same meaning and tone. Write only the expanded narration text — no labels, no quotes, no explanation.\n\n"
            f"Context: {json.dumps(context)}\n\n"
            f"Original narration:\n{narr}\n\n"
            f"Expanded narration ({min_words}+ words):"
        )

        payload = {
            "model": FAST_LLM_MODEL,
            "prompt": expand_prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_ctx": 4000},
        }

        try:
            data = post_with_retries(url, payload, timeout=120)
            expanded_text = (data.get("response") or "").strip()
            if expanded_text and len(expanded_text.split()) >= min_words:
                sec["narration"] = expanded_text
                sec["viewer_narration"] = expanded_text
                expanded += 1
        except Exception:
            pass  # Keep original if expansion fails

    if expanded:
        log(f"Expanded {expanded}/{len(sections)} short narrations to ≥{min_words} words.")

    return storyline


def validate_fragment_references(storyline: dict[str, Any], valid_fragment_ids: set[str]) -> list[str]:
    issues = []

    def check_value(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                if key == "fragment_id" and nested is not None and str(nested).strip() != "":
                    if str(nested) not in valid_fragment_ids:
                        issues.append(f"Invalid fragment_id at {path}.{key}: {nested}")
                else:
                    check_value(nested, f"{path}.{key}")
        elif isinstance(value, list):
            for idx, nested in enumerate(value):
                check_value(nested, f"{path}[{idx}]")

    check_value(storyline, "storyline")

    for idx, item in enumerate(storyline.get("export_timeline", []) or []):
        if "fragment_id" in item:
            value = str(item.get("fragment_id") or "").strip()
            if not value:
                issues.append(f"Empty fragment_id at storyline.export_timeline[{idx}]")
            elif value not in valid_fragment_ids:
                issues.append(f"Invalid export_timeline fragment_id at storyline.export_timeline[{idx}]: {value}")

    return issues


def validate_speaker_selection(selected: list[dict[str, Any]], intent: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    requested_speaker = intent.get("requested_speaker")
    if not requested_speaker:
        return selected, []

    filtered = [item for item in selected if item.get("speaker_match") is True]
    issues: list[str] = []
    if len(filtered) != len(selected):
        issues.append("Removed non-matching speaker candidates during final validation.")
    return filtered, issues


def enrich_storyline_with_director_plan(
    storyline: dict[str, Any],
    director_plan: dict[str, Any],
) -> dict[str, Any]:
    enriched = dict(storyline)
    enriched["director_plan"] = director_plan

    mode = director_plan.get("video_mode", "narration_led_explainer")
    default_section_roles = {
        "speaker_led_documentary": ["opening_context", "exact_evidence_clip", "exact_evidence_clip", "bridge_narration", "exact_evidence_clip", "synthesis_narration", "closing_takeaway"],
        "speaker_led_theme_summary": ["opening_context", "exact_evidence_clip", "exact_evidence_clip", "bridge_narration", "exact_evidence_clip", "closing_takeaway"],
        "low_evidence_speaker_brief": ["opening_context", "available_evidence_clip", "evidence_gap_narration", "closing_takeaway"],
        "no_evidence": ["opening_context", "evidence_gap_narration", "closing_takeaway"],
        "relay_of_ideas": ["opening_context", "exact_evidence_clip", "related_theme_clip", "related_theme_clip", "synthesis_narration", "related_governance_clip", "closing_takeaway"],
        "theme_summary_explainer": ["opening_context", "related_theme_clip", "related_theme_clip", "exact_evidence_clip", "related_governance_clip", "synthesis_narration", "closing_takeaway"],
        "narration_led_explainer": ["opening_context", "exact_evidence_clip", "related_theme_clip", "related_theme_clip", "synthesis_narration", "related_supporting_clip", "closing_takeaway"],
        "low_evidence_brief": ["opening_context", "available_evidence_clip", "evidence_gap_narration", "related_context", "related_context", "synthesis_narration", "closing_takeaway"],
    }
    role_sequence = default_section_roles.get(mode, default_section_roles["narration_led_explainer"])

    sections = []
    for idx, section in enumerate(enriched.get("sections", [])):
        section_copy = dict(section)
        has_speaker = bool(section_copy.get("speaker_soundbites"))
        has_visual = bool(section_copy.get("archive_visual_fragments"))

        section_copy.setdefault("director_role", role_sequence[idx] if idx < len(role_sequence) else "related_theme_clip")
        section_copy.setdefault("evidence_role", "exact" if has_speaker else ("related" if has_visual else "low_evidence"))
        section_copy.setdefault(
            "evidence_confidence",
            "high" if has_speaker else ("medium" if has_visual else "low"),
        )
        section_copy.setdefault("fallback_allowed", section_copy.get("evidence_role") != "exact")
        section_copy.setdefault(
            "planning_note",
            "Used as related supporting evidence because exact prompt clips were limited." if section_copy.get("evidence_role") != "exact" else "Anchored by an exact or near-exact evidence clip.",
        )
        if mode == "chronological_relay":
            for frag_list_key in ("speaker_soundbites", "archive_visual_fragments"):
                frags = section_copy.get(frag_list_key)
                if frags:
                    section_copy[frag_list_key] = sorted(
                        frags,
                        key=lambda f: (f.get("video_year") is None, f.get("video_year") or 0),
                    )
            all_years = [
                f.get("video_year")
                for key in ("speaker_soundbites", "archive_visual_fragments")
                for f in (section_copy.get(key) or [])
                if f.get("video_year") is not None
            ]
            if all_years:
                sorted_years = sorted(all_years)
                mid = len(sorted_years) // 2
                section_copy["year_slot"] = sorted_years[mid]
            else:
                section_copy["year_slot"] = None

        sections.append(section_copy)

    enriched["sections"] = sections
    enriched["planning_summary"] = {
        "video_mode": director_plan.get("video_mode"),
        "evidence_strength": director_plan.get("evidence_strength"),
        "exact_clip_count": director_plan.get("exact_clip_count"),
        "related_clip_count": director_plan.get("related_clip_count"),
        "strong_related_count": director_plan.get("strong_related_count"),
        "usable_speaker_clip_count": director_plan.get("usable_speaker_clip_count"),
        "rejected_clip_count": director_plan.get("rejected_clip_count"),
        "fallback_risk": director_plan.get("fallback_risk"),
        "planning_notes": director_plan.get("planning_notes", []),
    }
    return enriched


def ensure_storyline_export_timeline(
    storyline: dict[str, Any],
    selected_candidates: list[dict[str, Any]],
    prompt_profile: dict[str, Any],
    target_duration_seconds: int,
) -> dict[str, Any]:
    enriched = dict(storyline)
    timeline = list(enriched.get("export_timeline") or [])
    if not timeline and selected_candidates:
        timeline = build_export_timeline_from_candidates(
            selected_candidates=selected_candidates,
            prompt_profile=prompt_profile,
            target_duration_seconds=target_duration_seconds,
            title=enriched.get("title"),
        )
    enriched["export_timeline"] = timeline
    return enriched


PHASE4O_MOMENT_FIELDS = {
    "start", "end", "duration",
    "moment_text", "moment_selection_reason", "moment_scores", "speech_quality",
    "muted", "subtitle_kind",
}


def timeline_requires_moment_repair(export_timeline: list[dict]) -> bool:
    """Return True if any fragment-linked item is missing Phase 4O moment fields."""
    for item in export_timeline:
        fid = item.get("fragment_id")
        if not fid:
            continue
        # Null / missing type
        if not item.get("type") and not item.get("timeline_type"):
            return True
        # timeline_type present but type missing (raw LLM item)
        if item.get("timeline_type") and not item.get("type"):
            return True
        # Null start/end
        if item.get("start") is None or item.get("end") is None:
            return True
        # Any Phase 4O field missing
        for field in PHASE4O_MOMENT_FIELDS:
            if field not in item:
                return True
    return False


def repair_or_rebuild_timeline_with_moments(
    storyline: dict[str, Any],
    selected_candidates: list[dict[str, Any]],
    prompt_profile: dict[str, Any],
    target_duration_seconds: int,
    video_mode: str | None = None,
) -> dict[str, Any]:
    """
    Ensure every fragment-linked timeline item has Phase 4O moment fields.

    If the existing export_timeline is empty or contains raw LLM items that
    lack moment fields, rebuild it from scratch using `selected_candidates`
    via the Phase 4O moment selector.

    Title/closing cards from any existing timeline are preserved.
    """
    result = dict(storyline)
    existing = list(result.get("export_timeline") or [])

    if not existing:
        needs_rebuild = True
        raw_count = 0
    else:
        needs_rebuild = timeline_requires_moment_repair(existing)
        raw_count = len(existing)

    if not needs_rebuild:
        result["timeline_repair"] = {"required": False}
        return result

    log(f"Timeline requires Phase 4O moment repair (raw_items={raw_count}). Rebuilding from selected_candidates.")

    # Preserve existing title/closing card entries
    card_items = [
        item for item in existing
        if item.get("type") in ("title_card", "closing_card")
        or item.get("timeline_type") in ("title_card", "closing_card")
    ]

    rebuilt = build_export_timeline_from_candidates(
        selected_candidates=selected_candidates,
        prompt_profile=prompt_profile,
        target_duration_seconds=target_duration_seconds,
        title=result.get("title"),
        video_mode=video_mode,
    )

    # If rebuild produced title/closing cards, don't double-add from the existing timeline
    rebuilt_has_title = any(item.get("type") == "title_card" for item in rebuilt)
    rebuilt_has_closing = any(item.get("type") == "closing_card" for item in rebuilt)

    final: list[dict[str, Any]] = []
    if not rebuilt_has_title:
        title_cards = [item for item in card_items if item.get("type") == "title_card" or item.get("timeline_type") == "title_card"]
        final.extend(title_cards)

    final.extend(rebuilt)

    if not rebuilt_has_closing:
        closing_cards = [item for item in card_items if item.get("type") == "closing_card" or item.get("timeline_type") == "closing_card"]
        final.extend(closing_cards)

    result["export_timeline"] = final
    result["timeline_repair"] = {
        "required": True,
        "reason": "llm_timeline_missing_phase4o_moment_fields",
        "raw_timeline_items": raw_count,
        "rebuilt_timeline_items": len(final),
    }
    log(f"Timeline rebuilt: {raw_count} raw items → {len(final)} Phase 4O items.")
    return result


def detect_internal_pipeline_language(text: str) -> list[str]:
    """Detect internal pipeline/system terms in viewer-facing text."""
    issues = []
    if not text:
        return issues
    normalized = text.lower()
    for term in INTERNAL_PIPELINE_TERMS:
        if term in normalized:
            issues.append(f"internal_pipeline_language_detected: '{term}'")
    return issues


def repair_internal_pipeline_language(text: str) -> str:
    """Remove internal pipeline terms from viewer-facing text."""
    if not text:
        return text
    # Ordered list: longer / more-specific phrases first so they are replaced before
    # their sub-terms would be reached by a later pattern.
    replacements = [
        (r"\bconfirmed archive clips?\b", "recordings"),
        (r"\barchive fragments?\b", "recordings"),
        (r"\bselected clips?\b", "speakers"),
        (r"\bevidence packages?\b", "evidence"),
        (r"\bevidence package\b", "evidence"),
        (r"\bcandidate buckets?\b", "clips"),
        (r"\bvalidation issues?\b", "notes"),
        (r"\bbackground tasks?\b", ""),
        (r"\bqa report\b", ""),
        # Pipeline-specific verbs/nouns that are context-safe to remove
        (r"\bpipeline\b", ""),
        (r"\bfallback\b", ""),
        (r"\bllm\b", ""),
        (r"\bjson\b", ""),
        (r"\bmanifest\b", ""),
        (r"\bretrieval\b", "research"),
        # "candidate" used as a noun phrase for clips — only replace when clearly pipeline-context
        # (avoid replacing e.g. "candidate countries" in legitimate text)
        (r"\bcandidate clips?\b", "recordings"),
        (r"\bcandidate fragments?\b", "recordings"),
        (r"\bcandidate bucket\b", "clips"),
    ]
    result = text
    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    result = re.sub(r"  +", " ", result).strip()
    return result


VIEWER_FACING_FIELDS = {
    "title", "logline", "narration", "voiceover_text", "subtitle_text",
    "closing_meaning", "central_question", "narrative_thesis",
}

VIEWER_FACING_SECTION_FIELDS = {
    "title", "narration", "viewer_narration",
}


def validate_viewer_facing_text(storyline_or_manifest: dict) -> list[str]:
    """Validate all viewer-facing text fields for internal pipeline language."""
    issues = []

    def check_text(text: str, field_path: str) -> None:
        found = detect_internal_pipeline_language(text)
        for f in found:
            issues.append(f"{f} (field: {field_path})")

    for field in VIEWER_FACING_FIELDS:
        val = storyline_or_manifest.get(field)
        if isinstance(val, str):
            check_text(val, field)

    for section in (storyline_or_manifest.get("sections") or []):
        sid = section.get("section_id", "?")
        for field in VIEWER_FACING_SECTION_FIELDS:
            val = section.get(field)
            if isinstance(val, str):
                check_text(val, f"section[{sid}].{field}")

    for item in (storyline_or_manifest.get("voiceover_script") or []):
        val = item.get("text")
        if isinstance(val, str):
            check_text(val, f"voiceover_script[{item.get('section_id', '?')}].text")

    return issues


def _repair_text_field(text: str) -> tuple[str, bool]:
    """Repair a single viewer-facing text field. Returns (repaired_text, was_changed)."""
    if not text:
        return text, False
    repaired = repair_internal_pipeline_language(text)
    return repaired, repaired != text


def sanitize_viewer_facing_storyline_text(
    storyline: dict,
    prompt_profile: dict,
) -> tuple[dict, list[str]]:
    """
    Sanitize all viewer-facing text fields in a storyline.

    Applies repair_internal_pipeline_language to every field a viewer could see.
    Returns (sanitized_storyline, list_of_repair_notes).
    Internal diagnostic fields (evidence_fragment_ids, selection_reason, etc.) are untouched.
    """
    result = dict(storyline)
    repair_notes: list[str] = []

    # Top-level viewer-facing string fields
    top_level_fields = [
        "title", "logline", "closing_meaning", "central_question", "narrative_thesis",
    ]
    for field in top_level_fields:
        val = result.get(field)
        if isinstance(val, str):
            repaired, changed = _repair_text_field(val)
            if changed:
                result[field] = repaired
                repair_notes.append(f"repaired:{field}")

    # voiceover_script list
    if isinstance(result.get("voiceover_script"), list):
        repaired_vos = []
        for item in result["voiceover_script"]:
            item = dict(item)
            for field in ("text", "subtitle_text"):
                val = item.get(field)
                if isinstance(val, str):
                    repaired, changed = _repair_text_field(val)
                    if changed:
                        item[field] = repaired
                        repair_notes.append(f"repaired:voiceover_script[{item.get('section_id','?')}].{field}")
            repaired_vos.append(item)
        result["voiceover_script"] = repaired_vos

    # sections[]
    if isinstance(result.get("sections"), list):
        repaired_sections = []
        for section in result["sections"]:
            section = dict(section)
            for field in ("title", "narration", "viewer_narration", "subtitle_text", "voiceover_text"):
                val = section.get(field)
                if isinstance(val, str):
                    repaired, changed = _repair_text_field(val)
                    if changed:
                        section[field] = repaired
                        repair_notes.append(f"repaired:section[{section.get('section_id','?')}].{field}")
            repaired_sections.append(section)
        result["sections"] = repaired_sections

    # export_timeline[]
    if isinstance(result.get("export_timeline"), list):
        repaired_timeline = []
        for item in result["export_timeline"]:
            item = dict(item)
            for field in ("title", "subtitle_text", "voiceover_text"):
                val = item.get(field)
                if isinstance(val, str):
                    repaired, changed = _repair_text_field(val)
                    if changed:
                        item[field] = repaired
                        repair_notes.append(f"repaired:export_timeline[{item.get('fragment_id','?')}].{field}")
            repaired_timeline.append(item)
        result["export_timeline"] = repaired_timeline

    # Post-repair: check if any viewer-facing issues remain
    remaining = validate_viewer_facing_text(result)
    unrepairable = [r for r in remaining if "internal_pipeline_language_detected" in r]
    if unrepairable:
        repair_notes.extend(
            f"viewer_facing_text_unrepairable: {issue}" for issue in unrepairable
        )

    return result, repair_notes


def find_best_subclip_window(
    candidate: dict,
    prompt_profile: dict,
    desired_duration: float = 13.0,
) -> dict:
    """Find the best subclip window avoiding side-topic text, preferring prompt-term density."""
    start = float(candidate.get("start") or 0.0)
    end = float(candidate.get("end") or start + desired_duration)
    total_duration = end - start

    text = (candidate.get("text") or candidate.get("summary") or "").lower()
    if not text:
        return {"start": start, "end": end, "adjusted": False}

    topic_terms = set(prompt_profile.get("topic_terms") or [])
    expanded_terms = set(prompt_profile.get("expanded_terms") or [])
    all_prompt_terms = topic_terms | expanded_terms

    side_terms_lower = [t.lower() for t in SIDE_TOPIC_TERMS]
    has_side_topic = any(term in text for term in side_terms_lower)
    has_prompt_terms = any(term in text for term in all_prompt_terms)

    if not has_side_topic:
        return {"start": start, "end": end, "adjusted": False}

    words = text.split()
    total_words = len(words)
    if total_words < 10:
        return {"start": start, "end": end, "adjusted": False}

    first_40pct = " ".join(words[: int(total_words * 0.4)])
    latter_60pct = " ".join(words[int(total_words * 0.4):])

    side_in_first = any(term in first_40pct for term in side_terms_lower)
    prompt_in_latter = any(term in latter_60pct for term in all_prompt_terms)

    if side_in_first and prompt_in_latter and has_prompt_terms:
        offset = total_duration * 0.35
        new_start = round(start + offset, 2)
        new_end = round(new_start + desired_duration, 2)
        if new_end > end:
            new_end = end
        return {"start": new_start, "end": new_end, "adjusted": True, "reason": "side_topic_in_first_40pct_shifted_forward"}

    return {"start": start, "end": end, "adjusted": False}


INTRO_ADMIN_TERMS = [
    "hello everyone",
    "good morning",
    "good afternoon",
    "good evening",
    "thank you",
    "thanks for inviting me",
    "let me introduce",
    "next slide",
    "moving on",
    "today i will",
    "my name is",
    "it is an honor",
    "it is a pleasure",
    "housekeeping",
    "restroom",
    "closed captioning",
    "can you hear me",
    "sorry about",
    "welcome remarks",
    "i am delighted",
    "i am honored",
    "please welcome",
    "round of applause",
    "let me start by",
    "before i begin",
    "on behalf of",
]

SECTION_ROLE_KEYWORD_MAP: dict[str, dict[str, list[str]]] = {
    "deepfake": {
        "context": ["synthetic media", "fake video", "definition", "generated content", "digital manipulation", "what is"],
        "problem_or_tension": ["threat", "misinformation", "false", "fraud", "harm", "dangerous", "deceptive", "disinformation"],
        "human_perspective": ["ordinary", "women", "victims", "people", "families", "individuals", "targeted"],
        "expert_evidence": ["researcher", "expert", "study", "evidence", "data", "found", "shows", "analysis"],
        "solution_direction": ["regulation", "transparency", "platform", "law", "detect", "authenticate", "verify", "watermark"],
        "final_synthesis": ["trust", "democracy", "information integrity", "future", "society", "responsibility"],
    },
    "healthcare": {
        "context": ["diagnosis", "clinical", "care", "doctors", "treatment", "medical", "hospital"],
        "human_perspective": ["patients", "families", "access", "communities", "people living", "everyday"],
        "problem_or_tension": ["trust", "safety", "regulation", "risk", "overloaded", "waste", "burden"],
        "expert_evidence": ["researchers", "findings", "study", "evidence", "speakers", "data shows"],
        "solution_direction": ["governance", "ethics", "consent", "transparency", "accountability", "guidelines"],
        "final_synthesis": ["future", "measure", "human", "safer", "fairer", "dignity"],
    },
    "women": {
        "context": ["gender", "systems", "designed", "discrimination", "barriers"],
        "human_perspective": ["agency", "voice", "decision", "women", "girls", "lived experience"],
        "problem_or_tension": ["bias", "exclusion", "underrepresented", "inequality", "lack of"],
        "solution_direction": ["inclusion", "design", "participation", "representation", "leadership"],
        "final_synthesis": ["progress", "power", "real agency", "measure of", "change"],
    },
    "governance": {
        "context": ["rules", "policy", "institutions", "systems", "framework"],
        "problem_or_tension": ["accountability", "oversight", "challenge", "risk", "who decides"],
        "solution_direction": ["regulation", "rights", "transparency", "law", "participate"],
        "final_synthesis": ["responsibility", "future", "accountability", "protection"],
    },
    "crypto": {
        "context": ["blockchain", "cryptocurrency", "digital currency", "token", "web3"],
        "problem_or_tension": ["fraud", "scam", "volatile", "risk", "loss", "unregulated"],
        "solution_direction": ["consumer protection", "regulation", "transparency", "safety"],
        "final_synthesis": ["open financial", "trust", "future", "promise", "real benefit"],
    },
}


def _window_intro_penalty(window_text: str) -> float:
    """Return a penalty score (0–1) based on how much intro/admin language is in the window."""
    if not window_text:
        return 0.0
    normalized = window_text.lower()
    word_count = max(1, len(normalized.split()))
    intro_count = sum(1 for term in INTRO_ADMIN_TERMS if term in normalized)
    if intro_count == 0:
        return 0.0
    return min(1.0, intro_count * 0.35)


def _window_side_topic_penalty(window_text: str) -> float:
    """Return a penalty score based on side-topic term density."""
    if not window_text:
        return 0.0
    normalized = window_text.lower()
    count = sum(1 for term in SIDE_TOPIC_TERMS if term.lower() in normalized)
    return min(1.0, count * 0.25)


def _window_incomplete_sentence_penalty(window_text: str) -> float:
    """Penalise text that starts or ends mid-sentence."""
    if not window_text:
        return 0.0
    stripped = window_text.strip()
    penalty = 0.0
    # Starts mid-sentence: lowercase and not punctuated opener
    first_char = stripped[0] if stripped else ""
    if first_char.islower():
        penalty += 0.15
    # Ends without terminal punctuation
    last_char = stripped[-1] if stripped else ""
    if last_char not in ".!?\"'":
        penalty += 0.1
    return min(0.35, penalty)


def _window_prompt_score(window_text: str, prompt_profile: dict) -> float:
    """Score how much prompt-relevant content is in this window."""
    if not window_text:
        return 0.0
    normalized = window_text.lower()
    topic_terms = set(prompt_profile.get("topic_terms") or [])
    expanded_terms = set(prompt_profile.get("expanded_terms") or [])
    all_terms = topic_terms | expanded_terms
    hits = sum(1 for t in all_terms if t.lower() in normalized)
    return min(1.0, hits * 0.2)


def _window_section_fit_score(window_text: str, section_role: str | None, prompt_profile: dict) -> float:
    """Score how well this window fits a given section role."""
    if not window_text or not section_role:
        return 0.0
    normalized = window_text.lower()
    topic_terms = prompt_profile.get("topic_terms") or []
    primary_topic = topic_terms[0] if topic_terms else None

    role_map = SECTION_ROLE_KEYWORD_MAP.get(primary_topic or "", {})
    if not role_map:
        for topic in topic_terms:
            role_map = SECTION_ROLE_KEYWORD_MAP.get(topic, {})
            if role_map:
                break

    role_keywords = role_map.get(section_role, [])
    if not role_keywords:
        return 0.0

    hits = sum(1 for kw in role_keywords if kw.lower() in normalized)
    return min(1.0, hits * 0.3)


WEAK_MOMENT_STARTS = [
    "because",
    "and of course",
    "so if",
    "it's very interesting",
    "it is very interesting",
    "really heard today",
    "as i told you",
    "you know",
    "moving on",
    "next slide",
    "hello everyone",
    "good morning",
    "good afternoon",
    "good evening",
    "thank you",
    "welcome",
    "in conclusion",
    "and so",
    "so basically",
    "as i was saying",
    "right so",
    "okay so",
    "yeah so",
    "but anyway",
]


def is_meaningful_documentary_moment(
    moment_text: str,
    prompt_profile: dict,
    moment_scores: dict | None = None,
) -> tuple[bool, list[str]]:
    """
    Return (is_meaningful, reasons_for_rejection).

    A moment is meaningful if it starts cleanly, contains topic content,
    forms a complete thought, and scores above minimum thresholds.
    """
    reasons: list[str] = []
    if not moment_text or not moment_text.strip():
        reasons.append("empty_moment_text")
        return False, reasons

    stripped = moment_text.strip()
    normalized_lower = stripped.lower()

    # Starts with a weak phrase
    for phrase in WEAK_MOMENT_STARTS:
        if normalized_lower.startswith(phrase.lower()):
            reasons.append(f"starts_with_weak_phrase: {phrase!r}")
            break

    # Starts lowercase (mid-sentence) without being a proper noun opener
    first_char = stripped[0] if stripped else ""
    if first_char.islower():
        reasons.append("starts_lowercase_mid_sentence")

    # Contains intro/admin terms
    admin_hit = sum(1 for term in INTRO_ADMIN_TERMS if term in normalized_lower)
    if admin_hit >= 1:
        reasons.append(f"contains_admin_terms: {admin_hit}")

    # Fewer than 8 meaningful words (stop-words excluded)
    STOP_WORDS = {"a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "to", "and", "or", "but", "it", "this", "that", "i", "we", "you", "they", "he", "she", "so"}
    words = [w.strip(".,!?\"'") for w in stripped.split()]
    meaningful_words = [w for w in words if w.lower() not in STOP_WORDS and len(w) > 1]
    if len(meaningful_words) < 8:
        reasons.append(f"too_few_meaningful_words: {len(meaningful_words)}")

    # Does not contain topic/context terms
    topic_terms = set(prompt_profile.get("topic_terms") or [])
    expanded_terms = set(prompt_profile.get("expanded_terms") or [])
    all_terms = topic_terms | expanded_terms
    topic_hit = sum(1 for t in all_terms if t.lower() in normalized_lower)
    if all_terms and topic_hit == 0:
        reasons.append("no_topic_terms_found")

    # Ends mid-thought (no terminal punctuation)
    last_char = stripped[-1] if stripped else ""
    if last_char not in ".!?\"'":
        reasons.append("ends_without_terminal_punctuation")

    # Score thresholds
    if moment_scores:
        final_score = moment_scores.get("final_moment_score") or 0.0
        completeness = moment_scores.get("quote_completeness_score") or 0.0
        if float(final_score) < 0.35:
            reasons.append(f"final_moment_score_too_low: {final_score:.3f}")
        if float(completeness) < 0.55:
            reasons.append(f"quote_completeness_too_low: {completeness:.3f}")

    # Hard rejections: truly disqualifying — empty text, admin/greetings, not enough words
    # starts_lowercase is SOFT (conference transcripts often start mid-sentence in windows)
    hard_reasons = [r for r in reasons if any(r.startswith(p) for p in (
        "empty_", "contains_admin_terms", "too_few_meaningful_words",
    ))]

    # starts_with_weak_phrase is also a hard rejection when it matches admin/filler starts
    hard_weak = [r for r in reasons if r.startswith("starts_with_weak_phrase")]
    # Only treat weak_phrase start as hard if the phrase is truly admin (no topic terms present)
    if hard_weak and topic_hit > 0:
        hard_weak = []  # topic terms rescue a weak start

    is_meaningful = len(hard_reasons) == 0 and len(hard_weak) == 0
    return is_meaningful, reasons


def decide_clip_audio_role(
    moment: dict,
    candidate: dict,
    prompt_profile: dict,
) -> dict:
    """
    Decide the audio role for a selected moment window.

    Returns a dict with:
      segment_type: speaker_clip | narration_broll | archive_visual
      audio_mode:   source_audio | ai_voiceover | silent
      muted:        bool
      subtitle_kind: speaker | narration | none
      reason:       str
    """
    moment_text = moment.get("window_text") or moment.get("moment_text") or ""
    scores = {
        "final_moment_score": moment.get("final_moment_score") or 0.0,
        "quote_completeness_score": moment.get("quote_completeness_score") or 0.0,
    }
    speech = moment.get("speech_quality") or {}
    is_usable_speech = speech.get("is_usable_speech", True)
    intro_penalty = float(moment.get("intro_penalty") or 0.0)
    final_score = float(moment.get("final_moment_score") or 0.0)
    completeness = float(moment.get("quote_completeness_score") or 0.0)
    duration = float(moment.get("window_duration") or moment.get("duration") or 0.0)

    # Check meaningful moment (relaxed for Phase 4Q)
    is_meaningful, meaning_reasons = is_meaningful_documentary_moment(
        moment_text=moment_text,
        prompt_profile=prompt_profile,
        moment_scores=scores,
    )

    speaker_annotation_allowed = bool(
        candidate.get("speaker_soundbite_allowed")
        or candidate.get("speaker_clip_allowed")
    )

    # Phase 4Q: relax the speaker_allowed gate.
    # The original annotation sets speaker_clip_allowed=False when quote_strength < 0.65,
    # but conference clips at a topic-relevant event with clarity >= 0.75 and good moment
    # scores should still be eligible as speaker clips.
    # Override: allow as speaker_clip when moment score is meaningful even if the candidate
    # annotation gate didn't pre-approve it.
    clarity = float(candidate.get("clarity_score") or 0.0)
    final_video_allowed = candidate.get("final_video_allowed", True)
    speaker_allowed = (
        speaker_annotation_allowed
        or (
            final_video_allowed
            and clarity >= 0.70         # reasonable speech quality
            and final_score >= 0.20     # moment has content
            and is_meaningful           # not admin/filler
        )
    )

    # If the candidate annotation explicitly approved this as a speaker soundbite,
    # treat is_usable_speech as advisory rather than a hard gate — the heuristic
    # can be wrong (e.g. long fragments with correct window durations fail the old
    # duration check and get marked unusable even when the audio is fine).
    annotation_approved = bool(
        candidate.get("speaker_soundbite_allowed")
        or candidate.get("speaker_clip_allowed")
    )
    effective_usable_speech = is_usable_speech or annotation_approved

    # Phase 4Q speaker_clip conditions (relaxed from 4O):
    # - meaningful content (not admin/filler)
    # - usable speech quality (or annotation override)
    # - low intro penalty
    # - minimum meaningful score
    # - reasonable duration
    is_speaker_clip = (
        is_meaningful
        and effective_usable_speech
        and intro_penalty < 0.20
        and final_score >= 0.20        # lowered from 0.45 — conference transcripts score lower
        and completeness >= 0.35       # lowered from 0.65 — partial credit for mid-window starts
        and 5.0 <= duration <= 20.0
        and speaker_allowed
    )

    if is_speaker_clip:
        return {
            "segment_type": "speaker_clip",
            "audio_mode": "source_audio",
            "muted": False,
            "subtitle_kind": "speaker",
            "reason": f"meaningful_speaker_quote(score={final_score:.2f},completeness={completeness:.2f})",
        }

    # Not speaker quality — use as muted visual b-roll
    return {
        "segment_type": "archive_visual",
        "audio_mode": "silent",
        "muted": True,
        "subtitle_kind": "none",
        "reason": f"archive_visual(is_meaningful={is_meaningful},score={final_score:.2f},completeness={completeness:.2f},reasons={meaning_reasons[:2]})",
    }


def build_candidate_moment_windows(
    candidate: dict,
    prompt_profile: dict,
    window_seconds: float = 12.0,
    stride_seconds: float = 4.0,
) -> list[dict]:
    """
    Slide a window across a candidate fragment's transcript to find distinct
    moment windows. Returns a list of window dicts, each describing a time
    subrange and the text it covers.
    """
    fragment_id = candidate.get("fragment_id", "")
    source_video = candidate.get("source_video", "")
    start = float(candidate.get("start") or 0.0)
    end = float(candidate.get("end") or start + window_seconds)
    total_duration = max(0.1, end - start)

    text = (candidate.get("text") or candidate.get("summary") or "").strip()
    topic_terms = list((prompt_profile.get("topic_terms") or []) + (prompt_profile.get("expanded_terms") or []))

    windows: list[dict] = []

    if not text or total_duration < 4.0:
        # Single window covering the whole fragment
        windows.append({
            "fragment_id": fragment_id,
            "source_video": source_video,
            "window_start": round(start, 3),
            "window_end": round(end, 3),
            "window_duration": round(total_duration, 3),
            "window_text": text,
            "window_summary": text[:200] if text else "",
            "prompt_terms": [t for t in topic_terms if t.lower() in text.lower()],
            "section_role_candidates": [],
        })
        return windows

    words = text.split()
    total_words = len(words)
    seconds_per_word = total_duration / max(1, total_words)
    words_per_window = max(1, int(window_seconds / seconds_per_word))
    words_per_stride = max(1, int(stride_seconds / seconds_per_word))

    pos = 0
    while pos < total_words:
        w_end = min(pos + words_per_window, total_words)
        window_words = words[pos:w_end]
        window_text = " ".join(window_words)

        word_start_time = start + pos * seconds_per_word
        word_end_time = start + w_end * seconds_per_word
        word_end_time = min(word_end_time, end)

        win_duration = max(1.0, word_end_time - word_start_time)
        matched_terms = [t for t in topic_terms if t.lower() in window_text.lower()]

        windows.append({
            "fragment_id": fragment_id,
            "source_video": source_video,
            "window_start": round(word_start_time, 3),
            "window_end": round(word_end_time, 3),
            "window_duration": round(win_duration, 3),
            "window_text": window_text,
            "window_summary": window_text[:200],
            "prompt_terms": matched_terms,
            "section_role_candidates": [],
        })

        pos += words_per_stride
        if w_end >= total_words:
            break

    if not windows:
        windows.append({
            "fragment_id": fragment_id,
            "source_video": source_video,
            "window_start": round(start, 3),
            "window_end": round(end, 3),
            "window_duration": round(total_duration, 3),
            "window_text": text,
            "window_summary": text[:200],
            "prompt_terms": [t for t in topic_terms if t.lower() in text.lower()],
            "section_role_candidates": [],
        })

    return windows


def score_moment_window(
    window: dict,
    candidate: dict,
    prompt_profile: dict,
    section_role: str | None = None,
) -> dict:
    """
    Score a moment window for use as a speaker clip or archive visual.
    Returns a copy of `window` with scoring fields added.
    """
    scored = dict(window)
    wtext = window.get("window_text", "")

    moment_prompt_score = _window_prompt_score(wtext, prompt_profile)
    section_fit_score = _window_section_fit_score(wtext, section_role, prompt_profile)

    # Quote completeness: does it start and end like a complete thought?
    # Phase 4Q: partial credit when window starts mid-sentence but ends cleanly,
    # or when the candidate has high quote_strength (the source material is quotable).
    stripped = wtext.strip()
    starts_clean = stripped[:1].isupper() if stripped else False
    ends_clean = stripped[-1:] in ".!?\"'" if stripped else False
    candidate_quote_strength = float(candidate.get("quote_strength") or 0.5)
    if starts_clean and ends_clean:
        quote_completeness_score = 1.0
    elif ends_clean:
        # ends cleanly but starts mid-sentence: partial credit, boosted by quote_strength
        quote_completeness_score = 0.45 + 0.25 * candidate_quote_strength
    elif starts_clean:
        # starts cleanly but no terminal punctuation (window cut mid-sentence)
        quote_completeness_score = 0.35 + 0.15 * candidate_quote_strength
    else:
        # mid-sentence both ends: low but non-zero when quote_strength is high
        quote_completeness_score = 0.15 + 0.20 * candidate_quote_strength
    quote_completeness_score = round(min(1.0, quote_completeness_score), 4)

    # Speech quality: proxy from candidate metadata
    quote_strength = float(candidate.get("quote_strength") or 0.5)
    clarity_score = float(candidate.get("clarity_score") or 0.5)
    speech_quality_score = min(1.0, (quote_strength * 0.6) + (clarity_score * 0.4))

    # Documentary value: topic density and sentence quality
    word_count = len(wtext.split()) if wtext else 0
    density_score = min(1.0, len(window.get("prompt_terms", [])) * 0.2)
    length_score = min(1.0, word_count / 40.0)  # ~40 words ideal
    documentary_value_score = min(1.0, density_score * 0.6 + length_score * 0.4)

    # Penalties
    intro_penalty = _window_intro_penalty(wtext)
    silence_penalty = 0.0  # populated later by estimate_speech_presence if available
    side_topic_penalty = _window_side_topic_penalty(wtext)
    incomplete_sentence_penalty = _window_incomplete_sentence_penalty(wtext)

    final_moment_score = (
        0.30 * moment_prompt_score
        + 0.20 * section_fit_score
        + 0.20 * quote_completeness_score
        + 0.15 * speech_quality_score
        + 0.15 * documentary_value_score
        - intro_penalty
        - silence_penalty
        - side_topic_penalty
        - incomplete_sentence_penalty
    )
    final_moment_score = max(0.0, min(1.0, final_moment_score))

    # Build selection reason
    reasons = []
    if moment_prompt_score >= 0.3:
        reasons.append(f"topic-dense({moment_prompt_score:.2f})")
    if section_fit_score >= 0.3:
        reasons.append(f"role-fit({section_fit_score:.2f})")
    if quote_completeness_score >= 0.8:
        reasons.append("complete-quote")
    if intro_penalty >= 0.35:
        reasons.append("intro-language-penalty")
    if side_topic_penalty >= 0.25:
        reasons.append("side-topic-penalty")
    if not reasons:
        reasons.append(f"general-score({final_moment_score:.2f})")

    scored.update({
        "moment_prompt_score": round(moment_prompt_score, 4),
        "quote_completeness_score": round(quote_completeness_score, 4),
        "speech_quality_score": round(speech_quality_score, 4),
        "section_fit_score": round(section_fit_score, 4),
        "documentary_value_score": round(documentary_value_score, 4),
        "intro_penalty": round(intro_penalty, 4),
        "silence_penalty": round(silence_penalty, 4),
        "side_topic_penalty": round(side_topic_penalty, 4),
        "incomplete_sentence_penalty": round(incomplete_sentence_penalty, 4),
        "final_moment_score": round(final_moment_score, 4),
        "moment_selection_reason": ", ".join(reasons),
    })
    return scored


def infer_section_role_for_moment(window: dict, prompt_profile: dict) -> list[str]:
    """
    Return the list of section roles this moment window fits best.
    Used to suggest where in the documentary arc a clip belongs.
    """
    wtext = (window.get("window_text") or "").lower()
    topic_terms = prompt_profile.get("topic_terms") or []
    roles = []

    for topic in topic_terms:
        role_map = SECTION_ROLE_KEYWORD_MAP.get(topic, {})
        for role, keywords in role_map.items():
            if any(kw.lower() in wtext for kw in keywords):
                if role not in roles:
                    roles.append(role)

    # Fallback: intro/admin → not suitable for any speaking role
    if _window_intro_penalty(wtext) >= 0.35:
        return []

    if not roles:
        # Generic fallbacks
        if any(t in wtext for t in ["therefore", "conclusion", "in summary", "ultimately", "this means"]):
            roles.append("final_synthesis")
        elif any(t in wtext for t in ["however", "but", "challenge", "problem", "concern", "risk"]):
            roles.append("problem_or_tension")
        elif any(t in wtext for t in ["solution", "approach", "recommend", "propose", "should"]):
            roles.append("solution_direction")
        else:
            roles.append("expert_evidence")

    return roles


def estimate_speech_presence(candidate: dict, start: float, end: float) -> dict:
    """
    Estimate speech quality for a clip window.

    Since we don't have direct audio analysis here (that requires ffmpeg/librosa
    in the video export step), we use proxy signals from metadata:
    - quote_strength as speech density proxy
    - clarity_score as quality proxy
    - fragment text word count vs duration as speech ratio proxy

    Returns a dict compatible with the Phase 4O speech_quality field.
    """
    duration = max(0.1, end - start)
    text = (candidate.get("text") or candidate.get("summary") or "").strip()
    total_frag_start = float(candidate.get("start") or 0.0)
    total_frag_end = float(candidate.get("end") or total_frag_start + duration)
    total_frag_duration = max(0.1, total_frag_end - total_frag_start)

    # Proportional word count for this window
    total_words = len(text.split()) if text else 0
    fraction_of_frag = min(1.0, duration / total_frag_duration)
    estimated_words = int(total_words * fraction_of_frag)

    # Words-per-minute proxy: typical speech is 120-160 wpm
    words_per_minute = (estimated_words / duration) * 60 if duration > 0 else 0
    speech_ratio = min(1.0, words_per_minute / 150.0)  # 150 wpm = full speech

    quote_strength = float(candidate.get("quote_strength") or 0.5)
    clarity_score = float(candidate.get("clarity_score") or 0.5)

    # Weight: speech_ratio is proxy; also factor in metadata quality signals
    combined_ratio = min(1.0, (speech_ratio * 0.5) + (quote_strength * 0.3) + (clarity_score * 0.2))
    silence_ratio = max(0.0, 1.0 - combined_ratio)
    mean_volume = -18.0 - (1.0 - combined_ratio) * 12  # rough proxy in dBFS

    # Duration constraint is applied at the is_speaker_clip level (5.0 <= window_duration <= 20.0),
    # not here — the fragment duration can be long; we're evaluating the window we selected from it.
    is_usable = (
        combined_ratio >= 0.55
        and silence_ratio <= 0.35
    )

    return {
        "speech_ratio": round(combined_ratio, 3),
        "mean_volume": round(mean_volume, 1),
        "silence_ratio": round(silence_ratio, 3),
        "is_usable_speech": is_usable,
        "estimated_words": estimated_words,
        "duration": round(duration, 3),
    }


def select_best_moment_for_section(
    section: dict,
    candidates: list[dict],
    prompt_profile: dict,
    used_fragment_ids: set[str],
    section_role: str | None = None,
    window_seconds: float = 12.0,
) -> dict | None:
    """
    Select the best moment window for a story section from the available candidates.

    Returns a moment dict with all Phase 4O fields, or None if no usable window found.
    """
    effective_role = section_role or section.get("section_type") or section.get("role")

    best_window = None
    best_score = -999.0

    for candidate in candidates:
        fid = str(candidate.get("fragment_id") or "")
        if not fid or fid in used_fragment_ids:
            continue

        start_c = float(candidate.get("start") or 0.0)
        end_c = float(candidate.get("end") or start_c + window_seconds)
        total_dur = max(0.1, end_c - start_c)

        # Build windows for this candidate
        windows = build_candidate_moment_windows(
            candidate=candidate,
            prompt_profile=prompt_profile,
            window_seconds=min(window_seconds, total_dur),
            stride_seconds=max(2.0, window_seconds * 0.33),
        )

        for window in windows:
            scored = score_moment_window(
                window=window,
                candidate=candidate,
                prompt_profile=prompt_profile,
                section_role=effective_role,
            )
            # Assign section role candidates
            scored["section_role_candidates"] = infer_section_role_for_moment(window, prompt_profile)

            # Compute speech presence
            speech = estimate_speech_presence(
                candidate=candidate,
                start=window["window_start"],
                end=window["window_end"],
            )
            scored["speech_quality"] = speech

            # Update silence_penalty from speech estimate
            if not speech["is_usable_speech"] and speech["silence_ratio"] > 0.35:
                silence_pen = min(0.4, speech["silence_ratio"] * 0.5)
                scored["silence_penalty"] = round(silence_pen, 4)
                scored["final_moment_score"] = round(
                    max(0.0, scored["final_moment_score"] - silence_pen), 4
                )

            if scored["final_moment_score"] > best_score:
                best_score = scored["final_moment_score"]
                best_window = {
                    "candidate": candidate,
                    "window": scored,
                    "speech": speech,
                }

    if best_window is None:
        return None

    candidate = best_window["candidate"]
    window = best_window["window"]
    speech = best_window["speech"]

    # Phase 4Q: use decide_clip_audio_role for speaker vs archive decision
    # Pass speech quality into the window so decide_clip_audio_role can read it
    window["speech_quality"] = speech
    audio_role = decide_clip_audio_role(
        moment=window,
        candidate=candidate,
        prompt_profile=prompt_profile,
    )

    segment_type = audio_role["segment_type"]
    audio_mode = audio_role["audio_mode"]
    muted = audio_role["muted"]
    subtitle_kind = audio_role["subtitle_kind"]
    moment_text = window.get("window_text", "")
    subtitle_text = moment_text if segment_type == "speaker_clip" else ""

    if segment_type != "speaker_clip" and audio_role.get("reason"):
        window["moment_selection_reason"] = (
            window.get("moment_selection_reason", "") + f"; {audio_role['reason']}"
        )

    return {
        "type": segment_type,
        "segment_type": segment_type,
        "audio_mode": audio_mode,
        "fragment_id": candidate.get("fragment_id"),
        "source_video": candidate.get("source_video"),
        "start": window["window_start"],
        "end": window["window_end"],
        "duration": window["window_duration"],
        "moment_text": moment_text,
        "moment_summary": window.get("window_summary", ""),
        "moment_selection_reason": window.get("moment_selection_reason", ""),
        "section_role": effective_role,
        "muted": muted,
        "subtitle_text": subtitle_text,
        "subtitle_kind": subtitle_kind,
        "speech_quality": speech,
        "moment_scores": {
            "moment_prompt_score": window.get("moment_prompt_score"),
            "quote_completeness_score": window.get("quote_completeness_score"),
            "speech_quality_score": window.get("speech_quality_score"),
            "section_fit_score": window.get("section_fit_score"),
            "documentary_value_score": window.get("documentary_value_score"),
            "intro_penalty": window.get("intro_penalty"),
            "silence_penalty": window.get("silence_penalty"),
            "side_topic_penalty": window.get("side_topic_penalty"),
            "incomplete_sentence_penalty": window.get("incomplete_sentence_penalty"),
            "final_moment_score": window.get("final_moment_score"),
        },
        "use_as": candidate.get("use_as"),
        "relevance_role": candidate.get("relevance_role"),
        "selection_reason": candidate.get("selection_reason"),
        "speaker_clip_allowed": bool(candidate.get("speaker_clip_allowed") or candidate.get("speaker_soundbite_allowed")),
        "speaker_visual_allowed": bool(candidate.get("speaker_visual_allowed")),
        "subclip_adjusted": False,
    }


def build_documentary_arc(
    prompt_profile: dict,
    selected_candidates: list[dict],
    target_duration_seconds: int,
) -> dict:
    """Build a National Geographic-style documentary arc from prompt and evidence."""
    topic_terms = prompt_profile.get("topic_terms") or []
    expanded_terms = prompt_profile.get("expanded_terms") or []
    requested_speaker = prompt_profile.get("requested_speaker")

    primary_topic = topic_terms[0] if topic_terms else None
    topic_label = " ".join(topic_terms[:2]).title() if topic_terms else (requested_speaker or "Artificial Intelligence")

    hook = CINEMATIC_HOOKS.get(primary_topic, f"Across the world, conversations about artificial intelligence are raising a central question: what does it mean for technology to serve people?")
    tension = CINEMATIC_TENSIONS.get(primary_topic, "But the promise of AI is not automatic. Every benefit comes with a risk, and every advance leaves questions about who is left behind.")
    closing = CINEMATIC_CLOSINGS.get(primary_topic, f"The measure of AI will not be the speed of its systems, but whether the people it touches are better served, better heard, and more free.")

    if requested_speaker:
        central_question = f"What does {requested_speaker} contribute to our understanding of AI and its human stakes?"
        narrative_thesis = f"Through {requested_speaker}'s archive contributions, this documentary traces key questions in AI and society."
    elif primary_topic:
        central_questions = {
            "healthcare": "How can AI support doctors, patients, and families without weakening trust, safety, or human judgment?",
            "health": "How can AI support doctors, patients, and families without weakening trust, safety, or human judgment?",
            "women": "How can AI expand opportunity for women without reproducing the systems of exclusion it was meant to overcome?",
            "empowerment": "How can AI expand opportunity for women without reproducing the systems of exclusion it was meant to overcome?",
            "governance": "Who writes the rules for AI, who bears the risk, and how can accountability be built into the systems before harm arrives?",
            "youth": "What future are young people inheriting from AI systems, and do they have a say in how those systems are built?",
            "education": "Can AI help close the gap between those who have access to knowledge and those who have been shut out?",
            "crypto": "Can digital currencies deliver a more open financial system, or will the risks arrive before the benefits?",
            "accessibility": "Can AI become a genuine tool for access, or will it deepen the divides it was meant to bridge?",
        }
        central_question = central_questions.get(primary_topic, f"How is AI changing {topic_label}, and who decides whether that change is good?")
        narrative_thesis = f"This documentary draws on evidence from AI4IA speakers to examine the real promise and real risks of AI in {topic_label}."
    else:
        central_question = "How is AI changing the world, and what must change to ensure it serves everyone?"
        narrative_thesis = "This documentary draws on AI4IA conference evidence to examine AI's human stakes."

    fragment_ids_all = [c.get("fragment_id") for c in selected_candidates if c.get("fragment_id")]
    n_candidates = len(selected_candidates)
    sections_target = timeline_segment_count_for_duration(target_duration_seconds)
    chunk_size = max(1, n_candidates // max(1, sections_target - 2))

    sections = []

    hook_ids = fragment_ids_all[:2]
    sections.append({
        "role": "hook",
        "title": f"AI and {topic_label}" if not requested_speaker else f"{requested_speaker}: The Questions",
        "viewer_narration": hook,
        "evidence_fragment_ids": hook_ids,
    })

    context_candidates = selected_candidates[:chunk_size]
    context_ids = [c.get("fragment_id") for c in context_candidates if c.get("fragment_id")]
    context_texts = [truncate_text(c.get("text") or c.get("summary") or "", 200) for c in context_candidates if (c.get("text") or c.get("summary"))]

    if context_texts:
        context_narration = _synthesize_documentary_context(primary_topic, context_texts, topic_label)
    else:
        context_narration = f"Across the AI4IA conversations, {topic_label} emerges as a field where the stakes — for individuals, institutions, and communities — could not be higher."

    sections.append({
        "role": "human_context",
        "title": f"The Human Stakes of {topic_label}" if not requested_speaker else f"{requested_speaker}: Context",
        "viewer_narration": context_narration,
        "evidence_fragment_ids": context_ids,
    })

    tension_candidates = selected_candidates[chunk_size: chunk_size * 2]
    tension_ids = [c.get("fragment_id") for c in tension_candidates if c.get("fragment_id")]
    tension_texts = [truncate_text(c.get("text") or c.get("summary") or "", 200) for c in tension_candidates if (c.get("text") or c.get("summary"))]

    if tension_texts:
        tension_narration = _synthesize_documentary_tension(primary_topic, tension_texts, topic_label)
    else:
        tension_narration = tension

    sections.append({
        "role": "tension",
        "title": f"The Challenge" if not requested_speaker else f"{requested_speaker}: Tensions",
        "viewer_narration": tension_narration,
        "evidence_fragment_ids": tension_ids,
    })

    if sections_target >= 5 and n_candidates >= chunk_size * 3:
        expert_candidates = selected_candidates[chunk_size * 2: chunk_size * 3]
        expert_ids = [c.get("fragment_id") for c in expert_candidates if c.get("fragment_id")]
        expert_texts = [truncate_text(c.get("text") or c.get("summary") or "", 200) for c in expert_candidates if (c.get("text") or c.get("summary"))]
        expert_narration = _synthesize_documentary_evidence(primary_topic, expert_texts, topic_label)
        sections.append({
            "role": "expert_evidence",
            "title": "What the Speakers Say",
            "viewer_narration": expert_narration,
            "evidence_fragment_ids": expert_ids,
        })

    synthesis_ids = fragment_ids_all[-2:]
    sections.append({
        "role": "synthesis",
        "title": f"What It Means" if not requested_speaker else f"{requested_speaker}: The Meaning",
        "viewer_narration": closing,
        "evidence_fragment_ids": synthesis_ids,
    })

    # Expand narration to fill target duration.
    # Each section gets roughly (target - 10s cards) / num_sections worth of content.
    # At ~2.5 words/sec TTS pace, compute the target word count per section.
    _section_budget_s = max(8, (target_duration_seconds - 10) // max(1, len(sections)))
    _target_words = max(20, int(_section_budget_s * 2.5))

    # Build a flat lookup: fragment_id -> evidence text
    _frag_text: dict[str, str] = {}
    for c in selected_candidates:
        fid = c.get("fragment_id")
        txt = (c.get("text") or c.get("summary") or "").strip()
        if fid and txt:
            _frag_text[fid] = txt

    for sec in sections:
        narration = sec.get("viewer_narration", "")
        current_words = len(narration.split())
        if current_words >= _target_words:
            continue
        # Extend using evidence fragment texts for this section
        evidence_ids = sec.get("evidence_fragment_ids") or []
        extra_sentences = []
        for fid in evidence_ids:
            if current_words + len(" ".join(extra_sentences).split()) >= _target_words:
                break
            ftext = _frag_text.get(fid, "").strip()
            if not ftext:
                continue
            # Use the first 2 sentences of fragment text as supporting narration
            raw_sentences = [s.strip() for s in ftext.replace("\n", " ").split(".") if len(s.strip()) > 20]
            for sent in raw_sentences[:2]:
                extra_sentences.append(sent + ".")
                if current_words + len(" ".join(extra_sentences).split()) >= _target_words:
                    break
        if extra_sentences:
            sec["viewer_narration"] = narration.rstrip() + " " + " ".join(extra_sentences)

    return {
        "documentary_style": "national_geographic",
        "central_question": central_question,
        "narrative_thesis": narrative_thesis,
        "closing_meaning": closing,
        "sections": sections,
    }


def _synthesize_documentary_context(topic: str | None, texts: list[str], topic_label: str) -> str:
    """Build human-context narration from evidence texts."""
    combined = " ".join(texts[:3])
    combined_lower = combined.lower()

    if topic in ("healthcare", "health"):
        if any(w in combined_lower for w in ("digital", "online", "home", "monitor")):
            return "In everyday care, digital tools are already changing how people contact doctors, monitor health, and remain independent at home."
        if any(w in combined_lower for w in ("trust", "patient", "safety")):
            return "At the centre of these conversations is a question of trust. Patients, doctors, and families are asking whether AI systems can be relied upon when the stakes are a person's health."
        return f"Across the AI4IA conversations, {topic_label} emerges as one of the most contested frontiers of artificial intelligence — where the promise of better outcomes meets the reality of fragile systems and uneven access."

    if topic in ("women", "empowerment"):
        if any(w in combined_lower for w in ("agency", "voice", "decision")):
            return "For many women, the question is not just whether they can use AI tools, but whether they have a voice in how those tools are built and governed."
        return f"Across regions and sectors, women are engaging with AI on terms that are often not their own. The systems, the data, and the decisions often reflect the world as it has been — not as it should be."

    if topic == "governance":
        return "Governance of AI is not a technical matter alone. It involves institutions, rights, accountability, and the political will to hold powerful systems to account."

    if topic == "youth":
        return "Young people are not passive recipients of the AI era. They are users, critics, and — increasingly — the people who must live longest with the consequences of decisions made today."

    return f"Across the AI4IA conversations, {topic_label} is presented not as an abstraction, but as a challenge with concrete human consequences — for people who depend on these systems to work, and work fairly."


def _synthesize_documentary_tension(topic: str | None, texts: list[str], topic_label: str) -> str:
    """Build tension narration from evidence texts."""
    combined = " ".join(texts[:3]).lower()

    if topic in ("healthcare", "health"):
        if any(w in combined for w in ("replace", "doctor", "judgment")):
            return "The question is not whether machines will replace doctors. It is whether doctors, patients, and AI systems can work together in ways that keep human judgment at the center of care."
        if any(w in combined for w in ("waste", "overload", "verification", "staff")):
            return "Beyond diagnosis, AI also points to quieter problems inside healthcare systems — medicine waste, overloaded staff, and the need to make verification faster and safer."
        return "But the promise of AI in healthcare depends on more than speed. Speakers warn that trust, patient safety, ethical use, and regulation must shape how these systems enter clinical life."

    if topic in ("women", "empowerment"):
        return "But participation is not the same as power. Women can use digital tools and still find that the systems behind them — the data, the design, the decisions — were never made for them."

    if topic == "governance":
        return "Regulation is not the enemy of innovation. Speakers argue it is the condition for trust — and without trust, the systems that claim to help will eventually be rejected or misused."

    return f"But the arrival of AI in {topic_label} does not resolve the underlying tensions. It often sharpens them — exposing who has voice, who has access, and who bears the risks."


def _synthesize_documentary_evidence(topic: str | None, texts: list[str], topic_label: str) -> str:
    """Build expert-evidence narration from speaker evidence texts."""
    combined = " ".join(texts[:3]).lower()

    if topic in ("healthcare", "health"):
        if any(w in combined for w in ("regulation", "govern", "ethics", "ethic")):
            return "Speakers in the healthcare conversation return to a shared concern: without clear regulation, ethical frameworks, and patient consent, AI systems risk causing the very harms they were designed to prevent."
        return "Across these conversations, healthcare is not presented as a place where AI replaces people. It is presented as a field where trust, safety, and access decide whether the technology helps."

    if topic in ("women", "empowerment"):
        return "The speakers who discuss women and AI do not speak in abstractions. They describe specific moments where systems fail women — in hiring, in healthcare, in justice, in access to credit."

    if topic == "governance":
        return "Governance conversations in the AI4IA archive are marked by a shared urgency: the rules are being written now, and the people most affected by them are too often absent from the table."

    return f"Across these conversations, {topic_label} is not a distant aspiration. It is a field in motion — with real evidence, real failure, and real possibilities for change."


def _make_cinematic_title(topic_terms: list[str], topic_label: str) -> str:
    """Generate a cinematic, non-technical documentary title."""
    cinematic_titles = {
        "healthcare": "AI at the Bedside",
        "health": "AI at the Bedside",
        "women": "Agency and the Algorithm",
        "empowerment": "Agency and the Algorithm",
        "governance": "Who Writes the Rules",
        "youth": "The Next Generation",
        "education": "Learning in the Age of AI",
        "crypto": "The Digital Promise",
        "accessibility": "Access and the Invisible Divide",
    }
    for term in topic_terms:
        if term in cinematic_titles:
            return cinematic_titles[term]
    return f"AI and {topic_label}"


def detect_generic_fallback_narration(text: str, prompt_profile: dict) -> list[str]:
    """Detect generic AI4IA fallback phrases in narration text."""
    issues = []
    if not text:
        return issues
    normalized = text.lower()
    for phrase in GENERIC_FALLBACK_PHRASES:
        if phrase in normalized:
            issues.append(f"generic_fallback_narration_detected: '{phrase[:60]}'")
    return issues


def validate_narration_prompt_alignment(
    narration_text: str,
    prompt_profile: dict,
    selected_candidates: list[dict],
) -> list[str]:
    """Check that narration includes at least one prompt-specific term."""
    issues = []
    if not narration_text:
        return issues

    normalized = narration_text.lower()
    topic_terms = set(prompt_profile.get("topic_terms") or [])
    requested_speaker = prompt_profile.get("requested_speaker")

    if requested_speaker:
        name_parts = requested_speaker.lower().split()
        if not any(part in normalized for part in name_parts if len(part) > 3):
            issues.append("narration_not_aligned_to_prompt: speaker name not mentioned in narration")
        return issues

    found_any = False
    for term in topic_terms:
        required = PROMPT_TERM_REQUIRED_WORDS.get(term, {term})
        if any(req in normalized for req in required):
            found_any = True
            break

    if not found_any:
        expanded = set(prompt_profile.get("expanded_terms") or [])
        if any(term.lower() in normalized for term in expanded):
            found_any = True

    if not found_any and topic_terms:
        issues.append(f"narration_not_aligned_to_prompt: none of {sorted(topic_terms)[:5]} found in narration")

    return issues


def build_evidence_grounded_narration(
    prompt_profile: dict,
    selected_candidates: list[dict],
    evidence_package: dict,
    target_duration_seconds: int,
) -> dict:
    """Build narration strictly from selected fragment evidence. No generic AI4IA templates."""
    topic_terms = prompt_profile.get("topic_terms") or []
    expanded_terms = prompt_profile.get("expanded_terms") or []
    requested_speaker = prompt_profile.get("requested_speaker")
    evidence_gaps = evidence_package.get("evidence_gaps") or []
    must_not_claim = evidence_package.get("must_not_claim") or []

    all_terms = list(dict.fromkeys(topic_terms + expanded_terms))
    topic_label = " ".join(topic_terms[:3]).title() if topic_terms else (requested_speaker or "AI")

    if not selected_candidates:
        gap_text = " ".join(evidence_gaps) if evidence_gaps else f"The archive does not contain enough direct evidence for this prompt."
        disclaimer = must_not_claim[0] if must_not_claim else ""
        full_text = gap_text + (" " + disclaimer if disclaimer else "")
        return {
            "generation_mode": "low_evidence_disclosure",
            "fallback_reason": "no_selected_candidates",
            "narration_text": full_text,
            "sections": [
                {
                    "section_id": "S1",
                    "title": "Evidence Gap Disclosure",
                    "narration": full_text,
                    "evidence_fragment_ids": [],
                    "source_videos": [],
                    "evidence_terms": [],
                    "claim_type": "evidence_gap",
                }
            ],
        }

    evidence_summaries = []
    for c in selected_candidates[:12]:
        text = (c.get("text") or c.get("summary") or "").strip()
        if text:
            evidence_summaries.append({
                "fragment_id": c.get("fragment_id"),
                "source_video": c.get("source_video"),
                "speaker": c.get("speaker"),
                "primary_theme": c.get("primary_theme"),
                "text": truncate_text(text, 300),
                "direct_terms": c.get("direct_evidence_terms") or [],
                "expanded_terms": c.get("expanded_evidence_terms") or [],
                "selection_reason": c.get("selection_reason", ""),
            })

    sections = []
    fragment_ids_used = [e["fragment_id"] for e in evidence_summaries if e["fragment_id"]]
    source_videos_used = list(dict.fromkeys(e["source_video"] for e in evidence_summaries if e["source_video"]))
    sections_target = timeline_segment_count_for_duration(target_duration_seconds)
    chunk_size = max(1, len(evidence_summaries) // max(1, sections_target - 2))

    doc_arc = build_documentary_arc(prompt_profile, selected_candidates, target_duration_seconds)
    arc_sections = doc_arc.get("sections") or []

    if requested_speaker:
        speaker_label = requested_speaker
        themes_covered = list(dict.fromkeys(
            c.get("primary_theme") for c in selected_candidates if c.get("primary_theme")
        ))[:4]
        hook_narration = (
            f"{speaker_label} has been a voice in the AI4IA conversation across multiple themes. "
            + (f"Their contributions touch on {', '.join(themes_covered)}." if themes_covered else "The following recordings trace the questions they returned to most often.")
        )
        sections.append({
            "section_id": "S1",
            "title": f"{speaker_label}: The Contribution",
            "narration": hook_narration,
            "evidence_fragment_ids": fragment_ids_used[:2],
            "source_videos": source_videos_used[:2],
            "evidence_terms": [],
            "claim_type": "direct_evidence",
        })
        for i, ev in enumerate(evidence_summaries[:sections_target - 2], start=2):
            theme_label = ev.get("primary_theme") or "AI and Society"
            ev_text = ev["text"][:200].rstrip(".").strip()
            narration = f"On the theme of {theme_label}: {ev_text}."
            sections.append({
                "section_id": f"S{i}",
                "title": f"{speaker_label}: {theme_label}",
                "narration": narration,
                "evidence_fragment_ids": [ev["fragment_id"]] if ev["fragment_id"] else [],
                "source_videos": [ev["source_video"]] if ev["source_video"] else [],
                "evidence_terms": ev["direct_terms"] + ev["expanded_terms"],
                "claim_type": "direct_evidence",
            })
        primary_topic = (prompt_profile.get("topic_terms") or [None])[0]
        closing = CINEMATIC_CLOSINGS.get(primary_topic or "", f"The future of AI will be shaped by the questions people like {speaker_label} are willing to ask — and by whether institutions are willing to answer them.")
        sections.append({
            "section_id": f"S{len(sections) + 1}",
            "title": "The Meaning",
            "narration": closing,
            "evidence_fragment_ids": [],
            "source_videos": [],
            "evidence_terms": [],
            "claim_type": "synthesis",
        })
    else:
        # Use documentary arc sections with cinematic narration
        for i, arc_sec in enumerate(arc_sections):
            role = arc_sec.get("role", "context")
            arc_ids = arc_sec.get("evidence_fragment_ids") or []
            arc_sources = list(dict.fromkeys(
                c.get("source_video") for c in selected_candidates
                if c.get("fragment_id") in arc_ids and c.get("source_video")
            ))
            arc_terms = list(dict.fromkeys(
                t for c in selected_candidates
                if c.get("fragment_id") in arc_ids
                for t in ((c.get("direct_evidence_terms") or []) + (c.get("expanded_evidence_terms") or []))
            ))[:6]
            if role == "synthesis":
                claim = "synthesis"
            elif not arc_ids:
                claim = "synthesis"
            else:
                claim = "direct_evidence"
            sections.append({
                "section_id": f"S{i + 1}",
                "title": arc_sec.get("title", f"{topic_label}"),
                "narration": arc_sec.get("viewer_narration", ""),
                "evidence_fragment_ids": arc_ids,
                "source_videos": arc_sources,
                "evidence_terms": arc_terms or all_terms[:4],
                "claim_type": claim,
            })

    full_narration = " ".join(s["narration"] for s in sections)
    return {
        "generation_mode": "evidence_grounded_fallback",
        "fallback_reason": "llm_unavailable_or_failed",
        "narration_text": full_narration,
        "sections": sections,
    }


def validate_storyline_before_save(
    storyline: dict,
    prompt_profile: dict,
    selected_candidates: list[dict],
    video_mode: str,
    evidence_strength: str,
) -> list[str]:
    """Strict pre-save validation. Returns list of validation issues."""
    issues = []
    if not STRICT_NO_GENERIC_FALLBACK:
        return issues

    sections = storyline.get("sections") or []
    topic_terms = set(prompt_profile.get("topic_terms") or [])
    all_narration = " ".join(s.get("narration", "") for s in sections)

    generic_issues = detect_generic_fallback_narration(all_narration, prompt_profile)
    if generic_issues and selected_candidates:
        issues.extend(generic_issues)

    if selected_candidates and all_narration.strip():
        alignment_issues = validate_narration_prompt_alignment(all_narration, prompt_profile, selected_candidates)
        issues.extend(alignment_issues)

    pipeline_issues = validate_viewer_facing_text(storyline)
    issues.extend(pipeline_issues)

    side_topic_lower = [t.lower() for t in SIDE_TOPIC_TERMS]
    for section in sections:
        sid = section.get("section_id", "?")
        narration = (section.get("narration") or "").lower()
        for term in side_topic_lower:
            if term in narration:
                issues.append(f"side_topic_leakage_detected: '{term}' in section {sid} narration")
                break

    allowed_no_evidence = {"title_card", "closing_card", "evidence_gap_disclosure", "synthesis_narration", "final_synthesis"}
    allowed_claim_types_no_evidence = {"synthesis", "evidence_gap"}
    for section in sections:
        section_type = section.get("section_type", "")
        section_id = section.get("section_id", "?")
        claim_type = section.get("claim_type", "")
        if section_type in allowed_no_evidence:
            continue
        if claim_type in allowed_claim_types_no_evidence:
            continue
        has_frags = bool(
            section.get("evidence_fragment_ids")
            or section.get("archive_visual_fragments")
            or section.get("speaker_soundbites")
        )
        if not has_frags and selected_candidates:
            issues.append(f"section_missing_evidence_links: {section_id} ({section_type})")

    if len(selected_candidates) == 0 and video_mode not in (
        "low_evidence_brief", "low_evidence_speaker_brief", "no_evidence", "low_evidence_disclosure"
    ):
        issues.append(f"low_evidence_video_mode_mismatch: no selected_candidates but video_mode={video_mode}")

    # Phase 4O: moment-level validation on export_timeline
    export_timeline = storyline.get("export_timeline") or []
    for titem in export_timeline:
        if titem.get("type") in ("title_card", "closing_card"):
            continue
        fid = titem.get("fragment_id")
        seg_type = titem.get("type", "")
        if not fid:
            continue

        # speaker_clip must have moment_text
        if seg_type == "speaker_clip" and not titem.get("moment_text"):
            issues.append(f"selected_clip_missing_moment_text: {fid}")

        # speaker_clip must not have intro language
        moment_text = (titem.get("moment_text") or "").lower()
        if seg_type == "speaker_clip" and moment_text:
            intro_pen = _window_intro_penalty(moment_text)
            if intro_pen >= 0.35:
                issues.append(f"speaker_clip_has_intro_language: {fid}")

        # speaker_clip must not have low speech ratio
        speech_q = titem.get("speech_quality") or {}
        if seg_type == "speaker_clip" and speech_q:
            if not speech_q.get("is_usable_speech", True):
                issues.append(f"speaker_clip_has_low_speech_ratio: {fid}")

        # archive_visual should be muted
        if seg_type == "archive_visual" and titem.get("muted") is False:
            issues.append(f"archive_visual_not_muted: {fid}")

        # subtitle_text must not use full fragment text for speaker clips
        subtitle = titem.get("subtitle_text", "")
        if seg_type == "speaker_clip" and subtitle:
            # Warn if subtitle is > 3× the moment_text length (likely full-fragment text)
            moment = titem.get("moment_text", "")
            if moment and len(subtitle) > len(moment) * 3:
                issues.append(f"speaker_subtitle_uses_full_fragment_text: {fid}")

        # moment_selection_reason should be present
        if not titem.get("moment_selection_reason"):
            issues.append(f"moment_selection_missing_reason: {fid}")

    return issues


def build_fallback_storyline(
    user_prompt: str,
    duration: int,
    candidates: list[dict[str, Any]],
    style: str,
    director_plan: dict[str, Any],
    prompt_intent: dict[str, Any],
    evidence_package: dict[str, Any] | None = None,
) -> dict[str, Any]:
    requested_speaker = prompt_intent.get("requested_speaker")
    topic_terms = prompt_intent.get("topic_terms") or []
    evidence_gaps = (evidence_package or {}).get("evidence_gaps") or []
    must_not_claim = (evidence_package or {}).get("must_not_claim") or []
    topic_label = " ".join(topic_terms[:3]).title() if topic_terms else (requested_speaker or "AI")

    mode_to_roles = {
        "speaker_led_documentary": ["opening_context", "exact_evidence_clip", "exact_evidence_clip", "bridge_narration", "exact_evidence_clip", "synthesis_narration", "closing_takeaway"],
        "speaker_led_theme_summary": ["opening_context", "exact_evidence_clip", "exact_evidence_clip", "bridge_narration", "exact_evidence_clip", "closing_takeaway"],
        "low_evidence_speaker_brief": ["opening_context", "available_evidence_clip", "evidence_gap_narration", "closing_takeaway"],
        "no_evidence": ["opening_context", "evidence_gap_narration", "closing_takeaway"],
        "relay_of_ideas": ["opening_context", "exact_evidence_clip", "related_theme_clip", "related_theme_clip", "synthesis_narration", "related_governance_clip", "closing_takeaway"],
        "theme_summary_explainer": ["opening_context", "related_theme_clip", "related_theme_clip", "exact_evidence_clip", "related_governance_clip", "synthesis_narration", "closing_takeaway"],
        "narration_led_explainer": ["opening_context", "exact_evidence_clip", "related_theme_clip", "related_theme_clip", "synthesis_narration", "related_supporting_clip", "closing_takeaway"],
        "low_evidence_brief": ["opening_context", "available_evidence_clip", "evidence_gap_narration", "related_context", "related_context", "synthesis_narration", "closing_takeaway"],
    }
    _base_roles = mode_to_roles.get(director_plan.get("video_mode"), mode_to_roles["narration_led_explainer"])
    # Extend role_sequence to cover any number of sections by cycling the middle roles
    _middle = _base_roles[1:-1] if len(_base_roles) > 2 else _base_roles
    _needed = timeline_segment_count_for_duration(duration) + 4  # generous headroom
    role_sequence = [_base_roles[0]] + (_middle * ((_needed // max(1, len(_middle))) + 1))[:_needed] + [_base_roles[-1]]

    # No candidates: low_evidence_disclosure only
    if not candidates:
        if requested_speaker:
            disclosure_text = (
                f"The AI4IA archive does not contain direct clips from {requested_speaker}. "
                f"No documentary video was generated because using unrelated clips would be misleading."
            )
        elif "crypto" in set(topic_terms):
            disclosure_text = (
                f"The AI4IA archive does not contain direct evidence for AI and cryptocurrency. "
                f"Related clips discuss cryptography, cybersecurity, or broader AI governance, "
                f"which should not be presented as cryptocurrency evidence."
            )
        else:
            gap_detail = " ".join(evidence_gaps[:2]) if evidence_gaps else ""
            disclosure_text = (
                f"The archive does not contain enough direct evidence for this prompt ({topic_label}). "
                f"No final documentary video was generated because using unrelated clips would be misleading. "
                + gap_detail
            )
            if must_not_claim:
                disclosure_text += " " + must_not_claim[0]

        return {
            "title": f"Evidence Gap: {topic_label}",
            "logline": disclosure_text,
            "style": style,
            "target_duration_seconds": duration,
            "estimated_duration_seconds": 0,
            "generation_mode": "low_evidence_disclosure",
            "fallback_reason": "no_selected_candidates",
            "director_plan": director_plan,
            "narrative_arc": ["evidence_gap_disclosure"],
            "sections": [
                {
                    "section_id": "S1",
                    "section_type": "evidence_gap_disclosure",
                    "target_duration_seconds": 15,
                    "director_role": "evidence_gap_narration",
                    "evidence_role": "evidence_gap",
                    "evidence_confidence": "none",
                    "fallback_allowed": True,
                    "planning_note": "No archive evidence found; disclosure only.",
                    "narration": disclosure_text,
                    "evidence_fragment_ids": [],
                    "source_videos": [],
                    "evidence_terms": [],
                    "claim_type": "evidence_gap",
                    "archive_visual_fragments": [],
                    "speaker_soundbites": [],
                    "generated_assets_needed": [],
                    "internet_assets_needed": [],
                }
            ],
            "voiceover_script": [{"section_id": "S1", "text": disclosure_text, "tone": "calm informational narrator"}],
            "selected_speaker_soundbites": [],
            "asset_plan": {
                "real_archive_assets": [],
                "generated_assets": [],
                "internet_assets": [],
                "voiceover_required": True,
                "subtitles_required": False,
                "background_music_required": False,
            },
            "export_timeline": [],
            "editor_notes": [
                "low_evidence_disclosure: no archive evidence found for this prompt.",
                "should_export_video: false",
            ],
            "planning_summary": {
                "video_mode": director_plan.get("video_mode"),
                "evidence_strength": director_plan.get("evidence_strength"),
                "exact_clip_count": 0,
                "related_clip_count": 0,
                "fallback_risk": "high",
                "planning_notes": director_plan.get("planning_notes", []),
            },
            "should_export_video": False,
        }

    # Candidates exist: build evidence-grounded narration
    soundbites_cap = max(8, timeline_segment_count_for_duration(duration))
    soundbites = [
        item for item in candidates
        if "speaker_soundbite" in item.get("clip_usage_candidates", [])
    ][:soundbites_cap]
    visuals = candidates[:max(18, timeline_segment_count_for_duration(duration) * 2)]

    grounded = build_evidence_grounded_narration(
        prompt_profile=prompt_intent,
        selected_candidates=candidates,
        evidence_package=evidence_package or {},
        target_duration_seconds=duration,
    )
    grounded_sections = grounded.get("sections") or []

    section_types_sequence = [
        "opening_hook", "context", "problem_or_tension",
        "human_perspective", "expert_evidence", "solution_direction", "final_synthesis",
        # extended roles for longer videos (repeats cycle gracefully)
        "context", "human_perspective", "expert_evidence",
        "problem_or_tension", "solution_direction", "context",
        "expert_evidence", "human_perspective", "solution_direction",
        "context", "expert_evidence", "final_synthesis",
    ]
    built_sections = []

    for idx, gs in enumerate(grounded_sections):
        section_id = gs.get("section_id", f"S{idx + 1}")
        section_type = gs.get("section_type") or (section_types_sequence[idx] if idx < len(section_types_sequence) else "final_synthesis")
        narration = gs.get("narration", "")
        evidence_fragment_ids = gs.get("evidence_fragment_ids") or []
        source_videos = gs.get("source_videos") or []
        evidence_terms = gs.get("evidence_terms") or []
        claim_type = gs.get("claim_type", "direct_evidence")

        section_visuals = [v for v in visuals if v.get("fragment_id") in evidence_fragment_ids] or visuals[idx * 2: idx * 2 + 2]
        section_soundbites = soundbites[idx: idx + 1]
        section_role = role_sequence[idx] if idx < len(role_sequence) else "related_theme_clip"

        if claim_type == "evidence_gap":
            evidence_role = "evidence_gap"
            evidence_confidence = "none"
        elif section_soundbites:
            evidence_role = "exact"
            evidence_confidence = "high"
        elif section_visuals:
            evidence_role = "related"
            evidence_confidence = "medium"
        else:
            evidence_role = "low_evidence"
            evidence_confidence = "low"

        built_sections.append({
            "section_id": section_id,
            "section_type": section_type,
            "target_duration_seconds": round(duration / max(1, len(grounded_sections))),
            "director_role": section_role,
            "evidence_role": evidence_role,
            "evidence_confidence": evidence_confidence,
            "fallback_allowed": evidence_role != "exact",
            "planning_note": (
                f"Evidence-grounded narration from {len(evidence_fragment_ids)} fragment(s)."
                if evidence_fragment_ids else "Evidence-grounded narration; disclosure section."
            ),
            "narration": narration,
            "evidence_fragment_ids": evidence_fragment_ids,
            "source_videos": source_videos,
            "evidence_terms": evidence_terms,
            "claim_type": claim_type,
            "archive_visual_fragments": [
                {
                    "fragment_id": item["fragment_id"],
                    "usage": "evidence_visual",
                    "start": item.get("start"),
                    "end": item.get("end"),
                    "reason": item.get("summary", "Evidence-grounded archive visual."),
                }
                for item in section_visuals
            ],
            "speaker_soundbites": [
                {
                    "fragment_id": item["fragment_id"],
                    "start": item.get("start"),
                    "end": item.get("end"),
                    "quote_summary": item.get("summary"),
                    "reason": "Evidence-matched speaker soundbite.",
                }
                for item in section_soundbites
            ],
            "generated_assets_needed": [],
            "internet_assets_needed": [],
        })

    doc_arc = build_documentary_arc(prompt_intent, candidates, duration)

    if requested_speaker:
        title = f"{requested_speaker}: Archive Evidence"
        logline = doc_arc.get("narrative_thesis") or f"A documentary tracing {requested_speaker}'s contributions to AI4IA."
    elif topic_label:
        topic_term_0 = (topic_terms[:1] or [None])[0]
        hook = CINEMATIC_HOOKS.get(topic_term_0 or "", "")
        title = _make_cinematic_title(topic_terms, topic_label)
        logline = hook or doc_arc.get("narrative_thesis") or f"A documentary on AI and {topic_label}."
    else:
        title = "Artificial Intelligence and the Human Question"
        logline = doc_arc.get("narrative_thesis") or "A documentary on artificial intelligence and society."

    return {
        "title": title,
        "logline": logline,
        "closing_meaning": doc_arc.get("closing_meaning", ""),
        "central_question": doc_arc.get("central_question", ""),
        "style": style,
        "target_duration_seconds": duration,
        "estimated_duration_seconds": duration,
        "generation_mode": grounded.get("generation_mode", "evidence_grounded_fallback"),
        "fallback_reason": grounded.get("fallback_reason", "llm_unavailable_or_failed"),
        "director_plan": director_plan,
        "narrative_arc": [s["section_type"] for s in built_sections],
        "sections": built_sections,
        "voiceover_script": [
            {"section_id": s["section_id"], "text": s["narration"], "tone": "observational documentary narrator"}
            for s in built_sections if s.get("narration")
        ],
        "selected_speaker_soundbites": [
            {
                "fragment_id": item["fragment_id"],
                "source_video": item.get("source_video"),
                "start": item.get("start"),
                "end": item.get("end"),
                "quote_summary": item.get("summary"),
                "use_in_section": f"S{idx + 1}",
            }
            for idx, item in enumerate(soundbites[:7])
        ],
        "asset_plan": {
            "real_archive_assets": [item["fragment_id"] for item in visuals[:18]],
            "generated_assets": [],
            "internet_assets": [],
            "voiceover_required": True,
            "subtitles_required": True,
            "background_music_required": True,
        },
        "export_timeline": [],
        "editor_notes": [
            f"evidence_grounded_fallback: LLM narration not available; documentary arc built from {len(candidates)} selected recordings.",
            f"generation_mode: {grounded.get('generation_mode')}",
        ],
        "planning_summary": {
            "video_mode": director_plan.get("video_mode"),
            "evidence_strength": director_plan.get("evidence_strength"),
            "exact_clip_count": director_plan.get("exact_clip_count"),
            "related_clip_count": director_plan.get("related_clip_count"),
            "fallback_risk": director_plan.get("fallback_risk"),
            "planning_notes": director_plan.get("planning_notes", []),
        },
        "should_export_video": len(candidates) > 0,
    }


def save_storyline(
    storyline: dict[str, Any],
    user_prompt: str,
    prompt_intent: dict[str, Any],
    duration: int,
    style: str,
    retrieved: list[dict[str, Any]],
    buckets: dict[str, list[dict[str, Any]]],
    director_plan: dict[str, Any],
    evidence_metrics: dict[str, Any],
    validation_issues: list[str],
    evidence_package: dict[str, Any],
    retrieval_diagnostics: dict[str, Any],
    candidate_bucket_summary: dict[str, Any] | None = None,
    corpus_diagnostics: dict[str, Any] | None = None,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = STORYLINES_DIR / f"storyline_{timestamp}.json"

    selected = buckets.get("selected_candidates", [])
    supporting = buckets.get("supporting_candidates", [])
    context = buckets.get("context_candidates", [])
    rejected = buckets.get("rejected_candidates", [])

    payload = {
        "created_at": datetime.utcnow().isoformat(),
        "user_prompt": user_prompt,
        "prompt_profile": prompt_intent,
        "prompt_intent": prompt_intent,
        "target_duration_seconds": duration,
        "style": style,
        "corpus_version": _ACTIVE_CORPUS_VERSION,
        "embeddings_dir": str(_ACTIVE_EMBEDDINGS_DIR),
        "corpus_diagnostics": corpus_diagnostics or _CORPUS_DIAGNOSTICS,
        "generation_mode": storyline.get("generation_mode", "unknown"),
        "llm_parse_error": storyline.get("llm_parse_error"),
        "fallback_reason": storyline.get("fallback_reason"),
        "should_export_video": storyline.get("should_export_video", True),
        "models": {
            "embedding_model": EMBED_MODEL,
            "strong_llm_model": STRONG_LLM_MODEL,
            "fast_llm_model": FAST_LLM_MODEL,
        },
        "source_files": {
            "embedding_matrix": str(_ACTIVE_MATRIX_PATH),
            "embedding_records": str(_ACTIVE_JSONL_PATH),
        },
        "retrieval_summary": {
            "retrieved_count": len(retrieved),
            "selected_candidate_count": len(selected),
            "supporting_candidate_count": len(supporting),
            "context_candidate_count": len(context),
            "rejected_candidate_count": len(rejected),
            "top_semantic_score": retrieved[0]["semantic_score"] if retrieved else None,
        },
        "retrieval_diagnostics": retrieval_diagnostics,
        "candidate_bucket_summary": candidate_bucket_summary or {
            "selected": len(selected),
            "supporting": len(supporting),
            "context": len(context),
            "rejected": len(rejected),
        },
        "candidate_contract": {
            "selected_candidates": "final-video-safe evidence only",
            "supporting_candidates": "narration support only — not for speaker soundbites",
            "context_candidates": "background/context only — not for final video",
            "rejected_candidates": "not for final video",
        },
        "planning_summary": {
            "video_mode": director_plan.get("video_mode"),
            "evidence_strength": director_plan.get("evidence_strength"),
            "exact_clip_count": evidence_metrics.get("exact_clip_count"),
            "related_clip_count": evidence_metrics.get("related_clip_count"),
            "strong_related_count": evidence_metrics.get("strong_related_count"),
            "usable_speaker_clip_count": evidence_metrics.get("usable_speaker_clip_count"),
            "usable_speaker_visual_count": evidence_metrics.get("usable_speaker_visual_count", 0),
            "supporting_candidate_count": evidence_metrics.get("supporting_candidate_count"),
            "context_candidate_count": evidence_metrics.get("contextual_count"),
            "rejected_clip_count": evidence_metrics.get("rejected_clip_count"),
            "fallback_risk": director_plan.get("fallback_risk"),
            "planning_notes": director_plan.get("planning_notes", []),
        },
        "validation_issues": validation_issues,
        "timeline_repair": storyline.get("timeline_repair", {"required": False}),
        "evidence_package": evidence_package,
        "export_timeline": storyline.get("export_timeline", []),
        "selected_candidates": build_candidate_pack(selected),
        "supporting_candidates": build_candidate_pack(supporting[:10]),
        "context_candidates": build_candidate_pack(context[:5]),
        "rejected_candidates": build_candidate_pack(rejected[:5]),
        "debug_retrieved_candidates": build_candidate_pack(retrieved[:25]),
        "storyline": storyline,
    }

    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return output_path


def print_summary(storyline: dict[str, Any], output_path: Path) -> None:
    log("")
    log("Storyline generated")
    log(f"Output: {output_path}")

    title = storyline.get("title", "Untitled")
    log(f"Title: {title}")
    director_plan = storyline.get("director_plan", {})
    if director_plan:
        log(f"Video mode: {director_plan.get('video_mode')} | evidence: {director_plan.get('evidence_strength')} | exact={director_plan.get('exact_clip_count')} related={director_plan.get('related_clip_count')}")

    sections = storyline.get("sections", [])
    log(f"Sections: {len(sections)}")

    for section in sections:
        section_id = section.get("section_id")
        section_type = section.get("section_type")
        duration = section.get("target_duration_seconds")
        narration = truncate_text(section.get("narration", ""), 120)
        log(f"- {section_id} | {section_type} | {duration}s | {narration}")


def activate_corpus(
    embeddings_dir: str,
    corpus_version: str,
    min_transcript_quality: float,
    exclude_manual_review: bool,
) -> dict[str, Any]:
    """
    Configure the active embedding corpus globals, load and optionally filter
    records+matrix, and return corpus diagnostics.
    """
    global _ACTIVE_EMBEDDINGS_DIR, _ACTIVE_MATRIX_PATH, _ACTIVE_JSONL_PATH
    global _ACTIVE_CORPUS_VERSION, _FILTERED_RECORDS, _FILTERED_MATRIX, _CORPUS_DIAGNOSTICS

    active_dir = Path(embeddings_dir)
    _ACTIVE_EMBEDDINGS_DIR = active_dir
    _ACTIVE_MATRIX_PATH = active_dir / "fragment_embedding_matrix.npy"
    _ACTIVE_JSONL_PATH = active_dir / "classified_fragments_embeddings.jsonl"
    _ACTIVE_CORPUS_VERSION = corpus_version

    log(f"Embedding dir: {active_dir}")
    log(f"Corpus version: {corpus_version}")

    if _chroma_mode():
        matrix, all_records = _load_corpus_from_chroma()
    else:
        if not _ACTIVE_MATRIX_PATH.exists():
            raise FileNotFoundError(f"Missing embedding matrix: {_ACTIVE_MATRIX_PATH}")
        if not _ACTIVE_JSONL_PATH.exists():
            raise FileNotFoundError(f"Missing embedding JSONL: {_ACTIVE_JSONL_PATH}")
        all_records = load_embedding_records()
        matrix = np.load(_ACTIVE_MATRIX_PATH)

    if matrix.shape[0] != len(all_records):
        raise RuntimeError(
            f"Matrix/metadata mismatch: matrix rows={matrix.shape[0]}, records={len(all_records)}"
        )

    log(f"Embedding records loaded: {len(all_records)}")

    # Count version and audio_status distributions before filtering
    version_counts_before: dict[str, int] = {}
    audio_counts_before: dict[str, int] = {}
    for rec in all_records:
        v = rec.get("source_transcript_version") or "unknown"
        version_counts_before[v] = version_counts_before.get(v, 0) + 1
        a = rec.get("audio_quality_status") or "unknown"
        audio_counts_before[a] = audio_counts_before.get(a, 0) + 1

    embedding_dim = int(matrix.shape[1]) if matrix.ndim == 2 else 0

    # Determine which source_transcript_version values are allowed
    # "mixed_cleaned_v2" means use all records regardless of version label
    # "original_v1" or "cleaned_v2" restrict to that exact version
    # Any other string is treated as a metadata label only (no version filtering)
    if corpus_version == "mixed_cleaned_v2":
        allowed_source_versions = None  # no version filtering
    elif corpus_version in {"original_v1", "cleaned_v2"}:
        allowed_source_versions: set | None = {corpus_version}
    else:
        allowed_source_versions = None  # unknown label — don't drop records

    filtered_records: list[dict[str, Any]] = []
    filtered_vectors: list[Any] = []

    for idx, record in enumerate(all_records):
        if allowed_source_versions is not None:
            rec_version = record.get("source_transcript_version") or "original_v1"
            if rec_version not in allowed_source_versions:
                continue

        if exclude_manual_review and record.get("audio_quality_status") == "manual_review":
            continue

        quality = record.get("transcript_quality_score")
        if quality is None:
            quality = 1.0
        if float(quality) < min_transcript_quality:
            continue

        filtered_records.append(record)
        filtered_vectors.append(matrix[idx])

    records_after = len(filtered_records)
    log(f"Records after filtering: {records_after}")

    if filtered_vectors:
        _FILTERED_MATRIX = np.stack(filtered_vectors, axis=0)
    else:
        _FILTERED_MATRIX = np.empty((0, embedding_dim), dtype=matrix.dtype)

    _FILTERED_RECORDS = filtered_records

    log(f"Embedding matrix shape: {_FILTERED_MATRIX.shape}")

    # Count version and audio_status distributions after filtering
    version_counts_after: dict[str, int] = {}
    audio_counts_after: dict[str, int] = {}
    for rec in filtered_records:
        v = rec.get("source_transcript_version") or "unknown"
        version_counts_after[v] = version_counts_after.get(v, 0) + 1
        a = rec.get("audio_quality_status") or "unknown"
        audio_counts_after[a] = audio_counts_after.get(a, 0) + 1

    diagnostics = {
        "total_embedding_records": len(all_records),
        "records_after_filtering": records_after,
        "source_version_counts_before": version_counts_before,
        "source_version_counts_after": version_counts_after,
        "audio_status_counts_before": audio_counts_before,
        "audio_status_counts_after": audio_counts_after,
        "cleaned_v2_records": version_counts_before.get("cleaned_v2", 0),
        "original_v1_records": version_counts_before.get("original_v1", 0),
        "manual_review_records": audio_counts_before.get("manual_review", 0),
        "min_transcript_quality": min_transcript_quality,
        "exclude_manual_review": exclude_manual_review,
        "corpus_version": corpus_version,
        "embeddings_dir": str(active_dir),
        "embedding_dimension": embedding_dim,
    }
    _CORPUS_DIAGNOSTICS = diagnostics
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate documentary storyline from AI4IA fragments.")
    parser.add_argument(
        "--prompt",
        required=True,
        help="User prompt for the documentary storyline."
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=180,
        help="Target video duration in seconds."
    )
    parser.add_argument(
        "--style",
        default="National Geographic-style documentary explainer",
        help="Video style."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of fragments to retrieve before balancing."
    )
    parser.add_argument(
        "--final-candidates",
        type=int,
        default=DEFAULT_FINAL_CANDIDATES,
        help="Number of fragments sent to the storyline model."
    )
    parser.add_argument(
        "--fallback-only",
        action="store_true",
        help="Skip LLM storyline generation and use deterministic fallback."
    )
    parser.add_argument(
        "--embeddings-dir",
        default=str(EMBEDDINGS_DIR),
        help="Directory containing embedding matrix and JSONL. Default: data/embeddings/"
    )
    parser.add_argument(
        "--corpus-version",
        default="original_v1",
        help="Filter to a specific source_transcript_version (e.g. cleaned_v2, original_v1). Default: original_v1"
    )
    parser.add_argument(
        "--min-transcript-quality",
        type=float,
        default=0.0,
        help="Exclude fragments with transcript_quality_score below this threshold. Default: 0.0"
    )
    parser.add_argument(
        "--exclude-manual-review",
        action="store_true",
        help="Exclude fragments with audio_quality_status == 'manual_review'."
    )

    args = parser.parse_args()

    log("Starting Step 07: Generate Documentary Storyline")
    log(f"Prompt: {args.prompt}")
    log(f"Duration: {args.duration}")
    log(f"Style: {args.style}")
    log(f"Top K: {args.top_k}")
    log(f"Final candidates: {args.final_candidates}")

    corpus_diagnostics = activate_corpus(
        embeddings_dir=args.embeddings_dir,
        corpus_version=args.corpus_version,
        min_transcript_quality=args.min_transcript_quality,
        exclude_manual_review=args.exclude_manual_review,
    )

    warm_up_ollama()
    known_speakers = load_known_speakers()
    log(f"Loaded {len(known_speakers)} known speakers.")
    all_records = _FILTERED_RECORDS if _FILTERED_RECORDS is not None else load_embedding_records()
    prompt_intent = detect_prompt_intent(args.prompt, known_speakers, all_records, target_duration_seconds=args.duration)
    log(f"Prompt intent: {json.dumps(prompt_intent, ensure_ascii=False)}")

    raw_retrieved = retrieve_candidates(args.prompt, prompt_intent, top_k=args.top_k)
    retrieved = annotate_candidate_relevance(raw_retrieved, args.prompt, prompt_intent, known_speakers)

    buckets = bucket_candidates(retrieved, prompt_intent)
    selected = buckets["selected_candidates"]
    supporting = buckets["supporting_candidates"]
    context = buckets["context_candidates"]
    rejected_cands = buckets["rejected_candidates"]

    log(f"Retrieved: {len(retrieved)} | Selected: {len(selected)} | Supporting: {len(supporting)} | Context: {len(context)} | Rejected: {len(rejected_cands)}")

    selected = select_balanced_candidates(selected, limit=min(args.final_candidates, len(selected))) if selected else []
    buckets["selected_candidates"] = selected

    selected, selection_issues = validate_speaker_selection(selected, prompt_intent)
    buckets["selected_candidates"] = selected

    # Preliminary metrics to determine cap
    pre_metrics = compute_evidence_metrics(selected, supporting, context, rejected_cands)
    pre_plan = build_director_plan(pre_metrics, intent=prompt_intent)
    video_mode = pre_plan["video_mode"]
    evidence_strength = pre_plan["evidence_strength"]

    # Apply duration-aware cap with sort + diversity filter
    cap = get_selected_candidate_cap(args.duration, video_mode, evidence_strength)
    selected = sort_final_video_candidates(selected)
    selected = apply_diversity_cap(
        selected,
        max_per_source=2,
        max_per_speaker=2,
        requested_speaker=prompt_intent.get("requested_speaker"),
    )
    if evidence_strength == "low":
        cap_reason = "evidence_strength=low"
    elif video_mode in ("low_evidence_brief", "low_evidence_speaker_brief", "no_evidence"):
        cap_reason = f"video_mode={video_mode}"
    else:
        cap_reason = f"target_duration_seconds={args.duration}"
    pre_cap_count = len(selected)
    selected = selected[:cap]
    buckets["selected_candidates"] = selected

    candidate_bucket_summary = {
        "selected": len(selected),
        "supporting": len(supporting),
        "context": len(context),
        "rejected": len(rejected_cands),
        "selected_cap": cap,
        "pre_cap_count": pre_cap_count,
        "cap_reason": cap_reason,
    }
    log(f"Cap applied: {pre_cap_count} → {len(selected)} (cap={cap}, reason={cap_reason})")

    # Recompute metrics on capped selected
    evidence_metrics = compute_evidence_metrics(selected, supporting, context, rejected_cands)
    director_plan = build_director_plan(evidence_metrics, intent=prompt_intent)
    retrieval_diagnostics = compute_retrieval_diagnostics(prompt_intent, retrieved, buckets, args.top_k)

    # Build evidence-gap notes and append to planning notes
    prompt_terms = set(prompt_intent.get("topic_terms") or [])
    gap_notes = build_evidence_gap_notes(prompt_intent, prompt_terms, evidence_metrics)
    existing_notes = director_plan.get("planning_notes", [])
    merged_notes = existing_notes[:]
    for note in gap_notes:
        if note not in merged_notes:
            merged_notes.append(note)
    director_plan["planning_notes"] = merged_notes
    evidence_package = build_storyline_evidence_package(prompt_intent, buckets)

    valid_fragment_ids = {item.get("fragment_id") for item in selected}

    # For chronological_relay: pre-assign real fragments (grouped by year) to each
    # recommended section BEFORE the LLM call so the LLM receives concrete year_slot
    # values and actual clips to narrate around. Without this the LLM receives
    # year_slot=null for every section and writes narrations with no temporal anchors.
    if video_mode == "chronological_relay":
        relay_structure = director_plan.get("recommended_structure") or []
        if relay_structure:
            director_plan["recommended_structure"] = _assign_relay_sections_by_year(
                relay_structure, selected
            )
            log(f"chronological_relay: pre-assigned year slots to {len(relay_structure)} sections from {len(set(c.get('video_year') for c in selected if c.get('video_year') is not None))} distinct years")

    log(f"Selected candidates after cap: {len(selected)}")
    log(f"Evidence: exact={evidence_metrics['exact_clip_count']} strong={evidence_metrics['strong_related_count']} supporting={evidence_metrics.get('supporting_candidate_count', 0)} usable_speaker={evidence_metrics['usable_speaker_clip_count']}")

    storyline = None
    generation_mode = "llm"
    llm_parse_error = None
    fallback_reason = None

    if not args.fallback_only:
        try:
            storyline = generate_llm_storyline(
                user_prompt=args.prompt,
                duration=args.duration,
                candidates=selected,
                style=args.style,
                director_plan=director_plan,
                prompt_intent=prompt_intent,
                evidence_package=evidence_package,
            )
            storyline["generation_mode"] = "llm"
        except RuntimeError as exc:
            error_msg = str(exc)
            if "Could not parse JSON" in error_msg or "parse" in error_msg.lower():
                log(f"LLM JSON parse failed: {exc}")
                log("Falling back to evidence-grounded fallback (json parse failure).")
                generation_mode = "llm_json_parse_failed"
                llm_parse_error = error_msg
                fallback_reason = "llm_json_parse_failed_used_evidence_grounded_fallback"
            else:
                log(f"Strong LLM storyline failed: {exc}")
                generation_mode = "llm_failed"
                llm_parse_error = error_msg
                fallback_reason = "llm_request_failed"
        except Exception as exc:
            log(f"Strong LLM storyline failed: {exc}")
            generation_mode = "llm_failed"
            llm_parse_error = str(exc)
            fallback_reason = "llm_request_failed"

    if storyline is None:
        log(f"Using evidence-grounded fallback (mode={generation_mode}, reason={fallback_reason}).")
        storyline = build_fallback_storyline(
            user_prompt=args.prompt,
            duration=args.duration,
            candidates=selected,
            style=args.style,
            director_plan=director_plan,
            prompt_intent=prompt_intent,
            evidence_package=evidence_package,
        )
        if generation_mode in ("llm_json_parse_failed", "llm_failed"):
            storyline["generation_mode"] = generation_mode
        if llm_parse_error:
            storyline["llm_parse_error"] = llm_parse_error
        if fallback_reason:
            storyline["fallback_reason"] = fallback_reason

    # Phase 4O.3: If LLM returned no sections (or malformed sections), rebuild from
    # build_documentary_arc so that Step 08 can generate voiceover narration.
    if not storyline.get("sections") and selected:
        log("Sections missing from LLM output — rebuilding from build_documentary_arc.")
        arc = build_documentary_arc(
            prompt_profile=prompt_intent,
            selected_candidates=selected,
            target_duration_seconds=args.duration,
        )
        rebuilt_sections = arc.get("sections") or []
        # Stamp mandatory section fields
        for idx, sec in enumerate(rebuilt_sections, start=1):
            sec.setdefault("section_id", f"S{idx}")
            role = sec.get("role", "")
            sec.setdefault("section_type", role or f"section_{idx}")
            sec.setdefault("narration", sec.get("viewer_narration", ""))
            sec.setdefault("viewer_narration", sec.get("narration", ""))
        storyline["sections"] = rebuilt_sections
        storyline.setdefault("central_question", arc.get("central_question", ""))
        storyline.setdefault("narrative_thesis", arc.get("narrative_thesis", ""))
        storyline.setdefault("closing_meaning", arc.get("closing_meaning", ""))
        log(f"Rebuilt {len(rebuilt_sections)} sections from documentary arc.")

    # Backfill evidence_fragment_ids, claim_type, evidence_terms into sections that lack them
    for section in (storyline.get("sections") or []):
        if not section.get("evidence_fragment_ids"):
            frag_ids = []
            for frag in (section.get("archive_visual_fragments") or []):
                fid = frag.get("fragment_id")
                if fid:
                    frag_ids.append(fid)
            for sb in (section.get("speaker_soundbites") or []):
                fid = sb.get("fragment_id")
                if fid:
                    frag_ids.append(fid)
            section["evidence_fragment_ids"] = list(dict.fromkeys(frag_ids))
        if not section.get("claim_type"):
            has_evidence = bool(section.get("evidence_fragment_ids") or section.get("archive_visual_fragments") or section.get("speaker_soundbites"))
            section_type_val = section.get("section_type", "")
            if section_type_val == "evidence_gap_disclosure":
                section["claim_type"] = "evidence_gap"
            elif has_evidence:
                section["claim_type"] = "direct_evidence"
            else:
                section["claim_type"] = "synthesis"
        if not section.get("evidence_terms"):
            section_frags = [c for c in selected if c.get("fragment_id") in (section.get("evidence_fragment_ids") or [])]
            ev_terms = list(dict.fromkeys(
                t for c in section_frags
                for t in (c.get("direct_evidence_terms") or []) + (c.get("expanded_evidence_terms") or [])
            ))[:6]
            section["evidence_terms"] = ev_terms

    storyline = enrich_storyline_with_director_plan(storyline, director_plan)

    # Expand narrations that are shorter than the word budget
    storyline = expand_short_narrations(storyline, args.duration)

    # Sanitize all viewer-facing text before validation and save
    storyline, sanitize_notes = sanitize_viewer_facing_storyline_text(storyline, prompt_intent)
    if sanitize_notes:
        unrepairable = [n for n in sanitize_notes if n.startswith("viewer_facing_text_unrepairable")]
        repaired = [n for n in sanitize_notes if n.startswith("repaired:")]
        if repaired:
            log(f"Sanitized {len(repaired)} viewer-facing text field(s) of internal pipeline language.")
        if unrepairable:
            log(f"Warning: {len(unrepairable)} viewer-facing field(s) could not be fully repaired.")
            for note in unrepairable[:5]:
                log(f"  {note}")

    # Phase 4O: rebuild or repair the timeline so every fragment item has moment fields.
    # This replaces raw LLM timeline items unconditionally.
    storyline = repair_or_rebuild_timeline_with_moments(
        storyline=storyline,
        selected_candidates=selected,
        prompt_profile=prompt_intent,
        target_duration_seconds=args.duration,
        video_mode=director_plan.get("video_mode"),
    )

    # Strict pre-save validation (runs after sanitization + timeline repair)
    strict_issues = validate_storyline_before_save(
        storyline=storyline,
        prompt_profile=prompt_intent,
        selected_candidates=selected,
        video_mode=director_plan.get("video_mode", ""),
        evidence_strength=director_plan.get("evidence_strength", ""),
    )

    # Hard check: if any fragment item still lacks moment fields after repair, flag it
    final_timeline = storyline.get("export_timeline") or []
    if selected and timeline_requires_moment_repair(final_timeline):
        strict_issues.append("timeline_moment_repair_failed")

    if selected and not final_timeline:
        strict_issues.append("selected_candidates_present_but_export_timeline_empty")

    validation_issues = selection_issues + validate_fragment_references(storyline, valid_fragment_ids) + strict_issues

    if validation_issues:
        log("Validation issues found:")
        for issue in validation_issues[:20]:
            log(f"- {issue}")

    output_path = save_storyline(
        storyline=storyline,
        user_prompt=args.prompt,
        prompt_intent=prompt_intent,
        duration=args.duration,
        style=args.style,
        retrieved=retrieved,
        buckets=buckets,
        director_plan=director_plan,
        evidence_metrics=evidence_metrics,
        validation_issues=validation_issues,
        evidence_package=evidence_package,
        retrieval_diagnostics=retrieval_diagnostics,
        candidate_bucket_summary=candidate_bucket_summary,
        corpus_diagnostics=corpus_diagnostics,
    )

    print_summary(storyline, output_path)


if __name__ == "__main__":
    main()
