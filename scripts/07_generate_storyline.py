import argparse
import json
import re
import time
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
STORYLINES_DIR = PROJECT_ROOT / "data" / "storylines"

MATRIX_PATH = EMBEDDINGS_DIR / "fragment_embedding_matrix.npy"
JSONL_PATH = EMBEDDINGS_DIR / "classified_fragments_embeddings.jsonl"

STORYLINES_DIR.mkdir(parents=True, exist_ok=True)


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
    "empowerment": {"empowerment", "empower", "agency", "participation"},
    "empower": {"empowerment", "empower", "agency", "participation"},
    "healthcare": {"healthcare", "health", "medicine", "medical", "patient", "hospital", "doctor", "clinical"},
    "health": {"healthcare", "health", "medicine", "medical", "patient", "hospital", "doctor", "clinical"},
    "governance": {"governance", "policy", "regulation", "oversight", "accountability", "transparency", "responsible"},
    "ethics": {"ethics", "ethical", "fairness", "bias", "accountability", "transparency", "responsibility"},
    "ethical": {"ethics", "ethical", "fairness", "bias", "accountability", "transparency", "responsibility"},
    "youth": {"youth", "young", "students", "children", "generational"},
    "young": {"youth", "young", "students", "children", "generational"},
    "access": {"access", "accessibility"},
    "crypto": {"crypto", "cryptocurrency", "blockchain", "web3", "digital currency", "token", "defi"},
    "cryptocurrency": {"crypto", "cryptocurrency", "blockchain", "web3", "digital currency", "token"},
    "blockchain": {"crypto", "cryptocurrency", "blockchain", "web3"},
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
) -> dict[str, Any]:
    requested_speaker = detect_requested_speaker(prompt, known_speakers)
    topic_terms = extract_topic_terms(prompt, requested_speaker=requested_speaker)
    chapter = lookup_requested_chapter(prompt, records)
    theme = lookup_requested_theme(topic_terms, records)

    normalized_prompt = normalize_name(prompt)
    scope = "general"
    if requested_speaker and re.search(r"\ball themes?\b", normalized_prompt):
        scope = "all_themes"
    elif topic_terms or theme or chapter:
        scope = "specific_topic"

    if requested_speaker and scope == "all_themes":
        intent_type = "speaker_summary"
    elif requested_speaker:
        intent_type = "speaker_topic_summary"
    else:
        intent_type = "topic_summary"

    return {
        "intent_type": intent_type,
        "requested_speaker": requested_speaker,
        "topic_terms": sorted(topic_terms),
        "theme": theme,
        "chapter": chapter,
        "scope": scope,
    }


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
    usage_candidates = set(enriched.get("clip_usage_candidates") or [])

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

    # Classification logic
    if requested_speaker and scope == "all_themes":
        if chapter_match or theme_match or len(meaningful_overlap) >= 1 or semantic_score >= 0.58:
            role = "exact"
            reason = "Speaker matched; selected as one of the strongest clips across available themes."
            speaker_clip_allowed = (
                "speaker_soundbite" in usage_candidates
                and quote_strength >= 0.65
                and clarity_score >= 0.85
            )
        elif semantic_score >= 0.52:
            role = "strong_related"
            reason = f"Speaker matched with limited topic evidence; semantic {semantic_score:.3f}"
            speaker_clip_allowed = (
                "speaker_soundbite" in usage_candidates
                and quote_strength >= 0.70
                and clarity_score >= 0.85
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
            and quote_strength >= 0.65
            and clarity_score >= 0.85
        )
    elif theme_match and (len(meaningful_overlap) >= 1 or semantic_score >= 0.58):
        role = "exact"
        reason = "Speaker and requested theme matched."
        speaker_clip_allowed = (
            "speaker_soundbite" in usage_candidates
            and quote_strength >= 0.65
            and clarity_score >= 0.85
        )
    elif len(meaningful_overlap) >= 2 and semantic_score >= 0.62:
        role = "exact"
        reason = f"Strong meaningful overlap ({meaningful_overlap}) + semantic {semantic_score:.3f}"
        speaker_clip_allowed = (
            "speaker_soundbite" in usage_candidates
            and quote_strength >= 0.65
            and clarity_score >= 0.85
        )
    elif len(meaningful_overlap) >= 1 and semantic_score >= 0.66:
        role = "exact"
        reason = f"Meaningful overlap ({meaningful_overlap}) + high semantic {semantic_score:.3f}"
        speaker_clip_allowed = (
            "speaker_soundbite" in usage_candidates
            and quote_strength >= 0.65
            and clarity_score >= 0.85
        )
    elif len(expanded_overlap) >= 2 and semantic_score >= 0.65:
        role = "exact"
        reason = f"Non-generic expanded overlap ({expanded_overlap}) + semantic {semantic_score:.3f}"
        speaker_clip_allowed = (
            "speaker_soundbite" in usage_candidates
            and quote_strength >= 0.65
            and clarity_score >= 0.85
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
            and quote_strength >= 0.70
            and clarity_score >= 0.85
        )
    elif (total_theme >= 1 or total_meaningful >= 1 or len(expanded_overlap) >= 1) and semantic_score >= 0.62:
        role = "strong_related"
        reason = f"Topic/theme overlap at moderate semantic {semantic_score:.3f}: meaningful={meaningful_overlap}, theme={theme_overlap}, expanded={expanded_overlap[:3]}"
        speaker_clip_allowed = (
            "speaker_soundbite" in usage_candidates
            and quote_strength >= 0.70
            and clarity_score >= 0.85
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

    # speaker_soundbite_allowed = strict (original speaker_clip_allowed logic)
    speaker_soundbite_allowed = speaker_clip_allowed

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
        record.get("primary_theme", ""),
        record.get("primary_chapter", ""),
    ]
    return clean_text(" ".join(str(part) for part in parts if part)).lower()


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

    prompt_terms = set(intent.get("topic_terms") or []) or extract_meaningful_prompt_terms(prompt)
    expanded_terms = expand_prompt_terms(prompt_terms)
    requested_speaker = intent.get("requested_speaker")

    if requested_speaker:
        log(f"Speaker constraint detected: '{requested_speaker}'")
    if prompt_terms:
        log(f"Meaningful prompt terms: {sorted(prompt_terms)}")
    if expanded_terms - prompt_terms:
        log(f"Expanded synonym terms: {sorted(expanded_terms - prompt_terms)}")

    annotated = []
    for record in records:
        enriched = classify_candidate_relevance(record, prompt_terms, expanded_terms, intent)
        # Keep backward-compat field
        enriched["prompt_keyword_overlap"] = enriched.get("meaningful_prompt_overlap", [])
        enriched["prompt_keyword_overlap_count"] = len(enriched["prompt_keyword_overlap"])
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
            role_priority = 2
        elif role == "strong_related":
            role_priority = 1
        return (
            1 if c.get("speaker_soundbite_allowed") or c.get("speaker_clip_allowed") else 0,
            1 if c.get("speaker_visual_allowed") else 0,
            role_priority,
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
    if not MATRIX_PATH.exists():
        raise FileNotFoundError(f"Missing embedding matrix: {MATRIX_PATH}")

    if not JSONL_PATH.exists():
        raise FileNotFoundError(f"Missing embedding JSONL: {JSONL_PATH}")

    matrix = np.load(MATRIX_PATH)
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
    records = []

    with JSONL_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                # Keep output lightweight; matrix already stores embeddings.
                record.pop("embedding", None)
                record.pop("embedding_text", None)
                records.append(record)

    return records


def score_and_retrieve(prompt: str, top_k: int) -> list[dict[str, Any]]:
    return score_all_records(prompt)[:top_k]


def retrieve_candidates(
    prompt: str,
    intent: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    scored = score_all_records(prompt)
    requested_speaker = intent.get("requested_speaker")

    if not requested_speaker:
        # Hybrid retrieval: semantic top-K + keyword/chapter-boosted candidates
        semantic_pool = scored[:top_k]
        semantic_ids = {r.get("fragment_id") for r in semantic_pool}

        prompt_terms = set(intent.get("topic_terms") or [])
        expanded_terms = expand_prompt_terms(prompt_terms)
        # Combined search terms: prompt + non-generic expanded
        search_terms = prompt_terms | (expanded_terms - GENERIC_EXPANSION_TERMS)

        if search_terms:
            boost_pool: list[dict[str, Any]] = []
            for record in scored:
                if record.get("fragment_id") in semantic_ids:
                    continue
                blob = candidate_search_blob(record)
                blob_tokens = set(re.findall(r"[a-z0-9]+", blob))
                chapter_tokens = set(re.findall(r"[a-z0-9]+", (record.get("primary_chapter") or "").lower()))
                keyword_tokens = set(re.findall(r"[a-z0-9]+", " ".join(record.get("keywords") or []).lower()))
                # Count matches across blob, chapter, keywords
                text_hits = sum(1 for t in search_terms if t in blob_tokens)
                chapter_hits = sum(1 for t in search_terms if t in chapter_tokens)
                kw_hits = sum(1 for t in search_terms if t in keyword_tokens)
                total_hits = text_hits + chapter_hits * 2 + kw_hits
                if total_hits >= 1 and float(record.get("semantic_score") or 0) >= 0.55:
                    record = dict(record)
                    record["_boost_hits"] = total_hits
                    boost_pool.append(record)

            # Sort boost pool by hits then semantic score, take best
            boost_pool.sort(key=lambda r: (r.get("_boost_hits", 0), float(r.get("semantic_score") or 0)), reverse=True)
            # Merge up to top_k//2 boosted candidates
            boost_limit = top_k // 2
            combined = semantic_pool + boost_pool[:boost_limit]
            seen = set()
            deduped = []
            for r in combined:
                fid = r.get("fragment_id")
                if fid not in seen:
                    seen.add(fid)
                    deduped.append(r)
            log(f"Hybrid retrieval: {len(semantic_pool)} semantic + {min(len(boost_pool), boost_limit)} keyword-boosted = {len(deduped)} total")
            return deduped

        return semantic_pool

    speaker_pool = []
    for record in scored:
        speaker_match, speaker_match_reason = candidate_speaker_matches(record, requested_speaker)
        if not speaker_match:
            continue
        enriched = dict(record)
        enriched["requested_speaker"] = requested_speaker
        enriched["speaker_match"] = True
        enriched["speaker_match_reason"] = speaker_match_reason
        speaker_pool.append(enriched)

    if intent.get("scope") == "all_themes":
        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in speaker_pool:
            bucket = str(record.get("primary_chapter") or record.get("primary_theme") or "speaker_misc")
            grouped.setdefault(bucket, []).append(record)
        diversified: list[dict[str, Any]] = []
        for bucket in grouped.values():
            ranked_bucket = sorted(
                bucket,
                key=lambda item: (
                    float(item.get("quote_strength") or 0.0),
                    float(item.get("clarity_score") or 0.0),
                    float(item.get("semantic_score") or 0.0),
                ),
                reverse=True,
            )
            diversified.extend(ranked_bucket[:2])
        speaker_pool = sorted(
            diversified,
            key=lambda item: (
                float(item.get("quote_strength") or 0.0),
                float(item.get("clarity_score") or 0.0),
                float(item.get("semantic_score") or 0.0),
            ),
            reverse=True,
        )

    return speaker_pool[:top_k]


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


def select_balanced_candidates(retrieved: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    for record in retrieved:
        record["selection_score"] = round(
            float(record["semantic_score"]) + utility_boost(record),
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

        if used_count >= MAX_SOURCE_VIDEO_OCCURRENCES:
            continue

        if requested_speaker and source_video == "unknown" and record.get("speaker_match") is not True:
            continue

        if theme_count >= (2 if requested_speaker else 10):
            continue

        if speaker != "unknown" and speaker_count >= MAX_SPEAKER_OCCURRENCES:
            continue

        selected.append(record)
        used_videos[source_video] = used_count + 1
        theme_counts[theme] = theme_count + 1
        used_speakers[speaker] = speaker_count + 1

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
    packed = []

    for item in candidates:
        packed.append(
            {
                "fragment_id": item.get("fragment_id"),
                "source_video": item.get("source_video"),
                "video_id": item.get("video_id"),
                "start": item.get("start"),
                "end": item.get("end"),
                "duration": item.get("duration"),
                "primary_theme": item.get("primary_theme"),
                "primary_chapter": item.get("primary_chapter"),
                "summary": item.get("summary"),
                "keywords": item.get("keywords", [])[:10],
                "clip_usage_candidates": item.get("clip_usage_candidates", []),
                "quote_strength": item.get("quote_strength"),
                "clarity_score": item.get("clarity_score"),
                "narration_support_score": item.get("narration_support_score"),
                "semantic_score": item.get("semantic_score"),
                "prompt_relevance_score": item.get("prompt_relevance_score"),
                "selection_score": item.get("selection_score"),
                "relevance_role": item.get("relevance_role"),
                "prompt_keyword_overlap": item.get("prompt_keyword_overlap", [])[:8],
                "requested_speaker": item.get("requested_speaker"),
                "speaker_match": item.get("speaker_match"),
                "speaker_match_reason": item.get("speaker_match_reason"),
                "speaker_soundbite_allowed": item.get("speaker_soundbite_allowed", False),
                "speaker_visual_allowed": item.get("speaker_visual_allowed", False),
                "speaker_clip_allowed": item.get("speaker_clip_allowed", False),
                "broll_allowed": item.get("broll_allowed", True),
                "relevance_reason": item.get("relevance_reason", ""),
                "meaningful_prompt_overlap": item.get("meaningful_prompt_overlap", []),
                "final_video_allowed": item.get("final_video_allowed", False),
                "use_as": item.get("use_as", ""),
                "text": truncate_text(item.get("text", ""), 700),
            }
        )

    return packed


def generate_llm_storyline(
    user_prompt: str,
    duration: int,
    candidates: list[dict[str, Any]],
    style: str,
    director_plan: dict[str, Any],
    prompt_intent: dict[str, Any],
) -> dict[str, Any]:
    candidate_pack = build_candidate_pack(candidates)

    system_instruction = """
You are a senior documentary producer and AI research editor.

Create a production-ready storyline blueprint for a documentary-style video.
The source archive contains conference footage, speaker clips, panel discussions, and event visuals.
The final video should feel like a National Geographic / premium explainer documentary:
- narrator-led
- evidence-backed
- visually planned
- speaker soundbites used selectively
- generated or internet B-roll used where conference footage is insufficient

Important rules:
1. Do not invent speaker names unless they are visible in source_video text.
2. Use only fragment_id values from the provided candidate fragments.
3. Do not assign a permanent storyline_role to fragments.
4. Decide how each fragment is used only for this specific user prompt.
5. Separate real archive clips from generated/internet assets.
6. Keep narration cinematic but factual.
7. Return valid JSON only. No markdown. No backticks.
"""

    user_instruction = {
        "user_prompt": user_prompt,
        "prompt_intent": prompt_intent,
        "target_duration_seconds": duration,
        "style": style,
        "director_plan": director_plan,
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
            "num_ctx": 24000
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


def build_fallback_storyline(
    user_prompt: str,
    duration: int,
    candidates: list[dict[str, Any]],
    style: str,
    director_plan: dict[str, Any],
    prompt_intent: dict[str, Any],
) -> dict[str, Any]:
    soundbites = [
        item for item in candidates
        if "speaker_soundbite" in item.get("clip_usage_candidates", [])
    ][:8]

    visuals = candidates[:18]

    requested_speaker = prompt_intent.get("requested_speaker")
    title = f"{requested_speaker}: Speaker Summary" if requested_speaker else "AI for Inclusive Access to Information"

    sections = [
        ("S1", "opening_hook", "Artificial intelligence is no longer a distant idea. It is already shaping how people learn, communicate, govern and access knowledge."),
        ("S2", "context", "Across the AI4IA archive, speakers return to a shared question: how can AI expand access to information without deepening existing divides?"),
        ("S3", "problem_or_tension", "The promise of AI is powerful, but the risks are equally real: exclusion, weak infrastructure, language gaps, bias and loss of trust."),
        ("S4", "human_perspective", "The human story is about young people, communities, educators, policymakers and citizens trying to shape technology around real needs."),
        ("S5", "expert_evidence", "The archive shows experts connecting AI to governance, rights, media literacy, accessibility, local knowledge and sustainable development."),
        ("S6", "solution_direction", "The way forward is not only better tools. It is inclusive design, local data, ethical governance and public participation."),
        ("S7", "final_synthesis", "AI can become a bridge to knowledge, but only if societies choose to build it with transparency, trust and inclusion at the center."),
    ]

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
    role_sequence = mode_to_roles.get(director_plan.get("video_mode"), mode_to_roles["narration_led_explainer"])

    if requested_speaker and not candidates:
        return {
            "title": title,
            "logline": f"No archive evidence was found for {requested_speaker}.",
            "style": style,
            "target_duration_seconds": duration,
            "estimated_duration_seconds": max(20, min(duration, 45)),
            "director_plan": director_plan,
            "narrative_arc": ["opening_context", "evidence_gap_narration", "closing_takeaway"],
            "sections": [],
            "voiceover_script": [],
            "selected_speaker_soundbites": [],
            "asset_plan": {
                "real_archive_assets": [],
                "generated_assets": [],
                "internet_assets": [],
                "voiceover_required": True,
                "subtitles_required": True,
                "background_music_required": True
            },
            "export_timeline": [],
            "editor_notes": [f"No archive clips were found for {requested_speaker}. No unrelated speakers were used as filler."],
        }

    built_sections = []

    for idx, (section_id, section_type, narration) in enumerate(sections):
        section_visuals = visuals[idx * 2: idx * 2 + 2]
        section_soundbites = soundbites[idx: idx + 1]
        section_role = role_sequence[idx] if idx < len(role_sequence) else "related_theme_clip"
        evidence_role = "exact" if section_soundbites else ("related" if section_visuals else "low_evidence")
        evidence_confidence = "high" if section_soundbites else ("medium" if section_visuals else "low")

        built_sections.append(
            {
                "section_id": section_id,
                "section_type": section_type,
                "target_duration_seconds": round(duration / len(sections)),
                "director_role": section_role,
                "evidence_role": evidence_role,
                "evidence_confidence": evidence_confidence,
                "fallback_allowed": evidence_role != "exact",
                "planning_note": "Uses related evidence and narration support because exact prompt matches were limited." if evidence_role != "exact" else "Anchored by a higher-confidence speaker clip.",
                "narration": narration,
                "archive_visual_fragments": [
                    {
                        "fragment_id": item["fragment_id"],
                        "usage": "context_visual",
                        "start": item.get("start"),
                        "end": item.get("end"),
                        "reason": item.get("summary", "Relevant archive visual.")
                    }
                    for item in section_visuals
                ],
                "speaker_soundbites": [
                    {
                        "fragment_id": item["fragment_id"],
                        "start": item.get("start"),
                        "end": item.get("end"),
                        "quote_summary": item.get("summary"),
                        "reason": "Relevant speaker soundbite for this section."
                    }
                    for item in section_soundbites
                ],
                "generated_assets_needed": [
                    {
                        "asset_type": "transition_card",
                        "description": f"Minimal documentary transition card for {section_type}.",
                        "on_screen_text": section_type.replace("_", " ").title()
                    }
                ],
                "internet_assets_needed": [
                    {
                        "asset_type": "b_roll",
                        "description": "Documentary B-roll matching the section theme.",
                        "search_keywords": ["artificial intelligence", "digital inclusion", "access to information"]
                    }
                ]
            }
        )

    return {
        "title": title,
        "logline": "A documentary-style video exploring how AI can expand access to information while raising urgent questions of inclusion, trust, and governance.",
        "style": style,
        "target_duration_seconds": duration,
        "estimated_duration_seconds": duration,
        "director_plan": director_plan,
        "narrative_arc": [item[1] for item in sections],
        "sections": built_sections,
        "voiceover_script": [
            {
                "section_id": section_id,
                "text": narration,
                "tone": "calm documentary narrator"
            }
            for section_id, _, narration in sections
        ],
        "selected_speaker_soundbites": [
            {
                "fragment_id": item["fragment_id"],
                "source_video": item.get("source_video"),
                "start": item.get("start"),
                "end": item.get("end"),
                "quote_summary": item.get("summary"),
                "use_in_section": f"S{idx + 1}"
            }
            for idx, item in enumerate(soundbites[:7])
        ],
        "asset_plan": {
            "real_archive_assets": [item["fragment_id"] for item in visuals[:18]],
            "generated_assets": ["title cards", "transition cards", "map cards", "abstract AI visuals"],
            "internet_assets": ["youth coding B-roll", "AI infrastructure visuals", "public information access visuals"],
            "voiceover_required": True,
            "subtitles_required": True,
            "background_music_required": True
        },
        "export_timeline": [],
        "editor_notes": [
            "Fallback storyline generated without LLM JSON output.",
            "Use retrieved speaker clips as authentic evidence and generated/internet B-roll for missing visuals."
        ],
        "planning_summary": {
            "video_mode": director_plan.get("video_mode"),
            "evidence_strength": director_plan.get("evidence_strength"),
            "exact_clip_count": director_plan.get("exact_clip_count"),
            "related_clip_count": director_plan.get("related_clip_count"),
            "fallback_risk": director_plan.get("fallback_risk"),
            "planning_notes": director_plan.get("planning_notes", []),
        },
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
    candidate_bucket_summary: dict[str, Any] | None = None,
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
        "prompt_intent": prompt_intent,
        "target_duration_seconds": duration,
        "style": style,
        "models": {
            "embedding_model": EMBED_MODEL,
            "strong_llm_model": STRONG_LLM_MODEL,
            "fast_llm_model": FAST_LLM_MODEL,
        },
        "source_files": {
            "embedding_matrix": str(MATRIX_PATH),
            "embedding_records": str(JSONL_PATH),
        },
        "retrieval_summary": {
            "retrieved_count": len(retrieved),
            "selected_candidate_count": len(selected),
            "supporting_candidate_count": len(supporting),
            "context_candidate_count": len(context),
            "rejected_candidate_count": len(rejected),
            "top_semantic_score": retrieved[0]["semantic_score"] if retrieved else None,
        },
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
        "selected_candidates": build_candidate_pack(selected),
        "supporting_candidates": build_candidate_pack(supporting[:10]),
        "context_candidates": build_candidate_pack(context[:5]),
        "rejected_candidates": build_candidate_pack(rejected[:5]),
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

    args = parser.parse_args()

    log("Starting Step 07: Generate Documentary Storyline")
    log(f"Prompt: {args.prompt}")
    log(f"Duration: {args.duration}")
    log(f"Style: {args.style}")
    log(f"Top K: {args.top_k}")
    log(f"Final candidates: {args.final_candidates}")

    warm_up_ollama()
    known_speakers = load_known_speakers()
    log(f"Loaded {len(known_speakers)} known speakers.")
    all_records = load_embedding_records()
    prompt_intent = detect_prompt_intent(args.prompt, known_speakers, all_records)
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

    # Build evidence-gap notes and append to planning notes
    prompt_terms = set(prompt_intent.get("topic_terms") or [])
    gap_notes = build_evidence_gap_notes(prompt_intent, prompt_terms, evidence_metrics)
    existing_notes = director_plan.get("planning_notes", [])
    merged_notes = existing_notes[:]
    for note in gap_notes:
        if note not in merged_notes:
            merged_notes.append(note)
    director_plan["planning_notes"] = merged_notes

    valid_fragment_ids = {item.get("fragment_id") for item in selected}

    log(f"Selected candidates after cap: {len(selected)}")
    log(f"Evidence: exact={evidence_metrics['exact_clip_count']} strong={evidence_metrics['strong_related_count']} supporting={evidence_metrics.get('supporting_candidate_count', 0)} usable_speaker={evidence_metrics['usable_speaker_clip_count']}")

    storyline = None

    if not args.fallback_only:
        try:
            storyline = generate_llm_storyline(
                user_prompt=args.prompt,
                duration=args.duration,
                candidates=selected,
                style=args.style,
                director_plan=director_plan,
                prompt_intent=prompt_intent,
            )
        except Exception as exc:
            log(f"Strong LLM storyline failed: {exc}")
            log("Using fallback storyline.")

    if storyline is None:
        storyline = build_fallback_storyline(
            user_prompt=args.prompt,
            duration=args.duration,
            candidates=selected,
            style=args.style,
            director_plan=director_plan,
            prompt_intent=prompt_intent,
        )

    storyline = enrich_storyline_with_director_plan(storyline, director_plan)

    validation_issues = selection_issues + validate_fragment_references(storyline, valid_fragment_ids)

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
        candidate_bucket_summary=candidate_bucket_summary,
    )

    print_summary(storyline, output_path)


if __name__ == "__main__":
    main()
