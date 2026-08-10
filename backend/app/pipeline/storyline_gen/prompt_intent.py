import json
import re
from pathlib import Path
from typing import Any

from app.core.paths import PROJECT_ROOT
from app.pipeline.storyline_gen.text_utils import (
    clean_text,
    normalize_name,
    normalize_plural,
)


STOPWORDS = {
    "create", "make", "generate", "video", "documentary", "minute", "minutes",
    "one", "two", "three", "on", "about", "what", "how", "why", "who", "all",
    "the", "a", "an", "and", "or", "of", "for", "to", "in", "with", "from",
    "by", "as", "is", "are", "was", "were", "be", "been", "being", "ai",
    "ai4ia", "conference", "theme", "themes",
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
    # NOTE: artificial/intelligence/information/technology/technologies/data/
    # content/system/systems/tool/tools were removed from this set after
    # scripts/diagnostics/stopword_frequency_report.py showed each below the
    # 55% corpus document-frequency threshold (0.0%-50.5%), unlike "ai"
    # itself (100% -- every fragment mentions it, so it carries zero
    # discriminating signal and correctly stays stopped). These words are
    # this archive's actual subject-matter vocabulary; stopping them could
    # zero out topic_terms entirely for on-topic prompts built from them
    # (see NO_TOPIC_TERMS_SEMANTIC_FLOOR in candidate_scoring.py for the
    # remaining fallback when topic_terms still ends up empty).
}

SPEAKER_PROMPT_STOPWORDS = {
    "create", "make", "generate", "video", "documentary", "minute", "minutes",
    "what", "who", "said", "spoke", "about", "all", "theme", "themes",
    "chapter", "chapters", "the", "a", "an", "and", "or", "of", "for", "to",
    "in", "on", "with", "from", "by", "ai",
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
        "expanded_terms": ["woman", "female", "girls", "gender", "agency", "participation", "leadership", "youth", "empowerment", "inclusion", "equity", "access", "rights", "community"],
        "strictness": "medium",
    },
    "empowerment": {
        "expanded_terms": ["woman", "female", "girls", "gender", "agency", "participation", "leadership", "youth", "empowerment", "inclusion", "equity", "access", "rights", "community"],
        "strictness": "medium",
    },
    "youth": {
        "expanded_terms": ["young", "children", "students", "generation", "future", "education", "participation", "empowerment", "opportunity", "youth"],
        "strictness": "medium",
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
