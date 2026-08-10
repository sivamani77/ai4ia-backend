import json
import re
from typing import Any


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


def normalize_plural(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def parse_list_field(value: Any) -> list:
    """Normalize a metadata list field regardless of source shape.

    The local JSONL path stores these as native Python lists. ChromaDB
    metadata only accepts scalars, so import_embeddings_to_chroma.py's
    sanitize_metadata JSON-encodes lists (e.g. clip_usage_candidates,
    keywords, theme_tags, chapter_tags) into strings. Treating that string
    as already-a-list (set(), [:6], " ".join()) silently corrupts it —
    set() splits into individual characters, slicing takes characters not
    items, join() interleaves every character with spaces. Older Chroma
    data may still use the legacy comma-joined format, so fall back to
    that if the value isn't valid JSON.
    """
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        return [c.strip() for c in value.split(",") if c.strip()]
    return []


def normalize_name(value: str) -> str:
    lowered = (value or "").lower().replace("_", " ").replace("-", " ")
    lowered = re.sub(r"[^a-z0-9\s]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()
