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


# Download filenames are "<NNNN>_<11-char youtube id>_<human title>" — see
# scripts/01_download_videos.py and extract_video_id in 04_create_fragments.py.
# The id is always exactly 11 characters (video_naming.py), so match it by
# length rather than splitting on "_", which mis-parses the ids that contain
# an underscore themselves.
_SOURCE_VIDEO_PREFIX_RE = re.compile(r"^\d+_[A-Za-z0-9_-]{11}_")


def video_title_from_source(record: dict[str, Any]) -> str:
    """Best available human title for the video a fragment came from.

    Fragment records carry the title only inside `source_video` (the download
    filename stem) — `video_title` is defined on FragmentRecord but is not
    populated in the embedding/Chroma metadata, so it is tried first and
    falls back to the filename stem with its index/id prefix stripped.
    """
    explicit = clean_text(str(record.get("video_title") or ""))
    if explicit:
        return explicit
    stem = str(record.get("source_video") or "")
    if not stem:
        return ""
    return clean_text(_SOURCE_VIDEO_PREFIX_RE.sub("", stem).replace("_", " "))


def title_term_overlap(title: str, topic_terms: list[str] | set[str]) -> tuple[list[str], float]:
    """(matched terms, matched fraction) for topic_terms against a video title.

    Both sides are singularized before comparison. match_terms_in_text, which
    the keyword channel uses, normalizes only the search terms — so a prompt
    term like "algorithm" (extract_topic_terms singularizes it) misses the
    literal "Algorithms" in a title. Titles are short enough that the looser
    match costs nothing here.
    """
    terms = [t for t in (topic_terms or []) if t]
    if not terms:
        return [], 0.0
    title_tokens = {normalize_plural(tok) for tok in re.findall(r"[a-z0-9]+", normalize_name(title))}
    matched = [term for term in terms if normalize_plural(normalize_name(term)) in title_tokens]
    return matched, len(matched) / len(terms)
