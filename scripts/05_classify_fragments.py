import csv
import json
import math
import re
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Local dev layout: backend/app/. Docker image layout (see Dockerfile): app/
# directly under PROJECT_ROOT, with no backend/ directory at all.
_BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR if (_BACKEND_DIR / "app").is_dir() else PROJECT_ROOT))

from app.core.taxonomy_loader import THEMES, CHAPTERS  # noqa: E402

FRAGMENT_DIR = PROJECT_ROOT / "data" / "fragments_cleaned"
CLASSIFIED_DIR = PROJECT_ROOT / "data" / "fragments_classified"
SPEAKERS_METADATA_DIR = PROJECT_ROOT / "data" / "metadata" / "speakers"
REPORT_PATH = PROJECT_ROOT / "data" / "metadata" / "youtube" / "fragment_classification_report.csv"
CATALOG_PATH = PROJECT_ROOT / "data" / "metadata" / "youtube" / "classified_fragments_catalog.jsonl"

CLASSIFIED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "have", "has", "are", "was",
    "were", "you", "your", "our", "not", "but", "can", "will", "they", "their",
    "about", "into", "also", "more", "when", "what", "how", "why", "who", "where",
    "which", "there", "been", "being", "very", "much", "some", "such", "than",
    "then", "them", "these", "those", "would", "could", "should", "just", "like",
    "because", "through", "within", "without", "between", "across", "today", "thank",
    "thanks", "please", "ladies", "gentlemen"
}


IMPORTANT_PHRASES = [
    "artificial intelligence",
    "access to information",
    "information accessibility",
    "open data",
    "global south",
    "digital divide",
    "ai literacy",
    "human rights",
    "public trust",
    "information integrity",
    "local language",
    "data sovereignty",
    "national ai strategy",
    "media literacy",
    "deepfakes",
    "disinformation",
    "misinformation",
    "youth",
    "ethics",
    "governance",
    "policy",
    "regulation",
    "accessibility",
    "education",
    "health",
    "environment",
    "agriculture",
    "law",
    "copyright",
]


def log(message: str) -> None:
    print(message, flush=True)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z\-']{2,}", text.lower())
    return [word.strip("-'") for word in words if word not in STOPWORDS and len(word) > 2]


def phrase_count(text: str, phrase: str) -> int:
    return len(re.findall(r"\b" + re.escape(phrase.lower()) + r"\b", text.lower()))


def keyword_score(text: str, keywords: list[str]) -> tuple[float, list[str]]:
    text_norm = normalize(text)
    tokens = tokenize(text_norm)
    token_counter = Counter(tokens)

    score = 0.0
    matched = []

    for kw in keywords:
        kw_norm = kw.lower().strip()
        if " " in kw_norm or "-" in kw_norm:
            count = phrase_count(text_norm, kw_norm)
            if count > 0:
                score += count * 3.0
                matched.append(kw)
        else:
            count = token_counter.get(kw_norm, 0)
            if count > 0:
                score += count * 1.0
                matched.append(kw)

    word_count = max(len(tokens), 1)
    normalized_score = score / math.sqrt(word_count)

    return normalized_score, matched


def score_themes(text: str) -> list[dict]:
    raw_scores = []

    for theme_id, theme in THEMES.items():
        score, matched = keyword_score(text, theme["keywords"])
        if score > 0:
            raw_scores.append(
                {
                    "theme_id": theme_id,
                    "theme": theme["name"],
                    "score": score,
                    "matched_keywords": matched[:12],
                }
            )

    raw_scores.sort(key=lambda item: item["score"], reverse=True)

    if not raw_scores:
        return [
            {
                "theme_id": "T2",
                "theme": THEMES["T2"]["name"],
                "score": 0.15,
                "matched_keywords": [],
            }
        ]

    max_score = raw_scores[0]["score"]

    for item in raw_scores:
        item["score"] = round(min(item["score"] / max_score, 1.0), 3)

    return raw_scores[:3]


def score_chapters(text: str) -> list[dict]:
    raw_scores = []

    for chapter_id, chapter in CHAPTERS.items():
        score, matched = keyword_score(text, chapter["keywords"])
        if score > 0:
            theme_id = chapter["theme_id"]
            raw_scores.append(
                {
                    "chapter_id": chapter_id,
                    "chapter": f"{chapter_id}: {chapter['name']}",
                    "theme_id": theme_id,
                    "theme": THEMES[theme_id]["name"],
                    "score": score,
                    "matched_keywords": matched[:12],
                }
            )

    raw_scores.sort(key=lambda item: item["score"], reverse=True)

    if not raw_scores:
        return [
            {
                "chapter_id": "Ch 8",
                "chapter": "Ch 8: Regional & Satellite Conference Events",
                "theme_id": "T2",
                "theme": THEMES["T2"]["name"],
                "score": 0.15,
                "matched_keywords": [],
            }
        ]

    max_score = raw_scores[0]["score"]

    for item in raw_scores:
        item["score"] = round(min(item["score"] / max_score, 1.0), 3)

    return raw_scores[:5]


def extract_keywords(text: str, theme_tags: list[dict], chapter_tags: list[dict]) -> list[str]:
    text_norm = normalize(text)
    found = []

    for phrase in IMPORTANT_PHRASES:
        if phrase_count(text_norm, phrase) > 0:
            found.append(phrase)

    for tag in theme_tags:
        found.extend(tag.get("matched_keywords", []))

    for tag in chapter_tags:
        found.extend(tag.get("matched_keywords", []))

    tokens = tokenize(text_norm)
    frequent = [
        word for word, count in Counter(tokens).most_common(20)
        if count >= 2 and word not in found
    ]

    combined = []
    seen = set()

    for item in found + frequent:
        clean = item.strip().lower()
        if clean and clean not in seen:
            seen.add(clean)
            combined.append(item)

    return combined[:15]


def create_summary(text: str, max_words: int = 36) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", clean)

    selected = ""

    for sentence in sentences:
        if len(sentence.split()) >= 8:
            selected = sentence
            break

    if not selected:
        selected = clean

    words = selected.split()

    if len(words) <= max_words:
        return selected.strip()

    return " ".join(words[:max_words]).strip() + "..."


def estimate_clarity(text: str, duration: float, word_count: int) -> float:
    if not text.strip():
        return 0.0

    words_per_second = word_count / max(duration, 1)
    avg_sentence_length = word_count / max(len(re.split(r"[.!?]+", text)), 1)

    score = 0.65

    if 1.4 <= words_per_second <= 3.2:
        score += 0.15
    elif words_per_second > 4.0:
        score -= 0.15

    if 10 <= avg_sentence_length <= 35:
        score += 0.1
    elif avg_sentence_length > 50:
        score -= 0.1

    filler_terms = [
        "um", "uh", "you know", "sort of", "kind of", "okay", "right", "basically"
    ]
    filler_count = sum(phrase_count(text, term) for term in filler_terms)

    if filler_count > 6:
        score -= 0.15
    elif filler_count <= 2:
        score += 0.05

    return round(max(0.0, min(score, 1.0)), 3)


def estimate_quote_strength(text: str) -> float:
    text_norm = normalize(text)
    word_count = len(text.split())

    score = 0.30

    # Penalize intro/outro filler — not quotable as soundbites
    filler_openers = [
        "thank you", "thanks very much", "good morning", "good afternoon",
        "good evening", "it's a pleasure", "welcome to", "let me introduce",
        "our next speaker", "i would like to thank", "before i begin",
        "i'm delighted", "i'm honored", "thank you so much",
    ]
    if any(phrase_count(text_norm, p) > 0 for p in filler_openers):
        score -= 0.15

    # Strong claim signals — speaker asserting something specific and personal
    strong_claim_phrases = [
        "i argue", "i believe", "i am convinced", "we found", "our research",
        "the data shows", "the evidence shows", "this is not acceptable",
        "we cannot", "we must not", "i have seen", "i have witnessed",
        "the reality is", "the truth is", "what we know is",
        "this means that", "the consequence is",
    ]
    strong_hits = sum(1 for p in strong_claim_phrases if phrase_count(text_norm, p) > 0)
    score += min(strong_hits * 0.10, 0.25)

    # Moderate claim signals — declarative but less specific
    moderate_claim_phrases = [
        "we need", "we must", "we should", "it is essential", "it is critical",
        "the challenge is", "the opportunity is", "the question is",
        "not just", "more importantly", "most importantly",
        "this is why", "that is why",
    ]
    mod_hits = sum(1 for p in moderate_claim_phrases if phrase_count(text_norm, p) > 0)
    score += min(mod_hits * 0.06, 0.15)

    # Rhetorical question — engaging and quotable
    if "?" in text:
        score += 0.08

    # Concrete evidence: numbers, percentages, scale
    concrete_terms = ["percent", "million", "billion", "thousand", "majority", "minority"]
    if any(t in text_norm for t in concrete_terms):
        score += 0.06

    # Word count sweet spot for a standalone soundbite (50–150 words ideal)
    if 50 <= word_count <= 150:
        score += 0.12
    elif 35 <= word_count < 50 or 150 < word_count <= 200:
        score += 0.06

    return round(max(0.0, min(score, 1.0)), 3)


def estimate_narration_support(text: str, chapter_tags: list[dict]) -> float:
    text_norm = normalize(text)

    score = 0.35

    if len(chapter_tags) >= 2:
        score += 0.15

    if any(term in text_norm for term in [
        "context", "broader", "agenda", "foundation", "framework",
        "society", "communities", "future", "development", "access"
    ]):
        score += 0.2

    if any(term in text_norm for term in [
        "for example", "such as", "in other words", "this means", "therefore"
    ]):
        score += 0.1

    return round(max(0.0, min(score, 1.0)), 3)


def determine_clip_usage(text: str, quote_strength: float, clarity_score: float, narration_support_score: float) -> list[str]:
    text_norm = normalize(text)
    usage = []

    if quote_strength >= 0.35 and clarity_score >= 0.55:
        usage.append("speaker_soundbite")

    if narration_support_score >= 0.6:
        usage.append("narration_support")

    if any(term in text_norm for term in [
        "for example", "evidence", "data", "research", "study", "statistics",
        "we have seen", "we saw", "in zimbabwe", "in jamaica", "in africa"
    ]):
        usage.append("evidence_clip")

    if any(term in text_norm for term in [
        "welcome", "conference", "event", "panel", "speaker", "session",
        "ladies and gentlemen", "thank you"
    ]):
        usage.append("context_visual")

    if any(term in text_norm for term in [
        "we need", "we must", "call to action", "recommendations", "future",
        "should", "must ensure", "let us"
    ]):
        usage.append("synthesis_candidate")

    if not usage:
        usage.append("archive_visual")

    return usage[:4]


def load_speaker_lookup() -> dict[str, dict]:
    """video_id/source_video -> speaker roster record from 04b_extract_speakers.py output."""
    lookup: dict[str, dict] = {}

    if not SPEAKERS_METADATA_DIR.exists():
        return lookup

    for path in sorted(SPEAKERS_METADATA_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        source_video = data.get("source_video")
        video_id = data.get("video_id")

        if source_video:
            lookup[str(source_video)] = data
        if video_id:
            lookup[str(video_id)] = data

    return lookup


def classify_fragment(fragment: dict) -> dict:
    text = fragment.get("text", "")
    duration = float(fragment.get("duration", 0) or 0)
    word_count = int(fragment.get("word_count", 0) or 0)

    theme_tags = score_themes(text)
    chapter_tags = score_chapters(text)
    keywords = extract_keywords(text, theme_tags, chapter_tags)
    summary = create_summary(text)

    clarity_score = estimate_clarity(text, duration, word_count)
    quote_strength = estimate_quote_strength(text)
    narration_support_score = estimate_narration_support(text, chapter_tags)
    clip_usage_candidates = determine_clip_usage(
        text=text,
        quote_strength=quote_strength,
        clarity_score=clarity_score,
        narration_support_score=narration_support_score,
    )

    top_theme_score = theme_tags[0]["score"] if theme_tags else 0
    top_chapter_score = chapter_tags[0]["score"] if chapter_tags else 0
    classification_confidence = round((top_theme_score + top_chapter_score + clarity_score) / 3, 3)

    updated = dict(fragment)

    # Remove old fixed placeholders from Step 04 if present.
    updated.pop("theme", None)
    updated.pop("chapter", None)
    updated.pop("storyline_role", None)
    updated.pop("persona_relevance", None)
    updated.pop("classification_confidence", None)

    updated.update(
        {
            "theme_tags": theme_tags,
            "chapter_tags": chapter_tags,
            "primary_theme": theme_tags[0]["theme"] if theme_tags else None,
            "primary_chapter": chapter_tags[0]["chapter"] if chapter_tags else None,
            "keywords": keywords,
            "summary": summary,
            "clip_usage_candidates": clip_usage_candidates,
            "quote_strength": quote_strength,
            "clarity_score": clarity_score,
            "narration_support_score": narration_support_score,
            "classification_confidence": classification_confidence,
            "classification_method": "rule_based_mvp_v1",
            "classified_at": datetime.utcnow().isoformat(),
        }
    )

    return updated


def classify_fragment_file(path: Path, speaker_lookup: dict[str, dict] | None = None) -> tuple[dict, dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    speaker_lookup = speaker_lookup if speaker_lookup is not None else load_speaker_lookup()
    speaker_meta = (
        speaker_lookup.get(str(payload.get("source_video")))
        or speaker_lookup.get(str(payload.get("video_id")))
        or {}
    )
    fallback_speaker = speaker_meta.get("primary_speaker")
    fallback_speaker_attribution = speaker_meta.get("attribution")

    fragments = payload.get("fragments", [])
    classified_fragments = [classify_fragment(fragment) for fragment in fragments]

    if fallback_speaker:
        for fragment in classified_fragments:
            if not fragment.get("speaker"):
                fragment["speaker"] = fallback_speaker
            if not fragment.get("speaker_attribution"):
                fragment["speaker_attribution"] = fallback_speaker_attribution

    classified_payload = dict(payload)
    classified_payload["classification_method"] = "rule_based_mvp_v1"
    classified_payload["classified_at"] = datetime.utcnow().isoformat()
    classified_payload["fragments"] = classified_fragments

    theme_counter = Counter()
    chapter_counter = Counter()
    usage_counter = Counter()

    for fragment in classified_fragments:
        if fragment.get("primary_theme"):
            theme_counter[fragment["primary_theme"]] += 1
        if fragment.get("primary_chapter"):
            chapter_counter[fragment["primary_chapter"]] += 1
        for usage in fragment.get("clip_usage_candidates", []):
            usage_counter[usage] += 1

    report_row = {
        "fragment_file": path.name,
        "status": "classified",
        "fragment_count": len(classified_fragments),
        "top_theme": theme_counter.most_common(1)[0][0] if theme_counter else "",
        "top_chapter": chapter_counter.most_common(1)[0][0] if chapter_counter else "",
        "speaker_soundbite_count": usage_counter.get("speaker_soundbite", 0),
        "narration_support_count": usage_counter.get("narration_support", 0),
        "evidence_clip_count": usage_counter.get("evidence_clip", 0),
        "context_visual_count": usage_counter.get("context_visual", 0),
        "error": "",
    }

    return classified_payload, report_row


def write_report(rows: list[dict]) -> None:
    fieldnames = [
        "fragment_file",
        "status",
        "fragment_count",
        "top_theme",
        "top_chapter",
        "speaker_soundbite_count",
        "narration_support_count",
        "evidence_clip_count",
        "context_visual_count",
        "error",
    ]

    with REPORT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log(f"Report written to: {REPORT_PATH}")


def write_catalog(classified_payloads: list[dict]) -> None:
    with CATALOG_PATH.open("w", encoding="utf-8") as f:
        for payload in classified_payloads:
            for fragment in payload.get("fragments", []):
                record = {
                    "fragment_id": fragment.get("fragment_id"),
                    "source_video": fragment.get("source_video"),
                    "video_id": fragment.get("video_id"),
                    "video_year": fragment.get("video_year"),
                    "video_upload_date": fragment.get("video_upload_date"),
                    "start": fragment.get("start"),
                    "end": fragment.get("end"),
                    "duration": fragment.get("duration"),
                    "word_count": fragment.get("word_count"),
                    "primary_theme": fragment.get("primary_theme"),
                    "primary_chapter": fragment.get("primary_chapter"),
                    "keywords": fragment.get("keywords", []),
                    "summary": fragment.get("summary"),
                    "clip_usage_candidates": fragment.get("clip_usage_candidates", []),
                    "quote_strength": fragment.get("quote_strength"),
                    "clarity_score": fragment.get("clarity_score"),
                    "narration_support_score": fragment.get("narration_support_score"),
                    "classification_confidence": fragment.get("classification_confidence"),
                    "text": fragment.get("text"),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    log(f"Catalog written to: {CATALOG_PATH}")


def main() -> None:
    log("Starting Step 05: Classify Knowledge Fragments")
    log(f"Project root: {PROJECT_ROOT}")
    log(f"Input fragment directory: {FRAGMENT_DIR}")
    log(f"Output classified directory: {CLASSIFIED_DIR}")

    fragment_files = sorted(FRAGMENT_DIR.glob("*.fragments.json"))

    log(f"Found fragment files: {len(fragment_files)}")

    if not fragment_files:
        log("No fragment files found. Run scripts/04_create_fragments.py first.")
        return

    report_rows = []
    classified_payloads = []

    total_fragments = 0
    speaker_lookup = load_speaker_lookup()

    for path in fragment_files:
        output_path = CLASSIFIED_DIR / path.name.replace(".fragments.json", ".classified.json")

        log(f"CLASSIFYING: {path.name}")

        try:
            classified_payload, report_row = classify_fragment_file(path, speaker_lookup)

            output_path.write_text(
                json.dumps(classified_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            classified_payloads.append(classified_payload)
            report_rows.append(report_row)
            total_fragments += report_row["fragment_count"]

            log(f"DONE: {path.name} -> {report_row['fragment_count']} fragments classified")

        except Exception as exc:
            log(f"FAILED: {path.name}")
            log(str(exc))

            report_rows.append(
                {
                    "fragment_file": path.name,
                    "status": "failed",
                    "fragment_count": 0,
                    "top_theme": "",
                    "top_chapter": "",
                    "speaker_soundbite_count": 0,
                    "narration_support_count": 0,
                    "evidence_clip_count": 0,
                    "context_visual_count": 0,
                    "error": str(exc),
                }
            )

        write_report(report_rows)

    write_catalog(classified_payloads)

    failed_count = sum(1 for row in report_rows if row["status"] == "failed")

    log("")
    log("Summary")
    log(f"Fragment files processed: {len(report_rows)}")
    log(f"Total fragments classified: {total_fragments}")
    log(f"Failed: {failed_count}")
    log(f"Report: {REPORT_PATH}")
    log(f"Catalog: {CATALOG_PATH}")


if __name__ == "__main__":
    main()
