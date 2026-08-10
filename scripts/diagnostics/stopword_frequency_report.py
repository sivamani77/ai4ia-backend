#!/usr/bin/env python3
"""Stopword document-frequency diagnostic for STOPWORDS (prompt_intent.py:14-32).

Answers, before trimming any word off STOPWORDS: how much of the corpus does
this word actually appear in? A word present in most fragments can't
discriminate between candidates as a keyword filter regardless of whether
it's stopped or not, so only words below some frequency threshold are worth
un-stopping.

Reads the real retrieval corpus via ChromaDB (not the local
data/fragments_classified/ snapshot, which was confirmed stale relative to
the corpus retrieval.py actually queries -- 3,875 fragments there vs. 5,112
records in a real job's corpus_diagnostics). Requires local ChromaDB running
against the persisted data/chroma/ volume:

    docker-compose -f docker-compose.infra.yml up -d chromadb

Usage:
    CHROMA_MODE=http CHROMA_HOST=localhost CHROMA_PORT=8001 CHROMA_COLLECTION=fragments \\
        python scripts/diagnostics/stopword_frequency_report.py
"""
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR if (_BACKEND_DIR / "app").is_dir() else PROJECT_ROOT))

from app.pipeline.storyline_gen.prompt_intent import STOPWORDS  # noqa: E402
from app.pipeline.storyline_gen.retrieval import load_embedding_records  # noqa: E402
from app.pipeline.storyline_gen.text_utils import normalize_plural  # noqa: E402

OUT_PATH = PROJECT_ROOT / "data" / "reports" / "stopword_frequency_report.json"

# Confirmed-over-broad in the architectural review (root cause of the "AI for
# the Information Technology" incident) plus the additional candidates the
# review flagged as similarly domain-core: checked first/highlighted
# separately in the printed summary, but every STOPWORDS entry is measured.
SUSPECT_TERMS = [
    "ai", "artificial", "intelligence", "information", "technology",
    "technologies", "data", "content", "system", "systems", "tool", "tools",
]

# Discriminating-power cutoff: terms at/above this document frequency can't
# usefully distinguish between candidates as a keyword filter regardless of
# whether they're stopped, since keyword_score/expanded_keyword_score would
# saturate for most of the corpus either way. Provisional -- eyeball the real
# distribution before trusting this number.
DISCRIMINATING_FREQUENCY_THRESHOLD = 0.55


def build_blob(record: dict) -> str:
    # Mirrors candidate_search_blob (candidate_scoring.py:276-287) -- same
    # fields, same join -- so frequency here reflects what the pipeline's own
    # keyword matching actually sees.
    parts = [
        record.get("text", ""),
        record.get("summary", ""),
        " ".join(record.get("keywords") or []) if isinstance(record.get("keywords"), list) else str(record.get("keywords") or ""),
        record.get("source_video", ""),
        record.get("speaker", ""),
        record.get("primary_speaker", ""),
        record.get("primary_theme", ""),
        record.get("primary_chapter", ""),
    ]
    return " ".join(str(part) for part in parts if part).lower()


def tokenize(blob: str) -> set[str]:
    # normalize_plural mirrors extract_topic_terms's actual matching logic
    # (prompt_intent.py:211) so plural/singular variants aren't undercounted.
    return {normalize_plural(tok) for tok in re.findall(r"[a-z0-9]+", blob)}


def main() -> None:
    records = load_embedding_records()
    total = len(records)
    print(f"Loaded {total} records")
    if total == 0:
        print("No records loaded -- is ChromaDB running and reachable? "
              "(docker-compose -f docker-compose.infra.yml up -d chromadb, "
              "then set CHROMA_MODE=http CHROMA_HOST=localhost CHROMA_PORT=8001)")
        sys.exit(1)

    terms_to_check = sorted(set(STOPWORDS) | set(SUSPECT_TERMS))
    doc_counts = {term: 0 for term in terms_to_check}

    for record in records:
        token_set = tokenize(build_blob(record))
        for term in terms_to_check:
            if term in token_set:
                doc_counts[term] += 1

    frequencies = {
        term: round(count / total, 4)
        for term, count in doc_counts.items()
    }
    ranked = sorted(frequencies.items(), key=lambda kv: kv[1], reverse=True)

    below_threshold = [term for term, freq in ranked if freq < DISCRIMINATING_FREQUENCY_THRESHOLD]
    suspect_results = {term: frequencies[term] for term in SUSPECT_TERMS}

    report = {
        "total_records": total,
        "discriminating_frequency_threshold": DISCRIMINATING_FREQUENCY_THRESHOLD,
        "frequencies_all_stopwords": dict(ranked),
        "suspect_terms": suspect_results,
        "suspect_terms_below_threshold": [t for t in SUSPECT_TERMS if frequencies[t] < DISCRIMINATING_FREQUENCY_THRESHOLD],
        "all_stopwords_below_threshold": below_threshold,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n--- suspect terms (document frequency) ---")
    for term in SUSPECT_TERMS:
        freq = frequencies[term]
        flag = "UNSTOP CANDIDATE" if freq < DISCRIMINATING_FREQUENCY_THRESHOLD else "keep stopped (too common to discriminate)"
        print(f"  {term:15s} {freq:6.1%}   {flag}")

    print(f"\n--- all {len(terms_to_check)} STOPWORDS entries, ranked by frequency (top 20) ---")
    for term, freq in ranked[:20]:
        print(f"  {freq:6.1%}  {term}")

    print(f"\nSTOPWORDS entries below the {DISCRIMINATING_FREQUENCY_THRESHOLD:.0%} threshold "
          f"({len(below_threshold)} of {len(terms_to_check)}):")
    print(f"  {below_threshold}")

    print(f"\nFull report written to: {OUT_PATH}")


if __name__ == "__main__":
    main()
