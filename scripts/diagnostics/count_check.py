"""Check counting (issue #23, design §7.8) against an independent direct query.

The production path scores passages with retrieval.score_every_record and counts with
storyline_gen.counting.count_topic. This script pulls every stored embedding from Chroma,
scores them itself, groups passages by talk itself, and compares the two answers. The term
test is shared on purpose: it is the coverage test the count is defined by.

    scripts/localtest_run.sh .venv/bin/python scripts/diagnostics/count_check.py deepfakes regulation "quantum computing"
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.paths import EMBEDDINGS_DIR  # noqa: E402
from app.core.settings import settings  # noqa: E402
from app.pipeline.storyline_gen import counting, retrieval  # noqa: E402
from app.pipeline.storyline_gen.coverage import has_term  # noqa: E402
from app.pipeline.storyline_gen.prompt_intent import NON_TOPIC_WORDS, build_prompt_profile  # noqa: E402


def direct_count(topic: str, terms: list[str], floor: float) -> tuple[int, list[tuple[str, int]]]:
    collection = retrieval._get_chroma_collection()
    query = retrieval.embed_query(counting.count_question(topic))
    query = query / np.linalg.norm(query)
    passages: Counter = Counter()
    total, offset = collection.count(), 0
    while offset < total:
        batch = collection.get(limit=500, offset=offset, include=["embeddings", "metadatas"])
        for fid, vector, meta in zip(batch["ids"], batch["embeddings"], batch["metadatas"]):
            vector = np.asarray(vector, dtype=np.float32)
            if float(vector @ query / np.linalg.norm(vector)) < floor:
                continue
            record = {**(meta or {}), "fragment_id": fid}
            if any(has_term(record, t) for t in terms):
                passages[record.get("source_video") or ""] += 1
        offset += 500
    talks = [(sv, n) for sv, n in passages.most_common() if sv and n >= counting.MIN_PASSAGES]
    return len(talks), talks


def main() -> None:
    retrieval.activate_corpus(embeddings_dir=str(EMBEDDINGS_DIR), corpus_version=settings.storyline_corpus_version,
                              min_transcript_quality=0.0, exclude_manual_review=False)
    records = retrieval.get_filtered_records_or_load()
    floor = settings.archive_relevance_floor
    for asked in sys.argv[1:] or ["deepfakes"]:
        # As the router hands it over: the words as written, without filler such as "AI".
        topic = " ".join(w for w in asked.lower().split() if w not in NON_TOPIC_WORDS)
        profile = build_prompt_profile(topic, [])
        terms = counting.count_terms(profile["topic_terms"] + profile["expanded_terms"])
        facts = counting.count_topic(records, retrieval.score_every_record(counting.count_question(topic)),
                                     terms, floor)
        n_direct, talks = direct_count(topic, terms, floor)
        verdict = "MATCH" if n_direct == facts["talk_count"] else "DIFFERENT"
        print(f"{verdict} {asked!r} (as {topic!r}): production {facts['talk_count']} talks, direct {n_direct} | "
              f"speakers {facts['speaker_count']}, panels {facts['panel_count']}")
        for source_video, n in talks[:5]:
            print(f"    {n:3} passages  {source_video[:90]}")


if __name__ == "__main__":
    main()
