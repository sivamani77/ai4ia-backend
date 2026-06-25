"""
Step 03b: Compare Old vs Cleaned Transcripts

For each video that has both an original transcript (data/transcripts/) and
a cleaned transcript (data/transcripts_cleaned/), computes improvement metrics.

Output:
    data/reports/transcript_comparison_report.csv
    data/reports/transcript_comparison_report.json

Usage:
    .venv/bin/python scripts/03b_compare_transcripts.py
    .venv/bin/python scripts/03b_compare_transcripts.py --limit 5
    .venv/bin/python scripts/03b_compare_transcripts.py --video-id 0026
"""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ORIG_DIR = PROJECT_ROOT / "data" / "transcripts"
CLEAN_DIR = PROJECT_ROOT / "data" / "transcripts_cleaned"
REPORT_DIR = PROJECT_ROOT / "data" / "reports"

REPORT_CSV = REPORT_DIR / "transcript_comparison_report.csv"
REPORT_JSON = REPORT_DIR / "transcript_comparison_report.json"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(msg, flush=True)


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def load_transcript(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data
    except Exception:
        return None


def empty_segment_ratio(segments: list[dict]) -> float:
    if not segments:
        return 1.0
    empty = sum(1 for s in segments if not s.get("text", "").strip())
    return round(empty / len(segments), 4)


def repeated_phrase_ratio(segments: list[dict]) -> float:
    texts = [s.get("text", "").strip() for s in segments if s.get("text", "").strip()]
    if not texts:
        return 0.0
    counts = Counter(texts)
    repeated = sum(cnt - 1 for cnt in counts.values() if cnt > 1)
    return round(repeated / len(texts), 4)


def avg_confidence(segments: list[dict]) -> float:
    confs = [1.0 + s["avg_logprob"] for s in segments if s.get("avg_logprob") is not None]
    if not confs:
        return 0.0
    return round(sum(confs) / len(confs), 4)


def compute_improvement_score(old: dict, new: dict) -> float:
    """
    Score in [0, 1]. Higher = cleaned transcript is better.

    Components:
        word_count_gain   : cleaned has more words (more speech recovered)
        empty_ratio_drop  : fewer empty segments
        repeated_drop     : fewer repeated phrases
        confidence_gain   : higher avg confidence
    """
    old_wc = old.get("old_word_count", 0) or 0
    new_wc = new.get("new_word_count", 0) or 0
    wc_gain = min(1.0, max(0.0, (new_wc - old_wc) / max(old_wc, 1)))

    empty_drop = max(0.0, (old.get("old_empty_ratio", 0) or 0) - (new.get("new_empty_ratio", 0) or 0))
    repeat_drop = max(0.0, (old.get("old_repeated_phrase_ratio", 0) or 0) - (new.get("new_repeated_phrase_ratio", 0) or 0))
    conf_gain = max(0.0, (new.get("new_avg_confidence", 0) or 0) - (old.get("old_avg_confidence", 0) or 0))

    score = (
        0.30 * min(1.0, wc_gain) +
        0.25 * min(1.0, empty_drop * 5) +
        0.25 * min(1.0, repeat_drop * 5) +
        0.20 * min(1.0, conf_gain * 2)
    )
    return round(score, 4)


def decide_recommended_use(improvement_score: float, new_action: str, old_action: str) -> str:
    if new_action == "manual_review":
        return "manual_review"
    if improvement_score >= 0.15:
        return "cleaned"
    if improvement_score < 0.0:
        return "old"
    return "old"


def compare_pair(orig_path: Path, clean_path: Path) -> dict:
    video_id = orig_path.stem

    orig = load_transcript(orig_path)
    clean = load_transcript(clean_path)

    if not orig:
        return {
            "video_id": video_id,
            "status": "error",
            "error": "could not load original transcript",
        }
    if not clean:
        return {
            "video_id": video_id,
            "status": "error",
            "error": "could not load cleaned transcript",
        }

    orig_segs = orig.get("segments", [])
    clean_segs = clean.get("segments", [])

    orig_text = orig.get("text", "") or " ".join(s.get("text", "") for s in orig_segs)
    clean_text = clean.get("text", "") or " ".join(s.get("text", "") for s in clean_segs)

    old_wc = word_count(orig_text)
    new_wc = word_count(clean_text)

    old_empty = empty_segment_ratio(orig_segs)
    new_empty = empty_segment_ratio(clean_segs)

    old_repeated = repeated_phrase_ratio(orig_segs)
    new_repeated = repeated_phrase_ratio(clean_segs)

    old_conf = avg_confidence(orig_segs)
    new_conf = avg_confidence(clean_segs)

    old_quality = orig.get("quality", {})
    new_quality = clean.get("quality", {})
    old_action = old_quality.get("recommended_action", "unknown")
    new_action = new_quality.get("recommended_action", "unknown")

    row = {
        "video_id": video_id,
        "status": "compared",
        "old_word_count": old_wc,
        "new_word_count": new_wc,
        "word_count_delta": new_wc - old_wc,
        "old_segment_count": len(orig_segs),
        "new_segment_count": len(clean_segs),
        "old_empty_ratio": old_empty,
        "new_empty_ratio": new_empty,
        "old_repeated_phrase_ratio": old_repeated,
        "new_repeated_phrase_ratio": new_repeated,
        "old_avg_confidence": old_conf,
        "new_avg_confidence": new_conf,
        "old_recommended_action": old_action,
        "new_recommended_action": new_action,
        "error": "",
    }

    improvement_score = compute_improvement_score(row, row)
    recommended_use = decide_recommended_use(improvement_score, new_action, old_action)

    row["improvement_score"] = improvement_score
    row["recommended_use"] = recommended_use

    return row


def write_csv(rows: list[dict]) -> None:
    fieldnames = [
        "video_id", "status",
        "old_word_count", "new_word_count", "word_count_delta",
        "old_segment_count", "new_segment_count",
        "old_empty_ratio", "new_empty_ratio",
        "old_repeated_phrase_ratio", "new_repeated_phrase_ratio",
        "old_avg_confidence", "new_avg_confidence",
        "old_recommended_action", "new_recommended_action",
        "improvement_score", "recommended_use", "error",
    ]
    with REPORT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    log(f"CSV report: {REPORT_CSV}")


def write_json(rows: list[dict]) -> None:
    use_counts = Counter(r.get("recommended_use", "unknown") for r in rows)
    payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "total_compared": len(rows),
        "recommend_cleaned": use_counts.get("cleaned", 0),
        "recommend_old": use_counts.get("old", 0),
        "recommend_manual_review": use_counts.get("manual_review", 0),
        "files": rows,
    }
    REPORT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"JSON report: {REPORT_JSON}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare old vs cleaned transcripts")
    parser.add_argument("--video-id", help="Process only files matching this video ID substring")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    log("Starting Step 03b: Compare Transcripts")
    log(f"Original dir : {ORIG_DIR}")
    log(f"Cleaned dir  : {CLEAN_DIR}")

    orig_files = {p.stem: p for p in sorted(ORIG_DIR.glob("*.json"))}
    clean_files = {p.stem: p for p in sorted(CLEAN_DIR.glob("*.json"))}

    # Match by stem
    pairs = []
    for stem, orig_path in sorted(orig_files.items()):
        if stem in clean_files:
            pairs.append((orig_path, clean_files[stem]))

    if args.video_id:
        pairs = [(o, c) for o, c in pairs if args.video_id in o.stem]

    if args.limit:
        pairs = pairs[:args.limit]

    log(f"Pairs found  : {len(pairs)}")

    # Report originals without cleaned counterpart
    only_orig = set(orig_files) - set(clean_files)
    if only_orig:
        log(f"Original-only (no cleaned version yet): {len(only_orig)}")

    if not pairs:
        log("No matching pairs found. Run 03_retranscribe_cleaned.py first.")
        return

    rows = []
    for orig_path, clean_path in pairs:
        log(f"COMPARING: {orig_path.stem}")
        row = compare_pair(orig_path, clean_path)
        rows.append(row)

    write_csv(rows)
    write_json(rows)

    by_use = Counter(r.get("recommended_use", "unknown") for r in rows)
    log("")
    log("Summary")
    log(f"  Compared             : {len(rows)}")
    log(f"  Recommend cleaned    : {by_use.get('cleaned', 0)}")
    log(f"  Recommend old        : {by_use.get('old', 0)}")
    log(f"  Recommend manual     : {by_use.get('manual_review', 0)}")

    # Print a few notable improvements
    improved = sorted(
        [r for r in rows if r.get("recommended_use") == "cleaned"],
        key=lambda r: r.get("improvement_score", 0),
        reverse=True,
    )
    if improved:
        log("")
        log("Top improvements:")
        for r in improved[:5]:
            log(
                f"  {r['video_id']}: score={r['improvement_score']:.3f} "
                f"words {r['old_word_count']}→{r['new_word_count']} "
                f"conf {r['old_avg_confidence']:.3f}→{r['new_avg_confidence']:.3f}"
            )


if __name__ == "__main__":
    main()
