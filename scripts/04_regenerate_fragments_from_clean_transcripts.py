"""
Step 04 (Regenerate): Create Fragments from Cleaned Transcripts

Rebuilds knowledge fragments from cleaned (and optionally aligned) transcripts.
Input priority per video:
    1. data/transcripts_cleaned_aligned/{stem}.json  (word-level timing)
    2. data/transcripts_cleaned/{stem}.json          (cleaned, no word alignment)
    3. data/transcripts/{stem}.json                  (original — only if no cleaned version)

Output:
    data/fragments_cleaned/{stem}.fragments.json     (per-video fragment file)
    data/reports/fragment_regeneration_report.csv

Usage:
    .venv/bin/python scripts/04_regenerate_fragments_from_clean_transcripts.py
    .venv/bin/python scripts/04_regenerate_fragments_from_clean_transcripts.py --limit 5
    .venv/bin/python scripts/04_regenerate_fragments_from_clean_transcripts.py --video-id 0026
    .venv/bin/python scripts/04_regenerate_fragments_from_clean_transcripts.py --force
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Local dev layout: backend/app/. Docker image layout (see Dockerfile): app/
# directly under PROJECT_ROOT, with no backend/ directory at all.
_BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR if (_BACKEND_DIR / "app").is_dir() else PROJECT_ROOT))

from app.core.settings import settings  # noqa: E402
from app.services.video_naming import parse_source_video  # noqa: E402

ALIGNED_DIR = PROJECT_ROOT / "data" / "transcripts_cleaned_aligned"
CLEANED_DIR = PROJECT_ROOT / "data" / "transcripts_cleaned"
ORIG_DIR = PROJECT_ROOT / "data" / "transcripts"
FRAGMENT_DIR = PROJECT_ROOT / "data" / "fragments_cleaned"
REPORT_DIR = PROJECT_ROOT / "data" / "reports"

FRAGMENT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_CSV = REPORT_DIR / "fragment_regeneration_report.csv"

MIN_DURATION_SECONDS = settings.fragment_min_duration_seconds
TARGET_DURATION_SECONDS = settings.fragment_target_duration_seconds
MAX_DURATION_SECONDS = settings.fragment_max_duration_seconds

MIN_WORDS = settings.fragment_min_words
TARGET_WORDS = settings.fragment_target_words
MAX_WORDS = settings.fragment_max_words

STORYLINE_PLACEHOLDER = {
    "theme": None,
    "chapter": None,
    "storyline_role": None,
    "persona_relevance": [],
    "keywords": [],
    "summary": None,
    "classification_confidence": None,
}


def log(msg: str) -> None:
    print(msg, flush=True)


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()


def extract_video_id(stem: str) -> str:
    """0002_SvVu6k02Hes_Zimbabwe_-_AI4IA_2025_1 -> SvVu6k02Hes.
    Parsed by the fixed 11-character id length: ids can contain "_"."""
    return parse_source_video(stem)[0] or stem


def find_transcript(stem: str) -> tuple[Path | None, str]:
    """Returns (path, version) with input priority: aligned > cleaned > original."""
    aligned = ALIGNED_DIR / f"{stem}.json"
    if aligned.exists() and aligned.stat().st_size > 0:
        return aligned, "cleaned_v2_aligned"

    cleaned = CLEANED_DIR / f"{stem}.json"
    if cleaned.exists() and cleaned.stat().st_size > 0:
        return cleaned, "cleaned_v2"

    orig = ORIG_DIR / f"{stem}.json"
    if orig.exists() and orig.stat().st_size > 0:
        return orig, "original_v1"

    return None, "not_found"


def has_word_timestamps(segments: list[dict]) -> bool:
    return any(s.get("words") for s in segments)


def get_audio_quality_status(stem: str) -> str:
    report_json = REPORT_DIR / "audio_quality_report.json"
    if not report_json.exists():
        return "unknown"
    try:
        data = json.loads(report_json.read_text(encoding="utf-8"))
        for f in data.get("files", []):
            if stem in f.get("video_id", ""):
                return f.get("recommended_action", "unknown")
    except Exception:
        pass
    return "unknown"


def get_transcript_quality_score(transcript_data: dict) -> float:
    quality = transcript_data.get("quality", {})
    conf = quality.get("avg_confidence")
    empty = quality.get("empty_segment_ratio")
    repeated = quality.get("repeated_phrase_ratio")

    if conf is None:
        segs = transcript_data.get("segments", [])
        confs = [1.0 + s["avg_logprob"] for s in segs if s.get("avg_logprob") is not None]
        conf = round(sum(confs) / len(confs), 4) if confs else 0.0
        empty = sum(1 for s in segs if not s.get("text", "").strip()) / max(len(segs), 1)
        texts = [s.get("text", "").strip() for s in segs if s.get("text", "").strip()]
        from collections import Counter
        repeated = sum(cnt - 1 for cnt in Counter(texts).values() if cnt > 1) / max(len(texts), 1)

    score = (
        0.50 * min(1.0, conf) +
        0.25 * max(0.0, 1.0 - (empty or 0) * 3) +
        0.25 * max(0.0, 1.0 - (repeated or 0) * 5)
    )
    return round(score, 4)


def should_close_fragment(text: str, start: float, end: float) -> bool:
    duration = end - start
    wc = count_words(text)
    if duration >= MAX_DURATION_SECONDS:
        return True
    if wc >= MAX_WORDS:
        return True
    if duration >= TARGET_DURATION_SECONDS and wc >= MIN_WORDS:
        return True
    if wc >= TARGET_WORDS and duration >= MIN_DURATION_SECONDS:
        return True
    return False


def make_fragment(
    transcript_stem: str,
    video_id: str,
    fragment_index: int,
    start_time: float,
    end_time: float,
    segments: list[dict],
    source_version: str,
    audio_quality_status: str,
    transcript_quality_score: float,
    word_timing_available: bool,
) -> dict:
    text = clean_text(" ".join(s.get("text", "") for s in segments))
    wc = count_words(text)
    duration = round(end_time - start_time, 3)
    seg_ids = [s.get("id") for s in segments if s.get("id") is not None]

    return {
        "fragment_id": f"{transcript_stem}_{fragment_index:04d}",
        "source_video": transcript_stem,
        "video_id": video_id,
        "fragment_index": fragment_index,
        "start": round(start_time, 3),
        "end": round(end_time, 3),
        "duration": duration,
        "text": text,
        "word_count": wc,
        "source_segment_ids": seg_ids,
        "source_transcript_version": source_version,
        "audio_quality_status": audio_quality_status,
        "transcript_quality_score": transcript_quality_score,
        "word_timing_available": word_timing_available,
        "created_at": datetime.utcnow().isoformat(),
        **STORYLINE_PLACEHOLDER,
    }


def create_fragments(transcript_path: Path, source_version: str) -> tuple[list[dict], dict]:
    stem = transcript_path.stem
    video_id = extract_video_id(stem)

    data = json.loads(transcript_path.read_text(encoding="utf-8"))
    segments = data.get("segments", [])

    if not segments:
        return [], {
            "transcript_file": transcript_path.name,
            "source_version": source_version,
            "status": "no_segments",
            "fragment_count": 0,
            "total_words": 0,
            "error": "no segments",
        }

    audio_quality = get_audio_quality_status(stem)
    transcript_quality = get_transcript_quality_score(data)
    word_timing = has_word_timestamps(segments)

    fragments = []
    current_segs = []
    current_start = None
    current_end = None

    for seg in segments:
        text = clean_text(seg.get("text", ""))
        if not text:
            continue

        seg_start = float(seg.get("start", 0))
        seg_end = float(seg.get("end", seg_start))

        if current_start is None:
            current_start = seg_start

        current_end = seg_end
        current_segs.append(seg)

        current_text = " ".join(s.get("text", "") for s in current_segs)
        if should_close_fragment(current_text, current_start, current_end):
            frag = make_fragment(
                transcript_stem=stem,
                video_id=video_id,
                fragment_index=len(fragments) + 1,
                start_time=current_start,
                end_time=current_end,
                segments=current_segs,
                source_version=source_version,
                audio_quality_status=audio_quality,
                transcript_quality_score=transcript_quality,
                word_timing_available=word_timing,
            )
            if frag["word_count"] >= MIN_WORDS:
                fragments.append(frag)
            current_segs = []
            current_start = None
            current_end = None

    # Remaining segments
    if current_segs and current_start is not None and current_end is not None:
        final_frag = make_fragment(
            transcript_stem=stem,
            video_id=video_id,
            fragment_index=len(fragments) + 1,
            start_time=current_start,
            end_time=current_end,
            segments=current_segs,
            source_version=source_version,
            audio_quality_status=audio_quality,
            transcript_quality_score=transcript_quality,
            word_timing_available=word_timing,
        )
        if (
            fragments
            and final_frag["word_count"] < MIN_WORDS
            and final_frag["duration"] < MIN_DURATION_SECONDS
        ):
            # Merge into last fragment
            prev = fragments[-1]
            prev_ids = set(prev.get("source_segment_ids", []))
            merged_segs = [
                s for s in segments
                if s.get("id") in prev_ids or s in current_segs
            ]
            fragments[-1] = make_fragment(
                transcript_stem=stem,
                video_id=video_id,
                fragment_index=prev["fragment_index"],
                start_time=prev["start"],
                end_time=final_frag["end"],
                segments=merged_segs,
                source_version=source_version,
                audio_quality_status=audio_quality,
                transcript_quality_score=transcript_quality,
                word_timing_available=word_timing,
            )
        elif final_frag["word_count"] > 0:
            fragments.append(final_frag)

    return fragments, {
        "transcript_file": transcript_path.name,
        "source_version": source_version,
        "status": "fragmented",
        "fragment_count": len(fragments),
        "total_words": sum(f["word_count"] for f in fragments),
        "audio_quality_status": audio_quality,
        "transcript_quality_score": transcript_quality,
        "word_timing_available": word_timing,
        "error": "",
    }


def write_report(rows: list[dict]) -> None:
    fieldnames = [
        "transcript_file", "source_version", "status",
        "fragment_count", "total_words",
        "audio_quality_status", "transcript_quality_score",
        "word_timing_available", "error",
    ]
    with REPORT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    log(f"Report: {REPORT_CSV}")


def collect_stems(video_id: str | None) -> list[str]:
    """Collect all unique stems that have at least one transcript version."""
    stems: set[str] = set()
    for d in (ALIGNED_DIR, CLEANED_DIR, ORIG_DIR):
        for p in d.glob("*.json"):
            stems.add(p.stem)

    if video_id:
        stems = {s for s in stems if video_id in s}

    return sorted(stems)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate fragments from cleaned transcripts"
    )
    parser.add_argument("--video-id", help="Process only stems matching this substring")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even if fragment file already exists")
    parser.add_argument("--only-cleaned", action="store_true",
                        help="Skip stems that only have original transcripts")
    args = parser.parse_args()

    log("Starting Step 04 (Regenerate): Fragments from Cleaned Transcripts")
    log(f"Aligned dir  : {ALIGNED_DIR}")
    log(f"Cleaned dir  : {CLEANED_DIR}")
    log(f"Orig dir     : {ORIG_DIR}")
    log(f"Fragment dir : {FRAGMENT_DIR}")

    stems = collect_stems(args.video_id)

    if args.only_cleaned:
        stems = [
            s for s in stems
            if (ALIGNED_DIR / f"{s}.json").exists() or (CLEANED_DIR / f"{s}.json").exists()
        ]

    if args.limit:
        stems = stems[:args.limit]

    log(f"Stems to process: {len(stems)}")

    if not stems:
        log("No transcripts found. Run Step 03 or Step 03_retranscribe_cleaned.py first.")
        return

    report_rows = []
    total_fragments = 0

    for stem in stems:
        out_path = FRAGMENT_DIR / f"{stem}.fragments.json"

        if not args.force and out_path.exists() and out_path.stat().st_size > 0:
            log(f"SKIP (exists): {stem}")
            report_rows.append({
                "transcript_file": stem + ".json",
                "source_version": "skipped",
                "status": "skipped_exists",
                "fragment_count": 0,
                "total_words": 0,
                "error": "",
            })
            continue

        transcript_path, version = find_transcript(stem)
        if not transcript_path:
            log(f"SKIP (no transcript): {stem}")
            report_rows.append({
                "transcript_file": stem + ".json",
                "source_version": "not_found",
                "status": "no_transcript",
                "fragment_count": 0,
                "total_words": 0,
                "error": "no transcript found",
            })
            continue

        log(f"FRAGMENTING: {stem} [{version}]")

        try:
            fragments, row = create_fragments(transcript_path, version)

            payload = {
                "source_transcript": transcript_path.name,
                "source_transcript_version": version,
                "source_video": stem,
                "video_id": extract_video_id(stem),
                "fragment_count": len(fragments),
                "created_at": datetime.utcnow().isoformat(),
                "fragmenting_config": {
                    "min_duration_seconds": MIN_DURATION_SECONDS,
                    "target_duration_seconds": TARGET_DURATION_SECONDS,
                    "max_duration_seconds": MAX_DURATION_SECONDS,
                    "min_words": MIN_WORDS,
                    "target_words": TARGET_WORDS,
                    "max_words": MAX_WORDS,
                },
                "fragments": fragments,
            }

            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            total_fragments += len(fragments)
            report_rows.append(row)
            log(f"DONE: {stem} -> {len(fragments)} fragments [{version}]")

        except Exception as exc:
            log(f"FAILED: {stem}: {exc}")
            report_rows.append({
                "transcript_file": stem + ".json",
                "source_version": "unknown",
                "status": "failed",
                "fragment_count": 0,
                "total_words": 0,
                "error": str(exc),
            })

        write_report(report_rows)

    by_status = {}
    for r in report_rows:
        s = r.get("status", "unknown")
        by_status[s] = by_status.get(s, 0) + 1

    by_version = {}
    for r in report_rows:
        v = r.get("source_version", "unknown")
        by_version[v] = by_version.get(v, 0) + 1

    log("")
    log("Summary")
    log(f"  Total stems  : {len(stems)}")
    log(f"  Total frags  : {total_fragments}")
    for status, cnt in sorted(by_status.items()):
        log(f"  {status:20s}: {cnt}")
    log("")
    log("By transcript version:")
    for ver, cnt in sorted(by_version.items()):
        log(f"  {ver:25s}: {cnt}")


if __name__ == "__main__":
    main()
