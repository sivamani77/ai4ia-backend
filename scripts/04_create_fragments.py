import csv
import json
import re
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRANSCRIPT_DIR = PROJECT_ROOT / "data" / "transcripts"
FRAGMENT_DIR = PROJECT_ROOT / "data" / "fragments"
REPORT_PATH = PROJECT_ROOT / "data" / "metadata" / "youtube" / "fragment_creation_report.csv"

FRAGMENT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


# Fragmenting rules
MIN_DURATION_SECONDS = 35
TARGET_DURATION_SECONDS = 75
MAX_DURATION_SECONDS = 130

MIN_WORDS = 35
TARGET_WORDS = 120
MAX_WORDS = 230


STORYLINE_PLACEHOLDER = {
    "theme": None,
    "chapter": None,
    "storyline_role": None,
    "persona_relevance": [],
    "keywords": [],
    "summary": None,
    "classification_confidence": None,
}


def log(message: str) -> None:
    print(message, flush=True)


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_video_id(stem: str) -> str:
    """
    Expected filename:
    0002_SvVu6k02Hes_Zimbabwe_-_AI4IA_2025_1

    Returns:
    SvVu6k02Hes
    """
    parts = stem.split("_")
    if len(parts) >= 2:
        return parts[1]
    return stem


def should_close_fragment(
    current_text: str,
    start_time: float,
    end_time: float,
) -> bool:
    duration = end_time - start_time
    word_count = count_words(current_text)

    if duration >= MAX_DURATION_SECONDS:
        return True

    if word_count >= MAX_WORDS:
        return True

    if duration >= TARGET_DURATION_SECONDS and word_count >= MIN_WORDS:
        return True

    if word_count >= TARGET_WORDS and duration >= MIN_DURATION_SECONDS:
        return True

    return False


def make_fragment(
    transcript_stem: str,
    video_id: str,
    fragment_index: int,
    start_time: float,
    end_time: float,
    segments: list[dict],
) -> dict:
    text = clean_text(" ".join(segment.get("text", "") for segment in segments))
    word_count = count_words(text)
    duration = round(end_time - start_time, 3)

    segment_ids = [
        segment.get("id")
        for segment in segments
        if segment.get("id") is not None
    ]

    return {
        "fragment_id": f"{transcript_stem}_{fragment_index:04d}",
        "source_video": transcript_stem,
        "video_id": video_id,
        "fragment_index": fragment_index,
        "start": round(start_time, 3),
        "end": round(end_time, 3),
        "duration": duration,
        "text": text,
        "word_count": word_count,
        "source_segment_ids": segment_ids,
        "created_at": datetime.utcnow().isoformat(),
        **STORYLINE_PLACEHOLDER,
    }


def create_fragments_for_transcript(transcript_path: Path) -> tuple[list[dict], dict]:
    transcript_stem = transcript_path.stem
    video_id = extract_video_id(transcript_stem)

    with transcript_path.open("r", encoding="utf-8") as f:
        transcript = json.load(f)

    segments = transcript.get("segments", [])

    if not segments:
        return [], {
            "transcript_file": transcript_path.name,
            "status": "no_segments",
            "fragment_count": 0,
            "total_words": 0,
            "duration_seconds": transcript.get("duration_seconds", ""),
            "error": "Transcript contains no segments",
        }

    fragments = []

    current_segments = []
    current_start = None
    current_end = None

    for segment in segments:
        text = clean_text(segment.get("text", ""))
        if not text:
            continue

        segment_start = float(segment.get("start", 0))
        segment_end = float(segment.get("end", segment_start))

        if current_start is None:
            current_start = segment_start

        current_end = segment_end
        current_segments.append(segment)

        current_text = " ".join(s.get("text", "") for s in current_segments)

        if should_close_fragment(current_text, current_start, current_end):
            fragment = make_fragment(
                transcript_stem=transcript_stem,
                video_id=video_id,
                fragment_index=len(fragments) + 1,
                start_time=current_start,
                end_time=current_end,
                segments=current_segments,
            )

            if fragment["word_count"] >= MIN_WORDS:
                fragments.append(fragment)

            current_segments = []
            current_start = None
            current_end = None

    # Add remaining segments as final fragment.
    if current_segments and current_start is not None and current_end is not None:
        final_fragment = make_fragment(
            transcript_stem=transcript_stem,
            video_id=video_id,
            fragment_index=len(fragments) + 1,
            start_time=current_start,
            end_time=current_end,
            segments=current_segments,
        )

        # If final fragment is very small, merge it into previous fragment.
        if (
            fragments
            and final_fragment["word_count"] < MIN_WORDS
            and final_fragment["duration"] < MIN_DURATION_SECONDS
        ):
            previous = fragments[-1]
            merged_segments = []

            previous_segment_ids = set(previous.get("source_segment_ids", []))

            for segment in segments:
                segment_id = segment.get("id")
                if segment_id in previous_segment_ids or segment in current_segments:
                    merged_segments.append(segment)

            fragments[-1] = make_fragment(
                transcript_stem=transcript_stem,
                video_id=video_id,
                fragment_index=previous["fragment_index"],
                start_time=previous["start"],
                end_time=final_fragment["end"],
                segments=merged_segments,
            )
        elif final_fragment["word_count"] > 0:
            fragments.append(final_fragment)

    total_words = sum(fragment["word_count"] for fragment in fragments)

    return fragments, {
        "transcript_file": transcript_path.name,
        "status": "fragmented",
        "fragment_count": len(fragments),
        "total_words": total_words,
        "duration_seconds": transcript.get("duration_seconds", ""),
        "error": "",
    }


def write_report(rows: list[dict]) -> None:
    fieldnames = [
        "transcript_file",
        "status",
        "fragment_count",
        "total_words",
        "duration_seconds",
        "error",
    ]

    with REPORT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log(f"Report written to: {REPORT_PATH}")


def main() -> None:
    log("Starting Step 04: Create Knowledge Fragments")
    log(f"Project root: {PROJECT_ROOT}")
    log(f"Transcript directory: {TRANSCRIPT_DIR}")
    log(f"Fragment directory: {FRAGMENT_DIR}")

    transcript_files = sorted(TRANSCRIPT_DIR.glob("*.json"))

    log(f"Found transcript JSON files: {len(transcript_files)}")

    if not transcript_files:
        log("No transcript JSON files found. Run scripts/03_transcribe.py first.")
        return

    report_rows = []
    total_fragments = 0

    for transcript_path in transcript_files:
        output_path = FRAGMENT_DIR / f"{transcript_path.stem}.fragments.json"

        log(f"FRAGMENTING: {transcript_path.name}")

        try:
            fragments, report_row = create_fragments_for_transcript(transcript_path)

            payload = {
                "source_transcript": transcript_path.name,
                "source_video": transcript_path.stem,
                "video_id": extract_video_id(transcript_path.stem),
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

            output_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            total_fragments += len(fragments)
            report_rows.append(report_row)

            log(f"DONE: {transcript_path.name} -> {len(fragments)} fragments")

        except Exception as exc:
            log(f"FAILED: {transcript_path.name}")
            log(str(exc))

            report_rows.append(
                {
                    "transcript_file": transcript_path.name,
                    "status": "failed",
                    "fragment_count": 0,
                    "total_words": 0,
                    "duration_seconds": "",
                    "error": str(exc),
                }
            )

        write_report(report_rows)

    failed_count = sum(1 for row in report_rows if row["status"] == "failed")
    no_segment_count = sum(1 for row in report_rows if row["status"] == "no_segments")

    log("")
    log("Summary")
    log(f"Transcript files processed: {len(report_rows)}")
    log(f"Total fragments created: {total_fragments}")
    log(f"Failed: {failed_count}")
    log(f"No-segment transcripts: {no_segment_count}")
    log(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
