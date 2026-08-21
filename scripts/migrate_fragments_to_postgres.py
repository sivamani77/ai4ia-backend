#!/usr/bin/env python3
"""One-time (and safe-to-rerun) backfill: load all
data/fragments_classified/*.classified.json -> Postgres `fragments` table.

Successor to the old migrate_fragments_to_supabase.py. That script only
inserted a fragment_id once and skipped it forever after — so a fragment
re-classified with corrected data (e.g. a speaker roster fix) would never
get its DB row updated. This script always upserts (ON CONFLICT DO UPDATE,
see FragmentRepository.upsert_many), so re-running it after a pipeline
re-run picks up any local changes.

Run from the repo root:
    python scripts/migrate_fragments_to_postgres.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.core.paths import CLASSIFIED_FRAGMENTS_DIR, RAW_VIDEO_DIR, SPEAKERS_DIR
from app.db.repositories.fragment_repository import FragmentRepository
from app.services.enrichment import derive_facets

BATCH_SIZE = 200


def _safe_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _infer_topic(name: str) -> str:
    stem = Path(name).stem.replace(".classified", "").replace(".info", "")
    parts = stem.split("_")
    human = "_".join(parts[2:]) if len(parts) > 2 else stem
    return re.sub(r"\s+", " ", human.replace("_-_", " - ").replace("_", " ")).strip()


def _load_roster(video_id: str, source_video: str) -> dict | None:
    if not SPEAKERS_DIR.exists():
        return None
    for key in (video_id, source_video):
        path = SPEAKERS_DIR / f"{key}.json"
        if path.exists():
            return _safe_json(path)
    return None


def _to_row(frag: dict, *, source_video: str, video_id: str, video_title: str,
            speaker: str | None, speaker_attribution: str, facets: dict) -> dict:
    fid = frag.get("fragment_id") or frag.get("id")
    start = float(frag.get("start") or 0)
    end = float(frag.get("end") or 0)
    duration = frag.get("duration") or max(end - start, 0)

    return {
        "fragment_id": fid,
        "source_video": frag.get("source_video") or source_video,
        "video_id": frag.get("video_id") or video_id,
        "video_title": video_title,
        "speaker": frag.get("speaker") or speaker,
        "speaker_attribution": frag.get("speaker_attribution") or speaker_attribution,
        "text": frag.get("text") or "",
        "summary": frag.get("summary"),
        "start": start,
        "end": end,
        "duration": float(duration),
        "primary_theme": frag.get("primary_theme"),
        "primary_chapter": frag.get("primary_chapter"),
        "theme_tags": frag.get("theme_tags") or [],
        "chapter_tags": frag.get("chapter_tags") or [],
        "clip_usage_candidates": frag.get("clip_usage_candidates") or [],
        "quote_strength": float(frag.get("quote_strength") or 0),
        "clarity_score": float(frag.get("clarity_score") or 0),
        "year": facets.get("year"),
        "edition": facets.get("edition"),
        "session_type": facets.get("session_type"),
        "keywords": frag.get("keywords") or [],
        "extra": {},
    }


def main() -> None:
    files = sorted(CLASSIFIED_FRAGMENTS_DIR.glob("*.classified.json"))
    if not files:
        print("No classified fragment files found — nothing to migrate.")
        return

    print(f"Found {len(files)} classified JSON files.")
    repo = FragmentRepository()

    total_upserted = total_failed = 0

    for fpath in files:
        payload = _safe_json(fpath)
        if not isinstance(payload, dict):
            print(f"  SKIP (bad JSON) {fpath.name}")
            total_failed += 1
            continue

        source_video = payload.get("source_video") or fpath.stem.replace(".classified", "")
        video_id = payload.get("video_id") or source_video
        info = _safe_json(RAW_VIDEO_DIR / f"{source_video}.info.json")
        video_title = (info or {}).get("title") or _infer_topic(fpath.name)
        facets = derive_facets(video_title, source_video)

        roster = _load_roster(video_id, source_video)
        if roster:
            speaker = roster.get("primary_speaker")
            attribution = "video-level"
        else:
            speaker = _infer_topic(fpath.name)
            attribution = "filename-derived"

        fragments = payload.get("fragments", [])
        rows: list[dict] = []
        for frag in fragments:
            fid = frag.get("fragment_id") or frag.get("id")
            if not fid:
                total_failed += 1
                continue
            try:
                rows.append(_to_row(
                    frag, source_video=source_video, video_id=video_id,
                    video_title=video_title, speaker=speaker,
                    speaker_attribution=attribution, facets=facets,
                ))
            except Exception as e:
                print(f"    FAIL build row {fid}: {e}")
                total_failed += 1

        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i: i + BATCH_SIZE]
            try:
                repo.upsert_many(batch)
                total_upserted += len(batch)
            except Exception as e:
                print(f"    FAIL batch upsert ({len(batch)} rows): {e}")
                total_failed += len(batch)

        print(f"  {fpath.name[:70]}: upserted={len(rows)}")

    print(f"\nDone. Total upserted={total_upserted}  failed={total_failed}")


if __name__ == "__main__":
    main()
