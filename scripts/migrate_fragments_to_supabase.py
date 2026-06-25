#!/usr/bin/env python3
"""One-time migration: load all data/fragments_classified/*.classified.json → Supabase fragments table.

Applies the same speaker-roster enrichment and facet derivation as the old
fragment_store._build() so the DB rows are identical to what the in-memory
cache used to serve.

Run from the repo root:
    python scripts/migrate_fragments_to_supabase.py

Progress is printed per file. Fragments already present (by fragment_id) are
skipped so the script is safe to re-run.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.core.paths import CLASSIFIED_FRAGMENTS_DIR, RAW_VIDEO_DIR, SPEAKERS_DIR
from app.db.repositories.fragment_repository import FragmentRepository
from app.services.enrichment import derive_facets

SCHEMA_COLUMNS = {
    "fragment_id", "source_video", "video_id", "video_title",
    "speaker", "speaker_attribution", "text", "summary",
    "start", "end", "duration",
    "primary_theme", "primary_chapter",
    "theme_tags", "chapter_tags", "clip_usage_candidates",
    "quote_strength", "clarity_score",
    "year", "edition", "session_type", "keywords",
}

BATCH_SIZE = 100


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


def _name(tag, key: str) -> str | None:
    if tag is None:
        return None
    if isinstance(tag, str):
        return tag
    if isinstance(tag, dict):
        return tag.get(key) or tag.get("name") or tag.get("label") or tag.get(f"{key}_id")
    return str(tag)


def _to_row(frag: dict, *, source_video: str, video_id: str, video_title: str,
            speaker: str | None, speaker_attribution: str, facets: dict) -> dict:
    """Map a classified fragment dict to a DB row dict."""
    fid = frag.get("fragment_id") or frag.get("id")
    start = float(frag.get("start") or 0)
    end = float(frag.get("end") or 0)
    duration = frag.get("duration")
    if not duration:
        duration = max(end - start, 0)

    # Collect extra fields not in the schema
    extra = {k: v for k, v in frag.items() if k not in SCHEMA_COLUMNS and k not in {"id", "fragment_id"}}

    return {
        "fragment_id": fid,
        "source_video": frag.get("source_video") or source_video,
        "video_id": frag.get("video_id") or video_id,
        "video_title": video_title,
        "speaker": speaker,
        "speaker_attribution": speaker_attribution,
        "text": frag.get("text") or "",
        "summary": frag.get("summary"),
        "start": start,
        "end": end,
        "duration": float(duration),
        "primary_theme": _name(frag.get("primary_theme"), "theme"),
        "primary_chapter": _name(frag.get("primary_chapter"), "chapter"),
        "theme_tags": frag.get("theme_tags") or [],
        "chapter_tags": frag.get("chapter_tags") or [],
        "clip_usage_candidates": frag.get("clip_usage_candidates") or [],
        "quote_strength": float(frag.get("quote_strength") or 0),
        "clarity_score": float(frag.get("clarity_score") or 0),
        "year": facets.get("year"),
        "edition": facets.get("edition"),
        "session_type": facets.get("session_type"),
        "keywords": frag.get("keywords") or [],
        "extra": extra,
    }


def main() -> None:
    files = sorted(CLASSIFIED_FRAGMENTS_DIR.glob("*.classified.json"))
    if not files:
        print("No classified fragment files found — nothing to migrate.")
        return

    print(f"Found {len(files)} classified JSON files.")
    repo = FragmentRepository()

    total_migrated = total_skipped = total_failed = 0

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
        rows_to_insert: list[dict] = []
        skipped_in_file = 0

        for frag in fragments:
            fid = frag.get("fragment_id") or frag.get("id")
            if not fid:
                total_failed += 1
                continue
            try:
                existing = repo.get(fid)
            except Exception:
                existing = None
            if existing:
                skipped_in_file += 1
                total_skipped += 1
                continue
            try:
                row = _to_row(
                    frag,
                    source_video=source_video,
                    video_id=video_id,
                    video_title=video_title,
                    speaker=speaker,
                    speaker_attribution=attribution,
                    facets=facets,
                )
                rows_to_insert.append(row)
            except Exception as e:
                print(f"    FAIL build row {fid}: {e}")
                total_failed += 1

        # Batch upsert
        for i in range(0, len(rows_to_insert), BATCH_SIZE):
            batch = rows_to_insert[i: i + BATCH_SIZE]
            try:
                repo.upsert_many(batch)
                total_migrated += len(batch)
            except Exception as e:
                print(f"    FAIL batch upsert ({len(batch)} rows): {e}")
                total_failed += len(batch)

        status = f"migrated={len(rows_to_insert)} skipped={skipped_in_file}"
        print(f"  {fpath.name[:70]}: {status}")

    print(f"\nDone. Total migrated={total_migrated}  skipped={total_skipped}  failed={total_failed}")


if __name__ == "__main__":
    main()
