#!/usr/bin/env python3
"""04b — Extract speaker rosters per video using the LLM.

There is no diarization, so attribution is VIDEO-LEVEL: we identify who appears
in each video (primary speaker + roster) from the title and the opening
fragments, where speakers typically introduce themselves. Per-fragment
attribution to a specific person is a documented MVP v4 follow-up.

Output: data/metadata/speakers/{video_id}.json   (idempotent; --force to redo)

Usage:
    python scripts/04b_extract_speakers.py --limit 5      # prove it on 5 videos
    python scripts/04b_extract_speakers.py                # full run
    python scripts/04b_extract_speakers.py --force        # re-extract all
"""
import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Local dev layout: backend/app/. Docker image layout (see Dockerfile): app/
# directly under PROJECT_ROOT, with no backend/ directory at all.
_BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR if (_BACKEND_DIR / "app").is_dir() else PROJECT_ROOT))

from app.pipeline import s3_helpers  # noqa: E402
from app.services.llm_service import generate_json, SPEAKER_EXTRACTION_MODEL  # noqa: E402
from app.services.speaker_aliases import canonicalize as canonicalize_speaker  # noqa: E402
from app.services.speaker_names import description_roster  # noqa: E402
from app.services.speaker_registry import get_registry  # noqa: E402
from app.services.video_naming import parse_source_video  # noqa: E402

import os

CLASSIFIED_DIR = Path(os.getenv("AI4IA_CLASSIFIED_DIR", str(PROJECT_ROOT / "data" / "fragments_classified")))
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "youtube" / "ai4ia_channel"
SPEAKERS_DIR = PROJECT_ROOT / "data" / "metadata" / "speakers"

PROMPT = """You are analyzing an AI4IA conference video to identify its human speaker(s).

Evidence is ordered by reliability:
1. Title name hint (if present, this is almost always the correct primary speaker).
2. Video description (often names the speaker and their affiliation).
3. Opening transcript excerpt (self-introductions, "please welcome ...").

Use ONLY evidence below. Do NOT invent names. A region or country (e.g. "Zimbabwe")
is NOT a person's name. If no human name appears anywhere, return an empty list.

Title: {title}
Title name hint: {title_hint}
Description: {description}

Opening transcript excerpt:
\"\"\"{excerpt}\"\"\"

Return JSON exactly in this shape:
{{
  "speakers": [
    {{"name": "<full name>", "role": "Speaker|Moderator|Panelist|Host|Unknown",
      "affiliation": "<org/country or null>", "confidence": 0.0}}
  ],
  "primary_speaker": "<name or null>"
}}"""

# Tokens signalling a region/session rather than a person, anywhere in the title lead.
_NON_PERSON_WORDS = {"zimbabwe", "eswatini", "uganda", "kenya", "nigeria", "ghana",
                     "india", "pakistan", "nepal", "uruguay", "america", "latin",
                     "africa", "asia", "europe", "caribbean", "event", "events",
                     "conference", "forum", "webinar", "launch", "panel", "opening",
                     "closing", "satellite", "regional", "session", "keynote"}


def parse_title_name_hint(title: str) -> str | None:
    """If a title is '<Person Name> - <Topic>', return the name. Region/session -> None."""
    # Strip a leading track/session code like "(7G)" or "(9A)" — otherwise it fails
    # the all-tokens-capitalized check below and masks a real name right after it.
    title = re.sub(r"^\(\s*[0-9]+[A-Za-z]*\s*\)\s*", "", title.strip())
    # Accept ' - ', ' -', or 'word- ' separators.
    lead = re.split(r"\s+-\s*|\s*-\s+", title, maxsplit=1)[0].strip()
    if lead == title.strip():  # no separator found
        return None
    tokens = lead.split()
    # A person name is typically 2-4 capitalized tokens; single tokens are usually regions.
    if not (2 <= len(tokens) <= 4):
        return None
    if any(t.lower().strip(".") in _NON_PERSON_WORDS for t in tokens):
        return None
    if not all(t[:1].isupper() for t in tokens if t):
        return None
    return lead


def load_excerpt(payload: dict, n: int = 6, max_chars: int = 2500) -> str:
    texts = [f.get("text", "") for f in payload.get("fragments", [])[:n]]
    return " ".join(texts)[:max_chars].strip()


def extract_one(fpath: Path, model: str) -> tuple[str, str]:
    """Process one classified file. Returns (status, message). Writes the roster JSON."""
    payload = json.loads(fpath.read_text(encoding="utf-8"))
    source_video = payload.get("source_video") or fpath.stem.replace(".classified", "")
    video_id = parse_source_video(source_video)[0] or payload.get("video_id") or source_video
    out_path = SPEAKERS_DIR / f"{video_id}.json"

    info_path = RAW_DIR / f"{source_video}.info.json"
    title = source_video
    description = ""
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            title = info.get("title") or source_video
            description = (info.get("description") or "").strip()
        except Exception:  # noqa: BLE001
            pass

    title_hint = parse_title_name_hint(title)
    excerpt = load_excerpt(payload)
    result = generate_json(
        PROMPT.format(
            title=title,
            title_hint=title_hint or "(none)",
            description=(description[:1500] or "(none)"),
            excerpt=excerpt,
        ),
        model=model,
    )
    speakers = [s for s in (result.get("speakers") or []) if s.get("name")]
    primary = result.get("primary_speaker")

    # Deterministic safety net: a clean title name hint always wins.
    if title_hint:
        primary = title_hint
        if not any(s.get("name", "").lower() == title_hint.lower() for s in speakers):
            speakers.insert(0, {"name": title_hint, "role": "Speaker",
                                "affiliation": None, "confidence": 0.9})

    # Canonicalize known duplicate spellings immediately, at extraction
    # time — not just retroactively in the Postgres backfill — so a
    # newly-ingested video referencing an already-known speaker doesn't
    # add yet another spelling variant to clean up later.
    primary = canonicalize_speaker(primary)
    for s in speakers:
        s["name"] = canonicalize_speaker(s.get("name"))
    speakers = [s for s in speakers if s.get("name")]

    # The speaker registry wins for known videos. For new videos, speakers
    # listed in the description are taken as given, not left to the LLM.
    registry = get_registry()
    method = "llm+title"
    entry = registry.video(video_id)
    described = description_roster(description)
    if entry is not None:
        speakers = [{"name": n, "role": r, "affiliation": None, "confidence": 1.0} for n, r in entry.speakers]
        primary = entry.speakers[0][0] if entry.kind == "single" and entry.speakers else None
        method = "registry"
    elif len(described) >= 2:
        speakers = [{"name": registry.canonicalize(n) or n, "role": r, "affiliation": None, "confidence": 0.95}
                    for n, r in described]
        primary = None
        method = "description"
    # Names the registry does not know yet: review before adding them.
    unresolved = sorted({s["name"] for s in speakers if s["name"] not in registry.speakers})

    record = {
        "video_id": video_id,
        "source_video": source_video,
        "title": title,
        "speakers": speakers,
        "primary_speaker": primary,
        "title_name_hint": title_hint,
        "attribution": "Multiple speakers" if len(speakers) > 1 else "video-level",
        "method": method,
        "unresolved_names": unresolved,
        "model": model,
        "extracted_at": datetime.now().isoformat(),
    }
    out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    # 05_classify_fragments.py's speaker fallback (load_speaker_lookup) has
    # no automated way to see this roster once it runs in a different pod —
    # the automated ingest pipeline never runs this script at all, and
    # nothing else made this output durable. Upload alongside the local
    # write (kept, not deleted — the --force/skip-if-exists check above
    # relies on the local copy) so classify can pull it from S3 later.
    if s3_helpers.use_s3():
        try:
            s3_helpers.upload(out_path, f"raw/metadata/speakers/{video_id}.json")
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] S3 upload failed for {video_id}: {e}")

    return "ok", f"{video_id}: primary={primary!r} ({len(speakers)} speakers)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all videos")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=6, help="concurrent LLM calls")
    ap.add_argument("--model", default=SPEAKER_EXTRACTION_MODEL,
                    help=f"override model (default {SPEAKER_EXTRACTION_MODEL})")
    args = ap.parse_args()

    SPEAKERS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(CLASSIFIED_DIR.glob("*.classified.json"))
    if args.limit:
        files = files[: args.limit]

    todo = []
    skipped = 0
    for fpath in files:
        # Match extract_one()'s actual output key (payload video_id, falling back
        # to source_video, then the file stem) — checking against the filename
        # stem alone missed every video already saved under its short video_id.
        vid = fpath.stem.replace(".classified", "")
        if not args.force:
            try:
                payload = json.loads(fpath.read_text(encoding="utf-8"))
                vid = payload.get("video_id") or payload.get("source_video") or vid
            except Exception:  # noqa: BLE001
                pass
        if not args.force and (SPEAKERS_DIR / f"{vid}.json").exists():
            skipped += 1
            continue
        todo.append(fpath)

    print(f"Extracting {len(todo)} videos (skipped {skipped}) with "
          f"model={args.model} workers={args.workers}", flush=True)

    done = failed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(extract_one, f, args.model): f for f in todo}
        for fut in as_completed(futures):
            try:
                status, msg = fut.result()
                done += 1
                print(f"[{status}]   {msg}", flush=True)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"[fail] {futures[fut].name}: {exc}", flush=True)

    print(f"\nDone. extracted={done} skipped={skipped} failed={failed} "
          f"-> {SPEAKERS_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
