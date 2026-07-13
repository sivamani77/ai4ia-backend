"""Single source of truth for timeline duration math.

Shared by pipeline steps 07 / 07b / 08 / 09 (scripts import it as
``app.pipeline.timeline_budget``). Every stage must measure a timeline with
``timeline_total_seconds`` and attach the result of ``build_duration_budget``
to its output JSON under the ``duration_budget`` key, so a duration drift is
attributable to the stage that introduced it.

Duration invariant enforced by later stages: the final rendered video must be
at least ``target_duration_seconds`` long (floor, not target).
"""

import os
import re
import subprocess
from pathlib import Path
from typing import Any

# Fraction above target the plan may exceed before a stage flags it.
OVERSHOOT_WARN_RATIO = 1.25

# Plan slightly past target so the duration floor survives rounding.
DURATION_FILL_MARGIN = 1.02


def _project_root() -> Path:
    # Local dev layout: <root>/backend/app/pipeline/. Docker layout: <root>/app/pipeline/.
    here = Path(__file__).resolve()
    for ancestor in (here.parents[3], here.parents[2]):
        if (ancestor / "data").is_dir():
            return ancestor
    return here.parents[3]


RAW_VIDEO_DIR = Path(
    os.environ.get("AI4IA_RAW_VIDEO_DIR", str(_project_root() / "data" / "raw" / "youtube" / "ai4ia_channel"))
)

# Small sidecar of pre-probed real video-stream durations, built by
# scripts/build_source_duration_index.py. Stages 07/07b run without the raw
# video files present (only stage 09 downloads them) so they read this instead
# of probing locally — falls back to a local ffprobe when a key is missing.
SOURCE_DURATION_INDEX_S3_KEY = "raw/metadata/source_video_durations.json"
_env_index_path = os.environ.get("AI4IA_SOURCE_DURATION_INDEX")
SOURCE_DURATION_INDEX_PATH = (
    Path(_env_index_path) if _env_index_path
    else _project_root() / "data" / "metadata" / "source_video_durations.json"
)


def timeline_segment_count_for_duration(target_duration_seconds: int) -> int:
    if target_duration_seconds <= 60:
        return 4
    if target_duration_seconds <= 120:
        return 6
    if target_duration_seconds <= 300:
        return 8
    if target_duration_seconds <= 600:
        return 12
    return 15


def timeline_card_duration(target_duration_seconds: int) -> int:
    return 4 if target_duration_seconds <= 60 else 5


def classify_timeline_segment_type(candidate: dict[str, Any]) -> str:
    if candidate.get("speaker_clip_allowed") or candidate.get("speaker_soundbite_allowed"):
        return "speaker_clip"
    if candidate.get("speaker_visual_allowed"):
        return "speaker_visual"
    if candidate.get("use_as") == "archive_visual":
        return "archive_visual"
    if candidate.get("use_as") == "speaker_soundbite":
        return "speaker_clip"
    return "narration_broll"


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def normalize_candidate_time_bounds(candidate: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    start = candidate.get("start")
    end = candidate.get("end")
    duration = candidate.get("duration")

    if start is None:
        start = candidate.get("start_seconds")
    if end is None:
        end = candidate.get("end_seconds")

    start_value = _as_float(start)
    end_value = _as_float(end)
    duration_value = _as_float(duration)

    if duration_value is None and start_value is not None and end_value is not None:
        duration_value = max(0.0, end_value - start_value)
    if end_value is None and start_value is not None and duration_value is not None:
        end_value = start_value + duration_value
    if start_value is None and end_value is not None and duration_value is not None:
        start_value = max(0.0, end_value - duration_value)

    return start_value, end_value, duration_value


def segment_effective_duration(item: dict[str, Any]) -> float:
    """Seconds this timeline item will occupy in the rendered video.

    Precedence mirrors the renderer: an explicit render window always wins
    over planning-time fields.
    """
    render_duration = _as_float(item.get("render_duration"))
    if render_duration is not None and render_duration > 0:
        return render_duration

    render_start = _as_float(item.get("render_start"))
    render_end = _as_float(item.get("render_end"))
    if render_start is not None and render_end is not None and render_end > render_start:
        return render_end - render_start

    duration = _as_float(item.get("duration"))
    if duration is not None and duration > 0:
        return duration

    _, _, bounds_duration = normalize_candidate_time_bounds(item)
    if bounds_duration is not None and bounds_duration > 0:
        return bounds_duration

    return 0.0


def timeline_total_seconds(timeline: list[dict[str, Any]] | None) -> float:
    return round(sum(segment_effective_duration(item) for item in (timeline or [])), 3)


def build_duration_budget(
    timeline: list[dict[str, Any]] | None,
    target_duration_seconds: float | int | None,
    stage: str,
) -> dict[str, Any]:
    """Per-stage duration report attached to stage output JSON as ``duration_budget``."""
    items = timeline or []
    total = timeline_total_seconds(items)
    target = _as_float(target_duration_seconds)

    segments = []
    unmeasured = 0
    for index, item in enumerate(items, start=1):
        seconds = segment_effective_duration(item)
        if seconds <= 0:
            unmeasured += 1
        segments.append(
            {
                "index": index,
                "type": item.get("segment_type") or item.get("type") or "unknown",
                "section_id": item.get("section_id"),
                "fragment_id": item.get("fragment_id"),
                "seconds": round(seconds, 3),
            }
        )

    budget: dict[str, Any] = {
        "stage": stage,
        "target_seconds": target,
        "planned_total_seconds": total,
        "segment_count": len(items),
        "unmeasured_segment_count": unmeasured,
        "segments": segments,
    }
    if target is not None and target > 0:
        budget["meets_floor"] = total >= target
        budget["shortfall_seconds"] = round(max(0.0, target - total), 3)
        budget["overshoot_seconds"] = round(max(0.0, total - target), 3)
        budget["overshoot_warning"] = total > target * OVERSHOOT_WARN_RATIO
    return budget


# ---------------------------------------------------------------------------
# Duration floor: probe-clamp windows to real footage, fill timeline to target
# ---------------------------------------------------------------------------

_source_duration_cache: dict[str, float] = {}


_source_duration_index: dict[str, dict] | None = None


def _load_source_duration_index() -> dict[str, dict]:
    """Load the pre-probed duration sidecar once per process. Missing file → {}."""
    global _source_duration_index
    if _source_duration_index is not None:
        return _source_duration_index
    _source_duration_index = {}
    if SOURCE_DURATION_INDEX_PATH.exists():
        try:
            import json
            _source_duration_index = json.loads(SOURCE_DURATION_INDEX_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _source_duration_index = {}
    return _source_duration_index


def probe_source_video_duration(source_video: str | None, fragment_id: str | None = None) -> float:
    """Real video-stream duration of a source file, cached. 0.0 = unknown/not found.

    Checks the pre-built sidecar index first (works even when the raw video
    file itself isn't present, e.g. in the 07/07b planning containers), then
    falls back to probing a local file directly (dev/09-export environments).
    """
    key = str(source_video or "") or re.sub(r"_\d{4}$", "", str(fragment_id or ""))
    if not key:
        return 0.0
    if key in _source_duration_cache:
        return _source_duration_cache[key]

    indexed = _load_source_duration_index().get(key)
    if indexed and float(indexed.get("video_stream_seconds") or 0.0) > 0:
        duration = float(indexed["video_stream_seconds"])
        _source_duration_cache[key] = duration
        return duration

    path = RAW_VIDEO_DIR / f"{key}.mp4"
    if not path.exists():
        matches = sorted(RAW_VIDEO_DIR.glob(f"{key}*.mp4"))
        path = matches[0] if matches else None

    duration = 0.0
    if path is not None:
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=duration",
                    "-of", "csv=p=0",
                    str(path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            duration = float(result.stdout.strip() or 0.0)
        except (subprocess.SubprocessError, ValueError, OSError):
            duration = 0.0

    _source_duration_cache[key] = duration
    return duration


def clamp_timeline_windows_to_source(timeline: list[dict[str, Any]]) -> list[str]:
    """Clamp each clip window to the real video-stream length at PLAN time.

    This moves the "video track ends before the fragment window" discovery from
    render time (where it used to silently shorten clips) to planning, where the
    fill pass can still compensate. Items whose window has no video at all are
    removed. Mutates `timeline` in place; returns human-readable notes.
    """
    notes: list[str] = []
    kept: list[dict[str, Any]] = []
    for item in timeline:
        fragment_id = item.get("fragment_id")
        if not fragment_id or item.get("start") is None:
            kept.append(item)
            continue
        stream_dur = probe_source_video_duration(item.get("source_video"), fragment_id)
        if stream_dur <= 0.0:
            kept.append(item)  # unknown source length — renderer preflight will judge
            continue
        try:
            start = float(item["start"])
            end = float(item.get("end") or start)
        except (TypeError, ValueError):
            kept.append(item)
            continue
        if start >= stream_dur - 1.0:
            notes.append(
                f"dropped {str(fragment_id)[:50]!r}: window starts at {start:.1f}s but video "
                f"stream ends at {stream_dur:.1f}s"
            )
            continue
        if end > stream_dur:
            item["end"] = round(stream_dur - 0.1, 3)
            item["duration"] = round(item["end"] - start, 3)
            notes.append(
                f"clamped {str(fragment_id)[:50]!r}: end {end:.1f}s → {item['end']:.1f}s "
                f"(video stream is {stream_dur:.1f}s)"
            )
        # Speaker clips carry a separate PADDED render window (render_start/render_end,
        # set by Step 08's post-roll padding + quote expansion). That padding runs AFTER
        # the plan-time clamp above, so it can push render_end past real footage even when
        # start/end were in-bounds — clamp it here too so the renderer never gets a window
        # with no picture. render_start is only clamped if it somehow lands past the video.
        r_start = item.get("render_start")
        r_end = item.get("render_end")
        if r_start is not None and r_end is not None:
            try:
                rs = float(r_start)
                re_ = float(r_end)
            except (TypeError, ValueError):
                rs = re_ = None
            if rs is not None and re_ is not None and re_ > stream_dur:
                new_end = round(stream_dur - 0.1, 3)
                if new_end > rs:
                    item["render_end"] = new_end
                    item["render_duration"] = round(new_end - rs, 3)
                    notes.append(
                        f"clamped {str(fragment_id)[:50]!r}: render_end {re_:.1f}s → {new_end:.1f}s "
                        f"(video stream is {stream_dur:.1f}s)"
                    )
        kept.append(item)
    timeline[:] = kept
    return notes


def _candidate_is_speaker_eligible(candidate: dict[str, Any]) -> bool:
    return bool(
        candidate.get("speaker_soundbite_allowed")
        or candidate.get("speaker_clip_allowed")
        or classify_timeline_segment_type(candidate) == "speaker_clip"
    )


def _candidate_fill_segment(candidate: dict[str, Any], stage: str) -> dict[str, Any] | None:
    """Build a REAL speaker-clip segment from a pool candidate for duration top-up.

    Only speaker-eligible candidates qualify — they carry their own source audio,
    so they add genuine documentary content (a real voice), never silent filler.
    Non-speaker candidates return None: we would rather deliver a slightly shorter
    honest video than pad it with muted b-roll. (This is the deliberate policy
    change from the earlier fill behaviour that dumped silent archive_visual
    clips at the tail.)
    """
    fragment_id = str(candidate.get("fragment_id") or "").strip()
    if not fragment_id:
        return None
    if not _candidate_is_speaker_eligible(candidate):
        return None
    start, end, _ = normalize_candidate_time_bounds(candidate)
    if start is None or end is None or end - start < 3.0:
        return None

    stream_dur = probe_source_video_duration(candidate.get("source_video"), fragment_id)
    if stream_dur > 0.0:
        if start >= stream_dur - 1.0:
            return None
        end = min(end, stream_dur - 0.1)
        if end - start < 3.0:
            return None

    return {
        "type": "speaker_clip",
        "segment_type": "speaker_clip",
        "audio_mode": "source_audio",
        "muted": False,
        "fragment_id": fragment_id,
        "source_video": candidate.get("source_video"),
        "video_year": candidate.get("video_year"),
        "speaker": candidate.get("speaker"),
        "speaker_attribution": candidate.get("speaker_attribution"),
        "start": round(float(start), 3),
        "end": round(float(end), 3),
        "render_start": round(float(start), 3),
        "render_end": round(float(end), 3),
        "render_duration": round(float(end - start), 3),
        "duration": round(float(end - start), 3),
        "moment_text": candidate.get("moment_text") or candidate.get("text") or "",
        "subtitle_text": candidate.get("moment_text") or candidate.get("quote_text") or "",
        "relevance_role": candidate.get("relevance_role"),
        "added_by_duration_fill": True,
        "fill_stage": stage,
        "needs_validation": stage != "selected",
    }


def fill_timeline_to_target(
    timeline: list[dict[str, Any]],
    candidate_pool: dict[str, list[dict[str, Any]]],
    target_duration_seconds: float | int,
) -> dict[str, Any]:
    """Top up a planned timeline toward the duration floor by adding ONLY real,
    speaker-eligible clips (source audio) that aren't already used. Mutates
    `timeline` in place.

    Policy (per product decision): never pad with silent b-roll or invented
    narration. If the genuinely relevant speaker pool is exhausted before the
    floor is reached, stop and report `content_exhausted` — the caller ships
    the honest best-effort duration with a warning, it does NOT fail the job.

    Per-source cap is 2 (kept low so the film doesn't repeat the same talk).
    Returns a fill report.
    """
    floor = float(target_duration_seconds) * DURATION_FILL_MARGIN
    total_before = timeline_total_seconds(timeline)
    used_fids = {str(item.get("fragment_id")) for item in timeline if item.get("fragment_id")}
    source_counts: dict[str, int] = {}
    for item in timeline:
        if item.get("fragment_id"):
            source = str(item.get("source_video") or "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1

    # Insert new segments before a trailing closing_card
    insert_at = len(timeline)
    if timeline and (timeline[-1].get("type") or timeline[-1].get("segment_type")) == "closing_card":
        insert_at = len(timeline) - 1

    def _rank(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # speaker-eligible first, then by relevance
        return sorted(
            candidates,
            key=lambda c: (
                0 if _candidate_is_speaker_eligible(c) else 1,
                -float(c.get("final_prompt_score") or c.get("prompt_relevance_score") or 0.0),
                -float(c.get("selection_score") or 0.0),
            ),
        )

    selected = _rank(candidate_pool.get("selected") or [])
    extended = _rank((candidate_pool.get("supporting") or []) + (candidate_pool.get("context") or []))
    # Per-source cap held at 2 throughout — no relaxation to 3 that previously
    # let the same talk appear three times back-to-back.
    ladder: list[tuple[str, list[dict[str, Any]], int]] = [
        ("selected", selected, 2),
        ("supporting", extended, 2),
    ]

    added: list[dict[str, Any]] = []
    total = total_before
    for stage, pool, per_source_cap in ladder:
        if total >= floor:
            break
        for candidate in pool:
            if total >= floor:
                break
            fragment_id = str(candidate.get("fragment_id") or "").strip()
            if not fragment_id or fragment_id in used_fids:
                continue
            source = str(candidate.get("source_video") or "unknown")
            if source_counts.get(source, 0) >= per_source_cap:
                continue
            segment = _candidate_fill_segment(candidate, stage)
            if segment is None:
                continue
            timeline.insert(insert_at, segment)
            insert_at += 1
            used_fids.add(fragment_id)
            source_counts[source] = source_counts.get(source, 0) + 1
            total += float(segment["duration"])
            added.append({"fragment_id": fragment_id, "stage": stage, "seconds": segment["duration"]})

    total_after = timeline_total_seconds(timeline)
    if total_before >= floor:
        status = "already_sufficient"
    elif total_after >= floor:
        status = "filled"
    else:
        # Relevant speaker pool exhausted — ship best effort with a warning.
        status = "content_exhausted"

    return {
        "status": status,
        "target_seconds": float(target_duration_seconds),
        "floor_seconds": round(floor, 3),
        "total_before_seconds": round(total_before, 3),
        "total_after_seconds": round(total_after, 3),
        "segments_added": added,
        "shortfall_seconds": round(max(0.0, floor - total_after), 3),
    }
