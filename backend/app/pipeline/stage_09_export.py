from __future__ import annotations
import sys

import json
import re
from pathlib import Path
from typing import Any

from app.core.paths import EXPORTS_DIR, SCRIPTS_DIR, PROJECT_ROOT
from app.pipeline.base import PipelineContext, StageResult, run_script
from app.pipeline.s3_helpers import use_s3, download, upload, list_keys, tmp_dir, cleanup_tmp_dir, shared_cache_dir
from app.services.job_store import append_event, load_job, save_job
from app.services.video_job_runner import update_stage, _utc_now

PYTHON_BIN = sys.executable

_S3_RAW_VIDEO_PREFIX = "raw/youtube/ai4ia_channel/"
_S3_ASSETS_PREFIX = "generated_assets"
_S3_EXPORTS_PREFIX = "exports"


def _prepare_assets_for_export(manifest_path: str, job_id: str) -> tuple[Path, Path, Path]:
    """
    Download everything needed for the export script to run locally.
    Returns (manifest_local, raw_video_dir, exports_dir).
    """
    if not use_s3():
        return Path(manifest_path), PROJECT_ROOT / "data" / "raw" / "youtube" / "ai4ia_channel", EXPORTS_DIR

    work = tmp_dir(f"09_{job_id}")
    manifest_local = work / "generated_assets" / "latest_asset_manifest.json"

    # Download job-scoped manifest from S3 (avoids concurrent-job collision)
    if not manifest_local.exists():
        download(f"{_S3_ASSETS_PREFIX}/{job_id}_manifest.json", manifest_local)

    # Parse manifest to find which assets are needed
    try:
        mdata = json.loads(manifest_local.read_text(encoding="utf-8"))
    except Exception:
        mdata = {}

    # Download all generated assets referenced in manifest (audio, cards, visuals)
    _download_referenced_assets(mdata, work)

    # Rewrite manifest paths to point to /tmp local copies
    manifest_local = _rewrite_manifest_paths(mdata, manifest_local, work)

    # Download source videos needed for the timeline into a cache dir shared
    # across jobs (not this job's own /tmp), so a talk already fetched for a
    # prior export is reused as-is instead of being downloaded in full again.
    raw_video_dir = shared_cache_dir("raw_youtube_ai4ia_channel")
    _download_source_videos(mdata, raw_video_dir)

    exports_dir = work / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)

    return manifest_local, raw_video_dir, exports_dir


def _download_referenced_assets(mdata: dict, work: Path) -> None:
    """Download all audio/card/visual files referenced in the manifest."""
    from app.pipeline.s3_helpers import _s3, _bucket
    s3 = _s3()
    bucket = _bucket()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=_S3_ASSETS_PREFIX + "/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(_S3_ASSETS_PREFIX):].lstrip("/")
            local = work / "generated_assets" / rel
            if not local.exists():
                local.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, key, str(local))


def _rewrite_manifest_paths(mdata: dict, manifest_local: Path, work: Path) -> Path:
    """
    Rewrite all absolute paths in manifest that point to any generated_assets directory
    (whether from /app/data or a prior stage's /tmp) to point to this stage's /tmp equivalents.
    """
    new_assets_prefix = str(work / "generated_assets")
    manifest_str = json.dumps(mdata)

    # Replace any JSON string value containing .../generated_assets/...
    # Uses a quoted-string-aware pattern to avoid crossing field boundaries.
    manifest_str = re.sub(
        r'"([^"]*)/generated_assets/([^"]*)"',
        lambda m: f'"{new_assets_prefix}/{m.group(2)}"',
        manifest_str,
    )
    # Also handle storylines paths
    manifest_str = re.sub(
        r'"([^"]*)/storylines/([^"]*)"',
        lambda m: f'"{work / "storylines"}/{m.group(2)}"',
        manifest_str,
    )
    updated = json.loads(manifest_str)
    manifest_local.write_text(json.dumps(updated), encoding="utf-8")
    return manifest_local


def _get_needed_source_videos(mdata: dict) -> set[str]:
    """Collect all source_video values from every location in the manifest."""
    needed = set()

    def _collect(items: list) -> None:
        for item in items or []:
            sv = item.get("source_video")
            if sv:
                needed.add(str(sv))

    _collect(mdata.get("export_timeline"))
    _collect(mdata.get("archive_fragments"))

    storyline = mdata.get("storyline") or {}
    _collect(storyline.get("export_timeline"))
    _collect(storyline.get("selected_candidates"))

    # Also scan sections sub-arrays (legacy path via build_refined_timeline)
    for section in storyline.get("sections") or []:
        for sub_key in ("clips", "moments", "broll_candidates", "selected_candidates"):
            _collect(section.get(sub_key))

    return needed


def _export_timeout_for(target_seconds: float) -> int:
    """Timeout budget for stage 09 (render + concat + Whisper transcribe).

    Scales with the requested duration because render time does — and it is far
    slower on a CPU-only host (libx264) than a GPU one (nvenc). ~6x the target
    covers worst-case CPU encoding of the (possibly overshooting) video plus the
    concat re-encode plus Whisper; a GPU host finishes well inside this ceiling.
    Floored at 1800s so short requests still allow for Whisper on a slow host.
    """
    return max(1800, int(600 + target_seconds * 6))


_VIDEO_ID_PREFIX_RE = re.compile(r"^(\d{4}_[A-Za-z0-9_-]{6,})")


def _video_id_prefix(name: str) -> str | None:
    """Extract the stable '0254_RXXzSGoo0KQ' id prefix from a source_video or
    fragment_id string. This prefix comes straight from the yt-dlp playlist
    index + video id and is never touched by any downstream text processing —
    unlike the human-readable title suffix, which LLM round-trips (storyline
    generation, narrative assembly) can silently rewrite: fullwidth Unicode
    punctuation in real titles (：？) gets normalized to ASCII (:?) when an LLM
    echoes a fragment_id/title back in its JSON response, which breaks any
    match against the untouched S3 filename. Matching on the id prefix alone
    sidesteps that entire class of drift.
    """
    m = _VIDEO_ID_PREFIX_RE.match(name)
    return m.group(1) if m else None


def _download_source_videos(mdata: dict, raw_video_dir: Path) -> None:
    """Download only the source videos actually referenced in the manifest.

    Verifies each download actually landed on disk and raises immediately with
    the specific source_video name if not — rather than leaving a silent gap
    for 09_export_video.py's pre-render integrity check to discover far later
    with a less actionable "source video not found" message.
    """
    needed = _get_needed_source_videos(mdata)
    if not needed:
        return

    all_keys = list_keys(_S3_RAW_VIDEO_PREFIX)
    mp4_keys = [k for k in all_keys if k.endswith(".mp4")]
    unresolved: list[str] = []
    for source_video in needed:
        stem = source_video.replace(".mp4", "")

        # 1) Exact/substring match on the full title (fast path — works whenever
        #    the title text wasn't altered anywhere upstream).
        matched = [k for k in mp4_keys if stem in k]

        # 2) Fallback: match on the stable numeric/video-id prefix alone, which
        #    survives Unicode punctuation normalization in the title.
        if not matched:
            prefix = _video_id_prefix(stem)
            if prefix:
                matched = [k for k in mp4_keys if _video_id_prefix(Path(k).name) == prefix]

        if not matched:
            unresolved.append(f"{source_video!r}: no matching .mp4 key under {_S3_RAW_VIDEO_PREFIX}")
            continue
        key = matched[0]
        local = raw_video_dir / Path(key).name
        if not local.exists():
            download(key, local)
        if not local.exists() or local.stat().st_size == 0:
            unresolved.append(f"{source_video!r}: download from s3://{key} did not produce a usable file at {local}")

    if unresolved:
        raise RuntimeError(
            f"Failed to resolve {len(unresolved)} source video(s) needed for export:\n  - "
            + "\n  - ".join(unresolved)
        )


def run(job_id: str, request: dict[str, Any], context: PipelineContext) -> StageResult:
    manifest_path = context.asset_manifest_path
    update_stage(job_id, "09_export", "running", 5, "Starting final video export")
    append_event(job_id, "stage_started", "09_export", {"manifest_path": manifest_path})

    if use_s3():
        update_stage(job_id, "09_export", "running", 8, "Downloading assets and source videos from S3")

    manifest_local, raw_video_dir, exports_dir = _prepare_assets_for_export(manifest_path, job_id)

    # Determine output video path
    try:
        mdata = json.loads(manifest_local.read_text(encoding="utf-8"))
        raw_title = mdata.get("title") or mdata.get("storyline", {}).get("title") or ""
        if raw_title:
            safe_title = re.sub(r"[^a-z0-9]+", "_", raw_title.lower()).strip("_")[:40]
        else:
            safe_title = ""
    except Exception:
        safe_title = ""

    exports_dir.mkdir(parents=True, exist_ok=True)
    video_filename = f"{job_id}_{safe_title}.mp4" if safe_title else f"{job_id}.mp4"
    video_path = exports_dir / video_filename

    cmd = [
        PYTHON_BIN,
        str(SCRIPTS_DIR / "09_export_video.py"),
        "--manifest", str(manifest_local),
        "--output", str(video_path),
    ]
    if request.get("burn_subtitles"):
        cmd.append("--burn-subtitles")

    # Pass tmp paths to the script via env vars (safe: no os.environ mutation)
    env_overrides: dict[str, str] = {}
    if use_s3():
        env_overrides["AI4IA_RAW_VIDEO_DIR"] = str(raw_video_dir)
        env_overrides["AI4IA_EXPORTS_DIR"] = str(exports_dir)

    update_stage(job_id, "09_export", "running", 15, "Rendering video segments")
    append_event(job_id, "export", "09_export", {"progress": 15, "message": "Rendering video segments"})

    export_timeout = _export_timeout_for(float(request.get("target_duration") or 0))
    code, stdout, stderr = run_script(
        job_id, "09_export", cmd, extra_env=env_overrides or None, timeout=export_timeout,
    )

    if code != 0:
        raise RuntimeError(f"09_export_video.py exited {code}: {stderr[-4000:]}")

    if not video_path.exists():
        raise RuntimeError(f"Expected export video not found: {video_path}")

    export_manifest_path = exports_dir / f"{video_path.stem}_export_manifest.json"
    timeline_metadata_path = exports_dir / f"{video_path.stem}_timeline_metadata.json"
    srt_path = exports_dir / f"{video_path.stem}.srt"
    vtt_path = exports_dir / f"{video_path.stem}.vtt"

    # Upload all export outputs to S3
    if use_s3():
        update_stage(job_id, "09_export", "running", 90, "Uploading exported video to S3")
        for f in [video_path, export_manifest_path, srt_path, vtt_path, timeline_metadata_path]:
            if f.exists():
                upload(f, f"{_S3_EXPORTS_PREFIX}/{f.name}")

    # Parse export manifest
    try:
        emdata = json.loads(export_manifest_path.read_text(encoding="utf-8")) if export_manifest_path.exists() else {}
        qc = emdata.get("quality_checks") or {}
        gen_vis = emdata.get("generated_visual_segments", 0)
        muted_arc = emdata.get("muted_archive_narration_segments", 0)
        sub_cov = emdata.get("subtitles", {}).get("coverage", {})
    except Exception:
        qc, gen_vis, muted_arc, sub_cov = {}, 0, 0, {}

    job = load_job(job_id)
    job["paths"]["video_path"] = str(video_path)
    job["paths"]["export_manifest_path"] = str(export_manifest_path) if export_manifest_path.exists() else None
    job["paths"]["timeline_metadata_path"] = str(timeline_metadata_path) if timeline_metadata_path.exists() else None
    job["paths"]["srt_path"] = str(srt_path) if srt_path.exists() else None
    job["paths"]["vtt_path"] = str(vtt_path) if vtt_path.exists() else None
    if use_s3():
        job["paths"]["video_s3_key"] = f"{_S3_EXPORTS_PREFIX}/{video_path.name}"
        if export_manifest_path.exists():
            job["paths"]["export_manifest_s3_key"] = f"{_S3_EXPORTS_PREFIX}/{export_manifest_path.name}"
        if srt_path.exists():
            job["paths"]["srt_s3_key"] = f"{_S3_EXPORTS_PREFIX}/{srt_path.name}"
        if vtt_path.exists():
            job["paths"]["vtt_s3_key"] = f"{_S3_EXPORTS_PREFIX}/{vtt_path.name}"
    job["updated_at"] = _utc_now()
    save_job(job_id, job)

    update_stage(job_id, "09_export", "completed", 100,
                 f"Export complete — generated_visuals={gen_vis}, muted_archive={muted_arc}")
    append_event(job_id, "export", "09_export", {
        "progress": 100,
        "message": "Video export complete",
        "video_path": str(video_path),
        "generated_visual_segments": gen_vis,
        "muted_archive_narration_segments": muted_arc,
        "subtitle_coverage": sub_cov,
    })
    append_event(job_id, "stage_completed", "09_export", {
        "video_path": str(video_path),
        "quality_checks": qc,
    })

    context.video_path = str(video_path)
    context.export_manifest_path = str(export_manifest_path)
    context.srt_path = str(srt_path) if srt_path.exists() else None
    context.vtt_path = str(vtt_path) if vtt_path.exists() else None
    # Cleanup deferred to video_job_runner after S3 upload completes

    return StageResult(
        stage="09_export",
        status="completed",
        message=f"Export complete — generated_visuals={gen_vis}, muted_archive={muted_arc}",
        data={
            "video_path": str(video_path),
            "export_manifest_path": str(export_manifest_path),
            "generated_visual_segments": gen_vis,
            "muted_archive_narration_segments": muted_arc,
        },
    )
