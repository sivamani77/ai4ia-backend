from __future__ import annotations
import sys

import json
import logging
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from app.core.error_messages import ExportSubprocessError
from app.core.paths import EXPORTS_DIR, SCRIPTS_DIR, RAW_VIDEO_DIR, GENERATED_ASSETS_DIR
from app.core.settings import settings
from app.pipeline.base import PipelineContext, StageResult, run_script
from app.pipeline.s3_helpers import use_s3, download, upload, list_keys
from app.services.job_store import append_event, load_job, save_job
from app.services.video_job_runner import update_stage, _utc_now

_S3_HLS_PREFIX = "hls"

logger = logging.getLogger("ai4ia.pipeline.stage_09_export")

PYTHON_BIN = sys.executable

_S3_RAW_VIDEO_PREFIX = "raw/youtube/ai4ia_channel/"
_S3_ASSETS_PREFIX = "generated_assets"
_S3_EXPORTS_PREFIX = "exports"

# Cross-job caches. RAW_VIDEO_DIR / GENERATED_ASSETS_DIR are the SAME persistent
# local directories used in non-S3 mode — downloading into them here (instead of
# a per-job /tmp dir that gets thrown away after every run) means a job that
# reuses source footage or generated assets from an earlier job skips the
# network entirely. Only the raw-video S3 key listing is cached in-process
# (short TTL) since the archive rarely changes and re-listing the whole prefix
# on every job was a full bucket scan for no reason.
_RAW_VIDEO_KEY_CACHE: dict[str, Any] = {"keys": None, "fetched_at": 0.0}
_RAW_VIDEO_KEY_TTL_SECONDS = 600


_ASSET_RELPATH_RE = re.compile(r'"([^"]*)/generated_assets/([^"]*)"')


def _get_needed_asset_relpaths(mdata: dict) -> set[str]:
    """Every path the manifest references under a generated_assets/ directory,
    reduced to the relative part (audio/cards/visuals) — this is exactly the
    set of files this job's export actually needs, as opposed to every asset
    any job has ever produced."""
    manifest_str = json.dumps(mdata)
    return {m.group(2) for m in _ASSET_RELPATH_RE.finditer(manifest_str) if m.group(2)}


def _rewrite_manifest_asset_paths(manifest_local: Path) -> None:
    """Normalize every generated_assets/ path in the manifest to point at
    GENERATED_ASSETS_DIR, in place. Whatever prefix stage 08 originally wrote
    (a per-job /tmp dir in S3 mode, since deleted) is replaced — only the
    relative part (audio/cards/visuals/...) is kept."""
    manifest_str = manifest_local.read_text(encoding="utf-8")
    rewritten = _ASSET_RELPATH_RE.sub(
        lambda m: f'"{GENERATED_ASSETS_DIR / m.group(2)}"',
        manifest_str,
    )
    if rewritten != manifest_str:
        manifest_local.write_text(rewritten, encoding="utf-8")


def _prepare_assets_for_export(manifest_path: str, job_id: str) -> Path:
    """
    Ensure everything the export script needs is present locally, downloading
    only what's actually missing.

    Assets and source videos are cached in the SAME persistent local
    directories used in non-S3 mode (GENERATED_ASSETS_DIR, RAW_VIDEO_DIR)
    rather than a per-job /tmp dir that got downloaded from scratch and
    discarded on every run — so a job that reuses footage or assets from an
    earlier job (very common: the same source talks feed many videos) skips
    the network for those files entirely.

    Manifest paths still need rewriting: stage 08 in S3 mode writes assets
    under a per-job /tmp dir (via AI4IA_LOCAL_ASSETS_DIR) and deletes that
    dir right after uploading, so the manifest's embedded paths point at a
    directory that no longer exists by the time stage 09 runs. Every
    generated_assets/ path gets normalized to GENERATED_ASSETS_DIR — the
    same place _download_referenced_assets_cached just ensured the files
    are — before the rewritten manifest is handed to 09_export_video.py.
    RAW_VIDEO_DIR/EXPORTS_DIR need no such rewrite: 09_export_video.py
    resolves those from its own defaults, not from manifest fields.

    Returns manifest_local.
    """
    if not use_s3():
        return Path(manifest_path)

    _t0 = time.time()
    manifest_local = GENERATED_ASSETS_DIR / f"{job_id}_manifest.json"

    # Job-scoped manifest is unique per job — always fetch fresh, never cached.
    download(f"{_S3_ASSETS_PREFIX}/{job_id}_manifest.json", manifest_local)

    try:
        mdata = json.loads(manifest_local.read_text(encoding="utf-8"))
    except Exception:
        mdata = {}

    # This job's own assets may be left from a lost run, which the cache
    # below would reuse. Assets shared across jobs are kept.
    storyline_id = str(mdata.get("storyline_id") or "")
    if job_id in storyline_id:
        shutil.rmtree(GENERATED_ASSETS_DIR / storyline_id, ignore_errors=True)

    _download_referenced_assets_cached(mdata)
    _download_source_videos_cached(mdata)
    _rewrite_manifest_asset_paths(manifest_local)

    logger.info(
        "[%s] asset prep done in %.1fs (cache-first: assets+source videos reused where already local)",
        job_id, time.time() - _t0,
    )

    return manifest_local


def _download_referenced_assets_cached(mdata: dict) -> None:
    """Download only the specific audio/card/visual files this manifest
    references, skipping any already present in the persistent local cache.

    Unlike the old full-bucket mirror (which only ever iterated keys that
    genuinely existed, from a real S3 listing), keys here are derived from
    paths embedded in the manifest — a stale or never-uploaded reference is
    possible (e.g. a generation step that failed for one item). A 404 here
    must not crash the whole stage: 09_export_video.py's own pre-render
    integrity check already gives a clear, actionable error for a truly
    missing/required asset, so just log and move on.
    """
    needed_rel = _get_needed_asset_relpaths(mdata)
    missing = [rel for rel in needed_rel if not (GENERATED_ASSETS_DIR / rel).exists()]
    if not missing:
        return

    from concurrent.futures import ThreadPoolExecutor

    def _dl(rel: str) -> None:
        try:
            download(f"{_S3_ASSETS_PREFIX}/{rel}", GENERATED_ASSETS_DIR / rel)
        except Exception as exc:
            logger.warning("Could not download referenced asset %r from S3 (will surface later if actually required): %s", rel, exc)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(_dl, missing))


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


def _cached_raw_video_keys() -> list[str]:
    """S3 key listing for the raw-video prefix, cached in-process with a TTL.

    The archive of source talks only grows via ingestion runs, not per video
    job, so re-listing the entire prefix (paginated bucket scan) on every
    single export was pure waste — this makes repeat jobs within the TTL
    window skip the LIST call entirely.
    """
    now = time.time()
    if (
        _RAW_VIDEO_KEY_CACHE["keys"] is None
        or now - _RAW_VIDEO_KEY_CACHE["fetched_at"] > _RAW_VIDEO_KEY_TTL_SECONDS
    ):
        _RAW_VIDEO_KEY_CACHE["keys"] = [
            k for k in list_keys(_S3_RAW_VIDEO_PREFIX) if k.endswith(".mp4")
        ]
        _RAW_VIDEO_KEY_CACHE["fetched_at"] = now
    return _RAW_VIDEO_KEY_CACHE["keys"]


def _source_video_cached_locally(source_video: str) -> bool:
    stem = source_video.replace(".mp4", "")
    if (RAW_VIDEO_DIR / f"{stem}.mp4").exists():
        return True
    if any(RAW_VIDEO_DIR.glob(f"{stem}*.mp4")):
        return True
    prefix = _video_id_prefix(stem)
    if prefix and any(RAW_VIDEO_DIR.glob(f"{prefix}*.mp4")):
        return True
    return False


def _download_source_videos_cached(mdata: dict) -> None:
    """Download only the source videos this manifest needs AND that aren't
    already sitting in the persistent local cache from an earlier job.

    Verifies each download actually landed on disk and raises immediately with
    the specific source_video name if not — rather than leaving a silent gap
    for 09_export_video.py's pre-render integrity check to discover far later
    with a less actionable "source video not found" message.
    """
    needed = _get_needed_source_videos(mdata)
    missing_needed = [sv for sv in needed if not _source_video_cached_locally(sv)]
    if not missing_needed:
        return

    mp4_keys = _cached_raw_video_keys()
    unresolved: list[str] = []

    def _resolve_and_download(source_video: str) -> None:
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
            return
        key = matched[0]
        local = RAW_VIDEO_DIR / Path(key).name
        if not local.exists():
            try:
                download(key, local)
            except Exception as exc:
                unresolved.append(f"{source_video!r}: download from s3://{key} failed: {exc}")
                return
        if not local.exists() or local.stat().st_size == 0:
            unresolved.append(f"{source_video!r}: download from s3://{key} did not produce a usable file at {local}")

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(_resolve_and_download, missing_needed))

    if unresolved:
        raise RuntimeError(
            f"Failed to resolve {len(unresolved)} source video(s) needed for export:\n  - "
            + "\n  - ".join(unresolved)
        )


_HLS_EXTINF_RE = re.compile(r"^#EXTINF:", re.MULTILINE)
_HLS_WATCH_INTERVAL_SECONDS = 2.0


def _watch_hls_segments(job_id: str, hls_dir: Path, stop_event) -> None:
    """Poll the HLS playlist stage_09's subprocess is writing to, mirror new
    files to S3 when the API server doesn't share a disk with this worker,
    and emit a structured job event each time the segment count grows.

    09_export_video.py runs as a separate subprocess (via run_script, which
    blocks until it exits) — it has no way to call append_event itself
    without pulling in Supabase/job_store dependencies into a script that's
    designed to run standalone with zero DB/queue dependencies. Instead,
    this runs as a background thread in the same backend process as this
    stage module, watching the shared hls_dir on disk while run_script()
    blocks — the only structured (non-log-line) signal a frontend gets that
    a new segment is ready to play.

    S3 mirroring matters whenever the Celery worker and the API process are
    separate hosts/containers (the common case) — local-disk serving alone
    only works when they share a filesystem. Segments are uploaded once
    each (immutable once written); the playlist and subs.vtt are re-uploaded
    every time they change since they're rewritten in place.
    """
    last_count = 0
    uploaded_segments: set[str] = set()
    playlist_path = hls_dir / "master.m3u8"
    subs_path = hls_dir / "subs.vtt"
    mirror_to_s3 = use_s3()

    while not stop_event.is_set():
        try:
            if playlist_path.exists():
                playlist_text = playlist_path.read_text(encoding="utf-8")
                count = len(_HLS_EXTINF_RE.findall(playlist_text))

                if mirror_to_s3:
                    for seg_path in hls_dir.glob("seg_*.ts"):
                        if seg_path.name not in uploaded_segments:
                            upload(seg_path, f"{_S3_HLS_PREFIX}/{job_id}/{seg_path.name}")
                            uploaded_segments.add(seg_path.name)
                    upload(playlist_path, f"{_S3_HLS_PREFIX}/{job_id}/master.m3u8")
                    if subs_path.exists():
                        upload(subs_path, f"{_S3_HLS_PREFIX}/{job_id}/subs.vtt")

                if count > last_count:
                    last_count = count
                    append_event(job_id, "hls_segment_ready", "09_export", {
                        "segment_count": count,
                        "hls_playlist_url": f"/api/video-jobs/{job_id}/video/hls/master.m3u8",
                    })
        except Exception:
            pass
        stop_event.wait(_HLS_WATCH_INTERVAL_SECONDS)


def run(job_id: str, request: dict[str, Any], context: PipelineContext) -> StageResult:
    manifest_path = context.asset_manifest_path
    update_stage(job_id, "09_export", "running", 5, "Starting final video export")
    append_event(job_id, "stage_started", "09_export", {"manifest_path": manifest_path})
    # A rerun after a lost worker must not reuse the lost run's partial files.
    shutil.rmtree(EXPORTS_DIR / "hls" / job_id, ignore_errors=True)

    if use_s3():
        update_stage(job_id, "09_export", "running", 8, "Downloading assets and source videos from S3")

    manifest_local = _prepare_assets_for_export(manifest_path, job_id)

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

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    video_filename = f"{job_id}_{safe_title}.mp4" if safe_title else f"{job_id}.mp4"
    video_path = EXPORTS_DIR / video_filename

    cmd = [
        PYTHON_BIN,
        str(SCRIPTS_DIR / "09_export_video.py"),
        "--manifest", str(manifest_local),
        "--output", str(video_path),
    ]
    if request.get("burn_subtitles"):
        cmd.append("--burn-subtitles")

    hls_dir = EXPORTS_DIR / "hls" / job_id
    if settings.ai4ia_enable_hls_streaming:
        cmd += ["--hls-dir", str(hls_dir)]

    update_stage(job_id, "09_export", "running", 15, "Rendering video segments")
    append_event(job_id, "export", "09_export", {"progress": 15, "message": "Rendering video segments"})

    # No env override needed here: assets/videos are already in
    # RAW_VIDEO_DIR/EXPORTS_DIR, the same paths 09_export_video.py defaults to.
    export_timeout = _export_timeout_for(float(request.get("target_duration") or 0))

    watcher_thread = None
    stop_watcher = None
    if settings.ai4ia_enable_hls_streaming:
        stop_watcher = threading.Event()
        watcher_thread = threading.Thread(
            target=_watch_hls_segments, args=(job_id, hls_dir, stop_watcher), daemon=True,
        )
        watcher_thread.start()

    try:
        code, stdout, stderr = run_script(
            job_id, "09_export", cmd, timeout=export_timeout,
        )
    finally:
        if watcher_thread is not None:
            stop_watcher.set()
            watcher_thread.join(timeout=5)

    if code != 0:
        raise ExportSubprocessError(f"09_export_video.py exited {code}: {stderr[-4000:]}")

    if not video_path.exists():
        raise ExportSubprocessError(f"Expected export video not found: {video_path}")

    export_manifest_path = EXPORTS_DIR / f"{video_path.stem}_export_manifest.json"
    timeline_metadata_path = EXPORTS_DIR / f"{video_path.stem}_timeline_metadata.json"
    srt_path = EXPORTS_DIR / f"{video_path.stem}.srt"
    vtt_path = EXPORTS_DIR / f"{video_path.stem}.vtt"

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
    if settings.ai4ia_enable_hls_streaming and (hls_dir / "master.m3u8").exists():
        job["paths"]["hls_playlist_path"] = str(hls_dir / "master.m3u8")
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
