"""One-off speaker repair: makes every stored copy of a fragment's speaker
match the speaker registry (config/speaker_registry.json).

Stores touched: Postgres fragments, the Chroma collection, the speaker
tables, and the per-video roster and classified-fragment files (S3 or local).
See docs/speaker-cleanup-plan.md for the rollout.

Runs from API startup when a switch below is on, or from the CLI:

    python -m app.services.speaker_repair --dry-run
    python -m app.services.speaker_repair --apply
    python -m app.services.speaker_repair --rollback
    python -m app.services.speaker_repair --cleanup-preview
    python -m app.services.speaker_repair --cleanup

The CLI refuses non-local stores unless --target=cluster is passed.
Every run logs lines prefixed "[speaker_repair]" and ends with one RESULT line.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import socket
import sys
import threading
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import psycopg2
import psycopg2.extras

from app.core.paths import CLASSIFIED_FRAGMENTS_DIR, PROJECT_ROOT, SPEAKERS_DIR
from app.core.settings import settings
from app.pipeline import s3_helpers
from app.services.speaker_registry import (
    MULTIPLE_SPEAKERS,
    REGISTRY_PATH,
    VIDEO_LEVEL,
    SpeakerRegistry,
    get_registry,
)
from app.services.video_naming import parse_source_video

logger = logging.getLogger("ai4ia.speaker_repair")

# --- Deploy switches (docs/speaker-cleanup-plan.md, Phase 7) -----------------
REPAIR_NAME = "speaker_registry_v1"  # done-marker key; bump to run again
RUN_SPEAKER_REPAIR = True            # True for deploys 1 and 2, then back to False
SPEAKER_REPAIR_APPLY = False         # False = dry run and report only
RUN_SPEAKER_ROLLBACK = False         # restore the values saved before the repair
RUN_SPEAKER_CLEANUP = False          # Phase 6: remove old data once the repair has settled
SPEAKER_CLEANUP_APPLY = False        # False = preview only; True = delete (rollback no longer possible)
MIN_DAYS_BEFORE_CLEANUP = 7          # the real cleanup waits this long after the repair was applied

MAX_APPLY_ATTEMPTS = 3
_LOCK_KEY = 73_310_001
_FIELDS = ("video_id", "speaker", "speaker_attribution", "speaker_roster")
_CHROMA_PAGE = 500
_CHROMA_BATCH = 200

_S3_PREFIXES = {"roster": "raw/metadata/speakers/", "classified": "raw/fragments_classified/"}
_LOCAL_DIRS = {"roster": SPEAKERS_DIR, "classified": CLASSIFIED_FRAGMENTS_DIR}
_S3_BACKUP_PREFIX = f"backups/{REPAIR_NAME}/"
_S3_REPORT_PREFIX = f"reports/data_repairs/{REPAIR_NAME}/"
_LOCAL_ROOT = PROJECT_ROOT / "backups" / REPAIR_NAME

_MODE_STATUS = {"dry_run": "dry_run", "apply": "applied", "rollback": "rolled_back",
                "cleanup_preview": "cleanup_preview", "cleanup": "cleaned"}
_CLEANUP_MODES = ("cleanup_preview", "cleanup")
_CLEANUP_NAME = f"{REPAIR_NAME}_cleanup"


class RepairError(Exception):
    pass


def _log(message: str, *args: Any, level: int = logging.INFO) -> None:
    logger.log(level, "[speaker_repair] " + message, *args)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Fragment:
    fragment_id: str
    source_video: str
    raw: dict[str, str | None]  # stored values; None when the field is absent

    def value(self, field: str) -> str:
        return self.raw.get(field) or ""


# --- Stores ------------------------------------------------------------------

def _read_postgres(conn) -> dict[str, Fragment]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "select fragment_id, source_video, video_id, speaker, speaker_attribution, "
            "extra->>'speaker_roster' as speaker_roster from fragments"
        )
        rows = cur.fetchall()
    conn.commit()
    return {r["fragment_id"]: Fragment(r["fragment_id"], r["source_video"] or "",
                                       {f: r[f] for f in _FIELDS}) for r in rows}


def _chroma_collection():
    from app.services.embedding_index import _make_client, _make_http_collection

    if settings.chroma_mode == "http":
        return _make_http_collection(settings.chroma_collection)
    return _make_client().get_collection(settings.chroma_collection)


def _read_chroma(collection) -> dict[str, Fragment]:
    total = collection.count()
    fragments: dict[str, Fragment] = {}
    offset = 0
    while offset < total:
        page = collection.get(limit=_CHROMA_PAGE, offset=offset, include=["metadatas"])
        ids = page["ids"]
        if not ids:
            break
        for fragment_id, meta in zip(ids, page["metadatas"]):
            meta = meta or {}
            fragments[fragment_id] = Fragment(fragment_id, meta.get("source_video") or "",
                                              {f: meta.get(f) for f in _FIELDS})
        offset += len(ids)
    if len(fragments) != total:
        raise RepairError(f"read {len(fragments)} Chroma fragments, collection reports {total}")
    return fragments


# --- Plan --------------------------------------------------------------------

def _targets(fragments: dict[str, Fragment], registry: SpeakerRegistry) -> dict[str, dict[str, str]]:
    return {fid: registry.speaker_fields(f.source_video, f.value("speaker"),
                                         f.value("speaker_attribution"), f.value("video_id"))
            for fid, f in fragments.items()}


def _changes(fragments: dict[str, Fragment], targets: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    return {fid: t for fid, t in targets.items()
            if any(fragments[fid].value(k) != t[k] for k in _FIELDS)}


def _summarize(fragments: dict[str, Fragment], changes: dict[str, dict[str, str]]) -> dict[str, Any]:
    by_field = Counter(k for fid, t in changes.items() for k in _FIELDS if fragments[fid].value(k) != t[k])
    return {"fragments": len(fragments), "changed": len(changes), "by_field": dict(by_field)}


def _video_changes(fragments: dict[str, Fragment], changes: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for fid, t in changes.items():
        entry = grouped.setdefault(t["video_id"], {"video_id": t["video_id"], "old": Counter(),
                                                   "new_speaker": t["speaker"], "new_roster": t["speaker_roster"],
                                                   "new_attribution": t["speaker_attribution"], "fragments": 0})
        entry["old"][fragments[fid].value("speaker") or "∅"] += 1
        entry["fragments"] += 1
    return [dict(v, old=dict(v["old"])) for _, v in sorted(grouped.items())]


def _plan(pg: dict[str, Fragment], chroma: dict[str, Fragment], registry: SpeakerRegistry) -> dict[str, Any]:
    pg_targets, chroma_targets = _targets(pg, registry), _targets(chroma, registry)
    pg_changes, chroma_changes = _changes(pg, pg_targets), _changes(chroma, chroma_targets)
    known = set(registry.video_ids())
    unknown_videos = sorted({t["video_id"] for t in chroma_targets.values()} - known)
    names = Counter(t["speaker"] for t in chroma_targets.values() if t["speaker"])
    return {
        "pg_changes": pg_changes,
        "chroma_changes": chroma_changes,
        "report": {
            "postgres": _summarize(pg, pg_changes),
            "chroma": _summarize(chroma, chroma_changes),
            "only_in_postgres": len(set(pg) - set(chroma)),
            "only_in_chroma": len(set(chroma) - set(pg)),
            "videos_not_in_registry": unknown_videos,
            "speakers_not_in_registry": {n: c for n, c in names.items() if n not in registry.speakers},
            "video_changes": _video_changes(chroma, chroma_changes),
        },
    }


# --- Apply -------------------------------------------------------------------

def _backup(conn, store: str, fragments: dict[str, Fragment]) -> int:
    """Save current values; returns how many were newly saved."""
    rows = [(REPAIR_NAME, store, f.fragment_id, f.raw["video_id"], f.raw["speaker"],
             f.raw["speaker_attribution"], f.raw["speaker_roster"]) for f in fragments.values()]
    count_sql = "select count(*) from fragment_speaker_backup where repair_name = %s and store = %s"
    with conn.cursor() as cur:
        cur.execute(count_sql, (REPAIR_NAME, store))
        before = cur.fetchone()[0]
        # Insert-only: a re-run must never overwrite the first saved values.
        psycopg2.extras.execute_values(
            cur,
            "insert into fragment_speaker_backup (repair_name, store, fragment_id, video_id, speaker, "
            "speaker_attribution, speaker_roster) values %s on conflict do nothing",
            rows, page_size=1000,
        )
        cur.execute(count_sql, (REPAIR_NAME, store))
        after = cur.fetchone()[0]
    conn.commit()
    return after - before


def _apply_postgres(conn, changes: dict[str, dict[str, str]]) -> int:
    rows = [(fid, t["video_id"], t["speaker"], t["speaker_attribution"], t["speaker_roster"])
            for fid, t in changes.items()]
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            update fragments as f set
                video_id = nullif(v.video_id, ''),
                speaker = nullif(v.speaker, ''),
                speaker_attribution = nullif(v.attribution, ''),
                extra = coalesce(f.extra, '{}'::jsonb) || jsonb_build_object('speaker_roster', v.roster)
            from (values %s) as v(fragment_id, video_id, speaker, attribution, roster)
            where f.fragment_id = v.fragment_id
            """,
            rows, page_size=500,
        )
    conn.commit()
    return len(rows)


def _apply_chroma(collection, changes: dict[str, dict[str, str]]) -> int:
    ids = list(changes)
    for start in range(0, len(ids), _CHROMA_BATCH):
        batch = ids[start:start + _CHROMA_BATCH]
        # Chroma merges metadata on update, so only the speaker fields are sent.
        collection.update(ids=batch, metadatas=[{k: changes[i][k] for k in _FIELDS} for i in batch])
    return len(ids)


def _apply_tables(registry: SpeakerRegistry) -> dict[str, int]:
    from app.db.repositories.speaker_repository import SpeakerRepository

    repo = SpeakerRepository()
    upserted = repo.upsert_from_registry(registry)
    deleted = repo.delete_not_in_registry(registry)
    return {"speakers": upserted["speakers"], "aliases": upserted["aliases"],
            "deleted_speakers": deleted["speakers"], "deleted_aliases": deleted["aliases"],
            "video_speakers": repo.replace_video_speakers(registry)}


# --- Files (S3 or local) -----------------------------------------------------

class _Files:
    """Roster and classified-fragment files, in S3 or on local disk."""

    def __init__(self) -> None:
        self.s3 = s3_helpers.use_s3()
        self._client = s3_helpers._s3() if self.s3 else None
        self._bucket = s3_helpers._bucket() if self.s3 else ""

    def names(self, kind: str) -> list[str]:
        if self.s3:
            prefix = _S3_PREFIXES[kind]
            return [k[len(prefix):] for k in s3_helpers.list_keys(prefix) if k.endswith(".json")]
        directory = _LOCAL_DIRS[kind]
        return sorted(p.name for p in directory.glob("*.json")) if directory.exists() else []

    def read(self, kind: str, name: str) -> bytes | None:
        if self.s3:
            try:
                return self._client.get_object(Bucket=self._bucket, Key=_S3_PREFIXES[kind] + name)["Body"].read()
            except self._client.exceptions.NoSuchKey:
                return None
        path = _LOCAL_DIRS[kind] / name
        return path.read_bytes() if path.exists() else None

    def write(self, kind: str, name: str, data: bytes) -> None:
        if self.s3:
            self._client.put_object(Bucket=self._bucket, Key=_S3_PREFIXES[kind] + name, Body=data,
                                    ContentType="application/json")
            return
        path = _LOCAL_DIRS[kind] / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def delete(self, kind: str, name: str) -> None:
        if self.s3:
            self._client.delete_object(Bucket=self._bucket, Key=_S3_PREFIXES[kind] + name)
        else:
            (_LOCAL_DIRS[kind] / name).unlink(missing_ok=True)

    def _backup_ref(self, name: str) -> str | Path:
        return _S3_BACKUP_PREFIX + name if self.s3 else _LOCAL_ROOT / "files" / name

    def read_backup(self, name: str) -> bytes | None:
        ref = self._backup_ref(name)
        if self.s3:
            try:
                return self._client.get_object(Bucket=self._bucket, Key=ref)["Body"].read()
            except self._client.exceptions.NoSuchKey:
                return None
        return ref.read_bytes() if ref.exists() else None

    def write_backup(self, name: str, data: bytes, *, overwrite: bool = False) -> None:
        """Insert-only unless overwrite: the first saved copy is the original."""
        ref = self._backup_ref(name)
        if self.s3:
            if not overwrite and s3_helpers.key_exists(ref):
                return
            self._client.put_object(Bucket=self._bucket, Key=ref, Body=data)
            return
        if ref.exists() and not overwrite:
            return
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_bytes(data)

    def backup_names(self, sub: str) -> list[str]:
        """Saved copies under sub/, as names relative to the backup root."""
        if self.s3:
            return [k[len(_S3_BACKUP_PREFIX):] for k in s3_helpers.list_keys(f"{_S3_BACKUP_PREFIX}{sub}/")]
        root = _LOCAL_ROOT / "files" / sub
        return [f"{sub}/{p.name}" for p in sorted(root.glob("*"))] if root.exists() else []

    def delete_backup(self, name: str) -> None:
        ref = self._backup_ref(name)
        if self.s3:
            self._client.delete_object(Bucket=self._bucket, Key=ref)
        else:
            ref.unlink(missing_ok=True)


def _dump(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _roster_record(existing: dict[str, Any], video_id: str, source_video: str,
                   registry: SpeakerRegistry) -> dict[str, Any]:
    record = dict(existing)
    entry = registry.video(video_id)
    if entry is None:
        speakers = []
        for s in existing.get("speakers") or []:
            name = registry.canonicalize(s.get("name"))
            if name:
                speakers.append(dict(s, name=name))
        record["speakers"] = speakers
        record["primary_speaker"] = registry.canonicalize(existing.get("primary_speaker"))
        return record
    record.update({
        "video_id": video_id,
        "source_video": source_video or existing.get("source_video"),
        "speakers": [{"name": n, "role": r} for n, r in entry.speakers],
        "primary_speaker": entry.speakers[0][0] if entry.kind == "single" and entry.speakers else None,
        "attribution": MULTIPLE_SPEAKERS if entry.kind == "panel" else VIDEO_LEVEL,
        "method": "registry",
        "registry_version": registry.version,
    })
    return record


def _apply_files(files: _Files, registry: SpeakerRegistry, source_videos: dict[str, str]) -> dict[str, int]:
    marker = "_FILES_APPLIED"
    if files.read_backup(marker) is not None:
        _log("STEP files skipped: already rewritten by an earlier run")
        return {"skipped": 1}
    counts = Counter()

    # Classified fragments.
    for name in files.names("classified"):
        raw = files.read("classified", name)
        payload = json.loads(raw)
        payload_source = payload.get("source_video") or name.replace(".classified.json", "")
        for fragment in payload.get("fragments") or []:
            fragment.update(registry.speaker_fields(
                fragment.get("source_video") or payload_source, fragment.get("speaker"),
                fragment.get("speaker_attribution"), fragment.get("video_id")))
        parsed_id, _ = parse_source_video(payload_source)
        if parsed_id:
            payload["video_id"] = parsed_id
        new = _dump(payload)
        if new != raw:
            files.write_backup(f"classified/{name}", raw)
            files.write("classified", name, new)
            counts["classified_rewritten"] += 1

    # Rosters: rewrite existing files in place, create missing ones by video id.
    covered: set[str] = set()
    for name in files.names("roster"):
        raw = files.read("roster", name)
        existing = json.loads(raw)
        video_id = parse_source_video(existing.get("source_video") or "")[0] or existing.get("video_id") or ""
        covered.add(video_id)
        new = _dump(_roster_record(existing, video_id, source_videos.get(video_id, ""), registry))
        if new != raw:
            files.write_backup(f"roster/{name}", raw)
            files.write("roster", name, new)
            counts["rosters_rewritten"] += 1
    created = []
    for video_id in registry.video_ids():
        if video_id in covered or video_id not in source_videos:
            continue
        name = f"{video_id}.json"
        files.write("roster", name, _dump(_roster_record({}, video_id, source_videos[video_id], registry)))
        created.append(name)
    counts["rosters_created"] = len(created)
    files.write_backup("created_rosters.json", _dump(created), overwrite=True)
    files.write_backup(marker, _now().encode(), overwrite=True)
    return dict(counts)


def _rollback_files(files: _Files) -> dict[str, int]:
    counts = Counter()
    for kind in ("classified", "roster"):
        for name in files.names(kind):
            original = files.read_backup(f"{kind}/{name}")
            if original is not None and original != files.read(kind, name):
                files.write(kind, name, original)
                counts[f"{kind}_restored"] += 1
    created = files.read_backup("created_rosters.json")
    for name in json.loads(created) if created else []:
        files.delete("roster", name)
        counts["rosters_deleted"] += 1
    files.delete_backup("_FILES_APPLIED")
    return dict(counts)


# --- Checks ------------------------------------------------------------------

def _checks(pg: dict[str, Fragment], chroma: dict[str, Fragment], registry: SpeakerRegistry,
            before: dict[str, int]) -> list[dict[str, Any]]:
    results = []

    def check(name: str, ok: bool, detail: str) -> None:
        results.append({"name": name, "ok": ok, "detail": detail})
        _log("CHECK %s %s %s", name, "PASS" if ok else "FAIL", detail,
             level=logging.INFO if ok else logging.ERROR)

    pg_left = _changes(pg, _targets(pg, registry))
    chroma_left = _changes(chroma, _targets(chroma, registry))
    check("postgres_matches_registry", not pg_left, f"{len(pg_left)} fragments differ")
    check("chroma_matches_registry", not chroma_left, f"{len(chroma_left)} fragments differ")
    disagree = [fid for fid in set(pg) & set(chroma)
                if any(pg[fid].value(k) != chroma[fid].value(k) for k in _FIELDS)]
    check("stores_agree", not disagree, f"{len(disagree)} shared fragments differ between stores")
    cut = [f for f in (*pg.values(), *chroma.values())
           if parse_source_video(f.source_video)[0] not in (None, f.value("video_id"))]
    check("no_cut_short_video_ids", not cut, f"{len(cut)} fragments")
    check("fragment_counts_unchanged",
          before == {"postgres": len(pg), "chroma": len(chroma)},
          f"before={before} after={{'postgres': {len(pg)}, 'chroma': {len(chroma)}}}")
    return results


# --- Rollback ----------------------------------------------------------------

def _rollback(conn, report: dict[str, Any]) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            update fragments as f set
                video_id = b.video_id,
                speaker = b.speaker,
                speaker_attribution = b.speaker_attribution,
                extra = case when b.speaker_roster is null
                             then coalesce(f.extra, '{}'::jsonb) - 'speaker_roster'
                             else coalesce(f.extra, '{}'::jsonb) || jsonb_build_object('speaker_roster', b.speaker_roster)
                        end
            from fragment_speaker_backup b
            where b.repair_name = %s and b.store = 'postgres' and b.fragment_id = f.fragment_id
            """,
            (REPAIR_NAME,),
        )
        pg_restored = cur.rowcount
        cur.execute(
            "select fragment_id, video_id, speaker, speaker_attribution, speaker_roster "
            "from fragment_speaker_backup where repair_name = %s and store = 'chroma'",
            (REPAIR_NAME,),
        )
        chroma_rows = cur.fetchall()
    conn.commit()
    _log("STEP postgres restored=%d", pg_restored)

    collection = _chroma_collection()
    for start in range(0, len(chroma_rows), _CHROMA_BATCH):
        batch = chroma_rows[start:start + _CHROMA_BATCH]
        # None removes a key that did not exist before the repair.
        collection.update(ids=[r[0] for r in batch],
                           metadatas=[dict(zip(_FIELDS, r[1:])) for r in batch])
    _log("STEP chroma restored=%d", len(chroma_rows))

    files = _rollback_files(_Files())
    _log("STEP files %s", files)
    report["restored"] = {"postgres": pg_restored, "chroma": len(chroma_rows), "files": files}


# --- Marker, lock, report ----------------------------------------------------

def _registry_fingerprint(registry: SpeakerRegistry) -> str:
    try:
        digest = hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest()[:12]
    except FileNotFoundError:
        digest = "missing"
    return f"{registry.version}:{digest}"


def _read_marker(conn, name: str = REPAIR_NAME) -> dict[str, Any] | None:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("select * from data_repairs where name = %s", (name,))
        row = cur.fetchone()
    conn.commit()
    return dict(row) if row else None


def _write_marker(conn, status: str, mode: str, fingerprint: str, summary: dict[str, Any],
                  failed_apply: bool, name: str = REPAIR_NAME) -> None:
    conn.rollback()
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into data_repairs (name, status, mode, attempts, registry_version, summary, updated_at)
            values (%s, %s, %s, %s, %s, %s, now())
            on conflict (name) do update set
                status = excluded.status,
                mode = excluded.mode,
                attempts = case when excluded.status = 'applied' then data_repairs.attempts
                                else data_repairs.attempts + excluded.attempts end,
                registry_version = excluded.registry_version,
                summary = excluded.summary,
                updated_at = now()
            """,
            (name, status, mode, 1 if failed_apply else 0, fingerprint, psycopg2.extras.Json(summary)),
        )
    conn.commit()


def _skip_reason(mode: str, marker: dict[str, Any] | None, fingerprint: str, force: bool) -> str | None:
    if force:
        return None
    status = (marker or {}).get("status")
    if mode == "rollback":
        if status == "rolled_back":
            return "already_done"
        return None if status in ("applied", "failed") else "nothing_to_roll_back"
    if status == "applied":
        return "already_done"
    if mode == "apply" and status == "failed" and (marker or {}).get("attempts", 0) >= MAX_APPLY_ATTEMPTS:
        return "too_many_failures"
    if mode == "dry_run" and status == "dry_run" and (marker or {}).get("registry_version") == fingerprint:
        return "already_done"
    return None


def _cleanup_skip_reason(mode: str, repair: dict[str, Any] | None, cleanup: dict[str, Any] | None,
                         force: bool, now: datetime | None = None) -> str | None:
    if (repair or {}).get("status") != "applied":
        return "repair_not_applied"
    status = (cleanup or {}).get("status")
    if status == "cleaned":
        return "already_done"
    if force:
        return None
    if mode == "cleanup_preview":
        return "already_done" if status == "cleanup_preview" else None
    applied_at = (repair or {}).get("updated_at")
    if applied_at and (now or datetime.now(timezone.utc)) - applied_at < timedelta(days=MIN_DAYS_BEFORE_CLEANUP):
        return "too_soon_after_apply"
    return None


def _write_report(report: dict[str, Any], mode: str) -> str:
    data = json.dumps(report, indent=2, ensure_ascii=False, default=str).encode("utf-8")
    stamp = report["finished_at"].replace(":", "").replace("-", "")
    try:
        if s3_helpers.use_s3():
            client, bucket = s3_helpers._s3(), s3_helpers._bucket()
            for key in (f"{_S3_REPORT_PREFIX}{mode}.json", f"{_S3_REPORT_PREFIX}history/{mode}-{stamp}.json"):
                client.put_object(Bucket=bucket, Key=key, Body=data, ContentType="application/json")
            return f"s3://{bucket}/{_S3_REPORT_PREFIX}{mode}.json"
        directory = _LOCAL_ROOT / "reports"
        (directory / "history").mkdir(parents=True, exist_ok=True)
        (directory / f"{mode}.json").write_bytes(data)
        (directory / "history" / f"{mode}-{stamp}.json").write_bytes(data)
        return str(directory / f"{mode}.json")
    except Exception:  # noqa: BLE001
        logger.exception("[speaker_repair] could not write the report")
        return "unwritten"


# --- Entry points ------------------------------------------------------------

def run(mode: str, *, force: bool = False) -> dict[str, Any]:
    """Run one mode: "dry_run", "apply", "rollback", "cleanup_preview" or "cleanup".
    Returns the report."""
    if mode not in _MODE_STATUS:
        raise ValueError(f"unknown mode {mode!r}")
    registry = get_registry()
    if not registry.speakers:
        raise RepairError(f"speaker registry is empty or missing at {REGISTRY_PATH}")
    fingerprint = _registry_fingerprint(registry)
    report: dict[str, Any] = {"name": REPAIR_NAME, "mode": mode, "pod": socket.gethostname(),
                              "registry": fingerprint, "started_at": _now()}
    _log("START name=%s mode=%s pod=%s registry=%s", REPAIR_NAME, mode, report["pod"], fingerprint)

    conn = psycopg2.connect(settings.database_url)
    try:
        with conn.cursor() as cur:
            cur.execute("select pg_try_advisory_lock(%s)", (_LOCK_KEY,))
            locked = cur.fetchone()[0]
        conn.commit()
        if not locked:
            _log("RESULT status=skipped mode=%s reason=locked_by_other_pod", mode)
            return {**report, "status": "skipped", "reason": "locked_by_other_pod"}
        repair_marker, cleanup_marker = _read_marker(conn), _read_marker(conn, _CLEANUP_NAME)
        if mode in _CLEANUP_MODES:
            reason = _cleanup_skip_reason(mode, repair_marker, cleanup_marker, force)
        elif mode == "rollback" and (cleanup_marker or {}).get("status") == "cleaned":
            reason = "backup_removed_by_cleanup"
        else:
            reason = _skip_reason(mode, repair_marker, fingerprint, force)
        if reason:
            _log("RESULT status=skipped mode=%s reason=%s", mode, reason)
            return {**report, "status": "skipped", "reason": reason}

        status = _MODE_STATUS[mode]
        try:
            if mode == "rollback":
                _rollback(conn, report)
            elif mode in _CLEANUP_MODES:
                _cleanup(conn, registry, report, apply=(mode == "cleanup"))
                if any(not c["ok"] for c in report.get("checks", [])):
                    status = "failed"
            else:
                _repair(conn, registry, report, apply=(mode == "apply"))
                if any(not c["ok"] for c in report.get("checks", [])):
                    status = "failed"
        except Exception as exc:  # noqa: BLE001
            logger.exception("[speaker_repair] %s failed", mode)
            report["error"] = repr(exc)
            status = "failed"

        report["status"] = status
        report["finished_at"] = _now()
        location = _write_report(report, mode)
        summary = {k: v for k, v in report.items() if k != "video_changes"}
        summary["report"] = location
        _write_marker(conn, status, mode, fingerprint, summary,
                      failed_apply=(mode == "apply" and status == "failed"),
                      name=_CLEANUP_NAME if mode in _CLEANUP_MODES else REPAIR_NAME)
        _log("RESULT status=%s mode=%s postgres_changed=%s chroma_changed=%s report=%s",
             status, mode, report.get("postgres", {}).get("changed", "-"),
             report.get("chroma", {}).get("changed", "-"), location,
             level=logging.INFO if status != "failed" else logging.ERROR)
        return report
    finally:
        conn.close()  # also releases the advisory lock


def _repair(conn, registry: SpeakerRegistry, report: dict[str, Any], *, apply: bool) -> None:
    collection = _chroma_collection()
    pg = _read_postgres(conn)
    chroma = _read_chroma(collection)
    before = {"postgres": len(pg), "chroma": len(chroma)}
    _log("STEP read postgres=%d chroma=%d", len(pg), len(chroma))

    plan = _plan(pg, chroma, registry)
    report.update(plan["report"])
    _log("STEP plan postgres_changes=%d chroma_changes=%d videos_changed=%d videos_not_in_registry=%d",
         len(plan["pg_changes"]), len(plan["chroma_changes"]), len(report["video_changes"]),
         len(report["videos_not_in_registry"]))
    if not apply:
        return

    saved = _backup(conn, "postgres", pg) + _backup(conn, "chroma", chroma)
    _log("STEP backup saved=%d already_saved=%d", saved, len(pg) + len(chroma) - saved)
    _log("STEP postgres updated=%d", _apply_postgres(conn, plan["pg_changes"]))
    _log("STEP chroma updated=%d", _apply_chroma(collection, plan["chroma_changes"]))
    report["tables"] = _apply_tables(registry)
    _log("STEP tables %s", report["tables"])
    source_videos = {vid: f.source_video for f in chroma.values()
                     if (vid := parse_source_video(f.source_video)[0])}
    report["files"] = _apply_files(_Files(), registry, source_videos)
    _log("STEP files %s", report["files"])

    report["checks"] = _checks(_read_postgres(conn), _read_chroma(collection), registry, before)


def _cleanup_files(files: _Files, *, apply: bool) -> dict[str, Any]:
    """Rename roster files stored under a cut-short id, and delete the
    pre-repair file copies. The files-applied marker is kept."""
    renames, conflicts = [], []
    for name in files.names("roster"):
        data = files.read("roster", name)
        record = json.loads(data)
        video_id = parse_source_video(record.get("source_video") or "")[0] or record.get("video_id")
        target = f"{video_id}.json"
        if not video_id or name == target:
            continue
        existing = files.read("roster", target)
        (conflicts if existing is not None and existing != data else renames).append((name, target, data))
    copies = files.backup_names("classified") + files.backup_names("roster")
    if files.read_backup("created_rosters.json") is not None:
        copies.append("created_rosters.json")
    if apply:
        for name, target, data in renames:
            files.write("roster", target, data)
            files.delete("roster", name)
        for name in copies:
            files.delete_backup(name)
    return {"rosters_renamed": len(renames), "roster_name_conflicts": [c[0] for c in conflicts],
            "backup_copies_deleted": len(copies)}


def _cleanup(conn, registry: SpeakerRegistry, report: dict[str, Any], *, apply: bool) -> None:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("select video_id, source_video from videos")
        videos = cur.fetchall()
        cur.execute("select count(*) as n from fragment_speaker_backup where repair_name = %s", (REPAIR_NAME,))
        backup_rows = cur.fetchone()["n"]
    conn.commit()
    ids = {v["video_id"] for v in videos}
    wrong = [(v["video_id"], parse_source_video(v["source_video"] or "")[0]) for v in videos
             if parse_source_video(v["source_video"] or "")[0] not in (None, v["video_id"])]
    stale_videos = [old for old, correct in wrong if correct in ids]
    report["cleanup"] = {
        "videos_rows_cut_short_id": len(stale_videos),
        "videos_rows_without_correct_row": [old for old, correct in wrong if correct not in ids],
        "backup_rows": backup_rows,
        **_cleanup_files(_Files(), apply=False),
    }
    _log("STEP cleanup plan %s", report["cleanup"])
    if not apply:
        return

    collection = _chroma_collection()
    before = {"postgres": len(_read_postgres(conn)), "chroma": collection.count()}
    with conn.cursor() as cur:
        cur.execute("delete from videos where video_id = any(%s)", (stale_videos,))
        videos_deleted = cur.rowcount
        cur.execute("delete from fragment_speaker_backup where repair_name = %s", (REPAIR_NAME,))
        backup_deleted = cur.rowcount
    conn.commit()
    _log("STEP cleanup postgres videos_deleted=%d backup_rows_deleted=%d", videos_deleted, backup_deleted)
    files = _cleanup_files(_Files(), apply=True)
    _log("STEP cleanup files %s", files)
    report["cleanup_done"] = {"videos_deleted": videos_deleted, "backup_rows_deleted": backup_deleted, **files}

    checks = _checks(_read_postgres(conn), _read_chroma(collection), registry, before)
    left = _cleanup_files(_Files(), apply=False)
    with conn.cursor() as cur:
        cur.execute("select count(*) from fragment_speaker_backup where repair_name = %s", (REPAIR_NAME,))
        rows_left = cur.fetchone()[0]
    conn.commit()
    for name, ok, detail in (
        ("backup_rows_removed", rows_left == 0, f"{rows_left} rows left"),
        ("backup_copies_removed", left["backup_copies_deleted"] == 0, f"{left['backup_copies_deleted']} copies left"),
        ("rosters_named_by_video_id", left["rosters_renamed"] == 0, f"{left['rosters_renamed']} still misnamed"),
    ):
        checks.append({"name": name, "ok": ok, "detail": detail})
        _log("CHECK %s %s %s", name, "PASS" if ok else "FAIL", detail, level=logging.INFO if ok else logging.ERROR)
    report["checks"] = checks


def start_in_background() -> None:
    """Called from API startup. Does nothing unless a switch above is on."""
    if RUN_SPEAKER_ROLLBACK:
        mode = "rollback"
    elif RUN_SPEAKER_CLEANUP:
        mode = "cleanup" if SPEAKER_CLEANUP_APPLY else "cleanup_preview"
    elif RUN_SPEAKER_REPAIR:
        mode = "apply" if SPEAKER_REPAIR_APPLY else "dry_run"
    else:
        return

    def _target() -> None:
        try:
            run(mode)
        except Exception:  # noqa: BLE001
            logger.exception("[speaker_repair] crashed")
            _log("RESULT status=failed mode=%s reason=crashed", mode, level=logging.ERROR)

    threading.Thread(target=_target, name="speaker-repair", daemon=True).start()


def _stores_are_local() -> bool:
    local = {"localhost", "127.0.0.1"}
    hosts = {urlparse(settings.database_url).hostname}
    if settings.chroma_mode == "http":
        hosts.add(settings.chroma_host)
    return hosts <= local and not s3_helpers.use_s3()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", dest="mode", action="store_const", const="dry_run")
    group.add_argument("--apply", dest="mode", action="store_const", const="apply")
    group.add_argument("--rollback", dest="mode", action="store_const", const="rollback")
    group.add_argument("--cleanup-preview", dest="mode", action="store_const", const="cleanup_preview")
    group.add_argument("--cleanup", dest="mode", action="store_const", const="cleanup")
    ap.add_argument("--target", choices=["local", "cluster"], default="local")
    ap.add_argument("--force", action="store_true", help="ignore the done marker and the cleanup wait")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.target != "cluster" and not _stores_are_local():
        print("Refusing: stores are not local. Pass --target=cluster to run against them.", file=sys.stderr)
        return 2
    report = run(args.mode, force=args.force)
    return 1 if report.get("status") == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
