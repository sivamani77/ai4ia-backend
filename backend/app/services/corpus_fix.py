"""One-off corpus fix in Postgres (case matrix, 2026-09): multi-speaker videos
filed as solo talks, conference years, duplicate uploads.

Stores touched: fragments, videos, speakers and video_speakers (only the videos below). Chroma is replaced
separately from the cleaned local collection.

Runs from API startup when the switch below is on, or from the CLI:

    python -m app.services.corpus_fix --dry-run
    python -m app.services.corpus_fix --apply
    python -m app.services.corpus_fix --rollback

The CLI refuses a non-local database unless --target=cluster is passed.
Every run logs lines prefixed "[corpus_fix]" and ends with one RESULT line.
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
from collections import Counter
from typing import Any
from urllib.parse import urlparse

import psycopg2
import psycopg2.extras

from app.core.settings import settings
from app.services.conference_year import YEARS_PATH, known_years
from app.services.speaker_registry import SpeakerRegistry, get_registry

logger = logging.getLogger("ai4ia.corpus_fix")

# --- Deploy switches ----------------------------------------------------------
# Deploy 1: dry run only; check the PLAN and RESULT lines in the API log.
# Deploy 2: CORPUS_FIX_APPLY = True. Deploy 3 (done): RUN_CORPUS_FIX = False.
#
# Apply confirmed on uat 2026-09-28 (shared with release): the fix ran once,
# guarded by the advisory lock and the corpus_fix_2026_09 row in data_repairs,
# so the switch is off now. `python -m app.services.corpus_fix --apply` still
# runs on demand from the CLI if it's ever needed again.
FIX_NAME = "corpus_fix_2026_09"  # done-marker key; bump to run again
RUN_CORPUS_FIX = False
CORPUS_FIX_APPLY = True          # False = dry run and report only; moot while RUN_CORPUS_FIX is False

_LOCK_KEY = 73_310_002
# Multi-speaker sessions filed as one speaker's talk; rosters are in the registry.
PANEL_VIDEOS = ("srXR5cfLK4I", "B-B0ilBRyLo", "QP_5ovm6wn8", "llAJjuaCCVE", "wSKcjA2uFfw", "Fp4NLQAOLms")
# Removed upload -> the upload kept (same talk, identical fragments).
DUPLICATES = {"ea96KAytzSg": "aFtkmF49jCQ", "iuNzi9HZGWk": "8oNB2wBAP7M"}


def _log(msg: str, *args: Any, level: int = logging.INFO) -> None:
    logger.log(level, "[corpus_fix] " + msg, *args)


def load_years() -> dict[str, int]:
    years = known_years()
    if not years:
        raise RuntimeError(f"no conference years in {YEARS_PATH}")
    return years


def _current(row: dict[str, Any]) -> dict[str, Any]:
    return {"speaker": row.get("speaker") or "", "speaker_attribution": row.get("speaker_attribution") or "",
            "speaker_roster": (row.get("extra") or {}).get("speaker_roster") or "",
            "year": row.get("year"), "edition": row.get("edition")}


def plan(rows: list[dict[str, Any]], registry: SpeakerRegistry, years: dict[str, int]) -> dict[str, Any]:
    """Changes for the given fragment rows: {fragment_id: new values} and ids to delete."""
    updates: dict[str, dict[str, Any]] = {}
    deletes: list[str] = []
    counts: Counter = Counter()
    for row in rows:
        video_id = row.get("video_id") or ""
        if video_id in DUPLICATES:
            deletes.append(row["fragment_id"])
            continue
        old = _current(row)
        new = dict(old)
        if video_id in PANEL_VIDEOS:
            fields = registry.speaker_fields(row.get("source_video"), row.get("speaker"),
                                             row.get("speaker_attribution"), video_id)
            new.update({k: fields[k] for k in ("speaker", "speaker_attribution", "speaker_roster")})
        if video_id in years:
            new["year"], new["edition"] = years[video_id], f"AI4IA {years[video_id]}"
        changed = sorted(k for k in new if new[k] != old[k])
        if changed:
            updates[row["fragment_id"]] = new
            counts["changed " + ",".join(changed)] += 1
    return {"updates": updates, "deletes": deletes, "counts": dict(counts)}


def _read_marker(conn) -> dict[str, Any] | None:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("select * from data_repairs where name = %s", (FIX_NAME,))
        row = cur.fetchone()
    conn.commit()
    return dict(row) if row else None


def _write_marker(conn, status: str, mode: str, summary: dict[str, Any]) -> None:
    conn.rollback()
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into data_repairs (name, status, mode, summary, updated_at)
            values (%s, %s, %s, %s, now())
            on conflict (name) do update set status = excluded.status, mode = excluded.mode,
                summary = excluded.summary, updated_at = now()
            """,
            (FIX_NAME, status, mode, psycopg2.extras.Json(summary)),
        )
    conn.commit()


def _apply(conn, changes: dict[str, Any], registry: SpeakerRegistry) -> dict[str, int]:
    """One transaction: back up, then change fragments, videos and speaker tables.
    Nothing is kept if any step fails."""
    updates, deletes = changes["updates"], changes["deletes"]
    dup_videos = list(DUPLICATES)
    roster_videos = list(PANEL_VIDEOS) + dup_videos
    new_rosters = [(vid, name, role, i) for vid in PANEL_VIDEOS
                   for i, (name, role) in enumerate((registry.video(vid).speakers if registry.video(vid) else ()))]
    new_people = sorted({name for _, name, _, _ in new_rosters})
    with conn.cursor() as cur:
        # Original rows first, so a rollback can restore them exactly.
        cur.execute(
            """
            insert into corpus_fix_backup (fix_name, table_name, row_id, action, before)
            select %s, 'fragments', fragment_id,
                   case when fragment_id = any(%s) then 'delete' else 'update' end, to_jsonb(f)
            from fragments f where fragment_id = any(%s)
            on conflict do nothing
            """,
            (FIX_NAME, deletes, list(updates) + deletes),
        )
        cur.execute(
            """
            insert into corpus_fix_backup (fix_name, table_name, row_id, action, before)
            select %s, 'videos', video_id, 'delete', to_jsonb(v) from videos v where video_id = any(%s)
            on conflict do nothing
            """,
            (FIX_NAME, dup_videos),
        )
        cur.execute(
            """
            insert into corpus_fix_backup (fix_name, table_name, row_id, action, before)
            select %s, 'video_speakers', video_id || '|' || canonical_name, 'replace', to_jsonb(vs)
            from video_speakers vs where video_id = any(%s)
            on conflict do nothing
            """,
            (FIX_NAME, roster_videos),
        )
        rows = [(fid, n["speaker"], n["speaker_attribution"], n["speaker_roster"], n["year"], n["edition"])
                for fid, n in updates.items()]
        psycopg2.extras.execute_values(
            cur,
            """
            update fragments as f set
                speaker = nullif(v.speaker, ''),
                speaker_attribution = nullif(v.attribution, ''),
                extra = coalesce(f.extra, '{}'::jsonb) || jsonb_build_object('speaker_roster', v.roster),
                year = v.year::int,
                edition = v.edition
            from (values %s) as v(fragment_id, speaker, attribution, roster, year, edition)
            where f.fragment_id = v.fragment_id
            """,
            rows, page_size=500,
        )
        cur.execute("delete from fragments where fragment_id = any(%s)", (deletes,))
        deleted = cur.rowcount
        # Speakers named on the new rosters but missing from the table; recorded so a rollback removes them.
        cur.execute(
            """
            with added as (
                insert into speakers (canonical_name, status, source)
                select name, 'review', 'decision' from unnest(%s::text[]) as name
                on conflict (canonical_name) do nothing
                returning canonical_name
            )
            insert into corpus_fix_backup (fix_name, table_name, row_id, action, before)
            select %s, 'speakers', canonical_name, 'insert', '{}'::jsonb from added
            on conflict do nothing
            """,
            (new_people, FIX_NAME),
        )
        cur.execute("select count(*) from corpus_fix_backup where fix_name = %s and table_name = 'speakers'",
                    (FIX_NAME,))
        speakers_added = cur.fetchone()[0]
        # Only the six panels and the two duplicates: every other roster is left as it is.
        cur.execute("delete from video_speakers where video_id = any(%s)", (roster_videos,))
        psycopg2.extras.execute_values(
            cur, "insert into video_speakers (video_id, canonical_name, role, ordinal) values %s", new_rosters)
        cur.execute("delete from videos where video_id = any(%s)", (dup_videos,))
        videos_deleted = cur.rowcount
    conn.commit()
    return {"fragments_updated": len(rows), "fragments_deleted": deleted, "videos_deleted": videos_deleted,
            "speakers_added": speakers_added, "video_speakers_written": len(new_rosters)}


def _rollback(conn) -> dict[str, int]:
    """One transaction: restore every row this fix changed, deleted or added."""
    roster_videos = list(PANEL_VIDEOS) + list(DUPLICATES)
    with conn.cursor() as cur:
        cur.execute(
            """
            update fragments f set speaker = r.speaker, speaker_attribution = r.speaker_attribution,
                   extra = r.extra, year = r.year, edition = r.edition
            from corpus_fix_backup b cross join lateral jsonb_populate_record(null::fragments, b.before) r
            where b.fix_name = %s and b.table_name = 'fragments' and b.action = 'update'
              and f.fragment_id = b.row_id
            """,
            (FIX_NAME,),
        )
        restored = cur.rowcount
        cur.execute(
            """
            insert into fragments select (jsonb_populate_record(null::fragments, before)).*
            from corpus_fix_backup where fix_name = %s and table_name = 'fragments' and action = 'delete'
            on conflict (fragment_id) do nothing
            """,
            (FIX_NAME,),
        )
        reinserted = cur.rowcount
        cur.execute(
            """
            insert into videos select (jsonb_populate_record(null::videos, before)).*
            from corpus_fix_backup where fix_name = %s and table_name = 'videos'
            on conflict (video_id) do nothing
            """,
            (FIX_NAME,),
        )
        videos = cur.rowcount
        cur.execute("delete from video_speakers where video_id = any(%s)", (roster_videos,))
        cur.execute(
            """
            insert into video_speakers select (jsonb_populate_record(null::video_speakers, before)).*
            from corpus_fix_backup where fix_name = %s and table_name = 'video_speakers'
            on conflict do nothing
            """,
            (FIX_NAME,),
        )
        rosters = cur.rowcount
        cur.execute(
            """
            delete from speakers s using corpus_fix_backup b
            where b.fix_name = %s and b.table_name = 'speakers' and s.canonical_name = b.row_id
              and not exists (select 1 from video_speakers v where v.canonical_name = s.canonical_name)
              and not exists (select 1 from speaker_aliases a where a.canonical_name = s.canonical_name)
            """,
            (FIX_NAME,),
        )
        speakers = cur.rowcount
        # Rows restored; clear this fix's backup so a later apply starts clean.
        cur.execute("delete from corpus_fix_backup where fix_name = %s", (FIX_NAME,))
    conn.commit()
    return {"fragments_restored": restored, "fragments_reinserted": reinserted, "videos_reinserted": videos,
            "video_speakers_restored": rosters, "speakers_removed": speakers}


def run(mode: str, *, force: bool = False) -> dict[str, Any]:
    """Run one mode: "dry_run", "apply" or "rollback". Returns the report."""
    if mode not in ("dry_run", "apply", "rollback"):
        raise ValueError(f"unknown mode {mode!r}")
    conn = psycopg2.connect(settings.database_url)
    try:
        with conn.cursor() as cur:
            cur.execute("select pg_try_advisory_lock(%s)", (_LOCK_KEY,))
            locked = cur.fetchone()[0]
        conn.commit()
        if not locked:
            _log("RESULT status=skipped mode=%s reason=another_instance_running", mode)
            return {"status": "skipped", "reason": "another_instance_running"}
        status = (_read_marker(conn) or {}).get("status")
        if not force and mode != "rollback" and status == "applied":
            _log("RESULT status=skipped mode=%s reason=already_done", mode)
            return {"status": "skipped", "reason": "already_done"}
        if not force and mode == "rollback" and status != "applied":
            _log("RESULT status=skipped mode=rollback reason=nothing_to_roll_back")
            return {"status": "skipped", "reason": "nothing_to_roll_back"}

        if mode == "rollback":
            summary = _rollback(conn)
            _write_marker(conn, "rolled_back", mode, summary)
            _log("RESULT status=rolled_back %s", summary)
            return {"status": "rolled_back", **summary}

        registry, years = get_registry(), load_years()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("select fragment_id, video_id, source_video, speaker, speaker_attribution, extra, year, "
                        "edition from fragments")
            rows = [dict(r) for r in cur.fetchall()]
        conn.commit()
        changes = plan(rows, registry, years)
        missing = sorted({r["video_id"] for r in rows if r.get("video_id") and r["video_id"] not in years
                          and r["video_id"] not in DUPLICATES})
        summary = {"fragments": len(rows), "updates": len(changes["updates"]), "deletes": len(changes["deletes"]),
                   "counts": changes["counts"], "videos_without_year": missing[:20]}
        for key, count in changes["counts"].items():
            _log("PLAN %s: %d", key, count)
        if missing:
            _log("WARN %d videos have no conference year: %s", len(missing), missing[:20], level=logging.WARNING)
        if mode == "dry_run":
            _write_marker(conn, "dry_run", mode, summary)
            _log("RESULT status=dry_run fragments=%d updates=%d deletes=%d", len(rows),
                 summary["updates"], summary["deletes"])
            return {"status": "dry_run", **summary}

        summary.update(_apply(conn, changes, registry))
        _write_marker(conn, "applied", mode, summary)
        _log("RESULT status=applied %s", {k: v for k, v in summary.items() if k != "counts"})
        return {"status": "applied", **summary}
    except Exception as exc:
        _log("RESULT status=failed mode=%s error=%s", mode, exc, level=logging.ERROR)
        try:
            _write_marker(conn, "failed", mode, {"error": str(exc)[:500]})
        except Exception:  # noqa: BLE001
            pass
        raise
    finally:
        conn.close()  # also releases the advisory lock


def start_in_background() -> None:
    """Called from API startup. Does nothing unless the switch above is on."""
    if not RUN_CORPUS_FIX:
        return
    mode = "apply" if CORPUS_FIX_APPLY else "dry_run"

    def _target() -> None:
        try:
            run(mode)
        except Exception:  # noqa: BLE001
            logger.exception("[corpus_fix] crashed")

    threading.Thread(target=_target, name="corpus-fix", daemon=True).start()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", dest="mode", action="store_const", const="dry_run")
    group.add_argument("--apply", dest="mode", action="store_const", const="apply")
    group.add_argument("--rollback", dest="mode", action="store_const", const="rollback")
    ap.add_argument("--target", choices=["local", "cluster"], default="local")
    ap.add_argument("--force", action="store_true", help="ignore the done marker")
    args = ap.parse_args(argv)
    if args.target == "local" and urlparse(settings.database_url).hostname not in {"localhost", "127.0.0.1"}:
        print("refusing: DATABASE_URL is not local; pass --target=cluster to run against it", file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = run(args.mode, force=args.force)
    return 0 if report.get("status") in ("applied", "dry_run", "rolled_back", "skipped") else 1


if __name__ == "__main__":
    sys.exit(main())
