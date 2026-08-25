from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Mapping, TypedDict

import psycopg2.extras

from app.core.settings import settings
from app.db.postgres import pg_execute, pg_fetch_all, pg_fetch_one
from app.db.utils import normalize_value

logger = logging.getLogger("ai4ia.db.repositories.fragments")

_JSONB_FIELDS = ("theme_tags", "chapter_tags", "clip_usage_candidates", "keywords", "extra")

_COLUMNS = (
    "fragment_id, source_video, video_id, video_title, speaker, speaker_attribution, "
    'text, summary, start, "end", duration, primary_theme, primary_chapter, '
    "theme_tags, chapter_tags, clip_usage_candidates, quote_strength, clarity_score, "
    "year, edition, session_type, keywords, extra"
)


class FragmentRecord(TypedDict, total=False):
    fragment_id: str
    source_video: str | None
    video_id: str | None
    video_title: str | None
    speaker: str | None
    speaker_attribution: str | None
    text: str
    summary: str | None
    start: float
    end: float
    duration: float
    primary_theme: str | None
    primary_chapter: str | None
    theme_tags: list
    chapter_tags: list
    clip_usage_candidates: list
    quote_strength: float
    clarity_score: float
    year: int | None
    edition: str | None
    session_type: str | None
    keywords: list
    extra: dict


class FragmentRepository:
    """Fragments live in Postgres (SQL-queryable) with vectors/metadata mirrored
    in Chroma for search. Reads fall back to the legacy Supabase pipeline
    project for rows not yet backfilled into Postgres; writes go to Postgres
    only — see backend/app/db/schema.py for the table and
    scripts/migrate_fragments_to_postgres.py for the one-time backfill."""

    table_name = "fragments"

    # -- writes ---------------------------------------------------------

    def upsert_many(self, rows: list[Mapping[str, Any]]) -> int:
        if not rows:
            return 0
        self._log("upsert_many")
        payloads = [self._to_row_params(self._to_row(row)) for row in rows]
        for payload in payloads:
            pg_execute(
                f"""
                insert into fragments ({_COLUMNS})
                values (
                    %(fragment_id)s, %(source_video)s, %(video_id)s, %(video_title)s,
                    %(speaker)s, %(speaker_attribution)s, %(text)s, %(summary)s,
                    %(start)s, %(end)s, %(duration)s, %(primary_theme)s, %(primary_chapter)s,
                    %(theme_tags)s, %(chapter_tags)s, %(clip_usage_candidates)s,
                    %(quote_strength)s, %(clarity_score)s, %(year)s, %(edition)s,
                    %(session_type)s, %(keywords)s, %(extra)s
                )
                on conflict (fragment_id) do update set
                    source_video = excluded.source_video,
                    video_id = excluded.video_id,
                    video_title = excluded.video_title,
                    speaker = excluded.speaker,
                    speaker_attribution = excluded.speaker_attribution,
                    text = excluded.text,
                    summary = excluded.summary,
                    start = excluded.start,
                    "end" = excluded."end",
                    duration = excluded.duration,
                    primary_theme = excluded.primary_theme,
                    primary_chapter = excluded.primary_chapter,
                    theme_tags = excluded.theme_tags,
                    chapter_tags = excluded.chapter_tags,
                    clip_usage_candidates = excluded.clip_usage_candidates,
                    quote_strength = excluded.quote_strength,
                    clarity_score = excluded.clarity_score,
                    year = excluded.year,
                    edition = excluded.edition,
                    session_type = excluded.session_type,
                    keywords = excluded.keywords,
                    extra = excluded.extra
                """,
                payload,
            )
        return len(payloads)

    # -- reads ------------------------------------------------------------

    def get(self, fragment_id: str) -> FragmentRecord | None:
        self._log("select_one", fragment_id)
        row = pg_fetch_one(f"select {_COLUMNS} from fragments where fragment_id = %s", (fragment_id,))
        if row is not None:
            return self._from_row(row)
        return self._legacy_get(fragment_id)

    def get_many(self, fragment_ids: list[str]) -> list[FragmentRecord]:
        """Batched lookup — one round trip instead of one `get()` call per id."""
        if not fragment_ids:
            return []
        self._log("select_many")
        rows = pg_fetch_all(
            f"select {_COLUMNS} from fragments where fragment_id = any(%s)",
            (list(fragment_ids),),
        )
        found = [self._from_row(row) for row in rows]
        missing = set(fragment_ids) - {r["fragment_id"] for r in found}
        if missing:
            found.extend(self._legacy_get_many(sorted(missing)))
        return found

    def list_all(self, limit: int = 10000) -> list[FragmentRecord]:
        self._log("select_all")
        if not self._pg_has_data():
            return self._legacy_list_all(limit)
        rows = pg_fetch_all(f"select {_COLUMNS} from fragments order by fragment_id limit %s", (limit,))
        return [self._from_row(row) for row in rows]

    def search(
        self,
        *,
        query: str | None = None,
        theme: str | None = None,
        chapter: str | None = None,
        speaker: str | None = None,
        source_video: str | None = None,
        year: int | None = None,
        edition: str | None = None,
        clip_usage: str | None = None,
        min_duration: float | None = None,
        max_duration: float | None = None,
        start_after: float | None = None,
        end_before: float | None = None,
        limit: int = 200,
    ) -> list[FragmentRecord]:
        self._log("search")
        if not self._pg_has_data():
            return self._legacy_search(
                query=query, theme=theme, chapter=chapter, speaker=speaker,
                source_video=source_video, year=year, edition=edition,
                clip_usage=clip_usage, min_duration=min_duration, max_duration=max_duration,
                start_after=start_after, end_before=end_before, limit=limit,
            )

        clauses: list[str] = []
        params: dict[str, Any] = {"limit": limit}
        if theme:
            clauses.append("primary_theme ilike %(theme)s")
            params["theme"] = f"%{theme}%"
        if chapter:
            clauses.append("primary_chapter ilike %(chapter)s")
            params["chapter"] = f"%{chapter}%"
        if speaker:
            clauses.append("speaker ilike %(speaker)s")
            params["speaker"] = f"%{speaker}%"
        if source_video:
            clauses.append("source_video ilike %(source_video)s")
            params["source_video"] = f"%{source_video}%"
        if year:
            clauses.append("year = %(year)s")
            params["year"] = year
        if edition:
            clauses.append("edition ilike %(edition)s")
            params["edition"] = f"%{edition}%"
        if min_duration is not None:
            clauses.append("duration >= %(min_duration)s")
            params["min_duration"] = min_duration
        if max_duration is not None:
            clauses.append("duration <= %(max_duration)s")
            params["max_duration"] = max_duration
        if start_after is not None:
            clauses.append("start >= %(start_after)s")
            params["start_after"] = start_after
        if end_before is not None:
            clauses.append('"end" <= %(end_before)s')
            params["end_before"] = end_before

        where = f"where {' and '.join(clauses)}" if clauses else ""
        rows = pg_fetch_all(
            f"select {_COLUMNS} from fragments {where} limit %(limit)s",
            params,
        )
        results = [self._from_row(row) for row in rows]

        if query:
            ql = query.lower()
            results = [r for r in results if ql in self._search_haystack(r)]

        return results

    def list_fragment_ids_by_source_video(self, source_video: str) -> list[str]:
        self._log("select_ids_by_source_video", source_video)
        rows = pg_fetch_all(
            "select fragment_id from fragments where source_video = %s", (source_video,)
        )
        return [row["fragment_id"] for row in rows]

    # -- deletes ----------------------------------------------------------

    def delete_many(self, fragment_ids: list[str]) -> list[str]:
        """Delete by explicit id list (not by source_video again) so the
        caller can delete exactly the ids it already removed from Chroma —
        never a superset picked up by a second, later-running query."""
        if not fragment_ids:
            return []
        self._log("delete_many", f"{len(fragment_ids)} ids")
        rows = pg_fetch_all(
            "delete from fragments where fragment_id = any(%s) returning fragment_id",
            (list(fragment_ids),),
        )
        return [row["fragment_id"] for row in rows]

    def count(self) -> int:
        self._log("count")
        if not self._pg_has_data():
            return self._legacy_count()
        row = pg_fetch_one("select count(*) as n from fragments")
        return int(row["n"]) if row else 0

    def speakers_aggregate(self) -> list[dict[str, Any]]:
        """Group-by speaker in the DB — avoids loading all rows into Python."""
        self._log("speakers_aggregate")
        if not self._pg_has_data():
            return self._legacy_speakers_aggregate()
        rows = pg_fetch_all(
            """
            select speaker,
                   count(*) as fragment_count,
                   count(distinct source_video) as video_count,
                   max(speaker_attribution) as attribution
            from fragments
            where speaker is not null
            group by speaker
            order by fragment_count desc
            """
        )
        return [
            {
                "speaker": row["speaker"],
                "fragment_count": int(row["fragment_count"]),
                "video_count": int(row["video_count"]),
                "attribution": row["attribution"],
            }
            for row in rows
        ]

    def chapter_counts(self) -> dict[str, int]:
        """Return {chapter_name: count} aggregated in the DB."""
        self._log("chapter_counts")
        if not self._pg_has_data():
            return self._legacy_chapter_counts()
        rows = pg_fetch_all(
            """
            select primary_chapter, count(*) as n
            from fragments
            where primary_chapter is not null
            group by primary_chapter
            """
        )
        return {row["primary_chapter"]: int(row["n"]) for row in rows}

    # -- legacy Supabase fallback (read-only) ------------------------------

    def _legacy_table(self):
        """Returns the old Supabase pipeline `fragments` table, or None if no
        legacy project is configured (e.g. it's been fully decommissioned)."""
        if not settings.supabase_url or not settings.supabase_service_key:
            return None
        try:
            from app.db.supabase import get_supabase_client
            return get_supabase_client().table(self.table_name)
        except Exception:
            logger.exception("Legacy Supabase fragments client unavailable")
            return None

    def _pg_has_data(self) -> bool:
        row = pg_fetch_one("select exists(select 1 from fragments limit 1) as has_data")
        return bool(row and row.get("has_data"))

    def _legacy_get(self, fragment_id: str) -> FragmentRecord | None:
        table = self._legacy_table()
        if table is None:
            return None
        try:
            response = table.select("*").eq("fragment_id", fragment_id).limit(1).execute()
        except Exception:
            logger.exception("Legacy fragments select failed id=%s", fragment_id)
            return None
        return self._from_row(response.data[0]) if response.data else None

    def _legacy_get_many(self, fragment_ids: list[str]) -> list[FragmentRecord]:
        table = self._legacy_table()
        if table is None or not fragment_ids:
            return []
        try:
            response = table.select("*").in_("fragment_id", fragment_ids).execute()
        except Exception:
            logger.exception("Legacy fragments select_many failed count=%d", len(fragment_ids))
            return []
        return [self._from_row(row) for row in response.data or []]

    def _legacy_list_all(self, limit: int) -> list[FragmentRecord]:
        table = self._legacy_table()
        if table is None:
            return []
        PAGE = 1000
        all_rows: list[dict] = []
        offset = 0
        try:
            while True:
                response = (
                    table.select("*").order("fragment_id").range(offset, offset + PAGE - 1).execute()
                )
                page_data = response.data or []
                all_rows.extend(page_data)
                if len(page_data) < PAGE or len(all_rows) >= limit:
                    break
                offset += PAGE
        except Exception:
            logger.exception("Legacy fragments list_all failed at offset=%d", offset)
        return [self._from_row(row) for row in all_rows[:limit]]

    def _legacy_search(self, *, query, theme, chapter, speaker, source_video, year,
                        edition, clip_usage, min_duration, max_duration,
                        start_after, end_before, limit) -> list[FragmentRecord]:
        table = self._legacy_table()
        if table is None:
            return []
        try:
            q = table.select("*")
            if theme:
                q = q.ilike("primary_theme", f"%{theme}%")
            if chapter:
                q = q.ilike("primary_chapter", f"%{chapter}%")
            if speaker:
                q = q.ilike("speaker", f"%{speaker}%")
            if source_video:
                q = q.ilike("source_video", f"%{source_video}%")
            if year:
                q = q.eq("year", year)
            if edition:
                q = q.ilike("edition", f"%{edition}%")
            if min_duration is not None:
                q = q.gte("duration", min_duration)
            if max_duration is not None:
                q = q.lte("duration", max_duration)
            if start_after is not None:
                q = q.gte("start", start_after)
            if end_before is not None:
                q = q.lte("end", end_before)
            response = q.limit(limit).execute()
        except Exception:
            logger.exception("Legacy fragments search failed")
            return []

        rows = [self._from_row(row) for row in response.data or []]
        if query:
            ql = query.lower()
            rows = [r for r in rows if ql in self._search_haystack(r)]
        return rows

    def _legacy_count(self) -> int:
        table = self._legacy_table()
        if table is None:
            return 0
        try:
            response = table.select("fragment_id", count="exact").limit(1).execute()
            return response.count or 0
        except Exception:
            logger.exception("Legacy fragments count failed")
            return 0

    def _legacy_speakers_aggregate(self) -> list[dict[str, Any]]:
        table = self._legacy_table()
        if table is None:
            return []
        try:
            response = (
                table.select("speaker, speaker_attribution, source_video, fragment_id")
                .not_.is_("speaker", "null")
                .execute()
            )
            rows = response.data or []
        except Exception:
            logger.exception("Legacy fragments speakers_aggregate failed")
            return []

        counts: dict[str, dict] = {}
        for r in rows:
            sp = r.get("speaker")
            if not sp:
                continue
            rec = counts.setdefault(sp, {
                "speaker": sp, "fragment_count": 0, "videos": set(),
                "attribution": r.get("speaker_attribution"),
            })
            rec["fragment_count"] += 1
            rec["videos"].add(r.get("source_video"))

        out = []
        for rec in counts.values():
            rec["video_count"] = len(rec.pop("videos"))
            out.append(rec)
        return sorted(out, key=lambda r: r["fragment_count"], reverse=True)

    def _legacy_chapter_counts(self) -> dict[str, int]:
        table = self._legacy_table()
        if table is None:
            return {}
        try:
            response = table.select("primary_chapter, fragment_id").not_.is_("primary_chapter", "null").execute()
            rows = response.data or []
        except Exception:
            logger.exception("Legacy fragments chapter_counts failed")
            return {}
        counts: Counter = Counter()
        for r in rows:
            ch = r.get("primary_chapter")
            if ch:
                counts[ch] += 1
        return dict(counts)

    # -- row shaping --------------------------------------------------------

    def _search_haystack(self, row: FragmentRecord) -> str:
        return " ".join(filter(None, [
            row.get("text", ""),
            row.get("summary", ""),
            row.get("speaker", ""),
            row.get("primary_theme", ""),
            row.get("primary_chapter", ""),
            row.get("video_title", ""),
            " ".join(str(k) for k in (row.get("keywords") or [])),
        ])).lower()

    def _to_row(self, data: Mapping[str, Any]) -> dict[str, Any]:
        d = self._normalize(dict(data))
        return {
            "fragment_id": d.get("fragment_id") or d.get("id"),
            "source_video": d.get("source_video"),
            "video_id": d.get("video_id"),
            "video_title": d.get("video_title"),
            "speaker": d.get("speaker"),
            "speaker_attribution": d.get("speaker_attribution"),
            "text": d.get("text") or "",
            "summary": d.get("summary"),
            "start": float(d.get("start") or 0),
            "end": float(d.get("end") or 0),
            "duration": float(d.get("duration") or 0),
            "primary_theme": d.get("primary_theme"),
            "primary_chapter": d.get("primary_chapter"),
            "theme_tags": d.get("theme_tags") or [],
            "chapter_tags": d.get("chapter_tags") or [],
            "clip_usage_candidates": d.get("clip_usage_candidates") or [],
            "quote_strength": float(d.get("quote_strength") or 0),
            "clarity_score": float(d.get("clarity_score") or 0),
            "year": d.get("year"),
            "edition": d.get("edition"),
            "session_type": d.get("session_type"),
            "keywords": d.get("keywords") or [],
            "extra": d.get("extra") or {},
        }

    def _from_row(self, row: Mapping[str, Any]) -> FragmentRecord:
        d = self._normalize(dict(row))
        return FragmentRecord(
            fragment_id=d.get("fragment_id", ""),
            source_video=d.get("source_video"),
            video_id=d.get("video_id"),
            video_title=d.get("video_title"),
            speaker=d.get("speaker"),
            speaker_attribution=d.get("speaker_attribution"),
            text=d.get("text") or "",
            summary=d.get("summary"),
            start=d.get("start"),
            end=d.get("end"),
            duration=d.get("duration"),
            primary_theme=d.get("primary_theme"),
            primary_chapter=d.get("primary_chapter"),
            theme_tags=d.get("theme_tags") or [],
            chapter_tags=d.get("chapter_tags") or [],
            clip_usage_candidates=d.get("clip_usage_candidates") or [],
            quote_strength=d.get("quote_strength"),
            clarity_score=d.get("clarity_score"),
            year=d.get("year"),
            edition=d.get("edition"),
            session_type=d.get("session_type"),
            keywords=d.get("keywords") or [],
            extra=d.get("extra") or {},
        )

    def _to_row_params(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        params = dict(payload)
        for field in _JSONB_FIELDS:
            if field in params:
                params[field] = psycopg2.extras.Json(params[field])
        return params

    def _log(self, operation: str, record_id: str | None = None) -> None:
        logger.debug("Postgres repository operation: table=%s operation=%s id=%s", self.table_name, operation, record_id)

    def _normalize(self, value: Any) -> Any:
        return normalize_value(value)
