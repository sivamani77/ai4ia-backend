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
    in Chroma for search. Reads and writes both go to Postgres; the legacy
    Supabase read fallback was removed once the pipeline project was fully
    backfilled — see backend/app/db/schema.py for the table and
    scripts/migrate_fragments_to_postgres.py for that one-time backfill."""

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
        return self._from_row(row) if row is not None else None

    def get_many(self, fragment_ids: list[str]) -> list[FragmentRecord]:
        """Batched lookup — one round trip instead of one `get()` call per id."""
        if not fragment_ids:
            return []
        self._log("select_many")
        rows = pg_fetch_all(
            f"select {_COLUMNS} from fragments where fragment_id = any(%s)",
            (list(fragment_ids),),
        )
        return [self._from_row(row) for row in rows]

    def list_all(self, limit: int = 10000) -> list[FragmentRecord]:
        self._log("select_all")
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
        row = pg_fetch_one("select count(*) as n from fragments")
        return int(row["n"]) if row else 0

    def speakers_aggregate(self) -> list[dict[str, Any]]:
        """Group-by speaker in the DB — avoids loading all rows into Python."""
        self._log("speakers_aggregate")
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
        rows = pg_fetch_all(
            """
            select primary_chapter, count(*) as n
            from fragments
            where primary_chapter is not null
            group by primary_chapter
            """
        )
        return {row["primary_chapter"]: int(row["n"]) for row in rows}

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
