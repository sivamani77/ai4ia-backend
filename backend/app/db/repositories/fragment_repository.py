from __future__ import annotations

import logging
from typing import Any, Mapping, TypedDict

from app.db.supabase import get_supabase_client
from app.db.utils import normalize_value

logger = logging.getLogger("ai4ia.db.repositories.fragments")


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
    table_name = "fragments"

    def __init__(self) -> None:
        self._client = get_supabase_client()
        self._table = self._client.table(self.table_name)

    def get(self, fragment_id: str) -> FragmentRecord | None:
        self._log("select_one", fragment_id)
        try:
            response = (
                self._table.select("*").eq("fragment_id", fragment_id).limit(1).execute()
            )
        except Exception:
            logger.exception("fragments select failed id=%s", fragment_id)
            raise
        return self._from_row(response.data[0]) if response.data else None

    def list_all(self, limit: int = 10000) -> list[FragmentRecord]:
        self._log("select_all")
        try:
            response = (
                self._table.select("*")
                .order("fragment_id")
                .limit(limit)
                .execute()
            )
        except Exception:
            logger.exception("fragments list_all failed")
            raise
        return [self._from_row(row) for row in response.data or []]

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
        try:
            q = self._table.select("*")
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
            q = q.limit(limit)
            response = q.execute()
        except Exception:
            logger.exception("fragments search failed")
            raise

        rows = [self._from_row(row) for row in response.data or []]

        # Apply free-text query filter in Python (full-text search via ilike on
        # multiple columns would need OR chaining not easily expressible here).
        if query:
            ql = query.lower()
            filtered = []
            for row in rows:
                haystack = " ".join(filter(None, [
                    row.get("text", ""),
                    row.get("summary", ""),
                    row.get("speaker", ""),
                    row.get("primary_theme", ""),
                    row.get("primary_chapter", ""),
                    row.get("video_title", ""),
                    " ".join(str(k) for k in (row.get("keywords") or [])),
                ])).lower()
                if ql in haystack:
                    filtered.append(row)
            rows = filtered

        return rows

    def count(self) -> int:
        self._log("count")
        try:
            response = self._table.select("fragment_id", count="exact").limit(1).execute()
            return response.count or 0
        except Exception:
            logger.exception("fragments count failed")
            return 0

    def speakers_aggregate(self) -> list[dict[str, Any]]:
        """Group-by speaker in the DB — avoids loading all rows into Python."""
        self._log("speakers_aggregate")
        try:
            response = (
                self._table
                .select("speaker, speaker_attribution, source_video, fragment_id")
                .not_.is_("speaker", "null")
                .execute()
            )
            rows = response.data or []
        except Exception:
            logger.exception("fragments speakers_aggregate failed")
            return []

        counts: dict[str, dict] = {}
        for r in rows:
            sp = r.get("speaker")
            if not sp:
                continue
            rec = counts.setdefault(sp, {
                "speaker": sp,
                "fragment_count": 0,
                "videos": set(),
                "attribution": r.get("speaker_attribution"),
            })
            rec["fragment_count"] += 1
            rec["videos"].add(r.get("source_video"))

        out = []
        for rec in counts.values():
            rec["video_count"] = len(rec.pop("videos"))
            out.append(rec)
        return sorted(out, key=lambda r: r["fragment_count"], reverse=True)

    def chapter_counts(self) -> dict[str, int]:
        """Return {chapter_name: count} aggregated in the DB."""
        self._log("chapter_counts")
        try:
            response = (
                self._table
                .select("primary_chapter, fragment_id")
                .not_.is_("primary_chapter", "null")
                .execute()
            )
            rows = response.data or []
        except Exception:
            logger.exception("fragments chapter_counts failed")
            return {}

        counts: dict[str, int] = {}
        for r in rows:
            ch = r.get("primary_chapter")
            if ch:
                counts[ch] = counts.get(ch, 0) + 1
        return counts

    def upsert_many(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        self._log("upsert_many")
        try:
            self._table.upsert(rows).execute()
            return len(rows)
        except Exception:
            logger.exception("fragments upsert_many failed count=%d", len(rows))
            raise

    def _from_row(self, row: Mapping[str, Any]) -> FragmentRecord:
        d = self._normalize(dict(row))
        # Reconstruct stream_url dynamically — not stored in DB
        source_video = d.get("source_video", "")
        return FragmentRecord(
            fragment_id=d.get("fragment_id", ""),
            source_video=source_video,
            video_id=d.get("video_id"),
            video_title=d.get("video_title"),
            speaker=d.get("speaker"),
            speaker_attribution=d.get("speaker_attribution"),
            text=d.get("text", ""),
            summary=d.get("summary"),
            start=float(d.get("start") or 0),
            end=float(d.get("end") or 0),
            duration=float(d.get("duration") or 0),
            primary_theme=d.get("primary_theme"),
            primary_chapter=d.get("primary_chapter"),
            theme_tags=d.get("theme_tags") or [],
            chapter_tags=d.get("chapter_tags") or [],
            clip_usage_candidates=d.get("clip_usage_candidates") or [],
            quote_strength=float(d.get("quote_strength") or 0),
            clarity_score=float(d.get("clarity_score") or 0),
            year=d.get("year"),
            edition=d.get("edition"),
            session_type=d.get("session_type"),
            keywords=d.get("keywords") or [],
            extra=d.get("extra") or {},
            # Virtual field — callers that need stream_url get it from here
            stream_url=f"/api/archive/videos/{source_video}/stream",  # type: ignore[typeddict-unknown-key]
        )

    def _log(self, operation: str, record_id: str | None = None) -> None:
        logger.debug("fragments op=%s id=%s", operation, record_id)

    def _normalize(self, value: Any) -> Any:
        return normalize_value(value)
