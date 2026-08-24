from __future__ import annotations

import logging
from typing import Any, Mapping, TypedDict

from app.db.postgres import pg_execute, pg_fetch_all, pg_fetch_one
from app.db.utils import normalize_value

logger = logging.getLogger("ai4ia.db.repositories.videos")


class VideoRecord(TypedDict, total=False):
    video_id: str
    source_video: str | None
    title: str | None
    duration: float | None
    webpage_url: str | None
    s3_raw_key: str | None
    updated_at: str | None


class VideoRepository:
    """Canonical video registry — one row per video, populated by syncing
    the S3 raw-video prefix (see archive_service.list_videos()). Brand new
    table with no legacy data, so unlike FragmentRepository there's no
    Supabase fallback here."""

    table_name = "videos"

    def upsert_many(self, rows: list[Mapping[str, Any]]) -> int:
        if not rows:
            return 0
        self._log("upsert_many")
        for row in rows:
            payload = self._to_row(row)
            pg_execute(
                """
                insert into videos (video_id, source_video, title, duration, webpage_url, s3_raw_key, updated_at)
                values (%(video_id)s, %(source_video)s, %(title)s, %(duration)s, %(webpage_url)s, %(s3_raw_key)s, now())
                on conflict (video_id) do update set
                    source_video = excluded.source_video,
                    title = excluded.title,
                    duration = excluded.duration,
                    webpage_url = excluded.webpage_url,
                    s3_raw_key = excluded.s3_raw_key,
                    updated_at = now()
                """,
                payload,
            )
        return len(rows)

    def list_all(self) -> list[VideoRecord]:
        self._log("select_all")
        rows = pg_fetch_all("select * from videos order by video_id")
        return [self._from_row(row) for row in rows]

    def get(self, video_id: str) -> VideoRecord | None:
        self._log("select_one", video_id)
        row = pg_fetch_one(
            "select * from videos where video_id = %s or source_video = %s limit 1",
            (video_id, video_id),
        )
        return self._from_row(row) if row is not None else None

    def count(self) -> int:
        row = pg_fetch_one("select count(*) as n from videos")
        return int(row["n"]) if row else 0

    def _to_row(self, data: Mapping[str, Any]) -> dict[str, Any]:
        d = self._normalize(dict(data))
        return {
            "video_id": d.get("video_id"),
            "source_video": d.get("source_video"),
            "title": d.get("title"),
            "duration": float(d["duration"]) if d.get("duration") is not None else None,
            "webpage_url": d.get("webpage_url"),
            "s3_raw_key": d.get("s3_key") or d.get("s3_raw_key"),
        }

    def _from_row(self, row: Mapping[str, Any]) -> VideoRecord:
        d = self._normalize(dict(row))
        return VideoRecord(
            video_id=d.get("video_id", ""),
            source_video=d.get("source_video"),
            title=d.get("title"),
            duration=d.get("duration"),
            webpage_url=d.get("webpage_url"),
            s3_raw_key=d.get("s3_raw_key"),
            updated_at=d.get("updated_at"),
        )

    def _log(self, operation: str, record_id: str | None = None) -> None:
        logger.debug("Postgres repository operation: table=%s operation=%s id=%s", self.table_name, operation, record_id)

    def _normalize(self, value: Any) -> Any:
        return normalize_value(value)
