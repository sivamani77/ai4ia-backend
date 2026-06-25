from functools import lru_cache

from app.db.repositories.event_repository import EventRepository
from app.db.repositories.job_repository import JobRepository


@lru_cache(maxsize=1)
def get_job_repository() -> JobRepository:
    return JobRepository()


@lru_cache(maxsize=1)
def get_event_repository() -> EventRepository:
    return EventRepository()


__all__ = [
    "EventRepository",
    "JobRepository",
    "get_event_repository",
    "get_job_repository",
]
