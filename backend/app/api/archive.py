from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.core.config import SEMANTIC_TOP_K_DEFAULT
from app.core.paths import RAW_VIDEO_DIR
from app.services import embedding_index, fragment_store
from app.services.archive_service import (
    list_videos,
    get_video,
    search_fragments,
    get_fragment,
    get_themes_and_chapters,
)

router = APIRouter(prefix="/archive", tags=["Archive"])


class SemanticSearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500)
    top_k: int = Field(SEMANTIC_TOP_K_DEFAULT, ge=1, le=100)
    filters: dict | None = None


@router.post("/semantic-search")
def api_semantic_search(req: SemanticSearchRequest):
    """Embedding-based retrieval. Falls back to keyword search if Ollama is unreachable."""
    if not embedding_index.available():
        raise HTTPException(status_code=503, detail="Embedding matrix not available.")
    try:
        results = embedding_index.search(req.query, top_k=req.top_k, filters=req.filters)
        mode = "semantic"
    except Exception as exc:  # noqa: BLE001 — degrade gracefully so the demo never hard-fails
        results = search_fragments(query=req.query, limit=req.top_k)
        return {"mode": "keyword_fallback", "reason": str(exc),
                "count": len(results), "results": results}
    return {"mode": mode, "count": len(results), "results": results}


@router.get("/speakers")
def api_speakers():
    summary = fragment_store.speakers_summary()
    return {"count": len(summary), "speakers": summary}


@router.get("/videos")
def api_list_videos(
    query: str | None = None,
    limit: int = Query(100, ge=1, le=500),
):
    videos = list_videos()

    if query:
        q = query.lower().strip()
        videos = [
            video for video in videos
            if q in video.get("title", "").lower()
            or q in video.get("speaker_or_topic", "").lower()
            or q in video.get("video_id", "").lower()
        ]

    return videos[:limit]


@router.get("/videos/{video_id}")
def api_get_video(video_id: str):
    video = get_video(video_id)

    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    return video


@router.get("/videos/{video_id}/stream")
def api_stream_video(video_id: str):
    path = RAW_VIDEO_DIR / f"{video_id}.mp4"

    if not path.exists():
        matches = sorted(RAW_VIDEO_DIR.glob(f"*{video_id}*.mp4"))
        if not matches:
            raise HTTPException(status_code=404, detail="Video file not found")
        path = matches[0]

    return FileResponse(path, media_type="video/mp4", filename=path.name)


@router.get("/themes")
def api_themes():
    """Nested theme -> chapters hierarchy with counts (for tree/navigation UI)."""
    return get_themes_and_chapters()


@router.get("/fragments")
def api_fragments(
    query: str | None = None,
    theme: str | None = None,
    chapter: str | None = None,
    speaker: str | None = None,
    source_video: str | None = None,
    clip_usage: str | None = None,
    min_duration: float | None = None,
    max_duration: float | None = None,
    start_after: float | None = None,
    end_before: float | None = None,
    limit: int = Query(50, ge=1, le=200),
):
    return search_fragments(
        query=query,
        theme=theme,
        chapter=chapter,
        speaker=speaker,
        source_video=source_video,
        clip_usage=clip_usage,
        min_duration=min_duration,
        max_duration=max_duration,
        start_after=start_after,
        end_before=end_before,
        limit=limit,
    )


@router.get("/search")
def api_search(
    q: str | None = None,
    theme: str | None = None,
    chapter: str | None = None,
    speaker: str | None = None,
    source_video: str | None = None,
    clip_usage: str | None = None,
    min_duration: float | None = None,
    max_duration: float | None = None,
    start_after: float | None = None,
    end_before: float | None = None,
    limit: int = Query(50, ge=1, le=200),
):
    return search_fragments(
        query=q,
        theme=theme,
        chapter=chapter,
        speaker=speaker,
        source_video=source_video,
        clip_usage=clip_usage,
        min_duration=min_duration,
        max_duration=max_duration,
        start_after=start_after,
        end_before=end_before,
        limit=limit,
    )


@router.get("/fragments/{fragment_id}/preview")
def api_fragment_preview(fragment_id: str):
    fragment = fragment_store.get(fragment_id) or get_fragment(fragment_id)
    if not fragment:
        raise HTTPException(status_code=404, detail="Fragment not found")

    source_video = fragment.get("source_video")
    return {
        "fragment_id": fragment_id,
        "video_stream_url": f"/api/archive/videos/{source_video}/stream",
        "start": fragment.get("start"),
        "end": fragment.get("end"),
        "duration": fragment.get("duration"),
        "text": fragment.get("text"),
        "speaker": fragment.get("speaker"),
        "speaker_attribution": fragment.get("speaker_attribution"),
        "primary_theme": fragment.get("primary_theme"),
        "primary_chapter": fragment.get("primary_chapter"),
        "year": fragment.get("year"),
        "edition": fragment.get("edition"),
    }


@router.get("/fragments/{fragment_id}")
def api_get_fragment(fragment_id: str):
    fragment = fragment_store.get(fragment_id) or get_fragment(fragment_id)

    if not fragment:
        raise HTTPException(status_code=404, detail="Fragment not found")

    return fragment
