from fastapi import APIRouter

from app.services import fragment_store

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.post("/reindex")
def api_reindex():
    """Rebuild the in-memory fragment store (call after a pipeline run)."""
    result = fragment_store.reload()
    return {"status": "ok", **result}
