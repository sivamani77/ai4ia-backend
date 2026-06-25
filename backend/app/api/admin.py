from fastapi import APIRouter

from app.services import embedding_index, fragment_store

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.post("/reindex")
def api_reindex():
    """Reload fragment store count and embedding matrix from disk.

    Call after running scripts/06_generate_embeddings.py to pick up a
    new .npy matrix without restarting the server.
    """
    fs_result = fragment_store.reload()
    ei_result = embedding_index.reload()
    return {
        "status": "ok",
        "fragment_store": fs_result,
        "embedding_index": ei_result,
        "message": "Index reloaded from disk",
    }
