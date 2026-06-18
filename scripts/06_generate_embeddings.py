import csv
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np
import requests
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLASSIFIED_DIR = PROJECT_ROOT / "data" / "fragments_classified"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
REPORT_PATH = PROJECT_ROOT / "data" / "metadata" / "youtube" / "embedding_generation_report.csv"

JSONL_PATH = EMBEDDINGS_DIR / "classified_fragments_embeddings.jsonl"
MATRIX_PATH = EMBEDDINGS_DIR / "fragment_embedding_matrix.npy"
METADATA_PATH = EMBEDDINGS_DIR / "fragment_embedding_metadata.json"

EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


# Company Ollama endpoint
OLLAMA_BASE_URL = "http://ollama-mobius-sales.mobiusdtaas.ai"
EMBED_MODEL = "nomic-embed-text:v1.5"

# Preferred endpoint for modern Ollama.
# Script will fallback to /api/embeddings if /api/embed fails.
EMBED_ENDPOINT = "/api/embed"
FALLBACK_EMBED_ENDPOINT = "/api/embeddings"

BATCH_SIZE = 16
REQUEST_TIMEOUT_SECONDS = 120
MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 5


def log(message: str) -> None:
    print(message, flush=True)


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_text(text: str) -> str:
    return " ".join((text or "").split()).strip()


def build_embedding_text(fragment: dict[str, Any]) -> str:
    """
    Nomic embedding convention:
    - Store fragments as search_document.
    - Later user prompts should be embedded as search_query.
    """

    primary_theme = fragment.get("primary_theme") or ""
    primary_chapter = fragment.get("primary_chapter") or ""
    keywords = fragment.get("keywords") or []
    summary = fragment.get("summary") or ""
    text = fragment.get("text") or ""

    keyword_text = ", ".join(str(k) for k in keywords[:15])

    embedding_text = f"""
search_document:
Theme: {primary_theme}
Chapter: {primary_chapter}
Keywords: {keyword_text}
Summary: {summary}
Transcript fragment: {text}
"""

    return clean_text(embedding_text)


def load_classified_fragments() -> list[dict[str, Any]]:
    fragment_files = sorted(CLASSIFIED_DIR.glob("*.classified.json"))

    all_fragments = []

    for path in fragment_files:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        for fragment in payload.get("fragments", []):
            record = {
                "fragment_id": fragment.get("fragment_id"),
                "source_video": fragment.get("source_video"),
                "video_id": fragment.get("video_id"),
                "start": fragment.get("start"),
                "end": fragment.get("end"),
                "duration": fragment.get("duration"),
                "word_count": fragment.get("word_count"),
                "text": fragment.get("text"),
                "summary": fragment.get("summary"),
                "keywords": fragment.get("keywords", []),
                "primary_theme": fragment.get("primary_theme"),
                "primary_chapter": fragment.get("primary_chapter"),
                "theme_tags": fragment.get("theme_tags", []),
                "chapter_tags": fragment.get("chapter_tags", []),
                "clip_usage_candidates": fragment.get("clip_usage_candidates", []),
                "quote_strength": fragment.get("quote_strength"),
                "clarity_score": fragment.get("clarity_score"),
                "narration_support_score": fragment.get("narration_support_score"),
                "classification_confidence": fragment.get("classification_confidence"),
                "source_file": path.name,
            }

            record["embedding_text"] = build_embedding_text(fragment)
            record["embedding_text_hash"] = stable_hash(record["embedding_text"])

            all_fragments.append(record)

    return all_fragments


def post_with_retries(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

            response.raise_for_status()
            return response.json()

        except Exception as exc:
            last_error = exc
            log(f"Request failed attempt {attempt}/{MAX_RETRIES}: {exc}")

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS)

    raise RuntimeError(f"Embedding request failed after retries: {last_error}")


def embed_batch_modern(texts: list[str]) -> list[list[float]]:
    """
    Uses Ollama /api/embed:
    {
      "model": "...",
      "input": ["text1", "text2"]
    }

    Expected response:
    {
      "embeddings": [[...], [...]]
    }
    """

    url = OLLAMA_BASE_URL.rstrip("/") + EMBED_ENDPOINT

    payload = {
        "model": EMBED_MODEL,
        "input": texts,
    }

    data = post_with_retries(url, payload)

    embeddings = data.get("embeddings")

    if not embeddings:
        raise RuntimeError(f"No embeddings returned from /api/embed. Response keys: {list(data.keys())}")

    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"Embedding count mismatch. Sent {len(texts)}, received {len(embeddings)}"
        )

    return embeddings


def embed_single_fallback(text: str) -> list[float]:
    """
    Uses Ollama older /api/embeddings:
    {
      "model": "...",
      "prompt": "text"
    }

    Expected response:
    {
      "embedding": [...]
    }
    """

    url = OLLAMA_BASE_URL.rstrip("/") + FALLBACK_EMBED_ENDPOINT

    payload = {
        "model": EMBED_MODEL,
        "prompt": text,
    }

    data = post_with_retries(url, payload)

    embedding = data.get("embedding")

    if not embedding:
        raise RuntimeError(f"No embedding returned from /api/embeddings. Response keys: {list(data.keys())}")

    return embedding


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    First tries modern batch endpoint /api/embed.
    If that fails, falls back to single-item /api/embeddings.
    """

    try:
        return embed_batch_modern(texts)

    except Exception as exc:
        log(f"/api/embed failed. Falling back to /api/embeddings. Reason: {exc}")

        embeddings = []
        for text in texts:
            embeddings.append(embed_single_fallback(text))

        return embeddings


def normalize_vector(vector: list[float]) -> list[float]:
    arr = np.array(vector, dtype=np.float32)
    norm = np.linalg.norm(arr)

    if norm == 0:
        return arr.tolist()

    return (arr / norm).tolist()


def write_report(rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "status",
        "total_fragments",
        "embedded_fragments",
        "failed_fragments",
        "embedding_dimension",
        "model",
        "endpoint",
        "jsonl_path",
        "matrix_path",
        "metadata_path",
        "error",
    ]

    with REPORT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log(f"Report written to: {REPORT_PATH}")


def main() -> None:
    log("Starting Step 06: Generate Fragment Embeddings")
    log(f"Project root: {PROJECT_ROOT}")
    log(f"Input classified directory: {CLASSIFIED_DIR}")
    log(f"Embeddings directory: {EMBEDDINGS_DIR}")
    log(f"Ollama base URL: {OLLAMA_BASE_URL}")
    log(f"Embedding model: {EMBED_MODEL}")
    log(f"Batch size: {BATCH_SIZE}")

    fragments = load_classified_fragments()

    log(f"Loaded classified fragments: {len(fragments)}")

    if not fragments:
        log("No classified fragments found. Run scripts/05_classify_fragments.py first.")
        return

    embedded_records = []
    embedding_matrix = []

    failed = []

    # Clear previous JSONL before writing.
    if JSONL_PATH.exists():
        JSONL_PATH.unlink()

    for batch_start in tqdm(range(0, len(fragments), BATCH_SIZE), desc="Embedding batches"):
        batch = fragments[batch_start: batch_start + BATCH_SIZE]
        texts = [item["embedding_text"] for item in batch]

        try:
            embeddings = embed_texts(texts)

            with JSONL_PATH.open("a", encoding="utf-8") as f:
                for fragment, embedding in zip(batch, embeddings):
                    normalized_embedding = normalize_vector(embedding)

                    record = dict(fragment)
                    record["embedding_model"] = EMBED_MODEL
                    record["embedding_dimension"] = len(normalized_embedding)
                    record["embedding_created_at"] = datetime.utcnow().isoformat()
                    record["embedding"] = normalized_embedding

                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

                    embedded_records.append(record)
                    embedding_matrix.append(normalized_embedding)

        except Exception as exc:
            log(f"FAILED batch {batch_start} - {batch_start + len(batch) - 1}: {exc}")

            for fragment in batch:
                failed.append(
                    {
                        "fragment_id": fragment.get("fragment_id"),
                        "source_video": fragment.get("source_video"),
                        "error": str(exc),
                    }
                )

    if embedding_matrix:
        matrix = np.array(embedding_matrix, dtype=np.float32)
        np.save(MATRIX_PATH, matrix)

        embedding_dimension = int(matrix.shape[1])
    else:
        embedding_dimension = 0

    metadata_payload = {
        "created_at": datetime.utcnow().isoformat(),
        "embedding_model": EMBED_MODEL,
        "ollama_base_url": OLLAMA_BASE_URL,
        "source_directory": str(CLASSIFIED_DIR),
        "jsonl_path": str(JSONL_PATH),
        "matrix_path": str(MATRIX_PATH),
        "total_fragments": len(fragments),
        "embedded_fragments": len(embedded_records),
        "failed_fragments": len(failed),
        "embedding_dimension": embedding_dimension,
        "batch_size": BATCH_SIZE,
        "note": "Embeddings are L2-normalized. Use dot product or cosine similarity for retrieval.",
        "failed": failed,
    }

    METADATA_PATH.write_text(
        json.dumps(metadata_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    status = "success" if not failed else "partial_success"

    write_report(
        [
            {
                "status": status,
                "total_fragments": len(fragments),
                "embedded_fragments": len(embedded_records),
                "failed_fragments": len(failed),
                "embedding_dimension": embedding_dimension,
                "model": EMBED_MODEL,
                "endpoint": OLLAMA_BASE_URL,
                "jsonl_path": str(JSONL_PATH),
                "matrix_path": str(MATRIX_PATH),
                "metadata_path": str(METADATA_PATH),
                "error": "" if not failed else f"{len(failed)} fragments failed",
            }
        ]
    )

    log("")
    log("Summary")
    log(f"Total fragments: {len(fragments)}")
    log(f"Embedded fragments: {len(embedded_records)}")
    log(f"Failed fragments: {len(failed)}")
    log(f"Embedding dimension: {embedding_dimension}")
    log(f"JSONL: {JSONL_PATH}")
    log(f"Matrix: {MATRIX_PATH}")
    log(f"Metadata: {METADATA_PATH}")

    if failed:
        log("Some fragments failed. Check metadata failed list.")


if __name__ == "__main__":
    main()
