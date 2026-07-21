import argparse
import csv
import json
import time
import hashlib
import sys
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np
import requests
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Local dev layout: backend/app/. Docker image layout (see Dockerfile): app/
# directly under PROJECT_ROOT, with no backend/ directory at all.
_BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(_BACKEND_DIR if (_BACKEND_DIR / "app").is_dir() else PROJECT_ROOT))

from app.core.settings import settings  # noqa: E402

CLASSIFIED_DIR = PROJECT_ROOT / "data" / "fragments_classified"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
REPORT_PATH = PROJECT_ROOT / "data" / "metadata" / "youtube" / "embedding_generation_report.csv"

JSONL_PATH = EMBEDDINGS_DIR / "classified_fragments_embeddings.jsonl"
MATRIX_PATH = EMBEDDINGS_DIR / "fragment_embedding_matrix.npy"
METADATA_PATH = EMBEDDINGS_DIR / "fragment_embedding_metadata.json"

EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


# Company Ollama endpoint
OLLAMA_BASE_URL = settings.ollama_base_url
EMBED_MODEL = settings.embedding_model

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


def load_classified_fragments(source_dir: Path | None = None) -> list[dict[str, Any]]:
    search_dir = source_dir or CLASSIFIED_DIR
    fragment_files = sorted(search_dir.glob("*.classified.json"))

    all_fragments = []

    for path in fragment_files:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        for fragment in payload.get("fragments", []):
            record = {
                "fragment_id": fragment.get("fragment_id"),
                "source_video": fragment.get("source_video"),
                "video_id": fragment.get("video_id"),
                "video_year": fragment.get("video_year"),
                "video_upload_date": fragment.get("video_upload_date"),
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
                "source_transcript_version": fragment.get("source_transcript_version", ""),
                "audio_quality_status": fragment.get("audio_quality_status", ""),
                "transcript_quality_score": fragment.get("transcript_quality_score"),
                "speaker": fragment.get("speaker"),
                "speaker_attribution": fragment.get("speaker_attribution"),
                "source_file": path.name,
            }

            record["embedding_text"] = build_embedding_text(fragment)
            record["embedding_text_hash"] = stable_hash(record["embedding_text"])

            all_fragments.append(record)

    return all_fragments


def load_fragments_jsonl(jsonl_path: Path) -> list[dict[str, Any]]:
    """Load fragments from a JSONL file (e.g. data/fragments_cleaned/*.fragments.json lines)."""
    fragments = []

    if not jsonl_path.exists():
        return fragments

    # Support both JSONL and standard JSON fragment files
    text = jsonl_path.read_text(encoding="utf-8").strip()
    if not text:
        return fragments

    # Check if it's a standard JSON fragment file (not JSONL)
    if text.startswith("{"):
        try:
            payload = json.loads(text)
            raw_fragments = payload.get("fragments", [])
            for fragment in raw_fragments:
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
                    "source_transcript_version": fragment.get("source_transcript_version", ""),
                    "audio_quality_status": fragment.get("audio_quality_status", ""),
                    "transcript_quality_score": fragment.get("transcript_quality_score"),
                    "speaker": fragment.get("speaker"),
                    "speaker_attribution": fragment.get("speaker_attribution"),
                    "source_file": jsonl_path.name,
                    "embedding_model": "",
                    "text_hash": stable_hash(fragment.get("text", "")),
                    "created_at": fragment.get("created_at", ""),
                }
                record["embedding_text"] = build_embedding_text(fragment)
                record["embedding_text_hash"] = stable_hash(record["embedding_text"])
                fragments.append(record)
            return fragments
        except json.JSONDecodeError:
            pass

    # True JSONL
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            fragment = json.loads(line)
            record = dict(fragment)
            record.setdefault("source_transcript_version", "")
            record.setdefault("audio_quality_status", "")
            record["embedding_text"] = build_embedding_text(fragment)
            record["embedding_text_hash"] = stable_hash(record["embedding_text"])
            record["text_hash"] = stable_hash(fragment.get("text", ""))
            fragments.append(record)
        except json.JSONDecodeError:
            continue

    return fragments


def load_fragments_from_dir(fragment_dir: Path) -> list[dict[str, Any]]:
    """Load all fragment JSON files from a directory (fragments_cleaned layout)."""
    all_fragments = []
    for path in sorted(fragment_dir.glob("*.fragments.json")):
        frags = load_fragments_jsonl(path)
        all_fragments.extend(frags)
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


def embed_texts(texts: list[str]) -> list[list[float] | None]:
    """
    First tries modern batch endpoint /api/embed.
    If that fails, falls back to single-item /api/embeddings, isolating each
    item's failure (e.g. one oversized fragment exceeding the model's context
    length) so it doesn't take out every other item in the same batch — a
    failed item comes back as None rather than aborting the whole call.
    """

    try:
        return embed_batch_modern(texts)

    except Exception as exc:
        log(f"/api/embed failed. Falling back to /api/embeddings. Reason: {exc}")

        embeddings: list[list[float] | None] = []
        for text in texts:
            try:
                embeddings.append(embed_single_fallback(text))
            except Exception as item_exc:
                log(f"  Item failed in fallback: {item_exc}")
                embeddings.append(None)

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
    parser = argparse.ArgumentParser(description="Generate fragment embeddings via Ollama")
    parser.add_argument(
        "--fragments",
        type=Path,
        default=None,
        help=(
            "Path to a JSONL file, a .fragments.json file, or a directory of "
            "*.fragments.json files (e.g. data/fragments_cleaned/). "
            "When omitted, loads from the default fragments_classified directory."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for embeddings (default: data/embeddings/)",
    )
    args = parser.parse_args()

    # Resolve output directory
    out_dir = args.output or EMBEDDINGS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / "classified_fragments_embeddings.jsonl"
    matrix_path = out_dir / "fragment_embedding_matrix.npy"
    metadata_path = out_dir / "fragment_embedding_metadata.json"

    log("Starting Step 06: Generate Fragment Embeddings")
    log(f"Project root: {PROJECT_ROOT}")
    log(f"Ollama base URL: {OLLAMA_BASE_URL}")
    log(f"Embedding model: {EMBED_MODEL}")
    log(f"Batch size: {BATCH_SIZE}")
    log(f"Output dir: {out_dir}")

    # Load fragments from the specified source
    if args.fragments is not None:
        frag_path = args.fragments
        if frag_path.is_dir():
            log(f"Loading from directory: {frag_path}")
            fragments = load_fragments_from_dir(frag_path)
        else:
            log(f"Loading from file: {frag_path}")
            fragments = load_fragments_jsonl(frag_path)
        log(f"Loaded fragments from --fragments: {len(fragments)}")
    else:
        log(f"Input classified directory: {CLASSIFIED_DIR}")
        fragments = load_classified_fragments()
        log(f"Loaded classified fragments: {len(fragments)}")

    if not fragments:
        log("No fragments found. Check --fragments path or run scripts/05_classify_fragments.py first.")
        return

    embedded_records = []
    embedding_matrix = []

    failed = []

    # Reuse embeddings from a previous run when the embedding text is unchanged,
    # keyed by embedding_text_hash, so re-runs don't re-embed everything from scratch.
    cache_by_hash: dict[str, dict[str, Any]] = {}
    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    old_record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text_hash = old_record.get("embedding_text_hash")
                if text_hash and old_record.get("embedding_model") == EMBED_MODEL and old_record.get("embedding"):
                    cache_by_hash[text_hash] = old_record
        jsonl_path.unlink()

    to_embed = [f for f in fragments if f["embedding_text_hash"] not in cache_by_hash]
    log(f"Reusing {len(fragments) - len(to_embed)} cached embedding(s); embedding {len(to_embed)} fragment(s)")

    with jsonl_path.open("a", encoding="utf-8") as f:
        for fragment in fragments:
            cached = cache_by_hash.get(fragment["embedding_text_hash"])
            if cached is not None:
                record = dict(fragment)
                record["embedding_model"] = cached["embedding_model"]
                record["embedding_dimension"] = cached["embedding_dimension"]
                record["embedding_created_at"] = cached["embedding_created_at"]
                record["text_hash"] = stable_hash(fragment.get("text", ""))
                record["embedding"] = cached["embedding"]

                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                embedded_records.append(record)
                embedding_matrix.append(record["embedding"])

        for batch_start in tqdm(range(0, len(to_embed), BATCH_SIZE), desc="Embedding batches"):
            batch = to_embed[batch_start: batch_start + BATCH_SIZE]
            texts = [item["embedding_text"] for item in batch]

            try:
                embeddings = embed_texts(texts)

                for fragment, embedding in zip(batch, embeddings):
                    if embedding is None:
                        failed.append(
                            {
                                "fragment_id": fragment.get("fragment_id"),
                                "source_video": fragment.get("source_video"),
                                "error": "embedding request failed for this item",
                            }
                        )
                        continue

                    normalized_embedding = normalize_vector(embedding)

                    record = dict(fragment)
                    record["embedding_model"] = EMBED_MODEL
                    record["embedding_dimension"] = len(normalized_embedding)
                    record["embedding_created_at"] = datetime.utcnow().isoformat()
                    record["text_hash"] = stable_hash(fragment.get("text", ""))
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
        np.save(matrix_path, matrix)

        embedding_dimension = int(matrix.shape[1])
    else:
        embedding_dimension = 0

    source_desc = str(args.fragments) if args.fragments else str(CLASSIFIED_DIR)

    metadata_payload = {
        "created_at": datetime.utcnow().isoformat(),
        "embedding_model": EMBED_MODEL,
        "ollama_base_url": OLLAMA_BASE_URL,
        "source": source_desc,
        "jsonl_path": str(jsonl_path),
        "matrix_path": str(matrix_path),
        "total_fragments": len(fragments),
        "embedded_fragments": len(embedded_records),
        "failed_fragments": len(failed),
        "embedding_dimension": embedding_dimension,
        "batch_size": BATCH_SIZE,
        "note": "Embeddings are L2-normalized. Use dot product or cosine similarity for retrieval.",
        "failed": failed,
    }

    metadata_path.write_text(
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
                "jsonl_path": str(jsonl_path),
                "matrix_path": str(matrix_path),
                "metadata_path": str(metadata_path),
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
    log(f"JSONL: {jsonl_path}")
    log(f"Matrix: {matrix_path}")
    log(f"Metadata: {metadata_path}")

    if failed:
        log("Some fragments failed. Check metadata failed list.")


if __name__ == "__main__":
    main()
