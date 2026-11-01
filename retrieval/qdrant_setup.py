"""Validate the local Qwen cache, create a Qdrant collection, and upsert chunks.

Run with ``uv run python retrieval/qdrant_setup.py --check-only`` to validate
that the local chunk order and embedding cache match. Omit ``--check-only``
to create/validate the remote collection and upload the vectors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import httpx
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_DIR = ROOT / "preprocessing" / "data" / "chunks"
EMBEDDING_DIR = ROOT / "preprocessing" / "data" / "embeddings" / "qwen3-8B-4bit"
EMBEDDINGS_FILE = EMBEDDING_DIR / "embeddings.npy"
EMBEDDING_METADATA_FILE = EMBEDDING_DIR / "metadata.json"
DEFAULT_BATCH_SIZE = 128
MAX_UPSERT_ATTEMPTS = 4
RETRY_BASE_DELAY_SECONDS = 1.0
RETRYABLE_HTTP_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_chunks() -> list[dict[str, Any]]:
    """Load chunks in the same deterministic order used by embeddings.py."""
    files = sorted(CHUNKS_DIR.glob("*_chunks.json"))
    if not files:
        raise FileNotFoundError(f"No chunk files found in {CHUNKS_DIR}")

    chunks = []
    for path in files:
        document = json.loads(path.read_text(encoding="utf-8"))
        document_id = document.get("document_id")
        if not document_id:
            raise ValueError(f"Chunk file {path} has no document_id.")
        for item in document.get("chunks", []):
            text = item.get("text")
            chunk_id = item.get("chunk_id")
            if not isinstance(text, str) or not text.strip() or not chunk_id:
                raise ValueError(f"Chunk file {path} contains a chunk without an ID or text.")
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "section_title": item.get("section_title", ""),
                    "section_index": item.get("section_index"),
                    "chunk_index": item.get("chunk_index"),
                    "token_count": item.get("token_count"),
                    "overlap_prev_tokens": item.get("overlap_prev_tokens", 0),
                    "text": text,
                    "fingerprint": sha256_text(text),
                }
            )

    ids = [chunk["chunk_id"] for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Chunk files contain duplicate chunk_id values.")
    return chunks


def load_and_validate_cache() -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    chunks = load_chunks()
    if not EMBEDDINGS_FILE.exists() or not EMBEDDING_METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Expected Qwen embeddings and metadata under {EMBEDDING_DIR}. "
            "Generate or repair the embedding cache before uploading."
        )

    metadata = json.loads(EMBEDDING_METADATA_FILE.read_text(encoding="utf-8"))
    vectors = np.load(EMBEDDINGS_FILE, mmap_mode="r", allow_pickle=False)
    if metadata.get("model") != "Qwen/Qwen3-Embedding-8B":
        raise ValueError(f"Unexpected embedding model in cache: {metadata.get('model')!r}.")

    dimension = metadata.get("embedding_dimension")
    if not isinstance(dimension, int) or dimension <= 0:
        raise ValueError("Embedding metadata has no valid embedding_dimension.")
    if vectors.ndim != 2 or vectors.shape != (len(chunks), dimension):
        raise ValueError(
            f"Embedding matrix shape is {vectors.shape}; current corpus requires "
            f"({len(chunks)}, {dimension}). Refusing to pair vectors with the wrong chunks."
        )

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    fingerprints = [chunk["fingerprint"] for chunk in chunks]
    if metadata.get("chunk_ids") != chunk_ids or metadata.get("chunk_fingerprints") != fingerprints:
        raise ValueError(
            "Embedding cache IDs/text do not exactly match the current chunk files. "
            "Run preprocessing/embeddings.py to safely rebase matching vectors and encode changed chunks."
        )
    if metadata.get("embedded_chunk_ids") != chunk_ids or metadata.get("embedded_chunk_fingerprints") != fingerprints:
        raise ValueError("Embedding cache is incomplete or its embedded-chunk manifest does not match.")
    if metadata.get("embedded_chunks") != len(chunks):
        raise ValueError("Embedding cache metadata reports an incomplete corpus.")
    if not np.isfinite(vectors).all():
        raise ValueError("Embedding matrix contains non-finite values.")

    return chunks, vectors, metadata


def collection_distance() -> models.Distance:
    name = os.getenv("QDRANT_DISTANCE", "Cosine").strip().upper()
    try:
        return models.Distance[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported QDRANT_DISTANCE {name!r}.") from exc


def ensure_collection(
    client: QdrantClient, collection_name: str, dimension: int, distance: models.Distance
) -> None:
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={"dense": models.VectorParams(size=dimension, distance=distance)},
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
        client.create_payload_index(
            collection_name=collection_name,
            field_name="corpus_version",
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )
        print(f"Created collection {collection_name!r} with dense dimension {dimension}.")
        return

    info = client.get_collection(collection_name)
    configured = info.config.params.vectors
    dense = configured.get("dense") if isinstance(configured, dict) else None
    if dense is None:
        raise ValueError(
            f"Collection {collection_name!r} does not have the expected named 'dense' vector."
        )
    if dense.size != dimension or dense.distance != distance:
        raise ValueError(
            f"Collection {collection_name!r} has dense size/distance "
            f"{dense.size}/{dense.distance}; expected {dimension}/{distance}."
        )
    client.create_payload_index(
        collection_name=collection_name,
        field_name="corpus_version",
        field_schema=models.PayloadSchemaType.KEYWORD,
        wait=True,
    )
    print(f"Using existing compatible collection {collection_name!r}.")


def corpus_version(chunks: list[dict[str, Any]], metadata: dict[str, Any]) -> str:
    identity = {
        "model": metadata["model"],
        "embedding_dimension": metadata["embedding_dimension"],
        "max_seq_length": metadata.get("max_seq_length"),
        "embedding_config_version": metadata.get("embedding_config_version"),
        "model_kwargs": metadata.get("model_kwargs"),
        "normalize_embeddings": metadata.get("normalize_embeddings"),
        "prompt_name": metadata.get("prompt_name"),
        "chunk_ids": [chunk["chunk_id"] for chunk in chunks],
        "chunk_fingerprints": [chunk["fingerprint"] for chunk in chunks],
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def is_retryable_upsert_error(error: Exception) -> bool:
    """Retry network failures and transient server responses, not bad requests/auth errors."""
    if isinstance(error, UnexpectedResponse):
        return error.status_code in RETRYABLE_HTTP_STATUS_CODES
    if isinstance(error, ResponseHandlingException):
        return isinstance(error.source, httpx.TransportError)
    return isinstance(error, httpx.TransportError)


def upsert_batch_with_retry(
    client: QdrantClient,
    collection_name: str,
    points: list[models.PointStruct],
    start: int,
    stop: int,
    total: int,
) -> None:
    """Upload one batch sequentially, retrying transient failures with backoff.

    Point IDs are deterministic, so retrying is safe even if Qdrant accepted the
    request but the client did not receive its acknowledgment.
    """
    for attempt in range(1, MAX_UPSERT_ATTEMPTS + 1):
        try:
            client.upsert(collection_name=collection_name, points=points, wait=True)
            print(f"Uploaded chunks {start + 1}-{stop}/{total} (attempt {attempt}).")
            return
        except Exception as error:
            if not is_retryable_upsert_error(error) or attempt == MAX_UPSERT_ATTEMPTS:
                if is_retryable_upsert_error(error):
                    raise RuntimeError(
                        f"Batch chunks {start + 1}-{stop}/{total} failed after "
                        f"{MAX_UPSERT_ATTEMPTS} attempts: {error}"
                    ) from error
                raise

            delay = RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            print(
                f"Transient error uploading chunks {start + 1}-{stop}/{total} "
                f"(attempt {attempt}/{MAX_UPSERT_ATTEMPTS}): {error}. "
                f"Retrying in {delay:g}s."
            )
            time.sleep(delay)


def upload_chunks(
    client: QdrantClient,
    collection_name: str,
    chunks: list[dict[str, Any]],
    vectors: np.ndarray,
    version: str,
    batch_size: int,
) -> None:
    for start in range(0, len(chunks), batch_size):
        stop = min(start + batch_size, len(chunks))
        points = []
        for index in range(start, stop):
            chunk = chunks[index]
            payload = {
                key: chunk[key]
                for key in (
                    "chunk_id",
                    "document_id",
                    "section_title",
                    "section_index",
                    "chunk_index",
                    "token_count",
                    "overlap_prev_tokens",
                    "text",
                )
            }
            payload["corpus_version"] = version
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["chunk_id"])),
                    vector={"dense": np.asarray(vectors[index], dtype=np.float32).tolist()},
                    payload=payload,
                )
            )
        upsert_batch_with_retry(
            client, collection_name, points, start, stop, len(chunks)
        )

    verify_uploaded_corpus(client, collection_name, len(chunks), version)


def verify_uploaded_corpus(
    client: QdrantClient, collection_name: str, expected_count: int, version: str
) -> None:
    version_filter = models.Filter(
        must=[models.FieldCondition(key="corpus_version", match=models.MatchValue(value=version))]
    )
    count = client.count(collection_name=collection_name, count_filter=version_filter, exact=True).count
    if count != expected_count:
        raise RuntimeError(f"Qdrant verification found {count} points for this corpus; expected {expected_count}.")
    print(f"Verified {count} points for corpus version {version[:12]}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Validate local files without contacting Qdrant.")
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Create or validate the Qdrant collection without loading or uploading embeddings.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Create/validate the collection index and verify an existing upload without upserting vectors.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.setup_only and args.verify_only:
        parser.error("--setup-only and --verify-only cannot be used together")

    if args.setup_only:
        load_dotenv(ROOT / ".env")
        required = ("QDRANT_URL", "QDRANT_API_KEY", "QDRANT_COLLECTION")
        missing = [key for key in required if not os.getenv(key)]
        if missing:
            raise ValueError(f"Missing Qdrant environment settings: {', '.join(missing)}")
        dimension = int(os.getenv("QDRANT_VECTOR_SIZE", "4096"))
        distance = collection_distance()
        client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])
        collection_name = os.environ["QDRANT_COLLECTION"]
        ensure_collection(client, collection_name, dimension, distance)
        print(f"Qdrant collection setup complete: {collection_name!r}.")
        return

    chunks, vectors, metadata = load_and_validate_cache()
    dimension = int(metadata["embedding_dimension"])
    print(f"Validated {len(chunks)} chunks and a {dimension}-dimensional Qwen embedding matrix.")
    version = corpus_version(chunks, metadata)
    if args.check_only:
        print(f"Corpus version: {version}")
        return

    load_dotenv(ROOT / ".env")
    required = ("QDRANT_URL", "QDRANT_API_KEY", "QDRANT_COLLECTION")
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise ValueError(f"Missing Qdrant environment settings: {', '.join(missing)}")

    configured_dimension = int(os.getenv("QDRANT_VECTOR_SIZE", str(dimension)))
    if configured_dimension != dimension:
        raise ValueError(
            f"QDRANT_VECTOR_SIZE={configured_dimension} does not match embedding dimension {dimension}."
        )
    distance = collection_distance()
    client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])
    collection_name = os.environ["QDRANT_COLLECTION"]
    ensure_collection(client, collection_name, dimension, distance)
    if args.verify_only:
        verify_uploaded_corpus(client, collection_name, len(chunks), version)
        return
    upload_chunks(client, collection_name, chunks, vectors, version, args.batch_size)
    print(f"Qdrant ingestion complete: {collection_name!r}.")


if __name__ == "__main__":
    main()
