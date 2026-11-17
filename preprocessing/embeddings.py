"""Generate resumable Qwen3 embeddings for the complete chunk corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ARXIV_CHUNKS_DIR = ROOT / "preprocessing" / "data" / "chunks"
ARXIV_CACHE_DIR = ROOT / "preprocessing" / "data" / "embeddings" / "qwen3-8B-4bit"
CHUNKS_DIR = ARXIV_CHUNKS_DIR
CACHE_DIR = ARXIV_CACHE_DIR
EMBEDDINGS_FILE = CACHE_DIR / "embeddings.npy"
METADATA_FILE = CACHE_DIR / "metadata.json"
PROGRESS_FILE = CACHE_DIR / "progress.json"
PARTIAL_EMBEDDINGS_FILE = CACHE_DIR / "embeddings.partial.npy"
PARTIAL_METADATA_FILE = CACHE_DIR / "metadata.partial.json"
PENDING_BATCH_FILE = CACHE_DIR / "pending_batch.npz"

MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
BATCH_SIZE = 2
MAX_SEQ_LENGTH = 2000
CONFIG_VERSION = 1
MODEL_KWARGS = {
    "quantization_config": {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "float16",
    },
    "torch_dtype": "float16",
}


def configure_corpus(corpus: str) -> None:
    """Select corpus-local chunk and embedding-cache paths.

    The default arXiv paths are retained verbatim for existing callers.
    """
    global CHUNKS_DIR, CACHE_DIR, EMBEDDINGS_FILE, METADATA_FILE, PROGRESS_FILE
    global PARTIAL_EMBEDDINGS_FILE, PARTIAL_METADATA_FILE, PENDING_BATCH_FILE
    if corpus == "arxiv":
        CHUNKS_DIR = ARXIV_CHUNKS_DIR
        CACHE_DIR = ARXIV_CACHE_DIR
        EMBEDDINGS_FILE = CACHE_DIR / "embeddings.npy"
        METADATA_FILE = CACHE_DIR / "metadata.json"
        PROGRESS_FILE = CACHE_DIR / "progress.json"
        PARTIAL_EMBEDDINGS_FILE = CACHE_DIR / "embeddings.partial.npy"
        PARTIAL_METADATA_FILE = CACHE_DIR / "metadata.partial.json"
        PENDING_BATCH_FILE = CACHE_DIR / "pending_batch.npz"
        return
    if corpus != "peerqa":
        raise ValueError(f"Unsupported corpus {corpus!r}.")
    CHUNKS_DIR = ROOT / "download_corpus" / "peerqa" / "chunks"
    CACHE_DIR = ROOT / "preprocessing" / "data" / "peerqa" / "embeddings" / "qwen3-8B-4bit"
    EMBEDDINGS_FILE = CACHE_DIR / "embeddings.npy"
    METADATA_FILE = CACHE_DIR / "metadata.json"
    PROGRESS_FILE = CACHE_DIR / "progress.json"
    PARTIAL_EMBEDDINGS_FILE = CACHE_DIR / "embeddings.partial.npy"
    PARTIAL_METADATA_FILE = CACHE_DIR / "metadata.partial.json"
    PENDING_BATCH_FILE = CACHE_DIR / "pending_batch.npz"


def load_chunks() -> list[dict[str, Any]]:
    """Load corpus chunks in deterministic file and source order."""
    files = sorted(CHUNKS_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No chunk JSON files found in {CHUNKS_DIR}")

    chunks = []
    for path in files:
        doc = json.loads(path.read_text(encoding="utf-8"))
        for item in doc.get("chunks", []):
            chunk_id = item["chunk_id"]
            chunks.append(
                {
                    "id": chunk_id,
                    "text": item["text"],
                    "token_count": item.get("token_count"),
                }
            )
    counts: dict[str, int] = {}
    for chunk in chunks:
        counts[chunk["id"]] = counts.get(chunk["id"], 0) + 1
    duplicate_count = sum(count - 1 for count in counts.values())
    if duplicate_count:
        print(
            f"Warning: {duplicate_count} repeated chunk IDs found; preserving every source record "
            "and validating its ordered text fingerprint when resuming."
        )
    return chunks


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, array)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def cache_signature(chunks: list[dict[str, Any]], dimension: int) -> dict:
    return {
        "model": MODEL_NAME,
        "embedding_dimension": dimension,
        "total_chunks": len(chunks),
        "chunk_ids": [chunk["id"] for chunk in chunks],
        "chunk_fingerprints": [fingerprint(chunk["text"]) for chunk in chunks],
        "batch_size": BATCH_SIZE,
        "max_seq_length": MAX_SEQ_LENGTH,
        "embedding_config_version": CONFIG_VERSION,
        "model_kwargs": MODEL_KWARGS,
        "normalize_embeddings": True,
        "prompt_name": None,
    }


def embedding_config(dimension: int) -> dict:
    """Fields that change the meaning or shape of an embedding vector."""
    return {
        "model": MODEL_NAME,
        "embedding_dimension": dimension,
        "max_seq_length": MAX_SEQ_LENGTH,
        "embedding_config_version": CONFIG_VERSION,
        "model_kwargs": MODEL_KWARGS,
        "normalize_embeddings": True,
        "prompt_name": None,
    }


def validate_cached_matrix(metadata: dict, matrix: np.ndarray, dimension: int) -> None:
    expected = embedding_config(dimension)
    actual = {key: metadata.get(key) for key in expected}
    # A changed context limit can still reuse exact chunks that fit under both
    # limits. create_partial_matrix filters out any longer chunks for re-encoding.
    same_except_context = {
        key: value for key, value in actual.items() if key != "max_seq_length"
    } == {key: value for key, value in expected.items() if key != "max_seq_length"}
    context_limits_are_valid = all(
        isinstance(value, int) and value > 0
        for value in (actual.get("max_seq_length"), expected.get("max_seq_length"))
    )
    if actual != expected and not (same_except_context and context_limits_are_valid):
        raise ValueError("Embedding cache model/configuration/dimension is incompatible; refusing reuse.")
    ids = metadata.get("embedded_chunk_ids", [])
    hashes = metadata.get("embedded_chunk_fingerprints", [])
    if matrix.ndim != 2 or matrix.shape[1] != dimension:
        raise ValueError("Embedding cache matrix has an unexpected shape.")
    if len(ids) != len(hashes) or len(ids) != matrix.shape[0]:
        raise ValueError("Embedding cache metadata and matrix disagree; refusing unsafe reuse.")


def write_progress(signature: dict, completed: list[int], dimension: int) -> None:
    atomic_json(
        PROGRESS_FILE,
        {
            "model": MODEL_NAME,
            "total_chunks": len(signature["chunk_ids"]),
            "embedded_chunks": len(completed),
            "embedded_chunk_ids": [signature["chunk_ids"][i] for i in completed],
            "embedded_chunk_fingerprints": [signature["chunk_fingerprints"][i] for i in completed],
            "batch_size": BATCH_SIZE,
            "max_seq_length": MAX_SEQ_LENGTH,
            "embedding_dimension": dimension,
            "embedding_config_version": CONFIG_VERSION,
        },
    )


def save_partial_metadata(signature: dict, completed: list[int], dimension: int) -> None:
    atomic_json(
        PARTIAL_METADATA_FILE,
        {
            **signature,
            "completed_indices": sorted(set(completed)),
            "embedded_chunks": len(set(completed)),
        },
    )
    write_progress(signature, sorted(set(completed)), dimension)


def apply_pending_batch(matrix: np.ndarray, signature: dict, completed: list[int], dimension: int) -> list[int]:
    """Idempotently apply a durable pending batch, then commit its manifest."""
    with np.load(PENDING_BATCH_FILE, allow_pickle=False) as pending:
        indices = pending["indices"].astype(np.int64).tolist()
        ids = pending["chunk_ids"].tolist()
        hashes = pending["chunk_fingerprints"].tolist()
        vectors = pending["vectors"].astype(np.float32)
    for row, index in enumerate(indices):
        if not 0 <= index < len(signature["chunk_ids"]):
            raise ValueError("Pending checkpoint refers to an invalid corpus index.")
        if ids[row] != signature["chunk_ids"][index] or hashes[row] != signature["chunk_fingerprints"][index]:
            raise ValueError("Pending checkpoint does not match the current corpus; refusing reuse.")
    if vectors.shape != (len(indices), dimension) or not np.isfinite(vectors).all():
        raise ValueError("Pending checkpoint has invalid embedding vectors.")
    matrix[indices] = vectors
    matrix.flush()
    completed = sorted(set(completed).union(indices))
    save_partial_metadata(signature, completed, dimension)
    PENDING_BATCH_FILE.unlink(missing_ok=True)
    return completed


def create_partial_matrix(
    chunks: list[dict[str, Any]], signature: dict, dimension: int, cached_vectors: np.ndarray | None,
    cached_metadata: dict | None,
) -> tuple[np.memmap, list[int]]:
    """Create current-order working storage, reusing vectors for identical text."""
    count = len(chunks)
    fd, temp_name = tempfile.mkstemp(prefix=".embeddings.partial.", suffix=".npy", dir=CACHE_DIR)
    os.close(fd)
    try:
        matrix = np.lib.format.open_memmap(temp_name, mode="w+", dtype=np.float32, shape=(count, dimension))
        matrix[:] = 0
        completed: list[int] = []
        reusable = defaultdict(deque)
        if cached_vectors is not None and cached_metadata is not None:
            # IDs may be deliberately revised while chunk text and embedding
            # configuration stay unchanged. Text fingerprints let us carry
            # those vectors forward. A deque preserves multiplicity when the
            # corpus contains repeated identical text.
            old_limit = cached_metadata.get("max_seq_length", signature["max_seq_length"])
            current_limit = signature["max_seq_length"]
            if old_limit == current_limit:
                for row, text_hash in enumerate(cached_metadata["embedded_chunk_fingerprints"]):
                    reusable[text_hash].append(row)
            else:
                safe_counts = defaultdict(int)
                safe_limit = min(old_limit, current_limit)
                for chunk in chunks:
                    token_count = chunk.get("token_count")
                    if isinstance(token_count, int) and token_count <= safe_limit:
                        safe_counts[fingerprint(chunk["text"])] += 1
                for row, text_hash in enumerate(cached_metadata["embedded_chunk_fingerprints"]):
                    if safe_counts[text_hash] > 0:
                        reusable[text_hash].append(row)
                        safe_counts[text_hash] -= 1
        for index, text_hash in enumerate(signature["chunk_fingerprints"]):
            candidates = reusable[text_hash]
            if candidates:
                matrix[index] = cached_vectors[candidates.popleft()]
                completed.append(index)
        matrix.flush()
        del matrix
        os.replace(temp_name, PARTIAL_EMBEDDINGS_FILE)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    matrix = np.load(PARTIAL_EMBEDDINGS_FILE, mmap_mode="r+")
    save_partial_metadata(signature, completed, dimension)
    return matrix, completed


def encode_corpus(
    chunks: list[dict[str, Any]],
    encoder: Callable[[list[str]], np.ndarray],
    dimension: int,
) -> np.ndarray:
    """Resume unchanged records and checkpoint newly encoded batches safely."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    signature = cache_signature(chunks, dimension)
    config = embedding_config(dimension)

    # A partial manifest is authoritative after a run has begun. Its corpus
    # snapshot must match exactly; final caches can be safely rebased below.
    partial_matrix = None
    completed: list[int] = []
    if PARTIAL_METADATA_FILE.exists() and PARTIAL_EMBEDDINGS_FILE.exists():
        partial_metadata = json.loads(PARTIAL_METADATA_FILE.read_text(encoding="utf-8"))
        if {key: partial_metadata.get(key) for key in config} != config:
            raise ValueError("Partial cache model/configuration mismatch; refusing unsafe resume.")
        partial_matrix = np.load(PARTIAL_EMBEDDINGS_FILE, mmap_mode="r+")
        old_ids = partial_metadata.get("chunk_ids", [])
        old_fingerprints = partial_metadata.get("chunk_fingerprints", [])
        if partial_matrix.shape != (len(old_ids), dimension) or len(old_ids) != len(old_fingerprints):
            raise ValueError("Partial embedding matrix has an unexpected shape.")
        completed = sorted(set(partial_metadata.get("completed_indices", [])))
        if any(not isinstance(i, int) or i < 0 or i >= len(old_ids) for i in completed):
            raise ValueError("Partial checkpoint contains invalid completed indices.")
        if PENDING_BATCH_FILE.exists():
            old_signature = {**partial_metadata, "chunk_ids": old_ids, "chunk_fingerprints": old_fingerprints}
            completed = apply_pending_batch(partial_matrix, old_signature, completed, dimension)

        corpus_changed = old_ids != signature["chunk_ids"] or old_fingerprints != signature["chunk_fingerprints"]
        if corpus_changed:
            old_vectors = np.asarray(partial_matrix[completed], dtype=np.float32)
            old_cache_metadata = {
                **config,
                "embedded_chunk_ids": [old_ids[i] for i in completed],
                "embedded_chunk_fingerprints": [old_fingerprints[i] for i in completed],
            }
            del partial_matrix
            partial_matrix, completed = create_partial_matrix(
                chunks, signature, dimension, old_vectors, old_cache_metadata
            )
            print("Corpus changed during the interrupted run; rebased exact ID/text matches safely.")
        print(f"Found existing {MODEL_NAME} partial cache.")
    else:
        # An incomplete pair can only be an interrupted initialization. Rebuild
        # from the last committed final cache rather than trusting loose files.
        PARTIAL_EMBEDDINGS_FILE.unlink(missing_ok=True)
        PARTIAL_METADATA_FILE.unlink(missing_ok=True)
        PENDING_BATCH_FILE.unlink(missing_ok=True)

        cached_vectors = None
        cached_metadata = None
        if EMBEDDINGS_FILE.exists() or METADATA_FILE.exists():
            if not (EMBEDDINGS_FILE.exists() and METADATA_FILE.exists()):
                raise ValueError("Final embedding cache is incomplete; refusing unsafe reuse.")
            cached_metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
            cached_vectors = np.load(EMBEDDINGS_FILE, allow_pickle=False)
            validate_cached_matrix(cached_metadata, cached_vectors, dimension)

            # Exact complete cache: no migration or re-encoding is necessary.
            cached_config = {key: cached_metadata.get(key) for key in config}
            if (cached_metadata.get("chunk_ids") == signature["chunk_ids"]
                    and cached_metadata.get("chunk_fingerprints") == signature["chunk_fingerprints"]
                    and cached_config == config
                    and len(cached_metadata["embedded_chunk_ids"]) == len(chunks)):
                print(f"Found complete {MODEL_NAME} embedding cache: {len(chunks)} / {len(chunks)} chunks.")
                write_progress(signature, list(range(len(chunks))), dimension)
                return cached_vectors.astype(np.float32, copy=False)

            print(f"Found existing {MODEL_NAME} embedding cache with {len(cached_vectors)} vectors.")

        partial_matrix, completed = create_partial_matrix(
            chunks, signature, dimension, cached_vectors, cached_metadata
        )
        if completed:
            print(f"Reused {len(completed)} exact chunk ID/text matches in the current corpus.")

    missing = [i for i in range(len(chunks)) if i not in set(completed)]
    print(f"{len(completed)} / {len(chunks)} chunks already embedded.")
    if missing:
        print(f"Encoding {len(missing)} new or changed chunks in batches of {BATCH_SIZE}...")

    for offset in range(0, len(missing), BATCH_SIZE):
        indices = missing[offset : offset + BATCH_SIZE]
        batch = [chunks[i] for i in indices]
        vectors = np.asarray(encoder([chunk["text"] for chunk in batch]), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape != (len(batch), dimension):
            raise ValueError(f"Encoder returned shape {vectors.shape}; expected {(len(batch), dimension)}")
        if not np.isfinite(vectors).all():
            raise ValueError("Encoder returned non-finite values; batch was not checkpointed.")

        fd, temp_name = tempfile.mkstemp(prefix=".pending_batch.", suffix=".npz", dir=CACHE_DIR)
        try:
            with os.fdopen(fd, "wb") as f:
                np.savez(
                    f,
                    indices=np.asarray(indices, dtype=np.int64),
                    chunk_ids=np.asarray([signature["chunk_ids"][i] for i in indices]),
                    chunk_fingerprints=np.asarray([signature["chunk_fingerprints"][i] for i in indices]),
                    vectors=vectors,
                )
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_name, PENDING_BATCH_FILE)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        completed = apply_pending_batch(partial_matrix, signature, completed, dimension)
        print(f"Checkpoint saved: {len(completed)} / {len(chunks)} chunks.")

    if len(completed) != len(chunks):
        raise RuntimeError(f"Only {len(completed)} of {len(chunks)} chunks were embedded.")

    del partial_matrix
    final_vectors = np.load(PARTIAL_EMBEDDINGS_FILE, mmap_mode="r")
    atomic_npy(EMBEDDINGS_FILE, np.asarray(final_vectors, dtype=np.float32))
    del final_vectors
    final_metadata = {
        **signature,
        **config,
        "embedded_chunk_ids": signature["chunk_ids"],
        "embedded_chunk_fingerprints": signature["chunk_fingerprints"],
        "embedded_chunks": len(chunks),
    }
    atomic_json(METADATA_FILE, final_metadata)
    write_progress(signature, list(range(len(chunks))), dimension)
    PARTIAL_EMBEDDINGS_FILE.unlink(missing_ok=True)
    PARTIAL_METADATA_FILE.unlink(missing_ok=True)
    PENDING_BATCH_FILE.unlink(missing_ok=True)
    return np.load(EMBEDDINGS_FILE, allow_pickle=False)


def load_model():
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import BitsAndBytesConfig

    quantization = dict(MODEL_KWARGS["quantization_config"])
    quantization["bnb_4bit_compute_dtype"] = torch.float16
    model_kwargs = {
        "quantization_config": BitsAndBytesConfig(**quantization),
        "torch_dtype": torch.float16,
    }
    model = SentenceTransformer(
        MODEL_NAME,
        model_kwargs=model_kwargs,
        device="cuda",
        tokenizer_kwargs={"padding_side": "left"},
    )
    model.max_seq_length = MAX_SEQ_LENGTH
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not found; this Qwen quantization configuration requires CUDA.")
    dimension = model.get_sentence_embedding_dimension()
    if not dimension:
        raise ValueError("Could not determine embedding dimension from the loaded model.")
    return model, dimension


def main() -> None:
    global BATCH_SIZE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=("arxiv", "peerqa"), default="arxiv")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Chunks per checkpoint batch")
    args = parser.parse_args()
    configure_corpus(args.corpus)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    BATCH_SIZE = args.batch_size

    chunks = load_chunks()
    print(f"Loaded {len(chunks):,} chunks from {CHUNKS_DIR}.")
    model, dimension = load_model()
    embeddings = encode_corpus(
        chunks,
        lambda texts: model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dimension,
    )
    print(f"Complete: saved {embeddings.shape[0]:,} x {embeddings.shape[1]} embeddings to {EMBEDDINGS_FILE}.")


if __name__ == "__main__":
    main()
