# Repository agent instructions

## Project objective

This repository is a research codebase for studying how corpus and query characteristics affect RAG architecture performance. Its objective is to identify when architectural choices produce measurable benefits and what quality, latency, storage, and compute trade-offs they introduce. Treat this as a controlled comparison project, not merely a demonstration that RAG works.

## Research principles

- Compare retrieval quality and evidence relevance. Measure answer quality and faithfulness only where an answer-generation evaluation actually exists.
- Consider latency, index/storage overhead, embedding and retrieval compute, robustness, generalization, corpus structure, and query requirements.
- The implemented arXiv pipeline provides acquisition, extraction, chunking, embeddings, Qdrant indexing, and dense/BM25/hybrid retrieval. The repository does not currently contain a complete answer-generation or answer-faithfulness evaluation path.
- Keep measured results, historical reports, hypotheses, and estimates clearly distinguished. Preserve fixed datasets, configurations, and baselines for fair comparisons.
- Qdrant quantization and automated storage snapshots are planned experiments, not current capabilities. The current uploader stores dense vectors as float32. For quantization comparisons, preserve an unquantized baseline, record storage from the actual Qdrant collection, and keep corpus, queries, retrieval settings, K, and required reranking fixed while comparing retrieval metrics and latency.

## Repository architecture

- `download_corpus/download_pdfs.py`: arXiv paper list, PDF download/retries, title checks, pypdf extraction, and per-paper extracted-text JSON/metadata.
- `preprocessing/chunking.py`: arXiv GROBID TEI or extracted-text input, flat section/block construction, tokenizer-aware chunking, and chunk JSON output. GROBID is needed only for uncached TEI in GROBID mode.
- `preprocessing/embeddings.py`: resumable Qwen3-Embedding-8B embedding cache for arXiv chunks.
- `retrieval/qdrant_setup.py`: embedding-cache validation, Qdrant schema, chunk payload upload, retries, and upload-count verification.
- `retrieval/search.py`: dense, Qdrant text/BM25, and hybrid retrieval; every mode cross-encoder reranks the retrieved candidates. `colbert_encode` is a helper, not a complete indexing/search mode.
- `scripts/ingest_corpus.py`: orchestrates arXiv acquisition, chunking, embedding, and indexing. `scripts/query.py` runs query-time retrieval; `scripts/corpus_status.py` prints local acquisition/chunk counts.
- `preprocessing/diagnostics.py`: audits cached arXiv TEI/chunks and benchmark-tokenizer lengths.
- `evals/embeddings/compare_embeddings.py` and `_v2.py`: separate model-comparison experiments. Saved artifacts/reports include historical results and must not be presented as current reruns without verification.
- `tests/test_retrieval_pipeline.py`: unit coverage for the embedding cache and mocked retrieval query construction. It does not validate real arXiv acquisition, GROBID, GPU embedding, or Qdrant service behavior.

Typical data flow:

```text
download_corpus/download_pdfs.py -> PDFs + extracted-text JSON
  -> preprocessing/chunking.py -> chunks JSON
  -> preprocessing/embeddings.py -> embedding cache
  -> retrieval/qdrant_setup.py -> Qdrant collection
  -> retrieval/search.py -> top-K hits
```

## Data integrity and provenance

- Preserve arXiv IDs as document IDs and retain section/chunk metadata through chunk files, embedding-cache alignment, Qdrant payloads, and retrieval results.
- Do not silently change document IDs, chunk IDs, cache signatures, collection payloads, or corpus-version computation. Explain migrations and effects on cached embeddings and indexes.
- Treat source PDFs and extracted-text records as inputs. Write generated artifacts to their established output paths; do not edit source data in place.
- `preprocessing/chunking.py` does not currently retain source sentence IDs or source offsets in its arXiv chunks. Do not claim sentence-level provenance that the output does not contain.

## Experimental integrity

- Make dataset versions, chunking parameters, tokenizer/embedding models, retrieval parameters, and evaluation conditions explicit and reproducible.
- Keep datasets, query sets, and evidence labels consistent across compared systems. Avoid changing multiple experimental variables at once without justification.
- Preserve baseline implementations and distinguish the implemented retrieval modes from planned architectures in `README.MD`.
- Report measured results separately from hypotheses, estimates, and historical artifacts. Never fabricate benchmark results or claim tests/integration runs that were not performed.

## Coding rules

- Inspect before modifying; trace callers and data flow before changing interfaces.
- Prefer extending existing abstractions over duplicating implementations, but avoid premature abstractions.
- Keep changes scoped. Do not reorganize large modules solely for aesthetics or remove downstream functionality because it is currently unused.
- Preserve backward compatibility unless a deliberate migration is justified and documented.
- Add focused tests for behavior changes and update documentation when architecture or interfaces change.

## Testing and external validation

- The inspected suite uses `unittest` in `tests/test_retrieval_pipeline.py`. From an activated project environment, run `python -m unittest discover -s tests` from the repository root. Do not claim it passed unless it is run.
- Tests use temporary files and mocks for embedding-cache and retrieval behavior. They are not validation against live arXiv/GROBID, GPU model downloads, Qdrant Cloud, or an end-to-end benchmark.
- Real arXiv acquisition needs network access. GROBID mode needs a reachable service if TEI is uncached or overwritten. Embedding generation uses CUDA and the configured Hugging Face model. Qdrant upload/query needs valid `QDRANT_URL` and `QDRANT_API_KEY` settings.
- If a test or integration stage cannot be run, say which one and why. Do not call mocked tests a live integration test.

## Agent communication

At completion, report files inspected and modified, observed behavior, architectural choices and rationale, tests actually executed and results, known limitations, and assumptions or unresolved questions. State explicitly when source data was left untouched.
