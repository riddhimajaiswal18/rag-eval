---
name: rag-corpus-engineering
description: Inspect, extend, or evaluate this repository's arXiv corpus acquisition, preprocessing, chunking, provenance, embeddings, Qdrant indexing, retrieval, and evaluation pipeline.
---

# RAG corpus engineering in RagAndEval

Load this skill for work involving the arXiv corpus, PDF processing, extraction and normalization, document construction, chunking, provenance, embeddings, vector indexing, retrieval, or RAG architecture evaluation in this repository. Read root `AGENTS.md` first for project-wide research and integrity rules.

## 1. Inspect before acting

1. Trace the requested entry point and its callers. `scripts/ingest_corpus.py` shows the current end-to-end arXiv ingestion flow.
2. Check the existing output schema and ID generation before changing transformations. Chunk IDs affect embedding reuse and Qdrant corpus versions.
3. Inspect `tests/test_retrieval_pipeline.py` for covered contracts. Mocked tests do not establish behavior against real arXiv, GROBID, GPU models, or Qdrant.
4. Preserve source PDFs and extracted-text JSON as inputs. Generated artifacts belong in the established data directories.

## 2. End-to-end arXiv flow

### Acquisition and text extraction

- **Input:** `download_corpus/download_pdfs.py` contains a fixed `PAPERS` list of `(arxiv_id, category, expected_title)` tuples.
- **Work:** `fetch_titles` requests arXiv metadata. `download_paper` streams each PDF to a temporary file, retries transient failures, checks the `%PDF-` prefix and content length where available, and reuses a valid existing PDF. `ingest_papers` checks titles when metadata is available, extracts pages with pypdf through `extract_text_from_pdf`, and writes per-paper JSON through `save_extracted_text`. `write_metadata` rebuilds aggregate metadata from saved text records.
- **Output:** `download_corpus/rag_corpus/pdfs/`, `download_corpus/rag_corpus/extracted_text/<id>.json`, `metadata.json`, and the downloader log. Text JSON includes `document_id`/`arxiv_id`, title, category, timestamp, text, and page/word/token estimates. Extracted text includes page markers; `load_extracted_text` removes them before chunking.
- **Failure/resume:** Invalid or missing PDFs, exhausted download retries, and empty text extraction are handled as failures. Title mismatches are logged but are not fatal. Ctrl+C is caught and the same command can be rerun to reuse completed files; failures are not persisted in a separate retry queue.

### Document construction and chunking

- **Inputs:** `preprocessing/chunking.py` defaults to PDFs plus metadata and GROBID at `http://localhost:8070`. `--source extracted-text` instead loads the downloader's JSON and wraps it in a single `Full text` section. GROBID mode caches TEI in `preprocessing/data/grobid_tei/`.
- **GROBID structure:** `extract_flat_sections` emits an abstract and flat top-level body sections; references/bibliographies are skipped. `direct_body_blocks` walks nested TEI divisions in document order. Tables retain caption/context/header/row structure while chunking; figures retain captions/descriptions. This is not a hierarchical parent-child representation; `HIERARCHY_ENABLED` is false.
- **Boundaries:** `split_sentences` uses a regex after `.`, `!`, or `?` before uppercase letters/digits; it is not a language-aware sentence parser. `TokenBudget` counts with all tokenizers in `TOKENIZER_MODELS`, including special tokens, and enforces the maximum count. Defaults are `CHUNK_SIZE_TOKENS=1900`, `OVERLAP_TOKENS=100`, and `MAX_SEQ_LEN=2000`. Chunk packing is section-local. Later chunks reserve up to 100 tokens for sentence overlap; tables may split by row while repeating context; oversized text splits at whitespace.
- **Output:** `build_chunk_document` writes `preprocessing/data/chunks/<id>_chunks.json`. Chunk IDs are `<document_id>::s<section_index>::<slug>::c<index>`. Records contain section title/index, chunk index, text, token count, and overlap metadata/configuration. They do not retain source sentence IDs or source offsets.
- **Failure:** Missing/invalid input, unreachable GROBID when uncached TEI is required, malformed TEI, or no usable sections cause a per-paper failure. `main` reports failures and exits nonzero when any selected paper fails.

### Embedding generation

- `preprocessing/embeddings.py::load_chunks` reads sorted chunk JSON files and yields ordered chunk ID, text, and token-count records.
- `load_model` loads `Qwen/Qwen3-Embedding-8B` with 4-bit NF4/float16 settings, normalized embeddings, CUDA, and a 2000-token limit. The default checkpoint batch size is 2; model dimension is obtained from the model.
- `encode_corpus` fingerprints text and validates model/configuration/dimensions before cache reuse. It can reuse unchanged text, writes partial matrices and metadata atomically, checkpoints pending batches, and resumes an interrupted run. Cache metadata must remain aligned to the exact ordered chunk list.
- Outputs are under `preprocessing/data/embeddings/qwen3-8B-4bit/`. Do not alter chunk ordering, IDs, or cache signature without considering re-embedding and migration behavior.

### Qdrant indexing

- `retrieval/qdrant_setup.py::load_chunks` reloads chunk files in deterministic order, validates text/IDs, and computes fingerprints. `load_and_validate_cache` ensures embedding rows match those chunk IDs and fingerprints and rejects wrong dimensions, incomplete metadata, or non-finite vectors.
- `ensure_collection` creates/validates a named dense vector collection and indexes `corpus_version`, `document_id`, and text. The text index is used by the Qdrant text/BM25 query path.
- `upload_chunks` creates deterministic UUID5 point IDs from `chunk_id`, uploads dense vectors and chunk payloads, retries transient failures, then verifies exact point count for the current version. There is no explicit cleanup of older corpus versions.
- Configuration is read from `.env`; do not expose or commit credentials. The example settings are in `.env.example`.

### Retrieval and evaluation

- `retrieval/search.py::Retriever.search` supports dense, BM25 text search, and hybrid dense+text reciprocal-rank fusion. Results are limited by `limit` (top-K) and filtered by corpus version; an optional document ID adds a document filter. Dense/hybrid queries encode with Qwen. The CLI optionally reranks returned hits with a cross-encoder.
- `rerank` is a real second-stage cross-encoder implementation. `colbert_encode` is only an encoding helper; no ColBERT index/search flow is wired into Qdrant or `Retriever.search`.
- The repository does not currently provide an arXiv answer generator or an end-to-end answer-faithfulness evaluator. `evals/embeddings/compare_embeddings.py` and `_v2.py` are model comparison scripts; `evals/reports/embedding_model_selection.txt` says its saved rankings predate current section-qualified chunk IDs and should be treated as historical pending rerun.

## 3. Provenance contract

Keep `document_id` (arXiv ID), section title/index, chunk ID/index, and text aligned through chunk JSON, the embedding matrix, Qdrant payload, and returned hits. The current arXiv output does not identify exact source sentences or offsets, so do not claim a retrieved chunk has sentence-level traceability.

When changing chunking:

- Record the strategy, tokenizer(s), size unit, limits, section-boundary policy, overlap, and ID scheme.
- Ensure overlap text remains part of each chunk's text and record overlap metadata accurately.
- Keep embedding order and cache IDs/fingerprints synchronized with chunk files.
- Preserve the existing baseline and make experimental strategies selectable/configured rather than silently changing its behavior.
- If sentence-level evidence evaluation is added, first add a stable mapping from chunks back to source sentences/locations; do not infer that mapping from chunk IDs.

## 4. Safe workflows

### Change acquisition

1. Trace `PAPERS`, `download_paper`, `ingest_papers`, and callers in `scripts/ingest_corpus.py` and `start.ps1`.
2. Preserve retry, temporary-file, PDF validation, resume, title-check, and extraction semantics unless the request changes them.
3. Add focused unit coverage for changed behavior; describe network/arXiv integration checks separately.

### Add or change a chunking strategy

1. Document tokenizer, size unit, section constraints, overlap, sentence/table/figure treatment, and IDs.
2. Preserve current chunking as a baseline and do not invalidate existing caches silently.
3. Check empty, oversized, malformed, and boundary cases and keep chunk output consumable by `embeddings.py` and `qdrant_setup.py`.
4. Record the configuration so experiments are reproducible.

### Change or add an embedding model

1. Set model, dimension, max sequence length, normalization, prompts, and dtype/quantization explicitly.
2. Update cache signatures and compatibility checks so incompatible vectors are rejected and safe exact reuse is possible.
3. Preserve partial-batch crash recovery and verify row order/fingerprints remain aligned.
4. Compare on the same fixed chunks and query set as the baseline.

### Change indexing or retrieval

1. Trace payload fields, named vectors, filters, and version generation across `qdrant_setup.py` and `search.py`.
2. Keep version and document filters consistent through all candidate-generation stages.
3. Add focused mocked tests for query construction and filtering; use live Qdrant integration only when service credentials are available.
4. Report retrieval mode, top-K, corpus version, latency, and candidate budget when measured.

### Add evaluation

1. Define the evidence/relevance labels and eligibility rules before implementing metrics.
2. Add deterministic metric tests for relevant, irrelevant, and empty cases.
3. Keep retrieval metrics separate from generated-answer quality and faithfulness; do not claim unimplemented metrics.
4. Report skipped examples, dataset version, query filters, and limits.

## 5. Validation and external dependencies

The current unittest suite is `tests/test_retrieval_pipeline.py`. From an activated project environment, run `python -m unittest discover -s tests` from the repository root. This skill does not claim that the suite has been run. Existing tests are local/mock-based and do not validate real arXiv downloads, GROBID, GPU embeddings, live Qdrant, or full evaluation.

Network is required for arXiv acquisition/model downloads. GROBID mode needs a reachable server when TEI must be generated. Embedding generation requires CUDA under the configured 4-bit model path. Qdrant indexing/query requires valid `.env` credentials and service access. State explicitly which integration stages were not run.

## 6. Confirmed limitations

| Location | Current behavior | Why it matters |
| --- | --- | --- |
| `preprocessing/chunking.py::build_chunk_document` | Chunk records do not contain sentence IDs or source offsets. | Exact sentence-level evidence attribution cannot be derived from current arXiv chunks alone. |
| `retrieval/qdrant_setup.py::upload_chunks` | Uploads dense vectors; collection setup includes a sparse-vector slot but does not upload sparse vectors. | BM25 is provided via Qdrant's text index/query API; the sparse-vector slot is not a sparse embedding implementation. |
| `retrieval/qdrant_setup.py::upload_chunks` | Verifies the active corpus version but does not delete old versions. | Old points may remain in the collection; active search filters by version. |
| `retrieval/search.py::colbert_encode` | Helper is not connected to collection indexing or `Retriever.search`. | Do not present ColBERT as an available end-to-end strategy. |
| `start.ps1` | The final console message may claim query-time retrieval is not implemented even though `retrieval/search.py` provides it. | Keep user-facing status text aligned with the actual entry points. |

These are code-inspection findings; do not infer their measured impact without running a suitable experiment.

## 7. Experimental design

- Freeze and record the paper IDs, chunk artifacts/configuration, query set, embedding model/configuration, index version, retrieval mode, filters, and top-K.
- Preserve a baseline and vary one main factor at a time; use ablations to isolate chunking, embedding, retrieval, and reranking effects.
- Report retrieval evidence results separately from answer-generation quality. Measure latency, index size, embedding/indexing time, and compute only when instrumented.
- Treat saved reports as historical unless their dataset, chunk IDs, and configuration match the current artifacts. Explain uncertainty and limits on generalization from a single corpus.
