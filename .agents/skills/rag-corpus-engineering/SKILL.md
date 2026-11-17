---
name: rag-corpus-engineering
description: Inspect, extend, or evaluate this repository's arXiv and PeerQA corpus pipelines, provenance, chunking, embeddings, Qdrant indexing, retrieval, and evidence evaluation.
---

# RAG corpus engineering in RagAndEval

Load this skill for work involving corpus acquisition, PDF processing, extraction/normalization, document construction, chunking, provenance, embeddings, vector indexing, retrieval, dataset integration, or RAG architecture evaluation in this repository. Read the repository-root `AGENTS.md` first for project-wide integrity and collaboration rules.

## 1. Inspect before acting

1. Read the relevant code and its callers. `scripts/ingest_corpus.py` is the clearest top-level workflow map; trace actual calls into the implementation before inferring behavior from comments or names.
2. Identify which corpus is selected. Keep arXiv and PeerQA data paths and Qdrant collections isolated.
3. Check current output schema and ID generation before changing any transform. A change to IDs affects embedding cache matching and Qdrant versioning.
4. Inspect `tests/test_peerqa_pipeline.py` for covered contracts. Its fixtures/mocks do not prove behavior against the real PeerQA archive or external services.
5. Keep raw source records immutable. Generated data goes to established corpus-specific output paths.

## 2. Actual end-to-end flow

### arXiv acquisition and extraction

- **Input:** `download_corpus/download_pdfs.py` contains a fixed `PAPERS` list of `(arxiv_id, category, expected_title)` tuples. There is no manifest-driven arXiv catalogue in this script.
- **Work:** `fetch_titles` requests arXiv metadata; `download_paper` streams a PDF to a temporary file, retries transient failures, checks the `%PDF-` prefix and content length where available, and reuses an already valid PDF. `ingest_papers` compares titles when the metadata request succeeds, extracts pages with `extract_text_from_pdf` using pypdf, and writes extracted text/metadata with `save_extracted_text`. `write_metadata` rebuilds aggregate metadata from text JSON files. Ctrl+C is caught and existing files are reused on rerun; failures are not written to a retry queue.
- **Output:** `download_corpus/rag_corpus/pdfs/`, `download_corpus/rag_corpus/extracted_text/<id>.json`, `metadata.json`, and the downloader log. Text records carry `document_id`/`arxiv_id`, title, category, timestamp, text, and page/word/token estimates. Page markers are included in the extracted text and later removed by the extracted-text chunking input loader.
- **Failure/validation:** bad/missing PDFs, exhausted download retries, title mismatches (logged, not fatal), and empty extraction are handled separately. A PDF can be structurally valid by signature without its actual paper identity being verified; title comparison is the identity cross-check when available.

### arXiv document construction and chunking

- **Input:** `preprocessing/chunking.py` defaults to PDFs plus metadata and GROBID at `http://localhost:8070`. With `--source extracted-text`, it reads downloader JSON instead. `process_paper` creates a single `Full text` section for extracted-text input. In GROBID mode it reads/caches TEI, extracts title and abstract and flattens top-level body sections; references/bibliographies are skipped. Nested TEI divisions are traversed into the containing section's ordered blocks. Tables are represented as caption/context/headers/rows and figures as captions/descriptions or context blocks.
- **Chunking:** `split_sentences` uses a punctuation-and-uppercase/digit regex, not a linguistic sentence parser. `TokenBudget` measures text with each of the three `TOKENIZER_MODELS` tokenizers (including special tokens) and enforces the largest count. Defaults: `CHUNK_SIZE_TOKENS=1900`, `OVERLAP_TOKENS=100`, and embedding `MAX_SEQ_LEN=2000`. Atoms are packed within each section; later chunks reserve 100 tokens for new content and get up to 100 tokens of prior sentence overlap. Tables split by rows while repeating table context; oversized sentences/blocks split at whitespace. Figure descriptions may be combined with the next ordinary text block.
- **Output:** `preprocessing/data/chunks/<id>_chunks.json`. `build_chunk_document` creates IDs as `<document_id>::s<section_index>::<slug>::c<index>`, and stores section title/index, chunk index, text, token count, and prior-overlap count plus chunk configuration. The resulting chunk records do not contain source sentence IDs or offsets. Hierarchical chunking is disabled (`HIERARCHY_ENABLED=False`).
- **Failure:** invalid/missing extracted text, missing PDFs, unavailable GROBID when uncached TEI is required, malformed TEI, or no usable sections fail the paper; `main` reports per-paper failures and exits nonzero if any fail. Existing TEI may be reused unless overwritten.

### PeerQA acquisition and normalization

- **Input:** `peerqa/corpus.py` expects official `papers.jsonl` and labelled `qa.jsonl` in the acquired v1.0 archive. The code validates for 90 distinct papers (82 `nlpeer`, 8 `egu`) and 579 labelled QA rows by default. The source-row schema it consumes is `paper_id`, integer `idx`, optional integer/null `pidx` and `sidx`, `content`, `type`, and `last_heading`. QA evidence is read from `answer_evidence_mapped`; the question record also retains raw/annotated evidence, free-form answer, `answerable`, and `answerable_mapped` fields when present. These are code expectations; fixture tests are not proof of all real-release edge cases.
- **Acquisition:** `download_archive` tries two official URLs, retries selected status/network failures, checks content length if supplied, ZIP CRC, and presence of required JSONL files, then records SHA-256. `extract_official_files` only writes safe basenames for papers/QA and optional unlabeled QA JSONL; rejects traversal and duplicates; records checksums. `acquire_and_build` reuses already extracted official files unless forced download is requested, then invokes `build_corpus` and licensed PDF acquisition.
- **PDFs:** `_official_egu_pdf_source` derives a deterministic Copernicus URL from EGU IDs. Other sources require `download_corpus/peerqa/pdf_sources.json` entries with included original paper ID, HTTPS URL, explicit `license_allows_download: true`, and license note. `download_permitted_pdfs` ignores IDs outside the 90-paper manifest, retries selected failures, validates content length if supplied and `%PDF-`, and stores files by SHA-256 of paper ID. `acquire_and_build` adds PDF hash, URL, license/status fields to the manifest. Existing PDFs are reused based on size and signature. The PDFs are not parsed to replace PeerQA's official source text and there is no PDF-to-source-text version/alignment verification in this flow.
- **Output and failures:** source files live below `download_corpus/peerqa/source/`; documents, chunks, questions, evidence maps, manifest, and status live below `download_corpus/peerqa/`. `_iter_jsonl` rejects invalid JSON and non-object records. `_normalize_source_record` rejects missing required fields, blank paper IDs, noninteger indices, and duplicate `idx` values per paper; it coerces non-string content to a string (null becomes empty). `build_corpus` rejects unexpected default corpus counts/composition and duplicate/missing question IDs. Acquisition failure is recorded in `acquisition_status.json`; per-paper PDF statuses are reflected in the manifest.

### PeerQA document, evidence, and chunk construction

- `_normalize_source_record` standardizes row fields without changing official `idx`, `pidx`, or `sidx`.
- `canonical_document_id` returns `peerqa:` plus SHA-256 of the original `paper_id`; `original_paper_id` remains a separate field.
- `_row_sentence_id` returns `<document_id>:i<idx>:p<pidx-or-na>:s<sidx-or-na>`. `_build_document` sorts by `idx`, creates an `idx_map` from **every** normalized row, and retains `source_location={idx,pidx,sidx}` and source type/text. It groups records into paragraphs using `pidx` (falling back to `idx` when null), groups consecutive paragraphs by heading into sections, and excludes empty content plus `title` and `heading` rows from paragraph records. The document records extraction method `peerqa_official_papers_jsonl`.
- `_section_id` hashes document ID, heading, and first paragraph index. Paragraph IDs are `<document_id>:p<pidx>`. Repeated heading segments after another section get a separate section ID.
- `map_question_evidence` accepts QA records plus a mapping from original paper ID to `(document, idx_map)`. For every `answer_evidence_mapped[].idx`, it maps the official integer index to the row's sentence ID and pidx/sidx or retains an explicit unmapped status (outside subset, absent index, invalid record, or missing row). It de-duplicates gold sentence IDs while preserving order and returns a question record and an evidence-map record. It does not use chunk IDs or boundaries.
- `_chunks_for_document` walks section then paragraph records. Default is 900 whitespace-delimited words and zero overlap; an oversized source row is split at whitespace and every fragment keeps the original sentence ID/source ref. Chunk IDs use `<document_id>::<paragraph_id>::c<part>`. Payload carries `sentence_ids`, `source_refs`, source text/location, section/paragraph IDs, and original paper ID.
- `build_corpus` first builds all documents and index maps, creates chunks, then maps each QA, writes JSONL question/evidence files, and writes counts/checksums. Evidence mapping is source-index based and remains meaningful when the chunking strategy changes.

**Important edge case confirmed by code:** `idx_map` includes all source rows, but chunk document records exclude empty/title/heading rows. Thus an official evidence index can map to a source row whose sentence ID is not present in any current chunk. Keep that distinction visible; do not silently claim every mapped gold ID is retrievable. The current fixture tests do not establish how often this occurs in the full release.

### Embeddings, Qdrant, retrieval, evaluation

- `preprocessing/embeddings.py` uses `configure_corpus` to retain the legacy arXiv cache path or select PeerQA's isolated chunk/cache path. `load_chunks` sorts JSON files and extracts only chunk ID, text, and optional token count. `encode_corpus` uses Qwen3-Embedding-8B with normalized embeddings, 4-bit NF4/float16 configuration, CUDA, and default batch size 2. It fingerprints text, uses atomic cache writes and a pending-batch checkpoint, reuses exact text matches, and validates model/configuration/dimensions before reuse. The cache preserves chunk order and IDs in metadata.
- `retrieval/qdrant_setup.py` re-loads chunk JSON with the retrieval payload fields and verifies order, IDs, text fingerprints, cache completeness, and finite vectors. `ensure_collection` configures named dense vectors and a sparse-vector slot plus keyword/text payload indexes. `upload_chunks` assigns deterministic UUID5 point IDs from chunk IDs, uploads dense vectors with chunk metadata and `corpus_version`, retries transient network/server errors, and checks exact point count for that version. It does not generate/upload sparse vectors. Uploading a new version does not explicitly delete old-version points; query filters select one version. PeerQA index status is written after verification/upload.
- `retrieval/search.py` provides dense Qwen query embedding, Qdrant text/BM25 queries, and hybrid dense+text reciprocal-rank fusion, with corpus-version and optional document-ID filters. CLI `--rerank` applies a cross-encoder after retrieval. `colbert_encode` exists but is not connected to the collection upload or `Retriever.search`; do not describe ColBERT as a working end-to-end mode. There is no answer-generation implementation in this request path.
- `scripts/evaluate_peerqa.py` loads labelled questions and evidence mappings and evaluates only questions with a document ID and mapped gold sentence IDs. It filters retrieval to the known target document, compares returned chunks' `sentence_ids` with gold IDs, and reports mean recall, precision@k, MRR, and nDCG. Precision divides by requested k, even when fewer than k hits return. Gold is not sent to retrieval. The evaluator skips unmapped/out-of-corpus evidence and does not measure answer correctness, faithfulness, abstention, latency, index cost, or full cross-document retrieval.

## 3. Corpus separation and schemas

### Keep arXiv and PeerQA contracts distinct

The arXiv document ID is the arXiv ID. Chunk JSON has a top-level document ID, configuration, and section-qualified chunk IDs. Its current chunk payload has no source sentence IDs. PeerQA canonical IDs are hashes of original IDs; its chunk text is built from source rows and each chunk carries official sentence IDs and source locations. `scripts/ingest_corpus.py` keeps separate acquisition/chunk generation flows but runs the same embedding, Qdrant, and retrieval implementations with corpus-specific directories/collections.

Do not replace the PeerQA chunker with `preprocessing/chunking.py` without an explicit experiment and provenance-aware mapping. The chunkers differ in units, boundary constraints, overlap, IDs, and evidence fields.

## 4. Provenance contract for future work

Every retrievable chunk must remain traceable to its source. For PeerQA preserve `original_paper_id`, canonical `document_id`, `section_id`/title, `paragraph_id`, each `sentence_id`, and each source `idx`/`pidx`/`sidx` reference through the serialized chunk, embedding-index pairing, Qdrant payload, and evaluation hit.

- **A source sentence split:** Each fragment may repeat the same official sentence ID and source reference. Chunk text fragments can differ, but do not invent new official source IDs. Current PeerQA oversized-row behavior repeats the ID/ref on each fragment.
- **Overlap:** A source sentence included in multiple chunks must appear in the provenance of every such chunk. Overlap is retrieval context; it does not create a new source sentence.
- **Several sentences packed:** A chunk carries the ordered collection of all source sentence IDs and source refs that contributed text. Keep their association/order recoverable.
- **Gold points to title, heading, or empty row:** Preserve the official mapping and mark whether it appears in any retrievable chunk. Do not convert a source-level mapping into “retrieved” merely because an ID is in `idx_map`.
- **Chunk strategy changes:** Preserve source IDs; create new chunk IDs only according to a documented scheme/config. Evaluation should compare chunk hits by source sentence IDs, not assume old chunk IDs or boundaries survive.
- **Text edits/normalization:** Do not silently alter source row text while claiming exact official evidence text. Record extraction/normalization method and ensure evidence mapping still targets the same source row.

## 5. Embedding/index details and constraints

Current embedding configuration is centralized in `preprocessing/embeddings.py`: `MODEL_NAME`, `MAX_SEQ_LENGTH`, `BATCH_SIZE`, `MODEL_KWARGS`, `cache_signature`, `embedding_config`, and `encode_corpus`. An embedding-model change must update/review cache configuration and validation, not just `load_model`. Qwen's dimension is obtained from the loaded model; `.env.example` documents default Qdrant vector size 4096 and distance Cosine. A model/config/cache change requires re-embedding where needed.

Qdrant settings read from `.env` include endpoint, key, collection, vector size, and distance. `.env.example` distinguishes default/PeerQA collection variables; never read, reproduce, or commit `.env` secrets. Collection names are selected in `retrieval/qdrant_setup.py::collection_name_for`. Payload identity includes deterministic point ID derived from `chunk_id`, the `corpus_version`, and source chunk metadata. Version identity includes model configuration and ordered chunk IDs/text fingerprints (plus PeerQA corpus ID). There is no automatic old-version deletion/retention policy in the inspected uploader.

Known coupling: embedding/Qdrant corpus selection is implemented by mutating module-level paths through `configure_corpus`; setup and retrieval functions rely on consistent selection. Qdrant setup imports PeerQA manifest/status helpers lazily to write PeerQA indexing status. Preserve existing defaults and test corpus path isolation when touching these interfaces.

## 6. Retrieval and evaluation interpretation

Current CLI modes are `dense`, `bm25`, and `hybrid`. Dense/hybrid use the Qwen query encoder; BM25 uses Qdrant's text query API over the indexed text field. Hybrid uses Qdrant RRF over dense and text prefetch results. `document_id` and `corpus_version` are filters. Optional cross-encoder reranking is a second stage after the selected search. ColBERT's encoder helper alone is not a retrieval implementation.

The current PeerQA evaluation is retrieval-only and uses sentence overlap as relevance. It restricts each question to its known paper, includes only questions with mapped evidence, and reports mean recall, precision@k, MRR, and nDCG. It does not execute a language model to answer, so README's answer-quality/faithfulness research dimensions remain goals rather than measurements in this script. Be cautious when comparing reported values: corpus version, K, retrieval mode, filters, number of eligible questions, unmapped evidence, and chunking strategy all matter.

Saved embedding model comparison scripts/results under `evals/embeddings/` are separate experiments. `evals/reports/embedding_model_selection.txt` explicitly says its saved scores predate current section-qualified chunk IDs and should be treated as historical until rerun. `evals/reports/qdrant_ingestion_status.md` records a dated arXiv ingestion snapshot, not necessarily current live state.

## 7. Safe workflows

### Add a corpus

1. Inspect official source schema, licensing, identity fields, split semantics, and gold evidence granularity.
2. Add corpus-local acquisition, source checksums, normalized output, and status without editing source data.
3. Define stable original and canonical IDs; specify provenance mapping before chunking.
4. Add a corpus selector/path in embeddings and Qdrant only if the chunk schema satisfies their contracts; preserve existing arXiv defaults.
5. Add fixture tests for source parsing, IDs, provenance, and isolated paths; validate against a small authorized real sample separately and label that as integration validation.
6. Run prepare-only/local checks before model or remote service stages. Record which stages were actually executed.

### Add a chunking strategy

1. Keep the current strategy as a named baseline and record unit, tokenizer, size, overlap, section/paragraph constraints, and ID scheme.
2. Carry source IDs through sentence splitting, oversized-row splitting, packing, and overlap. Add a mapping from each output chunk to source references.
3. Include strategy/configuration in corpus/chunk artifact metadata and versioning as appropriate. Do not alter the existing default silently.
4. Add tests for boundary, maximum size, overlap, empty input, oversized source records, stable IDs, and complete provenance. Compare on fixed source data and fixed queries.

### Add an embedding model

1. Make model, tokenizer/query behavior, dimension, max sequence length, normalization, dtype/quantization, and prompt conventions explicit.
2. Extend cache signatures and validation so incompatible vectors are rejected and exact reusable chunks can be safely identified.
3. Test matrix shape, finiteness, interrupted/pending-batch resume, changed text/IDs, and reuse. Then run a documented benchmark on the same chunks/query set as the baseline.
4. Do not rely on `.env` vector dimension defaults when model output says otherwise; validate before indexing.

### Add a retrieval strategy

1. Trace payload fields and Qdrant named-vector/index configuration in `retrieval/qdrant_setup.py` and query flow in `retrieval/search.py`.
2. Keep corpus-version/document filters consistent across candidate-generation stages. Ensure returned payload retains PeerQA sentence/source references.
3. Add mocked unit tests for query construction and filters, then perform live Qdrant integration validation when credentials/services are available.
4. Compare with fixed version, corpus, query set, K, and candidate budget; report latency and cost if measured.

### Add an evaluation metric

1. Define relevance from the existing gold structure before coding. Specify treatment of unmapped/out-of-subset evidence and fewer-than-K results.
2. Add small deterministic metric tests (including no relevant items and tied/missing cases where applicable).
3. Keep answer-generation metrics separate from retrieval-only metrics. Record eligible/skipped counts and evaluator filters.
4. Never present a fixture result as a benchmark result.

### Integrate source-level evidence

1. Preserve the source dataset's original IDs and index semantics.
2. Build source-index-to-stable-evidence-ID mapping independently from chunks, following `peerqa/corpus.py::map_question_evidence` as the local example.
3. Store explicit statuses for absent, malformed, outside-corpus, or unrepresented evidence; retain source references rather than dropping them.
4. Verify that every expected gold reference either appears in one or more retrievable chunks or is explicitly reported as unmapped/unretrievable.

### Refactor shared processing

1. First document both corpus input/output contracts, chunk units/boundaries, IDs, provenance, and failure behavior.
2. Separate reusable mechanics from corpus semantics. Similar function names do not establish equivalence.
3. Keep old paths and baselines available while adding an opt-in shared route. Preserve arXiv cache paths, IDs, and outputs unless a reviewed migration says otherwise.
4. Test old-output compatibility and new provenance propagation at each transformation. Run a real sample comparison before making the shared route a default.

## 8. Tests and external validation

The existing test file is `tests/test_peerqa_pipeline.py` and uses `unittest`; run from the repository root with `.\.venv\Scripts\python.exe -m unittest discover -s tests`. Its archive, downloader, embedding, Qdrant, retrieval, and metric cases primarily use temporary directories, fixtures, and mocks. No separate arXiv tests were present in the inspected `tests/` tree. This audit did not run tests and did not inspect/validate the real PeerQA archive.

External requirements differ by stage: archive/arXiv downloads need network; uncached GROBID extraction needs a reachable GROBID service; embedding generation loads Qwen3 and requires CUDA under the configured 4-bit path; Qdrant setup/upload/query needs valid `.env` credentials and a reachable service. PeerQA `--prepare-only` performs acquisition/normalization without embeddings or Qdrant.

## 9. Confirmed limitations and technical debt

| Location | Observed behavior | Why it matters / direction |
| --- | --- | --- |
| `peerqa/corpus.py::_build_document` and `map_question_evidence` | `idx_map` includes every row, but document paragraph records omit empty/title/heading rows; mapping can therefore yield a gold sentence ID absent from chunk payloads. | Track mapped-but-not-retrievable evidence explicitly and inspect counts on real PeerQA data before interpreting recall. Fixture tests do not establish frequency. |
| `peerqa/corpus.py::download_permitted_pdfs` | Reused PDFs are accepted by file size and magic signature; acquired PDF is hashed, but there is no PDF/text alignment or paper-identity version check. PDFs do not supply indexed PeerQA text. | Avoid claiming PDF acquisition verifies the indexed text version. Keep official JSONL text as evidence source unless alignment is validated. |
| `preprocessing/chunking.py` vs `peerqa/corpus.py::_chunks_for_document` | Different chunk units, limits, boundaries, overlap, IDs, and provenance. | Do not unify by swapping implementations; any shared path must retain corpus-specific policy and source traceability. |
| `preprocessing/embeddings.py::load_chunks` | Embedding input view keeps only ID, text, and token count; provenance is restored from source chunk files by Qdrant loading, with strict ordered ID/fingerprint checks. | Preserve the pairing invariant. A future embedding format must not reorder or mismatch metadata and vectors. |
| `retrieval/qdrant_setup.py::upload_chunks` | Creates dense vectors and a sparse-vector collection slot, but uploads dense vectors only; versioned points are verified, while old versions are not explicitly deleted. | Do not call the current path sparse-vector retrieval. Establish retention/deletion deliberately if storage growth becomes a concern. |
| `retrieval/search.py::colbert_encode` | Helper is not wired into index creation or `Retriever.search`. | Treat as a utility/prototype rather than an available retrieval mode. |
| `scripts/evaluate_peerqa.py` | Scores only questions with mapped evidence and filters retrieval to the known target document; no answer generation or operational-cost measurement. | Results answer a constrained evidence-retrieval question, not full RAG answer quality or unrestricted retrieval. |
| `evals/reports/embedding_model_selection.txt` | Saved rankings predate current section-qualified chunk IDs and are marked historical pending rerun. | Do not cite them as current results without reproducing the benchmark. |

These are code-inspection findings, not claims about the prevalence or impact of each issue on all real corpus records.

## 10. Experimental design checklist

- Freeze and record corpus/dataset version, included documents, query IDs, evidence mapping version, and eligibility rules.
- Record chunking strategy and all parameters, including tokenizers, word/token units, section boundaries, overlap, and chunk ID version.
- Keep embedding model/configuration, index schema, retrieval mode, K/candidate budget, filters, and reranker settings explicit.
- Preserve an unchanged baseline and vary one main factor at a time; use ablations to isolate chunking, embedding, retrieval, and reranking effects.
- Report retrieval evidence metrics separately from generated-answer quality/faithfulness. Also measure latency, index size, embedding/indexing time, and compute only when actually instrumented.
- Include skipped/unmapped evidence counts, query/paper filters, uncertainty or sample limits, and known provenance gaps.
- Compare across structurally different corpora only with their different ground-truth coverage and source semantics stated. A good score on one benchmark is not evidence of universal advantage.
