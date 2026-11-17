# Repository agent instructions

## 1. Project identity and objective

This repository is a research codebase for studying how corpus and query characteristics affect RAG architecture performance. Its objective is to identify when architectural choices produce measurable benefits and what quality, latency, storage, and compute trade-offs they introduce. Treat this as a controlled comparison project, not simply a demonstration that RAG works.

## 2. Research principles

- Compare retrieval quality and evidence relevance; measure answer quality and faithfulness only where an answer-generation evaluation actually exists.
- Consider latency, index/storage overhead, embedding and retrieval compute, robustness, generalization, corpus structure, and query requirements.
- The current PeerQA evaluator measures evidence retrieval (recall, precision@k, MRR, and nDCG) for mapped evidence. It does not generate answers or measure answer faithfulness. README research questions and planned architectures are not evidence that those capabilities exist.
- Keep measured results, historical reports, hypotheses, and estimates clearly distinguished. Preserve fixed datasets, configurations, and baselines for fair comparisons.

## 3. Repository architecture

- `download_corpus/download_pdfs.py`: arXiv paper list, PDF download/retries, title checks, pypdf extraction, and per-paper extracted-text JSON/metadata.
- `preprocessing/chunking.py`: arXiv GROBID TEI or extracted-text input, flat section/block construction, tokenizer-aware chunking, and chunk JSON output. GROBID is needed only for uncached TEI in the GROBID mode.
- `peerqa/corpus.py`: PeerQA archive/source acquisition, source-row normalization, document and paragraph reconstruction, official evidence-index mapping, paragraph chunks, manifest/status, and licensed PDF acquisition. PeerQA's searchable text remains the official `papers.jsonl` text; acquired PDFs do not replace it.
- `preprocessing/embeddings.py`: corpus-selectable chunk loading and resumable Qwen3-Embedding-8B vector cache. The arXiv paths remain its defaults; `--corpus peerqa` selects isolated PeerQA paths.
- `retrieval/qdrant_setup.py`: cache validation, corpus/version-specific collection selection, Qdrant schema, chunk payload upload, retries, and upload count verification.
- `retrieval/search.py`: shared dense, Qdrant text/BM25, and hybrid retrieval, optional cross-encoder reranking, and an exposed ColBERT encoding helper. ColBERT is not wired as an end-to-end indexing/search mode.
- `scripts/ingest_corpus.py`: orchestrates the arXiv or PeerQA stages. `scripts/query.py` delegates to `retrieval/search.py`; `scripts/corpus_status.py` prints local status; `scripts/evaluate_peerqa.py` evaluates PeerQA evidence retrieval.
- `preprocessing/diagnostics.py`: audits cached arXiv TEI/chunks and benchmark-tokenizer lengths.
- `evals/embeddings/compare_embeddings.py` and `_v2.py`: separate model-comparison experiments. `evals/embeddings/*` and `evals/reports/*` include saved artifacts/reports; some are historical and should not be presented as current rerun results without verification.
- `tests/test_peerqa_pipeline.py`: unittest coverage focused on PeerQA and mocked/local embedding, Qdrant, and retrieval behavior. There is no separate arXiv test suite in the inspected `tests/` tree.

Typical data flow:

```text
arXiv list -> download_corpus/download_pdfs.py -> PDFs + extracted_text JSON
  -> preprocessing/chunking.py -> chunks JSON
PeerQA archive -> peerqa/corpus.py -> official source rows -> documents + chunks
  -> preprocessing/embeddings.py -> embedding cache
  -> retrieval/qdrant_setup.py -> corpus-versioned Qdrant collection
  -> retrieval/search.py -> hits
  -> scripts/evaluate_peerqa.py -> evidence retrieval metrics
```

The two corpora have separate acquisition and chunking logic. Embedding cache, Qdrant setup, and retrieval are shared by selecting corpus-specific paths/collections. Do not assume the corpus-specific chunkers or their output metadata are interchangeable.

## 4. Data integrity and provenance

- Preserve original paper IDs separately from canonical document IDs and preserve official question IDs and source `idx`, `pidx`, and `sidx` values.
- Preserve sentence, paragraph, section, and source-location metadata wherever the source provides it. Never silently discard provenance during splitting, overlap, embedding, payload conversion, or evaluation.
- PeerQA maps official evidence `idx` values to source-row sentence IDs independently of chunk boundaries. Keep gold mapping independent of chunk boundaries wherever possible.
- Do not silently change document IDs, chunk IDs, cache signatures, collection payloads, or version computation. Explain migrations and their effects on cached embeddings and indexes.
- Treat raw source files as inputs. Write normalized and generated artifacts to their established output paths; do not edit source data in place.
- Do not infer a PDF URL or permission from a title. PeerQA PDF acquisition requires a supported EGU ID rule or explicit HTTPS source and license fields.

## 5. Experimental integrity

- Make dataset versions, corpus IDs, chunking parameters, tokenizer/embedding models, retrieval parameters, and evaluation conditions explicit and reproducible.
- Keep datasets, query sets, and evidence labels consistent across compared systems. Avoid changing multiple experimental variables at once without explaining why.
- Preserve baseline implementations and corpus isolation. A PeerQA experiment must not overwrite arXiv chunks, embedding cache, or collection settings.
- Report measured results separately from hypotheses, estimates, and historical artifacts. Never fabricate benchmark results or claim tests/integration runs that were not performed.
- Interpret PeerQA's current evidence metrics in light of its evaluator's document filter and mapped-evidence subset; they do not establish answer-generation quality or unrestricted corpus-wide retrieval performance.

## 6. Existing architecture and implementation constraints

- The arXiv path starts from a fixed `PAPERS` list in `download_corpus/download_pdfs.py`. It downloads and validates PDF magic bytes, checks titles against the arXiv API when available, extracts page text with pypdf, and reuses files on reruns. `preprocessing/chunking.py` can instead parse GROBID TEI into flat sections and text/table/figure blocks. Its 1900-token limit is measured as the maximum across three benchmark tokenizers and its overlap is up to 100 tokens by sentence.
- PeerQA expects official source rows with `paper_id`, integer `idx`, optional integer/null `pidx` and `sidx`, `content`, `type`, and `last_heading`. `_build_document` constructs canonical IDs, section/paragraph records, and an `idx_map`; `_row_sentence_id` embeds the official row indices in a stable source sentence ID. `map_question_evidence` maps `answer_evidence_mapped[].idx` through that `idx_map`, retains unmapped reasons, and preserves answerability fields. `_chunks_for_document` uses paragraph boundaries and a 900-word default with no overlap, and stores `sentence_ids`/`source_refs`.
- The PeerQA and arXiv chunkers have different boundaries, units, and provenance models. Do not silently substitute one for the other. `preprocessing/embeddings.py` currently reduces chunks to ID/text/token count for embedding; Qdrant reloads and uploads chunk metadata as payload.
- Any future shared chunking path must preserve provenance through source-row splitting, sentence splitting, packing, overlap, and multiple chunks containing the same source evidence. Gold evidence IDs must remain stable when chunk boundaries change.
- PeerQA PDF downloads validate response size when available and the `%PDF-` signature and record a hash, but the downloaded PDF is not used to replace official PeerQA extracted text. This is not source-version or paper-identity verification. Do not describe it as such.

## 7. Coding rules for future agents

- Inspect before modifying; trace callers and data flow before changing interfaces.
- Prefer extending existing abstractions over creating duplicate implementations, but avoid premature shared abstractions when corpus semantics differ.
- Keep changes scoped. Do not reorganize large modules solely for aesthetics, remove downstream functionality because it is currently unused, or change application/data/configuration outside the requested scope.
- Preserve backward compatibility unless a deliberate migration is justified and documented.
- Add focused tests for behavior changes and update documentation when architecture or interfaces change.

## 8. Testing and validation

- The inspected suite uses `unittest` in `tests/test_peerqa_pipeline.py`. A repository-local invocation is `.\.venv\Scripts\python.exe -m unittest discover -s tests` from the repository root. This command was identified from the test framework; do not claim it passed unless it is run.
- The existing tests use temporary files, fixture rows, and mocks for archive/PDF requests, embedding cache, Qdrant payload loading, retrieval filters, and metric helpers. They are unit/contract tests, not validation against the real PeerQA archive, live arXiv/GROBID, GPU embeddings, Qdrant Cloud, or a complete live evaluation.
- Real acquisition needs network access. GROBID mode needs a reachable GROBID service if TEI is uncached or overwritten. Embedding generation uses CUDA and the configured Hugging Face model. Qdrant upload/query needs valid `QDRANT_URL` and `QDRANT_API_KEY`. `--prepare-only` on PeerQA avoids embedding and Qdrant stages.
- If a test or integration stage cannot be run, say which one and why. Do not invent commands or call mocked tests a live integration test.

## 9. Agent communication requirements

At completion, report files inspected and modified, observed behavior, architectural choices and rationale, tests actually executed and their results, known limitations, and assumptions or unresolved questions. State explicitly when application code or source data was left untouched.
