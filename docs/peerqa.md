# PeerQA corpus

PeerQA is acquired from the official v1.0 archive and stored independently from
the existing arXiv corpus. The pipeline processes `papers.jsonl` (the 90-paper
permissible subset) and all labelled rows in `qa.jsonl`. It does not invoke the
official `papers-all` builder, which downloads the additional 118 OpenReview
papers.

## Commands

```powershell
python scripts/ingest_corpus.py --corpus peerqa
python scripts/query.py --corpus peerqa --query "What methodology did the authors use?"
python scripts/evaluate_peerqa.py --mode hybrid --limit 10
python scripts/ingest_corpus.py --corpus peerqa --status
```

The full ingestion command downloads or reuses the official archive, validates
its ZIP CRC, records SHA-256 checksums, normalizes papers, creates paragraph
chunks, resumes the Qwen embedding cache, and indexes a separate Qdrant
collection. It is safe to rerun. `--prepare-only` stops before GPU embedding
and Qdrant indexing.

PeerQA rows preserve `idx`, `pidx`, and `sidx` under `source_location` and
`source_refs`. Canonical document, section, paragraph, sentence, and chunk IDs
are stable. The original `paper_id` and `question_id` remain separate fields.
Question/evidence files are stored outside the retrieval collection. During
evaluation, only the question and its known target document ID are passed to
retrieval; gold answers and evidence are examined after hits return.

The official Hugging Face builder exposes the release as one `test` split, not
published train/dev/test partitions. The pipeline retains that source split and
does not invent partitions. If the benchmark later supplies official splits,
the ingestion step should be updated to preserve those labels and group them by
paper.

## PDFs and licenses

The official release bundles extracted text, not original PDFs. The PeerQA
repository supplies a deterministic source URL rule for EGU paper IDs, and the
Copernicus license page states that its articles use CC BY 4.0 (or the earlier
CC BY 3.0 for older articles). The pipeline uses that rule for EGU papers only;
it does not guess URLs from titles or download OpenReview papers. Other PDF
sources require an explicit URL and paper license note in the optional source
manifest.

PeerQA's EGU downloader constructs URLs from the original
`egu/{journal}/{volume-page-year}` ID, and the acquisition script honors its
one-second request interval. An optional
`download_corpus/peerqa/pdf_sources.json` can provide other verified sources.
Each entry must contain `original_paper_id`, an HTTPS `pdf_url`,
`license_allows_download: true`, and a license note. The downloader ignores
records for papers outside the 90-paper manifest and records without all
required fields.

The official `papers.jsonl` text remains the indexed source for all papers,
including EGU papers with acquired PDFs, because PeerQA's evidence indices are
aligned to that text. PDF extraction cannot replace it until sentence alignment
has been checked. The manifest records PDF hashes separately from the official
text checksums and marks the extraction method for each indexed document.

The source dataset is licensed CC BY-NC-SA 4.0. That dataset license does not
automatically grant rights to original article PDFs.

## Corpus separation and status

PeerQA chunks and embeddings live under `download_corpus/peerqa/` and
`preprocessing/data/peerqa/`. The default collection is
`peerqa_qwen3_embeddings`, configurable with `QDRANT_COLLECTION_PEERQA`. The
legacy `QDRANT_COLLECTION` remains the arXiv collection setting. Retrieval
selects one collection and corpus version at a time.

`manifest.json` reports paper acquisition, extraction, PDF, chunk, question,
and mapping counts. `index_status.json` is written after Qdrant verifies the
complete PeerQA upload. Questions outside the 90-paper subset are retained and
marked `paper_outside_90_paper_corpus`; unmapped evidence indices are retained
with explicit reasons. `answerable` and `answerable_mapped` remain distinct.
