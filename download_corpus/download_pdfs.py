#!/usr/bin/env python3
"""
arXiv paper downloader + text extractor for the RAG corpus.

For each paper in PAPERS it:
  1. downloads the PDF (streamed to a .pdf.tmp file, renamed only when complete),
  2. checks the arXiv title against the expected title written in PAPERS,
  3. extracts text with pypdf into extracted_text/<arxiv_id>.json for chunking.

Resuming: the files on disk are the record of progress. A paper whose PDF is
already valid is skipped, and a paper whose extracted_text JSON exists is
reused (use --force to re-extract). So Ctrl+C is a safe "pause"; rerun to resume.
Failures are NOT stored in a queue file: rerunning retries anything missing.

document_id (= arxiv_id) is the stable ID the chunker should use as the prefix
for every chunk_id it generates.

Usage:
    uv run python download_corpus/download_pdfs.py                  # download + extract all
    uv run python download_corpus/download_pdfs.py --limit 10       # first 10 only, for testing
    uv run python download_corpus/download_pdfs.py --skip-download  # extract from existing PDFs only
    uv run python download_corpus/download_pdfs.py --download-only  # PDFs only, no extraction
    uv run python download_corpus/download_pdfs.py --force          # re-extract even if JSON exists

Exit code: 0 if everything succeeded, 1 if any paper failed or the run was interrupted.
"""

import argparse
import json
import logging
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from pypdf import PdfReader

# ============================================================================
# CONFIG
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent / "rag_corpus"
PDF_DIR = BASE_DIR / "pdfs"
TEXT_DIR = BASE_DIR / "extracted_text"
METADATA_FILE = BASE_DIR / "metadata.json"
LOG_FILE = BASE_DIR / "download.log"

PDF_DIR.mkdir(parents=True, exist_ok=True)
TEXT_DIR.mkdir(parents=True, exist_ok=True)

REQUEST_DELAY_SECONDS = 3.0  # arXiv's own guidance; don't go lower than this
MAX_RETRIES = 3
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
TITLE_MATCH_THRESHOLD = 0.8  # 0..1 similarity between expected and arXiv title
ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
USER_AGENT = "rag-and-eval/0.1 (personal research project)"

# Plain ASCII tags only: emoji in log lines can crash on Windows consoles (cp1252).
# Console + file, both with timestamps.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

SESSION = requests.Session()
SESSION.headers["User-Agent"] = USER_AGENT

# ============================================================================
# PAPER LIST: (arxiv_id, category, expected_title)
# The expected title is checked against arXiv's metadata, so a wrong ID is
# flagged instead of silently downloading the wrong paper.
# ============================================================================

PAPERS: List[Tuple[str, str, str]] = [
    ("2005.11401", "RAG", "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"),
    ("2310.11511", "RAG", "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"),
    ("2401.15884", "RAG", "Corrective Retrieval Augmented Generation"),
    ("2401.18059", "RAG", "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"),
    ("2004.04906", "Retrieval", "Dense Passage Retrieval for Open-Domain Question Answering"),
    ("2306.07174", "RAG", "Augmenting Language Models with Long-Term Memory"),
    ("2002.08909", "RAG", "REALM: Retrieval-Augmented Language Model Pre-Training"),
    ("2212.09146", "RAG", "Can Retriever-Augmented Language Models Reason? The Blame Game Between the Retriever and the Language Model"),
    ("2112.04426", "RAG", "Improving language models by retrieving from trillions of tokens"),
    ("2208.03299", "RAG", "Atlas: Few-shot Learning with Retrieval Augmented Language Models"),
    ("2007.01282", "RAG", "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering"),
    ("2004.12832", "Retrieval", "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT"),
    ("2112.01488", "Reranking", "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction"),
    ("1908.10084", "Embeddings", "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"),
    ("2205.13147", "Embeddings", "Matryoshka Representation Learning"),
    ("2212.03533", "Embeddings", "Text Embeddings by Weakly-Supervised Contrastive Pre-training"),
    ("2201.10005", "Embeddings", "Text and Code Embeddings by Contrastive Pre-Training"),
    ("2402.01613", "Embeddings", "Nomic Embed: Training a Reproducible Long Context Text Embedder"),
    ("2402.03216", "Embeddings", "M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation"),
    ("2401.00368", "Embeddings", "Improving Text Embeddings with Large Language Models"),
    ("2210.07316", "Evaluation", "MTEB: Massive Text Embedding Benchmark"),
    ("2107.05720", "Retrieval", "SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking"),
    ("2004.13969", "Retrieval", "Complementing Lexical Retrieval with Semantic Residual Embedding"),
    ("2007.00808", "Retrieval", "Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval"),
    ("2112.09118", "Retrieval", "Contriever: Unsupervised Dense Information Retrieval with Contrastive Learning"),
    ("2212.10496", "Retrieval", "Precise Zero-Shot Dense Retrieval without Relevance Labels"),
    ("2303.07678", "Retrieval", "Query2doc: Query Expansion with Large Language Models"),
    ("2212.10509", "Advanced", "Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions"),
    ("2305.06983", "Advanced", "Active Retrieval Augmented Generation"),
    ("2210.03629", "Advanced", "ReAct: Synergizing Reasoning and Acting in Language Models"),
    ("2404.16130", "Advanced", "From Local to Global: A Graph RAG Approach to Query-Focused Summarization"),
    ("2403.14403", "Advanced", "Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity"),
    ("2402.07630", "Advanced", "G-Retriever: Retrieval-Augmented Generation for Textual Graphs"),
    ("2301.12652", "Advanced", "REPLUG: Retrieval-Augmented Black-Box Language Models"),
    ("2311.09476", "Evaluation", "ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems"),
    ("2408.08067", "Evaluation", "RAGCHECKER: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation"),
    ("2401.00396", "Evaluation", "RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models"),
    ("2309.15217", "Evaluation", "Ragas: Automated Evaluation of Retrieval Augmented Generation"),
    ("2504.14891", "Evaluation", "Retrieval Augmented Generation Evaluation in the Era of Large Language Models: A Comprehensive Survey"),
    ("2405.07437", "Evaluation", "Evaluation of Retrieval-Augmented Generation: A Survey"),
    ("2404.10981", "Survey", "A Survey on Retrieval-Augmented Text Generation for Large Language Models"),
    ("2312.10997", "Survey", "Retrieval-Augmented Generation for Large Language Models: A Survey"),
    ("2104.08663", "Evaluation", "BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models"),
    ("1611.09268", "Evaluation", "MS MARCO: A Human Generated MAchine Reading COmprehension Dataset"),
    ("1809.09600", "Evaluation", "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering"),
    ("2009.02252", "Evaluation", "KILT: a Benchmark for Knowledge Intensive Language Tasks"),
    ("1603.09320", "Indexing", "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"),
    ("1702.08734", "Indexing", "Billion-scale similarity search with GPUs"),
    ("1908.10396", "Indexing", "Accelerating Large-Scale Inference with Anisotropic Vector Quantization"),
    ("2210.10634", "Reranking", "RankT5: Fine-Tuning T5 for Text Ranking with Ranking Losses"),
    ("1910.10683", "LLM", "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"),
    ("1706.03762", "LLM", "Attention Is All You Need"),
    ("1901.02860", "LLM", "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"),
    ("2210.11416", "LLM", "Scaling Instruction-Finetuned Language Models"),
    ("2201.11903", "LLM", "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"),
    ("2305.10601", "LLM", "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"),
    ("2205.10625", "LLM", "Least-to-Most Prompting Enables Complex Reasoning in Large Language Models"),
    ("2212.08073", "LLM", "Constitutional AI: Harmlessness from AI Feedback"),
    ("2106.09685", "LLM", "LoRA: Low-Rank Adaptation of Large Language Models"),
    ("2104.08691", "LLM", "The Power of Scale for Parameter-Efficient Prompt Tuning"),
    ("2309.06180", "Production", "Efficient Memory Management for Large Language Model Serving with PagedAttention"),
    ("2205.14135", "Production", "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"),
    ("2305.05176", "Production", "FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance"),
    ("2405.14831", "Advanced", "HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models"),
]

_seen_ids = set()
_dupes = sorted({p[0] for p in PAPERS if p[0] in _seen_ids or _seen_ids.add(p[0])})
if _dupes:
    raise ValueError(f"Duplicate arXiv IDs in PAPERS: {', '.join(_dupes)}")


# ============================================================================
# HELPERS
# ============================================================================


def write_json_atomic(path: Path, data: Dict) -> None:
    """Write JSON to <path>.tmp, then rename, so a crash never leaves a half-written file."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def is_valid_pdf(path: Path) -> bool:
    """True only if the file exists, is non-empty, and starts with the PDF magic bytes."""
    try:
        if not path.exists() or path.stat().st_size == 0:
            return False
        with open(path, "rb") as f:
            return f.read(5) == b"%PDF-"
    except OSError:
        return False


def _normalize(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def titles_match(expected: str, actual: str) -> bool:
    ratio = SequenceMatcher(None, _normalize(expected), _normalize(actual)).ratio()
    return ratio >= TITLE_MATCH_THRESHOLD


# ============================================================================
# TITLES (one batched request for the whole run)
# ============================================================================


def fetch_titles(arxiv_ids: List[str]) -> Dict[str, str]:
    """Fetch official arXiv titles for all IDs in a single API call."""
    titles: Dict[str, str] = {}
    if not arxiv_ids:
        return titles

    try:
        resp = SESSION.get(
            ARXIV_API_URL,
            params={"id_list": ",".join(arxiv_ids), "max_results": len(arxiv_ids)},
            timeout=(10, 60),
        )
    except requests.RequestException as e:
        logger.warning("TITLES | request failed | %s: %s", type(e).__name__, e)
        return titles

    if resp.status_code != 200:
        logger.warning("TITLES | HTTP %d %s | titles unavailable this run", resp.status_code, resp.reason)
        return titles

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as e:
        logger.warning("TITLES | could not parse arXiv response: %s", e)
        return titles

    wanted = set(arxiv_ids)
    for entry in root.findall("atom:entry", ATOM_NS):
        id_el = entry.find("atom:id", ATOM_NS)
        title_el = entry.find("atom:title", ATOM_NS)
        if id_el is None or id_el.text is None or title_el is None or not title_el.text:
            continue
        arxiv_id = re.sub(r"v\d+$", "", id_el.text.rsplit("/abs/", 1)[-1])
        if arxiv_id in wanted:
            titles[arxiv_id] = " ".join(title_el.text.split())

    logger.info("TITLES | fetched %d/%d titles from arXiv (HTTP 200)", len(titles), len(arxiv_ids))
    time.sleep(REQUEST_DELAY_SECONDS)
    return titles


# ============================================================================
# DOWNLOAD
# ============================================================================


def _retry_wait(resp: requests.Response, attempt: int) -> float:
    """Honor Retry-After if the server sent it, otherwise back off exponentially."""
    retry_after = resp.headers.get("Retry-After", "")
    if retry_after.isdigit():
        return min(int(retry_after), 120)
    return 3 * 2 ** attempt  # 6s, 12s, 24s


def _attempt_download(url: str, tmp_path: Path, attempt: int) -> Tuple[str, str, float]:
    """
    One download attempt, streamed to tmp_path.
    Returns (outcome, detail, wait): outcome is "ok", "retry", or "fail".
    """
    try:
        with SESSION.get(url, stream=True, timeout=(10, 30)) as resp:
            status = resp.status_code
            if status in RETRYABLE_STATUSES:
                return "retry", f"HTTP {status} {resp.reason}", _retry_wait(resp, attempt)
            if status != 200:
                return "fail", f"HTTP {status} {resp.reason}", 0

            content_type = resp.headers.get("Content-Type", "unknown")
            written = 0
            first_chunk = True
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    if first_chunk:
                        first_chunk = False
                        if chunk[:5] != b"%PDF-":
                            # arXiv sometimes returns an HTML error page with HTTP 200.
                            return "fail", f"HTTP 200 but not a PDF (Content-Type: {content_type})", 0
                    f.write(chunk)
                    written += len(chunk)

            if written == 0:
                return "fail", "HTTP 200 but empty response", 0

            expected = resp.headers.get("Content-Length")
            if expected and not resp.headers.get("Content-Encoding") and written != int(expected):
                return "retry", f"incomplete download ({written} of {expected} bytes)", 2 ** attempt

            return "ok", f"HTTP 200 | {written / 1024:.1f} KB", 0

    except (
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
        requests.exceptions.ChunkedEncodingError,
    ) as e:
        return "retry", f"{type(e).__name__}: {e}", 2 ** attempt
    except requests.RequestException as e:
        return "fail", f"{type(e).__name__}: {e}", 0


def download_paper(arxiv_id: str) -> Tuple[str, str]:
    """
    Download one PDF. Returns (status, detail) where status is
    "downloaded", "skipped", or "failed".
    """
    pdf_path = PDF_DIR / f"{arxiv_id}.pdf"
    tmp_path = pdf_path.with_suffix(".pdf.tmp")

    if is_valid_pdf(pdf_path):
        logger.info("%s | SKIPPED    | already on disk", arxiv_id)
        return "skipped", "already on disk"
    if pdf_path.exists():
        logger.warning("%s | INVALID    | existing file is not a valid PDF, re-downloading", arxiv_id)

    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    for attempt in range(1, MAX_RETRIES + 1):
        started = time.time()
        outcome, detail, wait = _attempt_download(url, tmp_path, attempt)

        if outcome == "ok":
            tmp_path.replace(pdf_path)  # rename only after the whole file is written
            logger.info("%s | DOWNLOADED | %s | %.1fs", arxiv_id, detail, time.time() - started)
            return "downloaded", detail

        tmp_path.unlink(missing_ok=True)

        if outcome == "fail":
            logger.error("%s | FAILED     | %s", arxiv_id, detail)
            return "failed", detail

        if attempt < MAX_RETRIES:
            logger.warning(
                "%s | RETRY      | attempt %d/%d | %s | waiting %.0fs",
                arxiv_id, attempt, MAX_RETRIES, detail, wait,
            )
            time.sleep(wait)
        else:
            logger.error("%s | FAILED     | %s | gave up after %d attempts", arxiv_id, detail, MAX_RETRIES)
            return "failed", f"{detail} (after {MAX_RETRIES} attempts)"

    return "failed", "unknown error"


# ============================================================================
# EXTRACTION
# ============================================================================


def extract_text_from_pdf(pdf_path: Path) -> Tuple[str, int]:
    """Extract text page by page with pypdf. Returns (text, num_pages)."""
    try:
        reader = PdfReader(str(pdf_path))
        parts = []
        for page_num, page in enumerate(reader.pages, 1):
            page_text = page.extract_text()
            if page_text:
                parts.append(f"\n--- Page {page_num} ---\n{page_text}")
        return "".join(parts), len(reader.pages)
    except Exception as e:
        logger.error("Failed to extract text from %s: %s: %s", pdf_path.name, type(e).__name__, e)
        return "", 0


def save_extracted_text(
    arxiv_id: str, title: Optional[str], category: str, text: str, num_pages: int
) -> Dict:
    """Save extracted text + metadata to JSON for the chunking step."""
    num_words = len(text.split())
    stats = {
        "num_chars": len(text),
        "num_words": num_words,
        "num_pages": num_pages,
        "approx_tokens": round(num_words * 1.3),  # ~1.3 tokens per English word
    }
    data = {
        "document_id": arxiv_id,  # stable ID the chunker should prefix chunk_ids with
        "arxiv_id": arxiv_id,
        "title": title,
        "category": category,
        "timestamp": datetime.now().isoformat(),
        "text": text,
        "stats": stats,
    }
    write_json_atomic(TEXT_DIR / f"{arxiv_id}.json", data)
    return stats


def write_metadata() -> Dict:
    """Rebuild metadata.json from the extracted_text files on disk (always complete, even after --limit or Ctrl+C)."""
    papers: Dict[str, Dict] = {}
    total_tokens = 0
    for arxiv_id, _category, _title in PAPERS:
        text_path = TEXT_DIR / f"{arxiv_id}.json"
        if not text_path.exists():
            continue
        try:
            data = json.loads(text_path.read_text(encoding="utf-8"))
            papers[arxiv_id] = {
                "category": data.get("category"),
                "title": data.get("title"),
                "stats": data["stats"],
            }
            total_tokens += data["stats"]["approx_tokens"]
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning("%s | METADATA  | could not read %s: %s", arxiv_id, text_path.name, e)

    metadata = {
        "generated_at": datetime.now().isoformat(),
        "total_papers": len(PAPERS),
        "successful": len(papers),
        "failed": len(PAPERS) - len(papers),
        "total_tokens": total_tokens,
        "papers": papers,
    }
    write_json_atomic(METADATA_FILE, metadata)
    return metadata


# ============================================================================
# PIPELINE
# ============================================================================


def ingest_papers(
    paper_list: List[Tuple[str, str, str]],
    skip_download: bool = False,
    download_only: bool = False,
    force: bool = False,
) -> Dict:
    results: Dict = {
        "downloaded": 0,
        "skipped": 0,
        "extracted": 0,
        "reused": 0,
        "failed": [],  # list of (arxiv_id, reason)
        "title_mismatches": [],
        "interrupted": False,
    }
    total = len(paper_list)
    logger.info(
        "Processing %d papers (skip_download=%s, download_only=%s, force=%s)",
        total, skip_download, download_only, force,
    )

    try:
        titles = fetch_titles([p[0] for p in paper_list])

        for index, (arxiv_id, category, expected_title) in enumerate(paper_list, 1):
            logger.info("[%02d/%02d] %s", index, total, arxiv_id)
            pdf_path = PDF_DIR / f"{arxiv_id}.pdf"

            # ---- download ----
            if skip_download:
                if not is_valid_pdf(pdf_path):
                    logger.warning("%s | MISSING    | no valid local PDF (--skip-download)", arxiv_id)
                    results["failed"].append((arxiv_id, "no valid local PDF"))
                    continue
            else:
                status, detail = download_paper(arxiv_id)
                if status != "skipped":
                    time.sleep(REQUEST_DELAY_SECONDS)  # be polite after any network request
                if status == "failed":
                    results["failed"].append((arxiv_id, detail))
                    continue
                results[status] += 1

            # ---- title check ----
            actual_title = titles.get(arxiv_id)
            if actual_title is None:
                logger.warning("%s | TITLE      | not available from arXiv this run", arxiv_id)
            elif not titles_match(expected_title, actual_title):
                logger.warning(
                    "%s | TITLE      | MISMATCH | expected: %s | arXiv: %s",
                    arxiv_id, expected_title, actual_title,
                )
                results["title_mismatches"].append(arxiv_id)

            if download_only:
                continue

            # ---- extraction ----
            text_path = TEXT_DIR / f"{arxiv_id}.json"
            if text_path.exists() and not force:
                logger.info("%s | REUSED     | extracted text already saved", arxiv_id)
                results["reused"] += 1
                continue

            text, num_pages = extract_text_from_pdf(pdf_path)
            if not text.strip():
                logger.error("%s | FAILED     | no text extracted (scanned or unreadable PDF?)", arxiv_id)
                results["failed"].append((arxiv_id, "no text extracted"))
                continue

            stats = save_extracted_text(arxiv_id, actual_title, category, text, num_pages)
            logger.info(
                "%s | EXTRACTED  | %d pages | %d words | ~%d tokens",
                arxiv_id, stats["num_pages"], stats["num_words"], stats["approx_tokens"],
            )
            results["extracted"] += 1

    except KeyboardInterrupt:
        results["interrupted"] = True
        logger.warning("INTERRUPTED | progress is saved on disk; rerun the same command to resume")
    finally:
        metadata = write_metadata()

    logger.info("=" * 70)
    logger.info("RUN SUMMARY%s", " (INTERRUPTED)" if results["interrupted"] else "")
    logger.info("=" * 70)
    logger.info("Downloaded now:   %d", results["downloaded"])
    logger.info("Already on disk:  %d", results["skipped"])
    logger.info("Text extracted:   %d (reused existing: %d)", results["extracted"], results["reused"])
    logger.info("Failed:           %d", len(results["failed"]))
    for arxiv_id, reason in results["failed"]:
        logger.info("  - %s: %s", arxiv_id, reason)
    if results["title_mismatches"]:
        logger.info("Title mismatches (check these IDs): %s", ", ".join(results["title_mismatches"]))
    logger.info(
        "Corpus on disk:   %d/%d papers extracted, ~%s tokens",
        metadata["successful"], metadata["total_papers"], f"{metadata['total_tokens']:,}",
    )
    logger.info("PDFs: %s | Text: %s | Log: %s", PDF_DIR, TEXT_DIR, LOG_FILE)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and extract papers from arXiv")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N papers (for testing)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract text even if extracted_text/<id>.json already exists",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloads; extract from PDFs already in rag_corpus/pdfs",
    )
    mode.add_argument(
        "--download-only",
        action="store_true",
        help="Download PDFs and check titles, but skip text extraction",
    )
    args = parser.parse_args()

    # Clean up partial downloads left by a previous interrupted run.
    for stale in PDF_DIR.glob("*.pdf.tmp"):
        stale.unlink(missing_ok=True)

    paper_list = PAPERS[: args.limit] if args.limit else PAPERS
    logger.info("Script: download_pdfs.py | papers in list: %d | processing: %d", len(PAPERS), len(paper_list))

    results = ingest_papers(
        paper_list,
        skip_download=args.skip_download,
        download_only=args.download_only,
        force=args.force,
    )
    return 1 if (results["failed"] or results["interrupted"]) else 0


if __name__ == "__main__":
    sys.exit(main())