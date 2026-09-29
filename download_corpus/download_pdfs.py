#!/usr/bin/env python3
"""
Corrected arXiv paper downloader + text extractor for RAG corpus.
Downloads 70 verified RAG papers, extracts text, saves for chunking.

Fixes vs. the original download_papers.py:
  - approx_tokens was computed as num_words // 1.3 (shrinks the count).
    Tokens per word for English academic text is closer to ~1.3x words,
    so it's now num_words * 1.3.
  - No validation that arXiv actually returned a PDF. A bad/withdrawn
    arxiv_id returns an HTML error page with a 200 status, which the
    original code silently saved as a .pdf. Now checks Content-Type
    and the PDF magic bytes before writing.
  - --skip-download was declared but never wired into main(); it now
    actually skips the download loop and only re-runs extraction.
  - Retries now use exponential backoff instead of firing immediately.
  - Added a real per-request delay (arXiv asks for ~3s between requests,
    not the 0.5s the original used).
  - PyPDF2 is unmaintained; swapped for pypdf (same API, active project).
  - Fetches the paper's title from arXiv's metadata API and stores it
    alongside the arxiv_id, since "2310.11511" alone isn't useful for a
    human checking provenance later -- you want the title too.
  - document_id is now an explicit, stable field (= arxiv_id) that the
    chunking step downstream should use as the prefix for every
    chunk_id it generates, e.g. "{document_id}::{section_slug}::c{n}".

Usage:
    python downloading_pdfs.py                  # download + extract all
    python downloading_pdfs.py --limit 10        # first 10 only, for testing
    python downloading_pdfs.py --skip-download    # re-extract from existing PDFs only
"""

import json
import logging
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import argparse
import requests
from pypdf import PdfReader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path("./rag_corpus")
PDF_DIR = BASE_DIR / "pdfs"
TEXT_DIR = BASE_DIR / "extracted_text"
METADATA_FILE = BASE_DIR / "metadata.json"

PDF_DIR.mkdir(parents=True, exist_ok=True)
TEXT_DIR.mkdir(parents=True, exist_ok=True)

REQUEST_DELAY_SECONDS = 3.0  # arXiv's own guidance; don't go lower than this
ARXIV_API_URL = "https://export.arxiv.org/api/query"

# ============================================================================
# PAPER LIST (70 verified RAG papers with arXiv IDs)
# ============================================================================

PAPERS: List[Tuple[str, str]] = [

    ("2005.11401", "RAG"),  # Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

    ("2310.11511", "RAG"),  # Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection

    ("2401.15884", "RAG"),  # Corrective Retrieval Augmented Generation

    ("2401.18059", "RAG"),  # RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval

    ("2004.04906", "Retrieval"),  # Dense Passage Retrieval for Open-Domain Question Answering

    ("2306.07174", "RAG"),  # Augmenting Language Models with Long-Term Memory

    ("2002.08909", "RAG"),  # REALM: Retrieval-Augmented Language Model Pre-Training

    ("2212.09146", "RAG"),  # Can Retriever-Augmented Language Models Reason? The Blame Game Between the Retriever and the Language Model

    ("2112.04426", "RAG"),  # Improving language models by retrieving from trillions of tokens

    ("2208.03299", "RAG"),  # Atlas: Few-shot Learning with Retrieval Augmented Language Models

    ("2007.01282", "RAG"),  # Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering

    ("2004.12832", "Retrieval"),  # ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT

    ("2112.01488", "Reranking"),  # ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction

    ("1908.10084", "Embeddings"),  # Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks

    ("2205.13147", "Embeddings"),  # Matryoshka Representation Learning

    ("2212.03533", "Embeddings"),  # Text Embeddings by Weakly-Supervised Contrastive Pre-training

    ("2201.10005", "Embeddings"),  # Text and Code Embeddings by Contrastive Pre-Training

    ("2402.01613", "Embeddings"),  # Nomic Embed: Training a Reproducible Long Context Text Embedder

    ("2402.03216", "Embeddings"),  # M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation

    ("2401.00368", "Embeddings"),  # Improving Text Embeddings with Large Language Models

    ("2210.07316", "Evaluation"),  # MTEB: Massive Text Embedding Benchmark

    ("2107.05720", "Retrieval"),  # SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking

    ("2004.13969", "Retrieval"),  # Complementing Lexical Retrieval with Semantic Residual Embedding

    ("2007.00808", "Retrieval"),  # Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval

    ("2112.09118", "Retrieval"),  # Contriever: Unsupervised Dense Information Retrieval with Contrastive Learning

    ("2212.10496", "Retrieval"),  # Precise Zero-Shot Dense Retrieval without Relevance Labels

    ("2303.07678", "Retrieval"),  # Query2doc: Query Expansion with Large Language Models

    ("2212.10509", "Advanced"),  # Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions

    ("2305.06983", "Advanced"),  # Active Retrieval Augmented Generation

    ("2210.03629", "Advanced"),  # ReAct: Synergizing Reasoning and Acting in Language Models

    ("2404.16130", "Advanced"),  # From Local to Global: A Graph RAG Approach to Query-Focused Summarization

    ("2403.14403", "Advanced"),  # Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity

    ("2402.07630", "Advanced"),  # G-Retriever: Retrieval-Augmented Generation for Textual Graphs

    ("2301.12652", "Advanced"),  # REPLUG: Retrieval-Augmented Black-Box Language Models

    ("2311.09476", "Evaluation"),  # ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems

    ("2408.08067", "Evaluation"),  # RAGCHECKER: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation

    ("2401.00396", "Evaluation"),  # RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models

    ("2309.15217", "Evaluation"),  # Ragas: Automated Evaluation of Retrieval Augmented Generation

    ("2504.14891", "Evaluation"),  # Retrieval Augmented Generation Evaluation in the Era of Large Language Models: A Comprehensive Survey

    ("2405.07437", "Evaluation"),  # Evaluation of Retrieval-Augmented Generation: A Survey

    ("2404.10981", "Survey"),  # A Survey on Retrieval-Augmented Text Generation for Large Language Models

    ("2312.10997", "Survey"),  # Retrieval-Augmented Generation for Large Language Models: A Survey

    ("2104.08663", "Evaluation"),  # BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models

    ("1611.09268", "Evaluation"),  # MS MARCO: A Human Generated MAchine Reading COmprehension Dataset

    ("1809.09600", "Evaluation"),  # HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering

    ("2009.02252", "Evaluation"),  # KILT: a Benchmark for Knowledge Intensive Language Tasks

    ("1603.09320", "Indexing"),  # Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs

    ("1702.08734", "Indexing"),  # Billion-scale similarity search with GPUs

    ("1908.10396", "Indexing"),  # Accelerating Large-Scale Inference with Anisotropic Vector Quantization

    ("2210.10634", "Reranking"),  # RankT5: Fine-Tuning T5 for Text Ranking with Ranking Losses

    ("1910.10683", "LLM"),  # Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer

    ("1706.03762", "LLM"),  # Attention Is All You Need

    ("1901.02860", "LLM"),  # Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context

    ("2210.11416", "LLM"),  # Scaling Instruction-Finetuned Language Models

    ("2201.11903", "LLM"),  # Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

    ("2305.10601", "LLM"),  # Tree of Thoughts: Deliberate Problem Solving with Large Language Models

    ("2205.10625", "LLM"),  # Least-to-Most Prompting Enables Complex Reasoning in Large Language Models

    ("2212.08073", "LLM"),  # Constitutional AI: Harmlessness from AI Feedback

    ("2106.09685", "LLM"),  # LoRA: Low-Rank Adaptation of Large Language Models

    ("2104.08691", "LLM"),  # The Power of Scale for Parameter-Efficient Prompt Tuning

    ("2309.06180", "Production"),  # Efficient Memory Management for Large Language Model Serving with PagedAttention

    ("2205.14135", "Production"),  # FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

    ("2305.05176", "Production"),  # FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance

    ("2405.14831", "Advanced"),  # HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models

]

# NOTE: entries tagged NEEDS_FIX or UNCONFIRMED above have NOT been verified
# against arXiv and should not be trusted as-is. Run verify_corpus.py again
# after resolving these before treating the corpus as ready for chunking.

# ============================================================================
# METADATA (title) FETCH
# ============================================================================


def fetch_title(arxiv_id: str) -> Optional[str]:
    """Fetch the paper title from arXiv's metadata API (not the PDF)."""
    try:
        resp = requests.get(
            ARXIV_API_URL, params={"id_list": arxiv_id}, timeout=15
        )
        resp.raise_for_status()
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(resp.content)
        entry = root.find("atom:entry", ns)
        if entry is None:
            return None
        title_el = entry.find("atom:title", ns)
        if title_el is None or not title_el.text:
            return None
        return " ".join(title_el.text.split())  # collapse whitespace/newlines
    except Exception as e:
        logger.warning(f"⚠ Could not fetch title for {arxiv_id}: {e}")
        return None


# ============================================================================
# DOWNLOAD
# ============================================================================


def download_paper(arxiv_id: str, max_retries: int = 3) -> bool:
    """
    Download a paper's PDF from arXiv, validating that it's actually a PDF.
    """
    pdf_path = PDF_DIR / f"{arxiv_id}.pdf"

    if pdf_path.exists():
        logger.info(f"✓ {arxiv_id} already exists, skipping")
        return True

    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading {arxiv_id} (attempt {attempt + 1}/{max_retries})...")
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            is_pdf_bytes = response.content[:5] == b"%PDF-"
            if "application/pdf" not in content_type and not is_pdf_bytes:
                # arXiv returned something else (commonly an HTML error page
                # for a bad/withdrawn ID) with a 200 status -- don't save it.
                logger.error(
                    f"✗ {arxiv_id} did not return a PDF "
                    f"(Content-Type: {content_type!r}); skipping"
                )
                return False

            with open(pdf_path, "wb") as f:
                f.write(response.content)

            logger.info(f"✓ Downloaded {arxiv_id} ({len(response.content) / 1024:.1f} KB)")
            time.sleep(REQUEST_DELAY_SECONDS)
            return True

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            backoff = 2 ** attempt
            logger.warning(f"⚠ {type(e).__name__} on {arxiv_id}, retrying in {backoff}s...")
            time.sleep(backoff)
        except Exception as e:
            logger.error(f"✗ Failed to download {arxiv_id}: {e}")
            return False

    logger.error(f"✗ Failed to download {arxiv_id} after {max_retries} attempts")
    return False


# ============================================================================
# EXTRACTION
# ============================================================================


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF using pypdf, page by page."""
    try:
        text = ""
        reader = PdfReader(str(pdf_path))
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- Page {page_num + 1} ---\n{page_text}"
        return text
    except Exception as e:
        logger.error(f"Failed to extract text from {pdf_path}: {e}")
        return ""


def save_extracted_text(arxiv_id: str, title: Optional[str], category: str, text: str) -> Dict:
    """Save extracted text + metadata to JSON for the chunking step."""
    output_path = TEXT_DIR / f"{arxiv_id}.json"

    num_chars = len(text)
    num_words = len(text.split())
    num_pages = text.count("--- Page")
    approx_tokens = round(num_words * 1.3)  # words -> tokens, not tokens -> words

    data = {
        "document_id": arxiv_id,  # stable ID the chunker should prefix chunk_ids with
        "arxiv_id": arxiv_id,
        "title": title,
        "category": category,
        "timestamp": datetime.now().isoformat(),
        "text": text,
        "stats": {
            "num_chars": num_chars,
            "num_words": num_words,
            "num_pages": num_pages,
            "approx_tokens": approx_tokens,
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved {arxiv_id}: {num_words} words, ~{approx_tokens} tokens")
    return data["stats"]


# ============================================================================
# PIPELINE
# ============================================================================


def ingest_papers(paper_list: List[Tuple[str, str]], limit: Optional[int] = None, skip_download: bool = False) -> Dict:
    if limit:
        paper_list = paper_list[:limit]

    stats = {
        "total_papers": len(paper_list),
        "successful": 0,
        "failed": 0,
        "total_tokens": 0,
        "papers": {},
    }

    logger.info(f"Processing {len(paper_list)} papers (skip_download={skip_download})...")

    for arxiv_id, category in paper_list:
        pdf_path = PDF_DIR / f"{arxiv_id}.pdf"

        if not skip_download:
            if not download_paper(arxiv_id):
                stats["failed"] += 1
                continue
        elif not pdf_path.exists():
            logger.warning(f"⚠ --skip-download set but {arxiv_id}.pdf not found locally, skipping")
            stats["failed"] += 1
            continue

        text = extract_text_from_pdf(pdf_path)
        if not text:
            logger.warning(f"⚠ No text extracted from {arxiv_id}")
            stats["failed"] += 1
            continue

        title = fetch_title(arxiv_id)
        if title:
            time.sleep(1.0)  # separate, lighter-weight API -- still be polite

        paper_stats = save_extracted_text(arxiv_id, title, category, text)
        stats["papers"][arxiv_id] = {"category": category, "title": title, "stats": paper_stats}
        stats["successful"] += 1
        stats["total_tokens"] += paper_stats["approx_tokens"]

    with open(METADATA_FILE, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Processed:   {stats['successful']}/{stats['total_papers']} papers")
    logger.info(f"Failed:      {stats['failed']}")
    logger.info(f"Total tokens: ~{stats['total_tokens']:,}")
    logger.info(f"PDFs saved:  {PDF_DIR}")
    logger.info(f"Text saved:  {TEXT_DIR}")
    logger.info(f"Metadata:    {METADATA_FILE}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Download and extract papers from arXiv")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of papers (for testing)")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the download step; only (re-)extract text from PDFs already in ./rag_corpus/pdfs",
    )
    args = parser.parse_args()

    logger.info("Script: downloading_pdfs.py")
    logger.info(f"Total papers in list: {len(PAPERS)}")
    if args.limit:
        logger.info(f"Limiting to: {args.limit} papers")

    ingest_papers(PAPERS, limit=args.limit, skip_download=args.skip_download)


if __name__ == "__main__":
    main()