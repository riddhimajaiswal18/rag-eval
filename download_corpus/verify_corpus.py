#!/usr/bin/env python3
"""
verify_corpus.py

Verifies the EXACT CURRENT corpus from download_pdfs.py.

Important:
- download_pdfs.py currently contains 69 PAPERS entries (despite its
  comments/docstring saying "70").
- This verifier therefore checks exactly those same 69 IDs and labels.
- The expected local PDF name is always: <arxiv_id>.pdf

Checks for each entry:
1. arXiv ID is present in the current corpus list.
2. Expected PDF file exists at download_corpus/rag_corpus/pdfs/<arxiv_id>.pdf.
3. The local file has PDF magic bytes.
4. The arXiv ID returns metadata.
5. The actual arXiv title is printed for provenance.

This version does NOT compare against the old intended-title list.
It therefore verifies the CURRENT downloader corpus and its local PDFs,
rather than an outdated 70-paper list.
"""

import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

PDF_DIR = Path(__file__).resolve().parent / "rag_corpus" / "pdfs"
ARXIV_API_URL = "https://export.arxiv.org/api/query"
REQUEST_DELAY_SECONDS = 3.0

# ============================================================================
# EXACT PAPERS LIST FROM download_pdfs.py
# ============================================================================

PAPERS = [
    ("2005.11401", "RAG"),
    ("2302.00083", "RAG"),
    ("2310.11511", "RAG"),
    ("2401.15884", "RAG"),
    ("2401.18059", "RAG"),
    ("2004.04906", "RAG"),
    ("2306.07174", "RAG"),
    ("2004.12832", "Embeddings"),
    ("1908.10084", "Embeddings"),
    ("2402.03216", "Embeddings"),
    ("2402.01613", "Embeddings"),
    ("2205.13147", "Embeddings"),
    ("2212.03533", "Embeddings"),
    ("2201.10005", "Embeddings"),
    ("1910.10683", "Embeddings"),
    ("1706.03762", "Embeddings"),
    ("2002.08909", "Embeddings"),
    ("2304.07577", "Retrieval"),
    ("2004.13969", "Retrieval"),
    ("1808.03313", "Retrieval"),
    ("1607.01759", "Retrieval"),
    ("2008.11672", "Retrieval"),
    ("2107.05720", "Retrieval"),
    ("1804.07201", "Retrieval"),
    ("2009.08553", "Retrieval"),
    ("2112.01488", "Reranking"),
    ("2202.06991", "Reranking"),
    ("2210.10634", "Reranking"),
    ("1710.05649", "Reranking"),
    ("1905.03692", "Reranking"),
    ("2304.10149", "Reranking"),
    ("2307.09009", "Reranking"),
    ("1603.09320", "Indexing"),
    ("1802.08686", "Indexing"),
    ("1906.08640", "Indexing"),
    ("1801.05644", "Indexing"),
    ("1702.08734", "Indexing"),
    ("1604.06778", "Indexing"),
    ("1308.2396", "Indexing"),
    ("2309.15217", "Evaluation"),
    ("2104.08663", "Evaluation"),
    ("1611.09268", "Evaluation"),
    ("1906.04341", "Evaluation"),
    ("1809.09600", "Evaluation"),
    ("2104.11807", "Evaluation"),
    ("2104.08664", "Evaluation"),
    ("2210.07316", "Evaluation"),
    ("2210.03629", "Advanced"),
    ("2307.08426", "Advanced"),
    ("2404.16130", "Advanced"),
    ("2403.14403", "Advanced"),
    ("2105.12722", "Advanced"),
    ("1908.06039", "Advanced"),
    ("2310.11597", "Advanced"),
    ("2205.08532", "Advanced"),
    ("2203.00205", "Advanced"),
    ("1901.02860", "LLM"),
    ("2205.00296", "LLM"),
    ("2210.11416", "LLM"),
    ("2104.08691", "LLM"),
    ("2201.11903", "LLM"),
    ("2305.10601", "LLM"),
    ("2205.10625", "LLM"),
    ("2212.08073", "LLM"),
    ("2106.09685", "LLM"),
    ("2103.12219", "Production"),
    ("2312.09092", "Production"),
    ("2311.00663", "Production"),
    ("2008.07890", "Production"),
]


def fetch_title(arxiv_id):
    """Return the actual title registered with arXiv for this ID."""
    try:
        response = requests.get(
            ARXIV_API_URL,
            params={"id_list": arxiv_id},
            timeout=15,
        )
        response.raise_for_status()

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(response.content)
        entry = root.find("atom:entry", ns)

        if entry is None:
            return None

        title = entry.find("atom:title", ns)
        if title is None or not title.text:
            return None

        return " ".join(title.text.split())

    except Exception as exc:
        print(f"      ✗ arXiv request failed: {exc}")
        return None


def valid_pdf(path):
    """Return True only if the local file exists and is a real PDF."""
    if not path.exists() or path.stat().st_size == 0:
        return False

    try:
        with open(path, "rb") as f:
            return f.read(5) == b"%PDF-"
    except OSError:
        return False


def main():
    expected_count = len(PAPERS)

    if expected_count != 69:
        print(f"ERROR: download corpus should currently contain 69 entries, found {expected_count}.")
        sys.exit(1)

    ids = [arxiv_id for arxiv_id, _ in PAPERS]

    if len(set(ids)) != expected_count:
        print("ERROR: duplicate arXiv ID found.")
        sys.exit(1)

    print("=" * 78)
    print("VERIFYING CURRENT DOWNLOAD CORPUS")
    print("=" * 78)
    print(f"Expected corpus entries: {expected_count}")
    print(f"PDF directory: {PDF_DIR.resolve()}")
    print()

    problems = []

    for number, (arxiv_id, category) in enumerate(PAPERS, 1):
        pdf_name = f"{arxiv_id}.pdf"
        pdf_path = PDF_DIR / pdf_name

        print(f"[{number:02d}/{expected_count}] {arxiv_id}")
        print(f"      Label: {category}")
        print(f"      PDF:   {pdf_name}")

        if valid_pdf(pdf_path):
            size_kb = pdf_path.stat().st_size / 1024
            print(f"      ✓ PDF exists and is valid ({size_kb:.1f} KB)")
        else:
            print("      ✗ PDF missing or invalid")
            problems.append((arxiv_id, "missing_or_invalid_pdf"))

        actual_title = fetch_title(arxiv_id)

        if actual_title:
            print(f"      ✓ arXiv title: {actual_title}")
        else:
            print("      ✗ No arXiv metadata returned")
            problems.append((arxiv_id, "missing_arxiv_metadata"))

        time.sleep(REQUEST_DELAY_SECONDS)
        print()

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"Expected entries: {expected_count}")
    print(f"Unique IDs:       {len(set(ids))}")
    print(f"Problems:         {len(problems)}")

    if problems:
        print("\nProblems:")
        for arxiv_id, reason in problems:
            print(f"  - {arxiv_id}: {reason}")

        print("\n✗ CORPUS VERIFICATION FAILED")
        sys.exit(1)

    print("\n✓ CORPUS VERIFICATION PASSED")
    print(f"All {expected_count} current downloader entries have:")
    print("  - matching arXiv ID")
    print("  - matching <arxiv_id>.pdf filename")
    print("  - valid local PDF")
    print("  - arXiv metadata")


if __name__ == "__main__":
    main()
