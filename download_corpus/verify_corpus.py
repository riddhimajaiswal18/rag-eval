"""
verify_corpus.py

Verifies the current corpus defined in download_pdfs.py.

The verifier reads the downloader's PAPERS literal directly, so the two
scripts cannot silently drift into separate ID lists. The expected local PDF
name is always: <arxiv_id>.pdf

Checks for each entry:
1. arXiv ID is present in the current corpus list.
2. Expected PDF file exists at download_corpus/rag_corpus/pdfs/<arxiv_id>.pdf.
3. The local file has PDF magic bytes.
4. The arXiv ID returns metadata.
5. The actual arXiv title is printed for provenance.

This version does NOT compare against the old intended-title list.
"""

import ast
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

PDF_DIR = Path(__file__).resolve().parent / "rag_corpus" / "pdfs"
ARXIV_API_URL = "https://export.arxiv.org/api/query"
REQUEST_DELAY_SECONDS = 3.0

def load_downloader_papers():
    """Read the downloader's literal PAPERS list without importing its dependencies."""
    downloader = Path(__file__).with_name("download_pdfs.py")
    tree = ast.parse(downloader.read_text(encoding="utf-8"), filename=str(downloader))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "PAPERS":
            papers = ast.literal_eval(node.value)
            if not isinstance(papers, list):
                raise ValueError("download_pdfs.py PAPERS must be a list")
            return papers
    raise ValueError("Could not find PAPERS in download_pdfs.py")


PAPERS = load_downloader_papers()


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
