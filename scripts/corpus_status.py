#!/usr/bin/env python3
"""Print local ingestion/indexing status for PeerQA or arXiv."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peerqa.corpus import read_status

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--corpus", choices=("peerqa", "arxiv"), default="peerqa")
args = parser.parse_args()
if args.corpus == "peerqa":
    result = read_status()
else:
    result = {
        "corpus_id": "arxiv",
        "pdfs": len(list((ROOT / "download_corpus" / "rag_corpus" / "pdfs").glob("*.pdf"))),
        "extracted_documents": len(list((ROOT / "download_corpus" / "rag_corpus" / "extracted_text").glob("*.json"))),
        "chunks": len(list((ROOT / "preprocessing" / "data" / "chunks").glob("*_chunks.json"))),
        "note": "Use the existing arXiv metadata and qdrant_setup.py --verify-only for indexed point totals.",
    }
print(json.dumps(result, ensure_ascii=False, indent=2))
