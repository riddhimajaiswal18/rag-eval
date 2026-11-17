#!/usr/bin/env python3
"""Print local arXiv acquisition and chunk counts."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "download_corpus" / "rag_corpus"

result = {
    "corpus_id": "arxiv",
    "pdfs": len(list((CORPUS_DIR / "pdfs").glob("*.pdf"))),
    "extracted_documents": len(list((CORPUS_DIR / "extracted_text").glob("*.json"))),
    "chunks": len(list((ROOT / "preprocessing" / "data" / "chunks").glob("*_chunks.json"))),
    "note": "Use retrieval/qdrant_setup.py --verify-only to verify the indexed corpus version and point count.",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
