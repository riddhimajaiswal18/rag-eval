#!/usr/bin/env python3
"""Acquire, prepare, embed, and index a corpus with one command."""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peerqa.corpus import acquire_and_build, read_status


def run_step(description: str, command: list[str]) -> None:
    print(f"\n{description}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=("peerqa", "arxiv"), required=True)
    parser.add_argument("--status", action="store_true", help="Print local corpus status without downloading or indexing.")
    parser.add_argument("--force-download", action="store_true", help="Re-download the official PeerQA archive.")
    parser.add_argument("--batch-size", type=int, help="Embedding batch size; defaults to the existing pipeline setting.")
    parser.add_argument("--prepare-only", action="store_true", help="Acquire and normalize data without embeddings or Qdrant indexing.")
    parser.add_argument("--arxiv-source", choices=("grobid", "extracted-text"), default="grobid")
    args = parser.parse_args()

    if args.status:
        if args.corpus == "peerqa":
            print(json.dumps(read_status(), ensure_ascii=False, indent=2))
        else:
            print(json.dumps({"corpus_id": "arxiv", "status": "see existing download/preprocessing manifests"}, indent=2))
        return 0

    if args.corpus == "peerqa":
        manifest = acquire_and_build(force_download=args.force_download)
        print(
            "PeerQA preparation complete: "
            f"{manifest['counts']['papers']} papers, "
            f"{manifest['counts']['labelled_questions']} labelled questions, "
            f"{manifest['counts']['chunks']} paragraph chunks."
        )
        if args.prepare_only:
            return 0
        embedding_command = [sys.executable, "preprocessing/embeddings.py", "--corpus", "peerqa"]
        if args.batch_size is not None:
            embedding_command.extend(("--batch-size", str(args.batch_size)))
        run_step("Generating or resuming PeerQA embeddings", embedding_command)
        run_step("Indexing PeerQA into its separate Qdrant collection", [
            sys.executable, "retrieval/qdrant_setup.py", "--corpus", "peerqa"
        ])
        return 0

    run_step("Downloading and extracting the existing arXiv corpus", [
        sys.executable, "download_corpus/download_pdfs.py"
    ])
    run_step("Chunking the existing arXiv corpus", [
        sys.executable, "preprocessing/chunking.py", "--source", args.arxiv_source
    ])
    embedding_command = [sys.executable, "preprocessing/embeddings.py", "--corpus", "arxiv"]
    if args.batch_size is not None:
        embedding_command.extend(("--batch-size", str(args.batch_size)))
    run_step("Generating or resuming arXiv embeddings", embedding_command)
    run_step("Indexing arXiv into its existing Qdrant collection", [
        sys.executable, "retrieval/qdrant_setup.py", "--corpus", "arxiv"
    ])
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    raise SystemExit(main())
