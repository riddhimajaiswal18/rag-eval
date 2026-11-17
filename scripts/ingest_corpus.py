#!/usr/bin/env python3
"""Acquire, prepare, embed, and index the arXiv corpus with one command."""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_step(description: str, command: list[str]) -> None:
    print(f"\n{description}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, help="Embedding batch size; defaults to the existing pipeline setting.")
    parser.add_argument("--arxiv-source", choices=("grobid", "extracted-text"), default="grobid")
    args = parser.parse_args()

    run_step("Downloading and extracting the arXiv corpus", [
        sys.executable, "download_corpus/download_pdfs.py"
    ])
    run_step("Chunking the arXiv corpus", [
        sys.executable, "preprocessing/chunking.py", "--source", args.arxiv_source
    ])
    embedding_command = [sys.executable, "preprocessing/embeddings.py"]
    if args.batch_size is not None:
        embedding_command.extend(("--batch-size", str(args.batch_size)))
    run_step("Generating or resuming arXiv embeddings", embedding_command)
    run_step("Indexing arXiv into its Qdrant collection", [
        sys.executable, "retrieval/qdrant_setup.py"
    ])
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    raise SystemExit(main())
