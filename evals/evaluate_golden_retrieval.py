"""Evaluate paper-level retrieval against the draft golden JSONL datasets."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval import qdrant_setup
from retrieval.search import Retriever


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def paper_id(value: Any) -> str:
    result = str(value)
    return result.removeprefix("arxiv:")


def score_record(record: dict[str, Any], hits: list[dict[str, Any]], k: int) -> dict[str, Any]:
    relevant = {paper_id(item) for item in record["relevant_doc_ids"]}
    top_hits = hits[:k]
    retrieved_ids = [paper_id(item.get("document_id", "")) for item in top_hits]
    if record.get("answerability") == "unanswerable":
        return {
            "query_id": record.get("query_id"),
            "answerability": "unanswerable",
            "retrieved_anything_at_k": bool(top_hits),
            "retrieved_paper_ids": retrieved_ids,
        }
    true_positives = sum(item in relevant for item in retrieved_ids)
    precision = true_positives / k
    recall = len(set(retrieved_ids) & relevant) / len(relevant) if relevant else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    required_sets = record.get("required_doc_sets", [])
    required_set_covered = any(
        {paper_id(item) for item in required} <= set(retrieved_ids)
        for required in required_sets
    ) if required_sets else None
    return {
        "query_id": record.get("query_id"),
        "answerability": record.get("answerability", "answerable"),
        "challenge_tags": record.get("challenge_tags", []),
        "precision_at_k": precision,
        "recall_at_k": recall,
        "f1_at_k": f1,
        "required_doc_set_covered": required_set_covered,
        "retrieved_paper_ids": retrieved_ids,
    }


def evaluate(path: Path, retriever: Retriever, mode: str, k: int) -> dict[str, Any]:
    records = read_jsonl(path)
    results = []
    for record in records:
        hits = retriever.search(record["query"], mode=mode, limit=k)
        results.append(score_record(record, hits, k))
    summary: dict[str, Any] = {"dataset": path.name, "queries": len(results), "k": k, "mode": mode}
    scored = [row for row in results if "precision_at_k" in row]
    summary["answerable_queries_scored"] = len(scored)
    summary["unanswerable_queries"] = len(results) - len(scored)
    for metric in ("precision_at_k", "recall_at_k", "f1_at_k"):
        summary[metric] = sum(row[metric] for row in scored) / len(scored) if scored else 0.0
    coverage = [row["required_doc_set_covered"] for row in scored if row["required_doc_set_covered"] is not None]
    if coverage:
        summary["required_doc_set_coverage"] = sum(coverage) / len(coverage)
    unanswerable = [row for row in results if row.get("answerability") == "unanswerable"]
    if unanswerable:
        summary["unanswerable_retrieval_rate"] = sum(row["retrieved_anything_at_k"] for row in unanswerable) / len(unanswerable)
    summary["per_query"] = results
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", type=Path, help="Golden JSONL dataset paths")
    parser.add_argument("--mode", choices=("dense", "bm25", "hybrid"), default="hybrid")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--version", help="Corpus version; inferred from the local chunk cache by default")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be at least 1")

    version = args.version
    if not version:
        chunks, _vectors, metadata = qdrant_setup.load_and_validate_cache()
        version = qdrant_setup.corpus_version(chunks, metadata)
    if not version:
        raise RuntimeError("Could not determine corpus version; pass --version explicitly")

    retriever = Retriever(qdrant_setup.collection_name_for(), version)
    report = [evaluate(path, retriever, args.mode, args.k) for path in args.datasets]
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
