#!/usr/bin/env python3
"""Evaluate evidence retrieval against PeerQA's mapped source sentence IDs."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from peerqa.corpus import EVIDENCE_MAP_PATH, QUESTIONS_PATH, read_status
from retrieval.qdrant_setup import collection_name_for
from retrieval.search import Retriever


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def reciprocal_rank(relevant: list[bool]) -> float:
    for rank, value in enumerate(relevant, 1):
        if value:
            return 1.0 / rank
    return 0.0


def ndcg(relevant: list[bool], k: int) -> float:
    dcg = sum((1.0 / math.log2(rank + 1)) for rank, value in enumerate(relevant[:k], 1) if value)
    ideal_count = min(k, sum(relevant))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / idcg if idcg else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dense", "bm25", "hybrid"), default="hybrid")
    parser.add_argument("--limit", type=int, default=10, help="Retrieve this many chunks per question.")
    parser.add_argument("--max-questions", type=int, help="Optional cap for a quick smoke evaluation.")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    if not QUESTIONS_PATH.exists() or not EVIDENCE_MAP_PATH.exists():
        raise FileNotFoundError("Run scripts/ingest_corpus.py --corpus peerqa --prepare-only first.")
    load_dotenv(ROOT / ".env")
    status = read_status()
    version = (status.get("index") or {}).get("corpus_version")
    if not version:
        raise RuntimeError("PeerQA is not indexed. Run scripts/ingest_corpus.py --corpus peerqa first.")

    questions = iter_jsonl(QUESTIONS_PATH)
    mappings = {row["question_id"]: row for row in iter_jsonl(EVIDENCE_MAP_PATH)}
    all_evaluable = [q for q in questions if q.get("document_id") and mappings.get(q["question_id"], {}).get("gold_sentence_ids")]
    candidates = all_evaluable
    if args.max_questions:
        candidates = candidates[:args.max_questions]
    retriever = Retriever(collection_name_for("peerqa"), version)
    per_question: list[dict[str, Any]] = []
    for question in candidates:
        mapping = mappings[question["question_id"]]
        gold = set(mapping["gold_sentence_ids"])
        # Only the question text and known document ID reach retrieval; gold data
        # is examined after the hit list is returned.
        hits = retriever.search(
            question["question"], args.mode, args.limit, document_id=question["document_id"]
        )
        relevant = [bool(gold.intersection(hit.get("sentence_ids", []))) for hit in hits]
        hit_sentence_ids = {
            sentence_id
            for hit in hits
            for sentence_id in hit.get("sentence_ids", [])
        }
        per_question.append({
            "question_id": question["question_id"],
            "original_paper_id": question["original_paper_id"],
            "answerable": question.get("answerable"),
            "answerable_mapped": question.get("answerable_mapped"),
            "gold_sentence_count": len(gold),
            "unmapped_evidence_count": len(mapping.get("unmapped_evidence", [])),
            "recall": len(gold & hit_sentence_ids) / len(gold),
            "precision_at_k": sum(relevant[:args.limit]) / args.limit,
            "mrr": reciprocal_rank(relevant),
            "ndcg": ndcg(relevant, args.limit),
        })
    if not per_question:
        raise RuntimeError("No included PeerQA questions have mapped gold evidence to score.")
    metrics = {
        metric: sum(row[metric] for row in per_question) / len(per_question)
        for metric in ("recall", "precision_at_k", "mrr", "ndcg")
    }
    report = {
        "corpus_id": "peerqa",
        "corpus_version": version,
        "mode": args.mode,
        "k": args.limit,
        "evaluated_questions": len(per_question),
        "questions_with_mapped_evidence_in_corpus": len(all_evaluable),
        "questions_skipped_for_evidence_metrics": len(questions) - len(all_evaluable),
        "questions_sampled": len(candidates),
        "questions_with_unmapped_evidence": sum(bool(m.get("unmapped_evidence")) for m in mappings.values()),
        "metrics": metrics,
        "per_question": per_question,
        "gold_answers_and_evidence_were_not_passed_to_retrieval": True,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
