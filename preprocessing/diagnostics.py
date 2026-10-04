#!/usr/bin/env python3
"""Audit TEI-to-chunk retention and benchmark-model token lengths.

Run with ``uv run python preprocessing/diagnostics.py``. Reads cached GROBID
TEI and chunk JSON files, then loads only the benchmark tokenizers (not the
embedding models). Counts include each tokenizer's special tokens. Benchmark
corpus inputs have no model-specific prefixes and all three scripts set a
512-token SentenceTransformers limit.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
TEI_DIR = ROOT / "preprocessing" / "data" / "grobid_tei"
CHUNKS_DIR = ROOT / "preprocessing" / "data" / "chunks"
MODELS = {
    "Yuan": "IEITYuan/Yuan-embedding-2.0-en",
    "E5": "intfloat/e5-mistral-7b-instruct",
    "Qwen": "Qwen/Qwen3-Embedding-8B",
}
EFFECTIVE_LIMIT = 512
NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def quantile(values: list[int], p: float) -> float:
    """Linear-interpolated percentile (same convention as NumPy default)."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    index = (len(ordered) - 1) * p
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def text(el: ET.Element | None) -> str:
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def tei_inventory(path: Path) -> dict[str, Any]:
    root = ET.fromstring(path.read_bytes())
    body = root.find(".//tei:text/tei:body", NS)
    divs = body.findall(".//tei:div", NS) if body is not None else []
    top = body.findall("tei:div", NS) if body is not None else []
    nested = [d for d in divs if d not in top]
    return {
        "top_level_sections": [text(d.find("tei:head", NS)) or "Body" for d in top],
        "nested_sections": [text(d.find("tei:head", NS)) or "(untitled)" for d in nested],
        "paragraphs": len(root.findall(".//tei:text/tei:body//tei:p", NS)),
        "tables": len(root.findall(".//tei:text/tei:body//tei:table", NS)),
        "figures": len(root.findall(".//tei:text/tei:body//tei:figure", NS)),
        "equations": len(root.findall(".//tei:text/tei:body//tei:formula", NS)),
    }


def load_tokenizers():
    from transformers import AutoTokenizer
    return {
        label: AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        for label, name in MODELS.items()
    }


def audit(tokenizers) -> dict[str, Any]:
    reports = []
    lengths = {label: [] for label in tokenizers}
    for chunk_path in sorted(CHUNKS_DIR.glob("*_chunks.json")):
        data = json.loads(chunk_path.read_text(encoding="utf-8"))
        tei_path = TEI_DIR / f"{data['document_id']}.tei.xml"
        inventory = tei_inventory(tei_path) if tei_path.exists() else None
        chunks = data.get("chunks", [])
        per_model_lengths = {
            label: [len(tokenizer.encode(c["text"], add_special_tokens=True)) for c in chunks]
            for label, tokenizer in tokenizers.items()
        }
        for label, counts in per_model_lengths.items():
            lengths[label].extend(counts)
        retained = {
            "sections": data.get("num_sections", 0),
            "chunks": len(chunks),
            "chunk_sections": sorted({c.get("section_title", "") for c in chunks}),
            "characters": sum(len(c.get("text", "")) for c in chunks),
        }
        reports.append({"paper": data["document_id"], **(inventory or {"tei": "missing"}), "retained_by_chunking": retained})
    distributions = {}
    for label, values in lengths.items():
        distributions[label] = {
            "checkpoint": MODELS[label],
            "effective_limit": EFFECTIVE_LIMIT,
            "chunks": len(values),
            "min": min(values) if values else None,
            "median": median(values) if values else None,
            "p95": quantile(values, .95) if values else None,
            "p99": quantile(values, .99) if values else None,
            "max": max(values) if values else None,
            ">400": sum(n > 400 for n in values),
            ">480": sum(n > 480 for n in values),
            ">limit-32": sum(n > EFFECTIVE_LIMIT - 32 for n in values),
            ">limit": sum(n > EFFECTIVE_LIMIT for n in values),
        }
    return {"token_counts_include_special_tokens": True,
            "corpus_prefixes": "none for all three benchmark configurations",
            "distributions": distributions, "papers": reports}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Also write the complete report as JSON.")
    args = parser.parse_args()
    report = audit(load_tokenizers())
    print("Benchmark tokenizer audit (counts include special tokens; effective limit 512)")
    for label, stats in report["distributions"].items():
        print(f"\n{label}: {stats['checkpoint']}")
        for key, value in stats.items():
            if key != "checkpoint":
                print(f"{key}: {value}")
    print("\nPer-paper TEI inventory and chunk retention:")
    for paper in report["papers"]:
        print(json.dumps(paper, ensure_ascii=False))
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
