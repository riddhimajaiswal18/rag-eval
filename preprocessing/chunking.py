#!/usr/bin/env python3
"""
chunking.py - RagAndEval V1 preprocessing: PDF -> GROBID TEI -> flat sections
-> sentence-aware token chunks (400 tok, ~100 tok sentence overlap) -> JSON.

INPUT:    download_corpus/rag_corpus/pdfs/   (metadata: .../extracted_text/)
OUTPUT:   preprocessing/data/grobid_tei/, preprocessing/data/chunks/
GROBID:   http://localhost:8070

    uv run python preprocessing/chunking.py --ids 2005.11401 2310.11511 --overwrite
    uv run python preprocessing/chunking.py --overwrite
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional
from xml.etree import ElementTree as ET

import requests
import tiktoken

# ---------------------------------------------------------------- config/paths
ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "download_corpus" / "rag_corpus"
PDF_DIR = CORPUS_DIR / "pdfs"
METADATA_DIR = CORPUS_DIR / "extracted_text"
DATA_DIR = ROOT / "preprocessing" / "data"
TEI_DIR = DATA_DIR / "grobid_tei"
CHUNKS_DIR = DATA_DIR / "chunks"
TEI_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_GROBID_URL = "http://localhost:8070"
CHUNK_SIZE_TOKENS = 400
OVERLAP_TOKENS = 100
ENCODING_NAME = "cl100k_base"
HIERARCHY_ENABLED = False  # V1: flat sections only
REFERENCE_HEADINGS = {"references", "bibliography", "works cited", "literature cited"}
NS = {"tei": "http://www.tei-c.org/ns/1.0"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- helpers
def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def slugify(text: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", normalize_space(text).lower())).strip("-")[:80] or "section"


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def clean_inline_text(el: ET.Element) -> str:
    return normalize_space("".join(el.itertext()))


# ---------------------------------------------------------------- PDF/metadata
def _find_one(directory: Path, document_id: str, ext: str, exact: List[Path]) -> Optional[Path]:
    for p in exact:
        if p.exists():
            return p
    matches = list(directory.glob(f"*{document_id}*.{ext}"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1 and ext == "pdf":
        raise RuntimeError(f"Multiple PDFs matched {document_id}: {[p.name for p in matches]}")
    return None


def find_pdf(document_id: str) -> Path:
    path = _find_one(PDF_DIR, document_id, "pdf",
                     [PDF_DIR / f"{document_id}.pdf", PDF_DIR / f"{document_id.replace('.', '_')}.pdf"])
    if path is None:
        raise FileNotFoundError(f"Could not find PDF for {document_id} in:\n{PDF_DIR}")
    return path


def load_metadata(document_id: str, pdf_path: Path) -> Dict:
    path = _find_one(METADATA_DIR, document_id, "json", [METADATA_DIR / f"{document_id}.json"])
    if path is None:
        logger.warning("No metadata JSON found for %s", document_id)
        return {"document_id": document_id, "arxiv_id": document_id, "title": "",
                "category": "", "source_file": pdf_path.name}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "document_id": data.get("document_id", document_id),
        "arxiv_id": data.get("arxiv_id", document_id),
        "title": data.get("title", ""),
        "category": data.get("category", ""),
        "timestamp": data.get("timestamp"),
        "source_file": pdf_path.name,
    }


# ---------------------------------------------------------------------- GROBID
def check_grobid(grobid_url: str) -> None:
    try:
        requests.get(grobid_url.rstrip("/") + "/api/isalive", timeout=10).raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"GROBID is not reachable at {grobid_url}.\n"
                           f"Start GROBID before running chunking.py.") from exc
    logger.info("GROBID is alive.")


def process_pdf_with_grobid(pdf_path: Path, grobid_url: str, timeout: int = 300, retries: int = 3) -> bytes:
    url = grobid_url.rstrip("/") + "/api/processFulltextDocument"
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            with pdf_path.open("rb") as f:
                resp = requests.post(url, files={"input": (pdf_path.name, f, "application/pdf")},
                                     data={"segmentSentences": "1"}, timeout=timeout)
            if resp.status_code != 200:
                raise RuntimeError(f"GROBID returned HTTP {resp.status_code}: {resp.text[:500]}")
            if b"<TEI" not in resp.content and b":TEI" not in resp.content:
                raise RuntimeError("GROBID returned HTTP 200 but the response does not appear to be TEI XML.")
            return resp.content
        except (OSError, requests.RequestException, RuntimeError) as exc:
            last_error = exc
            logger.warning("GROBID attempt %d/%d failed for %s: %s", attempt, retries, pdf_path.name, exc)
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"GROBID failed after {retries} attempts for {pdf_path.name}: {last_error}")


def generate_tei(document_id: str, pdf_path: Path, grobid_url: str, overwrite: bool) -> Path:
    """Generate TEI XML (cached unless overwrite)."""
    tei_path = TEI_DIR / f"{document_id}.tei.xml"
    if tei_path.exists() and not overwrite:
        logger.info("Using cached TEI: %s", tei_path.name)
        return tei_path
    logger.info("Running GROBID on %s...", pdf_path.name)
    tei_path.write_bytes(process_pdf_with_grobid(pdf_path, grobid_url))
    logger.info("Saved TEI: %s", tei_path)
    return tei_path


# ------------------------------------------------------------------ TEI parsing
def extract_title(root: ET.Element) -> str:
    title = root.find(".//tei:teiHeader//tei:titleStmt/tei:title", NS)
    return clean_inline_text(title) if title is not None else ""


def extract_abstract(root: ET.Element) -> str:
    abstract = root.find(".//tei:text/tei:front/tei:abstract", NS)
    if abstract is None:
        abstract = root.find(".//tei:abstract", NS)
    if abstract is None:
        return ""
    paras = [t for t in (clean_inline_text(p) for p in abstract.findall(".//tei:p", NS)) if t]
    return "\n\n".join(paras) if paras else clean_inline_text(abstract)


def direct_body_blocks(div: ET.Element) -> List[str]:
    """Text directly in this section; nested <div>s are skipped (flat V1)."""
    blocks: List[str] = []
    for child in div:
        name = local_name(child.tag)
        if name in {"p", "formula", "note", "item"}:
            items = [child]
        elif name in {"list", "listBibl"}:
            items = [i for i in child if local_name(i.tag) in {"item", "p"}]
        else:
            continue
        blocks += [t for t in (clean_inline_text(i) for i in items) if t]
    return blocks


def extract_flat_sections(root: ET.Element) -> List[Dict]:
    """Flat list of top-level sections (abstract + body divs), references dropped."""
    sections: List[Dict] = []

    def add(title: str, text: str) -> None:
        sections.append({
    "section_title": title,
    "section_index": len(sections),
    "text": text,
})

    abstract = extract_abstract(root)
    if abstract:
        add("Abstract", abstract)

    body = root.find(".//tei:text/tei:body", NS)
    if body is None:
        raise RuntimeError("GROBID TEI contains no <body> element.")

    for div in body.findall("tei:div", NS):
        head = div.find("tei:head", NS)
        title = clean_inline_text(head) if head is not None else ""
        if title.lower() in REFERENCE_HEADINGS:
            logger.info("Skipping References section.")
            continue
        text = "\n\n".join(direct_body_blocks(div)).strip()
        if text:
            add(title or "Body", text)

    if not sections:  # fallback: loose paragraphs directly under <body>
        paras = [t for t in (clean_inline_text(p) for p in body.findall("tei:p", NS)) if t]
        if paras:
            add("Body", "\n\n".join(paras))

    if not sections:
        raise RuntimeError("No usable sections could be extracted from TEI.")
    return sections


# -------------------------------------------------------------------- chunking
def split_sentences(text: str) -> List[str]:
    text = normalize_space(text)
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text) if p.strip()] if text else []


def split_large_sentence(sentence: str, encoding, chunk_size: int) -> List[str]:
    """Last resort: hard-split a sentence longer than chunk_size."""
    toks = encoding.encode(sentence)
    pieces = (encoding.decode(toks[i:i + chunk_size]).strip() for i in range(0, len(toks), chunk_size))
    return [p for p in pieces if p]


def chunk_section(section: Dict, encoding, chunk_size: int = CHUNK_SIZE_TOKENS,
                  overlap_size: int = OVERLAP_TOKENS) -> List[Dict]:
    """Pack sentences into <=chunk_size chunks, prepending sentence-level overlap
    from the previous chunk only if the result still fits in chunk_size."""
    ntok = lambda s: len(encoding.encode(s))

    sentences: List[str] = []
    for s in split_sentences(section["text"].strip()):
        sentences += [s] if ntok(s) <= chunk_size else split_large_sentence(s, encoding, chunk_size)

    # 1) pack
    base: List[str] = []
    cur: List[str] = []
    cur_tokens = 0
    for s in sentences:
        n = ntok(s)
        if cur and cur_tokens + n > chunk_size:
            base.append(" ".join(cur).strip())
            cur, cur_tokens = [], 0
        cur.append(s)
        cur_tokens += n
    if cur:
        base.append(" ".join(cur).strip())

    # 2) overlap
    chunks: List[Dict] = []
    for i, text in enumerate(base):
        if i > 0:
            overlap: List[str] = []
            used = 0
            for s in reversed(split_sentences(base[i - 1])):
                n = ntok(s)
                if used + n > overlap_size:
                    break
                overlap.insert(0, s)
                used += n
            if overlap:
                candidate = " ".join(overlap) + " " + text
                if ntok(candidate) <= chunk_size:
                    text = candidate
        count = ntok(text)
        chunks.append({"text": text, "token_count": count,
                       "overlap_prev_tokens": 0 if i == 0 else min(overlap_size, count)})
    return chunks


def build_chunk_document(structured: Dict, encoding) -> Dict:
    doc_id = structured["document_id"]

    all_chunks: List[Dict] = []

    for section in structured["sections"]:
        title = section["section_title"] or "Body"
        slug = slugify(title)

        for i, chunk in enumerate(
            chunk_section(section, encoding)
        ):
            all_chunks.append({
                "chunk_id": f"{doc_id}::{slug}::c{i:03d}",
                "section_title": title,
                "section_index": section["section_index"],
                "chunk_index": i,
                "text": chunk["text"],
                "token_count": chunk["token_count"],
            })

    return {
        "document_id": doc_id,
        "num_sections": len(structured["sections"]),
        "num_chunks": len(all_chunks),

        "chunking_config": {
            "chunk_size_tokens": CHUNK_SIZE_TOKENS,
            "overlap_tokens": OVERLAP_TOKENS,
            "tokenizer": ENCODING_NAME,
            "overlap_unit": "sentence",
        },

        "chunks": all_chunks,
    }


# ------------------------------------------------------------------- pipeline
def process_paper(document_id: str, encoding, grobid_url: str, overwrite: bool) -> Dict:
    pdf_path = find_pdf(document_id)
    logger.info("PDF: %s", pdf_path.name)
    metadata = load_metadata(document_id, pdf_path)
    tei_path = generate_tei(document_id, pdf_path, grobid_url, overwrite)

    try:
        root = ET.fromstring(tei_path.read_bytes())
    except ET.ParseError as exc:
        raise RuntimeError(f"Invalid TEI XML:\n{tei_path}") from exc

    structured = {
    "document_id": metadata.get(
        "document_id",
        document_id,
    ),
    "sections": extract_flat_sections(root),
}
    result = build_chunk_document(structured, encoding)

    (CHUNKS_DIR / f"{document_id}_chunks.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("✓ %s: %d sections → %d chunks", document_id, result["num_sections"], result["num_chunks"])
    return result


def available_ids() -> List[str]:
    return [p.stem for p in sorted(PDF_DIR.glob("*.pdf"))]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="GROBID-backed RAG chunking pipeline")
    ap.add_argument("--ids", nargs="+", help="Specific document IDs to process.")
    ap.add_argument("--grobid-url", default=DEFAULT_GROBID_URL, help=f"GROBID URL (default: {DEFAULT_GROBID_URL})")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-run GROBID and overwrite existing TEI and chunk files.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not PDF_DIR.exists():
        raise FileNotFoundError(f"PDF directory does not exist:\n{PDF_DIR}")
    document_ids = args.ids or available_ids()
    if not document_ids:
        raise RuntimeError(f"No PDFs found in:\n{PDF_DIR}")

    encoding = tiktoken.get_encoding(ENCODING_NAME)
    bar = "=" * 70
    for line in (bar, "RAG CHUNKING", bar,
                 f"PDF directory: {PDF_DIR}", f"TEI directory: {TEI_DIR}",
                 f"Output directory: {CHUNKS_DIR}", f"Papers: {len(document_ids)}",
                 f"Chunk size: {CHUNK_SIZE_TOKENS} tokens", f"Overlap: {OVERLAP_TOKENS} tokens",
                 "Hierarchy: disabled (V1)", "Embeddings: not generated here", ""):
        logger.info(line)

    check_grobid(args.grobid_url)

    ok = total_sections = total_chunks = 0
    failures: List[tuple] = []
    for doc_id in document_ids:
        try:
            r = process_paper(doc_id, encoding, args.grobid_url, args.overwrite)
            ok += 1
            total_sections += r["num_sections"]
            total_chunks += r["num_chunks"]
        except Exception as exc:
            failures.append((doc_id, str(exc)))
            logger.exception("✗ Failed %s", doc_id)

    for line in ("", bar, "CHUNKING COMPLETE", bar, f"Successful: {ok}", f"Failed: {len(failures)}",
                 f"Total sections: {total_sections}", f"Total chunks: {total_chunks}",
                 f"TEI output: {TEI_DIR}", f"Chunk output: {CHUNKS_DIR}"):
        logger.info(line)

    if failures:
        logger.error("\nFailed papers:")
        for doc_id, err in failures:
            logger.error("  %s: %s", doc_id, err)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
