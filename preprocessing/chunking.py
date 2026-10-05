#!/usr/bin/env python3
"""
chunking.py - RagAndEval V1 preprocessing: PDF -> GROBID TEI -> flat sections
-> sentence-aware chunks (<=1900 tokens under every benchmark tokenizer,
with 100-token sentence overlap) -> JSON.

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
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional
from xml.etree import ElementTree as ET

import requests
from transformers import AutoTokenizer

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
CHUNK_SIZE_TOKENS = 1900
OVERLAP_TOKENS = 100
MAX_SEQ_LEN = 2000
TOKENIZER_MODELS = {
    "yuan-0.8B-fp16": "IEITYuan/Yuan-embedding-2.0-en",
    "e5-mistral-7B-4bit": "intfloat/e5-mistral-7b-instruct",
    "qwen3-8B-4bit": "Qwen/Qwen3-Embedding-8B",
}
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


def element_children(el: ET.Element, name: str) -> List[ET.Element]:
    """Return descendants with a matching local tag name."""
    return [child for child in el.iter() if local_name(child.tag) == name]


def caption_text(el: ET.Element) -> str:
    """Build a readable caption from a TEI label and heading."""
    parts = []
    for name in ("label", "head"):
        child = next((c for c in el if local_name(c.tag) == name), None)
        value = clean_inline_text(child) if child is not None else ""
        if value:
            parts.append(value)
    return ": ".join(parts)


def table_parts(table: ET.Element, outer_caption: str = "", context: str = "") -> Dict:
    """Extract table context and rows for context-preserving chunking."""
    caption = caption_text(table)
    caption = ": ".join(part for part in (outer_caption, caption) if part)

    rows = element_children(table, "row")
    if not rows:
        content = clean_inline_text(table)
        return {"type": "table", "caption": caption, "context": context,
                "headers": [], "rows": [content] if content else []}

    table_rows = []
    for row in rows:
        cells = [cell for cell in row if local_name(cell.tag) == "cell"]
        values = [clean_inline_text(cell) for cell in cells]
        if any(values):
            is_header = (
                row.attrib.get("role", "").lower() in {"label", "head", "header"}
                or bool(cells) and all(
                    cell.attrib.get("role", "").lower() in {"label", "head", "header"}
                    for cell in cells
                )
            )
            table_rows.append((values, is_header))
    if not table_rows:
        return {"type": "table", "caption": caption, "headers": [], "rows": []}

    # GROBID commonly marks headers with role="label". When it does not,
    # use the first row as the header so values retain their column context.
    header_index = next((i for i, (_, is_header) in enumerate(table_rows) if is_header), None)
    header_index = 0 if header_index is None else header_index
    headers = table_rows[header_index][0]
    data_rows = [row for i, (row, _) in enumerate(table_rows) if i != header_index]
    return {"type": "table", "caption": caption, "context": context,
            "headers": headers, "rows": data_rows}


def figure_text(figure: ET.Element) -> str:
    """Keep a figure's caption and explanatory description searchable."""
    parts = []
    caption = caption_text(figure)
    if caption:
        parts.append("Figure: " + caption)
    descriptions = [clean_inline_text(child) for child in figure
                    if local_name(child.tag) in {"figDesc", "p"}]
    parts.extend("Description: " + value for value in descriptions if value)
    return "\n".join(parts)


def serialize_table(table: Dict) -> str:
    lines = [table["caption"]] if table.get("caption") else []
    if table.get("context"):
        lines.append("Description: " + table["context"])
    if table.get("headers"):
        lines.append("Headers: " + " | ".join(table["headers"]))
    lines.extend("Row: " + " | ".join(row) for row in table.get("rows", []))
    return "\n".join(lines)


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


def direct_body_blocks(div: ET.Element) -> List[Dict]:
    """Read section blocks in document order, including nested div content."""
    blocks: List[Dict] = []

    def visit(parent: ET.Element, include_head: bool = False) -> None:
        for child in parent:
            name = local_name(child.tag)
            if name == "div":
                visit(child, include_head=True)
            elif name == "head" and include_head:
                value = clean_inline_text(child)
                if value:
                    blocks.append({"type": "text", "text": value})
            elif name == "figure":
                cap = caption_text(child)
                desc = " ".join(clean_inline_text(c) for c in child
                                if local_name(c.tag) == "figDesc")
                tables = [c for c in child if local_name(c.tag) == "table"]
                if tables:
                    blocks.extend(table_parts(table, cap, desc) for table in tables)
                else:
                    value = figure_text(child)
                    if value:
                        blocks.append({"type": "text", "text": value, "figure": True})
            elif name == "table":
                blocks.append(table_parts(child))
            elif name in {"p", "formula", "note", "item"}:
                value = clean_inline_text(child)
                if value:
                    blocks.append({"type": "text", "text": value})
            elif name in {"list", "listBibl"}:
                for item in child:
                    if local_name(item.tag) in {"item", "p"}:
                        value = clean_inline_text(item)
                        if value:
                            blocks.append({"type": "text", "text": value})

    visit(div)
    return blocks


def extract_flat_sections(root: ET.Element) -> List[Dict]:
    """Flat list of top-level sections (abstract + body divs), references dropped."""
    sections: List[Dict] = []

    abstract = extract_abstract(root)
    if abstract:
        sections.append({"section_title": "Abstract", "section_index": len(sections),
                         "text": abstract, "blocks": [{"type": "text", "text": abstract}]})

    body = root.find(".//tei:text/tei:body", NS)
    if body is None:
        raise RuntimeError("GROBID TEI contains no <body> element.")

    for div in body.findall("tei:div", NS):
        head = div.find("tei:head", NS)
        title = clean_inline_text(head) if head is not None else ""
        if title.lower() in REFERENCE_HEADINGS:
            logger.info("Skipping References section.")
            continue
        blocks = direct_body_blocks(div)
        text = "\n\n".join(
            serialize_table(block) if block["type"] == "table" else block["text"]
            for block in blocks
        ).strip()
        if text:
            sections.append({"section_title": title or "Body", "section_index": len(sections),
                             "text": text, "blocks": blocks})

    if not sections:  # fallback: loose paragraphs directly under <body>
        paras = [t for t in (clean_inline_text(p) for p in body.findall("tei:p", NS)) if t]
        if paras:
            text = "\n\n".join(paras)
            sections.append({"section_title": "Body", "section_index": len(sections),
                             "text": text, "blocks": [{"type": "text", "text": text}]})

    if not sections:
        raise RuntimeError("No usable sections could be extracted from TEI.")
    return sections


# -------------------------------------------------------------------- chunking
def split_sentences(text: str) -> List[str]:
    text = normalize_space(text)
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text) if p.strip()] if text else []


class TokenBudget:
    """Count with each retrieval model tokenizer and enforce the largest count."""

    def __init__(self, tokenizers: Dict[str, object]):
        self.tokenizers = tokenizers

    @lru_cache(maxsize=100_000)
    def _count_cached(self, text: str) -> int:
        return max(len(tokenizer.encode(text, add_special_tokens=True))
                   for tokenizer in self.tokenizers.values())

    def count(self, text: str) -> int:
        return self._count_cached(text)


def load_tokenizers() -> Dict[str, object]:
    return {
        label: AutoTokenizer.from_pretrained(model_name)
        for label, model_name in TOKENIZER_MODELS.items()
    }


def split_large_sentence(sentence: str, budget: TokenBudget, chunk_size: int) -> List[str]:
    """Split oversized text at whitespace while respecting every model tokenizer."""
    remaining = sentence.strip()
    pieces: List[str] = []
    while remaining:
        if budget.count(remaining) <= chunk_size:
            pieces.append(remaining)
            break
        low, high, best = 1, len(remaining), 0
        while low <= high:
            middle = (low + high) // 2
            if budget.count(remaining[:middle].strip()) <= chunk_size:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        if best == 0:
            raise ValueError("A single character exceeds the configured chunk token limit.")
        boundary = remaining.rfind(" ", 0, best + 1)
        if boundary > 0:
            best = boundary
        piece = remaining[:best].strip()
        if not piece:
            piece = remaining[:best].strip()
        pieces.append(piece)
        remaining = remaining[best:].strip()
    return pieces


def _overlap_suffix(text: str, budget: TokenBudget, overlap_size: int) -> str:
    sentences = split_sentences(text)
    selected: List[str] = []
    for sentence in reversed(sentences):
        candidate = " ".join([sentence, *selected])
        if budget.count(candidate) > overlap_size:
            break
        selected.insert(0, sentence)
    if selected:
        return " ".join(selected)
    # Preserve a short tail even when the last sentence exceeds the overlap budget.
    low, high, best = 1, len(text), 0
    while low <= high:
        middle = (low + high) // 2
        suffix = text[-middle:].strip()
        if budget.count(suffix) <= overlap_size:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    if best:
        suffix = text[-best:].strip()
        boundary = suffix.find(" ")
        return suffix[boundary + 1:] if boundary >= 0 else suffix
    return ""


def chunk_section(section: Dict, budget: TokenBudget, chunk_size: int = CHUNK_SIZE_TOKENS,
                  overlap_size: int = OVERLAP_TOKENS) -> List[Dict]:
    """Pack semantic text units under the largest of the benchmark token counts."""
    atoms: List[str] = []
    blocks = section.get("blocks") or [{"type": "text", "text": section.get("text", "")}]
    block_index = 0
    while block_index < len(blocks):
        block = blocks[block_index]
        if block.get("type") == "table":
            serialized = serialize_table(block)
            if budget.count(serialized) <= chunk_size:
                atoms.append(serialized)
            else:
                context = [block["caption"]] if block.get("caption") else []
                if block.get("context"):
                    context.append("Description: " + block["context"])
                if block.get("headers"):
                    context.append("Headers: " + " | ".join(block["headers"]))
                context_text = "\n".join(context)
                rows = block.get("rows", [])
                if not rows:
                    atoms.extend(split_large_sentence(context_text, budget, chunk_size))
                for row in rows:
                    row_text = "Row: " + " | ".join(row)
                    value = "\n".join(part for part in (context_text, row_text) if part)
                    atoms.extend(split_large_sentence(value, budget, chunk_size))
        elif block.get("figure"):
            value = block.get("text", "")
            next_block = blocks[block_index + 1] if block_index + 1 < len(blocks) else None
            if next_block and next_block.get("type") != "table" and not next_block.get("figure"):
                value = "\n\n".join(part for part in (value, next_block.get("text", "")) if part)
                block_index += 1
            atoms.extend(split_large_sentence(value, budget, chunk_size))
        else:
            for sentence in split_sentences(block.get("text", "")):
                atoms.extend(split_large_sentence(sentence, budget, chunk_size))
        block_index += 1

    base: List[str] = []
    current: List[str] = []
    for atom in atoms:
        candidate = " ".join(current + [atom]).strip()
        if current and budget.count(candidate) > chunk_size:
            base.append(" ".join(current).strip())
            current = [atom]
        else:
            current.append(atom)
    if current:
        base.append(" ".join(current).strip())

    chunks: List[Dict] = []
    for index, text in enumerate(base):
        overlap = _overlap_suffix(base[index - 1], budget, overlap_size) if index else ""
        if overlap:
            candidate = overlap + " " + text
            if budget.count(candidate) <= chunk_size:
                text = candidate
        count = budget.count(text)
        if count > chunk_size:
            raise ValueError(f"Chunk contains {count} tokens; configured maximum is {chunk_size}.")
        chunks.append({"text": text, "token_count": count,
                       "overlap_prev_tokens": budget.count(overlap) if overlap else 0})
    return chunks


def build_chunk_document(structured: Dict, budget: TokenBudget) -> Dict:
    doc_id = structured["document_id"]

    all_chunks: List[Dict] = []

    for section in structured["sections"]:
        title = section["section_title"] or "Body"
        slug = slugify(title)

        for i, chunk in enumerate(
            chunk_section(section, budget)
        ):
            all_chunks.append({
                "chunk_id": f"{doc_id}::s{section['section_index']:03d}::{slug}::c{i:03d}",
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
            "tokenizer": "maximum count across benchmark model tokenizers",
            "tokenizers": TOKENIZER_MODELS,
            "max_sequence_length": MAX_SEQ_LEN,
            "overlap_unit": "sentence",
        },

        "chunks": all_chunks,
    }


# ------------------------------------------------------------------- pipeline
def process_paper(document_id: str, budget: TokenBudget, grobid_url: str, overwrite: bool) -> Dict:
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
    result = build_chunk_document(structured, budget)

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

    budget = TokenBudget(load_tokenizers())
    bar = "=" * 70
    for line in (bar, "RAG CHUNKING", bar,
                 f"PDF directory: {PDF_DIR}", f"TEI directory: {TEI_DIR}",
                 f"Output directory: {CHUNKS_DIR}", f"Papers: {len(document_ids)}",
                 f"Chunk size: at most {CHUNK_SIZE_TOKENS} tokens under all benchmark tokenizers",
                 f"Overlap: {OVERLAP_TOKENS} tokens", f"Embedding input limit: {MAX_SEQ_LEN}",
                 "Hierarchy: disabled (V1)", "Embeddings: not generated here", ""):
        logger.info(line)

    needs_grobid = args.overwrite or any(not (TEI_DIR / f"{doc_id}.tei.xml").exists()
                                         for doc_id in document_ids)
    if needs_grobid:
        check_grobid(args.grobid_url)

    ok = total_sections = total_chunks = 0
    failures: List[tuple] = []
    for doc_id in document_ids:
        try:
            r = process_paper(doc_id, budget, args.grobid_url, args.overwrite)
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
