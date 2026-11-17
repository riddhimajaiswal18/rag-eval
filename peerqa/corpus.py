"""Download the official PeerQA release and build a provenance-preserving corpus.

This module acquires the official dataset archive and EGU PDFs with a
source-specific URL rule supplied by the PeerQA repository. It does not
download OpenReview papers or infer PDF URLs from titles. Other PDFs require
explicit per-paper source and license records.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import requests

from . import CORPUS_ID, DATASET_VERSION

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "download_corpus" / "peerqa"
SOURCE_DIR = CORPUS_DIR / "source"
ARCHIVE_PATH = CORPUS_DIR / "peerqa-data-v1.0.zip"
PAPERS_PATH = SOURCE_DIR / "papers.jsonl"
QA_PATH = SOURCE_DIR / "qa.jsonl"
DOCS_DIR = CORPUS_DIR / "documents"
CHUNKS_DIR = CORPUS_DIR / "chunks"
QUESTIONS_PATH = CORPUS_DIR / "questions.jsonl"
EVIDENCE_MAP_PATH = CORPUS_DIR / "evidence_mapping.jsonl"
MANIFEST_PATH = CORPUS_DIR / "manifest.json"
STATUS_PATH = CORPUS_DIR / "index_status.json"
ACQUISITION_STATUS_PATH = CORPUS_DIR / "acquisition_status.json"

OFFICIAL_ARCHIVE_URL = (
    "https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/4467/"
    "peerqa-data-v1.0.zip?sequence=1&isAllowed=y"
)
OFFICIAL_ARCHIVE_URLS = (
    OFFICIAL_ARCHIVE_URL,
    "https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/4467/"
    "peerqa-data-v1.0.zip?sequence=5&isAllowed=y",
)
EXPECTED_PAPERS = 90
EXPECTED_LABELLED_QAS = 579
EXPECTED_PAPER_SOURCES = {"nlpeer": 82, "egu": 8}
MAX_RETRIES = 4
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
USER_AGENT = "RagAndEval/0.1 (PeerQA research corpus acquisition)"
LOGGER = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_document_id(original_paper_id: str) -> str:
    """Return a stable canonical ID derived solely from the official paper ID."""
    digest = hashlib.sha256(original_paper_id.encode("utf-8")).hexdigest()
    return f"peerqa:{digest}"


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write(path, (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path.name}:{line_no}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object at {path.name}:{line_no}.")
            yield record


def _valid_archive(path: Path) -> bool:
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        with zipfile.ZipFile(path) as archive:
            return archive.testzip() is None and all(
                any(Path(name).name == required for name in archive.namelist())
                for required in ("papers.jsonl", "qa.jsonl")
            )
    except (OSError, zipfile.BadZipFile):
        return False


def download_archive(force: bool = False) -> dict[str, Any]:
    """Fetch and validate the official v1.0 ZIP, resuming from a valid local copy."""
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    if not force and _valid_archive(ARCHIVE_PATH):
        return {"path": str(ARCHIVE_PATH), "sha256": sha256_file(ARCHIVE_PATH), "status": "reused"}

    temp_path = ARCHIVE_PATH.with_suffix(ARCHIVE_PATH.suffix + ".tmp")
    headers = {"User-Agent": USER_AGENT}
    last_error: Exception | None = None
    for source_url in OFFICIAL_ARCHIVE_URLS:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                with requests.get(source_url, headers=headers, stream=True, timeout=(15, 120)) as response:
                    if response.status_code >= 400 and response.status_code not in RETRYABLE_STATUS:
                        last_error = requests.HTTPError(
                            f"HTTP {response.status_code} from official PeerQA source {source_url}"
                        )
                        LOGGER.warning("PeerQA source unavailable: %s", last_error)
                        temp_path.unlink(missing_ok=True)
                        break
                    if response.status_code in RETRYABLE_STATUS:
                        raise requests.HTTPError(
                            f"retryable HTTP {response.status_code}", response=response
                        )
                    response.raise_for_status()
                    total = 0
                    with temp_path.open("wb") as handle:
                        for block in response.iter_content(1024 * 256):
                            if block:
                                handle.write(block)
                                total += len(block)
                        handle.flush()
                        os.fsync(handle.fileno())
                    expected = response.headers.get("Content-Length")
                    if expected and total != int(expected):
                        raise IOError(f"Incomplete download: received {total} of {expected} bytes.")
                if not _valid_archive(temp_path):
                    raise zipfile.BadZipFile("Downloaded archive failed ZIP CRC or required-file checks.")
                os.replace(temp_path, ARCHIVE_PATH)
                result = {
                    "path": str(ARCHIVE_PATH),
                    "sha256": sha256_file(ARCHIVE_PATH),
                    "source_url": source_url,
                    "status": "downloaded",
                }
                _atomic_json(ACQUISITION_STATUS_PATH, {"corpus_id": CORPUS_ID, **result})
                return result
            except (OSError, requests.RequestException, zipfile.BadZipFile) as exc:
                last_error = exc
                temp_path.unlink(missing_ok=True)
                if attempt == MAX_RETRIES:
                    break
                wait = min(2 ** attempt, 30)
                LOGGER.warning("PeerQA archive attempt %d/%d failed (%s); retrying in %ds", attempt, MAX_RETRIES, exc, wait)
                time.sleep(wait)
    failure = {
        "corpus_id": CORPUS_ID,
        "dataset_version": DATASET_VERSION,
        "status": "acquisition_failed",
        "sources_tried": list(OFFICIAL_ARCHIVE_URLS),
        "error": str(last_error),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _atomic_json(ACQUISITION_STATUS_PATH, failure)
    raise RuntimeError(f"Could not acquire official PeerQA archive after {len(OFFICIAL_ARCHIVE_URLS)} official source URLs: {last_error}")


def _safe_zip_member(name: str) -> str | None:
    """Return a required basename only for safe regular members of the ZIP."""
    posix_path = PurePosixPath(name)
    if posix_path.is_absolute() or ".." in posix_path.parts:
        raise ValueError(f"Unsafe path in PeerQA archive: {name!r}")
    base = posix_path.name
    return base if base in {"papers.jsonl", "qa.jsonl", "qa-unlabeled.jsonl"} else None


def extract_official_files(archive_path: Path = ARCHIVE_PATH) -> dict[str, str]:
    """Extract only the official paper and labelled QA JSONL files."""
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    checksums: dict[str, str] = {}
    with zipfile.ZipFile(archive_path) as archive:
        members: dict[str, str] = {}
        for name in archive.namelist():
            base = _safe_zip_member(name)
            if base and not name.endswith("/"):
                if base in members:
                    raise ValueError(f"Duplicate {base} entries in PeerQA archive.")
                members[base] = name
        for required in ("papers.jsonl", "qa.jsonl"):
            if required not in members:
                raise FileNotFoundError(f"Official PeerQA archive does not contain {required}.")
        for base, member in members.items():
            data = archive.read(member)
            path = SOURCE_DIR / base
            if not path.exists() or sha256_file(path) != hashlib.sha256(data).hexdigest():
                _atomic_write(path, data)
            checksums[base] = hashlib.sha256(data).hexdigest()
    return checksums


def _normalize_source_record(raw: dict[str, Any]) -> dict[str, Any]:
    for key in ("paper_id", "idx", "content"):
        if key not in raw:
            raise ValueError(f"PeerQA paper row is missing required field {key!r}.")
    paper_id = str(raw["paper_id"])
    if not paper_id.strip():
        raise ValueError("PeerQA paper row has an empty paper_id.")
    idx = raw["idx"]
    if not isinstance(idx, int):
        raise ValueError(f"PeerQA row idx must be an integer for {paper_id!r}.")
    pidx = raw.get("pidx")
    sidx = raw.get("sidx")
    if pidx is not None and not isinstance(pidx, int):
        raise ValueError(f"PeerQA row pidx must be an integer or null for {paper_id!r}.")
    if sidx is not None and not isinstance(sidx, int):
        raise ValueError(f"PeerQA row sidx must be an integer or null for {paper_id!r}.")
    content = raw.get("content")
    if not isinstance(content, str):
        content = "" if content is None else str(content)
    return {
        "paper_id": paper_id,
        "idx": idx,
        "pidx": pidx,
        "sidx": sidx,
        "type": str(raw.get("type") or "sentence"),
        "content": content,
        "last_heading": str(raw.get("last_heading") or ""),
    }


def _section_id(document_id: str, heading: str, first_pidx: int) -> str:
    identity = f"{document_id}\x1f{heading.strip()}\x1f{first_pidx}"
    return f"{document_id}:sec:{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:16]}"


def _row_sentence_id(document_id: str, row: dict[str, Any]) -> str:
    pidx = row["pidx"] if row["pidx"] is not None else "na"
    sidx = row["sidx"] if row["sidx"] is not None else "na"
    return f"{document_id}:i{row['idx']}:p{pidx}:s{sidx}"


def _build_document(original_paper_id: str, source_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    document_id = canonical_document_id(original_paper_id)
    rows = sorted(source_rows, key=lambda row: row["idx"])
    title = next((row["content"].strip() for row in rows if row["type"].lower() == "title" and row["content"].strip()), original_paper_id)
    idx_map: dict[int, dict[str, Any]] = {}
    paragraphs: dict[int, dict[str, Any]] = {}
    for row in rows:
        idx = row["idx"]
        if idx in idx_map:
            raise ValueError(f"Duplicate official idx={idx} for paper_id={original_paper_id!r}.")
        sentence_id = _row_sentence_id(document_id, row)
        location = {"idx": idx, "pidx": row["pidx"], "sidx": row["sidx"]}
        normalized = {
            "sentence_id": sentence_id,
            "sentence_index": row["sidx"],
            "source_text": row["content"],
            "source_location": location,
            "source_type": row["type"],
        }
        idx_map[idx] = normalized
        pidx = row["pidx"] if row["pidx"] is not None else idx
        paragraph = paragraphs.setdefault(pidx, {
            "paragraph_id": f"{document_id}:p{pidx}",
            "original_pidx": row["pidx"],
            "section_title": row["last_heading"] or "Body",
            "records": [],
        })
        if not paragraph["section_title"] and row["last_heading"]:
            paragraph["section_title"] = row["last_heading"]
        if row["content"].strip() and row["type"].lower() not in {"title", "heading"}:
            paragraph["records"].append(normalized)

    sections: list[dict[str, Any]] = []
    for pidx in sorted(paragraphs):
        paragraph = paragraphs[pidx]
        heading = paragraph["section_title"] or "Body"
        if not paragraph["records"]:
            continue
        # The first paragraph index makes repeated headings distinct and stable.
        # A heading reappearing after another section is a distinct section.
        previous_heading = sections[-1]["section_title"] if sections else None
        if previous_heading != heading:
            section = {
                "section_id": _section_id(document_id, heading, pidx),
                "section_title": heading,
                "paragraphs": [],
            }
            sections.append(section)
        else:
            section = sections[-1]
        paragraph["section_id"] = section["section_id"]
        section["paragraphs"].append(paragraph)

    document = {
        "corpus_id": CORPUS_ID,
        "document_id": document_id,
        "original_paper_id": original_paper_id,
        "title": title,
        "source": original_paper_id.split("/", 1)[0],
        "dataset_version": DATASET_VERSION,
        "extraction_method": "peerqa_official_papers_jsonl",
        "sections": sections,
        "source_record_count": len(rows),
    }
    return document, idx_map


def _chunks_for_document(document: dict[str, Any], max_words: int = 900) -> dict[str, Any]:
    """Create deterministic paragraph chunks while preserving sentence provenance."""
    chunks: list[dict[str, Any]] = []
    for section in document["sections"]:
        for paragraph in section["paragraphs"]:
            records = paragraph["records"]
            current: list[dict[str, Any]] = []
            current_words = 0
            part = 0

            def flush() -> None:
                nonlocal current, current_words, part
                if not current:
                    return
                text = " ".join(record["source_text"].strip() for record in current if record["source_text"].strip())
                refs = [record["source_location"] for record in current]
                sentence_ids = [record["sentence_id"] for record in current]
                chunks.append({
                    "chunk_id": f"{document['document_id']}::{paragraph['paragraph_id']}::c{part:03d}",
                    "corpus_id": CORPUS_ID,
                    "document_id": document["document_id"],
                    "original_paper_id": document["original_paper_id"],
                    "title": document["title"],
                    "section_id": section["section_id"],
                    "section_title": section["section_title"],
                    "paragraph_id": paragraph["paragraph_id"],
                    "sentence_ids": sentence_ids,
                    "source_refs": refs,
                    "source_text": text,
                    "source_location": refs,
                    "extraction_method": document["extraction_method"],
                    "text": text,
                    "token_count": len(text.split()),
                    "chunk_index": len(chunks),
                    "paragraph_part_index": part,
                })
                part += 1
                current = []
                current_words = 0

            for record in records:
                words = max(1, len(record["source_text"].split()))
                if words > max_words:
                    flush()
                    source_words = record["source_text"].split()
                    for offset in range(0, len(source_words), max_words):
                        fragment = {**record, "source_text": " ".join(source_words[offset : offset + max_words])}
                        current.append(fragment)
                        current_words = len(fragment["source_text"].split())
                        flush()
                    continue
                if current and current_words + words > max_words:
                    flush()
                current.append(record)
                current_words += words
            flush()
    return {
        "corpus_id": CORPUS_ID,
        "document_id": document["document_id"],
        "original_paper_id": document["original_paper_id"],
        "title": document["title"],
        "chunking_config": {"strategy": "paragraph", "max_words": max_words, "overlap_words": 0, "version": 1},
        "num_chunks": len(chunks),
        "chunks": chunks,
    }


def map_question_evidence(
    qa: dict[str, Any],
    documents: dict[str, tuple[dict[str, Any], dict[int, dict[str, Any]]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Map official evidence indices exactly; retain every unmapped source reference."""
    paper_id = str(qa.get("paper_id", ""))
    in_corpus = paper_id in documents
    idx_map = documents[paper_id][1] if in_corpus else {}
    mapped_evidence = qa.get("answer_evidence_mapped")
    mappings: list[dict[str, Any]] = []
    gold_sentence_ids: list[str] = []
    missing_refs: list[dict[str, Any]] = []

    if isinstance(mapped_evidence, list) and mapped_evidence:
        for evidence_index, evidence in enumerate(mapped_evidence):
            if not isinstance(evidence, dict):
                missing_refs.append({"evidence_index": evidence_index, "reason": "invalid_evidence_record", "raw": evidence})
                continue
            idx_values = evidence.get("idx")
            if idx_values is None:
                idx_values = [None]
            elif not isinstance(idx_values, list):
                idx_values = [idx_values]
            for original_idx in idx_values:
                ref = {
                    "evidence_index": evidence_index,
                    "evidence_sentence": evidence.get("sentence"),
                    "idx": original_idx,
                    "pidx": None,
                    "sidx": None,
                    "sentence_id": None,
                    "status": "unmapped",
                }
                if not in_corpus:
                    ref["status"] = "paper_outside_90_paper_corpus"
                elif original_idx is None:
                    ref["status"] = "official_evidence_has_no_idx"
                elif isinstance(original_idx, int) and original_idx in idx_map:
                    source = idx_map[original_idx]
                    ref.update({
                        "pidx": source["source_location"]["pidx"],
                        "sidx": source["source_location"]["sidx"],
                        "sentence_id": source["sentence_id"],
                        "status": "mapped",
                    })
                    gold_sentence_ids.append(source["sentence_id"])
                else:
                    ref["status"] = "source_idx_not_found"
                mappings.append(ref)
                if ref["status"] != "mapped":
                    missing_refs.append(ref.copy())
    else:
        missing_refs.append({
            "reason": "no_official_mapped_evidence_record",
            "status": "unmapped_or_unavailable",
            "source_value": mapped_evidence,
        })

    unique_gold = list(dict.fromkeys(gold_sentence_ids))
    mapped_record = {
        "corpus_id": CORPUS_ID,
        "question_id": qa.get("question_id"),
        "original_paper_id": paper_id,
        "document_id": documents[paper_id][0]["document_id"] if in_corpus else None,
        "source_split": qa.get("split", "test"),
        "answerable": qa.get("answerable"),
        "answerable_mapped": qa.get("answerable_mapped"),
        "evidence_mappings": mappings,
        "gold_sentence_ids": unique_gold,
        "mapped_evidence_count": len(unique_gold),
        "unmapped_evidence": missing_refs,
        "evidence_mapping_status": "mapped" if unique_gold else ("outside_corpus" if not in_corpus else "unmapped_or_empty"),
    }
    question_record = {
        "corpus_id": CORPUS_ID,
        "question_id": qa.get("question_id"),
        "original_paper_id": paper_id,
        "document_id": mapped_record["document_id"],
        "source_split": mapped_record["source_split"],
        "question": qa.get("question"),
        "raw_answer_evidence": qa.get("raw_answer_evidence"),
        "answer_evidence_sent": qa.get("answer_evidence_sent"),
        "answer_free_form": qa.get("answer_free_form"),
        "answerable": qa.get("answerable"),
        "answerable_mapped": qa.get("answerable_mapped"),
        "answer_evidence_mapped": qa.get("answer_evidence_mapped"),
        "gold_sentence_ids": unique_gold,
        "unmapped_evidence": missing_refs,
    }
    return question_record, mapped_record


def _license_note() -> str:
    return (
        "PeerQA dataset release is CC BY-NC-SA 4.0. This does not establish the "
        "license or redistribution permission for the original paper PDF."
    )


def _official_egu_pdf_source(paper_id: str) -> dict[str, Any] | None:
    """Build the source URL used by the official PeerQA EGU downloader."""
    parts = paper_id.split("/")
    if len(parts) != 3 or parts[0] != "egu":
        return None
    journal, article_code = parts[1], parts[2]
    code_parts = article_code.split("-")
    if len(code_parts) != 3 or not all(part.isdigit() for part in code_parts):
        return None
    volume, page, year = code_parts
    if not re.fullmatch(r"[a-z0-9]+", journal, flags=re.IGNORECASE):
        return None
    url = (
        f"https://{journal}.copernicus.org/articles/{volume}/{page}/{year}/"
        f"{journal}-{article_code}.pdf"
    )
    return {
        "original_paper_id": paper_id,
        "pdf_url": url,
        "license_allows_download": True,
        "license_note": (
            "Official EGU/Copernicus publication policy: articles are distributed under "
            "CC BY 4.0 (older content may use the preceding CC BY 3.0 license)."
        ),
        "license_source": "https://publications.copernicus.org/for_authors/licence_and_copyright.html",
        "source_script": "https://github.com/UKPLab/PeerQA/blob/main/download_egu.py",
        "seconds_between_requests": 1,
    }


def build_corpus(*, expected_papers: int | None = EXPECTED_PAPERS, expected_qas: int | None = EXPECTED_LABELLED_QAS) -> dict[str, Any]:
    """Normalize all 90 official text papers and retain mappings for all labelled QAs."""
    paper_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in _iter_jsonl(PAPERS_PATH):
        row = _normalize_source_record(raw)
        paper_rows[row["paper_id"]].append(row)
    if expected_papers is not None and len(paper_rows) != expected_papers:
        raise ValueError(f"Expected {expected_papers} distinct papers in official papers.jsonl; found {len(paper_rows)}.")
    source_counts = Counter(paper_id.split("/", 1)[0] for paper_id in paper_rows)
    if expected_papers == EXPECTED_PAPERS and dict(source_counts) != EXPECTED_PAPER_SOURCES:
        raise ValueError(
            "Official 90-paper subset has unexpected source composition: "
            f"expected {EXPECTED_PAPER_SOURCES}, found {dict(source_counts)}."
        )

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    docs_for_evidence: dict[str, tuple[dict[str, Any], dict[int, dict[str, Any]]]] = {}
    paper_manifest: list[dict[str, Any]] = []
    total_paragraphs = 0
    total_sentences = 0
    total_chunks = 0
    for original_paper_id, rows in sorted(paper_rows.items()):
        document, idx_map = _build_document(original_paper_id, rows)
        docs_for_evidence[original_paper_id] = (document, idx_map)
        chunks = _chunks_for_document(document)
        safe_id = hashlib.sha256(original_paper_id.encode("utf-8")).hexdigest()
        _atomic_json(DOCS_DIR / f"{safe_id}.json", document)
        _atomic_json(CHUNKS_DIR / f"{safe_id}_chunks.json", chunks)
        paragraphs = [p for section in document["sections"] for p in section["paragraphs"]]
        sentences = [
            r for p in paragraphs for r in p["records"]
            if r.get("source_type", "").casefold() == "sentence"
        ]
        source_digest = hashlib.sha256(
            "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows).encode("utf-8")
        ).hexdigest()
        paper_manifest.append({
            "corpus_id": CORPUS_ID,
            "dataset_version": DATASET_VERSION,
            "original_paper_id": original_paper_id,
            "document_id": document["document_id"],
            "title": document["title"],
            "source": document["source"],
            "original_pdf_url": None,
            "pdf_download_status": "not_attempted_no_official_url_or_paper_license_record",
            "license_or_availability": _license_note(),
            "extraction_status": "extracted_from_official_papers_jsonl",
            "extraction_method": document["extraction_method"],
            "dataset_inclusion_status": "included_in_90_paper_subset",
            "source_rows": len(rows),
            "source_sha256": source_digest,
            "paragraphs": len(paragraphs),
            "sentences": len(sentences),
            "chunks": chunks["num_chunks"],
        })
        total_paragraphs += len(paragraphs)
        total_sentences += len(sentences)
        total_chunks += chunks["num_chunks"]

    questions: list[dict[str, Any]] = []
    evidence_maps: list[dict[str, Any]] = []
    seen_question_ids: set[str] = set()
    for qa in _iter_jsonl(QA_PATH):
        question_id = str(qa.get("question_id", ""))
        if not question_id:
            raise ValueError("A labelled PeerQA record is missing question_id.")
        if question_id in seen_question_ids:
            raise ValueError(f"Duplicate PeerQA question_id {question_id!r}.")
        seen_question_ids.add(question_id)
        q_record, mapping = map_question_evidence(qa, docs_for_evidence)
        questions.append(q_record)
        evidence_maps.append(mapping)
    if expected_qas is not None and len(questions) != expected_qas:
        raise ValueError(f"Expected {expected_qas} labelled QA records; found {len(questions)}.")

    _atomic_write(QUESTIONS_PATH, "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in questions).encode("utf-8"))
    _atomic_write(EVIDENCE_MAP_PATH, "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in evidence_maps).encode("utf-8"))
    qa_inside = sum(q["document_id"] is not None for q in questions)
    successfully_mapped = sum(bool(q["gold_sentence_ids"]) for q in questions)
    manifest = {
        "corpus_id": CORPUS_ID,
        "dataset_version": DATASET_VERSION,
        "dataset_source": OFFICIAL_ARCHIVE_URL,
        "dataset_license": "CC BY-NC-SA 4.0",
        "archive_sha256": sha256_file(ARCHIVE_PATH) if ARCHIVE_PATH.exists() else None,
        "source_file_sha256": {
            "papers.jsonl": sha256_file(PAPERS_PATH),
            "qa.jsonl": sha256_file(QA_PATH),
        },
        "papers": paper_manifest,
        "expected_papers": expected_papers,
        "paper_source_counts": dict(source_counts),
        "expected_labelled_qas": expected_qas,
        "counts": {
            "papers": len(paper_manifest),
            "papers_downloaded": len(paper_manifest),
            "failed_papers": 0,
            "labelled_questions": len(questions),
            "questions_for_included_papers": qa_inside,
            "questions_with_mapped_evidence": successfully_mapped,
            "questions_with_unmapped_evidence": sum(bool(q["unmapped_evidence"]) for q in evidence_maps),
            "papers_extracted": len(paper_manifest),
            "papers_indexed": 0,
            "pdfs_downloaded": 0,
            "pdfs_missing_or_not_authorized": len(paper_manifest),
            "fallback_extracted_text": len(paper_manifest),
            "paragraphs": total_paragraphs,
            "sentences": total_sentences,
            "chunks": total_chunks,
        },
        "split_note": (
            "The official PeerQA builder publishes the release as a single test split; "
            "qa.jsonl does not provide train/dev/test partitions. source_split is retained as test."
        ),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _atomic_json(MANIFEST_PATH, manifest)
    return manifest


def read_status() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        failure = json.loads(ACQUISITION_STATUS_PATH.read_text(encoding="utf-8")) if ACQUISITION_STATUS_PATH.exists() else {}
        return {
            "corpus_id": CORPUS_ID,
            "dataset_version": DATASET_VERSION,
            "status": failure.get("status", "not_acquired"),
            "expected_papers": EXPECTED_PAPERS,
            "counts": {
                "papers": 0,
                "papers_downloaded": 0,
                "papers_extracted": 0,
                "papers_indexed": 0,
                "failed_papers": None,
                "pdfs_downloaded": 0,
                "pdfs_missing_or_not_authorized": None,
                "fallback_extracted_text": 0,
                "paragraphs": 0,
                "sentences": 0,
                "chunks": 0,
                "labelled_questions": 0,
                "questions_with_mapped_evidence": 0,
            },
            "papers_downloaded": 0,
            "papers_extracted": 0,
            "papers_indexed": 0,
            "failed_papers": None,
            "last_acquisition_error": failure.get("error"),
            "dataset_archive": str(ARCHIVE_PATH),
        }
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    counts = manifest.get("counts", {})
    if STATUS_PATH.exists():
        index = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        counts["papers_indexed"] = index.get("papers_indexed", 0)
        counts["indexed_chunks"] = index.get("indexed_chunks", 0)
        manifest["index"] = index
    manifest["status"] = "ready" if counts.get("papers") == EXPECTED_PAPERS else "incomplete"
    return manifest


def download_permitted_pdfs() -> dict[str, int]:
    """Download licensed, source-specific PDFs for included PeerQA papers.

    The official repository defines the URL rule for EGU IDs. Other sources
    need explicit entries in ``pdf_sources.json``. Neither path searches by
    title, and records outside the official 90-paper manifest are ignored.
    """
    source_manifest_path = CORPUS_DIR / "pdf_sources.json"
    included_ids: set[str] = set()
    current_manifest: dict[str, Any] = {}
    if MANIFEST_PATH.exists():
        current_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        included_ids = {row["original_paper_id"] for row in current_manifest.get("papers", [])}
    records = json.loads(source_manifest_path.read_text(encoding="utf-8")) if source_manifest_path.exists() else []
    records_by_id = {
        row.get("original_paper_id"): row
        for row in records
        if isinstance(row, dict) and row.get("original_paper_id")
    }
    # The official PeerQA repository supplies a source-specific PDF URL rule
    # for EGU papers. Its IDs, unlike titles, give an exact and reproducible URL.
    for paper in current_manifest.get("papers", []):
        if paper.get("source") == "egu":
            source = _official_egu_pdf_source(paper["original_paper_id"])
            if source:
                records_by_id[paper["original_paper_id"]] = source
    records = list(records_by_id.values())
    downloaded = failed = 0
    per_paper: dict[str, str] = {}
    pdf_dir = CORPUS_DIR / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        paper_id = str(record.get("original_paper_id", ""))
        url = record.get("pdf_url")
        if (
            not paper_id
            or paper_id not in included_ids
            or not isinstance(url, str)
            or not url.startswith("https://")
            or record.get("license_allows_download") is not True
            or not str(record.get("license_note", "")).strip()
        ):
            if paper_id:
                per_paper[paper_id] = (
                    "skipped_paper_outside_90_paper_corpus"
                    if paper_id not in included_ids
                    else "skipped_missing_url_or_license_permission"
                )
            continue
        out = pdf_dir / f"{hashlib.sha256(paper_id.encode('utf-8')).hexdigest()}.pdf"
        if out.exists() and out.stat().st_size > 5:
            with out.open("rb") as existing:
                if existing.read(5) == b"%PDF-":
                    downloaded += 1
                    per_paper[paper_id] = "reused_existing_pdf"
                    continue
        temp = out.with_suffix(".pdf.tmp")
        last_error: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                with requests.get(url, headers={"User-Agent": USER_AGENT}, stream=True, timeout=(15, 90)) as response:
                    if response.status_code in RETRYABLE_STATUS:
                        raise requests.HTTPError(
                            f"retryable HTTP {response.status_code}", response=response
                        )
                    response.raise_for_status()
                    with temp.open("wb") as handle:
                        for block in response.iter_content(1024 * 128):
                            if block:
                                handle.write(block)
                    expected = response.headers.get("Content-Length")
                    if expected and temp.stat().st_size != int(expected):
                        actual = temp.stat().st_size
                        raise IOError(f"Incomplete PDF download ({actual} of {expected} bytes).")
                    if temp.stat().st_size < 6:
                        raise ValueError("Source returned an empty or truncated PDF file.")
                    with temp.open("rb") as handle:
                        if handle.read(5) != b"%PDF-":
                            raise ValueError("Source did not return a PDF file.")
                    os.replace(temp, out)
                downloaded += 1
                per_paper[paper_id] = "downloaded"
                last_error = None
                break
            except (OSError, requests.RequestException, ValueError) as exc:
                last_error = exc
                temp.unlink(missing_ok=True)
                if isinstance(exc, requests.HTTPError) and getattr(exc.response, "status_code", None) not in RETRYABLE_STATUS:
                    break
                if attempt < MAX_RETRIES:
                    time.sleep(min(2 ** attempt, 20))
        if last_error is not None:
            LOGGER.error("PDF download failed for %s: %s", paper_id, last_error)
            failed += 1
            per_paper[paper_id] = f"failed:{type(last_error).__name__}"
        delay = record.get("seconds_between_requests", 0)
        if isinstance(delay, (int, float)) and delay > 0 and record is not records[-1]:
            time.sleep(delay)
    for paper_id in included_ids:
        per_paper.setdefault(paper_id, "not_attempted_no_official_url_or_paper_license_record")
    skipped = sum(
        status.startswith(("skipped_", "not_attempted_"))
        for paper_id, status in per_paper.items()
        if paper_id in included_ids
    )
    return {"downloaded": downloaded, "skipped": skipped, "failed": failed, "per_paper": per_paper}


def acquire_and_build(*, force_download: bool = False) -> dict[str, Any]:
    if not force_download and PAPERS_PATH.is_file() and QA_PATH.is_file():
        download = {"path": str(ARCHIVE_PATH), "sha256": None, "status": "reused_extracted_official_files"}
        checksums = {"papers.jsonl": sha256_file(PAPERS_PATH), "qa.jsonl": sha256_file(QA_PATH)}
    else:
        download = download_archive(force=force_download)
        checksums = extract_official_files()
    manifest = build_corpus()
    manifest["archive_download"] = download
    manifest["extracted_file_checksums"] = checksums
    manifest["dataset_source"] = download.get("source_url", OFFICIAL_ARCHIVE_URL)
    _atomic_json(MANIFEST_PATH, manifest)
    pdf_result = download_permitted_pdfs()
    manifest["pdf_acquisition"] = pdf_result
    source_records = []
    source_manifest_path = CORPUS_DIR / "pdf_sources.json"
    if source_manifest_path.exists():
        source_records = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_by_id = {row.get("original_paper_id"): row for row in source_records if row.get("original_paper_id")}
    for paper in manifest.get("papers", []):
        if paper.get("source") == "egu":
            source_by_id[paper["original_paper_id"]] = _official_egu_pdf_source(paper["original_paper_id"]) or {}
    for paper in manifest.get("papers", []):
        paper_id = paper["original_paper_id"]
        pdf_source = source_by_id.get(paper_id, {})
        paper["original_pdf_url"] = pdf_source.get("pdf_url")
        paper["license_or_availability"] = pdf_source.get("license_note", paper["license_or_availability"])
        if pdf_source.get("license_source"):
            paper["license_source"] = pdf_source["license_source"]
        if pdf_source.get("source_script"):
            paper["pdf_source_script"] = pdf_source["source_script"]
        paper["pdf_download_status"] = pdf_result.get("per_paper", {}).get(
            paper_id, "not_attempted_no_official_url_or_paper_license_record"
        )
        pdf_path = CORPUS_DIR / "pdfs" / f"{hashlib.sha256(paper_id.encode('utf-8')).hexdigest()}.pdf"
        paper["pdf_sha256"] = sha256_file(pdf_path) if pdf_path.is_file() else None
    manifest["counts"]["pdfs_downloaded"] = pdf_result["downloaded"]
    manifest["counts"]["pdfs_missing_or_not_authorized"] = max(0, manifest["counts"]["papers"] - pdf_result["downloaded"])
    manifest["counts"]["fallback_extracted_text"] = max(0, manifest["counts"]["papers"] - pdf_result["downloaded"])
    manifest["counts"]["official_text_documents"] = manifest["counts"]["papers"]
    _atomic_json(MANIFEST_PATH, manifest)
    _atomic_json(ACQUISITION_STATUS_PATH, {
        "corpus_id": CORPUS_ID,
        "dataset_version": DATASET_VERSION,
        "status": "ready",
        "archive_sha256": manifest.get("archive_sha256"),
        "papers": manifest["counts"]["papers"],
        "labelled_questions": manifest["counts"]["labelled_questions"],
    })
    return manifest
