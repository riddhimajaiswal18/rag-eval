from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from peerqa import corpus
from preprocessing import embeddings
from retrieval import qdrant_setup
from scripts.evaluate_peerqa import ndcg, reciprocal_rank


def source_rows() -> list[dict]:
    return [
        {"paper_id": "nlpeer/ARR-2023/p1", "idx": 0, "pidx": 0, "sidx": 0,
         "type": "title", "content": "A Stable Paper Title", "last_heading": ""},
        {"paper_id": "nlpeer/ARR-2023/p1", "idx": 1, "pidx": 1, "sidx": 0,
         "type": "sentence", "content": "The first evidence sentence.", "last_heading": "Methods"},
        {"paper_id": "nlpeer/ARR-2023/p1", "idx": 2, "pidx": 1, "sidx": 1,
         "type": "sentence", "content": "The second evidence sentence.", "last_heading": "Methods"},
        {"paper_id": "egu/ESurf/p2", "idx": 0, "pidx": 0, "sidx": 0,
         "type": "title", "content": "Second Paper", "last_heading": ""},
        {"paper_id": "egu/ESurf/p2", "idx": 3, "pidx": 2, "sidx": 0,
         "type": "sentence", "content": "A different paper sentence.", "last_heading": "Results"},
    ]


def qa_rows() -> list[dict]:
    return [
        {
            "question_id": "q-1", "paper_id": "nlpeer/ARR-2023/p1", "question": "What was done?",
            "raw_answer_evidence": ["The first evidence sentence."],
            "answer_evidence_sent": ["The first evidence sentence."],
            "answer_evidence_mapped": [{"sentence": "The first evidence sentence.", "idx": [1, None]}],
            "answer_free_form": "The authors ran a study.", "answerable": True, "answerable_mapped": False,
        },
        {
            "question_id": "q-2", "paper_id": "openreview/ICLR-2022-conf/paper-x", "question": "Outside?",
            "answer_evidence_mapped": [{"sentence": "Not in subset.", "idx": [12]}],
            "answer_free_form": "", "answerable": True, "answerable_mapped": True,
        },
    ]


class PeerQAPipelineTests(unittest.TestCase):
    def test_canonical_document_id_is_stable_and_preserves_source_identity_separately(self):
        paper_id = "nlpeer/ARR-2023/p1"
        expected = "peerqa:" + hashlib.sha256(paper_id.encode()).hexdigest()
        self.assertEqual(corpus.canonical_document_id(paper_id), expected)
        self.assertNotEqual(corpus.canonical_document_id(paper_id), corpus.canonical_document_id("egu/ESurf/p1"))

    def test_official_archive_extraction_validates_and_writes_only_official_jsonl(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_dir = root / "source"
            archive_path = root / "official.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("peerqa-data-v1.0/papers.jsonl", "{}\n")
                archive.writestr("peerqa-data-v1.0/qa.jsonl", "{}\n")
                archive.writestr("peerqa-data-v1.0/qa-unlabeled.jsonl", "{}\n")
                archive.writestr("peerqa-data-v1.0/ignore.bin", b"ignore")
            with patch.object(corpus, "SOURCE_DIR", source_dir):
                checksums = corpus.extract_official_files(archive_path)
            self.assertEqual(set(checksums), {"papers.jsonl", "qa.jsonl", "qa-unlabeled.jsonl"})
            self.assertEqual((source_dir / "papers.jsonl").read_text(), "{}\n")
            self.assertFalse((source_dir / "ignore.bin").exists())

    def test_official_archive_download_is_validated_and_checksummed(self):
        archive_buffer = io.BytesIO()
        with zipfile.ZipFile(archive_buffer, "w") as archive:
            archive.writestr("papers.jsonl", "{}\n")
            archive.writestr("qa.jsonl", "{}\n")
        archive_bytes = archive_buffer.getvalue()
        response = MagicMock()
        response.__enter__.return_value = response
        response.status_code = 200
        response.headers = {"Content-Length": str(len(archive_bytes))}
        response.iter_content.return_value = [archive_bytes]
        response.raise_for_status.return_value = None
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive_path = root / "peerqa.zip"
            status_path = root / "acquisition_status.json"
            with patch.object(corpus, "CORPUS_DIR", root), patch.object(corpus, "ARCHIVE_PATH", archive_path), \
                    patch.object(corpus, "ACQUISITION_STATUS_PATH", status_path), \
                    patch.object(corpus, "OFFICIAL_ARCHIVE_URLS", ("https://official.example/peerqa.zip",)), \
                    patch.object(corpus.requests, "get", return_value=response) as get:
                result = corpus.download_archive()
            get.assert_called_once()
            self.assertTrue(corpus._valid_archive(archive_path))
            self.assertEqual(result["sha256"], hashlib.sha256(archive_bytes).hexdigest())
            self.assertEqual(result["status"], "downloaded")

    def test_official_archive_403_is_recorded_without_fake_dataset_counts(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.status_code = 403
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            status_path = root / "acquisition_status.json"
            urls = ("https://official.example/sequence-1", "https://official.example/sequence-5")
            with patch.object(corpus, "CORPUS_DIR", root), patch.object(corpus, "ARCHIVE_PATH", root / "peerqa.zip"), \
                    patch.object(corpus, "ACQUISITION_STATUS_PATH", status_path), \
                    patch.object(corpus, "OFFICIAL_ARCHIVE_URLS", urls), \
                    patch.object(corpus.requests, "get", return_value=response) as get:
                with self.assertRaisesRegex(RuntimeError, "Could not acquire official PeerQA archive"):
                    corpus.download_archive()
            self.assertEqual(get.call_count, 2)
            status = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "acquisition_failed")
            self.assertEqual(status["sources_tried"], list(urls))
            self.assertNotIn("papers", status)

    def test_rejects_zip_path_traversal(self):
        with self.assertRaises(ValueError):
            corpus._safe_zip_member("../../papers.jsonl")

    def test_document_ids_and_sentence_ids_do_not_depend_on_input_iteration_order(self):
        rows = [corpus._normalize_source_record(row) for row in source_rows()[:3]]
        one, map_one = corpus._build_document("nlpeer/ARR-2023/p1", rows)
        two, map_two = corpus._build_document("nlpeer/ARR-2023/p1", list(reversed(rows)))
        self.assertEqual(one["document_id"], two["document_id"])
        self.assertEqual(map_one[1]["sentence_id"], map_two[1]["sentence_id"])
        self.assertEqual(map_one[2]["source_location"], {"idx": 2, "pidx": 1, "sidx": 1})

    def test_paragraph_chunks_preserve_exact_source_sentence_provenance(self):
        rows = [corpus._normalize_source_record(row) for row in source_rows()[:3]]
        document, _ = corpus._build_document("nlpeer/ARR-2023/p1", rows)
        chunks = corpus._chunks_for_document(document)
        self.assertEqual(chunks["num_chunks"], 1)
        chunk = chunks["chunks"][0]
        self.assertEqual(chunk["sentence_ids"], [
            corpus._row_sentence_id(document["document_id"], rows[1]),
            corpus._row_sentence_id(document["document_id"], rows[2]),
        ])
        self.assertEqual([ref["idx"] for ref in chunk["source_refs"]], [1, 2])
        self.assertEqual(chunk["paragraph_id"], f"{document['document_id']}:p1")

    def test_oversized_source_sentence_splits_without_losing_its_source_reference(self):
        rows = [corpus._normalize_source_record(row) for row in source_rows()[:3]]
        rows[1]["content"] = " ".join(f"word{i}" for i in range(12))
        document, _ = corpus._build_document("nlpeer/ARR-2023/p1", rows)
        chunks = corpus._chunks_for_document(document, max_words=5)["chunks"]
        first_sentence_id = corpus._row_sentence_id(document["document_id"], rows[1])
        fragments = [chunk for chunk in chunks if first_sentence_id in chunk["sentence_ids"]]
        self.assertEqual(len(fragments), 3)
        self.assertTrue(all(chunk["source_refs"][0]["idx"] == 1 for chunk in fragments))
        self.assertEqual(" ".join(chunk["text"] for chunk in fragments), rows[1]["content"])

    def test_evidence_mapping_retains_unmapped_refs_and_answerability_fields(self):
        records = [corpus._normalize_source_record(row) for row in source_rows()[:3]]
        document, idx_map = corpus._build_document("nlpeer/ARR-2023/p1", records)
        docs = {"nlpeer/ARR-2023/p1": (document, idx_map)}
        question, mapping = corpus.map_question_evidence(qa_rows()[0], docs)
        self.assertEqual(question["answerable"], True)
        self.assertEqual(question["answerable_mapped"], False)
        self.assertEqual(len(question["gold_sentence_ids"]), 1)
        self.assertEqual(mapping["evidence_mappings"][0]["idx"], 1)
        self.assertEqual(mapping["evidence_mappings"][0]["pidx"], 1)
        self.assertEqual(mapping["evidence_mappings"][1]["status"], "official_evidence_has_no_idx")
        self.assertEqual(len(mapping["unmapped_evidence"]), 1)

    def test_questions_for_unincluded_papers_are_preserved_as_unmapped(self):
        question, mapping = corpus.map_question_evidence(qa_rows()[1], {})
        self.assertIsNone(question["document_id"])
        self.assertEqual(mapping["evidence_mappings"][0]["status"], "paper_outside_90_paper_corpus")
        self.assertEqual(question["answerable"], True)
        self.assertEqual(question["answerable_mapped"], True)

    def test_unlicensed_or_unknown_pdf_source_is_not_requested(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "pdf_sources.json").write_text(json.dumps([{
                "original_paper_id": "nlpeer/ARR-2023/p1",
                "pdf_url": "https://publisher.example/paper.pdf",
                "license_allows_download": True,
                # Missing explicit license note: this must remain skipped.
            }]), encoding="utf-8")
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({"papers": [{"original_paper_id": "nlpeer/ARR-2023/p1"}]}), encoding="utf-8")
            with patch.object(corpus, "CORPUS_DIR", root), patch.object(corpus, "MANIFEST_PATH", manifest_path), patch.object(corpus.requests, "get") as get:
                result = corpus.download_permitted_pdfs()
            get.assert_not_called()
            self.assertEqual(result["downloaded"], 0)
            self.assertEqual(result["skipped"], 1)

    def test_official_egu_source_rule_matches_paper_ids(self):
        source = corpus._official_egu_pdf_source("egu/ESurf/6-1-2011")
        self.assertIsNotNone(source)
        self.assertEqual(
            source["pdf_url"],
            "https://ESurf.copernicus.org/articles/6/1/2011/ESurf-6-1-2011.pdf",
        )
        self.assertTrue(source["license_allows_download"])
        self.assertIsNone(corpus._official_egu_pdf_source("egu/ESurf/not-a-source-id"))

    def test_egu_pdf_is_auto_registered_only_for_included_papers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({"papers": [
                {"original_paper_id": "egu/ESurf/6-1-2011", "source": "egu"},
            ]}), encoding="utf-8")
            response = MagicMock()
            response.__enter__.return_value = response
            response.status_code = 200
            response.headers = {}
            response.iter_content.return_value = [b"%PDF-1.7\ncontent"]
            with patch.object(corpus, "CORPUS_DIR", root), patch.object(corpus, "MANIFEST_PATH", manifest_path), \
                    patch.object(corpus.requests, "get", return_value=response) as get:
                result = corpus.download_permitted_pdfs()
            get.assert_called_once()
            self.assertEqual(get.call_args.args[0], "https://ESurf.copernicus.org/articles/6/1/2011/ESurf-6-1-2011.pdf")
            self.assertEqual(result["downloaded"], 1)
            self.assertEqual(result["per_paper"]["egu/ESurf/6-1-2011"], "downloaded")

    def test_pdf_source_cannot_add_the_118_openreview_papers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "pdf_sources.json").write_text(json.dumps([{
                "original_paper_id": "openreview/ICLR-2022-conf/not-in-subset",
                "pdf_url": "https://openreview.net/pdf?id=not-in-subset",
                "license_allows_download": True,
                "license_note": "verified",
            }]), encoding="utf-8")
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({"papers": [{"original_paper_id": "nlpeer/ARR-2023/p1"}]}), encoding="utf-8")
            with patch.object(corpus, "CORPUS_DIR", root), patch.object(corpus, "MANIFEST_PATH", manifest_path), patch.object(corpus.requests, "get") as get:
                result = corpus.download_permitted_pdfs()
            get.assert_not_called()
            self.assertEqual(result["downloaded"], 0)
            self.assertTrue(result["per_paper"]["openreview/ICLR-2022-conf/not-in-subset"].startswith("skipped_paper_outside"))

    def test_qdrant_loader_keeps_sentence_and_location_payload(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            chunk = {
                "chunk_id": "peerqa:doc::p1::c000", "text": "A sentence.",
                "corpus_id": "peerqa", "document_id": "peerqa:doc",
                "original_paper_id": "nlpeer/ARR-2023/p1", "title": "Paper",
                "section_id": "sec-1", "section_title": "Methods", "paragraph_id": "para-1",
                "sentence_ids": ["sent-1"], "source_refs": [{"idx": 4, "pidx": 1, "sidx": 0}],
                "source_text": "A sentence.", "source_location": [{"idx": 4, "pidx": 1, "sidx": 0}],
                "extraction_method": "peerqa_official_papers_jsonl",
            }
            (root / "p_chunks.json").write_text(json.dumps({"document_id": "peerqa:doc", "chunks": [chunk]}), encoding="utf-8")
            with patch.object(qdrant_setup, "CHUNKS_DIR", root), patch.object(qdrant_setup, "CORPUS_ID", "peerqa"):
                loaded = qdrant_setup.load_chunks()
            self.assertEqual(loaded[0]["sentence_ids"], ["sent-1"])
            self.assertEqual(loaded[0]["source_refs"][0]["idx"], 4)
            self.assertEqual(loaded[0]["original_paper_id"], "nlpeer/ARR-2023/p1")

    def test_build_is_incremental_and_stable_on_rerun(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_dir = root / "source"
            source_dir.mkdir()
            papers = "".join(json.dumps(row) + "\n" for row in source_rows())
            qas = "".join(json.dumps(row) + "\n" for row in qa_rows())
            (source_dir / "papers.jsonl").write_text(papers, encoding="utf-8")
            (source_dir / "qa.jsonl").write_text(qas, encoding="utf-8")
            overrides = {
                "CORPUS_DIR": root,
                "SOURCE_DIR": source_dir,
                "PAPERS_PATH": source_dir / "papers.jsonl",
                "QA_PATH": source_dir / "qa.jsonl",
                "DOCS_DIR": root / "documents",
                "CHUNKS_DIR": root / "chunks",
                "QUESTIONS_PATH": root / "questions.jsonl",
                "EVIDENCE_MAP_PATH": root / "evidence.jsonl",
                "MANIFEST_PATH": root / "manifest.json",
                "ARCHIVE_PATH": root / "official.zip",
            }
            with patch.multiple(corpus, **overrides):
                first = corpus.build_corpus(expected_papers=2, expected_qas=2)
                first_chunks = sorted((root / "chunks").glob("*_chunks.json"))
                ids_before = [chunk["chunk_id"] for path in first_chunks for chunk in json.loads(path.read_text())["chunks"]]
                second = corpus.build_corpus(expected_papers=2, expected_qas=2)
                second_chunks = sorted((root / "chunks").glob("*_chunks.json"))
                ids_after = [chunk["chunk_id"] for path in second_chunks for chunk in json.loads(path.read_text())["chunks"]]
            self.assertEqual(len(first_chunks), 2)
            self.assertEqual(len(second_chunks), 2)
            self.assertEqual(ids_before, ids_after)
            self.assertEqual(first["counts"]["chunks"], second["counts"]["chunks"])
            self.assertEqual(second["counts"]["labelled_questions"], 2)
            self.assertEqual(second["counts"]["questions_for_included_papers"], 1)
            self.assertTrue(all(p["extraction_method"] == "peerqa_official_papers_jsonl" for p in second["papers"]))
            self.assertTrue(all(p["pdf_download_status"].startswith("not_attempted") for p in second["papers"]))

    def test_embedding_cache_does_not_reencode_unchanged_chunks(self):
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp)
            paths = {
                "CACHE_DIR": cache,
                "EMBEDDINGS_FILE": cache / "embeddings.npy",
                "METADATA_FILE": cache / "metadata.json",
                "PROGRESS_FILE": cache / "progress.json",
                "PARTIAL_EMBEDDINGS_FILE": cache / "embeddings.partial.npy",
                "PARTIAL_METADATA_FILE": cache / "metadata.partial.json",
                "PENDING_BATCH_FILE": cache / "pending_batch.npz",
            }
            chunks = [{"id": "peerqa:doc::p1::c000", "text": "Evidence text.", "token_count": 2}]
            calls = []

            def encoder(texts):
                calls.extend(texts)
                return np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

            with patch.multiple(embeddings, **paths), patch.object(embeddings, "BATCH_SIZE", 1):
                first = embeddings.encode_corpus(chunks, encoder, 4)
                second = embeddings.encode_corpus(chunks, encoder, 4)
            self.assertEqual(first.shape, (1, 4))
            self.assertTrue(np.array_equal(first, second))
            self.assertEqual(calls, ["Evidence text."])

    def test_corpus_paths_and_qdrant_collections_are_isolated(self):
        arxiv_chunks = qdrant_setup.CHUNKS_DIR
        arxiv_embedding_cache = embeddings.CACHE_DIR
        with patch.dict(os.environ, {"QDRANT_COLLECTION": "arxiv-existing", "QDRANT_COLLECTION_PEERQA": "peerqa-only"}):
            qdrant_setup.configure_corpus("peerqa")
            embeddings.configure_corpus("peerqa")
            self.assertNotEqual(qdrant_setup.CHUNKS_DIR, arxiv_chunks)
            self.assertNotEqual(embeddings.CACHE_DIR, arxiv_embedding_cache)
            self.assertEqual(qdrant_setup.collection_name_for("peerqa"), "peerqa-only")
            self.assertEqual(qdrant_setup.collection_name_for("arxiv"), "arxiv-existing")
        qdrant_setup.configure_corpus("arxiv")
        embeddings.configure_corpus("arxiv")

    def test_retrieval_filter_can_scope_peerqa_to_the_known_document(self):
        from retrieval import search

        retriever = object.__new__(search.Retriever)
        retriever.collection = "peerqa-only"
        retriever.version = "version-123"
        query_filter = retriever._filter("peerqa:doc")
        self.assertEqual(len(query_filter.must), 2)
        self.assertEqual(query_filter.must[0].key, "corpus_version")
        self.assertEqual(query_filter.must[1].key, "document_id")
        self.assertEqual(query_filter.must[1].match.value, "peerqa:doc")

    def test_bm25_retrieval_returns_title_ids_score_and_source_refs(self):
        from retrieval import search

        payload = {
            "title": "Paper title", "original_paper_id": "nlpeer/ARR-2023/p1",
            "sentence_ids": ["sent-1"], "source_refs": [{"idx": 4, "pidx": 1, "sidx": 0}],
        }
        point = SimpleNamespace(id="point-1", score=0.75, payload=payload)

        retriever = object.__new__(search.Retriever)
        retriever.collection = "peerqa-only"
        retriever.version = "version-123"
        bodies = []

        def fake_query(body):
            bodies.append(body)
            return [{"id": str(point.id), "score": point.score, **point.payload}]

        retriever._query_text_api = fake_query
        hits = retriever.search("what happened?", mode="bm25", limit=3, document_id="peerqa:doc")
        self.assertEqual(hits[0]["title"], "Paper title")
        self.assertEqual(hits[0]["original_paper_id"], "nlpeer/ARR-2023/p1")
        self.assertEqual(hits[0]["score"], 0.75)
        self.assertEqual(hits[0]["source_refs"][0]["idx"], 4)
        self.assertEqual(bodies[0]["query"], {"text": "what happened?"})
        self.assertEqual(bodies[0]["filter"]["must"][1]["key"], "document_id")

    def test_hybrid_retrieval_fuses_dense_and_bm25_with_paper_scope(self):
        from retrieval import search

        retriever = object.__new__(search.Retriever)
        retriever.collection = "peerqa-only"
        retriever.version = "version-123"
        retriever.encode_query = lambda _query: [0.1, 0.2]
        bodies = []
        retriever._query_text_api = lambda body: (bodies.append(body) or [])
        retriever.search("what happened?", mode="hybrid", limit=5, document_id="peerqa:doc")
        self.assertEqual(bodies[0]["query"], {"rrf": {}})
        self.assertEqual(len(bodies[0]["prefetch"]), 2)
        self.assertEqual(bodies[0]["prefetch"][0]["using"], "dense")
        self.assertEqual(bodies[0]["prefetch"][1]["query"], {"text": "what happened?"})
        self.assertEqual(bodies[0]["filter"]["must"][1]["match"]["value"], "peerqa:doc")

    def test_retrieval_metrics_handle_ranked_binary_evidence(self):
        self.assertEqual(reciprocal_rank([False, True]), 0.5)
        self.assertEqual(reciprocal_rank([False, False]), 0.0)
        self.assertAlmostEqual(ndcg([True, False], 2), 1.0)


if __name__ == "__main__":
    unittest.main()
