from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from preprocessing import embeddings
from retrieval import search


class RetrievalPipelineTests(unittest.TestCase):
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
            chunks = [{
                "id": "2005.11401::s000::introduction::c000",
                "text": "A reproducible evidence sentence.",
                "token_count": 5,
            }]
            encoded_texts = []

            def encoder(texts):
                encoded_texts.extend(texts)
                return np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

            with patch.multiple(embeddings, **paths), patch.object(embeddings, "BATCH_SIZE", 1):
                first = embeddings.encode_corpus(chunks, encoder, 4)
                second = embeddings.encode_corpus(chunks, encoder, 4)

            self.assertEqual(first.shape, (1, 4))
            self.assertTrue(np.array_equal(first, second))
            self.assertEqual(encoded_texts, ["A reproducible evidence sentence."])

    def test_retrieval_filter_can_scope_to_a_document(self):
        retriever = object.__new__(search.Retriever)
        retriever.collection = "qwen3_embeddings"
        retriever.version = "version-123"

        query_filter = retriever._filter("2005.11401")

        self.assertEqual(len(query_filter.must), 2)
        self.assertEqual(query_filter.must[0].key, "corpus_version")
        self.assertEqual(query_filter.must[1].key, "document_id")
        self.assertEqual(query_filter.must[1].match.value, "2005.11401")

    def test_bm25_retrieval_returns_ranked_chunk_payload(self):
        payload = {
            "chunk_id": "2005.11401::s000::introduction::c000",
            "document_id": "2005.11401",
            "section_title": "Introduction",
            "text": "A passage relevant to the query.",
        }
        retriever = object.__new__(search.Retriever)
        retriever.collection = "qwen3_embeddings"
        retriever.version = "version-123"
        bodies = []
        retriever._query_text_api = lambda body: (bodies.append(body) or [{"id": "point-1", "score": 0.75, **payload}])

        with patch.object(search, "rerank", side_effect=lambda _query, candidates: candidates) as reranker:
            hits = retriever.search("what happened?", mode="bm25", limit=3, document_id="2005.11401")

        reranker.assert_called_once_with("what happened?", [{"id": "point-1", "score": 0.75, **payload}])
        self.assertEqual(hits[0]["chunk_id"], payload["chunk_id"])
        self.assertEqual(hits[0]["text"], payload["text"])
        self.assertEqual(hits[0]["score"], 0.75)
        self.assertEqual(bodies[0]["query"], {"text": "what happened?"})
        self.assertEqual(bodies[0]["filter"]["must"][1]["key"], "document_id")

    def test_hybrid_retrieval_fuses_dense_and_bm25_with_document_scope(self):
        retriever = object.__new__(search.Retriever)
        retriever.collection = "qwen3_embeddings"
        retriever.version = "version-123"
        retriever.encode_query = lambda _query: [0.1, 0.2]
        bodies = []
        retriever._query_text_api = lambda body: (bodies.append(body) or [])

        with patch.object(search, "rerank", side_effect=lambda _query, candidates: candidates) as reranker:
            retriever.search("what happened?", mode="hybrid", limit=5, document_id="2005.11401")

        reranker.assert_called_once_with("what happened?", [])
        self.assertEqual(bodies[0]["query"], {"rrf": {}})
        self.assertEqual(len(bodies[0]["prefetch"]), 2)
        self.assertEqual(bodies[0]["prefetch"][0]["using"], "dense")
        self.assertEqual(bodies[0]["prefetch"][1]["query"], {"text": "what happened?"})
        self.assertEqual(bodies[0]["filter"]["must"][1]["match"]["value"], "2005.11401")


if __name__ == "__main__":
    unittest.main()
