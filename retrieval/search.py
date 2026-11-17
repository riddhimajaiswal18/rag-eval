"""Dense, BM25, hybrid, ColBERT multivector, and cross-encoder retrieval."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import httpx
try:  # Support both ``python retrieval/search.py`` and package imports.
    from .qdrant_setup import collection_name_for
except ImportError:
    from qdrant_setup import collection_name_for
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from sentence_transformers import CrossEncoder, SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
QWEN_MODEL = "Qwen/Qwen3-Embedding-8B"
COLBERT_MODEL = "colbert-ir/colbertv2.0"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Retriever:
    def __init__(self, collection: str, corpus_version: str, model_name: str = QWEN_MODEL):
        load_dotenv(ROOT / ".env")
        self.client = QdrantClient(
            url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"]
        )
        self.collection = collection
        self.version = corpus_version
        self.model_name = model_name
        self.encoder = None

    def encode_query(self, query: str) -> list[float]:
        if self.encoder is None:
            self.encoder = SentenceTransformer(self.model_name, trust_remote_code=True)
        vector = self.encoder.encode_query(query, normalize_embeddings=True)
        return np.asarray(vector, dtype=np.float32).tolist()

    def _filter(self, document_id: str | None = None) -> models.Filter:
        conditions = [models.FieldCondition(
            key="corpus_version", match=models.MatchValue(value=self.version)
        )]
        if document_id:
            conditions.append(models.FieldCondition(
                key="document_id", match=models.MatchValue(value=document_id)
            ))
        return models.Filter(must=conditions)

    def _filter_json(self, document_id: str | None = None) -> dict[str, Any]:
        must: list[dict[str, Any]] = [
            {"key": "corpus_version", "match": {"value": self.version}}
        ]
        if document_id:
            must.append({"key": "document_id", "match": {"value": document_id}})
        return {"must": must}

    def _query_text_api(self, body: dict[str, Any]) -> list[dict[str, Any]]:
        """Call Qdrant's Query API for BM25 text queries.

        qdrant-client versions before their TextQuery model was generated still
        support the server API; send its documented JSON directly in that case.
        """
        base_url = os.environ["QDRANT_URL"].rstrip("/")
        headers = {"api-key": os.environ["QDRANT_API_KEY"], "Content-Type": "application/json"}
        url = f"{base_url}/collections/{self.collection}/points/query"
        response = httpx.post(url, json=body, headers=headers, timeout=60.0)
        response.raise_for_status()
        result = response.json().get("result", {})
        points = result.get("points", []) if isinstance(result, dict) else []
        return [{"id": str(point.get("id")), "score": float(point.get("score", 0)), **(point.get("payload") or {})} for point in points]

    def _text_search(self, query: str, limit: int, document_id: str | None) -> list[dict[str, Any]]:
        return self._query_text_api({
            "query": {"text": query},
            "using": "text",
            "filter": self._filter_json(document_id),
            "limit": limit,
            "with_payload": True,
        })

    def _hybrid_search(
        self, query: str, vector: list[float], limit: int, document_id: str | None
    ) -> list[dict[str, Any]]:
        query_filter = self._filter_json(document_id)
        return self._query_text_api({
            "prefetch": [
                {"query": vector, "using": "dense", "filter": query_filter, "limit": limit},
                {"query": {"text": query}, "using": "text", "filter": query_filter, "limit": limit},
            ],
            "query": {"rrf": {}},
            "filter": query_filter,
            "limit": limit,
            "with_payload": True,
        })

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        limit: int = 100,
        document_id: str | None = None,
    ) -> list[dict[str, Any]]:
        common = dict(query_filter=self._filter(document_id), limit=limit, with_payload=True)
        if mode == "dense":
            qvec = self.encode_query(query)
            response = self.client.query_points(
                collection_name=self.collection, query=qvec, using="dense", **common
            )
        elif mode == "bm25":
            return self._text_search(query, limit, document_id)
        elif mode == "hybrid":
            qvec = self.encode_query(query)
            return self._hybrid_search(query, np.asarray(qvec, dtype=np.float32).tolist(), limit, document_id)
        else:
            raise ValueError("mode must be dense, bm25, or hybrid")
        return [{"id": str(p.id), "score": float(p.score), **(p.payload or {})} for p in response.points]


def rerank(query: str, candidates: list[dict[str, Any]], model_name: str = RERANKER_MODEL) -> list[dict[str, Any]]:
    """Cross-encoder rerank over a retrieved candidate set."""
    if not candidates:
        return []
    model = CrossEncoder(model_name)
    pairs = [(query, item["text"]) for item in candidates]
    scores = model.predict(pairs)
    return [item for _, item in sorted(zip(scores, candidates), key=lambda pair: float(pair[0]), reverse=True)]


def colbert_encode(text: str, model_name: str = COLBERT_MODEL) -> list[list[float]]:
    """Encode token vectors for Qdrant MAX_SIM multivector indexing/search.

    Requires FastEmbed's ColBERT support. PLAID's compressed index is a separate
    ColBERT engine and is not implemented by Qdrant's generic multivector path.
    """
    from fastembed import LateInteractionTextEmbedding

    model = LateInteractionTextEmbedding(model_name)
    return next(model.query_embed(text)).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="Query text (positional form retained for compatibility).")
    parser.add_argument("--query", dest="query_text", help="Query text.")
    parser.add_argument("--mode", choices=("dense", "bm25", "hybrid"), default="hybrid")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--version", help="Optional corpus_version; inferred from the selected local corpus cache by default.")
    args = parser.parse_args()
    query = args.query_text or args.question
    if not query:
        parser.error("provide --query or a positional query")
    if args.query_text and args.question:
        parser.error("provide either --query or a positional query, not both")
    load_dotenv(ROOT / ".env")
    collection = collection_name_for()
    version = args.version
    if not version:
        try:
            from . import qdrant_setup
        except ImportError:
            import qdrant_setup

        chunks, _vectors, metadata = qdrant_setup.load_and_validate_cache()
        version = qdrant_setup.corpus_version(chunks, metadata)
    if not version:
        raise RuntimeError("Could not determine the selected corpus version; pass --version explicitly.")
    retriever = Retriever(collection, version)
    hits = retriever.search(query, args.mode, args.limit)
    if args.rerank:
        hits = rerank(query, hits)
    print(json.dumps(hits, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
