# Qdrant ingestion status

Last updated: 2026-10-29

## Completed

- Current corpus: 64 papers and 1,707 chunks; the configured cap is 1,900 tokens and the largest recorded chunk is 1,899 tokens.
- Current embedding matrix: 1,707 Qwen3-Embedding-8B vectors with 4,096 dimensions and the configured 2,000-token input limit.
- Chunking now reserves capacity for overlap. All 68 adjacent same-section boundaries have positive overlap metadata, with a maximum recorded chunk length of 1,899 tokens.
- The overlap-aware embedding refresh reused 1,664 matching vectors and encoded the remaining 43.
- Qdrant collection `qwen3_embeddings` is configured with a compatible named `dense` vector.
- All 1,707 points were uploaded and verified for current corpus version `3d64f850b4d7`.
- A keyword payload index on `corpus_version` is now created before upload/verification so the exact-count filter works with Qdrant Cloud strict mode.

## Resolved issues

- The original cache had 2,906 rows and a 512-token limit; it did not align with the 1,707 current chunks. The resumable embedding pipeline aligned it, then reused matching vectors again after the overlap update.
- The 1,900-token chunk cap is below the 2,000-token model input limit. The chunker counts with all three model tokenizers, including Qwen, and includes special tokens.
- Qdrant connectivity succeeded after setting the cluster endpoint to use port 6333.
- The initial upload completed, but its final count check failed because `corpus_version` lacked a payload index. The index was added and verification-only confirmed the existing upload without re-uploading vectors.

## Next stage

Implement query-time retrieval: encode a question with Qwen, search the named `dense` vector, and filter results to the current `corpus_version`. The collection also has a sparse-vector slot, but no sparse embeddings have been generated or uploaded.
