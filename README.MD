# RagAndEval

**An experimental platform for evaluating RAG architectures, retrieval strategies, and their quality–performance–cost trade-offs.**

RAG systems are built by combining document processing, chunking strategies, embedding models, vector databases, and retrieval techniques. However, choosing an architecture involves more than simply retrieving relevant documents.

Different workloads introduce different requirements. A configuration that performs well on one corpus or query distribution may perform poorly on another. Improvements in retrieval quality may also introduce additional indexing costs, storage overhead, or query latency.

**RagAndEval explores these trade-offs through reproducible experiments, retrieval evaluation, and architectural comparisons.**

The goal is to understand not just *which configuration performs better*, but:

- Under what conditions does it perform better?
- What does the improvement cost?
- Which architectural choices are justified for a particular workload?
- How can developers make informed decisions instead of relying on default configurations?

---

## 1. Problem Statement

There is no universally optimal RAG architecture.

Its performance depends on several interacting variables:

- **Corpus characteristics:** document length, structure, content distribution, and information density.
- **Query characteristics:** semantic queries, keyword-heavy queries, mathematical queries, and multi-hop questions.
- **Chunking:** chunk boundaries, size, overlap, and structural preservation.
- **Retrieval:** dense, sparse, hybrid, and reranking strategies.
- **Infrastructure:** embedding computation, index construction, retrieval latency, and storage.
- **Resource constraints:** compute availability, throughput, and inference costs.

A retrieval strategy that improves recall may also increase latency or storage requirements. Similarly, more sophisticated chunking does not necessarily justify its preprocessing and indexing overhead.

RagAndEval investigates these interactions rather than assuming that more complex architectures are always better.

## 2. Areas of Investigation

### Retrieval quality

- How do dense, BM25, and hybrid retrieval compare?
- When does hybrid retrieval provide meaningful improvements over individual retrieval methods?
- What is the effect of reranking on retrieval quality and latency?
- How do different query characteristics influence retrieval performance?

### Chunking and document representation

- How do chunk size and overlap affect evidence retrieval?
- Does structure-aware chunking improve retrieval over flat chunking?
- How does changing chunk boundaries affect the ability to retrieve relevant evidence?
- What information is lost when documents are converted into plain text?

### Multimodal and visual retrieval

Text extraction can lose information from mathematical equations, diagrams, and complex document layouts.

An experimental direction is to investigate **ColPali as an alternative visual retrieval path** when conventional text-based retrieval is insufficient.

The objective is to understand:

- Whether visual retrieval recovers relevant evidence missed by text retrieval.
- Which query types benefit from visual document representations.
- Whether retrieval improvements justify additional inference and storage costs.
- When a visual retrieval path is preferable to conventional text retrieval.

ColPali is an experimental direction, not a currently completed retrieval capability.

### Infrastructure and resource efficiency

- How does corpus size affect indexing and retrieval performance?
- How do vector quantisation strategies affect storage and retrieval quality?
- How does query concurrency affect latency and throughput?
- What are the trade-offs between embedding generation, index construction, and query-time costs?
- Can retrieval quality be improved without proportionally increasing compute and inference costs?

## 3. Architecture

The current implementation consists of a document acquisition and processing pipeline, followed by shared embedding, indexing, and retrieval infrastructure.

### High-level architecture

```text
              ARXIV RESEARCH PAPERS
                        |
                 PDF Acquisition
                        |
                PDF Text Extraction
                        |
                 GROBID / TEI
                        |
             Document Structure Parsing
                        |
                    Chunking
                        |
                Embedding Pipeline
                        |
                 Embedding Cache
                        |
                  Qdrant Index
                        |
                 Retrieval Layer
                        |
          +-------------+-------------+
          |             |             |
         Dense         BM25         Hybrid
          |             |             |
          +-------------+-------------+
                        |
                   Reranking
                   (required for every retrieval mode)
                        |
                 Retrieved Evidence
                        |
                Evaluation Pipeline
                        |
               Retrieval Metrics
                        |
                Experiment Reports
```

The architecture separates document processing, embedding generation, indexing, retrieval, and evaluation so that individual components can be investigated independently.

## 4. Current Implementation

### Corpus processing

The current corpus pipeline supports research-paper acquisition and processing.

- Acquisition of arXiv research-paper PDFs.
- PDF text extraction.
- GROBID-based TEI/XML extraction.
- Section and document-block extraction.
- Tokenizer-aware chunking.
- Metadata preservation.

The implementation supports extracted-text input as an alternative to GROBID processing.

### Embedding experiments

The project includes experiments comparing three embedding models.

The embedding pipeline supports:

- Configurable embedding generation.
- Tokenizer-aware input handling.
- Persistent embedding caches.
- Resumable processing.
- Cache validation.

Qwen3-Embedding-8B is also used in the existing embedding infrastructure, with a 4-bit model configuration.

Embedding model comparisons and historical results should be distinguished from newly reproduced benchmark results.

### Vector indexing

Qdrant is used as the vector database.

The indexing infrastructure includes:

- Embedding cache validation.
- Vector and payload storage.
- Collection/version separation.
- Upload retries.
- Upload count verification.

The current indexing code uploads dense vectors as float32 and does not configure Qdrant quantization or report collection storage usage. Quantization is a planned experiment, not an implemented feature.

### Retrieval

The current retrieval implementation supports:

- Dense retrieval.
- BM25 / sparse text retrieval.
- Hybrid retrieval.
- Cross-encoder reranking is applied to every retrieved candidate list.

ColBERT-related encoding experimentation exists, but should not be interpreted as a complete end-to-end ColBERT retrieval implementation.

### Evaluation

The project includes embedding comparison experiments and a retrieval evaluation pipeline.

The evaluation dimensions include:

| Metric | Purpose |
|---|---|
| Recall@K | Whether relevant evidence is retrieved |
| Precision@K | How much of the retrieved set is relevant |
| MRR | How highly the first relevant result is ranked |
| nDCG@K | Ranking quality with graded relevance where available |

The evaluation infrastructure focuses on retrieval and evidence relevance. It does not currently establish generated-answer quality, faithfulness, or end-to-end RAG performance.

## 5. Evaluation Methodology

A central principle of RagAndEval is to separate retrieval quality from infrastructure performance.

A retrieval configuration should not be considered better solely because it achieves a higher retrieval score.

The experiments aim to measure multiple dimensions:

| Dimension | Measurements |
|---|---|
| Retrieval quality | Recall@K, Precision@K, MRR, nDCG |
| Latency | Mean, p50, p95 |
| Throughput | Queries per second |
| Storage | Embedding and index size |
| Compute | Embedding generation and indexing time |
| Cost | Compute and inference cost per workload |

Experiments should control the corpus version, query set, relevance labels, embedding model, and retrieval parameters wherever possible.

For the planned Qdrant quantization comparison, capture a storage snapshot for the unquantized baseline and the quantized collection. Record the collection configuration, corpus version, point count, vector and index storage (where the server exposes those fields), payload storage, and RAM usage. Compare retrieval with the same query set, retrieval mode, K, filters, embedding model, and required cross-encoder reranking. Report Recall@K, Precision@K, and F1@K alongside storage and latency; do not treat expected compression ratios as measured results. Keep the unquantized collection as the baseline and use a separate collection or restorable snapshot for the quantized condition.

Results from different corpus versions or evaluation datasets should not be compared as though they were equivalent experiments.

### Evaluation data

A major challenge in RAG evaluation is establishing reliable ground-truth evidence for retrieval experiments.

The project aims to maintain consistent evaluation datasets, stable evidence identifiers, and reproducible experimental configurations.

Synthetic questions may help expand experimental coverage, but they should not automatically be treated as equivalent to human-validated ground truth.

Evaluation results are meaningful only when the underlying evidence mapping and corpus version are verified.

## 6. Planned Experiments

The following are research and engineering directions, not claims about completed implementation.

| Area | Planned investigation |
|---|---|
| Retrieval | More controlled dense, BM25, hybrid and reranking comparisons |
| Chunking | Recursive, semantic and structure-aware strategies |
| Visual retrieval | ColPali as a fallback for mathematical and diagram-based content |
| Storage | Capture Qdrant storage snapshots, then compare scalar/binary vector quantisation and index-size trade-offs |
| Performance | Concurrent query load testing and latency profiling |
| Evaluation | Stable evidence mapping and stronger query coverage |
| Efficiency | Embedding and indexing cost optimisation |

## 7. Cost and Efficiency Considerations

Efficiency is treated as an architectural requirement rather than an afterthought.

The project explores several opportunities:

- Reusing cached embeddings instead of recomputing them.
- Separating indexing costs from query-time costs.
- Measuring the storage overhead of different representations.
- Investigating quantisation.
- Avoiding unnecessary LLM invocations during retrieval-only evaluation.
- Comparing retrieval improvements against their computational overhead.

An important distinction is that retrieval evaluation does not require answer generation for every query.

This allows retrieval experiments to measure evidence quality without introducing unnecessary generation costs.

## 8. Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| Document extraction | GROBID, TEI/XML |
| Embeddings | Hugging Face embedding models |
| Vector database | Qdrant |
| Retrieval | Dense, BM25, hybrid |
| Reranking | Cross-encoder |
| Evaluation | Custom evaluation pipelines |
| Data processing | JSON, JSONL |
| Experimentation | Reproducible scripts and benchmark artifacts |

## 9. Repository Structure

```text
RagAndEval/
│
├── download_corpus/
│   └── download_pdfs.py
│
├── preprocessing/
│   ├── chunking.py
│   ├── embeddings.py
│   ├── diagnostics.py
│   └── data/
│
├── retrieval/
│   ├── qdrant_setup.py
│   └── search.py
│
├── evals/
│   ├── embeddings/
│   └── reports/
│
├── scripts/
│   ├── ingest_corpus.py
│   └── query.py
│
├── tests/
│
├── docs/
│
├── pyproject.toml
└── README.md
```

Generated data, embedding caches, credentials, and corpus artifacts should be managed separately from source code.

## 10. Current Limitations

Current limitations include:

- No sufficiently validated, general-purpose gold evaluation dataset for the current corpus.
- Historical embedding evaluation artifacts require consistency checks and reproduction against the current corpus.
- Chunk identifiers and evidence mappings require careful version management.
- ColPali visual retrieval is not yet integrated.
- Vector quantisation trade-offs require controlled benchmarking.
- Comprehensive concurrency and load-testing results are not yet established.
- Answer generation and answer faithfulness are outside the current retrieval evaluator's scope.

## 11. Project Objective

The long-term objective is to move from isolated RAG experiments towards reproducible evaluation and architectural comparison infrastructure.

Rather than providing another predefined RAG pipeline, RagAndEval aims to investigate:

**How should a retrieval architecture be selected when quality, latency, storage, compute and workload requirements compete with one another?**

The intended outcome is a system that helps developers understand architectural trade-offs, reproduce experiments, and make evidence-based retrieval configuration decisions.

---

**Status:** Active experimental development(project is still in progress)
