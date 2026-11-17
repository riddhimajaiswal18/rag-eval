# Corpus-Aware RAG Evaluation Framework

> Understanding when, why, and under what conditions different RAG architectures work.

## 1. Overview

Retrieval-Augmented Generation (RAG) systems can be built using several architectural choices: fixed-size chunking, semantic chunking, parent-child retrieval, hybrid search, reranking, graph retrieval, hierarchical retrieval, and more.

However, there is no universally optimal RAG architecture.

The effectiveness of each technique depends on multiple factors, including:
- The structure and characteristics of the corpus.
- How information and evidence are distributed across documents.
- The complexity and nature of user queries.
- The relationships between entities and concepts in the corpus.
- Computational constraints such as latency, storage and retrieval cost.

A technique that performs exceptionally well on one corpus may provide little benefit or introduce unnecessary complexity on another.

This project aims to systematically investigate these relationships through controlled experiments and reproducible evaluations.

## 2. Core Objective

**To investigate how corpus characteristics and query requirements influence the effectiveness of different RAG architectures, and identify the conditions under which particular architectural choices provide measurable benefits.**

Rather than searching for a single best-performing RAG architecture, the project seeks to understand:

- When does a particular retrieval technique improve performance?
- What corpus characteristics make that technique effective?
- Which query types benefit from different retrieval architectures?
- When does additional architectural complexity provide little or no improvement?
- What trade-offs exist between retrieval quality, answer quality, latency, storage and computational cost?
- To what extent do experimental findings generalize across different corpora?

The ultimate goal is to develop an evidence-backed understanding of RAG architecture selection.

## 3. Research Questions

The project will investigate the following questions:

### RQ1. Retrieval Architecture

How do different retrieval architectures perform under different corpus and query conditions?

Architectures to investigate:
- BM25 / lexical retrieval
- Dense vector retrieval
- Hybrid retrieval
- Cross-encoder reranking
- ColBERT / multi-vector retrieval
- Graph-based retrieval
- Hierarchical retrieval approaches such as RAPTOR

### RQ2. Chunking Strategies

How do different chunking strategies affect evidence retrieval and downstream answer quality?

Strategies to investigate:
- Fixed-size chunking
- Recursive chunking
- Semantic chunking
- Structure-aware chunking
- Parent-child chunking
- Multi-level hierarchical chunking

### RQ3. Corpus Characteristics

Which properties of a corpus influence the effectiveness of different RAG architectures?

Potential characteristics:
- Document structure and hierarchy.
- Average document and section length.
- Information density.
- Evidence distribution across paragraphs and sections.
- Entity and relationship density.
- Cross-document dependencies.
- Consistency of document formatting.

### RQ4. Query Characteristics

How do different query requirements influence the choice of retrieval architecture?

Potential categories:
- Factual and direct retrieval.
- Exact terminology and keyword-dependent queries.
- Conceptual and semantic queries.
- Multi-hop reasoning.
- Cross-section evidence retrieval.
- Cross-document synthesis.
- Unanswerable queries.

### RQ5. Performance and Trade-offs

Do improvements in retrieval quality justify the additional computational and architectural complexity?

Evaluate:
- Retrieval effectiveness.
- Answer correctness and faithfulness.
- Retrieval and end-to-end latency.
- Indexing time.
- Storage requirements.
- Computational and inference costs.

### RQ6. Generalization

Do techniques that perform well on one corpus retain their effectiveness on structurally different corpora?

Investigate which findings generalize, which depend on corpus-specific characteristics, and why.

## 4. Guiding Principles

The project follows these principles:

1. **No universally optimal architecture:** No retrieval or chunking technique is assumed to be superior across all corpora and queries.

2. **Evidence-driven architecture selection:** Every architectural choice must have an identifiable hypothesis and measurable justification.

3. **Corpus-aware evaluation:** Corpus characteristics must be considered experimental variables, not merely implementation details.

4. **Query-aware evaluation:** Retrieval effectiveness must be evaluated against the information requirements of the queries.

5. **Controlled experimentation:** Change one major experimental variable at a time wherever possible, keeping other conditions consistent.

6. **Quality versus complexity:** An improvement in retrieval quality must be considered alongside its latency, storage and computational overhead.

7. **Reproducibility:** Experiments must preserve configurations, dataset versions, provenance, parameters and evaluation results.

8. **Generalization over benchmark optimization:** A technique should not be considered universally effective simply because it performs well on one benchmark.

## 5. Initial Datasets

The project will initially use two research-paper corpora with different roles.

### PeerQA

**Primary evaluation benchmark**

PeerQA provides human-annotated scientific question-answer pairs, evidence sentences and answerability labels.

It will initially be used to:
- Establish retrieval baselines.
- Evaluate evidence retrieval.
- Compare retrieval and chunking configurations.
- Evaluate answerability and abstention.
- Reproduce selected published retrieval baselines.

The initial implementation will use the 90-paper subset with bundled extracted text.

### Existing arXiv Research Corpus

**Secondary experimental corpus**

The existing collection of research papers will be retained for:
- Testing whether findings transfer beyond PeerQA.
- Exploring different document structures.
- Conducting additional robustness experiments.
- Investigating the effect of corpus characteristics on retrieval performance.

Synthetic questions may be used as supplementary experimental data, but their results will be distinguished from human-grounded benchmark results.

The two corpora will not be treated as interchangeable benchmarks. Their roles and available ground truth will be documented separately.

## 6. Evaluation Methodology

The framework will support controlled comparisons across different configurations.

### Retrieval Metrics
- Evidence Recall@K
- Precision@K
- Mean Reciprocal Rank (MRR)
- nDCG

### Generation Metrics
- Answer correctness
- Faithfulness
- Evidence and citation accuracy

### Operational Metrics
- Retrieval latency
- End-to-end latency
- Indexing time
- Index storage
- Context token consumption
- Computational and inference cost

### Experimental Method

For each experiment:

1. Define the research hypothesis.
2. Identify the relevant corpus and query characteristics.
3. Establish a baseline.
4. Implement the architectural variation.
5. Keep unrelated variables controlled.
6. Evaluate retrieval and generation quality.
7. Measure operational overhead.
8. Analyze where the technique helps, fails or introduces trade-offs.
9. Record reproducible findings.

## 7. Initial Implementation Priorities

The initial development will focus on establishing a reliable evaluation foundation.

| Priority | Component |
|---|---|
| 1 | Stable document, paragraph and sentence provenance |
| 2 | Dataset ingestion and standardized representation |
| 3 | BM25 retrieval baseline |
| 4 | Dense retrieval baseline |
| 5 | Hybrid retrieval and reranking |
| 6 | Chunking strategy comparisons |
| 7 | Parent-child and hierarchical retrieval |
| 8 | Evaluation and experiment tracking |
| 9 | Cross-corpus evaluation |

Graph retrieval, RAPTOR and other specialized architectures will be investigated when the corpus characteristics and query requirements provide a meaningful reason to test them.

They will not be implemented merely to increase the number of techniques in the framework.

## 8. Expected Contribution

The project does not initially aim to invent another standalone retrieval algorithm.

Its intended contribution is a reproducible evaluation framework and empirical investigation that:

- Compares different RAG architectures under controlled conditions.
- Connects retrieval performance to corpus and query characteristics.
- Identifies when architectural complexity provides measurable value.
- Studies the relationship between retrieval quality and answer quality.
- Quantifies operational trade-offs.
- Investigates the generalization of architectural choices across corpora.

The longer-term goal is to produce evidence-backed insights that help developers and researchers make better RAG architecture decisions.

## 9. What This Project Is Not

- It is not another basic RAG chatbot.
- It is not an attempt to combine every existing RAG technique.
- It is not an attempt to establish one universally optimal retrieval architecture.
- It is not primarily a synthetic benchmark generation project.
- It is not an implementation of GraphRAG or RAPTOR for the sake of using them.

**The central question remains:**

*Under what corpus characteristics and query requirements does a particular RAG architecture work better, why does it work, and is the improvement worth its additional complexity?*