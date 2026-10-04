"""
Compare embedding models on quality, latency, VRAM and GPU temperature in one run.

Benchmarks the 5 shortlisted candidates (open-weight / self-hostable only --
Voyage-3-m-exp was dropped earlier since it's API-only):
    - Yuan-embedding-2.0-en  (0.8B, fp16, no quantization needed)
    - E5-Mistral-7B-Instruct (7B, 4-bit)
    - Qwen3-Embedding-8B     (8B, 4-bit)
    

NOTE on the two "custom" models: GritLM and NV-Embed-v2 do not load through
sentence-transformers -- they need their own package / API, so they use
separate load_*/encode_* functions below (see the LOADERS dict). The
sentence-transformers path (Yuan, E5-Mistral, Qwen3) is the well-trodden one;
the GritLM and NV-Embed-v2 paths are more likely to need small fixes on your
machine (exact quantization kwarg support, gritlm package version, etc.) --
test each with --limit-chunks small before a full run.

These metrics (recall@k, MRR, paper-hit@5) are pure vector-similarity ranking
against a known-correct chunk -- the right, cheap tool for choosing an
embedding model. This is deliberately separate from DeepEval, which comes in
later for full RAG *generation* quality (faithfulness, answer relevancy) once
a generator is wired in -- that needs an LLM-as-judge call per question, which
would be the wrong (slow, noisy) tool for this narrower comparison.

Inputs:
  CHUNKS_DIR/*.json : output of chunking.py, one file per paper, each with a
                       top-level "chunks" list of {"chunk_id", "document_id",
                       "text", ...}.
  evals/embeddings/evals_v2.jsonl : 50 questions with gold chunk/paper IDs and
                       an explicit list of candidate paper IDs. All chunks in
                       those 64 paper files are used as retrieval candidates.

This benchmark includes multi-chunk and cross-paper evidence questions over
the complete 64-paper corpus.

future scope - add a pause button to allow the user to pause the script if the GPU is overheating 
and add time stamps everywhere to track the time taken for each step of the process.

Run:  uv run python evals/embeddings/compare_embeddings_v2.py
Outputs: evals/embeddings/results_v2.csv, evals/embeddings/ranks_v2.csv,
         evals/embeddings/embeddings_2/<model>_chunks_v2.npy, evals/embeddings/logs/
"""
import gc
import hashlib
import json
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, BitsAndBytesConfig

# ----------------------------------------------------------------------------
# PATHS -- adjust CHUNKS_DIR if this script doesn't sit where assumed
# relative to preprocessing/data/chunks/ (see chunking.py's layout)
# ----------------------------------------------------------------------------
EVALS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EVALS_DIR.parents[1]
CHUNKS_DIR = PROJECT_DIR / "preprocessing" / "data" / "chunks"
EVAL_FILE = EVALS_DIR / "evals_v2.jsonl"
LOGS_DIR = EVALS_DIR / "logs"

# ----------------------------------------------------------------------------
# CONFIG: the 5 shortlisted models
# ----------------------------------------------------------------------------
MODELS = [
    {
        "label": "yuan-0.8B-fp16",
        "name": "IEITYuan/Yuan-embedding-2.0-en",
        "loader": "sentence_transformers",
        "4bit": False,
        "batch_size": 16,
        "query_prefix": "",  # no instruction needed for this model
    },
    {
        "label": "e5-mistral-7B-4bit",
        "name": "intfloat/e5-mistral-7b-instruct",
        "loader": "sentence_transformers",
        "4bit": True,
        "batch_size": 2,  # 6GB card -- keep conservative
        # Queries get the Instruct prefix; corpus chunks do NOT.
        "query_prefix": "Instruct: Given a question about a research paper, retrieve the passage that answers it\nQuery: ",
    },
    {
        "label": "qwen3-8B-4bit",
        "name": "Qwen/Qwen3-Embedding-8B",
        "loader": "sentence_transformers",
        "4bit": True,
        "batch_size": 2,
        "prompt_name": "query",  # sentence-transformers applies the built-in query instruction
        "tokenizer_kwargs": {"padding_side": "left"},
    },
]

MAX_SEQ_LEN = 512  # chunks are ~400 tokens under chunking.py's neutral tiktoken count; a model's own
                    # tokenizer can run a bit higher on the same text, so this leaves headroom
KS = (1, 5, 10)
N_LATENCY_QUERIES = 20  # queries timed one at a time (realistic RAG usage)
COOLDOWN_S = 45  # laptop GPU thermals under sustained 7-8B loads -- bump this if you see throttling
OUT_DIR = EVALS_DIR / "embeddings_2"
OUT_DIR.mkdir(exist_ok=True)
RUN_DIR = EVALS_DIR


# ----------------------------------------------------------------------------
# GPU temperature sampler (polls nvidia-smi in the background)
# ----------------------------------------------------------------------------
class TempSampler:
    def __init__(self, interval=2.0):
        self.interval = interval
        self.temps = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _read(self):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                text=True,
            )
            return int(out.strip().splitlines()[0])
        except Exception:
            return None

    def _run(self):
        while not self._stop.is_set():
            t = self._read()
            if t is not None:
                self.temps.append(t)
            self._stop.wait(self.interval)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()
        return max(self.temps) if self.temps else float("nan")


# ----------------------------------------------------------------------------
# Data + metrics
# ----------------------------------------------------------------------------
def load_data():
    """
    V2 multi-evidence evaluation.

    Loads candidate chunks from every paper listed in evals_v2.jsonl, then validates
    the gold chunk and paper IDs against that full candidate corpus.
    """
    if not EVAL_FILE.exists():
        raise FileNotFoundError(
            f"{EVAL_FILE} not found. Create evals_v2.jsonl with candidate_paper_ids "
            f"and questions containing question, gold_chunk_ids, and gold_paper_ids."
        )

    records = [json.loads(line) for line in EVAL_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) < 2 or records[0].get("record_type") != "dataset_metadata":
        raise ValueError("evals_v2.jsonl must start with dataset metadata, followed by question records.")
    dataset = records[0]
    evals = records[1:]
    target_papers = dataset.get("candidate_paper_ids")
    if not isinstance(target_papers, list) or not target_papers:
        raise ValueError("evals_v2.jsonl metadata must contain a non-empty candidate_paper_ids array.")
    if len(target_papers) != len(set(target_papers)):
        raise ValueError("candidate_paper_ids contains duplicates.")
    if len(evals) != 50:
        raise ValueError(f"evals_v2.jsonl must contain exactly 50 questions; found {len(evals)}.")

    chunk_files = {path.stem.removesuffix("_chunks"): path for path in CHUNKS_DIR.glob("*_chunks.json")}
    if set(target_papers) != set(chunk_files):
        missing = sorted(set(target_papers) - set(chunk_files))
        unlisted = sorted(set(chunk_files) - set(target_papers))
        raise ValueError(f"Candidate paper list must match all chunk files. Missing files: {missing}; unlisted files: {unlisted}")

    chunks = []

    for paper_id in target_papers:
        path = chunk_files[paper_id]

        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("document_id") != paper_id:
            raise ValueError(f"Chunk file {path} has document_id {doc.get('document_id')!r}, expected {paper_id!r}.")
        for c in doc.get("chunks", []):
            chunks.append({
                "id": c["chunk_id"],
                "paper": doc["document_id"],
                "text": c["text"],
            })

    if not chunks:
        raise ValueError("No candidate chunks found for the papers listed in evals_v2.jsonl.")
    expected_chunk_count = dataset.get("candidate_chunk_count")
    if expected_chunk_count is not None and len(chunks) != expected_chunk_count:
        raise ValueError(
            f"Candidate corpus has {len(chunks)} chunks; evals_v2.jsonl expects {expected_chunk_count}."
        )

    chunk_ids = [c["id"] for c in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Chunk corpus contains duplicate chunk_id values; fix the chunk IDs before evaluating.")
    id_to_idx = {c["id"]: i for i, c in enumerate(chunks)}

    # Do not silently drop malformed/stale questions.
    for i, e in enumerate(evals):
        if not e.get("question") or not e.get("gold_chunk_ids") or not e.get("gold_paper_ids"):
            raise ValueError(f"evals_v2.jsonl question record {i + 1} needs question, gold_chunk_ids, and gold_paper_ids.")
        missing_ids = [cid for cid in e["gold_chunk_ids"] if cid not in id_to_idx]
        if missing_ids:
            raise ValueError(
                f"evals_v2.jsonl question record {i + 1} references missing chunk IDs: {missing_ids}"
            )

        missing_papers = [
            pid for pid in e["gold_paper_ids"]
            if pid not in target_papers
        ]
        if missing_papers:
            raise ValueError(
                f"evals_v2.jsonl question record {i + 1} has invalid gold_paper_ids: {missing_papers}"
            )
        actual_papers = {chunks[id_to_idx[cid]]["paper"] for cid in e["gold_chunk_ids"]}
        if actual_papers != set(e["gold_paper_ids"]):
            raise ValueError(
                f"evals_v2.jsonl question record {i + 1} gold_paper_ids do not match the papers of its gold chunks."
            )

    gold_indices = [
        [id_to_idx[cid] for cid in e["gold_chunk_ids"]]
        for e in evals
    ]
    papers = np.array([c["paper"] for c in chunks])

    return chunks, evals, gold_indices, papers


def score(chunk_embs, query_embs, gold_indices, papers):
    """
    Multi-evidence retrieval metrics.

    Evidence recall/coverage = fraction of required gold chunks retrieved.
    Evidence precision = fraction of retrieved top-k chunks that are gold.
    Paper coverage = fraction of required gold papers represented in top-k.
    """
    scores = query_embs @ chunk_embs.T
    ranked = np.argsort(-scores, axis=1)

    out = {}

    for k in KS:
        topk = ranked[:, :k]

        evidence_recalls = []
        evidence_precisions = []
        paper_coverages = []

        for i, gold in enumerate(gold_indices):
            gold_set = set(gold)
            retrieved_set = set(topk[i].tolist())

            hits = len(gold_set & retrieved_set)
            evidence_recalls.append(hits / len(gold_set))
            evidence_precisions.append(hits / k)

            required_papers = set(papers[g] for g in gold)
            retrieved_papers = set(papers[j] for j in topk[i])
            paper_coverages.append(
                len(required_papers & retrieved_papers) / len(required_papers)
            )

        out[f"evidence_recall@{k}"] = float(np.mean(evidence_recalls))
        out[f"evidence_precision@{k}"] = float(np.mean(evidence_precisions))

        # Coverage is defined as the fraction of required evidence chunks
        # covered by the retrieved set; retained as a named metric for V2.
        out[f"evidence_coverage@{k}"] = float(np.mean(evidence_recalls))
        out[f"paper_coverage@{k}"] = float(np.mean(paper_coverages))

    return out, ranked

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bnb_config():
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)


# ----------------------------------------------------------------------------
# Loader / encoder pairs, one per backend
# ----------------------------------------------------------------------------
def load_st(cfg):
    model_kwargs, st_kwargs = {}, {}
    if cfg["4bit"]:
        model_kwargs["quantization_config"] = bnb_config()
        # Qwen3-Embedding-8B was verified on the RTX 3050 6GB
        # without device_map; device_map="auto" caused the earlier
        # dispatch validation error.
        if cfg["label"] != "qwen3-8B-4bit":
            model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.float16
        st_kwargs["device"] = "cuda"
    if "tokenizer_kwargs" in cfg:
        st_kwargs["tokenizer_kwargs"] = cfg["tokenizer_kwargs"]
    model = SentenceTransformer(cfg["name"], model_kwargs=model_kwargs, **st_kwargs)
    model.max_seq_length = MAX_SEQ_LEN
    return model


def encode_st(model, cfg, texts, batch_size, is_query):
    kwargs = dict(batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
    if is_query:
        if "prompt_name" in cfg:
            kwargs["prompt_name"] = cfg["prompt_name"]
        elif cfg.get("query_prefix"):
            texts = [cfg["query_prefix"] + t for t in texts]
    return model.encode(texts, **kwargs)



LOADERS = {
    "sentence_transformers": (load_st, encode_st),
}


# ----------------------------------------------------------------------------
# Cache helpers
# ----------------------------------------------------------------------------
def load_cached_result(label, num_chunks, num_evals):
    chunk_cache = OUT_DIR / f"{label}_chunks_v2.npy"
    results_file = RUN_DIR / "results_v2.csv"
    ranks_file = RUN_DIR / "ranks_v2.csv"

    if not chunk_cache.exists() or not results_file.exists() or not ranks_file.exists():
        return None

    try:
        chunk_embs = np.load(chunk_cache, mmap_mode="r")
        if chunk_embs.ndim != 2 or chunk_embs.shape[0] != num_chunks:
            return None

        results_df = pd.read_csv(results_file)
        result_rows = results_df[results_df["model"] == label]
        if result_rows.empty:
            return None

        ranks_df = pd.read_csv(ranks_file)
        if label not in ranks_df.columns or len(ranks_df) != num_evals:
            return None

        row = result_rows.iloc[0].to_dict()
        ranks = ranks_df[label].to_numpy()

        print(f"Using cached results for {label} -- model will not be loaded or recomputed.")
        return row, ranks
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Benchmark one model
# ----------------------------------------------------------------------------
def run_model(cfg, chunks, evals, gold_indices, papers):
    label = cfg["label"]

    cached = load_cached_result(label, len(chunks), len(evals))
    if cached is not None:
        return cached

    print(f"\n=== {label} ===")
    load_fn, encode_fn = LOADERS[cfg["loader"]]
    chunk_texts = [c["text"] for c in chunks]
    fingerprint = hashlib.sha256("\n".join(c["id"] + c["text"] for c in chunks).encode()).hexdigest()
    questions = [e["question"] for e in evals]

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated()

    # ---- load
    t0 = time.perf_counter()
    model = load_fn(cfg)
    sync()
    load_s = time.perf_counter() - t0
    log_event(f"{label}: model load complete in {load_s:.1f}s")
    vram_loaded_gb = (torch.cuda.memory_allocated() - base_mem) / 1024**3

    # ---- warm-up (kernel compilation, cache) so timings are fair
    encode_fn(model, cfg, chunk_texts[:8], cfg["batch_size"], is_query=False)
    sync()
    log_event(f"{label}: warm-up complete")
    torch.cuda.reset_peak_memory_stats()

    # ---- encode all chunks (timed, with temperature sampling)
    sampler = TempSampler()
    sampler.start()
    encode_s = 0.0
    checkpoint = OUT_DIR / f"{label}_resume.npy"
    checkpoint_meta = OUT_DIR / f"{label}_resume.json"
    completed = 0
    chunk_embs = None
    if checkpoint.exists() and checkpoint_meta.exists():
        try:
            meta = json.loads(checkpoint_meta.read_text(encoding="utf-8"))
            if meta.get("fingerprint") == fingerprint and meta.get("num_chunks") == len(chunks):
                completed = min(int(meta.get("completed", 0)), len(chunks))
                encode_s = float(meta.get("encode_seconds", 0.0))
                chunk_embs = np.load(checkpoint, mmap_mode="r+")
                if chunk_embs.shape[0] != len(chunks):
                    completed, chunk_embs = 0, None
        except Exception:
            completed, chunk_embs = 0, None
    log_event(f"{label}: resuming chunk embeddings at {completed}/{len(chunks)}")
    group_size = cfg["batch_size"] * 16
    for start in range(completed, len(chunks), group_size):
        stop = min(start + group_size, len(chunks))
        batch_t0 = time.perf_counter()
        group_embs = np.asarray(encode_fn(model, cfg, chunk_texts[start:stop], cfg["batch_size"], is_query=False))
        sync()
        encode_s += time.perf_counter() - batch_t0
        if chunk_embs is None or completed == 0:
            chunk_embs = np.lib.format.open_memmap(checkpoint, mode="w+", dtype=group_embs.dtype,
                                                  shape=(len(chunks), group_embs.shape[1]))
        chunk_embs[start:stop] = group_embs
        chunk_embs.flush()
        checkpoint_meta.write_text(json.dumps({"fingerprint": fingerprint, "num_chunks": len(chunks),
                                              "completed": stop, "encode_seconds": encode_s}), encoding="utf-8")
        completed = stop
        log_event(f"{label}: checkpointed chunks {start + 1}-{stop}/{len(chunks)}")
    peak_temp = sampler.stop()
    vram_peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    log_event(f"{label}: chunk encoding complete in {encode_s:.1f}s; peak GPU temperature {peak_temp} C")

    # ---- query embeddings for scoring (batched)
    query_embs = encode_fn(model, cfg, questions, cfg["batch_size"], is_query=True)
    log_event(f"{label}: encoded {len(questions)} evaluation questions")

    # ---- single-query latency (what a RAG app would experience)
    lat = []
    for q in questions[:N_LATENCY_QUERIES]:
        t0 = time.perf_counter()
        encode_fn(model, cfg, [q], 1, is_query=True)
        sync()
        lat.append((time.perf_counter() - t0) * 1000)
    log_event(f"{label}: measured single-question latency for {len(lat)} questions")

    # ---- quality
    metrics, ranks = score(np.asarray(chunk_embs), np.asarray(query_embs), gold_indices, papers)
    log_event(f"{label}: scoring complete")
    np.save(OUT_DIR / f"{label}_chunks_v2.npy", chunk_embs)

    row = {
        "model": label,
        "dim": np.asarray(chunk_embs).shape[1],
        **{k: round(v, 3) for k, v in metrics.items()},
        "chunks/s": round(len(chunks) / encode_s, 1),
        "encode_total_s": round(encode_s, 1),
        "query_ms(median)": round(float(np.median(lat)), 1),
        "load_s": round(load_s, 1),
        "vram_loaded_GB": round(vram_loaded_gb, 2),
        "vram_peak_GB": round(vram_peak_gb, 2),
        "peak_gpu_temp_C": peak_temp,
    }

    # ---- free the GPU before the next model
    del model, chunk_embs, query_embs
    gc.collect()
    torch.cuda.empty_cache()
    return row, ranks


def main():
    assert torch.cuda.is_available(), "CUDA GPU not found"
    chunks, evals, gold_indices, papers = load_data()
    print(f"{len(chunks)} chunks from {len(set(papers))} candidate papers, {len(evals)} eval questions")

    rows, all_ranks = [], {}
    for cfg in MODELS:
        try:
            row, ranks = run_model(cfg, chunks, evals, gold_indices, papers)
            rows.append(row)
            all_ranks[cfg["label"]] = ranks
        except Exception as e:  # e.g. CUDA OOM, or a gated-model auth error: keep going with the other models
            log_event(f"!! {cfg['label']} failed: {type(e).__name__}: {e}")
            gc.collect()
            torch.cuda.empty_cache()
        log_event(f"Cooling down for {COOLDOWN_S}s")
        time.sleep(COOLDOWN_S)
        log_event("Cooldown complete")

    df = pd.DataFrame(rows)
    df.to_csv(RUN_DIR / "results_v2.csv", index=False)
    log_event(f"Wrote results to {RUN_DIR / 'results_v2.csv'}")
    print("\n" + df.to_string(index=False))

    # per-question ranks, handy for looking at where models disagree
    rk = pd.DataFrame({
    "question": [e["question"] for e in evals],
    "question_type": [e["question_type"] for e in evals],
    "gold_chunk_ids": [
        json.dumps(e["gold_chunk_ids"]) for e in evals
    ]
})

    for model, ranked in all_ranks.items():
       rk[model] = [
        json.dumps([chunks[j]["id"] for j in row[:10]])
        for row in ranked
       ]

    rk.to_csv(RUN_DIR / "ranks_v2.csv", index=False)
    log_event(f"Wrote ranks to {RUN_DIR / 'ranks_v2.csv'}")

    print(f"\nV2 multi-evidence results written to {RUN_DIR / 'results_v2.csv'} and {RUN_DIR / 'ranks_v2.csv'}")


if __name__ == "__main__":
    from datetime import datetime
    import sys

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, value):
            for stream in self.streams:
                stream.write(value)
                stream.flush()

        def flush(self):
            for stream in self.streams:
                stream.flush()

    def log_event(message):
        line = f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}"
        print(line)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"compare_embeddings_v2_{datetime.now():%Y%m%d_%H%M%S}.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(original_stdout, log_file), Tee(original_stderr, log_file)
        try:
            log_event("Run started; using the full candidate corpus specified in evals_v2.jsonl")
            main()
            log_event("Run completed")
        except BaseException as exc:
            log_event(f"Run interrupted or failed: {type(exc).__name__}: {exc}")
            raise
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr
