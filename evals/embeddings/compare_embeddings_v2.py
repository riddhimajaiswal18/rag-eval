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
  evals/embeddings/eval_v2.json : [{"question": "...", "gold_chunk_ids": [...], "gold_paper_ids": [...]}, ...]
                       -- NOT YET CREATED. This still needs to be built before
                       this script can run; see the note in load_data().

cuurently we only test on these papers -
1706.03762 
2005.11401 
2309.06180 
2310.11511
2312.10997

this benchmark is only for multi-evidence questions; the corpus is restricted to the papers listed in eval_v2.json

future scope - add a pause button to allow the user to pause the script if the GPU is overheating 
and add time stamps everywhere to track the time taken for each step of the process.

Run:  uv run python evals/embeddings/compare_embeddings_v2.py
Outputs: evals/embeddings/results_v2.csv, evals/embeddings/ranks_v2.csv,
         evals/embeddings/embeddings_2/<model>_chunks_v2.npy, evals/embeddings/logs/
"""
import gc
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
EVAL_FILE = EVALS_DIR / "eval_v2.json"
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

    Loads only the paper chunk files explicitly referenced by eval_v2.json.
    This keeps the benchmark corpus restricted to the papers represented in
    this evaluation set instead of automatically using every chunk file.
    """
    if not EVAL_FILE.exists():
        raise FileNotFoundError(
            f"{EVAL_FILE} not found. Create eval_v2.json with "
            f"question, gold_chunk_ids, and gold_paper_ids."
        )

    evals = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    if not isinstance(evals, list) or not evals:
        raise ValueError("eval_v2.json must contain a non-empty list.")

    target_papers = sorted({
        paper_id
        for e in evals
        for paper_id in e["gold_paper_ids"]
    })

    chunks = []
    missing_files = []

    for paper_id in target_papers:
        path = CHUNKS_DIR / f"{paper_id}_chunks.json"
        if not path.exists():
            missing_files.append(str(path))
            continue

        doc = json.loads(path.read_text(encoding="utf-8"))
        for c in doc.get("chunks", []):
            chunks.append({
                "id": c["chunk_id"],
                "paper": doc["document_id"],
                "text": c["text"],
            })

    if missing_files:
        raise FileNotFoundError(
            "Missing chunk files referenced by eval_v2.json:\n"
            + "\n".join(missing_files)
        )

    if not chunks:
        raise ValueError("No chunks found for the papers in eval_v2.json.")

    id_to_idx = {c["id"]: i for i, c in enumerate(chunks)}

    # Do not silently drop malformed/stale questions in V2.
    for i, e in enumerate(evals):
        missing_ids = [cid for cid in e["gold_chunk_ids"] if cid not in id_to_idx]
        if missing_ids:
            raise ValueError(
                f"eval_v2.json entry {i} references missing chunk IDs: {missing_ids}"
            )

        missing_papers = [
            pid for pid in e["gold_paper_ids"]
            if pid not in target_papers
        ]
        if missing_papers:
            raise ValueError(
                f"eval_v2.json entry {i} has invalid gold_paper_ids: {missing_papers}"
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
    results_file = EVALS_DIR / "results_v2.csv"
    ranks_file = EVALS_DIR / "ranks_v2.csv"

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
    questions = [e["question"] for e in evals]

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated()

    # ---- load
    t0 = time.perf_counter()
    model = load_fn(cfg)
    sync()
    load_s = time.perf_counter() - t0
    vram_loaded_gb = (torch.cuda.memory_allocated() - base_mem) / 1024**3

    # ---- warm-up (kernel compilation, cache) so timings are fair
    encode_fn(model, cfg, chunk_texts[:8], cfg["batch_size"], is_query=False)
    sync()
    torch.cuda.reset_peak_memory_stats()

    # ---- encode all chunks (timed, with temperature sampling)
    sampler = TempSampler()
    sampler.start()
    t0 = time.perf_counter()
    chunk_embs = encode_fn(model, cfg, chunk_texts, cfg["batch_size"], is_query=False)
    sync()
    encode_s = time.perf_counter() - t0
    peak_temp = sampler.stop()
    vram_peak_gb = torch.cuda.max_memory_allocated() / 1024**3

    # ---- query embeddings for scoring (batched)
    query_embs = encode_fn(model, cfg, questions, cfg["batch_size"], is_query=True)

    # ---- single-query latency (what a RAG app would experience)
    lat = []
    for q in questions[:N_LATENCY_QUERIES]:
        t0 = time.perf_counter()
        encode_fn(model, cfg, [q], 1, is_query=True)
        sync()
        lat.append((time.perf_counter() - t0) * 1000)

    # ---- quality
    metrics, ranks = score(np.asarray(chunk_embs), np.asarray(query_embs), gold_indices, papers)
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
    print(f"{len(chunks)} chunks, {len(evals)} eval questions")

    rows, all_ranks = [], {}
    for cfg in MODELS:
        try:
            row, ranks = run_model(cfg, chunks, evals, gold_indices, papers)
            rows.append(row)
            all_ranks[cfg["label"]] = ranks
        except Exception as e:  # e.g. CUDA OOM, or a gated-model auth error: keep going with the other models
            print(f"!! {cfg['label']} failed: {type(e).__name__}: {e}")
            gc.collect()
            torch.cuda.empty_cache()
        print(f"Cooling down {COOLDOWN_S}s...")
        time.sleep(COOLDOWN_S)

    df = pd.DataFrame(rows)
    df.to_csv(EVALS_DIR / "results_v2.csv", index=False)
    print("\n" + df.to_string(index=False))

    # per-question ranks, handy for looking at where models disagree
    rk = pd.DataFrame({
    "question": [e["question"] for e in evals],
    "gold_chunk_ids": [
        json.dumps(e["gold_chunk_ids"]) for e in evals
    ]
})

    for model, ranked in all_ranks.items():
       rk[model] = [
        json.dumps([chunks[j]["id"] for j in row[:10]])
        for row in ranked
       ]

    rk.to_csv(EVALS_DIR / "ranks_v2.csv", index=False)  
    rk.insert(0, "question", [e["question"] for e in evals])
    rk.insert(1, "gold_chunk_ids", [json.dumps(e["gold_chunk_ids"]) for e in evals])
    rk.to_csv(EVALS_DIR / "ranks_v2.csv", index=False)

    print("\nV2 multi-evidence results written to results_v2.csv and ranks_v2.csv")


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

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"compare_embeddings_v2_{datetime.now():%Y%m%d_%H%M%S}.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(original_stdout, log_file), Tee(original_stderr, log_file)
        try:
            main()
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr
