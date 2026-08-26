"""
One-shot FK-grade cache builder.

For each (task, cue) prompts parquet, compute Flesch-Kincaid grade per row in
parallel and save to:
  fk_cache/{task}_{cue}.parquet

The cache stores all identity columns from the prompts parquet plus
`flesch_kincaid_grade`, so a downstream merge needs only the natural keys for
that cue (e.g., prompt_id+name).

Multiprocessing across CPU cores. Skips files where the cache already exists.
"""
import os, sys
from multiprocessing import Pool, cpu_count

import pandas as pd
import textstat

HERE       = os.path.dirname(os.path.abspath(__file__))            # the repo root
REPO_ROOT  = os.environ.get("CUES_ROOT", os.path.dirname(HERE))
DATA_DIR   = os.environ.get("CUES_DATA_DIR", os.path.join(REPO_ROOT, "data"))
PROMPT_DIR = os.path.join(DATA_DIR, "prompts")
CACHE_DIR  = os.path.join(HERE, "fk_cache")

from cues_io import require_dir, require_file, require_any, require_produced
require_dir(PROMPT_DIR, "prompt directory (data/prompts)")
os.makedirs(CACHE_DIR, exist_ok=True)

CUES = [
    "dialect", "explicit",
    "name_specific_rosenman", "name_specific_hayes_elder", "name_specific_tzioumis",
    "name_specific_an",
    "convo_prefix", "convo_prefix_prism",
    "neutral",
]
TASKS = ["medical_advice", "legal_advice", "salary_rec"]


def _fk(text):
    if not isinstance(text, str) or not text.strip():
        return float("nan")
    try:
        return float(textstat.flesch_kincaid_grade(text))
    except Exception:
        return float("nan")


def _fk_chunk(texts):
    return [_fk(t) for t in texts]


def build_one(task: str, cue: str, n_proc: int):
    f = os.path.join(PROMPT_DIR, task, f"{task}_{cue}_constrained.parquet")
    out = os.path.join(CACHE_DIR, f"{task}_{cue}.parquet")
    if not os.path.exists(f):
        return f"[skip] {task}/{cue} (no prompts)"
    if os.path.exists(out):
        return f"[cached] {task}/{cue}"
    df = pd.read_parquet(f)
    if "prompt" not in df.columns:
        return f"[skip] {task}/{cue} (no prompt col)"

    n = len(df)
    texts = df["prompt"].tolist()
    chunk_size = max(1, n // (n_proc * 4))
    chunks = [texts[i:i+chunk_size] for i in range(0, n, chunk_size)]

    with Pool(processes=n_proc) as pool:
        results = pool.map(_fk_chunk, chunks)

    fk = [v for r in results for v in r]
    df["flesch_kincaid_grade"] = fk
    df = df.drop(columns=["prompt"])
    df.to_parquet(out, index=False)
    return f"[ok ] {task}/{cue}  n={n:,}  saved {out}"


def main():
    n_proc = max(1, cpu_count() - 1)
    print(f"Using {n_proc} processes")
    built = 0
    for task in TASKS:
        for cue in CUES:
            msg = build_one(task, cue, n_proc)
            print(msg, flush=True)
            built += not msg.startswith("[skip]")
    require_produced(built, "cache files")


if __name__ == "__main__":
    main()
