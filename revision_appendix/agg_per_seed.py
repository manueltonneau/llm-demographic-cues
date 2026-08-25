"""
Single streaming pass over LLaMA-3.1 / OLMo2 cleaned responses.
Builds a compact per-(model, task, cue, race, seed, prompt_id) aggregate
(sum of outcome, n) that powers:
  - Task 1: per-seed B/W outcome ratios + within-cue cross-source correlations
  - Task 3: per-job-profile baseline salary tiers

Outcome convention matches the figure pipeline:
  - binary tasks (medical/legal): y = 1[response_final.lower()=="yes"], count ALL rows
  - salary: salary_final (numeric), dropna
Excludes name_specific_an (excluded everywhere in the paper) and gpt52/qwen3.
"""
import os, re, glob
import numpy as np
import pandas as pd

HERE      = os.path.dirname(os.path.abspath(__file__))             # replication/revision_appendix
REPO_ROOT = os.environ.get("CUES_ROOT", os.path.dirname(os.path.dirname(HERE)))
DATA_DIR  = os.environ.get("CUES_DATA_DIR", os.path.join(REPO_ROOT, "data"))
DATA = os.path.join(DATA_DIR, "decoder_model_responses_cleaned")
OUT  = HERE

MODELS = {"llama3.1", "olmo2"}
EXCLUDE_METHODS = {"name_specific_an"}
BINARY_TASKS = {"medical_advice", "legal_advice"}
SALARY_TASK = "salary_rec"

FNAME_RE = re.compile(r"^(medical_advice|legal_advice|salary_rec)_(.+)_constrained_(.+)_seed_(\d+)\.csv$")

def norm_race(r):
    r = str(r).strip().lower()
    if r in ("black", "black or african american"):
        return "black"
    if r == "white":
        return "white"
    if r == "none":
        return "none"
    return r

frames = []
paths = sorted(glob.glob(os.path.join(DATA, "*.csv")))
n_done = 0
for path in paths:
    m = FNAME_RE.match(os.path.basename(path))
    if not m:
        continue
    task, method, model, seed = m.group(1), m.group(2), m.group(3), int(m.group(4))
    if model not in MODELS or method in EXCLUDE_METHODS:
        continue

    if task == SALARY_TASK:
        df = pd.read_csv(path, usecols=["prompt_id", "race", "salary_final"], low_memory=False)
        df["race"] = df["race"].map(norm_race)
        df = df.dropna(subset=["salary_final"])
        df["val"] = df["salary_final"].astype(float)
    else:
        df = pd.read_csv(path, usecols=["prompt_id", "race", "response_final"])
        df["race"] = df["race"].map(norm_race)
        df["val"] = (df["response_final"].astype(str).str.lower() == "yes").astype(float)

    g = df.groupby(["prompt_id", "race"], sort=False)["val"].agg(["sum", "count"]).reset_index()
    g["model"] = model; g["task"] = task; g["method"] = method; g["seed"] = seed
    frames.append(g)
    n_done += 1
    print(f"[{n_done}] {os.path.basename(path)}  rows->{len(g)}", flush=True)
    del df, g

agg = pd.concat(frames, ignore_index=True)
agg = agg.rename(columns={"sum": "sum_val", "count": "n"})
agg = agg[["model", "task", "method", "seed", "prompt_id", "race", "sum_val", "n"]]
out_path = os.path.join(OUT, "per_seed_agg.parquet")
agg.to_parquet(out_path, index=False)
print("SAVED", out_path, agg.shape, flush=True)
print(agg.groupby(["model", "task"]).seed.nunique().to_string(), flush=True)
