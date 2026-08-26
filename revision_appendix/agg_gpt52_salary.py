"""GPT-5.2 salary aggregation (seed 0 only) at (cue,race,prompt_id) granularity,
to extend the Task 3 job-tier figure to all three models. Mirrors agg_per_seed.py."""
import os, re, glob
import pandas as pd

HERE      = os.path.dirname(os.path.abspath(__file__))             # revision_appendix
REPO_ROOT = os.environ.get("CUES_ROOT", os.path.dirname(os.path.dirname(HERE)))
DATA_DIR  = os.environ.get("CUES_DATA_DIR", os.path.join(REPO_ROOT, "data"))
DATA = os.path.join(DATA_DIR, "decoder_model_responses_cleaned")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cues_io import require_dir, require_file, require_any
require_dir(DATA, "cleaned responses (data/decoder_model_responses_cleaned)")
OUT = os.path.join(HERE, "gpt52_salary_agg.parquet")
RE = re.compile(r"^salary_rec_(.+)_constrained_gpt52_seed_0\.csv$")

def norm_race(r):
    r = str(r).strip().lower()
    if r in ("black", "black or african american"): return "black"
    if r == "white": return "white"
    if r == "none": return "none"
    return r

frames = []
for path in sorted(glob.glob(os.path.join(DATA, "salary_rec_*_gpt52_seed_0.csv"))):
    m = RE.match(os.path.basename(path))
    if not m: continue
    method = m.group(1)
    if method == "name_specific_an": continue
    df = pd.read_csv(path, usecols=["prompt_id", "race", "salary_final"], low_memory=False)
    df["race"] = df["race"].map(norm_race)
    df = df.dropna(subset=["salary_final"])
    df["val"] = df["salary_final"].astype(float)
    g = df.groupby(["prompt_id", "race"], sort=False)["val"].agg(["sum", "count"]).reset_index()
    g["model"] = "gpt52"; g["task"] = "salary_rec"; g["method"] = method; g["seed"] = 0
    frames.append(g.rename(columns={"sum": "sum_val", "count": "n"}))
    print(f"done {method}: {len(g)} rows", flush=True)
    del df, g

out = pd.concat(frames, ignore_index=True)[
    ["model", "task", "method", "seed", "prompt_id", "race", "sum_val", "n"]]
out.to_parquet(OUT, index=False)
print("SAVED", OUT, out.shape, flush=True)
