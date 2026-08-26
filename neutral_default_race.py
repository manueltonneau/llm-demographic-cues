"""
For each (model, task) on neutral (unmarked) prompts, compute the
race_pred distribution: what does the model 'default' to?
"""
import os, glob
import pandas as pd

HERE      = os.path.dirname(os.path.abspath(__file__))             # the repo root
REPO_ROOT = os.environ.get("CUES_ROOT", os.path.dirname(HERE))
DATA_DIR  = os.environ.get("CUES_DATA_DIR", os.path.join(REPO_ROOT, "data"))
RP_DIR = os.path.join(DATA_DIR, "decoder_model_responses_race_pred")
OUT    = os.path.join(HERE, "results_recall")

from cues_io import require_dir, require_file, require_any
require_dir(RP_DIR, "race-prediction responses (data/decoder_model_responses_race_pred)")
os.makedirs(OUT, exist_ok=True)

VALID = {"Black", "White", "Unknown"}
MODELS = ["llama3.1", "olmo2", "gpt52"]
TASKS  = ["medical_advice", "legal_advice", "salary_rec"]

rows = []
for m in MODELS:
    for t in TASKS:
        f = os.path.join(RP_DIR, f"{t}_neutral_constrained_{m}_seed_0.csv")
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f, low_memory=False)
        df["pred"] = df["response_text"].astype(str).str.strip()
        df_valid = df[df["pred"].isin(VALID)]
        n = len(df_valid)
        n_raw = len(df)
        vc = df_valid["pred"].value_counts(normalize=True).to_dict()
        rows.append({
            "model": m, "task": t,
            "n_raw": n_raw, "n_valid": n,
            "pct_other": round(100*(n_raw - n)/n_raw, 2) if n_raw else 0,
            "pct_White":   round(100*vc.get("White", 0), 2),
            "pct_Black":   round(100*vc.get("Black", 0), 2),
            "pct_Unknown": round(100*vc.get("Unknown", 0), 2),
        })

out = pd.DataFrame(rows)
out.to_csv(os.path.join(OUT, "neutral_default_race.csv"), index=False)
print(out.to_string(index=False))

# Average across tasks per model
print("\n=== Avg across the 3 tasks (neutral / unmarked prompts) ===")
agg = out.groupby("model")[["pct_White", "pct_Black", "pct_Unknown"]].mean().round(2)
print(agg.to_string())
agg.to_csv(os.path.join(OUT, "neutral_default_race_by_model.csv"))
