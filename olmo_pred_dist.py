"""
For OLMo-2 across all cues × tasks: what is the race_pred distribution
overall, and conditional on the cued (true) race?
"""
import os
import pandas as pd

HERE      = os.path.dirname(os.path.abspath(__file__))             # the repo root
REPO_ROOT = os.environ.get("CUES_ROOT", os.path.dirname(HERE))
DATA_DIR  = os.environ.get("CUES_DATA_DIR", os.path.join(REPO_ROOT, "data"))
RP_DIR = os.path.join(DATA_DIR, "decoder_model_responses_race_pred")
OUT    = os.path.join(HERE, "results_recall")

from cues_io import require_dir, require_file, require_any
require_dir(RP_DIR, "race-prediction responses (data/decoder_model_responses_race_pred)")

VALID = {"Black", "White", "Unknown"}
TASKS = ["medical_advice", "legal_advice", "salary_rec"]
CUES  = [
    "neutral", "explicit", "dialect",
    "name_specific_rosenman", "name_specific_hayes_elder", "name_specific_tzioumis",
    "convo_prefix", "convo_prefix_prism",
]

def normalize_race(r):
    if pd.isna(r):
        return "None"
    s = str(r).strip().lower()
    if s in {"black", "black or african american"}:
        return "Black"
    if s == "white":
        return "White"
    if s == "none":
        return "None"
    return None

rows = []
for task in TASKS:
    for cue in CUES:
        f = os.path.join(RP_DIR, f"{task}_{cue}_constrained_olmo2_seed_0.csv")
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f, low_memory=False)
        df["pred"] = df["response_text"].astype(str).str.strip()
        df = df[df["pred"].isin(VALID)].copy()
        if "race" in df.columns:
            df["true"] = df["race"].apply(normalize_race)
        else:
            df["true"] = None

        for true_grp in ["Black", "White", "None", "ALL"]:
            sub = df if true_grp == "ALL" else df[df["true"] == true_grp]
            if len(sub) == 0:
                continue
            vc = sub["pred"].value_counts(normalize=True).to_dict()
            rows.append({
                "task": task, "cue": cue, "cued": true_grp, "n": len(sub),
                "pct_White":   round(100*vc.get("White", 0), 2),
                "pct_Black":   round(100*vc.get("Black", 0), 2),
                "pct_Unknown": round(100*vc.get("Unknown", 0), 2),
            })

out = pd.DataFrame(rows)
out.to_csv(os.path.join(OUT, "olmo_pred_dist.csv"), index=False)

# Aggregate across tasks: per cue × cued race
print("=== OLMo-2: race_pred distribution by cue, conditional on cued group ===")
agg = (out[out["cued"].isin(["Black","White","None","ALL"])]
       .groupby(["cue","cued"])[["pct_White","pct_Black","pct_Unknown"]]
       .mean().round(2).reset_index())
print(agg.to_string(index=False))
agg.to_csv(os.path.join(OUT, "olmo_pred_dist_avg.csv"), index=False)

# Headline: among rows with cued=Black, how often does OLMo say White vs Unknown vs Black?
print("\n=== When the cue is *Black* (true=Black), OLMo predicts: ===")
black_only = (out[out["cued"]=="Black"]
              .groupby("cue")[["pct_White","pct_Black","pct_Unknown"]]
              .mean().round(2))
print(black_only.to_string())
