"""
Black recall (and full per-class P/R/F1) for race_pred across:
  - models: llama3.1, olmo2, gpt52
  - tasks:  medical_advice, legal_advice, salary_rec
  - cues:   dialect, explicit, name_{rosenman,hayes_elder,tzioumis},
            convo_prefix (CAD), convo_prefix_prism (PRISM)

Output: one long CSV with columns
  model, task, cue, n, n_black_cued, n_white_cued,
  black_recall, white_recall,
  black_precision, white_precision,
  black_f1, white_f1
plus per-(model,task,cue) confusion-matrix CSVs.

Cued-race normalization:
  {"black", "Black or African American"} -> "Black"
  {"white", "White"}                     -> "White"
  "none"                                 -> "None"  (excluded from B/W recall)
"""
import os, glob, sys, json
import pandas as pd
import numpy as np
from collections import Counter

HERE      = os.path.dirname(os.path.abspath(__file__))             # the repo root
REPO_ROOT = os.environ.get("CUES_ROOT", os.path.dirname(HERE))
DATA_DIR  = os.environ.get("CUES_DATA_DIR", os.path.join(REPO_ROOT, "data"))
RP_DIR = os.path.join(DATA_DIR, "decoder_model_responses_race_pred")
OUT    = os.path.join(HERE, "results_recall")

from cues_io import require_dir, require_file, require_any
require_dir(RP_DIR, "race-prediction responses (data/decoder_model_responses_race_pred)")
os.makedirs(OUT, exist_ok=True)

VALID_PRED = {"Black", "White", "Unknown"}

CUES = [
    "dialect", "explicit",
    "name_specific_rosenman", "name_specific_hayes_elder", "name_specific_tzioumis",
    "convo_prefix", "convo_prefix_prism",
    "neutral",
]
TASKS  = ["medical_advice", "legal_advice", "salary_rec"]
MODELS = ["llama3.1", "olmo2", "gpt52"]

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
    return None  # unexpected → drop


def per_class_metrics(df_b, df_w):
    """For binary (Black vs White) cued classes, compute per-class P/R/F1."""
    out = {}
    out["n_black_cued"] = len(df_b)
    out["n_white_cued"] = len(df_w)

    if len(df_b):
        b_correct = (df_b["pred"] == "Black").sum()
        out["black_recall"] = b_correct / len(df_b)
    else:
        out["black_recall"] = np.nan

    if len(df_w):
        w_correct = (df_w["pred"] == "White").sum()
        out["white_recall"] = w_correct / len(df_w)
    else:
        out["white_recall"] = np.nan

    # Precision: among rows predicted Black/White, how many were actually so
    all_df = pd.concat([df_b, df_w], ignore_index=True) if (len(df_b) or len(df_w)) else None
    if all_df is None or len(all_df) == 0:
        out.update({"black_precision": np.nan, "white_precision": np.nan,
                    "black_f1": np.nan, "white_f1": np.nan})
        return out

    pb = all_df["pred"] == "Black"
    pw = all_df["pred"] == "White"
    tb = all_df["true"] == "Black"
    tw = all_df["true"] == "White"

    bp = (pb & tb).sum() / pb.sum() if pb.sum() else np.nan
    wp = (pw & tw).sum() / pw.sum() if pw.sum() else np.nan
    out["black_precision"] = bp
    out["white_precision"] = wp

    out["black_f1"] = (2*bp*out["black_recall"] / (bp + out["black_recall"])
                      if (pd.notna(bp) and pd.notna(out["black_recall"]) and (bp+out["black_recall"])>0) else np.nan)
    out["white_f1"] = (2*wp*out["white_recall"] / (wp + out["white_recall"])
                      if (pd.notna(wp) and pd.notna(out["white_recall"]) and (wp+out["white_recall"])>0) else np.nan)
    return out


def collect_one(model, task, cue):
    f = os.path.join(RP_DIR, f"{task}_{cue}_constrained_{model}_seed_0.csv")
    if not os.path.exists(f):
        return None
    df = pd.read_csv(f, low_memory=False)
    if "response_text" not in df.columns or "race" not in df.columns:
        return None
    df["pred"] = df["response_text"].astype(str).str.strip()
    df = df[df["pred"].isin(VALID_PRED)].copy()
    df["true"] = df["race"].apply(normalize_race)
    df = df.dropna(subset=["true"])
    return df


def confusion(df_all):
    """Confusion counts for Black/White/None cued × Black/White/Unknown predicted."""
    rows = ["Black", "White", "None"]
    cols = ["Black", "White", "Unknown"]
    cm = pd.DataFrame(0, index=rows, columns=cols, dtype=int)
    for r in rows:
        sub = df_all[df_all["true"] == r]
        for c in cols:
            cm.loc[r, c] = int((sub["pred"] == c).sum())
    return cm


def main():
    long_rows = []
    for model in MODELS:
        for task in TASKS:
            confs = {}
            for cue in CUES:
                df = collect_one(model, task, cue)
                if df is None or len(df) == 0:
                    continue
                df_b = df[df["true"] == "Black"]
                df_w = df[df["true"] == "White"]
                rec  = per_class_metrics(df_b, df_w)

                rec.update({
                    "model": model, "task": task, "cue": cue,
                    "n": len(df),
                })
                long_rows.append(rec)
                confs[cue] = confusion(df)

            # Save confusion matrices for this (model, task)
            mt_dir = os.path.join(OUT, f"{model}_{task}")
            os.makedirs(mt_dir, exist_ok=True)
            for cue, cm in confs.items():
                cm.to_csv(os.path.join(mt_dir, f"confusion_{cue}.csv"))

    df = pd.DataFrame(long_rows)
    cols = ["model","task","cue","n","n_black_cued","n_white_cued",
            "black_recall","white_recall",
            "black_precision","white_precision",
            "black_f1","white_f1"]
    df = df[cols]
    df.sort_values(["model","task","cue"], inplace=True)
    df.to_csv(os.path.join(OUT, "race_pred_recall_long.csv"), index=False)

    # Wide pivot: black recall per (model, task) × cue
    wide_br = df.pivot_table(index=["model","task"], columns="cue",
                             values="black_recall").round(4)
    wide_br.to_csv(os.path.join(OUT, "black_recall_wide.csv"))
    print("=== Black recall (rows: model × task, cols: cue) ===")
    print(wide_br.to_string())
    print()
    print("=== Long table head ===")
    print(df.head(30).to_string(index=False))
    print(f"\n{len(df)} rows total. Saved to {OUT}")


if __name__ == "__main__":
    main()
