"""
Replicate the regressions referenced in the paper section "Impact on model behavior"
(\\S\\ref{app:reg_analysis}). Adapted from analyze_regressions_medical.py and
analyze_regressions_salary_legal.py, but reads the parquet master files locally instead
of MySQL.

For each (task, model), runs the same eight nested OLS specs:
  1. race_pred
  2. race_pred + prompt_id FE      (within transform / demeaning)
  3. race_pred + race
  4. race_pred + id_cue
  5. race_pred + readability
  6. race_pred + sentiment
  7. race_pred + readability + sentiment
  8. Full model + prompt_id FE     (within transform / demeaning)
"""

import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

HERE      = os.path.dirname(os.path.abspath(__file__))             # the repo root
REPO_ROOT = os.environ.get("CUES_ROOT", os.path.dirname(HERE))
DATA_DIR  = os.environ.get("CUES_DATA_DIR", os.path.join(REPO_ROOT, "data"))
# Pre-built master parquets (medical_master.parquet, legal_salary_master.parquet);
# see README for how to obtain/build them.
SHARE_DIR = os.environ.get("CUES_MASTER_DIR", os.path.join(DATA_DIR, "masters"))
OUT_ROOT = os.path.join(HERE, "results")

from cues_io import require_dir, require_file, require_any
require_dir(SHARE_DIR, "master-table directory (data/masters)")
require_any([os.path.join(SHARE_DIR, "medical_master.parquet"),
             os.path.join(SHARE_DIR, "legal_salary_master.parquet")], "master tables")
os.makedirs(OUT_ROOT, exist_ok=True)

READABILITY_COLS = ["type_token_ratio", "flesch_kincaid_grade", "avg_sentence_length"]


def normalize_salary(text):
    if pd.isna(text):
        return None
    s = re.sub(r"[$,\s]", "", str(text).strip()).rstrip(".")
    try:
        return float(s)
    except ValueError:
        m = re.search(r"\d+", s)
        return float(m.group()) if m else None


def load_medical():
    df = pd.read_parquet(os.path.join(SHARE_DIR, "medical_master.parquet"))
    df["id_cue"] = df["id_cue"].fillna("neutral")
    df["response"] = (df["response_final"].astype(str).str.lower() == "yes").astype(float)
    df["category"] = "medical_advice"
    return df


def load_legal_salary_subset(category, model):
    """Load only the rows we need for one (category, model) cell, with light dtypes."""
    cols = [
        "category", "model", "prompt_id", "race", "race_pred", "id_cue",
        "response_text", "response_final",
        "type_token_ratio", "flesch_kincaid_grade", "avg_sentence_length",
        "prompt_vader",
    ]
    df = pd.read_parquet(
        os.path.join(SHARE_DIR, "legal_salary_master.parquet"),
        columns=cols,
        filters=[("category", "==", category), ("model", "==", model)],
    )
    df["id_cue"] = df["id_cue"].fillna("neutral")
    if category == "legal_advice":
        df["response"] = (df["response_final"].astype(str).str.lower() == "yes").astype(float)
    else:
        print(f"    normalizing {len(df):,} salary rows…")
        df["response"] = df["response_text"].apply(normalize_salary)
    df = df.drop(columns=["response_text", "response_final"])
    return df


def demean_by_prompt(df, columns):
    out = df.copy()
    grp = df.groupby("prompt_id")
    for col in columns:
        if col in df.columns:
            out[col] = df[col] - grp[col].transform("mean")
    return out


def run_specs(df_sub, out_dir, label):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n--- {label} | n={len(df_sub):,} ---")
    if len(df_sub) == 0:
        print("  empty, skipping")
        return None

    valid_race_pred = {"Black", "White", "Unknown"}
    n_before = len(df_sub)
    df_sub = df_sub[df_sub["race_pred"].isin(valid_race_pred)].copy()
    if len(df_sub) < n_before:
        print(f"  dropped {n_before - len(df_sub):,} rows with unparseable race_pred")

    cols_to_dummy = [c for c in ["race_pred", "race", "id_cue"] if df_sub[c].nunique() > 1]
    df_d = pd.get_dummies(df_sub, columns=cols_to_dummy, drop_first=True, dtype=float)

    race_pred_dummies = [c for c in df_d.columns if c.startswith("race_pred_")]
    race_dummies = [
        c for c in df_d.columns if c.startswith("race_") and c not in race_pred_dummies
    ]
    id_cue_dummies = [c for c in df_d.columns if c.startswith("id_cue_")]

    readability_cols = [
        c for c in READABILITY_COLS
        if c in df_d.columns and not df_d[c].isna().all()
    ]
    sentiment_col = ["prompt_vader"] if "prompt_vader" in df_d.columns and not df_d["prompt_vader"].isna().all() else []

    cols_to_demean = (
        ["response"] + race_pred_dummies + race_dummies + id_cue_dummies
        + readability_cols + sentiment_col
    )
    df_demeaned = demean_by_prompt(df_d, cols_to_demean)

    specs = []
    if race_pred_dummies:
        specs.append(("1. race_pred", f"response ~ {' + '.join(race_pred_dummies)}", False))
        specs.append(("2. race_pred + prompt_id FE",
                     f"response ~ {' + '.join(race_pred_dummies)} - 1", True))
        if race_dummies:
            specs.append(("3. race_pred + race",
                         f"response ~ {' + '.join(race_pred_dummies)} + {' + '.join(race_dummies)}",
                         False))
        if id_cue_dummies:
            specs.append(("4. race_pred + id_cue",
                         f"response ~ {' + '.join(race_pred_dummies)} + {' + '.join(id_cue_dummies)}",
                         False))
        if readability_cols:
            specs.append(("5. race_pred + readability",
                         f"response ~ {' + '.join(race_pred_dummies)} + {' + '.join(readability_cols)}",
                         False))
        if sentiment_col:
            specs.append(("6. race_pred + sentiment",
                         f"response ~ {' + '.join(race_pred_dummies)} + prompt_vader", False))
            if readability_cols:
                specs.append(("7. race_pred + readability + sentiment",
                             f"response ~ {' + '.join(race_pred_dummies)} + {' + '.join(readability_cols)} + prompt_vader",
                             False))

        full = f"response ~ {' + '.join(race_pred_dummies)}"
        if readability_cols:
            full += f" + {' + '.join(readability_cols)}"
        if sentiment_col:
            full += " + prompt_vader"
        if id_cue_dummies:
            full += f" + {' + '.join(id_cue_dummies)}"
        if race_dummies:
            full += f" + {' + '.join(race_dummies)}"
        full += " - 1"
        specs.append(("8. Full model + prompt_id FE", full, True))

    summary = []
    for name, formula, use_fe in specs:
        data = df_demeaned if use_fe else df_d
        print(f"  fitting: {name} (n={len(data):,}, vars≈{formula.count('+')+1})")
        try:
            res = smf.ols(formula, data=data).fit()
            summary.append({"spec": name, "r2": res.rsquared, "n": int(res.nobs)})

            safe = (name.lower().replace(" ", "_").replace(".", "")
                    .replace("+", "plus").replace("(", "").replace(")", ""))
            coef_df = pd.DataFrame({
                "variable": res.params.index,
                "coef": res.params.values,
                "std_err": res.bse.values,
                "t": res.tvalues.values,
                "p": res.pvalues.values,
            })
            coef_df.to_csv(os.path.join(out_dir, f"coef_{safe}.csv"), index=False)
        except Exception as e:
            print(f"    ERROR: {e}")
            summary.append({"spec": name, "r2": np.nan, "n": 0, "error": str(e)})

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    print(summary_df.to_string(index=False))
    return summary_df


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else ["medical", "legal", "salary"]

    if "medical" in targets:
        print("\n========== MEDICAL ==========")
        df_med = load_medical()
        df_med = df_med.dropna(subset=["response"])
        for m in sorted(df_med["model"].unique()):
            run_specs(df_med[df_med["model"] == m].copy(),
                      os.path.join(OUT_ROOT, "medical_advice", m),
                      f"medical_advice | {m}")
        del df_med

    if "legal" in targets or "salary" in targets:
        print("\n========== LEGAL + SALARY ==========")
        cats = []
        if "legal" in targets: cats.append("legal_advice")
        if "salary" in targets: cats.append("salary_rec")
        for cat in cats:
            for m in ("llama3.1", "olmo2"):
                print(f"\n[load] {cat} | {m}")
                sub = load_legal_salary_subset(cat, m)
                sub = sub.dropna(subset=["response"])
                run_specs(sub, os.path.join(OUT_ROOT, cat, m), f"{cat} | {m}")
                del sub
                import gc; gc.collect()


if __name__ == "__main__":
    main()
