"""
Replicate the three regression tables in the paper appendix:
  Table reg_medical (LLaMA-3.1, n=4,551,000)
  Table reg_legal   (LLaMA-3.1, n=5,125,000)
  Table reg_salary  (LLaMA-3.1, n=5,122,434)

Specs:
  (1) y ~ race_pred                              + prompt FE
  (2) y ~ race_pred + race                       + prompt FE
  (3) y ~ race_pred + race + flesch_kincaid_grade+ prompt FE

Reference categories:
  race_pred:  White
  race:       White

Race normalization to {Black, White, None}.
race_pred filtered to {Black, White, Unknown}; reference White.
For medical, drops the "an" name source (paper uses 3 name sources, not 4).
Salary outcome = numeric salary in USD, parsed from response_text.

Prompt fixed effects implemented by demeaning. Specs are fit without
intercept (- 1), so statsmodels reports the "uncentered R^2" matching
the paper.
"""
import os
import re
import gc
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from cues.paths import MASTER_DIR as SHARE, REGRESSIONS_LLAMA as OUT, require_dir, require_any
# Pre-built master parquets (medical_master.parquet, legal_salary_master.parquet);
# see README for how to obtain/build them.

require_dir(SHARE, "master-table directory (data/masters)")
require_any([os.path.join(SHARE, "medical_master.parquet"),
             os.path.join(SHARE, "legal_salary_master.parquet")], "master tables")
os.makedirs(OUT, exist_ok=True)

VALID_RACE_PRED = {"Black", "White", "Unknown"}


def normalize_race(s):
    if pd.isna(s):
        return "None"
    s = str(s).strip()
    sl = s.lower()
    if sl in {"black", "black or african american"}:
        return "Black"
    if sl == "white":
        return "White"
    if sl == "none":
        return "None"
    return None  # other unexpected values → drop


def parse_salary(text):
    if pd.isna(text):
        return np.nan
    s = re.sub(r"[$,\s]", "", str(text).strip()).rstrip(".")
    try:
        return float(s)
    except ValueError:
        m = re.search(r"\d+", s)
        return float(m.group()) if m else np.nan


def demean(df, cols):
    out = df.copy()
    g = df.groupby("prompt_id", sort=False)
    for c in cols:
        out[c] = df[c] - g[c].transform("mean")
    return out


def add_dummies(df, ref_race_pred="White", ref_race="White"):
    """Create dummies with explicit reference categories."""
    rp = df["race_pred"].astype("category")
    rp = rp.cat.set_categories([ref_race_pred] + [c for c in rp.cat.categories if c != ref_race_pred])
    df["race_pred_Black"]   = (df["race_pred"] == "Black").astype(float)
    df["race_pred_Unknown"] = (df["race_pred"] == "Unknown").astype(float)
    df["race_Black"]        = (df["race"] == "Black").astype(float)
    df["race_None"]         = (df["race"] == "None").astype(float)
    return df


def fit_spec(df, label, formula, demean_cols, out_dir, label_short):
    df_d = demean(df, demean_cols)
    print(f"  fitting {label_short}: {formula}  (n={len(df_d):,})")
    res = smf.ols(formula, data=df_d).fit()
    coefs = pd.DataFrame({
        "variable": res.params.index,
        "coef":     res.params.values,
        "std_err":  res.bse.values,
        "t":        res.tvalues.values,
        "p":        res.pvalues.values,
    })
    safe = label_short.replace(" ", "_").replace("+", "plus")
    coefs.to_csv(os.path.join(out_dir, f"coef_{safe}.csv"), index=False)
    return {
        "spec": label,
        "n": int(res.nobs),
        "r2_uncentered": res.rsquared,
        "aic": res.aic,
        "bic": res.bic,
        "params": res.params.to_dict(),
        "ses":    res.bse.to_dict(),
    }


def run_paper_specs(df, y_col, out_dir, task_label):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n========== {task_label}  n={len(df):,} ==========")

    df = add_dummies(df)
    base_dummies = ["race_pred_Black", "race_pred_Unknown"]
    race_dummies = ["race_Black", "race_None"]
    fk = "flesch_kincaid_grade"

    specs = [
        ("(1) Inferred Race Only",
         f"{y_col} ~ {' + '.join(base_dummies)} - 1",
         [y_col] + base_dummies,
         "spec1"),
        ("(2) + Actual Race",
         f"{y_col} ~ {' + '.join(base_dummies + race_dummies)} - 1",
         [y_col] + base_dummies + race_dummies,
         "spec2"),
        ("(3) + Actual Race + FK",
         f"{y_col} ~ {' + '.join(base_dummies + race_dummies + [fk])} - 1",
         [y_col] + base_dummies + race_dummies + [fk],
         "spec3"),
    ]
    rows = []
    for label, formula, dcols, short in specs:
        rows.append(fit_spec(df, label, formula, dcols, out_dir, short))

    # Side-by-side coefficient table
    keys = base_dummies + race_dummies + [fk]
    rows_df = []
    for k in keys:
        rec = {"variable": k}
        for i, r in enumerate(rows):
            c = r["params"].get(k, np.nan)
            s = r["ses"].get(k, np.nan)
            rec[f"({i+1}) coef"] = c
            rec[f"({i+1}) se"]   = s
        rows_df.append(rec)
    side = pd.DataFrame(rows_df)
    side.to_csv(os.path.join(out_dir, "side_by_side_coefficients.csv"), index=False)

    summary = pd.DataFrame([
        {"spec": r["spec"], "n": r["n"], "r2_uncentered": r["r2_uncentered"],
         "aic": r["aic"], "bic": r["bic"]}
        for r in rows
    ])
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    print(side.to_string(index=False))
    print(summary.to_string(index=False))
    return side, summary


def load_medical_llama():
    cols = ["message_id", "prompt_id", "race", "race_pred", "id_cue",
            "response_final", "flesch_kincaid_grade", "model"]
    df = pd.read_parquet(
        os.path.join(SHARE, "medical_master.parquet"),
        columns=cols,
        filters=[("model", "==", "llama3.1")],
    )
    print(f"  loaded {len(df):,} medical (llama3.1)")

    # Drop the "an" name source not used by paper (3 sources only)
    is_an = df["message_id"].str.startswith("name_specific_an_")
    print(f"  dropping {is_an.sum():,} rows from name source 'an'")
    df = df[~is_an].copy()

    df["race"]      = df["race"].apply(normalize_race)
    df = df[df["race"].notna()]
    df = df[df["race_pred"].isin(VALID_RACE_PRED)]
    df["response"]  = (df["response_final"].astype(str).str.lower() == "yes").astype(float)
    return df


def load_legalsalary_llama(category):
    cols = ["message_id", "prompt_id", "race", "race_pred", "id_cue",
            "response_text", "response_final",
            "flesch_kincaid_grade", "model", "category"]
    df = pd.read_parquet(
        os.path.join(SHARE, "legal_salary_master.parquet"),
        columns=cols,
        filters=[("model", "==", "llama3.1"), ("category", "==", category)],
    )
    print(f"  loaded {len(df):,} {category} (llama3.1)")

    df["race"] = df["race"].apply(normalize_race)
    df = df[df["race"].notna()]
    df = df[df["race_pred"].isin(VALID_RACE_PRED)]

    if category == "legal_advice":
        df["response"] = (df["response_final"].astype(str).str.lower() == "yes").astype(float)
    else:
        df["response"] = df["response_text"].apply(parse_salary)
        before = len(df)
        df = df[df["response"].notna()].copy()
        print(f"  dropped {before - len(df):,} salary rows that failed numeric parse")
    return df


def main():
    print("=== MEDICAL ===")
    df = load_medical_llama()
    run_paper_specs(df, "response", os.path.join(OUT, "medical"), "medical_advice | llama3.1")
    del df; gc.collect()

    print("\n=== LEGAL ===")
    df = load_legalsalary_llama("legal_advice")
    run_paper_specs(df, "response", os.path.join(OUT, "legal"), "legal_advice | llama3.1")
    del df; gc.collect()

    print("\n=== SALARY ===")
    df = load_legalsalary_llama("salary_rec")
    run_paper_specs(df, "response", os.path.join(OUT, "salary"), "salary_rec | llama3.1")


if __name__ == "__main__":
    main()
