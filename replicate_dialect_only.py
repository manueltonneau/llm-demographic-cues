"""
Re-run the regression analysis filtered to dialect (AAVE) cue only,
LLaMA-3.1.

Caveats specific to the dialect subset:
- Only seed=0 is present (no seed replications), so each prompt is observed once
  and prompt fixed effects would absorb all variation.
- Cued race is constant (all rows are AAVE → cued Black), so the actual cued race
  coefficient is not identifiable.

We therefore run plain OLS across dialect prompts (no FE) for:
  (1)  y ~ race_pred                              [White ref]
  (2)  y ~ race_pred + flesch_kincaid_grade
"""
import os, re, gc, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

HERE      = os.path.dirname(os.path.abspath(__file__))             # the repo root
REPO_ROOT = os.environ.get("CUES_ROOT", os.path.dirname(HERE))
DATA_DIR  = os.environ.get("CUES_DATA_DIR", os.path.join(REPO_ROOT, "data"))
# Pre-built master parquets (medical_master.parquet, legal_salary_master.parquet);
# see README for how to obtain/build them.
SHARE = os.environ.get("CUES_MASTER_DIR", os.path.join(DATA_DIR, "masters"))
OUT   = os.path.join(HERE, "results_dialect")

from cues_io import require_dir, require_file, require_any
require_dir(SHARE, "master-table directory (data/masters)")
require_any([os.path.join(SHARE, "medical_master.parquet"),
             os.path.join(SHARE, "legal_salary_master.parquet")], "master tables")
os.makedirs(OUT, exist_ok=True)

VALID_RACE_PRED = {"Black", "White", "Unknown"}


def parse_salary(text):
    if pd.isna(text):
        return np.nan
    s = re.sub(r"[$,\s]", "", str(text).strip()).rstrip(".")
    try:
        return float(s)
    except ValueError:
        m = re.search(r"\d+", s)
        return float(m.group()) if m else np.nan


def add_dummies(df):
    df["race_pred_Black"]   = (df["race_pred"] == "Black").astype(float)
    df["race_pred_Unknown"] = (df["race_pred"] == "Unknown").astype(float)
    return df


def run_specs(df, y_col, out_dir, label):
    os.makedirs(out_dir, exist_ok=True)
    df = add_dummies(df)
    print(f"\n========== {label}  n={len(df):,} ==========")
    print(f"  race_pred dist: {df['race_pred'].value_counts().to_dict()}")
    print(f"  FK grade: mean={df['flesch_kincaid_grade'].mean():.2f}  std={df['flesch_kincaid_grade'].std():.2f}")

    base = ["race_pred_Black", "race_pred_Unknown"]
    fk = "flesch_kincaid_grade"

    specs = [
        ("(1) Inferred Race Only",
         f"{y_col} ~ {' + '.join(base)}", "spec1"),
        ("(2) + FK grade",
         f"{y_col} ~ {' + '.join(base)} + {fk}", "spec2"),
    ]

    rows = []
    for label_, formula, short in specs:
        print(f"  fitting {short}: {formula}")
        res = smf.ols(formula, data=df).fit()
        coefs = pd.DataFrame({
            "variable": res.params.index,
            "coef":     res.params.values,
            "std_err":  res.bse.values,
            "t":        res.tvalues.values,
            "p":        res.pvalues.values,
        })
        coefs.to_csv(os.path.join(out_dir, f"coef_{short}.csv"), index=False)
        rows.append({"spec": label_, "n": int(res.nobs), "r2": res.rsquared,
                     "params": res.params.to_dict(), "ses": res.bse.to_dict(),
                     "ps": res.pvalues.to_dict()})

    keys = ["Intercept"] + base + [fk]
    side_rows = []
    for k in keys:
        rec = {"variable": k}
        for i, r in enumerate(rows):
            rec[f"({i+1}) coef"] = r["params"].get(k, np.nan)
            rec[f"({i+1}) se"]   = r["ses"].get(k, np.nan)
            rec[f"({i+1}) p"]    = r["ps"].get(k, np.nan)
        side_rows.append(rec)
    side = pd.DataFrame(side_rows)
    side.to_csv(os.path.join(out_dir, "side_by_side.csv"), index=False)
    summary = pd.DataFrame([{"spec": r["spec"], "n": r["n"], "r2": r["r2"]} for r in rows])
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    print()
    print(side.to_string(index=False))
    print(summary.to_string(index=False))
    return side, summary


def load_medical_dialect():
    df = pd.read_parquet(
        os.path.join(SHARE, "medical_master.parquet"),
        columns=["prompt_id", "race_pred", "id_cue", "response_final",
                 "flesch_kincaid_grade", "model"],
        filters=[("model", "==", "llama3.1"), ("id_cue", "==", "dialect")],
    )
    df = df[df["race_pred"].isin(VALID_RACE_PRED)]
    df["response"] = (df["response_final"].astype(str).str.lower() == "yes").astype(float)
    return df


def load_legalsalary_dialect(category):
    df = pd.read_parquet(
        os.path.join(SHARE, "legal_salary_master.parquet"),
        columns=["prompt_id", "race_pred", "id_cue", "response_text",
                 "response_final", "flesch_kincaid_grade", "model", "category"],
        filters=[("model", "==", "llama3.1"),
                 ("category", "==", category),
                 ("id_cue", "==", "dialect")],
    )
    df = df[df["race_pred"].isin(VALID_RACE_PRED)]
    if category == "legal_advice":
        df["response"] = (df["response_final"].astype(str).str.lower() == "yes").astype(float)
    else:
        df["response"] = df["response_text"].apply(parse_salary)
        df = df[df["response"].notna()].copy()
    return df


def main():
    print("=== MEDICAL (dialect only) ===")
    df = load_medical_dialect()
    run_specs(df, "response", os.path.join(OUT, "medical"), "medical_advice | llama3.1 | dialect")
    del df; gc.collect()

    print("\n=== LEGAL (dialect only) ===")
    df = load_legalsalary_dialect("legal_advice")
    run_specs(df, "response", os.path.join(OUT, "legal"), "legal_advice | llama3.1 | dialect")
    del df; gc.collect()

    print("\n=== SALARY (dialect only) ===")
    df = load_legalsalary_dialect("salary_rec")
    run_specs(df, "response", os.path.join(OUT, "salary"), "salary_rec | llama3.1 | dialect")


if __name__ == "__main__":
    main()
