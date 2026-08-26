"""
Dialect (AAVE)-only regression, extended to all three audited models
(LLaMA-3.1 8B, OLMo2-7B, GPT-5.2).

Mirrors replicate_dialect_only.py (LLaMA-only) but sources from the same
per-(task, cue, model) CSVs + FK cache used by build_master_and_regress.py,
so it is consistent with the cross-model regression tables (Tables 9-10).

Caveats specific to the dialect subset (identical to the LLaMA-only script):
- Only seed=0 is present, so each prompt is observed once and prompt fixed
  effects would absorb all variation -> no FE.
- Cued race is constant (all rows are AAVE -> cued Black), so the cued-race
  coefficient is not identifiable and is dropped.

We therefore run plain OLS across dialect prompts for:
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
RP_DIR = os.path.join(DATA_DIR, "decoder_model_responses_race_pred")
RS_DIR = os.path.join(DATA_DIR, "decoder_model_responses_cleaned")
FK_DIR = os.path.join(HERE, "fk_cache")
OUT    = os.path.join(HERE, "results_dialect_allmodels")
os.makedirs(OUT, exist_ok=True)

VALID_RACE_PRED = {"Black", "White", "Unknown"}
MODELS = ["llama3.1", "olmo2", "gpt52"]
TASKS  = ["medical_advice", "legal_advice", "salary_rec"]


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
    df["race_pred_White"]   = (df["race_pred"] == "White").astype(float)
    df["race_pred_Unknown"] = (df["race_pred"] == "Unknown").astype(float)
    return df


def load_task_model(task, model):
    # race prediction (inferred race)
    rp_f = os.path.join(RP_DIR, f"{task}_dialect_constrained_{model}_seed_0.csv")
    rp = pd.read_csv(rp_f, low_memory=False).rename(columns={"response_text": "race_pred"})
    rp["race_pred"] = rp["race_pred"].astype(str).str.strip()
    rp = rp[rp["race_pred"].isin(VALID_RACE_PRED)][["prompt_id", "race_pred"]]
    rp = rp.drop_duplicates(subset=["prompt_id"])

    # response
    rs_f = os.path.join(RS_DIR, f"{task}_dialect_constrained_{model}_seed_0.csv")
    rs = pd.read_csv(rs_f, low_memory=False)
    if task == "salary_rec":
        rs["response"] = rs["salary_final"].apply(parse_salary)
    else:
        rs["response"] = (rs["response_final"].astype(str).str.lower().str.strip() == "yes").astype(float)
    rs = rs[["prompt_id", "response"]].dropna(subset=["response"]).drop_duplicates(subset=["prompt_id"])

    # FK grade (per-task, shared across models)
    fk = pd.read_parquet(os.path.join(FK_DIR, f"{task}_dialect.parquet"))
    fk = fk[["prompt_id", "flesch_kincaid_grade"]].drop_duplicates(subset=["prompt_id"])

    df = rp.merge(rs, on="prompt_id", how="inner").merge(fk, on="prompt_id", how="inner")
    return df


def run_cell(df, model, task):
    """Run y ~ inferred-race (+ FK) on dialect prompts for one model x task.

    Reference category = modal race_pred. Only non-reference categories with
    >0 observations enter the design (avoids the rank-deficiency seen when a
    model never predicts a category on dialect prompts). Returns the Black
    coefficient (relative to ref) and the FK coefficient from the +FK spec.
    """
    df = add_dummies(df)
    counts = df["race_pred"].value_counts()
    ref = counts.idxmax()
    others = [c for c in ["Black", "White", "Unknown"]
              if c != ref and counts.get(c, 0) > 0]
    race_terms = [f"race_pred_{c}" for c in others]
    fk = "flesch_kincaid_grade"

    rec = {"model": model, "task": task, "n": len(df), "ref": ref,
           "race_pred_dist": counts.to_dict()}

    # spec 1: inferred race only ; spec 2: + FK
    for short, terms in [("spec1", race_terms), ("spec2", race_terms + [fk])]:
        rhs = " + ".join(terms) if terms else "1"
        res = smf.ols(f"response ~ {rhs}", data=df).fit()
        for term in terms:
            rec[f"{short}_{term}_coef"] = res.params.get(term, np.nan)
            rec[f"{short}_{term}_p"]    = res.pvalues.get(term, np.nan)
        rec[f"{short}_r2"] = res.rsquared

    # convenience extraction for the table
    rec["black_estimable"] = "race_pred_Black" in race_terms
    rec["black_coef"]  = rec.get("spec2_race_pred_Black_coef", np.nan)
    rec["black_p"]     = rec.get("spec2_race_pred_Black_p", np.nan)
    rec["fk_coef"]     = rec.get("spec2_flesch_kincaid_grade_coef", np.nan)
    rec["fk_p"]        = rec.get("spec2_flesch_kincaid_grade_p", np.nan)
    return rec


def main():
    recs = []
    for model in MODELS:
        for task in TASKS:
            df = load_task_model(task, model)
            rec = run_cell(df, model, task)
            recs.append(rec)
            be = "yes" if rec["black_estimable"] else "NO (Black has 0 obs)"
            print(f"\n== {model:9s} {task:14s} n={rec['n']:5d}  ref={rec['ref']:8s}  "
                  f"dist={rec['race_pred_dist']}")
            print(f"   Black estimable: {be}")
            if rec["black_estimable"]:
                print(f"   Black coef (vs {rec['ref']}): {rec['black_coef']:+.5g}  p={rec['black_p']:.4g}")
            print(f"   FK coef: {rec['fk_coef']:+.5g}  p={rec['fk_p']:.4g}")
            del df; gc.collect()
    out = pd.DataFrame(recs)
    out.to_csv(os.path.join(OUT, "dialect_allmodels_summary.csv"), index=False)
    print(f"\nwrote {os.path.join(OUT, 'dialect_allmodels_summary.csv')}")


if __name__ == "__main__":
    main()
