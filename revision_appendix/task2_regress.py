"""Task 2 step 2: re-estimate the prompt-FE regressions with extended linguistic
controls and report the inferred-race coefficient (race_pred_Black) before/after.

Base spec  = published spec (3): race_pred_{Black,Unknown} + race_{Black,None} + FK
Extended   = base + token_len + ttr + dep_depth + vader + polite
All within-prompt demeaned (prompt fixed effects), OLS with no intercept.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

HERE     = os.path.dirname(os.path.abspath(__file__))              # revision_appendix
RA       = HERE
REPL_DIR = os.path.dirname(HERE)                                   # the repo root
MASTER = {
    "llama3.1": os.path.join(REPL_DIR, "results_all_models/llama3.1/{task}/master.parquet"),
    "olmo2":    os.path.join(REPL_DIR, "results_all_models/olmo2/{task}/master.parquet"),
    "gpt52":    os.path.join(REPL_DIR, "results_all_models/gpt52/{task}/master.parquet"),
}
MODELS = ["llama3.1", "olmo2", "gpt52"]
TASKS = ["medical_advice", "legal_advice", "salary_rec"]
NEWF = ["token_len", "ttr", "dep_depth", "vader", "polite"]

def norm_race(r):
    s = str(r).strip().lower()
    if s in ("black", "black or african american"):
        return "Black"
    if s == "white":
        return "White"
    if s == "none":
        return "None"
    return s

def add_dummies(df):
    df["race_pred_Black"] = (df["race_pred"] == "Black").astype(float)
    df["race_pred_Unknown"] = (df["race_pred"] == "Unknown").astype(float)
    df["race_Black"] = (df["race"] == "Black").astype(float)
    df["race_None"] = (df["race"] == "None").astype(float)
    return df

def demean(df, cols):
    out = df.copy()
    g = df.groupby("prompt_id", sort=False)
    for c in cols:
        out[c] = df[c] - g[c].transform("mean")
    return out

def fit(df, cols):
    formula = "response ~ " + " + ".join(cols) + " - 1"
    res = smf.ols(formula, data=demean(df, cols)).fit()
    return res

rows = []
for model in MODELS:
    for task in TASKS:
        mf = MASTER[model].format(task=task)
        if not os.path.exists(mf):
            print(f"[skip] no master {model}/{task}")
            continue
        m = pd.read_parquet(mf)
        m["race"] = m["race"].map(norm_race)
        feat = pd.read_parquet(os.path.join(RA, f"task2_features_{task}.parquet"))
        feat["race"] = feat["race"].map(norm_race)
        feat = feat[["id_cue", "prompt_id", "race"] + NEWF].drop_duplicates(["id_cue", "prompt_id", "race"])
        m = m.merge(feat, on=["id_cue", "prompt_id", "race"], how="left")
        m = add_dummies(m)

        base_cols = ["race_pred_Black", "race_pred_Unknown", "race_Black", "race_None", "flesch_kincaid_grade"]
        ext_cols = base_cols + NEWF
        d = m.dropna(subset=base_cols + NEWF + ["response", "prompt_id"]).copy()
        # standardize continuous controls (FK + new) for numerical stability; does not affect race_pred_Black
        for c in ["flesch_kincaid_grade"] + NEWF:
            sd = d[c].std()
            if sd and sd > 0:
                d[c] = (d[c] - d[c].mean()) / sd

        rb = fit(d, base_cols)
        re_ = fit(d, ext_cols)
        rec = dict(model=model, task=task, n=int(rb.nobs),
                   # inferred race (race_pred_Black)
                   inf_base_coef=rb.params["race_pred_Black"], inf_base_p=rb.pvalues["race_pred_Black"],
                   inf_ext_coef=re_.params["race_pred_Black"], inf_ext_p=re_.pvalues["race_pred_Black"],
                   # cued race (race_Black)
                   cue_base_coef=rb.params["race_Black"], cue_base_p=rb.pvalues["race_Black"],
                   cue_ext_coef=re_.params["race_Black"], cue_ext_p=re_.pvalues["race_Black"])
        rows.append(rec)
        print(f"{model:9s} {task:14s} n={rec['n']:>9,}  "
              f"INF base={rec['inf_base_coef']:+.4g}(p{rec['inf_base_p']:.0e})->ext={rec['inf_ext_coef']:+.4g}(p{rec['inf_ext_p']:.0e})  "
              f"CUE base={rec['cue_base_coef']:+.4g}->ext={rec['cue_ext_coef']:+.4g}", flush=True)
        del m, d

res = pd.DataFrame(rows)
res.to_csv(os.path.join(RA, "task2_extended_controls.csv"), index=False)
print("\nSAVED task2_extended_controls.csv")
print(res.to_string(index=False))
