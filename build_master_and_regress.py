"""
Build a per-(model, task) master DataFrame and run the paper's regression specs:
  (1) y ~ race_pred_Black + race_pred_Unknown                          + prompt FE
  (2) + race_Black + race_None                                         + prompt FE
  (3) + flesch_kincaid_grade                                           + prompt FE

Reference categories:  race_pred=White, race=White
Race normalization:    {Black, Black or African American} → Black
                       White → White
                       none → None  (only for cued race; race_pred uses Unknown)
race_pred filter:      {Black, White, Unknown}

Convention (matches the LLaMA paper appendix):
- seed 0 only
- Drop the `an` name source for medical_advice
- For salary, response = numeric salary in USD (salary_final column)

Inputs:
  - Prompts:   data/prompts/{task}/{task}_{cue}_constrained.parquet  (has prompt text + identity columns)
  - Race-pred: data/decoder_model_responses_race_pred/{task}_{cue}_constrained_{model}_seed_0.csv
  - Response:  data/decoder_model_responses_cleaned/{task}_{cue}_constrained_{model}_seed_0.csv

Outputs:
  - replication/results_all_models/{model}/{task}/  (coefficients, summary, side-by-side)
"""
import os
import sys
import gc
import warnings
from typing import List

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import textstat

warnings.filterwarnings("ignore")

HERE       = os.path.dirname(os.path.abspath(__file__))            # the replication/ dir
REPO_ROOT  = os.environ.get("CUES_ROOT", os.path.dirname(HERE))
DATA_DIR   = os.environ.get("CUES_DATA_DIR", os.path.join(REPO_ROOT, "data"))
PROMPT_DIR = os.path.join(DATA_DIR, "prompts")
RP_DIR     = os.path.join(DATA_DIR, "decoder_model_responses_race_pred")
RS_DIR     = os.path.join(DATA_DIR, "decoder_model_responses_cleaned")
OUT_ROOT   = os.path.join(HERE, "results_all_models")
FK_CACHE   = os.path.join(HERE, "fk_cache")
os.makedirs(OUT_ROOT, exist_ok=True)

CUES = [
    "dialect",
    "explicit",
    "name_specific_rosenman",
    "name_specific_hayes_elder",
    "name_specific_tzioumis",
    "name_specific_an",   # dropped for medical to match paper
    "convo_prefix",
    "convo_prefix_prism",
    "neutral",
]
TASKS = ["medical_advice", "legal_advice", "salary_rec"]
VALID_PRED = {"Black", "White", "Unknown"}


def keys_for_cue(cue: str) -> List[str]:
    if cue.startswith("name_specific_"):
        return ["prompt_id", "name"]
    if cue.startswith("convo_prefix"):
        return ["prompt_id", "convo_concat_id"]
    if cue == "explicit":
        return ["prompt_id", "race", "gender"]
    return ["prompt_id"]   # dialect, neutral


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


def fk_grade(text):
    if not isinstance(text, str) or not text.strip():
        return np.nan
    try:
        return float(textstat.flesch_kincaid_grade(text))
    except Exception:
        return np.nan


def load_prompts(task: str, cue: str, with_text: bool = False) -> pd.DataFrame:
    """Prefer the FK cache (replication/fk_cache/<task>_<cue>.parquet) which
    contains all identity columns + flesch_kincaid_grade pre-computed. Falls
    back to the prompts parquet if cache is absent (and computes FK on demand
    in the caller)."""
    cache_f = os.path.join(FK_CACHE, f"{task}_{cue}.parquet")
    if os.path.exists(cache_f) and not with_text:
        return pd.read_parquet(cache_f)

    f = os.path.join(PROMPT_DIR, task, f"{task}_{cue}_constrained.parquet")
    if not os.path.exists(f):
        return None
    df = pd.read_parquet(f)
    if "prompt" not in df.columns:
        return None
    if not with_text:
        df = df.drop(columns=["prompt"])
    return df


def load_race_pred(task: str, cue: str, model: str) -> pd.DataFrame:
    f = os.path.join(RP_DIR, f"{task}_{cue}_constrained_{model}_seed_0.csv")
    if not os.path.exists(f):
        return None
    df = pd.read_csv(f, low_memory=False)
    if "response_text" not in df.columns:
        return None
    df = df.rename(columns={"response_text": "race_pred"})
    df["race_pred"] = df["race_pred"].astype(str).str.strip()
    return df[df["race_pred"].isin(VALID_PRED)].copy()


def load_response(task: str, cue: str, model: str, seed: int = 0) -> pd.DataFrame:
    f = os.path.join(RS_DIR, f"{task}_{cue}_constrained_{model}_seed_{seed}.csv")
    if not os.path.exists(f):
        return None
    df = pd.read_csv(f, low_memory=False)
    return df


def build_master(model: str, task: str) -> pd.DataFrame:
    print(f"\n=== building master: {model} | {task} ===")
    pieces = []
    cues = list(CUES)
    if task == "medical_advice":
        cues = [c for c in cues if c != "name_specific_an"]   # paper drops `an`

    for cue in cues:
        rp = load_race_pred(task, cue, model)
        if rp is None or len(rp) == 0:
            print(f"  [skip {cue}] no race_pred for {model}")
            continue
        rs = load_response(task, cue, model, seed=0)
        if rs is None:
            print(f"  [skip {cue}] no response for {model} seed 0")
            continue
        # Prefer the FK cache (already has FK + identity columns)
        prompts = load_prompts(task, cue, with_text=False)
        if prompts is None:
            print(f"  [skip {cue}] no prompts parquet")
            continue
        has_fk = "flesch_kincaid_grade" in prompts.columns

        keys = keys_for_cue(cue)

        # response column varies by task
        if task == "salary_rec":
            if "salary_final" not in rs.columns:
                print(f"  [skip {cue}] no salary_final col")
                continue
            rs_resp = rs[keys + ["salary_final"]].copy()
            rs_resp = rs_resp.rename(columns={"salary_final": "response"})
        else:
            if "response_final" not in rs.columns:
                print(f"  [skip {cue}] no response_final col")
                continue
            rs_resp = rs[keys + ["response_final"]].copy()
            rs_resp["response"] = (rs_resp["response_final"].astype(str).str.lower()
                                   .str.strip() == "yes").astype(float)
            rs_resp = rs_resp[keys + ["response"]]

        # race_pred slim
        rp_slim = rp[keys + ["race_pred"]].drop_duplicates(subset=keys)

        keep = list(keys)
        if has_fk:
            keep.append("flesch_kincaid_grade")
        for c in ["race", "gender"]:
            if c not in keep and c in prompts.columns:
                keep.append(c)
        # If the cache has prompt text we'd need it; otherwise we rely on cache FK
        if not has_fk and "prompt" in prompts.columns:
            keep.append("prompt")
        prompts_slim = prompts[keep].drop_duplicates(subset=keys)

        # Merge: prompts × race_pred × response (filters to rows we actually need)
        m = prompts_slim.merge(rp_slim, on=keys, how="inner")
        m = m.merge(rs_resp, on=keys, how="inner")
        m["id_cue"] = cue
        m = m.dropna(subset=["response"])

        if not has_fk:
            # Fallback: compute FK on the merged subset
            m["flesch_kincaid_grade"] = m["prompt"].apply(fk_grade)
            if "prompt" in m.columns:
                m = m.drop(columns=["prompt"])
        # Make sure prompt_id is consistent type
        m["prompt_id"] = m["prompt_id"].astype("int64")
        # Keep stable subset of columns
        out_cols = ["prompt_id", "id_cue", "race", "gender",
                    "race_pred", "flesch_kincaid_grade", "response"]
        for c in out_cols:
            if c not in m.columns:
                m[c] = np.nan
        pieces.append(m[out_cols])
        print(f"  [+ {cue}] n={len(m):,}")

    if not pieces:
        return None
    master = pd.concat(pieces, ignore_index=True)
    master["race"] = master["race"].apply(normalize_race)
    master = master[master["race"].notna()].copy()
    master = master[master["race_pred"].isin(VALID_PRED)].copy()
    print(f"  master rows after race-norm: {len(master):,}")
    return master


# ---------------------------------------------------------------- regression
def add_dummies(df):
    df = df.copy()
    df["race_pred_Black"]   = (df["race_pred"] == "Black").astype(float)
    df["race_pred_Unknown"] = (df["race_pred"] == "Unknown").astype(float)
    df["race_Black"]        = (df["race"] == "Black").astype(float)
    df["race_None"]         = (df["race"] == "None").astype(float)
    return df


def demean(df, cols):
    out = df.copy()
    g = df.groupby("prompt_id", sort=False)
    for c in cols:
        out[c] = df[c] - g[c].transform("mean")
    return out


def fit_spec(df_d, formula, label, out_dir, short):
    print(f"  fitting {short}: {formula} (n={len(df_d):,})")
    res = smf.ols(formula, data=df_d).fit()
    coefs = pd.DataFrame({
        "variable": res.params.index,
        "coef":     res.params.values,
        "std_err":  res.bse.values,
        "t":        res.tvalues.values,
        "p":        res.pvalues.values,
    })
    coefs.to_csv(os.path.join(out_dir, f"coef_{short}.csv"), index=False)
    return {
        "spec": label, "n": int(res.nobs), "r2_uncentered": res.rsquared,
        "aic": res.aic, "bic": res.bic,
        "params": res.params.to_dict(), "ses": res.bse.to_dict(),
    }


def run_paper_specs(master, out_dir, label):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n========== {label}  n={len(master):,} ==========")
    df = add_dummies(master)
    base = ["race_pred_Black", "race_pred_Unknown"]
    race_d = ["race_Black", "race_None"]
    fk = "flesch_kincaid_grade"

    specs_def = [
        ("(1) Inferred Race Only",
         f"response ~ {' + '.join(base)} - 1",
         ["response"] + base, "spec1"),
        ("(2) + Actual Race",
         f"response ~ {' + '.join(base + race_d)} - 1",
         ["response"] + base + race_d, "spec2"),
        ("(3) + Actual Race + FK",
         f"response ~ {' + '.join(base + race_d + [fk])} - 1",
         ["response"] + base + race_d + [fk], "spec3"),
    ]

    rows = []
    for label_, formula, dcols, short in specs_def:
        # Skip variables that are constant 0 (would be perfectly collinear / dropped)
        if any(c in dcols and df[c].nunique() <= 1 for c in base + race_d):
            # we still try; statsmodels will drop rank-deficient columns
            pass
        df_d = demean(df, dcols)
        try:
            rows.append(fit_spec(df_d, formula, label_, out_dir, short))
        except Exception as e:
            print(f"    ERROR: {e}")
            rows.append({"spec": label_, "n": 0, "r2_uncentered": np.nan,
                         "aic": np.nan, "bic": np.nan, "params": {}, "ses": {}})

    keys = ["race_pred_Black", "race_pred_Unknown",
            "race_Black", "race_None",
            "flesch_kincaid_grade"]
    side_rows = []
    for k in keys:
        rec = {"variable": k}
        for i, r in enumerate(rows):
            rec[f"({i+1}) coef"] = r["params"].get(k, np.nan)
            rec[f"({i+1}) se"]   = r["ses"].get(k, np.nan)
        side_rows.append(rec)
    side = pd.DataFrame(side_rows)
    side.to_csv(os.path.join(out_dir, "side_by_side.csv"), index=False)

    summ = pd.DataFrame([{"spec": r["spec"], "n": r["n"],
                           "r2_uncentered": r["r2_uncentered"],
                           "aic": r["aic"], "bic": r["bic"]} for r in rows])
    summ.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    print(side.to_string(index=False))
    print(summ.to_string(index=False))


def run_for_model(model: str, tasks=TASKS, save_master=True):
    for task in tasks:
        master = build_master(model, task)
        if master is None or len(master) == 0:
            print(f"  no master for {model} {task}, skipping")
            continue
        out_dir = os.path.join(OUT_ROOT, model, task)
        os.makedirs(out_dir, exist_ok=True)
        if save_master:
            master.to_parquet(os.path.join(out_dir, "master.parquet"), index=False)
        run_paper_specs(master, out_dir, f"{task} | {model}")
        del master
        gc.collect()


def main():
    models = sys.argv[1:] if len(sys.argv) > 1 else ["olmo2"]
    for m in models:
        run_for_model(m)


if __name__ == "__main__":
    main()
