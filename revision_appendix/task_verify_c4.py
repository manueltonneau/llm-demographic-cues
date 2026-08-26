#!/usr/bin/env python3
"""
Verify cross-model mechanism numbers (Appendix C.4).

Confirms that the C.4 mechanism claims hold. Two checks, both re-extracted from the *source CSVs* (not the hand-synced .tex), then
compared against what the appendix currently prints:

  Check 1 -- regression significance counts (paper Sec. 4 / Tables reg_*):
    In how many of the 9 (model x task) cells is (a) inferred race
    [race_pred_Black] significant, and (b) Flesch-Kincaid grade significant,
    in the full spec (3) with prompt FE? The paper claims
    7/9 (inferred race) and 9/9 (readability).
      Sources: LLaMA-3.1 -> results_paper/{medical,legal,salary}/
                             side_by_side_coefficients.csv (col "(3)")
               GPT-5.2/OLMo-2 -> results_all_models/{gpt52,olmo2}/
                             {task}/side_by_side.csv (spec-3 only)

  Check 2 -- Black-recall-by-cue table (Table black_recall_by_model):
    Confirm every printed cell (Explicit 99.4 / AAVE 14.5 / ...) equals the
    simple mean of the 3 per-task recalls x 100 from
    results_recall/black_recall_wide.csv.

Significance: t = coef/se, two-sided normal; *** p<.001, ** p<.01, * p<.05.

Outputs (written next to this script):
    task_verify_c4_regression.csv   - per-cell coef/t/sig for race & FK
    task_verify_c4_recall.csv       - per (model,cue) CSV-mean vs printed value
    task_verify_c4_summary.txt      - headline confirmation
"""
import os
import pandas as pd
from scipy.stats import norm

HERE     = os.path.dirname(os.path.abspath(__file__))              # revision_appendix
REPL_DIR = os.path.dirname(HERE)                                   # the repo root

MODELS = ["LLaMA-3.1", "OLMo-2", "GPT-5.2"]
TASKS  = ["medical", "legal", "salary"]
# task-name spelling differs between the LLaMA and all-models trees
TASK_ALL = {"medical": "medical_advice", "legal": "legal_advice", "salary": "salary_rec"}

REG_SRC = {
    "LLaMA-3.1": lambda t: f"{REPL_DIR}/results_paper/{t}/side_by_side_coefficients.csv",
    "GPT-5.2":   lambda t: f"{REPL_DIR}/results_all_models/gpt52/{TASK_ALL[t]}/side_by_side.csv",
    "OLMo-2":    lambda t: f"{REPL_DIR}/results_all_models/olmo2/{TASK_ALL[t]}/side_by_side.csv",
}

RACE_VAR = "race_pred_Black"
FK_VAR   = "flesch_kincaid_grade"


def star(t):
    p = 2 * (1 - norm.cdf(abs(t)))
    s = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    return s, p


def spec3_cols(df):
    ccol = [c for c in df.columns if "(3) coef" in c][0]
    scol = [c for c in df.columns if "(3) se" in c][0]
    return ccol, scol


# ---------- Check 1: regression significance ----------
rows = []
race_sig = fk_sig = 0
for model in MODELS:
    for task in TASKS:
        df = pd.read_csv(REG_SRC[model](task))
        ccol, scol = spec3_cols(df)
        d = df.set_index("variable")
        rec = {"model": model, "task": task}
        for var, tag in [(RACE_VAR, "race"), (FK_VAR, "fk")]:
            c = float(d.loc[var, ccol]); s = float(d.loc[var, scol])
            t = c / s; st, p = star(t)
            rec[f"{tag}_coef"] = c; rec[f"{tag}_t"] = t; rec[f"{tag}_sig"] = st
        if rec["race_sig"] != "ns": race_sig += 1
        if rec["fk_sig"] != "ns":   fk_sig += 1
        rows.append(rec)
reg = pd.DataFrame(rows)
reg.to_csv(f"{HERE}/task_verify_c4_regression.csv", index=False)

# ---------- Check 2: Black-recall-by-cue ----------
rec_df = pd.read_csv(f"{REPL_DIR}/results_recall/black_recall_wide.csv")
CUES = {  # csv column -> (source label, printed value per model in black_recall_by_model.tex)
    "explicit":                   ("Explicit",       {"llama3.1": 99.4, "olmo2": 65.1, "gpt52": 99.3}),
    "dialect":                    ("AAVE",           {"llama3.1": 14.5, "olmo2":  0.0, "gpt52":  9.7}),
    "name_specific_rosenman":     ("Rosenman",       {"llama3.1": 11.2, "olmo2":  0.0, "gpt52": 35.6}),
    "name_specific_hayes_elder":  ("Elder--Hayes",   {"llama3.1":  4.7, "olmo2":  0.0, "gpt52":  8.6}),
    "name_specific_tzioumis":     ("Tzioumis",       {"llama3.1":  4.7, "olmo2":  0.0, "gpt52": 11.9}),
    "convo_prefix":               ("CAD",            {"llama3.1": 0.13, "olmo2":  0.0, "gpt52": 0.20}),
    "convo_prefix_prism":         ("PRISM",          {"llama3.1":  1.7, "olmo2":  0.0, "gpt52": 0.23}),
}
MODEL_KEY = {"LLaMA-3.1": "llama3.1", "OLMo-2": "olmo2", "GPT-5.2": "gpt52"}
rrows = []
recall_mismatch = 0
for model in MODELS:
    sub = rec_df[rec_df.model == MODEL_KEY[model]]
    for col, (label, printed) in CUES.items():
        mean = sub[col].mean() * 100
        val = round(mean, 2) if mean < 1 else round(mean, 1)
        want = printed[MODEL_KEY[model]]
        ok = abs(val - want) < 0.06
        if not ok:
            recall_mismatch += 1
        rrows.append({"model": model, "cue": label, "csv_mean_pct": round(mean, 3),
                      "rounded": val, "printed_tex": want, "match": ok})
rec_out = pd.DataFrame(rrows)
rec_out.to_csv(f"{HERE}/task_verify_c4_recall.csv", index=False)

# ---------- summary ----------
lines = []
lines.append("T2. VERIFY CROSS-MODEL MECHANISM NUMBERS (Appendix C.4)")
lines.append("=" * 60)
lines.append("")
lines.append("CHECK 1 -- regression significance, full spec (3), prompt FE.")
lines.append(f"  Inferred race (race_pred_Black) significant: {race_sig}/9  (claim: 7/9)")
lines.append(f"  Flesch-Kincaid grade significant:            {fk_sig}/9  (claim: 9/9)")
notsig = reg[reg.race_sig == "ns"][["model", "task"]].values.tolist()
lines.append(f"  Inferred race NOT significant in: {', '.join(f'{m}/{t}' for m, t in notsig)}")
lines.append("")
for _, r in reg.iterrows():
    lines.append(f"    {r.model:9} {r.task:8} | inferred-race {r.race_coef:>10.4f} "
                 f"(t={r.race_t:6.1f}) {r.race_sig:3} | FK {r.fk_coef:>10.4f} "
                 f"(t={r.fk_t:7.1f}) {r.fk_sig:3}")
lines.append("")
lines.append("CHECK 2 -- Black-recall-by-cue (Table black_recall_by_model).")
lines.append(f"  Cells compared: {len(rec_out)} ; mismatches vs printed .tex: {recall_mismatch}")
lines.append("  (each printed value = simple mean of the 3 per-task recalls x 100)")
lines.append("")
lines.append("--- VERDICT ---")
ok1 = (race_sig == 7 and fk_sig == 9)
ok2 = (recall_mismatch == 0)
lines.append(f"  7/9 & 9/9 claim: {'CONFIRMED' if ok1 else 'MISMATCH -- see above'}")
lines.append(f"  Recall table matches source CSV & PDF: {'CONFIRMED' if ok2 else 'MISMATCH'}")

summary = "\n".join(lines)
with open(f"{HERE}/task_verify_c4_summary.txt", "w") as fh:
    fh.write(summary + "\n")
print(summary)
