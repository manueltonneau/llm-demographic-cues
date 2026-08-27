#!/usr/bin/env python3
"""
Dialect-only (AAVE) regression table (appendix).

The analysis itself lives in `replicate_dialect_only_allmodels.py`, which writes
`results/dialect_all_models/dialect_allmodels_summary.csv`; it was omitted from the
main tables for space. This script formats it into a compact LaTeX table plus a
text summary with exact numbers.

Spec (per model x task, OLS over AAVE-only prompts; no prompt FE because dialect
has one seed, and cued race is dropped because every dialect row is cued-Black):
    (2)  y ~ inferred_race (Black/Unknown vs White ref) + flesch_kincaid_grade
We report the spec-(2) inferred-Black coefficient and the FK coefficient with
significance stars. Where a model predicts White/Unknown for essentially all
AAVE prompts, the Black coefficient is not estimable ("--").

Outputs (next to this script):
    task_dialect_only_table.tex      - drop-in LaTeX table
    task_dialect_only_summary.txt    - headline counts
"""
import os
import pandas as pd

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from cues.paths import APPENDIX as HERE, APPENDIX as RA, REPO_ROOT as REPL_DIR, DATA_DIR, PROMPTS, PROMPTS as PROMPT_DIR, RACE_INFERENCE as RECALL_DIR, RESPONSES as DATA, RACE_PRED, PLOT_DATA as PLOT, GPT_RUN, FIGURES, RACE_INFERENCE, REGRESSIONS_LLAMA, REGRESSIONS_ALL_MODELS, DIALECT_ALL_MODELS, require_dir, require_file, require_any, fail
SRC = os.path.join(DIALECT_ALL_MODELS, "dialect_allmodels_summary.csv")

MODEL_LBL = {"llama3.1": "LLaMA-3.1", "olmo2": "OLMo-2", "gpt52": "GPT-5.2"}
TASK_LBL = {"medical_advice": "Medical", "legal_advice": "Legal", "salary_rec": "Salary"}


def stars(p):
    return "" if pd.isna(p) else "$^{***}$" if p < 0.001 else "$^{**}$" if p < 0.01 else "$^{*}$" if p < 0.05 else ""


def fmt(coef, p):
    if pd.isna(coef):
        return "--"
    c = f"{coef:.4f}" if abs(coef) < 100 else f"{coef:,.0f}"
    return f"{c}{stars(p)}"


d = pd.read_csv(SRC)

rows = []
fk_sig = race_sig = race_est = both = 0
for _, r in d.iterrows():
    fk_ok = (not pd.isna(r.spec2_flesch_kincaid_grade_p)) and r.spec2_flesch_kincaid_grade_p < 0.05
    fk_sig += fk_ok
    est = bool(r.black_estimable)
    race_est += est
    r_ok = est and r.spec2_race_pred_Black_p < 0.05
    race_sig += r_ok
    both += (fk_ok and r_ok)
    rows.append((MODEL_LBL[r.model], TASK_LBL[r.task], int(r["n"]),
                 fmt(r.spec2_race_pred_Black_coef if est else float("nan"),
                     r.spec2_race_pred_Black_p if est else float("nan")),
                 fmt(r.spec2_flesch_kincaid_grade_coef, r.spec2_flesch_kincaid_grade_p)))

# ---- LaTeX table ----
tex = [
    r"\begin{table}[t]", r"\centering", r"\small",
    r"\begin{tabular}{llrrr}", r"\toprule",
    r"Model & Task & $N$ & Inferred Black & FK grade \\",
    r"\midrule",
]
for m, t, n, rc, fk in rows:
    tex.append(f"{m} & {t} & {n:,} & {rc} & {fk} \\\\")
tex += [
    r"\bottomrule",
    r"\end{tabular}",
    (r"\caption{Dialect-only (AAVE) regression: model response on inferred race "
     r"(Black vs.\ White ref.) and Flesch--Kincaid grade, per model and task "
     r"(OLS; no prompt FE, cued race dropped as constant). ``--'' = inferred Black "
     r"not estimable (model predicts White/Unknown for nearly all AAVE prompts). "
     r"$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.}"),
    r"\label{tab:dialect_only_regression}",
    r"\end{table}",
]
with open(f"{HERE}/task_dialect_only_table.tex", "w") as fh:
    fh.write("\n".join(tex) + "\n")

# ---- summary ----
both_cells = ", ".join(f"{MODEL_LBL[r.model]}/{TASK_LBL[r.task]}"
                       for _, r in d.iterrows()
                       if r.black_estimable and r.spec2_race_pred_Black_p < 0.05
                       and r.spec2_flesch_kincaid_grade_p < 0.05)
S = [
    "DIALECT-ONLY (AAVE) REGRESSION",
    "=" * 52,
    f"Flesch--Kincaid (linguistic-form channel) significant: {fk_sig}/9 cells.",
    f"Inferred race (demographic-signal channel) estimable: {race_est}/9;",
    f"  of which significant (net of FK): {race_sig}/9.",
    f"Both channels significant simultaneously ({both}/9): {both_cells}.",
    "Inferred Black not estimable in the 3 OLMo-2 cells: the model predicts",
    "  White/Unknown for essentially all AAVE prompts (itself evidence that",
    "  AAVE's demographic signal is weak and model-dependent).",
    "",
    "TAKEAWAY: the AAVE effect is not reducible to readability — FK matters in",
    "8/9 cells, and inferred race adds independent, significant signal wherever",
    "the model actually infers race from AAVE (LLaMA-3.1 medical & salary,",
    "GPT-5.2 salary). It is a mixture we decompose, not pure linguistic form.",
    "",
    "Source: results/dialect_all_models/dialect_allmodels_summary.csv",
    "        (replicate_dialect_only_allmodels.py). Table -> task_dialect_only_table.tex",
]
summary = "\n".join(S)
with open(f"{HERE}/task_dialect_only_summary.txt", "w") as fh:
    fh.write(summary + "\n")
print(summary)
