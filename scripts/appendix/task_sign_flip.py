#!/usr/bin/env python3
"""
Sign-flip / conclusion-instability analysis (paper §4.1 and appendix).

Are single-cue conclusions about intergroup disparity stable across cues?
This script quantifies that instability directly from the per-cue Black/White and
Female/Male outcome ratios that feed Figure 3 (race) and Figure 11 (gender),
i.e. the "*_likelihood_ratios_ALL_MODELS.csv" / "salary_lr_results.csv" files
under data/plot_data/. Each row there is one (model, comparison, cue) with:
    lr  = pooled disparity ratio (1 = parity; >1 favors Black/Female)
    lo,hi = 95% bootstrap CI (same seed-pooled bootstrap as the figures).

A "cell" is a (task x model x comparison) combination -> 9 race cells and
9 gender cells. A "cue" is one operationalisation (method): the 4 name lists,
2 dialog-history sources, explicit descriptor, and (race only) dialect.

Metrics
-------
A. Sign-flip cells: a cell flips if >=1 pair of cues has point ratios on
   opposite sides of 1 AND non-overlapping 95% CIs (the higher cue's lo >
   the lower cue's hi). We report X/9 for race and X/9 for gender and list
   the offending cue pairs.

B. Single-cue error rate: for each cell we build a multi-cue REFERENCE
   conclusion two ways (robustness check):
     (1) majority  - modal 3-way class across the cue's own CIs
                     (ties broken toward parity);
     (2) pooled    - inverse-variance (precision) weighted pooled ratio in
                     log space, classified by its own 95% CI.
   Each single cue is classified {favors Black/Female, parity (CI covers 1),
   favors White/Male}. A cue is an "error" if its class != the reference
   class. We report the error rate over all (cell,cue) combinations, race and
   gender separately, for both references. We also report a stricter
   "hard sign error" rate (cue significant in the direction opposite to a
   significant reference).

C. Range: per cell, min-max of the cue-level point ratios (the range we
   report alongside the point estimates).

Salary is broken out separately in every metric.

Outputs (written next to this script, under scripts/appendix/):
    task_sign_flip_per_cue.csv     - (cell,cue) rows with classification
    task_sign_flip_per_cell.csv    - one row per cell (flip?, range, #discordant)
    task_sign_flip_table.tex       - compact LaTeX table for the appendix
    task_sign_flip_summary.txt     - headline text summary
"""

import os
import itertools
import numpy as np
import pandas as pd

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from cues.paths import APPENDIX as HERE, APPENDIX as RA, REPO_ROOT as REPL_DIR, DATA_DIR, PROMPTS, PROMPTS as PROMPT_DIR, RACE_INFERENCE as RECALL_DIR, RESPONSES as DATA, RACE_PRED, PLOT_DATA as PLOT, GPT_RUN, FIGURES, RACE_INFERENCE, REGRESSIONS_LLAMA, REGRESSIONS_ALL_MODELS, DIALECT_ALL_MODELS, require_dir, require_file, require_any, fail

import sys
require_dir(PLOT, "outcome-ratio tables (data/plot_data)")

TASK_FILES = {
    "medical": "medical_advice_likelihood_ratios_ALL_MODELS.csv",
    "legal":   "legal_advice_likelihood_ratios_ALL_MODELS.csv",
    "salary":  "salary_lr_results.csv",
}
TASK_ORDER = ["medical", "legal", "salary"]
MODEL_ORDER = ["gpt52", "llama3.1", "olmo2"]
MODEL_LABEL = {"gpt52": "GPT-5.2", "llama3.1": "LLaMA-3.1", "olmo2": "OLMo-2"}
TASK_LABEL = {"medical": "Medical", "legal": "Legal", "salary": "Salary"}

CUE_LABEL = {
    "name_specific_an": "Name (An)",
    "name_specific_hayes_elder": "Name (Hayes-Elder)",
    "name_specific_tzioumis": "Name (Tzioumis)",
    "name_specific_rosenman": "Name (Rosenman)",
    "convo_prefix": "Dialog (CAD)",
    "convo_prefix_prism": "Dialog (PRISM)",
    "explicit": "Explicit",
    "dialect": "Dialect",
}

Z = 1.959963984540054  # 95%


# ------------------------------------------------------------------ helpers
def classify(lr, lo, hi, favor_high, favor_low):
    """3-way class of a single ratio+CI. favor_high/low are labels for the
    two directions (e.g. 'Black' / 'White')."""
    if lo > 1.0:
        return favor_high        # CI entirely above 1
    if hi < 1.0:
        return favor_low         # CI entirely below 1
    return "Parity"             # CI covers 1


def direction(lr):
    return "up" if lr > 1.0 else ("down" if lr < 1.0 else "eq")


def pooled_ratio(sub):
    """Precision-weighted pooled ratio in log space, with its own 95% CI.
    var_i inferred from the reported CI: se_i = (ln hi - ln lo)/(2Z)."""
    lr = sub["lr"].to_numpy(float)
    lo = sub["lo"].to_numpy(float)
    hi = sub["hi"].to_numpy(float)
    log_lr = np.log(lr)
    se = (np.log(hi) - np.log(lo)) / (2 * Z)
    se = np.where(se <= 0, np.nan, se)
    w = 1.0 / (se ** 2)
    good = np.isfinite(w) & np.isfinite(log_lr)
    w, log_lr = w[good], log_lr[good]
    if w.sum() == 0:
        return np.nan, np.nan, np.nan
    mu = np.sum(w * log_lr) / np.sum(w)
    se_mu = np.sqrt(1.0 / np.sum(w))
    return np.exp(mu), np.exp(mu - Z * se_mu), np.exp(mu + Z * se_mu)


def majority_class(classes, favor_high, favor_low):
    """Modal 3-way class; ties broken toward Parity, then toward the
    numerically larger group. Documented, deterministic."""
    order = [favor_high, favor_low, "Parity"]
    counts = {c: classes.count(c) for c in order}
    mx = max(counts.values())
    winners = [c for c in order if counts[c] == mx]
    if len(winners) == 1:
        return winners[0]
    return "Parity" if "Parity" in winners else winners[0]


# ------------------------------------------------------------------ load
frames = []
for task, fname in TASK_FILES.items():
    d = pd.read_csv(os.path.join(PLOT, fname))
    d["task"] = task
    frames.append(d)
df = pd.concat(frames, ignore_index=True)

df["demo"] = df["comparison"].map(
    {"Black / White": "race", "Female / Male": "gender"}
)
FAVOR = {"race": ("Black", "White"), "gender": ("Female", "Male")}

# ------------------------------------------------------------------ per-cue + per-cell
per_cue_rows = []
per_cell_rows = []

for demo in ["race", "gender"]:
    fav_hi, fav_lo = FAVOR[demo]
    for task in TASK_ORDER:
        for model in MODEL_ORDER:
            sub = df[(df.demo == demo) & (df.task == task) & (df.model == model)].copy()
            if sub.empty:
                continue
            sub = sub.sort_values("method")

            # per-cue classification
            sub["cue_class"] = [
                classify(r.lr, r.lo, r.hi, fav_hi, fav_lo)
                for r in sub.itertuples()
            ]

            # ---- references
            classes = sub["cue_class"].tolist()
            ref_maj = majority_class(classes, fav_hi, fav_lo)
            p_lr, p_lo, p_hi = pooled_ratio(sub)
            ref_pool = classify(p_lr, p_lo, p_hi, fav_hi, fav_lo)

            # ---- Metric A: sign-flip via any opposite-sign, non-overlapping pair
            flip_pairs = []
            recs = list(sub.itertuples())
            for a, b in itertools.combinations(recs, 2):
                if (a.lr - 1.0) * (b.lr - 1.0) < 0:            # opposite sides of 1
                    hi_rec, lo_rec = (a, b) if a.lr > b.lr else (b, a)
                    if hi_rec.lo > lo_rec.hi:                  # non-overlapping CIs
                        flip_pairs.append(
                            f"{CUE_LABEL.get(hi_rec.method, hi_rec.method)}"
                            f"({hi_rec.lr:.3f}) vs "
                            f"{CUE_LABEL.get(lo_rec.method, lo_rec.method)}"
                            f"({lo_rec.lr:.3f})"
                        )
            is_flip = len(flip_pairs) > 0

            # ---- Metric B: disagreements with each reference
            n_disc_maj = int((sub["cue_class"] != ref_maj).sum())
            n_disc_pool = int((sub["cue_class"] != ref_pool).sum())

            # hard sign errors: cue significantly opposite a significant ref
            def hard_errors(ref_class):
                if ref_class not in (fav_hi, fav_lo):
                    return 0
                opp = fav_lo if ref_class == fav_hi else fav_hi
                return int((sub["cue_class"] == opp).sum())
            n_hard_maj = hard_errors(ref_maj)
            n_hard_pool = hard_errors(ref_pool)

            # ---- Metric C: range of point ratios
            rmin, rmax = sub["lr"].min(), sub["lr"].max()

            for r in sub.itertuples():
                per_cue_rows.append(dict(
                    demo=demo, task=task, model=model,
                    cue=r.method, cue_label=CUE_LABEL.get(r.method, r.method),
                    lr=r.lr, lo=r.lo, hi=r.hi, cue_class=r.cue_class,
                    ref_majority=ref_maj, ref_pooled=ref_pool,
                    err_majority=int(r.cue_class != ref_maj),
                    err_pooled=int(r.cue_class != ref_pool),
                ))

            per_cell_rows.append(dict(
                demo=demo, task=task, model=model,
                n_cues=len(sub),
                flip=is_flip, n_flip_pairs=len(flip_pairs),
                flip_pairs="; ".join(flip_pairs),
                ratio_min=rmin, ratio_max=rmax, ratio_range=rmax - rmin,
                ref_majority=ref_maj, ref_pooled=ref_pool,
                pooled_lr=p_lr, pooled_lo=p_lo, pooled_hi=p_hi,
                n_discordant_majority=n_disc_maj,
                n_discordant_pooled=n_disc_pool,
                n_hard_error_majority=n_hard_maj,
                n_hard_error_pooled=n_hard_pool,
            ))

per_cue = pd.DataFrame(per_cue_rows)
per_cell = pd.DataFrame(per_cell_rows)

per_cue.to_csv(os.path.join(HERE, "task_sign_flip_per_cue.csv"), index=False)
per_cell.to_csv(os.path.join(HERE, "task_sign_flip_per_cell.csv"), index=False)


# ------------------------------------------------------------------ headline numbers
def summarize(demo):
    cell = per_cell[per_cell.demo == demo]
    cue = per_cue[per_cue.demo == demo]
    out = {}
    out["n_cells"] = len(cell)
    out["n_flip"] = int(cell["flip"].sum())
    out["flip_cells"] = [
        (r.task, r.model, r.flip_pairs)
        for r in cell.itertuples() if r.flip
    ]
    out["n_cue_combos"] = len(cue)
    out["err_maj"] = int(cue["err_majority"].sum())
    out["err_pool"] = int(cue["err_pooled"].sum())
    out["hard_maj"] = int(cell["n_hard_error_majority"].sum())
    out["hard_pool"] = int(cell["n_hard_error_pooled"].sum())
    return out


race = summarize("race")
gender = summarize("gender")

# salary-only breakout
def summarize_salary(demo):
    cell = per_cell[(per_cell.demo == demo) & (per_cell.task == "salary")]
    cue = per_cue[(per_cue.demo == demo) & (per_cue.task == "salary")]
    return dict(
        n_cells=len(cell), n_flip=int(cell["flip"].sum()),
        n_cue_combos=len(cue),
        err_maj=int(cue["err_majority"].sum()),
        err_pool=int(cue["err_pooled"].sum()),
    )

race_sal = summarize_salary("race")
gender_sal = summarize_salary("gender")


# ------------------------------------------------------------------ LaTeX table
def build_latex():
    lines = []
    lines.append(r"% Auto-generated by scripts/appendix/task_sign_flip.py")
    lines.append(r"% Requires \usepackage{pifont} and \usepackage{booktabs} in the preamble.")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{llccc}")
    lines.append(r"\toprule")
    lines.append(r"Task & Model & Flip? & Ratio range & \# disc.\ cues \\")
    lines.append(r"\midrule")
    for demo, header in [("race", r"\multicolumn{5}{l}{\emph{Race (Black/White)}}\\"),
                         ("gender", r"\multicolumn{5}{l}{\emph{Gender (Female/Male)}}\\")]:
        lines.append(r"\midrule")
        lines.append(header)
        cell = per_cell[per_cell.demo == demo]
        for task in TASK_ORDER:
            for model in MODEL_ORDER:
                r = cell[(cell.task == task) & (cell.model == model)]
                if r.empty:
                    continue
                r = r.iloc[0]
                flip = r"\ding{51}" if r.flip else r"\ding{55}"
                rng = f"[{r.ratio_min:.3f}, {r.ratio_max:.3f}]"
                disc = f"{int(r.n_discordant_majority)}/{int(r.n_cues)}"
                lines.append(
                    f"{TASK_LABEL[task]} & {MODEL_LABEL[model]} & {flip} & {rng} & {disc} \\\\"
                )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{\textbf{Conclusion instability across cues.} "
        r"For each task$\times$model cell we test whether single-cue "
        r"Black/White (Female/Male) disparity conclusions are stable across "
        r"the 7--8 cues plotted in Figures~\ref{fig:outcome_ratios} and "
        r"\ref{fig:outcome_ratios_gender}. \emph{Flip?}: \ding{51} if at least "
        r"one cue pair has oppositely-signed ratios with non-overlapping 95\% "
        r"bootstrap CIs. \emph{Ratio range}: min--max cue-level ratio. "
        r"\emph{\# disc.\ cues}: single cues whose 3-way conclusion "
        r"(favors Black/Female / parity / favors White/Male) disagrees with "
        r"the cell's majority-cue reference, out of the cues available.}"
    )
    lines.append(r"\label{tab:sign_flip}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

with open(os.path.join(HERE, "task_sign_flip_table.tex"), "w") as fh:
    fh.write(build_latex() + "\n")


# ------------------------------------------------------------------ text summary
def pct(n, d):
    return f"{100.0 * n / d:.1f}\\%" if d else "n/a"

lines = []
lines.append("T1. SIGN-FLIP / CONCLUSION-INSTABILITY ANALYSIS")
lines.append("=" * 60)
lines.append("")
lines.append("Data: per-cue Black/White (race) and Female/Male (gender) outcome")
lines.append("ratios + 95% bootstrap CIs feeding Figures 3 and 11 "
             "(data/plot_data/*_likelihood_ratios / salary_lr_results).")
lines.append("Cells = task x model (3 x 3 = 9 per demographic). Cues = the 7-8")
lines.append("operationalisations per cell (4 name lists, 2 dialog sources,")
lines.append("explicit; race adds dialect).")
lines.append("")
for demo, s, sal, fav in [("RACE", race, race_sal, "Black"),
                          ("GENDER", gender, gender_sal, "Female")]:
    lines.append(f"--- {demo} ---")
    lines.append(f"Metric A (sign-flip cells): {s['n_flip']}/{s['n_cells']} "
                 f"cells contain >=1 oppositely-signed, non-overlapping-CI cue pair.")
    for task, model, pairs in s["flip_cells"]:
        lines.append(f"    * {TASK_LABEL[task]} / {MODEL_LABEL[model]}: {pairs}")
    lines.append(f"Metric B (single-cue error rate over {s['n_cue_combos']} "
                 f"(cell,cue) combos):")
    lines.append(f"    majority reference: {s['err_maj']}/{s['n_cue_combos']} "
                 f"= {pct(s['err_maj'], s['n_cue_combos'])} discordant "
                 f"(hard opposite-sign errors: {s['hard_maj']}).")
    lines.append(f"    pooled reference:   {s['err_pool']}/{s['n_cue_combos']} "
                 f"= {pct(s['err_pool'], s['n_cue_combos'])} discordant "
                 f"(hard opposite-sign errors: {s['hard_pool']}).")
    lines.append(f"  Salary only: {sal['n_flip']}/{sal['n_cells']} flip cells; "
                 f"single-cue error {sal['err_maj']}/{sal['n_cue_combos']} "
                 f"= {pct(sal['err_maj'], sal['n_cue_combos'])} (majority ref), "
                 f"{pct(sal['err_pool'], sal['n_cue_combos'])} (pooled ref).")
    lines.append("")

# ready-to-paste headline sentence
lines.append("--- HEADLINE (paste-ready) ---")
lines.append(
    f"Across the 9 race cells, {race['n_flip']} contain a genuine sign flip "
    f"(oppositely-signed cue ratios with non-overlapping 95% CIs), and "
    f"{pct(race['err_maj'], race['n_cue_combos'])} of single-cue conclusions "
    f"disagree with the majority-cue reference in the same cell "
    f"({pct(race['err_pool'], race['n_cue_combos'])} against a "
    f"precision-weighted pooled reference). For gender the corresponding "
    f"figures are {gender['n_flip']}/9 flip cells and "
    f"{pct(gender['err_maj'], gender['n_cue_combos'])} "
    f"({pct(gender['err_pool'], gender['n_cue_combos'])}) discordant "
    f"single-cue conclusions."
)

summary = "\n".join(lines)
with open(os.path.join(HERE, "task_sign_flip_summary.txt"), "w") as fh:
    fh.write(summary + "\n")

print(summary)
print()
print("Wrote: task_sign_flip_per_cue.csv, task_sign_flip_per_cell.csv,")
print("       task_sign_flip_table.tex, task_sign_flip_summary.txt")
