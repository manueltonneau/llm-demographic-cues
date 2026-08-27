"""Regenerate the three LLaMA-3.1 regression tables from results/regressions_llama/.

The paper's tables/reg_{medical,legal,salary}.tex used to be hand-synced from the
result CSVs, which is how they drifted out of step with the pipeline. This script
rewrites the numbers in place, leaving every label, caption and \\label untouched,
so the tables can be regenerated instead of retyped.

    python make_paper_tables.py /path/to/cues_emnlp_draft/tables

Significance stars are recomputed from coef/SE (two-sided normal):
*** p<.001, ** p<.01, * p<.05.
"""
import os
import re
import sys

import pandas as pd
from scipy.stats import norm


import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from cues.paths import REGRESSIONS_LLAMA as SRC, require_dir, require_file

TASKS = ["medical", "legal", "salary"]
# (csv variable, the label line that introduces its block, how many specs it spans)
BLOCKS = [
    ("race_pred_Black",       r"Inferred Black (vs.\ White)",   (1, 2, 3)),
    ("race_pred_Unknown",     r"Inferred Unknown (vs.\ White)", (1, 2, 3)),
    ("race_Black",            r"$\quad$ Black",                 (2, 3)),
    ("race_None",             r"$\quad$ None",                  (2, 3)),
    ("flesch_kincaid_grade",  r"Readability (Flesch--Kincaid Grade)", (3,)),
]


def _tail(line):
    """The trailing row terminator and whitespace of an existing line, so
    rewriting the numbers leaves the file's formatting byte-identical."""
    m = re.match(r"^.*?(\s*\\\\)?(\s*)$", line)
    return (m.group(1) or "") + (m.group(2) or "")


def stars(coef, se):
    if pd.isna(coef) or pd.isna(se) or se == 0:
        return ""
    p = 2 * (1 - norm.cdf(abs(coef / se)))
    return "$^{***}$" if p < 0.001 else "$^{**}$" if p < 0.01 else "$^{*}$" if p < 0.05 else ""


def thousands(s):
    """LaTeX-safe thousands separator, matching the existing tables."""
    return re.sub(r"(?<=\d),(?=\d)", "{,}", f"{s}")


def fmt_coef(v, task):
    if pd.isna(v):
        return ""
    return thousands(f"{v:,.2f}") if task == "salary" else f"{v:.4f}"


def fmt_se(v, task):
    if pd.isna(v):
        return ""
    if task == "salary":
        return thousands(f"{v:,.2f}")
    s = f"{v:.4f}"
    if float(s) == 0:                      # e.g. the FK standard errors
        s = f"{v:.5f}"
    return s


def sig4(x):
    """Round to four significant digits, the convention used for AIC/BIC."""
    if pd.isna(x):
        return ""
    from math import floor, log10
    if x == 0:
        return "0"
    d = 4 - int(floor(log10(abs(x)))) - 1
    return f"{round(x, d):,.0f}"


def sync(task, tables_dir):
    coef = pd.read_csv(require_file(os.path.join(SRC, task, "side_by_side_coefficients.csv"),
                                    f"{task} coefficients")).set_index("variable")
    summ = pd.read_csv(require_file(os.path.join(SRC, task, "summary.csv"),
                                    f"{task} summary"))
    path = require_file(os.path.join(tables_dir, f"reg_{task}.tex"), f"reg_{task}.tex")
    lines = open(path, encoding="utf-8").read().split("\n")

    def get(var, spec, what):
        if var not in coef.index:
            return float("nan")
        return coef.loc[var, f"({spec}) {what}"]

    out, i, changed = [], 0, 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        for var, label, specs in BLOCKS:
            if label not in line:
                continue
            cs = [get(var, s, "coef") for s in specs]
            ss = [get(var, s, "se") for s in specs]
            cells = [f"{fmt_coef(c, task)}{stars(c, s)}" for c, s in zip(cs, ss)]
            pad = ["" for _ in range(3 - len(specs))]
            n_lines = 4 if len(specs) == 3 else 2
            originals = lines[i + 1:i + 1 + n_lines]
            if len(specs) == 3:            # one coef per line, then a shared SE line
                bodies = [f"& {c}" for c in cells]
                bodies.append("& " + " & ".join(f"({fmt_se(s, task)})" for s in ss))
            else:                          # coefs and SEs each on one line
                bodies = ["& " + " & ".join(pad + cells),
                          "& " + " & ".join(pad + [f"({fmt_se(s, task)})" for s in ss])]
            for body, original in zip(bodies, originals):
                out.append(body + _tail(original))
            i += n_lines
            changed += 1
            break
        i += 1

    text = "\n".join(out)
    n = f"{int(summ.n.iloc[0]):,}"
    text = re.sub(r"(Observations)( & [^\\]*)\\\\", rf"\1 & {n} & {n} & {n} \\\\", text)
    r2 = " & ".join(f"{v:.3f}" for v in summ.r2_uncentered)
    text = re.sub(r"(\$R\^2\$ \(uncentered\))( & [^\\]*)\\\\", rf"\1 & {r2} \\\\", text)
    for stat in ("AIC", "BIC"):
        vals = " & ".join(sig4(v) for v in summ[stat.lower()])
        text = re.sub(rf"^({stat})( & [^\\]*)\\\\", rf"\1 & {vals} \\\\", text, flags=re.M)
    open(path, "w", encoding="utf-8").write(text)
    return changed, path



# ---------------------------------------------------------------- summary tables
# reg_summary.tex and spec_ladder.tex restate the same LLaMA-3.1 coefficients at
# lower precision. Only the LLaMA rows come from results/regressions_llama/; the OLMo-2 and
# GPT-5.2 rows come from results/regressions_all_models/ and are left alone.
SUMMARY_ROWS = {"medical": "Healthcare", "legal": "Legal", "salary": "Salary"}


def _short(v, task, digits=3):
    if pd.isna(v):
        return ""
    return f"{v:,.0f}" if task == "salary" else f"{v:.{digits}f}"


def _cell(coef, se, task):
    return f"${_short(coef, task)}{'^{***}' if stars(coef, se) == '$^{***}$' else ''}$"


def sync_summaries(tables_dir):
    vals = {}
    for task in TASKS:
        c = pd.read_csv(os.path.join(SRC, task, "side_by_side_coefficients.csv")).set_index("variable")
        vals[task] = c

    # --- reg_summary.tex: spec (3) inferred / cued / FK, LLaMA rows only
    path = os.path.join(tables_dir, "reg_summary.tex")
    if os.path.exists(path):
        text = open(path, encoding="utf-8").read()
        for task, label in SUMMARY_ROWS.items():
            c = vals[task]
            # the Cued column is printed without significance stars throughout
            # this table (all three models), so it is rendered bare here too.
            cells = [_cell(c.loc["race_pred_Black", "(3) coef"], c.loc["race_pred_Black", "(3) se"], task),
                     f'${_short(c.loc["race_Black", "(3) coef"], task)}$',
                     _cell(c.loc["flesch_kincaid_grade", "(3) coef"], c.loc["flesch_kincaid_grade", "(3) se"], task)]
            pat = re.compile(rf"(^\s*(?:LLaMA-3\.1 &|&) *{re.escape(label)}(?:\$\^\{{\\\$\}}\$)? *&)[^\\]*(\\\\)",
                             re.M)
            new = rf"\1 {cells[0]} & {cells[1]} & {cells[2]} \2"
            text, n = pat.subn(new, text, count=1)
        open(path, "w", encoding="utf-8").write(text)

    # --- spec_ladder.tex: inferred-Black across specs 1-3, LLaMA rows only
    path = os.path.join(tables_dir, "spec_ladder.tex")
    if os.path.exists(path):
        lines = open(path, encoding="utf-8").read().split("\n")
        in_llama = False
        for i, line in enumerate(lines):
            if "textit{LLaMA-3.1}" in line:
                in_llama = True
                continue
            if in_llama and ("textit{" in line or "addlinespace" in line):
                in_llama = False
            if not in_llama:
                continue
            for task, label in (("medical", "Medical"), ("legal", "Legal"), ("salary", "Salary")):
                if not line.strip().startswith(label):
                    continue
                c = vals[task]
                cells = [_cell(c.loc["race_pred_Black", f"({s}) coef"],
                               c.loc["race_pred_Black", f"({s}) se"], task) for s in (1, 2, 3)]
                head = line.split("&")[0]
                lines[i] = head + "& " + " & ".join(cells) + " \\\\"
        open(path, "w", encoding="utf-8").write("\n".join(lines))


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python make_paper_tables.py /path/to/cues_emnlp_draft/tables")
    tables_dir = require_dir(sys.argv[1], "the paper's tables/ directory")
    for task in TASKS:
        n, path = sync(task, tables_dir)
        print(f"  {os.path.basename(path)}: {n} coefficient blocks rewritten")
    sync_summaries(tables_dir)
    print("  reg_summary.tex / spec_ladder.tex: LLaMA-3.1 rows rewritten")


if __name__ == "__main__":
    main()
