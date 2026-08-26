"""Task 3: Black/White salary ratios by job tier.
Tiers = quartiles of the no-cue baseline salary distribution over ~100 job
profiles (one tiering, defined on the across-model mean no-cue salary per
profile). B/W ratio per cue x tier x model, plotted in Figure-3 style.
"""
import os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE      = os.path.dirname(os.path.abspath(__file__))             # revision_appendix
RA        = HERE
REPL_DIR  = os.path.dirname(HERE)                                  # the repo root
REPO_ROOT = os.environ.get("CUES_ROOT", os.path.dirname(REPL_DIR))
DATA_DIR  = os.environ.get("CUES_DATA_DIR", os.path.join(REPO_ROOT, "data"))

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cues_io import require_dir, require_file, require_any
require_file(os.path.join(DATA_DIR, "prompts", "salary_rec",
                          "salary_rec_neutral_constrained.parquet"),
             "no-cue salary prompts")

# ---- load per-seed salary aggregates (llama, olmo) + gpt52 ----
agg = pd.read_parquet(os.path.join(RA, "per_seed_agg.parquet"))
agg = agg[agg.task == "salary_rec"]
gpt = pd.read_parquet(os.path.join(RA, "gpt52_salary_agg.parquet"))
sal = pd.concat([agg, gpt], ignore_index=True)
# pool across seeds: sum sum_val and n
sal = sal.groupby(["model", "method", "race", "prompt_id"], as_index=False)[["sum_val", "n"]].sum()

# ---- prompt_id -> job profile (mask location) ----
neu = pd.read_parquet(os.path.join(DATA_DIR, "prompts/salary_rec/salary_rec_neutral_constrained.parquet"),
                      columns=["prompt", "prompt_id"])
loc_re = re.compile(r" in [A-Z][A-Za-z .\-]+?, [A-Z]{2}\b")
neu["profile"] = neu["prompt"].map(lambda t: loc_re.sub(" in <LOC>", t, count=1))
prof_map = dict(zip(neu.prompt_id, neu.profile))
profiles = sorted(neu.profile.unique())
print(f"{len(profiles)} job profiles")

sal["profile"] = sal.prompt_id.map(prof_map)
sal = sal[sal.profile.notna()]

# ---- baseline salary per profile (no-cue), averaged across models ----
neu_rows = sal[sal.method == "neutral"].copy()
neu_rows["mean"] = neu_rows.sum_val / neu_rows.n
prof_base = neu_rows.groupby(["model", "profile"])["mean"].mean().groupby("profile").mean()
# quartile tiers
q = pd.qcut(prof_base, 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])
tier_of = dict(zip(prof_base.index, q))
sal["tier"] = sal.profile.map(tier_of)
print("\nbaseline salary range by tier:")
print(prof_base.groupby(q).agg(["min", "max", "size"]).to_string())

# ---- pooled mean helper ----
def pooled(df, model, method, race, tier):
    s = df[(df.model == model) & (df.method == method) & (df.race == race) & (df.tier == tier)]
    return s.sum_val.sum() / s.n.sum() if s.n.sum() > 0 else np.nan

MODELS = [("llama3.1", "LLaMA-3.1 8B"), ("olmo2", "OLMo2 7B"), ("gpt52", "GPT-5.2")]
TIERS = ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
# cue rows (label, method, white-reference method/race)
CUES = [
    ("Name (R)", "name_specific_rosenman", "white"),
    ("Name (EH)", "name_specific_hayes_elder", "white"),
    ("Name (T)", "name_specific_tzioumis", "white"),
    ("Dialog (CAD)", "convo_prefix", "white"),
    ("Dialog (PRISM)", "convo_prefix_prism", "white"),
    ("Explicit", "explicit", "white"),
    ("Dialect (AAVE)", "dialect", "neutral"),  # white ref = no-cue
]

rows = []
for model, _ in MODELS:
    for label, method, wref in CUES:
        for tier in TIERS:
            b = pooled(sal, model, method, "black", tier)
            w = pooled(sal, model, "neutral", "none", tier) if wref == "neutral" \
                else pooled(sal, model, method, "white", tier)
            rows.append(dict(model=model, cue=label, tier=tier,
                             ratio=(b / w if (w and not np.isnan(w)) else np.nan)))
res = pd.DataFrame(rows)
res.to_csv(os.path.join(RA, "task3_salary_tiers.csv"), index=False)
print("\nratios sample:\n", res.head(12).to_string(index=False))

# ---- plot ----
plt.rcParams.update({"font.family": "serif", "font.size": 10, "text.usetex": False})
tier_colors = ["#762a83", "#af8dc3", "#7fbf7b", "#1b7837"]  # purple->green seq
cue_labels = [c[0] for c in CUES]
y_pos = np.arange(len(cue_labels))[::-1]

fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.6), sharey=True)
for ax, (model, mlabel) in zip(axes, MODELS):
    ax.axvline(1.0, color="0.5", ls="--", lw=0.9, zorder=0)
    for ti, tier in enumerate(TIERS):
        sub = res[(res.model == model) & (res.tier == tier)].set_index("cue")
        xs = [sub.loc[c, "ratio"] if c in sub.index else np.nan for c in cue_labels]
        offset = (ti - 1.5) * 0.16
        ax.scatter(xs, y_pos + offset, s=22, color=tier_colors[ti],
                   edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_yticks(y_pos); ax.set_yticklabels(cue_labels)
    ax.set_title(mlabel, fontsize=10)
    ax.set_xlabel("Black / White salary ratio")
    ax.tick_params(axis="x", labelsize=8)
    ax.margins(y=0.06)

handles = [Line2D([0], [0], marker="o", ls="", color=tier_colors[i],
                  markeredgecolor="white", label=TIERS[i]) for i in range(4)]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
           frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=[0, 0.04, 1, 1])
out = os.path.join(REPL_DIR, "figures", "salary_tiers.pdf")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight")
print("SAVED", out)
