"""Cue-group association strength -> main-text figure for 4.2.

Uses Figure 3's encoding grammar, legend, and rendering style so the two figures
are read the same way and the legend is learned once. Matches the notebook's
matplotlib default rendering (Type 3 CMUSerif fonts) rather than using pgf/LaTeX.

The x-axis is symlog (linear below 0.1, logarithmic above) so a share of
exactly zero sits at a real position on the axis, and small values are visible.
"""
import os
import numpy as np
import pandas as pd
from matplotlib import rc, font_manager
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Load CMU Serif so this figure matches the typeface of the others.
# Point CUES_FONT_DIR at a directory of .ttf files if CMU Serif is not
# installed system-wide; without it matplotlib falls back to DejaVu Serif.
font_dir = os.environ.get("CUES_FONT_DIR")
if font_dir and os.path.isdir(font_dir):
    for font_file in font_manager.findSystemFonts(fontpaths=font_dir, fontext="ttf"):
        font_manager.fontManager.addfont(font_file)

# Use the notebook's font setup exactly
rc('font', family='serif', serif=['CMU Serif', 'DejaVu Serif'])
rc('text', usetex=False)
plt.rcParams.update({
    'font.size': 12,
    'text.usetex': False,
})

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get(
    "CUES_FIG_OUT",
    os.path.normpath(os.path.join(HERE, "..", "figures", "recall_by_cue.pdf")))
os.makedirs(os.path.dirname(OUT), exist_ok=True)

BASE = 14                      # BASE_FONTSIZE in the Fig. 3 cell
TITLE = BASE + 1.5             # TITLE_FONTSIZE
AXIS_LABEL = BASE              # AXIS_LABEL_FONTSIZE
TICK = BASE - 2                # TICK_FONTSIZE
LEGEND = BASE - 0.5            # LEGEND_FONTSIZE
MARKER_SIZE = 7.0
MARKER_EDGE_WIDTH = 0.6
CI_LINE_WIDTH = 1.1
CI_CAP_SIZE = 3
CI_ALPHA = 0.9

COLORS = {"name": "#0072B2", "dialog": "#D55E00",
          "dialect": "#009E73", "explicit": "#E69F00"}
NAME_MARKERS = {"name_specific_rosenman": "^", "name_specific_hayes_elder": "s", "name_specific_tzioumis": "D"}
CONVO_MARKERS = {"convo_prefix": "o", "convo_prefix_prism": "s"}
MODEL_ORDER = ["LLaMA-3.1 8B", "OLMo2 7B", "GPT-5.2"]

# Values and 95% bootstrap CIs are derived from the per-cell confusion
# matrices, so the figure cannot drift from Table 5.
RECALL_DIR = os.path.normpath(os.path.join(HERE, "..", "results_recall"))
MODEL_KEY = {"LLaMA-3.1 8B": "llama3.1", "OLMo2 7B": "olmo2", "GPT-5.2": "gpt52"}
TASKS = ["medical_advice", "legal_advice", "salary_rec"]
CUES = [  # (family, marker, cue key), top row first
    ("explicit",     "o",                                    "explicit"),
    ("dialect",      "o",                                    "dialect"),
    ("name",         NAME_MARKERS["name_specific_rosenman"],     "name_specific_rosenman"),
    ("name",         NAME_MARKERS["name_specific_hayes_elder"],  "name_specific_hayes_elder"),
    ("name",         NAME_MARKERS["name_specific_tzioumis"],     "name_specific_tzioumis"),
    ("dialog",       CONVO_MARKERS["convo_prefix_prism"],       "convo_prefix_prism"),
    ("dialog",       CONVO_MARKERS["convo_prefix"],             "convo_prefix"),
]

def recall_ci(model, cue, n_boot=4000, seed=0):
    """Mean over tasks of recall for Black, with a percentile bootstrap CI.

    Each task contributes a binomial proportion (k Black-cued prompts inferred
    Black out of n); we resample each task's count and average the three, which
    matches how the point estimate in Table 5 is formed.
    """
    rng = np.random.default_rng(seed)
    kn = []
    for t in TASKS:
        f = os.path.join(RECALL_DIR, f"{MODEL_KEY[model]}_{t}", f"confusion_{cue}.csv")
        d = pd.read_csv(f, index_col=0)
        kn.append((d.loc["Black", "Black"], d.loc["Black"].sum()))
    point = float(np.mean([k / n for k, n in kn])) * 100
    draws = np.column_stack([rng.binomial(n, k / n, n_boot) / n
                             for k, n in kn]).mean(axis=1) * 100
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)

XMAX = 400
LEGEND_Y = -0.11   # shared top anchor for all three legend groups

fig, axes = plt.subplots(1, 3, figsize=(5.87, 3.3), sharey=True,
                         gridspec_kw={"wspace": 0.12})
ys = list(range(len(CUES)))[::-1]

for j, model in enumerate(MODEL_ORDER):
    ax = axes[j]
    for y, (fam, marker, cue) in zip(ys, CUES):
        ax.axhline(y, color="0.92", linewidth=0.6, zorder=0)
        v, lo, hi = recall_ci(model, cue)
        ax.errorbar(v, y, xerr=[[max(v - lo, 0)], [max(hi - v, 0)]],
                    fmt=marker, color=COLORS[fam], capsize=CI_CAP_SIZE,
                    elinewidth=CI_LINE_WIDTH, alpha=CI_ALPHA,
                    markersize=MARKER_SIZE, markeredgecolor="black",
                    markeredgewidth=MARKER_EDGE_WIDTH, zorder=3)
    ax.set_xscale("symlog", linthresh=0.1, linscale=0.85)
    ax.set_xlim(-0.05, XMAX)
    ax.set_ylim(-0.7, len(CUES) - 0.35)
    ax.set_xticks([0, 1, 100])
    ax.set_xticklabels(["0", "1", "100"])
    ax.set_title(model, fontsize=TITLE, pad=5)
    ax.tick_params(axis="x", labelsize=TICK, pad=2)
    ax.tick_params(axis="y", left=False, labelleft=False)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_linewidth(0.8)

fig.text(0.5, -0.07, r"Black-cued prompts inferred as Black (%)",
         ha="center", fontsize=AXIS_LABEL)

def mk(marker, fam, label):
    return Line2D([], [], marker=marker, linestyle="none",
                  markerfacecolor=COLORS[fam], markeredgecolor="black",
                  markeredgewidth=MARKER_EDGE_WIDTH,
                  markersize=MARKER_SIZE, label=label)

# LEGENDS (MATCH FIGURE 3 EXACTLY)
legend_cues = [
    Patch(facecolor=COLORS["explicit"], label="Explicit"),
    Patch(facecolor=COLORS["dialect"], label="Dialect"),
]

legend_dialog = [
    Line2D([], [], marker=CONVO_MARKERS["convo_prefix"], linestyle="none",
           markerfacecolor=COLORS["dialog"], markeredgecolor="black",
           markeredgewidth=MARKER_EDGE_WIDTH,
           markersize=MARKER_SIZE, label="CAD"),
    Line2D([], [], marker=CONVO_MARKERS["convo_prefix_prism"], linestyle="none",
           markerfacecolor=COLORS["dialog"], markeredgecolor="black",
           markeredgewidth=MARKER_EDGE_WIDTH,
           markersize=MARKER_SIZE, label="PRISM"),
]

legend_names = [
    Line2D([], [], marker=NAME_MARKERS["name_specific_hayes_elder"],
           linestyle="none", markerfacecolor=COLORS["name"],
           markeredgecolor="black",
           markeredgewidth=MARKER_EDGE_WIDTH,
           markersize=MARKER_SIZE, label="Elder & Hayes"),
    Line2D([], [], marker=NAME_MARKERS["name_specific_tzioumis"],
           linestyle="none", markerfacecolor=COLORS["name"],
           markeredgecolor="black",
           markeredgewidth=MARKER_EDGE_WIDTH,
           markersize=MARKER_SIZE, label="Tzioumis"),
    Line2D([], [], marker=NAME_MARKERS["name_specific_rosenman"],
           linestyle="none", markerfacecolor=COLORS["name"],
           markeredgecolor="black",
           markeredgewidth=MARKER_EDGE_WIDTH,
           markersize=MARKER_SIZE, label="Rosenman et al."),
]

fig.legend(
    handles=legend_cues,
    loc="upper left",
    bbox_to_anchor=(0.02, LEGEND_Y),
    frameon=False,
    fontsize=LEGEND,
    title_fontsize=LEGEND,
)

fig.legend(
    handles=legend_dialog,
    loc="upper center",
    bbox_to_anchor=(0.45, LEGEND_Y),
    frameon=False,
    fontsize=LEGEND,
    title="Dialog History",
    title_fontsize=LEGEND,
)

fig.legend(
    handles=legend_names,
    loc="upper right",
    bbox_to_anchor=(1.00, LEGEND_Y),
    frameon=False,
    fontsize=LEGEND,
    title="Name",
    title_fontsize=LEGEND,
)

plt.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.015)
print("wrote", OUT)
