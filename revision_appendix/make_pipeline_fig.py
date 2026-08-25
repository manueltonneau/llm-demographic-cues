"""Task 4: deviation-computation pipeline schematic -> images/pipeline.pdf"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif", "font.size": 9, "text.usetex": False})

HERE     = os.path.dirname(os.path.abspath(__file__))              # replication/revision_appendix
REPL_DIR = os.path.dirname(HERE)                                   # the replication/ dir
OUT = os.path.join(REPL_DIR, "figures", "pipeline.pdf")

# cue palette (matches paper)
C_NAME = "#0072B2"; C_EXP = "#E69F00"; C_DIA = "#009E73"; C_DLG = "#D55E00"
GREY = "#444444"; LIGHT = "#f2f2f2"

fig, ax = plt.subplots(figsize=(7.0, 3.9))
ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis("off")

def box(x, y, w, h, text, fc=LIGHT, ec=GREY, fs=9, tc="black", lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.5",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, wrap=True)

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                 lw=1.1, color=GREY, shrinkA=0, shrinkB=0))

# Stage 1: raw outputs
box(1, 30, 16, 12,
    "Raw model outputs\n(prompt, cue,\nmodel, seed)", fc="white", fs=8.5)

# Stage 2: averaging (with sub-note)
box(22, 30, 20, 12,
    "Average over\n(a) within-cue variants\n(b) random seeds", fc=LIGHT, fs=8.5)
ax.text(32, 27.5,
        "names: 50/group  |  dialog: 50 clusters\nexplicit: 23 variants  |  seeds 0–2",
        ha="center", va="top", fontsize=6.5, color=GREY, style="italic")

# Stage 3: per-prompt cue mean + no-cue baseline (stacked) -> deviation
box(47, 43, 22, 11, "Per-prompt cue mean", fc="white", fs=8.5)
box(47, 30, 22, 11, "No-cue baseline\n(same prompt)", fc="white", fs=8.5)
box(74, 34, 23, 13, "Cue-induced\ndeviation\n(cue − no-cue)", fc=LIGHT, fs=8.5, lw=1.4)

arrow(17, 36, 22, 36)         # raw -> average
arrow(42, 36, 47, 48)         # average -> cue mean
arrow(42, 36, 47, 35.5)       # average -> no-cue baseline
arrow(69, 48, 74, 42)         # cue mean -> deviation
arrow(69, 35.5, 74, 39)       # baseline -> deviation

# Stage 4: two analyses below the deviation box
box(40, 2, 26, 7.5,
    "Within-group Pearson\ncorrelations across cues\n(Fig. 2)", fc="white", fs=7.4)
box(70, 2, 27, 7.5,
    "Black/White outcome\nratios\n(Fig. 3)", fc="white", fs=7.4)
arrow(80, 34, 57, 9.5)        # deviation -> correlations
arrow(89, 34, 84, 9.5)        # deviation -> ratios

fig.tight_layout(pad=0.3)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print("SAVED", OUT)
