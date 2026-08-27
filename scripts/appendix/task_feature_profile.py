"""Per-cue linguistic feature profile: mean token length, TTR, dependency depth,
VADER sentiment, and politeness by cue, pooled across the three tasks.
Complements the single Flesch-Kincaid metric with five additional features."""
import os
import pandas as pd

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from cues.paths import APPENDIX as HERE, APPENDIX as RA, REPO_ROOT as REPL_DIR, DATA_DIR, PROMPTS, PROMPTS as PROMPT_DIR, RACE_INFERENCE as RECALL_DIR, RESPONSES as DATA, RACE_PRED, PLOT_DATA as PLOT, GPT_RUN, FIGURES, RACE_INFERENCE, REGRESSIONS_LLAMA, REGRESSIONS_ALL_MODELS, DIALECT_ALL_MODELS, require_dir, require_file, require_any, fail
TASKS = ["medical_advice", "legal_advice", "salary_rec"]
feat = pd.concat([pd.read_parquet(f"{RA}/task2_features_{t}.parquet") for t in TASKS], ignore_index=True)

LABEL = {
    "neutral": "No-cue (SAE)",
    "name_specific_rosenman": "Name (Rosenman)",
    "name_specific_hayes_elder": "Name (Hayes--Elder)",
    "name_specific_tzioumis": "Name (Tzioumis)",
    "convo_prefix": "Dialog (CAD)",
    "convo_prefix_prism": "Dialog (PRISM)",
    "explicit": "Explicit",
    "dialect": "Dialect (AAVE)",
}
ORDER = ["No-cue (SAE)", "Name (Rosenman)", "Name (Hayes--Elder)", "Name (Tzioumis)",
         "Dialog (CAD)", "Dialog (PRISM)", "Explicit", "Dialect (AAVE)"]
COLS = ["token_len", "ttr", "dep_depth", "vader", "polite"]

feat["cue"] = feat["id_cue"].map(LABEL)
prof = feat.groupby("cue")[COLS].mean().reindex(ORDER)
prof.columns = ["Tokens", "TTR", "DepDepth", "VADER", "Polite"]
print(prof.round(3).to_string())

# emit LaTeX rows
print("\n--- latex rows ---")
for cue, r in prof.iterrows():
    print(f"{cue} & {r.Tokens:.0f} & {r.TTR:.2f} & {r.DepDepth:.2f} & {r.VADER:+.2f} & {r.Polite:.2f} \\\\")
