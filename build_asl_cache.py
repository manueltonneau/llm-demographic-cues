"""Precompute average sentence length per prompt -> asl_cache/{task}_{cue}.parquet.

Companion to build_fk_cache.py. The eight nested specifications in
replicate_regressions.py control for three readability measures; two of them
(Flesch-Kincaid grade, type-token ratio) are already available from fk_cache/ and
revision_appendix/task2_features_*.parquet, and this script supplies the third.

Like fk_cache/, the output holds only identifiers and the derived number, never
prompt text, so it can be redistributed alongside the responses.

Run once before build_masters.py:
    python build_asl_cache.py
"""
import os
import re
import sys

import pandas as pd

from cues_io import require_dir

HERE       = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.environ.get("CUES_ROOT", os.path.dirname(HERE))
DATA_DIR   = os.environ.get("CUES_DATA_DIR", os.path.join(REPO_ROOT, "data"))
PROMPT_DIR = os.path.join(DATA_DIR, "prompts")
CACHE_DIR  = os.path.join(HERE, "asl_cache")

require_dir(PROMPT_DIR, "prompt directory (data/prompts)")

TASKS = ["medical_advice", "legal_advice", "salary_rec"]
CUES  = ["neutral", "explicit", "dialect",
         "name_specific_rosenman", "name_specific_hayes_elder",
         "name_specific_tzioumis", "name_specific_an",
         "convo_prefix", "convo_prefix_prism"]

# Identifier columns to carry through, mirroring fk_cache. Kept only if present.
ID_COLS = ["prompt_id", "convo_concat_id", "name", "race", "gender", "demo_group"]

SENT_END = re.compile(r"[.!?]+")
WORD = re.compile(r"\b\w+\b")


def avg_sentence_length(text):
    """Words per sentence. Mirrors the textstat convention: a trailing fragment
    without terminal punctuation still counts as one sentence."""
    if not isinstance(text, str) or not text.strip():
        return float("nan")
    n_words = len(WORD.findall(text))
    if not n_words:
        return float("nan")
    n_sent = max(1, len([p for p in SENT_END.split(text) if p.strip()]))
    return n_words / n_sent


def build_one(task, cue):
    src = os.path.join(PROMPT_DIR, task, f"{task}_{cue}_constrained.parquet")
    if not os.path.exists(src):
        return f"[skip] {task}/{cue} (no prompts)"
    dst = os.path.join(CACHE_DIR, f"{task}_{cue}.parquet")
    if os.path.exists(dst):
        return f"[have] {task}/{cue}"

    import pyarrow.parquet as pq
    have = set(pq.ParquetFile(src).schema.names)
    cols = [c for c in ID_COLS if c in have] + ["prompt"]
    df = pd.read_parquet(src, columns=cols)
    df["avg_sentence_length"] = df["prompt"].map(avg_sentence_length)
    df = df.drop(columns=["prompt"])
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_parquet(dst, index=False)
    return f"[ok]   {task}/{cue}  {len(df):,} rows"


def main():
    built = 0
    for task in TASKS:
        for cue in CUES:
            msg = build_one(task, cue)
            print(msg, flush=True)
            built += not msg.startswith("[skip]")
    if not built:
        sys.exit("[error] no prompt files found; nothing to cache")


if __name__ == "__main__":
    main()
