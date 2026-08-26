"""Rebuild the two master tables the legacy replicate_* scripts read.

replicate_paper_specs.py, replicate_regressions.py and replicate_dialect_only.py
were written against an earlier pipeline that produced

    <CUES_MASTER_DIR>/medical_master.parquet
    <CUES_MASTER_DIR>/legal_salary_master.parquet

which are not part of the data release. This script reconstructs them, one
(task, cue, model) cell at a time, from the files that are released:

    decoder_model_responses_cleaned/     response_text, response_final, salary_final
    decoder_model_responses_race_pred/   the model's inferred race
    fk_cache/                            flesch_kincaid_grade
    asl_cache/                           avg_sentence_length   (build_asl_cache.py)
    revision_appendix/task2_features_*   type_token_ratio (ttr), prompt_vader (vader)

Merge keys per cue follow build_master_and_regress.py, so the joins are the same
ones the maintained pipeline uses. Seed 0 only, matching that pipeline.

    python build_asl_cache.py     # once
    python build_masters.py       # -> <CUES_MASTER_DIR>/*.parquet

Output is written a row group at a time, so peak memory is one (task, cue, model)
cell rather than the whole table.
"""
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from cues_io import require_dir, fail

HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.environ.get("CUES_ROOT", os.path.dirname(HERE))
DATA_DIR  = os.environ.get("CUES_DATA_DIR", os.path.join(REPO_ROOT, "data"))
RS_DIR    = os.path.join(DATA_DIR, "decoder_model_responses_cleaned")
RP_DIR    = os.path.join(DATA_DIR, "decoder_model_responses_race_pred")


def _cache_dir(name):
    """Caches may sit in the clone (if built here) or in the data release."""
    local = os.path.join(HERE, name)
    return local if os.path.isdir(local) else os.path.join(DATA_DIR, name)


FK_DIR    = _cache_dir("fk_cache")
ASL_DIR   = _cache_dir("asl_cache")
FEAT_DIR  = os.path.join(HERE, "revision_appendix")
OUT_DIR   = os.environ.get("CUES_MASTER_DIR", os.path.join(DATA_DIR, "masters"))

require_dir(RS_DIR, "cleaned responses (data/decoder_model_responses_cleaned)")
require_dir(RP_DIR, "race-prediction responses (data/decoder_model_responses_race_pred)")
require_dir(FK_DIR, "Flesch-Kincaid cache (fk_cache/) -- ships with the data, or run build_fk_cache.py")
require_dir(ASL_DIR, "sentence-length cache (asl_cache/) -- ships with the data, or run build_asl_cache.py")

TASKS  = ["medical_advice", "legal_advice", "salary_rec"]
CUES   = ["neutral", "explicit", "dialect",
          "name_specific_rosenman", "name_specific_hayes_elder",
          "name_specific_tzioumis", "name_specific_an",
          "convo_prefix", "convo_prefix_prism"]
MODELS = ["llama3.1", "olmo2", "gpt52"]

OUT_COLS = ["message_id", "prompt_id", "race", "gender", "race_pred", "id_cue",
            "response_text", "response_final",
            "type_token_ratio", "flesch_kincaid_grade", "avg_sentence_length",
            "prompt_vader", "model", "category"]

SCHEMA = pa.schema([
    ("message_id", pa.string()), ("prompt_id", pa.int64()),
    ("race", pa.string()), ("gender", pa.string()), ("race_pred", pa.string()),
    ("id_cue", pa.string()), ("response_text", pa.string()),
    ("response_final", pa.string()), ("type_token_ratio", pa.float64()),
    ("flesch_kincaid_grade", pa.float64()), ("avg_sentence_length", pa.float64()),
    ("prompt_vader", pa.float64()), ("model", pa.string()), ("category", pa.string()),
])


def keys_for_cue(cue):
    """Identical to build_master_and_regress.keys_for_cue."""
    if cue.startswith("name_specific_"):
        return ["prompt_id", "name"]
    if cue.startswith("convo_prefix"):
        return ["prompt_id", "convo_concat_id"]
    if cue == "explicit":
        return ["prompt_id", "race", "gender"]
    return ["prompt_id"]


def _read_csv(path, **kw):
    return pd.read_csv(path, low_memory=False, **kw) if os.path.exists(path) else None


def _cache(dirname, task, cue, col):
    f = os.path.join(dirname, f"{task}_{cue}.parquet")
    if not os.path.exists(f):
        return None
    df = pd.read_parquet(f)
    return df if col in df.columns else None


_FEAT = {}


def features(task):
    """ttr / vader per (id_cue, prompt_id, race), from the committed feature file."""
    if task not in _FEAT:
        f = os.path.join(FEAT_DIR, f"task2_features_{task}.parquet")
        if not os.path.exists(f):
            _FEAT[task] = None
        else:
            d = pd.read_parquet(f)
            d = d.rename(columns={"ttr": "type_token_ratio", "vader": "prompt_vader"})
            d["race"] = d["race"].astype(str).str.lower()
            _FEAT[task] = d[["id_cue", "prompt_id", "race",
                             "type_token_ratio", "prompt_vader"]].drop_duplicates(
                                 subset=["id_cue", "prompt_id", "race"])
    return _FEAT[task]


def build_cell(task, cue, model):
    rs = _read_csv(os.path.join(RS_DIR, f"{task}_{cue}_constrained_{model}_seed_0.csv"))
    if rs is None:
        return None, f"[skip] {task}/{cue}/{model}: no responses"
    rp = _read_csv(os.path.join(RP_DIR, f"{task}_{cue}_constrained_{model}_seed_0.csv"))
    if rp is None:
        return None, f"[skip] {task}/{cue}/{model}: no race predictions"

    keys = [k for k in keys_for_cue(cue) if k in rs.columns and k in rp.columns]
    if "prompt_id" not in keys:
        return None, f"[skip] {task}/{cue}/{model}: no usable merge key"

    # race_pred lives in the prediction file's response_text column
    rp = rp.rename(columns={"response_text": "race_pred"})
    rp_slim = rp[keys + ["race_pred"]].drop_duplicates(subset=keys)

    keep = keys + [c for c in ["race", "gender", "response_text", "response_final",
                               "salary_final"] if c in rs.columns and c not in keys]
    m = rs[keep].merge(rp_slim, on=keys, how="inner")

    # salary answers are parsed downstream from response_text; keep both columns
    if "response_final" not in m.columns:
        m["response_final"] = m["salary_final"] if "salary_final" in m.columns else np.nan
    m = m.drop(columns=[c for c in ["salary_final"] if c in m.columns])

    for dirname, col in ((FK_DIR, "flesch_kincaid_grade"), (ASL_DIR, "avg_sentence_length")):
        cache = _cache(dirname, task, cue, col)
        if cache is None:
            m[col] = np.nan
            continue
        ck = [k for k in keys if k in cache.columns]
        m = m.merge(cache[ck + [col]].drop_duplicates(subset=ck), on=ck, how="left")

    feat = features(task)
    if feat is None:
        m["type_token_ratio"] = np.nan
        m["prompt_vader"] = np.nan
    else:
        m["_race_l"] = m["race"].astype(str).str.lower() if "race" in m.columns else ""
        f = feat[feat["id_cue"] == cue]
        m = m.merge(f.rename(columns={"race": "_race_l"}).drop(columns=["id_cue"]),
                    on=["prompt_id", "_race_l"], how="left").drop(columns=["_race_l"])

    m["id_cue"] = cue
    m["model"] = model
    m["category"] = task
    m["prompt_id"] = pd.to_numeric(m["prompt_id"], errors="coerce").astype("Int64")
    m = m[m["prompt_id"].notna()]
    m["prompt_id"] = m["prompt_id"].astype("int64")
    m["message_id"] = f"{cue}_constrained_" + m["prompt_id"].astype(str)

    for c in OUT_COLS:
        if c not in m.columns:
            m[c] = np.nan
    for c in ("race", "gender", "race_pred", "response_text", "response_final"):
        m[c] = m[c].astype("object").where(m[c].notna(), None)
        m[c] = m[c].map(lambda v: None if v is None else str(v))
    for c in ("type_token_ratio", "flesch_kincaid_grade", "avg_sentence_length", "prompt_vader"):
        m[c] = pd.to_numeric(m[c], errors="coerce").astype("float64")
    return m[OUT_COLS], f"[ok]   {task}/{cue}/{model}  {len(m):,} rows"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    targets = {
        "medical_master.parquet": ["medical_advice"],
        "legal_salary_master.parquet": ["legal_advice", "salary_rec"],
    }
    total = 0
    for fname, tasks in targets.items():
        path = os.path.join(OUT_DIR, fname)
        writer = None
        rows = 0
        try:
            for task in tasks:
                for cue in CUES:
                    for model in MODELS:
                        df, msg = build_cell(task, cue, model)
                        print(msg, flush=True)
                        if df is None or df.empty:
                            continue
                        table = pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False)
                        if writer is None:
                            writer = pq.ParquetWriter(path, SCHEMA, compression="snappy")
                        writer.write_table(table)
                        rows += len(df)
        finally:
            if writer is not None:
                writer.close()
        if not rows:
            fail(f"produced no rows for {fname}")
        print(f"WROTE {path}  {rows:,} rows  "
              f"{os.path.getsize(path)/2**20:.0f} MB\n", flush=True)
        total += rows
    print(f"done: {total:,} rows across {len(targets)} master tables")


if __name__ == "__main__":
    main()
