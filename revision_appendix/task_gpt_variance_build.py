#!/usr/bin/env python3
"""
GPT-5.2 repeated-run variance harness (Appendix B.3).

PREP ONLY -- this script builds and prices the OpenAI Batch job and parses its
output. It does NOT call the API. Submitting the batch spends money; that is a
human go/no-go decision (see the three HUMAN DECISIONS below). Nothing here
touches an API key.

Goal: quantify GPT-5.2 run-to-run (sampling) variance so we can state
that stochasticity is small relative to the cross-cue effects in Figure 3.
Design: temperature 1 (sampling stochasticity IS the quantity of interest; no
seed), k repetitions of a subsample of the SAME prompts used in the main
experiments.

PROMPT SOURCE (memory-safe by design): prompts are reused verbatim. The main
GPT-5.2 run already wrote its exact request JSONLs under data/gpt_run/*/batches/
(each file = 50k lines, homogeneous in condition). We index those files by
condition and sample n_base lines from ONE file per (task, condition) -- streamed
line-by-line, never loading the 0.7-1.3 GB prompt parquets (single 2.7 GB row
groups -> would blow up RAM). The two conditions with thin batch coverage
(no-cue, dialect/AAVE) fall back to their SMALL parquets (<=0.3 MB); any fallback
parquet above SAFE_PARQUET_MB is skipped with a warning rather than risking RAM.

Request format (matches the main GPT-5.2 generation batches exactly):
    url="/v1/chat/completions", model="gpt-5.2",
    messages=[{"role":"user","content": <prompt>}], temperature=1

Usage
-----
    python task_gpt_variance_build.py                 # dry-run: sample, price, 10-line preview
    python task_gpt_variance_build.py --n-base 150 --k 5
    python task_gpt_variance_build.py --emit           # ALSO write the full request JSONL
    python task_gpt_variance_build.py --parse OUT.jsonl # parse a completed batch output -> SD table

Outputs (next to this script):
    gpt_variance_manifest.csv        - custom_id -> (task, condition, prompt_id, race, rep)
    gpt_variance_cost.txt            - token + cost estimate and the design summary
    gpt_variance_requests.jsonl      - full batch input       (only with --emit)
    gpt_variance_requests.sample.jsonl - first 10 requests    (always, for inspection)
    gpt_variance_sd_by_condition.csv - run-to-run SD table    (only with --parse)

HUMAN DECISIONS (do not hard-code a spend):
    1. design size (--n-base / --k). Proposal: 150 x 8 x 3 x k=5 = 18,000 calls.
       Larger: 500/task/condition x k=3 = 36,000. Pick to budget.
    2. VERIFY CURRENT gpt-5.2 PRICING before submitting (set --p-in/--p-out).
    3. whether to submit at all, and via the Batch API (50% off, latency irrelevant).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.environ.get("CUES_ROOT", os.path.dirname(os.path.dirname(HERE)))
DATA_DIR  = os.environ.get("CUES_DATA_DIR", os.path.join(REPO_ROOT, "data"))
PROMPTS   = os.path.join(DATA_DIR, "prompts")
GPT_RUN   = os.path.join(DATA_DIR, "gpt_run")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cues_io import require_dir, require_file, require_any
require_dir(PROMPTS, "prompt directory (data/prompts)")
SAFE_PARQUET_MB = 50  # never read a fallback parquet larger than this into RAM

TASKS = ["medical_advice", "legal_advice", "salary_rec"]
# 8 cue conditions -> parquet stem (constrained variants, as in the main runs)
CONDITIONS = {
    "nocue":     "{t}_neutral_constrained",
    "explicit":  "{t}_explicit_constrained",
    "aave":      "{t}_dialect_constrained",
    "cad":       "{t}_convo_prefix_constrained",
    "prism":     "{t}_convo_prefix_prism_constrained",
    "rosenman":  "{t}_name_specific_rosenman_constrained",
    "hayeselder":"{t}_name_specific_hayes_elder_constrained",
    "tzioumis":  "{t}_name_specific_tzioumis_constrained",
}
MODEL = "gpt-5.2"
_enc = None


def n_tokens(texts):
    """Input token count via tiktoken o200k_base (fallback chars/4)."""
    global _enc
    try:
        if _enc is None:
            import tiktoken
            _enc = tiktoken.get_encoding("o200k_base")
        return sum(len(_enc.encode(t)) for t in texts)
    except Exception:
        return int(sum(len(t) for t in texts) / 4)


def index_batch_files():
    """Map (task, condition-stem) -> a batch JSONL file, by reading each file's first line."""
    idx = {}
    for root, _dirs, files in os.walk(GPT_RUN):
        if os.path.basename(root) != "batches":
            continue
        for fn in files:
            if not fn.endswith(".jsonl"):
                continue
            path = os.path.join(root, fn)
            try:
                with open(path) as fh:
                    cid = json.loads(fh.readline())["custom_id"]
            except Exception:
                continue
            stem = cid.split(".parquet")[0]           # e.g. legal_advice_name_specific_rosenman_constrained
            idx.setdefault(stem, path)                # first file per condition is enough (50k lines)
    return idx


def _reservoir(iterable, n, rng):
    """Streaming reservoir sample of n items (Algorithm R). Memory = O(n)."""
    res = []
    for i, item in enumerate(iterable):
        if i < n:
            res.append(item)
        else:
            j = int(rng.integers(0, i + 1))
            if j < n:
                res[j] = item
    return res


def _sample_from_batch(path, n, rng):
    """Yield up to n (content, source_row_id) sampled from a homogeneous batch JSONL."""
    def rows():
        with open(path) as fh:
            for line in fh:
                try:
                    o = json.loads(line)
                    yield (o["body"]["messages"][0]["content"], o["custom_id"].split("::")[-1])
                except Exception:
                    continue
    return _reservoir(rows(), n, rng)


def _sample_from_parquet(task, stem, n, rng):
    """Fallback for small conditions only (guarded by SAFE_PARQUET_MB)."""
    path = os.path.join(PROMPTS, task, stem + ".parquet")
    mb = os.path.getsize(path) / 1e6
    if mb > SAFE_PARQUET_MB:
        print(f"[skip] {stem}: parquet {mb:.0f} MB > {SAFE_PARQUET_MB} MB safe limit; "
              f"no batch file indexed. Sample this condition from data/gpt_run instead.")
        return []
    df = pd.read_parquet(path, columns=["prompt", "prompt_id"])
    take = min(n, len(df))
    sub = df.sample(take, random_state=int(rng.integers(1 << 31)))
    return [(r.prompt, str(int(r.prompt_id))) for _, r in sub.iterrows()]


def build(n_base, k, seed=0):
    rng = np.random.default_rng(seed)
    bidx = index_batch_files()
    rows, requests = [], []
    for ti, task in enumerate(TASKS):
        for ci, (cond, stem_t) in enumerate(CONDITIONS.items()):
            stem = stem_t.format(t=task)
            if stem in bidx:
                samples = _sample_from_batch(bidx[stem], n_base, rng)
                src = "batch"
            else:
                samples = _sample_from_parquet(task, stem, n_base, rng)
                src = "parquet"
            if not samples:
                continue
            for j, (content, row_id) in enumerate(samples):
                for rep in range(k):
                    cid = f"v{ti}{ci}_{j:04d}_{rep}"   # <=64 chars; metadata in manifest
                    requests.append({
                        "custom_id": cid, "method": "POST", "url": "/v1/chat/completions",
                        "body": {"model": MODEL,
                                 "messages": [{"role": "user", "content": content}],
                                 "temperature": 1},
                    })
                    rows.append({"custom_id": cid, "task": task, "condition": cond,
                                 "src_row": row_id, "rep": rep, "prompt_source": src})
    return pd.DataFrame(rows), requests


def parse_output(out_path, manifest_path):
    """Map a completed Batch output JSONL to run-to-run SD per (task, condition)."""
    man = pd.read_csv(manifest_path).set_index("custom_id")
    recs = []
    with open(out_path) as fh:
        for line in fh:
            o = json.loads(line)
            cid = o.get("custom_id")
            try:
                content = o["response"]["body"]["choices"][0]["message"]["content"].strip()
            except Exception:
                content = None
            recs.append({"custom_id": cid, "raw": content})
    out = pd.DataFrame(recs).set_index("custom_id").join(man, how="inner").reset_index()

    def to_outcome(row):
        raw = (row.raw or "").strip().lower()
        if row.task == "salary_rec":
            import re
            m = re.search(r"\d[\d,]*", raw.replace("$", ""))
            return float(m.group().replace(",", "")) if m else np.nan
        return 1.0 if raw.startswith("yes") else 0.0 if raw.startswith("no") else np.nan

    out["outcome"] = out.apply(to_outcome, axis=1)
    # run-to-run SD: SD across the k reps of each prompt, then mean over prompts
    per_prompt = (out.groupby(["task", "condition", "src_row"])["outcome"]
                    .agg(["mean", "std", "count"]).reset_index())
    by_cond = (per_prompt.groupby(["task", "condition"])
                 .agg(run_to_run_sd=("std", "mean"),
                      mean_outcome=("mean", "mean"),
                      n_prompts=("src_row", "nunique")).reset_index())
    by_cond.to_csv(os.path.join(HERE, "gpt_variance_sd_by_condition.csv"), index=False)
    print(by_cond.to_string(index=False))
    print("\nCompare run_to_run_sd against the smallest cross-cue effect in Figure 3.")
    return by_cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-base", type=int, default=150)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--emit", action="store_true", help="write the full request JSONL")
    ap.add_argument("--parse", metavar="OUT.jsonl", help="parse a completed batch output")
    ap.add_argument("--p-in", type=float, default=1.25, help="$/M input tokens (VERIFY)")
    ap.add_argument("--p-out", type=float, default=10.0, help="$/M output tokens (VERIFY)")
    args = ap.parse_args()

    if args.parse:
        parse_output(args.parse, os.path.join(HERE, "gpt_variance_manifest.csv"))
        return

    man, requests = build(args.n_base, args.k)
    man.to_csv(os.path.join(HERE, "gpt_variance_manifest.csv"), index=False)

    # always write a 10-line preview; full file only with --emit
    with open(os.path.join(HERE, "gpt_variance_requests.sample.jsonl"), "w") as fh:
        for req in requests[:10]:
            fh.write(json.dumps(req) + "\n")
    if args.emit:
        with open(os.path.join(HERE, "gpt_variance_requests.jsonl"), "w") as fh:
            for req in requests:
                fh.write(json.dumps(req) + "\n")

    # token + cost estimate from the ACTUAL sampled prompts
    in_tok = n_tokens([r["body"]["messages"][0]["content"] for r in requests])
    out_tok = 10 * len(requests)  # constrained Yes/No or a number ~<=10 tok/call
    cost = in_tok / 1e6 * args.p_in + out_tok / 1e6 * args.p_out
    batch_cost = cost * 0.5

    lines = []
    lines.append("T6. GPT-5.2 REPEATED-RUN VARIANCE -- design + cost (PREP ONLY)")
    lines.append("=" * 60)
    lines.append(f"  design: n_base={args.n_base}/task/condition x {len(CONDITIONS)} conditions "
                 f"x {len(TASKS)} tasks x k={args.k} reps")
    lines.append(f"  total requests: {len(requests):,}   (unique prompts: {len(requests)//args.k:,})")
    lines.append(f"  input tokens (measured on sampled prompts): {in_tok:,}")
    lines.append(f"  output tokens (<=10/call assumed):          {out_tok:,}")
    lines.append(f"  cost formula: in/1e6 * p_in + out/1e6 * p_out")
    lines.append(f"  at p_in=${args.p_in}/M, p_out=${args.p_out}/M  ->  ${cost:,.2f} "
                 f"(Batch API 50% off: ${batch_cost:,.2f})")
    lines.append("  *** VERIFY CURRENT gpt-5.2 PRICING before committing (set --p-in/--p-out). ***")
    lines.append("")
    lines.append("  wrote: gpt_variance_manifest.csv, gpt_variance_requests.sample.jsonl"
                 + (", gpt_variance_requests.jsonl" if args.emit else " (use --emit for full JSONL)"))
    lines.append("")
    lines.append("  NEXT (human): 1) pick design size  2) verify pricing  3) submit via Batch API:")
    lines.append("     files.create(purpose='batch') -> batches.create(endpoint='/v1/chat/completions')")
    lines.append("     then: python task_gpt_variance_build.py --parse <output>.jsonl")
    report = "\n".join(lines)
    with open(os.path.join(HERE, "gpt_variance_cost.txt"), "w") as fh:
        fh.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
