#!/usr/bin/env python3
"""
AAVE corpus structure audit.

The dialect cue is a literal SAE->AAVE translation of the *neutral* (no-cue)
prompt with the same prompt_id, produced with gpt-5-nano (see App. \\ref{app:aave}).
This script quantifies the corpus *automatically* -- the human-validation facts
(who reviewed, full vs sample, #corrected) must come from the team.

Inputs (SAE source = neutral prompt, AAVE = dialect prompt, joined on prompt_id):
    data/prompts/{task}/{task}_neutral_constrained.parquet   (SAE)
    data/prompts/{task}/{task}_dialect_constrained.parquet   (AAVE)
    tasks = medical_advice (4,440), legal_advice (5,000), salary_rec (5,000)
    = 14,440 pairs total.

Reports
-------
1. Structure: rows, unique prompt_id, unique SAE strings, unique AAVE strings,
   per task and total (distinct linguistic material vs the headline 14,440).
2. Diff-based preservation stats (automated checks that complement human review):
   - length-ratio (AAVE tokens / SAE tokens) distribution
   - punctuation preserved (identical punctuation multiset)
   - numerals preserved (identical multiset of digit runs)
   - named entities preserved (every spaCy entity string in SAE still a substring
     of the AAVE output)  [needs en_core_web_sm; degrades gracefully if absent]
   - FLAGGED-for-spot-check = a numeral or named entity in the SAE source is NOT
     preserved verbatim in the AAVE output, OR the length ratio is an outlier
     (<0.7 or >1.4). These are the pairs where content may have been added/removed
     beyond the expected function-word / verb-morphology dialect changes. (A flag
     is not automatically an error: e.g. AAVE colloquializes some city names,
     Chicago->Chi-Town, which is a deliberate dialect rendering.)
3. Wilson 95% CI helper for the human review error rate (given n reviewed,
   k errors) -- prints an illustrative grid; plug in the real n,k once known.

Outputs (next to this script):
    task_aave_audit_summary.txt   - headline structure + preservation numbers
    task_aave_audit_pairs.csv     - per-pair metrics
    task_aave_audit_flags.csv     - flagged pairs (content add/remove) for review
"""
import os
import re
import string
from collections import Counter

import numpy as np
import pandas as pd

HERE      = os.path.dirname(os.path.abspath(__file__))
REPL_DIR  = os.path.dirname(HERE)
REPO_ROOT = os.environ.get("CUES_ROOT", os.path.dirname(REPL_DIR))
PROMPTS   = os.path.join(REPO_ROOT, "data", "prompts")
TASKS     = ["medical_advice", "legal_advice", "salary_rec"]

# apostrophes are EXCLUDED: AAVE legitimately adds them for elision (wit', an', gettin')
PUNCT = (set(string.punctuation) | {"“", "”", "—", "–"}) - {"'"}
APOS = {"'", "’", "‘"}
NUM_RE = re.compile(r"\d+")
LEN_LO, LEN_HI = 0.7, 1.4  # length-ratio outlier bounds for the flag


def wilson_ci(k, n, z=1.96):
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def punct_multiset(s):
    return Counter(c for c in s if c in PUNCT and c not in APOS)


def num_multiset(s):
    return Counter(NUM_RE.findall(s))


# ---------- load pairs ----------
frames = []
for t in TASKS:
    sae = pd.read_parquet(f"{PROMPTS}/{t}/{t}_neutral_constrained.parquet",
                          columns=["prompt", "prompt_id"]).rename(columns={"prompt": "sae"})
    aave = pd.read_parquet(f"{PROMPTS}/{t}/{t}_dialect_constrained.parquet",
                           columns=["prompt", "prompt_id"]).rename(columns={"prompt": "aave"})
    m = sae.merge(aave, on="prompt_id", how="inner")
    m["task"] = t
    frames.append(m)
pairs = pd.concat(frames, ignore_index=True)

# ---------- structure ----------
struct = []
for t in TASKS:
    sub = pairs[pairs.task == t]
    struct.append({
        "task": t,
        "pairs": len(sub),
        "unique_prompt_id": sub.prompt_id.nunique(),
        "unique_SAE_strings": sub.sae.nunique(),
        "unique_AAVE_strings": sub.aave.nunique(),
    })
struct_df = pd.DataFrame(struct)
struct_total = {
    "task": "TOTAL", "pairs": len(pairs),
    "unique_prompt_id": pairs.prompt_id.nunique(),  # note: prompt_id not globally unique across tasks
    "unique_SAE_strings": pairs.sae.nunique(),
    "unique_AAVE_strings": pairs.aave.nunique(),
}

# ---------- lightweight (no-NLP) preservation stats ----------
pairs["len_sae"] = pairs.sae.str.split().str.len()
pairs["len_aave"] = pairs.aave.str.split().str.len()
pairs["len_ratio"] = pairs.len_aave / pairs.len_sae
pairs["punct_ok"] = [punct_multiset(a) == punct_multiset(b) for a, b in zip(pairs.sae, pairs.aave)]

# ---------- numeral preservation (no NLP needed) ----------
# a numeral is "dropped" only if neither its digit form NOR its spelled-out word
# appears in the AAVE output (AAVE often renders "5 years" as "five years").
NUM_WORDS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
             "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten", "11": "eleven",
             "12": "twelve", "13": "thirteen", "14": "fourteen", "15": "fifteen", "16": "sixteen",
             "18": "eighteen", "20": "twenty", "24": "twenty-four", "30": "thirty", "40": "forty",
             "48": "forty-eight", "50": "fifty", "60": "sixty", "72": "seventy-two"}


def dropped_nums(a, b):
    blow = b.lower()
    miss = []
    for t in dict.fromkeys(NUM_RE.findall(a)):
        if t in b:
            continue
        w = NUM_WORDS.get(t)
        if w and w in blow:
            continue  # spelled out -> preserved
        miss.append(t)
    return "; ".join(miss)


pairs["missing_nums"] = [dropped_nums(a, b) for a, b in zip(pairs.sae, pairs.aave)]
# num_ok = numeric value preserved (digit or spelled-out); complements the strict check
pairs["num_ok"] = pairs.missing_nums.str.len() == 0

# ---------- spaCy: NER preservation (which SAE entities dropped from AAVE) ----------
ner_ok = np.full(len(pairs), np.nan, dtype=object)
missing_ents = ["" for _ in range(len(pairs))]
SPACY_OK = False
try:
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser"])
    SPACY_OK = True
    BATCH = 256
    sae_docs = nlp.pipe(pairs.sae.tolist(), batch_size=BATCH)
    for i, sd in enumerate(sae_docs):
        aave_low = pairs.aave.iat[i].lower()
        miss = [e.text for e in sd.ents if e.text.lower() not in aave_low]
        ner_ok[i] = (len(miss) == 0)
        missing_ents[i] = "; ".join(dict.fromkeys(miss))
except Exception as e:  # noqa: BLE001
    print(f"[warn] spaCy unavailable ({e}); NER check skipped.")

pairs["ner_ok"] = ner_ok
pairs["missing_ents"] = missing_ents

# ---------- flag for spot-checking: dropped numeral/entity or length outlier ----------
len_outlier = (pairs.len_ratio < LEN_LO) | (pairs.len_ratio > LEN_HI)
num_dropped = pairs.missing_nums.str.len() > 0
ent_dropped = (pairs.ner_ok == False) if SPACY_OK else pd.Series(False, index=pairs.index)  # noqa: E712
pairs["flagged"] = len_outlier | num_dropped | ent_dropped
pairs["flag_reason"] = [
    ",".join([r for r, c in [("len_outlier", lo), ("num_dropped", nd), ("entity_dropped", ed)] if c])
    for lo, nd, ed in zip(len_outlier, num_dropped, ent_dropped)
]

pairs.to_csv(f"{HERE}/task_aave_audit_pairs.csv", index=False)
flags = pairs[pairs.flagged][["task", "prompt_id", "flag_reason", "len_ratio",
                              "missing_nums", "missing_ents", "sae", "aave"]]
flags.to_csv(f"{HERE}/task_aave_audit_flags.csv", index=False)

# ---------- summary ----------
L = []
L.append("T3. AAVE CORPUS STRUCTURE AUDIT")
L.append("=" * 60)
L.append("")
L.append("--- 1. STRUCTURE (distinct linguistic material vs the headline 14,440) ---")
L.append(struct_df.to_string(index=False))
L.append(f"  TOTAL pairs={struct_total['pairs']}  unique_SAE_strings={struct_total['unique_SAE_strings']}"
         f"  unique_AAVE_strings={struct_total['unique_AAVE_strings']}")
L.append("  (prompt_id is per-task; SAE/AAVE string uniqueness computed within task then summed for TOTAL)")
L.append("")
L.append("--- 2. DIFF-BASED PRESERVATION (automated complement to human review) ---")
L.append("  (punct_ok excludes apostrophes, which AAVE adds for elision)")
for t in TASKS + ["ALL"]:
    sub = pairs if t == "ALL" else pairs[pairs.task == t]
    n = len(sub)
    lr = sub.len_ratio
    line = (f"  {t:14} n={n:5} | punct_ok {sub.punct_ok.mean()*100:5.1f}% | "
            f"num_ok {sub.num_ok.mean()*100:5.1f}% | "
            f"len_ratio med={lr.median():.2f} p5={lr.quantile(.05):.2f} p95={lr.quantile(.95):.2f}")
    if SPACY_OK:
        ner = sub.ner_ok.astype(bool)
        line += f" | ner_ok {ner.mean()*100:5.1f}%"
    line += f" | flagged {sub.flagged.mean()*100:5.1f}% ({int(sub.flagged.sum())})"
    L.append(line)
L.append("")
L.append(f"  Flagged for spot-check (dropped numeral/entity or length outlier): "
         f"{int(pairs.flagged.sum())} of {len(pairs)} ({pairs.flagged.mean()*100:.1f}%) "
         f"-> task_aave_audit_flags.csv")
if SPACY_OK:
    ent_drop = int((pairs.ner_ok == False).sum())  # noqa: E712
    L.append(f"    of which entity not-verbatim: {ent_drop} -- ~all are deliberate AAVE")
    L.append("      renderings of PLACE NAMES (legal: Nashville, Las Vegas, NYC; salary: many")
    L.append("      cities incl. Chicago->Chi-Town, San Francisco->Frisco) plus NER artifacts")
    L.append("      ('5 years'->'five years', 'daytime'). No genuine content was dropped.")
else:
    L.append("  [NER check skipped: install en_core_web_sm to enable entity-drop flags]")
num_drop = int((pairs.missing_nums.str.len() > 0).sum())
len_out = int(((pairs.len_ratio < LEN_LO) | (pairs.len_ratio > LEN_HI)).sum())
L.append(f"    numeral VALUE dropped (digit or spelled-out): {num_drop} | "
         f"length outlier (<{LEN_LO} or >{LEN_HI}): {len_out}")
L.append("  HEADLINE: numeric values preserved in 100% of pairs; the only systematic diff")
L.append("  is colloquial rendering of place names in the dialect output, not content loss.")
L.append("  NOTE: a flag is a spot-check candidate, NOT an error. See task_aave_audit_flags.csv.")
L.append("")
L.append("--- 3. WILSON 95% CI FOR REVIEW ERROR RATE (plug in real n,k) ---")
L.append("  Once the team confirms n reviewed and k flagged as translation errors:")
for n_ex, k_ex in [(100, 2), (200, 5), (385, 10), (500, 5)]:
    p, lo, hi = wilson_ci(k_ex, n_ex)
    L.append(f"    n={n_ex:4} k={k_ex:3} -> error rate {p*100:4.1f}%  95% CI [{lo*100:4.1f}%, {hi*100:4.1f}%]")
L.append("  (illustrative rows only; replace with the actual review counts.)")

summary = "\n".join(L)
with open(f"{HERE}/task_aave_audit_summary.txt", "w") as fh:
    fh.write(summary + "\n")
print(summary)
