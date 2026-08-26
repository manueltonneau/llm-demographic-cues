"""Task 2 step 1: per-prompt linguistic features for the extended controls.

Features (computed on the cued prompt text):
  - token_len : whitespace token count
  - ttr       : type-token ratio (lowercased alphabetic tokens)
  - dep_depth : mean dependency-tree depth across sentences (spaCy en_core_web_sm,
                parsed on length-capped text to bound cost on long dialog prefixes)
  - vader     : VADER compound (vaderSentiment, same library as data/prompts_vader)
  - polite    : politeness-lexicon score (lexicon hits per 100 tokens)

Granularity: one representative realization per (task, id_cue, prompt_id, race)
(features are text-model-independent, so computed once and reused for all
behavioral models). Documented as an approximation in the appendix.
"""
import os, re, glob
import numpy as np
import pandas as pd
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

HERE      = os.path.dirname(os.path.abspath(__file__))             # revision_appendix
REPO_ROOT = os.environ.get("CUES_ROOT", os.path.dirname(os.path.dirname(HERE)))
DATA_DIR  = os.environ.get("CUES_DATA_DIR", os.path.join(REPO_ROOT, "data"))
PROMPT_DIR = os.path.join(DATA_DIR, "prompts")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cues_io import require_dir, require_file, require_any
require_dir(PROMPT_DIR, "prompt directory (data/prompts)")
OUT = HERE
TASKS = ["medical_advice", "legal_advice", "salary_rec"]
EXCLUDE = {"name_specific_an"}

POLITE_LEX = [
    "please", "thank you", "thanks", "could you", "would you", "i appreciate",
    "appreciate", "kindly", "sorry", "excuse me", "if you don't mind",
    "i was wondering", "grateful", "pardon", "may i",
]

nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "textcat"])
analyzer = SentimentIntensityAnalyzer()
WORD = re.compile(r"[a-z]+")

def token_len(t):
    return len(t.split())

def ttr(t):
    toks = WORD.findall(t.lower())
    return (len(set(toks)) / len(toks)) if toks else np.nan

def polite(t):
    tl = t.lower()
    hits = sum(tl.count(p) for p in POLITE_LEX)
    return 100.0 * hits / max(len(t.split()), 1)

def vader(t):
    return analyzer.polarity_scores(t)["compound"] if isinstance(t, str) and t else 0.0

def tree_depth(token):
    best = 1
    stack = [(token, 1)]
    while stack:
        tok, d = stack.pop()
        if d > best:
            best = d
        for c in tok.children:
            stack.append((c, d + 1))
    return best

def dep_depth(doc):
    depths = [tree_depth(s.root) for s in doc.sents]
    return float(np.mean(depths)) if depths else np.nan

for task in TASKS:
    rep_rows = []
    for f in sorted(glob.glob(os.path.join(PROMPT_DIR, task, f"{task}_*_constrained.parquet"))):
        cue = os.path.basename(f).replace(f"{task}_", "").replace("_constrained.parquet", "")
        if cue in EXCLUDE or "unconstrained" in f:
            continue
        df = pd.read_parquet(f, columns=["prompt", "prompt_id", "race"])
        df["race"] = df["race"].astype(str).str.lower()
        rep = df.drop_duplicates(subset=["prompt_id", "race"], keep="first").copy()
        rep["id_cue"] = cue
        rep_rows.append(rep[["id_cue", "prompt_id", "race", "prompt"]])
        del df
    reps = pd.concat(rep_rows, ignore_index=True)
    print(f"[{task}] {len(reps)} representative texts", flush=True)

    texts = reps["prompt"].fillna("").tolist()
    reps["token_len"] = [token_len(t) for t in texts]
    reps["ttr"] = [ttr(t) for t in texts]
    reps["polite"] = [polite(t) for t in texts]
    reps["vader"] = [vader(t) for t in texts]

    # dep_depth is structural: invariant to which name/label fills the memory
    # slot, so mask it before dedup (collapses the per-name/label explosion).
    # Length/TTR/sentiment/politeness above use the full original text.
    CAP = 1000
    mem_re = re.compile(r"\[MEMORY:[^\]]*\]")
    keyfn = lambda t: mem_re.sub("[MEMORY: X]", t)[:CAP]
    keys = [keyfn(t) for t in texts]
    uniq = list(dict.fromkeys(keys))
    print(f"  parsing {len(uniq)} unique masked-capped texts (of {len(keys)})", flush=True)
    depth_map = {}
    for j, doc in enumerate(nlp.pipe(uniq, batch_size=128, n_process=4)):
        depth_map[uniq[j]] = dep_depth(doc)
        if (j + 1) % 5000 == 0:
            print(f"  parsed {j+1}/{len(uniq)}", flush=True)
    reps["dep_depth"] = [depth_map[k] for k in keys]
    reps = reps.drop(columns=["prompt"])
    out_f = os.path.join(OUT, f"task2_features_{task}.parquet")
    reps.to_parquet(out_f, index=False)
    print(f"  SAVED {out_f} {reps.shape}", flush=True)
