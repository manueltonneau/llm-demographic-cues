"""Correlation-robustness appendix: shows the within-cue near-unity correlations
are not an artifact of low-variance deviation vectors, and hold under Spearman
and cosine in addition to Pearson.

Reuses per_seed_agg (seed-pooled) to build cue-induced deviation vectors per
(model, task, cue, race), then aggregates within-cue (same family, different
data source) vs cross-cue (different family) similarity, within race.
"""
import itertools
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))                  # revision_appendix
AGG = os.path.join(HERE, "per_seed_agg.parquet")
MODELS = ["llama3.1", "olmo2"]
TASKS = ["medical_advice", "legal_advice", "salary_rec"]
NAME = ["name_specific_rosenman", "name_specific_hayes_elder", "name_specific_tzioumis"]
DIALOG = ["convo_prefix", "convo_prefix_prism"]
FAMILY = {**{n: "name" for n in NAME}, **{d: "dialog" for d in DIALOG},
          "explicit": "explicit", "dialect": "dialect"}

df = pd.read_parquet(AGG)
# pool across seeds
df = df.groupby(["model", "task", "method", "race", "prompt_id"], as_index=False)[["sum_val", "n"]].sum()
df["mean"] = df["sum_val"] / df["n"]

def fisher_mean(rs):
    rs = np.clip(np.array([r for r in rs if not np.isnan(r)]), -0.999999, 0.999999)
    return np.tanh(np.nanmean(np.arctanh(rs))) if len(rs) else np.nan

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

within_p, within_s, within_c = [], [], []
cross_p, cross_s, cross_c = [], [], []
vec_sd = []   # deviation-vector SDs (to show non-degenerate)

for model in MODELS:
    for task in TASKS:
        d = df[(df.model == model) & (df.task == task)]
        neutral = d[d.method == "neutral"].set_index("prompt_id")["mean"]
        # build deviation vectors per (method, race)
        vecs = {}
        for method in FAMILY:
            for race in ["black", "white"]:
                s = d[(d.method == method) & (d.race == race)].set_index("prompt_id")["mean"]
                pids = s.index.intersection(neutral.index)
                if len(pids) < 10:
                    continue
                v = (s.loc[pids] - neutral.loc[pids])
                if v.std() > 0:
                    vecs[(method, race)] = v
                    vec_sd.append(v.std())
        keys = list(vecs)
        for (ka, kb) in itertools.combinations(keys, 2):
            ma, ra = ka; mb, rb = kb
            if ra != rb:
                continue  # within-race only
            pids = vecs[ka].index.intersection(vecs[kb].index)
            if len(pids) < 10:
                continue
            a, b = vecs[ka].loc[pids].values, vecs[kb].loc[pids].values
            r = np.corrcoef(a, b)[0, 1]
            rho = spearmanr(a, b).correlation
            cos = cosine(a, b)
            same_family = FAMILY[ma] == FAMILY[mb]
            diff_source = ma != mb
            if same_family and diff_source:           # within-cue, cross-source
                within_p.append(r); within_s.append(rho); within_c.append(cos)
            elif not same_family:                     # cross-cue-type
                cross_p.append(r); cross_s.append(rho); cross_c.append(cos)

print("=== Within-cue (same family, different data source) ===")
print(f"  Pearson  r   = {fisher_mean(within_p):.3f}  (n pairs={len(within_p)})")
print(f"  Spearman rho = {fisher_mean(within_s):.3f}")
print(f"  Cosine       = {np.mean(within_c):.3f}")
print("=== Cross-cue (different family) ===")
print(f"  Pearson  r   = {fisher_mean(cross_p):.3f}  (n pairs={len(cross_p)})")
print(f"  Spearman rho = {fisher_mean(cross_s):.3f}")
print(f"  Cosine       = {np.mean(cross_c):.3f}")
print("=== Deviation-vector SD (non-degeneracy check) ===")
vec_sd = np.array(vec_sd)
print(f"  n vectors={len(vec_sd)}  min={vec_sd.min():.4g}  median={np.median(vec_sd):.4g}  max={vec_sd.max():.4g}")
