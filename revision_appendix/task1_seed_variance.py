"""Task 1: seed-level variance of (i) within-cue cross-source correlations
and (ii) Black/White outcome ratios, for LLaMA-3.1 and OLMo2.

Replicates the figure pipelines per seed (instead of pooling across seeds):
  - deviation(method,race)[pid] = mean(cue) - mean(no-cue)  (per prompt)
  - within-cue r: Pearson between deviation vectors of different data SOURCES
    of the same cue family (names: R/EH/T; dialog: CAD/PRISM), within race.
  - B/W ratio: pooled mean(black)/mean(white) per cue; dialect vs no-cue.
Then mean & SD across seeds 0-2.
"""
import itertools
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))                  # revision_appendix
AGG = os.path.join(HERE, "per_seed_agg.parquet")
MODELS = ["llama3.1", "olmo2"]
TASKS = ["medical_advice", "legal_advice", "salary_rec"]
NAME_SRC = ["name_specific_rosenman", "name_specific_hayes_elder", "name_specific_tzioumis"]
DIALOG_SRC = ["convo_prefix", "convo_prefix_prism"]
RATIO_CUES = NAME_SRC + DIALOG_SRC + ["explicit"]  # cues with black & white

df = pd.read_parquet(AGG)
df["mean"] = df["sum_val"] / df["n"]

def cell(model, task, seed):
    return df[(df.model == model) & (df.task == task) & (df.seed == seed)]

def dev_vector(d, method, race, neutral_map):
    sub = d[(d.method == method) & (d.race == race)].set_index("prompt_id")["mean"]
    pids = sub.index.intersection(neutral_map.index)
    return (sub.loc[pids] - neutral_map.loc[pids])

records_corr = []   # within-cue cross-source correlations
records_ratio = []  # B/W ratios

for model in MODELS:
    for seed in [0, 1, 2]:
        within_rs = []
        ratios = []
        for task in TASKS:
            d = cell(model, task, seed)
            neutral_map = d[d.method == "neutral"].set_index("prompt_id")["mean"]
            # ---- within-cue cross-source correlations ----
            for race in ["black", "white"]:
                # names: all 3 source pairs
                for a, b in itertools.combinations(NAME_SRC, 2):
                    va = dev_vector(d, a, race, neutral_map)
                    vb = dev_vector(d, b, race, neutral_map)
                    pids = va.index.intersection(vb.index)
                    if len(pids) > 2 and va.loc[pids].std() > 0 and vb.loc[pids].std() > 0:
                        r = np.corrcoef(va.loc[pids], vb.loc[pids])[0, 1]
                        within_rs.append(r)
                        records_corr.append(dict(model=model, seed=seed, task=task,
                                                 family="name", race=race, pair=f"{a}|{b}", r=r))
                # dialog: CAD vs PRISM
                va = dev_vector(d, DIALOG_SRC[0], race, neutral_map)
                vb = dev_vector(d, DIALOG_SRC[1], race, neutral_map)
                pids = va.index.intersection(vb.index)
                if len(pids) > 2 and va.loc[pids].std() > 0 and vb.loc[pids].std() > 0:
                    r = np.corrcoef(va.loc[pids], vb.loc[pids])[0, 1]
                    within_rs.append(r)
                    records_corr.append(dict(model=model, seed=seed, task=task,
                                             family="dialog", race=race, pair="CAD|PRISM", r=r))
            # ---- B/W outcome ratios ----
            def pooled(method, race):
                s = d[(d.method == method) & (d.race == race)]
                return s.sum_val.sum() / s.n.sum() if s.n.sum() > 0 else np.nan
            white_neutral = pooled("neutral", "none")
            for cue in RATIO_CUES:
                rb, rw = pooled(cue, "black"), pooled(cue, "white")
                if rw and not np.isnan(rw):
                    ratios.append(rb / rw)
                    records_ratio.append(dict(model=model, seed=seed, task=task, cue=cue, ratio=rb / rw))
            # dialect vs no-cue (white reference = SAE neutral)
            rb = pooled("dialect", "black")
            if white_neutral and not np.isnan(white_neutral):
                ratios.append(rb / white_neutral)
                records_ratio.append(dict(model=model, seed=seed, task=task, cue="dialect", ratio=rb / white_neutral))

        # store per (model, seed) aggregate
        records_corr.append(dict(model=model, seed=seed, task="ALL", family="ALL",
                                 race="ALL", pair="AGG", r=np.mean(within_rs)))
        records_ratio.append(dict(model=model, seed=seed, task="ALL", cue="AGG", ratio=np.mean(ratios)))

corr_df = pd.DataFrame(records_corr)
ratio_df = pd.DataFrame(records_ratio)

# per-(model,seed) aggregates
agg_corr = corr_df[corr_df.pair == "AGG"].groupby("model")["r"].agg(["mean", "std"])
agg_ratio = ratio_df[ratio_df.cue == "AGG"].groupby("model")["ratio"].agg(["mean", "std"])

print("=== Within-cue cross-source r: mean & SD across seeds (per-seed aggregate) ===")
print(agg_corr.to_string())
print("\n=== B/W outcome ratio: mean & SD across seeds (per-seed aggregate) ===")
print(agg_ratio.to_string())

print("\n=== per-seed within-cue r values ===")
print(corr_df[corr_df.pair == "AGG"][["model", "seed", "r"]].to_string(index=False))
print("\n=== per-seed B/W ratio values ===")
print(ratio_df[ratio_df.cue == "AGG"][["model", "seed", "ratio"]].to_string(index=False))

# also family-level means (names vs dialog) for sanity vs paper (0.99 / 0.92)
print("\n=== family-level mean r (pooled over seeds/tasks/races) ===")
print(corr_df[corr_df.family.isin(["name", "dialog"])].groupby(["model", "family"])["r"].mean().to_string())

corr_df.to_csv(os.path.join(HERE, "task1_corr_detail.csv"), index=False)
ratio_df.to_csv(os.path.join(HERE, "task1_ratio_detail.csv"), index=False)
