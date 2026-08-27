# Different Demographic Cues Yield Inconsistent Conclusions About LLM Personalization and Bias

Replication package for the EMNLP 2026 paper.

> Manuel Tonneau, Neil K. R. Sehgal, Niyati Malhotra, Sharif Kazemi, Victor
> Orozco-Olvera, Ana María Muñoz Boudet, Lakshmi Subramanian, Samuel P.
> Fraiberger, Sharath Chandra Guntuku, Valentin Hofmann. *Different Demographic
> Cues Yield Inconsistent Conclusions About LLM Personalization and Bias.*
> EMNLP 2026.

The paper asks whether the demographic cues commonly used to probe LLMs, that is
explicit identity statements, names, dialog history and dialect, are
interchangeable operationalizations of the same underlying identity-conditioned
behavior. They are not: cue choice changes both the magnitude and the direction
of the estimated effect.

This repository holds the analysis code and the result tables behind every
number, table and figure in the paper. The scripts are **analysis-only**: they
read the released prompt and response data and recompute the results. They do
not run inference or call model APIs, with one documented exception
(`scripts/appendix/task_gpt_variance_*.py`, the repeated-run variance harness
for Appendix B.3).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Runs on the committed result tables alone, no data download needed:
python scripts/appendix/task_verify_c4.py

# Everything else needs the released data (see below):
CUES_DATA_DIR=/path/to/data python scripts/analysis/race_pred_recall.py
```

Tested with Python 3.12. `python -m spacy download en_core_web_sm` is needed
only by `scripts/appendix/task2_features.py`.

Any script whose inputs are missing stops immediately with one line naming the
missing path, before writing anything.

## Data

The data is released separately from this code package (it is far too large for
git):

**https://huggingface.co/datasets/manueltonneau/llm-demographic-cues**

```bash
hf download manueltonneau/llm-demographic-cues --repo-type dataset --local-dir ../data
```

By default the scripts look for a `data/` directory **next to this repository**
(i.e. `../data` relative to the clone). Set `CUES_DATA_DIR` to put it anywhere
else:

```
data/
├── prompts/<task>/<task>_<cue>_constrained.parquet          # prompt text + identity columns
├── decoder_model_responses_race_pred/<task>_<cue>_constrained_<model>_seed_0.csv
├── decoder_model_responses_cleaned/<task>_<cue>_constrained_<model>_seed_<0|1|2>.csv
├── plot_data/                                                # per-cue outcome ratios behind Figures 3 and 11
├── gpt_run/                                                  # GPT-5.2 batch requests (Appendix B.3 harness only)
└── masters/                                                  # pre-built master tables (optional)
    ├── medical_master.parquet
    └── legal_salary_master.parquet
```

with

- `<task>`  ∈ `medical_advice`, `legal_advice`, `salary_rec`
- `<cue>`   ∈ `neutral`, `explicit`, `dialect`, `name_specific_{rosenman,hayes_elder,tzioumis,an}`, `convo_prefix`, `convo_prefix_prism`
- `<model>` ∈ `llama3.1`, `olmo2`, `gpt52` (seeds 1–2 available for `llama3.1` / `olmo2`)

If the data lives elsewhere, point the scripts at it with environment variables;
no code edits are needed.

| Variable | Default | Used for |
|----------|---------|----------|
| `CUES_DATA_DIR` | `../data` | The data directory above. |
| `CUES_MASTER_DIR` | `<CUES_DATA_DIR>/masters` | The two pre-built `*_master.parquet` files. |
| `CUES_ROOT` | parent of the clone | Only used to derive the default `CUES_DATA_DIR`. |
| `CUES_FONT_DIR` | unset | Optional directory of CMU Serif `.ttf` files, so `scripts/appendix/make_recall_fig.py` matches the paper's typeface. |

The `masters/` files are required only by `scripts/analysis/replicate_paper_specs.py`,
`scripts/analysis/replicate_regressions.py` and `scripts/analysis/replicate_dialect_only.py`, and are **not part of
the data release** (they belong to an earlier pipeline). Rebuild them from the
released files with:

```bash
python scripts/build/build_asl_cache.py # once, needs data/prompts
python scripts/build/build_masters.py   # -> $CUES_MASTER_DIR/{medical,legal_salary}_master.parquet
```

The other scripts read the per-`(task, cue, model)` response files directly.

## Layout

```
cues/paths.py     every path this package reads or writes, in one place
scripts/
  build/          caches and master tables (run these first)
  analysis/       the analyses behind the paper's tables
  appendix/       appendix analyses and figures
  tables/         regenerates the paper's LaTeX tables
results/
  regressions_llama/       Tables reg_medical / reg_legal / reg_salary
  regressions_all_models/  Tables reg_gpt52 / reg_olmo2
  regressions_nested/      the eight nested specifications (appendix)
  dialect_llama/           dialect-only regression, LLaMA-3.1
  dialect_all_models/      dialect-only regression, all three models
  race_inference/          recall / precision / F1 and confusion matrices
  appendix/                appendix deliverables and bundled aggregates
figures/
```

| Script | What it does | Writes to |
|--------|--------------|-----------|
| `scripts/build/build_fk_cache.py` | Flesch–Kincaid grade per prompt | `fk_cache/` |
| `scripts/build/build_asl_cache.py` | Average sentence length per prompt | `asl_cache/` |
| `scripts/build/build_master_and_regress.py` | Per-(model, task) master tables and cross-model regressions | `results/regressions_all_models/` |
| `scripts/build/build_masters.py` | The two master tables the `replicate_*` scripts read | `$CUES_MASTER_DIR` |
| `scripts/analysis/replicate_paper_specs.py` | Three main regression tables, LLaMA-3.1 | `results/regressions_llama/` |
| `scripts/analysis/replicate_regressions.py` | Eight nested OLS specifications | `results/regressions_nested/` |
| `scripts/analysis/replicate_dialect_only.py` | Dialect (AAVE) subset regression, LLaMA-3.1 | `results/dialect_llama/` |
| `scripts/analysis/replicate_dialect_only_allmodels.py` | Same, all three models | `results/dialect_all_models/` |
| `scripts/analysis/race_pred_recall.py` | Per-class precision/recall/F1 for inferred race | `results/race_inference/` |
| `scripts/analysis/neutral_default_race.py` | Default inferred race on unmarked prompts | `results/race_inference/` |
| `scripts/analysis/olmo_pred_dist.py` | OLMo-2 inferred-race distribution by cue | `results/race_inference/` |
| `scripts/appendix/` | Seed variance, sign flips, extended controls, salary tiers, AAVE audit, figures. See [its README](results/appendix/README.md). | `results/appendix/` |
| `scripts/tables/make_paper_tables.py` | Rewrites the paper's five LLaMA tables from the CSVs | the paper's `tables/` |

Scripts run from any working directory; paths come from `cues/paths.py`, not from
where the script happens to sit.

## Running

`scripts/build/build_fk_cache.py` is a prerequisite for `scripts/build/build_master_and_regress.py` and
`scripts/analysis/replicate_dialect_only_allmodels.py`; otherwise the top-level scripts are
independent. A full run:

```bash
python scripts/build/build_fk_cache.py                       # -> fk_cache/   (once)
python scripts/build/build_asl_cache.py                      # -> asl_cache/  (once)
python scripts/build/build_master_and_regress.py llama3.1 olmo2 gpt52
python scripts/build/build_masters.py                        # -> $CUES_MASTER_DIR
python scripts/analysis/replicate_paper_specs.py
python scripts/analysis/replicate_regressions.py             # args: medical legal salary
python scripts/analysis/replicate_dialect_only.py
python scripts/analysis/replicate_dialect_only_allmodels.py
python scripts/analysis/race_pred_recall.py
python scripts/analysis/neutral_default_race.py
python scripts/analysis/olmo_pred_dist.py
```

`scripts/build/build_master_and_regress.py` and `scripts/analysis/replicate_regressions.py` accept optional
command-line arguments to restrict the run (models and tasks respectively); with
no arguments they use the defaults shown above.

## Citation

The paper is on arXiv at **https://arxiv.org/abs/2601.18486**; cite that until
the ACL Anthology entry exists.

```bibtex
@article{tonneau2026cues,
  title     = {Different Demographic Cues Yield Inconsistent Conclusions About {LLM} Personalization and Bias},
  author    = {Tonneau, Manuel and Sehgal, Neil K. R. and Malhotra, Niyati and
               Kazemi, Sharif and Orozco-Olvera, Victor and Mu{\~n}oz Boudet, Ana Mar{\'i}a and
               Subramanian, Lakshmi and Fraiberger, Samuel P. and
               Guntuku, Sharath Chandra and Hofmann, Valentin},
  journal   = {arXiv preprint arXiv:2601.18486},
  note      = {To appear at EMNLP 2026},
  year      = {2026}
}
```

## License

Code is released under the MIT License (see [LICENSE](LICENSE)).
