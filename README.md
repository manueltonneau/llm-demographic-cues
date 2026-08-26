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
(`revision_appendix/task_gpt_variance_*.py`, the repeated-run variance harness
for Appendix B.3).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Runs on the committed result tables alone, no data download needed:
python revision_appendix/task_verify_c4.py

# Everything else needs the released data (see below):
CUES_DATA_DIR=/path/to/data python race_pred_recall.py
```

Tested with Python 3.12. `python -m spacy download en_core_web_sm` is needed
only by `revision_appendix/task2_features.py`.

Any script whose inputs are missing stops immediately with one line naming the
missing path, before writing anything.

## Data

The data is released separately from this code package (it is far too large for
git). **TODO: add the data URL here.**

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
| `CUES_FONT_DIR` | unset | Optional directory of CMU Serif `.ttf` files, so `revision_appendix/make_recall_fig.py` matches the paper's typeface. |

The `masters/` files are required only by `replicate_paper_specs.py`,
`replicate_regressions.py` and `replicate_dialect_only.py`, and are **not part of
the data release** (they belong to an earlier pipeline). Rebuild them from the
released files with:

```bash
python build_asl_cache.py     # once, needs data/prompts
python build_masters.py       # -> $CUES_MASTER_DIR/{medical,legal_salary}_master.parquet
```

The other scripts read the per-`(task, cue, model)` response files directly.

## Contents

| Path | What it is |
|------|------------|
| `requirements.txt` | Python dependencies (pinned). |
| `cues_io.py` | Shared input-location helpers: every script checks its inputs up front and stops with one actionable line if they are missing. |
| `build_fk_cache.py` | Precomputes Flesch–Kincaid grade per prompt → `fk_cache/`. Run first. |
| `build_asl_cache.py` | Precomputes average sentence length per prompt → `asl_cache/`. Needed by `build_masters.py`. |
| `build_masters.py` | Rebuilds the two master tables the three `replicate_*` scripts read → `<CUES_MASTER_DIR>/`. |
| `build_master_and_regress.py` | Builds per-(model, task) master tables and the cross-model regressions → `results_all_models/`. |
| `replicate_paper_specs.py` | Three main regression tables for LLaMA-3.1 → `results_paper/`. |
| `replicate_regressions.py` | Eight nested OLS specifications (appendix) → `results/`. |
| `replicate_dialect_only.py` | Dialect (AAVE) subset regression, LLaMA-3.1 → `results_dialect/`. |
| `replicate_dialect_only_allmodels.py` | Same, extended to all three models → `results_dialect_allmodels/`. |
| `race_pred_recall.py` | Per-class precision/recall/F1 for inferred race → `results_recall/`. |
| `neutral_default_race.py` | Default inferred-race distribution on unmarked prompts → `results_recall/`. |
| `olmo_pred_dist.py` | OLMo-2 inferred-race distribution by cue → `results_recall/`. |
| `revision_appendix/` | Appendix analyses: seed variance, sign flips, extended linguistic controls, salary tiers, AAVE corpus audit, figures. See [its README](revision_appendix/README.md). |
| `results*/`, `figures/` | Committed outputs. The `*.csv` files are the source numbers for the corresponding paper tables; the LaTeX tables are synced from them by hand. |

Every script writes its outputs next to itself, creating the output directory if
needed.

### Not committed here

Two large generated artifacts are reproducible, so they ship with the data
release rather than with this repository:

- `fk_cache/` (~25 MB) — rebuild with `python build_fk_cache.py`
- `asl_cache/` (~25 MB) — rebuild with `python build_asl_cache.py`
- `results_all_models/**/master.parquet` (~38 MB) — rebuild with
  `python build_master_and_regress.py llama3.1 olmo2 gpt52`

The regression CSVs under `results_all_models/` *are* committed, so
`revision_appendix/task_verify_c4.py` runs out of the box.
`revision_appendix/task2_regress.py` needs the master tables and skips any
(model, task) whose master is absent.

## Running

`build_fk_cache.py` is a prerequisite for `build_master_and_regress.py` and
`replicate_dialect_only_allmodels.py`; otherwise the top-level scripts are
independent. A full run:

```bash
python build_fk_cache.py                                   # → fk_cache/  (run once)
python build_asl_cache.py                                  # → asl_cache/ (run once)
python build_masters.py                                    # → masters/   (for the replicate_* scripts)
python build_master_and_regress.py llama3.1 olmo2 gpt52    # → results_all_models/
python replicate_paper_specs.py                            # → results_paper/
python replicate_regressions.py                            # → results/        (args: medical legal salary)
python replicate_dialect_only.py                           # → results_dialect/
python replicate_dialect_only_allmodels.py                 # → results_dialect_allmodels/
python race_pred_recall.py                                 # → results_recall/
python neutral_default_race.py                             # → results_recall/
python olmo_pred_dist.py                                   # → results_recall/
```

`build_master_and_regress.py` and `replicate_regressions.py` accept optional
command-line arguments to restrict the run (models and tasks respectively); with
no arguments they use the defaults shown above.

## Citation

```bibtex
@inproceedings{tonneau2026cues,
  title     = {Different Demographic Cues Yield Inconsistent Conclusions About {LLM} Personalization and Bias},
  author    = {Tonneau, Manuel and Sehgal, Neil K. R. and Malhotra, Niyati and
               Kazemi, Sharif and Orozco-Olvera, Victor and Mu{\~n}oz Boudet, Ana Mar{\'i}a and
               Subramanian, Lakshmi and Fraiberger, Samuel P. and
               Guntuku, Sharath Chandra and Hofmann, Valentin},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026}
}
```

## License

Code is released under the MIT License (see [LICENSE](LICENSE)).
