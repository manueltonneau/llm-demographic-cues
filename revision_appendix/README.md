# Appendix analyses

Scripts behind the paper's appendix analyses. Each resolves its own paths from
`dirname(__file__)`, so it runs from any working directory as long as it stays
inside the repository (it imports `cues_io.py` from the repo root):

```bash
python revision_appendix/<script>.py
```

Inputs come from `data/plot_data/`, `results_all_models/` (build it first, see
the top-level README) and the bundled aggregates `per_seed_agg.parquet` /
`gpt52_salary_agg.parquet` / `task2_features_*.parquet`. The aggregates are
committed, so the analyses below run without the multi-GB raw response files;
rebuild them with `agg_per_seed.py`, `agg_gpt52_salary.py` and
`task2_features.py` if the raw responses change.

| Script | What it does | Outputs |
|--------|--------------|---------|
| `task_sign_flip.py` | Sign-flip / conclusion-instability analysis: does a single-cue disparity conclusion survive a different cue? | `task_sign_flip_{per_cue,per_cell}.csv`, `task_sign_flip_table.tex`, `task_sign_flip_summary.txt` |
| `task_verify_c4.py` | Re-derives the Appendix C.4 mechanism numbers from the source CSVs. | `task_verify_c4_{regression,recall}.csv`, `task_verify_c4_summary.txt` |
| `task1_seed_variance.py` | Seed-level variance of correlations and outcome ratios. | `task1_{corr,ratio}_detail.csv` |
| `task_corr_robustness.py` | Within-cue correlation robustness (Pearson / Spearman / cosine). | prints a table |
| `task2_features.py` → `task2_regress.py` | Extended linguistic controls beyond Flesch–Kincaid. | `task2_features_*.parquet`, `task2_extended_controls.csv` |
| `task_feature_profile.py` | Per-cue linguistic feature profile. | prints LaTeX |
| `task3_salary_tiers.py` | Black/White salary ratios by job tier. | `task3_salary_tiers.csv`, `../figures/salary_tiers.pdf` |
| `task_dialect_only_table.py` | Formats the dialect-only regression into a compact table. | `task_dialect_only_table.tex`, `task_dialect_only_summary.txt` |
| `task_aave_audit.py` | Structure and content-preservation audit of the SAE→AAVE prompt pairs. | `task_aave_audit_summary.txt`, plus per-pair CSVs (not committed, ~10 MB) |
| `task_gpt_variance_build.py`, `task_gpt_variance_submit.py` | GPT-5.2 repeated-run variance harness (Appendix B.3). **The only scripts here that call a model API**; `build` prices and prepares the batch, `submit` sends it. | `gpt_variance_sd_by_condition.csv`, `gpt_variance_summary.txt` |
| `make_pipeline_fig.py` | Pipeline schematic. | `../figures/pipeline.pdf` |
| `make_recall_fig.py` | Inferred-race recall by cue (paper Figure 4). Set `CUES_FONT_DIR` to a CMU Serif directory to match the paper's typeface. | `../figures/recall_by_cue.pdf` |

`task2_features.py` additionally needs `spacy` (`en_core_web_sm`) and
`vaderSentiment`.
