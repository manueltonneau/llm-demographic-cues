"""Single source of truth for every path this package reads or writes.

Scripts import the directory they need from here instead of deriving it from
their own location, so moving a script between folders cannot silently change
where it looks for data or writes results.

Layout::

    <repo>/
      cues/           this package
      scripts/        build/  analysis/  appendix/  tables/
      results/        regressions_llama/  regressions_all_models/  ...
      figures/
    <repo>/../data/   the released data (override with CUES_DATA_DIR)

Environment overrides: ``CUES_ROOT``, ``CUES_DATA_DIR``, ``CUES_MASTER_DIR``.
"""
import os
import sys

REPO_ROOT = os.environ.get(
    "CUES_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

# The released data sits next to the clone by default; see the README.
DATA_DIR   = os.environ.get("CUES_DATA_DIR", os.path.join(os.path.dirname(REPO_ROOT), "data"))
MASTER_DIR = os.environ.get("CUES_MASTER_DIR", os.path.join(DATA_DIR, "masters"))

PROMPTS   = os.path.join(DATA_DIR, "prompts")
RESPONSES = os.path.join(DATA_DIR, "decoder_model_responses_cleaned")
RACE_PRED = os.path.join(DATA_DIR, "decoder_model_responses_race_pred")
PLOT_DATA = os.path.join(DATA_DIR, "plot_data")
GPT_RUN   = os.path.join(DATA_DIR, "gpt_run")

RESULTS = os.path.join(REPO_ROOT, "results")
FIGURES = os.path.join(REPO_ROOT, "figures")

REGRESSIONS_LLAMA      = os.path.join(RESULTS, "regressions_llama")
REGRESSIONS_ALL_MODELS = os.path.join(RESULTS, "regressions_all_models")
REGRESSIONS_NESTED     = os.path.join(RESULTS, "regressions_nested")
DIALECT_LLAMA          = os.path.join(RESULTS, "dialect_llama")
DIALECT_ALL_MODELS     = os.path.join(RESULTS, "dialect_all_models")
RACE_INFERENCE         = os.path.join(RESULTS, "race_inference")
APPENDIX               = os.path.join(RESULTS, "appendix")


def _cache_dir(name):
    """Caches live in the clone if built here, otherwise in the data release."""
    local = os.path.join(REPO_ROOT, name)
    return local if os.path.isdir(local) else os.path.join(DATA_DIR, name)


FK_CACHE  = _cache_dir("fk_cache")
ASL_CACHE = _cache_dir("asl_cache")

# Where the build scripts *write* their caches: always inside the clone, never
# the (possibly read-only) data release the read side may fall back to.
FK_CACHE_OUT  = os.path.join(REPO_ROOT, "fk_cache")
ASL_CACHE_OUT = os.path.join(REPO_ROOT, "asl_cache")

HINT = ("Set CUES_DATA_DIR to the released data directory, or see the Data "
        "section of the README for the expected layout.")


def fail(msg):
    sys.exit(f"[error] {msg}\n        {HINT}")


def require_dir(path, what="input directory"):
    if not os.path.isdir(path):
        fail(f"missing {what}: {path}")
    return path


def require_file(path, what="input file"):
    if not os.path.isfile(path):
        fail(f"missing {what}: {path}")
    return path


def require_any(paths, what="input files"):
    """At least one of `paths` must exist. Returns the ones that do."""
    paths = list(paths)
    found = [p for p in paths if os.path.exists(p)]
    if not found:
        shown = "\n          ".join(str(p) for p in paths[:3])
        more = f"\n          ... ({len(paths) - 3} more)" if len(paths) > 3 else ""
        fail(f"none of the expected {what} exist, e.g.:\n          {shown}{more}")
    return found


def require_produced(n, what="outputs"):
    """Guard for build scripts: refuse to exit 0 having produced nothing."""
    if not n:
        fail(f"produced no {what} -- every input was missing or skipped")
