"""Shared input-location helpers.

Every script in this package reads inputs that ship separately from the code
(see the Data section of the README). These helpers turn a missing input into
one actionable line instead of a traceback, and they are called *before*
anything is written, so a run without the data cannot overwrite the committed
results with empty files.
"""
import os
import sys

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
