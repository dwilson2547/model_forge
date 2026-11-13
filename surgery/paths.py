"""Where the forge keeps files on a machine. Layout and the cold-archive rules: docs/storage.md.

One root per machine, `FORGE_ROOT` (default ~/forge), the same tree on the Altra and the desktop:
  base/<hf-org>/<repo>/                  upstream weights in use
  data/                                  corpora (wiki/fineweb/code .test.txt, mixed.train.txt, ...)
  runs/<family>/<experiment>/<arm>/      outputs + that arm's logs
  results/                               the ledgers (multicorpus.jsonl, bench.jsonl, longwindow.jsonl)

Importing this sets FORGE_ROOT in the environment if it is unset, so `${FORGE_ROOT}` in a
recipe and in a child process resolves the same way.
"""

import os
from pathlib import Path

ROOT = Path(os.environ.setdefault("FORGE_ROOT", str(Path.home() / "forge"))).expanduser()
BASE = ROOT / "base"
DATA = Path(os.environ.get("FORGE_DATA", str(ROOT / "data")))
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
LLAMA = Path(os.environ.get("FORGE_LLAMA", str(Path.home() / "llama.cpp")))


def expand(value):
    """${FORGE_ROOT}/~ expansion for recipe strings, recursively through lists and dicts."""
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [expand(v) for v in value]
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    return value
