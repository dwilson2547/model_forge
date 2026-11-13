"""Multi-corpus perplexity: the forge's tier-1 quality floor.

Same model through several domain-distinct corpora so domain-gaming shows up as
one corpus improving while others regress. Usage:
  python multicorpus.py <label> <model.gguf> [<label> <model.gguf> ...]
Appends rows to multicorpus-results.jsonl.

Paths come from surgery/paths.py (FORGE_ROOT, default ~/forge); overrides:
  FORGE_LLAMA    llama.cpp checkout with build/bin   (default ~/llama.cpp)
  FORGE_DATA     dir holding wiki/fineweb/code .test.txt (default $FORGE_ROOT/data)
  FORGE_RESULTS  results jsonl                       (default $FORGE_ROOT/results/multicorpus.jsonl)
  FORGE_NGL      -ngl for CUDA builds (see evaluate.py)
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surgery import evaluate, paths  # noqa: E402

LLAMA = str(paths.LLAMA)
DATA = paths.DATA
CORPORA = {
    "wikitext": str(DATA / "wiki.test.txt"),
    "fineweb": str(DATA / "fineweb.test.txt"),
    "code": str(DATA / "code.test.txt"),
}
OUT = Path(os.environ.get("FORGE_RESULTS", str(paths.RESULTS / "multicorpus.jsonl")))


def main():
    args = sys.argv[1:]
    if len(args) % 2 != 0:
        raise SystemExit("usage: multicorpus.py <label> <model.gguf> [...]")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(args), 2):
        label, model = args[i], args[i + 1]
        row = {"label": label, "model": model, "ngl": evaluate.NGL or None}
        for corpus, path in CORPORA.items():
            row[corpus] = evaluate.ppl(model, path, LLAMA, chunks=24)["ppl"]
            print(f"{label:24s} {corpus:10s} PPL {row[corpus]:.4f}", flush=True)
        with OUT.open("a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
