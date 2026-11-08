"""Multi-corpus perplexity: the forge's tier-1 quality floor.

Same model through several domain-distinct corpora so domain-gaming shows up as
one corpus improving while others regress. Usage:
  python multicorpus.py <label> <model.gguf> [<label> <model.gguf> ...]
Appends rows to results.jsonl in the output directory.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/daniel/model-forge")
from surgery import evaluate  # noqa: E402

LLAMA = "/home/daniel/llama.cpp"
CORPORA = {
    "wikitext": "/home/daniel/m0/data/wiki.test.txt",
    "fineweb": "/home/daniel/m0/data/fineweb.test.txt",
    "code": "/home/daniel/m0/data/code.test.txt",
}
OUT = Path("/home/daniel/m0/runs/multicorpus-results.jsonl")


def main():
    args = sys.argv[1:]
    if len(args) % 2 != 0:
        raise SystemExit("usage: multicorpus.py <label> <model.gguf> [...]")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(args), 2):
        label, model = args[i], args[i + 1]
        row = {"label": label, "model": model}
        for corpus, path in CORPORA.items():
            row[corpus] = evaluate.ppl(model, path, LLAMA, chunks=24)["ppl"]
            print(f"{label:24s} {corpus:10s} PPL {row[corpus]:.4f}", flush=True)
        with OUT.open("a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
