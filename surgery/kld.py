"""KL divergence vs the base: the floor row domain adaptation can't game (TODO M2).

Perplexity rewards predicting a corpus well; a model can beat the base on wikitext while
behaving nothing like it (wikirep, design §0c). KLD measures how far each candidate's
next-token distribution moved from the base's, on the same three corpora and chunking as
multicorpus.py, so the two ledgers line up row for row.

  python -m surgery.kld ref <base.gguf>                  # once per base: writes the reference logits
  python -m surgery.kld score <label> <model.gguf> [...]  # appends rows to results/kld.jsonl

Reference files: $FORGE_KLD_REF (default $FORGE_ROOT/runs/<family>/kld/ref-<corpus>.kld),
~3 GB each for Qwen3.5-4B at 24 chunks x 512 ctx (248k vocab). Delete them when done; they
rebuild in ~15 s on a GPU. Env as multicorpus.py (FORGE_LLAMA, FORGE_NGL, FORGE_DATA).
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surgery import evaluate, multicorpus, paths  # noqa: E402

REF = Path(os.environ.get("FORGE_KLD_REF", str(paths.ROOT / "runs/qwen35-4b/kld")))
OUT = Path(os.environ.get("FORGE_KLD_RESULTS", str(paths.RESULTS / "kld.jsonl")))
CHUNKS = 24  # must match multicorpus.py


def _run(args):
    cmd = [evaluate._bin(str(paths.LLAMA), "llama-perplexity"), "--chunks", str(CHUNKS),
           "-t", str(evaluate.THREADS)] + (["-ngl", evaluate.NGL] if evaluate.NGL else []) + args
    p = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True)
    text = p.stdout + p.stderr
    if p.returncode:
        raise RuntimeError(f"llama-perplexity rc={p.returncode}:\n{text[-800:]}")
    return text


def _num(text, pattern):
    m = re.search(pattern, text)
    if not m:
        raise RuntimeError(f"{pattern!r} not in output:\n{text[-800:]}")
    return float(m.group(1))


def ref(base):
    REF.mkdir(parents=True, exist_ok=True)
    for corpus, path in multicorpus.CORPORA.items():
        text = _run(["-m", base, "-f", path, "--kl-divergence-base", str(REF / f"ref-{corpus}.kld")])
        print(f"ref {corpus:10s} base PPL {_num(text, r'Final estimate: PPL = ([0-9.]+)'):.4f}", flush=True)


def score(label, model):
    row = {"label": label, "model": model, "ngl": evaluate.NGL or None}
    for corpus in multicorpus.CORPORA:
        text = _run(["-m", model, "--kl-divergence-base", str(REF / f"ref-{corpus}.kld"),
                     "--kl-divergence"])
        row[corpus] = {
            "ppl": _num(text, r"Mean PPL\(Q\)\s*:\s*([0-9.]+)"),
            "kld_mean": _num(text, r"Mean\s+KLD:\s*([0-9.]+)"),
            "kld_p99": _num(text, r"99\.0%\s+KLD:\s*([0-9.]+)"),
            "kld_median": _num(text, r"Median\s+KLD:\s*([0-9.]+)"),
            "same_top": _num(text, r"Same top p:\s*([0-9.]+)"),
        }
        c = row[corpus]
        print(f"{label:24s} {corpus:10s} PPL {c['ppl']:.4f}  KLD {c['kld_mean']:.4f}  "
              f"p99 {c['kld_p99']:.3f}  top1 {c['same_top']:.2f}%", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as f:
        f.write(json.dumps(row) + "\n")


def main():
    a = sys.argv[1:]
    if a[:1] == ["ref"] and len(a) == 2:
        ref(a[1])
    elif a[:1] == ["score"] and len(a) >= 3 and len(a) % 2 == 1:
        for i in range(1, len(a), 2):
            score(a[i], a[i + 1])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
