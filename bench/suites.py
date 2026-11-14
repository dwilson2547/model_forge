"""Board v2 item sets (TODO M2 "D2 tier"): public, pinned, model-agnostic.

The v1 board (bench/tasks.py, 12 items) saturated: base and every decent repair tie at
0.867, so it can no longer rank repairs. v2 scores hundreds of items, each a pure function
of the response (no LLM judge), across five tiers:

  humaneval  164 coding problems, unit tests run in the docker sandbox     (MIT)
  mbpp       100 of the 257 sanitized-test problems, same sandbox          (CC-BY-4.0)
  gsm8k      200 of the 1319 test word problems, exact final number        (MIT)
  ifeval     200 of the 541 verifiable-instruction prompts, Google checker (Apache-2.0)
  needle     30 generated long-context lookups: 6 keyed facts hidden in real fineweb
             text at 8k/16k/24k tokens, 2 asked per item, with near-miss distractors

These are public benchmarks, so the base model has likely seen them in training: absolute
scores flatter it. The board is for COMPARING a slice against its own base on the same
surface, where contamination cancels.

  python -m bench.suites fetch      # downloads the pinned revisions into $FORGE_DATA/bench
"""
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surgery import paths  # noqa: E402

DIR = paths.DATA / "bench"
SEED = 20261113  # fixes every subset and every generated needle; change = a new board

SOURCES = {  # name: (repo, revision, file)
    "humaneval": ("openai/openai_humaneval", "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
                  "openai_humaneval/test-00000-of-00001.parquet"),
    "mbpp": ("google-research-datasets/mbpp", "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
             "sanitized/test-00000-of-00001.parquet"),
    "gsm8k": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866",
              "main/test-00000-of-00001.parquet"),
    "ifeval": ("google/IFEval", "966cd89545d6b6acfd7638bc708b98261ca58e84",
               "ifeval_input_data.jsonl"),
}
SUBSET = {"humaneval": None, "mbpp": 100, "gsm8k": 200, "ifeval": 200}


def fetch():
    """Download each pinned file and write $FORGE_DATA/bench/<name>.jsonl (all rows)."""
    from huggingface_hub import hf_hub_download
    import pandas as pd
    DIR.mkdir(parents=True, exist_ok=True)
    for name, (repo, rev, fn) in SOURCES.items():
        p = hf_hub_download(repo, fn, repo_type="dataset", revision=rev)
        rows = (pd.read_parquet(p).to_dict("records") if fn.endswith(".parquet")
                else [json.loads(l) for l in open(p)])
        with open(DIR / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, default=list) + "\n")
        print(f"{name:10s} {len(rows):5d} rows  {repo}@{rev[:10]}")


def _rows(name):
    p = DIR / f"{name}.jsonl"
    if not p.exists():
        raise SystemExit(f"{p} missing: run `python -m bench.suites fetch` first")
    rows = [json.loads(l) for l in open(p)]
    n = SUBSET.get(name)
    if n:
        rows = random.Random(f"{SEED}-{name}").sample(rows, n)
    return rows


CODE_ASK = ("\n\nRespond with the complete Python solution (including any imports it needs) "
            "in a single ```python code block.")


def humaneval():
    out = []
    for r in _rows("humaneval"):
        out.append({"id": r["task_id"], "tier": "humaneval",
                    "prompt": "Complete this Python function.\n\n```python\n" + r["prompt"]
                              + "```" + CODE_ASK,
                    "max_tokens": 4096,
                    "header": "\n".join(l for l in r["prompt"].splitlines()
                                        if l.startswith(("import ", "from "))),
                    "tests": r["test"] + f"\n\ncheck({r['entry_point']})\n"})
    return out


def mbpp():
    out = []
    for r in _rows("mbpp"):
        tests = list(r["test_list"])
        out.append({"id": f"mbpp/{r['task_id']}", "tier": "mbpp",
                    "prompt": r["prompt"] + "\nYour code should pass this test:\n" + tests[0]
                              + CODE_ASK,
                    "max_tokens": 4096,
                    "header": "\n".join(r.get("test_imports") or []),
                    "tests": "\n".join(tests) + "\n"})
    return out


def gsm8k():
    out = []
    for i, r in enumerate(_rows("gsm8k")):
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        out.append({"id": f"gsm8k/{i}", "tier": "gsm8k",
                    "prompt": r["question"] + "\n\nSolve this step by step. End with a final "
                              "line of the form 'Answer: <number>'.",
                    "max_tokens": 4096, "gold": gold})
    return out


def ifeval():
    return [{"id": f"ifeval/{r['key']}", "tier": "ifeval", "prompt": r["prompt"],
             "max_tokens": 4096, "instruction_id_list": r["instruction_id_list"],
             "kwargs": r["kwargs"]} for r in _rows("ifeval")]


NAMES = ["Harrow", "Lindqvist", "Okafor", "Brennan", "Tamura", "Castell", "Volkov", "Achebe",
         "Moreau", "Sandoval", "Kowalski", "Iyer", "Fairbanks", "Nakamura", "Delacroix",
         "Abernathy", "Ruiz", "Thorsen", "Mbeki", "Grantham"]


def needle():
    """Keyed multi-needle retrieval in real text. Each item hides 6 'locker code' facts for
    6 different people at random depths, plus 2 decoys (same person, a 'former' code), and
    asks for two people's current codes. Score = fraction of the two answered right."""
    text = (paths.DATA / "fineweb.test.txt").read_text()
    rng = random.Random(f"{SEED}-needle")
    out = []
    for length in (8000, 16000, 24000):          # approx tokens (4 chars/token)
        for k in range(10):
            start = rng.randrange(0, len(text) - length * 4)
            hay = text[start:start + length * 4]
            people = rng.sample(NAMES, 6)
            codes = {p: str(rng.randrange(10000, 99999)) for p in people}
            facts = [f" The current locker code for {p} is {c}. " for p, c in codes.items()]
            for p in rng.sample(people, 2):
                facts.append(f" The former locker code for {p} was "
                             f"{rng.randrange(10000, 99999)}; it is no longer valid. ")
            cuts = sorted(rng.randrange(0, len(hay)) for _ in facts)
            rng.shuffle(facts)
            parts, prev = [], 0
            for c, f in zip(cuts, facts):
                parts += [hay[prev:c], f]
                prev = c
            parts.append(hay[prev:])
            ask = rng.sample(people, 2)
            out.append({"id": f"needle/{length // 1000}k-{k}", "tier": "needle",
                        "prompt": "".join(parts) + "\n\nFrom the text above: what are the "
                                  f"CURRENT locker codes for {ask[0]} and {ask[1]}? Answer "
                                  f"with two lines, '{ask[0]}: <code>' and '{ask[1]}: <code>'.",
                        "max_tokens": 4096, "gold": {p: codes[p] for p in ask},
                        "ctx_tokens": length})
    return out


TIERS = {"humaneval": humaneval, "mbpp": mbpp, "gsm8k": gsm8k, "ifeval": ifeval,
         "needle": needle}


def items(tiers=None):
    return [it for t in (tiers or TIERS) for it in TIERS[t]()]


if __name__ == "__main__":
    if sys.argv[1:] == ["fetch"]:
        fetch()
    else:
        raise SystemExit(__doc__)
