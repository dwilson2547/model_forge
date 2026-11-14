"""Prompt mixes for teacher datasets (see distill/__init__.py).

Sources are public, pinned, permissively licensed, and are train splits disjoint from the
board v2 test items:

  chat     UltraChat 200k train_sft, first user turn                       (MIT)
  ifollow  Tulu 3 persona instruction-following, IFEval-style constraints  (ODC-BY)
  code     StarCoder2 self-OSS-instruct exec-filtered, Python tasks        (ODC-BY)
  math     GSM8K train (the board uses test)                               (MIT)
  longctx  generated: keyed facts hidden in fineweb.TRAIN text, 2k-6k tok  (the board's
           needles use fineweb.test and a different fact template)

A mix is {kind: count}; the seed fixes which rows are drawn, so a mix name is a dataset.
"""
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surgery import paths  # noqa: E402

DIR = paths.DATA / "distill"
SRC = DIR / "src"
SEED = 20261113

SOURCES = {  # name: (repo, revision, file)
    "chat": ("HuggingFaceH4/ultrachat_200k", "8049631c405ae6576f93f445c6b8166f76f5505a",
             "data/train_sft-00000-of-00003-*.parquet"),
    "ifollow": ("allenai/tulu-3-sft-personas-instruction-following",
                "fe0c7d350c9b4542b8d829a6f1daa1c259f0ba0e", "data/train-00000-of-00001.parquet"),
    "code": ("bigcode/self-oss-instruct-sc2-exec-filter-50k",
             "356bb069eee815daa6e23e9a282eeefe1490ad44", "data/train-00000-of-00001.parquet"),
    "math": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866",
             "main/train-00000-of-00001.parquet"),
}

MIXES = {
    # ~4k prompts, weighted toward what a slice loses first (instructions, chat format)
    "default": {"chat": 1600, "ifollow": 1000, "code": 700, "math": 500, "longctx": 200},
    "smoke": {"chat": 8, "ifollow": 8, "code": 8, "math": 8, "longctx": 4},
}
MAX_TOKENS = {"chat": 4096, "ifollow": 4096, "code": 4096, "math": 4096, "longctx": 4096}
MAX_PROMPT_CHARS = 6000  # short-form kinds; longctx is built long on purpose


def fetch():
    from huggingface_hub import HfApi, hf_hub_download
    import pandas as pd
    SRC.mkdir(parents=True, exist_ok=True)
    for name, (repo, rev, fn) in SOURCES.items():
        if name == "chat":  # the shard name carries a hash: resolve it at the pinned revision
            fn = next(s.rfilename for s in HfApi().dataset_info(repo, revision=rev).siblings
                      if s.rfilename.startswith("data/train_sft-00000-of-"))
        p = hf_hub_download(repo, fn, repo_type="dataset", revision=rev)
        rows = pd.read_parquet(p).to_dict("records")
        with open(SRC / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, default=list) + "\n")
        print(f"{name:8s} {len(rows):6d} rows  {repo}@{rev[:10]}  {fn}")


# ---------------------------------------------------------------- decontamination

def _grams(text, n=13):
    w = re.findall(r"\w+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def board_grams():
    """13-word runs of every board v2 prompt except the needle filler (fineweb.test, disjoint
    from fineweb.train by construction). Our own instruction suffixes are excluded so they
    don't match themselves."""
    from bench import suites
    g = set()
    for it in suites.items([t for t in suites.TIERS if t != "needle"]):
        body = it["prompt"].replace(suites.CODE_ASK, "")
        g |= _grams(body)
    return g


# ---------------------------------------------------------------- per-kind prompts

def _src(name):
    p = SRC / f"{name}.jsonl"
    if not p.exists():
        raise SystemExit(f"{p} missing: run `python -m distill.prompts fetch` first")
    return [json.loads(l) for l in open(p)]


def chat(rows):
    for r in rows:
        u = next((m["content"] for m in r["messages"] if m["role"] == "user"), r["prompt"])
        yield f"chat/{r['prompt_id'][:16]}", u


def ifollow(rows):
    for r in rows:
        yield f"ifollow/{r['id']}", r["prompt"]


def code(rows):
    for r in rows:
        yield f"code/{r['id']}", r["instruction"]


MATH_ASK = ["Solve this step by step.", "Show your working, then give the final answer.",
            "Work it out and state the final number clearly.", ""]


def math(rows):
    for i, r in enumerate(rows):
        ask = MATH_ASK[i % len(MATH_ASK)]
        yield f"math/{i}", r["question"] + (f"\n\n{ask}" if ask else "")


CITIES = ["Aberdeen", "Bergamo", "Calgary", "Durban", "Esbjerg", "Fukuoka", "Gdansk", "Hobart",
          "Izmir", "Jaipur", "Kraków", "Lyon", "Malmö", "Nantes", "Oaxaca", "Porto", "Quito",
          "Recife", "Sapporo", "Tampere", "Utrecht", "Valencia", "Winnipeg", "Zagreb"]


def longctx(n, rng):
    """Keyed facts ('the shipment for <city> left from dock <n>') at random depths in real
    fineweb train text, with a superseded-fact decoy; asks for 1-3 of them."""
    text = (paths.DATA / "fineweb.train.txt").read_text()
    for k in range(n):
        tokens = rng.choice([2000, 3000, 4000, 5000, 6000])
        start = rng.randrange(0, len(text) - tokens * 4)
        hay = text[start:start + tokens * 4]
        cities = rng.sample(CITIES, 5)
        docks = {c: rng.randrange(100, 999) for c in cities}
        facts = [f" The shipment for {c} left from dock {d}. " for c, d in docks.items()]
        decoy = rng.choice(cities)
        facts.append(f" An earlier schedule listed dock {rng.randrange(100, 999)} for the "
                     f"{decoy} shipment, but that was changed. ")
        cuts = sorted(rng.randrange(0, len(hay)) for _ in facts)
        rng.shuffle(facts)
        parts, prev = [], 0
        for c, f in zip(cuts, facts):
            parts += [hay[prev:c], f]
            prev = c
        parts.append(hay[prev:])
        ask = rng.sample(cities, rng.randint(1, 3))
        q = ("Which dock did the " + ", ".join(ask[:-1]) + (" and " if len(ask) > 1 else "")
             + ask[-1] + " shipment" + ("s" if len(ask) > 1 else "") + " leave from?")
        yield (f"longctx/{k}", "".join(parts) + "\n\n" + q,
               {c: str(docks[c]) for c in ask})


# ---------------------------------------------------------------- build

def build(mix="default"):
    want = MIXES[mix]
    bad = board_grams()
    out, dropped = [], {}
    for kind, n in want.items():
        rng = random.Random(f"{SEED}-{kind}")
        if kind == "longctx":
            for pid, user, gold in longctx(n, rng):
                out.append({"id": pid, "kind": kind, "user": user,
                            "max_tokens": MAX_TOKENS[kind], "gold": gold})
            continue
        rows = _src(kind)
        if kind == "math":
            gold = {i: r["answer"].split("####")[-1].strip().replace(",", "")
                    for i, r in enumerate(rows)}
        cands = list(globals()[kind](rows))
        order = list(range(len(cands)))
        rng.shuffle(order)
        got = 0
        for i in order:
            if got == n:
                break
            pid, user = cands[i]
            if not user or len(user) > MAX_PROMPT_CHARS:
                dropped[f"{kind}:length"] = dropped.get(f"{kind}:length", 0) + 1
                continue
            if _grams(user) & bad:
                dropped[f"{kind}:board-overlap"] = dropped.get(f"{kind}:board-overlap", 0) + 1
                continue
            rec = {"id": pid, "kind": kind, "user": user, "max_tokens": MAX_TOKENS[kind]}
            if kind == "math":
                rec["gold"] = gold[int(pid.split("/")[1])]
            out.append(rec)
            got += 1
        if got < n:
            raise SystemExit(f"{kind}: only {got} of {n} prompts survived filtering")
    DIR.mkdir(parents=True, exist_ok=True)
    p = DIR / f"{mix}.prompts.jsonl"
    with open(p, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"{p}: {len(out)} prompts {dict((k, sum(r['kind'] == k for r in out)) for k in want)}"
          f"; dropped {dropped or 'none'}")
    return p


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["fetch"]:
        fetch()
    elif a[:1] == ["build"]:
        build(a[2] if a[1:2] == ["--mix"] else "default")
    else:
        raise SystemExit(__doc__)
