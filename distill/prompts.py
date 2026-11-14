"""Prompt mixes for teacher datasets (see distill/__init__.py).

Sources are public, pinned, permissively licensed, and are train splits disjoint from the
board v2 test items:

  chat     UltraChat 200k train_sft, first user turn                       (MIT)
  ifollow  Tulu 3 persona instruction-following, IFEval-style constraints  (ODC-BY)
  code     StarCoder2 self-OSS-instruct exec-filtered, Python tasks        (ODC-BY)
  math     GSM8K train (the board uses test)                               (MIT)
  longctx  generated: keyed facts hidden in fineweb.TRAIN text, 2k-6k tok  (the board's
           needles use fineweb.test and a different fact template)

Pool kinds read prompt pools already downloaded under $FORGE_POOLS (default
/mnt/models/datasets, the Altra's cold archive; docs/datasets.md), so mixes that use them are
built there:

  wildchat   Tulu 3 mixture: tulu_v3.9_wildchat_100k, real user prompts       (ODC-BY)
  general    Tulu 3 mixture: flan_v2, no_robots, oasst1, sciriff, table_gpt    (ODC-BY)
  codemore   Magicoder Evol-Instruct 110K + Tulu 3 evol_codealpaca, personahub_code
                                                                     (Apache-2.0, ODC-BY)
  mathmore   NuminaMath-CoT problems (cn_k12, olympiads, AMC, orca_math, ...)  (Apache-2.0)
  multiling  Tulu 3 mixture: tulu_v3.9_aya_100k                               (ODC-BY)
  safety     Tulu 3 mixture: wildguardmix, wildjailbreak, coconot (keeps refusals in the
             training distribution)                                          (ODC-BY)

A mix is {kind: count}; the seed fixes which rows are drawn, so a mix name is a dataset.

  python -m distill.prompts build --mix <mix>
  python -m distill.prompts subset <src> <name> <n> [kind=n ...]

`subset` takes the first n prompts of each kind of an already built mix (kind=n overrides
one kind, kind=0 drops it), so a pilot or a slower teacher answers the same prompts as the
mix it came from, without the pools.
"""
import glob
import json
import os
import random
import zlib
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

POOLS = Path(os.environ.get("FORGE_POOLS", "/mnt/models/datasets"))
POOL_REVS = {  # revisions of the local downloads, from their .cache/huggingface metadata
    "allenai/tulu-3-sft-mixture": None,   # ⚠ unknown: its download kept no metadata
    "ise-uiuc/Magicoder-Evol-Instruct-110K": "b0079beaa0361d82412520b873715bee59cc7dd4",
    "AI-MO/NuminaMath-CoT": "9d8d210c9f6a36c8f3cd84045668c9b7800ef517",
}
TULU = {
    "wildchat": ["ai2-adapt-dev/tulu_v3.9_wildchat_100k"],
    "general": ["ai2-adapt-dev/flan_v2_converted", "ai2-adapt-dev/no_robots_converted",
                "ai2-adapt-dev/oasst1_converted", "ai2-adapt-dev/tulu_v3.9_sciriff_10k",
                "ai2-adapt-dev/tulu_v3.9_table_gpt_5k"],
    "codemore": ["ai2-adapt-dev/evol_codealpaca_heval_decontaminated",
                 "ai2-adapt-dev/personahub_code_v2_34999"],
    "multiling": ["ai2-adapt-dev/tulu_v3.9_aya_100k"],
    "safety": ["ai2-adapt-dev/tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k",
               "ai2-adapt-dev/tulu_v3.9_wildjailbreak_decontaminated_50k",
               "ai2-adapt-dev/coconot_converted"],
}

MIXES = {
    # ~4k prompts, weighted toward what a slice loses first (instructions, chat format)
    "default": {"chat": 1600, "ifollow": 1000, "code": 700, "math": 500, "longctx": 200},
    "smoke": {"chat": 8, "ifollow": 8, "code": 8, "math": 8, "longctx": 4},
    # ~41k, for a slow shared GPU to work through over days; built on the Altra (pool kinds)
    "wide": {"chat": 6000, "ifollow": 6000, "code": 4000, "math": 3000, "longctx": 1000,
             "wildchat": 6000, "general": 3000, "codemore": 5000, "mathmore": 5000,
             "multiling": 1000, "safety": 1000},
}
MAX_TOKENS = {k: 4096 for k in ("chat", "ifollow", "code", "math", "longctx", "wildchat",
                                "general", "codemore", "mathmore", "multiling", "safety")}
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
    don't match themselves. FORGE_BOARD_GRAMS names a JSON list written by `python -m
    distill.prompts grams <file>` on a machine with the board's sources (the Altra lacks them)."""
    if f := os.environ.get("FORGE_BOARD_GRAMS"):
        return set(json.load(open(f)))
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


def _tulu_first_user(sources):
    import pyarrow.parquet as pq
    want = set(sources)
    for p in sorted(glob.glob(str(POOLS / "allenai/tulu-3-sft-mixture/data/*.parquet"))):
        t = pq.read_table(p, columns=["id", "messages", "source"])
        for i, m, src in zip(*(t.column(c).to_pylist() for c in ("id", "messages", "source"))):
            if src in want:
                u = next((x["content"] for x in m if x["role"] == "user"), None)
                yield i, u


def _pool_tulu(kind):
    for i, u in _tulu_first_user(TULU[kind]):
        yield f"{kind}/{i}", u


def wildchat(_):
    return _pool_tulu("wildchat")


def general(_):
    return _pool_tulu("general")


def multiling(_):
    return _pool_tulu("multiling")


def safety(_):
    return _pool_tulu("safety")


def codemore(_):
    p = POOLS / "ise-uiuc/Magicoder-Evol-Instruct-110K/data-evol_instruct-decontaminated.jsonl"
    for line in open(p):
        u = json.loads(line)["instruction"]
        yield f"codemore/magicoder-{zlib.crc32(u.encode()):08x}", u
    yield from _pool_tulu("codemore")


def mathmore(_):
    import pyarrow.parquet as pq
    for p in sorted(glob.glob(str(POOLS / "AI-MO/NuminaMath-CoT/data/train-*.parquet"))):
        for u in pq.read_table(p, columns=["problem"]).column("problem").to_pylist():
            yield f"mathmore/numina-{zlib.crc32(u.encode()):08x}", u


POOL_KINDS = {"wildchat", "general", "codemore", "mathmore", "multiling", "safety"}


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
    out, dropped, seen = [], {}, set()   # seen: prompt texts, so pools that share rows don't repeat
    for kind, n in want.items():
        rng = random.Random(f"{SEED}-{kind}")
        if kind == "longctx":
            for pid, user, gold in longctx(n, rng):
                out.append({"id": pid, "kind": kind, "user": user,
                            "max_tokens": MAX_TOKENS[kind], "gold": gold})
            continue
        rows = None if kind in POOL_KINDS else _src(kind)
        if kind == "math":
            gold = {i: r["answer"].split("####")[-1].strip().replace(",", "")
                    for i, r in enumerate(rows)}
        cands = list(dict(globals()[kind](rows)).items())   # pools repeat some prompts: dedupe by id
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
            if "\x00" in user:   # a NUL can't be stored as JSON text by a Postgres-backed queue
                dropped[f"{kind}:nul"] = dropped.get(f"{kind}:nul", 0) + 1
                continue
            if user in seen:
                dropped[f"{kind}:duplicate"] = dropped.get(f"{kind}:duplicate", 0) + 1
                continue
            if _grams(user) & bad:
                dropped[f"{kind}:board-overlap"] = dropped.get(f"{kind}:board-overlap", 0) + 1
                continue
            rec = {"id": pid, "kind": kind, "user": user, "max_tokens": MAX_TOKENS[kind]}
            if kind == "math":
                rec["gold"] = gold[int(pid.split("/")[1])]
            out.append(rec)
            seen.add(user)
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


def subset(src, name, n, overrides=()):
    """Write <name>.prompts.jsonl: the first n prompts of each kind of <src>.prompts.jsonl."""
    want = {k: int(v) for k, v in (o.split("=") for o in overrides)}
    taken, out = {}, []
    for line in open(DIR / f"{src}.prompts.jsonl"):
        kind = json.loads(line)["kind"]
        if taken.get(kind, 0) < want.get(kind, n):
            taken[kind] = taken.get(kind, 0) + 1
            out.append(line)
    (DIR / f"{name}.prompts.jsonl").write_text("".join(out))
    print(f"{name}: {len(out)} prompts from {src} {dict(sorted(taken.items()))}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["fetch"]:
        fetch()
    elif a[:1] == ["build"]:
        build(a[2] if a[1:2] == ["--mix"] else "default")
    elif a[:1] == ["subset"] and len(a) >= 4:
        subset(a[1], a[2], int(a[3]), a[4:])
    elif a[:1] == ["grams"] and len(a) == 2:
        json.dump(sorted(board_grams()), open(a[1], "w"))
    else:
        raise SystemExit(__doc__)
