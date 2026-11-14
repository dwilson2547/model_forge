"""Choose which teacher answers to train on: spread (prompt embeddings) and need (divergence).

  python -m distill.curate embed <dataset.jsonl>
  python -m distill.curate score <dataset.jsonl> --student <hf> [--base <hf>] [--ctx 8192]
  python -m distill.curate report <dataset.jsonl> --student <hf> [--k 256] [--dup 0.95]
  python -m distill.curate pick <dataset.jsonl> --student <hf> --n <N> --how random|balanced|need
                                [--k 256] [--dup 0.95] [--floor 2] [--out <file>]

<dataset.jsonl> is a datagen output (`datagen export ... --partial` works on a run still going).
Everything is cached per prompt or answer id under $FORGE_DATA/distill/curate/, so a rerun on a
grown dataset embeds and scores only the new records.

- embed: prompts (the user turn, first 1024 tokens) through Qwen3-Embedding-0.6B (Apache-2.0),
  last-token pooled, unit length. Prompts, not answers: an answer's embedding mostly repeats
  its prompt's topic, and what matters about an answer (right, finished) isn't in it.
- score: per answer, the mean per-token KL(base || student) over the answer positions, the same
  quantity --distill trains on and computed the same way: the chat-template rendering of
  recover_gpu.py's --sft-dataset path, the base in bf16 and the student in nf4 as in training.
  Also each model's mean NLL of the answer. High KL = the cut broke something here; near zero =
  the student already matches the base, so the record teaches little.
- report: near-duplicates, k-means clusters (spherical, fixed seed), and KL by kind and cluster.
- pick: N records. random (the baseline), balanced (round-robin over clusters), need (a floor of
  the highest-KL records in every cluster, then the rest by KL). Near-duplicates are dropped
  first in all three: a record whose prompt has cosine > --dup with an earlier one.

Selection only uses the training pool. Never steer it toward board prompts: that is training
on the test by another route.
"""
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surgery import paths  # noqa: E402

DIR = paths.DATA / "distill" / "curate"
EMBEDDER = paths.BASE / "Qwen" / "Qwen3-Embedding-0.6B"   # revision 97b0c614be4d
BASE = paths.BASE / "Qwen" / "Qwen3.5-4B"
NF4_SKIP = ["in_proj_a", "in_proj_b", "lm_head"]          # as recover_gpu.FP32_SKIP_MODULES


def load(dataset):
    return [json.loads(l) for l in open(dataset)]


def _tag(path):
    """A short stable name for a model directory, for cache file names."""
    p = Path(path).expanduser().resolve()
    return f"{p.parent.name}-{p.name}"[-60:] + "-" + hashlib.sha1(str(p).encode()).hexdigest()[:6]


# ---------------------------------------------------------------- embeddings

def _emb_path():
    return DIR / "emb.qwen3-embedding-0.6b.npz"


def _emb_cache():
    p = _emb_path()
    if not p.exists():
        return {}
    z = np.load(p, allow_pickle=False)
    return dict(zip(z["ids"].tolist(), z["vecs"]))


def embed(dataset, batch=32, max_len=1024):
    import torch
    from transformers import AutoModel, AutoTokenizer
    recs = load(dataset)
    cache = _emb_cache()
    todo = sorted({r["id"]: r["user"] for r in recs if r["id"] not in cache}.items(),
                  key=lambda kv: -len(kv[1]))
    print(f"[curate] embed: {len(cache)} cached, {len(todo)} to do", flush=True)
    if todo:
        tok = AutoTokenizer.from_pretrained(EMBEDDER, padding_side="left")
        model = AutoModel.from_pretrained(EMBEDDER, dtype=torch.bfloat16, device_map={"": 0}).eval()
        t0 = time.time()
        with torch.no_grad():
            for i in range(0, len(todo), batch):
                chunk = todo[i:i + batch]
                # Truncate before the EOS the tokenizer appends: pooling reads the last token.
                ids = [x[:max_len - 1] + [tok.eos_token_id] if len(x) > max_len else x
                       for x in tok([t for _, t in chunk])["input_ids"]]
                enc = tok.pad({"input_ids": ids}, return_tensors="pt").to(0)
                h = model(**enc).last_hidden_state[:, -1]          # left padding: last = EOS
                h = torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy().astype(np.float16)
                cache.update({pid: v for (pid, _), v in zip(chunk, h)})
        print(f"[curate] embedded {len(todo)} in {time.time() - t0:.0f}s", flush=True)
        DIR.mkdir(parents=True, exist_ok=True)
        ids = list(cache)
        tmp = _emb_path().with_suffix(".tmp.npz")
        np.savez(tmp, ids=np.array(ids), vecs=np.stack([cache[i] for i in ids]))
        tmp.replace(_emb_path())
    return cache


def vectors(recs):
    cache = _emb_cache()
    missing = [r["id"] for r in recs if r["id"] not in cache]
    if missing:
        raise SystemExit(f"{len(missing)} records have no embedding: run `embed` first")
    return np.stack([cache[r["id"]] for r in recs]).astype(np.float32)


def near_dups(x, thr):
    """Indices i with cosine > thr to some j < i (the earlier one is kept)."""
    import torch
    v = torch.from_numpy(x).to(0, torch.float16)
    dup = []
    for s in range(0, len(v), 4096):
        sim = v[s:s + 4096] @ v[:s + 4096].T
        rows = torch.arange(sim.shape[0], device=0)[:, None] + s
        sim[torch.arange(sim.shape[1], device=0)[None, :] >= rows] = -1   # only earlier j
        dup += (torch.nonzero(sim.max(1).values > thr).flatten() + s).tolist()
    return set(dup)


def kmeans(x, k, iters=40, seed=0):
    """Spherical k-means: assignments (len(x),) and unit centroids (k, d)."""
    import torch
    g = torch.Generator().manual_seed(seed)
    v = torch.from_numpy(x).to(0)
    c = v[torch.randperm(len(v), generator=g)[:k].to(0)].clone()
    for _ in range(iters):
        a = (v @ c.T).argmax(1)
        new = torch.zeros_like(c).index_add_(0, a, v)
        empty = new.norm(dim=1) == 0
        new[empty] = c[empty]
        c = torch.nn.functional.normalize(new, dim=1)
    return (v @ c.T).argmax(1).cpu().numpy(), c.cpu().numpy()


# ---------------------------------------------------------------- divergence

def render(tok, rec):
    """(prefix ids, target ids) exactly as recover_gpu.py's --sft-dataset path renders them."""
    user = [{"role": "user", "content": rec["user"]}]
    uprefix = tok.apply_chat_template(user, tokenize=False, add_generation_prompt=False)
    full = tok.apply_chat_template(user + [{"role": "assistant", "reasoning_content": rec["reasoning"],
                                            "content": rec["answer"]}],
                                   tokenize=False, add_generation_prompt=False)
    assert full.startswith(uprefix), "template prefix drift"
    return (tok(uprefix, add_special_tokens=False)["input_ids"],
            tok(full[len(uprefix):].rstrip("\n"), add_special_tokens=False)["input_ids"])


def _score_path(dataset, student):
    return DIR / f"score.{Path(dataset).name.split('.')[0]}.{_tag(student)}.jsonl"


def scores(dataset, student):
    p = _score_path(dataset, student)
    return {r["id"]: r for r in map(json.loads, open(p))} if p.exists() else {}


def score(dataset, student, base=BASE, ctx=8192, chunk=1024):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    recs = load(dataset)
    done = scores(dataset, student)
    todo = [r for r in recs if r["id"] not in done]
    print(f"[curate] score vs {student}: {len(done)} cached, {len(todo)} to do", flush=True)
    if not todo:
        return done
    tok = AutoTokenizer.from_pretrained(base)
    t = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map={"": 0}).eval()
    s = AutoModelForCausalLM.from_pretrained(
        student, dtype=torch.bfloat16, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16,
                                               bnb_4bit_use_double_quant=True,
                                               llm_int8_skip_modules=NF4_SKIP)).eval()
    if t.config.get_text_config().vocab_size != s.config.get_text_config().vocab_size:
        raise SystemExit("base and student vocabularies differ")
    print(f"[curate] models loaded; vram {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    DIR.mkdir(parents=True, exist_ok=True)
    t0, n = time.time(), 0
    with open(_score_path(dataset, student), "a") as f, torch.no_grad():
        for rec in todo:
            prefix, tgt = render(tok, rec)
            out = {"id": rec["id"], "kind": rec["kind"], "tokens": len(tgt)}
            if len(prefix) + len(tgt) > ctx:
                out["skip"] = "over-ctx"                       # training drops these too
            else:
                ids = torch.tensor([prefix + tgt], device=0)
                pos = torch.arange(len(prefix) - 1, ids.shape[1] - 1, device=0)   # predicts tgt
                labels = ids[0, pos + 1]
                th = t.model(input_ids=ids)[0][0, pos]
                sh = s.model(input_ids=ids)[0][0, pos]
                kl = nll_t = nll_s = 0.0
                for i in range(0, len(pos), chunk):
                    lt = F.log_softmax(t.lm_head(th[i:i + chunk]).float(), -1)
                    ls = F.log_softmax(s.lm_head(sh[i:i + chunk]).float(), -1)
                    y = labels[i:i + chunk, None]
                    kl += (lt.exp() * (lt - ls)).sum().item()
                    nll_t -= lt.gather(1, y).sum().item()
                    nll_s -= ls.gather(1, y).sum().item()
                out.update(kl=kl / len(pos), nll_base=nll_t / len(pos), nll_student=nll_s / len(pos))
            f.write(json.dumps(out) + "\n")
            f.flush()
            done[rec["id"]] = out
            n += 1
            if n % 200 == 0 or n == len(todo):
                dt = time.time() - t0
                print(f"  {n}/{len(todo)}  {dt:.0f}s  ({n / dt:.1f}/s)", flush=True)
    return done


# ---------------------------------------------------------------- report / pick

def _prepare(dataset, student, k, dup):
    recs = load(dataset)
    x = vectors(recs)
    dups = near_dups(x, dup)
    sc = scores(dataset, student) if student else {}
    keep = [i for i in range(len(recs)) if i not in dups]
    assign, _ = kmeans(x[keep], k)
    clusters = {i: int(c) for i, c in zip(keep, assign)}
    return recs, dups, sc, clusters


def report(dataset, student=None, k=256, dup=0.95):
    recs, dups, sc, clusters = _prepare(dataset, student, k, dup)
    kl = {i: sc[r["id"]]["kl"] for i, r in enumerate(recs) if "kl" in sc.get(r["id"], {})}
    out = {"dataset": str(dataset), "records": len(recs), "near_dups": len(dups), "dup_cos": dup,
           "k": k, "scored": len(kl)}
    by_kind = {}
    for i, r in enumerate(recs):
        d = by_kind.setdefault(r["kind"], {"n": 0, "dups": 0, "kl": []})
        d["n"] += 1
        d["dups"] += i in dups
        if i in kl:
            d["kl"].append(kl[i])
    out["by_kind"] = {kd: {"n": d["n"], "dups": d["dups"],
                           **({"kl_mean": round(float(np.mean(d["kl"])), 4),
                               "kl_p90": round(float(np.percentile(d["kl"], 90)), 4)} if d["kl"] else {})}
                      for kd, d in sorted(by_kind.items())}
    if kl:
        v = np.array(list(kl.values()))
        out["kl"] = {q: round(float(np.percentile(v, p)), 4)
                     for q, p in (("p10", 10), ("p50", 50), ("p90", 90), ("p99", 99))}
        out["kl"]["mean"] = round(float(v.mean()), 4)
    cl = {}
    for i, c in clusters.items():
        cl.setdefault(c, []).append(i)
    sizes = sorted(len(m) for m in cl.values())
    out["cluster_sizes"] = {"min": sizes[0], "p50": sizes[len(sizes) // 2], "max": sizes[-1]}
    rows = []
    for c, m in cl.items():
        ks = [kl[i] for i in m if i in kl]
        kinds = {}
        for i in m:
            kinds[recs[i]["kind"]] = kinds.get(recs[i]["kind"], 0) + 1
        rows.append({"cluster": c, "n": len(m), "kl_mean": round(float(np.mean(ks)), 4) if ks else None,
                     "kinds": dict(sorted(kinds.items(), key=lambda kv: -kv[1])[:3]),
                     "example": recs[m[0]]["user"][:140].replace("\n", " ")})
    rows.sort(key=lambda r: -(r["kl_mean"] or 0))
    out["clusters_by_kl"] = rows
    p = DIR / f"report.{Path(dataset).name.split('.')[0]}.{Path(dataset).name.split('.')[1]}.json"
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps({kk: vv for kk, vv in out.items() if kk != "clusters_by_kl"}, indent=1))
    for r in rows[:8] + [{"cluster": "..."}] + rows[-5:]:
        print(r)
    print(f"[curate] full report: {p}")
    return out


def pick(dataset, student, n, how, k=256, dup=0.95, floor=2, out=None, seed=0):
    recs, dups, sc, clusters = _prepare(dataset, student, k, dup)
    pool = [i for i in clusters if "skip" not in sc.get(recs[i]["id"], {})]
    if how == "need":
        missing = [i for i in pool if "kl" not in sc.get(recs[i]["id"], {})]
        if missing:
            raise SystemExit(f"{len(missing)} records are unscored: run `score` first")
    rng = random.Random(seed)
    if how == "random":
        chosen = rng.sample(pool, min(n, len(pool)))
    else:
        by = {}
        for i in pool:
            by.setdefault(clusters[i], []).append(i)
        for m in by.values():
            if how == "need":
                m.sort(key=lambda i: -sc[recs[i]["id"]]["kl"])
            else:
                rng.shuffle(m)
        if how == "balanced":
            chosen, r = [], 0
            while len(chosen) < min(n, len(pool)):
                chosen += [m[r] for m in by.values() if r < len(m)][:n - len(chosen)]
                r += 1
        else:
            chosen = [i for m in by.values() for i in m[:floor]][:n]
            rest = sorted(set(pool) - set(chosen), key=lambda i: -sc[recs[i]["id"]]["kl"])
            chosen += rest[:n - len(chosen)]
    chosen = sorted(chosen)                                     # dataset order
    stem = Path(dataset).name.removesuffix(".jsonl")
    out = Path(out) if out else Path(dataset).with_name(f"{stem}.{how}{n}.jsonl")
    with open(out, "w") as f:
        for i in chosen:
            f.write(json.dumps(recs[i]) + "\n")
    kinds = {}
    for i in chosen:
        kinds[recs[i]["kind"]] = kinds.get(recs[i]["kind"], 0) + 1
    print(f"[curate] {how}: {len(chosen)} of {len(pool)} (after {len(dups)} near-dups) -> {out}\n"
          f"  kinds {dict(sorted(kinds.items()))}")
    return out


def main():
    a = sys.argv[1:]
    if len(a) < 2:
        raise SystemExit(__doc__)
    cmd, dataset = a[0], a[1]
    o = dict(zip(a[2::2], a[3::2]))
    st = o.get("--student")
    if cmd == "embed":
        embed(dataset)
    elif cmd == "score" and st:
        score(dataset, st, base=o.get("--base", BASE), ctx=int(o.get("--ctx", 8192)))
    elif cmd == "report":
        report(dataset, st, k=int(o.get("--k", 256)), dup=float(o.get("--dup", 0.95)))
    elif cmd == "pick" and "--n" in o and "--how" in o:
        pick(dataset, st, int(o["--n"]), o["--how"], k=int(o.get("--k", 256)),
             dup=float(o.get("--dup", 0.95)), floor=int(o.get("--floor", 2)), out=o.get("--out"))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
