"""Chat-format distance from the base, on teacher answers no arm trained on.

  python -m distill.chatkld ref <dataset.jsonl> [--base <hf>] [--ctx 8192] [--max 800]
  python -m distill.chatkld score <label> <student hf> <dataset.jsonl> [--nf4]

Why (design §0j.15): raw-text KLD turned out a weak proxy for chat behaviour (§0j.13: a 34%
lower fineweb KLD bought +0.014 on board v2), and the board's code tiers flip ~30 HumanEval
items between any two arms. This scores the thing the repair is for, per token: KL(base ||
student) and top-1 agreement over the answer positions of held-out teacher answers, in the chat
template, split by prompt kind. It is the trainer's own --distill quantity on records
`recover_gpu --holdout` kept out of training (distill.is_holdout, ~2% of ids).

One model on the GPU at a time, so the student is scored in bf16, as shipped (curate.py's
score keeps both resident and so reads the student in nf4). `ref` runs the base once and
stores, per answer position, its top-K log-probs and the leftover tail mass (fp16), in
$FORGE_DATA/distill/chatkld/<dataset>.ref.pt, about K x 6 bytes per token. `score` then needs
only the student. KL is computed over the base's top-K tokens plus one bucket for everything
else; with K=256 the bucket holds little mass, and the estimate is a lower bound on the full KL
(merging outcomes never raises KL). --nf4 scores the student as training sees it.

Rows append to $FORGE_ROOT/results/chatkld.jsonl.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distill import is_holdout  # noqa: E402
from distill.curate import BASE, NF4_SKIP, load, render  # noqa: E402
from surgery import paths  # noqa: E402

DIR = paths.DATA / "distill" / "chatkld"
LEDGER = paths.RESULTS / "chatkld.jsonl"
TOPK = 256


def _ref_path(dataset):
    return DIR / (Path(dataset).name.removesuffix(".jsonl") + ".ref.pt")


def holdout(dataset, ctx, cap, tok):
    """[(record, prefix ids, target ids)] of the held-out records that fit ctx, in id order."""
    out = []
    for rec in sorted((r for r in load(dataset) if is_holdout(r["id"])), key=lambda r: r["id"]):
        prefix, tgt = render(tok, rec)
        if len(prefix) + len(tgt) <= ctx:
            out.append((rec, prefix, tgt))
    return out[:cap]


def _positions(model, prefix, tgt, chunk=1024):
    """Log-softmax rows for the positions that predict tgt, yielded in chunks (fp32, on GPU)."""
    import torch
    import torch.nn.functional as F
    ids = torch.tensor([prefix + tgt], device=0)
    pos = torch.arange(len(prefix) - 1, ids.shape[1] - 1, device=0)
    h = model.model(input_ids=ids)[0][0, pos]
    for i in range(0, len(pos), chunk):
        yield F.log_softmax(model.lm_head(h[i:i + chunk]).float(), -1)


def ref(dataset, base=BASE, ctx=8192, cap=800):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base)
    items = holdout(dataset, ctx, cap, tok)
    if not items:
        raise SystemExit("no held-out records in this dataset")
    t = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map={"": 0}).eval()
    t.config.use_cache = False
    recs, t0 = [], time.time()
    with torch.no_grad():
        for n, (rec, prefix, tgt) in enumerate(items, 1):
            lp, ix = [], []
            for rows in _positions(t, prefix, tgt):
                top = rows.topk(TOPK, -1)
                lp.append(top.values.half().cpu())
                ix.append(top.indices.int().cpu())
            recs.append({"id": rec["id"], "kind": rec["kind"], "n_prefix": len(prefix),
                         "ids": prefix + tgt, "logp": torch.cat(lp), "idx": torch.cat(ix)})
            if n % 100 == 0 or n == len(items):
                print(f"  ref {n}/{len(items)}  {time.time() - t0:.0f}s", flush=True)
    DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"base": str(base), "ctx": ctx, "topk": TOPK, "records": recs}, _ref_path(dataset))
    toks = sum(len(r["logp"]) for r in recs)
    print(f"[chatkld] {_ref_path(dataset)}: {len(recs)} held-out answers, {toks} answer tokens")


def kl_topk(t_logp, t_idx, s_rows):
    """Per-position KL(base || student) over the base's top-K tokens + one 'everything else'
    bucket, and whether the two agree on the top token. t_* are [N, K]; s_rows is [N, V] log-probs."""
    import torch
    t_logp = t_logp.float()
    s_logp = s_rows.gather(1, t_idx.long())
    tp = t_logp.exp()
    kl = (tp * (t_logp - s_logp)).sum(1)
    t_rest = (1 - tp.sum(1)).clamp_min(1e-9)
    s_rest = (1 - s_logp.exp().sum(1)).clamp_min(1e-9)
    kl = kl + t_rest * (t_rest.log() - s_rest.log())
    return kl.clamp_min(0), s_rows.argmax(1) == t_idx[:, 0]


def score(label, student, dataset, nf4=False):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    p = _ref_path(dataset)
    if not p.exists():
        raise SystemExit(f"{p} missing: python -m distill.chatkld ref {dataset}")
    refd = torch.load(p, weights_only=False)
    q = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
                            llm_int8_skip_modules=NF4_SKIP) if nf4 else None)
    s = AutoModelForCausalLM.from_pretrained(student, dtype=torch.bfloat16, device_map={"": 0},
                                             quantization_config=q).eval()
    s.config.use_cache = False
    by, t0 = {}, time.time()
    with torch.no_grad():
        for rec in refd["records"]:
            prefix, tgt = rec["ids"][:rec["n_prefix"]], rec["ids"][rec["n_prefix"]:]
            kls, same, at = [], [], 0
            for rows in _positions(s, prefix, tgt):
                n = rows.shape[0]
                k, a = kl_topk(rec["logp"][at:at + n].to(0), rec["idx"][at:at + n].to(0), rows)
                kls.append(k.cpu())
                same.append(a.cpu())
                at += n
            d = by.setdefault(rec["kind"], {"kl": [], "same": [], "n": 0})
            d["kl"].append(torch.cat(kls))
            d["same"].append(torch.cat(same))
            d["n"] += 1

    def summ(kl, same, n):
        return {"answers": n, "tokens": len(kl), "kld_mean": round(kl.mean().item(), 4),
                "kld_p99": round(kl.quantile(0.99).item(), 3),
                "same_top": round(100 * same.float().mean().item(), 2)}
    kinds = {k: summ(torch.cat(d["kl"]), torch.cat(d["same"]), d["n"]) for k, d in sorted(by.items())}
    allkl = torch.cat([x for d in by.values() for x in d["kl"]])
    allsame = torch.cat([x for d in by.values() for x in d["same"]])
    row = {"label": label, "student": str(student), "dataset": Path(dataset).name, "nf4": nf4,
           "base": refd["base"], "topk": refd["topk"],
           "all": summ(allkl, allsame, sum(d["n"] for d in by.values())), "kinds": kinds,
           "seconds": round(time.time() - t0), "date": time.strftime("%Y-%m-%dT%H:%M:%S")}
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"== {label}  KLD {row['all']['kld_mean']:.4f}  p99 {row['all']['kld_p99']:.3f}  "
          f"top-1 {row['all']['same_top']:.2f}%  ({row['all']['tokens']} tokens)")
    for k, v in kinds.items():
        print(f"  {k:10s} KLD {v['kld_mean']:.4f}  top-1 {v['same_top']:.2f}%  n={v['answers']}")
    return row


def main():
    a = sys.argv[1:]
    flags = {"--nf4"} & set(a)
    a = [x for x in a if x not in flags]
    if a[:1] == ["ref"] and len(a) >= 2:
        o = dict(zip(a[2::2], a[3::2]))
        ref(a[1], base=o.get("--base", BASE), ctx=int(o.get("--ctx", 8192)), cap=int(o.get("--max", 800)))
    elif a[:1] == ["score"] and len(a) == 4:
        score(a[1], a[2], a[3], nf4="--nf4" in flags)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
