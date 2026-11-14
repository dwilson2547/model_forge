"""Pick layers to drop by measured damage: drop each one, score KL divergence from the uncut model.

  python -m surgery.damage_scan <label> <model.gguf> [--corpora fineweb,code] [--chunks 8]
                                [--greedy 4] [--candidates linear|all] [--start 25 29] [--only 3 7] [--top 16]

Why (design §0j.15): the cosine scorer (ranking.py) scores how little a layer turns the
residual stream, not what removing it costs. On the 4B its [25,29] pick is a worse unrepaired
wound than [24,28]; on the 27B every pick was an early layer and wikitext went +15% -> +74%
between drop6 and drop8 with nothing in the scores to predict it. This measures the cost
directly: gguf_drop writes the slice (bookkeeping, no HF load), llama-perplexity scores
KLD against the uncut model's logits on the same tokens.

Round 1 scores every candidate dropped alone. With --greedy K, the lowest-KLD layer joins
the drop set and every remaining candidate is rescored on top of it, K rounds in all:
damage is not additive (that is the drop6 -> drop8 cliff), so a set is built one layer at a
time, not read off the single-drop table. --start seeds the set (e.g. to extend a known one);
--top N limits rounds after the first to the N cheapest single drops.

Works on any GGUF quant (tensors are copied byte for byte), so a 27B scan can run on the
IQ3_M that fits a 16 GB card. The reference is then the uncut model at that same quant.

Rows append to $FORGE_ROOT/results/damage.jsonl, one per scored drop set; a rerun skips sets
already scored for this label/corpora/chunks. One slice GGUF on disk at a time, in
$FORGE_SCAN_TMP (default $FORGE_ROOT/runs/damage-scan/<label>/), deleted after scoring.
Env as kld.py: FORGE_LLAMA, FORGE_NGL, FORGE_THREADS, FORGE_DATA.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surgery import evaluate, gguf_drop, multicorpus, paths  # noqa: E402

OUT = Path(os.environ.get("FORGE_DAMAGE_RESULTS", str(paths.RESULTS / "damage.jsonl")))


def _perplexity(args, chunks):
    cmd = [evaluate._bin(str(paths.LLAMA), "llama-perplexity"), "--chunks", str(chunks),
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


def parse(text):
    return {"ppl": _num(text, r"Mean PPL\(Q\)\s*:\s*([0-9.]+)"),
            "kld_mean": _num(text, r"Mean\s+KLD:\s*([0-9.]+)"),
            "kld_p99": _num(text, r"99\.0%\s+KLD:\s*([0-9.]+)"),
            "same_top": _num(text, r"Same top p:\s*([0-9.]+)")}


def key(label, corpora, chunks, drop):
    return json.dumps([label, sorted(corpora), chunks, sorted(drop)])


def load_done():
    done = {}
    if OUT.exists():
        for r in map(json.loads, open(OUT)):
            done[key(r["label"], r["corpora"], r["chunks"], r["drop"])] = r
    return done


def pick(rows):
    """The candidate row with the lowest mean KLD across the corpora."""
    return min(rows, key=lambda r: r["kld"])


def scan(label, model, corpora, chunks, greedy, candidates, start, tmp, only=None, top=0):
    n_main, rec = gguf_drop.layout(model)
    cand = [i for i in range(n_main) if (candidates == "all" or rec[i]) and (not only or i in only)]
    print(f"[scan] {label}: {n_main} main blocks, {sum(rec)} linear; "
          f"{len(cand)} candidates ({candidates}), corpora {corpora}, {chunks} chunks", flush=True)
    tmp.mkdir(parents=True, exist_ok=True)
    refs = {}
    for c in corpora:
        refs[c] = tmp / f"ref-{c}-{chunks}.kld"
        if not refs[c].exists():
            t0 = time.time()
            text = _perplexity(["-m", str(model), "-f", multicorpus.CORPORA[c],
                                "--kl-divergence-base", str(refs[c])], chunks)
            print(f"[scan] ref {c}: PPL {_num(text, r'Final estimate: PPL = ([0-9.]+)'):.4f} "
                  f"({time.time() - t0:.0f}s, {refs[c].stat().st_size / 2**30:.1f} GiB)", flush=True)
    done = load_done()
    chosen = sorted(start)
    slice_p = tmp / "slice.gguf"
    for rnd in range(1, greedy + 1):
        rows = []
        for i in [c for c in cand if c not in chosen]:
            drop = sorted(chosen + [i])
            k = key(label, corpora, chunks, drop)
            if k not in done:
                t0 = time.time()
                gguf_drop.drop(model, slice_p, drop, quiet=True)
                row = {"label": label, "model": str(model), "corpora": sorted(corpora),
                       "chunks": chunks, "drop": drop, "added": i, "base_set": chosen,
                       "type": "linear" if rec[i] else "full", "ngl": evaluate.NGL or None}
                for c in corpora:
                    row[c] = parse(_perplexity(["-m", str(slice_p), "--kl-divergence-base",
                                                str(refs[c]), "--kl-divergence"], chunks))
                slice_p.unlink()
                row["kld"] = round(sum(row[c]["kld_mean"] for c in corpora) / len(corpora), 5)
                row["seconds"] = round(time.time() - t0)
                OUT.parent.mkdir(parents=True, exist_ok=True)
                with OUT.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                done[k] = row
            r = done[k]
            rows.append(r)
            print(f"[scan] round {rnd} +{i:<3d} {r['type']:6s} KLD {r['kld']:.4f}  "
                  + "  ".join(f"{c} {r[c]['kld_mean']:.4f}/top1 {r[c]['same_top']:.1f}%"
                              for c in corpora), flush=True)
        if not rows:
            break
        best = pick(rows)
        chosen = sorted(chosen + [best["added"]])
        order = sorted(rows, key=lambda r: r["kld"])
        if rnd == 1 and top:
            cand = [r["added"] for r in order[:top]]
            print(f"[scan] later rounds rescore the {len(cand)} cheapest single drops only: "
                  f"{sorted(cand)}", flush=True)
        print(f"[scan] round {rnd} pick +{best['added']} -> {chosen}  KLD {best['kld']:.4f}; "
              f"next best: " + ", ".join(f"{r['added']} ({r['kld']:.4f})" for r in order[1:6]),
              flush=True)
    for p in refs.values():
        p.unlink(missing_ok=True)
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("model", type=Path)
    ap.add_argument("--corpora", default="fineweb,code")
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--greedy", type=int, default=1, help="rounds; 1 = the single-drop table only")
    ap.add_argument("--candidates", choices=["linear", "all"], default="all",
                    help="'linear' skips full-attention blocks (the old protection rule)")
    ap.add_argument("--start", type=int, nargs="*", default=[])
    ap.add_argument("--top", type=int, default=0,
                    help="after round 1, rescore only the N cheapest single drops (0 = all): "
                         "a 27B round over all 64 blocks is hours on a CPU")
    ap.add_argument("--only", type=int, nargs="*", default=[], help="score just these blocks")
    a = ap.parse_args()
    tmp = Path(os.environ.get("FORGE_SCAN_TMP", str(paths.RUNS / "damage-scan" / a.label)))
    chosen = scan(a.label, a.model, a.corpora.split(","), a.chunks, a.greedy, a.candidates,
                  a.start, tmp, only=a.only, top=a.top)
    print(f"[scan] {a.label}: greedy set after {a.greedy} round(s): {chosen}")


if __name__ == "__main__":
    main()
