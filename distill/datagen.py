"""Answer a prompt mix with a teacher GGUF and write a --sft-dataset for the trainer.

  python -m distill.datagen <label> <teacher.gguf> [--mix default] [--slots 4]
                            [--temp 0.6] [--top-p 0.95] [--top-k 20] [--think]

<label> names the teacher (e.g. qwen35-4b-base, gemma-...); output is
$FORGE_DATA/distill/<label>.<mix>.jsonl and resumes from it after a crash. Sampling
defaults are Qwen's published thinking-mode settings; set them per teacher family from
its model card (record them in the run's notes, they are not stored per record).

Thinking is OFF by default (enable_thinking=false), matching board v2: with it on, the
Qwen3.5-4B base runs out of a 4096-token budget re-checking itself on most IFEval-style
prompts (2026-11-14), which here would drop the hardest prompts from the dataset and fill
the rest with loops. --think keeps thinking traces; the label then gets a `-think` suffix
so the two datasets never share a file.

Kept only when the teacher finished on its own (stop = eos) with a non-empty answer: a
response cut at max_tokens would teach the student to never close. Dropped responses stay
in the raw file ($FORGE_DATA/distill/raw/...) with their reason, so a rerun does not
regenerate them. Where the prompt has a gold answer (math, longctx), `check` records
whether the teacher got it right; it does not filter by default, since logit distillation
matches the teacher's distribution, not a correct answer (pass --only-correct to filter).
"""
import json
import re
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.serve import LlamaServer  # noqa: E402
from distill import prompts  # noqa: E402


def _check(rec, answer):
    gold = rec.get("gold")
    if gold is None:
        return None
    if rec["kind"] == "math":
        nums = re.findall(r"-?[\d,]*\.?\d+", answer.replace("$", ""))
        try:
            return bool(nums) and abs(float(nums[-1].replace(",", "").rstrip(".")) - float(gold)) < 1e-6
        except ValueError:
            return False
    return all(re.search(rf"{re.escape(c)}\D{{0,40}}{d}\b", answer) for c, d in gold.items())


def run(label, teacher, mix="default", slots=4, slot_ctx=16384, temp=0.6, top_p=0.95,
        top_k=20, only_correct=False, think=False):
    if think:
        label += "-think"
    src = prompts.DIR / f"{mix}.prompts.jsonl"
    if not src.exists():
        raise SystemExit(f"{src} missing: python -m distill.prompts build --mix {mix}")
    todo_all = [json.loads(l) for l in open(src)]
    raw_dir = prompts.DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_p = raw_dir / f"{label}.{mix}.jsonl"
    done = {r["id"]: r for r in map(json.loads, open(raw_p))} if raw_p.exists() else {}
    todo = [r for r in todo_all if r["id"] not in done]
    todo.sort(key=lambda r: -len(r["user"]))       # long prompts first: no stragglers
    print(f"[datagen] {label} on {mix}: {len(done)} cached, {len(todo)} to generate "
          f"({slots} slots, T={temp} top_p={top_p} top_k={top_k})", flush=True)
    if todo:
        t0, toks = time.time(), 0
        with LlamaServer(teacher, ctx=slots * slot_ctx, parallel=slots) as srv, \
                raw_p.open("a") as f, ThreadPoolExecutor(slots) as pool:
            def one(it):
                r = srv.chat(it["user"], max_tokens=it["max_tokens"], temperature=temp,
                             top_k=top_k, top_p=top_p,
                             seed=zlib.crc32(it["id"].encode()) % 2**31, think=think)
                return {"id": it["id"], "reasoning": r["reasoning"].strip(),
                        "answer": r["content"].strip(), "stop": r["stop_type"],
                        "tokens": r["tokens"]}
            for n, rec in enumerate(pool.map(one, todo), 1):
                f.write(json.dumps(rec) + "\n")
                f.flush()
                done[rec["id"]] = rec
                toks += rec["tokens"]
                if n % 50 == 0 or n == len(todo):
                    dt = time.time() - t0
                    print(f"  {n}/{len(todo)}  {dt:.0f}s  {toks / dt:.0f} tok/s", flush=True)

    out_p = prompts.DIR / f"{label}.{mix}.jsonl"
    kept, why = 0, {}
    with open(out_p, "w") as f:
        for it in todo_all:
            r = done[it["id"]]
            reason = ("limit" if r["stop"] != "eos" else "empty-answer" if not r["answer"]
                      else None)
            chk = _check(it, r["answer"])
            if reason is None and only_correct and chk is False:
                reason = "wrong"
            if reason:
                why[f"{it['kind']}:{reason}"] = why.get(f"{it['kind']}:{reason}", 0) + 1
                continue
            f.write(json.dumps({"id": it["id"], "kind": it["kind"], "user": it["user"],
                                "reasoning": r["reasoning"], "answer": r["answer"],
                                "teacher": label, "stop": r["stop"], "tokens": r["tokens"],
                                "check": chk}) + "\n")
            kept += 1
    checks = [_check(it, done[it["id"]]["answer"]) for it in todo_all if it.get("gold")]
    print(f"[datagen] {out_p}: kept {kept}/{len(todo_all)}; dropped {why or 'none'}; "
          f"gold-checked correct {sum(c is True for c in checks)}/{len(checks)}", flush=True)
    return out_p


def main():
    a = sys.argv[1:]
    flags = {"--only-correct", "--think"} & set(a)
    a = [x for x in a if x not in flags]
    if len(a) < 2:
        raise SystemExit(__doc__)
    opts = dict(zip(a[2::2], a[3::2]))
    run(a[0], a[1], mix=opts.get("--mix", "default"), slots=int(opts.get("--slots", 4)),
        temp=float(opts.get("--temp", 0.6)), top_p=float(opts.get("--top-p", 0.95)),
        top_k=int(opts.get("--top-k", 20)), only_correct="--only-correct" in flags,
        think="--think" in flags)


if __name__ == "__main__":
    main()
