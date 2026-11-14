"""Board v2 runner: the bench/suites.py items through one llama-server, graded, ledgered.

  python -m bench.board2 run <label> <model.gguf> [--tiers humaneval,gsm8k] [--slots 4] [--think]
  python -m bench.board2 compare <label-a> <label-b>     # paired, item by item

Chat mode only (/v1/chat/completions, the GGUF's own template), greedy, so any family with a
chat template runs unchanged (Qwen, Gemma, ...). Thinking is OFF by default
(enable_thinking=false; templates without the switch ignore it): with it on, the Qwen3.5-4B
base burns its whole 4096-token budget re-checking itself on 136 of 200 IFEval prompts and
scores 0.30, while with it off it scores 0.92 on 24 of those same prompts (2026-11-14) - a
thinking-mode board measures loop rate, not the skill. --think runs the thinking-mode board
(label it e.g. `base-think@5080`): its `limit` counts are the runaway-thinking signal.
Answers are graded on the post-thinking content; a response that hits max_tokens scores 0
and is counted as `limit`. Raw responses go to $FORGE_ROOT/runs/board2/<label>.jsonl (a rerun resumes from
it); the summary row, with every item's score for paired comparison, appends to
$FORGE_ROOT/results/board2.jsonl.

N parallel slots batch the decode (~N x throughput on a GPU). Batched greedy decoding is not
bit-identical to single-stream, so run the base twice to read the board's own noise floor
before trusting a small gap. Rows are only comparable from the same surface (GPU vs CPU).
"""
import json
import math
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench import suites  # noqa: E402
from bench.serve import LlamaServer  # noqa: E402
from surgery import paths  # noqa: E402

RAW = paths.RUNS / "board2"
LEDGER = paths.RESULTS / "board2.jsonl"


# ---------------------------------------------------------------- grading

def _code_block(text):
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, re.DOTALL)
    blocks = [b for b in blocks if "def " in b] or blocks
    return blocks[-1] if blocks else None


def _sandbox(program, timeout=30):
    """Run a program in the docker sandbox (no network, 512 MB, non-root). True iff rc 0."""
    with tempfile.TemporaryDirectory() as td:
        Path(td, "prog.py").write_text(program)
        try:
            p = subprocess.run(
                ["docker", "run", "--rm", "--network", "none", "--memory", "512m",
                 "--user", "1000:1000", "-e", "PYTHONDONTWRITEBYTECODE=1", "-v", f"{td}:/t",
                 "-w", "/t", "python:3.12-slim", "timeout", str(timeout), "python", "prog.py"],
                capture_output=True, text=True, timeout=timeout + 60)
        except subprocess.TimeoutExpired:
            return False, "sandbox timeout"
        return p.returncode == 0, (p.stderr.strip().splitlines() or ["ok"])[-1][:160]


def g_code(item, content):
    code = _code_block(content)
    if code is None:
        return 0.0, "no code block"
    ok, why = _sandbox(item["header"] + "\n" + code + "\n\n" + item["tests"])
    return float(ok), why


def _num(s):
    try:
        return float(s.replace(",", "").replace("$", "").rstrip("."))
    except ValueError:
        return None


def g_gsm8k(item, content):
    m = re.findall(r"Answer:\s*\**\s*\$?\s*(-?[\d,]*\.?\d+)", content)
    got = m[-1] if m else (re.findall(r"-?[\d,]*\.?\d+", content) or [None])[-1]
    if got is None:
        return 0.0, "no number"
    a, g = _num(got), _num(item["gold"])
    return float(a is not None and abs(a - g) < 1e-6), f"got {got} want {item['gold']}"


def g_ifeval(item, content):
    """Instruction-level strict accuracy (Google's checker), so a prompt with 3 instructions
    gives partial credit; prompt-level 'all followed' is kept in the detail."""
    from bench.ifeval import instructions_registry
    ok = []
    for iid, kw in zip(item["instruction_id_list"], item["kwargs"]):
        inst = instructions_registry.INSTRUCTION_DICT[iid](iid)
        inst.build_description(**{k: v for k, v in kw.items() if v is not None})
        args = inst.get_instruction_args()
        if args and "prompt" in args:
            inst.build_description(prompt=item["prompt"])
        ok.append(bool(content.strip()) and bool(inst.check_following(content)))
    return sum(ok) / len(ok), f"{sum(ok)}/{len(ok)}" + (" all" if all(ok) else "")


def g_needle(item, content):
    hit = 0
    for name, code in item["gold"].items():
        m = re.findall(rf"{name}\W{{0,6}}(\d{{5}})", content)
        hit += bool(m) and m[-1] == code
    return hit / len(item["gold"]), f"{hit}/{len(item['gold'])}"


GRADERS = {"humaneval": g_code, "mbpp": g_code, "gsm8k": g_gsm8k, "ifeval": g_ifeval,
           "needle": g_needle}


# ---------------------------------------------------------------- run

def _generate(label, model, items, slots, slot_ctx, think=False):
    RAW.mkdir(parents=True, exist_ok=True)
    raw_path = RAW / f"{label}.jsonl"
    done = {}
    if raw_path.exists():
        done = {r["id"]: r for r in map(json.loads, open(raw_path))}
    todo = [it for it in items if it["id"] not in done]
    todo.sort(key=lambda it: -it.get("ctx_tokens", 0))  # long prompts first: no stragglers
    print(f"  {len(done)} cached, {len(todo)} to generate ({slots} slots)", flush=True)
    if not todo:
        return done
    t0 = time.time()
    with LlamaServer(model, ctx=slots * slot_ctx, parallel=slots) as srv, \
            raw_path.open("a") as f, ThreadPoolExecutor(slots) as pool:
        def one(it):
            r = srv.chat(it["prompt"], max_tokens=it["max_tokens"], think=think)
            return {"id": it["id"], "content": r["content"], "think_chars":
                    len(r["text"]) - len(r["content"]), "stop": r["stop_type"],
                    "tokens": r["tokens"]}
        for n, rec in enumerate(pool.map(one, todo), 1):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done[rec["id"]] = rec
            if n % 25 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
    return done


def run(label, model, tiers=None, slots=4, slot_ctx=32768, think=False):
    items = suites.items(tiers)
    t0 = time.time()
    raw = _generate(label, model, items, slots, slot_ctx, think)
    gen_s = time.time() - t0

    def grade(it):
        r = raw[it["id"]]
        score, detail = GRADERS[it["tier"]](it, r["content"])
        return it["id"], it["tier"], score, detail, r
    with ThreadPoolExecutor(6) as pool:
        graded = list(pool.map(grade, items))

    tiers_out, scores = {}, {}
    for t in dict.fromkeys(it["tier"] for it in items):
        rows = [g for g in graded if g[1] == t]
        s = [g[2] for g in rows]
        mean = sum(s) / len(s)
        se = math.sqrt(sum((x - mean) ** 2 for x in s) / (len(s) - 1) / len(s)) if len(s) > 1 else 0
        tiers_out[t] = {"score": round(mean, 4), "se": round(se, 4), "n": len(s),
                        "limit": sum(1 for g in rows if g[4]["stop"] == "limit"),
                        "mean_tokens": round(sum(g[4]["tokens"] for g in rows) / len(rows))}
        if t == "ifeval":
            tiers_out[t]["prompt_all"] = round(
                sum(1 for g in rows if g[3].endswith(" all")) / len(rows), 4)
    for g in graded:
        scores[g[0]] = round(g[2], 4)
    row = {"label": label, "model": str(model), "board": "v2", "seed": suites.SEED,
           "think": think,
           "slots": slots, "tiers": tiers_out,
           "composite": round(sum(v["score"] for v in tiers_out.values()) / len(tiers_out), 4),
           "gen_s": round(gen_s), "date": time.strftime("%Y-%m-%dT%H:%M:%S"), "items": scores}
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"\n== {label}  composite {row['composite']}  ({gen_s / 60:.0f} min generating)")
    for t, v in tiers_out.items():
        print(f"  {t:10s} {v['score']:.3f} ± {v['se']:.3f}  n={v['n']}  limit={v['limit']}  "
              f"~{v['mean_tokens']} tok" + (f"  prompt-level {v['prompt_all']}"
                                            if "prompt_all" in v else ""))
    return row


def _last(label):
    rows = [r for r in map(json.loads, open(LEDGER)) if r["label"] == label]
    if not rows:
        raise SystemExit(f"no board2 row for {label}")
    return rows[-1]


def compare(a, b):
    """Paired difference b - a per tier, with its standard error and the items that flipped."""
    ra, rb = _last(a), _last(b)
    print(f"| tier | {a} | {b} | {b} - {a} | ± se | worse / better |")
    print("|---|---:|---:|---:|---:|---:|")
    for t in ra["tiers"]:
        ids = [i for i in ra["items"] if i in rb["items"] and suites_tier(i) == t]
        d = [rb["items"][i] - ra["items"][i] for i in ids]
        if not d:
            continue
        m = sum(d) / len(d)
        se = math.sqrt(sum((x - m) ** 2 for x in d) / max(1, len(d) - 1) / len(d))
        print(f"| {t} | {ra['tiers'][t]['score']:.3f} | {rb['tiers'][t]['score']:.3f} | "
              f"{m:+.3f} | {se:.3f} | {sum(x < 0 for x in d)} / {sum(x > 0 for x in d)} |")


def suites_tier(item_id):
    head = item_id.split("/")[0]
    return "humaneval" if head == "HumanEval" else head


def main():
    a = sys.argv[1:]
    think = "--think" in a
    a = [x for x in a if x != "--think"]
    if a[:1] == ["run"] and len(a) >= 3:
        opts = dict(zip(a[3::2], a[4::2]))
        run(a[1], a[2], tiers=opts["--tiers"].split(",") if "--tiers" in opts else None,
            slots=int(opts.get("--slots", 4)), think=think)
    elif a[:1] == ["compare"] and len(a) == 3:
        compare(a[1], a[2])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
