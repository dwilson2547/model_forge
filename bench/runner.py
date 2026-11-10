"""Suite runner: models through the task tiers + metric pack -> leaderboard.

Usage:
  python -m bench.runner label1=/path/a.gguf label2=/path/b.gguf ...

Per model: server up (f16/Q4 both fine; same-family comparisons stay valid),
tasks (coding docker-graded, instruct, needle), loop-threshold sweep, metrics
(decode t/s on a timed gen, TTFT on a 4k prompt, peak RSS, size GiB).
Writes rows to bench-results.jsonl and prints a markdown leaderboard.
Composite = mean(coding, instruct, needle); loop-threshold + speed reported
aside (per design §4.4: quality composite separate from the Pareto axes).
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.serve import LlamaServer           # noqa: E402
from bench.tasks import ALL_TASKS, loop_threshold, run_task  # noqa: E402

RESULTS = Path.home() / "m0" / "runs" / "bench-results.jsonl"
TTFT_PROMPT_FILE = Path.home() / "m0" / "data" / "fineweb.test.txt"


def metrics_pack(server, results_path_size):
    out = {}
    # decode speed: timed 128-token gen on a fixed prompt
    t0 = time.time()
    gen = server.completion("Write a short paragraph about tides.", max_tokens=128)
    dt = time.time() - t0
    out["gen_tps"] = round(gen["tokens"] / dt, 1) if gen.get("tokens") else None
    out["gen_wall_s"] = round(dt, 1)
    # TTFT on a ~4k-token prompt (streamed)
    prompt = (TTFT_PROMPT_FILE.read_text()[:16000]
              + "\n\nSummarize the above in one sentence.")
    r = server.completion(prompt, max_tokens=48, stream_first_token=True)
    out["ttft_4k_s"] = round(r["ttft"], 2) if r.get("ttft") else None
    rss = server.peak_rss_gib()
    out["peak_rss_gib"] = round(rss, 1) if rss else None
    out["size_gib"] = round(results_path_size / 2**30, 2)
    return out


def run_model(label, model_path, tiers=None):
    model_path = Path(model_path)
    row = {"label": label, "model": str(model_path)}
    if tiers:
        row["tiers"] = tiers
    with LlamaServer(model_path) as server:
        task_rows = []
        for task in ALL_TASKS:
            if tiers and task["group"] not in tiers:
                continue
            task_rows.append(run_task(server, task))
            print(f"  {task['id']:18s} {task_rows[-1]['score']:.2f} "
                  f"({task_rows[-1]['detail']})", flush=True)
        row["tasks"] = task_rows
        groups = {}
        for g in ("coding", "instruct", "needle"):
            scores = [t["score"] for t in task_rows if t["group"] == g]
            groups[g] = round(sum(scores) / len(scores), 3) if scores else None
        row["groups"] = groups
        row["composite"] = round(
            sum(v for v in groups.values() if v is not None) /
            max(1, len([v for v in groups.values() if v is not None])), 3)
        thresh, temps = loop_threshold(server)
        row["loop_threshold"] = thresh
        row["loop_rows"] = temps
        if not tiers:
            row["metrics"] = metrics_pack(server, model_path.stat().st_size)
        else:
            row["metrics"] = {}
    qp = (row.get("metrics") or {}).get("size_gib")
    row["quality_per_gib"] = round(row["composite"] / qp, 3) if qp else None
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def leaderboard(rows):
    hdr = ("| model | coding | instruct | needle | composite | loop-thresh | "
           "gen t/s | TTFT@4k | RSS | GiB | q/GiB |")
    line = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    print("\n" + hdr + "\n" + line)
    for r in rows:
        m = r["metrics"]
        print(f"| {r['label']:16s} | {r['groups'].get('coding')} | {r['groups'].get('instruct')} "
              f"| {r['groups'].get('needle')} | **{r['composite']}** | {r['loop_threshold']} "
              f"| {m.get('gen_tps')} | {m.get('ttft_4k_s')}s | {m.get('peak_rss_gib')} "
              f"| {m.get('size_gib')} | {r.get('quality_per_gib')} |")


def main():
    args = sys.argv[1:]
    tiers = None
    if args and args[0] == "--tiers":
        tiers = args[1].split(",")
        args = args[2:]
    if not args:
        raise SystemExit(__doc__)
    jobs = [a.split("=", 1) for a in args]
    rows = []
    for label, path in jobs:
        print(f"== suite: {label} ({path})", flush=True)
        rows.append(run_model(label, path, tiers=tiers))
    leaderboard(rows)


if __name__ == "__main__":
    main()
