"""Answer a prompt mix through an OpenAI-compatible server (vLLM), in datagen's raw format.

  python -m distill.vgen <label> <mix> <served-model-name> [--conc 160] [--url http://127.0.0.1:8001]
                         [--temp 0.6] [--top-p 0.95] [--top-k 20] [--think] [--max-tokens N]
  then: python -m distill.datagen export <label> --mix <mix>

llama.cpp serves a few interactive streams; for bulk generation vLLM is 4-5x faster on the
same card (docs/datasets.md, 2026-11-15). Start the server first, e.g.
  vllm serve <hf dir> --served-model-name q27 --dtype bfloat16 --max-model-len 20480 \
      --gpu-memory-utilization 0.95 --port 8001
(vLLM needs `ninja` on PATH for its sampling kernels.) Writes
$FORGE_DATA/distill/raw/<label>.<mix>.jsonl and resumes from it. Without a --reasoning-parser
on the server, thinking text arrives inline at the start of `answer`.

Ported from the cloud session 2 script (cloud/session2/vgen.py), which produced the
2026-11-15 numbers; this module itself has not been run ⚠ unverified.
"""
import json
import sys
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distill import prompts  # noqa: E402


def main():
    a = sys.argv[1:]
    think = "--think" in a
    a = [x for x in a if x != "--think"]
    if len(a) < 3:
        raise SystemExit(__doc__)
    label, mix, model = a[:3]
    o = dict(zip(a[3::2], a[4::2]))
    url = o.get("--url", "http://127.0.0.1:8001") + "/v1/chat/completions"
    if think:
        label += "-think"
    todo_all = [json.loads(l) for l in open(prompts.DIR / f"{mix}.prompts.jsonl")]
    raw_p = prompts.DIR / "raw" / f"{label}.{mix}.jsonl"
    raw_p.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in open(raw_p)} if raw_p.exists() else set()
    todo = sorted((r for r in todo_all if r["id"] not in done), key=lambda r: -len(r["user"]))
    print(f"[vgen] {label} on {mix}: {len(done)} cached, {len(todo)} to generate", flush=True)

    def one(it):
        body = {"model": model, "messages": [{"role": "user", "content": it["user"]}],
                "max_tokens": int(o.get("--max-tokens", it["max_tokens"])),
                "temperature": float(o.get("--temp", 0.6)), "top_p": float(o.get("--top-p", 0.95)),
                "top_k": int(o.get("--top-k", 20)), "seed": zlib.crc32(it["id"].encode()) % 2**31,
                "chat_template_kwargs": {"enable_thinking": think}}
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(
                url, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600))
            c, m = r["choices"][0], r["choices"][0]["message"]
            stop = {"stop": "eos", "length": "limit"}.get(c["finish_reason"], c["finish_reason"])
            return {"id": it["id"], "reasoning": (m.get("reasoning_content") or "").strip(),
                    "answer": (m.get("content") or "").strip(), "stop": stop,
                    "tokens": r["usage"]["completion_tokens"]}
        except Exception as e:  # noqa: BLE001 - recorded per item; a rerun does not retry it
            return {"id": it["id"], "reasoning": "", "answer": "", "stop": f"error:{str(e)[:120]}",
                    "tokens": 0}

    t0, toks = time.time(), 0
    with raw_p.open("a") as f, ThreadPoolExecutor(int(o.get("--conc", 160))) as pool:
        for n, rec in enumerate(pool.map(one, todo), 1):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            toks += rec["tokens"]
            if n % 50 == 0 or n == len(todo):
                dt = time.time() - t0
                print(f"  {n}/{len(todo)}  {dt:.0f}s  {toks / dt:.0f} tok/s", flush=True)


if __name__ == "__main__":
    main()
