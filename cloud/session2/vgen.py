"""Bulk generation against an OpenAI-compatible server (vLLM). Same raw format as distill.datagen.
  python vgen.py <label> <mix> <model-name> <concurrency> [url]"""
import json, os, sys, time, zlib, urllib.request
from concurrent.futures import ThreadPoolExecutor
label, mix, model, conc = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
url = sys.argv[5] if len(sys.argv) > 5 else "http://127.0.0.1:8001/v1/chat/completions"
D = os.path.expanduser("~/forge-data/distill/")
todo = [json.loads(l) for l in open(D + f"{mix}.prompts.jsonl")]
todo.sort(key=lambda r: -len(r["user"]))
def one(it):
    body = {"model": model, "messages": [{"role": "user", "content": it["user"]}],
            "max_tokens": it["max_tokens"], "temperature": 0.6, "top_p": 0.95, "top_k": 20,
            "seed": zlib.crc32(it["id"].encode()) % 2**31,
            "chat_template_kwargs": {"enable_thinking": False}}
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request(
            url, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600))
        c = r["choices"][0]; m = c["message"]
        return {"id": it["id"], "reasoning": (m.get("reasoning_content") or m.get("reasoning") or "").strip(),
                "answer": (m.get("content") or "").strip(),
                "stop": "eos" if c["finish_reason"] == "stop" else "limit" if c["finish_reason"] == "length" else c["finish_reason"],
                "tokens": r["usage"]["completion_tokens"], "prompt_tokens": r["usage"]["prompt_tokens"]}
    except Exception as e:
        return {"id": it["id"], "reasoning": "", "answer": "", "stop": f"error:{str(e)[:120]}", "tokens": 0}
t0, toks = time.time(), 0
with open(D + f"raw/{label}.{mix}.jsonl", "w") as f, ThreadPoolExecutor(conc) as pool:
    for n, rec in enumerate(pool.map(one, todo), 1):
        f.write(json.dumps(rec) + "\n"); f.flush(); toks += rec["tokens"]
        if n % 50 == 0 or n == len(todo):
            dt = time.time() - t0
            print(f"  {n}/{len(todo)}  {dt:.0f}s  {toks / dt:.0f} tok/s", flush=True)
