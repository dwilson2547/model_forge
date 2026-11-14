"""Teacher answers through an HTTP GPU job queue instead of a local llama-server.

  FORGE_QUEUE_URL=... FORGE_QUEUE_TOKEN=... \
  python -m distill.datagen <label> queue:<model> [--mix default] [--think]

An in-cluster queue can be reached through `kubectl port-forward`; polling survives the
forward dropping and being restarted.

Submits every prompt not yet answered as a *background* job (bulk work that runs only while
nothing interactive is queued), saves the batch ids beside the raw cache so a rerun resumes
polling instead of resubmitting, then polls until every job has finished, appending each answer
to the same raw cache the local path uses. Stop it any time: the jobs keep running in the queue,
and the next run picks up their answers.

The queue's API: POST /v1/jobs {model, inputs, background} -> {batch_id, job_ids};
GET /v1/batches/<id> -> {complete, jobs: [{id, status, result, error}]}. Each input is
{"messages", "max_new_tokens", "thinking"}; a result is {"text", "reasoning"?, "output_tokens",
"finish"}. Decoding there is greedy, so the sampling options of the local path don't apply.
"""
import json
import os
import time
import urllib.error
import urllib.request

BATCH = 1000      # the queue's per-request input limit


def _api(method, path, body=None, tries=30):
    """One call, retried for ~30 min across a dropped connection (e.g. a restarted port-forward).
    A POST is retried only when it never reached the server, so a batch isn't submitted twice."""
    for t in range(tries):
        try:
            return _call(method, path, body)
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            if method == "POST" and not isinstance(getattr(e, "reason", None), ConnectionRefusedError):
                raise
            print(f"  queue unreachable ({e}); retry {t + 1}/{tries} in 60s", flush=True)
            time.sleep(60)
    return _call(method, path, body)


def _call(method, path, body=None):
    url = os.environ["FORGE_QUEUE_URL"].rstrip("/") + path
    req = urllib.request.Request(url, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    if tok := os.environ.get("FORGE_QUEUE_TOKEN"):
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def run(model, todo, raw_f, done, state_p, think=False, poll=60):
    """Answer `todo` (prompt records) on queue `model`; write raw records to `raw_f`."""
    state = json.loads(state_p.read_text()) if state_p.exists() else {"batches": {}}
    pending = {pid for b in state["batches"].values() for pid in b.values()}
    new = [it for it in todo if it["id"] not in pending]
    for i in range(0, len(new), BATCH):
        chunk = new[i:i + BATCH]
        r = _api("POST", "/v1/jobs", {
            "model": model, "background": True,
            "inputs": [{"messages": [{"role": "user", "content": it["user"]}],
                        "max_new_tokens": it["max_tokens"], "thinking": think} for it in chunk]})
        state["batches"][str(r["batch_id"])] = {str(j): it["id"]
                                               for j, it in zip(r["job_ids"], chunk)}
        state_p.write_text(json.dumps(state))
        print(f"  submitted {i + len(chunk)}/{len(new)} as batch {r['batch_id']}", flush=True)
    t0, n = time.time(), 0
    while True:
        open_batches = 0
        for bid, jobs in state["batches"].items():
            if all(pid in done for pid in jobs.values()):
                continue
            b = _api("GET", f"/v1/batches/{bid}")
            for j in b["jobs"]:
                pid = jobs[str(j["id"])]
                if pid in done or j["status"] not in ("done", "failed", "cancelled"):
                    continue
                res = j.get("result") or {}
                rec = {"id": pid, "reasoning": (res.get("reasoning") or "").strip(),
                       "answer": (res.get("text") or "").strip(),
                       "stop": ("eos" if res.get("finish") == "stop" else
                                "limit" if res.get("finish") == "length" else j["status"]),
                       "tokens": res.get("output_tokens") or 0}
                if j.get("error"):
                    rec["error"] = j["error"][:300]
                raw_f.write(json.dumps(rec) + "\n")
                raw_f.flush()
                done[pid] = rec
                n += 1
            open_batches += not b["complete"]
        left = sum(pid not in done for b in state["batches"].values() for pid in b.values())
        print(f"  {n} answered this run, {left} still queued  {time.time() - t0:.0f}s", flush=True)
        if not left:
            return
        time.sleep(poll)
