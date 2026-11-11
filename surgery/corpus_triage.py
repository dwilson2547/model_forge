"""Extract the incident-triage SFT corpus from the incident-operator Postgres DB.

Run on the desktop with the DB port-forwarded:
    kubectl port-forward svc/postgres -n postgres 15432:5432 &
    PGPASSWORD=... python surgery/corpus_triage.py --out ~/forge-gpu/data/ \
        --mixed /home/daniel/forge-gpu/data/mixed.train.txt

Output: sft-v3-dataset.jsonl - records {"user", "reasoning", "answer"} in the native
assistant format (training adds the think-close structure):
  - triage pairs: incident bundle (real model_turns.messages) -> reason (think block)
    + result JSON (answer). The gold triage corpus.
  - knowledge-layer pairs from the incident-operator repo: runbooks, operations, primer
    -> "what does this cover" summaries.
  - continuation pairs from the mixed repair diet (v2-style) so the repair floor stays.
The extracted data is homelab-private: never publish, never commit the JSONL.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def psql(query, dbname="incident_operator"):
    env = dict(os.environ)
    r = subprocess.run(
        ["psql", "-h", "127.0.0.1", "-p", "15432", "-U", "incident_operator",
         "-d", dbname, "-tA", "-c", query],
        capture_output=True, text=True, env=env, timeout=60)
    if r.returncode:
        raise SystemExit(f"psql failed: {r.stderr}")
    return r.stdout.strip()


def triage_pairs(holdout_ids):
    """Every completed turn: the real user bundle -> reason + result JSON."""
    # each row is one JSON line (row_to_json escapes embedded newlines)
    rows = psql("select jsonb_build_object('incident_id', incident_id, 'session', session, "
                "'messages', messages, 'result', result)::text "
                "from model_turns where status='done' order by incident_id, id")
    out = []
    for line in rows.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        inc = int(row["incident_id"])
        msgs = row["messages"]
        res = row["result"]
        user = msgs[-1]["content"] if msgs and isinstance(msgs[-1], dict) else None
        if not user:
            continue
        if row["session"] == "triage":
            # phase-1 protocol: freeform prose triage answer
            ans = (res or {}).get("text")
            if ans:
                out.append({"kind": "turn-triage", "incident": inc,
                            "holdout": inc in holdout_ids, "user": user,
                            "reasoning": "Analyze the bundle: where it failed, "
                                         "likely cause, what to check next.",
                            "answer": ans.strip(),
                            "repeat": 4 if inc not in holdout_ids else 0})
        else:
            rj = (res or {}).get("json") or {}
            why = rj.get("why") if row["session"] == "remediation" else \
                (rj.get("reason") or rj.get("findings", {}).get("summary", ""))
            if rj and (rj.get("action") or rj.get("requests") or rj.get("why")):
                out.append({"kind": f"turn-{row['session']}", "incident": inc,
                            "holdout": inc in holdout_ids, "user": user,
                            "reasoning": why or "Weigh the evidence and conclude.",
                            "answer": json.dumps(rj, indent=2),
                            "repeat": 4 if inc not in holdout_ids else 0})
    return out


def knowledge_pairs(repo, holdout_ids):
    """Runbooks, operations, primer -> what-is-this-doc pairs (domain vocab, cheap)."""
    out = []
    rb = sorted((repo / "helm/incident-operator/files/runbooks").glob("*.md"))
    ops = sorted((repo / "helm/incident-operator/files/operations").glob("*.yaml"))
    primer = repo / "helm/incident-operator/files/primer.md"
    for f in rb:
        body = f.read_text()[:4000]
        out.append({"kind": "runbook",
                    "user": f"Summarize what this runbook covers and the checks it "
                            f"uses:\n\n{body}",
                    "reasoning": f"Reading the {f.stem} runbook: identify triggering "
                                 f"alert, target, checks, and remediation steps.",
                    "answer": _first_para(body) or f"This is the {f.stem} runbook.",
                    "repeat": 3})
    for f in ops:
        body = f.read_text()[:2500]
        out.append({"kind": "operation",
                    "user": f"What read-only check does this operation run and what "
                            f"does it return?\n\n{body}",
                    "reasoning": f"Parsing operation {f.stem}: verb, target, result shape.",
                    "answer": _first_para(body) or f"This is the {f.stem} operation.",
                    "repeat": 3})
    if primer.exists():
        body = primer.read_text()
        out.append({"kind": "primer", "user": f"Summarize the incident triage protocol:\n\n{body}",
                    "reasoning": "Summarizing the triage protocol: rounds, allowed actions, JSON schema.",
                    "answer": _first_para(body) or "The triage protocol.", "repeat": 3})
    return out


def _first_para(text):
    paras = [p.strip() for p in text.split("\n\n") if p.strip() and not p.strip().startswith("#")]
    return paras[0][:600] if paras else ""


def continuation_pairs(mixed_path, n_chars, tok_estimate=4):
    """v2-style pairs from the mixed repair diet (keeping the repair floor)."""
    raw = Path(mixed_path).read_text()[:n_chars]
    out, pos, i, chunk = [], 0, 0, 2000
    GENERIC = ["Continue the following text in the same style.",
               "Write the next part of this text.",
               "Continue writing the document.",
               "Provide the continuation of this passage."]
    while pos + chunk < len(raw):
        block = raw[pos: pos + chunk]
        head, rest = block[:256], block[256:]
        if i % 2:
            out.append({"kind": "cont-paired", "user": f"Continue this text:\n\n{head}",
                        "reasoning": "Continuing the passage in the same style.",
                        "answer": rest, "repeat": 1})
        else:
            out.append({"kind": "cont-generic", "user": GENERIC[i // 2 % len(GENERIC)],
                        "reasoning": "Continuing the passage in the same style.",
                        "answer": block, "repeat": 1})
        pos += chunk
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output dir for sft-v3-dataset.jsonl")
    ap.add_argument("--mixed", default="/home/daniel/forge-gpu/data/mixed.train.txt")
    ap.add_argument("--repo",
                    default="/home/daniel/documents/workspace/infra/incident-operator")
    ap.add_argument("--mixed-chars", type=int, default=6_000_000)
    ap.add_argument("--holdout", default="10,28,46",
                    help="incident ids held out of training for eval")
    ap.add_argument("--holdout-out", default=None, help="where to write holdout bundles")
    args = ap.parse_args()

    holdout_ids = {int(x) for x in args.holdout.split(",") if x.strip()}
    triage = triage_pairs(holdout_ids)
    know = knowledge_pairs(Path(args.repo), holdout_ids)
    cont = continuation_pairs(args.mixed, args.mixed_chars)
    train = [t for t in triage if not t["holdout"]] + know + cont
    hold = [t for t in triage if t["holdout"]]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ds_path = out / "sft-v3-dataset.jsonl"
    with ds_path.open("w") as f:
        n = 0
        for rec in train:
            for _ in range(max(1, rec.get("repeat", 1))):
                f.write(json.dumps({k: rec[k] for k in ("kind", "user", "reasoning", "answer")}) + "\n")
                n += 1
    if hold:
        hp = out / "sft-v3-holdout.jsonl"
        with hp.open("w") as f:
            for rec in hold:
                f.write(json.dumps({k: rec[k] for k in ("kind", "incident", "user", "reasoning", "answer")}) + "\n")
    counts = {}
    for rec in train:
        counts[rec["kind"]] = counts.get(rec["kind"], 0) + max(1, rec.get("repeat", 1))
    print(f"dataset: {n} records -> {ds_path}")
    print("by kind:", json.dumps(counts, indent=1))
    print(f"holdout: {len(hold)} records held out (incidents {sorted(holdout_ids)})")


if __name__ == "__main__":
    main()
