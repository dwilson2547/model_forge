"""Teacher answers as chat-format plain text, a corpus for surgery/damage_scan.py.

  python -m distill.chattext <dataset.jsonl> <out.txt> [--kinds code,codemore,math,mathmore] [--max 400]

The raw-text scan (fineweb, code files) ranked layer sets the opposite way round from board v2
(design §0j.15). This writes the text the board is closer to: prompts and teacher answers in
the chat template, restricted to the task kinds the board grades. llama-perplexity tokenizes
the template's markers as plain text, not as special tokens, so this is chat-shaped text, not
an exact replay of the chat path.
"""
import json
import sys
from pathlib import Path

BASE = Path("~/forge/base/Qwen/Qwen3.5-4B").expanduser()


def main():
    a = sys.argv[1:]
    if len(a) < 2:
        raise SystemExit(__doc__)
    o = dict(zip(a[2::2], a[3::2]))
    kinds = set(o.get("--kinds", "code,codemore,math,mathmore").split(","))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(o.get("--base", str(BASE)))
    recs = [r for r in map(json.loads, open(a[0])) if r["kind"] in kinds]
    recs.sort(key=lambda r: r["id"])
    by = {}
    for r in recs:                                   # the same number of records per kind
        by.setdefault(r["kind"], []).append(r)
    n = min(int(o.get("--max", 400)) // len(by), min(map(len, by.values())))
    out = []
    for k in sorted(by):
        for r in by[k][:n]:
            out.append(tok.apply_chat_template(
                [{"role": "user", "content": r["user"]},
                 {"role": "assistant", "reasoning_content": r["reasoning"], "content": r["answer"]}],
                tokenize=False, add_generation_prompt=False))
    Path(a[1]).write_text("\n".join(out))
    print(f"[chattext] {a[1]}: {n} records x {len(by)} kinds {sorted(by)}, "
          f"{sum(map(len, out)):,} chars")


if __name__ == "__main__":
    main()
