# qwen35-4b / drop2-repair / mixfull

drop2 repaired on the whole mixed file, one epoch. Buys wikitext, not breadth: fineweb plateaus at the same ~12.4 as codefw.

- **Status:** archived to the cold share
- **Design:** `docs/design.md` §0d, §0g, §0h
- **Trained:** 2026-11-08, desktop RTX 5080
- **Old folder names:** desktop `drop2-repair-mixed-full/`; Altra `drop2-mixfull-repaired/`, `mixfull-recovered/`, `publish/mixfull`

## Recipe

| | |
|---|---|
| base | Qwen/Qwen3.5-4B |
| slice | drop2 [24,28]: layers 24 and 28 removed (old neighbour-cosine scorer), the `drop2-ranked` slice |
| method | QLoRA (NF4 base, AdamW8bit, cosine), merged to bf16 |
| data | `mixed.train.txt` (40.8M chars: code, then fineweb-edu, then wikitext, concatenated), whole file |
| tokens seen | 9.46M |
| epochs | 1 |
| LoRA rank | 16 |
| lr | 2e-4 |
| ctx / batch | 512 / 4 |
| targets | classic: q/k/v/o + gate/up/down (no linear-attention mixers) |

## Floor

Perplexity, lower is better (`surgery/multicorpus.py`, 24 chunks; ledger `results/multicorpus.jsonl`).

| | wikitext | fineweb | code |
|---|---:|---:|---:|
| base, f16 | 9.33 | 10.44 | 3.17 |
| unrepaired wound, f16 | 12.82 | 13.55 | 3.57 |
| **this arm, f16** | 8.423 | 12.387 | 3.363 |

## Verdict

The published one: HF `dwilson2547/Qwen3.5-4B-drop2-mixed-repair` (+ `-GGUF`). Chat board 0.594. Superseded on the floor by 2529-fresh1e.
