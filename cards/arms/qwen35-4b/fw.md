# qwen35-4b / drop2-repair / fw

drop2 repaired on fineweb-edu only. Heals broadly (74% of the wikitext wound without seeing wikitext) but leaves code untouched.

- **Status:** archived to the cold share
- **Design:** `docs/design.md` §0d
- **Trained:** 2026-11-08, desktop RTX 5080

## Recipe

| | |
|---|---|
| base | Qwen/Qwen3.5-4B |
| slice | drop2 [24,28]: layers 24 and 28 removed (old neighbour-cosine scorer), the `drop2-ranked` slice |
| method | QLoRA (NF4 base, AdamW8bit, cosine), merged to bf16 |
| data | fineweb-edu, ~4.3M tokens |
| tokens seen | ~4.3M |
| epochs | 2 |
| LoRA rank | 16 |
| lr | not recorded |
| ctx / batch | 512 / 4 |
| targets | classic: q/k/v/o + gate/up/down (no linear-attention mixers) |

## Floor

Perplexity, lower is better (`surgery/multicorpus.py`, 24 chunks; ledger `results/multicorpus.jsonl`).

| | wikitext | fineweb | code |
|---|---:|---:|---:|
| base, f16 | 9.33 | 10.44 | 3.17 |
| unrepaired wound, f16 | 12.82 | 13.55 | 3.57 |
| **this arm, f16** | 10.22 | 12.466 | 3.578 |

## Verdict

Superseded by codefw: code needs code.
