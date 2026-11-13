# qwen35-4b / drop2-repair / lr1e4

The codefw recipe at r16 with lr 1e-4 (half the default then). Best arm of the old [24,28] study; made 1e-4 the trainer default.

- **Status:** archived to the cold share
- **Design:** `docs/design.md` §0f
- **Trained:** 2026-11-09, desktop RTX 5080

## Recipe

| | |
|---|---|
| base | Qwen/Qwen3.5-4B |
| slice | drop2 [24,28]: layers 24 and 28 removed (old neighbour-cosine scorer), the `drop2-ranked` slice |
| method | QLoRA (NF4 base, AdamW8bit, cosine), merged to bf16 |
| data | `mixed.train.txt` (40.8M chars: code, then fineweb-edu, then wikitext, concatenated), head-capped at 12M chars, so ~78% llama.cpp C++ + fineweb (§0j fix 2) |
| tokens seen | 6.43M |
| epochs | 2 |
| LoRA rank | 16 |
| lr | 1e-4 |
| ctx / batch | 512 / 4 |
| targets | classic: q/k/v/o + gate/up/down (no linear-attention mixers) |

## Floor

Perplexity, lower is better (`surgery/multicorpus.py`, 24 chunks; ledger `results/multicorpus.jsonl`).

| | wikitext | fineweb | code |
|---|---:|---:|---:|
| base, f16 | 9.33 | 10.44 | 3.17 |
| unrepaired wound, f16 | 12.82 | 13.55 | 3.57 |
| **this arm, f16** | 9.873 | 12.217 | 3.333 |

## Verdict

Best [24,28] floor on the old diet. Superseded by the [25,29] arms.
