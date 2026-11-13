# qwen35-4b / drop2-repair / iso-classic

Isolation pair, classic targets: seeded, identical diet draw and shuffles to iso-all; only the LoRA target list differs.

- **Status:** archived to the cold share
- **Design:** `docs/design.md` §0j
- **Trained:** 2026-11-11, desktop RTX 5080

## Recipe

| | |
|---|---|
| base | Qwen/Qwen3.5-4B |
| slice | drop2 [24,28]: layers 24 and 28 removed (old neighbour-cosine scorer), the `drop2-ranked` slice |
| method | QLoRA (NF4 base, AdamW8bit, cosine), merged to bf16 |
| data | `mixed.train.txt` (40.8M chars: code, then fineweb-edu, then wikitext, concatenated), 12M chars sampled in stripes across the whole file, chunk order shuffled per epoch |
| tokens seen | 5.38M |
| epochs | 2 |
| LoRA rank | 16 |
| lr | 1e-4 |
| ctx / batch | 512 / 4 |
| targets | classic: q/k/v/o + gate/up/down (no linear-attention mixers) |
| seed | 1234 |

## Floor

Perplexity, lower is better (`surgery/multicorpus.py`, 24 chunks; ledger `results/multicorpus.jsonl`).

| | wikitext | fineweb | code |
|---|---:|---:|---:|
| base, f16 | 9.33 | 10.44 | 3.17 |
| unrepaired wound, f16 | 12.82 | 13.55 | 3.57 |
| **this arm, f16** | 8.365 | 12.104 | 3.506 |
| this arm, Q4_K_M | 8.566 | 12.452 | 3.556 |

## Verdict

Ties iso-all: repairing the mixers buys nothing on the floor. Also measured the Q4_K_M penalty (+0.20/+0.35/+0.05).
