# qwen35-4b / drop2-repair / iso-all

Isolation pair, all targets (mixers included). Also the arm used for the NF4-vs-bf16 merge-shift test.

- **Status:** archived to the cold share
- **Design:** `docs/design.md` §0j, §0j.2
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
| targets | all: classic + linear-attention mixers (in_proj_qkv/z/a/b, out_proj) |
| seed | 1234 |

## Floor

Perplexity, lower is better (`surgery/multicorpus.py`, 24 chunks; ledger `results/multicorpus.jsonl`).

| | wikitext | fineweb | code |
|---|---:|---:|---:|
| base, f16 | 9.33 | 10.44 | 3.17 |
| unrepaired wound, f16 | 12.82 | 13.55 | 3.57 |
| **this arm, f16** | 8.359 | 12.221 | 3.509 |
| this arm, Q4_K_M | 8.565 | 12.448 | 3.566 |

## Verdict

Mixer hypothesis dead; merged bf16 beats NF4+adapter on every corpus (merge-shift dead). The [24,28] comparator for 2529-pass1.
