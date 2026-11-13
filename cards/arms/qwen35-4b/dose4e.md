# qwen35-4b / drop2-repair / dose4e

iso-all repeated for 4 epochs over the same 12M-char draw. Train loss improves while held-out PPL explodes.

- **Status:** deleted 2026-11-13 (overfit negative result; numbers in the ledger, recipe reproducible)
- **Design:** `docs/design.md` §0j.3
- **Trained:** 2026-11-11, desktop RTX 5080

## Recipe

| | |
|---|---|
| base | Qwen/Qwen3.5-4B |
| slice | drop2 [24,28]: layers 24 and 28 removed (old neighbour-cosine scorer), the `drop2-ranked` slice |
| method | QLoRA (NF4 base, AdamW8bit, cosine), merged to bf16 |
| data | `mixed.train.txt` (40.8M chars: code, then fineweb-edu, then wikitext, concatenated), 12M chars sampled in stripes across the whole file, chunk order shuffled per epoch |
| tokens seen | 10.76M |
| epochs | 4 |
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
| **this arm, f16** | 10.284 | 15.386 | 4.079 |
| this arm, Q4_K_M | 10.445 | 15.636 | 4.187 |

## Verdict

Overfit: fineweb worse than the unrepaired wound. Tested repeated epochs, not fresh tokens (see 2529-fresh1e).
