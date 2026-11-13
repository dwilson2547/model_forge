# qwen35-4b / drop2-repair / r2fix-lr1e4

First run after review round 2's trainer fixes: striped diet and mixer targets changed at once, so it confounds both.

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
| tokens seen | 6.14M |
| epochs | 2 |
| LoRA rank | 16 |
| lr | 1e-4 |
| ctx / batch | 512 / 4 |
| targets | all: classic + linear-attention mixers (in_proj_qkv/z/a/b, out_proj) |
| seed | unseeded |

## Floor

Perplexity, lower is better (`surgery/multicorpus.py`, 24 chunks; ledger `results/multicorpus.jsonl`).

| | wikitext | fineweb | code |
|---|---:|---:|---:|
| base, f16 | 9.33 | 10.44 | 3.17 |
| unrepaired wound, f16 | 12.82 | 13.55 | 3.57 |
| this arm, Q4_K_M | 8.66 | 12.715 | 3.626 |

## Verdict

Inconclusive by design; replaced by the seeded iso pair. Its gap to iso-all measured seed noise (~±0.27 fineweb).
