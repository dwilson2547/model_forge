# qwen35-4b / drop2-repair / wikirep

The first repair of the drop2 wound, on wikitext only. Beats the base on wikitext and wrecks fineweb: the domain-trap example that proved single-corpus PPL lies.

- **Status:** archived to the cold share
- **Design:** `docs/design.md` §0c, §0d, §0g, §0h
- **Trained:** 2026-11-08, desktop RTX 5080
- **Old folder names:** desktop `drop2-repair/` then `v1/`; Altra `drop2-repaired/`

## Recipe

| | |
|---|---|
| base | Qwen/Qwen3.5-4B |
| slice | drop2 [24,28]: layers 24 and 28 removed (old neighbour-cosine scorer), the `drop2-ranked` slice |
| method | QLoRA (NF4 base, AdamW8bit, cosine), merged to bf16 |
| data | wikitext-103 train, ~4.8M tokens |
| tokens seen | ~4.8M |
| epochs | 2 |
| LoRA rank | 16 |
| lr | not recorded (trainer default then) |
| ctx / batch | 512 / 4 |
| targets | classic: q/k/v/o + gate/up/down (no linear-attention mixers) |

## Floor

Perplexity, lower is better (`surgery/multicorpus.py`, 24 chunks; ledger `results/multicorpus.jsonl`).

| | wikitext | fineweb | code |
|---|---:|---:|---:|
| base, f16 | 9.33 | 10.44 | 3.17 |
| unrepaired wound, f16 | 12.82 | 13.55 | 3.57 |
| **this arm, f16** | 8.163 | 19.446 | 3.898 |
| this arm, Q4_K_M | 8.34 | — | — |

## Verdict

Superseded. Loops forever in raw mode (loop-threshold 0.0, raw board 0.40); chat mode partly recovers it (0.667). Kept only as the reference specialisation failure.
