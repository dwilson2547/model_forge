# qwen35-4b / full-repair-2529 / lr1e-6

Full-weight repair (no LoRA) of the [25,29] slice at lr 1e-6, on a rented A100: fp32 master weights, bf16 autocast, AdamW8bit. Same data/split/seed as 2529-fresh1e.

- **Status:** archived to the cold share
- **Design:** `docs/design.md` §0j.9 (cloud session 1, Run 1)
- **Trained:** 2026-11-12, Brev A100-SXM4-80GB (cloud session 1, 2026-11-12)
- **Old folder names:** desktop `cloud-session1/run1-full-lr1e-6/`

## Recipe

| | |
|---|---|
| base | Qwen/Qwen3.5-4B |
| slice | drop2 [25,29]: layers 25 and 29 removed (fixed BlockInfluence scorer, §0j.1), the `drop2-explicit-2529` slice |
| method | full fine-tune |
| data | `mixed.train.txt` (40.8M chars: code, then fineweb-edu, then wikitext, concatenated), whole file, 1% held out |
| tokens seen | ~9.5M |
| epochs | 1 |
| LoRA rank | full weights |
| lr | 1e-6 |
| ctx / batch | 512 / 4 |
| targets | all weights |
| seed | 1234 |

## Floor

Perplexity, lower is better (`surgery/multicorpus.py`, 24 chunks; ledger `results/multicorpus.jsonl`).

| | wikitext | fineweb | code |
|---|---:|---:|---:|
| base, f16 | 9.33 | 10.44 | 3.17 |
| unrepaired wound, f16 | 12.823 | 13.954 | 3.61 |
| **this arm, f16** | 7.685 | 11.39 | 3.182 |

*CUDA llama.cpp e95426f; parity with the Altra within 0.006 at f16.*

## Verdict

Ties fresh1e on code, slightly better prose (inside the noise band). Net: full weights tie LoRA, so adapter capacity was not the wall.
