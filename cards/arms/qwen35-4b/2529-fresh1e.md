# qwen35-4b / drop2-repair / 2529-fresh1e

The [25,29] slice repaired on the WHOLE mixed file for one epoch: fresh tokens instead of repeated epochs.

- **Status:** archived to the cold share
- **Design:** `docs/design.md` §0j.9
- **Trained:** 2026-11-12, desktop RTX 5080

## Recipe

| | |
|---|---|
| base | Qwen/Qwen3.5-4B |
| slice | drop2 [25,29]: layers 25 and 29 removed (fixed BlockInfluence scorer, §0j.1), the `drop2-explicit-2529` slice |
| method | QLoRA (NF4 base, AdamW8bit, cosine), merged to bf16 |
| data | `mixed.train.txt` (40.8M chars: code, then fineweb-edu, then wikitext, concatenated), whole file, 1% held out (`--val-frac 0.01`) |
| tokens seen | 9.53M |
| epochs | 1 |
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
| unrepaired wound, f16 | 12.823 | 13.954 | 3.61 |
| **this arm, f16** | 7.785 | 11.556 | 3.186 |
| this arm, Q4_K_M | 7.923 | 11.816 | 3.227 |

## Board v2

694 items, chat mode, greedy, thinking off, desktop 5080 (design §0j.13; ledger
`results/board2.jsonl`).

| arm (f16, thinking off) | humaneval | mbpp | gsm8k | ifeval | needle | composite |
|---|---:|---:|---:|---:|---:|---:|
| base | 0.872 | 0.740 | 0.960 | 0.894 | 0.983 | 0.890 |
| unrepaired [25,29] | 0.701 | 0.690 | 0.875 | 0.840 | 0.983 | 0.818 |
| 2529-fresh1e | 0.732 | 0.660 | 0.875 | 0.847 | 0.917 | 0.806 |
| 2529-distill | 0.707 | 0.660 | 0.915 | 0.867 | 0.950 | 0.820 |

Against the wound: flat on every tier but needle, which it loses (-0.067 ± 0.032, 4 items worse /
0 better). Its IFEval answers run long (~643 tokens, 18 limit hits vs base's 7).

## Verdict

**Best QLoRA floor of the project.** Fresh data is a live lever; held-out loss was still falling. On the M5 shelf. The comparator for the full-weight arms.
