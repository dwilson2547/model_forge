# qwen35-4b / drop2-repair / 2529-distill

The [25,29] slice repaired by distillation from the unsliced base: fresh1e's recipe with the loss
swapped from next-token CE to KL(base || slice).

- **Status:** shelf pick for [25,29] (replaces fresh1e); archived to the cold share
- **Design:** `docs/design.md` §0j.12
- **Trained:** 2026-11-13, desktop RTX 5080 (2.1 h)

## Recipe

| | |
|---|---|
| base | Qwen/Qwen3.5-4B |
| slice | drop2 [25,29]: layers 25 and 29 removed, the `drop2-explicit-2529` slice |
| method | QLoRA (NF4 base, AdamW8bit, cosine), merged to bf16 |
| objective | `--distill`: per-position KL(teacher ‖ student) at T=1; teacher = the bf16 base, frozen, in the same process |
| data | `mixed.train.txt`, whole file, 1% held out (`--val-frac 0.01`): the same text as 2529-fresh1e |
| tokens seen | 9.53M |
| epochs | 1 |
| LoRA rank | 16 |
| lr | 1e-4 |
| ctx / batch | 512 / 2 x accum 2 (fresh1e's batch 4 spilled past 16 GB with the teacher resident) |
| targets | all: classic + linear-attention mixers |
| seed | 1234 |

Held-out KL fell 0.323 → 0.127 and was still falling at the end of the cosine.

## Floor

KL divergence from the base, lower is better (`surgery/kld.py`, 24 chunks x 512; ledger
`results/kld.jsonl`). PPL from the same run.

| f16 | wikitext PPL / KLD | fineweb PPL / KLD / top-1 | code PPL / KLD |
|---|---|---|---|
| base | 9.33 / 0 | 10.44 / 0 | 3.17 / 0 |
| unrepaired [25,29] | 12.81 / 0.316 | 13.95 / 0.300 / 76.6% | 3.610 / 0.149 |
| 2529-fresh1e | 7.78 / 0.561 | 11.55 / 0.223 / 80.3% | 3.185 / 0.212 |
| **this arm** | 9.99 / **0.141** | 11.80 / **0.147** / **84.2%** | 3.280 / **0.067** |

The 99th-percentile KLD falls with it (fineweb 1.84 vs fresh1e 2.40; code 0.80 vs 2.39).

## Board

Chat mode, desktop 5080, f16 (design §0j.12; ledger `results/bench.jsonl`, labels `@5080`).

| | coding | instruct | needle | composite | loop-thresh |
|---|---:|---:|---:|---:|---:|
| base | 1.0 | 0.6 | 1.0 | 0.867 | 0.7 |
| unrepaired [25,29] | 0.75 | 0.6 | 1.0 | 0.783 | 0.7 |
| 2529-fresh1e | 1.0 | 0.6 | 1.0 | 0.867 | 0.7 |
| **this arm** | 1.0 | 0.6 | 1.0 | **0.867** | 0.7 |

## Board v2

694 items, chat mode, greedy, thinking off, desktop 5080 (design §0j.13; ledger
`results/board2.jsonl`). Two base runs differ by 0.012 on IFEval and 0 elsewhere.

| arm (f16, thinking off) | humaneval | mbpp | gsm8k | ifeval | needle | composite |
|---|---:|---:|---:|---:|---:|---:|
| base | 0.872 | 0.740 | 0.960 | 0.894 | 0.983 | 0.890 |
| unrepaired [25,29] | 0.701 | 0.690 | 0.875 | 0.840 | 0.983 | 0.818 |
| 2529-fresh1e | 0.732 | 0.660 | 0.875 | 0.847 | 0.917 | 0.806 |
| 2529-distill | 0.707 | 0.660 | 0.915 | 0.867 | 0.950 | 0.820 |

Against the wound it gains GSM8K +0.040 (± 0.021, 13 items better / 5 worse) and IFEval +0.027
(± 0.022), loses needle -0.033, and is flat on code. It closes about half the wound's GSM8K loss
and none of its HumanEval loss (-0.165 vs base).

## Verdict

Board v2 (2026-11-14) qualifies the verdict below: the repair recovers some math and instruct
behaviour but none of the code, so KLD-close on raw text is not behaviour-close.

**Closest to the base of any repair, on every corpus, by a wide margin, and it ties base on the
board.** Its PPL is higher than fresh1e's because it no longer specializes on the training diet.
The board cannot tell it from fresh1e (both tie base), so KLD decides: this arm replaces fresh1e on
the shelf.
