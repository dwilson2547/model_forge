# qwen35-4b / drop2-repair / sftrep1

First SFT-format repair: chat template + completion-only loss, every example behind ONE fixed generic prompt.

- **Status:** archived to the cold share
- **Design:** `docs/design.md` §0h
- **Trained:** 2026-11-10, desktop RTX 5080
- **Old folder names:** desktop `drop2-repair-sft/`; Altra `drop2-repair-sft-hf/`

## Recipe

| | |
|---|---|
| base | Qwen/Qwen3.5-4B |
| slice | drop2 [24,28]: layers 24 and 28 removed (old neighbour-cosine scorer), the `drop2-ranked` slice |
| method | QLoRA (NF4 base, AdamW8bit, cosine), merged to bf16 |
| data | `mixed.train.txt` (40.8M chars: code, then fineweb-edu, then wikitext, concatenated) as chat pairs (format `sft-chat`) |
| tokens seen | 6.69M |
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
| this arm | not floored (judged on the task board) | | |

Task board: chat board composite 0.0 (coding 0, instruct 0, needle 0).

## Verdict

Collapsed: learned 'any user turn = continue a corpus'. Perfect EOS, zero instruction-following. Negative result.
