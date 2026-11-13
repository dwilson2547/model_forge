# qwen35-4b / drop2-repair / sftrep2

SFT v2: rotated prompts plus half paired continuations (the user turn carries the opening 32 tokens).

- **Status:** archived to the cold share
- **Design:** `docs/design.md` §0h
- **Trained:** 2026-11-10, desktop RTX 5080
- **Old folder names:** desktop `drop2-repair-sft2/`; Altra `drop2-repair-sft2-hf/`, `lmstudio-out/sftrep2`

## Recipe

| | |
|---|---|
| base | Qwen/Qwen3.5-4B |
| slice | drop2 [24,28]: layers 24 and 28 removed (old neighbour-cosine scorer), the `drop2-ranked` slice |
| method | QLoRA (NF4 base, AdamW8bit, cosine), merged to bf16 |
| data | `mixed.train.txt` (40.8M chars: code, then fineweb-edu, then wikitext, concatenated) as chat pairs (format `sft-chat-v2`) |
| tokens seen | 8.61M |
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

Task board: chat board composite 0.333 (coding 0, instruct 0, needle 1.0).

## Verdict

Fixed v1's semantic collapse, but the think block never closes: 'thought for a moment then died' in LM Studio. LM Studio build in `lmstudio-builds/sftrep2`.
