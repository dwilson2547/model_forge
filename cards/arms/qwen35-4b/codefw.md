# qwen35-4b / drop2-repair / codefw

drop2 repaired on the head of the mixed file: mostly C++ plus fineweb. The 'code is the multivitamin' arm: better than fw-only on every corpus.

- **Status:** archived to the cold share
- **Design:** `docs/design.md` §0d, §0g, §0h
- **Trained:** 2026-11-08, desktop RTX 5080
- **Old folder names:** desktop `drop2-repair-mixed/` (named for the file, but it is the codefw arm); Altra `drop2-codefw-repaired/`, `codefw-recovered/`

## Recipe

| | |
|---|---|
| base | Qwen/Qwen3.5-4B |
| slice | drop2 [24,28]: layers 24 and 28 removed (old neighbour-cosine scorer), the `drop2-ranked` slice |
| method | QLoRA (NF4 base, AdamW8bit, cosine), merged to bf16 |
| data | `mixed.train.txt` (40.8M chars: code, then fineweb-edu, then wikitext, concatenated), head-capped at 12M chars, so ~78% llama.cpp C++ + fineweb (§0j fix 2) |
| tokens seen | 3.21M |
| epochs | 1 |
| LoRA rank | 16 |
| lr | 2e-4 |
| ctx / batch | 512 / 4 |
| targets | classic: q/k/v/o + gate/up/down (no linear-attention mixers) |

## Floor

Perplexity, lower is better (`surgery/multicorpus.py`, 24 chunks; ledger `results/multicorpus.jsonl`).

| | wikitext | fineweb | code |
|---|---:|---:|---:|
| base, f16 | 9.33 | 10.44 | 3.17 |
| unrepaired wound, f16 | 12.82 | 13.55 | 3.57 |
| **this arm, f16** | 9.97 | 12.341 | 3.282 |

## Verdict

Best balanced repair of the first diet study. Boards: raw 0.483, chat 0.733 (ties unrepaired). Rambles in raw mode (EOS discipline eroded): the reason the SFT line exists. LM Studio build in `lmstudio-builds/codefw`.
