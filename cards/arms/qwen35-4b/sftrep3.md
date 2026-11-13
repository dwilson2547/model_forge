# qwen35-4b / drop2-repair / sftrep3

SFT v3: the private incident-triage corpus plus continuation pairs, rendered so the model learns to close its think block. Full card: `cards/qwen35-4b-drop2-sftv3-triage.md`.

- **Status:** archived to the cold share
- **Design:** `docs/design.md` §0i
- **Trained:** 2026-11-11, desktop RTX 5080
- **Old folder names:** desktop `drop2-repair-sft3/`; Altra `drop2-repair-sft3-hf/`, `lmstudio-out/sftrep3`

## Recipe

| | |
|---|---|
| base | Qwen/Qwen3.5-4B |
| slice | drop2 [24,28]: layers 24 and 28 removed (old neighbour-cosine scorer), the `drop2-ranked` slice |
| method | QLoRA (NF4 base, AdamW8bit, cosine), merged to bf16 |
| data | private incident-triage corpus (`surgery/corpus_triage.py`; 3416 examples, format `sft-chat-v3`) |
| tokens seen | 1.20M |
| epochs | 2 |
| LoRA rank | 16 |
| lr | 1e-4 |
| ctx / batch | 6144 / 1 |
| targets | classic: q/k/v/o + gate/up/down (no linear-attention mixers) |

## Floor

Perplexity, lower is better (`surgery/multicorpus.py`, 24 chunks; ledger `results/multicorpus.jsonl`).

| | wikitext | fineweb | code |
|---|---:|---:|---:|
| base, f16 | 9.33 | 10.44 | 3.17 |
| unrepaired wound, f16 | 12.82 | 13.55 | 3.57 |
| this arm | not floored (judged on the task board) | | |

Task board: chat board composite 0.733 (coding 1.0, instruct 0.2, needle 1.0); held-out incident 46 diagnosed correctly.

## Verdict

Best task-level arm of the SFT line. Confabulates evidence without a bundle; v4 diet restores system prompts. Trained on homelab-private data: never publish these weights. LM Studio build in `lmstudio-builds/sftrep3`.
