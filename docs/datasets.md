# Prompt pools for teacher datasets

The distill workflow (`distill/`) only takes the **prompts** from a dataset. The teacher writes
the answers, and with `--distill` the student learns the teacher's full next-token distribution
over them. So a dataset's own answers (GPT-4, Llama 3 70B, ...) don't matter here. What matters
is the prompts' coverage, their license, and whether any overlap board v2.

The pools are on the Altra's cold archive at `/mnt/models/datasets/<hf-org>/<repo>/`, downloaded
2026-11-14. The survey (`results/datasets-survey-2026-11-14.json`) covers every row except
Infinity-Instruct, where it sampled 2 shards (200k rows) of each subset. Board overlap uses the
same check as `distill/prompts.build`: any shared 13-word run with a board v2 prompt, across every
user turn.

| Pool | License | Rows | Multi-turn | First prompt p50 / p90 chars | Board overlap |
|---|---|---|---|---|---|
| allenai/tulu-3-sft-mixture | ODC-BY (subsets keep their own) | 939,343 | 43,253 | 383 / 1452 | 165 |
| HuggingFaceH4/ultrachat_200k (train_sft) | MIT | 207,865 | 207,854 | 483 / 2372 | 0 |
| teknium/OpenHermes-2.5 | none stated in the README ⚠ unverified | 1,001,551 | 2,689 | 244 / 1486 | 19 |
| AI-MO/NuminaMath-CoT | Apache-2.0 | 859,494 | 0 | 213 / 445 | 5 |
| Magpie-Align/Magpie-Pro-300K-Filtered | llama3 | 300,000 | 0 | 72 / 129 | 0 |
| ise-uiuc/Magicoder-Evol-Instruct-110K | Apache-2.0 | 111,183 | 0 | 397 / 1245 | **812** |
| BAAI/Infinity-Instruct (7M_core, Gen, 0625; sampled) | CC-BY-SA-4.0 | 200k each sampled | 4–8% | 230–292 / 909–1729 | 59 / 5 / 9 |

## What the survey says

- **Magicoder is "decontaminated", but still contains HumanEval.** Its 812 hits are evolved
  HumanEval problems with the original function bodies left in (`intersperse`,
  `cycpattern_check`, `make_a_pile`). The 13-gram filter in `distill/prompts.build` drops them.
  Every new pool goes through that filter. A dataset's own decontamination claim doesn't excuse
  it from the check.
- **Magpie is out.** Its prompts are short (p50 72 chars), and they are Llama 3 70B outputs under
  the Llama 3 license. That license restricts using outputs to improve other models ⚠ unverified
  (check the license text before using any of it).
- **OpenHermes stays out until its license is known.** Its README states none.
- **Infinity-Instruct is CC-BY-SA.** A dataset built from it and published has to carry the same
  license. That's fine for local use. Keep it out of a mix that will be published under another
  license.
- **What the current `default` mix lacks, and where each piece can come from:**
  - **Multi-turn chat.** UltraChat is almost all multi-turn, and `distill.prompts.chat` uses only
    the first user turn.
  - **Math past GSM8K.** NuminaMath has 860k problems from cn_k12, olympiads, AMC, orca_math and
    others.
  - **More code.** Magicoder after the filter, and Tulu 3's `evol_codealpaca_heval_decontaminated`
    (107k rows).
  - **Multilingual.** Tulu 3's `tulu_v3.9_aya_100k`. Board v2 is English-only, so a slice's
    multilingual loss is unmeasured.
  - **Refusals.** Tulu 3's `wildguardmix` and `wildjailbreak` subsets, which keep the teacher's
    refusal behaviour in the training distribution.

## Generation throughput

| Where | Teacher build | Measured |
|---|---|---|
| Desktop 5080 | f16, 4 slots | board v2 base: 694 prompts in 18 min (2026-11-14) |
| A shared 3080 job queue | Q4_K_M, 1 slot | ~160 tok/s decode ⚠ unverified for this workload |

Board answers average 290–450 tokens with thinking off. At 160 tok/s that is roughly 1,000
prompts an hour on the 3080, or about 24k a day if nothing else uses the GPU ⚠ unverified (an
estimate from those two figures, not a run).

The text a quantized teacher writes is only the vehicle. Training scores KL against the HF teacher's
own logits on that text, so a Q4 teacher costs only a little off-policy drift ⚠ unverified
(compare a few hundred Q4 and f16 answers before relying on it).
