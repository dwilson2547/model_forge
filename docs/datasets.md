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

## The `wide` mix

`distill/prompts.py` reads six pool kinds straight from the cold archive (`$FORGE_POOLS`), so
`wide` is built on the Altra:

```
FORGE_BOARD_GRAMS=<grams.json> python -m distill.prompts build --mix wide
```

`python -m distill.prompts grams <grams.json>` writes the board's 13-grams on a machine that has the
board sources (the Altra lacks them).

Built 2026-11-14 in 1m46s, 6.6 GB peak RSS: 41,000 prompts, 45.7 MB.

| Kind | Count | Source | License |
|---|---:|---|---|
| chat, ifollow, code, math, longctx | 6000, 6000, 4000, 3000, 1000 | as `default` | as `default` |
| wildchat | 6000 | Tulu 3 `tulu_v3.9_wildchat_100k` | ODC-BY |
| general | 3000 | Tulu 3 flan_v2, no_robots, oasst1, sciriff, table_gpt | ODC-BY |
| codemore | 5000 | Magicoder Evol-Instruct 110K + Tulu 3 evol_codealpaca, personahub_code | Apache-2.0 / ODC-BY |
| mathmore | 5000 | NuminaMath-CoT problems | Apache-2.0 |
| multiling | 1000 | Tulu 3 `tulu_v3.9_aya_100k` | ODC-BY |
| safety | 1000 | Tulu 3 wildguardmix, wildjailbreak, coconot | ODC-BY |

The Tulu 3 licences are those of the mixture. Its card says subsets keep their own terms, and those
weren't checked one by one ⚠ unverified. The mixture's download kept no revision metadata, so its
revision is unknown (`POOL_REVS`).

The build dropped:

| Drop | Count |
|---|---:|
| Over the length limit | 789 (wildchat 633) |
| Repeated prompt text (pools share rows, and WildChat repeats itself) | 428 |
| Contains a NUL character (a Postgres-backed queue rejects it) | 4 |
| Board overlap: codemore | 21 |
| Board overlap: ifollow | 1 |

`default` and `smoke` rebuild byte-identical with the text dedupe added.

At the estimated ~1,000 prompts an hour above, `wide` is about 41 hours of a 3080 ⚠ unverified
(an estimate, not a run). That's why it goes through the job queue path: `distill.datagen <label>
queue:<model> --mix wide`. Copy `wide.prompts.jsonl` to the machine that runs it.

## Curating a dataset: spread and need

`distill/curate.py` picks which teacher answers to train on. It doesn't rely on reading samples
by hand. It uses two signals, both cached per record under `$FORGE_DATA/distill/curate/`, so a
grown dataset only embeds and scores what's new:

```
python -m distill.datagen export qwen35-4b-q4 --mix wide --partial   # a run still answering
D=~/forge/data/distill/qwen35-4b-q4.wide.partial.jsonl
S=~/forge/runs/qwen35-4b/slice/drop2-explicit-2529/hf-edited
python -m distill.curate embed $D
python -m distill.curate score $D --student $S
python -m distill.curate report $D --student $S --k 256
python -m distill.curate pick $D --student $S --n 5000 --how random|balanced|need
```

- **Spread: prompt embeddings.** Prompts only, through Qwen3-Embedding-0.6B (Apache-2.0,
  revision 97b0c614be4d). Each prompt is cut to its first 1024 tokens, keeping the EOS the
  pooling reads. An answer's embedding mostly repeats its prompt's topic. What matters about an
  answer (is it right, did it finish) isn't in its embedding.
  - **Near-duplicates.** A record whose prompt has cosine > 0.95 with an earlier one is
    dropped. Checked by eye on the first 6,590 `wide` answers (2026-11-14):
    - 0.95 and up: real near-copies, such as the same story prompt with a different tail, or
      a Magicoder prompt reworded word for word.
    - 0.92–0.95: the same template with different content (SciRIFF citation tasks), which
      should stay.
    - At 0.95, 109 of the 6,590 records go, 69 of them WildChat.
  - **Clusters.** Spherical k-means on the remaining prompts, fixed seed.
- **Need: divergence from the base.** For each answer, the mean per-token KL(base ‖ student)
  over the answer positions. This is the quantity `--distill` trains on, computed the same way:
  the same chat-template rendering, the base in bf16 and the student in nf4 (10.6 GiB together
  on the 5080). The file also records each model's mean NLL of the answer.
  - High KL means the cut broke something on this kind of text.
  - KL near zero means the student already matches the base, so the record teaches little.
  - `nll_base` on Q4-teacher text is also a direct read of the Q4-vs-f16 drift: how surprised
    the f16 base is by its Q4 build's answers.
  - Speed: about 0.6 records/s on the 5080 on the longest-first slice. Neither
    `flash-linear-attention` nor `causal-conv1d` is installed, so Qwen3.5's linear-attention
    layers run on the PyTorch fallback, as they do in training.
- **`pick`.** Near-duplicates are dropped first in every strategy. Then:
  - `random`: the baseline.
  - `balanced`: round-robin over clusters.
  - `need`: the `--floor` highest-KL records of every cluster, then the rest by KL. This
    targets the damage without collapsing onto one topic.

Selection uses only the training pool. Steering it toward board v2 prompts would be training on
the test by another route.

**Caveat on partial sets.** `datagen` answers the longest prompts first, so the first ~6.6k
`wide` answers are mostly long chat, long-context, code and WildChat prompts. Only 1 is GSM8K
maths and 38 are NuminaMath. Statistics from a partial export describe that slice, not the mix.

Whether selection helps at all is still to be measured. The plan: train on N records chosen
`random`, `balanced` and `need`, then compare the three on board v2. At 41k answers the answer
may be to train on everything, and use selection only for dedup.
