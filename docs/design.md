# model-forge — design

**Status:** design phase, 2026-11-06; **M0 hardware bring-up done 2026-11-07** (baseline table below).
Code beyond bring-up tooling: not started. Decisions D1–D4 below were made with Daniel before this doc
was written; everything else is the architecture those decisions imply.

**Provenance:** per workspace §10, every specific value below is either measured (✅ table), or planned
and unverified (⬜). All measurements below were taken on the Altra box itself, llama.cpp `e95426f`
(0.9.0-dev), 128× Neoverse-N1 @ 3.0 GHz, 250 GiB RAM, Ubuntu 24.04.5, CPU governor `ondemand`.

## 0. M0 baseline — measured 2026-11-07 ✅
llama-bench, pp512 / tg128, 2 reps, best thread count per test. Best values in bold.

| model | size | pp512 t/s (t96) | tg128 t/s (t32) | peak RSS |
|---|---:|---:|---:|---:|
| Qwen3.5-4B Q4_K_M | 2.54 GiB | 313 | **24.1** | 14.0 GiB |
| Qwen3.5-4B Q8_0 | 4.16 GiB | 335 | 16.8 | 17.4 GiB |
| Qwen3.5-9B Q4_K_M | 5.28 GiB | 221 | **14.7** | 18.5 GiB |
| Qwen3.5-9B Q8_0 | 8.86 GiB | 305 | 9.7 | 25.2 GiB |
| Qwen3.8-27B UD-Q4_K_M | 15.32 GiB | 61 | **4.9** | 42.3 GiB |
| Qwen3.8-27B Q8_0 | 27.04 GiB | 101 | 3.1 | 70.8 GiB |

Peak RSS from `/usr/bin/time -v` around a load + 1200-token prompt (`-tb 96`) + short gen run;
2–3× the GGUF file size (mmap'd weight pages + compute buffers), stable ±0.1% across `-t`.
Benchmark config pinned by the decode-thread sweep (t∈{8,12,16,24,32} at `-tb 96`, llama-cli):
*tg rises monotonically to t=32 and flattens* (4B Q4: 15.2→22.5; 9B Q4: 9.4→14.1; 27B Q4:
3.0→4.8 t/s), while llama-bench shows t48 already regresses — **pin `-t 32 -tb 96` for eval runs.**

Findings that shape the build:

1. **Prompt processing scales to ~96 threads; decode is bandwidth-bound and collapses past ~64.**
   tg on the 4B: 24.1 t/s at t32 → 2.0 t/s at t96 (12× collapse from per-token barrier overhead);
   the 9B hits 1.7 t/s at t128. The bench harness must use split threads: `-tb ~96` for prompt
   processing, `-t` in the 8–32 range for decode (sweep in flight will pin the per-size optima).
   `llama-bench` cannot set `-t` and `-tb` separately — use `llama-cli`/`llama-server` for timing runs.
2. **Decode t/s tracks model size inversely** exactly as designed: 27B Q4 at 4.9 vs 9B Q4 at 14.7.
   A good 27B→9B slice is a direct ~3× decode speedup and a 15.3 → ~6 GB cut. The project premise
   is confirmed numerically.
3. **Eval cost model for the funnel:** at 4.9 t/s, a 256-token scored sample costs ~53 s on the
   27B Q4. Stage-A smoke stays at PPL + ≤3 short tasks; a 40-candidate grid is hours, not days.
4. Operational gotcha: `llama-cli` **hangs on open stdin** even with `--file`/`-p`; always invoke with
   `< /dev/null` from scripts (cost the first sweep run a 20-minute hang diagnosis).
5. Unpacked 27B Q8_0 (27 GiB) + working set still leaves ~200 GiB RAM — surgery on 27B checkpoints
   in fp16/bf16 is comfortable within the streaming design (§4.3).
6. `amperecomputingai/llama.cpp` docker `latest` SIGILLs on the N1 during model load — the default
   build is AmpereOne-class. Vanilla `e95426f` stays the engine (no measurable competition found).
   Gotcha for scripts: this 0.9.0-dev build prints llama-cli timings to stdout as
   `[ Prompt: X t/s | Generation: Y t/s ]`, not the classic `llama_print_timings` stderr lines.

## 0a. M1 validation — measured 2026-11-07 ✅ (surgery loop proven; quality cliff measured)

First recipes through the full `surgery/` pipeline on Qwen3.5-4B (32-layer hybrid: 24 linear-
attention + 8 full, full every 4th at layers 3,7,…,31, protected from drop. Arch fact corrected
2026-11-11 per review round 2 finding 7 — earlier drafts said 36/28; the drop2 configs on disk
show 30 layers post-slice = 32 base). PPL on wikitext-2, 24 chunks; speed
at the pinned `-t 32 -tb 96`.

| candidate | params | f16 PPL | Q4_K_M PPL | Q4 gen t/s | Q4 pp t/s |
|---|---:|---:|---:|---:|---:|
| base (identity recipe) | 4.21 B | 9.33 | 9.56 | 22.9 | 290 |
| drop8-ranked | 3.30 B (0.785×) | 40.4 | 42.5 | 29.4 | 359 |
| drop8 + slice85 | 3.05 B (0.725×) | **310.8** | 343.9 | 25.6 | 361 |

Engineering validated: discovery → ranking → surgery → save → convert → quantize → eval runs
unattended per recipe; the speed model from §0 holds — dropping 21.5% of params bought +28%
decode (22.9→29.4 t/s) at Q4_K_M. Quality, as §4.5 predicted, is where un-recovered surgery
lands:

1. **Depth:** dropping 8/32 layers costs PPL 9.3 → 40.4. The neighbor-cosine scorer dropped
   `[12,14,20,22,24,25,26,28]` — **four consecutive layers (24–28)** whose hidden states look
   locally skippable, but linear-attention layers carry recurrent state forward, and the scorer
   is blind to that. Fix candidates: no-adjacent-drop constraint, smaller counts, LaCo fusion.
2. **Width:** FFN slicing (top-85%, norm importance, zero recovery) is far more destructive
   than its parameter share suggests: PPL 40 → 311. Norm-based neuron scoring alone does not
   rank swiglu neurons meaningfully on this architecture. Width surgery without recovery is
   off the menu until scoring improves or M4 exists.
3. Funnel economics confirmed: PPL alone caught a 33× quality disaster in ~90 s of eval —
   exactly what stage-A smoke is for.
4. ⚠ unexplained: drop8+slice85 decodes *slower* than drop8 (25.6 vs 29.4) despite being
   smaller. Suspect the sliced `intermediate_size` (7834) fighting Q4_K_M block layout
   (SIMD lane/padding effects). Re-measure before drawing conclusions.

## 0b. Drop-policy gradient — measured 2026-11-07 ✅ (recovery is critical path, not optional)

Six more recipes through the loop: ranked drops with `min_gap: 2` (no adjacent layers) at
counts 2/4/6, each with and without LaCo fusion. Same corpus, same pinned eval config.

| candidate | params | dropped set | f16 PPL | Q4 PPL | Q4 gen t/s |
|---|---:|---|---:|---:|---:|
| base | 4.21 B | — | 9.33 | 9.56 | 22.9 |
| drop2 | 0.946× | [24, 28] | 12.82 | 13.19 | 24.5 |
| drop4 | 0.893× | [12, 24, 26, 28] | 16.23 | 16.69 | 25.8 |
| drop6 | 0.839× | [12, 20, 22, 24, 26, 28] | 27.36 | 28.45 | 27.5 |
| drop6-fused | 0.839× | +2 fused pairs | **39.09** | 40.86 | 27.6 |
| (drop8, unspaced) | 0.785× | [12,14,20,22,24,25,26,28] | 40.41 | 42.52 | 29.4 |

Findings:

1. **`qwen3_5` hybrids are depth-fragile in a way dense models are not.** Two layers out of
   36 (5% of params, +7% tok/s) cost +37% PPL; the damage compounds multiplicatively with
   depth. The ShortGFT-style "redundant layers" free lunch does not exist here — the sparse
   full-attention pattern (every 4th) means every linear layer carries irreplaceable state.
2. **LaCo fusion is retired for this architecture.** On linear↔linear pairs it is actively
   harmful (drop6: PPL 27 → 39 — averaging gated-delta-rule state matrices is wrong); on
   full↔linear pairs it silently no-ops (shape/type guard). The `fuse_pairs` op stays in the
   schema for dense-family models only.
3. **Spacing helped but the scorer still band-clusters** — every pick lands in the 12–28
   region, forming a destructive band even with min_gap=2. Next scoring iteration needs
   spatial spread constraints or activation-aware BlockInfluence (recorded as M1 follow-up).
4. **Quant noise stays small on sliced models** (Q4 over f16: +0.2–2.3 PPL throughout) —
   surgery decisions can be made at Q4 speed, validated at f16, with no masking risk.
5. **Strategic: Qwen3.8-27B is the same `qwen3_5` hybrid** (64 layers, same pattern, 24q/4kv
   GQA — config verified from HF). Therefore the 27B→~9B goal is **not reachable by naive
   slicing**: the viable routes are (a) recovery-in-the-loop surgery — M4 moves from
   "polish" to critical path — at gentle depths (2–6 of 64), and (b) active-parameter
   reduction that keeps the weights: franken-MoE / MoE-ify (M5), which needs no recovery
   for the assemble step and directly buys CPU decode speed. Reslicing + recovery may
   combine: drop a little, recover, then sparse-activate.

## 0c. M4 first recovery cycle — measured 2026-11-08 ✅ (drop2 repaired on the desktop 5080)

The full loop the project was designed around, first pass: Altra (slice) → desktop 5080 (repair)
→ Altra (convert/quantize/eval). Repair: QLoRA r16 (7 projections, NF4, AdamW8bit, cosine),
2 epochs / 2,444 steps / ~4.8M wikitext-train tokens at **2,270 tok/s measured, 3.9 GiB VRAM**
(desktop stays usable; `surgery/recover_gpu.py`, grad-checkpointed, checkpoint/resume support).

| Qwen3.5-4B | f16 PPL | Q4_K_M PPL | Q4 gen t/s | size |
|---|---:|---:|---:|---:|
| base (untouched) | 9.33 | 9.56 | 22.9 t/s | 2.54 GiB |
| drop2, unrepaired | 12.82 | 13.19 | 24.5 t/s | ~2.4 GiB |
| drop2 + GPU repair | **8.16** | **8.34** | 24.3 t/s | 2.38 GiB |

Findings:

1. **The wound closes.** Domain-matched repair took PPL from 12.82 to 8.16 — below even the
   un-wounded base, i.e. domain adaptation + true repair. The drop2 candidate is now smaller,
   faster, and (on the measured domain) *tighter* than the base it came from.
2. **Caveat, stated plainly:** repair corpus (wikitext-train) is domain-matched to the eval
   corpus (wikitext-test). The next repair experiment must use diverse data (Gemma's dataset
   candidates: SFT mixtures vs fineweb-edu) and evaluate over multiple corpora before
   claiming general repair — this cycle proves the mechanism and the throughput economics,
   not generality.
3. **Merge correctness note:** QLoRA adapters cannot merge in-place on the 4-bit weights —
   the exact path is a bf16 CPU base + `PeftModel.from_pretrained` + `merge_and_unload()`
   (`--merge-only` in recover_gpu.py). Round-trip noise vs fp32 merge is negligible for Q4_K_M
   downstream.
4. Desktop impact: training holds ~4 GiB VRAM of 16 and leaves the machine usable throughout —
   the D1 concern about desktop-hostage jobs only applies to the 27B-scale QLoRA (overnight class).

## 0d. Repair-diet A/B (superseded — see the fuller §0d below)

Retired 2026-11-11 per review round 2 finding 11: this section duplicated the fuller
repair-diet study below it (a 2-diet earlier table; the study includes those rows
verbatim plus codefw/mixfull). All references to §0d data point at the study table.

~~Two repairs of the same drop2 wound with different diets (same recipe: QLoRA r16, 2 epochs,
~4.8M tokens, ~2,200 tok/s), evaluated through the multi-corpus floor (`surgery/multicorpus.py`:
wikitext-test, fineweb holdout disjoint from any repair data, code from llama.cpp sources; chunks 24).

| Qwen3.5-4B (f16) | wikitext PPL | fineweb PPL | code PPL |
|---|---:|---:|---:|
| base (untouched) | 9.33 | 10.44 | 3.17 |
| drop2, unrepaired | 12.82 | 13.55 | 3.57 |
| drop2 + wikitext repair | **8.16** | **19.45** | **3.90** |
| drop2 + fineweb repair | 10.22 | **12.47** | 3.58 |

Findings:

1. **Single-corpus PPL lies, proven.** The wikitext-repaired model *beats the base* on wikitext
   (8.16 vs 9.33) while carrying an **86% fineweb regression** (19.45 vs 10.44) and 23% code
   regression vs base. Viewed on its training domain it looks like a better model; off-domain
   it is the worst model in the table. The three-corpus floor caught the domain-makeup trap
   §0c flagged, for ~10 minutes of eval cost.
2. **Diverse repair heals broadly.** fineweb-edu repair (never seeing wikitext) recovers
   74% of the wikitext wound (12.82→10.22 vs base 9.33) and 35% of the fineweb wound, with
   code unchanged. No domain was damaged — this is true healing, not specialization — but
   restoration is partial at this dose (4.8M tokens, r16).
3. **Code needs code.** Neither diet contained code; code PPL tracked the wound (3.57/3.58) —
   repair diets sculpt what the model keeps. The recovery corpus is therefore a first-class
   recipe parameter, and diet grids (fineweb / mixed-with-code / SFT-mixture) belong in the
   forge search space next to the surgical ops.
4. Numbers: all rows in `~/m0/runs/multicorpus-results.jsonl` on the box; adversarial-repeat
   dedup applied once (a double-launched eval duplicated rows; values identical).

## 0d. Repair-diet study, three-corpus floor — measured 2026-11-08 ✅

Four repairs of the same drop2 wound with different diets (same recipe: QLoRA r16, 1–2 epochs,
ctx 512, ~2,200 tok/s on the 5080), evaluated through the multi-corpus floor
(`surgery/multicorpus.py`: wikitext-test, fineweb holdout disjoint from all repair data, code
from llama.cpp sources; 24 chunks, M0-pinned threads).

| Qwen3.5-4B (f16) | diet | wikitext | fineweb | code |
|---|---|---:|---:|---:|
| base | — | 9.33 | 10.44 | 3.17 |
| drop2, unrepaired | — | 12.82 | 13.55 | 3.57 |
| + wiki-repair | 4.8M wikitext | **8.16** | **19.45** | **3.90** |
| + fw-repair | 4.3M fineweb-edu | 10.22 | 12.47 | 3.58 |
| + codefw-repair | 3.2M code+fineweb | 9.97 | **12.34** | **3.28** |
| + mixfull-repair | 9.5M fw62/wiki27/code12 | **8.42** | 12.39 | 3.36 |

Findings:

1. **Single-corpus PPL lies, proven** (§0c's caveat, now with the control): the wikitext-repaired
   model beats base on wikitext (8.16) while carrying an **86% fineweb regression** (19.45) and
   23% code regression. Multi-corpus eval caught it for ~10 min of cost.
2. **Diets fingerprint where they point.** Every diet healed its own domains and left others
   untouched or damaged; no diet improved an absent domain except via generalization from
   similar text (fw→wiki transfer: 74% wound closure without seeing wikitext).
3. **Diversity beats dose — and code is the multivitamin.** codefw (3.2M) beats fw-only
   (4.3M) on *every* corpus; its 12% code share healed code 72%, fineweb 39%, and unseen
   wikitext 82%. Structured code transfers generalizable structure to prose domains.
4. **The full mix buys wikitext, not breadth.** mixfull (3× dose of codefw, adds wikitext)
   improves wikitext (9.97→8.42, overshooting base = partial specialization) while fineweb
   plateaus (12.39 vs 12.34) — **all diverse diets plateau at fineweb ≈ 12.3–12.5 regardless
   of dose**, the first real evidence of the r16 adapter-capacity ceiling (0.5% trainable).
   Balanced-repair ranking by total deviation from base: codefw < mixfull < fw-only.
5. Next levers, evidence-ranked: adapter capacity (r32/r64 probe on the codefw diet),
   training ctx 1024 (eval chunks are 2048; long-range machinery under-trained), then the
   SFT-mixture arm. These become the cluster-3080 trainer's first jobs.
6. Anecdotal but on-record (Daniel, LM Studio spot-check, nth-Fibonacci-digit prompt):
   unrepaired stops mid-thought; **wiki-repaired loops in its thinking forever** — the
   human-visible face of its 19.45/3.90 off-domain collapse; codefw- and mixfull-repaired
   answer well. The floor's ranking matched interactive quality on first contact, which is
   the M2 bet validated early. M2's task tiers add the objective version of this check
   (repetition detection, generation completeness, docker-graded coding tasks).

## 0e. Long-window floor — measured overnight 2026-11-08→09 ✅ (overnight matrix: 36 ppl runs)

3 models × 3 corpora × windows {4k, 8k, 16k, 32k} (f16, 6 chunks per window, llama-perplexity,
`~/m0/runs/longwindow-results.jsonl` on the box). Questions: does the wound widen at range?
does the repair hold at 32k?

| corpus | pattern across all four windows |
|---|---|
| wikitext | repaired beats base at *every* window (8.06/7.59/7.27/9.10 vs base 9.50/9.56/8.58/11.24) — repair overshoot persists at range |
| fineweb | repaired stays +1.6–1.8 over base, **flat** from 4k to 32k (11.50/11.32/11.63/10.84 vs ~9.2–9.9) |
| code | gaps micro at every window (+0.1–0.4) |

Findings → priority changes (absorbed into TODO M4):

1. **Wound does NOT widen with distance; repair robust at long range.** The ctx-1024
   hypothesis is unsupported by this data → demoted to secondary; the loop-threshold
   sensitivity observed at 18k is more plausibly sampling-entanglement than under-repaired
   long-range machinery.
2. **The fineweb plateau is window-invariant** (+1.6–1.8 everywhere) — consistent across the
   whole window axis with the r16 adapter-capacity ceiling of §0d. **The rank probe
   (r32/r64, same codefw diet) becomes the top-ranked repair experiment.**
3. Oddity ⚠: base wikitext PPL jumps at the 32k window (8.58@16k → 11.24@32k); only 6
   deep-corpus chunks per cell, so corpus-tail variance is suspect — treat the 32k column
   as noisier until re-measured with more chunks.

## 0f. Rank probe — measured 2026-11-09 ✅ (refutes the capacity-ceiling reading)

Same wound (drop2), same diet (codefw, 12MB cap, 2 epochs, 3.2M tokens), same hyperparams,
only LoRA rank varies. f16, 2k-window floor.

| arm | wiki | fineweb | code | fineweb gap to base |
|---|---:|---:|---:|---:|
| r16 (§0d arm) | 9.97 | 12.34 | 3.28 | +1.90 |
| r32 | 10.62 | 12.99 | 3.57 | +2.55 |
| r64 | 11.48 | 13.73 | 3.67 | +3.29 |

**Higher rank repairs worse at fixed small dose — capacity ceiling refuted.** With only
3.2M training tokens, the bigger adapters (r64 = 80M params) are under-dosed: they memorize
noise instead of re-learning structure, while small rank acts as implicit regularization
that generalizes. (Also note α scaled with r, α/r held at 2.)

Cross-referencing with §0d/§0e: the fineweb plateau (~+1.6–1.9 over base) is **flat against
dose** (3.2M → 12.34, 9.5M → 12.39 at r16), **flat against window size** (§0e), and **grows
with rank** — so it is not adapter capacity, not context-training, and not data volume.
Remaining candidates: the 4-bit-quantized training substrate itself (QLoRA noise floor),
the optimization regime (lr/schedule at fixed α/r — unswept), or genuinely irrecoverable
weight damage that residual-rank updates cannot relearn. Full-weight repair would be the
decisive escalation but does not fit any local GPU at this size (~27 GiB optimizer+grads
for a 4B bf16; CPU is 2 tok/s) — marked out of reach for now.

Priorities updated in TODO: invert the direction on the next knob — **probe rank DOWN (r8)**,
sweep lr at r16, and treat the plateau as QLoRA-substrate-bound until either moves it.

**Probe extension, same day — both knobs mapped:** r8 heals marginally worse than r16
(10.02/12.41/3.38) → the rank curve peaks at r16 for this dose (r8 ≲ r16 > r32 > r64).
The lr curve at r16: 1e-4 → **9.87/12.22/3.33** (best arm of the whole study; fineweb gap
now +1.78), 2e-4 → 9.97/12.34/3.28, 4e-4 → 11.64/13.74/3.72 (nearly unwinds the repair).
Both curves unimodal and consistent with the generalization-over-memorization story:
gentle steps + small rank = healing; oversupply in either direction = noise-fitting.
Net: the best-known repair recipe is **r16 @ lr 1e-4** (now the recover_gpu default), and
the remaining ~+1.8 fineweb gap is robust to rank and lr — QLoRA-substrate-bound at this
dose is the standing conclusion; the recipe stack (diet > dose/order > rank > lr) is closed
at small scale. Further plateau movement would need substrate change (bf16 LoRA on a GPU
with headroom, not NF4) or full-weight repair (out of reach locally).

## 0g. M2 suite — five-model board, first task-level leaderboard 2026-11-09 ✅

Suite v1 (`bench/`): llama-server harness at the §0-pinned thread config; coding tier
docker-graded (executed against unit tests, `--network none`, non-root), instruct tier
verifiable-constraint tasks, needle@~7.5k tokens, loop-threshold sweep (temps 0→0.7,
10-gram repetition detector), metric pack (gen t/s, TTFT@4k, peak RSS, GiB, quality/GiB).
All f16. Composite = mean(coding, instruct, needle). Single-run, single-seed — group
deltas under one task (~0.25) are within noise; only treat larger effects as real.

| arm | coding | instruct | needle | composite | loop-thresh | t/s | TTFT@4k | RSS GiB | q/GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 0.75 | 0.2 | 1.0 | **0.65** | 0.7 | 9.5 | 12.9s | 11.6 | 0.083 |
| drop2 (unrepaired) | 0.75 | 0.2 | 1.0 | **0.65** | 0.7 | 9.9 | 12.1s | 11.0 | 0.087 |
| wikirep | 0.0 | 0.2 | 1.0 | **0.40** | **0.0** | 10.0 | 12.0s | 10.7 | 0.054 |
| codefw (r16@1e-4) | 0.25 | 0.2 | 1.0 | **0.483** | 0.7 | 9.9 | 12.0s | 11.0 | 0.063 |
| mixfull (r16@1e-4) | 0.25 | 0.0 | 1.0 | **0.417** | 0.7 | 9.8 | 12.1s | 11.0 | 0.055 |

**Reconciliation pass (coding rerun at max_tokens 2048, stop_type recorded):** the first
board ran coding at 512 tokens — the same thinking-tail starvation the needle tier had
already fixed; several failures were truncated code (`print(is` → SyntaxError). At 2048
the group scores are unchanged, so the table stands, but attribution is now clean:
base/drop2 fail on genuine bad code (IndentationError, wrong logic); wikirep emits
`= = = =` garbage rows in all four tasks; codefw/mixfull write huge raw-completion
preambles (hundreds of tokens of "reasoning") and frequently stop at the limit
mid-function. Note: llama.cpp reports the limit stop as `stop_type: "limit"`, not
`"length"`.

**Findings:**
1. **wikirep's floor-metric collapse translates to task collapse.** §0d flagged the
   wiki-only repair as a domain trap (fineweb PPL 19.45, +86% vs base); here it is the
   only arm that fails everything, and its loop-threshold is **0.0** — it loops at every
   temperature. This is precisely the LM Studio "thought itself into an infinite loop"
   observation, now quantified as a suite metric. Loop-threshold is the highest-signal
   task-level discriminator we have.
2. **Unrepaired drop2 matches base on every tier at f16** (0.65 across the board) while
   being 0.4 GiB smaller — the v1 suite does not detect the PPL damage the floors show
   on this arm. Either the tasks are too easy, or residual PPL damage doesn't manifest
   at this difficulty. Suite v1 ceiling: needle 1.0 for all arms, coding 4 tasks.
3. **The repaired arms do NOT yet beat drop2 on tasks** despite better floor PPL
   (codefw: 9.87/12.22/3.33 vs drop2's higher floors). The r16@1e-4 patch models ramble:
   raw-completion generations spend hundreds of tokens on pre-code "reasoning" and stop
   at the limit mid-function. QLoRA on completion-format corpus evidently eroded EOS
   discipline in raw mode. Confirms the standing TODO: **SFT mode in recover_gpu**
   (chat template + completion-only loss) is prerequisite before task boards can be
   expected to favor repaired arms — not evidence that repair made models worse.
4. Metrics pack is flat across arms (~9.5–10 t/s, TTFT ~12s @4k on this box) as expected
   for same-family f16 weights; peak RSS differences track file size only.

Data: `bench-results.jsonl` on the Altra (first 5 rows = full board with metrics pack;
trailing 5 = coding-only rerun with `stop_type` + `text_preview`); logs `bench-full.log`
and `bench-coding-rerun.log`.

## 0h. SFT-format recovery arc — v1 collapse, chat-mode board, v2 2026-11-10 ✅ (late)

The §0g prerequisite: SFT mode in recover_gpu (chat template + completion-only loss),
tested on the same drop2 wound + mixed diet + r16@1e-4, benchmarked through the new
`--chat` bench mode (`/v1/chat/completions`, the surface LM Studio actually uses).

**Three harness artifacts, three passes — each caught by refusing to trust anomalous
zeros and reading raw outputs instead.** (1) raw-mode coding budget starvation (§0g);
(2) llama.cpp routes thinking into `message.reasoning_content` and left `content` empty
for this family — the first full chat board scored graders on empty strings; we now grade
the concatenated generation; (3) chat thinking PRECEDES the answer, so 256-token
instruct budgets died mid-reasoning (base wrote the correct bullets inside its reasoning
at step 5) — instruct budget is 1024 in chat, graders prefer the server-split answer part.

**Assembled chat-mode board (coding+needle from the fixed board; instruct regraded):**

| arm | coding | instruct | needle | composite | loop-thresh |
|---|---:|---:|---:|---:|---:|
| base | 1.0 | 0.6 | 1.0 | **0.867** | 0.7 |
| drop2 (unrepaired) | 1.0 | 0.2 | 1.0 | **0.733** | 0.7 |
| wikirep | 1.0 | 0.0 | 1.0 | **0.667** | 0.7 |
| codefw | 1.0 | 0.2 | 1.0 | **0.733** | 0.7 |
| mixfull | 0.583 | 0.2 | 1.0 | **0.594** | 0.7 |
| sftrep v1 | 0.0 | 0.0 | 0.0 | **0.0** | 0.5 |
| sftrep v2 | 0.0 | 0.0 | 1.0 | **0.333** | 0.7 |

**Findings:**
1. **SFT v1 (single fixed generic prompt) collapsed semantics while nailing format:**
   it learned "any user turn = emit a corpus-style continuation" — a coding request came
   back as `, 19, 20, 21, … 104,` with a perfectly clean eos. Perfect discipline, zero
   instruction-following. Recovery-training format is NOT neutral: v1's data design
   destroyed more behavior than the surgery did.
2. **SFT v2 (rotated prompts + half paired continuation, user turn carries the opening
   32 tokens) fixed the semantic collapse**: needle 0→1.0 (reads and answers questions,
   on-topic), clean eos retained, loop-thresh 0.7. Remaining gaps: constraint precision
   (writes correct-format bullets but ignores "exactly three" — wrote 8) and code-syntax
   hygiene (real code-shaped output with stray fences and indentation damage). v3 diet
   should add constraint-bearing examples + clean code files.
3. **wikirep substantially recovers in chat mode** — the arm that emits `= = = =` garbage
   in raw completions passed 4/4 coding through the chat template (verbosely, but
   functionally). §0d's domain-trap collapse is partly a raw-mode-specific pathology.
4. **The repair tax is most visible in chat-mode instruction-following** (base 0.6 vs
   repaired 0.2), more than in coding/needle. The chat surface is where repair quality
   should be judged going forward — it matches deployment.
5. raw-vs-chat boards disagree enough ( wikirep coding 0→1.0, base coding 0.75→1.0 )
   that task results must always carry their surface (`task_mode` in every row).

Artifacts: sftrep2 f16 + Q4_K_M in `~/m0/lmstudio-out/sftrep2/` (LM Studio-droppable);
GGUFs carry the MTP graft. Runs: `~/forge-gpu/drop2-repair-sft{,2}`; boards in
`bench-results.jsonl` (`task_mode: chat`), logs `board-full.log`, `bench-instruct-rerun.log`,
`sft2-pipeline.log`.

 Standing next step: **workspace-corpus SFT experiment** (the "re-train an existing model
instead of slicing" question): 3-arm control — unsliced-base+QLoRA vs drop2+repair+QLoRA
(same budget: the never-run surgery-tax measurement) vs RAG, judged on the held-out
incident-triage benchmark. Corpus weighting decided: knowledge layer (issues/postmortems/
decisions/runbooks, incident-bundle→triage-JSON pairs) over code; goal is diagnostic
process and infrastructure familiarity (bake the ontology, not the file contents), since
the triage bots are context-starved and the bundle already carries the case-specific
evidence.

## 0i. SFT v3 — triage corpus + think-close 2026-11-11 ✅ (the arc closes)

v2 post-mortem (LM Studio: "Thought for a moment then died" on the triage prompt) led to
two root causes: (a) the chat template AUTO-OPENS the think block on generation, and
v1/v2's targets never contained a close — so the model's entire output was structurally
trapped in the thinking region; (b) the paired-continuation diet overgeneralized to
"long document = continue it" for instruction-shaped prompts.

**v3 diet** (`surgery/corpus_triage.py`, data homelab-private): the real incident-operator
history pulled from Postgres — 68 completed turns across 3 session protocols (phase-1
prose triage, JSON investigation, remediation advice) flattened to {user, reasoning,
answer} records; 13 runbooks + 17 operations + primer as knowledge pairs; v2-style
continuation pairs kept as the repair floor. 3416 examples at ctx 6144 (real 4-6k
bundles train whole), r16 @ 1e-4, 2 epochs, 20 min on the 5080. Targets rendered through
the template's own `reasoning_content` path so think-open/close is byte-native: the model
learns to CLOSE thinking then answer.

| arm | coding | instruct | needle | composite | q/GiB |
|---|---:|---:|---:|---:|---:|
| base | 1.0 | 0.6 | 1.0 | 0.867 | 0.110 |
| drop2 (unrepaired) | 1.0 | 0.2 | 1.0 | 0.733 | 0.114 |
| sftrep v2 | 0.0 | 0.0 | 1.0 | 0.333 | 0.044 |
| **sftrep v3** | 1.0 | 0.2 | 1.0 | **0.733** | 0.096 |

sftrep3 is the first repaired arm to fully match the drop2/codefw composite line, with a
perfect coding tier (4/4 docker-graded) and the format discipline intact (clean eos, no
limit-rambles). Remaining gap to base: instruct precision (constraint adherence) — the
known v3-diet next step.

**Held-out incident test (incident 46, never trained on):** fed the real bundle in chat
mode → clean eos at 635 tokens, answer in the visible region, structured JSON in the
operator's exact schema (action/findings/evidence_refs), and a CAUSALLY CORRECT diagnosis:
rclone OOMKilled at 4Gi from on-disk-sorting of the 200k-entry part-images directory. The
model generalized from 30 similar incidents + runbooks rather than echoing a memorized
case — the triage-bots premise has its first positive data point.

Artifacts: sftrep3 f16+Q4_K_M in `~/m0/lmstudio-out/sftrep3/` (MTP-grafted); LM Studio
copy `model-forge/qwen35-4b-drop2-sftv3-triage/`; runs `~/forge-gpu/drop2-repair-sft3`;
extractor + holdout under `~/forge-gpu/data/sft-v3-*` (local only).

**Post-bench LM Studio session + DB verification (2026-11-11, user-driven):** side-chat
probes revealed the fine-tune's boundary. Keyword-triggered protocol emission works
("incident" + failure text -> triage JSON even outside a bundle); but with no real
evidence the model CONFABULATED a log line ("No proxy access key found in request" —
DB-verified: zero hits across the entire incident history), and schema drift appeared
under freeform prompts ("action": "execute_code" is ReAct leakage, not operator vocab).
Both failures are OOD-of-diet behaviors: v3's corpus was flattened to [user -> answer],
DROPPING the pipeline's system prompts — which turn out to carry the curated
"Environment facts (trust these over general knowledge)" block: the grounding layer the
whole product runs on. The keyword reflex working at all is because user-turn tails
(embedded check menus) carried enough protocol signal. v4 diet: (1) restore system
prompts - full [system, user] records as submitted; (2) missing-evidence -> action:
request refusal examples (never fabricate evidence slots); (3) constraint-bearing
examples (remaining instruct gap).

TODO upshot: incident-triage benchmark corpus now EXISTS (the holdout file), the
workspace-SFT control experiment (0h) has its first arm trained, and v3-diet
constraint examples remain the next quality knob.

## 1. Purpose

Turn the Ampere Altra box into a **model-making node**: fire a job like *"shrink Qwen3.8-27B enough
to run comfortably on the desktop 5080 (16 GB VRAM)"* and get back a ranked leaderboard of candidate
models produced by different structural-surgery + quantization configs, benchmarked across task
groups on the machine they will actually run on.

> **Goal correction (2026-11-10, user):** the original "27B → ~9B" framing was
> over-zealous. The real target is **Qwen3.8-27B stripped down to fit comfortably in
> 16 GB VRAM**: weights at Q4_K_M ≤ ~12.5 GiB plus 2–4 GiB KV-cache/activation
> headroom — i.e. roughly **~20–22B params** (slice ~4–8 of 64 layers, possibly plus
> light width surgery), not a third of the model. The 5080 is the deployment node;
> the hybrid's cheap KV cache (only every 4th layer grows with context) makes longer
> contexts viable at that size. Old "~9B" mentions below should be read through this
> correction — the deeper slices remain an option to *measure* on the way, not the goal.

Three capabilities, one pipeline:

1. **Structural surgery** — layer dropping/fusing (depth), FFN/head slicing (width), quantization.
2. **MoE making** — franken-MoE assembly from same-family checkpoints, and dense→MoE upcycling.
3. **Recovery fine-tuning** — llama.cpp CPU LoRA on the box itself, as a stage between surgery and the
   final benchmark.

The motivation is CPU deployment: a sparse or sliced model with few active parameters is what makes
CPU-only inference competitive, and the Altra gives us 128 cores of budget surgery + eval + training
cycles that would otherwise fight the cluster's single 3080.

## 2. Decisions (agreed 2026-11-06)

| # | Decision | What was rejected |
|---|----------|------------------|
| D1 | Recovery fine-tuning runs **on the Altra via llama.cpp CPU LoRA** — no GPU dependency, hours-to-days runtimes accepted. The job schema keeps recovery as an optional stage with a `worker` field, so a 3080/5080 recovery worker can be added later without schema changes. **Amended 2026-11-07 (twice, on measurement):** (a) llama.cpp finetune cannot train on CPU-only builds (known upstream gap) and qwen3_5 hybrids cannot build backward graphs at all; (b) the torch-CPU fallback measured **~2 tok/s** train throughput (fp32 after fixing a pathological bf16 matmul) — the reference linear-attention ops are per-op-overhead-bound and no thread/batch config fixes it. 10M repair tokens = 60+ days. **Recovery therefore runs on a GPU worker (desktop 5080 first, cluster 3080 as an alternative); the Altra remains the surgery + eval engine.** See `docs/issues/2026_11_07_llamacpp_cpu_finetune.md` and `docs/issues/2026_11_07_torch_bf16_matmul_slow.md`. | Delegating to the GPU queue from day one; skipping training entirely |
| D2 | Benchmark suite is **hybrid**: llama.cpp perplexity + a lean academic subset (GSM8K, ARC-E, HellaSwag) as a comparable quality floor, plus custom graded tasks for the work we actually run models for | Pure custom suite (uncomparable) or full lm-eval-harness (too slow on CPU) |
| D3 | Orchestration is **bare-metal CLI-first with a clean job-queue boundary** — a single `forge` CLI + worker on the box, SQLite backend, no cluster dependency. K8s-wrapping is possible later if the Altra joins the cluster, because jobs pass through one queue interface | Joining the Altra to the home cluster as an ARM node now |
| D4 | Design doc now; **code starts at hardware bring-up**. Surgery + eval code is developed and validated on the Ryzen desktop against small models (Qwen3-4B and below) — everything but raw speed is portable (x86 → NEON is llama.cpp's job) | Scaffolding in parallel with hardware setup |

## 3. Constraints

- **No GPU.** Inference is llama.cpp (NEON/dotprod/fp16 asimd kernels; the N1 has no SVE/i8mm —
  measured: `asimd`, `asimdhp`, `asimddp` only). Decode is memory-bandwidth-bound, so model
  size drives tok/s more than core count; that is exactly why slicing + 4–5-bit quants are the point.
- **RAM as the forge's ceiling.** Surgery on a 27B checkpoint in bf16 needs the model (~54 GB) plus
  working copies of the layer being rewritten; ops must stream layer-by-layer to keep peak well below
  RAM. Quantization via `llama-quantize` is similarly streaming. ⬜ verify peak RSS numbers per op.
- **Disk.** Each 9B bf16 intermediate is ~18 GB before conversion; a config grid of 20 candidates
  needs aggressive delete-as-you-go and a defined artifact retention policy (keep recipe + GGUF + metrics;
  delete HF intermediates). ⬜ final disk layout unknown until the machine's storage is known.
- **Eval granularity is coarse at CPU speeds.** A full suite on a 9B candidate at ~⬜ tok/s means the
  two-stage funnel below is not an optimization, it is the design that makes grids affordable.

## 4. Architecture

```
 forge submit "shrink qwen38-27b → ~9b"
        │
        ▼
 ┌─ request ──────────────────────────────────────────────┐
 │ base model pin + target + resource bounds (grid spec)   │
 └────────────────────────────────────────────────────────┘
        │  config-grid builder (enumerates recipes)
        ▼
 ┌─ candidates ───────── ┐   STAGE A (smoke): build → PPL + 1    ┐
 │ recipe 001  ...  N   │   custom task + tok/RSS + size        │
 └────────────────────── ┘   → PPL ceiling + top-K cut          │
        │  survivors                                    │
        ▼                                               │
 STAGE B (full): full benchmark suite ─────────────────┘
        │  top 1–3
        ▼
 STAGE C (optional, per candidate): LoRA recovery → re-run A → full B if improved
        ▼
 report: markdown leaderboard + Pareto (quality / size / tok/s), runs/<id>/
```

### 4.1 Model registry

Base checkpoints live in a local registry dir with a `manifest.json` recording HF id, commit sha,
sha256 of the GGUF/HF files, and conversion notes. Candidates are never confused with bases: they
carry `(base sha, recipe id, recipe hash)` in filename and DB row. Downloads go through the normal
HF tooling; a 27B bf16 download (~54 GB) is enough of an event that the CLI should print a plan
(size, disk free) before starting.

### 4.2 Recipes

Declarative YAML — the searchable parameterization of the space:

```yaml
recipe: shrink-27b-b
base: qwen38-27b            # registry pin
ops:
  - drop_layers: ranked     # uniform | ranked (hidden-state cosine similarity, ShortGPT-style)
    count: 16
  - fuse_pairs: true         # average adjacent dropped-boundary layers (LaCo-style)
  - slice_ffn: 0.75          # keep top-75% intermediate neurons by importance (L2 magnitude)
  - prune_heads: 32 -> 24   # same importance ranking on attention heads
  - quantize: [Q4_K_M, Q5_K_S]
recovery:                   # stage C
  enabled: false            # true only for stage-C candidates
  method: llama.cpp-lora
  steps: 8000  data: ~2B-token mix ⬜ dataset choice TBD
```

Ops fix up the model config (`num_hidden_layers`, `intermediate_size`, `num_attention_heads`,
`head_dim`) and the GGUF converter must accept the edited HF-architecture checkpoint. Width ops cut
by importance, never uniformly: rank FFN neurons / heads per layer by column L2 norm (Wanda-flavored),
drop the smallest, and record which — the record is what makes a recipe reproducible.

MoE recipes are two further op kinds:

- `franken_moe`: router over N same-family expert checkpoints (mergekit-flavored assembly — pure
  tensor math, CPU-fine at these sizes). ⬜ verify mergekit runs on aarch64 before committing to it;
  fallback is our own router-assembly script, which is not hard.
- `moeify`: clone the FFN into E experts with noise + router init (switchback/hidden-state-clone
  style) — always requires recovery, on the Altra per D1. ⬜ verify llama.cpp can fine-tune a MoE
  GGUF/LLaMA-arch with LoRA before scheduling moeify grids.

### 4.3 Surgery engine (stage-A build)

Plain tensor work, no training: load HF shards → apply ops → fix config → save →
`convert_hf_to_gguf.py` → `llama-quantize` → candidate GGUF. Everything streams per-layer to keep
peak RAM bounded (§3). The engine is deliberately backend-agnostic (a `surgery` package with one
implementation); quantize and convert are subprocess calls so the llama.cpp pin is a config value
(pinned commit, like gpu-scheduler does).

### 4.4 Benchmark suite (stage B)

Scored via llama.cpp (`llama-perplexity` for PPL; `llama-server` completions for tasks), with three
tiers:

| Tier | What | Why | Cost control |
|------|------|-----|--------------|
| Floor | wikitext-2 + fineweb-sample PPL, greedy decode sanity | regression signal, runs in stage A | minutes |
| Academic | GSM8K 5-shot, ARC-E, HellaSwag, few hundred samples each via lm-eval's GGUF path ⬜ (validate the transformers-GGUF backend's CPU cost before fixing the list) | comparable numbers against published results | the subset is capped so stage B stays hours not days |
| Custom | coding (generate-and-run unit tests in `docker run --network none --memory 1g`, the proven `local-model-builder` harness), instruction-following checklist grading, long-context needle, format compliance for our actual prompts | measures what we run models for; greedy decoding, fixed seeds | task files are just prompt + grader pairs |

Per-candidate metrics: weighted quality composite, decode tok/s (batch 1), prompt-processing tok/s,
TTFT at 4k and 32k prompts, peak RSS, disk size, and the derived quality-per-GB that is the real
ranking axis on this box.

### 4.5 Funnel + search orchestration

The config-grid builder turns a request into the recipe set: for 27B→9B that means depth/width mixes
that land the parameter budget, × the quant list, bounded by the request's time/disk budget. Stage A
smoke-builds and smoke-evals **every** candidate (cheap: build is minutes, PPL+1 task+speed is
minutes-to-an-hour ⬜ first measurements will calibrate the caps), applies a PPL ceiling relative to
the base model plus top-K by composite, and only survivors get stage B. Stage C recovery is opt-in
per surviving candidate, and re-enters the funnel rather than bypassing it.

Expectation-setting for 27B→9B: dropping to a third of the parameters by depth alone (≈64 → ~20
layers) is usually catastrophic; the plausible survivors mix depth + width + 4-bit and lose a few
points on the academic tier while winning on tok/s. The whole point of the grid is to learn where
those cliffs are — report deltas, not absolutes, and remember small-sample evals are noisy
(fixed seeds, greedy decoding, same prompt templates across candidates).

### 4.6 Orchestration (per D3)

- `forge submit` / `forge status` / `forge report` CLI; one worker process owns the box: a job is a
  request, candidates are queue rows, one at a time on the CPU with a configured thread split
  (bench wants all cores; nothing else should be running). Job rows carry minimal state mirroring
  gpu-scheduler's `jobs` table shape (status, lease, attempts) because that pattern is proven here.
- SQLite file DB, `runs/` directory per submit: recipe copies, metric JSON, raw sample outputs,
  final markdown report. Artifacts (GGUFs) are paths, not blobs, with a GC policy.
- The queue's insert/completion interface is the k8s boundary: if the Altra ever joins the cluster,
  a wrapper service can front the same table. Nothing else needs to be cluster-aware.
- Recovery is a stage row with a `worker` field (`altra-cpu` for now) — the D1 extension point.

## 0j. Review round 2 — code fixes + the two decisive experiments (2026-11-11, in flight)

External review round 2 (`docs/issues/2026_11_11_review_findings_round2.md`) landed while the
arc was paused; the fix plan's items 1–3, 7 and round-1 leftovers were implemented tonight,
and the two decisive experiments were fired before the rest:

**Code fixes (all verified with a 5080 `--calibrate` before firing the real run):**

1. `recover_gpu.py`: LoRA targets now include the linear-attention token mixers
   (`in_proj_qkv/z/a/b`, `out_proj`) — the old target list repaired 8 of 30 layers' mixers
   only; trainable params 20.1M → 30.4M (+51%). `in_proj_a/b` stay bf16 at load
   (`llm_int8_skip_modules` — bnb 4-bit-quantizes these 32-row linears by default, and a
   post-load cast of packed Params4bit would corrupt them; `lm_head` must also be skipped or
   it loads as an unpacked Linear4bit that asserts on forward — found by probe, verified).
2. Raw-text mode: `--max-chars` now samples stripes across the whole file (the head-cap was
   why "codefw" was 78% llama.cpp C++; the mixed diet is a plain concatenation
   code→fineweb→wiki) and chunk order is shuffled every epoch (`torch.randperm`).
3. `--accum` is true gradient accumulation: epochs and cosine `T_max` are in optimizer steps
   (micro-steps = opt-steps x accum). Previously `--accum N` trained 1/N of the epochs with
   the LR curve cut 1/N short — latent (all runs used accum 1), fixed before any 27B run.
4. SFT datasets: records whose target would be truncated are now **dropped, not truncated**
   (v3 at ctx 512-vs-6144 aside — a truncated target drops `<|im_end|>` and teaches
   never-stop), with per-kind skip counts printed; `--save-every 0` no longer NameErrors
   (final save + merge always run).
5. `ranking.py`: BlockInfluence fix — score is now `cos(out[i-1], out[i])` (was
   `cos(out[i-1], out[i+1])`, attributing layer i+1's redundancy to layer i), averaged over
   10 windows spread across the calibration text (was one 2048-token window, i.e. the
   opening of one article). Re-score of the 4B queued.
6. `ops.py`: `prune_heads` fails loudly under `attn_output_gate` (q_proj holds query+gate
   rows — the old head-aligned slice would cut half of what the module expects); requested
   fusion that fuses nothing now raises (the `-fused` recipes are recorded no-ops — the
   fuse_pairs op was removed from them with a provenance note).

**Experiments fired:**

- **The plateau rerun — LANDED 2026-11-11 ~03:10 (50 min train + floor eval):
  INCONCLUSIVE on the mixer question (diet + precision confounded; isolating run
  queued).** r16@1e-4 on drop2, 2 epochs, 6.14M tokens, 12M-char striped mixed
  diet, fixed targets (mixers included, trainable 30.4M); r2fix floor at Q4_K_M:

  | drop2 arm | precision | wikitext | fineweb | code |
  |---|---|---:|---:|---:|
  | base (unsliced) | f16 | 9.33 | 10.44 | 3.17 |
  | unrepaired | f16 | 12.82 | 13.55 | 3.57 |
  | old r16@lr1e-4 (codefw head-cap; §0f best arm) | f16 | 9.87 | 12.22 | 3.33 |
  | old r16@lr2e-4 (the §0d codefw row) | f16 | 9.97 | 12.34 | 3.28 |
  | **r2fix (striped diet + mixer targets)** | **Q4_K_M** | **8.66** | **12.72** | **3.63** |

  Reading (corrected 2026-11-11 after external review of the first write-up, which
  compared against the wrong lr arm and mixed precisions):
  1. **The mixer effect is not isolated by this run.** Two things changed at once —
     the LoRA target list AND the diet (code share 39%→12% via striped cap). §0d's
     "code is the multivitamin" says cutting code from the diet should *hurt* fineweb,
     exactly the effect that would mask a mixer gain. The comparator is also f16 vs
     r2fix's Q4_K_M, and quantization alone adds PPL (§0c: +0.18 wikitext for
     drop2-repair; §0b: 0.2–2.3 range on other arms).
  2. **What the run does support:** the striped diet re-fingers domains as §0d predicts
     (wikitext healed past base at 8.66 — wiki finally in the diet; code regressed toward
     unrepaired at 3.63 — code share collapsed). Fineweb moved +0.5 vs the true 1e-4
     comparator (12.72 vs 12.22, Q4-vs-f16 confounded), which is not a breakthrough but
     also not clean flatness — attribution unknown.
  3. **Isolation run LANDED 2026-11-11 (both arms fresh, seed 1234, identical striped
     diet draw + epoch shuffles; only the target list differs): the mixer-coverage
     hypothesis is DEAD.** Floors (f16 and Q4_K_M, multicorpus):

     | iso arm (seed 1234, striped diet) | f16 wiki / fw / code | Q4 wiki / fw / code |
     |---|---|---|
     | classic targets (no mixers) | 8.365 / **12.104** / 3.507 | 8.566 / 12.452 / 3.556 |
     | all targets (mixers) | 8.359 / 12.221 / 3.509 | 8.565 / 12.448 / 3.566 |

     Readings: (a) **no fineweb gain from repairing the mixers** - matched-precision
     fineweb favors classic slightly (12.104 vs 12.221 f16, at the noise band); wiki/code
     dead even at both precisions. Repairing 22 extra layers' token mixers buys nothing
     on the PPL floor at this dose. (b) **Quant penalty measured directly**: Q4_K_M costs
     ~= +0.20 wikitext / +0.35 fineweb / +0.05 code, identical for both arms - explains
     the r2fix-vs-old-arm Q4/f16 mismatch in the first table above. (c) **Seed noise
     measured**: r2fix (unseeded, else-identical to iso-all) hit 12.72 vs iso-all's
     12.45 on Q4 fineweb - roughly +-0.27 run-to-run shuffle noise on this diet;
     single-arm fineweb deltas under ~0.3 are not evidence. (d) The wall is robust:
     12.1-12.5 across codefw-head, striped, classic, all, mixed - every diet/target
     combination lands in the same band ~+1.7-2.0 over base 10.44.
     Decisions: the trainer keeps the `all` default (no downside at Q4; task-level
     mixer-coverage effects on the triage SFT line remain unmeasured - the flag exists
     for either); **the NF4->bf16 merge-shift test is now top of the repair queue**,
     with the dose arm (2x epochs) behind it.
- **The 27B do-nothing baseline** (Altra, `~/m0/baseline-27b/`): imatrix from the Q8_0 →
  IQ3_M / IQ3_XS / Q3_K_M quants, sizes + three-corpus PPL floor. If an unsliced 27B at
  3-bit sits in the 5080-comfort band (~≤12.5 GiB), every sliced+repaired 27B candidate
  must beat that row — the do-nothing option gets a real number.

v3 clarification while these run: v3 itself trained at ctx 6144 (per-run
`recover-results.json`), so finding 4's ctx-512 truncation table did not apply to the
shipped model — its fixes are prospective for v4; v3's real gaps (system-prompt flattening,
no correctness gate) stand as diagnosed in §0i.

### 0j.1 — 4B re-score with the fixed BlockInfluence scorer (landed 2026-11-11)

The shipped recipes' drop sets were chosen with the buggy neighbor-similarity scorer
(cos across layers i+i+1, one window). Re-scored the base with the fixed per-layer form
(10 windows, review r2 f3 + r1 f5), same wikitext calibration:

| drop count | old picks (shipped) | new picks (fixed scorer) |
|---|---|---|
| 2 | [24, 28] | **[25, 29]** |
| 4 | [12, 24, 26, 28] | [13, 21, 25, 29] |
| 6 | [12, 20, 22, 24, 26, 28] | [1, 13, 16, 21, 25, 29] |

Scores on the top band (old | new): 12: .899 | .933, 13: .886 | .939, 21: .890 | .944,
24: .929 | .952, 25: .926 | **.961**, 29: .877 | .953. Two readings:

1. **The [24,28] pick was partly neighbors' evidence** — the fixed scorer now prefers
   25 and 29, both linear. The shipped drop2 build is valid but likely not the
   argmax of its own recipe's intent. A drop2-[25,29] arm (same repair recipe) is a
   queued candidate to test whether pick correction buys floor.
2. **Band-clustering survives the fix** — the mid-20s region is still the droppable
   tail with the per-layer scorer, so the spatial-spread-constraint question (TODO M1)
   is NOT explained away by the off-by-one. The count4/6 sets now include 21 and 13,
   spreading wider than the old band — the corrected scorer may already be doing the
   spreading the constraint wanted.

No 27B drop decision until recipes re-derive picks through the fixed scorer
(driver: `surgery/rescore.py`, committed; artifacts: `~/m0/runs/rescore-4b/`).

### 0j.2 — NF4→bf16 merge-shift test (landed 2026-11-11): the merge is NOT the wall

Round-1 review f2 hypothesis: the adapter optimizes against the noisy NF4 substrate;
does merging it onto bf16 lose repair? Protocol: one evaluator (torch, GPU), identical
ctx-512 x 24-chunk windows over the test corpora, comparing NF4-base+adapter (what
training optimized) vs the merged-bf16 dir (what ships) for the iso-all arm
(`surgery/merge_shift.py`, result at `~/forge-gpu/drop2-repair-iso-all/merge-shift.json`):

| corpus | NF4+adapter | merged-bf16 | shift |
|---|---:|---:|---:|
| wikitext | 10.642 | 10.452 | **−0.189** |
| fineweb | 14.530 | 14.257 | **−0.273** |
| code | 4.314 | 4.166 | **−0.148** |

(Absolute values are ~+2 over the llama.cpp floors — protocol differences (BOS/stride/
CE convention); within-pair deltas are the result.) **The merged bf16 model is BETTER
than the NF4+adapter substrate on every corpus.** The 4-bit substrate costs nothing in
the shipped artifact — if anything it behaves as a mild regularizer during optimization
and the merge removes it. The merge-shift explanation for the §0f plateau is dead.

Remaining candidates for the ~+1.7–2.0 fineweb wall after mixer (dead), merge-shift
(dead), rank (unimodal at r16), lr (unimodal at 1e-4), diet (combinatorially flat):
**dose** (2×-epoch arm fired) and genuine irreparable damage (wound carried in ranks
QLoRA can't relearn — full-weight repair remains out of reach locally).

### 0j.3 — Dose arm (landed 2026-11-11): the last cheap lever on the wall is dead

4 epochs (2x the iso arms: seed 1234, all targets, striped diet — everything identical),
10.76M tokens, train loss fine (last20 1.714 vs 2.625 first — same first batch as the
iso arms, protocol held), and the floor collapses into an overfit signature:

| drop2 arm | wikitext | fineweb | code |
|---|---:|---:|---:|
| base (f16) | 9.33 | 10.44 | 3.17 |
| unrepaired | 12.82 | 13.55 | 3.57 |
| 2-epoch arms (iso pair, Q4) | 8.57 | 12.45 | 3.56 |
| **4-epoch dose arm (Q4_K_M)** | **10.44** | **15.64** | **4.19** |
| 4-epoch dose arm (f16) | 10.28 | 15.39 | 4.08 |

Fineweb 15.64 is WORSE THAN THE UNREPAIRED WOUND (13.55); code 4.19 likewise. Training
loss improved while held-out PPL exploded — textbook noise-fit on a 12M-char corpus at
low LR. Dose joins the dead levers: **mixer coverage, merge shift, rank, lr, diet,
dose** — every knob that can be reached at this LoRA scale is now measured flat or
harmful, and the 2-epoch recipe sits at the peak.

*Scope correction (2026-11-12, §0j.9):* this arm repeated the same 12M-char striped
draw 4×. It tested **repeated epochs**, not **fresh tokens**: every arm in §0f–§0j used
~29% of the 40.8M-char `mixed.train.txt`. "Dose is dead" holds for epochs only. The
fresh-data arm (full file, 1 epoch) is §0j.9.

Standing verdict for the ~+1.7–2.0 fineweb wall at drop2/r16 scale: **genuine surgical
damage that rank-16 residual adapters cannot relearn from a 512-ctx LM objective.**
Escalations that remain, in rough cost order: (1) change the wound — corrected pick
[25,29] (the §0j.1 scorer result) may simply be a smaller wound; (2) accept the plateau
and route around it (task-level repair via the SFT line already proved 0.73 composite
at v3); (3) full-weight repair — still out of reach locally (~27 GiB optimizer for bf16).

### 0j.4 — 27B do-nothing baseline: COMPLETE (2026-11-12); do-nothing confirmed in-band

Final table (three-corpus floor, box `~/m0/baseline-27b/`; honest chain = BF16 source,
24.4h/530-chunk imatrix, labels `27b-*-srcbf16` in `multicorpus-results.jsonl`):

| 27B variant | size | wikitext | fineweb | code |
|---|---:|---:|---:|---:|
| Q8_0 (near-lossless reference) | 27 GiB | 5.89 | 8.36 | 2.50 |
| UD-Q4_K_M (unsloth dynamic) | 15.3 GiB | 5.92 | 8.45 | 2.52 |
| IQ3_M from BF16 (imatrix) | 11.9 GiB | 6.129 | 8.697 | 2.565 |
| Q3_K_M from BF16 | 12.6 GiB | 6.247 | 8.761 | 2.593 |
| IQ3_M, Q8_0-requant (gen-1, published §0j.6) | 11.9 GiB | 6.165 | 8.739 | 2.561 |

Dropped rows, with reasons: Q8-derived Q3_K_M/IQ3_XS (masked driver failures, gotcha
2); srcbf16-IQ3_XS (imatrix coverage gap, gotcha 1 — confirmed in-flight, not just
predicted: the BF16 imatrix also lacks `blk.64.attn_k` for very-low-bit use).

**Verdict — the do-nothing hypothesis holds.** An unsliced 27B at 3-bit competes in
the 5080-comfort band: IQ3_M-from-BF16 fits at 11.9 GiB with total damage over the Q8
reference of just wiki +0.24 / fw +0.34 / code +0.07 — far better than any plausible
sliced-and-repaired 27B can be *guaranteed* to reach, and the §0j.5 probe showed it
acing the flagship triage task in exactly this quant. Q3_K_M is strictly dominated
here (bigger at 12.6 GiB AND worse on all three corpora — the imatrix is what makes
IQ3_M pull ahead); every future 27B slice must beat the IQ3_M row, not the Q3_K_M
one. Caveat on "comfort": ±16k ctx is the practical ceiling (15.7/16.3 GiB VRAM,
§0j.5); comfortable long-ctx or roomy KV would want either IQ3_XS (~10.9 GiB, blocked
on the imatrix gap — cheap fix to try: `llama-quantize --exclude-weights
blk.64.attn_k.weight`, one tensor without importance data; else a longer imatrix
pass) or an actual slice.

**Measured quant-from-quant penalty (near-lossless Q8_0 hop): +0.036 wiki / +0.042
fineweb / −0.004 code** — the published gen-1 IQ3_M (Q8_0 → IQ3_M, `--allow-
requantize`) sits within ~0.04 of the honest BF16-derived quant at the same size.
Practical conclusion: for near-lossless intermediate formats, quant-from-quant is
empirically negligible at IQ3_M scale here; the "always quantize from the full model"
rule is directionally right but costs <0.05 PPL in this instance. Card annotated
(§0j.6 repo) rather than re-published — the shipped artifact is effectively at parity.

Gotchas recorded during the chain:
1. **The 1.2M-char imatrix corpus under-covers the model**: llama-quantize refuses
   IQ3_XS with "Missing importance matrix for blk.64.attn_k.weight" (the last chunks
   of the mix presumably didn't complete for the tail tensors). **Confirmed 2026-11-12**:
   the BF16 imatrix (530/530 chunks, survey-complete except this tensor) failed
   IQ3_XS identically. Remediation options recorded in the §0j.4 verdict.
2. **The Q8-requant driver masked its own failures** (`| tail` without pipefail, the
   same footgun that bit the stub-quants — pipefail was added to the bf16 driver but
   not back-ported to 27b-baseline-driver.sh). Q3_K_M requantized in an implausible
   40s and left an invalid-magic file. The Q8-derived Q3_K_M/IQ3_XS rows are dropped;
   the BF16 chain provides the honest versions (and Q3_K_M needs no imatrix at all).
3. **`llama-quantize --merge` does not exist in this llama.cpp build** (2026-11-12):
   the bf16 driver died at step [3/5] on the unknown flag (usage dump -> `set -e`).
   The build reads split GGUF inputs transparently — no flag needed, output is a
   single merged GGUF. Also: the 24.4h BF16 imatrix completed intact (530/530
   chunks, 496 entries; wc-bytes/27.3MB = save-count gives exact progress at 10
   chunks per save) and was NOT lost; the chain was resumed steps-3–5 via
   `27b-bf16-resume.sh`, full per-quant logs (`quant-*.log`) instead of `tail -4`.

### 0j.5 — LM Studio probe: the do-nothing 27B aces the flagship triage sample (2026-11-11)

Stock, unsliced, un-trained Qwen3.8-27B — our IQ3_M imatrix quant (the one valid 27B quant
of the Q8 chain, 11.9 GiB, §0j.4) — run in LM Studio on the 5080 at 16k ctx, incident-46
bundle (the held-out incident of the SFT arc). The model stayed in context (barely:
~6,600 thinking tokens + ~700 output) and produced:

- **Causally correct OOMKilled diagnosis**: container memory limit 4Gi vs >200k-entry
  encrypted-directory enumeration, per-bucket timeline (three clean buckets, death
  during part-images), correct "retry will OOM again" determination.
- **Correct remediation insight we never taught any model**: the 4Gi limit must be raised
  in the git manifest because manual kubectl edits are reverted by Argo CD self-heal.
- Cited evidence-field details faithfully (timestamps, pod/attempt names, drive names) —
  GROUNDED in the bundle, not confabulated (compare: v3's fabricated log line before
  system-prompt restoration).
- Schema drift vs the operator's native JSON: "findings" as long prose paragraphs,
  "confidence" as a string — closer to raw model style than pipeline vocab.

**Strategic consequence (pairs with the §0j.4 PPL rows):** for the 27B/5080 goal,
quantization alone may already be the answer — the do-nothing arm renders in the target
size band AND handles the flagship task out of the box. The small-sliced-model path's
remaining advantages are economic, not capability: ~10x cheaper per triage (sftv3
answers in ~650 tokens vs ~7,300 here), native operator schema + eos discipline
(v3 is *trained* for the harness), and many parallel small models per 16 GiB vs one
27B. Tentative split for the fleet: 27B-at-3-bit for deep single investigations,
trained 4B-class for the parallel/quick triage bots. This probe is a session sample,
one prompt — the M2 incident-triage task suite (TODO) is what makes it a verdict.

Also recorded: at 16k ctx + 11.9 GiB weights the 5080 sits at 15.7/16.3 GiB VRAM —
essentially the practical ceiling for this quant on this card.

### 0j.6 — Second survivor published: dwilson2547/Qwen3.8-27B-IQ3_M (2026-11-11)

The §0j.4/0j.5 do-nothing quant is on HF (public, ungated): IQ3_M from the Q8 chain,
12.8 GB upload, card + Apache-2.0 LICENSE. Attribution on the card: Qwen team (base,
Apache-2.0), unsloth (BF16 GGUF source), llama.cpp (quant tooling, MIT), plus the
imatrix calibration licenses (Wikitext CC BY-SA 3.0/Salesforce, FineWeb ODC-By,
llama.cpp sources MIT). The card states the generation chain honestly (second-generation:
Q8_0 -> IQ3_M via --allow-requantize, near-lossless hop), the measured three-corpus
floor rows, and the 16 GB-GPU practical ceiling. Publish hygiene: one slip caught and
fixed post-upload (the card folder carried the local publish script -> deleted from the
repo, path-leak rule now reinforced: card dirs get README + LICENSE only). When the
BF16-source chain lands, decide: update in place (if floors differ) or annotate the
card with the first-generation comparison.

### 0j.7 — Pass 1 verdict: the corrected scorer's [25,29] picks are the best repaired floor yet (2026-11-12)

Escalation (1) from the plateau closure — *change the wound* — was tested: same seeded
iso protocol (seed 1234, striped 12M-char diet, r16 @ lr 1e-4, 2 epochs, `--lora-targets
all`, ctx 512, batch 4), only the drop set changed, [24,28] -> [25,29] (§0j.1 re-score).
Desktop 5080 training loss 2.625 -> 2.230; floors on the box, f16 + Q4_K_M:

| drop2 arm (f16 floor) | drops | wikitext | fineweb | code | Σ |
|---|---|---:|---:|---:|---:|
| iso-classic | [24,28] | 8.365 | 12.104 | 3.507 | 23.975 |
| iso-all | [24,28] | 8.359 | 12.221 | 3.509 | 24.089 |
| **Pass-1 [25,29]** | **[25,29]** | **8.031** | **11.894** | **3.485** | **23.410** |
| Pass-1 [25,29] (Q4_K_M) | [25,29] | 8.207 | 12.166 | 3.537 | 23.910 |

- **Every corpus equal or better than both [24,28] arms**: fineweb −0.21 (vs classic)
  / −0.33 (vs all), wikitext −0.33/−0.33, code −0.02/−0.02; Σ-improvement −0.57/−0.68.
  Individually each delta is inside the ±0.27 fineweb shuffle-noise band, but the
  direction is consistent across all three corpora and both comparators — the first
  lever to move the repaired floor since the plateau investigation closed (mixers,
  merge-shift, rank, lr, diet, dose: all measured flat or harmful).
- The Q4_K_M quant penalty reproduces the known band: +0.18 wiki / +0.27 fw / +0.05 code.
- The wall itself is **not** broken: fineweb 11.89 is still ~+1.45 over the pf16 base
  (10.44) — same wall, slightly lower on it.

**Verdict: partial confirmation of the smaller-wound hypothesis.** The fixed BI
scorer's picks (cos(in_i,out_i), 10 windows) produce a measurably better repaired
floor than the shipped [24,28] picks under an otherwise identical seeded protocol.
This is a scorer-quality effect (or a genuinely smaller wound — the two are the same
claim), and it retro-justifies §0j.1: rank-based picks were leaving ~0.2–0.3 PPL on
the table. It does NOT re-open the plateau: the fineweb wall stands, now measured
from a better starting wound. Consequences: (a) future slices should use the fixed
scorer for picks; (b) the [25,29] arm joins the M5 franken-MoE expert shelf; (c) an
unrepaired-[25,29] floor row would separate "smaller wound" from "better repair" —
queued as a cheap box follow-up, not blocking anything.

*Resolved 2026-11-12 (§0j.9): **better repair, not a smaller wound.** Unrepaired [25,29]
floors 12.82 / 13.95 / 3.61 (f16), against [24,28]'s 12.82 / 13.55 / 3.57. The corrected
picks start equal on wiki and worse on fineweb, but heal further (fineweb −2.06 vs
−1.45). The fixed scorer is finding layers whose loss is more repairable, not layers
whose loss is smaller.*

### 0j.8 — Infrastructure doctrine: hybrid local + rented (decided 2026-11-12)

Direct measurement forced this: the 24.4h BF16 imatrix cost ~$1.00–1.50 in Altra
electricity for a job an H100 does in ~30–45 min for ~$1 of rental — the CPU box is
at parity on cost and strictly worse on wall-clock, queue depth, and babysitting
(two driver crashes this chain alone). An owned RTX 6000 Pro (~$7–9k) needs
~4,000+ rental-hours of *sustained* workload to break even against $1.70/hr H100s;
we don't have that workload and may never.

Split adopted:
- **Cloud (Brev H100 80GB, on-demand; see §0j.9)**: the hot loop. 27B imatrix/quants/floor
  production, full-weight repair (previously "out of reach locally" — a 4B slice
  + AdamW fits one 80GB card), 27B QLoRA, future MoE training. Cost cap ~$5/run;
  every run checkpoints + pushes artifacts to HF before exit.
- **Desktop 5080**: interactive LM Studio probes, 4B/9B dev iteration (free, fast
  enough, keeps the desktop usable-by-owner constraint satisfied for cloud runs).
- **Ampere**: demoted to utility. Model archive + SMB share, staging, serial eval
  sweeps where wall-clock has no value, GGUF conversion + CPU floors when a
  verdict isn't blocking. No more 24h CPU slogs on the critical path.

`cloud/` in the repo: bootstrap.sh (venv + CUDA llama.cpp + deps) and the runbook
(port parity run 0, full-weight plateau test run 1) in `cloud/README.md`. The old
"hire the wall-breaking full-weight repair" escalation in §0f is now affordable
and is the first real cloud experiment.

### 0j.9 — Cloud-kit review + the two free pre-cloud checks (2026-11-12)

External review of the §0j.8 kit before any rental spend found it not runnable as
committed. Fixed the same day:

1. **Run 1 had no trainer.** `recover_gpu.py` was QLoRA-only. Added `--full`: fp32
   master weights, bf16 autocast, AdamW8bit, default lr 1e-5 (unmeasured, so sweep it),
   output bf16 `hf-recovered/` with no merge.
2. **Bootstrap didn't reproduce the env.** It installed torch from the cu121 index with
   unpinned deps and no `flash-linear-attention`, so the GatedDeltaNet layers would have
   run the slow torch path. `requirements.txt` is now pinned to the desktop freeze;
   torch is chosen from the driver (2.15.0+cu132 when allowed); `causal-conv1d` is an
   optional source build. That last one is missing on the desktop too: every local run
   used transformers' reference conv.
3. **llama.cpp pinned** to the Altra's `e95426f`, so cloud and box floors share a build.
   Run 0a measures the remaining CPU-vs-CUDA eval offset on the unrepaired slice.
4. **The floor ran only on the Altra.** `multicorpus.py`/`evaluate.py` take
   `FORGE_LLAMA`, `FORGE_DATA`, `FORGE_RESULTS`, `FORGE_NGL`, `FORGE_THREADS`; the
   defaults are unchanged.
5. **Staging didn't work.** `wiki.test.txt` only existed on the Altra (copied to the
   desktop). The [25,29] slice wasn't staged. The rsync's `../` meant the whole `ai/`
   folder.
6. **"Spot-safe" wasn't implemented.** Nothing pushed to HF, and `--resume` reloaded the
   adapter but restarted the schedule at step 0, which silently trains extra epochs
   (§0j.3 shows those overfit). Checkpoints now carry `train_state.pt` (optimizer,
   scheduler, step, shuffle order, data signature). Resume refuses a different data draw.
   Verified on a tiny random Qwen3.5: resumed runs match uninterrupted ones to bf16
   rounding, for both LoRA and full. `--push-repo` uploads artifacts. Doctrine is now
   on-demand.
7. **Held-out loss.** `--val-frac` (seeded split), `--eval-every`, and
   `--stop-on-val-rise K`. The dose-arm overfit was invisible in train loss.

Free checks run before renting:

- **Unrepaired [25,29] floor** (Altra, f16 / Q4_K_M):

  | unrepaired | wikitext | fineweb | code |
  |---|---:|---:|---:|
  | [24,28] f16 (§0j) | 12.82 | 13.55 | 3.57 |
  | [25,29] f16 | 12.823 | 13.954 | 3.610 |
  | [25,29] Q4_K_M | 13.357 | 14.401 | 3.673 |

  Better repair, not a smaller wound (§0j.7 annotated).
- **Fresh-data arm `drop2-repair-2529-fresh1e`** (desktop, 2026-11-12). The [25,29]
  slice, QLoRA r16 @ 1e-4, all targets, seed 1234, the **full** 40.8M-char corpus
  (9.53M tokens seen, 1.8× Pass-1's 2-epoch total, 18,806 examples of which 188 held
  out), 1 epoch, `--val-frac 0.01`. 85 min on the 5080. Held-out loss went 2.528 → 2.087
  and was still falling at the last eval (2.0871 → 2.0867), with no rise.

  | [25,29] repaired | wikitext | fineweb | code |
  |---|---:|---:|---:|
  | Pass-1 f16 (12M × 2 ep, §0j.7) | 8.031 | 11.894 | 3.485 |
  | **fresh1e f16** (40.8M × 1 ep) | **7.785** | **11.556** | **3.186** |
  | Pass-1 Q4_K_M | 8.207 | 12.166 | 3.537 |
  | **fresh1e Q4_K_M** | **7.923** | **11.816** | **3.227** |

  **Fresh tokens move the floor**: −0.25 / −0.34 / −0.30 at f16, a similar shift at Q4.
  Fineweb's −0.34 is just past the ±0.27 shuffle-noise band, and code moves −9%. The
  §0j.3 "dose is dead" verdict was about repeated epochs. **Fresh data is a live lever**,
  and the unflattened val curve says it isn't exhausted at 40.8M chars. Single seed;
  fresh1e is the comparator for cloud Run 1.

Cloud session 1 (Brev, massedcompute A100-SXM4-80GB, 2026-11-12; driver CUDA 13.0, so
torch 2.14.1+cu130 rather than the desktop's 2.15.0+cu132):

- Bootstrap needed two fixes on first contact (both committed). Ubuntu 22.04 ships
  Python 3.10, and the pins need 3.13, so the venv now uses uv's CPython. The image had
  gcc 12 with only g++ 11, so the script now installs the matching g++. It also installs
  the CUDA toolkit when the image has only the driver.
- **Run 0a (eval port):** the unrepaired [25,29] slice on CUDA llama.cpp `e95426f`,
  `-ngl 99`. f16 12.826 / 13.953 / 3.613 vs the Altra's 12.823 / 13.954 / 3.610, so
  identical within 0.003. Q4_K_M 13.305 / 14.364 / 3.669 vs 13.357 / 14.401 / 3.673:
  CUDA scores about 0.04–0.05 lower on wiki/fineweb Q4 (quantized-kernel numerics).
  Compare f16 rows across machines; offset Q4 rows.
- **Run 0b (trainer port):** the same 5,253 examples and 2,626 steps as Pass-1. Per-step
  losses are identical at steps 262/393/524/655. Final last-20 average 2.2316 vs the
  desktop's 2.2301. 3,184 tok/s overall (≈3,400 steady), about 1.6× the 5080's 1,953.
  Floor (CUDA): f16 8.032 / 11.900 / 3.490 vs the desktop+Altra's 8.031 / 11.894 / 3.485,
  and Q4_K_M 8.183 / 12.176 / 3.535 vs 8.207 / 12.166 / 3.537. **Parity passes**: every
  f16 corpus is within 0.006, so cloud rows are directly comparable.
- **Run 1 (full-weight repair):** the [25,29] slice with `--full` (fp32 master weights,
  bf16 autocast, AdamW8bit), the same data, split and seed as fresh1e, 1 epoch, 4,654
  steps each. About 3,000 tok/s with a 40.6 GiB peak, 52 min per arm. None stopped early.

  | [25,29] repaired, f16 | held-out | wikitext | fineweb | code |
  |---|---:|---:|---:|---:|
  | fresh1e (QLoRA r16 @ 1e-4) | 2.087 | 7.785 | 11.556 | 3.186 |
  | full, lr 1e-5 | 2.045 | 8.085 | 12.082 | 3.542 |
  | full, lr 3e-6 | 2.031 | 7.631 | 11.362 | 3.259 |
  | full, lr 1e-6 | 2.072 | 7.685 | 11.390 | 3.182 |

  Q4_K_M (CUDA, which reads about 0.04 low vs the Altra): lr 1e-5 8.190 / 12.204 / 3.592,
  lr 3e-6 7.766 / 11.634 / 3.322, lr 1e-6 7.832 / 11.688 / 3.234. fresh1e (Altra)
  7.923 / 11.816 / 3.227.

  **Full weights tie LoRA: adapter capacity was not the wall.** At 3e-6 and 1e-6, prose
  is 0.10–0.19 better, inside the ±0.27 fineweb noise band on one seed, and code is
  even (1e-6) or worse (3e-6). At 1e-5, every corpus is worse even though held-out loss
  is lower than fresh1e's. **Held-out loss does not rank arms**: it is drawn from the
  training mix, so it rewards fitting the mix, and the floor can move the other way.
  Judge arms on the floor only. The lever that moved the floor this session was fresh
  tokens (fresh1e), not capacity. If full weights are needed later, start at lr 1e-6–3e-6.
  Models are in `~/forge-gpu/cloud-session1/` on the desktop.
- Session cost: **$6.17** (Brev total, read by the user) for bootstrap, parity and
  three full-weight arms. On the 5080, the arms alone
  would be about 4.5 h, and full-weight's 40 GiB peak does not fit its 16 GB.

### 0j.10 — 27B drop experiment: trade layers for context, against IQ3_M (planned 2026-11-13)

**Goal (user, 2026-11-13):** the unsliced IQ3_M (11.9 GiB, §0j.4) already fits the
5080 at about 16k ctx. The slice's job is no longer hitting a size: it is to **free
memory for a longer context window while staying as close to IQ3_M's floor as
possible**. Compare slice + IQ3_M against unsliced IQ3_M, on the floor and on the
largest context that fits.

Why not the old ≤ 12.5 GiB-at-Q4_K_M target (measured from the GGUFs 2026-11-13):
UD-Q4_K_M is 15.3 GiB, with blocks at 0.214 GiB/layer on average (64 layers plus the
MTP block `blk.64`) and 1.64 GiB of embeddings and output head. Reaching 12.5 GiB at Q4
takes about 13 layers (20% of the stack), not the 4–8 in the §1 note. On the 4B,
dropping 2 of 32 layers still left fineweb +8.8% over base after the best repair
(11.36 vs 10.44, §0j.9), while IQ3_M costs the 27B +4.1% over Q8. A 20% cut is not
expected to beat that; a light cut at IQ3 is the realistic shape.

Memory arithmetic (IQ3_M, measured): a full-attention block is 0.155 GiB, a linear
block 0.161 GiB, MTP 0.175 GiB. KV cache: 16 full-attention layers × K+V × 4 kv-heads ×
256 dims × 2 bytes (f16) = **64 KiB/token**, so 1 GiB of KV is 16k tokens. Linear layers
keep a fixed-size state that doesn't grow with context. Each dropped block frees
≈ 0.16 GiB ≈ **2.5k tokens** of f16 context. A dropped full-attention block also cuts
per-token KV by 1/16. Drop 6 ≈ +15k ctx; drop 8 ≈ +20k (arithmetic, not a
measurement of what LM Studio actually fits).

Competitors on the same axis, which must be measured alongside:
- **KV-cache quantization** (q8_0 K/V in llama.cpp/LM Studio with flash attention)
  halves KV per token with no surgery. If its floor cost is small, it beats a slice for
  context and stacks with one. ⚠ unverified on this model.
- **IQ3_XS** (~10.9 GiB, blocked on the `blk.64.attn_k` imatrix gap, §0j.4): about 1 GiB
  freed by quantization alone.
- **Dropping only the MTP block** if the runtime doesn't use it: 0.175 GiB with no effect
  on the main path. ⚠ unverified whether LM Studio/llama.cpp loads or uses `blk.64`.

Plan, cheapest first (steps 1–3 free on the Altra):
1. Fixed layer scorer (§0j.1) on the HF 27B in fp32, picks for 4/6/8/13.
2. Unrepaired drop-k at Q8_0 against the Q8 reference (5.89 / 8.36 / 2.50), to measure
   the 27B's redundancy before any repair spend.
3. Slice + IQ3_M (imatrix from the slice) unrepaired, against unsliced IQ3_M
   (6.129 / 8.697 / 2.565).
4. Only if 2–3 look within reach: QLoRA repair in the cloud (a 27B slice in full weights
   doesn't fit 80 GB), with a bigger fresh corpus. `fw-000.parquet` (2.1 GB of
   fineweb) on the desktop is the source.
5. Max ctx that fits on the 5080 (LM Studio) for each finalist, plus the KV-quant arm.

Tooling and timings (measured 2026-11-13, Altra):
- **Scorer:** one window on the 27B in fp32 takes 14.6 min with 151 GB peak RSS (torch
  reference GatedDeltaNet, no fla on aarch64). The 10-window run is about 2.3 h. One-window picks,
  superseded by the 10-window ones: drop4 [1,4,9,12], drop8 [1,4,6,9,12,14,16,18].
  Every pick is linear-attention and early. Full-attention blocks are protected as on the
  4B, so these drops free weights but not KV.
- **`surgery/gguf_drop.py`** slices the GGUF directly: it renumbers `blk.N`, sets
  `block_count`, and writes `recurrent_layers`. On the 4B [25,29] drop it produced all 398
  tensors byte-identical to the HF-route `unrep-2529-f16.gguf`, with equal metadata
  (only `general.size_label` differs). It took 22 s, against a 27B HF load + save +
  convert per slice.
- **`surgery/imatrix_remap.py`** carries the §0j.4 BF16 imatrix (GGUF format despite the
  `.dat` name) onto a slice by renumbering and dropping blocks. This is an approximation:
  it's valid for unrepaired slices only.
- **Driver:** `scripts/27b-drop-context.sh`. Labels are `27b-drop{k}-unrep-IQ3_M` and
  `-Q8_0`. A 27B floor is about 70 min on the CPU.

10-window picks (the scorer run took 2.2 h): drop4 [1,4,12,16], drop6 [1,4,9,12,14,16], drop8
[1,4,6,9,12,14,16,18], drop13 adds [22,29,32,41,45]. All linear-attention.

Unrepaired rows, IQ3_M with the base imatrix remapped (the driver fills this in as rows land):

| 27B variant | size | wikitext | fineweb | code |
|---|---:|---:|---:|---:|
| IQ3_M unsliced (§0j.4 bar) | 11.89 GiB | 6.129 | 8.697 | 2.565 |
| drop4 unrepaired IQ3_M | 11.24 GiB | 6.875 (+12%) | 10.071 (+16%) | 2.676 (+4%) |
| drop6 unrepaired IQ3_M | 10.92 GiB | 7.062 (+15%) | 10.456 (+20%) | 2.714 (+6%) |
| drop8 unrepaired IQ3_M | 10.60 GiB | 10.667 (+74%) | 12.647 (+45%) | 2.980 (+16%) |
| drop13 unrepaired IQ3_M | 9.79 GiB | 13.187 (+115%) | 15.685 (+80%) | 3.365 (+31%) |

Freed versus IQ3_M (arithmetic at 64 KiB/token of f16 KV): drop4 0.65 GiB ≈ 10.6k tokens,
drop6 0.97 ≈ 15.9k, drop8 1.29 ≈ 21.1k, drop13 2.10 ≈ 34.4k.

**There is a cliff between drop6 and drop8.** Adding blocks 6 and 18 takes wikitext from +15% to +74%.
Up to drop6 the damage grows slowly (each extra block costs about 1–2 points). drop6 is the
knee and the only slice worth a repair spend: about 16k tokens of headroom at +20% fineweb
unrepaired. Wikitext, the scorer's own calibration domain, is hit hardest, so the
scorer's cos-similarity ranking does not track wikitext loss well past the first few blocks.

Disk on the Altra (2026-11-13): the HF-format 27B (`~/m0/registry/Qwen3.8-27B`, 52 GB, used only
by the scorer, re-downloadable) was deleted, along with the drop4/8/13 IQ3_M slices (each takes
about 5 min to rebuild from the BF16 source with `surgery.gguf_drop` + remapped imatrix). drop6-IQ3_M stays as the repair
candidate. The BF16 source shards stay, since every slice is cut from them.

First read, written when only drop4 had landed: it frees 0.65 GiB, which is about 10k tokens of f16 KV by the §0j.10
arithmetic, for +16% fineweb. **Unrepaired, it does not beat the KV-quant route**, which should free
about 0.5 GiB at 16k ctx with no surgery (⚠ unverified on this model). The 27B is
clearly more redundant per dropped layer than the 4B, though. 4B drop2 (6% of layers)
unrepaired was +34% fineweb (13.95 vs 10.44); 27B drop4 (6%) is +16%. The repair question
stays open until the larger drops land.

**The other two routes landed (2026-11-13, Altra CPU floor, `results/multicorpus.jsonl`).**

| 27B variant | size | wikitext | fineweb | code | frees vs IQ3_M |
|---|---:|---:|---:|---:|---:|
| IQ3_M unsliced, f16 KV, flash attn (control) | 11.89 GiB | 6.129 | 8.697 | 2.565 | – |
| IQ3_M unsliced, **q8_0 KV** | 11.89 GiB | 6.135 | 8.687 | 2.562 | half the KV: 1 GiB at 32k ctx |
| **IQ3_XS** (mtp3s) | 11.32 GiB | 6.190 (+1.0%) | 8.788 (+1.0%) | 2.582 (+0.7%) | 0.57 GiB ≈ 9.4k tok f16 |
| drop6 unrepaired IQ3_M | 10.92 GiB | 7.062 (+15%) | 10.456 (+20%) | 2.714 (+6%) | 0.97 GiB ≈ 15.9k tok f16 |

- **q8_0 KV is free at this window:** every corpus within ±0.1% of its f16-KV control. It halves
  KV to 32 KiB/token, so at 32k ctx it frees 1 GiB, more than drop6, with no surgery. The floor
  is 512-token chunks; KV-quant error at long windows is ⚠ unverified (longwindow run next).
- **IQ3_XS frees 0.57 GiB, not the ~1 GiB estimated above,** for about +1% everywhere.
- **drop6 does not pay for a repair.** To match IQ3_XS per GiB it would need fineweb back to
  about 8.8, i.e. 94% of its gap closed. The best 4B repair, distillation (§0j.12), closed 61% of
  the [25,29] fineweb gap (13.95 → 11.80, base 10.44); 61% on drop6 gives about 9.38 (+8%)
  (⚠ extrapolated across model sizes, not measured).

**Verdict:** the context lever on the 27B is IQ3_M (or IQ3_XS when the extra 0.57 GiB matters) with
q8_0 KV. The drop6 cloud repair is shelved, not run. It reopens only if a measurement shows the 27B
recovering far better than the 4B did.

### 0j.11 — KLD floor: perplexity was scoring train/test overlap, not repair (2026-11-13)

**Why.** PPL rewards predicting a corpus; it ranked wikirep above the base on wikitext while
wikirep looped forever (§0c). KL divergence against the base measures how far each model's
next-token distribution moved, which domain adaptation can't game (TODO M2). Tool:
`surgery/kld.py` (same 3 corpora, 24 chunks x 512 ctx as `multicorpus.py`); sweep:
`scripts/qwen35-4b/kld-sweep.sh`; ledger `results/kld.jsonl`.

**Setup.** Desktop RTX 5080, llama.cpp e95426f (CUDA, the Altra's commit). Base reference =
`qwen35-4b-base-f16.gguf`; parity: base code PPL 3.1737 vs 3.17 on the Altra, and each arm's PPL
is within 0.03 of its ledger row (sftrep3 had no floor row). Arms are the same f16 GGUFs the floors used (Altra), except
lr3e-6, lr1e-6 and sftrep3, converted from their archived hf weights. Reference logits: 3.0 GB per
corpus, deleted after. 11 arms; the rank/lr sweeps and SFT v1/v2 skipped (already settled).

| arm | wiki PPL / KLD | fineweb PPL / KLD / top-1 | code PPL / KLD |
|---|---|---|---|
| unrep [24,28] | 12.82 / 0.284 | 13.55 / 0.272 / 79.0% | 3.572 / 0.142 |
| unrep [25,29] | 12.81 / 0.316 | 13.95 / 0.300 / 76.6% | 3.610 / 0.149 |
| wikirep | 8.17 / **0.660** | 19.47 / **0.715** / 65.2% | 3.901 / 0.268 |
| codefw | 9.98 / 0.377 | 12.35 / 0.261 / 79.5% | 3.282 / 0.222 |
| mixfull | 8.43 / 0.520 | 12.39 / 0.298 / 77.7% | 3.364 / 0.210 |
| iso-all | 8.36 / 0.634 | 12.22 / 0.281 / 78.5% | 3.511 / 0.162 |
| 2529-pass1 | 8.04 / 0.632 | 11.89 / 0.264 / 78.5% | 3.485 / **0.147** |
| 2529-fresh1e | 7.78 / 0.561 | 11.55 / 0.223 / 80.3% | 3.185 / 0.212 |
| full lr3e-6 | 7.63 / 0.573 | 11.36 / 0.227 / 80.2% | 3.257 / 0.247 |
| full lr1e-6 | 7.69 / 0.535 | 11.38 / **0.212** / **81.4%** | 3.181 / 0.184 |
| sftrep3 | 10.57 / 0.306 | 12.73 / 0.249 / 80.1% | 3.260 / 0.190 |

**Findings.**
1. **KLD catches the domain trap PPL missed.** wikirep: best-in-class wikitext PPL, worst KLD on
   every corpus (2.3x the unrepaired wound on its own training domain).
2. **The wikitext PPL wins are specialization.** Every repair beats the base on wikitext PPL
   (7.6–8.4 vs 9.33) while roughly doubling wikitext KLD over the unrepaired wound (0.52–0.66 vs
   0.28–0.32). The diets contain wikitext-103 train, so the wikitext row measured train/test overlap.
3. **Code is the same, more mildly.** Repairs that reach base code PPL (fresh1e 3.185, lr1e-6 3.181)
   do it by drifting further (KLD 0.21/0.18) than 2529-pass1 (0.147, worst-case p99 1.37, the lowest
   of any arm). code.train and code.test are both llama.cpp (review r2 f8).
4. **Fineweb is the honest row,** and there the repairs really do move toward the base: fresh1e
   0.223 and full lr1e-6 0.212 (vs 0.300 unrepaired), top-1 agreement up 4–5 points. The ranking
   holds there: fresh data and full weights at low lr are the best repairs; the lead is smaller
   than the wiki/code PPL suggested.
5. sftrep3 (SFT on the triage corpus) moves least on wikitext of any repair (0.306): chat-format
   training doesn't learn the corpus style.

**Consequences.** The floor reports KLD next to PPL from now on, and fineweb KLD is the headline
repair metric; wikitext and code PPL need a held-out source disjoint from the training diet before
they rank repairs again. Distillation (train on the base's distribution instead of corpus text,
the next experiment) is judged on fineweb KLD and top-1, not on PPL.

### 0j.12 — Distillation repair: the slice learns the base, not the corpus (2026-11-13)

**Why.** §0j.11 showed CE repair buys wikitext/code PPL by specializing on the diet. Training
toward the base's next-token distribution instead can't specialize: the target is the model we
cut. `recover_gpu.py --distill <base>`: loss = per-position KL(base ‖ slice) at T=1, the bf16 base
frozen on the same GPU, KL in checkpointed 32-position chunks.

**Setup.** fresh1e exactly (§0j.9: [25,29] slice, whole `mixed.train.txt`, 1 epoch, QLoRA r16
all targets, lr 1e-4, seed 1234, 1% held out), except batch 2 x accum 2: batch 4 with the
teacher resident peaked at 16.3 GiB and spilled to host memory (260 tok/s); 2x2 peaks at 13.0
GiB, 1245 tok/s, 2.1 h. Held-out KL 0.323 → 0.127, still falling. Card:
`cards/arms/qwen35-4b/2529-distill.md`.

| f16 | wiki PPL / KLD / p99 | fineweb PPL / KLD / p99 / top-1 | code PPL / KLD / p99 |
|---|---|---|---|
| unrepaired [25,29] | 12.81 / 0.316 / 3.11 | 13.95 / 0.300 / 3.50 / 76.6% | 3.610 / 0.149 / 1.70 |
| 2529-fresh1e (CE) | 7.78 / 0.561 / 6.52 | 11.55 / 0.223 / 2.40 / 80.3% | 3.185 / 0.212 / 2.39 |
| full lr1e-6 (CE) | 7.69 / 0.535 / 6.08 | 11.38 / 0.212 / 2.41 / 81.4% | 3.181 / 0.184 / 2.03 |
| **2529-distill (KL)** | 9.99 / **0.141** / **1.67** | 11.80 / **0.147** / **1.84** / **84.2%** | 3.280 / **0.067** / **0.80** |

**Findings.**
1. Same text, same steps, the objective alone: fineweb KLD 0.223 → 0.147 (-34%), top-1 agreement
   80.3% → 84.2%. It beats the full-weight A100 arms from a 4-bit QLoRA on a desktop.
2. Wikitext KLD 0.561 → 0.141 and code 0.212 → 0.067: the specialization §0j.11 found is gone.
   Code KLD is below the unrepaired wound's (0.149), the first repair to manage that; every CE
   repair made code drift further than the wound itself did.
3. The tails shrink too (p99 down 2-4x vs fresh1e), so it isn't a mean bought with a few bad tokens.
4. PPL now ranks it below fresh1e on every corpus. That is the §0j.11 lesson in one row: fresh1e's
   lower PPL was diet-fitting, not closeness to the base.

**Consequences.** Distillation is the repair objective from here, and the 27B drop6 repair should use
it (teacher = the uncut 27B, which does not fit beside the student on the 5080: top-k teacher logits
precomputed offline, or the cloud lane). Before it replaces fresh1e on the M5 shelf it needs the
task/chat board: KLD says it is closer to the base, the board says whether that buys behaviour.

**Board (2026-11-13, chat mode, desktop 5080, `-ngl 99`, f16; `scripts/qwen35-4b/board-2529.sh`).**
GPU rows are their own surface, so base and the wound were re-run beside the repairs:

| arm | coding | instruct | needle | composite | loop-thresh | gen t/s |
|---|---:|---:|---:|---:|---:|---:|
| base | 1.0 | 0.6 | 1.0 | **0.867** | 0.7 | 72.9 |
| unrepaired [25,29] | 0.75 | 0.6 | 1.0 | **0.783** | 0.7 | 83.2 |
| 2529-fresh1e | 1.0 | 0.6 | 1.0 | **0.867** | 0.7 | 87.8 |
| **2529-distill** | 1.0 | 0.6 | 1.0 | **0.867** | 0.7 | 86.4 |

Base on the GPU reproduced its CPU chat row exactly (§0h). The wound's only loss is a syntax
error (`import from collections` in code-wordfreq); both repairs fix it, and both miss the same two
instruct tasks base misses (exact word count, constraint set). **The board is saturated at this
wound:** it cannot separate the repairs from each other or from base. The distill arm costs nothing
on behaviour and is closer by every KLD row, so it replaces fresh1e on the shelf. Ranking repairs
against each other from here needs either a harder board or KLD; the 9-task v1 board only catches
gross damage.

## 5. Milestones

Tracked in [`TODO.md`](../TODO.md); summary:

- **M0 — bring-up (hardware):** OS, llama.cpp ARM build pin, baseline table: stock Qwen quants
  (4B/9B ⬜-sized models) × Q4_K_M/Q8_0, tok/s, TTFT, RSS at 8k/64k ctx. Every ⬜ in this doc that
  concerns the box gets measured here.
- **M1 — surgery loop (desktop, small models):** one recipe end-to-end: Qwen3-4B → drop+slice →
  convert → quantize → PPL+speed, validated on the Ryzen. Proves the engine before the box exists.
- **M2 — suite:** floor + academic subset + first custom tasks (coding, instruct), stage-A smoke.
- **M3 — funnel + reports:** grid builder, submit CLI, leaderboard/Pareto reports. First real run:
  27B → 5080-comfort (~20–22B params; goal corrected 2026-11-10, §1 note).
- **M4 — recovery:** llama.cpp LoRA stage C on a stage-A survivor, before-and-after scores.
- **M5 — MoE:** franken-MoE assembly; then moeify + recovery if llama.cpp MoE LoRA checks out.

## 6. Non-goals

- Not a serving platform — gpu-scheduler serves; the forge ships artifacts + recipes + rankings.
- No distributed/multi-node training. Single box, single worker (until proven too slow).
- No pre-training, no tokenizer surgery (vocab stays fixed so perplexities stay comparable).
