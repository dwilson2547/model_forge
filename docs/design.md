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

First recipes through the full `surgery/` pipeline on Qwen3.5-4B (36-layer hybrid: 28 linear-
attention + 8 full, full every 4th, protected from drop). PPL on wikitext-2, 24 chunks; speed
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

1. **Depth:** dropping 8/36 layers costs PPL 9.3 → 40.4. The neighbor-cosine scorer dropped
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

## 1. Purpose

Turn the Ampere Altra box into a **model-making node**: fire a job like *"shrink Qwen3.8-27B to a ~9B
model"* and get back a ranked leaderboard of candidate models produced by different structural-surgery +
quantization configs, benchmarked across task groups on the machine they will actually run on.

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
| D1 | Recovery fine-tuning runs **on the Altra via llama.cpp CPU LoRA** — no GPU dependency, hours-to-days runtimes accepted. The job schema keeps recovery as an optional stage with a `worker` field, so a 3080/5080 recovery worker can be added later without schema changes. **Amended 2026-11-07 during M4:** llama.cpp finetune cannot train on CPU-only builds (known upstream gap) and qwen3_5 hybrids cannot build backward graphs at all — recovery engine moves to **torch-CPU via transformers reference implementations** (same box, same no-GPU spirit); see `docs/issues/2026_11_07_llamacpp_cpu_finetune.md`. | Delegating to the GPU queue from day one; skipping training entirely |
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

## 5. Milestones

Tracked in [`TODO.md`](../TODO.md); summary:

- **M0 — bring-up (hardware):** OS, llama.cpp ARM build pin, baseline table: stock Qwen quants
  (4B/9B ⬜-sized models) × Q4_K_M/Q8_0, tok/s, TTFT, RSS at 8k/64k ctx. Every ⬜ in this doc that
  concerns the box gets measured here.
- **M1 — surgery loop (desktop, small models):** one recipe end-to-end: Qwen3-4B → drop+slice →
  convert → quantize → PPL+speed, validated on the Ryzen. Proves the engine before the box exists.
- **M2 — suite:** floor + academic subset + first custom tasks (coding, instruct), stage-A smoke.
- **M3 — funnel + reports:** grid builder, submit CLI, leaderboard/Pareto reports. First real run:
  27B → ~9B.
- **M4 — recovery:** llama.cpp LoRA stage C on a stage-A survivor, before-and-after scores.
- **M5 — MoE:** franken-MoE assembly; then moeify + recovery if llama.cpp MoE LoRA checks out.

## 6. Non-goals

- Not a serving platform — gpu-scheduler serves; the forge ships artifacts + recipes + rankings.
- No distributed/multi-node training. Single box, single worker (until proven too slow).
- No pre-training, no tokenizer surgery (vocab stays fixed so perplexities stay comparable).
