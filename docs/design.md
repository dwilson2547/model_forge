# model-forge — design

**Status:** design phase, 2026-11-06. No code yet — hardware bring-up first. Decisions D1–D4 below were
made with Daniel before this doc was written; everything else is the architecture those decisions imply.

**Provenance:** per workspace §10, every specific value below is either already measured (✅), or planned
and unverified (⬜). Nothing about the Altra machine is measured yet — not even core count and RAM. The
only verified facts are the playbook patterns borrowed from `gpu-scheduler` and `local-model-builder`.

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
| D1 | Recovery fine-tuning runs **on the Altra via llama.cpp CPU LoRA** — no GPU dependency, hours-to-days runtimes accepted. The job schema keeps recovery as an optional stage with a `worker` field, so a 3080/5080 recovery worker can be added later without schema changes | Delegating to the GPU queue from day one; skipping training entirely |
| D2 | Benchmark suite is **hybrid**: llama.cpp perplexity + a lean academic subset (GSM8K, ARC-E, HellaSwag) as a comparable quality floor, plus custom graded tasks for the work we actually run models for | Pure custom suite (uncomparable) or full lm-eval-harness (too slow on CPU) |
| D3 | Orchestration is **bare-metal CLI-first with a clean job-queue boundary** — a single `forge` CLI + worker on the box, SQLite backend, no cluster dependency. K8s-wrapping is possible later if the Altra joins the cluster, because jobs pass through one queue interface | Joining the Altra to the home cluster as an ARM node now |
| D4 | Design doc now; **code starts at hardware bring-up**. Surgery + eval code is developed and validated on the Ryzen desktop against small models (Qwen3-4B and below) — everything but raw speed is portable (x86 → NEON is llama.cpp's job) | Scaffolding in parallel with hardware setup |

## 3. Constraints

- **No GPU.** Inference is llama.cpp (NEON/dotprod/i8mm). Decode is memory-bandwidth-bound, so model
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
