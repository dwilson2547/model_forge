# model-forge — AI-Assisted Engineering Session Showcase
**November 2026 · Mistral Vibe CLI**

---

## What Is This?

[model-forge](../README.md) is a planned CPU-only "model-making" pipeline for an Ampere Altra
homelab server (128× Neoverse-N1 ARM cores, 250 GiB RAM, no GPU): take a base LLM checkpoint
(the driving example: shrink Qwen3.8-27B to a ~9B model), apply recipe-driven structural
surgery — layer dropping/fusing, FFN/head slicing, quantization — assemble MoE variants, and
rank every candidate against a measured-on-target benchmark funnel. This session took the
project from a conversational idea to a designed, decision-recorded repo, then brought the
Altra hardware up and completed milestone M0: a fully measured baseline table plus a
pinned benchmark methodology. Key technologies: llama.cpp (GGUF inference engine on
aarch64/NEON), Hugging Face model distribution, self-hosted git submodules, and Bash-driven
remote benchmarking over SSH.

---

## What Was Already in Place at Session Start

- The Ampere Altra machine (ASRock Rack ALTRAD8UD-1L2T board), not yet set up, earmarked as the future inference/model-making node
- A workspace superrepo with strict conventions (closed domain taxonomy, submodules, provenance rules for measured vs planned values)
- Adjacent `ai/` projects providing reusable playbook patterns: `gpu-scheduler` (job queue/lease shape, measured llama.cpp numbers on the cluster's RTX 3080) and `local-model-builder` (sandboxed generate-and-test grading harness)
- No prior work anywhere in the workspace on model surgery, quantization, evaluation, or the Altra itself

---

## Task 1 — Design the pipeline with decisions recorded

**Request:** *"I'd like a series of inference tests to rank models in different tasks… also interested in layer tuning / structural weight slicing, quantizing models myself, and making our own MoE models — fire off a job to size down Qwen 3.8 27B to a 9B model, running through a number of pruning/slicing configs and benchmarking each result."*

**What happened:**
- Searched workspace notes (`doc-indexer`) for prior art; found the reusable patterns but no pruning/eval work — confirmed greenfield
- Proposed the architecture: declarative surgery **recipes** → tensor-surgery engine → GGUF convert/quantize → **two-stage benchmark funnel** (cheap smoke eval for every grid candidate, full suite only for survivors) → leaderboard/Pareto report
- Asked four scope questions, all answered: recovery fine-tuning runs **on the Altra via llama.cpp CPU LoRA** (D1); benchmark suite is **hybrid** — llama.cpp perplexity + lean lm-eval subset + custom graded tasks (D2); orchestration is **bare-metal CLI-first** with a k8s-wrappable queue boundary (D3); **design doc first**, code at hardware bring-up (D4)
- Created the `ai/model-forge` submodule (new private GitHub repo `dwilson2547/model_forge`): `docs/design.md` (architecture, recipe format, funnel, milestones M0–M5), `README.md` with tier marker, `TODO.md`, `.gitignore`
- Every unmeasured value written with ⬜ markers per workspace provenance rules
- Fixed en route: `git submodule add` against an empty GitHub repo silently skipped `.gitmodules` registration — caught it, appended the entry, re-registered

**Outcome:** A decision-recorded design doc in a properly placed submodule, pushed, indexed by the workspace doc layer, with a graded milestone plan.

---

## Task 2 — Hardware bring-up, no sudo required

**Request:** *"Box is up and running, use `ssh ampere` to access it, let me know the sudo commands that need to be run."*

**What happened:**
- Inventoried the machine over SSH: Ubuntu 24.04.5 aarch64, 128× Neoverse-N1 @ 3.0 GHz single socket/single NUMA, 250 GiB RAM, 915 GB NVMe (730 GB free), `asimd`/`asimdhp`/`asimddp` SIMD (no SVE/i8mm — expected on N1)
- Found a complete build toolchain (gcc 13.3, cmake 3.28), a working `python3-venv`, and an existing `~/llama_env` venv already holding transformers, safetensors, huggingface-hub, and llama-cpp-python — docker usable without sudo
- **Reported zero sudo commands needed** and set an optional one aside (`cpupower` performance governor) in case of benchmark jitter
- Cloned and built llama.cpp pinned at `e95426f` (0.9.0-dev) on the box with the full tool suite (`llama-bench`, `-perplexity`, `-quantize`, `-imatrix`, `-export-lora` — tools for milestones M1–M4)

**Outcome:** The forge's inference engine built and running on the target hardware in one step, with no human intervention beyond the OS install Daniel had already done.

---

## Task 3 — M0 baseline: measured numbers and a pinned methodology

**Request:** *(implicit from the M0 milestone)* establish the Altra's real inference characteristics.

**What happened:**
- Downloaded six stock models into a registry on the box (`~/m0/registry/`, ~66 GB): Qwen3.5-4B and 9B in Q4_K_M + Q8_0, Qwen3.8-27B in UD-Q4_K_M + Q8_0 (canonical `unsloth` GGUFs)
- Ran the `llama-bench` grid (pp512/tg128, t32 vs t96). **Headline finding:** prompt processing scales near-linearly to ~96 threads (up to 335 t/s on the 4B), while decode is bandwidth-bound and *collapses* past ~64 threads — the 4B does 24 t/s at t=32 but **2 t/s at t=96** (12× degradation from per-token barrier overhead)
- Debugged two harness bugs that cost reruns, both now codified in the design doc: `llama-cli` **hangs forever on open stdin** when scripted; and this build prints timings to **stdout** as `[ Prompt: X t/s | Generation: Y t/s ]`, not the classic stderr format
- Ran the corrected decode-thread sweep (t ∈ {8..32} at `-tb 96`): tg rises monotonically to t=32 then flattens → **pinned eval config `-t 32 -tb 96`**
- Measured peak RSS per model (14.0–70.8 GiB, 2–3× GGUF file size): the 27B Q8 with full eval footprint still leaves ~175 GiB for concurrent surgery work
- Verified the project premise numerically: decode t/s tracks model size inversely (27B Q4: 4.9 t/s vs 9B Q4: 14.7) — a successful 27B→9B slice is a direct ~3× speedup plus a 15.3→~6 GiB footprint cut, with quality retention as the only open variable

**Outcome:** Milestone M0 fully measured — the design doc's §0 now carries a ✅ baseline table (pp512, tg128, peak RSS per model), the pinned benchmark methodology, and the eval cost model that justifies the two-stage funnel (a 256-token scored sample costs ~53 s on the 27B).

---

## Task 4 — Ampere's optimized images: evaluated and ruled out

**Request:** *"Might be worth looking into… seems like there's a pytorch optimized for ampere (`AmpereComputingAI/ampere_model_library`)."*

**What happened:**
- Catalogued what Ampere actually ships on Docker Hub: a benchmark model zoo (ResNet/BERT/Whisper-era) plus framework images — pytorch, tensorflow, onnxruntime, **and an actively maintained fork-build of llama.cpp and Ollama**
- PyTorch images: newest tag carries **torch 1.12.1** (2022-era, kept for their old zoo) — modern Qwen3.x architectures require torch 2.x and won't load, so the images are unusable for this project's surgery stack; fallback recorded (stock aarch64 torch 2.x, or Ampere's newer AOP wheels if ever installable without a login wall)
- llama.cpp image (`latest`, v8896/`905ccccf2`, Clang 19.1.7, with an `LICENSE_AMPERE.txt`-marked patch layer): pulled to the box and A/B'd against our vanilla build, including their `/.aio_init/init.sh` tuning — **SIGILL core dump during model load on the N1**: even the plain tag is built to AmpereOne/N2 baseline, so it cannot run on Neoverse-N1 at all
- Recorded both verdicts, saving future sessions from re-running this evaluation

**Outcome:** Confirmed: no benefit from the Ampere container ecosystem on this machine — vanilla llama.cpp `e95426f` stays the inference engine and stock torch 2.x the surgery stack; findings committed to TODO/design.

---

## Summary of Infrastructure Built

| Component | Technology | Notes |
|---|---|---|
| Altra node | 128× Neoverse-N1 @ 3.0 GHz, 250 GiB RAM, 730 GB NVMe free | The forge's target and only compute resource; single NUMA, governor `ondemand` |
| Inference engine | llama.cpp `e95426f` (0.9.0-dev), GNU 13.3 aarch64 | At `~/llama.cpp` on the box; full tool suite for M1–M4 |
| Model registry | 6 unsloth GGUFs, ~66 GB, `~/m0/registry` | Qwen3.5-4B/9B × Q4_K_M/Q8_0, Qwen3.8-27B × UD-Q4_K_M/Q8_0 |
| Pinned benchmark methodology | `-t 32 -tb 96`, `llama-cli` stdin guard | From measured thread-collapse behavior on 128 cores |
| Workspace docs layer | design doc §0 + TODO M0–M5 + this showcase | All measurements carry ✅/⬜ provenance per workspace §10 |
| Project home | `ai/model-forge` submodule → `dwilson2547/model_forge` (private) | `cluster_config`-style placement; design phase → M0 done |

## Commits in This Session

| Hash | Description |
|---|---|
| 8a8910f | Design doc: CPU model-surgery + benchmark pipeline for the Ampere Altra node |
| 67cffad | (superrepo) ai/model-forge: new submodule, design doc for the Altra model-making node |
| c3f4d23 | (superrepo) .gitmodules: register ai/model-forge (silently skipped by empty-repo checkout) |
| e56dea6 | M0 hardware bring-up: baseline table measured on the Altra |
| 14f496c | (superrepo) Submodule bump: model-forge M0 bring-up done, baseline table in design doc §0 |
| f9fb78c | M0 complete: decode-thread optima, peak RSS, Ampere images ruled out |
| 30b8c62 | (superrepo) Submodule bump: model-forge M0 complete |

---

*Document generated 2026-11-07. Repository: `dwilson2547/model_forge` (superrepo: `dwilson2547/workspace`).*
