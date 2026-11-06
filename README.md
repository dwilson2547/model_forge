---
tier: project
---

# model-forge

CPU model-making pipeline for the Ampere Altra node: structural surgery (layer drop/fuse, FFN/head
slicing), quantization, franken-MoE / upcycled MoE assembly, and a two-stage benchmark funnel that
ranks the candidate models on the hardware they'll run on. Fire a job — *"shrink Qwen3.8-27B to
~9B"* — and get back a leaderboard: which surgery+quant configs survived, measured on quality,
tok/s, RAM, and disk.

**Status: design phase (2026-11-06).** Hardware not yet set up; no code yet. Full architecture,
decisions, and provenance-marked plan: [`docs/design.md`](docs/design.md). Milestones and the
bring-up checklist: [`TODO.md`](TODO.md).

## Layout

- `docs/design.md` — the design (pipeline, recipe format, benchmark tiers, funnel, milestones)
- `TODO.md` — M0–M5 milestones; M0 is the hardware bring-up baseline
- Code (when M1 starts): `forge/` CLI + worker, `surgery/` ops, `bench/` suite, `runs/` artifacts

Plays well with the rest of the `ai` domain: serving stays on `gpu-scheduler`; recovery fine-tuning runs on the Altra itself, CPU-only, via llama.cpp.
