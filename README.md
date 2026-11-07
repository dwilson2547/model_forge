---
tier: project
---

# model-forge

CPU model-making pipeline for the Ampere Altra node: structural surgery (layer drop/fuse, FFN/head
slicing), quantization, franken-MoE / upcycled MoE assembly, and a two-stage benchmark funnel that
ranks the candidate models on the hardware they'll run on. Fire a job — *"shrink Qwen3.8-27B to
~9B"* — and get back a leaderboard: which surgery+quant configs survived, measured on quality,
tok/s, RAM, and disk.

**Status: design phase (2026-11-06); M0 hardware bring-up done (2026-11-07).** The Ampere Altra is up:
128× Neoverse-N1, 250 GiB RAM, llama.cpp `e95426f` built, baseline table measured — see
[`docs/design.md`](docs/design.md) §0. Code milestones M1–M5 are next; current state in
[`TODO.md`](TODO.md).

## Layout

- `docs/design.md` — the design (pipeline, recipe format, benchmark tiers, funnel, milestones)
- `TODO.md` — M0–M5 milestones; M0 is the hardware bring-up baseline
- Code (when M1 starts): `forge/` CLI + worker, `surgery/` ops, `bench/` suite, `runs/` artifacts

Plays well with the rest of the `ai` domain: serving stays on `gpu-scheduler`; recovery fine-tuning runs on the Altra itself, CPU-only, via llama.cpp.
