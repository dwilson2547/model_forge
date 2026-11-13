---
tier: project
---

# model-forge

CPU model-making pipeline for the Ampere Altra node: structural surgery (layer drop/fuse, FFN/head
slicing), quantization, franken-MoE / upcycled MoE assembly, and a two-stage benchmark funnel that
ranks the candidate models on the hardware they'll run on. Fire a job — *"shrink Qwen3.8-27B to
5080-comfort"* — and get back a leaderboard: which surgery+quant configs survived, measured on quality,
tok/s, RAM, and disk.

**Status: v3 shipped (2026-11-11).** The surgery loop is validated end-to-end (M1 ✅), GPU QLoRA
recovery is the working repair path (M4 pivot — CPU recovery measured out at ~2 tok/s), the first
survivor is published (`dwilson2547/Qwen3.5-4B-drop2-mixed-repair` + GGUF sibling), and the
triage-SFT arc landed v3 (`qwen35-4b-drop2-sftv3-triage`). Measured story so far in
[`docs/design.md`](docs/design.md) §0a–§0j; current state and next steps in
[`TODO.md`](TODO.md).

## Layout

- `docs/design.md` — the design + the measured record (§0a–§0j)
- `TODO.md` — milestones, review follow-ups, the v4 SFT diet plan
- `surgery/` — ranking, ops, recipes, GPU QLoRA recovery, triage-corpus extraction
- `bench/` — serve harness + task board (chat-mode grading)
- `recipes/`, `cards/` — provenance per candidate / keeper
- `scripts/<family>/` — experiment drivers; `scripts/legacy/` — the old loose box scripts, as a record
- `publish/` — HF publish scripts
- `results/` — the ledgers (multicorpus, bench, longwindow) behind every number in `design.md`
- `docs/storage.md` — where weights live: `~/forge/` working tree on each machine, `/mnt/models` cold
  archive, and the map from the old `~/m0` / `~/forge-gpu` paths
- `forge/` CLI + funnel (M3, not started)

Plays well with the rest of the `ai` domain: serving stays on `gpu-scheduler`; recovery fine-tuning runs on the desktop/cluster GPUs (QLoRA), surgery + eval + quantize on the Altra.
