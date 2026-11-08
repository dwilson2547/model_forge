# 2026-11-07: torch-on-aarch64 bf16 matmul is pathologically slow; training must be fp32

**Date:** 2026-11-07
**Component:** model-forge M4 torch-CPU recovery (forge_env torch 2.15.0+cu132, aarch64, Neoverse-N1, 128 cores)

## Symptom

The first LoRA recovery probe (bf16, peft, Qwen3.5-4B drop2 edit) ran **16.5 hours without
finishing 100 tiny steps** (~51k training tokens), pinned at ~4.6 cores of CPU. A controlled
bench (96 torch threads, batch 2 × ctx 256) sat **16 minutes without completing one step**.
A single 3072×3072 **bf16** matmul hung for >120 s — thousands of times slower than it should be.

## Root cause

torch on aarch64 (this build) dispatches bf16 `mm`/`addmm` to a fallback path that is
effectively scalar-per-element (compounded by torch's warning frames from
`aten/src/ATen/native/LinearAlgebra.cpp:1516` during model forward). It is not merely
"bf16 slower than fp32" — it is unusable (>1000× slowdown). This matches the same
pattern recorded for embeddings work on Zen 3 (`ai/docs/notes`: bf16 slower than fp32
on CPU) but at pathological scale on aarch64.

Measured on the box (32 threads, 3072×3072 matmul):

- **fp32: 1372.8 GFLOP/s (42.2 ms/op)** — healthy; NEON/dotprod path engaged
- **bf16: failed to complete a single op in 120 s** — pathological fallback

## Resolution

All torch-CPU training in the forge runs **fp32**, never bf16/fp16 (RAM headroom is not
a constraint on this box: 4B fp32 = 16 GiB weights vs 250 GiB RAM). `surgery/recover.py`
loads `dtype=torch.float32` and defaults to 96 threads. Any future torch stage (scoring
v2 activation probes, MoE-ification) must assume the same: fp32 compute, and if a bf16
load ever shows <10% CPU utilization while "running", suspect this before anything else.

**Status:** resolved 2026-11-07 in favor of fp32; the wasted bf16 probe run was killed.
