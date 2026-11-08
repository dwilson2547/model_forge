# 2026-11-07: llama.cpp finetune cannot train on CPU-only builds; qwen3_5 hybrids can't build backward at all

**Date:** 2026-11-07
**Component:** model-forge M4 recovery stage — `llama-finetune` (llama.cpp `e95426f` master and `v0.9.0` release, aarch64 N1, CPU-only)

## Symptom

`llama-finetune` aborts before completing a single optimizer step, in every configuration tried:

| model | type | failure |
|---|---|---|
| Qwen3.5-4B drop2 candidate (our edit, f16) | hybrid qwen3_5 | `ggml_compute_backward: unsupported unary op for backward pass: SIGMOID` → fatal in `ggml_build_backward_expand` |
| Qwen3-1.7B Q8_0 | dense qwen3 | `GGML_ASSERT(*cur_backend_id != -1) failed` (ggml-backend.cpp) |
| Qwen3-1.7B BF16 | dense qwen3 | same assert |

Same assert reproduces on both the `v0.9.0` stable tag and post-`v0.9.0` master. Falling back to
older releases does not help: `v0.8.0` already contains the rewritten ggml_opt-based finetune and
does **not** know the `qwen3_5` architecture at all — versions old enough to predate the rewrite
cannot load the models we need to repair.

## Root cause (two independent gaps)

1. **CPU-only builds cannot run the new finetune.** The assert fires in `ggml_backend_sched_split_graph`
   pass 4 — a node in the backward/optimizer graph cannot be assigned to any loaded backend. With only
   the CPU backend, ops in the backward graph have no CPU kernel in this release. This is a known
   issue class with independent upstream reports (reports match "crashes after *Starting epoch 0*
   on CPU-only builds"); all official finetune documentation runs with `-ngl 999` (GPU).
2. **qwen3_5 hybrids add a second blocker before that:** the gated-delta-rule layers' graph contains
   `GGML_OP_SIGMOID`, which has no backward implementation in `ggml_compute_backward` — hybrid models
   cannot build a backward graph regardless of (1).

## Resolution / workaround

**Recovery engine pivot:** M4 recovery moves from llama.cpp CPU finetune to
**PyTorch-CPU training via transformers' reference implementations**.

**Follow-up, same day — measured, and it kills the workaround too:** torch-CPU training on
this model/arch runs at **~2 tok/s** at the box's best (fp32 after the bf16 pathology was
found and fixed — see `2026_11_07_torch_bf16_matmul_slow.md`; thread/batch sweeps could not
move the needle: the reference `chunk_gated_delta_rule`/`causal_conv1d` implementations are
thousands of tiny sequential ops per chunk, and no CPU thread config fixes per-op overhead).
At 2 tok/s a modest 10M-token repair corpus is **60+ days** of exclusive box time. CPU
recovery — llama.cpp or torch — is quantitatively ruled out on this hardware, not just
broken: the recovery stage needs the desktop 5080 (or cluster 3080) as the training worker,
with the Altra remaining the surgery + eval engine. See design D1 (amended twice) and
TODO M4.

**Status:** llama.cpp GGUF-side recovery abandoned (2026-11-07); torch-CPU recovery
measured infeasible at 2 tok/s (2026-11-07); recovery stage points at a GPU worker.
