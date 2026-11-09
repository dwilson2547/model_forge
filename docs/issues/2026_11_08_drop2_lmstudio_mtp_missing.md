# 2026-11-08: forge exports crashed LM Studio — MTP block silently dropped by the surgery pipeline

**Date:** 2026-11-08
**Component:** model-forge build/export stage (`surgery/pipeline.py` save→convert path)

## Symptom

All four forge GGUFs shipped to LM Studio failed to load with engine fast-fail
(`exitCode=3221226505` = `0xC0000409`). LM Studio's server logs carried the exact assert:

    llm-engine\llama.cpp\src\models\qwen35.cpp:508:
    GGML_ASSERT(layer.nextn.eh_proj && "MTP block missing nextn.eh_proj") failed

The same GGUFs loaded and evaluated fine on our vanilla llama.cpp `e95426f` — the vanilla
runtime treats nextn tensors as optional, so every forge number remained valid.

## Root cause

`Qwen/Qwen3.5-4B` carries a multi-token-prediction block (15 tensors: `mtp.fc`,
`mtp.layers.0.*`) plus `mtp_num_hidden_layers: 1` in its config. Our pipeline loads via
`AutoModelForCausalLM.from_pretrained` — the **text model only, 426 of 738 tensors** — so
`save_pretrained` wrote edited checkpoints with *no* `mtp.*` tensors while the config still
declared `mtp_num_hidden_layers: 1`. The GGUF converter then emitted
`qwen35.nextn_predict_layers = 1` with zero `blk.N.nextn.*` tensors: structurally
inconsistent GGUFs. Tolerant runtimes ignore it; LM Studio's pinned runtime asserts.

Also learned: `nextn` presence is not a metadata flag — the loader infers it from
`blk.{n_layer_trunk - 1}.nextn.*` tensors (blk.32 for the 36-layer stock model with
`qwen35.block_count=33`; blk.30 for our 34-layer slices with block_count 31).

## Resolution

`surgery/graft_mtp.py` (box: `~/m0/graft_mtp.py`): transplant the 15 `mtp.*` tensors from the
pristine base checkpoint into any edited `model.safetensors` (config already carries the
flags), then re-convert normally — the converter then emits `blk.30.nextn.*` at the correct
slice-relative index. All four LM Studio models rebuilt this way; verified on vanilla
(generation clean) before shipping.

**MTP semantics after slicing:** the grafted draft weights belong to the *original* model, so
speculative-decode acceptance rates are approximate after surgery/recovery — draft outputs are
always verified by the main model, so correctness is unaffected; only speculative speedup may
vary. If a runtime misbehaves with MTP on, disable speculative decoding for that model.

**Pipeline debt (TODO):** bake the graft into `surgery/build.py` (`save_hf` → transplant from
the pinned base) so every future candidate exports LM-Studio-compatible by construction, and
re-quantize from the grafted HF, never from the raw edited dir.
