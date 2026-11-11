# qwen35-4b-drop2-sftv3-triage — how this model was made

*At-a-glance provenance card for the drop2 surgical slice of Qwen3.5-4B, LoRA-repaired
on a private incident-triage corpus and taught to close its own thinking block.
Benchmarked 2026-11-11; best repaired arm of the model-forge project to date.*

## Lineage

| stage | what happened | result |
|---|---|---|
| **0 · base** | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | 36 layers, hybrid attention (linear layers + full-attention every 4th), MTP draft block, ~4.0B params |
| **1 · surgery** | 2 layers dropped — **24 & 28** — by neighbor-cosine importance ranking; full-attention layers pattern-protected, min 2-layer spacing | 34 layers, **~3.76B params (94.6%)**; wiki PPL 9.33 → 12.82 unrepaired |
| **2 · repair** | QLoRA on a single RTX 5080, 20 min: NF4 base, **r16 @ lr 1e-4**, 2 epochs on the v3 triage corpus (below); merged to bf16 | PPL damage mostly healed; task parity restored |
| **3 · prosthetics** | MTP draft block re-grafted from the pristine base (15 tensors; pipeline drops it, runtimes assert on it) | LM Studio/llama.cpp loadable |
| **4 · quant** | f16 GGUF convert, **Q4_K_M** on the Ampere Altra (128-core aarch64) | 2.63 GB playable |

## The v3 triage corpus (the interesting part)

Private to the homelab — extracted from the **incident-operator's Postgres history**
(the pipeline that triages real cluster alerts with a local LLM):

- **62 real turns × 4 upweight** from 30 incidents, across three session protocols —
  prose triage, JSON investigation turns, remediation advice — flattened to
  `(bundle, reasoning, answer)` records
- **13 runbooks + 17 probe operations + the protocol primer** as knowledge pairs
- **~3,000 continuation pairs** from the standard repair diet
  (61% FineWeb-EDU / 27% wikitext-2 / 12% code) so the surgery wound stays healed
- **Held out, never trained on:** incidents 10, 28, and 46

Targets are rendered through the chat template's own `reasoning_content` path — the
model is explicitly taught to *close its think block and then answer*. That one detail
is the difference between "Thought for 1.23 seconds… then died" and a usable bot.

## Results at a glance (chat-mode task board, f16)

| arm | coding | instruct | needle | composite |
|---|---:|---:|---:|---:|
| base Qwen3.5-4B | 1.0 | 0.6 | 1.0 | 0.867 |
| drop2 unrepaired | 1.0 | 0.2 | 1.0 | 0.733 |
| **this model** | **1.0** | 0.2 | **1.0** | **0.733** |

- Coding tier is docker-graded: model-written code executed against real unit tests.
- **Held-out test** (incident 46, unseen in training): produced a causally correct
  diagnosis in the operator's exact JSON schema — rclone OOMKilled at its 4Gi limit
  from on-disk sorting of a 200k-entry directory, correctly excluding a hang — with the
  answer in the visible region and a clean stop at 635 tokens.
- Remaining gap vs base: instruction constraint precision (writes the right *shape*,
  overproduces on exact counts). Next knob: constraint-bearing diet examples.

## Where the infrastructure did the heavy lifting

- **Training:** single consumer GPU (5080, 16 GB), 3.2 GB VRAM in use, gradient
  checkpointing, 4-bit QLoRA. The whole run: 20 minutes.
- **Surgery + eval + conversion:** Ampere Altra (128× Neoverse-N1), CPU-only llama.cpp.
- Full numbers and the two failed predecessors (v1: format without semantics; v2:
  semantics without think-close) are in `docs/design.md` §0h–§0i of the model-forge repo.
