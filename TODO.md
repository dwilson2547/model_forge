# model-forge TODO

Design: [`docs/design.md`](docs/design.md). Status: design phase — nothing below is started.
M1 can begin on the Ryzen desktop before hardware bring-up if desired (D4 default: wait for M0).

## M0 — Altra bring-up baseline (fills the ⬜s in the design)

- [x] OS + storage layout decided; 128-core topology confirmed (Ubuntu 24.04.5, 128× Neoverse-N1 @ 3.0 GHz,
      250 GiB RAM, 915 GB NVMe / 730 GB free, single NUMA node; SIMD: asimd/asimdhp/asimddp, no SVE/i8mm)
- [x] llama.cpp pinned ARM build (`e95426f`, 0.9.0-dev, GNU 13.3 aarch64) at `~/llama.cpp`
- [x] Baseline table: Qwen3.5-4B/9B + Qwen3.8-27B (GGUFs from `unsloth`, `~/m0/registry/`) — pp512/tg128,
      thread scaling; results in `docs/design.md` §0
- [x] Peak-RSS measurements per model (✅ in design §0; `/usr/bin/time -v` around a pp+gen run)
- [x] Decode-thread optima per model size: **pin `-t 32 -tb 96`** (measured sweep; collapse past ~64 threads confirmed)
- [x] A/B `amperecomputingai/llama.cpp` vs vanilla `e95426f`: **result — incompatible.** Their
      plain `latest` (v8896/`905ccccf2`) SIGILLs on the N1 during model load (instructions beyond
      N1 — the whole tag line is AmpereOne/N2-class). Vanilla build stays the forge engine; revisit
      only if Ampere ships an Altra-compatible tag.

## M1 — surgery loop (desktop, small models)

- [ ] Ampere Model Library `pytorch` docker images are a dead end for the surgery stack: newest tag is
      torch **1.12.1** — Qwen3.x-era architectures need torch 2.x. Fallbacks: stock aarch64 PyTorch 2.x
      (surgery is streaming/memory-bound; acceptable), or Ampere's current AOP wheels from their OS
      portal if installable without a login wall. Decide at M1.
- [ ] `surgery/` package: drop_layers (uniform + ranked), slice_ffn, prune_heads; per-layer fixup;
      per-layer streaming to bound peak RAM
- [ ] HF → GGUF convert of a surgically-edited checkpoint (Qwen3-4B-scale first)
- [ ] llama-quantize calls wired as pinned-subprocess step
- [ ] One recipe end-to-end: Qwen3-4B → drop+slice → convert → quantize → PPL + tok/s

## M2 — benchmark suite

- [ ] Floor tier: wikitext-2 + fineweb-sample perplexity runner
- [ ] Validate lm-eval GGUF/transformers CPU cost; fix the academic subset (GSM8K, ARC-E, HellaSwag)
- [ ] Coding task: local-model-builder-style generate-and-run unit tests (`docker run --network none`)
- [ ] Instruct checklist + long-context needle tasks; fixed seeds, greedy decoding
- [ ] Metric pack: quality composite, tok/s, TTFT, RSS, size, quality-per-GB

## M3 — funnel + orchestration

- [ ] Config-grid builder (request → recipe set with time/disk bounds)
- [ ] `forge` CLI + worker, SQLite jobs table (gpu-scheduler shape), `runs/` artifacts + GC
- [ ] Two-stage funnel: smoke (stage A) → full suite (top-K), PPL ceiling cut
- [ ] Markdown leaderboard + Pareto report
- [ ] First real run: 27B → ~9B

## M4 — recovery (Altra CPU)

- [ ] llama.cpp LoRA fine-tune stage for a stage-A survivor; before/after scores through the funnel
- [ ] Recovery dataset choice (~2B-token mix) ⬜

## M5 — MoE making

- [ ] franken-MoE: mergekit on aarch64 (fallback: own router-assembly script) ⬜
- [ ] moeify recipe + recovery; llama.cpp MoE LoRA training feasibility ⬜
