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

## M1 — surgery loop ✅ done 2026-11-07 (see design §0a)

Loop validated end-to-end on Qwen3.5-4B; measured the first quality cliff: drop8 = PPL 40
(4.3x base), +slice85 = PPL 311 — recovery (M4) and drop policy are the levers, not the loop.

- [x] Drop-policy gradient run — done 2026-11-07, design §0b: depth-fragile hybrid, fusion
      retired for qwen3_5, recovery critical path, 27B confirmed same hybrid arch
- [ ] FFN importance scoring v2 (activation-aware, not norm-only); width cuts stay off the
      menu until it passes a re-test at 0.85–0.95
- [ ] Layer scoring v2: spatial-spread constraint (picks band-cluster in the 12–28 region even
      with min_gap=2) or activation-aware BlockInfluence; re-run the 2/4/6 gradient against
      the per-layer scores now recorded in each run's results.json
- [ ] (bookkeeping) AML pytorch images are torch 1.12.1 — dead end for Qwen3.x stack decisions;
      stock torch 2.x it is, per M1's forge_env (torch 2.15.0 works fine on the box)

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
- [ ] Publish stage for survivors: `hf upload` (box is authenticated as `dwilson2547`) of the
      edited HF dir + GGUFs, with an auto-generated model card carrying recipe provenance
      (recipe YAML, results.json, surgery deltas, license of the base — Qwen3.5 is Apache-2.0,
      verify per base before publishing; gated behind an explicit cli flag, never automatic)
- [ ] First real run: 27B → ~9B

## M4 — recovery (Altra CPU)

- [ ] llama.cpp LoRA fine-tune stage for a stage-A survivor; before/after scores through the funnel
- [ ] Recovery dataset choice ⬜ — Daniel is collecting HF dataset suggestions (from Gemma) for
      repairing sliced models. Candidate families to evaluate (all ⬜ untested): SFT mixtures
      (Tulu-3 SFT mix, smoltalk2, OpenHermes-2.5) for instruct-style LoRA, vs continued-pretraining
      corpora (fineweb-edu) for raw-PPL repair. Cost matters: CPU LoRA is slow, so pick a
      token-efficient set and validate the M4 stage itself on a 1–2B model first — don't sink a
      weekend of box time into the 27B before the stage works on something cheap.

## M5 — MoE making

- [ ] franken-MoE: mergekit on aarch64 (fallback: own router-assembly script) ⬜
- [ ] moeify recipe + recovery; llama.cpp MoE LoRA training feasibility ⬜
