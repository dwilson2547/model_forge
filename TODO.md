# model-forge TODO

Design: [`docs/design.md`](docs/design.md). Status: design phase — nothing below is started.
M1 can begin on the Ryzen desktop before hardware bring-up if desired (D4 default: wait for M0).

## M0 — Altra bring-up baseline (fills the ⬜s in the design)

- [ ] OS + storage layout decided; 128-core topology confirmed (cores, RAM, disk)
- [ ] llama.cpp pinned ARM build (NEON/dotprod/i8mm), benchmarked
- [ ] Baseline table: stock Qwen quants (4B / 9B / 27B-class if RAM allows) × Q4_K_M / Q8_0
      → decode tok/s, prompt-processing tok/s, TTFT at 4k/32k, peak RSS at 8k/64k ctx
- [ ] Peak-RSS measurements: surgery ops on a 27B checkpoint, llama-quantize
- [ ] Disk budget: candidate grid sizes, retention policy for intermediates

## M1 — surgery loop (desktop, small models)

- [ ] `surgery/` package: drop_layers (uniform + ranked), slice_ffn, prune_heads; config fixup;
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
