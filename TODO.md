# model-forge TODO

Design: [`docs/design.md`](docs/design.md). Status: v3 shipped (qwen35-4b-drop2-sftv3-triage,
2026-11-11); review round 2 fixes applied + two decisive experiments fired (see design §0j):
plateau rerun on the 5080, 27B 3-bit do-nothing baseline on the Altra. Next: v4 SFT diet (below).

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

## External review follow-ups (Claude, 2026-11-10 — fixed 2026-11-11 except the merge-shift test)

Full detail in `docs/issues/2026_11_10_external_review_findings.md`. All six verified;
fixes recorded in design §0j.

- [x] drop2-ranked-fused recipe: `fuse_pairs` verified no-op — op removed from all three
      `-fused` recipes with a provenance note; ops now raise on requested-fusion no-ops
- [ ] NF4→bf16 merge-shift test (adapter-on-NF4 PPL vs merged-bf16 PPL) — queued behind the
      plateau rerun (if the rerun moves the floor, the shift explanation is moot)
- [x] Diet-build audit: mixed.train.txt is a plain concatenation (code→fineweb→wiki,
      round-2 finding 2 settled it); recover_gpu now shuffles per epoch and samples the
      `--max-chars` cap across the whole file (striped)
- [x] Fix `--accum` (was: 1/N of the epochs + LR curve cut N/early; true accumulation now,
      schedule in optimizer steps)
- [x] Ranking v2: scores = cos(out[i-1], out[i]) averaged over 10 windows (round 2 f3 fix
      included); re-score of the 4B pending (pick-set comparison vs [24,28] etc.)
- [x] `--save-every 0` NameError (final save+merge decoupled); dead `ts` not in the repo
      copy (box drift — repo is source of truth)


## Review round 2 follow-ups (Claude, 2026-11-11 — items 1–3, 6 partially, 7 done)

Full detail + ordered fix plan in `docs/issues/2026_11_11_review_findings_round2.md`.
Fixes + fired experiments recorded in design §0j.

- [x] LoRA targets miss all `linear_attn.*` projections (22/30 layers) — added (+51%
      trainable params); r16@1e-4 rerun IN FLIGHT on the 5080 (`drop2-repair-r2fix-lr1e4`)
- [x] Per-epoch shuffle + whole-file striped cap sampling in the trainer. [ ] Interleaving
      diet builder still to commit (diet re-runs depend on it)
- [x] Layer scorer off-by-one fixed (cos(in_i, out_i), 10-window average); [ ] re-score
      the 4B + compare pick sets vs [24,28] / [12,24,26,28] before any 27B drop set
- [ ] v4 SFT pre-flight: drop-not-truncate + per-kind skip counts done in the trainer;
      correctness gate, alert-family holdout split, and the ctx decision belong to v4
- [ ] Chat grading: grade `content` only (no reasoning fallback), last fence for code —
      regrade §0h board if raw generations survive, else on next run
- [ ] Loop metric: per-temp loop rate over seeds; §0g wording fix
- [ ] KL-divergence vs base in the floor tier (llama-perplexity --kl-divergence)
- [x→in flight] Baseline row: unsliced 27B imatrix IQ3_M/IQ3_XS/Q3_K_M vs the 12.5 GiB
      target (Altra job `~/m0/baseline-27b/`, imatrix → quants → 3-corpus floor)
- [ ] Seed replicate of the kept repair arm; rsLoRA/fixed-α rank re-probe (repeat the
      r2fix arm with a second seed when its result lands)
- [x] `prune_heads` fails loudly under `attn_output_gate` (real fix deferred until a
      head-prune grid is actually scheduled)
- [x] Docs: arch corrected to 32 layers (24 linear + 8 full); status headers refreshed;
      duplicate §0d first copy retired with pointer

## M2 — benchmark suite

- [ ] Floor tier: wikitext-2 + fineweb-sample perplexity runner
- [ ] Incident-triage benchmark task (proof-of-concept 2026-11-08, LM Studio): candidate models
      analyze historical incident-operator reports (incident 46 = first sample: correct
      schema, correct checks, connected findings, 18k ctx) and are graded on JSON schema
      validity + factual grounding in the source bundle. Sampling protocol is part of the
      score: temperature at which thinking loops (loop-threshold) ranked alongside accuracy —
      sliced/repaired models loop at higher temps than stock (observed: 0.21 loops, 0.16 clean
      on mixfull-repaired). Corpus + grading rubric to lift from infra/incident-operator history
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
      verify per base before publishing; gated behind an explicit cli flag, never automatic).
      **First survivor published 2026-11-09 (manual end-to-end, scripts under
      `~/m0/publish/` on the box): `dwilson2547/Qwen3.5-4B-drop2-mixed-repair` (HF format, card +
      recipe-yaml provenance) and `-GGUF` sibling (Q4_K_M with MTP graft). Checklist for the
      stage code: license check, card generation, secrets/infra-path sanitization, corpus-flag
      guard (local-only variants refuse), idempotent re-runs. Token on the box is now
      write-scoped.**
- [ ] First real run: Qwen3.8-27B → 5080-comfort (~20–22B params: Q4_K_M ≤ ~12.5 GiB + KV headroom in 16 GB VRAM; goal corrected 2026-11-10, see design §1 note)

## M4 — recovery

**Measured verdict 2026-11-07:** recovery cannot run on the Altra. llama.cpp finetune is
blocked upstream (CPU backward kernels missing + qwen3_5 SIGMOID backward missing), and
torch-CPU measures **~2 tok/s** (bf16 pathology found and fixed to fp32; the reference
linear-attention ops are per-op bound — 10M repair tokens = 60+ days). The recovery stage
becomes a **GPU worker** job: desktop 5080 first, cluster 3080 as alternative. The Altra
keeps surgery + build + eval. Issue docs: `2026_11_07_llamacpp_cpu_finetune.md`,
`2026_11_07_torch_bf16_matmul_slow.md`.

- [x] llama.cpp recovery attempt — blocked upstream, documented, pivoted
- [x] torch-CPU recovery probe — works correctly but ~2 tok/s ceiling; ruled out on
      throughput (probe artifacts in ~/m0; recover-results superseded)
- [x] GPU recovery worker: `surgery/recover_gpu.py` — QLoRA trainer + merge + calibrate modes;
      first full cycle done 2026-11-08 (design §0c): drop2 PPL 12.82 -> 8.16, 2270 tok/s at
      3.9 GiB, desktop-usable. Checkpoint/resume wired for pausable runs.
- [ ] Next repair experiment: diverse repair data (SFT mixture vs fineweb-edu — Gemma's
      candidates) evaluated over *multiple* corpora to separate domain adaptation from true
      repair (§0c caveat #2) ⬜
- [ ] ctx-1024 repair arm (same codefw diet, ctx 512→1024): **demoted to secondary** by the
      long-window floor (§0e: wound does not widen at range; repair holds to 32k)
- [x] SFT mode in recover_gpu: DONE (v1+v2, 00a70h) - v3 next: constraint-bearing diet (exact-count/format examples) + clean code files; also fix stray fences in code output
      (raw-text trainer can't use instruction data) ⬜
- [ ] Incumbent-domain SFT corpus (small, LoRA-sized): resolved incident-operator cases as
      (incident bundle → known-good triage JSON) pairs, plus scraper postmortems + runbooks
      from scrape-stack — targets the observed failure mode ("everything but the cloudflare
      block wrong"). Train/eval split MANDATORY: incidents used in the benchmark corpus must
      stay out of the training corpus. Public "AIOps triage" datasets: treat as ⚠ unverified
      candidates only; the proprietary corpus is the real asset
- [ ] (stretch) Homelab-adaptation arm: raw-text diet from the workspace's own markdown —
      runbooks, design docs, domain notes (ai-notes layer included), conventions, wiki.
      No SFT mode needed (existing trainer works). Order of operations:
      1) secrets scan over the corpus before it touches any trainer (gitleaks-class scan;
         workspace grade depends on it)
      2) exclude the benchmark-incident bundles + their postmortems (eval contamination)
      3) publish policy: workspace-doc-trained variants are local-only — the M3 publish
         stage must refuse variants whose training corpus flag isn't explicitly cleared
         by name (infra detail leakage), even though HF auth exists
- [ ] Wire GPU recovery into the pipeline proper: `forge submit` extension with a
      `recovery worker: desktop-5080` stage that ssh-round-trips the candidate HF dir ⬜
- [ ] Recovery dataset choice ⬜ — Daniel is collecting HF dataset suggestions (from Gemma)
      for repairing sliced models. Candidate families (all ⬜ untested): SFT mixtures
      (Tulu-3 SFT mix, smoltalk2, OpenHermes-2.5) vs continued-pretraining (fineweb-edu).
      On GPU the token budget is 1000× cheaper, so both families can be A/B'd directly.

## M5 — MoE making

- [ ] franken-MoE: mergekit on aarch64 (fallback: own router-assembly script) ⬜
- [ ] moeify recipe + recovery; llama.cpp MoE LoRA training feasibility ⬜

## v4 SFT diet (planned 2026-11-11; user paused pending external review suggestions)

Grounded in the post-v3 LM Studio session + DB verification (design 0i addendum):

- [ ] extractor: keep FULL conversation prefixes - [system, user] as submitted by the
      pipeline (v3 flattening dropped the system prompts, losing the "Environment
      facts" grounding block); incident 46 holdout upgrades to system-included for free
- [ ] diet: missing-evidence -> action:request refusal examples (empty evidence slots
      get a check request, never a fabricated log line)
- [ ] diet: constraint-bearing examples (exact counts/formats) for the last instruct
      gap to base (0.2 vs 0.6)
- [ ] regression probes: the four LM Studio session prompts (2x incident-keyword,
      rotate-proxy Q, bare scrape-403) as fixtures - after v4: no fabricated evidence,
      request-mode instead of done-with-hallucinated-refs, operator-vocab actions only
- [ ] open question for review: with system prompts restored, does the model still
      emit protocol JSON from bare user turns (LM Studio side-chat) or should there be
      explicitly separate chat-mode vs triage-mode behaviors?
