# model-forge TODO

Design: [`docs/design.md`](docs/design.md). Status: v3 shipped (qwen35-4b-drop2-sftv3-triage,
2026-11-11); review round 2 fixes applied + two decisive experiments fired (see design §0j):
plateau rerun on the 5080, 27B 3-bit do-nothing baseline on the Altra. Next: v4 SFT diet (below).

## Cloud lane (Brev H100 — doctrine adopted 2026-11-12, design §0j.8)

- [x] Doctrine decided: hybrid local + rented. Cloud = hot loop (27B production,
      full-weight repair, MoE training), on-demand, ~$20–30/month budget;
      desktop = LM Studio probes + 4B/9B dev; Ampere = archive/staging/serial evals
      only (no 24h CPU slogs on the critical path). Pro 6000 purchase rejected on
      break-even math (~4,000+ rental hours needed).
- [x] `cloud/` kit reviewed and fixed 2026-11-12 (design §0j.9): pinned env + fla,
      llama.cpp @ e95426f, portable floor scripts, staging fixed, `--full`,
      `--val-frac`/`--stop-on-val-rise`, exact `--resume`, `--push-repo`.
- [x] Unrepaired [25,29] floor: 12.82 / 13.95 / 3.61. Better repair, not a smaller
      wound (§0j.9).
- [x] Fresh-data arm `drop2-repair-2529-fresh1e`: 7.785 / 11.556 / 3.186 f16 (Pass-1
      8.031 / 11.894 / 3.485). Fresh data is a live lever, and the val curve hadn't
      flattened (§0j.9).
- [x] User: first Brev instance (massedcompute A100 80GB, not stoppable, so a fresh
      instance each session).
- [x] Run 0a: CUDA f16 floor matches the Altra within 0.003. Q4 reads about 0.04 lower.
- [x] Run 0b: port parity PASSES (f16 8.032 / 11.900 / 3.490; train loss 2.2316 vs 2.2301). Was: §0j.7 arm, seed 1234, no `--val-frac`. Expect loss
      2.625→~2.230 and floors ≈ 8.031 / 11.894 / 3.485 after the 0a offset.
- [ ] Run 1: full-weight repair, lr 1e-5 and 3e-6, same data and split as fresh1e
      (full file, 1 epoch, val-frac 0.01). Fineweb clearly below fresh1e ⇒ the wall was
      LoRA capacity. The same ⇒ the wall is the wound.
- [ ] After parity: move 27B quants/imatrix/floor production to cloud; keep the
      src-bf16/ dir on the box until the IQ3_XS retry question is settled (user call).


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
- [x] Layer scoring v2 + 4B re-score landed (design §0j.1): the fixed scorer wants
      [25,29] for count2 (vs shipped [24,28]), and band-clustering SURVIVES the fix —
      spatial-spread constraint stays on the menu; count4/6 sets spread wider (21, 13)
      on their own. No 27B pick without the fixed scorer. Queued candidate: drop2-[25,29]
      + same repair, to see if the corrected pick buys floor — **LANDED 2026-11-12
      (§0j.7): [25,29] beats both [24,28] arms on every corpus (Σ −0.57/−0.68); fixed
      scorer is now the default for all future pick-sets. Follow-up queued (cheap):
      unrepaired-[25,29] floor row to separate "smaller wound" from "better repair."**
- [ ] (bookkeeping) AML pytorch images are torch 1.12.1 — dead end for Qwen3.x stack decisions;
      stock torch 2.x it is, per M1's forge_env (torch 2.15.0 works fine on the box)

## External review follow-ups (Claude, 2026-11-10 — fixed 2026-11-11 except the merge-shift test)

Full detail in `docs/issues/2026_11_10_external_review_findings.md`. All six verified;
fixes recorded in design §0j.

- [x] drop2-ranked-fused recipe: `fuse_pairs` verified no-op — op removed from all three
      `-fused` recipes with a provenance note; ops now raise on requested-fusion no-ops
- [x] Isolating run LANDED (design §0j): classic vs all at seed 1234, identical diet —
      fineweb 12.104 vs 12.221 (f16): no mixer gain (slight classic edge), wiki/code
      even; quant penalty measured (+0.35 fw at Q4); seed noise measured (~±0.27 fw)
      — single sub-0.3 fineweb deltas are not evidence. Mixer-coverage hypothesis dead.
- [x] NF4→bf16 merge-shift test LANDED (design §0j.2): merged-bf16 is BETTER than
      NF4+adapter on every corpus (shift −0.19/−0.27/−0.15) — the merge is not the wall;
      the 4-bit substrate behaves as a mild regularizer, costing nothing in the shipped
      artifact. Dose arm LANDED (§0j.3): 4 epochs OVERFITS — fineweb 15.64 vs
      unrepaired 13.55; worse than no repair. Dose joins the dead levers (mixer,
      merge, rank, lr, diet); 2-epoch recipe is the peak. Wall verdict: genuine
      surgical damage rank-16 LoRA can't relearn. Escalations: wound change ([25,29]),
      route around via SFT line, or full-weight repair (out of reach locally)
- [x] Diet-build audit: mixed.train.txt is a plain concatenation (code→fineweb→wiki,
      round-2 finding 2 settled it); recover_gpu now shuffles per epoch and samples the
      `--max-chars` cap across the whole file (striped)
- [x] Fix `--accum` (was: 1/N of the epochs + LR curve cut N/early; true accumulation now,
      schedule in optimizer steps)
- [x] Ranking v2: scores = cos(out[i-1], out[i]) averaged over 10 windows (round 2 f3 fix
      included); re-score of the 4B pending (pick-set comparison vs [24,28] etc.)
- [x] `--save-every 0` NameError (final save+merge decoupled); dead `ts` not in the repo
      copy (box drift — repo is source of truth)


## Review round 2 follow-ups (Claude, 2026-11-11 — fix plan items 1–3, 7 done)

Full detail + ordered fix plan in `docs/issues/2026_11_11_review_findings_round2.md`.
Fixes + fired experiments recorded in design §0j.

- [x] LoRA targets miss all `linear_attn.*` projections (22/30 layers) — added (+51%
      trainable params); r16@1e-4 rerun LANDED but is INCONCLUSIVE on the mixer question
      (design §0j corrected 2026-11-11: compared against the wrong lr arm + f16/Q4
      precision mixing; diet confound could mask a mixer gain). Isolating run landed — verdict in design §0j
- [x] Per-epoch shuffle + whole-file striped cap sampling in the trainer. [ ] Interleaving
      diet builder still to commit (diet re-runs depend on it)
- [x] Layer scorer off-by-one fixed (cos(in_i, out_i), 10-window average) + 4B re-score
      LANDED (§0j.1): picks shift to [25,29] / [13,21,25,29] / [1,13,16,21,25,29];
      fixes go into any 27B drop-set decision
- [ ] v4 SFT pre-flight: drop-not-truncate + per-kind skip counts done in the trainer;
      correctness gate, alert-family holdout split, and the ctx decision belong to v4
- [ ] Chat grading: grade `content` only (no reasoning fallback), last fence for code —
      regrade §0h board if raw generations survive, else on next run
- [ ] Loop metric: per-temp loop rate over seeds; §0g wording fix
- [ ] KL-divergence vs base in the floor tier (tracked in M2 above)
- [x] Baseline row: unsliced 27B imatrix IQ3_M/IQ3_XS/Q3_K_M vs the 12.5 GiB
      target — **COMPLETE 2026-11-12 (§0j.4)**: honest BF16 chain landed
      IQ3_M 6.129/8.697/2.565 @11.9 GiB and Q3_K_M 6.247/8.761/2.593 @12.6 GiB;
      do-nothing confirmed in-band (Q3_K_M strictly dominated);
      srcbf16-IQ3_XS blocked on imatrix coverage (cheap fix queued: `--exclude-weights
      blk.64.attn_k.weight`); quant-from-quant penalty measured at ~+0.04; HF card
      annotated with gen-1 vs honest-chain comparison. Cleanup pending user go:
      delete `src-bf16/` (54 GB) — but note an IQ3_XS retry or imatrix rework would
      need it re-downloaded, ~54 GB.
- [ ] Seed replicate of the kept repair arm; rsLoRA/fixed-α rank re-probe (repeat the
      r2fix arm with a second seed when its result lands)
- [x] `prune_heads` fails loudly under `attn_output_gate` (real fix deferred until a
      head-prune grid is actually scheduled)
- [x] Docs: arch corrected to 32 layers (24 linear + 8 full); status headers refreshed;
      duplicate §0d first copy retired with pointer

## M2 — benchmark suite

- [ ] Floor tier: wikitext-2 + fineweb-sample perplexity runner
- [ ] D2 tier (review r2 f9 — the current 12-item board moves in 0.25 steps and cannot
      resolve small gaps): coding subset through the existing Docker sandbox
      (MBPP/HumanEval-style, >=50–100 items, disjoint from any training corpus — r2 f8
      warns code.train and code.test are both llama.cpp), and an IFEval-style instruct
      set at the same scale; fixed seeds, greedy decoding
- [ ] KLD row in the floor (llama-perplexity --kl-divergence-base on base, --kl-divergence
      on candidates): distance-from-base that domain adaptation can't game
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

## M3 — funnel + orchestration — HOLD per review r2 f10 until r2 f5/f6/f9 land: the
12-item bench just ranks noise faster

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
- [x] Plateau rerun LANDED (`~/forge-gpu/drop2-repair-r2fix-lr1e4`, box:
      `~/m0/runs/drop2-repair-r2fix-lr1e4/` incl. Q4_K_M GGUF): diet re-fingers domains
      as §0d predicts; mixer effect INCONCLUSIVE (confounded — see §0j corrected reading
      + the isolating run above)
- [ ] Next repair experiment: diverse repair data (SFT mixture vs fineweb-edu — Gemma's
      candidates) evaluated over *multiple* corpora to separate domain adaptation from true
      repair (§0c caveat #2) ⬜
- [ ] Gotcha bank (design §0j, discovered 2026-11-11): bnb packs `in_proj_a/b` (32-row
      linears — NOT skipped by default); a post-load bf16 cast of packed Params4bit
      corrupts bytes, use `llm_int8_skip_modules`; that list must also carry `lm_head` or
      it loads as an unpacked Linear4bit that asserts on forward
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

- [ ] franken-MoE v1 from the Qwen3.5-4B repaired siblings (x86 desktop, mergekit):
      experts must be same-layer-count/same-shape — use the 30-layer sliced+repaired
      shelf only ([24,28]-repair arms + [25,29]-repair from Pass 1; diet variants give
      the experts their measured domain biases per 0d). Router: mergekit-moe static
      routing; eval on the 3-corpus floor + the chat board
- [ ] moeify recipe + recovery; llama.cpp MoE LoRA training feasibility ⬜
- [ ] (stretch) Second forge family: Gemma — port the forge for dense Gemma stock
      (arch/ops: simpler than qwen3.5 — dense full-attention, no linear/full hybrid;
      classic LoRA target list; Gemma-turn-template SFT variant is the real work —
      the v3 think-open/close fix was Qwen-template-specific). Licensing: Gemma Terms
      of Use, NOT Apache — Gemma license text travels with derivatives + attribution +
      use-policy acknowledgment before any publish (use the publish checklist, new
      license lane). HF repos are gated: Daniel must accept Gemma terms with his
      account before the box token can pull weights
- [ ] (stretch) MoE-native surgery for gemma-family A4B-class models: expert pruning
      (drop least-loaded expert per layer, router renormalizes — the MoE analog of
      layer-drop, far more contained wound) and expert width-slicing; layer-drop through
      MoE blocks is the known-bad path (nemotron intuition). First target: gemma 26B
      A4B — at ~3.3 bpw it may already fit the 5080 (~11-12 GiB, only 4B active, so
      decode stays fast); the imatrix machinery translates as-is. Measured question:
      expert-prune 26B A4B -> A3B-class with how much floor damage?

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
