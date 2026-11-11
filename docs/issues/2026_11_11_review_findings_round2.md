# Repo review, round 2 (Claude, 2026-11-11)

Status: **open. Fixes are planned for the next model build.** This follows
`2026_11_10_external_review_findings.md`, whose six findings still stand. One of them (#3, diet
order) is now settled by the data (finding 2 below).

How each finding was checked:
- **verified (artifact)**: read from files in `~/forge-gpu` on the desktop: safetensors headers,
  `config.json`, `adapter_config.json`, `recover-results.json`, the diet files, `sft-v3-dataset.jsonl`.
- **verified (code)**: read line by line, not run.
- **⚠ unverified**: a hypothesis or estimate. Measure it before relying on it.

## 1. Repair LoRA never touches the linear-attention token mixers. Verified (artifact)

The drop2 checkpoint (`drop2-repair-lr1e4/hf-recovered`) has 30 layers: 22 `linear_attn.*` blocks
(`in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`, `conv1d`, `norm`) and 8
`self_attn.*` blocks. `recover_gpu.py` targets only `q/k/v/o_proj` plus the MLP projections. The
saved `adapter_config.json` confirms it: `['down_proj','q_proj','o_proj','gate_proj','v_proj','k_proj','up_proj']`.
So the token mixers in 22 of 30 layers get no adapter. These are the layers §0b says carry the
irreplaceable recurrent state.

Impact: this is the leading candidate for the +1.8 fineweb plateau, which doesn't move with rank,
learning rate or dose (§0f). The standing conclusion that the plateau is "QLoRA-substrate-bound" is
**premature** until a run includes these modules. The NF4 merge-shift test (prior review #2)
should come after this one.

Fix: add `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b` and `out_proj` to the LoRA targets.
⚠ Check that bitsandbytes quantizes the small `in_proj_a/b` (32-row) linears cleanly, or leave them
out of the 4-bit quantization. Leave `conv1d` and `norm` alone. Report the new trainable-parameter
count. Re-run the r16@1e-4 arm on the same diet, with the shuffling fix from finding 2, and compare
it on the three-corpus floor.

## 2. The diet is a straight concatenation, and that confounds §0d. Verified (artifact)

This settles prior review #3. Byte offsets in `~/forge-gpu/data/mixed.train.txt` (40.8M chars):
code (`mixed/code.part.txt`, 4.7M) at 0, then fineweb (25.2M) at 4,717,822, then wiki (10.9M) at
29,906,957. The trainer reads it in file order and does not shuffle.

- **"codefw" is just the default 12M-char head cap of this file**: all the code plus the first
  ~7.3M chars of fineweb. The lr1e4 run's `tokens_seen` = 6.43M over 2 epochs matches the
  3.2M tokens/epoch quoted for codefw.
- **"mixfull" is the whole file.** The fw62/wiki27/code12 shares in §0d are exactly the
  byte-share split of the three parts. Each epoch runs code, then fineweb, then wiki, so the
  low-learning-rate end of the cosine schedule is pure wikitext. That alone explains §0d finding 4
  ("the full mix buys wikitext, not breadth"; wikitext overshoots base). The ordering confounds
  that finding. It doesn't show that dose was the cause.
- The loss values show it: `loss_first` 1.44 (a C++ batch) against `loss_last20_avg` 2.54 (prose).
  A loss that ends higher than it started is the order effect, not divergence.

Fix: shuffle chunk indices every epoch (`torch.randperm(n_chunks)`), and sample the `--max-chars` cap
across the whole file instead of taking the head. Better still, build diets interleaved at document
level and **commit the builder** (finding 9). §0d's diet conclusions need re-running before they
steer the next model's diet choice.

## 3. The layer scorer is off by one. Verified (code)

`surgery/ranking.py` computes `score(i) = cos(out[i-1], out[i+1])`. That is the similarity across
layers i **and** i+1 together. ShortGPT's per-layer score (Block Influence) uses
`cos(out[i-1], out[i])`, i.e. layer i's own input against its own output.

Consequences: the decision to drop layer 24 was made partly on layer 25's redundancy. Layer 26 was
scored partly on layer 27, a protected full-attention layer, so a cheap full-attention layer
makes its linear neighbour look droppable. ⚠ Unverified: this, plus the single 2048-token window
(prior review #5), may account for the band clustering in §0b.

Fix: score `cos(out[i-1], out[i])` (input vs output of layer i; the layer's input is the hook output of i-1,
or capture inputs with a forward pre-hook). Do this together with the multi-window fix from prior
review #5. Re-score the 4B and compare the picked sets against `[24,28]`, `[12,24,26,28]`, …
before trusting the ranked picks on the 27B.

## 4. The v3 SFT dataset fails at the default context. Verified (artifact), run not started

Measured from `~/forge-gpu/data/sft-v3-dataset.jsonl` (3,448 records). Token counts are chars ÷
~3.5 (⚠ estimate):

| kind | n | user chars (median) | fits ctx 512 | fits ~2k |
|---|---:|---:|---:|---:|
| cont-generic | 1500 | 41 (answer ~2000) | target truncated | yes |
| cont-paired | 1499 | 277 (answer ~1720) | target truncated | yes |
| turn-investigation | 124 | 6,243 (max 21,960) | 32 | 60 |
| turn-remediation | 96 | 1,293 | 64 | 96 |
| turn-triage | 28 | 4,958 | 8 | 28 |
| runbook / operation / primer | 39 / 159 / 3 | 2,799 / 614 / 1,876 | 0 / 147 / 0 | all |

Problems:
- **When the prompt fits but the target doesn't, the trainer truncates the target, which drops
  `<|im_end|>`.** At ctx 512 that hits almost all 3,000 continuation records, which then teach the
  model never to stop. That reverses what v2 fixed (v2 sized blocks to `ctx-130` so the EOS
  always fit).
- **When the prompt alone doesn't fit, the record is skipped and no count is reported.** Only
  104 of 248 triage-turn records survive at 512.
- **The domain signal is thin.** Triage turns are ~7% of the records: 62 unique turns × 4
  repeats. The continuation filler is the head 6M chars of the concatenated file (finding 2),
  so it is ~78% llama.cpp C++.
- **Nothing filters for correctness.** `triage_pairs` takes every `model_turns` row with
  `status='done'`. That means distilling the incumbent model's own answers, wrong ones included,
  which works against the TODO goal of fixing "everything but the cloudflare block wrong".
- The holdout is three incident IDs (10, 28, 46). Recurring alert families leak across a holdout
  split by ID.

Fix (before v3 runs):
1. Run v3 at `--ctx 2048` or higher, measuring VRAM with `--calibrate` first.
2. **Drop** records whose target would be truncated instead of cutting them, and print skip
   counts by `kind`.
3. Take continuation pairs from a shuffled, interleaved diet.
4. Add a correctness gate to `triage_pairs`, using any resolution or outcome signal
   incident-operator records. If none exists, hand-pick a gold subset.
5. Hold out by alert family as well as by incident ID.

## 5. Chat-mode grading leaks the reasoning trace. Verified (code), effect ⚠ unmeasured

In `bench/tasks.py` `run_task`:
- **Coding** grades `_extract_code(reasoning_content + content)`. That takes the *first* code
  fence, which in a thinking model is often a draft inside the reasoning, not the final answer.
- **Needle and instruct** fall back to the concatenated text when `content` is empty. The needle
  grader is `"64701" in text`, so echoing or reasoning about the prompt passes.
- The `--sft-dataset` help text says v1/v2 content "never left the think region", which is exactly
  this fallback case. So §0h finding 2 ("v2 fixed the semantic collapse: needle 0→1.0") is
  **weak evidence** until the run is regraded.

Fix: when the server splits reasoning out, grade `content` only. Score "no answer outside the
think block" as 0 and record a separate `answered` flag. For coding, extract the *last* fence
from `content`. Also match ```` ```py ```` and ```` ```Python ```` fences. Regrade the §0h board
from the stored outputs if they're still available. If not, regrade on the next run.

## 6. Loop-threshold: the doc misreads it, and it has saturated. Verified (code + doc)

`loop_threshold` returns the highest temperature that did *not* loop, or `None` if every
temperature looped. wikirep's raw-mode value of 0.0 therefore means it was clean at temperature 0
and looped at every higher one. §0g's "loops at every temperature" would be `None`. The value
is also non-monotone: an arm that loops only at temperature 0 scores 0.7. It takes one sample per
temperature at 192 tokens, and on the §0h chat board 6 of 7 arms score 0.7.

Fix: correct the §0g wording. Report a loop *rate* per temperature over N seeds (e.g. 5) and
longer generations (≥1024 in chat mode, where thinking runs long). Treat looping at temperature
0 as its own failure flag.

## 7. Latent op bugs and wrong architecture facts in the docs. Verified (artifact)

- **`apply_prune_heads` is broken for this family.** `attn_output_gate: true` makes `q_proj`
  `[8192, 2560]` = 2 × 16 heads × 256 (query + gate). The op slices `keep_h*head_dim` rows,
  which is half of what the module expects. It also keeps the *first* N heads, where design §4.2
  says it ranks them by importance. No recipe uses it yet. Fix it or make it fail loudly before
  it goes into any 27B grid.
- **The docs give the wrong architecture.** §0a and prior review #1 say 36 layers (28 linear +
  8 full) with full attention at 3…35. The drop2 config shows 30 layers with full attention at
  `[3,7,11,15,19,23,26,29]`. So the base is **32 layers: 24 linear + 8 full, full at 3…31**.
  The fusion no-op analysis still holds (23 and 27 are full-attention layers).

## 8. Code eval and code training may share a source. ⚠ Unverified

`code.part.txt` (training) is llama.cpp sources (`~/forge-gpu/data/llamacpp-src/src`, 70 files).
§0d says `code.test.txt` (eval, on the Altra) is also from llama.cpp sources. Check that the two
are disjoint at the file level. Even if they are, "code healed 72%" is adaptation to the same repo
and the same C++ style, while the M2 coding tier tests Python. Use an unrelated code corpus for
the eval, from a different repo and language mix.

## 9. Methodology gaps

- **The bench is too small to support the conclusions.** It has 12 items: 4 coding, 5 instruct,
  3 needle. Group scores move in steps of 0.25, 0.2 and 0.33, and needle is 1.0 for every arm
  except sftrep v1. "Instruct 0.6 vs 0.2" (§0h) is 2 tasks at one seed. Build the D2 tier: an
  MBPP/HumanEval-style subset through the existing Docker sandbox, and an IFEval-style instruct
  set, both large enough (≥50–100 items) to resolve small gaps.
- **There are no training seed replicates.** The §0f "best arm" and unimodal rank/lr curves rest
  on 0.07–0.12 perplexity differences (r8 vs r16, 1e-4 vs 2e-4). Repeat the arm you plan to keep
  with 2–3 seeds to measure the noise floor.
- **The rank probe is confounded.** α = 2r with Adam makes higher rank act like a higher
  effective learning rate, and §0f shows a higher rate hurts. Re-probe with rsLoRA scaling
  (α/√r) or a fixed α before concluding anything about rank.
- **Add KL divergence against the base.** Use `llama-perplexity --kl-divergence-base` on the
  base, then `--kl-divergence` on each candidate. This gives mean KLD and top-token agreement:
  distance from the original model, which domain adaptation can't game the way it gamed
  wikitext perplexity in §0d.

## 10. Strategy

- **The do-nothing baseline is missing.** The target (§1 goal correction) is ≤ ~12.5 GiB at Q4
  for the 5080. Unsliced Qwen3.8-27B with imatrix 3-bit quants (IQ3_M / IQ3_XS / Q3_K_M) may already
  land in that band (⚠ estimate scaled from the 15.32 GiB UD-Q4_K_M: roughly 11–13 GiB). Measure
  size, perplexity, KLD and the task board for it. Every sliced and repaired candidate has to beat
  that row.
- **Repair by distilling from the base.** Use the unsliced base's logits (top-k, precomputed or
  run alongside) as the target instead of cross-entropy on a text diet, the approach used in the
  Minitron / Sheared-LLaMA family of work. Diet fingerprinting (§0d) goes away by design because
  the target distribution *is* the base model.
- **Check that the 4B lessons transfer.** The Altra can run surgery plus the perplexity check on
  the 27B now. Run drop2/drop4 there before committing the next model's recipe to the 4B's lessons.
- **Hold off on M3 (funnel and orchestration)** until findings 5, 6 and 9 are fixed. Automating
  a 12-item bench just ranks noise faster.

## 11. Reproducibility and doc hygiene

- **Code that produced published results lives only on the boxes.** The diet builder, the MTP
  graft, `~/m0/publish/` and the long-window harness are not in the repo, and
  `surgery/multicorpus.py` hardcodes `/home/daniel/model-forge` and `/home/daniel/m0/...`.
  Commit them with the paths as arguments.
- **There are no tests.** A small set of tests on a tiny randomly initialised model with the same
  config would have caught the fusion no-op, the scorer offset (3) and the head-pruning shapes (7).
  Separately, ops should fail loudly when they were requested but did nothing (`fused_pairs == []`).
- **The docs have gone stale.** README, TODO and the design header still say "design phase /
  nothing started". The README lists a `forge/` directory that doesn't exist. design.md has §0d
  twice (the first copy is superseded). §1, §4.2 and §5 still describe llama.cpp CPU LoRA recovery,
  which D1 retired.

## Fix plan for the next model

The order is chosen so that each experiment's result can be trusted:

1. **Trainer:** linear-attention LoRA targets (1). Per-epoch shuffle and whole-file cap sampling
   (2). Fix `--accum` (prior #4). Fix `--save-every 0` (prior minor).
2. **Diet:** commit an interleaving diet builder (2, 11). Re-run the r16@1e-4 arm with 1 + 2 to
   establish the new floor.
3. **Ranking:** fix the off-by-one (3) and average over multiple windows (prior #5). Re-score
   before choosing the next model's drop set.
4. **Eval:** add KL divergence to the floor (9). Grade chat-mode answers on `content` only (5).
   Report loop rate (6).
5. **v3 SFT:** ctx ≥2048, drop records whose target would be truncated, a correctness gate, and a
   holdout split by alert family (4).
6. **Baselines:** the 27B imatrix 3-bit row (10), and a seed replicate of the kept arm (9).
7. Hygiene in passing: fix the architecture facts (7), retire the duplicate §0d, make
   `prune_heads` fail loudly.
