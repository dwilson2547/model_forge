# External review findings (Claude repo review, 2026-11-10)

Status: **verified against the code, not yet acted on.** Recorded here so nothing is lost —
priorities in TODO.md. Reviewer read the repo blind; all claims checked line-by-line.

## 1. `drop2-ranked-fused` recipe is a no-op fusion ✅ VERIFIED (with context)

Layer pattern is linear,linear,linear,full → full-attention at 3,7,…,35. The recipe drops
24 and 28; `apply_drop` fuses a dropped layer into its predecessor (23, 27) — and
23/27 are the full-attention layers while 24/28 are linear, so the
`type(layer) is type(nxt)` guard (surgery/ops.py:97) is false both times. **Nothing was
merged: drop2-fused is byte-identical to drop2.**

Context the reviewer didn't have: §0b finding 2 already recorded both halves of this —
fusion is genuinely harmful on linear↔linear pairs (drop6-fused: 27 → 39 PPL), and
silently no-ops on full↔linear. The harm conclusion is safe: it rests on drop6's pairs
([21,22], [25,26] fire per the guard arithmetic — the "+2 fused pairs" in the §0b table
checks out). The actual cost here: drop2-fused (and 3 of 5 pairs in drop4-fused) ran
full build+eval cycles as unknowing no-ops — wasted compute, not a wrong conclusion.
Action (low priority): delete the drop2-fused recipe (or its `fuse_pairs` op); keep
`fuse_pairs` in the schema for dense families only, as §0b already concluded.

## 2. NF4-trained adapter merged into bf16 base ✅ VERIFIED, experiment queued

`recover_gpu.py` trains against NF4 (double-quantized) weights; `_merge_to_hf` merges the
same adapter onto a bf16 base. Standard QLoRA caveat: the adapter learns to compensate
for quantization noise that then vanishes at merge time — a distribution shift nobody
has measured here. **Directly relevant to §0f's "QLoRA-substrate-bound plateau"** — part
of the stubborn +1.8 fineweb gap could live in the merge itself, not the training.
Proposed test: same PPL eval on (a) adapter-on-NF4 base, (b) merged bf16. Reviewer's
estimate: bf16 LoRA (no quantization) fits the 5080's 16 GB at batch 2–4 (logits ~2 GB
fp32 at vocab 248,320 × batch 4 × ctx 512, same again for grads, ~7.8 GiB weights) —
worth a `--calibrate` probe before the 27B work, would also bypass the NF4 question
entirely.

## 3. No chunk shuffling + head-of-file `--max-chars` cap ✅ VERIFIED (training side)

`recover_gpu.py` reads the corpus in file order, slices chunks sequentially, iterates
`s = (step*batch) % n_chunks` — no `randperm` per epoch, and the 12M-char cap keeps only
the head. **Reviewer couldn't see the diet builder (not in repo — `multicorpus.py` is the
eval helper, not the mixer).** If the mixed.train.txt diets were built by concatenating
sources, the cosine-tail only ever sees the last source and the cap amputates whichever
source is last (plausibly code). Must check the actual build order of the diet files on
the desktop before assigning blame — if docs were interleaved at build time the effect
is smaller. Either way: shuffle chunk order per epoch (`torch.randperm(n_chunks)`) and
sample the cap across the whole file. May partially explain §0d diet asymmetries.

## 4. `--accum` is broken (latent) ✅ VERIFIED, none of our runs affected

`steps_per_epoch = n_chunks // (batch*accum)` but each loop pass consumes one
micro-batch, so `--accum N` trains on 1/N of the requested epochs; separately,
`CosineAnnealingLR(T_max=total_steps)` counts micro-steps while `sched.step()` only
fires per optimizer step, so the LR curve stops decaying early. Every run so far used
the default accum=1 → **no existing result is invalid.** Fix before the 27B QLoRA, which
will need accumulation.

## 5. Layer ranking rests on one 2048-token window ✅ VERIFIED

`ranking.py:31` — `layer_skip_scores(max_tokens=2048)`, `input_ids[:, :max_tokens]`: every
drop decision in the project was made on the opening of one wikitext calibration file.
Cheap likely contributor to §0b's band-clustering and the wikitext-lean of drop
selections. Fix: average scores over 10–20 windows drawn from a mixed fineweb+code
calibration set, then revisit BlockInfluence.

## Minor (verified)

- `--save-every 0` → `NameError` at the report step (`report` only defined inside the
  save/merge branch).
- `ts = torch.tensor(list(model.get_input_embeddings().weight.shape))` is dead code.
