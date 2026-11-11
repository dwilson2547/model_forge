"""GPU recovery (M4 pivot): QLoRA repair of a surgically sliced checkpoint, desktop-5080 class.

Runs under forge-gpu-env on the WSL desktop (CUDA), *not* on the Ampere — see
docs/issues/2026_11_07_llamacpp_cpu_finetune.md for why recovery left the box.
Output convention matches surgery/recover.py: merged HF dir + GGUF are produced on
whichever machine hosts llama.cpp (the Altra does convert/quantize/eval via the
regular pipeline); this script's job ends with a merged HF dir + adapter checkpoints.

Pausable by design: --save-every N writes an adapter checkpoint; restart with
--resume <ckpt dir> to continue. VRAM: 4-bit base + LoRA ~6-8 GB at batch 4 ctx 512
on a 4B model, leaving the desktop usable while training.

Review round 2 fixes (docs/issues/2026_11_11_review_findings_round2.md):
- LoRA targets now include the linear-attention token mixers (in_proj_qkv/z/a/b,
  out_proj) — 22 of 30 drop2 layers previously got no repair on the modules that
  carry the recurrent state (finding 1).
- Raw-text mode samples --max-chars across the WHOLE file (striped), not the head;
  chunk order is shuffled every epoch (finding 2 / round-1 #3: "codefw" was 78%
  llama.cpp C++ purely because code sits at offset 0 of the concatenation).
- --accum is a true gradient-accumulation count: epochs and cosine T_max are in
  optimizer steps, each consuming accum micro-batches (round-1 #4).
- SFT datasets drop records whose target would be truncated (never teach a
  truncated <|im_end|>) and skip counts are reported per kind (finding 4).
- --save-every 0 disables periodic checkpoints only; final save + merge always run
  (round-1 minor: NameError on `report`).
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Qwen3.5 hybrid: q/k/v/o + MLP projections on every layer, plus the linear-attention
# token mixers on non-full-attention layers. conv1d/norm stay out of LoRA (norm layers
# aren't LoRA-compatible; conv1d is tiny and input-adjacent).
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]

# Modules kept out of the NF4 base at load time. in_proj_a/b are 32-row linears in
# the linear-attention mixers: bnb 4-bit-quantizes them by default and LoRA against
# packed NF4 there is risky, so they stay bf16 (the review r2 caveat). lm_head must
# also be listed — passing this list replaces transformers' default skip list, and
# without it lm_head loads as an unpacked Linear4bit that asserts on the first
# forward (verified 2026-11-11).
FP32_SKIP_MODULES = ["in_proj_a", "in_proj_b", "lm_head"]


def _merge_to_hf(hf_dir, adapter_dir, out_dir):
    """Exact merge, off the 4-bit weights: bf16 CPU base + PeftModel + merge_and_unload."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(hf_dir, dtype=torch.bfloat16)
    base = PeftModel.from_pretrained(base, adapter_dir, adapter_name="repair")
    merged = base.merge_and_unload()
    hf_out = Path(out_dir) / "hf-recovered"
    merged.save_pretrained(hf_out, safe_serialization=True)
    AutoTokenizer.from_pretrained(hf_dir).save_pretrained(hf_out)
    report = {"hf_dir": str(hf_dir), "adapter": str(adapter_dir), "hf_recovered": str(hf_out)}
    (Path(out_dir) / "merge-results.json").write_text(json.dumps(report, indent=2))
    print(f"[recover-gpu] merged -> {hf_out}", flush=True)
    return hf_out


def _cap_sample(raw: str, cap: int) -> str:
    """Sample `cap` chars from across the WHOLE file instead of taking its head.

    The head cap was why "codefw" was mostly llama.cpp: the mixed diet is a plain
    concatenation (code -> fineweb -> wiki), so the first 12M chars were all code
    plus early fineweb (review round 2, finding 2). Striped sampling keeps each
    ctx-window contiguous while spanning every source block.
    """
    if len(raw) <= cap:
        return raw
    L = len(raw)
    n_blocks = -(-L // cap)               # ceil
    seg = max(1, cap // n_blocks)
    parts = []
    for b in range(n_blocks):
        lo = b * L // n_blocks
        hi = (b + 1) * L // n_blocks
        s = random.randint(lo, max(lo, hi - seg))
        parts.append(raw[s: s + seg])
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_dir", help="edited HF checkpoint to repair")
    ap.add_argument("corpus", nargs="?", default=None, help="plain-text training file (not needed with --merge-only)")
    ap.add_argument("--out", default="runs/gpu-recovery")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=1,
                    help="micro-batches per optimizer step (true gradient accumulation)")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="QLoRA lr; 1e-4 measured best for repair-scale doses (2k-floor "
                         "sweep 2026-11-09: 1e-4 < 2e-4 << 4e-4, which nearly unwound the repair)")
    ap.add_argument("--save-every", type=int, default=100,
                    help="optimizer steps between adapter checkpoints (0 = periodic checkpoints off)")
    ap.add_argument("--resume", default="", help="path to adapter checkpoint dir to resume from")
    ap.add_argument("--max-chars", type=int, default=12_000_000)
    ap.add_argument("--calibrate", action="store_true",
                    help="measure throughput only: no checkpoints, no merge, no GGUF")
    ap.add_argument("--merge-only", default="", help="skip training; merge this adapter dir into a bf16 CPU copy of the base and save")
    ap.add_argument("--sft", action="store_true",
                    help="chat-template SFT on a plain-text corpus (v2 pair design)")
    ap.add_argument("--sft-dataset", default="",
                    help="messages-JSONL SFT (v3): records {user, reasoning, answer, kind}. "
                         "Targets render in the native assistant format "
                         "(prefix auto-opens the think block) + reasoning + "
                         "</think\\n\\n> + answer + <|im_end|>, loss on the whole target. "
                         "Records whose target would be truncated at --ctx are DROPPED "
                         "(never train toward a truncated <|im_end|>) and skip counts "
                         "are reported per kind.")
    args = ap.parse_args()

    if args.merge_only:
        _merge_to_hf(args.hf_dir, args.merge_only, args.out)
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from bitsandbytes.optim import AdamW8bit

    out = Path(args.out)
    (out / "ckpt").mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.hf_dir)
    print("[recover-gpu] loading 4-bit base…", flush=True)
    from transformers import BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True,
                             llm_int8_skip_modules=FP32_SKIP_MODULES)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_dir,
        quantization_config=bnb,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    # in_proj_a/b load as bf16 via llm_int8_skip_modules (verified 2026-11-11: without
    # it bnb 4-bit-quantizes the 32-row linears — and casting packed Params4bit bytes
    # to bf16 post-load would corrupt them, so the exclusion happens at load time).
    model.config.use_cache = False
    model.enable_input_require_grads()
    # Activations are the VRAM hog at ctx 512 x batch 8 (the 250k-vocab CE alone peaks
    # several GiB in fp32). Grad checkpointing is nearly free at <1% trainable params.
    model.gradient_checkpointing_enable()

    lcfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
                      target_modules=LORA_TARGETS,
                      bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg, adapter_name="repair")
    if args.resume:
        from peft import PeftModel
        model.load_adapter(args.resume, adapter_name="repair")
        print(f"[recover-gpu] resumed adapter from {args.resume}", flush=True)
    model.print_trainable_parameters()
    for name, m in model.named_modules():
        if hasattr(m, "in_proj_qkv"):
            probe = ", ".join(
                f"{p}={getattr(m, p).weight.dtype}".replace("torch.", "")
                for p in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b")
                if hasattr(m, p))
            print(f"[recover-gpu] linear-attn probe {name}: {probe}", flush=True)
            break

    skipped = {}
    cap_mode = "n/a (sft-dataset)"
    if args.sft_dataset:
        # v3: records {user, reasoning, answer, kind}. Target rendered through the
        # template's own reasoning_content path, so the think-open/close structure is
        # byte-exact native (never hand-assembled): the generation prompt auto-OPENS
        # the think block; the model must learn to CLOSE it — the missing piece from
        # v1/v2 (their content never left the think region; LM Studio showed
        # "Thought ... then died").
        import json as _json
        ds = [_json.loads(l) for l in Path(args.sft_dataset).open()]
        examples = []

        def _skip(kind, total, reason):
            total.setdefault(kind, [0, set()])
            total[kind][0] += 1
            total[kind][1].add(reason)

        for rec in ds:
            kind = rec.get("kind", "?")
            user_msgs = [{"role": "user", "content": rec["user"]}]
            uprefix = tok.apply_chat_template(
                user_msgs, tokenize=False, add_generation_prompt=False)
            full = tok.apply_chat_template(
                user_msgs + [{"role": "assistant",
                              "reasoning_content": rec["reasoning"],
                              "content": rec["answer"]}],
                tokenize=False, add_generation_prompt=False)
            assert full.startswith(uprefix), "template prefix drift"
            target_str = full[len(uprefix):].rstrip("\n")   # keep up to <|im_end|>
            prefix = tok(uprefix, add_special_tokens=False)["input_ids"]
            tgt = tok(target_str, add_special_tokens=False)["input_ids"]
            if len(prefix) > args.ctx - 24:
                _skip(kind, skipped, "prefix>ctx")          # bundle alone too long
                continue
            if len(prefix) + len(tgt) > args.ctx + 24:
                _skip(kind, skipped, "target-would-truncate")  # drop, never truncate
                continue
            if len(prefix) + len(tgt) < 24:                  # degenerate
                _skip(kind, skipped, "too-short")
                continue
            examples.append((torch.tensor(prefix + tgt),
                             torch.tensor([-100] * len(prefix) + tgt)))
        n_seen = len(examples) + sum(v[0] for v in skipped.values())
        print(f"[recover-gpu] SFT v3 dataset: {len(examples)} kept / {n_seen} records"
              + (f"; skipped by kind: "
                 + ", ".join(f"{k}={v[0]} ({', '.join(sorted(v[1]))})"
                             for k, v in sorted(skipped.items())) if skipped else "")
              + f"; longest kept {max(len(e[0]) for e in examples)} tok", flush=True)
        show = tok.decode(examples[0][0])
        print(f"[recover-gpu] v3 example[0] target tail: {repr(show[-160:])}", flush=True)
    else:
        raw = Path(args.corpus).read_text()
        if len(raw) > args.max_chars:
            print(f"[recover-gpu] corpus {len(raw):,} chars > --max-chars "
                  f"{args.max_chars:,}: striped sampling across the whole file "
                  f"(not the head — review r2 f2)", flush=True)
        text = _cap_sample(raw, args.max_chars)
        ids_all = tok(text, return_tensors="pt")["input_ids"][0]
        if args.sft:
            # v2: v1's single fixed generic prompt collapsed instruction-following
            # (chat-mode board 2026-11-10: sft arm emitted corpus-style counting with
            # perfect eos discipline but scored 0 everywhere - it learned "any user
            # turn = continue text", never learning to read the user content).
            # v2 alternates examples: generic prompts (6 rotations) / paired
            # continuation (user turn carries the first K tokens of the content as
            # text, assistant completes it - forces reading the user message).
            im_end = tok("<|im_end|>", add_special_tokens=False)["input_ids"]
            assert len(im_end) == 1, f"im_end not a single token: {im_end}"
            SFT_GENERIC = ["Continue the following text in the same style.",
                           "Write the next part of this text.",
                           "Continue writing the document.",
                           "Provide the continuation of this passage.",
                           "Complete the following text.",
                           "Extend this writing naturally."]
            B = args.ctx - 130          # content block tokens (fits any prefix <= 99)
            K = 32                      # paired-example user-context tokens
            ids_rows, lab_rows = [], []
            for bi in range(ids_all.shape[0] // B):
                block = ids_all[bi * B: (bi + 1) * B]
                if bi % 2:               # paired continuation: read the user turn
                    user_text = tok.decode(block[:K])
                    content = block[K:]
                else:                    # generic prompt, rotated
                    user_text = SFT_GENERIC[bi // 2 % len(SFT_GENERIC)]
                    content = block
                prompt_str = tok.apply_chat_template(
                    [{"role": "user", "content": user_text}],
                    tokenize=False, add_generation_prompt=True)
                prefix = tok(prompt_str, add_special_tokens=False)["input_ids"]
                ex = list(prefix) + list(content) + list(im_end)
                lab = [-100] * len(prefix) + list(content) + list(im_end)
                if len(ex) > args.ctx:            # rare long-prefix edge: truncate
                    ex, lab = ex[:args.ctx], lab[:args.ctx]
                pad = args.ctx - len(ex)          # causal mask ignores the pad tail
                ids_rows.append(ex + im_end * pad)
                lab_rows.append(lab + [-100] * pad)
            ids = torch.tensor(ids_rows)
            labels = torch.tensor(lab_rows)
            examples = [(ids[i], labels[i]) for i in range(ids.shape[0])]
            print(f"[recover-gpu] SFT v2: {ids.shape[0]} examples (half paired K={K}), "
                  f"block {B} tok; example[0]: {repr(tok.decode(examples[0][0])[:220])}",
                  flush=True)
        else:
            seq = (ids_all.shape[0] // args.ctx) * args.ctx
            ids = ids_all[:seq].view(-1, args.ctx)
            examples = [(ids[i], ids[i]) for i in range(ids.shape[0])]
        cap_mode = "striped" if len(raw) > args.max_chars else "full-file"
        print(f"[recover-gpu] {len(examples)} examples x <= {args.ctx} tok", flush=True)

    # ---- schedule: micro-steps (batches) vs optimizer steps (accum bundles) ----
    n_ex = len(examples)
    if n_ex == 0:
        raise RuntimeError("no trainable examples survived pre-flight (see skip counts)")
    micro_per_epoch = max(1, n_ex // args.batch)
    total_micro = int(micro_per_epoch * args.epochs)
    n_epochs_i = max(1, -(-total_micro // micro_per_epoch))
    order = []
    for _ in range(n_epochs_i):                 # per-epoch shuffle (review r2 f2)
        order += torch.randperm(n_ex).tolist()
    opt_total = max(1, -(-total_micro // args.accum))
    print(f"[recover-gpu] {micro_per_epoch} micro-steps/epoch x {args.epochs} epochs = "
          f"{total_micro} micro-steps == {opt_total} optimizer steps "
          f"(batch {args.batch}, accum {args.accum})", flush=True)

    pad_id = tok("<|im_end|>", add_special_tokens=False)["input_ids"]
    pad_id = pad_id[0] if pad_id else 0

    opt = AdamW8bit([p for p in model.parameters() if p.requires_grad],
                    lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=opt_total)

    def collate(sel):
        L = max(len(examples[i][0]) for i in sel)
        xs, ys = [], []
        for i in sel:
            x, y = examples[i]
            p = L - len(x)
            xs.append(torch.cat([x, x.new_full((p,), pad_id)]) if p else x)
            ys.append(torch.cat([y, y.new_full((p,), -100)]) if p else y)
        return torch.stack(xs), torch.stack(ys)

    t0 = time.time()
    tokens_seen = 0
    log_every = max(1, total_micro // 20)
    losses = []
    model.train()
    for micro in range(total_micro):
        sel = order[micro * args.batch: (micro + 1) * args.batch]
        if len(sel) < args.batch:               # tail micro-batch: drop, don't short-batch
            break
        xb, yb = collate(sel)
        xb, yb = xb.to(model.device), yb.to(model.device)
        logits = model(input_ids=xb).logits
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                               yb[:, 1:].reshape(-1), ignore_index=-100)
        (loss / args.accum).backward()
        losses.append(loss.item())
        tokens_seen += xb.numel()
        os_done = (micro + 1) // args.accum
        if (micro + 1) % args.accum == 0:
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            if args.save_every and os_done % args.save_every == 0:
                model.save_pretrained(out / "ckpt" / f"step{os_done}")
                print(f"[recover-gpu] checkpoint saved: optimizer step {os_done}",
                      flush=True)
        if (micro + 1) % log_every == 0 or micro == 0:
            dt = time.time() - t0
            print(f"[recover-gpu] micro {micro+1}/{total_micro} loss {loss.item():.4f} "
                  f"(avg {sum(losses[-log_every:])/len(losses[-log_every:]):.4f}) "
                  f"{tokens_seen/dt:.0f} tok/s {dt:.0f}s "
                  f"vrsm {torch.cuda.memory_allocated()/2**30:.1f}GiB", flush=True)

    if args.calibrate:
        rate = tokens_seen / (time.time() - t0)
        print(f"CALIBRATION: {rate:.0f} tok/s (batch {args.batch}, ctx {args.ctx}, "
              f"QLoRA r{args.lora_r}, accum {args.accum})", flush=True)
        return

    # final save + merge always run (periodic checkpoints were optional)
    model.save_pretrained(out / "ckpt" / "final")
    # in-place merge on the 4-bit bnb model is both lossy and un-castable; the
    # exact path reloads a bf16 CPU base with the saved adapter.
    adapter_dir = out / "ckpt" / "final" / "repair"
    model = None  # free GPU + grads before the RAM-heavy merge
    import gc; gc.collect()
    _merge_to_hf(args.hf_dir, str(adapter_dir), str(out))
    hf_out = out / "hf-recovered"
    wall = time.time() - t0

    report = {
        "hf_dir": str(args.hf_dir), "corpus": str(args.corpus),
        "params": {"epochs": args.epochs, "ctx": args.ctx, "batch": args.batch,
                   "accum": args.accum, "lora_r": args.lora_r, "lr": args.lr},
        "lora_targets": LORA_TARGETS,
        "cap_sampling": cap_mode,
        "chunk_shuffle": True,
        "format": ("sft-chat-v3" if args.sft_dataset else
                   "sft-chat-v2" if args.sft else "raw-completion"),
        "loss_first": losses[0], "loss_last20_avg": sum(losses[-20:]) / min(20, len(losses)),
        "tokens_seen": tokens_seen, "train_tok_per_s": tokens_seen / wall,
        "wall_seconds": wall, "hf_recovered": str(hf_out),
    }
    if skipped:
        report["skipped_by_kind"] = {k: v[0] for k, v in skipped.items()}
    (out / "recover-results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f"[recover-gpu] done -> {out}", flush=True)


if __name__ == "__main__":
    main()
