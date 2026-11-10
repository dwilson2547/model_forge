"""GPU recovery (M4 pivot): QLoRA repair of a surgically sliced checkpoint, desktop-5080 class.

Runs under forge-gpu-env on the WSL desktop (CUDA), *not* on the Ampere — see
docs/issues/2026_11_07_llamacpp_cpu_finetune.md for why recovery left the box.
Output convention matches surgery/recover.py: merged HF dir + GGUF are produced on
whichever machine hosts llama.cpp (the Altra does convert/quantize/eval via the
regular pipeline); this script's job ends with a merged HF dir + adapter checkpoints.

Pausable by design: --save-every N writes an adapter checkpoint; restart with
--resume <ckpt dir> to continue. VRAM: 4-bit base + LoRA ~5-7 GB at batch 8 ctx 512
on a 4B model, leaving the desktop usable while training.
"""

import argparse
import json
import time
from pathlib import Path

import torch


def _merge_to_hf(hf_dir, adapter_dir, out_dir):
    """Exact merge, off the 4-bit weights: bf16 CPU base + PeftModel + merge_and_unload."""
    import json
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(hf_dir, dtype=torch.bfloat16)
    base = PeftModel.from_pretrained(base, adapter_dir, adapter_name="repair")
    merged = base.merge_and_unload()
    hf_out = Path(out_dir) / "hf-recovered"
    merged.save_pretrained(hf_out, safe_serialization=True)
    AutoTokenizer.from_pretrained(hf_dir).save_pretrained(hf_out)
    report = {"hf_dir": str(hf_dir), "adapter": str(adapter_dir), "hf_recovered": str(hf_out)}
    (Path(out_dir) / "recover-results.json").write_text(json.dumps(report, indent=2))
    print(f"[recover-gpu] merged -> {hf_out}", flush=True)
    return hf_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_dir", help="edited HF checkpoint to repair")
    ap.add_argument("corpus", nargs="?", default=None, help="plain-text training file (not needed with --merge-only)")
    ap.add_argument("--out", default="runs/gpu-recovery")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="QLoRA lr; 1e-4 measured best for repair-scale doses (2k-floor "
                         "sweep 2026-11-09: 1e-4 < 2e-4 << 4e-4, which nearly unwound the repair)")
    ap.add_argument("--save-every", type=int, default=100, help="steps between adapter checkpoints")
    ap.add_argument("--resume", default="", help="path to adapter checkpoint dir to resume from")
    ap.add_argument("--max-chars", type=int, default=12_000_000)
    ap.add_argument("--calibrate", action="store_true",
                    help="measure throughput only: no checkpoints, no merge, no GGUF")
    ap.add_argument("--merge-only", default="", help="skip training; merge this adapter dir into a bf16 CPU copy of the base and save")
    ap.add_argument("--sft", action="store_true",
                    help="chat-template SFT: fixed generic user turn + corpus chunk as "
                         "assistant content + <|im_end|>; loss on assistant tokens only. "
                         "Fixes the raw-completion EOS-discipline erosion seen in 0g "
                         "(repaired arms ramble, stop at the limit mid-function).")
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
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_dir,
        quantization_config=bnb,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    # Activations are the VRAM hog at ctx 512 x batch 8 (the 250k-vocab CE alone peaks
    # several GiB in fp32). Grad checkpointing is nearly free at 0.5% trainable params.
    model.gradient_checkpointing_enable()

    lcfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg, adapter_name="repair")
    if args.resume:
        from peft import PeftModel
        model.load_adapter(args.resume, adapter_name="repair")
        print(f"[recover-gpu] resumed adapter from {args.resume}", flush=True)
    model.print_trainable_parameters()

    raw = Path(args.corpus).read_text()
    if len(raw) > args.max_chars:
        print(f"[recover-gpu] WARNING: corpus {len(raw):,} chars exceeds --max-chars "
              f"{args.max_chars:,}; training on the first {args.max_chars:,} only. "
              f"Raise --max-chars if this is unintended (it was, once: 2026-11-08).", flush=True)
    text = raw[: args.max_chars]
    ids_all = tok(text, return_tensors="pt")[ "input_ids" ][0]
    if args.sft:
        # One fixed generic user turn for every example (format anchoring; the assistant
        # content carries the repair signal). Slicing token ids directly avoids any
        # decode/re-encode boundary drift vs. the raw-mode diet.
        SFT_USER_PROMPT = "Continue the following text in the same style."
        prompt_str = tok.apply_chat_template(
            [{"role": "user", "content": SFT_USER_PROMPT}],
            tokenize=False, add_generation_prompt=True)
        prefix = tok(prompt_str, add_special_tokens=False)["input_ids"]
        im_end = tok("<|im_end|>", add_special_tokens=False)["input_ids"]
        assert len(im_end) == 1, f"im_end not a single token: {im_end}"
        P = len(prefix)
        C = args.ctx - P - 1                     # content tokens per example
        n_examples = ids_all.shape[0] // C
        ids = torch.stack([
            torch.cat([torch.tensor(prefix),
                    ids_all[i * C: (i + 1) * C],
                    torch.tensor(im_end)])
            for i in range(n_examples)])
        labels = ids.clone()
        labels[:, :P] = -100                     # prompt tokens are not targets
        print(f"[recover-gpu] SFT: prefix {P} tok + content {C} tok + im_end "
              f"(eos {im_end[0]}); {n_examples} examples; "
              f"loss on {C + 1}/{args.ctx} tokens/example", flush=True)
        print(f"[recover-gpu] SFT example[0] decode: {repr(tok.decode(ids[0])[:300])}",
              flush=True)
    else:
        seq = (ids_all.shape[0] // args.ctx) * args.ctx
        ids = ids_all[:seq].view(-1, args.ctx)
        labels = None
    n_chunks = ids.shape[0]
    steps_per_epoch = n_chunks // (args.batch * args.accum)
    total_steps = int(steps_per_epoch * args.epochs)
    print(f"[recover-gpu] {n_chunks} chunks x {args.ctx} tok; "
          f"{steps_per_epoch} steps/epoch; {total_steps} steps total", flush=True)

    ts = torch.tensor(list(model.get_input_embeddings().weight.shape))
    opt = AdamW8bit([p for p in model.parameters() if p.requires_grad],
                    lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_steps, 1))

    t0 = time.time()
    tokens_seen = 0
    log_every = max(1, total_steps // 20)
    losses = []
    model.train()
    for step in range(total_steps):
        s = (step * args.batch) % n_chunks
        xb = ids[s: s + args.batch].to(model.device)
        logits = model(input_ids=xb).logits
        if labels is not None:  # SFT: completion-only loss over assistant tokens
            yb = labels[s: s + args.batch].to(model.device)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                yb[:, 1:].reshape(-1), ignore_index=-100)
        else:
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                xb[:, 1:].reshape(-1))
        (loss / args.accum).backward()
        losses.append(loss.item())
        tokens_seen += xb.numel()
        if (step + 1) % args.accum == 0:
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
        if (step + 1) % log_every == 0 or step == 0:
            dt = time.time() - t0
            print(f"[recover-gpu] step {step+1}/{total_steps} loss {loss.item():.4f} "
                  f"(avg {sum(losses[-log_every:])/len(losses[-log_every:]):.4f}) "
                  f"{tokens_seen/dt:.0f} tok/s {dt:.0f}s "
                  f"vrsm {torch.cuda.memory_allocated()/2**30:.1f}GiB", flush=True)
        if args.save_every and (step + 1) % args.save_every == 0:
            model.save_pretrained(out / "ckpt" / f"step{step+1}")
            print(f"[recover-gpu] checkpoint saved: step {step+1}", flush=True)

    if args.calibrate:
        rate = tokens_seen / (time.time() - t0)
        print(f"CALIBRATION: {rate:.0f} tok/s (batch {args.batch}, ctx {args.ctx}, QLoRA r{args.lora_r})", flush=True)
        return

    if args.save_every and not args.calibrate:
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
                   "lora_r": args.lora_r, "lr": args.lr},
            "format": ("sft-chat" if args.sft else "raw-completion"),
        "loss_first": losses[0], "loss_last20_avg": sum(losses[-20:]) / min(20, len(losses)),
        "tokens_seen": tokens_seen, "train_tok_per_s": tokens_seen / wall,
        "wall_seconds": wall, "hf_recovered": str(hf_out),
    }
    (out / "recover-results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f"[recover-gpu] done -> {out}", flush=True)


if __name__ == "__main__":
    main()
