"""M4 recovery probe: torch-CPU LoRA repair of a surgically sliced HF checkpoint.

Pivot engine (see docs/issues/2026_11_07_llamacpp_cpu_finetune.md): llama.cpp finetune
cannot train on CPU-only builds, and qwen3_5 hybrids cannot build a backward graph at
all. transformers' reference implementations have full autograd, so recovery runs
here on the 128 N1 cores.

Dtype is fp32, NOT bf16: torch-on-aarch64 has a pathological bf16 matmul fallback
(>1000x slower; see docs/issues/2026_11_07_torch_bf16_matmul_slow.md). fp32 matmul
measures ~1.4 TFLOP/s on this box, and the extra RAM (4B fp32 = 16 GiB + activations)
is well within budget.

This is a *probe*, not the final stage: it measures training throughput (tok/s) and
loss trajectory on a small corpus, and emits a merged HF dir + GGUF for the standard
eval path so repair is comparable with surgery results.
"""

import argparse
import json
import time
from pathlib import Path

import torch


def load_for_training(hf_dir):
    import transformers
    model = transformers.AutoModelForCausalLM.from_pretrained(hf_dir, dtype=torch.float32)
    model.config.use_cache = False
    model.gradient_checkpointing_disable()   # false-by-default; LoRA probe wants speed
    model.train()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_dir", help="edited HF checkpoint to repair (e.g. runs/<cand>/hf-edited)")
    ap.add_argument("corpus", help="plain-text training file")
    ap.add_argument("--out", default=None, help="output run dir (default: <hf_dir>/../recovered)")
    ap.add_argument("--llamacpp-dir", default="/home/daniel/llama.cpp")
    ap.add_argument("--max-chars", type=int, default=300_000)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--threads", type=int, default=96)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    out = Path(args.out or (Path(args.hf_dir).parent / "recovered"))
    out.mkdir(parents=True, exist_ok=True)

    import transformers
    from peft import LoraConfig, get_peft_model

    tok = transformers.AutoTokenizer.from_pretrained(args.hf_dir)
    model = load_for_training(args.hf_dir)

    # LoRA over the projections that exist in this arch; hybrid linear-attention layers'
    # own projections (gdn names) are left alone for the probe.
    target_modules = [n for n in
                      ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")]
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
                      target_modules=target_modules, bias="none",
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    text = Path(args.corpus).read_text()[: args.max_chars]
    enc = tok(text, return_tensors="pt")
    ids = enc["input_ids"][0]
    seq = (ids.shape[0] // args.ctx) * args.ctx
    ids = ids[:seq].view(-1, args.ctx)          # [n_chunks, ctx]
    n_chunks = ids.shape[0]
    print(f"[recover] corpus {len(text)} chars -> {n_chunks} chunks of {args.ctx} tokens", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0)

    def batch_at(i):
        xs = ids[(i * args.batch) % n_chunks: (i * args.batch) % n_chunks + args.batch]
        return xs

    losses = []
    t0 = time.time()
    tokens_seen = 0
    interval = max(1, args.steps // 10)
    model.train()
    for step in range(args.steps):
        xb = batch_at(step)                       # [b, ctx]
        xb = xb.to(model.device)
        logits = model(input_ids=xb).logits
        shift_logits = logits[:, :-1].reshape(-1, logits.shape[-1])
        shift_labels = xb[:, 1:].reshape(-1)
        loss = torch.nn.functional.cross_entropy(shift_logits.float(), shift_labels)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
        tokens_seen += xb.numel()
        if (step + 1) % interval == 0 or step == 0:
            dt = time.time() - t0
            print(f"[recover] step {step+1}/{args.steps} loss {loss.item():.4f} "
                  f"({sum(losses[-interval:])/len(losses[-interval:]):.4f} avg) "
                  f"{tokens_seen/dt:.1f} tok/s train, {dt:.0f}s elapsed", flush=True)

    wall = time.time() - t0
    report = {
        "hf_dir": str(args.hf_dir),
        "corpus": str(args.corpus),
        "steps": args.steps, "ctx": args.ctx, "batch": args.batch,
        "lora_r": args.lora_r,
        "loss_first": losses[0],
        "loss_last_10_avg": sum(losses[-10:]) / len(losses[-10:]),
        "tokens_seen": tokens_seen,
        "train_tok_per_s": tokens_seen / wall,
        "wall_seconds": wall,
        "torch_threads": torch.get_num_threads(),
    }

    merged = model.merge_and_unload()
    merged.eval()
    hf_out = out / "hf-recovered"
    merged.save_pretrained(hf_out, safe_serialization=True)
    tok.save_pretrained(hf_out)
    merged.config.use_cache = True
    report["hf_recovered"] = str(hf_out)

    # Same build+eval path as surgery candidates
    from .build import to_gguf, quantize
    import sys
    f16_path = out / "recovered-f16.gguf"
    report["f16"] = to_gguf(hf_out, f16_path, args.llamacpp_dir, python=sys.executable)
    report["quants"] = {"Q4_K_M": quantize(f16_path, out / "recovered-Q4_K_M.gguf",
                                           "Q4_K_M", args.llamacpp_dir)}

    (out / "recover-results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f"[recover] done -> {out}", flush=True)


if __name__ == "__main__":
    main()
