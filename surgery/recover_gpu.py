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

Cloud-lane additions (design §0j.8, review 2026-11-12):
- --full: full-weight repair (no LoRA). fp32 master weights + bf16 autocast + AdamW8bit
  (~40 GB for a 4B slice before activations — an 80 GB card, not the 5080). Default lr
  drops to 1e-5; output is the bf16 model in hf-recovered/ (no merge step).
- --val-frac: hold out a seeded fraction of the training examples and log held-out loss
  every --eval-every optimizer steps; --stop-on-val-rise K stops after K consecutive
  evals worse than the best (the 4-epoch dose arm's overfit was invisible in train loss).
- Checkpoints carry train_state.pt (optimizer, scheduler, step, shuffle order, data
  signature) and --resume continues the SAME run — it used to restart the schedule at
  step 0 on top of the loaded adapter, i.e. silently train extra epochs.
- --distill <teacher hf dir>: knowledge distillation instead of next-token CE (design
  §0j.12). The loss is KL(teacher || student) per position at T=1, i.e. the student learns
  the base's whole next-token distribution on the corpus, not the corpus text itself, so
  the repair can't buy wikitext/code PPL by specializing (§0j.11). The teacher runs bf16,
  frozen, in the same process; the KL is computed in checkpointed position chunks so the
  two 248k-vocab fp32 softmaxes never coexist for a whole batch. Held-out loss is the same
  KL. With --sft-dataset (teacher-generated chats, distill/datagen.py) the KL is taken on
  the assistant positions only (prompt labels are -100): the student matches the teacher's
  whole distribution on the teacher's own answers, in the deployed chat format.
- --push-repo uploads the run's artifacts to a private HF repo after the results json is
  written (token from HF_TOKEN / the cached login, never from a file in the repo).
"""

import argparse
import fnmatch
import hashlib
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

# the pre-2026-11-11 list, for isolating the mixer-coverage experiment: it repaired only
# q/k/v/o + MLP, i.e. the full-attention mixers and the FFNs, never linear_attn.*
LORA_TARGETS_CLASSIC = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]

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


PUSH_PATTERNS = ["hf-recovered/*", "recover-results.json", "merge-results.json"]
PUSH_PATTERNS_LORA = ["ckpt/final/*"]          # the adapter itself (small)


def _push(out_dir, repo, full, dry_run=False):
    """Upload the run's shippable artifacts to a private HF model repo."""
    out_dir = Path(out_dir)
    patterns = PUSH_PATTERNS + ([] if full else PUSH_PATTERNS_LORA)
    files = sorted(p.relative_to(out_dir).as_posix() for p in out_dir.rglob("*")
                   if p.is_file() and any(fnmatch.fnmatch(p.relative_to(out_dir).as_posix(), pat)
                                          for pat in patterns))
    total = sum((out_dir / f).stat().st_size for f in files)
    print(f"[recover-gpu] push {'(dry run) ' if dry_run else ''}-> {repo}: "
          f"{len(files)} files, {total / 2**30:.2f} GiB", flush=True)
    for f in files:
        print(f"[recover-gpu]   {f}", flush=True)
    if dry_run:
        return files
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo, private=True, exist_ok=True)
    api.upload_folder(folder_path=str(out_dir), repo_id=repo, allow_patterns=patterns,
                      commit_message=f"forge run {out_dir.name}")
    print(f"[recover-gpu] pushed -> https://huggingface.co/{repo}", flush=True)
    return files


def _env_report():
    import importlib.metadata as md
    env = {"torch": torch.__version__,
           "cuda": torch.version.cuda,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    for pkg in ("transformers", "peft", "bitsandbytes", "flash-linear-attention"):
        try:
            env[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            env[pkg] = None
    return env


def _data_signature(examples, n_val):
    """Identity of the exact training set — resume refuses a different draw."""
    h = hashlib.sha1()
    for x, y in examples:
        h.update(x.numpy().tobytes())
        h.update(y.numpy().tobytes())
    h.update(str(n_val).encode())
    return h.hexdigest()


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
    ap.add_argument("--lr", type=float, default=None,
                    help="default 1e-4 for QLoRA (2k-floor sweep 2026-11-09: 1e-4 < 2e-4 << "
                         "4e-4, which nearly unwound the repair); default 1e-5 with --full "
                         "(unmeasured starting point — sweep it)")
    ap.add_argument("--full", action="store_true",
                    help="full-weight repair: no LoRA, fp32 master weights + bf16 autocast "
                         "(needs an 80 GB-class card for a 4B slice)")
    ap.add_argument("--save-every", type=int, default=100,
                    help="optimizer steps between checkpoints (0 = periodic checkpoints off). "
                         "--full keeps only the newest periodic checkpoint (~15 GB + optimizer each)")
    ap.add_argument("--resume", default="",
                    help="checkpoint dir (out/ckpt/stepN) to continue the same run from: "
                         "restores weights/adapter, optimizer, scheduler, step and shuffle order")
    ap.add_argument("--val-frac", type=float, default=0.0,
                    help="hold out this fraction of examples (seeded) for held-out loss. "
                         "Raw mode windows are contiguous ctx chunks, so neighbours of a val "
                         "window are in train: an overfit detector, not a clean eval")
    ap.add_argument("--val-max", type=int, default=256,
                    help="examples scored per held-out eval (fixed subset; bounds eval cost)")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="optimizer steps between held-out evals (0 = ~10 evals per run)")
    ap.add_argument("--stop-on-val-rise", type=int, default=0,
                    help="stop after K consecutive held-out evals worse than the best (0 = off)")
    ap.add_argument("--push-repo", default="",
                    help="private HF repo (user/name) to upload hf-recovered + results to")
    ap.add_argument("--push-dry-run", action="store_true",
                    help="list what --push-repo would upload, upload nothing")
    ap.add_argument("--max-chars", type=int, default=12_000_000)
    ap.add_argument("--calibrate", action="store_true",
                    help="measure throughput only: no checkpoints, no merge, no GGUF")
    ap.add_argument("--merge-only", default="", help="skip training; merge this adapter dir into a bf16 CPU copy of the base and save")
    ap.add_argument("--lora-targets", choices=["all", "classic"], default="all",
                    help="'all' = q/k/v/o + MLP + linear_attn mixers (r2fix, current default); "
                         "'classic' = the pre-2026-11-11 q/k/v/o + MLP list (isolation arms)")
    ap.add_argument("--distill", default="",
                    help="teacher HF dir (the unsliced base): train on KL(teacher||student) "
                         "instead of next-token CE (design §0j.12)")
    ap.add_argument("--kd-chunk", type=int, default=128,
                    help="positions per checkpointed KL chunk (VRAM vs speed)")
    ap.add_argument("--seed", type=int, default=0,
                    help="0 = unseeded; otherwise seeds torch+random so striped-cap sampling "
                         "and epoch shuffles are repeatable (all future arms should pass it)")
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

    if args.lr is None:
        args.lr = 1e-5 if args.full else 1e-4
    if args.seed:
        import random as _rand
        torch.manual_seed(args.seed)
        _rand.seed(args.seed)
    lora_targets = (LORA_TARGETS if args.lora_targets == "all" else LORA_TARGETS_CLASSIC)
    resume = Path(args.resume) if args.resume else None
    if resume and not (resume / "train_state.pt").exists():
        raise SystemExit(f"--resume {resume}: no train_state.pt — checkpoints written before "
                         f"2026-11-12 hold weights only and cannot continue a run's schedule")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from bitsandbytes.optim import AdamW8bit

    env = _env_report()
    print(f"[recover-gpu] env: {json.dumps(env)}", flush=True)
    if env["flash-linear-attention"] is None:
        print("[recover-gpu] WARNING: flash-linear-attention not installed — the 24 "
              "GatedDeltaNet layers run the slow torch fallback", flush=True)

    out = Path(args.out)
    (out / "ckpt").mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.hf_dir)
    if args.full:
        src = resume / "model" if resume else args.hf_dir
        print(f"[recover-gpu] loading fp32 base for full-weight repair from {src}…", flush=True)
        model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32,
                                                     device_map={"": 0})
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[recover-gpu] full-weight: {n_train:,} trainable params", flush=True)
    else:
        from peft import LoraConfig, get_peft_model
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
                          target_modules=lora_targets,
                          bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lcfg, adapter_name="repair")
        if resume:
            from peft.utils import set_peft_model_state_dict
            from safetensors.torch import load_file
            sd = load_file(str(resume / "repair" / "adapter_model.safetensors"))
            res = set_peft_model_state_dict(model, sd, adapter_name="repair")
            bad = [k for k in getattr(res, "unexpected_keys", []) if "lora" in k]
            if bad:
                raise SystemExit(f"--resume: adapter keys did not load: {bad[:3]}…")
            print(f"[recover-gpu] resumed adapter weights from {resume}", flush=True)
        model.print_trainable_parameters()
    for name, m in model.named_modules():
        if hasattr(m, "in_proj_qkv"):
            probe = ", ".join(
                f"{p}={getattr(m, p).weight.dtype}".replace("torch.", "")
                for p in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b")
                if hasattr(m, p))
            print(f"[recover-gpu] linear-attn probe {name}: {probe}", flush=True)
            break

    teacher = None
    if args.distill:
        if args.sft:
            raise SystemExit("--distill works on a raw corpus or an --sft-dataset, not --sft v2")
        print(f"[recover-gpu] loading bf16 teacher {args.distill}…", flush=True)
        teacher = AutoModelForCausalLM.from_pretrained(args.distill, dtype=torch.bfloat16,
                                                       device_map={"": 0})
        teacher.config.use_cache = False
        teacher.eval().requires_grad_(False)
        if teacher.config.get_text_config().vocab_size != model.config.get_text_config().vocab_size:
            raise SystemExit("--distill: teacher and student vocabularies differ")
        print(f"[recover-gpu] teacher loaded; vram {torch.cuda.memory_allocated()/2**30:.1f}GiB",
              flush=True)

    def _kd_part(s_lg, t_lg, m):
        ls = F.log_softmax(s_lg.float(), dim=-1)
        lt = F.log_softmax(t_lg.float(), dim=-1)
        return ((lt.exp() * (lt - ls)).sum(-1) * m).sum()

    def kd_loss(s_logits, xb, yb):
        """mean per-token KL(teacher || student) over the positions CE would score"""
        from torch.utils.checkpoint import checkpoint
        with torch.no_grad():
            t_logits = teacher(input_ids=xb).logits[:, :-1]
        s_logits = s_logits[:, :-1]
        m = (yb[:, 1:] != -100).to(torch.float32)
        tot = s_logits.new_zeros((), dtype=torch.float32)
        for i in range(0, s_logits.shape[1], args.kd_chunk):
            j = i + args.kd_chunk
            tot = tot + checkpoint(_kd_part, s_logits[:, i:j], t_logits[:, i:j], m[:, i:j],
                                   use_reentrant=False)
        return tot / m.sum().clamp(min=1)

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

    # ---- held-out split (seeded, separate generator: the main RNG stream is untouched) ----
    val_ex = []
    if args.val_frac > 0:
        g = torch.Generator().manual_seed(args.seed or 0)
        perm = torch.randperm(len(examples), generator=g).tolist()
        n_val = max(1, int(len(examples) * args.val_frac))
        vset = set(perm[:n_val])
        val_ex = [examples[i] for i in sorted(vset)][:args.val_max]
        examples = [e for i, e in enumerate(examples) if i not in vset]
        print(f"[recover-gpu] held out {n_val} examples ({args.val_frac:.1%}); "
              f"scoring {len(val_ex)} per eval", flush=True)
        n_val_total = n_val
    else:
        n_val_total = 0
    data_sig = _data_signature(examples, n_val_total)

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
    eval_every = args.eval_every or max(1, opt_total // 10)
    print(f"[recover-gpu] {micro_per_epoch} micro-steps/epoch x {args.epochs} epochs = "
          f"{total_micro} micro-steps == {opt_total} optimizer steps "
          f"(batch {args.batch}, accum {args.accum}, lr {args.lr}, "
          f"{'full-weight' if args.full else 'qlora r%d' % args.lora_r})", flush=True)

    pad_id = tok("<|im_end|>", add_special_tokens=False)["input_ids"]
    pad_id = pad_id[0] if pad_id else 0

    opt = AdamW8bit([p for p in model.parameters() if p.requires_grad],
                    lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=opt_total)

    start_micro, tokens_seen, losses, val_curve, wall_before = 0, 0, [], [], 0.0
    if resume:
        st = torch.load(resume / "train_state.pt", weights_only=False)
        if st["data_sig"] != data_sig:
            raise SystemExit("--resume: training data differs from the checkpointed run "
                             "(same corpus, --max-chars, --seed, --val-frac and ctx are required)")
        if st["full"] != args.full:
            raise SystemExit("--resume: --full must match the checkpointed run")
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        order, start_micro = st["order"], st["micro"]
        tokens_seen, losses = st["tokens_seen"], st["losses"]
        val_curve, wall_before = st["val_curve"], st["wall"]
        torch.set_rng_state(st["rng"])
        print(f"[recover-gpu] resumed run state at micro-step {start_micro}/{total_micro}",
              flush=True)

    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.full)

    def collate(pool, sel):
        L = max(len(pool[i][0]) for i in sel)
        xs, ys = [], []
        for i in sel:
            x, y = pool[i]
            p = L - len(x)
            xs.append(torch.cat([x, x.new_full((p,), pad_id)]) if p else x)
            ys.append(torch.cat([y, y.new_full((p,), -100)]) if p else y)
        return torch.stack(xs), torch.stack(ys)

    @torch.no_grad()
    def held_out_loss():
        model.eval()
        tot, n = 0.0, 0
        for s in range(0, len(val_ex), args.batch):
            sel = list(range(s, min(s + args.batch, len(val_ex))))
            xb, yb = collate(val_ex, sel)
            xb, yb = xb.to(model.device), yb.to(model.device)
            with amp:
                logits = model(input_ids=xb).logits
            tgt = yb[:, 1:].reshape(-1)
            k = (tgt != -100).sum().item()
            if teacher is not None:
                tot += kd_loss(logits, xb, yb).item() * k
            else:
                tot += F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(), tgt,
                                       ignore_index=-100, reduction="sum").item()
            n += k
        model.train()
        return tot / max(1, n)

    def save_ckpt(tag, micro_done):
        d = out / "ckpt" / tag
        if args.full:
            model.save_pretrained(d / "model", safe_serialization=True)
        else:
            model.save_pretrained(d)
        torch.save({"micro": micro_done, "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "order": order, "tokens_seen": tokens_seen, "losses": losses,
                    "val_curve": val_curve, "wall": wall_before + time.time() - t0,
                    "rng": torch.get_rng_state(), "data_sig": data_sig, "full": args.full},
                   d / "train_state.pt")
        if args.full and tag.startswith("step"):    # keep only the newest full checkpoint
            import shutil
            for old in (out / "ckpt").glob("step*"):
                if old != d:
                    shutil.rmtree(old, ignore_errors=True)
        print(f"[recover-gpu] checkpoint saved: {tag}", flush=True)

    t0 = time.time()
    log_every = max(1, total_micro // 20)
    best_val, rises, stopped_early = float("inf"), 0, False
    if val_ex and not resume:
        v = held_out_loss()
        val_curve.append({"opt_step": 0, "val_loss": v})
        best_val = v
        print(f"[recover-gpu] held-out loss @0: {v:.4f}", flush=True)
    elif val_curve:
        best_val = min(e["val_loss"] for e in val_curve)
    model.train()
    for micro in range(start_micro, total_micro):
        sel = order[micro * args.batch: (micro + 1) * args.batch]
        if len(sel) < args.batch:               # tail micro-batch: drop, don't short-batch
            break
        xb, yb = collate(examples, sel)
        xb, yb = xb.to(model.device), yb.to(model.device)
        with amp:
            logits = model(input_ids=xb).logits
            if teacher is not None:
                loss = kd_loss(logits, xb, yb)
            else:
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
            if val_ex and os_done % eval_every == 0:
                v = held_out_loss()
                val_curve.append({"opt_step": os_done, "val_loss": v})
                if v < best_val:
                    best_val, rises = v, 0
                else:
                    rises += 1
                print(f"[recover-gpu] held-out loss @{os_done}: {v:.4f} "
                      f"(best {best_val:.4f}, rises {rises})", flush=True)
                if args.stop_on_val_rise and rises >= args.stop_on_val_rise:
                    print(f"[recover-gpu] STOP: held-out loss rose {rises} evals in a row",
                          flush=True)
                    stopped_early = True
            if args.save_every and os_done % args.save_every == 0 and not args.calibrate:
                save_ckpt(f"step{os_done}", micro + 1)
        if (micro + 1) % log_every == 0 or micro == start_micro:
            dt = time.time() - t0
            print(f"[recover-gpu] micro {micro+1}/{total_micro} loss {loss.item():.4f} "
                  f"(avg {sum(losses[-log_every:])/len(losses[-log_every:]):.4f}) "
                  f"{tokens_seen/(wall_before + dt):.0f} tok/s {dt:.0f}s "
                  f"vrsm {torch.cuda.memory_allocated()/2**30:.1f}GiB "
                  f"peak {torch.cuda.max_memory_allocated()/2**30:.1f}GiB", flush=True)
        if stopped_early:
            break

    if args.calibrate:
        rate = tokens_seen / (time.time() - t0)
        print(f"CALIBRATION: {rate:.0f} tok/s (batch {args.batch}, ctx {args.ctx}, "
              f"{'full-weight' if args.full else 'QLoRA r%d' % args.lora_r}, accum {args.accum}, "
              f"peak {torch.cuda.max_memory_allocated()/2**30:.1f}GiB)", flush=True)
        return

    hf_out = out / "hf-recovered"
    if args.full:
        opt = teacher = None
        import gc; gc.collect(); torch.cuda.empty_cache()
        model.to(torch.bfloat16)                   # ship bf16, like the LoRA merge path
        model.save_pretrained(hf_out, safe_serialization=True)
        tok.save_pretrained(hf_out)
        print(f"[recover-gpu] saved full-weight model -> {hf_out}", flush=True)
    else:
        # final save + merge always run (periodic checkpoints were optional)
        model.save_pretrained(out / "ckpt" / "final")
        # in-place merge on the 4-bit bnb model is both lossy and un-castable; the
        # exact path reloads a bf16 CPU base with the saved adapter.
        adapter_dir = out / "ckpt" / "final" / "repair"
        model = teacher = None  # free GPU + grads before the RAM-heavy merge
        import gc; gc.collect()
        _merge_to_hf(args.hf_dir, str(adapter_dir), str(out))
    wall = wall_before + time.time() - t0

    report = {
        "hf_dir": str(args.hf_dir), "corpus": str(args.corpus),
        "mode": "full-weight" if args.full else "qlora",
        "params": {"epochs": args.epochs, "ctx": args.ctx, "batch": args.batch,
                   "accum": args.accum, "lora_r": None if args.full else args.lora_r,
                   "lr": args.lr, "max_chars": args.max_chars},
        "lora_targets": None if args.full else lora_targets,
        "lora_target_mode": None if args.full else args.lora_targets,
        "seed": args.seed,
        "cap_sampling": cap_mode,
        "chunk_shuffle": True,
        "objective": f"kd-kl(teacher={args.distill})" if args.distill else "ce",
        "format": ("sft-chat-v3" if args.sft_dataset else
                   "sft-chat-v2" if args.sft else "raw-completion"),
        "loss_first": losses[0], "loss_last20_avg": sum(losses[-20:]) / min(20, len(losses)),
        "val_frac": args.val_frac, "val_curve": val_curve,
        "stopped_early": stopped_early, "resumed_from": str(resume) if resume else None,
        "tokens_seen": tokens_seen, "train_tok_per_s": tokens_seen / wall,
        "wall_seconds": wall, "hf_recovered": str(hf_out), "env": env,
    }
    if skipped:
        report["skipped_by_kind"] = {k: v[0] for k, v in skipped.items()}
    (out / "recover-results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f"[recover-gpu] done -> {out}", flush=True)
    if args.push_repo:
        _push(out, args.push_repo, args.full, dry_run=args.push_dry_run)


if __name__ == "__main__":
    main()
