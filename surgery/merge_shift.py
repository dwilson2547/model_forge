"""NF4 -> bf16 merge-shift test (external review r1 f2 / design 0j):

Does the adapter trained against the 4-bit NF4 base land correctly when merged
onto a bf16 base? Compares, through ONE evaluator (torch, GPU):
  (a) NF4 base + adapter (PeftModel, unmerged)     - what training optimized
  (b) merged bf16 HF dir from _merge_to_hf          - what ships
Both sides evaluate identical ctx-512 windows over the test corpora, so any
delta is the merge shift, not an engine difference. Not directly comparable to
llama.cpp floors (tokenization/BOS details differ) - compare within the pair.

  python merge_shift.py --hf-dir <merged-bf16 out> --adapter <ckpt/final/repair> \
      --base <edited base hf dir> --data-dir ~/forge-gpu/data [--chunks 24]
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

CORPORA = ["wiki.test.txt", "fineweb.test.txt", "code.test.txt"]


@torch.no_grad()
def ppl(model, tok, path, ctx=512, chunks=24):
    ids = tok(Path(path).read_text(), return_tensors="pt")["input_ids"][0]
    n = ids.shape[0] // ctx
    if n < chunks:
        chunks = n
    tot, cnt = 0.0, 0
    for c in range(chunks):
        block = ids[c * ctx: (c + 1) * ctx].unsqueeze(0).to(model.device)
        logits = model(input_ids=block).logits
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                             block[:, 1:].reshape(-1), reduction="sum")
        tot += ce.item()
        cnt += ctx - 1
    return float(torch.exp(torch.tensor(tot / cnt)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="edited (unsliced? no - surgically edited) base HF dir")
    ap.add_argument("--merged", required=True, help="merged bf16 HF dir (hf-recovered)")
    ap.add_argument("--adapter", required=True, help="adapter dir (ckpt/final/repair)")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--chunks", type=int, default=24)
    args = ap.parse_args()

    from transformers import BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True,
                             llm_int8_skip_modules=["in_proj_a", "in_proj_b", "lm_head"])
    tok = AutoTokenizer.from_pretrained(args.base)

    print("[merge-shift] loading NF4 base + adapter…", flush=True)
    nf4 = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=bnb,
                                                dtype=torch.bfloat16, device_map={"": 0})
    from peft import PeftModel
    nf4 = PeftModel.from_pretrained(nf4, args.adapter, adapter_name="repair")
    nf4.eval()

    print("[merge-shift] loading merged bf16…", flush=True)
    merged = AutoModelForCausalLM.from_pretrained(args.merged, dtype=torch.bfloat16,
                                                  device_map={"": 0})
    merged.eval()

    out = {"adapter": args.adapter, "merged": args.merged, "ctx": args.ctx,
           "chunks": args.chunks, "corpora": {}}
    for c in CORPORA:
        p = str(Path(args.data_dir) / c)
        a = ppl(nf4, tok, p, args.ctx, args.chunks)
        b = ppl(merged, tok, p, args.ctx, args.chunks)
        out["corpora"][c] = {"nf4_plus_adapter": a, "merged_bf16": b,
                             "shift": round(b - a, 4)}
        print(f"[merge-shift] {c:18s} NF4+adapter {a:.4f} | merged-bf16 {b:.4f} "
              f"| shift {b - a:+.4f}", flush=True)
    print(json.dumps(out, indent=2), flush=True)
    Path(args.merged).parent.joinpath("merge-shift.json").write_text(json.dumps(out, indent=2))
    print("MERGE-SHIFT DONE", flush=True)


if __name__ == "__main__":
    main()
