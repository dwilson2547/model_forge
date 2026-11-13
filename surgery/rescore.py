"""Re-score layer redundancy with the BlockInfluence scorer (review r2 f3/r1 f5).

Compares the fixed per-layer scores cos(out[i-1], out[i]) averaged over multiple
calibration windows against a previous run's recorded layer_scores, and reports
the ranked pick sets for the requested drop counts. Example:

  python rescore.py --hf-dir ~/m0/registry/Qwen3.5-4B \
      --old ~/m0/runs/qwen35-4b-drop2-ranked/results.json \
      --calib ~/m0/data/wiki.train.txt --counts 2 4 6 \
      --out ~/m0/runs/rescore-4b/ranking-v2.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from surgery.arch import find_layer_stack, layer_types, text_config
from surgery.ranking import layer_skip_scores
from surgery.recipe import DropLayers
from surgery.ops import _drop_indices

ap = argparse.ArgumentParser()
ap.add_argument("--hf-dir", required=True)
ap.add_argument("--calib", required=True)
ap.add_argument("--old", default="", help="previous results.json with layer_scores")
ap.add_argument("--counts", type=int, nargs="+", default=[2, 4, 6])
ap.add_argument("--min-gap", type=int, default=2)
ap.add_argument("--windows", type=int, default=10)
ap.add_argument("--max-tokens", type=int, default=2048)
ap.add_argument("--threads", type=int, default=32)
# fp32 by default: bf16 matmuls on the Altra (Neoverse N1, no bf16 units) hit the slow
# path that M4 measured. A 27B in fp32 needs ~110 GB of RAM.
ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
ap.add_argument("--out", default="ranking-v2.json")
args = ap.parse_args()

torch.set_num_threads(args.threads)
model = AutoModelForCausalLM.from_pretrained(args.hf_dir, dtype=getattr(torch, args.dtype))
tok = AutoTokenizer.from_pretrained(args.hf_dir)
parent, attr, layers = find_layer_stack(model)
types = layer_types(text_config(model), len(layers))
print(f"layers: {len(layers)} (linear {types.count('linear_attention')}, "
      f"full {types.count('full_attention')})", flush=True)

old = json.load(open(args.old))["layer_scores"] if args.old else None
calib = Path(args.calib).read_text()[:400_000]
scores = layer_skip_scores(model, layers, calib, tok,
                           max_tokens=args.max_tokens, n_windows=args.windows)

out = {"n_layers": len(layers), "types": types, "old_scores": old,
       "new_scores": scores, "picks": {}}
for count in args.counts:
    op = DropLayers(count=count, strategy="ranked", min_gap=args.min_gap)
    pick = {"new": _drop_indices(len(layers), types, op, scores=scores)}
    if old:
        pick["old"] = sorted(_drop_indices(len(layers), types, op, scores=old))
    out["picks"][f"count{count}"] = pick
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(args.out, "w"), indent=2)
print(json.dumps(out["picks"], indent=2), flush=True)
top = sorted((i for i in range(len(scores)) if scores[i] != float("-inf")),
             key=lambda i: scores[i], reverse=True)[:8]
print("new top-8 droppable:", [(i, round(scores[i], 4), types[i]) for i in top],
      flush=True)
print("RESCORE DONE", flush=True)
