"""Re-score Qwen3.5-4B layer redundancy with the fixed BlockInfluence scorer
(review r2 f3 + r1 f5) and compare pick sets against the shipped recipes'
choices. Run on the box: forge_env, base HF dir from the registry.

Writes ~/m0/runs/rescore-4b/ranking-v2.json with per-layer scores old + new and
the ranked pick sets for count 2/4/6 (min_gap 2, full_attention protected).
"""
import json
import sys

sys.path.insert(0, "/home/daniel/model-forge")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from surgery.arch import find_layer_stack, layer_types, text_config
from surgery.ranking import layer_skip_scores
from surgery.recipe import DropLayers
from surgery.ops import _drop_indices

BASE = "/home/daniel/m0/registry/Qwen3.5-4B"
OLD = "/home/daniel/m0/runs/qwen35-4b-drop2-ranked/results.json"

torch.set_num_threads(32)
model = AutoModelForCausalLM.from_pretrained(BASE)
tok = AutoTokenizer.from_pretrained(BASE)
parent, attr, layers = find_layer_stack(model)
cfg = text_config(model)
types = layer_types(cfg, len(layers))
print(f"layers: {len(layers)} (linear {types.count('linear_attention')}, "
      f"full {types.count('full_attention')})", flush=True)

old = json.load(open(OLD))["layer_scores"]          # cos(out[i-1], out[i+1]), 1 window

calib = open("/home/daniel/m0/data/wiki.train.txt").read(400_000)
scores = layer_skip_scores(model, layers, calib, tok,
                           max_tokens=2048, n_windows=10)

out = {"n_layers": len(layers), "types": types, "old_scores": old,
       "new_scores": scores, "picks": {}}
for count in (2, 4, 6):
    op = DropLayers(count=count, strategy="ranked", min_gap=2)
    out["picks"][f"count{count}"] = {
        "old": sorted(_drop_indices(len(layers), types, op, scores=old or None)),
        "new": _drop_indices(len(layers), types, op, scores=scores),
    }
json.dump(out, open("/home/daniel/m0/runs/rescore-4b/ranking-v2.json", "w"),
          indent=2)
print(json.dumps({k: out["picks"][k] for k in out["picks"]}, indent=2), flush=True)
top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:8]
print("new top-8 droppable:", [(i, round(scores[i], 4), types[i]) for i in top],
      flush=True)
print("RESCORE-4B DONE", flush=True)
