import json, pathlib
from safetensors.torch import load_file, save_file

src_dir = pathlib.Path("/home/daniel/m0/registry/Qwen3.5-4B")
idx = json.load(open(src_dir/"model.safetensors.index.json"))
mtp_keys = [k for k in idx["weight_map"] if k.startswith("mtp.")]
shards = {}
for k in mtp_keys:
    shards.setdefault(idx["weight_map"][k], []).append(k)
mtp = {}
for shard, keys in shards.items():
    d = load_file(src_dir/shard)
    for k in keys: mtp[k] = d[k].contiguous()
print("transplanting", len(mtp), "mtp tensors:",
      sum(t.numel()*t.element_size() for t in mtp.values())/1e6, "MB", flush=True)

CANDS = [
    "drop2-repair-sft3-hf",
]
for cand in CANDS:
    tgt = pathlib.Path("/home/daniel/m0/runs") / cand / "model.safetensors"
    if not tgt.exists():
        print("SKIP", cand, "(no model.safetensors)", flush=True)
        continue
    d = load_file(tgt)
    before = len(d)
    d.update(mtp)
    save_file(d, str(tgt), metadata={"format": "pt"})
    print(cand, before, "->", len(d), flush=True)
