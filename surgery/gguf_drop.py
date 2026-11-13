"""Drop decoder blocks straight out of a GGUF: no HF load, save or re-convert.

  python -m surgery.gguf_drop <in.gguf | first split shard> <out.gguf> --drop 1 4 9 12

Why (design §0j.10): a 27B slice through the HF route is a 54 GB load, a 54 GB save and
a full re-convert per drop set. Dropping a block is pure bookkeeping: copy every other
tensor byte for byte, renumber `blk.N`, and fix the per-layer metadata. That also keeps
the slice on the exact source the §0j.4 baseline quants came from.

Metadata that encodes layers (qwen35/qwen3next family):
- `<arch>.block_count` counts every block, MTP (`nextn`) blocks included; they sit
  last, so they are renumbered but never dropped here.
- `<arch>.attention.recurrent_layers` (bool per block) is what llama.cpp reads for
  linear-vs-full attention. If the source only has `full_attention_interval`, the array
  is synthesized from it exactly as src/models/qwen35.cpp does, then the dropped entries
  are removed. Without that, the periodic fallback would mislabel every block after the
  first drop.
Split inputs (`-00001-of-0000N.gguf`) are read shard by shard and written as one file.
"""

import argparse
import re
import sys
from pathlib import Path

from gguf import GGUFReader, GGUFValueType, GGUFWriter, Keys

BLK = re.compile(r"^blk\.(\d+)\.")


def shards(first: Path) -> list:
    m = re.match(r"^(.*)-00001-of-(\d{5})\.gguf$", first.name)
    if not m:
        return [first]
    n = int(m.group(2))
    return [first.with_name(f"{m.group(1)}-{i:05d}-of-{n:05d}.gguf") for i in range(1, n + 1)]


def field_value(f):
    return f.contents()


def plan(readers, drop):
    r0 = readers[0]
    arch = bytes(r0.fields[Keys.General.ARCHITECTURE].parts[-1]).decode()
    n_all = field_value(r0.fields[f"{arch}.block_count"])
    nextn_f = r0.fields.get(f"{arch}.nextn_predict_layers")
    n_nextn = field_value(nextn_f) if nextn_f else 0
    n_main = n_all - n_nextn
    drop = sorted(set(drop))
    if any(not 0 <= i < n_main for i in drop):
        raise SystemExit(f"--drop must index main blocks 0..{n_main - 1} (MTP blocks stay)")
    keep = [i for i in range(n_all) if i not in drop]
    remap = {old: new for new, old in enumerate(keep)}

    rec_key = Keys.Attention.RECURRENT_LAYERS.format(arch=arch)
    if rec_key in r0.fields:
        rec = [bool(v) for v in field_value(r0.fields[rec_key])]
    else:
        iv_f = r0.fields.get(f"{arch}.full_attention_interval")
        iv = field_value(iv_f) if iv_f else 4
        rec = [(i < n_main) and ((i + 1) % iv != 0) for i in range(n_all)]
    if len(rec) != n_all:
        raise SystemExit(f"{rec_key} has {len(rec)} entries, block_count is {n_all}")
    return arch, n_all, remap, rec_key, [rec[i] for i in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--drop", type=int, nargs="+", required=True)
    args = ap.parse_args()

    readers = [GGUFReader(p) for p in shards(args.src)]
    arch, n_all, remap, rec_key, new_rec = plan(readers, args.drop)
    r0 = readers[0]

    w = GGUFWriter(args.out, arch)
    for f in r0.fields.values():
        name = f.name
        if name == Keys.General.ARCHITECTURE or name.startswith("GGUF.") or name.startswith("split."):
            continue
        if name in (f"{arch}.block_count", rec_key):
            continue
        vt = f.types[0]
        sub = f.types[-1] if vt == GGUFValueType.ARRAY else None
        w.add_key_value(name, field_value(f), vt, sub_type=sub)
    w.add_key_value(f"{arch}.block_count", len(remap), GGUFValueType.UINT32)
    w.add_key_value(rec_key, new_rec, GGUFValueType.ARRAY, sub_type=GGUFValueType.BOOL)

    tensors = []
    for r in readers:
        for t in r.tensors:
            m = BLK.match(t.name)
            if m:
                old = int(m.group(1))
                if old not in remap:
                    continue
                name = f"blk.{remap[old]}." + t.name[m.end():]
            else:
                name = t.name
            tensors.append((name, t, r))
    for name, t, _ in tensors:
        w.add_tensor_info(name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    total = sum(t.n_bytes for _, t, _ in tensors)
    done = 0
    for i, (_, t, r) in enumerate(tensors):
        w.write_tensor_data(t.data, tensor_endianess=r.endianess)
        done += t.n_bytes
        if i % 50 == 0:
            print(f"[gguf_drop] {done / 2**30:.1f}/{total / 2**30:.1f} GiB", flush=True)
    w.close()
    print(f"[gguf_drop] {args.out}: {len(remap)}/{n_all} blocks, dropped {sorted(set(args.drop))}, "
          f"{len(tensors)} tensors, {total / 2**30:.2f} GiB", flush=True)


if __name__ == "__main__":
    sys.exit(main())
