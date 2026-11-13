"""Carry a base model's imatrix over to a layer-dropped slice.

  python -m surgery.imatrix_remap base-imatrix.dat out-imatrix.gguf --drop 1 4 9 12

Why (design §0j.10): the 27B BF16 imatrix took 24.4 h on the Altra (§0j.4), so a fresh
one per drop set is not affordable. For an *unrepaired* slice, the kept blocks are the
base's blocks unchanged, and the dropped ones are the most residual-preserving
(cos ≈ 0.99 in/out), so each kept block's input statistics stay close to the base's. This is an
approximation: a repaired slice changes its weights and needs its own imatrix.

Input is llama-imatrix's GGUF format, which it writes by default even under a `.dat`
name (the 27B file starts with `GGUF`): per-weight tensors `<weight>.in_sum2` and
`<weight>.counts` plus a few general.* keys. The drop set must be in the same block
numbering as the slice's `surgery.gguf_drop` call.
"""

import argparse
import re

from gguf import GGUFReader, GGUFValueType, GGUFWriter

BLK = re.compile(r"^blk\.(\d+)\.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--drop", type=int, nargs="+", required=True)
    args = ap.parse_args()

    r = GGUFReader(args.src)
    drop = set(args.drop)
    olds = sorted({int(m.group(1)) for t in r.tensors if (m := BLK.match(t.name))})
    keep = [i for i in range(max(olds) + 1) if i not in drop]
    remap = {old: new for new, old in enumerate(keep)}

    arch_f = r.fields.get("general.architecture")
    w = GGUFWriter(args.out, arch_f.contents() if arch_f else "imatrix")
    for f in r.fields.values():
        if f.name.startswith("GGUF.") or f.name == "general.architecture":
            continue
        vt = f.types[0]
        sub = f.types[-1] if vt == GGUFValueType.ARRAY else None
        w.add_key_value(f.name, f.contents(), vt, sub_type=sub)

    out, dropped = [], 0
    for t in r.tensors:
        name = t.name
        m = BLK.match(name)
        if m:
            old = int(m.group(1))
            if old not in remap:
                dropped += 1
                continue
            name = f"blk.{remap[old]}." + name[m.end():]
        out.append((name, t))
    for name, t in out:
        w.add_tensor_info(name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for _, t in out:
        w.write_tensor_data(t.data, tensor_endianess=r.endianess)
    w.close()
    print(f"[imatrix_remap] {args.out}: {len(out)} tensors kept, {dropped} dropped with blocks "
          f"{sorted(drop)}", flush=True)


if __name__ == "__main__":
    main()
