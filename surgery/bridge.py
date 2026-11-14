"""Training-free repair of a layer drop: a least-squares linear map in place of each dropped layer.

  python -m surgery.bridge <base hf> <slice hf> <out hf> --dropped 25 29
                           [--calib <text>] [--windows 64] [--ctx 512] [--ridge 1e-2] [--cpu]

Why (design §0j.15): a dropped layer i added delta_i(h) to the residual stream. Deleting it
sets that to zero; this sets it to the best linear guess instead, at no cost in size or
speed. The map is folded into the layer before the cut, so the output is an ordinary
checkpoint of the slice's shape (convert, quantize and train it as usual).

How: the kept layer p just before the cut ends with out = h_mid + mlp(norm(h_mid)). Fit T so
that h_mid + T @ mlp_out is as close as possible to the base's hidden state after the dropped
layer(s), over calibration text, then replace down_proj.weight with T @ down_proj.weight.
T = I is the unpatched slice, and the ridge term pulls toward it. (The ReplaceMe recipe,
arXiv 2505.02819: checked here on this hybrid's linear-attention blocks, not assumed.)

Cuts are fitted in order. The first cut's inputs are the base's own (the models are identical
up to it); a later cut is fitted on the already-patched slice's activations against the base's
targets, so it also absorbs what the earlier patch left over.

One model on the device at a time: the base pass stores its targets on the CPU (windows x ctx
x hidden floats per cut), then the slice is loaded. The last 20% of windows are held out; the
report's `unexplained` is the share of the missing layer's effect still missing there after the
patch (1.0 = no better than the plain drop). `unexplained_stream` is the same for a map from
the whole residual stream, which can't be folded into the weights: it shows how much a linear
map could ever recover at that cut.
"""
import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surgery import paths  # noqa: E402
from surgery.arch import find_layer_stack  # noqa: E402


class _Stop(Exception):
    pass


def runs(dropped):
    """Consecutive dropped layers as (first, last) runs: [25, 26, 29] -> [(25, 26), (29, 29)]."""
    out = []
    for d in sorted(set(dropped)):
        if out and d == out[-1][1] + 1:
            out[-1] = (out[-1][0], d)
        else:
            out.append((d, d))
    return out


def windows(tok, text, n, ctx, seed):
    """n windows of ctx tokens from offsets spread across the whole text (it is a concatenation)."""
    rng = random.Random(seed)
    span = ctx * 8                                  # chars: enough for ctx tokens of any source
    out = []
    for k in range(n * 2):
        lo = k * (len(text) - span) // (n * 2)
        s = rng.randint(lo, lo + max(1, (len(text) - span) // (n * 2)) - 1)
        ids = tok(text[s:s + span], add_special_tokens=False)["input_ids"][1:ctx + 1]
        if len(ids) == ctx:
            out.append(torch.tensor(ids))
        if len(out) == n:
            break
    if len(out) < n:
        raise SystemExit(f"calibration text too short for {n} windows of {ctx} tokens")
    return torch.stack(out)


def _out(o):
    return o[0] if isinstance(o, tuple) else o


@torch.no_grad()
def capture(model, ids, hooks, stop_after, batch=4):
    """Run ids through the model in batches; hooks = {name: module}, returns {name: [N, ctx, H]}
    on the CPU in fp32. The forward is cut short after the module `stop_after` has run."""
    got = {k: [] for k in hooks}
    handles = [m.register_forward_hook(lambda _m, _a, o, k=k: got[k].append(_out(o).float().cpu()))
               for k, m in hooks.items()]

    def stop(_m, _a, _o):
        raise _Stop
    handles.append(stop_after.register_forward_hook(stop))
    dev = next(model.parameters()).device
    try:
        for i in range(0, len(ids), batch):
            try:
                model(input_ids=ids[i:i + batch].to(dev))
            except _Stop:
                pass
    finally:
        for h in handles:
            h.remove()
    return {k: torch.cat(v) for k, v in got.items()}


def solve(x, r, ridge):
    """T minimising |x T^T - r|^2 + lam |T - I|^2, lam = ridge x mean diag(x^T x). Rows = tokens."""
    x, r = x.double(), r.double()
    xtx = x.T @ x
    lam = ridge * xtx.diagonal().mean()
    eye = torch.eye(x.shape[1], dtype=torch.float64)
    return torch.linalg.solve(xtx + lam * eye, x.T @ r + lam * eye).T


def _load(path, cpu):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.float32 if cpu else torch.bfloat16, device_map={"": "cpu" if cpu else 0})
    m.config.use_cache = False
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("slice")
    ap.add_argument("out")
    ap.add_argument("--dropped", type=int, nargs="+", required=True,
                    help="base layer indices the slice removed")
    ap.add_argument("--calib", default=str(paths.DATA / "mixed.train.txt"))
    ap.add_argument("--windows", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base)
    ids = windows(tok, Path(a.calib).read_text(), a.windows, a.ctx, a.seed)
    n_fit = max(1, int(a.windows * 0.8))
    cuts = runs(a.dropped)
    t0 = time.time()

    base = _load(a.base, a.cpu)
    b_layers = find_layer_stack(base)[2]
    n_base = len(b_layers)
    if cuts[0][0] == 0:
        raise SystemExit("layer 0 has no layer before it to carry the map")
    targets = capture(base, ids, {last: b_layers[last] for _, last in cuts}, b_layers[cuts[-1][1]])
    del base, b_layers
    torch.cuda.empty_cache()
    print(f"[bridge] base targets for cuts {cuts} in {time.time() - t0:.0f}s", flush=True)

    model = _load(a.slice, a.cpu)
    layers = find_layer_stack(model)[2]
    kept = [i for i in range(n_base) if i not in set(a.dropped)]
    if len(kept) != len(layers):
        raise SystemExit(f"--dropped {a.dropped}: base {n_base} minus {len(set(a.dropped))} "
                         f"!= slice {len(layers)} layers")
    report = {"base": a.base, "slice": a.slice, "dropped": sorted(set(a.dropped)),
              "calib": a.calib, "windows": a.windows, "ctx": a.ctx, "ridge": a.ridge,
              "seed": a.seed, "cuts": []}
    for first, last in cuts:
        p = kept.index(first - 1)                   # the kept layer just before this cut
        layer = layers[p]
        got = capture(model, ids, {"mlp": layer.mlp, "out": layer}, layer)
        h = got["out"].shape[-1]
        x, out = got["mlp"].reshape(-1, h), got["out"].reshape(-1, h)
        y = targets[last].reshape(-1, h)
        r = y - (out - x)                           # what the MLP branch should have added
        k = n_fit * a.ctx
        t = solve(x[:k], r[:k], a.ridge)
        miss = (y[k:] - out[k:]).double()           # the plain drop's error on held-out windows
        left = r[k:].double() - x[k:].double() @ t.T
        # ceiling: a map from the whole stream, out + out B^T ~ y (not foldable into weights)
        b = solve(out[:k], y[:k], a.ridge)
        left_stream = y[k:].double() - out[k:].double() @ b.T
        cut = {"dropped": [first, last], "patched_student_layer": p,
               "unexplained": round((left.pow(2).sum() / miss.pow(2).sum()).item(), 4),
               "unexplained_stream": round((left_stream.pow(2).sum() / miss.pow(2).sum()).item(), 4),
               "drop_rel_err": round((miss.pow(2).sum() / y[k:].double().pow(2).sum()).item(), 5),
               "t_minus_i_norm": round((t - torch.eye(h, dtype=torch.float64)).norm().item(), 3)}
        report["cuts"].append(cut)
        print(f"[bridge] cut {first}-{last} -> student layer {p}: {cut}", flush=True)
        w = layer.mlp.down_proj.weight
        w.data = (t @ w.data.double().cpu()).to(w.device, w.dtype)
        if layer.mlp.down_proj.bias is not None:
            bias = layer.mlp.down_proj.bias
            bias.data = (t @ bias.data.double().cpu()).to(bias.device, bias.dtype)
        del got, x, out, y, r

    out_dir = Path(a.out)
    model.to(torch.bfloat16).save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    for extra in ("chat_template.jinja", "generation_config.json"):
        if (Path(a.slice) / extra).exists() and not (out_dir / extra).exists():
            shutil.copy(Path(a.slice) / extra, out_dir / extra)
    report["seconds"] = round(time.time() - t0)
    (out_dir / "bridge-results.json").write_text(json.dumps(report, indent=2))
    print(f"[bridge] saved {out_dir} ({report['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
