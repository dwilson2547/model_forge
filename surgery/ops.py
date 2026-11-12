"""Surgery ops: apply a recipe's structural operations to a loaded HF model.

All ops mutate in place on CPU tensors; nothing here launches training. Configs are fixed
up at the end so the edited checkpoint resumes "a normal model" to any downstream consumer
(save_pretrained, convert_hf_to_gguf, llama-quantize).
"""

import torch
import torch.nn as nn

from .arch import find_layer_stack, layer_types, swap_layer_stack, text_config
from .ranking import ffn_keep_indices, layer_skip_scores


def _shrink_linear(lin, keep_rows=None, keep_cols=None):
    """Return a fresh Linear with sliced weight/bias (no in-place Parameter hacks)."""
    w = lin.weight.data
    b = lin.bias.data if lin.bias is not None else None
    if keep_rows is not None:
        w = w[keep_rows]
        if b is not None:
            b = b[keep_rows]
    if keep_cols is not None:
        w = w[:, keep_cols]
    new = nn.Linear(w.shape[1], w.shape[0], bias=lin.bias is not None,
                    dtype=w.dtype, device=w.device)
    new.weight.data = w
    if b is not None:
        new.bias.data = b
    return new


def _drop_indices(n_layers, types, op, scores=None):
    """Which layer indices may be dropped, subject to pattern/edge protections and min_gap."""
    protected = set(op.protected_types)
    interior = set(range(1, n_layers - 1))          # never drop the first or last layer
    droppable = [i for i in sorted(interior) if types[i] not in protected]
    if op.count > len(droppable):
        raise ValueError(f"cannot drop {op.count} layers: only {len(droppable)} are droppable")

    if op.strategy == "explicit":
        if len(op.indices) != op.count:
            raise ValueError(
                f"explicit drop: op.count={op.count} does not match "
                f"indices={list(op.indices)} — make count and the pick list agree")
        picked = []
        for i in sorted(op.indices):
            if i not in droppable:
                raise ValueError(
                    f"explicit drop {i} is not droppable (edge-protected or "
                    f"type-protected: {types[i] if 0 <= i < len(types) else 'out of range'})")
            if any(abs(i - j) < op.min_gap for j in picked):
                raise ValueError(f"explicit drop {i} violates min_gap={op.min_gap}")
            picked.append(i)
        return picked

    def spaced_ok(picked, cand):
        return all(abs(cand - j) >= op.min_gap for j in picked)

    if op.strategy == "uniform":
        step = len(droppable) / op.count
        picked = []
        for k in range(op.count):
            i = droppable[min(int(k * step), len(droppable) - 1)]
            if spaced_ok(picked, i):
                picked.append(i)
        if len(picked) != op.count:
            raise ValueError(
                f"uniform drop of {op.count} layers with min_gap={op.min_gap} did not fit"
            )
        return sorted(picked)
    if op.strategy == "ranked":
        if scores is None:
            raise ValueError("ranked drop requires calibration scores")
        picked = []
        for i in sorted(droppable, key=lambda x: scores[x], reverse=True):
            if len(picked) == op.count:
                break
            if spaced_ok(picked, i):
                picked.append(i)
        if len(picked) != op.count:
            raise ValueError(
                f"ranked drop of {op.count} layers with min_gap={op.min_gap} found only "
                f"{len(picked)} spaced candidates"
            )
        return sorted(picked)
    raise ValueError(f"unknown drop strategy {op.strategy!r}")


def apply_drop(model, op, tokenizer, calib_text="", fuse=False):
    """Depth surgery. LaCo fusion: a dropped layer's weights are averaged into its
    retained predecessor before the stack is rebuilt (keeps residual magnitudes sane).
    """
    parent, attr, layers = find_layer_stack(model)
    cfg = text_config(model)
    types = layer_types(cfg, len(layers))

    scores = None
    if op.strategy == "ranked":
        if not calib_text:
            raise ValueError("ranked drop needs calib_file")
        scores = layer_skip_scores(model, layers, calib_text, tokenizer)

    drop = set(_drop_indices(len(layers), types, op, scores))
    keep = [i for i in range(len(layers)) if i not in drop]

    new_layers = []
    fused_with = []
    for i in keep:
        layer = layers[i]
        if fuse and (i + 1) in drop:
            nxt = layers[i + 1]
            same = type(layer) is type(nxt) and all(
                pn == qn and p.shape == q.shape
                for (pn, p), (qn, q) in zip(layer.named_parameters(), nxt.named_parameters())
            )
            if same:
                with torch.no_grad():
                    for (pn, p), (qn, q) in zip(layer.named_parameters(), nxt.named_parameters()):
                        p.copy_((p + q) / 2)
                fused_with.append((i, i + 1))
        new_layers.append(layer)

    new_types = [types[i] for i in keep]
    if hasattr(cfg, "layer_types"):
        cfg.layer_types = new_types
    cfg.num_hidden_layers = len(new_layers)
    swap_layer_stack(parent, attr, nn.ModuleList(new_layers))

    if fuse and not fused_with:
        # Round 1 finding 1 (2026-11-10): the type guard made fusion a silent no-op
        # for qwen3.5's hybrid pattern (dropped layers' predecessors were
        # full-attention). A requested op that did nothing must fail loudly, not
        # ship a recipe whose name promises something the weights don't contain.
        raise RuntimeError(
            "fuse_pairs was requested but nothing fused: every dropped layer's "
            "predecessor had a different type or shape. This recipe's fusion is a "
            "no-op - remove the fuse_pairs op (the -fused recipes recorded this on "
            "2026-11-10: dropped 24/28 are linear, predecessors 23/27 are "
            "full-attention)")

    return {
        "dropped": sorted(drop),
        "kept_layers": len(new_layers),
        "fused_pairs": fused_with,
        "type_counts": {t: new_types.count(t) for t in set(new_types)},
        "scores_used": scores is not None,
        "layer_scores": scores,
    }


def apply_slice_ffn(model, op):
    """Width surgery on the FFN: keep the top-`keep_frac` neurons by importance, per layer.
    All layers must end at the same intermediate size (the config expresses exactly one).
    """
    parent, attr, layers = find_layer_stack(model)
    cfg = text_config(model)
    sizes = set()
    touched = 0
    for layer in layers:
        keep = ffn_keep_indices(layer, op.keep_frac)
        if keep is None:
            sizes.add(getattr(cfg, "intermediate_size", None))
            continue
        mlp = layer.mlp
        mlp.gate_proj = _shrink_linear(mlp.gate_proj, keep_rows=keep)
        mlp.up_proj = _shrink_linear(mlp.up_proj, keep_rows=keep)
        mlp.down_proj = _shrink_linear(mlp.down_proj, keep_cols=keep)
        sizes.add(mlp.gate_proj.weight.shape[0])
        touched += 1
    if len(sizes) != 1:
        raise RuntimeError(
            f"FFN slice produced heterogeneous intermediate sizes {sizes}; "
            "config can only express one intermediate_size"
        )
    cfg.intermediate_size = sizes.pop()
    return {"kept_frac": op.keep_frac, "layers_touched": touched,
            "new_intermediate_size": cfg.intermediate_size}


def apply_prune_heads(model, op):
    """Query-head pruning on full-attention layers, aligned to KV groups (GQA-safe)."""
    parent, attr, layers = find_layer_stack(model)
    cfg = text_config(model)
    types = layer_types(cfg, len(layers))

    n_heads = getattr(cfg, "num_attention_heads", None)
    n_kv = getattr(cfg, "num_key_value_heads", n_heads)
    head_dim = getattr(cfg, "head_dim", None)
    if not n_heads or not head_dim:
        return {"skipped": "no attention head geometry in config"}
    # Round 2 finding 7 (2026-11-11): with attn_output_gate=true, q_proj carries
    # 2 x n_heads x head_dim rows (query + gate). Slicing keep_h*head_dim rows
    # is exactly half of what the module expects. Not implemented — fail loudly
    # rather than produce a silently-broken checkpoint (no recipe uses it yet).
    if getattr(cfg, "attn_output_gate", False):
        raise NotImplementedError(
            "prune_heads does not support attn_output_gate models yet: q_proj holds "
            "query AND gate rows (2 x n_heads x head_dim), so a head-aligned slice "
            "must cut both halves in step. Write it before any 27B head-prune grid.")
    if n_heads and n_kv != n_heads and n_heads % n_kv != 0:
        return {"skipped": f"GQA geometry H={n_heads} KV={n_kv} not divisible"}
    group = n_heads // n_kv
    if op.keep_heads % group != 0:
        return {"skipped": f"keep_heads {op.keep_heads} not kv-aligned (group={group})"}

    keep_h = op.keep_heads
    keep_kv = keep_h // group
    keep_rows = torch.arange(keep_h * head_dim)
    kv_rows = torch.arange(keep_kv * head_dim)

    touched = 0
    for layer, ltype in zip(layers, types):
        sa = getattr(layer, "self_attn", None)
        if sa is None or not hasattr(sa, "q_proj") or (ltype != "full_attention"):
            continue
        sa.q_proj = _shrink_linear(sa.q_proj, keep_rows=keep_rows)
        sa.k_proj = _shrink_linear(sa.k_proj, keep_rows=kv_rows)
        sa.v_proj = _shrink_linear(sa.v_proj, keep_rows=kv_rows)
        sa.o_proj = _shrink_linear(sa.o_proj, keep_cols=keep_rows)
        touched += 1
    if touched == 0:
        return {"skipped": "no qualifying full-attention layers"}
    cfg.num_attention_heads = keep_h
    cfg.num_key_value_heads = keep_kv
    return {"keep_heads": keep_h, "keep_kv_heads": keep_kv, "head_dim": head_dim,
            "layers_touched": touched}


def apply_recipe(model, recipe, tokenizer):
    """Apply every op in recipe order. Returns a report dict (core of results.json)."""
    from .recipe import DropLayers, FusePairs, PruneHeads, SliceFFN

    report = {"recipe": recipe.to_dict(), "ops": []}
    calib_text = ""
    if recipe.calib_file:
        calib_text = open(recipe.calib_file).read(400_000)

    fuse = next((o for o in recipe.ops if isinstance(o, FusePairs) and o.enabled), None)
    for op in recipe.ops:
        if isinstance(op, DropLayers):
            info = apply_drop(model, op, tokenizer, calib_text, fuse=bool(fuse))
            report["ops"].append({"op": "drop_layers",
                                  **{k: v for k, v in info.items() if k != "layer_scores"}})
            report["layer_scores"] = info.get("layer_scores")
        elif isinstance(op, FusePairs):
            continue
        elif isinstance(op, SliceFFN):
            report["ops"].append({"op": "slice_ffn", **apply_slice_ffn(model, op)})
        elif isinstance(op, PruneHeads):
            report["ops"].append({"op": "prune_heads", **apply_prune_heads(model, op)})
        else:
            continue  # Quantize is handled by the build stage
    return report
