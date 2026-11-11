"""Importance scoring: which FFN neurons / which layers can be cut first.

Scores are deliberately cheap (norm-based for width, BlockInfluence-cosine for
depth). M1 record: depth scoring originally used a neighbor-similarity variant
cos(out[i-1], out[i+1]) that attributed layer i+1's redundancy to layer i; fixed
to the per-layer form cos(out[i-1], out[i]) and averaged over multiple
calibration windows after the 2026-11-11 external review (findings 3 and r1#5).
"""

import torch
import torch.nn.functional as F


def ffn_keep_indices(layer, keep_frac: float):
    """Rows/columns of the FFN intermediate dim to keep (importance-ranked). None if no mlp."""
    mlp = getattr(layer, "mlp", None)
    if mlp is None or not hasattr(mlp, "gate_proj"):
        return None
    gate = mlp.gate_proj.weight    # [I, H]
    up = mlp.up_proj.weight        # [I, H]
    down = mlp.down_proj.weight    # [H, I]
    # Output contribution of neuron i, scaled by how strongly it is driven.
    score = down.float().norm(dim=0) * (gate.float().norm(dim=1) + up.float().norm(dim=1))
    k = max(1, int(round(score.numel() * keep_frac)))
    return torch.topk(score, k).indices.sort().values.long()


def _first_output(out):
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def layer_skip_scores(model, layers, calib_text: str, tokenizer,
                      max_tokens: int = 2048, n_windows: int = 10) -> list:
    """BlockInfluence-flavored per-layer score: score(i) = cos(out[i-1], out[i]) —
    layer i's own input (the residual stream it receives, i.e. the previous
    layer's output) against its own output. High score = the layer changes the
    stream little. Layers 1..L-1 are scored; layer 0 gets -inf (no pre-layer
    capture without input hooks).

    Review round 2, finding 3 (2026-11-11): this used to compute cos(out[i-1],
    out[i+1]) — similarity *around* the layer, which attributed layer i+1's
    redundancy to layer i, and let a cheap full-attention neighbour make its
    linear counterpart look droppable. Round 1 finding 5: also average over
    n_windows windows spaced across the calibration text instead of trusting
    the opening of one article.

    Runs the model forward once per window with output hooks on every decoder
    layer; scores are averaged across windows. Memory: one window's captures
    at a time (n_layers x max_tokens x hidden floats on CPU).
    """
    n = len(layers)
    if n < 3:
        raise ValueError("need at least 3 layers to score")

    ids = tokenizer(calib_text, return_tensors="pt")["input_ids"][0]
    if ids.shape[0] < max_tokens:
        raise ValueError(
            f"calibration text is only {ids.shape[0]} tokens; need >= {max_tokens} "
            f"per window x {n_windows} windows "
            "(raise the recipe's calib read budget or the calib file size)")
    last_start = ids.shape[0] - max_tokens
    starts = [round(k * last_start / max(1, n_windows - 1)) for k in range(n_windows)]

    sums = [0.0] * n
    counts = [0] * n
    for s in starts:
        input_ids = ids[s: s + max_tokens].unsqueeze(0)
        captured = {}

        def hook_factory(idx):
            def hook(_module, _args, out):
                if idx not in captured:
                    o = _first_output(out).detach().float()
                    captured[idx] = o.reshape(-1, o.shape[-1]).cpu()
            return hook

        handles = [layers[i].register_forward_hook(hook_factory(i)) for i in range(n)]
        try:
            model(input_ids=input_ids)
        finally:
            for h in handles:
                h.remove()
        if len(captured) != n:
            missing = sorted(set(range(n)) - set(captured))
            raise RuntimeError(f"hooks failed to capture layer outputs for indices {missing}")

        for i in range(1, n):
            a = captured[i - 1]
            b = captured[i]
            sums[i] += F.cosine_similarity(a, b, dim=1).mean().item()
            counts[i] += 1
        del captured

    scores = [float("-inf")] * n
    for i in range(1, n):
        if counts[i]:
            scores[i] = sums[i] / counts[i]
    return scores
