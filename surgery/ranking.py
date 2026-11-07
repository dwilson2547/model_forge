"""Importance scoring: which FFN neurons / which layers can be cut first.

Scores are deliberately cheap (norm-based for width, neighbor-cosine for depth, ShortGPT
flavored). M1 record: neighbor-similarity replaces the published BlockInfluence formula;
refine when recoverable-quality data says it matters.
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
def layer_skip_scores(model, layers, calib_text: str, tokenizer, max_tokens: int = 2048) -> list:
    """score(i) = cos(out[i-1], out[i+1]): how similar the residual stream is *around*
    layer i — high score means removing layer i changes little. Interior layers only
    (1..L-2); the rest get -inf.

    Runs the full model forward once with forward hooks on every decoder layer.
    """
    n = len(layers)
    if n < 3:
        raise ValueError("need at least 3 layers to score")

    inputs = tokenizer(calib_text, return_tensors="pt")
    input_ids = inputs["input_ids"][:, :max_tokens]

    captured = {}

    def hook_factory(idx):
        def hook(_module, _args, out):
            if idx not in captured:
                h = _first_output(out).detach().float().reshape(-1, _first_output(out).shape[-1])
                captured[idx] = h.cpu()
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

    scores = [float("-inf")] * n
    for i in range(1, n - 1):
        a = captured[i - 1]
        b = captured[i + 1]
        scores[i] = F.cosine_similarity(a, b, dim=1).mean().item()
    return scores
