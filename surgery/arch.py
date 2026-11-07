"""Architecture introspection: locate the decoder stack and its config, generically.

Qwen3.5 checkpoints arrive wrapped (`Qwen3_5ForConditionalGeneration` with a vision tower and
a `text_config`); other families may be plain. Everything here is discovery, not hardcoding:
the surgery ops never assume a fixed attribute path.
"""

import torch.nn as nn


def text_config(model) -> object:
    """The config object that describes the decoder stack (unwraps multimodal text_config)."""
    cfg = model.config
    return getattr(cfg, "text_config", cfg)


def find_layer_stack(model):
    """Return (parent, attr_name, module_list) for the decoder-block ModuleList.

    The ModuleList must have as many elements as the text config's num_hidden_layers and
    elements must look like decoder blocks (have mlp or self_attn submodules).
    """
    n_layers = getattr(text_config(model), "num_hidden_layers", None)
    best = None

    def walk(mod, path, depth):
        nonlocal best
        if best is not None or depth > 4:
            return
        for name, child in mod.named_children():
            if isinstance(child, nn.ModuleList):
                if len(child) == n_layers and all(
                    hasattr(l, "mlp") or hasattr(l, "self_attn") for l in child
                ):
                    best = (mod, name, child)
                    return
            walk(child, path + (name,), depth + 1)

    walk(model, (), 0)
    if best is None:
        raise RuntimeError(
            f"could not locate a decoder-block ModuleList of length {n_layers} "
            f"under {type(model).__name__}"
        )
    return best


def layer_types(cfg, n_layers) -> list:
    """Attention pattern of the stack, e.g. hybrid linear/full Qwen3.5; uniform if unlisted."""
    lt = getattr(cfg, "layer_types", None)
    if lt:
        return list(lt)
    return ["full_attention"] * n_layers


def swap_layer_stack(parent, attr_name, new_layers):
    """Replace the stack in place and make sure the same attribute now holds new_layers."""
    setattr(parent, attr_name, new_layers)
