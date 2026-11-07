"""Recipe model: declarative surgery specs parsed into dataclasses."""

from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class DropLayers:
    count: int
    strategy: str = "uniform"   # uniform | ranked (neighbor-similarity scores on calibration text)
    # Hybrid-attention safety: layers whose type appears in `protected_types` are never dropped.
    protected_types: tuple = ("full_attention",)


@dataclass
class FusePairs:
    """LaCo-style: when a layer is dropped, average its weights into its retained predecessor."""
    enabled: bool = False


@dataclass
class SliceFFN:
    keep_frac: float            # fraction of FFN intermediate neurons kept (importance-ranked)


@dataclass
class PruneHeads:
    keep_heads: int             # query heads kept in full-attention layers (kv-group aligned)


@dataclass
class Quantize:
    quants: list = field(default_factory=lambda: ["Q4_K_M"])


OPS = {
    "drop_layers": DropLayers,
    "fuse_pairs": FusePairs,
    "slice_ffn": SliceFFN,
    "prune_heads": PruneHeads,
    "quantize": Quantize,
}


@dataclass
class Recipe:
    name: str
    base_dir: str               # HF checkpoint on disk
    out_dir: str                # work dir for HF edit + GGUFs + results
    ops: list
    calib_file: str = ""         # text file used for ranked scoring
    llamacpp_dir: str = ""       # path to llama.cpp checkout (bin/ + convert_hf_to_gguf.py)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "base_dir": self.base_dir,
            "out_dir": self.out_dir,
            "ops": [vars(op) for op in self.ops],
            "calib_file": self.calib_file,
            "llamacpp_dir": self.llamacpp_dir,
        }


def load_recipe(path_or_dict: Any) -> Recipe:
    if isinstance(path_or_dict, (str, bytes)):
        import pathlib
        raw = yaml.safe_load(pathlib.Path(path_or_dict).read_text())
    else:
        raw = path_or_dict

    ops = []
    for entry in raw["ops"]:
        if isinstance(entry, dict):
            name, kwargs = entry["op"], {k: v for k, v in entry.items() if k != "op"}
        else:
            raise ValueError(f"op entry must be a mapping with an 'op' key: {entry!r}")
        if name not in OPS:
            raise ValueError(f"unknown op {name!r}; known: {sorted(OPS)}")
        ops.append(OPS[name](**kwargs))

    return Recipe(
        name=raw["name"],
        base_dir=raw["base_dir"],
        out_dir=raw["out_dir"],
        ops=ops,
        calib_file=raw.get("calib_file", ""),
        llamacpp_dir=raw.get("llamacpp_dir", ""),
    )
