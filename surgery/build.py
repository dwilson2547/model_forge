"""Build stage: save the edited HF checkpoint, convert to GGUF, quantize.

llama.cpp and llama-quantize are pinned subprocesses — the toolchain versions are a config
value, never ambient PATH state.
"""

import json
import subprocess
import time
from pathlib import Path


class BuildError(RuntimeError):
    pass


def _run(cmd, cwd=None, tail=25):
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        blob = "\n".join((proc.stdout + "\n" + proc.stderr).splitlines()[-tail:])
        raise BuildError(f"command failed ({proc.returncode}): {' '.join(map(str, cmd))}\n{blob}")
    return time.time() - t0


def save_hf(model, tokenizer, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    if tokenizer is not None:
        tokenizer.save_pretrained(out)
    return out


def to_gguf(hf_dir, out_gguf, llamacpp_dir, outtype="f16", python="python3"):
    """convert_hf_to_gguf.py on the edited checkpoint."""
    out_gguf = Path(out_gguf)
    out_gguf.parent.mkdir(parents=True, exist_ok=True)
    convert = Path(llamacpp_dir) / "convert_hf_to_gguf.py"
    if not convert.exists():
        raise BuildError(f"convert script not found: {convert}")
    took = _run([python, str(convert), str(hf_dir),
                 "--outfile", str(out_gguf), "--outtype", outtype])
    return {"gguf": str(out_gguf), "outtype": outtype, "seconds": round(took, 1)}


def quantize(in_gguf, out_gguf, quant, llamacpp_dir):
    binpath = Path(llamacpp_dir) / "build" / "bin" / "llama-quantize"
    if not binpath.exists():
        raise BuildError(f"llama-quantize not found: {binpath}")
    out_gguf = Path(out_gguf)
    out_gguf.parent.mkdir(parents=True, exist_ok=True)
    took = _run([str(binpath), "--allow-requantize", "--leave-output-tensor",
                 str(in_gguf), str(out_gguf), quant])
    return {"gguf": str(out_gguf), "quant": quant, "seconds": round(took, 1),
            "bytes": out_gguf.stat().st_size}


def build(recipe, report, model, tokenizer, python="python3"):
    """Persist + convert + quantize per recipe. Appends a 'build' section to report."""
    llamacpp = recipe.llamacpp_dir
    work = Path(recipe.out_dir)
    work.mkdir(parents=True, exist_ok=True)

    steps = {}
    hf_dir = save_hf(model, tokenizer, work / "hf-edited")
    steps["hf_dir"] = str(hf_dir)

    f16 = to_gguf(hf_dir, work / f"{recipe.name}-f16.gguf", llamacpp, python=python)
    steps["f16"] = f16

    from .recipe import Quantize
    qop = next((o for o in recipe.ops if isinstance(o, Quantize)), None)
    quants = {}
    if qop:
        for q in qop.quants:
            quants[q] = quantize(f16["gguf"], work / f"{recipe.name}-{q}.gguf", q, llamacpp)
    steps["quants"] = quants

    report["build"] = steps
    (work / "results.json").write_text(json.dumps(report, indent=2, default=str))
    return report
