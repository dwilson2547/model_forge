"""Quick evaluation: perplexity + decode/prompt speed via llama.cpp binaries.

Methodology pinned from the M0 sweep: `-t 32 -tb 96` for llama-cli speed runs;
perplexity via llama-perplexity on a fixed text file + chunk count so candidates
are always compared like-for-like. stdout/stderr discipline: llama-cli timings are
printed on stdout as "[ Prompt: X t/s | Generation: Y t/s ]"; stdin must be
/dev/null or it hangs.
"""

import re
import subprocess
from pathlib import Path


def _bin(llamacpp_dir, name):
    p = Path(llamacpp_dir) / "build" / "bin" / name
    if not p.exists():
        raise FileNotFoundError(p)
    return str(p)


def speed(model_gguf, prompt_file, llamacpp_dir, n=64, t=32, tb=96):
    """Decode + prompt speed via llama-cli single-shot. Same method as the M0 sweep."""
    proc = subprocess.run(
        [_bin(llamacpp_dir, "llama-cli"), "-m", str(model_gguf),
         "--file", str(prompt_file), "-n", str(n), "-t", str(t), "-tb", str(tb),
         "--no-warmup"],
        stdin=subprocess.DEVNULL, capture_output=True, text=True,
    )
    text = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(f"llama-cli failed rc={proc.returncode}:\n{text[-800:]}")
    m = re.search(r"\[ Prompt: ([0-9.]+) t/s \| Generation: ([0-9.]+) t/s \]", text)
    if not m:
        raise RuntimeError(f"timing line not found in output:\n{text[-800:]}")
    return {"prompt_tps": float(m.group(1)), "gen_tps": float(m.group(2)),
            "threads": t, "threads_batch": tb}


def ppl(model_gguf, data_file, llamacpp_dir, chunks=24, threads=32):
    """Perplexity via llama-perplexity. Fixed corpus + chunk count = comparable scores."""
    proc = subprocess.run(
        [_bin(llamacpp_dir, "llama-perplexity"), "-m", str(model_gguf),
         "-f", str(data_file), "--chunks", str(chunks), "-t", str(threads)],
        stdin=subprocess.DEVNULL, capture_output=True, text=True,
    )
    text = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(f"llama-perplexity failed rc={proc.returncode}:\n{text[-800:]}")
    m = re.search(r"Final estimate: PPL = ([0-9.]+)", text)
    if not m:
        raise RuntimeError(f"PPL line not found in output:\n{text[-800:]}")
    return {"ppl": float(m.group(1)), "chunks": chunks}
