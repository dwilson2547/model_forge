"""One-off: dataset links in main card + push both cards to HF (run on the Altra)."""
from pathlib import Path

from huggingface_hub import HfApi

MAIN = "dwilson2547/Qwen3.5-4B-drop2-mixed-repair"
GGUF = "dwilson2547/Qwen3.5-4B-drop2-mixed-repair-GGUF"

FW = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu"
WT = "https://huggingface.co/datasets/Salesforce/wikitext"
LC = "https://github.com/ggml-org/llama.cpp"
PY = "https://github.com/python/cpython"

p = Path("/home/daniel/m0/publish/mixfull/README.md")
s = p.read_text()
s = s.replace(
    "61% FineWeb-EDU / 27% wikitext-2-train / 12% code (llama.cpp C/C++ + CPython stdlib).",
    f"61% [FineWeb-EDU]({FW}) / 27% [wikitext-2]({WT}) (train split) / "
    f"12% code ([llama.cpp]({LC}) C/C++ + [CPython]({PY}) stdlib).")
s = s.replace(
    "Training data: FineWeb-EDU, wikitext-2 (CC BY-SA 3.0, Salesforce), publicly\n"
    "available open-source code.",
    f"Training data: [FineWeb-EDU]({FW}), [wikitext-2]({WT}) "
    "(CC BY-SA 3.0, Salesforce),\nand publicly available open-source code "
    f"([llama.cpp]({LC}), [CPython]({PY}) stdlib).")
assert s.count(WT) >= 2 and FW in s, "replacement failed"
p.write_text(s)
print("card updated", flush=True)

api = HfApi()
api.upload_file(repo_id=GGUF,
                path_or_fileobj="/home/daniel/m0/publish/gguf-README.md",
                path_in_repo="README.md", repo_type="model",
                commit_message="Add model card: usage, MTP note, provenance links")
print("gguf card up", flush=True)
api.upload_file(repo_id=MAIN,
                path_or_fileobj=str(p),
                path_in_repo="README.md", repo_type="model",
                commit_message="Link training corpora to their HF dataset pages")
print("main card up", flush=True)
