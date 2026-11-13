import json, sys
sys.path.insert(0, "/home/daniel/model-forge")
from surgery.build import to_gguf, quantize
from surgery import evaluate

LLAMA = "/home/daniel/llama.cpp"
HF = "/home/daniel/m0/runs/qwen35-4b-drop2-ranked/recovered-hf"
OUT = "/home/daniel/m0/runs/drop2-repaired"
WIKI = "/home/daniel/m0/data/wiki.test.txt"
PROMPT = "/home/daniel/m0/prompt.txt"

f16 = to_gguf(HF, f"{OUT}/repaired-f16.gguf", LLAMA, python=sys.executable)
q4 = quantize(f16["gguf"], f"{OUT}/repaired-Q4_K_M.gguf", "Q4_K_M", LLAMA)
res = {
    "f16": {**evaluate.ppl(f16["gguf"], WIKI, LLAMA, chunks=24),
            **evaluate.speed(f16["gguf"], PROMPT, LLAMA)},
    "Q4_K_M": {**evaluate.ppl(q4["gguf"], WIKI, LLAMA, chunks=24),
               **evaluate.speed(q4["gguf"], PROMPT, LLAMA)},
    "sizes": {"f16_bytes": f16 is None or None},
}
res["sizes"] = {"q4_bytes": q4["bytes"]}
print(json.dumps(res, indent=2))
