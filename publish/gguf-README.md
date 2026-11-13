---
license: apache-2.0
base_model: dwilson2547/Qwen3.5-4B-drop2-mixed-repair
library_name: gguf
tags:
- gguf
- experimental
- model-surgery
- structured-pruning
- recovery-finetuning
---

# Qwen3.5-4B · drop2 · mixed-diet repair — GGUF

GGUF builds of [dwilson2547/Qwen3.5-4B-drop2-mixed-repair](https://huggingface.co/dwilson2547/Qwen3.5-4B-drop2-mixed-repair):
[Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) with 2 of its 36 decoder layers
surgically removed (importance-ranked by neighbor-cosine similarity; the hybrid
architecture's full-attention layers untouched), then repaired with QLoRA (r=16) on a
small mixed corpus. See the main repo for the full write-up (surgery, repair recipe,
perplexity results across three unrelated corpora, and honest limitations).

## Files

- `Qwen3.5-4B-drop2-mixed-repair-Q4_K_M.gguf` (2.63 GB) — Q4_K_M quantization. The MTP
  (multi-token-prediction) draft block from the base checkpoint is embedded in the file
  after conversion; draft tensors predate the slice, spec-decode acceptance is approximate,
  and drafts are always verified by the main model — output correctness is unaffected.

## Usage

Standard GGUF runtimes: llama.cpp, LM Studio, and compatible servers. The Qwen chat
template is embedded; use chat/completions mode for best behavior.

```
llama-server -m Qwen3.5-4B-drop2-mixed-repair-Q4_K_M.gguf --port 8080
```

In our measurements the Q4-vs-f16 perplexity gap on this family is small; the dominant
quality factor is the surgery + repair, not the quantization.

## Caveats

Partial repair on diverse corpora, no academic-benchmark numbers yet, and long-context
loop behavior prefers slightly lower sampling temperatures. Research artifact, not a
production model — full limitation list on the main repo.

## License & attribution

Apache-2.0, derived from [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B).
Training corpora: [FineWeb-EDU](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
and [wikitext-2](https://huggingface.co/datasets/Salesforce/wikitext) (CC BY-SA 3.0,
Salesforce), plus publicly available open-source code. LICENSE file in this repo.
