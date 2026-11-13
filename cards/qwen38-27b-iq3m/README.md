---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
- qwen
- gguf
- quantization
- imatrix
- llama.cpp
pipeline_tag: text-generation
---

# Qwen3.8-27B — IQ3_M (importance-matrix quant)

An IQ3_M quantization of **Qwen3.8-27B** (25.5B parameters), built with
[`llama.cpp`](https://github.com/ggml-org/llama.cpp) from an importance matrix
computed on a mixed-domain calibration set. Aimed at 16 GB-class GPUs: it renders
at ~11.9 GiB, leaving room for a 16k-token context (measured at 15.7 / 16.3 GiB
total on an RTX 5080).

## Attribution

- **Base model:** [Qwen3.8-27B](https://huggingface.co/Qwen) by the Alibaba **Qwen
  team** — Apache-2.0. This repo contains *no modified weights* of the base model,
  only a quantized rendition.
- **GGUF source:** [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
  — the BF16 GGUF conversion this quant derives from.
- **Quant tooling:** [llama.cpp](https://github.com/ggml-org/llama.cpp)
  (`llama-imatrix` + `llama-quantize`), MIT license.

## Provenance (read this before comparing quants)

This file is a **second-generation quant**: it was quantized from the unsloth
**Q8_0** GGUF (itself derived from the BF16), using `--allow-requantize`. Q8_0 is
near-lossless (~8.5 bpw), so the extra generation costs little in practice — but if
you need the strictest single-generation chain, first-generation BF16-sourced builds
are straightforward to make with the recipe below.

Measured (2026-11-12): an honest BF16-source IQ3_M (same imatrix recipe, same
floor harness) lands at 6.129 / 8.697 / 2.565 — a second-generation penalty of
just +0.036 / +0.042 / -0.004 versus this file. The quant-from-quant hop is
empirically negligible at this size/precision; see the perplexity table for the
full comparison.

## Quant recipe

```
# importance matrix (mixed-domain calibration, 1.2M chars ≈ 320k tokens)
llama-imatrix -m Qwen3.8-27B-Q8_0.gguf -f imatrix-mix.txt -o imatrix-27b.dat
# quantize
llama-quantize --allow-requantize --imatrix imatrix-27b.dat \
    Qwen3.8-27B-Q8_0.gguf Qwen3.8-27B-IQ3_M.gguf IQ3_M
```

### Calibration data & licenses

The importance matrix was computed over a ~1.2M-character mixed-domain sample:

| component | share | source | license |
|---|---:|---|---|
| Wiki-style encyclopedic prose | ~42% | Wikitext (Salesforce) | CC BY-SA 3.0 |
| Web prose | ~42% | FineWeb | ODC-By |
| C/C++ source | ~17% | llama.cpp sources | MIT |

No calibration text is included in this repo; only the activation statistics of the
importance matrix were used, and they are embedded in the quantized weights rather
than shipped as a separate artifact.

## Perplexity (three-corpus floor, 24 × 512-token chunks per corpus)

| Qwen3.8-27B variant | size | wikitext | fineweb | code |
|---|---:|---:|---:|---:|
| Q8_0 (near-lossless reference) | 27 GiB | 5.89 | 8.36 | 2.50 |
| Unsloth Dynamic Q4_K_M (UD) | 15.3 GiB | 5.92 | 8.45 | 2.52 |
| **this repo — IQ3_M (from Q8_0, imatrix)** | **11.9 GiB** | **6.17** | **8.74** | **2.56** |
| IQ3_M from BF16 (single-generation, same recipe) | 11.9 GiB | 6.13 | 8.70 | 2.57 |

Reading: at roughly 3.3 bpw this costs +0.28 to +0.38 perplexity over the Q8
reference while shaving 3.4 GiB versus the dynamic Q4_K_M — a reasonable point on
the size/quality curve for a 3-bit build.

## Practical notes

- Runs comfortably on a 16 GB GPU; a 16k-token context with the default KV cache
  lands at ~15.7 GiB total — effectively the practical ceiling on a 16 GB card.
- This is a thinking-style model: expect long reasoning sections. Budget context
  accordingly.
- Generation quality at low temperature is unaffected by this quant's errors;
  at very high sampling temperatures the lower precision shows faster than Q4.

## License

Apache-2.0 (inherited from the base model). See the `LICENSE` file in this repo.
Quantization of Apache-2.0 model weights is a derivative rendition; the same
license terms carry over.
