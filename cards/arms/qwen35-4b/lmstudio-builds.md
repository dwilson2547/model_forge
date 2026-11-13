# qwen35-4b / lmstudio-builds

GGUFs built for LM Studio spot-checks: f16 and Q4_K_M, each with the MTP draft block grafted back
from the base (the slice pipeline drops it, and llama.cpp/LM Studio refuse to load without it).
Each one is a quant of a trained arm; the arm's card says what it is.

| folder | arm card | what |
|---|---|---|
| `unrepaired` | — | drop2 [24,28] slice, no repair (wound 12.82 / 13.55 / 3.57) |
| `wikirep` | [wikirep](wikirep.md) | wikitext-only repair, the domain trap |
| `codefw` | [codefw](codefw.md) | code+fineweb repair |
| `mixfull` | [mixfull](mixfull.md) | whole-mix repair; the published one |
| `sftrep` | [sftrep1](sftrep1.md) | SFT v1, collapsed |
| `sftrep2` | [sftrep2](sftrep2.md) | SFT v2, never closes its think block |
| `sftrep3` | [sftrep3](sftrep3.md) | SFT v3 triage model; private data, never publish |
