# forge-cloud: the rented-GPU hot loop (Brev lane)

Doctrine (2026-11-12, design.md §0j.8): heavy jobs (27B imatrix/quants/floors,
full-weight repair, MoE training) run on rented H100-class GPUs — ~$1-5 per
experiment. Local stays: 5080 for interactive LM Studio probes + 4B/9B dev;
Ampere for archive/staging/serial evals where wall-clock has no value.

## 0. Instance

- Target: 1× H100 80GB (or 2× if a run needs it), spot is fine — Runs must be
  restart-safe (checkpoint to /straw checkpoint dir + push HF on completion).
- Staging from the desktop is ONE rsync: repo + data (below). Models that exist
  on HF (base Qwen, our published repos) pull in-cloud — never stage those.

```
rsync -a --exclude .git ../  brev-host:~/forge/          # repo (surgery/, bench/, recipes/, cloud/)
mkdir -p ~/forge-data                                     # on the desktop:
rsync -a ~/forge-gpu/data/mixed.train.txt \
          ~/forge-gpu/data/wiki.test.txt  \
          ~/forge-gpu/data/fineweb.test.txt \
          ~/forge-gpu/data/code.test.txt  brev-host:~/forge-data/
```

## 1. Bootstrap (~5-10 min, once per instance)

```
bash cloud/bootstrap.sh          # venv (forge-env/), python deps, cuda llama.cpp -> ~/llama.cpp
source forge-env/bin/activate
```

## 2. First-run protocol (cost-capped, in order)

Run 0 — PORT PARITY (~$0.50): re-run the [25,29] repair arm, same seed/protocol
  as design §0j.7 (`surgery/recover_gpu.py`, seed 1234, striped 12M, r16@1e-4,
  2 epochs, all targets). Expect loss curve ≈ 2.625→2.230. Floor on the box vs
  8.031/11.894/3.485; parity within noise = the port is verified.

Run 1 — THE PLATEAU ESCALATION (~$1-2): full-weight repair of the same sliced
  base (drops LoRA entirely — the capability written off as "out of reach
  locally" in §0f/§0j). Same corpus, same 2-epoch protocol; optionally a 1-epoch
  companion arm against the dose-overfit risk. Floor vs 11.894 fineweb decides:
    - substantially below → the wall was a rank-16 ceiling, the repair pipeline
      is viable, cloud lane becomes the default for all repairs;
    - same wall → the wall is genre damage in the corpus at this scale, and the
      slicing line pins its hopes on task-level repair (SFT route) instead.

## 3. Cost discipline

- Verdict-experiments only; no idle instance overnight — shut down after the
  run + artifact push, or use scheduled teardown.
- Every run pushes its artifacts (adapter/hf-recovered/recipe/provenance json)
  to a private HF repo before exit, so an interrupted spot instance loses
  nothing beyond time already paid.
- Full-weight 27B work (Qwen3.8-27B QLoRA, sliced-27B repair) stays cloud-only.

## 4. What does NOT move

- LM Studio probes, interactive inference (desktop 5080).
- 4B/9B experiment iteration (desktop; free).
- Long serial eval sweeps + model archive + SMB share (Ampere).
- GGUF conversion + CPU floors are fine on the Ampere when nobody's waiting
  for them; use the H100's CUDA llama-perplexity (ngl) when a verdict is blocking.
