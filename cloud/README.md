# forge-cloud: the rented-GPU hot loop (Brev lane)

Doctrine (design §0j.8): rent H100-class time where it saves hours. The budget is about
$20–30 a month: at $1.66/hr (cheapest Brev A100 the user saw 2026-11-12), that is
~12–18 GPU-hours. Local
stays: the 5080 for LM Studio probes and 4B/9B dev, the Ampere for archive, staging and
serial evals where wall-clock doesn't matter.

**On-demand, not spot.** Runs are an hour or two. `--resume` does continue a run
exactly (optimizer, schedule, shuffle order restored), but a preemption still costs the
reload and the lost minutes. Use spot only for a long job, and only with `--save-every`
set.

## 0. Instance

1× A100 **80GB** or H100 80GB, VM mode (full machine, ssh + sudo; Jupyter is irrelevant). Full-weight Run 1 needs ~40 GB before activations, so a
40GB A100 only fits the QLoRA runs. On an A100, run bootstrap as
`CUDA_ARCH=80 bash ~/forge/cloud/bootstrap.sh`; the default 90 builds llama.cpp for H100
only. If the image has only the driver, bootstrap installs the matching CUDA toolkit
from NVIDIA's apt repo. Disk: 256 GB covers the 4B work (≈100 GB worst case); 27B work
needs more.
A driver reporting CUDA ≥ 13.2 gets the desktop's exact torch (2.15.0+cu132). Older
drivers get the nearest wheel and a warning, and Run 0 then also tests that torch.

## 1. Stage (from the desktop)

Run everything from the **repo root**. `../` from here is the whole `ai/` domain.

```bash
cd ~/documents/workspace/ai/model-forge
B=<instance ssh alias>          # whatever `brev` set up in ~/.ssh/config
rsync -a --exclude .git --exclude __pycache__ ./ $B:forge/
ssh $B 'mkdir -p forge-data forge-runs'
rsync -a ~/forge-gpu/data/{mixed.train.txt,wiki.test.txt,fineweb.test.txt,code.test.txt} $B:forge-data/
# the [25,29] slice (7.5 GB; not on HF). Re-upload per instance, or push it once to a
# private HF repo and `hf download` it in-cloud instead (faster after the first time):
rsync -a ~/forge-gpu/runs/qwen35-4b-drop2-explicit-2529/hf-edited/ $B:forge-runs/2529-hf-edited/
```

`wiki.test.txt` is the Altra's `~/m0/data/wiki.test.txt`, copied to the desktop
2026-11-12. All four test corpora must be the floor's exact files.

## 2. Bootstrap (once per instance)

```bash
bash ~/forge/cloud/bootstrap.sh     # venv ~/forge-env, llama.cpp @ e95426f (CUDA), ~/forge.env
source ~/forge.env                  # every new shell
export HF_TOKEN=...                 # in the session only; never in a file under ~/forge
```

The sanity step prints whether `fla` (required) and `causal_conv1d` (optional, source
build) are present. `recover_gpu.py` records the full env in `recover-results.json`.

## 3. Runs

### Run 0a: eval port (minutes)

Floor the **unrepaired** slice on the H100 and compare against the Altra CPU row
`drop2-2529-unrepaired-f16/-q4` (design §0j.9). This separates "CUDA llama.cpp scores
differently" from "the trainer ported differently" before any training money is spent.

```bash
cd ~/forge-runs
python ~/llama.cpp/convert_hf_to_gguf.py 2529-hf-edited --outfile unrep-f16.gguf --outtype f16
~/llama.cpp/build/bin/llama-quantize unrep-f16.gguf unrep-Q4_K_M.gguf Q4_K_M
python ~/forge/surgery/multicorpus.py cloud-unrep-f16 $PWD/unrep-f16.gguf cloud-unrep-q4 $PWD/unrep-Q4_K_M.gguf
```

### Run 0b: trainer port parity

The §0j.7 arm, same protocol. **No `--val-frac`**: holding out data changes the training
set.

```bash
cd ~/forge
python -m surgery.recover_gpu ~/forge-runs/2529-hf-edited ~/forge-data/mixed.train.txt \
  --seed 1234 --epochs 2 --max-chars 12000000 --lr 1e-4 --lora-targets all --ctx 512 --batch 4 \
  --out ~/forge-runs/run0-parity --push-repo <hf-user>/forge-run0-parity
```

Expect the loss to go 2.625 → ~2.230 and floors near 8.031 / 11.894 / 3.485 (f16) and
8.207 / 12.166 / 3.537 (Q4_K_M), after any eval offset Run 0a measured. Within ~0.1 per
corpus means the port is good. Beyond the ±0.27 fineweb shuffle-noise band (§0j) means
something is broken: stop and diff `env` in `recover-results.json` against the desktop.

### Run 1: full-weight repair (the plateau escalation)

The comparator is the **desktop fresh-data arm** `drop2-repair-2529-fresh1e`: QLoRA on
the full 40.8M-char corpus, 1 epoch, seed 1234, `--val-frac 0.01` (§0j.9). Run 1 uses the
same data and split, and changes only LoRA → full weights. Two lr arms, since 1e-5 is a
starting guess:

```bash
for LR in 1e-5 3e-6; do
python -m surgery.recover_gpu ~/forge-runs/2529-hf-edited ~/forge-data/mixed.train.txt \
  --full --lr $LR --seed 1234 --epochs 1 --max-chars 41000000 --val-frac 0.01 \
  --stop-on-val-rise 2 --save-every 0 --ctx 512 --batch 4 \
  --out ~/forge-runs/run1-full-lr$LR --push-repo <hf-user>/forge-run1-full-lr$LR
done
```

Memory: about 40 GB before activations for a 4B slice (fp32 master weights + grads +
8-bit Adam). Watch `peak` in the log. `--stop-on-val-rise` stops at the current weights,
not the best ones, so read `val_curve`.

How to read the result, on the f16 fineweb floor against the fresh1e arm:
- **Clearly lower:** the wall was LoRA capacity, and full-weight becomes the repair lane.
- **The same:** the wall is the wound, not the adapter. Route around it via the SFT
  line or a smaller wound.

## 4. Cost discipline

- Verdict experiments only. Stop the instance as soon as the artifacts are pushed.
- `--push-repo` uploads `hf-recovered/`, the results json and (QLoRA) the final adapter
  to a private repo once the run finishes. If the push fails, everything is still on the
  instance disk.
- Do a `--calibrate` run (a few minutes) before any new run shape, to get tok/s and peak
  memory before committing an hour.

## 5. What stays local

- LM Studio probes and interactive inference (5080).
- 4B/9B iteration (5080, free).
- Long serial eval sweeps, model archive, SMB share (Ampere).
