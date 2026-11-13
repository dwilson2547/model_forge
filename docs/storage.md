# Storage: where forge files live

Three tiers share one tree, so a path means the same thing on each of them. Paths in code
come from `surgery/paths.py`, and scripts use `${FORGE_ROOT:-~/forge}`.

| Tier | Where | What |
|---|---|---|
| Repo | `ai/model-forge` | All code, recipes, cards, and the result ledgers in `results/`. No weights. |
| Working | `~/forge/` on the Altra and the desktop | What is in use now. Local NVMe. |
| Cold archive | `/mnt/models/` on the Altra (CIFS share, spinning disk) | Bulk storage we may want back. |

## The tree

```
~/forge/                                   (working; same shape on both machines)
  base/<hf-org>/<repo>/                    upstream weights in use, e.g. base/Qwen/Qwen3.5-4B
  data/                                    corpora: wiki/fineweb/code .test.txt, mixed.train.txt, sft-v3-*
  runs/<family>/<experiment>/<arm>/        outputs + that arm's logs
  results/                                 ledgers: multicorpus.jsonl, bench.jsonl, longwindow.jsonl

/mnt/models/                               (cold)
  hf/<hf-org>/<repo>/                      anything downloaded from HF: the user's LM Studio
                                           downloads and the forge's upstream weights alike
  forge/<family>/<experiment>/<arm>/       archived runs and builds, same paths as ~/forge/runs/
```

- **Families:** `qwen35-4b`, `qwen38-27b`. A new model gets its own family.
- **Experiments:** `slice` (unrepaired cuts, by recipe name), `drop2-repair` (one arm per
  repair config), `baseline` (the do-nothing quants and their imatrix), `drop-context`
  (design §0j.10), `rescore`, `cloud-session1`, `lmstudio-builds` (MTP-grafted probe GGUFs).
- **Ledgers:** the working copy is in `~/forge/results/` on the machine that runs the floors.
  After a run, copy it into the repo's `results/` and commit, so every number in
  `design.md` is versioned. The Altra's repo copy is an rsync target, not a git clone, so
  the ledger cannot live there.

## The cold share: rules

The share is CIFS on spinning disk: slow, large, and reliable enough for archives.

- **Never read from it during a run.** Copy an arm back to `~/forge/` first.
- **Archive (copy, verify, then delete the local copy):** anything expensive to rebuild.
  That covers the 27B imatrix (24.4 h), the BF16 source, repaired weights (a GPU run each),
  published and probe builds, and upstream downloads.
- **Delete without archiving:** anything rebuildable in under about an hour, such as
  unrepaired slices, ranked/fused cuts and quant intermediates. Their numbers are already
  in the ledgers.
- **Each archived arm carries its card.** Every trained arm has a card in the repo,
  `cards/arms/<family>/<arm>.md`: what it is, its old folder names, the recipe, the floor
  and the verdict. `cards/arms/<family>/README.md` is the one-line-per-arm index. Write
  the card when the arm lands, not at archive time.
- **Archive with `scripts/storage/archive-arm.sh <family> <experiment> <arm>`.** It copies
  `hf/`, `adapter/`, any `.gguf` and the run json, writes `SHA256SUMS` and a `README.md`
  (the card plus the commit and hashes), and re-hashes on the share. `--delete-local`
  removes the local weights only after every hash matches.

## Where the old paths went (migration of 2026-11-13)

The Altra's move is `scripts/storage/migrate-ampere.sh` (renames only; it waits for the
old-path drivers to finish). The desktop was moved by hand the same day.

`design.md`, `TODO.md` and `docs/issues/` keep their historical paths; this table maps them.

| Old | New |
|---|---|
| Altra `~/m0/` | Altra `~/forge/` |
| desktop `~/forge-gpu/` | desktop `~/forge/` |
| `~/m0/registry/Qwen3.5-4B`, desktop `registry/Qwen3.5-4B` | `base/Qwen/Qwen3.5-4B` |
| `~/m0/registry/*.gguf` (upstream quants) | `base/<org>/<repo>/` |
| `~/m0/data/`, desktop `data/` | `data/` |
| `~/m0/runs/*-results.jsonl` | `results/{multicorpus,bench,longwindow}.jsonl` |
| `~/m0/baseline-27b/` | `runs/qwen38-27b/baseline/` (`imatrix-27b-bf16.dat` → `imatrix-27b-bf16.gguf`: it was always GGUF) |
| `~/m0/runs/drop27/` | `runs/qwen38-27b/drop-context/` |
| `~/m0/runs/rescore-27b/`, `rescore-4b/` | `runs/qwen38-27b/rescore/`, `runs/qwen35-4b/rescore/` |
| `~/m0/runs/qwen35-4b-<name>/`, `drop2-unrep-2529/` | `runs/qwen35-4b/slice/<name>/`, `slice/drop2-unrep-2529/` |
| `~/m0/runs/drop2-repair-<arm>/`, desktop `drop2-repair-<arm>/` | `runs/qwen35-4b/drop2-repair/<arm>/`, renamed to the design.md arm names: each card lists its old names |
| desktop `cloud-session1/run1-full-lr<lr>/` | `runs/qwen35-4b/full-repair-2529/lr<lr>/` |
| desktop `drop2-repair/` (no suffix), Altra `drop2-repaired/` | `runs/qwen35-4b/drop2-repair/wikirep/` (the first repair) |
| desktop `calib/`, `calib-r2fixes/` | `runs/qwen35-4b/drop2-repair/calib/`, `.../calib-r2fixes/` |
| desktop loose `*.log`, `*.yaml` in `forge-gpu/` | `runs/qwen35-4b/drop2-repair/_logs/` |
| Altra `~/m0/registry/Qwen3-1.7B-*.gguf` | `base/_origin-unrecorded/` (the source repo was never written down) |
| Altra loose logs in `~/m0/`, `~/m0/runs/`, `~/m0/publish/` | `runs/_legacy-logs/` |
| `~/m0/runs/drop2-<arm>-repaired/`, `<arm>-recovered/` | `runs/qwen35-4b/drop2-repair/<arm>/gguf/`, `.../<arm>/recovered/` |
| desktop `hf-edited/`, `runs/qwen35-4b-drop2-explicit-2529/` | `runs/qwen35-4b/slice/drop2-ranked/hf-edited/`, `runs/qwen35-4b/slice/drop2-explicit-2529/` |
| `~/m0/lmstudio-out/sftrep{2,3}/`, share `sftrep*/ codefw/ mixfull/ unrepaired/ wikirep/` | `forge/qwen35-4b/lmstudio-builds/<name>/` |
| desktop `cloud-session1/` | `runs/qwen35-4b/cloud-session1/` |
| share top-level publisher dirs | `/mnt/models/hf/<publisher>/` |
| loose scripts in `~/m0/`, `~/` (Altra) and `cloud-session1/` | repo `scripts/legacy/` (committed as-is; paths inside are the old ones) |
| `~/m0/publish/*.py`, the 27B card | repo `publish/`, `cards/qwen38-27b-iq3m/` |
| `~/m0/sweep*/`, `bench_grid.txt`, rss files | repo `results/bench/` |
