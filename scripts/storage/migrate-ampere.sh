#!/bin/bash
# One-off: move the Altra's ~/m0 into the ~/forge tree (docs/storage.md). Renames only, on one
# filesystem. Nothing is copied or deleted, except empty directories. Waits for the old-path
# drivers (27b-drop-context.sh, 27b-context-arms.sh) to finish, because they write into ~/m0.
#   nohup bash ~/model-forge/scripts/storage/migrate-ampere.sh > ~/migrate-ampere.log 2>&1 &
set -euo pipefail
trap 'echo "=== FAILED at line $LINENO: $BASH_COMMAND $(date -Is) ==="' ERR

while pgrep -f "^bash [^ ]*27b-(drop-context|context-arms)\.sh" > /dev/null; do sleep 120; done
echo "=== start $(date -Is) ==="

O=~/m0
F=~/forge
R4=$F/runs/qwen35-4b
R27=$F/runs/qwen38-27b
[ -e "$F" ] && { echo "$F already exists; refusing to merge into it"; exit 1; }
mkdir -p $F/base/Qwen $F/results $R4/slice $R4/drop2-repair $R4/lmstudio-builds $R27 $F/runs/_legacy-logs

mvv() { [ -e "$1" ] || { echo "skip (absent): $1"; return 0; }; mkdir -p "$(dirname "$2")"; mv "$1" "$2"; echo "mv $1 -> $2"; }

# upstream weights
mvv $O/registry/Qwen3.5-4B $F/base/Qwen/Qwen3.5-4B
for f in Qwen3.5-4B-Q4_K_M Qwen3.5-4B-Q8_0; do mvv $O/registry/$f.gguf $F/base/unsloth/Qwen3.5-4B-GGUF/$f.gguf; done
for f in Qwen3.5-9B-Q4_K_M Qwen3.5-9B-Q8_0; do mvv $O/registry/$f.gguf $F/base/unsloth/Qwen3.5-9B-GGUF/$f.gguf; done
for f in Qwen3.8-27B-Q8_0 Qwen3.8-27B-UD-Q4_K_M; do mvv $O/registry/$f.gguf $F/base/unsloth/Qwen3.8-27B-GGUF/$f.gguf; done
# the 1.7B's source repo is not recorded anywhere
for f in Qwen3-1.7B-BF16 Qwen3-1.7B-Q8_0; do mvv $O/registry/$f.gguf $F/base/_origin-unrecorded/$f.gguf; done
mvv $O/registry/dl-27b.log $F/runs/_legacy-logs/dl-27b.log
mvv $O/data $F/data

# ledgers
mvv $O/runs/multicorpus-results.jsonl $F/results/multicorpus.jsonl
mvv $O/runs/bench-results.jsonl $F/results/bench.jsonl
mvv $O/runs/longwindow-results.jsonl $F/results/longwindow.jsonl

# 27B
mvv $O/baseline-27b $R27/baseline
mvv $R27/baseline/imatrix-27b-bf16.dat $R27/baseline/imatrix-27b-bf16.gguf
mvv $O/runs/drop27 $R27/drop-context
mvv $O/runs/rescore-27b $R27/rescore

# 4B
mvv $O/runs/rescore-4b $R4/rescore
mvv $O/runs/longwindow-logs $R4/longwindow
for d in $O/runs/qwen35-4b-*; do [ -e "$d" ] || continue; n=${d##*/qwen35-4b-}; mvv "$d" "$R4/slice/$n"; done
# after the loop: slice/base must be the moved qwen35-4b-base dir, not a fresh mkdir
mvv $O/runs/base-with-mtp-f16.gguf $R4/slice/base/base-with-mtp-f16.gguf
mvv $O/runs/drop2-unrep-2529 $R4/slice/drop2-unrep-2529
# arm names follow design.md (cards/arms/qwen35-4b/), not the old folder names
mvv $O/runs/drop2-repair-2529 $R4/drop2-repair/2529-pass1
for s in sft:sftrep1 sft2:sftrep2 sft3:sftrep3; do mvv $O/runs/drop2-repair-${s%%:*}-hf $R4/drop2-repair/${s##*:}/hf; done
for d in $O/runs/drop2-repair-*; do [ -e "$d" ] || continue; n=${d##*/drop2-repair-}; mvv "$d" "$R4/drop2-repair/$n"; done
mvv $O/runs/drop2-repaired $R4/drop2-repair/wikirep/gguf
for d in $O/runs/drop2-*-repaired; do [ -e "$d" ] || continue; n=${d##*/drop2-}; n=${n%-repaired}; mvv "$d" "$R4/drop2-repair/$n/gguf"; done
for d in $O/runs/*-recovered; do [ -e "$d" ] || continue; n=${d##*/}; n=${n%-recovered}; mvv "$d" "$R4/drop2-repair/$n/recovered"; done
mvv $O/runs/iso-eval.log $F/runs/_legacy-logs/iso-eval.log
mvv $O/runs/2529-eval.log $F/runs/_legacy-logs/2529-eval.log
for d in $O/lmstudio-out/*; do [ -e "$d" ] || continue; mvv "$d" "$R4/lmstudio-builds/${d##*/}"; done
mvv $O/publish/mixfull $R4/lmstudio-builds/mixfull-publish

# everything left in ~/m0 is loose logs and scripts (scripts are committed in scripts/legacy)
for f in $O/* $O/publish/* ~/*.sh ~/rescore-4b.py; do
  [ -f "$f" ] || continue
  case "$f" in "$HOME/Anaconda"*|"$HOME/mount"*) continue ;; esac
  mvv "$f" "$F/runs/_legacy-logs/$(basename "$(dirname "$f")")-$(basename "$f")"
done
for d in $O/sweep $O/sweep2 $O/publish/27b-iq3m-card; do mvv "$d" "$F/runs/_legacy-logs/${d##*/}"; done

rmdir $O/registry $O/runs $O/lmstudio-out $O/publish $O 2>/dev/null || true
echo "=== left behind in ~/m0 (should be nothing):"; find $O 2>/dev/null | head -20 || true
echo "=== MIGRATION DONE $(date -Is) ==="
