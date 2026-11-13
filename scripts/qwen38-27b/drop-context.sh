#!/bin/bash
# 27B drop-for-context experiment, unrepaired rows (design §0j.10), on the Altra.
#   nohup bash ~/model-forge/scripts/qwen38-27b/drop-context.sh > ~/forge/runs/qwen38-27b/drop-context/driver.log 2>&1 &
# 1. fixed scorer, 10 windows, picks for 4/6/8/13          (~2.3 h, ~150 GB RAM)
# 2. per k: GGUF slice of the §0j.4 BF16 source -> IQ3_M with the base imatrix remapped
#    -> floor. This is the user's comparison: against unsliced IQ3_M 6.129 / 8.697 / 2.565.
# 3. per k: re-slice -> Q8_0 -> floor -> delete. The redundancy rows, against Q8 5.89 / 8.36 / 2.50.
# Floors take ~70 min each on CPU. Re-running skips finished rows (by label in the results
# jsonl), so after a failure, fix and re-launch the same command.
set -euo pipefail
trap 'echo "=== FAILED at line $LINENO: $BASH_COMMAND $(date -Is) ==="' ERR

ROOT=${FORGE_ROOT:-~/forge}
export FORGE_ROOT=$ROOT
B=$ROOT/runs/qwen38-27b/baseline
OUT=$ROOT/runs/qwen38-27b/drop-context
LLAMA=~/llama.cpp/build/bin
SRC=$B/src-bf16/Qwen3.8-27B-BF16-00001-of-00002.gguf
IMAT=$B/imatrix-27b-bf16.gguf
RESULTS=$ROOT/results/multicorpus.jsonl
SCORES=$OUT/rescore-10win.json
COUNTS="4 6 8 13"
mkdir -p "$OUT"
source ~/forge_env/bin/activate
cd ~/model-forge
export PYTHONPATH=~/llama.cpp/gguf-py

log() { echo "=== $* $(date -Is) ==="; }
done_row() { grep -q "\"label\": \"$1\"" "$RESULTS" 2> /dev/null; }
picks() { python -c "import json,sys; print(*json.load(open('$SCORES'))['picks']['count$1']['new'])"; }
slice_bf16() {
  local k=$1
  python -m surgery.gguf_drop "$SRC" "$OUT/drop$k-bf16.gguf" --drop $(picks "$k") | tail -1
}

if [ ! -s "$SCORES" ]; then
  log "[1] rescore 10 windows"
  python surgery/rescore.py --hf-dir $ROOT/base/Qwen/Qwen3.8-27B --calib $ROOT/data/wiki.train.txt \
    --counts $COUNTS --windows 10 --threads 96 --out "$SCORES.tmp" > "$OUT/rescore-10win.log" 2>&1
  mv "$SCORES.tmp" "$SCORES"
fi
for k in $COUNTS; do echo "drop$k: $(picks "$k")"; done

for k in $COUNTS; do
  label=27b-drop$k-unrep-IQ3_M
  done_row "$label" && continue
  q=$OUT/drop$k-IQ3_M.gguf
  if [ ! -s "$q" ]; then
    log "[2] drop$k slice + IQ3_M"
    slice_bf16 "$k"
    python -m surgery.imatrix_remap "$IMAT" "$OUT/drop$k-imatrix.gguf" --drop $(picks "$k")
    $LLAMA/llama-quantize --imatrix "$OUT/drop$k-imatrix.gguf" "$OUT/drop$k-bf16.gguf" "$q.tmp" IQ3_M \
      > "$OUT/quant-drop$k-IQ3_M.log" 2>&1
    mv "$q.tmp" "$q"
    rm -f "$OUT/drop$k-bf16.gguf"
  fi
  ls -l "$q" | awk '{print $9, $5}'
  log "[2] drop$k IQ3_M floor"
  python surgery/multicorpus.py "$label" "$q"
done

for k in $COUNTS; do
  label=27b-drop$k-unrep-Q8_0
  done_row "$label" && continue
  q=$OUT/drop$k-Q8_0.gguf
  log "[3] drop$k slice + Q8_0"
  [ -s "$q" ] || {
    slice_bf16 "$k"
    $LLAMA/llama-quantize "$OUT/drop$k-bf16.gguf" "$q.tmp" Q8_0 > "$OUT/quant-drop$k-Q8_0.log" 2>&1
    mv "$q.tmp" "$q"
    rm -f "$OUT/drop$k-bf16.gguf"
  }
  ls -l "$q" | awk '{print $9, $5}'
  log "[3] drop$k Q8_0 floor"
  python surgery/multicorpus.py "$label" "$q"
  rm -f "$q"
done
log "ALL DONE"
