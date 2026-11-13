#!/bin/bash
# Long-context check of q8_0 KV on the 27B IQ3_M (design §0j.10): kvq8 was only floor-tested at
# 512-token chunks, where the cache is tiny. Per window, the f16-KV run (-fa on) writes reference
# logits; the q8_0-KV run scores KLD against them on the same tokens. PPL alone is too blunt.
# On the Altra (CPU). Ref ~0.5 MB/token scored (248k vocab), deleted after each window.
#   nohup bash scripts/qwen38-27b/kvq8-longwindow.sh > ~/forge/runs/qwen38-27b/kvq8-lw/run.log 2>&1 &
set -euo pipefail
source ~/forge_env/bin/activate
B=~/llama.cpp/build/bin
M=~/forge/runs/qwen38-27b/baseline/srcbf16-IQ3_M.gguf
W=~/forge/runs/qwen38-27b/kvq8-lw
OUT=~/forge/results/longwindow.jsonl
CORPUS=${CORPUS:-fineweb}; F=~/forge/data/$CORPUS.test.txt
CHUNKS=${CHUNKS:-2}
mkdir -p $W
num() { grep -oE "$1" "$2" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | tail -1; }
for C in ${WINDOWS:-8192 32768}; do
  if grep -q "\"label\": \"27b-IQ3_M-kvq8\", \"corpus\": \"$CORPUS\", \"window\": $C," $OUT 2>/dev/null; then continue; fi
  echo "=== $CORPUS ctx $C x $CHUNKS: f16-KV ref $(date -Is)"
  t0=$(date +%s)
  $B/llama-perplexity -m $M -f $F -c $C -b 2048 --chunks $CHUNKS -t 96 -fa on \
      --kl-divergence-base $W/ref-$C.kld < /dev/null > $W/f16-$C.log 2>&1
  t1=$(date +%s)
  echo "=== $CORPUS ctx $C: q8_0-KV score $(date -Is) (ref took $((t1-t0))s)"
  $B/llama-perplexity -m $M -c $C -b 2048 --chunks $CHUNKS -t 96 -fa on -ctk q8_0 -ctv q8_0 \
      --kl-divergence-base $W/ref-$C.kld --kl-divergence < /dev/null > $W/q8-$C.log 2>&1
  t2=$(date +%s)
  python3 - "$CORPUS" "$C" "$CHUNKS" "$W/f16-$C.log" "$W/q8-$C.log" $((t1-t0)) $((t2-t1)) >> $OUT <<'PY'
import json, re, sys
corpus, c, n, f16, q8, s1, s2 = sys.argv[1:]
a, b = open(f16).read(), open(q8).read()
g = lambda pat, t: float(re.findall(pat, t)[-1])
print(json.dumps({"label": "27b-IQ3_M-kvq8", "corpus": corpus, "window": int(c), "chunks": int(n),
  "ppl_f16kv": g(r"Final estimate: PPL = ([0-9.]+)", a),
  "ppl_q8kv": g(r"Mean PPL\(Q\)\s*:\s*([0-9.]+)", b),
  "kld_mean": g(r"Mean\s+KLD:\s*([0-9.]+)", b), "kld_p99": g(r"99\.0%\s+KLD:\s*([0-9.]+)", b),
  "same_top": g(r"Same top p:\s*([0-9.]+)", b), "secs": [int(s1), int(s2)],
  "model": "~/forge/runs/qwen38-27b/baseline/srcbf16-IQ3_M.gguf", "flags": "-fa on; q8: -ctk q8_0 -ctv q8_0"}))
PY
  tail -1 $OUT
  rm -f $W/ref-$C.kld
done
echo "=== KVQ8-LW DONE $(date -Is)"
