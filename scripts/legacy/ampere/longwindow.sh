#!/usr/bin/env bash
# Long-window perplexity floor: does the repair hold as the evaluation window stretches?
# Resumable: skips (model, corpus, window) rows already in the results file.
set -uo pipefail
PPL=~/llama.cpp/build/bin/llama-perplexity
OUT=/home/daniel/m0/runs/longwindow-results.jsonl
LOG_DIR=/home/daniel/m0/runs/longwindow-logs; mkdir -p $LOG_DIR; touch $OUT
MODELS="base:/home/daniel/m0/runs/qwen35-4b-base/qwen35-4b-base-f16.gguf drop2:/home/daniel/m0/runs/qwen35-4b-drop2-ranked/qwen35-4b-drop2-ranked-f16.gguf mixfull:/home/daniel/m0/lmstudio-out/mixfull/mixfull-f16-mtp.gguf"
CORPORA="wikitext:/home/daniel/m0/data/wiki.test.txt fineweb:/home/daniel/m0/data/fineweb.test.txt code:/home/daniel/m0/data/code.test.txt"
WINDOWS="4096 8192 16384 32768"
for m in $MODELS; do
  MNAME=${m%%:*}; MFILE=${m#*:}
  for c in $CORPORA; do
    CNAME=${c%%:*}; CFILE=${c#*:}
    for W in $WINDOWS; do
      if grep -q "\"$MNAME\",\"$CNAME\",$W" $OUT 2>/dev/null; then echo "SKIP $MNAME/$CNAME/$W"; continue; fi
      L=$LOG_DIR/$MNAME-$CNAME-$W.log
      timeout 3600 $PPL -m $MFILE -f $CFILE --chunks 6 -c $W -t 32 > $L 2>&1 < /dev/null
      PPLV=$(grep -oE "Final estimate: PPL = [0-9.]+" $L | grep -oE "[0-9.]+$" | tail -1)
      if [ -n "$PPLV" ]; then
        printf '{"model":"%s","corpus":"%s","window":%s,"chunks":6,"ppl":%s}\n' "$MNAME" "$CNAME" "$W" "$PPLV" >> $OUT
        echo "DONE $MNAME/$CNAME/$W PPL=$PPLV"
      else
        echo "FAIL $MNAME/$CNAME/$W (see $L)"
      fi
    done
  done
done
echo LONGWINDOW_ALL_DONE
