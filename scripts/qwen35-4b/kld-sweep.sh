#!/bin/bash
# KLD-vs-base sweep over the arms that matter (design §0j.11), on the desktop GPU.
# Streams one arm at a time (fetch -> score -> delete) to keep the desktop's peak disk small:
# the WSL vhdx never shrinks, so no bulk copies (docs/storage.md). Peak ~35 GB.
#   bash scripts/qwen35-4b/kld-sweep.sh
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
K=${FORGE_ROOT:-~/forge}/runs/qwen35-4b/kld
G=$K/gguf
PY=${FORGE_PY:-~/forge-gpu-env/bin/python}
export FORGE_LLAMA=${FORGE_LLAMA:-~/src/llama.cpp} FORGE_NGL=${FORGE_NGL:-99} FORGE_KLD_REF=$K
ALTRA=/home/daniel/m0/runs                       # pre-migration paths; f16 GGUFs already floored there
SHARE=/mnt/models/forge/qwen35-4b                # arms with no f16 GGUF: convert from archived hf
mkdir -p $G

guard() {  # stop before eating the Windows drive under the vhdx
  local free; free=$(df -BG --output=avail /mnt/c | tail -1 | tr -dc 0-9)
  [ "$free" -ge 200 ] || { echo "C: has ${free}G free (<200G): stopping"; exit 1; }
}
score() { (cd "$REPO" && $PY -m surgery.kld score "$1" "$2"); rm -f "$2"; }

[ -f $K/ref-wikitext.kld ] && [ -f $K/ref-fineweb.kld ] && [ -f $K/ref-code.kld ] || \
  (cd "$REPO" && $PY -m surgery.kld ref $G/base-f16.gguf)

# label : path under ~/m0/runs on the Altra
for spec in unrep-2428:qwen35-4b-drop2-ranked/qwen35-4b-drop2-ranked-f16.gguf \
            unrep-2529:drop2-unrep-2529/unrep-2529-f16.gguf \
            wikirep:drop2-repaired/repaired-f16.gguf \
            codefw:drop2-codefw-repaired/codefw-f16.gguf \
            mixfull:drop2-mixfull-repaired/mixfull-f16.gguf \
            iso-all:drop2-repair-iso-all/iso-all-f16.gguf \
            2529-pass1:drop2-repair-2529/2529-f16.gguf \
            2529-fresh1e:drop2-repair-2529-fresh1e/fresh1e-f16.gguf; do
  a=${spec%%:*}; guard
  [ -f $G/$a-f16.gguf ] || rsync -t ampere:$ALTRA/${spec#*:} $G/$a-f16.gguf
  score $a $G/$a-f16.gguf
done

# label : archived arm dir on the share (hf/ -> f16 GGUF here, then both deleted)
for spec in lr3e-6:full-repair-2529/lr3e-6 lr1e-6:full-repair-2529/lr1e-6 sftrep3:drop2-repair/sftrep3; do
  a=${spec%%:*}; guard
  rsync -rt ampere:$SHARE/${spec#*:}/hf/ $G/$a-hf/
  $PY $FORGE_LLAMA/convert_hf_to_gguf.py $G/$a-hf --outtype f16 --outfile $G/$a-f16.gguf 2>&1 | tail -1
  rm -rf $G/$a-hf
  score $a $G/$a-f16.gguf
done
echo "=== SWEEP DONE $(date -Is)"
