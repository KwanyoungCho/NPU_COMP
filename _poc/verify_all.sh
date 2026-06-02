#!/bin/bash
set -u
ROOT=/home/chokwans99/NPU_cmodel; POC=$ROOT/_poc
g++ -O2 -std=c++17 "$POC/mysim.cpp" -o "$POC/mysim" || { echo BUILD_FAIL; exit 1; }
WORK=$(mktemp -d); cp "$ROOT/a_npu/G_buffer_data.bin" "$WORK/"; cp "$POC/mysim" "$WORK/"
cd "$WORK"
pass=0; fail=0; fails=""
for g in "$POC"/goldens/*.txt; do
  name=$(basename "$g" .txt)
  cp "$ROOT/b_program/$name/program_memory.bin" "$WORK/program_memory.bin" 2>/dev/null || continue
  ./mysim > mine.txt 2>/dev/null
  L=$(wc -l < mine.txt)
  if diff -q <(head -$L "$g") mine.txt >/dev/null; then pass=$((pass+1));
  else fail=$((fail+1)); fails="$fails $name"; fi
done
echo "PASS=$pass FAIL=$fail"
[ -n "$fails" ] && echo "FAILED:$fails"
cd /; rm -rf "$WORK"
