#!/bin/bash
# Capture golden traces for every b_program/inst_* example using the canonical
# reference simulator (a_npu/a.out) + the default ramp G-buffer.
set -u
ROOT=/home/chokwans99/NPU_cmodel
POC=$ROOT/_poc
REF=$ROOT/a_npu/a.out
GBUF=$ROOT/a_npu/G_buffer_data.bin
LIB=/home/chokwans99/anaconda3/envs/ssd/lib
mkdir -p "$POC/goldens"
WORK=$(mktemp -d)
cp "$REF" "$WORK/ref_sim"; cp "$GBUF" "$WORK/G_buffer_data.bin"
cd "$WORK"
n=0
for d in "$ROOT"/b_program/inst_*/; do
  name=$(basename "$d")
  [ -f "$d/program_memory.bin" ] || continue
  cp "$d/program_memory.bin" "$WORK/program_memory.bin"
  timeout 4 script -qec "env LD_LIBRARY_PATH=$LIB ./ref_sim" /dev/null 2>/dev/null \
    | tr -d '\r' | head -350 > "$POC/goldens/$name.txt"
  l=$(wc -l < "$POC/goldens/$name.txt")
  echo "$name : $l lines"
  n=$((n+1))
done
echo "captured $n goldens"
cd /; rm -rf "$WORK"
