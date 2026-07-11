#!/usr/bin/env bash
# Validate our updated mysim.cpp (0710 ISA) byte-exact against the vendor a.out
# on every b_program example. See report/report_0710.md.
#
#   bash d_compiler/validate_isa_0710.sh
#
# Needs: 0710_npu_update/p006b.tar (extracted to _extract/), a C++17 compiler,
# and a libstdc++ with GLIBCXX_3.4.32 (conda npu-tvm env works).
set -e
ROOT=/home/chokwans99/NPU_cmodel
UPD=$ROOT/0710_npu_update
EX=$UPD/_extract
CONDA_LIB=${CONDA_LIB:-/home/chokwans99/anaconda3/envs/npu-tvm/lib}

[ -d "$EX" ] || { mkdir -p "$EX"; tar xf "$UPD/p006b.tar" -C "$EX"; }
g++ -O2 -std=c++17 "$ROOT/_poc/mysim.cpp" -o /tmp/mysim0710

pass=0; fail=0; faillist=""
for dir in "$EX"/b_program/inst_*/; do
  ex=$(basename "$dir"); [ -f "$dir/program_memory.bin" ] || continue
  T=$(mktemp -d)
  cp "$dir/program_memory.bin" "$T/"; cp "$EX/c_hex_data/G_buffer_data.bin" "$T/"
  cp "$EX/a_npu/a.out" "$T/npu"; chmod +x "$T/npu"
  ( cd "$T" && LD_LIBRARY_PATH=$CONDA_LIB timeout 20 ./npu 2>/dev/null | grep -E "PE_in_data|PE_out_array" > v.txt )
  pn=$(( $(stat -c%s "$T/program_memory.bin") / 4 ))
  ( cd "$T" && /tmp/mysim0710 --run "$pn" 2>/dev/null | grep -E "PE_in_data|PE_out_array" > o.txt )
  if diff -q "$T/v.txt" "$T/o.txt" >/dev/null 2>&1; then pass=$((pass+1)); else fail=$((fail+1)); faillist="$faillist $ex"; fi
  rm -rf "$T"
done
echo "PASS=$pass FAIL=$fail"
[ -n "$faillist" ] && { echo "FAIL:$faillist"; exit 1; } || echo "ALL b_program examples byte-exact ✓"
