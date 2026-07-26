#!/usr/bin/env bash
# Regression gate for the refactor: full test suite + byte-exact vs vendor.
# Every refactor Stage must pass this before commit. See REFACTOR_PLAN.md.
#   bash d_compiler/run_gate.sh
set -e
PY=/home/chokwans99/anaconda3/envs/npu-tvm/bin/python
ROOT=/home/chokwans99/NPU_cmodel
cd "$ROOT/d_compiler"
fail=0
for t in test_isa test_layout test_matmul test_rmsnorm test_swiglu test_elementwise test_tiling \
         test_runtime test_layer test_tir_backend test_import test_attention \
         test_real_layer test_decode test_v2; do
  out=$(timeout 300 $PY tests/$t.py 2>&1 | tail -1)
  case "$out" in
    *PASS*|*passed*) echo "  ok   $t" ;;
    *) echo "  FAIL $t :: $out"; fail=1 ;;
  esac
done
echo "-- byte-exact vs vendor --"
bash "$ROOT/d_compiler/validate_isa_0710.sh" 2>&1 | tail -1
echo "-- gate result: $([ $fail -eq 0 ] && echo GREEN || echo RED) --"
exit $fail
