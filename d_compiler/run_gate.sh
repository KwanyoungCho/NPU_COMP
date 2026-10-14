#!/usr/bin/env bash
# Canonical v09 gate; use --tests to select suites.
set -euo pipefail
NPU_GATE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NPU_GATE_PYTHON="${NPU_GATE_PYTHON:-/home/chokwans99/anaconda3/envs/npu-tvm/bin/python}"
cd "$NPU_GATE_ROOT"
exec "$NPU_GATE_PYTHON" d_compiler/run_gate.py "$@"
