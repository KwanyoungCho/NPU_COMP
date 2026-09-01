#!/bin/bash
# Build the native v09 codegen against the installed TVM.
#
# Out-of-tree on purpose: this links libtvm.so and registers its entry points
# through TVM's global registry, so it never needs TVM itself rebuilt.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TVM_HOME="${TVM_HOME:-$HOME/tvm-src}"
OUT="${1:-$HERE/libnpu_codegen.so}"

if [ ! -f "$TVM_HOME/build/libtvm.so" ]; then
  echo "no libtvm.so under $TVM_HOME/build -- set TVM_HOME" >&2
  exit 1
fi

# TVM builds with its own logging library; without this dmlc-core installs a
# second set of the same macros and every translation unit warns
g++ -std=c++17 -O2 -fPIC -shared -Wall \
    -DDMLC_USE_LOGGING_LIBRARY='<tvm/runtime/logging.h>' \
    -I"$HERE" \
    -I"$TVM_HOME/include" \
    -I"$TVM_HOME/3rdparty/dlpack/include" \
    -I"$TVM_HOME/3rdparty/dmlc-core/include" \
    "$HERE/codegen_v09.cc" "$HERE/v09_walker.cc" \
    -L"$TVM_HOME/build" -ltvm \
    -Wl,-rpath,"$TVM_HOME/build" \
    -o "$OUT"

echo "built $OUT"
