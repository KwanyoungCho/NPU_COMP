// TVM entry points for the native v09 codegen.
//
// The Python walker in npu_compiler/tir_codegen_v09.py stays the definition of
// what a scheduled kernel lowers to; this module is the same lowering written
// where it does not have to cross the FFI once per emitted word.  Every entry
// point here is gated by a bit-exactness test against its Python counterpart,
// so the two can never drift silently.
//
// Words leave as a uint32 NDArray: a program for a 3B model is tens of
// millions of them, and boxing each one as an Integer would put back the
// per-word crossing this exists to remove.
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <cstring>
#include <vector>

#include "v09_asm.h"

namespace npu {
namespace {

using tvm::runtime::NDArray;

NDArray ToNDArray(const std::vector<Word>& words) {
  NDArray array = NDArray::Empty({static_cast<int64_t>(words.size())},
                                 DLDataType{kDLUInt, 32, 1},
                                 DLDevice{kDLCPU, 0});
  if (!words.empty()) {
    std::memcpy(static_cast<char*>(array->data) + array->byte_offset,
                words.data(), words.size() * sizeof(Word));
  }
  return array;
}

// Exercises every encoder and emitter method with arguments chosen to put a
// one in each field that matters -- both descriptor halves, both operands,
// dtypes other than the default, a DMA whose row count exceeds the field, an
// odd byte length, a strided region.  The Python mirror in
// tests/test_native_codegen.py runs the identical script through the Python
// encoders; the two word streams must be identical.
std::vector<Word> EncodeSelfTest() {
  Asm a;
  SramEmitter stage(&a);

  a.nop();
  a.vlen(0xFFFF);
  a.addr(kSrc1, 0x12345678, kPartial);
  a.addr(kSrc2, 0x0000ABCD, kMain);
  a.addr(kDst, 0, kPartial);
  a.shape(kSrc1, 64, 3072, kMain, kFp16);
  a.shape(kSrc2, 7, 64, kPartial, kInt8);
  a.shape(kDst, 1, 9728, kPartial, kFp32);

  a.load(0, kSrc1, 0, 0, 0);
  a.load(1, kSrc2, 1, 63, 5);
  a.save(0, 0, 0, 0);
  a.save(1, 1, 64, 7);

  a.v_add(kVector);
  a.v_sub(kImm, 0x1234);
  a.v_mul(kScalar, 7);
  a.v_div(kImm, 127);
  a.v_max(kVector);
  a.v_min(kImm, 3);
  a.v_sqrt();
  a.v_exp();
  a.v_cos();
  a.v_sin();
  a.v_sign_inv();
  a.v_copy();
  a.v_reduce_sum();
  a.v_reduce_max();
  a.v_broadcast_addr(0xDEADBEEF);

  a.m_mul(kVector, 0, kActOff, false);
  a.m_mul(kVector, 0, kActOff, true);
  a.m_mul(kImm, 9, kActSilu, true);

  a.vquant();
  a.vdequant();
  a.ascale(0x1FFFF8);
  a.wscale(8);

  // emitter paths: descriptors, then the three DMA shapes
  stage.vector(kSrc1, 4096);
  stage.broadcast(2048);
  stage.region(kSrc2, 8192, 3072, 64, 64);
  stage.strided(kDst, 512, 128, 64);
  stage.dma_in(0, 0, 4096);
  stage.dma_out(4 * 1024, 8 * 1024, 260);          // odd tail, cell-rounded
  stage.dma_2d(64, 6144, 16384, 64, 128, true);    // one 2D transfer
  stage.dma_2d(0, 8, 0, 70000, 4, false);          // row count over the field
  a.halt();
  return a.words;
}

TVM_REGISTER_GLOBAL("npu.encode_selftest").set_body_typed([]() -> NDArray {
  return ToNDArray(EncodeSelfTest());
});

TVM_REGISTER_GLOBAL("npu.native_available").set_body_typed([]() -> bool {
  return true;
});

}  // namespace
}  // namespace npu
