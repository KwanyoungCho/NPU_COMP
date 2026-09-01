// v09 instruction word encoders.
//
// A direct port of npu_compiler/isa_0818.py and isa_v09.py.  The Python
// modules stay the definition of the format; this header exists because the
// codegen that calls them runs once per emitted word, and a fully unrolled
// program for a 3B model is tens of millions of words.
//
// The range checks are part of the port, not decoration: the vendor encoding
// masks silently, and a vlen that did not fit its 16-bit field once made a
// [7, 128256] copy move 45,696 elements without complaint.
#ifndef NPU_CODEGEN_V09_ISA_H_
#define NPU_CODEGEN_V09_ISA_H_

#include <cstdint>
#include <stdexcept>
#include <string>

namespace npu {

using Word = uint32_t;

// operands, operand modes, descriptor halves -- isa_0818.py
enum Operand : int { kSrc1 = 0, kSrc2 = 1, kDst = 2 };
enum Mode : int { kImm = 0, kScalar = 1, kVector = 2 };
enum Half : int { kMain = 0, kPartial = 1 };
enum Activation : int { kActOff = 0, kActReserved = 1, kActSilu = 2,
                        kActGelu = 3 };
// dtype bits live in the v09 descriptor words -- isa_v09.py
enum DType : int { kFp16 = 0, kFp32 = 1, kInt8 = 2, kInt4 = 3 };

constexpr int64_t kSramNibbles = 8LL * 1024 * 1024 * 2;  // 8 MiB
constexpr int kDmaWords = 4;

class EncodeError : public std::runtime_error {
 public:
  explicit EncodeError(const std::string& what) : std::runtime_error(what) {}
};

inline Word u16(int64_t x) { return static_cast<Word>(x) & 0xFFFFu; }

inline int64_t check(int64_t value, int bits, const char* name) {
  if (value < 0 || value >= (int64_t(1) << bits)) {
    throw EncodeError(std::string(name) + "=" + std::to_string(value) +
                      " does not fit in " + std::to_string(bits) + " bits");
  }
  return value;
}

// ---------------------------------------------------------------- control

inline Word enc_nop() { return 0; }
inline Word enc_finish() { return 0xF0; }
inline Word enc_snapshot() { return 0xF0; }
inline Word enc_halt() { return 0xFF; }

// ------------------------------------------------------------ descriptors

inline Word enc_addr_half(int operand, int64_t value, bool high, int partial) {
  Word half = high ? u16(value >> 16) : u16(value);
  return ((Word(operand) & 3u) << 30) | (Word(high ? 1 : 0) << 29) |
         ((Word(partial) & 1u) << 28) | (half << 8) | 0x80u;
}

inline Word enc_vlen(int64_t n) { return (u16(n) << 8) | 0x82u; }

// ver.08 shape words plus the v09 dtype in spare bits [26:25]
inline Word enc_mrows(int operand, int64_t rows, int partial, int dtype) {
  return ((Word(operand) & 3u) << 30) | ((Word(partial) & 1u) << 29) |
         (u16(rows) << 8) | 0x88u | ((Word(dtype) & 3u) << 25);
}

inline Word enc_mcols(int operand, int64_t cols, int partial, int dtype) {
  return ((Word(operand) & 3u) << 30) | ((Word(partial) & 1u) << 29) |
         (u16(cols) << 8) | 0x89u | ((Word(dtype) & 3u) << 25);
}

// --------------------------------------------------------- load and save

inline Word enc_load(int matrix, int operand, int strided, int ncols,
                     int start) {
  return ((Word(matrix) & 1u) << 31) | ((Word(operand) & 1u) << 30) |
         ((Word(strided) & 1u) << 29) | ((Word(ncols) & 0xFFu) << 16) |
         ((Word(start) & 0xFFu) << 8) | 0x90u;
}

inline Word enc_save(int matrix, int strided, int ncols, int start) {
  return ((Word(matrix) & 1u) << 31) | ((Word(strided) & 1u) << 29) |
         ((Word(ncols) & 0xFFu) << 16) | ((Word(start) & 0xFFu) << 8) | 0x98u;
}

// ------------------------------------------------------------ vector unit

inline Word enc_simple(Word op, int mode, int64_t imm) {
  return ((Word(mode) & 3u) << 30) | (u16(imm) << 8) | (op & 0xFFu);
}

inline Word enc_add(int mode, int64_t imm) { return enc_simple(0x01, mode, imm); }
inline Word enc_sub(int mode, int64_t imm) { return enc_simple(0x02, mode, imm); }
inline Word enc_mul(int mode, int64_t imm) { return enc_simple(0x0A, mode, imm); }
inline Word enc_div(int mode, int64_t imm) { return enc_simple(0x0B, mode, imm); }
inline Word enc_muladd(int mode, int64_t imm) { return enc_simple(0x0C, mode, imm); }
inline Word enc_move(int mode, int64_t imm) { return enc_simple(0x0D, mode, imm); }
inline Word enc_compare(int mode, int64_t imm) { return enc_simple(0x11, mode, imm); }
inline Word enc_sqrt() { return 0x0E; }
inline Word enc_exp() { return 0x0F; }
inline Word enc_reduce_sum() { return 0x14; }
inline Word enc_reduce_max() { return 0x19; }
inline Word enc_sign_inv() { return 0x16; }
inline Word enc_copy() { return 0x17; }
inline Word enc_cossin(bool is_sin) { return (Word(is_sin ? 1 : 0) << 27) | 0x18u; }

inline Word enc_minmax(bool is_max, int mode, int64_t imm) {
  return ((Word(mode) & 3u) << 30) | (Word(is_max ? 1 : 0) << 28) |
         (u16(imm) << 8) | 0x12u;
}

inline Word enc_broadcast(int mode, int64_t imm, bool high) {
  return ((Word(mode) & 3u) << 30) | (Word(high ? 1 : 0) << 29) |
         (u16(imm) << 8) | 0x15u;
}

// ------------------------------------------------------------ matrix unit

inline Word enc_matrix(Word op, int mode, int64_t imm, int activation,
                       bool mac) {
  return ((Word(mode) & 3u) << 30) | ((Word(activation) & 3u) << 28) |
         (Word(mac ? 1 : 0) << 27) | (u16(imm) << 8) | (op & 0xFFu);
}

inline Word enc_m_add(int mode, int64_t imm, int activation) {
  return enc_matrix(0x40, mode, imm, activation, false);
}

inline Word enc_m_mul(int mode, int64_t imm, int activation, bool mac) {
  return enc_matrix(0x42, mode, imm, activation, mac);
}

// -------------------------------------------------------- v09 quantization

inline Word enc_vquant() { return 0x1A; }
inline Word enc_vdequant() { return 0x1B; }

inline Word enc_scale_addr(Word which, int64_t value, bool high) {
  if (which != 0x8Au && which != 0x8Bu) {
    throw EncodeError("scale opcode must be 0x8A/0x8B");
  }
  Word half = high ? u16(value >> 16) : u16(value);
  return (Word(high ? 1 : 0) << 29) | (half << 8) | which;
}

// ------------------------------------------------------------------- DMA

// Emits the four words of one DMA instruction into `out`.
template <typename Sink>
inline void enc_dma(Word opcode, int64_t g_addr, int64_t g_stride,
                    int64_t sram_addr, int64_t rows, int64_t cols, Sink&& out) {
  check(g_addr, 32, "g_addr");
  check(g_stride, 32, "g_stride");
  if (sram_addr < 0 || sram_addr >= kSramNibbles) {
    throw EncodeError("sram_addr=" + std::to_string(sram_addr) +
                      " outside 2^24 nibbles");
  }
  if (sram_addr % 8 != 0) {
    throw EncodeError("sram_addr=" + std::to_string(sram_addr) +
                      " not 8-nibble aligned");
  }
  check(rows, 16, "rows");
  check(cols, 16, "cols");
  if (rows == 0 || cols == 0) {
    throw EncodeError("rows and cols must be nonzero");
  }
  // the 24-bit SRAM address rides in the opcode word's reserved field
  out(Word(sram_addr) << 8 | opcode);
  out(Word(g_addr));
  out(Word(g_stride));
  out((Word(rows) << 16) | Word(cols));
}

constexpr Word kOpGload = 0xA0;
constexpr Word kOpGstore = 0xA8;

}  // namespace npu

#endif  // NPU_CODEGEN_V09_ISA_H_
