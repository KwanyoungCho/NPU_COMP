// The v09 assembler and its SRAM/DMA emitter.
//
// Ports npu_compiler/isa_0818.py's ``Asm``, backend_v09.py's ``V09Asm`` and
// tir_codegen_v09.py's ``SramEmitter``.  Words accumulate in a vector the
// caller drains once, rather than crossing a language boundary per word.
//
// Descriptor state is deliberately NOT tracked here.  Every emission writes
// the descriptors it needs, and the peephole pass removes the writes that
// turn out to be redundant -- the straight-line stream makes that decidable
// exactly, which is why it is safe to keep this side stateless.
#ifndef NPU_CODEGEN_V09_ASM_H_
#define NPU_CODEGEN_V09_ASM_H_

#include <cstdint>
#include <vector>

#include "v09_isa.h"

namespace npu {

class Asm {
 public:
  std::vector<Word> words;

  void emit(Word word) { words.push_back(word); }

  // -- control
  void nop() { emit(enc_nop()); }
  void halt() { emit(enc_halt()); }
  void snapshot() { emit(enc_snapshot()); }

  // -- descriptors
  void addr(int operand, int64_t value, int partial) {
    emit(enc_addr_half(operand, value, false, partial));
    emit(enc_addr_half(operand, value, true, partial));
  }

  void shape(int operand, int64_t rows, int64_t cols, int partial,
             int dtype = kFp16) {
    emit(enc_mrows(operand, rows, partial, dtype));
    emit(enc_mcols(operand, cols, partial, dtype));
  }

  void vlen(int64_t n) {
    // the field is 16-bit and the encoder masks silently; a longer vector
    // would quietly process only the low bits' worth of lanes
    if (n <= 0 || n > 0xFFFF) {
      throw EncodeError("vlen=" + std::to_string(n) +
                        " does not fit in 16 bits");
    }
    emit(enc_vlen(n));
  }

  // -- transfers between SRAM and the unit registers
  void load(int matrix, int operand, int strided = 0, int ncols = 0,
            int start = 0) {
    emit(enc_load(matrix, operand, strided, ncols, start));
  }
  void save(int matrix, int strided = 0, int ncols = 0, int start = 0) {
    emit(enc_save(matrix, strided, ncols, start));
  }

  // -- vector unit
  void v_add(int mode, int64_t imm = 0) { emit(enc_add(mode, imm)); }
  void v_sub(int mode, int64_t imm = 0) { emit(enc_sub(mode, imm)); }
  void v_mul(int mode, int64_t imm = 0) { emit(enc_mul(mode, imm)); }
  void v_div(int mode, int64_t imm = 0) { emit(enc_div(mode, imm)); }
  void v_max(int mode, int64_t imm = 0) { emit(enc_minmax(true, mode, imm)); }
  void v_min(int mode, int64_t imm = 0) { emit(enc_minmax(false, mode, imm)); }
  void v_sqrt() { emit(enc_sqrt()); }
  void v_exp() { emit(enc_exp()); }
  void v_cos() { emit(enc_cossin(false)); }
  void v_sin() { emit(enc_cossin(true)); }
  void v_sign_inv() { emit(enc_sign_inv()); }
  void v_copy() { emit(enc_copy()); }
  void v_reduce_sum() { emit(enc_reduce_sum()); }
  void v_reduce_max() { emit(enc_reduce_max()); }

  // 0x15 updates one address half and also executes, so the low half's result
  // is meant to be overwritten by the high half's; both are always emitted so
  // stale high state cannot survive into a long program
  void v_broadcast_addr(int64_t address) {
    emit(enc_broadcast(kScalar, address, false));
    emit(enc_broadcast(kScalar, address >> 16, true));
  }

  // -- matrix unit
  void m_add(int mode, int64_t imm = 0, int activation = kActOff) {
    emit(enc_m_add(mode, imm, activation));
  }
  void m_mul(int mode, int64_t imm = 0, int activation = kActOff,
             bool mac = false) {
    emit(enc_m_mul(mode, imm, activation, mac));
  }

  // -- v09 quantization
  void vquant() { emit(enc_vquant()); }
  void vdequant() { emit(enc_vdequant()); }
  void ascale(int64_t nibble) {
    emit(enc_scale_addr(0x8A, nibble, false));
    emit(enc_scale_addr(0x8A, nibble, true));
  }
  void wscale(int64_t nibble) {
    emit(enc_scale_addr(0x8B, nibble, false));
    emit(enc_scale_addr(0x8B, nibble, true));
  }

  // -- DMA
  void gload(int64_t g_addr, int64_t g_stride, int64_t sram, int64_t rows,
             int64_t cols) {
    enc_dma(kOpGload, g_addr, g_stride, sram, rows, cols,
            [this](Word w) { emit(w); });
  }
  void gstore(int64_t g_addr, int64_t g_stride, int64_t sram, int64_t rows,
              int64_t cols) {
    enc_dma(kOpGstore, g_addr, g_stride, sram, rows, cols,
            [this](Word w) { emit(w); });
  }
};

// Descriptor and DMA emission for the SRAM-staged schedule.  Addresses arrive
// in the unit each side counts in: SRAM operands as nibbles, global operands
// as byte offsets, which the DMA turns into 32-bit cell addresses.
class SramEmitter {
 public:
  explicit SramEmitter(Asm* assembler) : a_(assembler) {}

  // rows and cols are 16-bit instruction fields; anything larger is split
  static constexpr int64_t kMaxCells = 0xFFFF;

  void vector(int operand, int64_t nibble) { a_->addr(operand, nibble, kPartial); }

  void broadcast(int64_t nibble) { a_->v_broadcast_addr(nibble); }

  void region(int operand, int64_t nibble, int64_t stride, int64_t rows,
              int64_t cols) {
    a_->addr(operand, nibble, kMain);
    a_->shape(operand, rows, stride, kMain);
    a_->addr(operand, nibble, kPartial);
    a_->shape(operand, rows, cols, kPartial);
  }

  void strided(int operand, int64_t nibble, int64_t stride, int64_t count) {
    a_->addr(operand, nibble, kMain);
    a_->shape(operand, count, stride, kMain);
    a_->addr(operand, nibble, kPartial);
    a_->shape(operand, count, 1, kPartial);
  }

  void dma_in(int64_t global_byte, int64_t sram_nibble, int64_t nbytes) {
    dma_1d(true, global_byte, sram_nibble, nbytes);
  }

  void dma_out(int64_t global_byte, int64_t sram_nibble, int64_t nbytes) {
    dma_1d(false, global_byte, sram_nibble, nbytes);
  }

  // ``rows`` rows of ``nbytes`` bytes spaced ``global_stride`` bytes apart in
  // global memory, to or from a packed SRAM block.
  void dma_2d(int64_t global_byte, int64_t global_stride, int64_t sram_nibble,
              int64_t rows, int64_t nbytes, bool to_sram) {
    if (nbytes / 4 > kMaxCells) {          // a row alone overflows the field
      for (int64_t r = 0; r < rows; ++r) {
        dma_1d(to_sram, global_byte, sram_nibble, nbytes);
        global_byte += global_stride;
        sram_nibble += nbytes * 2;
      }
      return;
    }
    while (rows > 0) {
      int64_t batch = rows < kMaxCells ? rows : kMaxCells;
      int64_t cells = check_cells(global_byte, nbytes, sram_nibble);
      if (to_sram) {
        a_->gload(global_byte / 4, global_stride / 4, sram_nibble, batch, cells);
      } else {
        a_->gstore(global_byte / 4, global_stride / 4, sram_nibble, batch, cells);
      }
      global_byte += batch * global_stride;
      sram_nibble += batch * nbytes * 2;
      rows -= batch;
    }
  }

 private:
  void dma_1d(bool to_sram, int64_t global_byte, int64_t sram_nibble,
              int64_t nbytes) {
    while (nbytes > 0) {
      int64_t piece = nbytes < kMaxCells * 4 ? nbytes : kMaxCells * 4;
      int64_t cells = check_cells(global_byte, piece, sram_nibble);
      if (to_sram) {
        a_->gload(global_byte / 4, cells, sram_nibble, 1, cells);
      } else {
        a_->gstore(global_byte / 4, cells, sram_nibble, 1, cells);
      }
      global_byte += piece;
      sram_nibble += piece * 2;
      nbytes -= piece;
    }
  }

  // allocations are cell-rounded, so a short tail only reaches into the
  // tensor's own padding; an unaligned start would shift the SRAM image and
  // is rejected instead
  static int64_t check_cells(int64_t global_byte, int64_t nbytes,
                             int64_t sram_nibble) {
    if (global_byte % 4 != 0) {
      throw EncodeError("DMA row must start on a 32-bit cell");
    }
    if (sram_nibble % 8 != 0) {
      throw EncodeError("DMA SRAM address must be 8-nibble aligned");
    }
    return (nbytes + 3) / 4;
  }

  Asm* a_;
};

}  // namespace npu

#endif  // NPU_CODEGEN_V09_ASM_H_
