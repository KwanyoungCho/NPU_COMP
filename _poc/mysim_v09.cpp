// ISA ver.09 C-model (our own next-generation design; not vendor parity).
//
// Machine model per d_compiler/ISA_V09.md:
//   * Global memory: 32-bit addresses in 32-bit cells (up to 16 GiB), DMA-only
//     storage with no dtype semantics.  Actual size = input file size.
//   * SRAM: 8 MiB shared scratchpad addressed in 4-bit nibbles (24 effective
//     bits).  All compute units read/write SRAM only.  Zero-initialized.
//   * Cell<->nibble mapping is little-endian: cell bits [4k+3:4k] are SRAM
//     nibble base+k, matching host little-endian files byte for byte.
//   * ver.08 instruction encodings are retained: 0x80 addresses are SRAM
//     nibble addresses; rows/cols/stride/vlen keep their ver.08 element-count
//     values; each operand descriptor carries a 2-bit dtype in the spare bits
//     [26:25] of 0x88/0x89 (00=FP16 01=FP32 10=INT8 11=INT4, so an untouched
//     ver.08 program runs as dtype FP16).
//   * Arithmetic contract: FP16 storage / FP32 compute / RNE on FP16 store.
//     Feeders do lossless format conversion only; scale restoration (dequant)
//     happens once at the matrix drain.  ver.08 fixes: seeded reduce-max,
//     signed int16 immediates, standard tanh-GELU on activation code 1.
//   * Out-of-range or misaligned access is a hard error (no silent corruption).
//   * HALT (0xFF) is the only normal termination and appends the global image
//     to the output file; SNAPSHOT (0xF0) appends a mid-run checkpoint.
//     Falling off the end of the program without HALT is an error.
//
// Files (cwd): global_memory.bin (in), program_memory.bin (in),
//              saved_global_memory.bin (out), perf_counters.txt (out).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kSramBytes = 8u << 20;            // 8 MiB
constexpr std::size_t kSramNibbles = kSramBytes * 2;    // 2^24
constexpr std::uint32_t kMaxVlen = 256;                 // vector unit lanes

enum Dtype : unsigned { FP16 = 0, FP32 = 1, INT8 = 2, INT4 = 3 };

unsigned dtype_width_nibbles(unsigned dtype) {
    static const unsigned widths[4] = {4, 8, 2, 1};
    return widths[dtype & 3];
}

float half_to_float(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    std::uint32_t exp = (h >> 10) & 0x1fu;
    std::uint32_t mantissa = h & 0x3ffu;
    std::uint32_t bits;
    if (exp == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int shift = -1;
            do {
                ++shift;
                mantissa <<= 1;
            } while ((mantissa & 0x400u) == 0);
            mantissa &= 0x3ffu;
            bits = sign | ((127 - 15 - shift) << 23) | (mantissa << 13);
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mantissa << 13);
    }
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

std::uint16_t float_to_half(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    int exp = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
    std::uint32_t mantissa = bits & 0x7fffffu;
    if (((bits >> 23) & 0xffu) == 0xffu) {
        return static_cast<std::uint16_t>(sign | 0x7c00u | (mantissa ? 0x200u : 0));
    }
    if (exp >= 0x1f) {
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    }
    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<std::uint16_t>(sign);
        }
        mantissa |= 0x800000u;
        std::uint32_t rounded = mantissa >> (14 - exp);
        const std::uint32_t remainder = mantissa & ((1u << (14 - exp)) - 1);
        const std::uint32_t halfway = 1u << (13 - exp);
        if (remainder > halfway || (remainder == halfway && (rounded & 1u))) {
            ++rounded;
        }
        return static_cast<std::uint16_t>(sign | rounded);
    }
    std::uint32_t rounded = mantissa >> 13;
    const std::uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (rounded & 1u))) {
        ++rounded;
    }
    std::uint32_t result = (static_cast<std::uint32_t>(exp) << 10) | rounded;
    if (rounded == 0x400u) {
        result = static_cast<std::uint32_t>(exp + 1) << 10;
    }
    return static_cast<std::uint16_t>(sign | result);
}

float sigmoid(float value) { return 1.0f / (1.0f + std::exp(-value)); }

// Activation codes ([29:28] of matrix ops): 0=off, 1=standard tanh-GELU (v09
// fix; ver.08 aliased this to its own GELU), 2=SiLU, 3=legacy vendor "GELU"
// x*sigmoid(2x) kept for compatibility experiments.
float activate(float value, unsigned activation) {
    if (activation == 1) {
        const float inner = 0.7978845608028654f *
                            (value + 0.044715f * value * value * value);
        // tanh evaluated in double so the reference (float64 tanh, same libm)
        // reproduces the result bit-exactly; single rounding back to float.
        const float t = static_cast<float>(std::tanh(static_cast<double>(inner)));
        return 0.5f * value * (1.0f + t);
    }
    if (activation == 2) {
        return value * sigmoid(value);
    }
    if (activation == 3) {
        return value * sigmoid(2.0f * value);
    }
    return value;
}

struct Descriptor {
    std::uint32_t main_address = 0;      // SRAM nibbles
    std::uint32_t partial_address = 0;   // SRAM nibbles
    std::uint32_t main_rows = 0;         // element counts
    std::uint32_t main_cols = 0;
    std::uint32_t partial_rows = 0;
    std::uint32_t partial_cols = 0;
    unsigned dtype = FP16;               // spare bits [26:25] of 0x88/0x89
};

struct Counters {
    std::uint64_t words_executed = 0;
    std::uint64_t nop = 0;
    std::uint64_t snapshot = 0;
    std::uint64_t halt = 0;
    std::uint64_t gload = 0;
    std::uint64_t gstore = 0;
    std::uint64_t dma_cells_loaded = 0;
    std::uint64_t dma_cells_stored = 0;
    std::uint64_t loads = 0;
    std::uint64_t saves = 0;
    std::uint64_t vector_ops = 0;
    std::uint64_t matrix_ops = 0;
    std::uint64_t vquant = 0;
    std::uint64_t vdequant = 0;

    void dump(const char* path) const {
        std::ofstream file(path);
        file << "words_executed " << words_executed << '\n'
             << "nop " << nop << '\n'
             << "snapshot " << snapshot << '\n'
             << "halt " << halt << '\n'
             << "gload " << gload << '\n'
             << "gstore " << gstore << '\n'
             << "dma_cells_loaded " << dma_cells_loaded << '\n'
             << "dma_cells_stored " << dma_cells_stored << '\n'
             << "loads " << loads << '\n'
             << "saves " << saves << '\n'
             << "vector_ops " << vector_ops << '\n'
             << "matrix_ops " << matrix_ops << '\n'
             << "vquant " << vquant << '\n'
             << "vdequant " << vdequant << '\n';
    }
};

class Machine {
public:
    int run() {
        if (!read_global() || !read_program()) {
            return 1;
        }
        sram_.assign(kSramBytes, 0);
        std::remove("saved_global_memory.bin");
        for (pc_ = 0; pc_ < program_.size(); ++pc_) {
            const std::uint32_t word = program_[pc_];
            ++counters_.words_executed;
            execute(word);
            if (halted_) {
                return finish(0);
            }
        }
        fail("program ended without HALT (0xFF)");
        return 2;  // unreachable; fail() exits
    }

private:
    std::vector<std::uint32_t> global_;
    std::vector<std::uint8_t> sram_;
    std::vector<std::uint32_t> program_;
    std::size_t pc_ = 0;
    bool halted_ = false;
    Counters counters_;

    Descriptor desc_[3];
    std::uint32_t vector_length_ = 0;
    std::uint32_t broadcast_address_ = 0;   // SRAM nibbles
    std::uint32_t a_scale_address_ = 0;     // SRAM nibbles, FP32 vector
    std::uint32_t w_scale_address_ = 0;     // SRAM nibbles, FP32 vector
    std::vector<float> input1_;
    std::vector<float> input2_;
    unsigned in_dtype_[2] = {FP16, FP16};
    std::vector<float> output_;
    std::vector<long long> iacc_;           // INT32-model matmul accumulator
    bool int_result_ = false;               // last matmul used integer path
    bool result_src0_int_ = false;          // apply a_scale at drain
    bool result_src1_int_ = false;          // apply w_scale at drain
    unsigned pending_activation_ = 0;       // int path: activation after dequant
    std::uint32_t output_rows_ = 0;
    std::uint32_t output_cols_ = 0;
    std::vector<float> drain_acc_;          // FP32 dequant accumulator
    float reduce_carry_ = 0.0f;             // FP32 chunk carry for reduces

    int finish(int code) {
        counters_.dump("perf_counters.txt");
        return code;
    }

    void fail(const std::string& message) {
        std::cerr << "v09 error @pc=" << pc_;
        if (pc_ < program_.size()) {
            char buffer[16];
            std::snprintf(buffer, sizeof(buffer), "%08x", program_[pc_]);
            std::cerr << " word=0x" << buffer;
        }
        std::cerr << ": " << message << '\n';
        finish(2);
        std::exit(2);
    }

    void require(bool condition, const char* message) {
        if (!condition) {
            fail(message);
        }
    }

    bool read_global() {
        std::ifstream file("global_memory.bin", std::ios::binary);
        if (!file) {
            std::cerr << "cannot open global_memory.bin\n";
            return false;
        }
        file.seekg(0, std::ios::end);
        const std::size_t bytes = static_cast<std::size_t>(file.tellg());
        file.seekg(0);
        if (bytes % 4 != 0) {
            std::cerr << "global_memory.bin size " << bytes
                      << " is not a multiple of 4 (32-bit cells)\n";
            return false;
        }
        global_.resize(bytes / 4);
        if (!global_.empty()) {
            file.read(reinterpret_cast<char*>(global_.data()), bytes);
        }
        return true;
    }

    bool read_program() {
        std::ifstream file("program_memory.bin", std::ios::binary);
        if (!file) {
            std::cerr << "cannot open program_memory.bin\n";
            return false;
        }
        file.seekg(0, std::ios::end);
        const std::size_t bytes = static_cast<std::size_t>(file.tellg());
        file.seekg(0);
        if (bytes % 4 != 0) {
            std::cerr << "program_memory.bin size " << bytes
                      << " is not a multiple of 4\n";
            return false;
        }
        program_.resize(bytes / 4);
        if (!program_.empty()) {
            file.read(reinterpret_cast<char*>(program_.data()), bytes);
        }
        return true;
    }

    // ------------------------------------------------------------- SRAM

    void check_span(std::uint64_t nibble, std::uint64_t width, const char* who) {
        if (nibble + width > kSramNibbles) {
            fail(std::string(who) + ": SRAM access out of bounds");
        }
    }

    float read_elem(unsigned dtype, std::uint64_t nibble) {
        const unsigned width = dtype_width_nibbles(dtype);
        check_span(nibble, width, "read");
        switch (dtype) {
            case FP16: {
                std::uint16_t bits;
                std::memcpy(&bits, sram_.data() + nibble / 2, 2);
                return half_to_float(bits);
            }
            case FP32: {
                float value;
                std::memcpy(&value, sram_.data() + nibble / 2, 4);
                return value;
            }
            case INT8:
                return static_cast<float>(
                    static_cast<std::int8_t>(sram_[nibble / 2]));
            default: {  // INT4
                const std::uint8_t byte = sram_[nibble / 2];
                const unsigned raw = (nibble % 2) ? (byte >> 4) : (byte & 0xF);
                const int value = raw >= 8 ? static_cast<int>(raw) - 16
                                           : static_cast<int>(raw);
                return static_cast<float>(value);
            }
        }
    }

    void write_fp16(std::uint64_t nibble, float value) {
        check_span(nibble, 4, "write");
        const std::uint16_t bits = float_to_half(value);
        std::memcpy(sram_.data() + nibble / 2, &bits, 2);
    }

    void write_fp32(std::uint64_t nibble, float value) {
        check_span(nibble, 8, "write");
        std::memcpy(sram_.data() + nibble / 2, &value, 4);
    }

    void write_packed_int(std::uint64_t nibble, int value, bool int4) {
        check_span(nibble, int4 ? 1 : 2, "write");
        if (int4) {
            std::uint8_t& byte = sram_[nibble / 2];
            const std::uint8_t raw = static_cast<std::uint8_t>(value & 0xF);
            byte = (nibble % 2) ? ((byte & 0x0F) | (raw << 4))
                                : ((byte & 0xF0) | raw);
        } else {
            sram_[nibble / 2] =
                static_cast<std::uint8_t>(static_cast<std::int8_t>(value));
        }
    }

    void check_alignment(std::uint32_t nibble, unsigned dtype, const char* who) {
        if (nibble % dtype_width_nibbles(dtype) != 0) {
            fail(std::string(who) + ": SRAM address not aligned to dtype width");
        }
    }

    float read_scale(std::uint32_t base, std::uint32_t index, const char* who) {
        check_alignment(base, FP32, who);
        return read_elem(FP32, static_cast<std::uint64_t>(base) + 8ull * index);
    }

    // ------------------------------------------------------------- DMA

    // GLOAD (0xA0) / GSTORE (0xA8), 4 words (ISA_V09.md section 3):
    //   w0 [31:8] SRAM nibble address (24-bit, 8-nibble aligned) | [7:0] opcode
    //   w1 global cell address | w2 global row stride (cells)
    //   w3 rows[31:16] cols[15:0]  (cols in cells)
    // The SRAM address is exactly 24 bits wide, so it rides in the opcode
    // word's reserved field -- no separate address word, and the address
    // travels atomically with the instruction.
    // dtype-blind synchronous copy; SRAM rows land contiguously
    // (row r occupies nibbles addr + r*cols*8 .. +cols*8).
    void dma(bool store) {
        if (pc_ + 3 >= program_.size()) {
            fail("truncated DMA instruction (needs 4 words)");
        }
        const std::uint32_t sram_addr = program_[pc_] >> 8;
        const std::uint64_t g_addr = program_[pc_ + 1];
        const std::uint64_t g_stride = program_[pc_ + 2];
        const std::uint32_t rows = program_[pc_ + 3] >> 16;
        const std::uint32_t cols = program_[pc_ + 3] & 0xffffu;
        pc_ += 3;
        if (sram_addr % 8 != 0) {
            fail("DMA SRAM address not 8-nibble aligned");
        }
        if (rows == 0 || cols == 0) {
            fail("DMA with zero rows or cols");
        }
        const std::uint64_t sram_end =
            static_cast<std::uint64_t>(sram_addr) +
            static_cast<std::uint64_t>(rows) * cols * 8;
        if (sram_end > kSramNibbles) {
            fail("DMA SRAM range out of bounds");
        }
        const std::uint64_t g_end =
            g_addr + static_cast<std::uint64_t>(rows - 1) * g_stride + cols;
        if (g_end > global_.size()) {
            fail("DMA global range out of bounds");
        }
        for (std::uint32_t row = 0; row < rows; ++row) {
            std::uint32_t* g = global_.data() + g_addr + row * g_stride;
            std::uint8_t* s = sram_.data() + (static_cast<std::size_t>(sram_addr) / 2) +
                              static_cast<std::size_t>(row) * cols * 4;
            if (store) {
                std::memcpy(g, s, static_cast<std::size_t>(cols) * 4);
            } else {
                std::memcpy(s, g, static_cast<std::size_t>(cols) * 4);
            }
        }
        const std::uint64_t cells = static_cast<std::uint64_t>(rows) * cols;
        (store ? counters_.dma_cells_stored : counters_.dma_cells_loaded) += cells;
    }

    // ------------------------------------------------------------- load/save

    static std::uint32_t replace_half(std::uint32_t old, std::uint16_t value,
                                      bool high) {
        return high ? ((old & 0x0000ffffu) | (static_cast<std::uint32_t>(value) << 16))
                    : ((old & 0xffff0000u) | value);
    }

    std::vector<float>& input(unsigned operand) {
        return operand == 0 ? input1_ : input2_;
    }

    void check_vlen(const char* who) {
        if (vector_length_ == 0 || vector_length_ > kMaxVlen) {
            fail(std::string(who) + ": vlen must be in [1, 256]");
        }
    }

    // ver.08 load semantics over SRAM: access uses the PARTIAL address and the
    // MAIN cols stride, all scaled by the operand dtype width in nibbles.
    void load(unsigned matrix, unsigned operand, bool strided, unsigned ncols,
              unsigned start) {
        ++counters_.loads;
        const Descriptor& d = desc_[operand];
        const unsigned dtype = d.dtype;
        const unsigned width = dtype_width_nibbles(dtype);
        std::vector<float>& target = input(operand);
        target.clear();
        in_dtype_[operand] = dtype;
        check_alignment(d.partial_address, dtype, "load");
        const std::uint64_t base = d.partial_address;
        if (!matrix) {
            check_vlen("vector load");
            require(dtype == FP16,
                    "vector load requires FP16 (INT data enters via VDEQUANT)");
            target.reserve(vector_length_);
            for (std::uint32_t i = 0; i < vector_length_; ++i) {
                target.push_back(read_elem(dtype, base + std::uint64_t(i) * width));
            }
        } else {
            require(dtype != FP32, "matrix operands cannot be FP32");
            if (strided) {
                target.reserve(static_cast<std::size_t>(d.partial_rows) * ncols);
                for (unsigned col = start; col < start + ncols; ++col) {
                    for (std::uint32_t row = 0; row < d.partial_rows; ++row) {
                        const std::uint64_t index =
                            std::uint64_t(row) * d.main_cols + col;
                        target.push_back(read_elem(dtype, base + index * width));
                    }
                }
            } else {
                target.reserve(static_cast<std::size_t>(d.partial_rows) *
                               d.partial_cols);
                for (std::uint32_t row = 0; row < d.partial_rows; ++row) {
                    for (std::uint32_t col = 0; col < d.partial_cols; ++col) {
                        const std::uint64_t index =
                            std::uint64_t(row) * d.main_cols + col;
                        target.push_back(read_elem(dtype, base + index * width));
                    }
                }
            }
        }
    }

    // Save: vector form writes the previous result's length (V3-006 fix) as
    // FP16, or raw FP32 with flag [25] (scale production; inert in FP16 mode).
    // Matrix form is the drain: dequant by the scale vectors when the matmul
    // consumed INT operands, with [27]=carry-in / [26]=hold chaining partial
    // results through the FP32 drain accumulator (00 = plain ver.08 store).
    void save(std::uint32_t instruction) {
        ++counters_.saves;
        const bool matrix = (instruction >> 31) & 1u;
        const bool strided = (instruction >> 29) & 1u;
        const unsigned ncols = (instruction >> 16) & 0xffu;
        const unsigned start = (instruction >> 8) & 0xffu;
        const Descriptor& d = desc_[2];
        if (!matrix) {
            const bool fp32_store = (instruction >> 25) & 1u;
            const unsigned dtype = fp32_store ? FP32 : FP16;
            const unsigned width = dtype_width_nibbles(dtype);
            check_alignment(d.partial_address, dtype, "save");
            for (std::size_t i = 0; i < output_.size(); ++i) {
                const std::uint64_t nibble =
                    std::uint64_t(d.partial_address) + std::uint64_t(i) * width;
                if (fp32_store) {
                    write_fp32(nibble, output_[i]);
                } else {
                    write_fp16(nibble, output_[i]);
                }
            }
            return;
        }
        const bool carry_in = (instruction >> 27) & 1u;
        const bool hold = (instruction >> 26) & 1u;
        const bool dequant = int_result_ || result_src0_int_ || result_src1_int_;
        check_alignment(d.partial_address, FP16, "save");
        if (strided) {
            require(!dequant && !carry_in && !hold,
                    "strided matrix save does not support dequant/carry");
            const std::uint32_t rows =
                output_rows_ ? output_rows_ : desc_[0].partial_rows;
            for (unsigned col = 0; col < ncols; ++col) {
                for (std::uint32_t row = 0; row < rows; ++row) {
                    const std::size_t source =
                        static_cast<std::size_t>(col) * rows + row;
                    if (source < output_.size()) {
                        const std::uint64_t index =
                            std::uint64_t(row) * d.main_cols + start + col;
                        write_fp16(std::uint64_t(d.partial_address) + index * 4,
                                   output_[source]);
                    }
                }
            }
            return;
        }
        const std::uint32_t rows = output_rows_ ? output_rows_ : d.partial_rows;
        const std::uint32_t cols = output_cols_ ? output_cols_ : d.partial_cols;
        const std::uint32_t stride = d.main_cols ? d.main_cols : cols;
        const std::size_t total = static_cast<std::size_t>(rows) * cols;
        if (!carry_in) {
            drain_acc_.assign(total, 0.0f);
        } else {
            require(drain_acc_.size() == total,
                    "drain carry-in with mismatched tile shape");
        }
        for (std::uint32_t row = 0; row < rows; ++row) {
            for (std::uint32_t col = 0; col < cols; ++col) {
                const std::size_t source = static_cast<std::size_t>(row) * cols + col;
                if (source >= (int_result_ ? iacc_.size() : output_.size())) {
                    continue;
                }
                float value = int_result_
                                  ? static_cast<float>(iacc_[source])
                                  : output_[source];
                if (result_src1_int_) {
                    value *= read_scale(w_scale_address_, col, "w_scale");
                }
                if (result_src0_int_) {
                    value *= read_scale(a_scale_address_, row, "a_scale");
                }
                if (dequant && pending_activation_) {
                    value = activate(value, pending_activation_);
                }
                if (carry_in) {
                    // adding an unconditional 0.0f would turn -0.0 into +0.0
                    // and break FP16-mode bit-exactness
                    value += drain_acc_[source];
                }
                if (hold) {
                    drain_acc_[source] = value;
                } else {
                    const std::uint64_t index = std::uint64_t(row) * stride + col;
                    write_fp16(std::uint64_t(d.partial_address) + index * 4, value);
                }
            }
        }
    }

    // ------------------------------------------------------------- compute

    // Signed int16 immediate (v09 fix of the ver.08 uint32 reinterpretation).
    float rhs(unsigned mode, std::uint16_t immediate, std::size_t index) {
        if (mode == 2) {
            return index < input2_.size() ? input2_[index] : 0.0f;
        }
        if (mode == 1) {
            check_alignment(immediate, FP16, "scalar operand");
            return read_elem(FP16, immediate);
        }
        return static_cast<float>(static_cast<std::int16_t>(immediate));
    }

    void matmul(std::uint32_t instruction, unsigned activation) {
        const std::uint32_t rows = desc_[0].partial_rows;
        const std::uint32_t inner = desc_[0].partial_cols;
        const std::uint32_t cols = desc_[1].partial_cols;
        const bool mac = ((instruction >> 27) & 1u) != 0;
        const unsigned a_dtype = in_dtype_[0];
        const unsigned w_dtype = in_dtype_[1];
        const bool a_int = a_dtype == INT8 || a_dtype == INT4;
        const bool w_int = w_dtype == INT8 || w_dtype == INT4;
        // Legal combination table (ISA_V09.md section 6).
        const bool int_path = a_int && w_int;
        require(!(a_dtype == INT4), "matmul src0 (activation) cannot be INT4");
        require(!(a_int && !w_int), "matmul INT activation requires INT weight");
        const std::size_t total = static_cast<std::size_t>(rows) * cols;
        if (int_path) {
            if (!mac || iacc_.size() != total || !int_result_) {
                require(!mac || iacc_.size() == total,
                        "MAC chain switched arithmetic path or tile shape");
                iacc_.assign(total, 0);
            }
        } else {
            require(!mac || (!int_result_ && output_.size() == total),
                    "MAC chain switched arithmetic path or tile shape");
            if (!mac || output_.size() != total) {
                output_.assign(total, 0.0f);
            }
        }
        for (std::uint32_t row = 0; row < rows; ++row) {
            for (std::uint32_t col = 0; col < cols; ++col) {
                const std::size_t out = static_cast<std::size_t>(row) * cols + col;
                if (int_path) {
                    long long sum = 0;
                    for (std::uint32_t k = 0; k < inner; ++k) {
                        sum += static_cast<long long>(
                                   input1_[std::size_t(row) * inner + k]) *
                               static_cast<long long>(
                                   input2_[std::size_t(k) * cols + col]);
                    }
                    iacc_[out] += sum;
                } else {
                    float sum = 0.0f;
                    for (std::uint32_t k = 0; k < inner; ++k) {
                        sum += input1_[std::size_t(row) * inner + k] *
                               input2_[std::size_t(k) * cols + col];
                    }
                    output_[out] += sum;
                }
            }
        }
        int_result_ = int_path;
        result_src0_int_ = a_int;
        result_src1_int_ = w_int;
        pending_activation_ = 0;
        if (activation) {
            if (int_path || w_int) {
                // dequant precedes activation: defer to the drain.
                pending_activation_ = activation;
            } else {
                for (float& value : output_) {
                    value = activate(value, activation);
                }
            }
        }
        output_rows_ = rows;
        output_cols_ = cols;
    }

    void compute(std::uint32_t instruction) {
        const unsigned op = instruction & 0xffu;
        const unsigned mode = (instruction >> 30) & 3u;
        const std::uint16_t immediate = (instruction >> 8) & 0xffffu;
        const bool matrix = op >= 0x40 && op <= 0x43;
        const unsigned activation = matrix ? ((instruction >> 28) & 3u) : 0;
        if (matrix) {
            ++counters_.matrix_ops;
            if (op == 0x42 && mode == 2) {
                matmul(instruction, activation);
                return;
            }
            require(in_dtype_[0] == FP16 && (mode != 2 || in_dtype_[1] == FP16),
                    "elementwise matrix ops require FP16 operands");
        } else {
            ++counters_.vector_ops;
            check_vlen("vector op");
        }
        const std::size_t count = matrix ? input1_.size() : vector_length_;
        if (output_.size() != count) {
            output_.assign(count, 0.0f);
        }
        for (std::size_t i = 0; i < count; ++i) {
            const float a = i < input1_.size() ? input1_[i] : 0.0f;
            const float b = rhs(mode, immediate, i);
            float result = a;
            switch (op) {
            case 0x01: case 0x40: result = a + b; break;
            case 0x02: case 0x41: result = a - b; break;
            case 0x0a: case 0x42: result = a * b; break;
            case 0x0b: result = a / b; break;
            case 0x0c: result = output_[i] + a * b; break;
            case 0x0d: case 0x43: result = b; break;
            case 0x0e: result = std::sqrt(a); break;
            case 0x0f: result = std::exp(a); break;
            case 0x11: result = a == b ? 1.0f : 0.0f; break;
            case 0x12: result = ((instruction >> 28) & 1u) ? std::max(a, b)
                                                           : std::min(a, b); break;
            case 0x13: result = static_cast<float>(static_cast<long>(a)); break;
            case 0x16: result = 0.0f - a; break;
            case 0x17: result = a; break;
            case 0x09: {
                std::int16_t amount = static_cast<std::int16_t>(immediate);
                if (mode == 2 && i < input2_.size()) {
                    amount = static_cast<std::int16_t>(
                        static_cast<long>(input2_[i]));
                }
                result = a * std::pow(2.0f, static_cast<float>(amount));
                break;
            }
            case 0x08: {
                const unsigned sub = (instruction >> 27) & 7u;
                const long ia = static_cast<long>(a);
                const long ib = static_cast<long>(b);
                if (sub == 0) result = static_cast<float>(ia & ib);
                else if (sub == 1) result = static_cast<float>(ia | ib);
                else if (sub == 2) result = static_cast<float>(~ia);
                else if (sub == 3) result = static_cast<float>(ia ^ ib);
                else if (sub == 4) result = static_cast<float>(~(ia & ib));
                else if (sub == 5) result = static_cast<float>(~(ia | ib));
                break;
            }
            default:
                fail("unknown compute opcode");
            }
            output_[i] = matrix ? activate(result, activation) : result;
        }
        int_result_ = false;
        result_src0_int_ = result_src1_int_ = false;
        pending_activation_ = 0;
        if (matrix) {
            output_rows_ = desc_[0].partial_rows;
            output_cols_ = desc_[0].partial_cols;
        } else {
            output_rows_ = 1;
            output_cols_ = static_cast<std::uint32_t>(count);
        }
    }

    // Reduces (0x14 sum / 0x19 max) run the 256-lane chunk in flat order with
    // an FP32 carry register; [27]=carry-in continues the previous chunk.
    // reduce-max seeds from the first element (V3-003 fix), never from zero.
    void reduce(bool is_max, bool carry_in) {
        ++counters_.vector_ops;
        check_vlen("reduce");
        require(!input1_.empty(), "reduce over empty input");
        float acc;
        std::size_t first = 0;
        if (carry_in) {
            acc = reduce_carry_;
        } else if (is_max) {
            acc = input1_[0];
            first = 1;
        } else {
            acc = 0.0f;
        }
        for (std::size_t i = first; i < input1_.size(); ++i) {
            acc = is_max ? std::max(acc, input1_[i]) : acc + input1_[i];
        }
        reduce_carry_ = acc;
        output_.assign(1, acc);
        int_result_ = false;
        result_src0_int_ = result_src1_int_ = false;
        output_rows_ = output_cols_ = 1;
    }

    // VQUANT (0x1A): FP16 vector at src0 -> symmetric round-to-nearest-even
    // integers packed at dst, divided by the FP32 scale a_scale[0].
    // VDEQUANT (0x1B): packed integers at src0 -> FP16-representable floats in
    // the output register, multiplied by a_scale[0].  [27]=1 selects INT4.
    void vquant(std::uint32_t instruction) {
        ++counters_.vquant;
        check_vlen("vquant");
        const bool int4 = (instruction >> 27) & 1u;
        const int limit = int4 ? 7 : 127;
        const unsigned dtype = int4 ? INT4 : INT8;
        const unsigned width = dtype_width_nibbles(dtype);
        const float scale = read_scale(a_scale_address_, 0, "vquant scale");
        require(scale != 0.0f, "vquant with zero scale");
        const Descriptor& src = desc_[0];
        const Descriptor& dst = desc_[2];
        check_alignment(src.partial_address, FP16, "vquant src");
        check_alignment(dst.partial_address, dtype, "vquant dst");
        for (std::uint32_t i = 0; i < vector_length_; ++i) {
            const float x =
                read_elem(FP16, std::uint64_t(src.partial_address) + 4ull * i);
            float q = std::nearbyint(x / scale);
            q = std::min(std::max(q, static_cast<float>(-limit)),
                         static_cast<float>(limit));
            write_packed_int(std::uint64_t(dst.partial_address) +
                                 std::uint64_t(i) * width,
                             static_cast<int>(q), int4);
        }
    }

    void vdequant(std::uint32_t instruction) {
        ++counters_.vdequant;
        check_vlen("vdequant");
        const bool int4 = (instruction >> 27) & 1u;
        const unsigned dtype = int4 ? INT4 : INT8;
        const unsigned width = dtype_width_nibbles(dtype);
        const float scale = read_scale(a_scale_address_, 0, "vdequant scale");
        const Descriptor& src = desc_[0];
        check_alignment(src.partial_address, dtype, "vdequant src");
        output_.assign(vector_length_, 0.0f);
        for (std::uint32_t i = 0; i < vector_length_; ++i) {
            const float q = read_elem(
                dtype, std::uint64_t(src.partial_address) + std::uint64_t(i) * width);
            output_[i] = q * scale;
        }
        int_result_ = false;
        result_src0_int_ = result_src1_int_ = false;
        output_rows_ = 1;
        output_cols_ = vector_length_;
    }

    // ------------------------------------------------------------- decode

    void execute(std::uint32_t word) {
        const unsigned op = word & 0xffu;
        switch (op) {
            case 0x00:
                require(word == 0, "NOP with nonzero reserved bits");
                ++counters_.nop;
                return;
            case 0xF0: ++counters_.snapshot; write_image(); return;
            case 0xFF: ++counters_.halt; write_image(); halted_ = true; return;
            case 0xA0: ++counters_.gload; dma(false); return;
            case 0xA8: ++counters_.gstore; dma(true); return;
            case 0x80: {
                const unsigned operand = (word >> 30) & 3u;
                if (operand < 3) {
                    const bool high = ((word >> 29) & 1u) != 0;
                    const bool partial = ((word >> 28) & 1u) != 0;
                    const std::uint16_t value = (word >> 8) & 0xffffu;
                    std::uint32_t& address = partial
                        ? desc_[operand].partial_address
                        : desc_[operand].main_address;
                    address = replace_half(address, value, high);
                    require((address >> 24) == 0,
                            "descriptor address exceeds 24-bit nibble space");
                }
                return;
            }
            case 0x82:
                vector_length_ = (word >> 8) & 0xffffu;
                require(vector_length_ <= kMaxVlen, "vlen exceeds 256 lanes");
                return;
            case 0x88: case 0x89: {
                const unsigned operand = (word >> 30) & 3u;
                if (operand < 3) {
                    const bool partial = ((word >> 29) & 1u) != 0;
                    const std::uint32_t value = (word >> 8) & 0xffffu;
                    Descriptor& d = desc_[operand];
                    if (op == 0x88) {
                        (partial ? d.partial_rows : d.main_rows) = value;
                    } else {
                        (partial ? d.partial_cols : d.main_cols) = value;
                    }
                    d.dtype = (word >> 25) & 3u;   // spare bits [26:25]
                }
                return;
            }
            case 0x8A: case 0x8B: {
                const bool high = ((word >> 29) & 1u) != 0;
                const std::uint16_t value = (word >> 8) & 0xffffu;
                std::uint32_t& address =
                    op == 0x8A ? a_scale_address_ : w_scale_address_;
                address = replace_half(address, value, high);
                require((address >> 24) == 0,
                        "scale address exceeds 24-bit nibble space");
                return;
            }
            case 0x90:
                load((word >> 31) & 1u, (word >> 30) & 1u,
                     ((word >> 29) & 1u) != 0,
                     (word >> 16) & 0xffu, (word >> 8) & 0xffu);
                return;
            case 0x98: save(word); return;
            case 0x14: reduce(false, ((word >> 27) & 1u) != 0); return;
            case 0x19: reduce(true, ((word >> 27) & 1u) != 0); return;
            case 0x15: {
                ++counters_.vector_ops;
                check_vlen("broadcast");
                const unsigned mode = (word >> 30) & 3u;
                const bool high = ((word >> 29) & 1u) != 0;
                const std::uint16_t value = (word >> 8) & 0xffffu;
                float scalar;
                if (mode == 1) {
                    broadcast_address_ = replace_half(broadcast_address_, value, high);
                    check_alignment(broadcast_address_, FP16, "broadcast");
                    scalar = read_elem(FP16, broadcast_address_);
                } else {
                    // signed int16 immediate (V3-030 fix)
                    scalar = static_cast<float>(static_cast<std::int16_t>(value));
                }
                output_.assign(vector_length_, scalar);
                int_result_ = false;
                result_src0_int_ = result_src1_int_ = false;
                output_rows_ = 1;
                output_cols_ = vector_length_;
                return;
            }
            case 0x18: {
                ++counters_.vector_ops;
                const bool sine = ((word >> 27) & 1u) != 0;
                output_.resize(input1_.size());
                for (std::size_t i = 0; i < input1_.size(); ++i) {
                    output_[i] = sine ? std::sin(input1_[i]) : std::cos(input1_[i]);
                }
                int_result_ = false;
                result_src0_int_ = result_src1_int_ = false;
                output_rows_ = 1;
                output_cols_ = static_cast<std::uint32_t>(output_.size());
                return;
            }
            case 0x1A: vquant(word); return;
            case 0x1B: vdequant(word); return;
            default:
                if ((op >= 0x01 && op <= 0x17) || (op >= 0x40 && op <= 0x43)) {
                    compute(word);
                    return;
                }
                fail("unknown opcode");
        }
    }

    void write_image() {
        std::ofstream file("saved_global_memory.bin",
                           std::ios::binary | std::ios::app);
        if (!global_.empty()) {
            file.write(reinterpret_cast<const char*>(global_.data()),
                       global_.size() * 4);
        }
    }
};

}  // namespace

int main() {
    Machine machine;
    return machine.run();
}
