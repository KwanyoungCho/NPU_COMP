// ISA ver.09 C-model (our own next-generation design; not vendor parity).
//
// Machine model per d_compiler/ISA_V09.md:
//   * Global memory: 32-bit addresses in 32-bit cells (up to 16 GiB), DMA-only
//     storage with no dtype semantics.  Actual size = input file size.
//   * SRAM: 8 MiB shared scratchpad addressed in 4-bit nibbles (24 effective
//     bits).  All compute units read/write SRAM only.  Zero-initialized.
//   * Cell<->nibble mapping is little-endian: cell bits [4k+3:4k] are SRAM
//     nibble base+k, matching host little-endian files byte for byte.
//   * Out-of-range access is a hard error (no silent corruption).
//   * HALT (0xFF) is the only normal termination and appends the global image
//     to the output file; SNAPSHOT (0xF0) appends a mid-run checkpoint.
//     Falling off the end of the program without HALT is an error.
//
// Files (cwd): global_memory.bin (in), program_memory.bin (in),
//              saved_global_memory.bin (out), perf_counters.txt (out).
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

struct Counters {
    std::uint64_t words_executed = 0;
    std::uint64_t nop = 0;
    std::uint64_t snapshot = 0;
    std::uint64_t halt = 0;
    std::uint64_t gload = 0;
    std::uint64_t gstore = 0;
    std::uint64_t dma_cells_loaded = 0;
    std::uint64_t dma_cells_stored = 0;

    void dump(const char* path) const {
        std::ofstream file(path);
        file << "words_executed " << words_executed << '\n'
             << "nop " << nop << '\n'
             << "snapshot " << snapshot << '\n'
             << "halt " << halt << '\n'
             << "gload " << gload << '\n'
             << "gstore " << gstore << '\n'
             << "dma_cells_loaded " << dma_cells_loaded << '\n'
             << "dma_cells_stored " << dma_cells_stored << '\n';
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
            switch (word & 0xffu) {
                case 0x00: ++counters_.nop; break;
                case 0xF0: ++counters_.snapshot; write_image(); break;
                case 0xFF: ++counters_.halt; write_image(); return finish(0);
                case 0xA0: ++counters_.gload; dma(false); break;
                case 0xA8: ++counters_.gstore; dma(true); break;
                default:
                    return fail("unknown opcode");
            }
        }
        return fail("program ended without HALT (0xFF)");
    }

private:
    std::vector<std::uint32_t> global_;
    std::vector<std::uint8_t> sram_;
    std::vector<std::uint32_t> program_;
    std::size_t pc_ = 0;
    Counters counters_;

    int finish(int code) {
        counters_.dump("perf_counters.txt");
        return code;
    }

    int fail(const std::string& message) {
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

    // GLOAD (0xA0) / GSTORE (0xA8), 5 words (ISA_V09.md section 3):
    //   w0 opcode | w1 global cell address | w2 global row stride (cells)
    //   w3 SRAM nibble address (24 effective bits, 8-nibble aligned)
    //   w4 rows[31:16] cols[15:0]  (cols in cells)
    // dtype-blind synchronous copy; SRAM rows land contiguously
    // (row r occupies nibbles w3 + r*cols*8 .. +cols*8).
    void dma(bool store) {
        if (pc_ + 4 >= program_.size()) {
            fail("truncated DMA instruction (needs 5 words)");
        }
        const std::uint64_t g_addr = program_[pc_ + 1];
        const std::uint64_t g_stride = program_[pc_ + 2];
        const std::uint32_t sram_addr = program_[pc_ + 3];
        const std::uint32_t rows = program_[pc_ + 4] >> 16;
        const std::uint32_t cols = program_[pc_ + 4] & 0xffffu;
        pc_ += 4;
        if (sram_addr >= kSramNibbles) {
            fail("DMA SRAM address exceeds 24-bit nibble space");
        }
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
