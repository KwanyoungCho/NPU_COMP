// Source-level model of the supplied 0818/ver.08 NPU C-model.
//
// This file intentionally follows the vendor executable, including observable
// quirks.  It is an analysis/reference model, not an "improved" simulator:
//   * G-buffer and program memory are fixed at 8192 FP16 and 32768 words.
//   * 0xF0 writes saved_G_buffer_data.bin and execution continues.
//   * loads/saves consume the PARTIAL address/shape descriptors.
//   * reduce-max starts from 0 (therefore all-negative inputs reduce to 0).
//   * GELU is the vendor approximation x * sigmoid(2*x).
//
// Inputs and output use the same fixed filenames as the vendor binary.  The
// arithmetic datapath is float32; values are rounded to FP16 on G-buffer save.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

constexpr std::size_t kGBufferSize = 8192;
constexpr std::size_t kProgramSize = 32768;

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

float round_half(float value) { return half_to_float(float_to_half(value)); }
float sigmoid(float value) { return 1.0f / (1.0f + std::exp(-value)); }

float activate(float value, unsigned activation) {
    if (activation == 2) {
        return value * sigmoid(value);       // SiLU
    }
    if (activation == 1 || activation == 3) {
        return value * sigmoid(2.0f * value);  // vendor "GELU"
    }
    return value;
}

struct MatrixDescriptor {
    std::uint32_t main_address = 0;
    std::uint32_t partial_address = 0;
    std::uint32_t main_rows = 0;
    std::uint32_t main_cols = 0;
    std::uint32_t partial_rows = 0;
    std::uint32_t partial_cols = 0;
};

class Model {
public:
    int run() {
        if (!read_gbuffer() || !read_program()) {
            return 1;
        }
        print_header();
        for (std::size_t pc = 0; pc < program_words_; ++pc) {
            execute(pc, program_[pc], pc + 1 == program_words_);
        }
        if (snapshot_written_) {
            snapshot_file_.put('\n');
            snapshot_file_.close();
        }
        return 0;
    }

private:
    std::vector<float> gbuffer_ = std::vector<float>(kGBufferSize, 0.0f);
    std::vector<std::uint32_t> program_ = std::vector<std::uint32_t>(kProgramSize, 0);
    std::size_t gbuffer_bytes_ = 0;
    std::size_t program_bytes_ = 0;
    std::size_t program_words_ = 0;
    MatrixDescriptor desc_[3];
    std::uint32_t vector_length_ = 0;
    std::uint32_t broadcast_address_ = 0;
    std::uint32_t output_rows_ = 0;
    std::uint32_t output_cols_ = 0;
    std::vector<float> input1_;
    std::vector<float> input2_;
    std::vector<float> output_;
    bool snapshot_written_ = false;
    std::ofstream snapshot_file_;

    bool read_gbuffer() {
        std::ifstream file("G_buffer_data.bin", std::ios::binary);
        if (!file) {
            std::cerr << "cannot open G_buffer_data.bin\n";
            return false;
        }
        file.seekg(0, std::ios::end);
        gbuffer_bytes_ = static_cast<std::size_t>(file.tellg());
        file.seekg(0);
        std::vector<unsigned char> raw(gbuffer_bytes_);
        file.read(reinterpret_cast<char*>(raw.data()), raw.size());
        const std::size_t count = std::min(kGBufferSize, raw.size() / 2);
        for (std::size_t i = 0; i < count; ++i) {
            const std::uint16_t bits = static_cast<std::uint16_t>(raw[2 * i]) |
                                       (static_cast<std::uint16_t>(raw[2 * i + 1]) << 8);
            gbuffer_[i] = half_to_float(bits);
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
        program_bytes_ = static_cast<std::size_t>(file.tellg());
        file.seekg(0);
        std::vector<unsigned char> raw(program_bytes_);
        file.read(reinterpret_cast<char*>(raw.data()), raw.size());
        program_words_ = std::min(kProgramSize, raw.size() / 4);
        for (std::size_t i = 0; i < program_words_; ++i) {
            program_[i] = static_cast<std::uint32_t>(raw[4 * i]) |
                          (static_cast<std::uint32_t>(raw[4 * i + 1]) << 8) |
                          (static_cast<std::uint32_t>(raw[4 * i + 2]) << 16) |
                          (static_cast<std::uint32_t>(raw[4 * i + 3]) << 24);
        }
        return true;
    }

    void print_header() const {
        std::cout << "G_buffer size :  " << gbuffer_bytes_ << "\n\n\n\n0\n";
        std::cout << "Program memory size :  " << std::hex << program_bytes_ << std::dec
                  << "\n\n\n\n0\n";
        for (int i = 0; i < 20; ++i) {
            std::cout << std::hex << program_[i] << std::dec << '\n';
        }
        std::cout << "\n\n\n";
    }

    static std::uint32_t replace_half(std::uint32_t old, std::uint16_t value, bool high) {
        return high ? ((old & 0x0000ffffu) | (static_cast<std::uint32_t>(value) << 16))
                    : ((old & 0xffff0000u) | value);
    }

    float read_g(std::uint32_t address) const {
        return address < gbuffer_.size() ? gbuffer_[address] : 0.0f;
    }

    void write_g(std::uint32_t address, float value) {
        if (address < gbuffer_.size()) {
            gbuffer_[address] = round_half(value);
        }
    }

    std::vector<float>& input(unsigned operand) { return operand == 0 ? input1_ : input2_; }

    void print_values(const std::vector<float>& values) const {
        for (float value : values) {
            std::cout << value << '\n';
        }
    }

    void load(unsigned matrix, unsigned operand, bool strided, unsigned ncols, unsigned start) {
        const MatrixDescriptor& d = desc_[operand];
        std::vector<float>& target = input(operand);
        target.clear();
        if (!matrix) {
            target.reserve(vector_length_);
            for (std::uint32_t i = 0; i < vector_length_; ++i) {
                target.push_back(read_g(d.partial_address + i));
            }
        } else if (strided) {
            target.reserve(static_cast<std::size_t>(d.partial_rows) * ncols);
            for (unsigned col = start; col < start + ncols; ++col) {
                for (std::uint32_t row = 0; row < d.partial_rows; ++row) {
                    target.push_back(read_g(d.partial_address + row * d.main_cols + col));
                }
            }
        } else {
            target.reserve(static_cast<std::size_t>(d.partial_rows) * d.partial_cols);
            for (std::uint32_t row = 0; row < d.partial_rows; ++row) {
                for (std::uint32_t col = 0; col < d.partial_cols; ++col) {
                    target.push_back(read_g(d.partial_address + row * d.main_cols + col));
                }
            }
        }
        const char* label = operand == 0 ? "PE_in_data_1_array :  " : "PE_in_data_2_array :  ";
        for (float value : target) {
            std::cout << label << value << '\n';
        }
    }

    void save(unsigned matrix, bool strided, unsigned ncols, unsigned start) {
        const MatrixDescriptor& d = desc_[2];
        if (!matrix) {
            for (std::size_t i = 0; i < vector_length_; ++i) {
                write_g(d.partial_address + static_cast<std::uint32_t>(i),
                        i < output_.size() ? output_[i] : 0.0f);
            }
        } else if (strided) {
            const std::uint32_t rows = output_rows_ ? output_rows_ : desc_[0].partial_rows;
            for (unsigned col = 0; col < ncols; ++col) {
                for (std::uint32_t row = 0; row < rows; ++row) {
                    const std::size_t source = static_cast<std::size_t>(col) * rows + row;
                    if (source < output_.size()) {
                        write_g(d.partial_address + row * d.main_cols + start + col, output_[source]);
                    }
                }
            }
        } else {
            const std::uint32_t rows = output_rows_ ? output_rows_ : desc_[2].partial_rows;
            const std::uint32_t cols = output_cols_ ? output_cols_ : desc_[2].partial_cols;
            const std::uint32_t stride = d.main_cols ? d.main_cols : cols;
            for (std::uint32_t row = 0; row < rows; ++row) {
                for (std::uint32_t col = 0; col < cols; ++col) {
                    const std::size_t source = static_cast<std::size_t>(row) * cols + col;
                    if (source < output_.size()) {
                        write_g(d.partial_address + row * stride + col, output_[source]);
                    }
                }
            }
        }
        const std::size_t printed = matrix ? output_.size() : vector_length_;
        for (std::size_t i = 0; i < printed; ++i) {
            std::cout << "PE_out_array :  " << (i < output_.size() ? output_[i] : 0.0f) << '\n';
        }
    }

    void snapshot() {
        if (!snapshot_written_) {
            snapshot_file_.open("saved_G_buffer_data.bin", std::ios::binary | std::ios::trunc);
            snapshot_written_ = true;
        }
        for (float value : gbuffer_) {
            const std::uint16_t bits = float_to_half(value);
            snapshot_file_.put(static_cast<char>(bits & 0xffu));
            snapshot_file_.put(static_cast<char>((bits >> 8) & 0xffu));
        }
    }

    float rhs(unsigned mode, std::uint16_t immediate, std::size_t index) const {
        if (mode == 2) {
            return index < input2_.size() ? input2_[index] : 0.0f;
        }
        if (mode == 1) {
            return read_g(immediate);
        }
        return static_cast<float>(static_cast<std::int16_t>(immediate));
    }

    void matmul(std::uint32_t instruction, unsigned activation) {
        const std::uint32_t rows = desc_[0].partial_rows;
        const std::uint32_t inner = desc_[0].partial_cols;
        const std::uint32_t cols = desc_[1].partial_cols;
        const bool mac = ((instruction >> 27) & 1u) != 0;
        if (!mac || output_.size() != static_cast<std::size_t>(rows) * cols) {
            output_.assign(static_cast<std::size_t>(rows) * cols, 0.0f);
        }
        for (std::uint32_t row = 0; row < rows; ++row) {
            for (std::uint32_t col = 0; col < cols; ++col) {
                float sum = 0.0f;
                for (std::uint32_t k = 0; k < inner; ++k) {
                    sum += input1_[static_cast<std::size_t>(row) * inner + k] *
                           input2_[static_cast<std::size_t>(k) * cols + col];
                }
                output_[static_cast<std::size_t>(row) * cols + col] += sum;
            }
        }
        if (activation) {
            for (float& value : output_) {
                value = activate(value, activation);
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
        if (matrix && op == 0x42 && mode == 2) {
            matmul(instruction, activation);
            print_values(output_);
            return;
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
            case 0x12: result = ((instruction >> 28) & 1u) ? std::max(a, b) : std::min(a, b); break;
            case 0x13: result = static_cast<float>(static_cast<long>(a)); break;
            case 0x16: result = 0.0f - a; break;
            case 0x17: result = a; break;
            case 0x09: {
                std::int16_t amount = static_cast<std::int16_t>(immediate);
                if (mode == 2 && i < input2_.size()) {
                    amount = static_cast<std::int16_t>(static_cast<long>(input2_[i]));
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
            default: break;
            }
            output_[i] = matrix ? activate(result, activation) : result;
        }
        if (matrix) {
            output_rows_ = desc_[0].partial_rows;
            output_cols_ = desc_[0].partial_cols;
        } else {
            output_rows_ = 1;
            output_cols_ = static_cast<std::uint32_t>(count);
        }
        print_values(output_);
    }

    void execute(std::size_t pc, std::uint32_t instruction, bool final_instruction) {
        const unsigned op = instruction & 0xffu;
        const unsigned mode = (instruction >> 30) & 3u;
        std::cout << "p_counter :  " << pc << "\ninstruction :  " << std::hex
                  << instruction << std::dec << '\n';
        if (instruction == 0) {
            std::cout << "NOP --- \n";
        } else if (op == 0x80) {
            const unsigned operand = (instruction >> 30) & 3u;
            if (operand < 3) {
                const bool high = ((instruction >> 29) & 1u) != 0;
                const bool partial = ((instruction >> 28) & 1u) != 0;
                const std::uint16_t value = (instruction >> 8) & 0xffffu;
                std::uint32_t& address = partial ? desc_[operand].partial_address
                                                 : desc_[operand].main_address;
                address = replace_half(address, value, high);
            }
        } else if (op == 0x82) {
            vector_length_ = (instruction >> 8) & 0xffffu;
        } else if (op == 0x88 || op == 0x89) {
            const unsigned operand = (instruction >> 30) & 3u;
            if (operand < 3) {
                const bool partial = ((instruction >> 29) & 1u) != 0;
                const std::uint32_t value = (instruction >> 8) & 0xffffu;
                if (op == 0x88) {
                    (partial ? desc_[operand].partial_rows : desc_[operand].main_rows) = value;
                } else {
                    (partial ? desc_[operand].partial_cols : desc_[operand].main_cols) = value;
                }
            }
        } else if (op == 0x90) {
            load((instruction >> 31) & 1u, (instruction >> 30) & 1u,
                 ((instruction >> 29) & 1u) != 0,
                 (instruction >> 16) & 0xffu, (instruction >> 8) & 0xffu);
        } else if (op == 0x98) {
            save((instruction >> 31) & 1u, ((instruction >> 29) & 1u) != 0,
                 (instruction >> 16) & 0xffu, (instruction >> 8) & 0xffu);
        } else if (op == 0x14) {
            float sum = 0.0f;
            for (float value : input1_) sum += value;
            output_.assign(1, sum);
            output_rows_ = output_cols_ = 1;
            print_values(output_);
        } else if (op == 0x19) {
            float maximum = 0.0f;  // observed vendor behavior
            for (float value : input1_) maximum = std::max(maximum, value);
            // The vendor leaves a vector-length output buffer and writes the
            // reduced value only to lane zero.
            output_.assign(vector_length_, 0.0f);
            if (!output_.empty()) output_[0] = maximum;
            output_rows_ = output_cols_ = 1;
            std::cout << maximum << '\n';
        } else if (op == 0x15) {
            const bool high = ((instruction >> 29) & 1u) != 0;
            const std::uint16_t value = (instruction >> 8) & 0xffffu;
            float scalar;
            if (mode == 1) {
                broadcast_address_ = replace_half(broadcast_address_, value, high);
                scalar = read_g(broadcast_address_);
            } else {
                scalar = static_cast<float>(static_cast<std::int16_t>(value));
            }
            output_.assign(vector_length_, scalar);
            output_rows_ = 1;
            output_cols_ = vector_length_;
            print_values(output_);
        } else if (op == 0x18) {
            const bool sine = ((instruction >> 27) & 1u) != 0;
            output_.resize(input1_.size());
            for (std::size_t i = 0; i < input1_.size(); ++i) {
                output_[i] = sine ? std::sin(input1_[i]) : std::cos(input1_[i]);
            }
            output_rows_ = 1;
            output_cols_ = static_cast<std::uint32_t>(output_.size());
            print_values(output_);
        } else if (op == 0xf0) {
            snapshot();
        } else {
            compute(instruction);
        }
        if (!final_instruction) {
            std::cout << "\n\n\n";
        }
    }
};

}  // namespace

int main() {
    Model model;
    return model.run();
}
