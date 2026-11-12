// Compare the native MXFP4 grouped kernels against the original dot kernel
// with initialized scale tables, real expert-pack weights and full matrix dimensions.
#include "ggml.h"
#include "ggml-cpu.h"
#include "quants.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

struct matrix {
    std::vector<uint8_t> bytes;
    int columns;
    int rows;
};

static matrix load_matrix(const char * path, const char * role, int columns, int rows) {
    std::ifstream pack(path, std::ios::binary | std::ios::ate);
    require(bool(pack), "cannot open expert pack");
    const auto pack_size = uint64_t(pack.tellg());
    pack.seekg(0);
    char magic[16] = {};
    uint32_t version = 0, header = 0;
    uint64_t count = 0, data_start = 0;
    pack.read(magic, sizeof(magic));
    pack.read(reinterpret_cast<char *>(&version), sizeof(version));
    pack.read(reinterpret_cast<char *>(&header), sizeof(header));
    pack.read(reinterpret_cast<char *>(&count), sizeof(count));
    pack.read(reinterpret_cast<char *>(&data_start), sizeof(data_start));
    require(bool(pack) && !std::memcmp(magic, "GGMLMOEPACKv1", 12) &&
            version == 1 && header == 40 && count < 1000000 &&
            data_start >= 40 + 152 * count && data_start <= pack_size, "bad expert-pack header");
    const std::string target = std::string("blk.0.ffn_") + role + "_exps.weight";
    const uint64_t expected = ggml_row_size(GGML_TYPE_MXFP4, columns) * rows;
    uint64_t location = 0;
    for (uint64_t i = 0; i < count; ++i) {
        char name[129] = {};
        int32_t expert = -1;
        uint32_t reserved = 0;
        uint64_t offset = 0, size = 0;
        pack.read(name, 128);
        pack.read(reinterpret_cast<char *>(&expert), 4);
        pack.read(reinterpret_cast<char *>(&reserved), 4);
        pack.read(reinterpret_cast<char *>(&offset), 8);
        pack.read(reinterpret_cast<char *>(&size), 8);
        require(bool(pack), "truncated expert-pack index");
        if (expert == 0 && target == name) {
            require(size == expected && offset >= data_start &&
                    offset <= pack_size && size <= pack_size - offset, "wrong expert shape or extent");
            location = offset;
            break;
        }
    }
    require(location != 0, "expert 0 role not found");
    matrix result { std::vector<uint8_t>(expected), columns, rows };
    pack.seekg(location);
    pack.read(reinterpret_cast<char *>(result.bytes.data()), expected);
    require(bool(pack), "short expert read");
    return result;
}

static uint64_t check(const matrix & weights, int tokens, std::mt19937 & rng) {
    const size_t input_stride = ggml_row_size(GGML_TYPE_Q8_0, weights.columns);
    const size_t weight_stride = ggml_row_size(GGML_TYPE_MXFP4, weights.columns);
    std::vector<uint8_t> input(input_stride * tokens);
    std::vector<float> floats(weights.columns);
    std::normal_distribution<float> sample(0.0f, 0.1f);
    for (int token = 0; token < tokens; ++token) {
        for (auto & value : floats) { value = sample(rng); }
        quantize_row_q8_0(floats.data(), input.data() + token * input_stride, weights.columns);
    }
    uint64_t compared = 0;
    bool nonzero = false;
    for (int row = 0; row < weights.rows; ++row) {
        const void * weight = weights.bytes.data() + row * weight_stride;
        for (int token = 0; token < tokens; token += 4) {
            const void * x[4];
            float old[4] = {}, two[4] = {}, four[4] = {};
            for (int column = 0; column < 4; ++column) {
                x[column] = input.data() + std::min(token + column, tokens - 1) * input_stride;
                ggml_vec_dot_mxfp4_q8_0(weights.columns, &old[column], 0, weight, 0, x[column], 0, 1);
            }
            ggml_vec_dot_mxfp4_q8_0_2x(weights.columns, two + 0, weight, x[0], x[1]);
            ggml_vec_dot_mxfp4_q8_0_2x(weights.columns, two + 2, weight, x[2], x[3]);
            ggml_vec_dot_mxfp4_q8_0_4x(weights.columns, four, weight, x[0], x[1], x[2], x[3]);
            if (std::memcmp(old, two, sizeof(old)) || std::memcmp(old, four, sizeof(old))) {
                std::fprintf(stderr, "mismatch columns=%d rows=%d tokens=%d row=%d token=%d\n",
                    weights.columns, weights.rows, tokens, row, token);
                throw std::runtime_error("MXFP4 grouped output differs bitwise");
            }
            for (int column = 0; column < std::min(4, tokens - token); ++column) {
                nonzero |= old[column] != 0.0f;
            }
            compared += std::min(4, tokens - token);
        }
    }
    require(nonzero, "MXFP4 test outputs are all zero; ggml_cpu_init may be missing");
    return compared;
}

static double bench(const matrix & weights, int tokens, int width) {
    const size_t input_stride = ggml_row_size(GGML_TYPE_Q8_0, weights.columns);
    const size_t weight_stride = ggml_row_size(GGML_TYPE_MXFP4, weights.columns);
    std::vector<uint8_t> input(input_stride * tokens);
    std::vector<float> floats(weights.columns);
    std::mt19937 rng(20260924u + unsigned(weights.columns) * 13u + unsigned(tokens));
    std::normal_distribution<float> sample(0.0f, 0.1f);
    for (int token = 0; token < tokens; ++token) {
        for (auto & value : floats) { value = sample(rng); }
        quantize_row_q8_0(floats.data(), input.data() + token * input_stride, weights.columns);
    }
    volatile float checksum = 0;
    const int rows = 16;
    int repeats = 0;
    const auto start = std::chrono::steady_clock::now();
    do {
        for (int row = 0; row < rows; ++row) {
            const void * weight = weights.bytes.data() + row * weight_stride;
            for (int token = 0; token < tokens; token += width) {
                const void * x0 = input.data() + token * input_stride;
                const void * x1 = input.data() + (token + 1) * input_stride;
                float out[4];
                if (width == 2) {
                    ggml_vec_dot_mxfp4_q8_0_2x(weights.columns, out, weight, x0, x1);
                } else {
                    const void * x2 = input.data() + (token + 2) * input_stride;
                    const void * x3 = input.data() + (token + 3) * input_stride;
                    ggml_vec_dot_mxfp4_q8_0_4x(weights.columns, out, weight, x0, x1, x2, x3);
                }
                checksum = checksum + out[0];
            }
        }
        repeats++;
    } while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < 0.25);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::fprintf(stderr, "bench_checksum=%g\n", double(checksum));
    require(checksum != 0.0f, "MXFP4 benchmark outputs are all zero");
    return 1e9 * seconds / (double(repeats) * rows * tokens);
}

static void dot_columns(const matrix & weights, const uint8_t * input, size_t stride,
        const void * weight, int token, int count, float * out) {
    const void * x0 = input + size_t(token) * stride;
    if (count == 4) {
        ggml_vec_dot_mxfp4_q8_0_4x(weights.columns, out, weight, x0,
                input + size_t(token + 1) * stride, input + size_t(token + 2) * stride,
                input + size_t(token + 3) * stride);
    } else {
        if (count >= 2) {
            ggml_vec_dot_mxfp4_q8_0_2x(weights.columns, out, weight, x0,
                    input + size_t(token + 1) * stride);
        }
        if (count == 1 || count == 3) {
            const int last = count == 1 ? 0 : 2;
            ggml_vec_dot_mxfp4_q8_0(weights.columns, out + last, 0, weight, 0,
                    input + size_t(token + last) * stride, 0, 1);
        }
    }
}

static std::vector<uint8_t> make_input(const matrix & weights, int tokens, std::mt19937 & rng) {
    const size_t stride = ggml_row_size(GGML_TYPE_Q8_0, weights.columns);
    std::vector<uint8_t> input(stride * tokens);
    std::vector<float> floats(weights.columns);
    std::normal_distribution<float> sample(0.0f, 0.1f);
    for (int token = 0; token < tokens; ++token) {
        for (auto & value : floats) { value = sample(rng); }
        quantize_row_q8_0(floats.data(), input.data() + size_t(token) * stride, weights.columns);
    }
    return input;
}

static uint64_t check_column_blocks(const matrix & weights, int tokens, std::mt19937 & rng) {
    const size_t input_stride = ggml_row_size(GGML_TYPE_Q8_0, weights.columns);
    const size_t weight_stride = ggml_row_size(GGML_TYPE_MXFP4, weights.columns);
    const auto input = make_input(weights, tokens, rng);
    std::vector<float> reference(size_t(tokens) * weights.rows);
    std::vector<float> blocked(reference.size());
    bool nonzero = false;
    auto compute = [&](int row, int first, int last, std::vector<float> & output) {
        const void * weight = weights.bytes.data() + size_t(row) * weight_stride;
        for (int token = first; token < last; token += 4) {
            const int count = std::min(4, last - token);
            float values[4] = {};
            dot_columns(weights, input.data(), input_stride, weight, token, count, values);
            for (int offset = 0; offset < count; ++offset) {
                output[size_t(token + offset) * weights.rows + row] = values[offset];
                nonzero |= values[offset] != 0.0f;
            }
        }
    };
    for (int row = 0; row < weights.rows; ++row) { compute(row, 0, tokens, reference); }
    require(nonzero, "MXFP4 column-block outputs are all zero");
    for (int block : { 16, 32, 64, 128, 256 }) {
        for (int first = 0; first < tokens; first += block) {
            const int last = std::min(tokens, first + block);
            for (int row = 0; row < weights.rows; ++row) { compute(row, first, last, blocked); }
        }
        if (std::memcmp(reference.data(), blocked.data(), reference.size() * sizeof(float))) {
            std::fprintf(stderr, "column-block mismatch columns=%d rows=%d tokens=%d block=%d\n",
                    weights.columns, weights.rows, tokens, block);
            throw std::runtime_error("MXFP4 column-block output differs bitwise");
        }
    }
    return reference.size();
}

static double bench_cache(const matrix & weights, int tokens, int block) {
    const size_t stride = ggml_row_size(GGML_TYPE_Q8_0, weights.columns);
    const size_t weight_stride = ggml_row_size(GGML_TYPE_MXFP4, weights.columns);
    std::mt19937 rng(20260924u + unsigned(weights.columns) * 13u + unsigned(tokens));
    const auto input = make_input(weights, tokens, rng);
    volatile float checksum = 0;
    const int rows = 16;
    int repeats = 0;
    const auto start = std::chrono::steady_clock::now();
    do {
        auto compute = [&](int row, int first, int last) {
            const void * weight = weights.bytes.data() + size_t(row) * weight_stride;
            for (int token = first; token < last; token += 4) {
                float values[4];
                ggml_vec_dot_mxfp4_q8_0_4x(weights.columns, values, weight,
                        input.data() + size_t(token) * stride,
                        input.data() + size_t(token + 1) * stride,
                        input.data() + size_t(token + 2) * stride,
                        input.data() + size_t(token + 3) * stride);
                checksum = checksum + values[0];
            }
        };
        if (block == 0) {
            for (int row = 0; row < rows; ++row) { compute(row, 0, tokens); }
        } else {
            for (int first = 0; first < tokens; first += block) {
                const int last = std::min(tokens, first + block);
                for (int row = 0; row < rows; ++row) { compute(row, first, last); }
            }
        }
        repeats++;
    } while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < 0.15);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    require(checksum != 0.0f, "MXFP4 cache benchmark outputs are all zero");
    return 1e9 * seconds / (double(repeats) * rows * tokens);
}

int main(int argc, char ** argv) {
    try {
        require(argc == 2 || (argc == 3 && (!std::strcmp(argv[2], "--bench") ||
                !std::strcmp(argv[2], "--cache-bench"))),
                "usage: test-ssd-mxfp4-gemm EXPERT_PACK [--bench|--cache-bench]");
        ggml_cpu_init();
        const matrix up = load_matrix(argv[1], "up", 5120, 2304);
        const matrix gate = load_matrix(argv[1], "gate", 5120, 2304);
        const matrix down = load_matrix(argv[1], "down", 2304, 5120);
        std::mt19937 rng(20260924);
        if (argc == 3 && !std::strcmp(argv[2], "--cache-bench")) {
            for (int role = 0; role < 2; ++role) {
                const char * name = role == 0 ? "up" : "down";
                const matrix & shape = role == 0 ? up : down;
                for (int tokens : { 16, 32, 64, 128, 256, 512, 1024, 2048 }) {
                    for (int block : { 16, 32, 64, 128, 256 }) {
                        const double row1 = bench_cache(shape, tokens, 0);
                        const double blocked1 = bench_cache(shape, tokens, block);
                        const double blocked2 = bench_cache(shape, tokens, block);
                        const double row2 = bench_cache(shape, tokens, 0);
                        const double row = (row1 + row2) / 2;
                        const double blocked = (blocked1 + blocked2) / 2;
                        std::printf("MXFP4_CACHE_BENCH role=%s m=%d block=%d row_ns=%.3f blocked_ns=%.3f speedup=%.4f row_spread=%.4f blocked_spread=%.4f\n",
                                name, tokens, block, row, blocked, row / blocked,
                                std::abs(row1 - row2) / row, std::abs(blocked1 - blocked2) / blocked);
                        std::fflush(stdout);
                    }
                }
            }
            return 0;
        }
        if (argc == 3) {
            for (int role = 0; role < 2; ++role) {
                const char * name = role == 0 ? "up" : "down";
                const matrix & shape = role == 0 ? up : down;
                for (int tokens : { 16, 32, 64, 128, 256, 512, 1024, 2048 }) {
                    const double two1 = bench(shape, tokens, 2);
                    const double four1 = bench(shape, tokens, 4);
                    const double four2 = bench(shape, tokens, 4);
                    const double two2 = bench(shape, tokens, 2);
                    const double two = (two1 + two2) / 2;
                    const double four = (four1 + four2) / 2;
                    std::printf("MXFP4_BENCH role=%s m=%d two_ns=%.3f four_ns=%.3f speedup=%.4f two_spread=%.4f four_spread=%.4f\n",
                            name, tokens, two, four, two / four,
                            std::abs(two1 - two2) / two, std::abs(four1 - four2) / four);
                }
            }
            return 0;
        }
        uint64_t total = 0;
        for (int tokens : { 1, 2, 3, 4, 5, 8, 64, 512, 2048 }) {
            const uint64_t compared = check(up, tokens, rng) +
                check(gate, tokens, rng) + check(down, tokens, rng);
            const uint64_t reordered = check_column_blocks(up, tokens, rng) +
                check_column_blocks(gate, tokens, rng) + check_column_blocks(down, tokens, rng);
            require(reordered == compared, "column-block test shape mismatch");
            std::printf("MXFP4_GEMM_BITWISE_OK tokens=%d outputs=%llu\n",
                tokens, (unsigned long long) compared);
            total += compared;
        }
        std::printf("MXFP4_GEMM_TEST_OK outputs=%llu\n", (unsigned long long) total);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "MXFP4_GEMM_TEST_FAILED: %s\n", error.what());
        return 1;
    }
}
