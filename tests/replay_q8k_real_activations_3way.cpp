#define Q8K_BENCH_ONLY
#include "test_q8k_kernels.cpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <limits>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct Mapping {
    int fd = -1;
    size_t size = 0;
    const uint8_t* data = nullptr;

    explicit Mapping(const std::string& path) {
        fd = open(path.c_str(), O_RDONLY);
        if (fd < 0)
            throw std::runtime_error("Cannot open " + path);

        struct stat st {};
        if (fstat(fd, &st) != 0 || st.st_size <= 0)
            throw std::runtime_error("Invalid file size: " + path);

        size = static_cast<size_t>(st.st_size);
        void* p = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);

        if (p == MAP_FAILED) {
            data = nullptr;
            throw std::runtime_error("mmap failed: " + path);
        }

        data = static_cast<const uint8_t*>(p);
    }

    ~Mapping() {
        if (data) munmap(const_cast<uint8_t*>(data), size);
        if (fd >= 0) close(fd);
    }

    Mapping(const Mapping&) = delete;
    Mapping& operator=(const Mapping&) = delete;
};

static uint64_t parity_blocks = 0;
static uint64_t parity_scale_mismatches = 0;
static uint64_t parity_qs_mismatches = 0;
static uint64_t parity_bsum_mismatches = 0;

struct Record {
    int id;
    std::string operation;
    int type;
    int k;
    int nb;
    int rows;
    int row_bytes;
    uint64_t activation_offset;
    uint64_t weight_offset;
};

static std::vector<Record> load_records(
    const std::string& path,
    int limit
) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("Cannot open descriptor file: " + path);

    std::vector<Record> records;
    std::string line;

    while (std::getline(in, line)) {
        if (line.empty()) continue;

        std::istringstream ss(line);
        Record r {};

        if (!(ss >> r.id
                 >> r.operation
                 >> r.type
                 >> r.k
                 >> r.nb
                 >> r.rows
                 >> r.row_bytes
                 >> r.activation_offset
                 >> r.weight_offset)) {
            throw std::runtime_error(
                "Malformed descriptor at line " +
                std::to_string(records.size() + 1)
            );
        }

        if ((r.type != 13 && r.type != 14) ||
            r.k <= 0 || r.k % 256 != 0 ||
            r.nb <= 0 || r.nb > 64 ||
            r.rows <= 0 || r.row_bytes <= 0) {
            throw std::runtime_error(
                "Invalid descriptor dimensions at record " +
                std::to_string(r.id)
            );
        }

        records.push_back(r);
        if (limit > 0 && static_cast<int>(records.size()) >= limit)
            break;
    }

    return records;
}

static void run_baseline(
    const Record& r,
    const uint8_t* weights,
    const float* activation,
    int rows,
    std::vector<float>& output
) {
    const int blocks = r.k / 256;
    const int block_bytes = r.type == 13 ? 176 : 210;

    output.assign(static_cast<size_t>(r.nb) * rows, 0.0f);

#pragma omp parallel
    {
        float tmp[256];

#pragma omp for schedule(dynamic, 16)
        for (int row = 0; row < rows; ++row) {
            const uint8_t* wr =
                weights + static_cast<size_t>(row) * r.row_bytes;

            float acc[64];
            for (int t = 0; t < r.nb; ++t)
                acc[t] = 0.0f;

            for (int b = 0; b < blocks; ++b) {
                const uint8_t* wb =
                    wr + static_cast<size_t>(b) * block_bytes;

                if (r.type == 13)
                    dq_q5_k(wb, tmp);
                else
                    dq_q6_k(wb, tmp);

                for (int t = 0; t < r.nb; ++t) {
                    const float* xb =
                        activation +
                        static_cast<size_t>(t) * r.k +
                        static_cast<size_t>(b) * 256;

                    acc[t] += dotf(tmp, xb, 256);
                }
            }

            for (int t = 0; t < r.nb; ++t)
                output[static_cast<size_t>(t) * rows + row] = acc[t];
        }
    }
}


// Q8_K activation quantization followed by FP32 dequantized dot.
// This isolates activation quantization from the integer dot kernel.
static void run_q8_dequant_reference(
    const Record& r,
    const uint8_t* weights,
    const float* activation,
    int rows,
    std::vector<float>& output
) {
    const int blocks = r.k / 256;
    const int block_bytes = r.type == 13 ? 176 : 210;

    std::vector<float> q8_float(
        static_cast<size_t>(r.nb) * r.k
    );

    for (int t = 0; t < r.nb; ++t) {
        for (int b = 0; b < blocks; ++b) {
            const float* xb =
                activation +
                static_cast<size_t>(t) * r.k +
                static_cast<size_t>(b) * 256;

            Q8K q{};
            quantize_q8k(xb, q);

            float* dst =
                q8_float.data() +
                static_cast<size_t>(t) * r.k +
                static_cast<size_t>(b) * 256;

            for (int i = 0; i < 256; ++i)
                dst[i] = q8_value(q, i);
        }
    }

    output.assign(static_cast<size_t>(r.nb) * rows, 0.0f);

#pragma omp parallel
    {
        float tmp[256];

#pragma omp for schedule(dynamic, 16)
        for (int row = 0; row < rows; ++row) {
            const uint8_t* wr =
                weights + static_cast<size_t>(row) * r.row_bytes;

            float acc[64];
            for (int t = 0; t < r.nb; ++t)
                acc[t] = 0.0f;

            for (int b = 0; b < blocks; ++b) {
                const uint8_t* wb =
                    wr + static_cast<size_t>(b) * block_bytes;

                if (r.type == 13)
                    dq_q5_k(wb, tmp);
                else
                    dq_q6_k(wb, tmp);

                for (int t = 0; t < r.nb; ++t) {
                    const float* xb =
                        q8_float.data() +
                        static_cast<size_t>(t) * r.k +
                        static_cast<size_t>(b) * 256;

                    acc[t] += dotf(tmp, xb, 256);
                }
            }

            for (int t = 0; t < r.nb; ++t)
                output[static_cast<size_t>(t) * rows + row] = acc[t];
        }
    }
}

static void run_candidate(
    const Record& r,
    const uint8_t* weights,
    const float* activation,
    int rows,
    std::vector<float>& output
) {
    const int blocks = r.k / 256;
    const int block_bytes = r.type == 13 ? 176 : 210;

    std::vector<Q8K> q8(static_cast<size_t>(r.nb) * blocks);

    // Quantize each token and each 256-element block exactly once.
    for (int t = 0; t < r.nb; ++t) {
        for (int b = 0; b < blocks; ++b) {
            const float* xb =
                activation +
                static_cast<size_t>(t) * r.k +
                static_cast<size_t>(b) * 256;

            Q8K expected{};
            Q8K& actual = q8[static_cast<size_t>(t) * blocks + b];

            quantize_q8k(xb, actual);
            ggml_reference_quantize_q8k(xb, expected);

            ++parity_blocks;
            if (std::memcmp(&actual.d, &expected.d, sizeof(float)) != 0)
                ++parity_scale_mismatches;
            if (std::memcmp(actual.qs, expected.qs, sizeof(actual.qs)) != 0)
                ++parity_qs_mismatches;
            if (std::memcmp(actual.bsums, expected.bsums,
                            sizeof(actual.bsums)) != 0)
                ++parity_bsum_mismatches;
        }
    }

    output.assign(static_cast<size_t>(r.nb) * rows, 0.0f);

#pragma omp parallel for schedule(dynamic, 16)
    for (int row = 0; row < rows; ++row) {
        const uint8_t* wr =
            weights + static_cast<size_t>(row) * r.row_bytes;

        float acc[64];
        for (int t = 0; t < r.nb; ++t)
            acc[t] = 0.0f;

        for (int b = 0; b < blocks; ++b) {
            const uint8_t* wb =
                wr + static_cast<size_t>(b) * block_bytes;

            for (int t = 0; t < r.nb; ++t) {
                const Q8K& qb =
                    q8[static_cast<size_t>(t) * blocks + b];

                if (r.type == 13)
                    acc[t] += dot_q5k_q8k(wb, qb);
                else
                    acc[t] += dot_q6k_q8k(wb, qb);
            }
        }

        for (int t = 0; t < r.nb; ++t)
            output[static_cast<size_t>(t) * rows + row] = acc[t];
    }
}

int main(int argc, char** argv) {
    if (argc != 7) {
        std::fprintf(
            stderr,
            "Usage: %s weights.bin activations.f32 descriptors.tsv "
            "record_limit max_rows threads\n"
            "Use 0 for no record or row limit.\n",
            argv[0]
        );
        return 2;
    }

    try {
        const int record_limit = std::atoi(argv[4]);
        const int max_rows = std::atoi(argv[5]);
        const int threads = std::atoi(argv[6]);

        if (record_limit < 0 || max_rows < 0 || threads < 1) {
            throw std::runtime_error("Invalid limit or thread count");
        }

        Mapping weights(argv[1]);
        Mapping activations(argv[2]);

        const auto records = load_records(argv[3], record_limit);

        if (records.empty())
            throw std::runtime_error("No replay records loaded");

        omp_set_dynamic(0);
        omp_set_num_threads(threads);
        init_types();

        std::printf("===== Q8_K REAL ACTIVATION REPLAY =====\n");
        std::printf("Weight file bytes: %zu\n", weights.size);
        std::printf("Activation file bytes: %zu\n", activations.size);
        std::printf("Records selected: %zu\n", records.size());
        std::printf("Maximum rows per record: %d\n", max_rows);
        std::printf("OpenMP threads: %d\n\n", threads);

        double total_error_sq = 0.0;
        double total_reference_sq = 0.0;
        double total_max_abs = 0.0;
        uint64_t total_outputs = 0;
        uint64_t total_macs = 0;
        uint64_t errors = 0;

        for (const Record& r : records) {
            const int block_bytes = r.type == 13 ? 176 : 210;
            const uint64_t weight_bytes =
                static_cast<uint64_t>(r.rows) * r.row_bytes;
            const uint64_t activation_bytes =
                static_cast<uint64_t>(r.nb) * r.k * sizeof(float);

            if (r.row_bytes != (r.k / 256) * block_bytes)
                throw std::runtime_error(
                    "Row-byte mismatch at record " + std::to_string(r.id)
                );

            if (r.weight_offset > weights.size ||
                weight_bytes > weights.size - r.weight_offset)
                throw std::runtime_error(
                    "Weight bounds error at record " + std::to_string(r.id)
                );

            if (r.activation_offset > activations.size ||
                activation_bytes > activations.size - r.activation_offset)
                throw std::runtime_error(
                    "Activation bounds error at record " +
                    std::to_string(r.id)
                );

            const int rows =
                max_rows > 0 ? std::min(max_rows, r.rows) : r.rows;

            const uint8_t* wp = weights.data + r.weight_offset;
            const float* xp = reinterpret_cast<const float*>(
                activations.data + r.activation_offset
            );

            if (reinterpret_cast<uintptr_t>(xp) % alignof(float) != 0)
                throw std::runtime_error("Unaligned activation offset");

            std::vector<float> baseline;
            std::vector<float> q8_reference;
            std::vector<float> candidate;

            run_baseline(r, wp, xp, rows, baseline);
            run_q8_dequant_reference(r, wp, xp, rows, q8_reference);
            run_candidate(r, wp, xp, rows, candidate);

            double err_sq = 0.0;
            double activation_quant_error_sq = 0.0;
            double integer_kernel_error_sq = 0.0;
            double q8_reference_sq = 0.0;
            double ref_sq = 0.0;
            double candidate_sq = 0.0;
            double cross = 0.0;
            double max_abs = 0.0;
            std::vector<double> absolute_errors;
            absolute_errors.reserve(baseline.size());

            for (size_t i = 0; i < baseline.size(); ++i) {
                const double ref = baseline[i];
                const double q8ref = q8_reference[i];
                const double got = candidate[i];

                const double diff = std::fabs(ref - got);
                const double activation_diff = ref - q8ref;
                const double kernel_diff = q8ref - got;

                if (!std::isfinite(ref) ||
                    !std::isfinite(q8ref) ||
                    !std::isfinite(got))
                    ++errors;

                activation_quant_error_sq +=
                    activation_diff * activation_diff;
                integer_kernel_error_sq +=
                    kernel_diff * kernel_diff;
                q8_reference_sq += q8ref * q8ref;

                max_abs = std::max(max_abs, diff);
                absolute_errors.push_back(diff);
                err_sq += diff * diff;
                ref_sq += ref * ref;
                candidate_sq += got * got;
                cross += ref * got;
            }

            const double normalized =
                std::sqrt(err_sq / std::max(ref_sq, 1e-30)) * 100.0;
            const double activation_quant_normalized =
                std::sqrt(activation_quant_error_sq /
                          std::max(ref_sq, 1e-30)) * 100.0;
            const double integer_kernel_normalized =
                std::sqrt(integer_kernel_error_sq /
                          std::max(q8_reference_sq, 1e-30)) * 100.0;
            const double error_rms =
                std::sqrt(err_sq / std::max<size_t>(1, baseline.size()));
            const double reference_rms =
                std::sqrt(ref_sq / std::max<size_t>(1, baseline.size()));

            std::sort(absolute_errors.begin(), absolute_errors.end());
            const size_t p99_index = absolute_errors.empty() ? 0 :
                std::min(absolute_errors.size() - 1,
                         static_cast<size_t>(0.99 * absolute_errors.size()));
            const double p99_abs = absolute_errors.empty() ? 0.0 :
                absolute_errors[p99_index];

            const double cosine =
                cross / std::sqrt(std::max(ref_sq * candidate_sq, 1e-60));

            total_error_sq += err_sq;
            total_reference_sq += ref_sq;
            total_max_abs = std::max(total_max_abs, max_abs);
            total_outputs += baseline.size();
            total_macs += static_cast<uint64_t>(rows) * r.k * r.nb;

            std::printf(
                "record=%5d op=%-16s type=Q%d K=%4d nb=%2d "
                "rows=%4d/%4d outputs=%8zu max_abs=%10.6g "
                "err_RMS=%9.6g ref_RMS=%9.6g p99_abs=%9.6g "
                "cosine=% .8f A_vs_B=%8.5f%% "
                "B_vs_C=%8.5f%% A_vs_C=%8.5f%%\n",
                r.id,
                r.operation.c_str(),
                r.type == 13 ? 5 : 6,
                r.k,
                r.nb,
                rows,
                r.rows,
                baseline.size(),
                max_abs,
                error_rms,
                reference_rms,
                p99_abs,
                cosine,
                activation_quant_normalized,
                integer_kernel_normalized,
                normalized
            );
        }

        const double aggregate_normalized =
            std::sqrt(
                total_error_sq /
                std::max(total_reference_sq, 1e-30)
            ) * 100.0;

        std::printf("\n===== AGGREGATE CORRECTNESS =====\n");
        std::printf("Records executed: %zu\n", records.size());
        std::printf("Output values compared: %llu\n",
                    static_cast<unsigned long long>(total_outputs));
        std::printf("Matrix MACs executed: %llu\n",
                    static_cast<unsigned long long>(total_macs));
        std::printf("Maximum absolute error: %.9g\n", total_max_abs);
        std::printf("Aggregate normalized RMS: %.8f%%\n",
                    aggregate_normalized);
        std::printf("Non-finite output pairs: %llu\n",
                    static_cast<unsigned long long>(errors));

        std::printf("\n===== REAL ACTIVATION Q8_K PARITY =====\n");
        std::printf("Blocks checked: %llu\n",
                    static_cast<unsigned long long>(parity_blocks));
        std::printf("Scale mismatches: %llu\n",
                    static_cast<unsigned long long>(parity_scale_mismatches));
        std::printf("Quantized-value mismatches: %llu\n",
                    static_cast<unsigned long long>(parity_qs_mismatches));
        std::printf("Block-sum mismatches: %llu\n",
                    static_cast<unsigned long long>(parity_bsum_mismatches));

        const bool parity_pass =
            parity_scale_mismatches == 0 &&
            parity_qs_mismatches == 0 &&
            parity_bsum_mismatches == 0;

        std::printf("Q8_K QUANTIZER PARITY: %s\n",
                    parity_pass ? "PASS" : "FAIL");

        std::printf(
            "\nREPLAY CORRECTNESS: %s\n",
            errors == 0 ? "COMPLETED - FINITE OUTPUTS" : "FAIL"
        );

        return (errors == 0 && parity_pass) ? 0 : 1;
    }
    catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
