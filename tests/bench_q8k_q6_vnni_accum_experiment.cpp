#define Q8K_BENCH_ONLY
#include "test_q8k_kernels.cpp"
#include <immintrin.h>

#include <algorithm>
#include <chrono>
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

            quantize_q8k(
                xb,
                q8[static_cast<size_t>(t) * blocks + b]
            );
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


// ---------------------------------------------------------------------------
// Predecoded Q5_K / Q6_K weight blocks.
// Packed weights are decoded once per output-row/block and reused for every
// token in the batch. Q8_K quantization and integer accumulation are unchanged.
// ---------------------------------------------------------------------------

struct PredecodedQ5 {
    int8_t q[256];
    uint8_t scales[8];
    uint8_t mins[8];
    float d;
    float dmin;
};

struct PredecodedQ6 {
    int8_t q[256];
    int8_t scales[16];
    float d;
};

static inline void decode_q5_predecoded(
    const uint8_t* b, PredecodedQ5& out
) {
    out.d = rd_f16(b);
    out.dmin = rd_f16(b + 2);

    const uint8_t* packed_scales = b + 4;
    const uint8_t* qh = b + 16;
    const uint8_t* ql = b + 48;

    for (int group = 0; group < 8; ++group) {
        scale_min_k4(
            group, packed_scales,
            &out.scales[group], &out.mins[group]
        );

        const uint8_t* ql_group = ql + (group / 2) * 32;
        const uint8_t high_mask = static_cast<uint8_t>(1u << group);

        for (int j = 0; j < 32; ++j) {
            const int low = (group & 1)
                ? (ql_group[j] >> 4)
                : (ql_group[j] & 0x0f);

            const int high = (qh[j] & high_mask) ? 16 : 0;
            out.q[group * 32 + j] =
                static_cast<int8_t>(low + high);
        }
    }
}


static inline float dot_q5_predecoded(
    const PredecodedQ5& w, const Q8K& x
) {
    float result = 0.0f;

    for (int group = 0; group < 8; ++group) {
        const __m256i qw = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(
                w.q + group * 32
            )
        );

        const __m256i qx = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(
                x.qs + group * 32
            )
        );

        const __m256i acc = _mm256_dpbusd_epi32(
            _mm256_setzero_si256(), qw, qx
        );

        alignas(32) int32_t lanes[8];
        _mm256_store_si256(
            reinterpret_cast<__m256i*>(lanes), acc
        );

        int32_t dot = 0;
        for (int i = 0; i < 8; ++i)
            dot += lanes[i];

        const int activation_sum =
            x.bsums[group * 2] + x.bsums[group * 2 + 1];

        result += (w.d * x.d *
                   static_cast<float>(w.scales[group])) *
                  static_cast<float>(dot);

        result -= (w.dmin * x.d *
                   static_cast<float>(w.mins[group])) *
                  static_cast<float>(activation_sum);
    }

    return result;
}

static inline void decode_q6_predecoded(
    const uint8_t* b, PredecodedQ6& out
) {
    const uint8_t* ql = b;
    const uint8_t* qh = b + 128;
    const int8_t* scales =
        reinterpret_cast<const int8_t*>(b + 192);

    out.d = rd_f16(b + 208);

    for (int half = 0; half < 2; ++half) {
        const uint8_t* ql_half = ql + half * 64;
        const uint8_t* qh_half = qh + half * 32;

        for (int j = 0; j < 32; ++j) {
            const uint8_t h = qh_half[j];

            out.q[half * 128 + j] =
                static_cast<int8_t>(
                    static_cast<int>((ql_half[j] & 0x0f) |
                        (((h >> 0) & 3) << 4)) - 32
                );

            out.q[half * 128 + j + 32] =
                static_cast<int8_t>(
                    static_cast<int>((ql_half[j + 32] & 0x0f) |
                        (((h >> 2) & 3) << 4)) - 32
                );

            out.q[half * 128 + j + 64] =
                static_cast<int8_t>(
                    static_cast<int>((ql_half[j] >> 4) |
                        (((h >> 4) & 3) << 4)) - 32
                );

            out.q[half * 128 + j + 96] =
                static_cast<int8_t>(
                    static_cast<int>((ql_half[j + 32] >> 4) |
                        (((h >> 6) & 3) << 4)) - 32
                );
        }
    }

    for (int i = 0; i < 16; ++i)
        out.scales[i] = scales[i];
}



static inline float dot_q6_predecoded(
    const PredecodedQ6& w, const Q8K& x
) {
    float result = 0.0f;

    const __m256i bias = _mm256_set1_epi8(
        static_cast<char>(0x80)
    );

    // Each iteration processes two adjacent 16-element Q6 groups.
    for (int pair = 0; pair < 8; ++pair) {
        const int group = pair * 2;

        const __m256i signed_weights = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(
                w.q + group * 16
            )
        );

        const __m256i unsigned_weights =
            _mm256_xor_si256(signed_weights, bias);

        const __m256i activations = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(
                x.qs + group * 16
            )
        );

        const __m256i acc = _mm256_dpbusd_epi32(
            _mm256_setzero_si256(),
            unsigned_weights,
            activations
        );

        alignas(32) int32_t lanes[8];
        _mm256_store_si256(
            reinterpret_cast<__m256i*>(lanes), acc
        );

        // First four lanes correspond to the first 16-element group.
        int32_t unsigned_dot_0 = 0;
        for (int i = 0; i < 4; ++i)
            unsigned_dot_0 += lanes[i];

        // Last four lanes correspond to the second group.
        int32_t unsigned_dot_1 = 0;
        for (int i = 4; i < 8; ++i)
            unsigned_dot_1 += lanes[i];

        const int32_t signed_dot_0 =
            unsigned_dot_0 - 128 * x.bsums[group];

        const int32_t signed_dot_1 =
            unsigned_dot_1 - 128 * x.bsums[group + 1];

        // Keep the original group-specific scales and arithmetic.
        result +=
            (w.d * x.d *
             static_cast<float>(w.scales[group])) *
            static_cast<float>(signed_dot_0);

        result +=
            (w.d * x.d *
             static_cast<float>(w.scales[group + 1])) *
            static_cast<float>(signed_dot_1);
    }

    return result;
}




// Experimental Q6 kernel: AVX2 horizontal reduction of VNNI lanes.
// Preserves the original VNNI products, signed correction and group scales.
static inline float dot_q6_predecoded_hadd(
    const PredecodedQ6& w, const Q8K& x
) {
    float result = 0.0f;

    const __m256i bias = _mm256_set1_epi8(
        static_cast<char>(0x80)
    );

    for (int pair = 0; pair < 8; ++pair) {
        const int group = pair * 2;

        const __m256i signed_weights = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(
                w.q + group * 16
            )
        );

        const __m256i unsigned_weights =
            _mm256_xor_si256(signed_weights, bias);

        const __m256i activations = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(
                x.qs + group * 16
            )
        );

        const __m256i acc = _mm256_dpbusd_epi32(
            _mm256_setzero_si256(),
            unsigned_weights,
            activations
        );

        // Reduce each 128-bit half independently.
        const __m256i h1 = _mm256_hadd_epi32(acc, acc);
        const __m256i h2 = _mm256_hadd_epi32(h1, h1);

        const int32_t unsigned_dot_0 =
            _mm_cvtsi128_si32(_mm256_castsi256_si128(h2));

        const int32_t unsigned_dot_1 =
            _mm_cvtsi128_si32(_mm256_extracti128_si256(h2, 1));

        const int32_t signed_dot_0 =
            unsigned_dot_0 - 128 * x.bsums[group];

        const int32_t signed_dot_1 =
            unsigned_dot_1 - 128 * x.bsums[group + 1];

        result +=
            (w.d * x.d *
             static_cast<float>(w.scales[group])) *
            static_cast<float>(signed_dot_0);

        result +=
            (w.d * x.d *
             static_cast<float>(w.scales[group + 1])) *
            static_cast<float>(signed_dot_1);
    }

    return result;
}

static void run_candidate_predecoded(
    const Record& r,
    const uint8_t* weights,
    const float* activation,
    int rows,
    std::vector<float>& output
) {
    const int blocks = r.k / 256;
    const int block_bytes = r.type == 13 ? 176 : 210;

    std::vector<Q8K> q8(static_cast<size_t>(r.nb) * blocks);

    // Keep the exact same activation quantization path as the original.
    for (int t = 0; t < r.nb; ++t) {
        for (int b = 0; b < blocks; ++b) {
            const float* xb =
                activation +
                static_cast<size_t>(t) * r.k +
                static_cast<size_t>(b) * 256;

            quantize_q8k(
                xb,
                q8[static_cast<size_t>(t) * blocks + b]
            );
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

            if (r.type == 13) {
                PredecodedQ5 decoded {};
                decode_q5_predecoded(wb, decoded);

                for (int t = 0; t < r.nb; ++t) {
                    const Q8K& qb =
                        q8[static_cast<size_t>(t) * blocks + b];
                    acc[t] += dot_q5_predecoded(decoded, qb);
                }
            } else {
                PredecodedQ6 decoded {};
                decode_q6_predecoded(wb, decoded);

                for (int t = 0; t < r.nb; ++t) {
                    const Q8K& qb =
                        q8[static_cast<size_t>(t) * blocks + b];
                    acc[t] += dot_q6_predecoded_hadd(decoded, qb);
                }
            }
        }

        for (int t = 0; t < r.nb; ++t)
            output[static_cast<size_t>(t) * rows + row] = acc[t];
    }
}


static double median_us(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    if (values.empty()) return 0.0;
    const size_t n = values.size();
    return n % 2 ? values[n / 2]
                 : (values[n / 2 - 1] + values[n / 2]) * 0.5;
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
        constexpr int BENCH_REPEATS = 5;
        double baseline_total_us = 0.0;
        double candidate_total_us = 0.0;

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
            std::vector<float> candidate;

            run_baseline(r, wp, xp, rows, baseline);
            run_candidate(r, wp, xp, rows, candidate);
            const std::vector<float> existing_candidate = candidate;
            run_candidate_predecoded(r, wp, xp, rows, candidate);

            double parity_max_abs = 0.0;
            uint64_t parity_mismatches = 0;
            for (size_t i = 0; i < candidate.size(); ++i) {
                const double delta = std::fabs(
                    static_cast<double>(existing_candidate[i]) - candidate[i]
                );
                parity_max_abs = std::max(parity_max_abs, delta);
                if (delta > 2.0e-5) ++parity_mismatches;
            }
            std::printf(
                "predecode_parity_max_abs=%.9g mismatches=%llu\\n",
                parity_max_abs,
                static_cast<unsigned long long>(parity_mismatches)
            );
            if (parity_mismatches != 0) {
                throw std::runtime_error(
                    "Predecoded candidate differs from existing Q8_K candidate"
                );
            }

            std::vector<double> bt, ct;
            bt.reserve(BENCH_REPEATS);
            ct.reserve(BENCH_REPEATS);

            using Clock = std::chrono::steady_clock;

            for (int rep = 0; rep < BENCH_REPEATS; ++rep) {
                if ((rep & 1) == 0) {
                    auto a = Clock::now();
                    run_baseline(r, wp, xp, rows, baseline);
                    auto b = Clock::now();
                    bt.push_back(
                        std::chrono::duration<double, std::micro>(b-a).count()
                    );

                    a = Clock::now();
                    run_candidate_predecoded(r, wp, xp, rows, candidate);
                    b = Clock::now();
                    ct.push_back(
                        std::chrono::duration<double, std::micro>(b-a).count()
                    );
                } else {
                    auto a = Clock::now();
                    run_candidate_predecoded(r, wp, xp, rows, candidate);
                    auto b = Clock::now();
                    ct.push_back(
                        std::chrono::duration<double, std::micro>(b-a).count()
                    );

                    a = Clock::now();
                    run_baseline(r, wp, xp, rows, baseline);
                    b = Clock::now();
                    bt.push_back(
                        std::chrono::duration<double, std::micro>(b-a).count()
                    );
                }
            }

            const double baseline_median = median_us(bt);
            const double candidate_median = median_us(ct);
            const double speedup = baseline_median /
                std::max(candidate_median, 1e-30);

            baseline_total_us += baseline_median;
            candidate_total_us += candidate_median;

            double err_sq = 0.0;
            double ref_sq = 0.0;
            double max_abs = 0.0;

            for (size_t i = 0; i < baseline.size(); ++i) {
                const double ref = baseline[i];
                const double got = candidate[i];
                const double diff = std::fabs(ref - got);

                if (!std::isfinite(ref) || !std::isfinite(got))
                    ++errors;

                max_abs = std::max(max_abs, diff);
                err_sq += diff * diff;
                ref_sq += ref * ref;
            }

            const double normalized =
                std::sqrt(err_sq / std::max(ref_sq, 1e-30)) * 100.0;

            total_error_sq += err_sq;
            total_reference_sq += ref_sq;
            total_max_abs = std::max(total_max_abs, max_abs);
            total_outputs += baseline.size();
            total_macs += static_cast<uint64_t>(rows) * r.k * r.nb;

            std::printf(
                "record=%5d op=%-16s type=Q%d K=%4d nb=%2d "
                "rows=%4d/%4d outputs=%8zu max_abs=%10.6g "
                "normalized_RMS=%8.5f%% "
                "baseline_us=%10.2f candidate_us=%10.2f speedup=%6.3fx\n",
                r.id,
                r.operation.c_str(),
                r.type == 13 ? 5 : 6,
                r.k,
                r.nb,
                rows,
                r.rows,
                baseline.size(),
                max_abs,
                normalized,
                baseline_median,
                candidate_median,
                speedup
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

        std::printf("\n===== PERFORMANCE SUMMARY =====\n");
        std::printf("Repetitions per path: %d\n", BENCH_REPEATS);
        std::printf("Sum baseline medians: %.2f us\n", baseline_total_us);
        std::printf("Sum candidate medians: %.2f us\n", candidate_total_us);
        std::printf("Median-sum speedup: %.4fx\n",
                    baseline_total_us / std::max(candidate_total_us, 1e-30));
        std::printf("Candidate timing includes allocation and Q8_K quantization.\n");

        std::printf(
            "\nREPLAY CORRECTNESS: %s\n",
            errors == 0 ? "COMPLETED - FINITE OUTPUTS" : "FAIL"
        );

        return errors == 0 ? 0 : 1;
    }
    catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
