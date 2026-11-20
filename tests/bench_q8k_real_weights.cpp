#define Q8K_BENCH_ONLY
#include "test_q8k_kernels.cpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <stdexcept>
#include <vector>

namespace {

constexpr int ITERS = 30;
constexpr int SAMPLES = 7;

using Clock = std::chrono::steady_clock;

template <typename Fn>
double measure_us(Fn&& fn) {
    auto start = Clock::now();
    for (int i = 0; i < ITERS; ++i) fn();
    auto stop = Clock::now();

    return std::chrono::duration<double, std::micro>(
        stop - start
    ).count() / ITERS;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

void report_error(
    const std::vector<float>& baseline,
    const std::vector<float>& candidate
) {
    double max_abs = 0.0;
    double error_sq = 0.0;
    double reference_sq = 0.0;

    for (size_t i = 0; i < baseline.size(); ++i) {
        double a = baseline[i];
        double b = candidate[i];
        double d = std::fabs(a - b);

        max_abs = std::max(max_abs, d);
        error_sq += d * d;
        reference_sq += a * a;
    }

    double error_rms = std::sqrt(error_sq / baseline.size());
    double reference_rms = std::sqrt(reference_sq / baseline.size());
    double normalized = std::sqrt(
        error_sq / std::max(reference_sq, 1e-30)
    );

    std::printf(
        "  max_abs=%g error_RMS=%g reference_RMS=%g "
        "normalized_RMS=%g%%\n",
        max_abs,
        error_rms,
        reference_rms,
        normalized * 100.0
    );
}

void run_benchmark(
    const std::string& name,
    const std::vector<uint8_t>& weights,
    int rows,
    int row_bytes,
    int type,
    int threads
) {
    const int block_bytes = (type == 13) ? 176 : 210;
    const int blocks = row_bytes / block_bytes;
    const int elements = blocks * K;

    if (row_bytes % block_bytes != 0)
        throw std::runtime_error("Invalid row byte count");

    if (type != 13 && type != 14)
        throw std::runtime_error("Only Q5_K and Q6_K are supported");

    std::mt19937 rng(0xEA11F);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> activation(elements);
    for (float& x : activation) x = dist(rng);

    std::vector<Q8K> q8(blocks);
    std::vector<float> baseline_output(rows);
    std::vector<float> candidate_output(rows);

    auto baseline = [&]() {
#pragma omp parallel
        {
            float tmp[K];

#pragma omp for schedule(dynamic, 16)
            for (int r = 0; r < rows; ++r) {
                float acc = 0.0f;
                const uint8_t* row =
                    weights.data() + static_cast<size_t>(r) * row_bytes;

                for (int b = 0; b < blocks; ++b) {
                    const uint8_t* block =
                        row + static_cast<size_t>(b) * block_bytes;

                    if (type == 13)
                        dq_q5_k(block, tmp);
                    else
                        dq_q6_k(block, tmp);

                    acc += dotf(
                        tmp,
                        activation.data() + static_cast<size_t>(b) * K,
                        K
                    );
                }

                baseline_output[r] = acc;
            }
        }
    };

    auto candidate = [&]() {
        // Include activation quantization in the measured candidate path.
        for (int b = 0; b < blocks; ++b) {
            quantize_q8k(
                activation.data() + static_cast<size_t>(b) * K,
                q8[b]
            );
        }

#pragma omp parallel
        {
#pragma omp for schedule(dynamic, 16)
            for (int r = 0; r < rows; ++r) {
                float acc = 0.0f;
                const uint8_t* row =
                    weights.data() + static_cast<size_t>(r) * row_bytes;

                for (int b = 0; b < blocks; ++b) {
                    const uint8_t* block =
                        row + static_cast<size_t>(b) * block_bytes;

                    if (type == 13)
                        acc += dot_q5k_q8k(block, q8[b]);
                    else
                        acc += dot_q6k_q8k(block, q8[b]);
                }

                candidate_output[r] = acc;
            }
        }
    };

    omp_set_num_threads(threads);
    omp_set_dynamic(0);

    baseline();
    candidate();

    std::vector<double> base_times;
    std::vector<double> candidate_times;

    for (int s = 0; s < SAMPLES; ++s) {
        if (s & 1) {
            candidate_times.push_back(measure_us(candidate));
            base_times.push_back(measure_us(baseline));
        } else {
            base_times.push_back(measure_us(baseline));
            candidate_times.push_back(measure_us(candidate));
        }
    }

    double base_us = median(base_times);
    double q8_us = median(candidate_times);

    baseline();
    candidate();

    std::printf(
        "%-15s rows=%3d blocks=%2d K=%4d threads=%2d "
        "baseline=%9.2f us Q8=%9.2f us speedup=%6.3fx\n",
        name.c_str(), rows, blocks, elements, threads,
        base_us, q8_us, base_us / q8_us
    );

    report_error(baseline_output, candidate_output);
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(
            stderr,
            "Usage: %s <q5|q6> <binary_file> <rows> <row_bytes>\n",
            argv[0]
        );
        return 2;
    }

    const std::string format = argv[1];
    const std::string path = argv[2];
    const int rows = std::atoi(argv[3]);
    const int row_bytes = std::atoi(argv[4]);
    const int type = (format == "q5") ? 13 : (format == "q6") ? 14 : 0;

    if (!type || rows <= 0 || row_bytes <= 0) {
        std::fprintf(stderr, "Invalid format or dimensions\n");
        return 2;
    }

    const size_t expected =
        static_cast<size_t>(rows) * static_cast<size_t>(row_bytes);

    std::vector<uint8_t> weights(expected);
    std::ifstream input(path, std::ios::binary);

    if (!input) {
        std::fprintf(stderr, "Cannot open: %s\n", path.c_str());
        return 1;
    }

    input.read(
        reinterpret_cast<char*>(weights.data()),
        static_cast<std::streamsize>(expected)
    );

    if (input.gcount() != static_cast<std::streamsize>(expected)) {
        std::fprintf(stderr, "Dataset size does not match dimensions\n");
        return 1;
    }

    char extra;
    if (input.read(&extra, 1)) {
        std::fprintf(stderr, "Dataset contains unexpected extra bytes\n");
        return 1;
    }

    init_types();

    std::printf("\n===== REAL GGUF WEIGHT BENCHMARK =====\n");
    std::printf("Dataset: %s\n", path.c_str());
    std::printf("Format: %s\n", format.c_str());
    std::printf("Rows: %d\n", rows);
    std::printf("Row bytes: %d\n", row_bytes);
    std::printf("Activation: deterministic synthetic uniform[-1,1]\n");
    std::printf("Candidate timing includes Q8_K quantization\n");
    std::printf("Iterations/sample: %d, samples: %d\n\n", ITERS, SAMPLES);

    for (int threads : {1, 8}) {
        run_benchmark(
            path, weights, rows, row_bytes, type, threads
        );
    }

    std::printf("\nREAL-WEIGHT BENCHMARK COMPLETE\n");
    return 0;
}
