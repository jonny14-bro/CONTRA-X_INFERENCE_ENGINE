#define Q8K_BENCH_ONLY
#include "test_q8k_kernels.cpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr int BENCH_BLOCKS = 16;
constexpr int BENCH_K = 256;
constexpr int BENCH_ITERS = 20;
constexpr int BENCH_SAMPLES = 5;

using Clock = std::chrono::steady_clock;
using Q5Block = std::array<uint8_t, 176>;
using Q6Block = std::array<uint8_t, 210>;

template <typename Fn>
static double measure_us(Fn&& fn) {
    const auto start = Clock::now();

    for (int i = 0; i < BENCH_ITERS; ++i)
        fn();

    const auto stop = Clock::now();

    return std::chrono::duration<double, std::micro>(
        stop - start
    ).count() / BENCH_ITERS;
}

static double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

static void report_error(
    const char* type,
    int rows,
    const std::vector<float>& baseline,
    const std::vector<float>& candidate
) {
    double max_abs = 0.0;
    double error_sq = 0.0;
    double reference_sq = 0.0;

    for (int r = 0; r < rows; ++r) {
        const double a = baseline[r];
        const double b = candidate[r];
        const double diff = std::fabs(a - b);

        max_abs = std::max(max_abs, diff);
        error_sq += diff * diff;
        reference_sq += a * a;
    }

    const double normalized_rms =
        std::sqrt(error_sq / std::max(reference_sq, 1e-30));

    std::printf(
        "  %-4s rows=%2d max_abs=%10.6g normalized_RMS=%10.6g%%\n",
        type, rows, max_abs, normalized_rms * 100.0
    );
}

template <typename Block, typename MakeBlock,
          typename Decode, typename IntegerDot>
static void benchmark_type(
    const char* name,
    int rows,
    int threads,
    const std::vector<float>& activation,
    std::mt19937& rng,
    MakeBlock make_block,
    Decode decode,
    IntegerDot integer_dot
) {
    std::vector<Block> weights(
        static_cast<size_t>(rows) * BENCH_BLOCKS
    );

    for (auto& block : weights)
        make_block(block, rng);

    std::vector<Q8K> quantized(BENCH_BLOCKS);
    std::vector<float> baseline_output(rows, 0.0f);
    std::vector<float> candidate_output(rows, 0.0f);
    std::vector<float> temporary(BENCH_K);

    auto baseline = [&]() {
#pragma omp parallel
        {
            float tmp[BENCH_K];

#pragma omp for schedule(dynamic, 16)
            for (int r = 0; r < rows; ++r) {
                float acc = 0.0f;

                for (int b = 0; b < BENCH_BLOCKS; ++b) {
                    const Block& block =
                        weights[static_cast<size_t>(r) *
                                BENCH_BLOCKS + b];

                    decode(block.data(), tmp);

                    acc += dotf(
                        tmp,
                        activation.data() +
                            static_cast<size_t>(b) * BENCH_K,
                        BENCH_K
                    );
                }

                baseline_output[r] = acc;
            }
        }
    };

    auto candidate = [&]() {
        // Quantization is included in the timed candidate path.
        // Each activation block is quantized once and reused by all rows.
        for (int b = 0; b < BENCH_BLOCKS; ++b) {
            quantize_q8k(
                activation.data() +
                    static_cast<size_t>(b) * BENCH_K,
                quantized[b]
            );
        }

#pragma omp parallel
        {
#pragma omp for schedule(dynamic, 16)
            for (int r = 0; r < rows; ++r) {
                float acc = 0.0f;

                for (int b = 0; b < BENCH_BLOCKS; ++b) {
                    const Block& block =
                        weights[static_cast<size_t>(r) *
                                BENCH_BLOCKS + b];

                    acc += integer_dot(
                        block.data(),
                        quantized[b]
                    );
                }

                candidate_output[r] = acc;
            }
        }
    };

    omp_set_num_threads(threads);
    omp_set_dynamic(0);

    // Warm-up both implementations.
    baseline();
    candidate();

    std::vector<double> baseline_times;
    std::vector<double> candidate_times;

    for (int sample = 0; sample < BENCH_SAMPLES; ++sample) {
        // Alternate order to reduce systematic ordering bias.
        if (sample & 1) {
            candidate_times.push_back(measure_us(candidate));
            baseline_times.push_back(measure_us(baseline));
        } else {
            baseline_times.push_back(measure_us(baseline));
            candidate_times.push_back(measure_us(candidate));
        }
    }

    const double base_us = median(baseline_times);
    const double q8_us = median(candidate_times);
    const double speedup = base_us / q8_us;

    // Final outputs for numerical comparison.
    baseline();
    candidate();

    std::printf(
        "%-4s rows=%2d threads=%2d "
        "baseline=%9.2f us Q8=%9.2f us speedup=%6.3fx\n",
        name, rows, threads, base_us, q8_us, speedup
    );

    report_error(
        name, rows, baseline_output, candidate_output
    );
}

} // namespace

int main() {
    init_types();

    constexpr int total_elements = BENCH_BLOCKS * BENCH_K;

    std::mt19937 rng(0xEA11F);
    std::uniform_real_distribution<float> activation_dist(-1.0f, 1.0f);

    std::vector<float> activation(total_elements);

    for (float& value : activation)
        value = activation_dist(rng);

    const int row_counts[] = {1, 8, 16, 32, 64};
    const int thread_counts[] = {1, 8};

    std::printf("\n========== Q8_K CPU MICROBENCHMARK ==========\n");
    std::printf("CPU threads tested: 1 and 8\n");
    std::printf("Blocks per row: %d\n", BENCH_BLOCKS);
    std::printf("Elements per row: %d\n", total_elements);
    std::printf("Iterations per sample: %d\n", BENCH_ITERS);
    std::printf("Samples per configuration: %d\n", BENCH_SAMPLES);
    std::printf("OpenMP schedule: dynamic, chunk=16\n");
    std::printf("Candidate timing includes Q8_K activation quantization.\n");
    std::printf("Synthetic packed weights; not a model-level benchmark.\n\n");

    for (int threads : thread_counts) {
        for (int rows : row_counts) {
            std::printf("\n--- threads=%d rows=%d ---\n", threads, rows);

            benchmark_type<Q5Block>(
                "Q5_K", rows, threads, activation, rng,
                [](Q5Block& b, std::mt19937& r) {
                    make_q5_block(b, r);
                },
                [](const uint8_t* b, float* out) {
                    dq_q5_k(b, out);
                },
                [](const uint8_t* b, const Q8K& q8) {
                    return dot_q5k_q8k(b, q8);
                }
            );

            benchmark_type<Q6Block>(
                "Q6_K", rows, threads, activation, rng,
                [](Q6Block& b, std::mt19937& r) {
                    make_q6_block(b, r);
                },
                [](const uint8_t* b, float* out) {
                    dq_q6_k(b, out);
                },
                [](const uint8_t* b, const Q8K& q8) {
                    return dot_q6k_q8k(b, q8);
                }
            );
        }
    }

    std::printf("\n========== BENCHMARK COMPLETE ==========\n");
    return 0;
}
