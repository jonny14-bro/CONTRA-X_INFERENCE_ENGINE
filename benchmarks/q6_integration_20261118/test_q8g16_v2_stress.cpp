#include "q6_groupwise_q8.h"
#include "q6_groupwise_q8_simd.h"
#include "q6_groupwise_q8_v2.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <immintrin.h>

struct Stats {
    double max_abs = 0.0;
    double max_rel = 0.0;
    double sum_sq = 0.0;
    double ref_sq = 0.0;
    size_t count = 0;
    size_t nonfinite = 0;

    void add(double ref, double value) {
        if (!std::isfinite(ref) || !std::isfinite(value)) {
            ++nonfinite;
            return;
        }

        const double error = std::fabs(ref - value);
        const double relative =
            error / std::max(std::fabs(ref), 1e-30);

        max_abs = std::max(max_abs, error);
        max_rel = std::max(max_rel, relative);
        sum_sq += error * error;
        ref_sq += ref * ref;
        ++count;
    }

    void print(const char* name) const {
        const double rmse = count
            ? std::sqrt(sum_sq / count) : 0.0;

        const double nrmse = ref_sq > 0
            ? std::sqrt(sum_sq / ref_sq) : 0.0;

        std::printf(
            "%-18s max_abs=%.9g max_rel=%.9g "
            "RMSE=%.9g NRMSE=%.9g nonfinite=%zu\n",
            name, max_abs, max_rel, rmse, nrmse, nonfinite);
    }
};

static float random_magnitude(std::mt19937& rng) {
    static std::uniform_real_distribution<float> exponent(-20.0f, 20.0f);
    return std::pow(10.0f, exponent(rng));
}

int main() {
    if (!q6vnni::available() || !q6g16_simd::available()) {
        std::fprintf(stderr, "Required AVX-VNNI/AVX2 unavailable\n");
        return 2;
    }

    // Match the scalar nearest_int reference's normal rounding behavior.
    _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);

    constexpr size_t ITERATIONS = 100000;

    std::mt19937 rng(0xEA6A2026);

    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    std::uniform_int_distribution<int> qdist(-127, 127);
    std::uniform_int_distribution<int> wdist(-32, 31);
    std::uniform_int_distribution<int> sdist(-128, 127);
    std::uniform_int_distribution<int> pattern_dist(0, 6);

    Stats v1_stats;
    Stats v2_stats;
    Stats v1_v2_stats;

    size_t quantizer_mismatches = 0;
    size_t first_mismatch = ITERATIONS;
    size_t zero_blocks = 0;

    for (size_t iteration = 0; iteration < ITERATIONS; ++iteration) {
        float input[256]{};

        const int pattern = pattern_dist(rng);
        const float magnitude = random_magnitude(rng);

        for (int g = 0; g < 16; ++g) {
            for (int j = 0; j < 16; ++j) {
                const int i = g * 16 + j;

                switch (pattern) {
                    case 0:
                        input[i] = unit(rng) * magnitude;
                        break;

                    case 1:
                        input[i] = (j % 2 ? 1.0f : -1.0f)
                                   * magnitude;
                        break;

                    case 2:
                        input[i] = j == 0
                            ? magnitude * (unit(rng) < 0 ? -1.0f : 1.0f)
                            : 0.0f;
                        break;

                    case 3:
                        input[i] = j < 4
                            ? unit(rng) * magnitude : 0.0f;
                        break;

                    case 4:
                        input[i] = (unit(rng) * 0.001f) * magnitude;
                        break;

                    case 5:
                        input[i] = (j % 3 == 0)
                            ? 0.0f : unit(rng) * magnitude;
                        break;

                    default:
                        input[i] = 0.0f;
                        break;
                }
            }
        }

        q6g16::Q8G16 scalar{};
        q6g16::Q8G16 simd{};

        q6g16::quantize(input, scalar);
        q6g16_simd::quantize(input, simd);

        if (std::memcmp(&scalar, &simd, sizeof(scalar)) != 0) {
            ++quantizer_mismatches;
            if (first_mismatch == ITERATIONS)
                first_mismatch = iteration;
        }

        if (pattern == 6)
            ++zero_blocks;

        q6vnni::PredecodedQ6 weights{};

        {
            static std::uniform_real_distribution<float> weight_exp(-8.0f, 2.0f);
            weights.d = std::pow(10.0f, weight_exp(rng));
        }

        for (int i = 0; i < 256; ++i)
            weights.q[i] = static_cast<int8_t>(wdist(rng));

        for (int g = 0; g < 16; ++g)
            weights.scales[g] = static_cast<int8_t>(sdist(rng));

        const float v1 = q6g16::dot_q6_g16(weights, simd);
        const float v2 = q6g16_v2::dot_q6_g16_v2(weights, simd);

        // Independent FP64 reference using the exact integer
        // quantized values and the mathematical groupwise formula.
        double reference = 0.0;

        for (int g = 0; g < 16; ++g) {
            int64_t integer_dot = 0;

            for (int j = 0; j < 16; ++j) {
                const int i = g * 16 + j;

                integer_dot +=
                    static_cast<int>(weights.q[i]) *
                    static_cast<int>(simd.qs[i]);
            }

            reference +=
                static_cast<double>(weights.d) *
                static_cast<double>(weights.scales[g]) *
                static_cast<double>(simd.d[g]) *
                static_cast<double>(integer_dot);
        }

        v1_stats.add(reference, v1);
        v2_stats.add(reference, v2);
        v1_v2_stats.add(v1, v2);
    }

    std::printf("Q8G16 V2 randomized stress test\n");
    std::printf("Iterations: %zu\n", ITERATIONS);
    std::printf("Zero-pattern blocks: %zu\n", zero_blocks);
    std::printf("AVX2 available: yes\n");
    std::printf("AVX-VNNI available: yes\n");
    std::printf("MXCSR rounding: nearest\n\n");

    std::printf(
        "Quantizer byte mismatches: %zu / %zu\n",
        quantizer_mismatches, ITERATIONS);

    if (quantizer_mismatches)
        std::printf("First mismatch iteration: %zu\n", first_mismatch);

    v1_stats.print("V1 vs FP64");
    v2_stats.print("V2 vs FP64");
    v1_v2_stats.print("V1 vs V2");

    const bool pass =
        quantizer_mismatches == 0 &&
        v1_stats.nonfinite == 0 &&
        v2_stats.nonfinite == 0 &&
        v1_v2_stats.nonfinite == 0;

    std::printf("\nRESULT: %s\n", pass ? "PASS" : "FAIL");

    return pass ? 0 : 1;
}
