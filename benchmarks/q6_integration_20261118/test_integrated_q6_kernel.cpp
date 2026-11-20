#define Q8K_BENCH_ONLY
#include "../../tests/test_q8k_kernels.cpp"
#include "../../q6_avxvnni.h"

#include <array>
#include <cstdio>
#include <random>
#include <cmath>
#include <cstring>

int main() {
    init_types();

    std::printf("AVX-VNNI available: %s\n",
                q6vnni::available() ? "yes" : "no");

    if (!q6vnni::available()) {
        std::printf("SKIP: CPU does not support AVX-VNNI\n");
        return 77;
    }

    std::mt19937 rng(0xEA11F);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    int quant_scale_errors = 0;
    int quant_value_errors = 0;
    int quant_sum_errors = 0;
    int dot_errors = 0;
    int nonfinite = 0;
    double max_abs = 0.0;

    constexpr int trials = 10000;

    for (int trial = 0; trial < trials; ++trial) {
        std::array<float, 256> activation{};
        for (float& v : activation)
            v = dist(rng);

        if (trial % 100 == 0)
            activation.fill(0.0f);

        Q8K reference_q8{};
        q6vnni::Q8K integrated_q8{};

        quantize_q8k(activation.data(), reference_q8);
        q6vnni::quantize_q8k(activation.data(), integrated_q8);

        if (std::memcmp(&reference_q8.d, &integrated_q8.d,
                        sizeof(float)) != 0)
            ++quant_scale_errors;

        if (std::memcmp(reference_q8.qs, integrated_q8.qs,
                        sizeof(reference_q8.qs)) != 0)
            ++quant_value_errors;

        if (std::memcmp(reference_q8.bsums, integrated_q8.bsums,
                        sizeof(reference_q8.bsums)) != 0)
            ++quant_sum_errors;

        std::array<uint8_t, 210> packed{};
        make_q6_block(packed, rng);

        q6vnni::PredecodedQ6 decoded{};
        q6vnni::decode_q6(
            packed.data(), rd_f16(packed.data() + 208), decoded
        );

        const float expected =
            dot_q6k_q8k(packed.data(), reference_q8);

        const float actual =
            q6vnni::dot_q6_hadd(decoded, integrated_q8);

        if (!std::isfinite(actual) || !std::isfinite(expected)) {
            ++nonfinite;
            continue;
        }

        const double error = std::fabs(
            static_cast<double>(actual) - expected
        );

        max_abs = std::max(max_abs, error);

        if (error > 2.0e-5)
            ++dot_errors;
    }

    std::printf("\n========== INTEGRATED Q6 KERNEL VALIDATION ==========\n");
    std::printf("Randomized blocks: %d\n", trials);
    std::printf("Q8 scale mismatches: %d\n", quant_scale_errors);
    std::printf("Q8 value mismatches: %d\n", quant_value_errors);
    std::printf("Q8 subgroup-sum mismatches: %d\n", quant_sum_errors);
    std::printf("Q6 dot mismatches (>2e-5): %d\n", dot_errors);
    std::printf("Nonfinite results: %d\n", nonfinite);
    std::printf("Maximum absolute dot difference: %.12g\n", max_abs);

    const bool passed =
        quant_scale_errors == 0 &&
        quant_value_errors == 0 &&
        quant_sum_errors == 0 &&
        dot_errors == 0 &&
        nonfinite == 0;

    std::printf("\nINTEGRATED Q6 KERNEL: %s\n",
                passed ? "PASS" : "FAIL");

    return passed ? 0 : 1;
}
