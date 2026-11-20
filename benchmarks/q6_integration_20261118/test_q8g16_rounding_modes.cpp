#include "q6_groupwise_q8_checked.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <cmath>
#include <immintrin.h>

struct Mode {
    const char* name;
    unsigned int bits;
};

int main() {
    const Mode modes[] = {
        {"nearest", _MM_ROUND_NEAREST},
        {"down", _MM_ROUND_DOWN},
        {"up", _MM_ROUND_UP},
        {"toward_zero", _MM_ROUND_TOWARD_ZERO}
    };

    constexpr int ITERATIONS = 10000;

    size_t total_mismatches = 0;
    size_t total_rejected = 0;
    size_t total_restore_errors = 0;

    for (const auto& mode : modes) {
        std::mt19937 rng(0xA17E2026);
        std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
        std::uniform_real_distribution<float> magnitude(0.01f, 100.0f);

        size_t mismatches = 0;
        size_t rejected = 0;
        size_t restore_errors = 0;

        for (int iteration = 0; iteration < ITERATIONS; ++iteration) {
            _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);

            float input[256]{};

            for (int g = 0; g < 16; ++g) {
                const float mag = magnitude(rng);

                for (int j = 0; j < 16; ++j) {
                    const int i = g * 16 + j;

                    if (iteration % 2 == 0) {
                        const float boundary =
                            static_cast<float>((j % 127) + 0.5f);

                        float value = (boundary / 127.0f) * mag;

                        if (j % 3 == 0)
                            value = std::nextafter(value, -INFINITY);
                        else if (j % 3 == 1)
                            value = std::nextafter(value, INFINITY);

                        input[i] = (j & 1) ? -value : value;
                    } else {
                        input[i] = unit(rng) * mag;
                    }
                }
            }

            q6g16::Q8G16 reference{}, scalar{}, simd{};

            const bool reference_ok =
                q6g16_checked::quantize_scalar(input, reference);

            _MM_SET_ROUNDING_MODE(mode.bits);
            const unsigned int expected_mode =
                _mm_getcsr() & _MM_ROUND_MASK;

            const bool scalar_ok =
                q6g16_checked::quantize_scalar(input, scalar);

            const unsigned int scalar_mode =
                _mm_getcsr() & _MM_ROUND_MASK;

            const bool simd_ok =
                q6g16_checked::quantize_simd(input, simd);

            const unsigned int simd_mode =
                _mm_getcsr() & _MM_ROUND_MASK;

            if (scalar_mode != expected_mode ||
                simd_mode != expected_mode)
                ++restore_errors;

            if (!reference_ok || !scalar_ok || !simd_ok) {
                ++rejected;
                continue;
            }

            if (std::memcmp(&reference, &scalar, sizeof(reference)) != 0 ||
                std::memcmp(&reference, &simd, sizeof(reference)) != 0)
                ++mismatches;
        }

        std::printf(
            "%-14s mismatches=%zu rejected=%zu "
            "restore_errors=%zu / %d\n",
            mode.name, mismatches, rejected,
            restore_errors, ITERATIONS);

        total_mismatches += mismatches;
        total_rejected += rejected;
        total_restore_errors += restore_errors;
    }

    _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);

    std::printf("\nTotal mismatches: %zu\n", total_mismatches);
    std::printf("Total rejected: %zu\n", total_rejected);
    std::printf("MXCSR restoration errors: %zu\n", total_restore_errors);

    const bool pass =
        total_mismatches == 0 &&
        total_rejected == 0 &&
        total_restore_errors == 0;

    std::printf("RESULT: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
