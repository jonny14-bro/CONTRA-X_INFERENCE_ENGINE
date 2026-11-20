#include "q6_groupwise_q8.h"
#include "q6_groupwise_q8_simd.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <immintrin.h>

struct Mode {
    const char* name;
    unsigned int bits;
};

int main() {
    const unsigned int original_mxcsr = _mm_getcsr();

    const Mode modes[] = {
        {"nearest", _MM_ROUND_NEAREST},
        {"down", _MM_ROUND_DOWN},
        {"up", _MM_ROUND_UP},
        {"toward_zero", _MM_ROUND_TOWARD_ZERO}
    };

    float input[256];
    std::mt19937 rng(0xA17E2026);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    std::uniform_real_distribution<float> magnitude(0.01f, 100.0f);

    for (int g = 0; g < 16; ++g) {
        const float mag = magnitude(rng);
        for (int j = 0; j < 16; ++j)
            input[g * 16 + j] = unit(rng) * mag;
    }

    q6g16::Q8G16 baseline{};
    _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
    q6g16::quantize(input, baseline);

    size_t total_mismatches = 0;
    size_t total_restore_errors = 0;

    for (const auto& mode : modes) {
        _MM_SET_ROUNDING_MODE(mode.bits);
        const unsigned int expected = _mm_getcsr();

        q6g16::Q8G16 scalar{};
        q6g16::Q8G16 simd{};

        q6g16::quantize(input, scalar);
        const unsigned int scalar_after =
            _mm_getcsr();

        q6g16_simd::quantize(input, simd);
        const unsigned int simd_after =
            _mm_getcsr();

        const bool scalar_equal =
            std::memcmp(&baseline, &scalar, sizeof(baseline)) == 0;
        const bool simd_equal =
            std::memcmp(&baseline, &simd, sizeof(baseline)) == 0;
        const bool scalar_simd_equal =
            std::memcmp(&scalar, &simd, sizeof(scalar)) == 0;

        const bool restored =
            scalar_after == expected && simd_after == expected;

        const size_t mismatches =
            (!scalar_equal) + (!simd_equal) + (!scalar_simd_equal);

        total_mismatches += mismatches;
        total_restore_errors += !restored;

        std::printf(
            "%-14s scalar_baseline=%s simd_baseline=%s "
            "scalar_simd=%s mxcsr_restored=%s\n",
            mode.name,
            scalar_equal ? "YES" : "NO",
            simd_equal ? "YES" : "NO",
            scalar_simd_equal ? "YES" : "NO",
            restored ? "YES" : "NO");
    }

    _mm_setcsr(original_mxcsr);

    const bool pass =
        total_mismatches == 0 && total_restore_errors == 0;

    std::printf("\nTotal parity failures: %zu\n", total_mismatches);
    std::printf("MXCSR restoration failures: %zu\n", total_restore_errors);
    std::printf("RESULT: %s\n", pass ? "PASS" : "FAIL");

    return pass ? 0 : 1;
}
