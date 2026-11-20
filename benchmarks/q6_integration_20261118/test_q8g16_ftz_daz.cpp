#include "q6_groupwise_q8_checked.h"

#include <cstdio>
#include <cstring>
#include <immintrin.h>

constexpr unsigned FTZ = 1u << 15;
constexpr unsigned DAZ = 1u << 6;
constexpr unsigned CONTROL = FTZ | DAZ;

int main() {
    float input[256];
    for (float& v : input)
        v = 4e-37f;

    const unsigned original = _mm_getcsr();

    q6g16::Q8G16 baseline{};

    _mm_setcsr(original & ~CONTROL);

    if (!q6g16_checked::quantize_scalar(input, baseline)) {
        std::puts("Baseline quantization failed");
        _mm_setcsr(original);
        return 1;
    }

    struct Mode {
        const char* name;
        unsigned bits;
    };

    const Mode modes[] = {
        {"FTZ_OFF_DAZ_OFF", 0},
        {"FTZ_ON_DAZ_OFF", FTZ},
        {"FTZ_OFF_DAZ_ON", DAZ},
        {"FTZ_ON_DAZ_ON", CONTROL}
    };

    size_t failures = 0;

    for (const auto& mode : modes) {
        _mm_setcsr((original & ~CONTROL) | mode.bits);
        const unsigned before = _mm_getcsr();

        q6g16::Q8G16 scalar{}, simd{};

        const bool scalar_ok =
            q6g16_checked::quantize_scalar(input, scalar);

        const bool simd_ok =
            q6g16_checked::quantize_simd(input, simd);

        const unsigned after = _mm_getcsr();

        const bool pair_equal =
            std::memcmp(&scalar, &simd, sizeof(scalar)) == 0;

        const bool baseline_equal =
            std::memcmp(&baseline, &scalar, sizeof(scalar)) == 0 &&
            std::memcmp(&baseline, &simd, sizeof(simd)) == 0;

        const bool restored =
            (before & CONTROL) == (after & CONTROL);

        const bool pass =
            scalar_ok && simd_ok &&
            pair_equal && baseline_equal && restored;

        std::printf(
            "%-20s scalar=%s simd=%s pair=%s "
            "baseline=%s restored=%s %s\n",
            mode.name,
            scalar_ok ? "ACCEPT" : "REJECT",
            simd_ok ? "ACCEPT" : "REJECT",
            pair_equal ? "YES" : "NO",
            baseline_equal ? "YES" : "NO",
            restored ? "YES" : "NO",
            pass ? "PASS" : "FAIL"
        );

        failures += !pass;
    }

    _mm_setcsr(original);

    std::printf("\nFailures: %zu\n", failures);
    std::printf("RESULT: %s\n", failures ? "FAIL" : "PASS");

    return failures ? 1 : 0;
}
