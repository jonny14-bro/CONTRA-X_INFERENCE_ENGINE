#include "q6_groupwise_q8.h"
#include "q6_groupwise_q8_simd.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <limits>
#include <vector>
#include <immintrin.h>

struct Category {
    const char* name;
    std::vector<float> values;
};

int main() {
    if (!q6g16_simd::available()) {
        std::puts("AVX2 unavailable");
        return 2;
    }

    _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);

    const float tiny = 1e-36f;
    const float huge = 1e36f;

    std::vector<Category> tests;

    tests.push_back({"all_zero", std::vector<float>(256, 0.0f)});

    tests.push_back({"tiny_normal", std::vector<float>(256, tiny)});
    tests.push_back({"large_finite", std::vector<float>(256, huge)});

    tests.push_back({"mixed_sign", std::vector<float>(256)});

    tests.push_back({"near_zero", std::vector<float>(256)});

    tests.push_back({"rounding_boundaries", std::vector<float>(256)});

    for (int i = 0; i < 256; ++i) {
        tests[3].values[i] = (i & 1) ? -1e-10f : 1e-10f;

        tests[4].values[i] =
            (i % 4 == 0) ? 0.0f :
            (i % 4 == 1) ? tiny :
            (i % 4 == 2) ? -tiny : tiny * 0.5f;

        // Values around half-integer quantization boundaries.
        const float boundary = (i % 127) + 0.5f;
        tests[5].values[i] =
            boundary * (1.0f / 127.0f);
    }

    size_t total_mismatches = 0;
    size_t failed_categories = 0;

    for (const auto& test : tests) {
        q6g16::Q8G16 scalar{};
        q6g16::Q8G16 simd{};

        q6g16::quantize(test.values.data(), scalar);
        q6g16_simd::quantize(test.values.data(), simd);

        const bool identical =
            std::memcmp(&scalar, &simd, sizeof(scalar)) == 0;

        size_t mismatches = 0;

        for (int i = 0; i < 256; ++i)
            mismatches += scalar.qs[i] != simd.qs[i];

        for (int i = 0; i < 16; ++i) {
            mismatches += scalar.d[i] != simd.d[i];
            mismatches += scalar.bsums[i] != simd.bsums[i];
        }

        total_mismatches += mismatches;

        if (!identical)
            ++failed_categories;

        std::printf(
            "%-24s byte_equal=%s mismatches=%zu\n",
            test.name,
            identical ? "YES" : "NO",
            mismatches
        );

        for (int g = 0; g < 16; ++g) {
            if (!std::isfinite(scalar.d[g]) ||
                !std::isfinite(simd.d[g])) {
                std::printf(
                    "  INVALID SCALE: group=%d scalar=%g simd=%g\n",
                    g, scalar.d[g], simd.d[g]
                );
                ++failed_categories;
                break;
            }
        }
    }

    std::printf("\nTotal mismatches: %zu\n", total_mismatches);
    std::printf("Failed categories: %zu\n", failed_categories);

    const bool pass =
        total_mismatches == 0 &&
        failed_categories == 0;

    std::printf("RESULT: %s\n", pass ? "PASS" : "FAIL");

    return pass ? 0 : 1;
}
