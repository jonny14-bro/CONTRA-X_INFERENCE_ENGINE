#include "q6_groupwise_q8.h"
#include "q6_groupwise_q8_simd.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>
#include <immintrin.h>

int main() {
    if (!q6g16_simd::available())
        return 2;

    _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);

    const float denorm =
        std::numeric_limits<float>::denorm_min();
    const float min_normal =
        std::numeric_limits<float>::min();

    std::printf("FP32 denorm_min: %.9e\n", denorm);
    std::printf("FP32 min_normal: %.9e\n\n", min_normal);

    size_t mismatches = 0;
    size_t failures = 0;

    // Mixed subnormal values with a safe normal maximum.
    {
        float input[256]{};

        for (int i = 0; i < 256; ++i) {
            switch (i % 4) {
                case 0: input[i] = denorm; break;
                case 1: input[i] = -denorm; break;
                case 2: input[i] = 4e-37f; break;
                default: input[i] = -4e-37f; break;
            }
        }

        q6g16::Q8G16 scalar{};
        q6g16::Q8G16 simd{};

        q6g16::quantize(input, scalar);
        q6g16_simd::quantize(input, simd);

        const bool equal =
            std::memcmp(&scalar, &simd, sizeof(scalar)) == 0;

        std::printf("mixed_subnormal: %s\n",
                    equal ? "PASS" : "FAIL");

        mismatches += !equal;

        for (int g = 0; g < 16; ++g) {
            if (!std::isfinite(scalar.d[g]) ||
                !std::isfinite(simd.d[g])) {
                ++failures;
            }
        }
    }

    // A normal maximum close to the inverse-scale overflow boundary.
    {
        float input[256];
        std::fill(input, input + 256, 4e-37f);

        q6g16::Q8G16 scalar{};
        q6g16::Q8G16 simd{};

        q6g16::quantize(input, scalar);
        q6g16_simd::quantize(input, simd);

        const bool equal =
            std::memcmp(&scalar, &simd, sizeof(scalar)) == 0;

        std::printf("near_scale_overflow_boundary: %s\n",
                    equal ? "PASS" : "FAIL");

        mismatches += !equal;

        if (!std::isfinite(scalar.d[0]) ||
            !std::isfinite(simd.d[0]))
            ++failures;
    }

    // Inspect subnormal-only groups without invoking unsafe quantization.
    {
        const float maximum = denorm;
        const float inverse_scale = -127.0f / maximum;

        std::printf(
            "subnormal_only_probe: maximum=%.9e "
            "inverse_scale=%g finite=%s\n",
            maximum,
            inverse_scale,
            std::isfinite(inverse_scale) ? "YES" : "NO"
        );

        if (std::isfinite(inverse_scale))
            ++failures;
    }

    std::printf("\nMismatches: %zu\n", mismatches);
    std::printf("Unexpected conditions: %zu\n", failures);

    const bool pass = mismatches == 0 && failures == 0;
    std::printf("RESULT: %s\n", pass ? "PASS" : "FAIL");

    return pass ? 0 : 1;
}
