#pragma once

#include "q6_groupwise_q8.h"

#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace q6g16_simd {

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#define Q6_G16_AVX2 __attribute__((target("avx2")))
#else
#define Q6_G16_AVX2
#endif

static inline bool available() {
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}

Q6_G16_AVX2
static inline void quantize(
    const float* x, q6g16::Q8G16& y) {

    q6g16::NearestRoundingGuard rounding;
    std::memset(&y, 0, sizeof(y));

    alignas(32) int32_t converted[8];

    const __m256i lower = _mm256_set1_epi32(-127);
    const __m256i upper = _mm256_set1_epi32(127);

    for (int group = 0; group < q6g16::GROUPS; ++group) {
        const int offset = group * q6g16::GROUP;

        // Preserve the scalar reference's maximum-selection semantics.
        float maximum = 0.0f;
        float absolute_maximum = 0.0f;

        for (int j = 0; j < q6g16::GROUP; ++j) {
            const float value = x[offset + j];
            const float av = std::fabs(value);

            if (av > absolute_maximum) {
                absolute_maximum = av;
                maximum = value;
            }
        }

        if (absolute_maximum == 0.0f)
            continue;

        const float inverse_scale = -127.0f / maximum;
        y.d[group] = 1.0f / inverse_scale;

        int sum = 0;

        for (int half = 0; half < 2; ++half) {
            const int base = offset + half * 8;

            const __m256 values = _mm256_loadu_ps(x + base);
            const __m256 scale = _mm256_set1_ps(inverse_scale);
            const __m256 scaled = _mm256_mul_ps(values, scale);

            // Round according to the current MXCSR rounding mode.
            __m256i q = _mm256_cvtps_epi32(scaled);

            q = _mm256_max_epi32(q, lower);
            q = _mm256_min_epi32(q, upper);

            _mm256_store_si256(
                reinterpret_cast<__m256i*>(converted), q);

            for (int j = 0; j < 8; ++j) {
                const int8_t value =
                    static_cast<int8_t>(converted[j]);

                y.qs[base + j] = value;
                sum += value;
            }
        }

        y.bsums[group] = static_cast<int16_t>(sum);
    }
}

#undef Q6_G16_AVX2

} // namespace q6g16_simd
