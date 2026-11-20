#pragma once

#include "q6_groupwise_q8.h"
#include <immintrin.h>
#include <cstdint>

namespace q6g16_v2 {

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#define Q6_G16_V2_TARGET __attribute__((target("avx2,avxvnni")))
#else
#define Q6_G16_V2_TARGET
#endif

Q6_G16_V2_TARGET
static inline float dot_q6_g16_v2(
    const q6vnni::PredecodedQ6& w,
    const q6g16::Q8G16& x
) {
    alignas(32) int32_t group_dot[16];

    const __m256i bias =
        _mm256_set1_epi8(static_cast<char>(0x80));

    // Integer dot products and signed-weight correction.
    // This deliberately preserves the original V1 calculation.
    for (int pair = 0; pair < 8; ++pair) {
        const int group = pair * 2;

        const __m256i signed_weights = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(w.q + group * 16));

        const __m256i unsigned_weights =
            _mm256_xor_si256(signed_weights, bias);

        const __m256i activations = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(x.qs + group * 16));

        const __m256i acc = _mm256_dpbusd_epi32(
            _mm256_setzero_si256(),
            unsigned_weights,
            activations);

        const __m256i h1 = _mm256_hadd_epi32(acc, acc);
        const __m256i h2 = _mm256_hadd_epi32(h1, h1);

        const int32_t dot0 =
            _mm_cvtsi128_si32(_mm256_castsi256_si128(h2));

        const int32_t dot1 =
            _mm_cvtsi128_si32(_mm256_extracti128_si256(h2, 1));

        group_dot[group] =
            dot0 - 128 * static_cast<int32_t>(x.bsums[group]);

        group_dot[group + 1] =
            dot1 - 128 * static_cast<int32_t>(x.bsums[group + 1]);
    }

    // Vectorized scale conversion and floating-point accumulation.
    __m256 result = _mm256_setzero_ps();
    const __m256 wd = _mm256_set1_ps(w.d);

    for (int base = 0; base < 16; base += 8) {
        const __m128i scale_bytes = _mm_loadl_epi64(
            reinterpret_cast<const __m128i*>(w.scales + base));

        const __m256i scales_i32 =
            _mm256_cvtepi8_epi32(scale_bytes);

        const __m256 scales =
            _mm256_cvtepi32_ps(scales_i32);

        const __m256 activation_scales =
            _mm256_loadu_ps(x.d + base);

        const __m256 dots =
            _mm256_cvtepi32_ps(
                _mm256_load_si256(
                    reinterpret_cast<const __m256i*>(
                        group_dot + base)));

        const __m256 factors =
            _mm256_mul_ps(
                _mm256_mul_ps(wd, scales),
                activation_scales);

        result = _mm256_add_ps(
            result,
            _mm256_mul_ps(dots, factors));
    }

    // Horizontal reduction of eight accumulated group results.
    const __m256 h1 = _mm256_hadd_ps(result, result);
    const __m256 h2 = _mm256_hadd_ps(h1, h1);
    const __m128 low = _mm256_castps256_ps128(h2);
    const __m128 high = _mm256_extractf128_ps(h2, 1);

    const __m128 total = _mm_add_ss(low, high);

    return _mm_cvtss_f32(total);
}

#undef Q6_G16_V2_TARGET

} // namespace q6g16_v2
