#pragma once

#include "q6_avxvnni.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

namespace q6g16 {

constexpr int K = 256;
constexpr int GROUP = 16;
constexpr int GROUPS = K / GROUP;

struct Q8G16 {
    int8_t qs[K];
    float d[GROUPS];
    int16_t bsums[GROUPS];
};

// Make quantization deterministic regardless of the caller's MXCSR
// rounding mode. Restore the complete original MXCSR state on exit.
class NearestRoundingGuard {
    unsigned int original_;

public:
    NearestRoundingGuard() : original_(_mm_getcsr()) {
        _mm_setcsr((original_ & ~_MM_ROUND_MASK) | _MM_ROUND_NEAREST);
    }

    ~NearestRoundingGuard() {
        _mm_setcsr(original_);
    }

    NearestRoundingGuard(const NearestRoundingGuard&) = delete;
    NearestRoundingGuard& operator=(const NearestRoundingGuard&) = delete;
};

static inline void quantize(const float* x, Q8G16& y) {
    NearestRoundingGuard rounding;
    std::memset(&y, 0, sizeof(y));

    for (int group = 0; group < GROUPS; ++group) {
        const int offset = group * GROUP;

        float maximum = 0.0f;
        float absolute_maximum = 0.0f;

        for (int j = 0; j < GROUP; ++j) {
            const float value = x[offset + j];
            const float av = std::fabs(value);

            if (av > absolute_maximum) {
                absolute_maximum = av;
                maximum = value;
            }
        }

        if (absolute_maximum == 0.0f) {
            y.d[group] = 0.0f;
            continue;
        }

        const float inverse_scale = -127.0f / maximum;
        y.d[group] = 1.0f / inverse_scale;

        int sum = 0;

        for (int j = 0; j < GROUP; ++j) {
            const int q = q6vnni::nearest_int(
                inverse_scale * x[offset + j]);

            const int clipped = std::max(-127, std::min(127, q));

            y.qs[offset + j] = static_cast<int8_t>(clipped);
            sum += clipped;
        }

        y.bsums[group] = static_cast<int16_t>(sum);
    }
}

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#define Q6_G16_TARGET __attribute__((target("avx2,avxvnni")))
#else
#define Q6_G16_TARGET
#endif

Q6_G16_TARGET
static inline float dot_q6_g16(
    const q6vnni::PredecodedQ6& w,
    const Q8G16& x
) {
    float result = 0.0f;

    const __m256i bias = _mm256_set1_epi8(static_cast<char>(0x80));

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

        const int32_t unsigned_dot_0 =
            _mm_cvtsi128_si32(_mm256_castsi256_si128(h2));

        const int32_t unsigned_dot_1 =
            _mm_cvtsi128_si32(_mm256_extracti128_si256(h2, 1));

        const int32_t signed_dot_0 =
            unsigned_dot_0 - 128 * x.bsums[group];

        const int32_t signed_dot_1 =
            unsigned_dot_1 - 128 * x.bsums[group + 1];

        result += (w.d * w.scales[group] * x.d[group]) *
                  static_cast<float>(signed_dot_0);

        result += (w.d * w.scales[group + 1] * x.d[group + 1]) *
                  static_cast<float>(signed_dot_1);
    }

    return result;
}

#undef Q6_G16_TARGET

} // namespace q6g16
