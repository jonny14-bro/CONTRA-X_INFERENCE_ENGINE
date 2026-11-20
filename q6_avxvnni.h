#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

namespace q6vnni {

constexpr int K = 256;

struct Q8K {
    float d;
    int8_t qs[K];
    int16_t bsums[K / 16];
};

static_assert(sizeof(Q8K) == 292, "Q8K layout mismatch");

struct PredecodedQ6 {
    int8_t q[256];
    int8_t scales[16];
    float d;
};

static inline int nearest_int(float f) {
    assert(std::fabs(f) <= 4194303.0f);
    float v = f + 12582912.0f;
    int i;
    std::memcpy(&i, &v, sizeof(i));
    return (i & 0x007fffff) - 0x00400000;
}

static inline void quantize_q8k(const float* x, Q8K& y) {
    float maximum = 0.0f;
    float absolute_maximum = 0.0f;

    for (int i = 0; i < K; ++i) {
        const float ax = std::fabs(x[i]);
        if (ax > absolute_maximum) {
            absolute_maximum = ax;
            maximum = x[i];
        }
    }

    if (absolute_maximum == 0.0f) {
        y.d = 0.0f;
        std::memset(y.qs, 0, sizeof(y.qs));
        std::memset(y.bsums, 0, sizeof(y.bsums));
        return;
    }

    const float inverse_scale = -127.0f / maximum;

    for (int i = 0; i < K; ++i) {
        const int q = nearest_int(inverse_scale * x[i]);
        y.qs[i] = static_cast<int8_t>(std::min(127, q));
    }

    for (int group = 0; group < K / 16; ++group) {
        int sum = 0;
        for (int j = 0; j < 16; ++j)
            sum += y.qs[group * 16 + j];
        y.bsums[group] = static_cast<int16_t>(sum);
    }

    y.d = 1.0f / inverse_scale;
}

static inline void decode_q6(
    const uint8_t* b, float d, PredecodedQ6& out
) {
    const uint8_t* ql = b;
    const uint8_t* qh = b + 128;
    const int8_t* scales =
        reinterpret_cast<const int8_t*>(b + 192);

    out.d = d;

    for (int half = 0; half < 2; ++half) {
        const uint8_t* ql_half = ql + half * 64;
        const uint8_t* qh_half = qh + half * 32;

        for (int j = 0; j < 32; ++j) {
            const uint8_t h = qh_half[j];

            out.q[half * 128 + j] =
                static_cast<int8_t>(
                    static_cast<int>((ql_half[j] & 0x0f) |
                        (((h >> 0) & 3) << 4)) - 32
                );

            out.q[half * 128 + j + 32] =
                static_cast<int8_t>(
                    static_cast<int>((ql_half[j + 32] & 0x0f) |
                        (((h >> 2) & 3) << 4)) - 32
                );

            out.q[half * 128 + j + 64] =
                static_cast<int8_t>(
                    static_cast<int>((ql_half[j] >> 4) |
                        (((h >> 4) & 3) << 4)) - 32
                );

            out.q[half * 128 + j + 96] =
                static_cast<int8_t>(
                    static_cast<int>((ql_half[j + 32] >> 4) |
                        (((h >> 6) & 3) << 4)) - 32
                );
        }
    }

    for (int i = 0; i < 16; ++i)
        out.scales[i] = scales[i];
}

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#define Q6_VNNI_TARGET __attribute__((target("avx2,avxvnni")))
#else
#define Q6_VNNI_TARGET
#endif

Q6_VNNI_TARGET
static inline float dot_q6_hadd(
    const PredecodedQ6& w, const Q8K& x
) {
    float result = 0.0f;

    const __m256i bias = _mm256_set1_epi8(static_cast<char>(0x80));

    for (int pair = 0; pair < 8; ++pair) {
        const int group = pair * 2;

        const __m256i signed_weights = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(w.q + group * 16)
        );

        const __m256i unsigned_weights =
            _mm256_xor_si256(signed_weights, bias);

        const __m256i activations = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(x.qs + group * 16)
        );

        const __m256i acc = _mm256_dpbusd_epi32(
            _mm256_setzero_si256(),
            unsigned_weights,
            activations
        );

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

        result +=
            (w.d * x.d * static_cast<float>(w.scales[group])) *
            static_cast<float>(signed_dot_0);

        result +=
            (w.d * x.d * static_cast<float>(w.scales[group + 1])) *
            static_cast<float>(signed_dot_1);
    }

    return result;
}

static inline bool available() {
#ifdef EAHMMF_DISABLE_Q6_VNNI
    return false;
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") &&
           __builtin_cpu_supports("avxvnni");
#else
    return false;
#endif
}

#undef Q6_VNNI_TARGET

} // namespace q6vnni
