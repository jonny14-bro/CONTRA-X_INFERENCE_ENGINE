#pragma once

#include "q6_groupwise_q8.h"
#include "q6_groupwise_q8_simd.h"

#include <cmath>
#include <cstring>
#include <immintrin.h>

namespace q6g16_checked {

constexpr unsigned int FTZ_MASK = 1u << 15;
constexpr unsigned int DAZ_MASK = 1u << 6;

constexpr unsigned int CONTROL_MASK =
    _MM_ROUND_MASK | FTZ_MASK | DAZ_MASK;

class NearestRoundingGuard {
    unsigned int original_;

public:
    NearestRoundingGuard() : original_(_mm_getcsr()) {
        const unsigned int controlled =
            (original_ & ~CONTROL_MASK) | _MM_ROUND_NEAREST;

        _mm_setcsr(controlled);
    }

    ~NearestRoundingGuard() {
        _mm_setcsr(original_);
    }

    NearestRoundingGuard(const NearestRoundingGuard&) = delete;
    NearestRoundingGuard& operator=(const NearestRoundingGuard&) = delete;
};

static inline bool validate(const float* x) {
    for (int i = 0; i < q6g16::K; ++i) {
        if (!std::isfinite(x[i]))
            return false;
    }

    for (int g = 0; g < q6g16::GROUPS; ++g) {
        float maximum = 0.0f;

        for (int j = 0; j < q6g16::GROUP; ++j) {
            const float value = x[g * q6g16::GROUP + j];

            if (std::fabs(value) > std::fabs(maximum))
                maximum = value;
        }

        if (maximum == 0.0f)
            continue;

        const float inverse_scale = -127.0f / maximum;
        const float scale = 1.0f / inverse_scale;

        if (!std::isfinite(inverse_scale) ||
            !std::isfinite(scale) ||
            scale == 0.0f)
            return false;
    }

    return true;
}

static inline bool quantize_scalar(
    const float* x,
    q6g16::Q8G16& output
) {
    NearestRoundingGuard rounding;
    std::memset(&output, 0, sizeof(output));

    if (!validate(x))
        return false;

    q6g16::quantize(x, output);
    return true;
}

static inline bool quantize_simd(
    const float* x,
    q6g16::Q8G16& output
) {
    NearestRoundingGuard rounding;
    std::memset(&output, 0, sizeof(output));

    if (!q6g16_simd::available() || !validate(x))
        return false;

    q6g16_simd::quantize(x, output);
    return true;
}

} // namespace q6g16_checked
