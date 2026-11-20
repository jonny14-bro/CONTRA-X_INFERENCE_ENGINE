#define QWEN3MOE_NO_MAIN
#include "../qwen3moe.cpp"

#include <array>
#include <random>
#include <limits>
#include <cassert>

namespace {

constexpr int K = 256;

struct Q8K {
    float d;
    int8_t qs[K];
    int16_t bsums[K / 16];
};

static_assert(sizeof(Q8K) == 292, "Q8K layout mismatch");

static int test_nearest_int(float f) {
    assert(std::fabs(f) <= 4194303.0f);
    float v = f + 12582912.0f;
    int i;
    std::memcpy(&i, &v, sizeof(i));
    return (i & 0x007fffff) - 0x00400000;
}

static int ggml_reference_nearest_int(float fval) {
    assert(std::fabs(fval) <= 4194303.0f);
    float val = fval + 12582912.0f;
    int i;
    std::memcpy(&i, &val, sizeof(i));
    return (i & 0x007fffff) - 0x00400000;
}

static void ggml_reference_quantize_q8k(
    const float* x, Q8K& y
) {
    float max = 0.0f;
    float amax = 0.0f;

    for (int j = 0; j < K; ++j) {
        const float ax = std::fabs(x[j]);
        if (ax > amax) {
            amax = ax;
            max = x[j];
        }
    }

    if (!amax) {
        y.d = 0.0f;
        std::memset(y.qs, 0, K);
        return;
    }

    const float iscale = -127.0f / max;

    for (int j = 0; j < K; ++j) {
        const int v = ggml_reference_nearest_int(iscale * x[j]);
        y.qs[j] = static_cast<int8_t>(std::min(127, v));
    }

    for (int j = 0; j < K / 16; ++j) {
        int sum = 0;
        for (int ii = 0; ii < 16; ++ii)
            sum += y.qs[j * 16 + ii];
        y.bsums[j] = static_cast<int16_t>(sum);
    }

    y.d = 1.0f / iscale;
}

struct QuantizerParityStats {
    int blocks = 0;
    int scale_mismatches = 0;
    int qs_mismatches = 0;
    int bsum_mismatches = 0;

    void add(const Q8K& actual, const Q8K& expected) {
        ++blocks;

        if (std::memcmp(&actual.d, &expected.d, sizeof(float)) != 0)
            ++scale_mismatches;

        if (std::memcmp(actual.qs, expected.qs, sizeof(actual.qs)) != 0)
            ++qs_mismatches;

        if (std::memcmp(actual.bsums, expected.bsums,
                        sizeof(actual.bsums)) != 0)
            ++bsum_mismatches;
    }

    bool passed() const {
        return scale_mismatches == 0 &&
               qs_mismatches == 0 &&
               bsum_mismatches == 0;
    }

    void print() const {
        std::printf(
            "Q8_K parity: blocks=%d scale_mismatches=%d "
            "qs_mismatches=%d bsum_mismatches=%d\n",
            blocks, scale_mismatches, qs_mismatches, bsum_mismatches
        );
    }
};

static void quantize_q8k(const float* x, Q8K& y) {
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
        const int q = test_nearest_int(inverse_scale * x[i]);
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

static float q8_value(const Q8K& q, int i) {
    return q.d * static_cast<float>(q.qs[i]);
}

struct DotReference {
    double value = 0.0;
    double l1 = 0.0;
};

static DotReference reference_dot_double(
    const float* weights,
    const Q8K& x
) {
    DotReference ref;

    for (int i = 0; i < K; ++i) {
        const double term =
            static_cast<double>(weights[i]) *
            static_cast<double>(x.qs[i]) *
            static_cast<double>(x.d);

        ref.value += term;
        ref.l1 += std::fabs(term);
    }

    return ref;
}

static float reference_dot(const float* weights, const Q8K& x) {
    float sum = 0.0f;
    for (int i = 0; i < K; ++i)
        sum += weights[i] * q8_value(x, i);
    return sum;
}

static float reference_float_dot(const float* weights, const float* x) {
    float sum = 0.0f;
    for (int i = 0; i < K; ++i)
        sum += weights[i] * x[i];
    return sum;
}

// Q5_K x Q8_K.
// The Q5_K block is decoded in eight 32-element groups.
// Each group has a scale and minimum correction.
static float dot_q5k_q8k(const uint8_t* b, const Q8K& x) {
    const float d = rd_f16(b);
    const float dmin = rd_f16(b + 2);

    const uint8_t* packed_scales = b + 4;
    const uint8_t* qh = b + 16;
    const uint8_t* ql = b + 48;

    float result = 0.0f;

    for (int group = 0; group < 8; ++group) {
        uint8_t scale, minimum;
        scale_min_k4(group, packed_scales, &scale, &minimum);

        const uint8_t* ql_group = ql + (group / 2) * 32;
        const uint8_t high_mask = static_cast<uint8_t>(1u << group);

        int32_t dot = 0;
        int32_t activation_sum = 0;

        for (int j = 0; j < 32; ++j) {
            const int low = (group & 1)
                ? (ql_group[j] >> 4)
                : (ql_group[j] & 0x0f);

            const int high = (qh[j] & high_mask) ? 16 : 0;
            const int weight_q = low + high;
            const int activation_q = x.qs[group * 32 + j];

            dot += weight_q * activation_q;
            activation_sum += activation_q;
        }

        // Also validate the stored Q8_K subgroup sums.
        const int stored_sum =
            x.bsums[group * 2] + x.bsums[group * 2 + 1];

        assert(stored_sum == activation_sum);

        result += (d * x.d * static_cast<float>(scale)) *
                  static_cast<float>(dot);

        result -= (dmin * x.d * static_cast<float>(minimum)) *
                  static_cast<float>(activation_sum);
    }

    return result;
}

// Q6_K x Q8_K.
// Q6_K contains sixteen 16-element groups with signed scales.
static float dot_q6k_q8k(const uint8_t* b, const Q8K& x) {
    const uint8_t* ql = b;
    const uint8_t* qh = b + 128;
    const int8_t* scales = reinterpret_cast<const int8_t*>(b + 192);
    const float d = rd_f16(b + 208);

    int qweights[K];

    for (int half = 0; half < 2; ++half) {
        const uint8_t* ql_half = ql + half * 64;
        const uint8_t* qh_half = qh + half * 32;

        for (int j = 0; j < 32; ++j) {
            const uint8_t h = qh_half[j];

            qweights[half * 128 + j] =
                static_cast<int>((ql_half[j] & 0x0f) |
                                 (((h >> 0) & 3) << 4)) - 32;

            qweights[half * 128 + j + 32] =
                static_cast<int>((ql_half[j + 32] & 0x0f) |
                                 (((h >> 2) & 3) << 4)) - 32;

            qweights[half * 128 + j + 64] =
                static_cast<int>((ql_half[j] >> 4) |
                                 (((h >> 4) & 3) << 4)) - 32;

            qweights[half * 128 + j + 96] =
                static_cast<int>((ql_half[j + 32] >> 4) |
                                 (((h >> 6) & 3) << 4)) - 32;
        }
    }

    float result = 0.0f;

    for (int group = 0; group < 16; ++group) {
        int32_t dot = 0;

        for (int j = 0; j < 16; ++j) {
            const int i = group * 16 + j;
            dot += qweights[i] * static_cast<int>(x.qs[i]);
        }

        result += (d * x.d * static_cast<float>(scales[group])) *
                  static_cast<float>(dot);
    }

    return result;
}


static int verify_q5_decode(const uint8_t* b, const float* reference) {
    const float d = rd_f16(b);
    const float dmin = rd_f16(b + 2);
    const uint8_t* sc = b + 4;
    const uint8_t* qh = b + 16;
    const uint8_t* ql = b + 48;
    int mismatches = 0;

    for (int group = 0; group < 8; ++group) {
        uint8_t scale, minimum;
        scale_min_k4(group, sc, &scale, &minimum);

        for (int j = 0; j < 32; ++j) {
            const uint8_t packed = ql[(group / 2) * 32 + j];
            const int low = (group & 1) ? (packed >> 4) : (packed & 15);
            const int high = (qh[j] & (1u << group)) ? 16 : 0;
            const float decoded =
                d * scale * (low + high) - dmin * minimum;

            const int index = group * 32 + j;
            const float error = std::fabs(decoded - reference[index]);

            if (error > 1.0e-5f * std::max(1.0f, std::fabs(reference[index]))) {
                if (mismatches < 3) {
                    std::fprintf(stderr,
                        "[Q5 DECODE MISMATCH] group=%d index=%d "
                        "decoded=%.9g reference=%.9g error=%.9g\n",
                        group, index, decoded, reference[index], error);
                }
                ++mismatches;
            }
        }
    }
    return mismatches;
}

static int verify_q6_decode(const uint8_t* b, const float* reference) {
    const uint8_t* ql = b;
    const uint8_t* qh = b + 128;
    const int8_t* scales = reinterpret_cast<const int8_t*>(b + 192);
    const float d = rd_f16(b + 208);
    int mismatches = 0;

    for (int half = 0; half < 2; ++half) {
        const uint8_t* ql_half = ql + half * 64;
        const uint8_t* qh_half = qh + half * 32;

        for (int j = 0; j < 32; ++j) {
            const uint8_t h = qh_half[j];

            const int q[4] = {
                int((ql_half[j] & 15) | (((h >> 0) & 3) << 4)) - 32,
                int((ql_half[j + 32] & 15) | (((h >> 2) & 3) << 4)) - 32,
                int((ql_half[j] >> 4) | (((h >> 4) & 3) << 4)) - 32,
                int((ql_half[j + 32] >> 4) | (((h >> 6) & 3) << 4)) - 32
            };

            const int index[4] = {
                half * 128 + j,
                half * 128 + j + 32,
                half * 128 + j + 64,
                half * 128 + j + 96
            };

            for (int lane = 0; lane < 4; ++lane) {
                const int scale_index =
                    half * 8 + (lane * 2) + (j / 16);
                const float decoded =
                    d * scales[scale_index] * q[lane];
                const float error =
                    std::fabs(decoded - reference[index[lane]]);

                if (error > 1.0e-5f *
                    std::max(1.0f, std::fabs(reference[index[lane]]))) {
                    if (mismatches < 3) {
                        std::fprintf(stderr,
                            "[Q6 DECODE MISMATCH] index=%d scale_index=%d "
                            "decoded=%.9g reference=%.9g error=%.9g\n",
                            index[lane], scale_index, decoded,
                            reference[index[lane]], error);
                    }
                    ++mismatches;
                }
            }
        }
    }
    return mismatches;
}

struct ScaledErrorStats {
    double max_abs = 0.0;
    double max_scaled = 0.0;
    double sum_scaled = 0.0;
    int count = 0;
    int over_1e_6 = 0;

    void add(float actual, const DotReference& expected) {
        const double error = std::fabs(
            static_cast<double>(actual) - expected.value
        );

        const double scaled =
            error / std::max(1.0, expected.l1);

        max_abs = std::max(max_abs, error);
        max_scaled = std::max(max_scaled, scaled);
        sum_scaled += scaled;
        ++count;

        if (!std::isfinite(actual) ||
            !std::isfinite(expected.value) ||
            !std::isfinite(expected.l1) ||
            scaled > 1.0e-6) {
            ++over_1e_6;
        }
    }

    void print(const char* name) const {
        std::printf(
            "%-30s count=%d over_1e-6=%d "
            "max_abs=%.10g max_scaled=%.10g mean_scaled=%.10g\\n",
            name, count, over_1e_6, max_abs, max_scaled,
            count ? sum_scaled / count : 0.0
        );
    }
};

struct ErrorStats {
    double max_abs = 0.0;
    double max_rel = 0.0;
    double sum_abs = 0.0;
    int count = 0;
    int failures = 0;

    void add(float actual, float expected, double tolerance) {
        const double error = std::fabs(
            static_cast<double>(actual) - expected);

        const double relative = error /
            std::max(1.0e-12, std::fabs(static_cast<double>(expected)));

        max_abs = std::max(max_abs, error);
        max_rel = std::max(max_rel, relative);
        sum_abs += error;
        ++count;

        if (!std::isfinite(actual) || !std::isfinite(expected) ||
            error > tolerance * std::max(1.0, std::fabs(static_cast<double>(expected)))) {
            if (failures < 5) {
                std::fprintf(stderr,
                    "[DIAGNOSTIC] actual=%.9g expected=%.9g "
                    "abs_error=%.9g relative_error=%.9g threshold=%.9g\\n",
                    actual, expected, error, relative,
                    tolerance * std::max(1.0, std::fabs(static_cast<double>(expected))));
            }
            ++failures;
        }
    }

    void print(const char* name) const {
        std::printf(
            "%-30s count=%d failures=%d max_abs=%.8g "
            "max_rel=%.8g mean_abs=%.8g\n",
            name, count, failures, max_abs, max_rel,
            count ? sum_abs / count : 0.0);
    }
};

static void set_fp16_one(uint8_t* p) {
    const uint16_t one = 0x3c00;
    std::memcpy(p, &one, sizeof(one));
}

static void make_q5_block(std::array<uint8_t, 176>& b,
                          std::mt19937& rng) {
    for (auto& v : b)
        v = static_cast<uint8_t>(rng() & 0xff);

    set_fp16_one(b.data());
    // dmin = 0.25
    const uint16_t quarter = 0x3400;
    std::memcpy(b.data() + 2, &quarter, sizeof(quarter));
}

static void make_q6_block(std::array<uint8_t, 210>& b,
                          std::mt19937& rng) {
    for (auto& v : b)
        v = static_cast<uint8_t>(rng() & 0xff);

    set_fp16_one(b.data() + 208);
}

} // namespace

#ifndef Q8K_BENCH_ONLY
int main() {
    init_types();

    std::mt19937 rng(0xEA11F);

    std::uniform_real_distribution<float> activation_dist(-2.0f, 2.0f);

    int q5_decode_mismatches = 0;
    int q6_decode_mismatches = 0;

    QuantizerParityStats q8_parity;

    ScaledErrorStats q5_double_reference;
    ScaledErrorStats q6_double_reference;
    ErrorStats q5_kernel;
    ErrorStats q6_kernel;
    ErrorStats q5_quantization;
    ErrorStats q6_quantization;

    constexpr int trials = 5000;

    std::array<float, K> activation{};
    std::array<float, K> dequantized_weights{};

    for (int trial = 0; trial < trials; ++trial) {
        for (float& v : activation)
            v = activation_dist(rng);

        if (trial % 100 == 0)
            activation.fill(0.0f);

        Q8K q8{};
        quantize_q8k(activation.data(), q8);

        Q8K q8_reference{};
        ggml_reference_quantize_q8k(
            activation.data(), q8_reference
        );
        q8_parity.add(q8, q8_reference);

        std::array<uint8_t, 176> q5{};
        make_q5_block(q5, rng);

        dq_q5_k(q5.data(), dequantized_weights.data());
        q5_decode_mismatches += verify_q5_decode(q5.data(), dequantized_weights.data());

        const float q5_actual = dot_q5k_q8k(q5.data(), q8);
        const float q5_expected =
            reference_dot(dequantized_weights.data(), q8);

        const float q5_original =
            reference_float_dot(dequantized_weights.data(),
                                activation.data());

        q5_kernel.add(q5_actual, q5_expected, 2.0e-5);

        const DotReference q5_ref =
            reference_dot_double(dequantized_weights.data(), q8);
        q5_double_reference.add(q5_actual, q5_ref);
        q5_quantization.add(q5_actual, q5_original, 1.0);

        std::array<uint8_t, 210> q6{};
        make_q6_block(q6, rng);

        dq_q6_k(q6.data(), dequantized_weights.data());
        q6_decode_mismatches += verify_q6_decode(q6.data(), dequantized_weights.data());

        const float q6_actual = dot_q6k_q8k(q6.data(), q8);
        const float q6_expected =
            reference_dot(dequantized_weights.data(), q8);

        const float q6_original =
            reference_float_dot(dequantized_weights.data(),
                                activation.data());

        q6_kernel.add(q6_actual, q6_expected, 2.0e-5);

        const DotReference q6_ref =
            reference_dot_double(dequantized_weights.data(), q8);
        q6_double_reference.add(q6_actual, q6_ref);
        q6_quantization.add(q6_actual, q6_original, 1.0);
    }


    // Deterministic Q8_K edge cases.
    std::array<float, K> edge{};

    auto check_edge = [&](const char* name) {
        Q8K actual{};
        Q8K expected{};

        quantize_q8k(edge.data(), actual);
        ggml_reference_quantize_q8k(edge.data(), expected);
        q8_parity.add(actual, expected);

        std::printf("Edge case checked: %s\n", name);
    };

    edge.fill(0.0f);
    check_edge("all zero");

    edge.fill(1.0f);
    check_edge("all positive");

    edge.fill(-1.0f);
    check_edge("all negative");

    for (int i = 0; i < K; ++i)
        edge[i] = (i & 1) ? -1.0f : 1.0f;
    check_edge("alternating signs");

    edge.fill(0.0f);
    edge[137] = 1.0f;
    edge[19] = -0.5f;
    check_edge("sparse activations");

    for (int i = 0; i < K; ++i)
        edge[i] = (static_cast<float>(i) - 127.5f) / 127.5f;
    check_edge("linear ramp");

    edge[0] = 1.0f;
    for (int i = 1; i < K; ++i) {
        const int q = (i % 253) - 126;
        edge[i] = (static_cast<float>(q) + 0.5f) / 127.0f;
    }
    check_edge("rounding boundaries");

    std::printf("\n========== Q8_K INTEGER DOT VALIDATION ==========\n");
    std::printf("Randomized blocks: %d\n", trials);
    std::printf("Q5 element decode mismatches: %d\n", q5_decode_mismatches);
    std::printf("Q6 element decode mismatches: %d\n", q6_decode_mismatches);
    std::printf("Seed: 0xEA11F\n\n");

    q8_parity.print();
    q5_kernel.print("Q5_K legacy FP32 diagnostic");
    q6_kernel.print("Q6_K legacy FP32 diagnostic");

    std::printf("\\n========== DOUBLE REFERENCE VALIDATION ==========\\n");
    q5_double_reference.print("Q5_K double reference");
    q6_double_reference.print("Q6_K double reference");

    std::printf("\n========== Q8_K ACTIVATION APPROXIMATION ==========\n");
    q5_quantization.print("Q5_K quantized vs FP32");
    q6_quantization.print("Q6_K quantized vs FP32");

    const bool passed =
        q8_parity.passed() &&
        q5_decode_mismatches == 0 &&
        q6_decode_mismatches == 0 &&
        q5_double_reference.over_1e_6 == 0 &&
        q6_double_reference.over_1e_6 == 0;

    std::printf("\nKERNEL CORRECTNESS: %s\n",
                passed ? "PASS" : "FAIL");

    return passed ? 0 : 1;
}
#endif // Q8K_BENCH_ONLY
