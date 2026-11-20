#include "q6_groupwise_q8_checked.h"

#include <cstdio>
#include <cstring>
#include <limits>

struct Test {
    const char* name;
    float value;
    bool expected;
};

int main() {
    const float denorm =
        std::numeric_limits<float>::denorm_min();

    const Test tests[] = {
        {"all_zero", 0.0f, true},
        {"normal", 1.0f, true},
        {"tiny_normal", 1e-36f, true},
        {"large_finite", 1e36f, true},
        {"max_finite", std::numeric_limits<float>::max(), true},
        {"subnormal_only", denorm, false},
        {"positive_infinity",
            std::numeric_limits<float>::infinity(), false},
        {"negative_infinity",
            -std::numeric_limits<float>::infinity(), false},
        {"quiet_nan",
            std::numeric_limits<float>::quiet_NaN(), false}
    };

    int failures = 0;

    for (const auto& test : tests) {
        float input[256];
        for (float& v : input)
            v = test.value;

        q6g16::Q8G16 scalar{};
        q6g16::Q8G16 simd{};

        const bool scalar_ok =
            q6g16_checked::quantize_scalar(input, scalar);

        const bool simd_ok =
            q6g16_checked::quantize_simd(input, simd);

        const bool bytes_equal =
            std::memcmp(&scalar, &simd, sizeof(scalar)) == 0;

        q6g16::Q8G16 zero_reference{};

        const bool output_zero =
            !scalar_ok &&
            !simd_ok &&
            std::memcmp(&scalar, &zero_reference,
                        sizeof(scalar)) == 0 &&
            std::memcmp(&simd, &zero_reference,
                        sizeof(simd)) == 0;

        const bool pass =
            scalar_ok == test.expected &&
            simd_ok == test.expected &&
            bytes_equal &&
            (test.expected || output_zero);

        std::printf(
            "%-22s scalar=%s simd=%s equal=%s %s\n",
            test.name,
            scalar_ok ? "ACCEPT" : "REJECT",
            simd_ok ? "ACCEPT" : "REJECT",
            bytes_equal ? "YES" : "NO",
            pass ? "PASS" : "FAIL"
        );

        failures += !pass;
    }

    std::printf("\nRESULT: %s\n",
                failures ? "FAIL" : "PASS");

    return failures ? 1 : 0;
}
