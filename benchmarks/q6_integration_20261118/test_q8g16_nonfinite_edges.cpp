#include <cmath>
#include <cstdio>
#include <limits>

static bool valid_input(const float* x, int count) {
    for (int i = 0; i < count; ++i) {
        if (!std::isfinite(x[i]))
            return false;
    }
    return true;
}

int main() {
    constexpr int K = 256;

    float input[K]{};

    struct Test {
        const char* name;
        int index;
        float value;
        bool expected_valid;
    };

    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();

    const Test tests[] = {
        {"finite", 0, 1.0f, true},
        {"positive_infinity", 17, inf, false},
        {"negative_infinity", 128, -inf, false},
        {"quiet_nan", 255, nan, false},
        {"negative_zero", 64, -0.0f, true},
        {"max_finite", 100, std::numeric_limits<float>::max(), true}
    };

    int failures = 0;

    for (const auto& test : tests) {
        for (int i = 0; i < K; ++i)
            input[i] = 1.0f;

        input[test.index] = test.value;

        const bool actual = valid_input(input, K);
        const bool pass = actual == test.expected_valid;

        std::printf("%-24s expected=%s actual=%s %s\n",
            test.name,
            test.expected_valid ? "VALID" : "REJECT",
            actual ? "VALID" : "REJECT",
            pass ? "PASS" : "FAIL");

        failures += !pass;
    }

    std::printf("\nRESULT: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
