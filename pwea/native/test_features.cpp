#include "pwea_features.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static uint32_t read_u32(std::ifstream& in) {
    unsigned char b[4];

    if (!in.read(reinterpret_cast<char*>(b), 4)) {
        throw std::runtime_error("Unexpected end of fixture");
    }

    return uint32_t(b[0]) |
           (uint32_t(b[1]) << 8) |
           (uint32_t(b[2]) << 16) |
           (uint32_t(b[3]) << 24);
}

int main(int argc, char** argv) {
    const std::string path = argc > 1
        ? argv[1]
        : "pwea/native/tests/fixtures/pwea_feature_reference.bin";

    std::ifstream in(path, std::ios::binary);

    if (!in) {
        std::cerr << "Cannot open fixture: " << path << '\n';
        return 1;
    }

    char magic[8];

    if (!in.read(magic, 8) ||
        std::string(magic, 8) != "PWEAFEAT") {
        std::cerr << "Invalid fixture header\n";
        return 1;
    }

    const uint32_t cases = read_u32(in);
    const uint32_t dimensions = read_u32(in);

    if (dimensions != pwea::features::TOTAL_DIM) {
        std::cerr << "Unexpected feature dimensions\n";
        return 1;
    }

    std::cout << "Cases: " << cases
              << " Dimensions: " << dimensions << "\n\n";

    constexpr double TOLERANCE = 1e-12;
    bool all_passed = true;

    for (uint32_t c = 0; c < cases; ++c) {
        const uint32_t prompt_size = read_u32(in);

        std::string prompt(prompt_size, '\0');

        if (!in.read(prompt.data(), prompt_size)) {
            throw std::runtime_error("Truncated prompt");
        }

        std::vector<double> reference(dimensions);

        if (!in.read(
            reinterpret_cast<char*>(reference.data()),
            dimensions * sizeof(double)
        )) {
            throw std::runtime_error("Truncated reference vector");
        }

        const auto actual =
            pwea::features::transform_balanced_hybrid(prompt);

        double max_abs = 0.0;
        double squared_error = 0.0;
        size_t mismatches = 0;

        for (size_t i = 0; i < dimensions; ++i) {
            const double diff = std::abs(actual[i] - reference[i]);

            max_abs = std::max(max_abs, diff);
            squared_error += diff * diff;

            if (diff > TOLERANCE) {
                ++mismatches;
            }
        }

        const double rmse =
            std::sqrt(squared_error / dimensions);

        const bool passed = mismatches == 0;
        all_passed &= passed;

        std::cout
            << "Case " << (c + 1)
            << ": " << (passed ? "PASS" : "FAIL")
            << " | max_abs=" << std::scientific << max_abs
            << " | rmse=" << rmse
            << " | mismatches=" << mismatches
            << '\n';
    }

    std::cout << "\n"
              << (all_passed
                  ? "ALL FEATURE FIXTURES PASSED"
                  : "FEATURE FIXTURE MISMATCHES DETECTED")
              << '\n';

    return all_passed ? 0 : 2;
}
