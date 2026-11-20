#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace pwea::model {

constexpr uint32_t FORMAT_VERSION = 1;
constexpr uint32_t N_SAMPLES = 140;
constexpr uint32_t INPUT_DIM = 32768;
constexpr uint32_t OUTPUT_DIM = 6144;
constexpr uint32_t N_LAYER = 48;
constexpr uint32_t N_EXPERT = 128;
constexpr size_t HEADER_SIZE = 108;

struct Model {
    std::vector<double> X_train;
    std::vector<double> dual_coefficients;

    std::array<uint8_t, 32> payload_sha256{};
    std::array<uint8_t, 32> source_npz_sha256{};
};

Model load_model(const std::string& path);

}
