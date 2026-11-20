#include "pwea_model.h"
#include "pwea_hash.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace pwea::model {
namespace {

uint32_t read_u32_le(const uint8_t* p) {
    return uint32_t(p[0])
        | (uint32_t(p[1]) << 8)
        | (uint32_t(p[2]) << 16)
        | (uint32_t(p[3]) << 24);
}

uint64_t read_u64_le(const uint8_t* p) {
    uint64_t value = 0;

    for (unsigned i = 0; i < 8; ++i) {
        value |= uint64_t(p[i]) << (8 * i);
    }

    return value;
}

double read_f64_le(const uint8_t* p) {
    static_assert(sizeof(double) == sizeof(uint64_t));

    const uint64_t bits = read_u64_le(p);
    double value;

    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

bool hashes_equal(
    const std::array<uint8_t, 32>& a,
    const std::array<uint8_t, 32>& b
) {
    uint8_t difference = 0;

    for (size_t i = 0; i < a.size(); ++i) {
        difference |= a[i] ^ b[i];
    }

    return difference == 0;
}

} // namespace

Model load_model(const std::string& path) {
    std::ifstream file(path, std::ios::binary);

    if (!file) {
        throw std::runtime_error("Cannot open PWEA model: " + path);
    }

    file.seekg(0, std::ios::end);
    const std::streamoff file_size = file.tellg();

    if (file_size < static_cast<std::streamoff>(HEADER_SIZE)) {
        throw std::runtime_error("PWEA model file is shorter than its header");
    }

    file.seekg(0, std::ios::beg);

    uint8_t header[HEADER_SIZE]{};

    if (!file.read(
        reinterpret_cast<char*>(header),
        static_cast<std::streamsize>(HEADER_SIZE)
    )) {
        throw std::runtime_error("Cannot read PWEA model header");
    }

    const uint8_t expected_magic[8] = {
        'P', 'W', 'E', 'A', 'M', 'D', 'L', '1'
    };

    if (std::memcmp(header, expected_magic, 8) != 0) {
        throw std::runtime_error("Invalid PWEA model magic");
    }

    const uint32_t version = read_u32_le(header + 8);
    const uint32_t samples = read_u32_le(header + 12);
    const uint32_t input_dim = read_u32_le(header + 16);
    const uint32_t output_dim = read_u32_le(header + 20);
    const uint32_t layers = read_u32_le(header + 24);
    const uint32_t experts = read_u32_le(header + 28);
    const uint32_t dtype_code = read_u32_le(header + 32);
    const uint64_t payload_bytes = read_u64_le(header + 36);

    if (version != FORMAT_VERSION ||
        samples != N_SAMPLES ||
        input_dim != INPUT_DIM ||
        output_dim != OUTPUT_DIM ||
        layers != N_LAYER ||
        experts != N_EXPERT ||
        dtype_code != 1) {
        throw std::runtime_error("Unsupported PWEA model header");
    }

    constexpr uint64_t expected_payload =
        uint64_t(N_SAMPLES) * INPUT_DIM * sizeof(double)
        + uint64_t(N_SAMPLES) * OUTPUT_DIM * sizeof(double);

    if (payload_bytes != expected_payload) {
        throw std::runtime_error("Unexpected PWEA payload size");
    }

    if (static_cast<uint64_t>(file_size) !=
        uint64_t(HEADER_SIZE) + payload_bytes) {
        throw std::runtime_error("PWEA model file size mismatch");
    }

    std::array<uint8_t, 32> expected_payload_hash{};
    std::array<uint8_t, 32> source_hash{};

    std::memcpy(expected_payload_hash.data(), header + 44, 32);
    std::memcpy(source_hash.data(), header + 76, 32);

    std::vector<uint8_t> payload(
        static_cast<size_t>(payload_bytes)
    );

    if (!file.read(
        reinterpret_cast<char*>(payload.data()),
        static_cast<std::streamsize>(payload.size())
    )) {
        throw std::runtime_error("Cannot read PWEA model payload");
    }

    const auto actual_hash = pwea::crypto::sha256(
        payload.data(),
        payload.size()
    );

    if (!hashes_equal(actual_hash, expected_payload_hash)) {
        throw std::runtime_error("PWEA payload SHA-256 verification failed");
    }

    constexpr size_t X_COUNT =
        size_t(N_SAMPLES) * INPUT_DIM;

    constexpr size_t C_COUNT =
        size_t(N_SAMPLES) * OUTPUT_DIM;

    Model model;

    model.payload_sha256 = actual_hash;
    model.source_npz_sha256 = source_hash;

    model.X_train.resize(X_COUNT);
    model.dual_coefficients.resize(C_COUNT);

    size_t offset = 0;

    for (size_t i = 0; i < X_COUNT; ++i, offset += 8) {
        const double value = read_f64_le(payload.data() + offset);

        if (!std::isfinite(value)) {
            throw std::runtime_error("Non-finite value in X_train");
        }

        model.X_train[i] = value;
    }

    for (size_t i = 0; i < C_COUNT; ++i, offset += 8) {
        const double value = read_f64_le(payload.data() + offset);

        if (!std::isfinite(value)) {
            throw std::runtime_error(
                "Non-finite value in dual_coefficients"
            );
        }

        model.dual_coefficients[i] = value;
    }

    if (offset != payload.size()) {
        throw std::runtime_error("Internal PWEA payload offset mismatch");
    }

    return model;
}

} // namespace pwea::model
