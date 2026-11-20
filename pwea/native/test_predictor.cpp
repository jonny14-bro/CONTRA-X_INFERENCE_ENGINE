#include "pwea_model.h"
#include "pwea_predictor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static uint32_t read_u32(std::ifstream& in) {
    unsigned char b[4];
    in.read(reinterpret_cast<char*>(b), 4);

    if (!in) {
        throw std::runtime_error("Failed to read uint32");
    }

    return static_cast<uint32_t>(b[0])
        | (static_cast<uint32_t>(b[1]) << 8)
        | (static_cast<uint32_t>(b[2]) << 16)
        | (static_cast<uint32_t>(b[3]) << 24);
}

static double read_f64(std::ifstream& in) {
    unsigned char b[8];
    in.read(reinterpret_cast<char*>(b), 8);

    if (!in) {
        throw std::runtime_error("Failed to read float64");
    }

    uint64_t bits = 0;

    for (int i = 0; i < 8; ++i) {
        bits |= static_cast<uint64_t>(b[i]) << (8 * i);
    }

    double value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

int main() {
    try {
        const std::string model_path =
            "pwea/native/models/pwea_model_v1.bin";

        const std::string fixture_path =
            "pwea/native/tests/fixtures/pwea_prediction_reference.bin";

        std::cout << "Loading native PWEA model...\n";

        const auto model = pwea::model::load_model(model_path);

        std::ifstream in(fixture_path, std::ios::binary);

        if (!in) {
            throw std::runtime_error(
                "Cannot open prediction reference fixture"
            );
        }

        char magic[8];
        in.read(magic, 8);

        if (!in || std::memcmp(magic, "PWEAPRF1", 8) != 0) {
            throw std::runtime_error("Invalid prediction fixture magic");
        }

        const uint32_t cases = read_u32(in);

        if (cases != 8) {
            throw std::runtime_error("Unexpected reference case count");
        }

        constexpr size_t OUTPUT_DIM =
            pwea::predictor::OUTPUT_DIM;

        constexpr size_t N_LAYER =
            pwea::predictor::N_LAYER;

        constexpr size_t N_EXPERT =
            pwea::predictor::N_EXPERT;

        double global_max_abs = 0.0;
        long double global_squared_error = 0.0;
        size_t global_values = 0;

        for (uint32_t c = 0; c < cases; ++c) {
            const uint32_t prompt_size = read_u32(in);

            std::string prompt(prompt_size, '\0');
            in.read(prompt.data(), prompt_size);

            if (!in) {
                throw std::runtime_error("Failed to read reference prompt");
            }

            std::vector<double> reference(OUTPUT_DIM);

            for (double& value : reference) {
                value = read_f64(in);
            }

            const auto actual =
                pwea::predictor::predict_prompt(model, prompt);

            if (actual.size() != OUTPUT_DIM) {
                throw std::runtime_error("Native prediction size mismatch");
            }

            double max_abs = 0.0;
            long double squared_error = 0.0;

            for (size_t i = 0; i < OUTPUT_DIM; ++i) {
                const double error =
                    std::abs(actual[i] - reference[i]);

                max_abs = std::max(max_abs, error);
                squared_error +=
                    static_cast<long double>(error) * error;
            }

            const double rmse = std::sqrt(
                static_cast<double>(squared_error / OUTPUT_DIM)
            );

            double max_simplex_error = 0.0;

            for (size_t layer = 0; layer < N_LAYER; ++layer) {
                double sum = 0.0;

                for (size_t expert = 0; expert < N_EXPERT; ++expert) {
                    sum += actual[layer * N_EXPERT + expert];
                }

                max_simplex_error = std::max(
                    max_simplex_error,
                    std::abs(sum - 1.0)
                );
            }

            global_max_abs =
                std::max(global_max_abs, max_abs);

            global_squared_error += squared_error;
            global_values += OUTPUT_DIM;

            const bool pass =
                max_abs <= 1e-10 &&
                max_simplex_error <= 1e-10;

            std::cout
                << "Case " << (c + 1)
                << ": " << (pass ? "PASS" : "FAIL")
                << " | max_abs=" << std::scientific << max_abs
                << " | rmse=" << rmse
                << " | simplex_error=" << max_simplex_error
                << '\n';

            if (!pass) {
                throw std::runtime_error(
                    "Native prediction differs from Python reference"
                );
            }
        }

        char extra;

        if (in.read(&extra, 1)) {
            throw std::runtime_error(
                "Unexpected trailing data in prediction fixture"
            );
        }

        const double global_rmse = std::sqrt(
            static_cast<double>(
                global_squared_error / global_values
            )
        );

        std::cout << "\nGLOBAL MAX ABS ERROR: "
                  << global_max_abs << '\n';

        std::cout << "GLOBAL RMSE: "
                  << global_rmse << '\n';

        std::cout << "\nNATIVE PREDICTION DIFFERENTIAL TEST: PASS\n";

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "\nTEST FAILURE: " << e.what() << '\n';
        return 1;
    }
}
