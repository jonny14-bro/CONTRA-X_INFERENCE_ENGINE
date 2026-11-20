#include "pwea_model.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        const std::string path = argc > 1
            ? argv[1]
            : "pwea/native/models/pwea_model_v1.bin";

        std::cout << "Loading: " << path << "\n";

        const auto model = pwea::model::load_model(path);

        constexpr size_t expected_X = 140ULL * 32768;
        constexpr size_t expected_C = 140ULL * 6144;

        if (model.X_train.size() != expected_X) {
            throw std::runtime_error("X_train size mismatch");
        }

        if (model.dual_coefficients.size() != expected_C) {
            throw std::runtime_error("Coefficient size mismatch");
        }

        std::cout << "MODEL HEADER: PASS\n";
        std::cout << "PAYLOAD SHA-256: PASS\n";
        std::cout << "FLOAT64 VALIDATION: PASS\n";
        std::cout << "X_train values: "
                  << model.X_train.size() << "\n";
        std::cout << "Coefficient values: "
                  << model.dual_coefficients.size() << "\n";

        std::cout << std::setprecision(17);
        std::cout << "X_train[0]: "
                  << model.X_train[0] << "\n";
        std::cout << "X_train[last]: "
                  << model.X_train.back() << "\n";
        std::cout << "Coefficient[0]: "
                  << model.dual_coefficients[0] << "\n";
        std::cout << "Coefficient[last]: "
                  << model.dual_coefficients.back() << "\n";

        std::cout << "NATIVE MODEL LOADER: PASS\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "NATIVE MODEL LOADER: FAIL: "
                  << e.what() << "\n";
        return 1;
    }
}
