#include "pwea_model.h"
#include "pwea_predictor.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr
            << "Usage: export_prediction MODEL PROMPT_FILE OUTPUT_CSV\n";
        return 2;
    }

    try {
        const std::string model_path = argv[1];
        const std::string prompt_path = argv[2];
        const std::string output_path = argv[3];

        std::ifstream prompt_file(prompt_path);

        if (!prompt_file) {
            throw std::runtime_error("Cannot open prompt file");
        }

        const std::string prompt(
            (std::istreambuf_iterator<char>(prompt_file)),
            std::istreambuf_iterator<char>()
        );

        if (prompt.empty()) {
            throw std::runtime_error("Prompt is empty");
        }

        std::cout << "Loading native PWEA model...\n";

        const auto model =
            pwea::model::load_model(model_path);

        std::cout << "Running native predict_prompt()...\n";

        const auto prediction =
            pwea::predictor::predict_prompt(model, prompt);

        constexpr size_t LAYERS = 48;
        constexpr size_t EXPERTS = 128;
        constexpr size_t EXPECTED = LAYERS * EXPERTS;

        if (prediction.size() != EXPECTED) {
            throw std::runtime_error(
                "Prediction output dimension mismatch"
            );
        }

        for (size_t layer = 0; layer < LAYERS; ++layer) {
            double sum = 0.0;

            for (size_t expert = 0; expert < EXPERTS; ++expert) {
                const double score =
                    prediction[layer * EXPERTS + expert];

                if (!std::isfinite(score) || score < 0.0) {
                    throw std::runtime_error(
                        "Invalid prediction value"
                    );
                }

                sum += score;
            }

            if (std::abs(sum - 1.0) > 1e-10) {
                throw std::runtime_error(
                    "Layer simplex validation failed"
                );
            }
        }

        std::ofstream out(output_path);

        if (!out) {
            throw std::runtime_error("Cannot create output CSV");
        }

        out << "layer,expert_id,predicted_probability\n";
        out << std::setprecision(17);

        for (size_t layer = 0; layer < LAYERS; ++layer) {
            for (size_t expert = 0; expert < EXPERTS; ++expert) {
                out << layer << ","
                    << expert << ","
                    << prediction[layer * EXPERTS + expert]
                    << "\n";
            }
        }

        std::cout << "Prediction export successful.\n";
        std::cout << "Layers: " << LAYERS << "\n";
        std::cout << "Experts per layer: " << EXPERTS << "\n";
        std::cout << "Total predictions: " << prediction.size() << "\n";
        std::cout << "Simplex validation: PASSED\n";
        std::cout << "Output: " << output_path << "\n";

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
