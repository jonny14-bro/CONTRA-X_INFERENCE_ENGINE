#include "pwea_predictor.h"
#include "pwea_features.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace pwea::predictor {

std::vector<double> project_simplex(
    const double* values,
    size_t size
) {
    if (values == nullptr || size == 0) {
        throw std::invalid_argument(
            "Simplex input must be non-empty"
        );
    }

    std::vector<double> ordered(values, values + size);

    for (double value : ordered) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "Simplex input contains non-finite values"
            );
        }
    }

    std::sort(
        ordered.begin(),
        ordered.end(),
        std::greater<double>()
    );

    double cumulative = 0.0;
    size_t rho = 0;
    bool found = false;

    for (size_t i = 0; i < size; ++i) {
        cumulative += ordered[i];

        const double threshold =
            (cumulative - 1.0) / double(i + 1);

        if (ordered[i] - threshold > 0.0) {
            rho = i;
            found = true;
        }
    }

    if (!found) {
        throw std::runtime_error("Simplex projection failed");
    }

    cumulative = 0.0;

    for (size_t i = 0; i <= rho; ++i) {
        cumulative += ordered[i];
    }

    const double threshold =
        (cumulative - 1.0) / double(rho + 1);

    std::vector<double> result(size, 0.0);
    double sum = 0.0;

    for (size_t i = 0; i < size; ++i) {
        result[i] = std::max(values[i] - threshold, 0.0);
        sum += result[i];
    }

    if (!std::isfinite(sum) || sum <= 0.0) {
        throw std::runtime_error(
            "Invalid simplex projection normalization"
        );
    }

    for (double& value : result) {
        value /= sum;
    }

    return result;
}

std::vector<double> predict(
    const model::Model& model,
    const std::vector<double>& features
) {
    constexpr size_t SAMPLES = model::N_SAMPLES;
    constexpr size_t INPUT = model::INPUT_DIM;
    constexpr size_t OUTPUT = model::OUTPUT_DIM;

    if (features.size() != INPUT) {
        throw std::invalid_argument(
            "PWEA feature dimension mismatch"
        );
    }

    if (model.X_train.size() != SAMPLES * INPUT ||
        model.dual_coefficients.size() != SAMPLES * OUTPUT) {
        throw std::invalid_argument(
            "PWEA model matrix dimensions are invalid"
        );
    }

    for (double value : features) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "PWEA features contain non-finite values"
            );
        }
    }

    // Python reference:
    // kernel = X @ X_train.T
    std::vector<double> kernel(SAMPLES, 0.0);

    for (size_t sample = 0; sample < SAMPLES; ++sample) {
        const double* train =
            model.X_train.data() + sample * INPUT;

        double dot = 0.0;

        for (size_t feature = 0; feature < INPUT; ++feature) {
            dot += features[feature] * train[feature];
        }

        if (!std::isfinite(dot)) {
            throw std::runtime_error(
                "Non-finite PWEA kernel value"
            );
        }

        kernel[sample] = dot;
    }

    // Python reference:
    // raw = kernel @ dual_coefficients
    std::vector<double> raw(OUTPUT, 0.0);

    for (size_t sample = 0; sample < SAMPLES; ++sample) {
        const double weight = kernel[sample];

        const double* coefficients =
            model.dual_coefficients.data() + sample * OUTPUT;

        for (size_t output = 0; output < OUTPUT; ++output) {
            raw[output] += weight * coefficients[output];
        }
    }

    for (double value : raw) {
        if (!std::isfinite(value)) {
            throw std::runtime_error(
                "PWEA raw prediction contains non-finite values"
            );
        }
    }

    // Independently project each layer onto its 128-expert simplex.
    std::vector<double> result(OUTPUT);

    for (size_t layer = 0; layer < N_LAYER; ++layer) {
        const size_t offset = layer * N_EXPERT;

        const auto projected = project_simplex(
            raw.data() + offset,
            N_EXPERT
        );

        std::copy(
            projected.begin(),
            projected.end(),
            result.begin() + offset
        );
    }

    for (size_t layer = 0; layer < N_LAYER; ++layer) {
        double sum = 0.0;

        for (size_t expert = 0; expert < N_EXPERT; ++expert) {
            const double value =
                result[layer * N_EXPERT + expert];

            if (!std::isfinite(value) || value < 0.0) {
                throw std::runtime_error(
                    "Invalid PWEA simplex output"
                );
            }

            sum += value;
        }

        if (std::abs(sum - 1.0) > 1e-10) {
            throw std::runtime_error(
                "PWEA layer simplex validation failed"
            );
        }
    }

    return result;
}

std::vector<double> predict_prompt(
    const model::Model& model,
    const std::string& prompt
) {
    if (prompt.empty()) {
        throw std::invalid_argument(
            "PWEA prompt must not be empty"
        );
    }

    const auto features =
        pwea::features::transform_balanced_hybrid(prompt);

    return predict(model, features);
}

}
