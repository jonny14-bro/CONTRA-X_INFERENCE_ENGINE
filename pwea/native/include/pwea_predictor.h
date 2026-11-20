#pragma once

#include "pwea_model.h"

#include <string>
#include <vector>

namespace pwea::predictor {

constexpr size_t N_LAYER = 48;
constexpr size_t N_EXPERT = 128;
constexpr size_t OUTPUT_DIM = N_LAYER * N_EXPERT;

std::vector<double> project_simplex(
    const double* values,
    size_t size
);

std::vector<double> predict(
    const model::Model& model,
    const std::vector<double>& features
);

std::vector<double> predict_prompt(
    const model::Model& model,
    const std::string& prompt
);

}
