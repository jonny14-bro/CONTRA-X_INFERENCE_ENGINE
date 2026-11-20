#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace pwea::features {

constexpr size_t WORD_DIM = 16384;
constexpr size_t CHAR_DIM = 16384;
constexpr size_t TOTAL_DIM = WORD_DIM + CHAR_DIM;

std::vector<double> transform_balanced_hybrid(
    const std::string& prompt
);

}
