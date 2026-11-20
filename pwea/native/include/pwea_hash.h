#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace pwea::crypto {

std::array<uint8_t, 32> sha256(
    const uint8_t* data,
    size_t size
);

}
