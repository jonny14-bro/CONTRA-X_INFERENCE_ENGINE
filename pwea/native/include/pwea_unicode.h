#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace pwea::unicode {

bool decode_utf8(const std::string& input, std::vector<uint32_t>& output);
void encode_utf8(uint32_t cp, std::string& output);
bool is_word(uint32_t cp);
bool is_space(uint32_t cp);
std::string lowercase(const std::string& input);

}
