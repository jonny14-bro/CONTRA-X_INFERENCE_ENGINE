#include "pwea_unicode.h"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static bool from_hex(const std::string& hex, std::string& bytes) {
    if (hex.size() % 2 != 0) return false;

    bytes.clear();

    for (size_t i = 0; i < hex.size(); i += 2) {
        unsigned int value = 0;
        std::istringstream stream(hex.substr(i, 2));

        if (!(stream >> std::hex >> value)) return false;

        bytes.push_back(static_cast<char>(value));
    }

    return true;
}

static std::string to_hex(const std::string& bytes) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');

    for (unsigned char byte : bytes) {
        out << std::setw(2) << static_cast<unsigned int>(byte);
    }

    return out.str();
}

int main() {
    std::string line;

    while (std::getline(std::cin, line)) {
        std::string input;

        if (!from_hex(line, input)) {
            std::cout << "ERROR\n";
            continue;
        }

        try {
            std::vector<uint32_t> cps;

            if (!pwea::unicode::decode_utf8(input, cps)) {
                std::cout << "ERROR\n";
                continue;
            }

            std::cout << to_hex(pwea::unicode::lowercase(input));

            if (cps.size() == 1) {
                std::cout << '\t'
                          << pwea::unicode::is_word(cps[0])
                          << '\t'
                          << pwea::unicode::is_space(cps[0]);
            }

            std::cout << '\n';
        } catch (...) {
            std::cout << "ERROR\n";
        }
    }

    return 0;
}
