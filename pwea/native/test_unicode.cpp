#include "pwea_unicode.h"

#include <iostream>
#include <string>

int main() {
    const std::string tests[] = {
        "Hello WORLD 123",
        "Caf\xc3\xa9 na\xc3\xafve",
        "\xce\x9f\xce\xa3",                 // ΟΣ
        "\xce\x9f\xce\xa3\xce\x91",         // ΟΣΑ
        "A\xce\xa3",
        "A\xce\xa3\xcc\x81",                // AΣ + combining acute
        "\xc4\xb0stanbul",                  // İstanbul
        "\xe1\xba\x9e",                     // ẞ
        "\xef\xac\x83",                     // ﬃ
        "\xe0\xa4\xa8\xe0\xa4\xae\xe0\xa4\xb8"
        "\xe0\xa5\x8d\xe0\xa4\xa4\xe0\xa5\x87"
    };

    for (const auto& input : tests) {
        try {
            std::cout << pwea::unicode::lowercase(input) << '\n';
        } catch (const std::exception& e) {
            std::cerr << "ERROR: " << e.what() << '\n';
            return 1;
        }
    }

    std::cout << "WORD underscore: "
              << pwea::unicode::is_word('_') << '\n';

    std::cout << "WORD combining acute: "
              << pwea::unicode::is_word(0x0301) << '\n';

    std::cout << "SPACE NBSP: "
              << pwea::unicode::is_space(0x00A0) << '\n';

    return 0;
}
