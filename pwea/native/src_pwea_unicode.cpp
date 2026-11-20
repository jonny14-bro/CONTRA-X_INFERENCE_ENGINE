#include "pwea_unicode.h"
#include "pwea_unicode_tables.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace pwea::unicode {

template <size_t N>
static bool in_ranges(
    const uint32_t (&ranges)[N][2],
    size_t count,
    uint32_t cp
) {
    size_t lo = 0;
    size_t hi = count;

    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;

        if (cp < ranges[mid][0]) {
            hi = mid;
        } else if (cp > ranges[mid][1]) {
            lo = mid + 1;
        } else {
            return true;
        }
    }

    return false;
}

static const LowerEntry* find_lower(uint32_t cp) {
    size_t lo = 0;
    size_t hi = LOWER_MAP_N;

    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;

        if (LOWER_MAP[mid].codepoint < cp) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    if (lo < LOWER_MAP_N && LOWER_MAP[lo].codepoint == cp) {
        return &LOWER_MAP[lo];
    }

    return nullptr;
}

bool is_word(uint32_t cp) {
    return in_ranges(PY_WORD, PY_WORD_N, cp);
}

bool is_space(uint32_t cp) {
    return in_ranges(PY_SPACE, PY_SPACE_N, cp);
}

bool decode_utf8(
    const std::string& input,
    std::vector<uint32_t>& output
) {
    output.clear();
    output.reserve(input.size());

    size_t i = 0;

    while (i < input.size()) {
        const uint8_t first = static_cast<uint8_t>(input[i]);
        uint32_t cp = 0;
        size_t length = 0;

        if (first <= 0x7F) {
            cp = first;
            length = 1;
        } else if (first >= 0xC2 && first <= 0xDF) {
            cp = first & 0x1F;
            length = 2;
        } else if (first >= 0xE0 && first <= 0xEF) {
            cp = first & 0x0F;
            length = 3;
        } else if (first >= 0xF0 && first <= 0xF4) {
            cp = first & 0x07;
            length = 4;
        } else {
            return false;
        }

        if (i + length > input.size()) {
            return false;
        }

        for (size_t j = 1; j < length; ++j) {
            const uint8_t next =
                static_cast<uint8_t>(input[i + j]);

            if ((next & 0xC0) != 0x80) {
                return false;
            }

            cp = (cp << 6) | (next & 0x3F);
        }

        // Reject overlong encodings, surrogates and out-of-range values.
        if ((length == 2 && cp < 0x80) ||
            (length == 3 && cp < 0x800) ||
            (length == 4 && cp < 0x10000) ||
            (cp >= 0xD800 && cp <= 0xDFFF) ||
            cp > 0x10FFFF) {
            return false;
        }

        output.push_back(cp);
        i += length;
    }

    return true;
}

void encode_utf8(uint32_t cp, std::string& output) {
    if (cp <= 0x7F) {
        output.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        output.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        output.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        output.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        output.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        output.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        output.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        output.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        output.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        output.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

static bool is_cased(uint32_t cp) {
    return in_ranges(
        UNICODE_CASED,
        UNICODE_CASED_N,
        cp
    );
}

static bool is_case_ignorable(uint32_t cp) {
    return in_ranges(
        UNICODE_CASE_IGNORABLE,
        UNICODE_CASE_IGNORABLE_N,
        cp
    );
}

static bool is_final_sigma(
    const std::vector<uint32_t>& codepoints,
    size_t index
) {
    bool preceded_by_cased = false;
    bool followed_by_cased = false;

    // Search backwards, skipping Case_Ignorable characters.
    for (size_t i = index; i > 0;) {
        --i;

        const uint32_t cp = codepoints[i];

        if (is_case_ignorable(cp)) {
            continue;
        }

        preceded_by_cased = is_cased(cp);
        break;
    }

    if (!preceded_by_cased) {
        return false;
    }

    // Search forwards, skipping Case_Ignorable characters.
    for (size_t i = index + 1; i < codepoints.size(); ++i) {
        const uint32_t cp = codepoints[i];

        if (is_case_ignorable(cp)) {
            continue;
        }

        followed_by_cased = is_cased(cp);
        break;
    }

    return !followed_by_cased;
}

std::string lowercase(const std::string& input) {
    std::vector<uint32_t> codepoints;

    if (!decode_utf8(input, codepoints)) {
        throw std::invalid_argument("Invalid UTF-8 input");
    }

    std::string output;
    output.reserve(input.size());

    for (size_t i = 0; i < codepoints.size(); ++i) {
        const uint32_t cp = codepoints[i];

        // Unicode SpecialCasing: Final_Sigma.
        if (cp == 0x03A3) {
            encode_utf8(
                is_final_sigma(codepoints, i) ? 0x03C2 : 0x03C3,
                output
            );
            continue;
        }

        const LowerEntry* entry = find_lower(cp);

        if (entry == nullptr) {
            encode_utf8(cp, output);
            continue;
        }

        for (uint8_t j = 0; j < entry->length; ++j) {
            encode_utf8(
                LOWER_DATA[entry->offset + j],
                output
            );
        }
    }

    return output;
}

}
