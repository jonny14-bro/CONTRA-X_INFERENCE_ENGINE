#include "pwea_features.h"
#include "pwea_unicode.h"
#include "pwea_hash.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace pwea::features {

static constexpr uint32_t K[64] = {
    0x428A2F98,0x71374491,0xB5C0FBCF,0xE9B5DBA5,
    0x3956C25B,0x59F111F1,0x923F82A4,0xAB1C5ED5,
    0xD807AA98,0x12835B01,0x243185BE,0x550C7DC3,
    0x72BE5D74,0x80DEB1FE,0x9BDC06A7,0xC19BF174,
    0xE49B69C1,0xEFBE4786,0x0FC19DC6,0x240CA1CC,
    0x2DE92C6F,0x4A7484AA,0x5CB0A9DC,0x76F988DA,
    0x983E5152,0xA831C66D,0xB00327C8,0xBF597FC7,
    0xC6E00BF3,0xD5A79147,0x06CA6351,0x14292967,
    0x27B70A85,0x2E1B2138,0x4D2C6DFC,0x53380D13,
    0x650A7354,0x766A0ABB,0x81C2C92E,0x92722C85,
    0xA2BFE8A1,0xA81A664B,0xC24B8B70,0xC76C51A3,
    0xD192E819,0xD6990624,0xF40E3585,0x106AA070,
    0x19A4C116,0x1E376C08,0x2748774C,0x34B0BCB5,
    0x391C0CB3,0x4ED8AA4A,0x5B9CCA4F,0x682E6FF3,
    0x748F82EE,0x78A5636F,0x84C87814,0x8CC70208,
    0x90BEFFFA,0xA4506CEB,0xBEF9A3F7,0xC67178F2
};

static uint32_t rotr(uint32_t x, unsigned n) {
    return (x >> n) | (x << (32 - n));
}

static std::array<uint8_t, 32> sha256(const std::string& input) {
    std::vector<uint8_t> data(input.begin(), input.end());
    const uint64_t bit_length = uint64_t(data.size()) * 8;

    data.push_back(0x80);

    while (data.size() % 64 != 56) {
        data.push_back(0);
    }

    for (int i = 7; i >= 0; --i) {
        data.push_back(static_cast<uint8_t>(
            bit_length >> (i * 8)
        ));
    }

    uint32_t h[8] = {
        0x6A09E667,0xBB67AE85,0x3C6EF372,0xA54FF53A,
        0x510E527F,0x9B05688C,0x1F83D9AB,0x5BE0CD19
    };

    for (size_t offset = 0; offset < data.size(); offset += 64) {
        uint32_t w[64]{};

        for (int i = 0; i < 16; ++i) {
            const size_t p = offset + i * 4;
            w[i] =
                (uint32_t(data[p]) << 24) |
                (uint32_t(data[p + 1]) << 16) |
                (uint32_t(data[p + 2]) << 8) |
                uint32_t(data[p + 3]);
        }

        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 =
                rotr(w[i - 15], 7) ^
                rotr(w[i - 15], 18) ^
                (w[i - 15] >> 3);

            const uint32_t s1 =
                rotr(w[i - 2], 17) ^
                rotr(w[i - 2], 19) ^
                (w[i - 2] >> 10);

            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }

        uint32_t a=h[0], b=h[1], c=h[2], d=h[3];
        uint32_t e=h[4], f=h[5], g=h[6], hh=h[7];

        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 =
                rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);

            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t temp1 = hh + S1 + ch + K[i] + w[i];

            const uint32_t S0 =
                rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);

            const uint32_t maj =
                (a & b) ^ (a & c) ^ (b & c);

            const uint32_t temp2 = S0 + maj;

            hh=g; g=f; f=e; e=d+temp1;
            d=c; c=b; b=a; a=temp1+temp2;
        }

        h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d;
        h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
    }

    std::array<uint8_t, 32> digest{};

    for (int i = 0; i < 8; ++i) {
        digest[i*4]     = uint8_t(h[i] >> 24);
        digest[i*4 + 1] = uint8_t(h[i] >> 16);
        digest[i*4 + 2] = uint8_t(h[i] >> 8);
        digest[i*4 + 3] = uint8_t(h[i]);
    }

    return digest;
}

static void add_feature(
    std::vector<double>& block,
    const std::string& feature
) {
    const auto digest = sha256(feature);

    uint64_t hash = 0;

    for (int i = 0; i < 8; ++i) {
        hash = (hash << 8) | digest[i];
    }

    const size_t index = hash % WORD_DIM;
    const double sign = (digest[8] & 1) ? 1.0 : -1.0;

    block[index] += sign;
}

static std::string encode_codepoints(
    const std::vector<uint32_t>& cps,
    size_t begin,
    size_t end
) {
    std::string result;

    for (size_t i = begin; i < end; ++i) {
        pwea::unicode::encode_utf8(cps[i], result);
    }

    return result;
}

static std::vector<std::string> tokenize(
    const std::vector<uint32_t>& cps
) {
    std::vector<std::string> tokens;
    std::vector<uint32_t> current;

    auto flush = [&]() {
        if (!current.empty()) {
            tokens.push_back(encode_codepoints(
                current, 0, current.size()
            ));
            current.clear();
        }
    };

    for (uint32_t cp : cps) {
        if (pwea::unicode::is_word(cp)) {
            current.push_back(cp);
        } else {
            flush();

            if (!pwea::unicode::is_space(cp)) {
                std::vector<uint32_t> one{cp};
                tokens.push_back(encode_codepoints(one, 0, 1));
            }
        }
    }

    flush();
    return tokens;
}

static std::vector<uint32_t> normalize_character_text(
    const std::vector<uint32_t>& cps
) {
    std::vector<uint32_t> result;
    bool pending_space = false;

    for (uint32_t cp : cps) {
        if (pwea::unicode::is_space(cp)) {
            if (!result.empty()) {
                pending_space = true;
            }
            continue;
        }

        if (pending_space) {
            result.push_back(0x20);
            pending_space = false;
        }

        result.push_back(cp);
    }

    return result;
}

static void normalize_block(std::vector<double>& block) {
    double squared_norm = 0.0;

    for (double value : block) {
        squared_norm += value * value;
    }

    if (squared_norm > 0.0) {
        const double inverse_norm = 1.0 / std::sqrt(squared_norm);

        for (double& value : block) {
            value *= inverse_norm;
        }
    }

    constexpr double SCALE = 0.70710678118654752440;

    for (double& value : block) {
        value *= SCALE;
    }
}

std::vector<double> transform_balanced_hybrid(
    const std::string& prompt
) {
    if (prompt.empty()) {
        throw std::invalid_argument("Prompt must not be empty");
    }

    const std::string lowered =
        pwea::unicode::lowercase(prompt);

    std::vector<uint32_t> cps;

    if (!pwea::unicode::decode_utf8(lowered, cps)) {
        throw std::invalid_argument("Invalid UTF-8 prompt");
    }

    std::vector<double> word_block(WORD_DIM, 0.0);
    std::vector<double> char_block(CHAR_DIM, 0.0);

    // Word unigrams and bigrams.
    const auto tokens = tokenize(cps);

    for (const auto& token : tokens) {
        add_feature(word_block, "word:" + token);
    }

    for (size_t i = 0; i + 1 < tokens.size(); ++i) {
        add_feature(
            word_block,
            "word:" + tokens[i] + " " + tokens[i + 1]
        );
    }

    // Character n-grams over normalized Unicode codepoints.
    const auto normalized = normalize_character_text(cps);

    for (size_t n = 3; n <= 5; ++n) {
        if (normalized.size() < n) continue;

        for (size_t i = 0; i + n <= normalized.size(); ++i) {
            const std::string feature =
                "char:" + encode_codepoints(normalized, i, i + n);

            add_feature(char_block, feature);
        }
    }

    normalize_block(word_block);
    normalize_block(char_block);

    std::vector<double> result;
    result.reserve(TOTAL_DIM);

    result.insert(
        result.end(),
        word_block.begin(),
        word_block.end()
    );

    result.insert(
        result.end(),
        char_block.begin(),
        char_block.end()
    );

    return result;
}

}


namespace pwea::crypto {

std::array<uint8_t, 32> sha256(
    const uint8_t* data,
    size_t size
) {
    if (size != 0 && data == nullptr) {
        throw std::invalid_argument("SHA-256 input is null");
    }

    const std::string input(
        reinterpret_cast<const char*>(data),
        size
    );

    return pwea::features::sha256(input);
}

}
