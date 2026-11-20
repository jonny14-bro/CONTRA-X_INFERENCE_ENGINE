#include "q6_groupwise_q8_simd.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct Record {
    int id;
    int k;
    int rows;
    long long activation_offset;
    std::string operation;
};

struct Selection {
    int row;
    int block;
};

int main() {
    const std::string descriptor =
        "benchmarks/real_activation_weight_replay_20261118/"
        "replay_descriptors_v1.tsv";

    const std::string activation_file =
        "benchmarks/real_activations_q8k_full_20261118/"
        "activations.f32";

    const std::vector<int> wanted = {
        8297, 8494, 8623, 8702, 8810
    };

    std::ifstream desc(descriptor);
    std::ifstream activation(activation_file, std::ios::binary);

    if (!desc || !activation) {
        std::cerr << "ERROR: Cannot open input files\n";
        return 1;
    }

    std::vector<Record> records;
    std::string line;

    // Descriptor file has no header.
    while (std::getline(desc, line)) {
        std::stringstream ss(line);
        std::string field;
        std::vector<std::string> fields;

        while (std::getline(ss, field, '\t')) {
            fields.push_back(field);
        }

        if (fields.size() != 9)
            continue;

        int id = std::stoi(fields[0]);

        if (std::find(wanted.begin(), wanted.end(), id)
            == wanted.end()) {
            continue;
        }

        records.push_back({
            id,
            std::stoi(fields[3]),
            std::stoi(fields[5]),
            std::stoll(fields[7]),
            fields[1]
        });
    }

    struct Target {
        int id;
        Selection selections[2];
    };

    const Target targets[] = {
        {8297, {{304, 0}, {434, 5}}},
        {8494, {{0, 0}, {745, 2}}},
        {8623, {{834, 1}, {86, 1}}},
        {8702, {{1, 1}, {1670, 1}}},
        {8810, {{1, 1}, {967, 1}}}
    };

    if (!q6g16_simd::available()) {
        std::cerr << "ERROR: AVX2 unavailable\n";
        return 2;
    }

    for (const auto& target : targets) {
        auto it = std::find_if(
            records.begin(),
            records.end(),
            [&](const Record& r) {
                return r.id == target.id;
            }
        );

        if (it == records.end()) {
            std::cerr << "ERROR: Missing record "
                      << target.id << "\n";
            return 3;
        }

        const Record& r = *it;

        std::cout << "\n========================================\n";
        std::cout << "RECORD " << r.id
                  << " | " << r.operation << "\n";

        for (const auto& selection : target.selections) {
            const int row = selection.row;
            const int block = selection.block;

            if (row < 0 || row >= r.rows ||
                block < 0 || block >= r.k / 256) {
                std::cerr << "ERROR: Invalid selection\n";
                return 4;
            }

            float x[256];

            // Descriptor activation_offset is in bytes.
            const long long byte_offset =
                r.activation_offset +
                (static_cast<long long>(row) * r.k +
                 static_cast<long long>(block) * 256) * 4;

            activation.clear();
            activation.seekg(byte_offset);

            if (!activation.read(
                    reinterpret_cast<char*>(x),
                    sizeof(x))) {
                std::cerr << "ERROR: Activation read failed\n";
                return 5;
            }

            q6g16::Q8G16 q;

            // Actual production SIMD quantizer.
            q6g16_simd::quantize(x, q);

            std::cout << "\nROW=" << row
                      << " BLOCK=" << block << "\n";

            std::cout
                << "GROUP  SCALE       ZERO  SATURATED"
                << "  MAX_ERROR     RMSE          NRMSE\n";

            for (int g = 0; g < 16; ++g) {
                double error_sq = 0.0;
                double ref_sq = 0.0;
                double max_error = 0.0;

                int zeros = 0;
                int saturated = 0;

                for (int j = 0; j < 16; ++j) {
                    const int index = g * 16 + j;

                    const float original = x[index];
                    const int8_t code = q.qs[index];

                    const float reconstructed =
                        q.d[g] * static_cast<float>(code);

                    const double error =
                        static_cast<double>(original) -
                        reconstructed;

                    error_sq += error * error;
                    ref_sq +=
                        static_cast<double>(original) * original;

                    max_error = std::max(
                        max_error, std::abs(error));

                    zeros += (code == 0);

                    saturated += (
                        std::abs(static_cast<int>(code)) == 127
                    );
                }

                const double rmse =
                    std::sqrt(error_sq / 16.0);

                const double ref_rms =
                    std::sqrt(ref_sq / 16.0);

                const double nrmse =
                    ref_rms > 0.0
                    ? 100.0 * rmse / ref_rms
                    : 0.0;

                std::printf(
                    "%02d     %-11.6g %3d   %3d"
                    "       %-11.6g %-12.6g %.6f%%\n",
                    g,
                    q.d[g],
                    zeros,
                    saturated,
                    max_error,
                    rmse,
                    nrmse
                );
            }

            // Find the subgroup with the largest top-1
            // energy concentration.
            int worst = 0;
            double worst_ratio = -1.0;

            for (int g = 0; g < 16; ++g) {
                double energy = 0.0;
                double maximum = 0.0;

                for (int j = 0; j < 16; ++j) {
                    const double v = x[g * 16 + j];

                    energy += v * v;
                    maximum = std::max(maximum, std::abs(v));
                }

                const double ratio =
                    energy > 0.0
                    ? maximum * maximum / energy
                    : 0.0;

                if (ratio > worst_ratio) {
                    worst_ratio = ratio;
                    worst = g;
                }
            }

            std::cout << "\nWorst concentration subgroup: "
                      << worst << "\n";

            std::cout
                << "INDEX  ORIGINAL       Q8_VALUE  RECONSTRUCTED"
                << "    ERROR\n";

            for (int j = 0; j < 16; ++j) {
                const int index = worst * 16 + j;

                const float original = x[index];
                const int code = q.qs[index];

                const float reconstructed =
                    q.d[worst] * static_cast<float>(code);

                std::printf(
                    "%02d     %13.7g  %5d     %13.7g"
                    "  %13.7g\n",
                    j,
                    original,
                    code,
                    reconstructed,
                    original - reconstructed
                );
            }
        }
    }

    std::cout
        << "\nExact SIMD quantization inspection complete.\n";

    return 0;
}
