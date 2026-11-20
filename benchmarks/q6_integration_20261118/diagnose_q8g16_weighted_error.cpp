#include "compare_q6_fiveway_embedded.cpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

struct Target {
    int id;
    int output_rows[2];
};

struct OutputRow {
    int token;
    double reference;
    double quantized;
    double zero_error;
    double rounding_error;
    double total_error;
};

struct ComponentStats {
    double max_abs = 0.0;
    double error_sq = 0.0;
    double reference_sq = 0.0;
    size_t count = 0;
    size_t nonfinite = 0;

    void add(double reference, double error) {
        if (!std::isfinite(reference) || !std::isfinite(error)) {
            ++nonfinite;
            return;
        }

        max_abs = std::max(max_abs, std::abs(error));
        error_sq += error * error;
        reference_sq += reference * reference;
        ++count;
    }

    void print(const char* label) const {
        const double rmse =
            count ? std::sqrt(error_sq / count) : 0.0;

        const double ref_rms =
            count ? std::sqrt(reference_sq / count) : 0.0;

        const double nrmse =
            ref_rms > 0.0 ? 100.0 * rmse / ref_rms : 0.0;

        std::printf(
            "%-24s count=%zu max_abs=%.9g "
            "RMSE=%.9g NRMSE=%.6f%% nonfinite=%zu\n",
            label, count, max_abs, rmse, nrmse, nonfinite
        );
    }
};

int main() {
    try {
        const char* weights_path =
            "benchmarks/real_activation_weight_replay_20261118/weights.bin";

        const char* activations_path =
            "benchmarks/real_activations_q8k_full_20261118/activations.f32";

        const char* descriptors_path =
            "benchmarks/real_threeway_validation_20261120/q6_only_descriptors.tsv";

        const Target targets[] = {
            {8297, {304, 434}},
            {8494, {0, 745}},
            {8623, {834, 86}},
            {8702, {1, 1670}},
            {8810, {1, 967}}
        };

        Mapping weights(weights_path);
        Mapping activations(activations_path);

        const auto records = load_records(descriptors_path);

        init_types();

        if (!q6g16_simd::available()) {
            throw std::runtime_error("Q8G16 SIMD unavailable");
        }

        std::printf("Q8G16 WEIGHTED OUTPUT ERROR ATTRIBUTION\n");
        std::printf("Records loaded: %zu\n", records.size());
        std::printf("Selected output rows: 10\n");
        std::printf("Weight source: actual captured Q6 weights\n");
        std::printf("Activation source: captured FP32 activations\n");

        for (const auto& target : targets) {
            auto it = std::find_if(
                records.begin(), records.end(),
                [&](const Record& r) {
                    return r.id == target.id;
                }
            );

            if (it == records.end()) {
                throw std::runtime_error(
                    "Missing target record " + std::to_string(target.id)
                );
            }

            const Record& r = *it;
            const TypeInfo& ti = g_types[r.type];

            const int64_t nblk = r.k / 256;

            const uint64_t weight_bytes =
                static_cast<uint64_t>(r.rows) * r.row_bytes;

            const uint64_t activation_bytes =
                static_cast<uint64_t>(r.nb) * r.k * sizeof(float);

            if (r.weight_offset > weights.size ||
                weight_bytes > weights.size - r.weight_offset) {
                throw std::runtime_error("Weight bounds failure");
            }

            if (r.activation_offset > activations.size ||
                activation_bytes > activations.size - r.activation_offset) {
                throw std::runtime_error("Activation bounds failure");
            }

            const uint8_t* wdata =
                weights.data + r.weight_offset;

            const float* x =
                reinterpret_cast<const float*>(
                    activations.data + r.activation_offset
                );

            std::printf(
                "\n==================================================\n"
                "RECORD %d | %s | K=%d | output_rows=%d\n",
                r.id, r.operation.c_str(), r.k, r.rows
            );


            for (int selected = 0; selected < 2; ++selected) {
                const int output_row = target.output_rows[selected];

                if (output_row < 0 || output_row >= r.rows) {
                    throw std::runtime_error("Invalid selected output row");
                }

                const uint8_t* wr =
                    wdata + static_cast<size_t>(output_row) * r.row_bytes;

                ComponentStats zero_stats;
                ComponentStats rounding_stats;
                ComponentStats total_stats;

                std::vector<OutputRow> token_results;
                token_results.reserve(r.nb);

                double closure_max = 0.0;
                double closure_sq = 0.0;

                for (int token = 0; token < r.nb; ++token) {
                    const float* token_x =
                        x + static_cast<size_t>(token) * r.k;

                    std::vector<q6g16::Q8G16> quantized(nblk);

                    for (int64_t b = 0; b < nblk; ++b) {
                        q6g16_simd::quantize(
                            token_x + b * 256,
                            quantized[b]
                        );
                    }

                    double reference_output = 0.0;
                    double quantized_output = 0.0;
                    double zero_error = 0.0;
                    double rounding_error = 0.0;

                    std::array<float, 256> decoded_weights{};
                    std::array<float, 256> reconstructed{};
                    std::array<float, 256> zero_loss{};
                    std::array<float, 256> rounding_loss{};

                    for (int64_t b = 0; b < nblk; ++b) {
                        const uint8_t* wb = wr + b * ti.bytes;

                        ti.dq(wb, decoded_weights.data());

                        const auto& q = quantized[b];
                        const float* original = token_x + b * 256;

                        for (int i = 0; i < 256; ++i) {
                            const int code = q.qs[i];

                            const float rec =
                                q.d[i / 16] * static_cast<float>(code);

                            reconstructed[i] = rec;

                            if (code == 0) {
                                zero_loss[i] = original[i];
                                rounding_loss[i] = 0.0f;
                            } else {
                                zero_loss[i] = 0.0f;
                                rounding_loss[i] = original[i] - rec;
                            }
                        }

                        reference_output += dotf(
                            decoded_weights.data(), original, 256
                        );

                        quantized_output += dotf(
                            decoded_weights.data(), reconstructed.data(), 256
                        );

                        zero_error += dotf(
                            decoded_weights.data(), zero_loss.data(), 256
                        );

                        rounding_error += dotf(
                            decoded_weights.data(), rounding_loss.data(), 256
                        );
                    }

                    const double total_error =
                        reference_output - quantized_output;

                    const double reconstructed_error =
                        zero_error + rounding_error;

                    const double closure =
                        total_error - reconstructed_error;

                    closure_max = std::max(
                        closure_max, std::abs(closure)
                    );

                    closure_sq += closure * closure;

                    zero_stats.add(reference_output, zero_error);
                    rounding_stats.add(reference_output, rounding_error);
                    total_stats.add(reference_output, total_error);

                    token_results.push_back({
                        token,
                        reference_output,
                        quantized_output,
                        zero_error,
                        rounding_error,
                        total_error
                    });
                }

                std::printf(
                    "\n-----------------------------------------------\n"
                    "OUTPUT ROW=%d | K=%d | ACTIVATION TOKENS=%d\n",
                    output_row, r.k, r.nb
                );

                std::printf(
                    "\n--- ERROR COMPONENTS (normalized to FP32 output RMS) ---\n"
                );

                zero_stats.print("Zeroing contribution");
                rounding_stats.print("Rounding contribution");
                total_stats.print("Total Q8G16 error");

                const double closure_rmse =
                    token_results.empty()
                    ? 0.0
                    : std::sqrt(
                        closure_sq / token_results.size()
                    );

                std::printf(
                    "Decomposition closure: max_abs=%.12g RMSE=%.12g\n",
                    closure_max, closure_rmse
                );

                std::sort(
                    token_results.begin(),
                    token_results.end(),
                    [](const OutputRow& a, const OutputRow& b) {
                        return std::abs(a.total_error) >
                               std::abs(b.total_error);
                    }
                );

                std::printf(
                    "\n--- TOKENS RANKED BY ABSOLUTE OUTPUT ERROR ---\n"
                );

                std::printf(
                    "TOKEN     FP32_REF       Q8G16_OUT"
                    "      ZERO_ERROR    ROUND_ERROR   TOTAL_ERROR\n"
                );

                const size_t limit =
                    std::min<size_t>(10, token_results.size());

                for (size_t i = 0; i < limit; ++i) {
                    const auto& o = token_results[i];

                    std::printf(
                        "%-8d %13.6g %13.6g "
                        "%13.6g %13.6g %13.6g\n",
                        o.token,
                        o.reference,
                        o.quantized,
                        o.zero_error,
                        o.rounding_error,
                        o.total_error
                    );
                }

                std::printf(
                    "\nOutput row %d attribution complete.\n",
                    output_row
                );
            }
        }

        std::printf(
            "\nALL SELECTED WEIGHTED OUTPUT DIAGNOSTICS COMPLETE.\n"
        );

        return 0;
    }
    catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
