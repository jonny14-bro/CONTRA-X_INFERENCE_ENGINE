#include "compare_q6_fiveway_embedded.cpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <filesystem>

struct Metric {
    double max_abs = 0.0;
    long double error_sq = 0.0;
    long double ref_sq = 0.0;
    uint64_t count = 0;
    uint64_t nonfinite = 0;

    void add(double reference, double error) {
        if (!std::isfinite(reference) || !std::isfinite(error)) {
            ++nonfinite;
            return;
        }

        max_abs = std::max(max_abs, std::abs(error));
        error_sq += static_cast<long double>(error) * error;
        ref_sq += static_cast<long double>(reference) * reference;
        ++count;
    }

    void merge(const Metric& other) {
        max_abs = std::max(max_abs, other.max_abs);
        error_sq += other.error_sq;
        ref_sq += other.ref_sq;
        count += other.count;
        nonfinite += other.nonfinite;
    }

    double rmse() const {
        return count ? std::sqrt(static_cast<double>(error_sq / count)) : 0.0;
    }

    double ref_rms() const {
        return count ? std::sqrt(static_cast<double>(ref_sq / count)) : 0.0;
    }

    double nrmse() const {
        return ref_rms() > 0.0 ? 100.0 * rmse() / ref_rms() : 0.0;
    }
};

struct Bucket {
    Metric zeroing;
    Metric rounding;
    Metric total;

    double closure_max = 0.0;
    long double closure_sq = 0.0;
    uint64_t closure_count = 0;
    uint64_t closure_nonfinite = 0;

    uint64_t outputs = 0;

    void add(
        double reference,
        double zero_error,
        double rounding_error,
        double total_error,
        double closure
    ) {
        ++outputs;

        zeroing.add(reference, zero_error);
        rounding.add(reference, rounding_error);
        total.add(reference, total_error);

        if (std::isfinite(closure)) {
            closure_max = std::max(closure_max, std::abs(closure));
            closure_sq += static_cast<long double>(closure) * closure;
            ++closure_count;
        } else {
            ++closure_nonfinite;
        }
    }

    void merge(const Bucket& other) {
        zeroing.merge(other.zeroing);
        rounding.merge(other.rounding);
        total.merge(other.total);

        closure_max = std::max(closure_max, other.closure_max);
        closure_sq += other.closure_sq;
        closure_count += other.closure_count;
        closure_nonfinite += other.closure_nonfinite;

        outputs += other.outputs;
    }

    double closure_rmse() const {
        return closure_count
            ? std::sqrt(static_cast<double>(
                closure_sq / closure_count))
            : 0.0;
    }
};

struct RecordResult {
    int id;
    std::string operation;
    int tokens;
    int rows;
    Bucket stats;
};

static void write_metric_json(
    std::ostream& out,
    const Metric& m
) {
    out << "{"
        << "\"count\":" << m.count << ","
        << "\"nonfinite\":" << m.nonfinite << ","
        << "\"max_abs\":" << m.max_abs << ","
        << "\"rmse\":" << m.rmse() << ","
        << "\"reference_rms\":" << m.ref_rms() << ","
        << "\"nrmse_percent\":" << m.nrmse()
        << "}";
}

static void write_bucket_json(
    std::ostream& out,
    const Bucket& b
) {
    out << "{"
        << "\"outputs\":" << b.outputs << ","
        << "\"zeroing\":";

    write_metric_json(out, b.zeroing);

    out << ",\"rounding\":";
    write_metric_json(out, b.rounding);

    out << ",\"total\":";
    write_metric_json(out, b.total);

    out << ",\"closure\":{"
        << "\"max_abs\":" << b.closure_max << ","
        << "\"rmse\":" << b.closure_rmse() << ","
        << "\"count\":" << b.closure_count << ","
        << "\"nonfinite\":" << b.closure_nonfinite
        << "}}";
}

static void write_bucket_tsv(
    std::ostream& out,
    const Bucket& b
) {
    out << std::setprecision(17);
    out << b.outputs << '\t';

    const Metric* metrics[] = {
        &b.zeroing,
        &b.rounding,
        &b.total
    };

    for (const Metric* m : metrics) {
        out << m->max_abs << '\t'
            << m->rmse() << '\t'
            << m->ref_rms() << '\t'
            << m->nrmse() << '\t'
            << m->nonfinite << '\t';
    }

    out << b.closure_max << '\t'
        << b.closure_rmse() << '\t'
        << b.closure_nonfinite;
}

int main(int argc, char** argv) {
    try {
        if (argc != 5) {
            throw std::runtime_error(
                "Usage: executable weights activations descriptors threads"
            );
        }

        const int threads = std::atoi(argv[4]);

        if (threads < 1)
            throw std::runtime_error("Invalid thread count");

        Mapping weights(argv[1]);
        Mapping activations(argv[2]);

        const auto records = load_records(argv[3]);

        init_types();

        omp_set_dynamic(0);
        omp_set_num_threads(threads);

        if (!q6g16_simd::available())
            throw std::runtime_error("Q8G16 SIMD unavailable");

        std::printf("Q8G16 DATASET-WIDE WEIGHTED ERROR ATTRIBUTION\n");
        std::printf("Records: %zu\n", records.size());
        std::printf("Threads: %d\n", threads);
        std::printf("Q8G16 SIMD: available\n");
        std::printf("Weight source: captured Q6 weights\n");
        std::printf("Activation source: captured FP32 activations\n");

        const std::string output_dir =
            "benchmarks/real_threeway_validation_20261120/"
            "weighted_error_dataset_20261120";

        std::filesystem::create_directories(output_dir);

        std::set<int> record_ids;
        std::map<std::string, Bucket> operations;
        std::vector<RecordResult> record_results;

        Bucket global;

        uint64_t total_tokens = 0;
        uint64_t total_output_rows = 0;

        for (size_t ri = 0; ri < records.size(); ++ri) {
            const Record& r = records[ri];

            if (!record_ids.insert(r.id).second)
                throw std::runtime_error("Duplicate descriptor ID");

            if (r.type != 14 || r.k <= 0 || r.k % 256 ||
                r.nb <= 0 || r.rows <= 0)
                throw std::runtime_error("Invalid Q6 descriptor");

            const TypeInfo& ti = g_types[r.type];
            const int64_t nblk = r.k / 256;

            const uint64_t expected_row_bytes =
                static_cast<uint64_t>(nblk) * ti.bytes;

            const uint64_t weight_bytes =
                static_cast<uint64_t>(r.rows) * r.row_bytes;

            const uint64_t activation_bytes =
                static_cast<uint64_t>(r.nb) * r.k * sizeof(float);

            if (r.row_bytes != expected_row_bytes)
                throw std::runtime_error("Row-byte mismatch");

            if (r.weight_offset > weights.size ||
                weight_bytes > weights.size - r.weight_offset)
                throw std::runtime_error("Weight bounds failure");

            if (r.activation_offset > activations.size ||
                activation_bytes > activations.size - r.activation_offset)
                throw std::runtime_error("Activation bounds failure");

            const uint8_t* wdata =
                weights.data + r.weight_offset;

            const float* x =
                reinterpret_cast<const float*>(
                    activations.data + r.activation_offset
                );

            const size_t quantized_count =
                static_cast<size_t>(r.nb) * nblk;

            std::vector<q6g16::Q8G16> q8g(quantized_count);

            // Quantize each activation token exactly once.
            for (int token = 0; token < r.nb; ++token) {
                for (int64_t block = 0; block < nblk; ++block) {
                    const float* xb =
                        x + static_cast<size_t>(token) * r.k
                          + block * 256;

                    q6g16_simd::quantize(
                        xb,
                        q8g[static_cast<size_t>(token) * nblk + block]
                    );
                }
            }

            const int64_t output_count =
                static_cast<int64_t>(r.nb) * r.rows;

            std::vector<Bucket> thread_buckets(threads);

            // Each pair (token, output row) is calculated once.
            #pragma omp parallel
            {
                std::array<float, 256> decoded_weights{};
                std::array<float, 256> reconstructed{};
                std::array<float, 256> zero_loss{};
                std::array<float, 256> rounding_loss{};

                const int tid = omp_get_thread_num();
                Bucket& local = thread_buckets[tid];

                #pragma omp for schedule(dynamic, 32)
                for (int64_t oi = 0; oi < output_count; ++oi) {
                    const int token =
                        static_cast<int>(oi / r.rows);

                    const int row =
                        static_cast<int>(oi % r.rows);

                    const float* token_x =
                        x + static_cast<size_t>(token) * r.k;

                    const uint8_t* wr =
                        wdata + static_cast<size_t>(row) * r.row_bytes;

                    double reference_output = 0.0;
                    double quantized_output = 0.0;
                    double zero_error = 0.0;
                    double rounding_error = 0.0;

                    for (int64_t block = 0; block < nblk; ++block) {
                        const uint8_t* wb =
                            wr + block * ti.bytes;

                        ti.dq(wb, decoded_weights.data());

                        const float* original =
                            token_x + block * 256;

                        const auto& q =
                            q8g[static_cast<size_t>(token) * nblk + block];

                        for (int i = 0; i < 256; ++i) {
                            const int code = q.qs[i];

                            const float reconstructed_value =
                                q.d[i / 16] * static_cast<float>(code);

                            reconstructed[i] = reconstructed_value;

                            if (code == 0) {
                                zero_loss[i] = original[i];
                                rounding_loss[i] = 0.0f;
                            } else {
                                zero_loss[i] = 0.0f;
                                rounding_loss[i] =
                                    original[i] - reconstructed_value;
                            }
                        }

                        reference_output += dotf(
                            decoded_weights.data(),
                            original,
                            256
                        );

                        quantized_output += dotf(
                            decoded_weights.data(),
                            reconstructed.data(),
                            256
                        );

                        zero_error += dotf(
                            decoded_weights.data(),
                            zero_loss.data(),
                            256
                        );

                        rounding_error += dotf(
                            decoded_weights.data(),
                            rounding_loss.data(),
                            256
                        );
                    }

                    const double total_error =
                        reference_output - quantized_output;

                    const double closure =
                        total_error - zero_error - rounding_error;

                    local.add(
                        reference_output,
                        zero_error,
                        rounding_error,
                        total_error,
                        closure
                    );
                }
            }

            Bucket record_bucket;

            for (const auto& b : thread_buckets)
                record_bucket.merge(b);

            operations[r.operation].merge(record_bucket);
            global.merge(record_bucket);

            total_tokens += r.nb;
            total_output_rows += r.rows;

            record_results.push_back({
                r.id,
                r.operation,
                r.nb,
                r.rows,
                record_bucket
            });

            if ((ri + 1) % 250 == 0 || ri + 1 == records.size()) {
                std::printf(
                    "Progress: %zu/%zu records | outputs=%llu\n",
                    ri + 1,
                    records.size(),
                    static_cast<unsigned long long>(global.outputs)
                );
                std::fflush(stdout);
            }
        }

        // Coverage checks.
        uint64_t expected_outputs = 0;

        for (const auto& r : records) {
            expected_outputs +=
                static_cast<uint64_t>(r.nb) * r.rows;
        }

        if (global.outputs != expected_outputs)
            throw std::runtime_error("Output coverage mismatch");

        if (global.total.nonfinite != 0 ||
            global.zeroing.nonfinite != 0 ||
            global.rounding.nonfinite != 0 ||
            global.closure_nonfinite != 0)
            throw std::runtime_error("Nonfinite output detected");

        // Operation-level TSV.
        const std::string operation_path =
            output_dir + "/operation_summary.tsv";

        std::ofstream operation_file(operation_path);
        operation_file << std::setprecision(17);

        operation_file
            << "operation\trecords\toutputs\t"
            << "zero_max_abs\tzero_rmse\tzero_ref_rms\tzero_nrmse_pct\tzero_nonfinite\t"
            << "round_max_abs\tround_rmse\tround_ref_rms\tround_nrmse_pct\tround_nonfinite\t"
            << "total_max_abs\ttotal_rmse\ttotal_ref_rms\ttotal_nrmse_pct\ttotal_nonfinite\t"
            << "closure_max_abs\tclosure_rmse\tclosure_nonfinite\n";

        for (const auto& [name, bucket] : operations) {
            size_t count = 0;

            for (const auto& record : record_results) {
                if (record.operation == name)
                    ++count;
            }

            operation_file << name << '\t' << count << '\t';
            write_bucket_tsv(operation_file, bucket);
            operation_file << '\n';
        }

        operation_file.close();

        // Per-record TSV.
        const std::string record_path =
            output_dir + "/record_summary.tsv";

        std::ofstream record_file(record_path);
        record_file << std::setprecision(17);

        record_file
            << "record_id\toperation\ttokens\trows\t"
            << "outputs\t"
            << "zero_max_abs\tzero_rmse\tzero_ref_rms\tzero_nrmse_pct\tzero_nonfinite\t"
            << "round_max_abs\tround_rmse\tround_ref_rms\tround_nrmse_pct\tround_nonfinite\t"
            << "total_max_abs\ttotal_rmse\ttotal_ref_rms\ttotal_nrmse_pct\ttotal_nonfinite\t"
            << "closure_max_abs\tclosure_rmse\tclosure_nonfinite\n";

        for (const auto& record : record_results) {
            record_file
                << record.id << '\t'
                << record.operation << '\t'
                << record.tokens << '\t'
                << record.rows << '\t';

            write_bucket_tsv(record_file, record.stats);
            record_file << '\n';
        }

        record_file.close();

        // JSON report.
        const std::string json_path =
            output_dir + "/summary.json";

        std::ofstream json(json_path);
        json << std::setprecision(17);

        json << "{\n"
             << "\"records\":" << records.size() << ",\n"
             << "\"activation_tokens\":" << total_tokens << ",\n"
             << "\"output_rows_sum\":" << total_output_rows << ",\n"
             << "\"output_pairs\":" << global.outputs << ",\n"
             << "\"operations_count\":" << operations.size() << ",\n"
             << "\"global\":";

        write_bucket_json(json, global);

        json << ",\n\"operations\":[";

        bool first = true;

        for (const auto& [name, bucket] : operations) {
            if (!first)
                json << ',';

            first = false;

            json << "\n{\"operation\":\"" << name << "\",\"records\":";

            size_t count = 0;

            for (const auto& record : record_results) {
                if (record.operation == name)
                    ++count;
            }

            json << count << ",\"metrics\":";
            write_bucket_json(json, bucket);
            json << "}";
        }

        json << "\n]}\n";
        json.close();

        std::printf("\n============================================\n");
        std::printf("DATASET-WIDE ATTRIBUTION COMPLETE\n");
        std::printf("Records: %zu\n", records.size());
        std::printf("Output pairs: %llu\n",
            static_cast<unsigned long long>(global.outputs));

        std::printf("\nGLOBAL ZEROING\n");
        std::printf("RMSE: %.10g | NRMSE: %.6f%%\n",
            global.zeroing.rmse(), global.zeroing.nrmse());

        std::printf("\nGLOBAL ROUNDING\n");
        std::printf("RMSE: %.10g | NRMSE: %.6f%%\n",
            global.rounding.rmse(), global.rounding.nrmse());

        std::printf("\nGLOBAL TOTAL ERROR\n");
        std::printf("RMSE: %.10g | NRMSE: %.6f%%\n",
            global.total.rmse(), global.total.nrmse());

        std::printf("\nDECOMPOSITION CLOSURE\n");
        std::printf("Max: %.12g | RMSE: %.12g\n",
            global.closure_max, global.closure_rmse());

        std::printf("\nOperation summary:\n");

        for (const auto& [name, bucket] : operations) {
            std::printf(
                "%-20s outputs=%llu zero=%.5f%% "
                "round=%.5f%% total=%.5f%%\n",
                name.c_str(),
                static_cast<unsigned long long>(bucket.outputs),
                bucket.zeroing.nrmse(),
                bucket.rounding.nrmse(),
                bucket.total.nrmse()
            );
        }

        std::printf("\nArtifacts:\n%s\n", output_dir.c_str());
        std::printf("  operation_summary.tsv\n");
        std::printf("  record_summary.tsv\n");
        std::printf("  summary.json\n");

        return 0;
    }
    catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
