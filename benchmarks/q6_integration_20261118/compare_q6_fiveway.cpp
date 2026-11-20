#define main qwen3moe_embedded_main
#include "qwen3moe_q6.cpp"
#undef main
#include "q6_groupwise_q8.h"
#include "q6_groupwise_q8_simd.h"
#include "q6_groupwise_q8_v2.h"

#include <array>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <iomanip>

struct Mapping {
    int fd = -1;
    size_t size = 0;
    const uint8_t* data = nullptr;

    explicit Mapping(const char* path) {
        fd = open(path, O_RDONLY);
        if (fd < 0) throw std::runtime_error(path);

        struct stat st {};
        if (fstat(fd, &st) || st.st_size <= 0)
            throw std::runtime_error("Invalid file");

        size = static_cast<size_t>(st.st_size);
        void* p = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (p == MAP_FAILED) throw std::runtime_error("mmap failed");
        data = static_cast<const uint8_t*>(p);
    }

    ~Mapping() {
        if (data) munmap(const_cast<uint8_t*>(data), size);
        if (fd >= 0) close(fd);
    }

    Mapping(const Mapping&) = delete;
    Mapping& operator=(const Mapping&) = delete;
};

struct Record {
    int id, type, k, nb, rows, row_bytes;
    std::string operation;
    uint64_t activation_offset, weight_offset;
};

static std::vector<Record> load_records(const char* path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open descriptors");

    std::vector<Record> records;
    std::string line;

    while (std::getline(in, line)) {
        if (line.empty()) continue;

        std::istringstream ss(line);
        Record r{};

        if (!(ss >> r.id >> r.operation >> r.type >> r.k >> r.nb
                 >> r.rows >> r.row_bytes
                 >> r.activation_offset >> r.weight_offset))
            throw std::runtime_error("Malformed descriptor");

        if (r.type != 14 || r.k <= 0 || r.k % 256 ||
            r.nb <= 0 || r.rows <= 0)
            throw std::runtime_error("Invalid Q6 descriptor");

        records.push_back(r);
    }

    return records;
}

struct Metrics {
    double max_abs = 0;
    double sum_sq = 0;
    double ref_sq = 0;
    double dot = 0;
    double other_sq = 0;
    size_t count = 0;
    size_t nonfinite = 0;

    void add(float ref, float other) {
        if (!std::isfinite(ref) || !std::isfinite(other)) {
            ++nonfinite;
            return;
        }

        const double d = static_cast<double>(ref) - other;
        max_abs = std::max(max_abs, std::fabs(d));
        sum_sq += d * d;
        ref_sq += static_cast<double>(ref) * ref;
        other_sq += static_cast<double>(other) * other;
        dot += static_cast<double>(ref) * other;
        ++count;
    }

    void print(const char* label) const {
        const double rmse = count ? std::sqrt(sum_sq / count) : 0;
        const double ref_rms = count ? std::sqrt(ref_sq / count) : 0;
        const double cosine =
            ref_sq > 0 && other_sq > 0
            ? dot / std::sqrt(ref_sq * other_sq) : 0;

        std::printf(
            "%-22s count=%zu max_abs=%.9g RMSE=%.9g "
            "ref_RMS=%.9g NRMSE=%.6f%% cosine=%.9f nonfinite=%zu\n",
            label, count, max_abs, rmse, ref_rms,
            ref_rms > 0 ? 100.0 * rmse / ref_rms : 0,
            cosine, nonfinite);
    }
};

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr,
            "Usage: %s weights.bin activations.f32 descriptors.tsv threads\n",
            argv[0]);
        return 2;
    }

    try {
        const int threads = std::atoi(argv[4]);
        if (threads < 1) throw std::runtime_error("Invalid threads");

        Mapping weights(argv[1]);
        Mapping activations(argv[2]);
        const auto records = load_records(argv[3]);

        init_types();
        omp_set_dynamic(0);
        omp_set_num_threads(threads);

        std::printf("AVX-VNNI available: %s\n",
                    q6vnni::available() ? "yes" : "no");
        std::printf("Threads: %d\nRecords: %zu\n",
                    threads, records.size());

        if (!q6vnni::available())
            throw std::runtime_error("AVX-VNNI unavailable");

        for (const auto& r : records) {
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

            const uint8_t* wdata = weights.data + r.weight_offset;
            const float* x = reinterpret_cast<const float*>(
                activations.data + r.activation_offset);

            std::printf(
                "\n========== RECORD %d: %s ==========\n"
                "K=%d tokens=%d rows=%d blocks=%lld\n",
                r.id, r.operation.c_str(), r.k, r.nb, r.rows,
                static_cast<long long>(nblk));

            // Quantize activations once, exactly as the production path.
            std::vector<q6vnni::Q8K> q8(
                static_cast<size_t>(r.nb) * nblk);

            std::vector<float> reconstructed(
                static_cast<size_t>(r.nb) * r.k);

            std::vector<q6g16::Q8G16> q8g(
                static_cast<size_t>(r.nb) * nblk);

            std::vector<float> reconstructed_g16(
                static_cast<size_t>(r.nb) * r.k);

            for (int t = 0; t < r.nb; ++t) {
                for (int64_t b = 0; b < nblk; ++b) {
                    const float* xb =
                        x + static_cast<size_t>(t) * r.k + b * 256;

                    auto& qb = q8[static_cast<size_t>(t) * nblk + b];
                    q6vnni::quantize_q8k(xb, qb);

                    float* dst =
                        reconstructed.data() +
                        static_cast<size_t>(t) * r.k + b * 256;

                    for (int i = 0; i < 256; ++i)
                        dst[i] = qb.d * static_cast<float>(qb.qs[i]);

                    auto& qg =
                        q8g[static_cast<size_t>(t) * nblk + b];

                    q6g16_simd::quantize(xb, qg);

                    float* dst_g =
                        reconstructed_g16.data() +
                        static_cast<size_t>(t) * r.k + b * 256;

                    for (int i = 0; i < 256; ++i) {
                        const int group = i / 16;
                        dst_g[i] = qg.d[group] *
                                   static_cast<float>(qg.qs[i]);
                    }
                }
            }

            const size_t output_count =
                static_cast<size_t>(r.nb) * r.rows;

            std::vector<float> A(output_count, 0.0f);
            std::vector<float> B(output_count, 0.0f);
            std::vector<float> C(output_count, 0.0f);
            std::vector<float> D(output_count, 0.0f);
            std::vector<float> E(output_count, 0.0f);

            // Same Q6 decoding and dotf as the original FP32 fallback.
            // B uses reconstructed Q8 activations instead of original FP32.
            // C uses the production AVX-VNNI dot product.
            #pragma omp parallel
            {
                std::array<float, 256> tmp{};

                #pragma omp for schedule(dynamic, 8)
                for (int row = 0; row < r.rows; ++row) {
                    const uint8_t* wr =
                        wdata + static_cast<size_t>(row) * r.row_bytes;

                    for (int64_t b = 0; b < nblk; ++b) {
                        const uint8_t* wb = wr + b * ti.bytes;

                        ti.dq(wb, tmp.data());

                        q6vnni::PredecodedQ6 decoded{};
                        q6vnni::decode_q6(
                            wb, rd_f16(wb + 208), decoded);

                        for (int t = 0; t < r.nb; ++t) {
                            const size_t out_index =
                                static_cast<size_t>(t) * r.rows + row;

                            const float* original =
                                x + static_cast<size_t>(t) * r.k + b * 256;

                            const float* quantized =
                                reconstructed.data() +
                                static_cast<size_t>(t) * r.k + b * 256;

                            const float* quantized_g =
                                reconstructed_g16.data() +
                                static_cast<size_t>(t) * r.k + b * 256;

                            const auto& qb =
                                q8[static_cast<size_t>(t) * nblk + b];

                            const auto& qg =
                                q8g[static_cast<size_t>(t) * nblk + b];

                            A[out_index] += dotf(
                                tmp.data(), original, 256);

                            B[out_index] += dotf(
                                tmp.data(), quantized, 256);

                            C[out_index] += dotf(
                                tmp.data(), quantized_g, 256);

                            D[out_index] +=
                                q6g16::dot_q6_g16(decoded, qg);

                            E[out_index] +=
                                q6g16_v2::dot_q6_g16_v2(decoded, qg);
                        }
                    }
                }
            }

            Metrics baseline_activation_error;
            Metrics groupwise_activation_error;
            Metrics groupwise_integer_error;
            Metrics groupwise_total_error;
            Metrics groupwise_v2_integer_error;
            Metrics groupwise_v2_total_error;

            for (size_t i = 0; i < output_count; ++i) {
                baseline_activation_error.add(A[i], B[i]);
                groupwise_activation_error.add(A[i], C[i]);
                groupwise_integer_error.add(C[i], D[i]);
                groupwise_total_error.add(A[i], D[i]);
                groupwise_v2_integer_error.add(C[i], E[i]);
                groupwise_v2_total_error.add(A[i], E[i]);
            }

            baseline_activation_error.print("A vs B baseline Q8");
            groupwise_activation_error.print("A vs C group-wise Q8");
            groupwise_integer_error.print("C vs D V1 VNNI");
            groupwise_total_error.print("A vs D V1 total");
            groupwise_v2_integer_error.print("C vs E V2 VNNI");
            groupwise_v2_total_error.print("A vs E V2 total");

            std::printf("Record %d complete.\n", r.id);
        }

        return 0;
    }
    catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
