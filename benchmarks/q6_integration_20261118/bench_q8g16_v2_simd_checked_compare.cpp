#define main qwen3moe_embedded_main
#include "qwen3moe_q6.cpp"
#undef main
#include "q6_groupwise_q8.h"
#include "q6_groupwise_q8_v2.h"
#include "q6_groupwise_q8_simd.h"
#include "q6_groupwise_q8_checked.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

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
    int id = 0, type = 0, k = 0, nb = 0, rows = 0, row_bytes = 0;
    std::string operation;
    uint64_t activation_offset = 0, weight_offset = 0;
};

static std::vector<Record> load_records(const char* path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open descriptors");

    std::vector<Record> records;
    std::string line;

    while (std::getline(in, line)) {
        if (line.empty()) continue;

        std::istringstream ss(line);
        Record r;

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

template<class F>
static double median_us(F&& fn) {
    constexpr int WARMUPS = 2;
    constexpr int TRIALS = 7;

    for (int i = 0; i < WARMUPS; ++i) fn();

    std::vector<double> samples;
    samples.reserve(TRIALS);

    for (int i = 0; i < TRIALS; ++i) {
        const auto begin = std::chrono::steady_clock::now();
        fn();
        const auto end = std::chrono::steady_clock::now();

        samples.push_back(
            std::chrono::duration<double, std::micro>(end - begin).count());
    }

    std::sort(samples.begin(), samples.end());
    return samples[TRIALS / 2];
}

static void quantize_q8k(
    const float* x, int nb, int64_t nblk,
    std::vector<q6vnni::Q8K>& q8, int k) {

    for (int t = 0; t < nb; ++t) {
        for (int64_t b = 0; b < nblk; ++b) {
            const size_t index = static_cast<size_t>(t) * nblk + b;
            q6vnni::quantize_q8k(
                x + static_cast<size_t>(t) * k + b * 256, q8[index]);
        }
    }
}

static void quantize_g16(
    const float* x, int nb, int64_t nblk,
    std::vector<q6g16::Q8G16>& q8g, int k) {

    for (int t = 0; t < nb; ++t) {
        for (int64_t b = 0; b < nblk; ++b) {
            const size_t index = static_cast<size_t>(t) * nblk + b;
            if (!q6g16_checked::quantize_simd(
                    x + static_cast<size_t>(t) * k + b * 256,
                    q8g[index])) {
                throw std::runtime_error(
                    "Checked Q8G16 quantization rejected an activation block");
            }
        }
    }
}

static void dot_q8k(
    const Record& r, const uint8_t* weights,
    const TypeInfo& ti, const std::vector<q6vnni::Q8K>& q8,
    float* output) {

    const int64_t nblk = r.k / 256;

    #pragma omp parallel for schedule(dynamic, 8)
    for (int row = 0; row < r.rows; ++row) {
        const uint8_t* wr =
            weights + static_cast<size_t>(row) * r.row_bytes;

        q6vnni::PredecodedQ6 decoded[r.k / 256];

        for (int64_t b = 0; b < nblk; ++b) {
            const uint8_t* wb = wr + b * ti.bytes;
            q6vnni::decode_q6(wb, rd_f16(wb + 208), decoded[b]);
        }

        for (int t = 0; t < r.nb; ++t) {
            float acc = 0.0f;

            for (int64_t b = 0; b < nblk; ++b) {
                const size_t index = static_cast<size_t>(t) * nblk + b;
                acc += q6vnni::dot_q6_hadd(decoded[b], q8[index]);
            }

            output[static_cast<size_t>(t) * r.rows + row] = acc;
        }
    }
}

static void dot_g16(
    const Record& r, const uint8_t* weights,
    const TypeInfo& ti, const std::vector<q6g16::Q8G16>& q8g,
    float* output) {

    const int64_t nblk = r.k / 256;

    #pragma omp parallel for schedule(dynamic, 8)
    for (int row = 0; row < r.rows; ++row) {
        const uint8_t* wr =
            weights + static_cast<size_t>(row) * r.row_bytes;

        q6vnni::PredecodedQ6 decoded[r.k / 256];

        for (int64_t b = 0; b < nblk; ++b) {
            const uint8_t* wb = wr + b * ti.bytes;
            q6vnni::decode_q6(wb, rd_f16(wb + 208), decoded[b]);
        }

        for (int t = 0; t < r.nb; ++t) {
            float acc = 0.0f;

            for (int64_t b = 0; b < nblk; ++b) {
                const size_t index = static_cast<size_t>(t) * nblk + b;
                acc += q6g16::dot_q6_g16(decoded[b], q8g[index]);
            }

            output[static_cast<size_t>(t) * r.rows + row] = acc;
        }
    }
}


static void dot_g16_v2(
    const Record& r, const uint8_t* weights,
    const TypeInfo& ti, const std::vector<q6g16::Q8G16>& q8g,
    float* output) {

    const int64_t nblk = r.k / 256;

    #pragma omp parallel for schedule(dynamic, 8)
    for (int row = 0; row < r.rows; ++row) {
        const uint8_t* wr =
            weights + static_cast<size_t>(row) * r.row_bytes;

        q6vnni::PredecodedQ6 decoded[r.k / 256];

        for (int64_t b = 0; b < nblk; ++b) {
            const uint8_t* wb = wr + b * ti.bytes;
            q6vnni::decode_q6(wb, rd_f16(wb + 208), decoded[b]);
        }

        for (int t = 0; t < r.nb; ++t) {
            float acc = 0.0f;

            for (int64_t b = 0; b < nblk; ++b) {
                const size_t index = static_cast<size_t>(t) * nblk + b;
                acc += q6g16_v2::dot_q6_g16_v2(decoded[b], q8g[index]);
            }

            output[static_cast<size_t>(t) * r.rows + row] = acc;
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 6) {
        std::fprintf(stderr,
            "Usage: %s weights.bin activations.f32 descriptors.tsv threads label\n",
            argv[0]);
        return 2;
    }

    try {
        const int threads = std::atoi(argv[4]);
        if (threads < 1) throw std::runtime_error("Invalid thread count");

        Mapping weights(argv[1]);
        Mapping activations(argv[2]);
        const auto records = load_records(argv[3]);

        init_types();
        omp_set_dynamic(0);
        omp_set_num_threads(threads);

        if (!q6vnni::available())
            throw std::runtime_error("AVX-VNNI unavailable");

        std::printf("Q6 activation benchmark | %s | threads=%d\n",
                    argv[5], threads);
        std::printf("Warmups=2 measured_trials=7 records=%zu\n",
                    records.size());
        std::printf("Q8K=%zu bytes Q8G16=%zu bytes\n",
                    sizeof(q6vnni::Q8K), sizeof(q6g16::Q8G16));

        for (const auto& r : records) {
            const TypeInfo& ti = g_types[r.type];
            const int64_t nblk = r.k / 256;
            const size_t activation_count =
                static_cast<size_t>(r.nb) * nblk;
            const size_t output_count =
                static_cast<size_t>(r.nb) * r.rows;

            const uint64_t weight_bytes =
                static_cast<uint64_t>(r.rows) * r.row_bytes;
            const uint64_t activation_bytes =
                static_cast<uint64_t>(r.nb) * r.k * sizeof(float);

            if (r.row_bytes != nblk * ti.bytes ||
                r.weight_offset > weights.size ||
                weight_bytes > weights.size - r.weight_offset ||
                r.activation_offset > activations.size ||
                activation_bytes > activations.size - r.activation_offset)
                throw std::runtime_error("Replay bounds/layout validation failed");

            const uint8_t* wdata = weights.data + r.weight_offset;
            const float* x = reinterpret_cast<const float*>(
                activations.data + r.activation_offset);

            if (reinterpret_cast<uintptr_t>(x) % alignof(float))
                throw std::runtime_error("Unaligned activation data");

            std::vector<q6vnni::Q8K> q8(activation_count);
            std::vector<q6g16::Q8G16> q8g(activation_count);
            std::vector<float> out_q8(output_count);
            std::vector<float> out_g16(output_count);
            std::vector<float> out_g16_v2(output_count);

            const double q8_quant_us = median_us([&] {
                quantize_q8k(x, r.nb, nblk, q8, r.k);
            });

            const double g16_quant_us = median_us([&] {
                quantize_g16(x, r.nb, nblk, q8g, r.k);
            });

            const double q8_dot_us = median_us([&] {
                dot_q8k(r, wdata, ti, q8, out_q8.data());
            });

            const double g16_dot_us = median_us([&] {
                dot_g16(r, wdata, ti, q8g, out_g16.data());
            });

            const double g16_v2_dot_us = median_us([&] {
                dot_g16_v2(r, wdata, ti, q8g, out_g16_v2.data());
            });

            double max_abs = 0.0;
            double squared_error = 0.0;
            double squared_reference = 0.0;

            for (size_t i = 0; i < output_count; ++i) {
                const double a = out_g16[i];
                const double b = out_g16_v2[i];
                const double diff = std::fabs(a - b);

                max_abs = std::max(max_abs, diff);
                squared_error += diff * diff;
                squared_reference += a * a;
            }

            const double rmse =
                std::sqrt(squared_error / output_count);
            const double reference_rms =
                std::sqrt(squared_reference / output_count);
            const double nrmse =
                reference_rms > 0.0 ? rmse / reference_rms : 0.0;

            // Actual existing production Q8K matmul.
            const double q8_production_us = median_us([&] {
                matmul(
                    WeightView{
                        .data = wdata,
                        .bytes = static_cast<size_t>(weight_bytes),
                        .type = static_cast<uint32_t>(r.type),
                        .ne0 = r.k,
                        .ne1 = r.rows,
                        .row_bytes = static_cast<size_t>(r.row_bytes)
                    },
                    x, r.nb, r.k, out_q8.data(), r.rows);
            });

            // G16 end-to-end: allocate, quantize, decode weights and dot.
            const double g16_full_us = median_us([&] {
                std::vector<q6g16::Q8G16> fresh(activation_count);
                quantize_g16(x, r.nb, nblk, fresh, r.k);
                dot_g16(r, wdata, ti, fresh, out_g16.data());
            });

            const double g16_v2_full_us = median_us([&] {
                std::vector<q6g16::Q8G16> fresh(activation_count);
                quantize_g16(x, r.nb, nblk, fresh, r.k);
                dot_g16_v2(r, wdata, ti, fresh, out_g16_v2.data());
            });

            const double macs =
                static_cast<double>(r.rows) * r.k * r.nb;

            std::printf(
                "\nrecord=%d op=%s K=%d tokens=%d rows=%d "
                "outputs=%zu MACs=%.0f\n",
                r.id, r.operation.c_str(), r.k, r.nb, r.rows,
                output_count, macs);

            std::printf(
                "  quantize: Q8K=%9.3f us | G16=%9.3f us | ratio=%.3fx\n",
                q8_quant_us, g16_quant_us,
                q8_quant_us > 0 ? g16_quant_us / q8_quant_us : 0);

            std::printf(
                "  dot-only: Q8K=%9.3f us | G16=%9.3f us | ratio=%.3fx\n",
                q8_dot_us, g16_dot_us,
                q8_dot_us > 0 ? g16_dot_us / q8_dot_us : 0);

            std::printf(
                "  full:     Q8K-production=%9.3f us | G16=%9.3f us | "
                "ratio=%.3fx\n",
                q8_production_us, g16_full_us,
                q8_production_us > 0 ? g16_full_us / q8_production_us : 0);

            std::printf(
                "  V1/V2 dot: %9.3f / %9.3f us | V2/V1=%.3fx\n",
                g16_dot_us, g16_v2_dot_us,
                g16_dot_us > 0 ? g16_v2_dot_us / g16_dot_us : 0);

            std::printf(
                "  V1/V2 full: %9.3f / %9.3f us | V2/V1=%.3fx\n",
                g16_full_us, g16_v2_full_us,
                g16_full_us > 0 ? g16_v2_full_us / g16_full_us : 0);

            std::printf(
                "  V1/V2 error: max_abs=%.9g RMSE=%.9g NRMSE=%.9g\n",
                max_abs, rmse, nrmse);

            std::printf(
                "  throughput GMAC/s: Q8K=%.3f G16=%.3f\n",
                q8_production_us > 0
                    ? macs / (q8_production_us * 1000.0) : 0,
                g16_full_us > 0
                    ? macs / (g16_full_us * 1000.0) : 0);
        }

        return 0;
    }
    catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
