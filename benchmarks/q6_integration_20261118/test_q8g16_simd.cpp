#include "q6_groupwise_q8_simd.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

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

        if (r.type != 14 || r.k <= 0 || r.k % 256 || r.nb <= 0)
            throw std::runtime_error("Invalid descriptor");

        records.push_back(r);
    }

    return records;
}

template<class F>
static double median_us(F&& fn) {
    for (int i = 0; i < 2; ++i) fn();

    std::vector<double> samples;

    for (int i = 0; i < 7; ++i) {
        const auto begin = std::chrono::steady_clock::now();
        fn();
        const auto end = std::chrono::steady_clock::now();

        samples.push_back(
            std::chrono::duration<double, std::micro>(
                end - begin).count());
    }

    std::sort(samples.begin(), samples.end());
    return samples[3];
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr,
            "Usage: %s activations.f32 descriptors.tsv\n", argv[0]);
        return 2;
    }

    try {
        if (!q6g16_simd::available())
            throw std::runtime_error("AVX2 unavailable");

        Mapping activations(argv[1]);
        const auto records = load_records(argv[2]);

        uint64_t total_blocks = 0;
        uint64_t mismatches = 0;
        uint64_t nonfinite = 0;

        std::printf("AVX2 available: yes\n");
        std::printf("Scalar Q8G16 size: %zu bytes\n",
                    sizeof(q6g16::Q8G16));

        for (const auto& r : records) {
            const int64_t nblk = r.k / 256;
            const uint64_t bytes =
                static_cast<uint64_t>(r.nb) * r.k * sizeof(float);

            if (r.activation_offset > activations.size ||
                bytes > activations.size - r.activation_offset)
                throw std::runtime_error("Activation bounds failure");

            const float* x = reinterpret_cast<const float*>(
                activations.data + r.activation_offset);

            if (reinterpret_cast<uintptr_t>(x) % alignof(float))
                throw std::runtime_error("Unaligned activation pointer");

            const size_t count =
                static_cast<size_t>(r.nb) * nblk;

            std::vector<q6g16::Q8G16> scalar(count);
            std::vector<q6g16::Q8G16> simd(count);

            for (int t = 0; t < r.nb; ++t) {
                for (int64_t b = 0; b < nblk; ++b) {
                    const size_t index =
                        static_cast<size_t>(t) * nblk + b;

                    const float* block =
                        x + static_cast<size_t>(t) * r.k + b * 256;

                    for (int i = 0; i < 256; ++i)
                        if (!std::isfinite(block[i])) ++nonfinite;

                    q6g16::quantize(block, scalar[index]);
                    q6g16_simd::quantize(block, simd[index]);

                    if (std::memcmp(
                            &scalar[index], &simd[index],
                            sizeof(q6g16::Q8G16)) != 0)
                        ++mismatches;
                }
            }

            total_blocks += count;

            const double scalar_us = median_us([&] {
                for (int t = 0; t < r.nb; ++t)
                    for (int64_t b = 0; b < nblk; ++b)
                        q6g16::quantize(
                            x + static_cast<size_t>(t) * r.k + b * 256,
                            scalar[static_cast<size_t>(t) * nblk + b]);
            });

            const double simd_us = median_us([&] {
                for (int t = 0; t < r.nb; ++t)
                    for (int64_t b = 0; b < nblk; ++b)
                        q6g16_simd::quantize(
                            x + static_cast<size_t>(t) * r.k + b * 256,
                            simd[static_cast<size_t>(t) * nblk + b]);
            });

            std::printf(
                "record=%d op=%s blocks=%zu "
                "scalar=%.3f us SIMD=%.3f us speedup=%.3fx\n",
                r.id, r.operation.c_str(), count,
                scalar_us, simd_us,
                simd_us > 0 ? scalar_us / simd_us : 0);
        }

        std::printf("\nTotal activation blocks: %llu\n",
                    static_cast<unsigned long long>(total_blocks));
        std::printf("Nonfinite activation values: %llu\n",
                    static_cast<unsigned long long>(nonfinite));
        std::printf("Byte-level mismatches: %llu\n",
                    static_cast<unsigned long long>(mismatches));

        return (mismatches == 0 && nonfinite == 0) ? 0 : 1;
    }
    catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
