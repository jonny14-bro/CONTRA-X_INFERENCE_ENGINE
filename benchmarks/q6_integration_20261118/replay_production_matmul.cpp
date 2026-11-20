#define main qwen3moe_embedded_main
#include "qwen3moe_q6.cpp"
#undef main

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

struct ReplayMapping {
    int fd = -1;
    size_t size = 0;
    const uint8_t* data = nullptr;

    explicit ReplayMapping(const char* path) {
        fd = open(path, O_RDONLY);
        if (fd < 0)
            throw std::runtime_error(std::string("Cannot open: ") + path);

        struct stat st {};
        if (fstat(fd, &st) != 0 || st.st_size <= 0)
            throw std::runtime_error(std::string("Invalid file: ") + path);

        size = static_cast<size_t>(st.st_size);
        void* p = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (p == MAP_FAILED) {
            data = nullptr;
            close(fd);
            fd = -1;
            throw std::runtime_error(std::string("mmap failed: ") + path);
        }
        data = static_cast<const uint8_t*>(p);
    }

    ~ReplayMapping() {
        if (data)
            munmap(const_cast<uint8_t*>(data), size);
        if (fd >= 0)
            close(fd);
    }

    ReplayMapping(const ReplayMapping&) = delete;
    ReplayMapping& operator=(const ReplayMapping&) = delete;
};

struct ReplayRecord {
    int id = 0;
    std::string operation;
    int type = 0;
    int k = 0;
    int nb = 0;
    int rows = 0;
    int row_bytes = 0;
    uint64_t activation_offset = 0;
    uint64_t weight_offset = 0;
};

static std::vector<ReplayRecord> load_records(const char* path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("Cannot open descriptor file");

    std::vector<ReplayRecord> records;
    std::string line;

    while (std::getline(in, line)) {
        if (line.empty())
            continue;

        std::istringstream ss(line);
        ReplayRecord r;

        if (!(ss >> r.id >> r.operation >> r.type >> r.k >> r.nb
                 >> r.rows >> r.row_bytes
                 >> r.activation_offset >> r.weight_offset))
            throw std::runtime_error("Malformed replay descriptor");

        if ((r.type != 13 && r.type != 14) ||
            r.k <= 0 || r.k % 256 != 0 ||
            r.nb <= 0 || r.nb > MAXB ||
            r.rows <= 0 || r.row_bytes <= 0)
            throw std::runtime_error("Invalid replay dimensions");

        records.push_back(r);
    }

    return records;
}

int main(int argc, char** argv) {
    if (argc != 6) {
        std::fprintf(stderr,
            "Usage: %s weights.bin activations.f32 descriptors.tsv output.bin threads\n",
            argv[0]);
        return 2;
    }

    try {
        const int threads = std::atoi(argv[5]);
        if (threads < 1)
            throw std::runtime_error("Invalid thread count");

        ReplayMapping weights(argv[1]);
        ReplayMapping activations(argv[2]);
        const auto records = load_records(argv[3]);

        if (records.empty())
            throw std::runtime_error("No records loaded");

        init_types();
        omp_set_dynamic(0);
        omp_set_num_threads(threads);

        std::ofstream output_file(argv[4], std::ios::binary);
        std::ofstream metadata(std::string(argv[4]) + ".tsv");
        if (!output_file || !metadata)
            throw std::runtime_error("Cannot create output files");

#ifdef EAHMMF_DISABLE_Q6_VNNI
        const char* mode = "FORCED_FP32_FALLBACK";
#else
        const char* mode = q6vnni::available()
            ? "RUNTIME_AVX_VNNI" : "RUNTIME_FP32_FALLBACK";
#endif

        std::printf("Mode: %s\n", mode);
        std::printf("AVX-VNNI CPU support: %s\n",
                    q6vnni::available() ? "yes" : "no");
        std::printf("Threads: %d\n", threads);
        std::printf("Records: %zu\n", records.size());

        uint64_t total_outputs = 0;
        uint64_t total_macs = 0;

        for (const auto& r : records) {
            const int block_bytes = r.type == 13 ? 176 : 210;
            const uint64_t weight_bytes =
                static_cast<uint64_t>(r.rows) * r.row_bytes;
            const uint64_t activation_bytes =
                static_cast<uint64_t>(r.nb) * r.k * sizeof(float);

            if (r.row_bytes != (r.k / 256) * block_bytes)
                throw std::runtime_error("Row-byte mismatch");

            if (r.weight_offset > weights.size ||
                weight_bytes > weights.size - r.weight_offset)
                throw std::runtime_error("Weight bounds error");

            if (r.activation_offset > activations.size ||
                activation_bytes > activations.size - r.activation_offset)
                throw std::runtime_error("Activation bounds error");

            const float* x = reinterpret_cast<const float*>(
                activations.data + r.activation_offset);

            if (reinterpret_cast<uintptr_t>(x) % alignof(float))
                throw std::runtime_error("Unaligned activation pointer");

            WeightView W{};
            W.data = weights.data + r.weight_offset;
            W.bytes = static_cast<size_t>(weight_bytes);
            W.type = static_cast<uint32_t>(r.type);
            W.ne0 = r.k;
            W.ne1 = r.rows;
            W.row_bytes = static_cast<size_t>(r.row_bytes);

            std::vector<float> result(
                static_cast<size_t>(r.nb) * r.rows);

            // Invoke the actual production matmul implementation.
            matmul(W, x, r.nb, r.k, result.data(), r.rows);

            const size_t bytes = result.size() * sizeof(float);
            output_file.write(
                reinterpret_cast<const char*>(result.data()),
                static_cast<std::streamsize>(bytes));

            if (!output_file)
                throw std::runtime_error("Output write failed");

            metadata << r.id << '\t' << r.operation << '\t'
                     << r.type << '\t' << r.k << '\t' << r.nb
                     << '\t' << r.rows << '\t' << result.size()
                     << '\n';

            total_outputs += result.size();
            total_macs += static_cast<uint64_t>(r.rows) * r.k * r.nb;

            std::printf("record=%d op=%s type=Q%d K=%d nb=%d rows=%d outputs=%zu\n",
                r.id, r.operation.c_str(), r.type == 13 ? 5 : 6,
                r.k, r.nb, r.rows, result.size());
        }

        std::printf("\nProduction matmul replay completed.\n");
        std::printf("Outputs: %llu\n",
                    static_cast<unsigned long long>(total_outputs));
        std::printf("MACs: %llu\n",
                    static_cast<unsigned long long>(total_macs));
        std::printf("Output file: %s\n", argv[4]);

        return 0;
    }
    catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
