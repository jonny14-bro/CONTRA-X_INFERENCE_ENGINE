#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>

class Q8KActivationCapture {
    std::mutex mutex_;
    FILE* data_ = nullptr;
    FILE* manifest_ = nullptr;
    bool enabled_ = false;
    uint64_t records_ = 0;
    uint64_t bytes_ = 0;
    uint64_t max_records_ = 1000;
    uint64_t max_bytes_ = 64ULL * 1024 * 1024;

    static uint64_t env_limit(const char* name, uint64_t fallback) {
        const char* s = std::getenv(name);
        if (!s || !*s) return fallback;
        char* end = nullptr;
        unsigned long long v = std::strtoull(s, &end, 10);
        return (end && *end == '\0') ? static_cast<uint64_t>(v) : fallback;
    }

public:
    Q8KActivationCapture() {
        const char* enabled = std::getenv("EAHMMF_ACTIVATION_CAPTURE");
        if (!enabled || std::strcmp(enabled, "1") != 0) return;

        max_records_ = env_limit("EAHMMF_CAPTURE_MAX_RECORDS", 1000);
        max_bytes_ = env_limit("EAHMMF_CAPTURE_MAX_BYTES", 64ULL * 1024 * 1024);

        const char* dir_env = std::getenv("EAHMMF_CAPTURE_DIR");
        const char* dir = (dir_env && *dir_env)
            ? dir_env : "benchmarks/real_activations_q8k";

        try {
            std::filesystem::create_directories(dir);
        } catch (...) {
            std::fprintf(stderr, "[q8k-capture] cannot create directory: %s\n", dir);
            return;
        }

        const std::string data_path = std::string(dir) + "/activations.f32";
        const std::string manifest_path = std::string(dir) + "/manifest.jsonl";

        data_ = std::fopen(data_path.c_str(), "wb");
        manifest_ = std::fopen(manifest_path.c_str(), "w");

        if (!data_ || !manifest_) {
            std::fprintf(stderr, "[q8k-capture] cannot open output files\n");
            if (data_) std::fclose(data_);
            if (manifest_) std::fclose(manifest_);
            data_ = manifest_ = nullptr;
            return;
        }

        enabled_ = true;
        std::fprintf(stderr,
            "[q8k-capture] enabled: dir=%s max_records=%llu max_bytes=%llu\n",
            dir,
            static_cast<unsigned long long>(max_records_),
            static_cast<unsigned long long>(max_bytes_));
    }

    ~Q8KActivationCapture() {
        std::lock_guard<std::mutex> lock(mutex_);

        if (manifest_) {
            std::fflush(manifest_);
            std::fclose(manifest_);
        }
        if (data_) {
            std::fflush(data_);
            std::fclose(data_);
        }

        if (enabled_) {
            std::fprintf(stderr,
                "[q8k-capture] finished: records=%llu payload_bytes=%llu\n",
                static_cast<unsigned long long>(records_),
                static_cast<unsigned long long>(bytes_));
        }
    }

    void record(
        uint32_t type,
        int64_t k,
        const float* x,
        int nb,
        int ldx,
        const char* operation,
        int layer,
        int expert,
        const char* backend)
    {
        if (!enabled_ || !x) return;
        if (type != 13 && type != 14) return;
        if (k <= 0 || nb <= 0 || ldx < k) return;

        const uint64_t elements = static_cast<uint64_t>(k) *
                                  static_cast<uint64_t>(nb);

        if (elements > UINT64_MAX / sizeof(float)) return;
        const uint64_t payload = elements * sizeof(float);

        std::lock_guard<std::mutex> lock(mutex_);

        if (!enabled_ || records_ >= max_records_) return;
        if (payload > max_bytes_ - (bytes_ > max_bytes_ ? max_bytes_ : bytes_))
            return;

        const long offset = std::ftell(data_);
        if (offset < 0) return;

        for (int t = 0; t < nb; ++t) {
            const float* row = x + static_cast<size_t>(t) * ldx;
            if (std::fwrite(row, sizeof(float), static_cast<size_t>(k), data_)
                    != static_cast<size_t>(k)) {
                enabled_ = false;
                std::fprintf(stderr, "[q8k-capture] binary write failed; capture disabled\n");
                return;
            }
        }

        if (std::fprintf(manifest_,
            "{\"record\":%llu,\"type\":%u,\"operation\":\"%s\","
            "\"layer\":%d,\"expert\":%d,\"backend\":\"%s\","
            "\"k\":%lld,\"nb\":%d,\"ldx\":%d,"
            "\"elements\":%llu,\"offset_bytes\":%ld}\n",
            static_cast<unsigned long long>(records_),
            type, operation, layer, expert, backend,
            static_cast<long long>(k), nb, ldx,
            static_cast<unsigned long long>(elements), offset) < 0) {
            enabled_ = false;
            std::fprintf(stderr, "[q8k-capture] manifest write failed; capture disabled\n");
            return;
        }

        std::fflush(data_);
        std::fflush(manifest_);

        records_++;
        bytes_ += payload;
    }
};

inline Q8KActivationCapture& q8k_activation_capture() {
    static Q8KActivationCapture capture;
    return capture;
}
