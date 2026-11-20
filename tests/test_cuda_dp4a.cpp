#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

extern "C" bool eahmmf_cuda_matmul(
    const uint8_t*, int, int, int, size_t,
    const float*, int, int, float*, int);

extern "C" bool eahmmf_cuda_matmul_pair(
    const uint8_t*, int, int, int, size_t,
    const uint8_t*, int, int, int, size_t,
    const float*, int, int, float*, int, float*, int);

static void put_f16_one(uint8_t* p) {
    p[0] = 0x00;
    p[1] = 0x3c; // IEEE-754 half 1.0
}

static void put_f16_quarter(uint8_t* p) {
    p[0] = 0x00;
    p[1] = 0x34; // IEEE-754 half 0.25
}

static size_t bytes_per_256(int type) {
    switch (type) {
        case 8: return 272;
        case 12: return 144;
        case 13: return 176;
        case 14: return 210;
        default: return 0;
    }
}

static std::vector<uint8_t> make_weights(
    int type, int rows, std::mt19937& rng)
{
    const size_t rb = bytes_per_256(type);
    std::vector<uint8_t> w((size_t)rows * rb, 0);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::uniform_int_distribution<int> q8_dist(-127, 127);

    for (int row = 0; row < rows; ++row) {
        uint8_t* b = w.data() + (size_t)row * rb;
        if (type == 8) {
            for (int g = 0; g < 8; ++g) {
                uint8_t* q = b + g * 34;
                put_f16_one(q);
                for (int i = 0; i < 32; ++i)
                    q[2 + i] = (uint8_t)(int8_t)q8_dist(rng);
            }
        } else if (type == 12 || type == 13) {
            put_f16_one(b);
            put_f16_quarter(b + 2); // nonzero dmin exercises affine correction
            // All eight 32-value groups: scale=1, min=1.
            b[4] = b[5] = b[6] = b[7] = 1;
            b[8] = b[9] = b[10] = b[11] = 1;
            b[12] = b[13] = b[14] = b[15] = 0x11;
            const size_t qoff = type == 12 ? 16 : 48;
            const size_t qbytes = type == 12 ? 128 : 128;
            for (size_t i = 0; i < qbytes; ++i)
                b[qoff + i] = (uint8_t)byte_dist(rng);
            if (type == 13)
                for (int i = 0; i < 32; ++i)
                    b[16 + i] = (uint8_t)byte_dist(rng);
        } else if (type == 14) {
            for (int i = 0; i < 192; ++i)
                b[i] = (uint8_t)byte_dist(rng);
            for (int i = 0; i < 16; ++i)
                b[192 + i] = 1; // signed Q6 scales
            put_f16_one(b + 208);
        }
    }
    return w;
}

struct ErrorStats {
    double rms_relative = 0.0;
    double max_abs = 0.0;
    double max_reference = 0.0;
    bool finite = true;
};

static ErrorStats compare(
    const std::vector<float>& ref,
    const std::vector<float>& got)
{
    ErrorStats s;
    double e2 = 0.0, r2 = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(ref[i]) || !std::isfinite(got[i])) {
            s.finite = false;
            continue;
        }
        const double e = (double)got[i] - (double)ref[i];
        e2 += e * e;
        r2 += (double)ref[i] * (double)ref[i];
        s.max_abs = std::max(s.max_abs, std::fabs(e));
        s.max_reference = std::max(s.max_reference, std::fabs((double)ref[i]));
    }
    s.rms_relative = r2 > 0.0 ? std::sqrt(e2 / r2) : std::sqrt(e2);
    return s;
}

static bool run_single(int type, std::mt19937& rng) {
    constexpr int K = 256, ROWS = 16, BATCH = 4;
    const size_t rb = bytes_per_256(type);
    auto host_w = make_weights(type, ROWS, rng);
    uint8_t* device_w = nullptr;
    if (cudaMalloc(&device_w, host_w.size()) != cudaSuccess) return false;
    if (cudaMemcpy(device_w, host_w.data(), host_w.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(device_w);
        return false;
    }

    std::uniform_real_distribution<float> xdist(-1.0f, 1.0f);
    std::vector<float> x((size_t)BATCH * K);
    for (float& v : x) v = xdist(rng);
    std::vector<float> baseline((size_t)BATCH * ROWS);
    std::vector<float> dp4a((size_t)BATCH * ROWS);

    unsetenv("EAHMMF_DP4A");
    bool ok = eahmmf_cuda_matmul(
        device_w, type, K, ROWS, rb, x.data(), BATCH, K,
        baseline.data(), ROWS);
    if (ok) {
        setenv("EAHMMF_DP4A", "1", 1);
        ok = eahmmf_cuda_matmul(
            device_w, type, K, ROWS, rb, x.data(), BATCH, K,
            dp4a.data(), ROWS);
    }
    unsetenv("EAHMMF_DP4A");
    cudaFree(device_w);
    if (!ok) return false;

    const ErrorStats e = compare(baseline, dp4a);
    std::printf(
        "single type=%d rms_rel=%.8f max_abs=%.8g max_ref=%.8g finite=%s\n",
        type, e.rms_relative, e.max_abs, e.max_reference,
        e.finite ? "yes" : "NO");
    return e.finite && e.rms_relative <= 0.02;
}

static bool run_pair(std::mt19937& rng) {
    constexpr int K = 256, ROWS = 16, BATCH = 4;
    const int gate_type = 12;
    const int up_type = 14;
    const size_t gate_rb = bytes_per_256(gate_type);
    const size_t up_rb = bytes_per_256(up_type);
    auto gate_host = make_weights(gate_type, ROWS, rng);
    auto up_host = make_weights(up_type, ROWS, rng);
    uint8_t *gate_dev = nullptr, *up_dev = nullptr;
    if (cudaMalloc(&gate_dev, gate_host.size()) != cudaSuccess ||
        cudaMalloc(&up_dev, up_host.size()) != cudaSuccess) {
        if (gate_dev) cudaFree(gate_dev);
        if (up_dev) cudaFree(up_dev);
        return false;
    }
    if (cudaMemcpy(gate_dev, gate_host.data(), gate_host.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(up_dev, up_host.data(), up_host.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(gate_dev);
        cudaFree(up_dev);
        return false;
    }

    std::uniform_real_distribution<float> xdist(-1.0f, 1.0f);
    std::vector<float> x((size_t)BATCH * K);
    for (float& v : x) v = xdist(rng);
    std::vector<float> gate_ref((size_t)BATCH * ROWS);
    std::vector<float> up_ref((size_t)BATCH * ROWS);
    std::vector<float> gate_got((size_t)BATCH * ROWS);
    std::vector<float> up_got((size_t)BATCH * ROWS);

    unsetenv("EAHMMF_DP4A");
    bool ok = eahmmf_cuda_matmul_pair(
        gate_dev, gate_type, K, ROWS, gate_rb,
        up_dev, up_type, K, ROWS, up_rb,
        x.data(), BATCH, K,
        gate_ref.data(), ROWS, up_ref.data(), ROWS);
    if (ok) {
        setenv("EAHMMF_DP4A", "1", 1);
        ok = eahmmf_cuda_matmul_pair(
            gate_dev, gate_type, K, ROWS, gate_rb,
            up_dev, up_type, K, ROWS, up_rb,
            x.data(), BATCH, K,
            gate_got.data(), ROWS, up_got.data(), ROWS);
    }
    unsetenv("EAHMMF_DP4A");
    cudaFree(gate_dev);
    cudaFree(up_dev);
    if (!ok) return false;

    const ErrorStats eg = compare(gate_ref, gate_got);
    const ErrorStats eu = compare(up_ref, up_got);
    std::printf(
        "pair gate_type=%d rms_rel=%.8f max_abs=%.8g finite=%s\n",
        gate_type, eg.rms_relative, eg.max_abs, eg.finite ? "yes" : "NO");
    std::printf(
        "pair up_type=%d rms_rel=%.8f max_abs=%.8g finite=%s\n",
        up_type, eu.rms_relative, eu.max_abs, eu.finite ? "yes" : "NO");
    return eg.finite && eu.finite &&
           eg.rms_relative <= 0.02 && eu.rms_relative <= 0.02;
}

int main() {
    int device = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        std::fprintf(stderr, "CUDA device query failed\n");
        return 2;
    }
    std::printf("GPU: %s (compute %d.%d)\n",
                prop.name, prop.major, prop.minor);
    if (prop.major * 10 + prop.minor < 61) {
        std::fprintf(stderr, "DP4A requires compute capability 6.1 or newer\n");
        return 2;
    }

    std::mt19937 rng(20261119);
    bool ok = true;
    for (int type : {8, 12, 13, 14})
        ok = run_single(type, rng) && ok;
    ok = run_pair(rng) && ok;
    std::printf("DP4A kernel differential test: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
