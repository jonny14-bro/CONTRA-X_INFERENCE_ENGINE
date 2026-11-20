#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

extern "C" bool eahmmf_cuda_matmul(
    const uint8_t*, int, int, int, size_t,
    const float*, int, int, float*, int);
extern "C" bool eahmmf_cuda_moe_ffn(
    const uint8_t*, int, int, int, size_t,
    const uint8_t*, int, int, int, size_t,
    const uint8_t*, int, int, int, size_t,
    const float*, int, int, float*, int);

namespace {

constexpr int K = 4096;
constexpr int ROWS = 2048;
constexpr int BATCH = 8;
constexpr size_t ROW_BYTES = (K / 32) * 34;

void put_f16_one(uint8_t* p) {
    p[0] = 0x00;
    p[1] = 0x3c;
}

bool run_matmul(
    const uint8_t* weights, const std::vector<float>& x,
    std::vector<float>& output)
{
    return eahmmf_cuda_matmul(
        weights, 8, K, ROWS, ROW_BYTES, x.data(), BATCH, K,
        output.data(), ROWS);
}

double elapsed_ms(
    const uint8_t* weights, const std::vector<float>& x,
    std::vector<float>& output, int repetitions)
{
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repetitions; ++i) {
        if (!run_matmul(weights, x, output)) return -1.0;
    }
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() /
           repetitions;
}

std::vector<uint8_t> make_q8_weights(int rows, int columns, std::mt19937& rng) {
    const size_t row_bytes = (size_t)(columns / 256) * 272;
    std::vector<uint8_t> weights((size_t)rows * row_bytes, 0);
    std::uniform_int_distribution<int> qdist(-32, 32);
    for (int row = 0; row < rows; ++row) {
        uint8_t* wr = weights.data() + (size_t)row * row_bytes;
        for (int block = 0; block < columns / 32; ++block) {
            uint8_t* b = wr + (size_t)block * 34;
            b[0] = 0;
            b[1] = 0x20; // FP16 1/128
            for (int i = 0; i < 32; ++i)
                b[2 + i] = (uint8_t)(int8_t)qdist(rng);
        }
    }
    return weights;
}

float q8_weight(const uint8_t* row, int k) {
    constexpr float scale = 1.0f / 128.0f;
    const uint8_t* block = row + (size_t)(k / 32) * 34;
    return scale * (float)(int8_t)block[2 + (k % 32)];
}

bool run_resident_ffn_test(std::mt19937& rng) {
    constexpr int D = 256, FF = 256, NB = 4;
    constexpr size_t ROW_BYTES = 272;
    auto gate = make_q8_weights(FF, D, rng);
    auto up = make_q8_weights(FF, D, rng);
    auto down = make_q8_weights(D, FF, rng);
    std::vector<float> x((size_t)NB * D);
    std::uniform_real_distribution<float> xdist(-1.0f, 1.0f);
    for (float& value : x) value = xdist(rng);

    uint8_t *dg = nullptr, *du = nullptr, *dd = nullptr;
    if (cudaMalloc((void**)&dg, gate.size()) != cudaSuccess ||
        cudaMalloc((void**)&du, up.size()) != cudaSuccess ||
        cudaMalloc((void**)&dd, down.size()) != cudaSuccess) {
        if (dg) cudaFree(dg);
        if (du) cudaFree(du);
        if (dd) cudaFree(dd);
        return false;
    }
    bool ok =
        cudaMemcpy(dg, gate.data(), gate.size(), cudaMemcpyHostToDevice) == cudaSuccess &&
        cudaMemcpy(du, up.data(), up.size(), cudaMemcpyHostToDevice) == cudaSuccess &&
        cudaMemcpy(dd, down.data(), down.size(), cudaMemcpyHostToDevice) == cudaSuccess;

    std::vector<float> reference((size_t)NB * D);
    std::vector<float> baseline((size_t)NB * D);
    std::vector<float> tensor_core((size_t)NB * D);
    std::vector<float> gate_values((size_t)NB * FF);
    std::vector<float> up_values((size_t)NB * FF);
    for (int batch = 0; batch < NB; ++batch) {
        for (int row = 0; row < FF; ++row) {
            float gv = 0.0f, uv = 0.0f;
            const uint8_t* gr = gate.data() + (size_t)row * ROW_BYTES;
            const uint8_t* ur = up.data() + (size_t)row * ROW_BYTES;
            for (int k = 0; k < D; ++k) {
                gv += q8_weight(gr, k) * x[(size_t)batch * D + k];
                uv += q8_weight(ur, k) * x[(size_t)batch * D + k];
            }
            gate_values[(size_t)batch * FF + row] =
                gv / (1.0f + std::exp(-gv));
            up_values[(size_t)batch * FF + row] = uv;
        }
        for (int row = 0; row < D; ++row) {
            float value = 0.0f;
            const uint8_t* dr = down.data() + (size_t)row * ROW_BYTES;
            for (int k = 0; k < FF; ++k)
                value += q8_weight(dr, k) *
                    gate_values[(size_t)batch * FF + k] *
                    up_values[(size_t)batch * FF + k];
            reference[(size_t)batch * D + row] = value;
        }
    }

    auto run = [&](float* output) {
        return eahmmf_cuda_moe_ffn(
            dg, 8, D, FF, ROW_BYTES, du, 8, D, FF, ROW_BYTES,
            dd, 8, FF, D, ROW_BYTES, x.data(), NB, D, output, D);
    };
    if (ok) {
        unsetenv("EAHMMF_DP4A");
        unsetenv("EAHMMF_TENSOR_CORES");
        ok = run(baseline.data());
    }
    if (ok) {
        if (setenv("EAHMMF_TENSOR_CORES", "1", 1) != 0) ok = false;
        else ok = run(tensor_core.data());
    }
    unsetenv("EAHMMF_TENSOR_CORES");
    cudaFree(dg);
    cudaFree(du);
    cudaFree(dd);
    if (!ok) return false;

    auto normalized_error = [&](const std::vector<float>& candidate) {
        double error_squared = 0.0, reference_squared = 0.0;
        for (size_t i = 0; i < reference.size(); ++i) {
            const double error = (double)candidate[i] - reference[i];
            error_squared += error * error;
            reference_squared += (double)reference[i] * reference[i];
        }
        return reference_squared > 0.0
            ? std::sqrt(error_squared / reference_squared)
            : std::sqrt(error_squared);
    };
    const double baseline_error = normalized_error(baseline);
    const double tensor_core_error = normalized_error(tensor_core);
    std::printf(
        "GPU-resident Q8_0 FFN normalized_rms: CUDA-core=%.8g Tensor-Core=%.8g\n",
        baseline_error, tensor_core_error);
    return std::isfinite(baseline_error) &&
           std::isfinite(tensor_core_error) &&
           baseline_error <= 0.005 && tensor_core_error <= 0.005;
}

} // namespace

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
    if (prop.major < 7) {
        std::printf("Tensor Core differential benchmark: SKIP (SM 7.0 required)\n");
        return 0;
    }

    std::mt19937 rng(20261119);
    std::uniform_int_distribution<int> qdist(-96, 96);
    std::uniform_real_distribution<float> xdist(-0.5f, 0.5f);
    std::vector<uint8_t> host_weights((size_t)ROWS * ROW_BYTES, 0);
    for (int row = 0; row < ROWS; ++row) {
        uint8_t* wr = host_weights.data() + (size_t)row * ROW_BYTES;
        for (int block = 0; block < K / 32; ++block) {
            uint8_t* b = wr + (size_t)block * 34;
            put_f16_one(b);
            for (int i = 0; i < 32; ++i)
                b[2 + i] = (uint8_t)(int8_t)qdist(rng);
        }
    }

    std::vector<float> x((size_t)BATCH * K);
    for (float& v : x) v = xdist(rng);
    std::vector<float> reference((size_t)BATCH * ROWS);
    std::vector<float> tensor_core((size_t)BATCH * ROWS);

    uint8_t* device_weights = nullptr;
    if (cudaMalloc((void**)&device_weights, host_weights.size()) != cudaSuccess)
        return 2;
    if (cudaMemcpy(device_weights, host_weights.data(), host_weights.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(device_weights);
        return 2;
    }

    unsetenv("EAHMMF_DP4A");
    unsetenv("EAHMMF_TENSOR_CORES");
    bool ok = run_matmul(device_weights, x, reference);
    if (ok) {
        if (setenv("EAHMMF_TENSOR_CORES", "1", 1) != 0) ok = false;
        else ok = run_matmul(device_weights, x, tensor_core);
    }
    if (!ok) {
        cudaFree(device_weights);
        std::fprintf(stderr, "Tensor Core matmul execution failed\n");
        return 2;
    }

    double error_squared = 0.0, reference_squared = 0.0;
    double max_abs = 0.0;
    bool finite = true;
    for (size_t i = 0; i < reference.size(); ++i) {
        if (!std::isfinite(reference[i]) || !std::isfinite(tensor_core[i])) {
            finite = false;
            continue;
        }
        const double diff = (double)tensor_core[i] - reference[i];
        error_squared += diff * diff;
        reference_squared += (double)reference[i] * reference[i];
        max_abs = std::max(max_abs, std::fabs(diff));
    }
    const double rms_relative = reference_squared > 0.0
        ? std::sqrt(error_squared / reference_squared) : std::sqrt(error_squared);

    constexpr int repetitions = 20;
    const double tensor_core_ms =
        elapsed_ms(device_weights, x, tensor_core, repetitions);
    unsetenv("EAHMMF_TENSOR_CORES");
    const double cuda_core_ms =
        elapsed_ms(device_weights, x, reference, repetitions);

    std::printf(
        "Q8_0 K=%d rows=%d batch=%d rms_rel=%.8g max_abs=%.8g finite=%s\n",
        K, ROWS, BATCH, rms_relative, max_abs, finite ? "yes" : "NO");
    std::printf(
        "mean end-to-end latency over %d runs: CUDA-core=%.4f ms "
        "Tensor-Core=%.4f ms latency_ratio(Tensor/CUDA)=%.3fx\n",
        repetitions, cuda_core_ms, tensor_core_ms,
        cuda_core_ms > 0.0 ? tensor_core_ms / cuda_core_ms : 0.0);
    cudaFree(device_weights);

    const bool passed = finite && std::isfinite(rms_relative) &&
                        rms_relative <= 0.005 &&
                        cuda_core_ms > 0.0 && tensor_core_ms > 0.0;
    const bool resident_ffn_passed = run_resident_ffn_test(rng);
    std::printf("Tensor Core differential benchmark: %s\n",
                passed ? "PASS" : "FAIL");
    std::printf("GPU-resident FFN staging differential test: %s\n",
                resident_ffn_passed ? "PASS" : "FAIL");
    return passed && resident_ffn_passed ? 0 : 1;
}
