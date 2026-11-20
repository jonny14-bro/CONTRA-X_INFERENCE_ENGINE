

// =============================================================================
// GPU-resident expert FFN (roadmap item A) + persistent/pinned scratch (item E)
//
//   host x --H2D--> [ gate | up ] -> silu(gate)*up -> down --D2H--> host y
//
// Gate/up results, the SiLU*up activation and the down-projection input never
// leave the device. One activation upload, one result download, one stream
// synchronization per expert. Pinned staging buffers make both copies true
// DMA transfers instead of pageable-memory copies.
// =============================================================================

__global__ static void silu_mul_kernel(float* gate, const float* up, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gate[i];
    gate[i] = g / (1.0f + expf(-g)) * up[i];
}

namespace {

struct FfnScratch {
    float* x = nullptr;   size_t cx = 0;    // device: expert input   [nb, D]
    float* g = nullptr;   size_t cg = 0;    // device: gate / act     [nb, FF]
    float* u = nullptr;   size_t cu = 0;    // device: up             [nb, FF]
    float* y = nullptr;   size_t cy = 0;    // device: down output    [nb, D]
    int8_t* qx = nullptr; size_t cqx = 0;   // device: Q8_K activations
    float* qs = nullptr;  size_t cqs = 0;   // device: Q8_K inverse scales
    float* hx = nullptr;  size_t chx = 0;   // pinned host staging (in)
    float* hy = nullptr;  size_t chy = 0;   // pinned host staging (out)
    cudaStream_t stream = nullptr;
};

template <typename T>
bool grow_device(T** ptr, size_t* cap, size_t need)
{
    if (need <= *cap) return true;
    if (*ptr) { cudaFree(*ptr); *ptr = nullptr; *cap = 0; }
    if (!CUDA_OK(cudaMalloc((void**)ptr, need))) return false;
    *cap = need;
    return true;
}

// Pinned allocation is an optimization: on failure *ptr stays null and the
// caller transfers directly from/to the caller's (pageable) buffer.
bool grow_pinned(float** ptr, size_t* cap, size_t need)
{
    if (need <= *cap) return *ptr != nullptr;
    if (*ptr) { cudaFreeHost(*ptr); *ptr = nullptr; *cap = 0; }
    if (!CUDA_OK(cudaMallocHost((void**)ptr, need))) {
        *ptr = nullptr;
        cudaGetLastError();  // clear the sticky error from the failed attempt
        return false;
    }
    *cap = need;
    return true;
}

size_t quant_bytes_per_256(int t)
{
    return t == 8 ? 272 : t == 12 ? 144 : t == 13 ? 176 : t == 14 ? 210 : 0;
}

bool type_supported(int t) { return quant_bytes_per_256(t) != 0; }

bool dp4a_eligible(int type, int ne0, size_t row_bytes)
{
    const size_t per = quant_bytes_per_256(type);
    return per != 0 && (ne0 % 256) == 0 &&
           row_bytes == (size_t)(ne0 / 256) * per && dp4a_requested();
}

// out[nb, ldo] = W[ne1, ne0] * x[nb, ldx], everything device-resident.
// With use_dp4a, (qx, qs) must already hold the Q8_K form of x.
bool matmul_device(
    cudaStream_t stream, const uint8_t* w, int type, int ne0, int ne1,
    size_t row_bytes, const float* x, int nb, int ldx, float* out, int ldo,
    bool use_dp4a, const int8_t* qx, const float* qs)
{
    if (use_dp4a) {
        const int total = ne1 * nb;
        quantized_dp4a_matmul_kernel<<<(total + 127) / 128, 128, 0, stream>>>(
            w, qx, qs, out, type, ne0, ne1, row_bytes, nb, ldo, ne0 / 256);
        return CUDA_OK(cudaGetLastError());
    }
    static const bool use_old_kernel = getenv("EAHMMF_OLD_KERNEL") != nullptr;
    if (use_old_kernel || (ne0 & 255)) {
        const int total = ne1 * nb;
        expert_matmul_kernel<<<(total + 127) / 128, 128, 0, stream>>>(
            w, x, out, type, ne0, ne1, row_bytes, nb, ldx, ldo);
    } else {
        constexpr int TB = 8;
        dim3 grid((ne1 + 3) / 4, (nb + TB - 1) / TB);
        expert_matmul_warp_kernel<TB><<<grid, 128, 0, stream>>>(
            w, x, out, type, ne0, ne1, row_bytes, nb, ldx, ldo);
    }
    return CUDA_OK(cudaGetLastError());
}

bool quantize_device(cudaStream_t stream, const float* x, int ne0, int ldx,
                     int nb, int8_t* qx, float* qs)
{
    const int blocks = ne0 / 256;
    quantize_q8k_activations_kernel<<<nb * blocks, 256, 0, stream>>>(
        x, qx, qs, ne0, ldx, blocks);
    return CUDA_OK(cudaGetLastError());
}

} // namespace

// gate/up: [FF rows, D cols]; down: [D rows, FF cols]. Single-threaded caller.
// Returns false (without writing host_out) when the configuration is
// unsupported, so the caller can fall back to the per-matmul path.
extern "C" bool eahmmf_cuda_expert_ffn(
    const uint8_t* gw, int gt, size_t grb,
    const uint8_t* uw, int ut, size_t urb,
    const uint8_t* dw, int dt, size_t drb,
    int D, int FF,
    const float* host_x, int nb, int ldx,
    float* host_out, int ldo)
{
    if (!gw || !uw || !dw || !host_x || !host_out ||
        !type_supported(gt) || !type_supported(ut) || !type_supported(dt) ||
        D <= 0 || FF <= 0 || nb <= 0 || ldx != D || ldo != D)
        return false;

    static FfnScratch S;

    const size_t xb = (size_t)nb * D * sizeof(float);
    const size_t fb = (size_t)nb * FF * sizeof(float);

    const bool gu_dp4a = dp4a_eligible(gt, D, grb) && dp4a_eligible(ut, D, urb);
    const bool dn_dp4a = dp4a_eligible(dt, FF, drb);
    const int qmax = std::max(gu_dp4a ? D : 0, dn_dp4a ? FF : 0);

    if (!S.stream &&
        !CUDA_OK(cudaStreamCreateWithFlags(&S.stream, cudaStreamNonBlocking)))
        return false;
    cudaStream_t stream = S.stream;

    if (!grow_device(&S.x, &S.cx, xb) || !grow_device(&S.g, &S.cg, fb) ||
        !grow_device(&S.u, &S.cu, fb) || !grow_device(&S.y, &S.cy, xb))
        return false;
    if (qmax > 0) {
        if (!grow_device(&S.qx, &S.cqx, (size_t)nb * qmax) ||
            !grow_device(&S.qs, &S.cqs, (size_t)nb * (qmax / 256) * sizeof(float)))
            return false;
    }
    const bool pin_in = grow_pinned(&S.hx, &S.chx, xb);
    const bool pin_out = grow_pinned(&S.hy, &S.chy, xb);

    static bool announced = false;
    if (!announced) {
        fprintf(stderr,
            "cuda-ffn: GPU-resident expert FFN enabled (dp4a gate/up=%d down=%d, pinned staging in=%d out=%d)\n",
            (int)gu_dp4a, (int)dn_dp4a, (int)pin_in, (int)pin_out);
        announced = true;
    }

    const float* src = host_x;
    if (pin_in) { memcpy(S.hx, host_x, xb); src = S.hx; }

    bool ok = CUDA_OK(cudaMemcpyAsync(S.x, src, xb, cudaMemcpyHostToDevice, stream));

    // gate + up share one activation quantization
    if (ok && gu_dp4a) ok = quantize_device(stream, S.x, D, D, nb, S.qx, S.qs);
    if (ok) ok = matmul_device(stream, gw, gt, D, FF, grb, S.x, nb, D, S.g, FF,
                               gu_dp4a, S.qx, S.qs);
    if (ok) ok = matmul_device(stream, uw, ut, D, FF, urb, S.x, nb, D, S.u, FF,
                               gu_dp4a, S.qx, S.qs);

    if (ok) {
        const int n = nb * FF;
        silu_mul_kernel<<<(n + 255) / 256, 256, 0, stream>>>(S.g, S.u, n);
        ok = CUDA_OK(cudaGetLastError());
    }

    // down consumes the activation in place on the device
    if (ok && dn_dp4a) ok = quantize_device(stream, S.g, FF, FF, nb, S.qx, S.qs);
    if (ok) ok = matmul_device(stream, dw, dt, FF, D, drb, S.g, nb, FF, S.y, D,
                               dn_dp4a, S.qx, S.qs);

    float* dst = pin_out ? S.hy : host_out;
    if (ok) ok = CUDA_OK(cudaMemcpyAsync(dst, S.y, xb, cudaMemcpyDeviceToHost, stream));

    // Always drain: the caller releases the expert lease after this returns.
    if (!CUDA_OK(cudaStreamSynchronize(stream))) ok = false;

    if (ok && pin_out) memcpy(host_out, S.hy, xb);
    return ok;
}
