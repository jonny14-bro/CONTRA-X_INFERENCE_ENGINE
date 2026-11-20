#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <vector>

#define CUDA_OK(call) ((call) == cudaSuccess)

__device__ static float half_to_float(uint16_t h) {
    unsigned sign = h >> 15;
    unsigned exp = (h >> 10) & 31;
    unsigned mant = h & 1023;
    float v;
    if (exp == 0)
        v = mant ? ldexpf((float)mant, -24) : 0.0f;
    else if (exp == 31)
        v = mant ? NAN : INFINITY;
    else
        v = ldexpf((float)(mant | 1024), (int)exp - 25);
    return sign ? -v : v;
}

__device__ static uint16_t read_u16(const uint8_t* p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

__device__ static void scale_min_k4(
    int j, const uint8_t* q, uint8_t* d, uint8_t* m)
{
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 15) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

__device__ static float decode_q4k(const uint8_t* b, int k) {
    float d = half_to_float(read_u16(b));
    float mn = half_to_float(read_u16(b + 2));
    const uint8_t* sc = b + 4;
    const uint8_t* q = b + 16;

    int group64 = k / 64;
    int within = k % 64;
    int group32 = k / 32;
    uint8_t s, m;
    scale_min_k4(group32, sc, &s, &m);

    int packed = q[group64 * 32 + (within % 32)];
    int quant = within < 32 ? (packed & 15) : (packed >> 4);
    return d * s * quant - mn * m;
}

__device__ static float decode_q5k(const uint8_t* b, int k) {
    float d = half_to_float(read_u16(b));
    float mn = half_to_float(read_u16(b + 2));
    const uint8_t* sc = b + 4;
    const uint8_t* qh = b + 16;
    const uint8_t* ql = b + 48;

    int group64 = k / 64;
    int within = k % 64;
    int group32 = k / 32;
    uint8_t s, m;
    scale_min_k4(group32, sc, &s, &m);

    uint8_t packed = ql[group64 * 32 + (within % 32)];
    int quant = within < 32 ? (packed & 15) : (packed >> 4);
    int bit = 2 * group64 + (within >= 32 ? 1 : 0);
    if (qh[within % 32] & (1u << bit))
        quant += 16;

    return d * s * quant - mn * m;
}

__device__ static float decode_q6k(const uint8_t* b, int k) {
    const uint8_t* ql = b;
    const uint8_t* qh = b + 128;
    const int8_t* sc = (const int8_t*)(b + 192);
    float d = half_to_float(read_u16(b + 208));

    int half = k / 128;
    int pos = k % 128;
    int quarter = pos / 32;
    int l = pos % 32;

    const uint8_t* low = ql + half * 64;
    const uint8_t* high = qh + half * 32;
    int packed = (quarter == 0 || quarter == 2)
        ? low[l] : low[l + 32];

    int quant = (quarter < 2)
        ? (packed & 15) : (packed >> 4);

    quant |= ((high[l] >> (2 * quarter)) & 3) << 4;
    quant -= 32;

    int scale_index = half * 8 + (l / 16) + 2 * quarter;
    return d * sc[scale_index] * quant;
}

__device__ static float decode_q8_0(const uint8_t* b, int k) {
    float d = half_to_float(read_u16(b));
    return d * (float)((const int8_t*)(b + 2))[k];
}

__device__ static float decode_weight(
    const uint8_t* row, int type, int k)
{
    if (type == 8)
        return decode_q8_0(row + (k / 32) * 34, k % 32);
    const int block = k / 256;
    const int within = k % 256;

    if (type == 12)
        return decode_q4k(row + block * 144, within);
    if (type == 13)
        return decode_q5k(row + block * 176, within);
    if (type == 14)
        return decode_q6k(row + block * 210, within);
    return 0.0f;
}

// Tensor Core path for quantized expert weights. Quantized values and
// activations are converted to FP16 in shared-memory tiles, then multiplied
// with FP32 accumulation. This is opt-in because conversion overhead and
// reduced input precision can outweigh the Tensor Core throughput.
__global__ static void expert_matmul_tensor_core_kernel(
    const uint8_t* weights, const float* x, float* out,
    int type, int ne0, int ne1, size_t row_bytes,
    int nb, int ldx, int ldo)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    using namespace nvcuda;
    constexpr int TILE = 16;
    const int row0 = blockIdx.x * TILE;
    const int batch0 = blockIdx.y * TILE;
    const int lane = threadIdx.x;

    __shared__ __align__(32) half a[TILE * TILE];
    __shared__ __align__(32) half b[TILE * TILE];
    __shared__ __align__(32) float c[TILE * TILE];

    wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, half, wmma::col_major> bf;
    wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> cf;
    wmma::fill_fragment(cf, 0.0f);

    for (int k0 = 0; k0 < ne0; k0 += TILE) {
        for (int i = lane; i < TILE * TILE; i += warpSize) {
            const int row = row0 + i / TILE;
            const int k = k0 + i % TILE;
            a[i] = row < ne1
                ? __float2half(decode_weight(
                    weights + (size_t)row * row_bytes, type, k))
                : __float2half(0.0f);

            const int n = i / TILE;
            const int bk = k0 + i % TILE;
            b[n * TILE + (i % TILE)] =
                batch0 + n < nb && bk < ne0
                    ? __float2half(x[(size_t)(batch0 + n) * ldx + bk])
                    : __float2half(0.0f);
        }
        __syncthreads();

        wmma::load_matrix_sync(af, a, TILE);
        wmma::load_matrix_sync(bf, b, TILE);
        wmma::mma_sync(cf, af, bf, cf);
        __syncthreads();
    }

    wmma::store_matrix_sync(c, cf, TILE, wmma::mem_row_major);
    __syncthreads();
    for (int i = lane; i < TILE * TILE; i += warpSize) {
        const int row = row0 + i / TILE;
        const int batch = batch0 + i % TILE;
        if (row < ne1 && batch < nb)
            out[(size_t)batch * ldo + row] = c[i];
    }
#else
    (void)weights; (void)x; (void)out; (void)type; (void)ne0; (void)ne1;
    (void)row_bytes; (void)nb; (void)ldx; (void)ldo;
#endif
}

__global__ static void expert_matmul_kernel(
    const uint8_t* weights,
    const float* x,
    float* out,
    int type,
    int ne0,
    int ne1,
    size_t row_bytes,
    int nb,
    int ldx,
    int ldo)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = ne1 * nb;
    if (index >= total) return;

    int row = index % ne1;
    int batch = index / ne1;
    const uint8_t* wr = weights + (size_t)row * row_bytes;
    const float* xr = x + (size_t)batch * ldx;

    float acc = 0.0f;
    for (int k = 0; k < ne0; ++k)
        acc += decode_weight(wr, type, k) * xr[k];

    out[(size_t)batch * ldo + row] = acc;
}

__device__ static __forceinline__ float h2f(uint16_t h) { return __half2float(__ushort_as_half(h)); }

// One warp per output row. Lane l handles values [8l, 8l+8) of every 256-value superblock, so the
// block header (d, dmin, scale/min) is decoded once per 8 weights instead of once per weight.
// Requires ne0 % 256 == 0. Up to TB tokens per launch (token tiles on grid.y).
template <int TB>
__global__ static void expert_matmul_warp_kernel(
    const uint8_t* weights, const float* x, float* out,
    int type, int ne0, int ne1, size_t row_bytes, int nb, int ldx, int ldo)
{
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= ne1) return;
    const int t0 = blockIdx.y * TB;
    const int ntb = min(TB, nb - t0);
    const uint8_t* wr = weights + (size_t)row * row_bytes;
    const int nsb = ne0 >> 8;
    const int k0 = lane * 8;

    float acc[TB];
#pragma unroll
    for (int t = 0; t < TB; t++) acc[t] = 0.f;

    for (int sb = 0; sb < nsb; sb++) {
        float w[8];
        if (type == 12 || type == 13) {
            const uint8_t* b = wr + (size_t)sb * (type == 12 ? 144 : 176);
            const float d = h2f(read_u16(b));
            const float mn = h2f(read_u16(b + 2));
            uint8_t s, m;
            scale_min_k4(lane >> 2, b + 4, &s, &m);
            const float ds = d * s, mm = mn * m;
            const int g64 = lane >> 3;
            const bool hi = ((lane >> 2) & 1) != 0;
            const int l0 = (lane & 3) * 8;
            const uint8_t* q = b + (type == 12 ? 16 : 48) + g64 * 32 + l0;
            const uint8_t* qh = b + 16 + l0;
            const int bit = 2 * g64 + (hi ? 1 : 0);
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const int packed = q[i];
                int quant = hi ? (packed >> 4) : (packed & 15);
                if (type == 13 && ((qh[i] >> bit) & 1)) quant += 16;
                w[i] = ds * quant - mm;
            }
        } else if (type == 14) {
            const uint8_t* b = wr + (size_t)sb * 210;
            const float d = h2f(read_u16(b + 208));
            const int half = lane >> 4;
            const int quarter = (lane & 15) >> 2;
            const int l0 = (lane & 3) * 8;
            const uint8_t* low = b + half * 64 + ((quarter == 0 || quarter == 2) ? 0 : 32);
            const uint8_t* high = b + 128 + half * 32;
            const int sidx = half * 8 + (l0 >> 4) + 2 * quarter;
            const float ds = d * (float)((const int8_t*)(b + 192))[sidx];
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const int l = l0 + i;
                const int packed = low[l];
                int quant = (quarter < 2) ? (packed & 15) : (packed >> 4);
                quant |= ((high[l] >> (2 * quarter)) & 3) << 4;
                quant -= 32;
                w[i] = ds * quant;
            }
        } else {  // type 8 (Q8_0): 8 blocks of 32 values per 256
            const uint8_t* b = wr + (size_t)(sb * 8 + (lane >> 2)) * 34;
            const float d = h2f(read_u16(b));
            const int8_t* q = (const int8_t*)(b + 2) + (lane & 3) * 8;
#pragma unroll
            for (int i = 0; i < 8; i++) w[i] = d * (float)q[i];
        }
#pragma unroll
        for (int t = 0; t < TB; t++) {
            if (t < ntb) {
                const float* xp = x + (size_t)(t0 + t) * ldx + (size_t)sb * 256 + k0;
                float s = 0.f;
#pragma unroll
                for (int i = 0; i < 8; i++) s += w[i] * xp[i];
                acc[t] += s;
            }
        }
    }
#pragma unroll
    for (int t = 0; t < TB; t++) {
        if (t < ntb) {
            float v = acc[t];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
            if (lane == 0) out[(size_t)(t0 + t) * ldo + row] = v;
        }
    }
}

// -----------------------------------------------------------------------------
// Quantized-weight DP4A path.
//
// Activations are quantized per 256-value block using the signed-scale
// convention from q6_avxvnni.h. Q4_K/Q5_K use centered integer weights plus
// an exact affine-offset correction; Q6_K and Q8_0 use signed integer weights.
// Four signed int8 products are accumulated per __dp4a. This path is opt-in
// through EAHMMF_DP4A=1; the existing FP32 CUDA kernels remain the fallback.
// -----------------------------------------------------------------------------


__global__ static void silu_mul_inplace_kernel(
    float* gate, const float* up, int count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float g = gate[i];
    gate[i] = (g / (1.0f + expf(-g))) * up[i];
}

__global__ static void quantize_q8k_activations_kernel(
    const float* x, int8_t* qx, float* x_inverse_scales,
    int ne0, int ldx, int nblocks)
{
    const int work = blockIdx.x;
    const int batch = work / nblocks;
    const int block = work - batch * nblocks;
    const int tid = threadIdx.x;
    const int base = block * 256;
    const float* xb = x + (size_t)batch * ldx + base;

    __shared__ float abs_values[256];
    abs_values[tid] = fabsf(xb[tid]);
    __syncthreads();

    // Preserve the reference quantizer's first-maximum tie behavior.
    if (tid == 0) {
        float amax = 0.0f;
        int max_index = -1;
        for (int i = 0; i < 256; ++i) {
            if (abs_values[i] > amax) {
                amax = abs_values[i];
                max_index = i;
            }
        }

        float inverse_scale = 0.0f;
        if (max_index >= 0) {
            const float maximum = xb[max_index];
            inverse_scale = -127.0f / maximum;
        }
        x_inverse_scales[(size_t)batch * nblocks + block] = inverse_scale;
    }
    __syncthreads();

    const float inverse_scale = x_inverse_scales[(size_t)batch * nblocks + block];
    if (inverse_scale == 0.0f) {
        qx[(size_t)batch * ne0 + base + tid] = 0;
    } else {
        int q = __float2int_rn(inverse_scale * xb[tid]);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        qx[(size_t)batch * ne0 + base + tid] = (int8_t)q;
    }
}

__device__ static __forceinline__ int q4k_quant_at(
    const uint8_t* b, int k)
{
    const int group64 = k / 64;
    const int within = k % 64;
    const uint8_t* q = b + 16;
    const int packed = q[group64 * 32 + (within % 32)];
    return within < 32 ? (packed & 15) : (packed >> 4);
}

__device__ static __forceinline__ int q5k_quant_at(
    const uint8_t* b, int k)
{
    const int group64 = k / 64;
    const int within = k % 64;
    const uint8_t* qh = b + 16;
    const uint8_t* ql = b + 48;
    const uint8_t packed = ql[group64 * 32 + (within % 32)];
    int q = within < 32 ? (packed & 15) : (packed >> 4);
    const int bit = 2 * group64 + (within >= 32 ? 1 : 0);
    if (qh[within % 32] & (1u << bit)) q += 16;
    return q;
}

__device__ static __forceinline__ int q6k_quant_at(
    const uint8_t* b, int k)
{
    const int half = k >> 7;
    const int pos = k & 127;
    const int quarter = pos >> 5;
    const int l = pos & 31;
    const uint8_t* low = b + half * 64 +
        ((quarter == 0 || quarter == 2) ? 0 : 32);
    const uint8_t* high = b + 128 + half * 32;

    const int packed = low[l];
    int q = (quarter < 2) ? (packed & 15) : (packed >> 4);
    q |= ((high[l] >> (2 * quarter)) & 3) << 4;
    return q - 32;
}

__device__ static __forceinline__ int pack_s8x4(
    int a, int b, int c, int d)
{
    const uint32_t packed =
        ((uint32_t)(a & 0xff)      ) |
        ((uint32_t)(b & 0xff) <<  8) |
        ((uint32_t)(c & 0xff) << 16) |
        ((uint32_t)(d & 0xff) << 24);
    // Convert the packed 32-bit bit pattern to signed int without relying
    // on a CUDA-specific bit-cast intrinsic.
    return (packed & 0x80000000u)
        ? (int)((int64_t)packed - 0x100000000LL)
        : (int)packed;
}

__global__ static void quantized_dp4a_matmul_kernel(
    const uint8_t* weights, const int8_t* qx, const float* x_inverse_scales,
    float* out, int type, int ne0, int ne1, size_t row_bytes,
    int nb, int ldo, int nblocks)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = ne1 * nb;
    if (index >= total) return;

    const int row = index % ne1;
    const int batch = index / ne1;
    const uint8_t* wr = weights + (size_t)row * row_bytes;
    const int8_t* xrow = qx + (size_t)batch * ne0;

    float result = 0.0f;
    for (int block = 0; block < nblocks; ++block) {
        const int8_t* xb = xrow + (size_t)block * 256;
        const float inverse_scale =
            x_inverse_scales[(size_t)batch * nblocks + block];
        if (inverse_scale == 0.0f) continue;
        const float xd = 1.0f / inverse_scale;

        if (type == 14) { // Q6_K: signed 6-bit values, one scale per 16
            const uint8_t* wb = wr + (size_t)block * 210;
            const int8_t* scales = (const int8_t*)(wb + 192);
            const float wd = half_to_float(read_u16(wb + 208));

            for (int group = 0; group < 16; ++group) {
                int32_t dot = 0;
                const int k0 = group * 16;
#pragma unroll
                for (int j = 0; j < 16; j += 4) {
                    const int k = k0 + j;
                    const int wp = pack_s8x4(
                        q6k_quant_at(wb, k + 0),
                        q6k_quant_at(wb, k + 1),
                        q6k_quant_at(wb, k + 2),
                        q6k_quant_at(wb, k + 3));
                    const int xp = pack_s8x4(
                        (int)xb[k + 0], (int)xb[k + 1],
                        (int)xb[k + 2], (int)xb[k + 3]);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
                    dot = __dp4a(wp, xp, dot);
#else
                    const signed char* wv = (const signed char*)&wp;
                    const signed char* xv = (const signed char*)&xp;
                    dot += (int)wv[0] * (int)xv[0] +
                           (int)wv[1] * (int)xv[1] +
                           (int)wv[2] * (int)xv[2] +
                           (int)wv[3] * (int)xv[3];
#endif
                }
                result += (wd * (float)scales[group] * xd) * (float)dot;
            }
        } else if (type == 12 || type == 13) { // Q4_K / Q5_K
            const size_t block_bytes = type == 12 ? 144 : 176;
            const uint8_t* wb = wr + (size_t)block * block_bytes;
            const float d = half_to_float(read_u16(wb));
            const float dm = half_to_float(read_u16(wb + 2));
            const uint8_t* packed_scales = wb + 4;
            const int center = type == 12 ? 8 : 16;

            for (int group = 0; group < 8; ++group) {
                uint8_t scale_q, min_q;
                scale_min_k4(group, packed_scales, &scale_q, &min_q);
                const float ds = d * (float)scale_q;
                const float offset = ds * (float)center -
                                     dm * (float)min_q;
                int32_t dot = 0;
                int32_t xsum = 0;
                const int k0 = group * 32;
#pragma unroll
                for (int j = 0; j < 32; j += 4) {
                    const int k = k0 + j;
                    const int q0 = (type == 12)
                        ? q4k_quant_at(wb, k + 0) - center
                        : q5k_quant_at(wb, k + 0) - center;
                    const int q1 = (type == 12)
                        ? q4k_quant_at(wb, k + 1) - center
                        : q5k_quant_at(wb, k + 1) - center;
                    const int q2 = (type == 12)
                        ? q4k_quant_at(wb, k + 2) - center
                        : q5k_quant_at(wb, k + 2) - center;
                    const int q3 = (type == 12)
                        ? q4k_quant_at(wb, k + 3) - center
                        : q5k_quant_at(wb, k + 3) - center;
                    const int x0 = (int)xb[k + 0];
                    const int x1 = (int)xb[k + 1];
                    const int x2 = (int)xb[k + 2];
                    const int x3 = (int)xb[k + 3];
                    const int wp = pack_s8x4(q0, q1, q2, q3);
                    const int xp = pack_s8x4(x0, x1, x2, x3);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
                    dot = __dp4a(wp, xp, dot);
#else
                    const signed char* wv = (const signed char*)&wp;
                    const signed char* xv = (const signed char*)&xp;
                    dot += (int)wv[0] * (int)xv[0] +
                           (int)wv[1] * (int)xv[1] +
                           (int)wv[2] * (int)xv[2] +
                           (int)wv[3] * (int)xv[3];
#endif
                    xsum += x0 + x1 + x2 + x3;
                }
                result += (ds * (float)dot + offset * (float)xsum) * xd;
            }
        } else if (type == 8) { // Q8_0: eight 32-value blocks per 256 values
            const uint8_t* w256 = wr + (size_t)block * (8 * 34);
            for (int group = 0; group < 8; ++group) {
                const uint8_t* wb = w256 + (size_t)group * 34;
                const float wd = half_to_float(read_u16(wb));
                const int8_t* wq = (const int8_t*)(wb + 2);
                int32_t dot = 0;
#pragma unroll
                for (int j = 0; j < 32; j += 4) {
                    const int k = group * 32 + j;
                    const int wp = pack_s8x4(
                        (int)wq[j + 0], (int)wq[j + 1],
                        (int)wq[j + 2], (int)wq[j + 3]);
                    const int xp = pack_s8x4(
                        (int)xb[k + 0], (int)xb[k + 1],
                        (int)xb[k + 2], (int)xb[k + 3]);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
                    dot = __dp4a(wp, xp, dot);
#else
                    const signed char* wv = (const signed char*)&wp;
                    const signed char* xv = (const signed char*)&xp;
                    dot += (int)wv[0] * (int)xv[0] +
                           (int)wv[1] * (int)xv[1] +
                           (int)wv[2] * (int)xv[2] +
                           (int)wv[3] * (int)xv[3];
#endif
                }
                result += (wd * xd) * (float)dot;
            }
        }
    }

    out[(size_t)batch * ldo + row] = result;
}

static bool cuda_device_supports_dp4a()
{
    static int cached = -1;
    if (cached >= 0) return cached != 0;

    int device = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        cached = 0;
    } else {
        cached = (prop.major * 10 + prop.minor) >= 61 ? 1 : 0;
    }
    return cached != 0;
}

static bool dp4a_requested()
{
    const char* flag = std::getenv("EAHMMF_DP4A");
    return flag && flag[0] != '\0' && flag[0] != '0' &&
           cuda_device_supports_dp4a();
}

static bool tensor_cores_requested()
{
    const char* flag = std::getenv("EAHMMF_TENSOR_CORES");
    if (!flag || flag[0] == '\0' || flag[0] == '0') return false;

    static int supported = -1;
    if (supported < 0) {
        int device = 0;
        cudaDeviceProp prop{};
        supported = cudaGetDevice(&device) == cudaSuccess &&
                    cudaGetDeviceProperties(&prop, device) == cudaSuccess &&
                    prop.major >= 7;
    }
    return supported != 0;
}

static bool tensor_core_eligible(
    int ne0, int ne1, int nb, size_t row_bytes, int type)
{
    if (!tensor_cores_requested() || ne0 < 256 || ne1 < 16 || nb <= 0 ||
        ne0 % 16 != 0)
        return false;

    const size_t block_bytes =
        type == 8 ? 34 : type == 12 ? 144 : type == 13 ? 176 :
        type == 14 ? 210 : 0;
    const int block_length = type == 8 ? 32 : 256;
    return block_bytes != 0 && ne0 % block_length == 0 &&
           row_bytes == (size_t)(ne0 / block_length) * block_bytes;
}

static bool grow_pinned(float** ptr, size_t* capacity, size_t need)
{
    if (need <= *capacity) return *ptr != nullptr;
    if (*ptr) {
        if (!CUDA_OK(cudaFreeHost(*ptr))) return false;
        *ptr = nullptr;
        *capacity = 0;
    }
    if (!CUDA_OK(cudaMallocHost((void**)ptr, need))) {
        *ptr = nullptr;
        cudaGetLastError();
        return false;
    }
    *capacity = need;
    return true;
}


// C interface called from qwen3moe.cpp.
// Supports GGML type IDs 12 (Q4_K), 13 (Q5_K), and 14 (Q6_K).
extern "C" bool eahmmf_cuda_matmul(
    const uint8_t* device_weights,
    int type,
    int ne0,
    int ne1,
    size_t row_bytes,
    const float* host_x,
    int nb,
    int ldx,
    float* host_out,
    int ldo)
{
    if (!device_weights || !host_x || !host_out ||
        ne0 <= 0 || ne1 <= 0 || nb <= 0 ||
        ldx < ne0 || ldo < ne1 ||
        (type != 8 && type != 12 && type != 13 && type != 14))
        return false;

    // Persistent device scratch (grown on demand, reused across calls; single-threaded caller).
    static float* sx = nullptr;
    static float* so = nullptr;
    static int8_t* sqx = nullptr;
    static float* s_q8_inverse_scales = nullptr;
    static size_t cx = 0, co = 0, cqx = 0, c_q8_inverse_scales = 0;
    size_t xbytes = (size_t)nb * ldx * sizeof(float);
    size_t obytes = (size_t)nb * ldo * sizeof(float);
    const int q8_blocks = ne0 / 256;
    const size_t qx_bytes = (size_t)nb * ne0;
    const size_t q8_inverse_scale_bytes = (size_t)nb * q8_blocks * sizeof(float);
    if (xbytes > cx) {
        if (sx) cudaFree(sx);
        sx = nullptr; cx = 0;
        if (!CUDA_OK(cudaMalloc(&sx, xbytes))) return false;
        cx = xbytes;
    }
    if (obytes > co) {
        if (so) cudaFree(so);
        so = nullptr; co = 0;
        if (!CUDA_OK(cudaMalloc(&so, obytes))) return false;
        co = obytes;
    }
    float* dx = sx;
    float* dout = so;

    const size_t dp4a_weight_bytes_per_256 =
        type == 8 ? 272 : type == 12 ? 144 : type == 13 ? 176 :
        type == 14 ? 210 : 0;
    bool use_dp4a = dp4a_weight_bytes_per_256 != 0 &&
                    (ne0 % 256) == 0 &&
                    row_bytes == (size_t)q8_blocks * dp4a_weight_bytes_per_256 &&
                    dp4a_requested();
    if (use_dp4a) {
        if (qx_bytes > cqx) {
            if (sqx) cudaFree(sqx);
            sqx = nullptr; cqx = 0;
            if (CUDA_OK(cudaMalloc(&sqx, qx_bytes))) cqx = qx_bytes;
            else use_dp4a = false;
        }
        if (use_dp4a && q8_inverse_scale_bytes > c_q8_inverse_scales) {
            if (s_q8_inverse_scales) cudaFree(s_q8_inverse_scales);
            s_q8_inverse_scales = nullptr; c_q8_inverse_scales = 0;
            if (CUDA_OK(cudaMalloc(&s_q8_inverse_scales, q8_inverse_scale_bytes)))
                c_q8_inverse_scales = q8_inverse_scale_bytes;
            else use_dp4a = false;
        }
    }
    if (dp4a_weight_bytes_per_256 != 0 && dp4a_requested() && !use_dp4a) {
        static bool warned = false;
        if (!warned) {
            fprintf(stderr, "cuda-dp4a: requested but unsupported dimensions or scratch allocation; using FP32 CUDA fallback\n");
            warned = true;
        }
    }
    if (use_dp4a) {
        static bool announced = false;
        if (!announced) {
            fprintf(stderr, "cuda-dp4a: quantized-weight DP4A path enabled (Q8_K activation quantization)\n");
            announced = true;
        }
    }

    // Dedicated stream for activation copies, kernel execution, and results.
    // The output is ready for the CPU when this function returns.
    static cudaStream_t stream = nullptr;
    static bool stream_initialized = false;

    if (!stream_initialized) {
        if (!CUDA_OK(cudaStreamCreateWithFlags(
                &stream, cudaStreamNonBlocking))) {
            return false;
        }
        stream_initialized = true;
    }

    bool ok = CUDA_OK(cudaMemcpyAsync(
        dx, host_x, xbytes, cudaMemcpyHostToDevice, stream));

    if (ok) {
        if (use_dp4a) {
            quantize_q8k_activations_kernel<<<nb * q8_blocks, 256, 0, stream>>>(
                dx, sqx, s_q8_inverse_scales, ne0, ldx, q8_blocks);
            ok = CUDA_OK(cudaGetLastError());
            if (ok) {
                const int total = ne1 * nb;
                quantized_dp4a_matmul_kernel<<<(total + 127) / 128, 128, 0, stream>>>(
                    device_weights, sqx, s_q8_inverse_scales, dout,
                    type, ne0, ne1, row_bytes, nb, ldo, q8_blocks);
                ok = CUDA_OK(cudaGetLastError());
            }
        } else {
            static const bool use_old_kernel =
                getenv("EAHMMF_OLD_KERNEL") != nullptr;

            if (!use_old_kernel &&
                tensor_core_eligible(ne0, ne1, nb, row_bytes, type)) {
                static bool announced = false;
                if (!announced) {
                    fprintf(stderr,
                        "cuda-tensor-cores: experimental FP16 WMMA expert matmul enabled\n");
                    announced = true;
                }
                dim3 grid((ne1 + 15) / 16, (nb + 15) / 16);
                expert_matmul_tensor_core_kernel<<<grid, 32, 0, stream>>>(
                    device_weights, dx, dout, type, ne0, ne1,
                    row_bytes, nb, ldx, ldo);
            } else if (use_old_kernel || (ne0 & 255)) {
                int total = ne1 * nb;
                expert_matmul_kernel<<<
                    (total + 127) / 128, 128, 0, stream>>>(
                    device_weights, dx, dout, type, ne0, ne1,
                    row_bytes, nb, ldx, ldo);
            } else {
                constexpr int TB = 8;
                dim3 grid((ne1 + 3) / 4, (nb + TB - 1) / TB);
                expert_matmul_warp_kernel<TB><<<grid, 128, 0, stream>>>(
                    device_weights, dx, dout, type, ne0, ne1,
                    row_bytes, nb, ldx, ldo);
            }
            ok = CUDA_OK(cudaGetLastError());
        }
    }

    if (ok) {
        ok = CUDA_OK(cudaMemcpyAsync(
            host_out, dout, obytes, cudaMemcpyDeviceToHost, stream));
    }

    // Drain queued work even after an error before the caller releases
    // the expert lease or this function's scratch can be reused.
    cudaError_t sync_status = cudaStreamSynchronize(stream);
    if (sync_status != cudaSuccess) ok = false;

    return ok;
}


// Paired gate/up projection: one input transfer, two kernels, two output
// transfers, and one stream synchronization. Scratch is persistent.
// The caller is single-threaded, matching eahmmf_cuda_matmul().
extern "C" bool eahmmf_cuda_matmul_pair(
    const uint8_t* gw, int gt, int gne0, int gne1, size_t grb,
    const uint8_t* uw, int ut, int une0, int une1, size_t urb,
    const float* host_x, int nb, int ldx,
    float* host_g, int ldg, float* host_u, int ldu)
{
    auto supported = [](int t) {
        return t == 8 || t == 12 || t == 13 || t == 14;
    };
    if (!gw || !uw || !host_x || !host_g || !host_u ||
        !supported(gt) || !supported(ut) ||
        gne0 <= 0 || gne1 <= 0 || une0 != gne0 || une1 != gne1 ||
        nb <= 0 || ldx < gne0 || ldg < gne1 || ldu < une1)
        return false;

    const size_t xb = (size_t)nb * ldx * sizeof(float);
    const size_t gb = (size_t)nb * ldg * sizeof(float);
    const size_t ub = (size_t)nb * ldu * sizeof(float);

    static float* sx = nullptr;
    static float* sg = nullptr;
    static float* su = nullptr;
    static int8_t* sqx = nullptr;
    static float* s_q8_inverse_scales = nullptr;
    static size_t cx = 0, cg = 0, cu = 0, cqx = 0, c_q8_inverse_scales = 0;

    auto grow = [](float** ptr, size_t* cap, size_t need) {
        if (need <= *cap) return true;
        if (*ptr) {
            if (!CUDA_OK(cudaFree(*ptr))) return false;
            *ptr = nullptr;
            *cap = 0;
        }
        if (!CUDA_OK(cudaMalloc((void**)ptr, need))) return false;
        *cap = need;
        return true;
    };

    if (!grow(&sx, &cx, xb) ||
        !grow(&sg, &cg, gb) ||
        !grow(&su, &cu, ub))
        return false;

    const int q8_blocks = gne0 / 256;
    const size_t qx_bytes = (size_t)nb * gne0;
    const size_t q8_inverse_scale_bytes = (size_t)nb * q8_blocks * sizeof(float);
    auto dp4a_bytes_per_256 = [](int t) -> size_t {
        return t == 8 ? 272 : t == 12 ? 144 : t == 13 ? 176 :
               t == 14 ? 210 : 0;
    };
    const size_t gate_dp4a_bytes = dp4a_bytes_per_256(gt);
    const size_t up_dp4a_bytes = dp4a_bytes_per_256(ut);
    bool use_dp4a = gate_dp4a_bytes != 0 && up_dp4a_bytes != 0 &&
                    (gne0 % 256) == 0 &&
                    grb == (size_t)q8_blocks * gate_dp4a_bytes &&
                    urb == (size_t)q8_blocks * up_dp4a_bytes &&
                    dp4a_requested();
    if (use_dp4a) {
        if (qx_bytes > cqx) {
            if (sqx) cudaFree(sqx);
            sqx = nullptr; cqx = 0;
            if (CUDA_OK(cudaMalloc(&sqx, qx_bytes))) cqx = qx_bytes;
            else use_dp4a = false;
        }
        if (use_dp4a && q8_inverse_scale_bytes > c_q8_inverse_scales) {
            if (s_q8_inverse_scales) cudaFree(s_q8_inverse_scales);
            s_q8_inverse_scales = nullptr; c_q8_inverse_scales = 0;
            if (CUDA_OK(cudaMalloc(&s_q8_inverse_scales, q8_inverse_scale_bytes)))
                c_q8_inverse_scales = q8_inverse_scale_bytes;
            else use_dp4a = false;
        }
    }
    if (gate_dp4a_bytes != 0 && up_dp4a_bytes != 0 &&
        dp4a_requested() && !use_dp4a) {
        static bool warned = false;
        if (!warned) {
            fprintf(stderr, "cuda-dp4a: paired path unavailable for these dimensions or scratch allocation; using FP32 CUDA fallback\n");
            warned = true;
        }
    }
    if (use_dp4a) {
        static bool announced = false;
        if (!announced) {
            fprintf(stderr, "cuda-dp4a: paired quantized gate/up path enabled (one Q8_K activation quantization)\n");
            announced = true;
        }
    }

    static cudaStream_t stream = nullptr;
    static bool stream_initialized = false;
    if (!stream_initialized) {
        if (!CUDA_OK(cudaStreamCreateWithFlags(
                &stream, cudaStreamNonBlocking)))
            return false;
        stream_initialized = true;
    }

    bool ok = CUDA_OK(cudaMemcpyAsync(
        sx, host_x, xb, cudaMemcpyHostToDevice, stream));

    static const bool use_old_kernel =
        getenv("EAHMMF_OLD_KERNEL") != nullptr;

    auto launch = [&](const uint8_t* weights, int type,
                      int ne0, int ne1, size_t row_bytes,
                      float* output, int ldo) {
        if (!use_old_kernel &&
            tensor_core_eligible(ne0, ne1, nb, row_bytes, type)) {
            static bool announced = false;
            if (!announced) {
                fprintf(stderr,
                    "cuda-tensor-cores: experimental FP16 WMMA expert matmul enabled\n");
                announced = true;
            }
            dim3 grid((ne1 + 15) / 16, (nb + 15) / 16);
            expert_matmul_tensor_core_kernel<<<grid, 32, 0, stream>>>(
                weights, sx, output, type, ne0, ne1,
                row_bytes, nb, ldx, ldo);
        } else if (use_old_kernel || (ne0 & 255)) {
            int total = ne1 * nb;
            expert_matmul_kernel<<<
                (total + 127) / 128, 128, 0, stream>>>(
                weights, sx, output, type, ne0, ne1,
                row_bytes, nb, ldx, ldo);
        } else {
            constexpr int TB = 8;
            dim3 grid((ne1 + 3) / 4, (nb + TB - 1) / TB);
            expert_matmul_warp_kernel<TB><<<grid, 128, 0, stream>>>(
                weights, sx, output, type, ne0, ne1,
                row_bytes, nb, ldx, ldo);
        }
        return CUDA_OK(cudaGetLastError());
    };

    if (ok && use_dp4a) {
        quantize_q8k_activations_kernel<<<nb * q8_blocks, 256, 0, stream>>>(
            sx, sqx, s_q8_inverse_scales, gne0, ldx, q8_blocks);
        ok = CUDA_OK(cudaGetLastError());
        if (ok) {
            const int total = gne1 * nb;
            quantized_dp4a_matmul_kernel<<<(total + 127) / 128, 128, 0, stream>>>(
                gw, sqx, s_q8_inverse_scales, sg,
                gt, gne0, gne1, grb, nb, ldg, q8_blocks);
            ok = CUDA_OK(cudaGetLastError());
        }
        if (ok) {
            const int total = une1 * nb;
            quantized_dp4a_matmul_kernel<<<(total + 127) / 128, 128, 0, stream>>>(
                uw, sqx, s_q8_inverse_scales, su,
                ut, une0, une1, urb, nb, ldu, q8_blocks);
            ok = CUDA_OK(cudaGetLastError());
        }
    } else if (ok) {
        ok = launch(gw, gt, gne0, gne1, grb, sg, ldg);
        if (ok) ok = launch(uw, ut, une0, une1, urb, su, ldu);
    }

    if (ok) {
        ok = CUDA_OK(cudaMemcpyAsync(
            host_g, sg, gb, cudaMemcpyDeviceToHost, stream));
    }
    if (ok) {
        ok = CUDA_OK(cudaMemcpyAsync(
            host_u, su, ub, cudaMemcpyDeviceToHost, stream));
    }

    // Drain queued work even after an error, before scratch can be reused.
    cudaError_t sync_status = cudaStreamSynchronize(stream);
    if (sync_status != cudaSuccess) ok = false;

    return ok;
}


// Fused expert FFN: one H2D input, all three projections and SiLU on
// device, one D2H output. The caller is single-threaded, as are the
// existing persistent-scratch entry points.

struct ResidentFFNProfiler {
    static constexpr int N = 6;
    static constexpr int E = 12;
    const char* names[N] = {"h2d", "gate_up", "silu", "down", "d2h", "total"};
    cudaEvent_t events[E]{};
    bool enabled = false, ready = false;
    uint64_t count[N]{};
    double sum[N]{};
    float low[N]{}, high[N]{};

    ResidentFFNProfiler() {
        const char* v = getenv("EAHMMF_GPU_RESIDENT_FFN_PROFILE");
        enabled = v && v[0] == '1' && v[1] == '\0';
        if (!enabled) return;
        int made = 0;
        for (; made < E; ++made)
            if (cudaEventCreate(&events[made]) != cudaSuccess) break;
        if (made != E) {
            for (int i = 0; i < made; ++i) cudaEventDestroy(events[i]);
            enabled = false;
            fprintf(stderr, "resident-ffn-profile: event creation failed; disabled\n");
            return;
        }
        ready = true;
    }

    void mark(int i, cudaStream_t stream) {
        if (enabled && ready &&
            cudaEventRecord(events[i], stream) != cudaSuccess)
            ready = false;
    }

    void collect() {
        if (!enabled || !ready) return;
        for (int i = 0; i < N; ++i) {
            float ms = 0;
            if (cudaEventElapsedTime(&ms, events[2*i], events[2*i+1])
                != cudaSuccess) continue;
            if (!count[i] || ms < low[i]) low[i] = ms;
            if (!count[i] || ms > high[i]) high[i] = ms;
            sum[i] += ms;
            ++count[i];
        }
    }

    ~ResidentFFNProfiler() {
        if (enabled) {
            fprintf(stderr, "\nresident-ffn-profile:\n");
            for (int i = 0; i < N; ++i) {
                if (!count[i]) {
                    fprintf(stderr, "  %-8s samples=0\n", names[i]);
                    continue;
                }
                fprintf(stderr,
                    "  %-8s samples=%llu total_ms=%.3f avg_ms=%.6f min_ms=%.6f max_ms=%.6f\n",
                    names[i], (unsigned long long)count[i], sum[i],
                    sum[i] / count[i], low[i], high[i]);
            }
        }
        if (ready)
            for (int i = 0; i < E; ++i) cudaEventDestroy(events[i]);
    }
};

extern "C" bool eahmmf_cuda_moe_ffn(
    const uint8_t* gw, int gt, int gne0, int gne1, size_t grb,
    const uint8_t* uw, int ut, int une0, int une1, size_t urb,
    const uint8_t* dw, int dt, int dne0, int dne1, size_t drb,
    const float* host_x, int nb, int ldx,
    float* host_out, int ldo)
{
    static ResidentFFNProfiler profiler;

    auto supported = [](int t) {
        return t == 8 || t == 12 || t == 13 || t == 14;
    };

    if (!gw || !uw || !dw || !host_x || !host_out ||
        !supported(gt) || !supported(ut) || !supported(dt) ||
        gne0 <= 0 || gne1 <= 0 ||
        une0 != gne0 || une1 != gne1 ||
        dne0 != gne1 || dne1 != gne0 ||
        nb <= 0 || ldx < gne0 || ldo != dne1)
        return false;

    const int D = gne0;
    const int FF = gne1;

    const size_t xb = (size_t)nb * ldx * sizeof(float);
    const size_t fb = (size_t)nb * FF * sizeof(float);
    const size_t ob = (size_t)nb * ldo * sizeof(float);

    static float *sx = nullptr, *sg = nullptr, *su = nullptr, *so = nullptr;
    static size_t cx = 0, cg = 0, cu = 0, co = 0;
    static float *host_x_staging = nullptr, *host_out_staging = nullptr;
    static size_t host_x_capacity = 0, host_out_capacity = 0;

    static int8_t *sqx = nullptr, *sqh = nullptr;
    static size_t cqx = 0, cqh = 0;

    static float *sxs = nullptr, *shs = nullptr;
    static size_t cxs = 0, chs = 0;

    static cudaStream_t stream = nullptr;
    static bool stream_initialized = false;

    auto grow_float = [](float*& p, size_t& cap, size_t need) {
        if (need <= cap) return true;
        if (p) {
            if (!CUDA_OK(cudaFree(p))) return false;
            p = nullptr;
            cap = 0;
        }
        if (!CUDA_OK(cudaMalloc((void**)&p, need))) return false;
        cap = need;
        return true;
    };

    auto grow_i8 = [](int8_t*& p, size_t& cap, size_t need) {
        if (need <= cap) return true;
        if (p) {
            if (!CUDA_OK(cudaFree(p))) return false;
            p = nullptr;
            cap = 0;
        }
        if (!CUDA_OK(cudaMalloc((void**)&p, need))) return false;
        cap = need;
        return true;
    };

    if (!grow_float(sx, cx, xb) ||
        !grow_float(sg, cg, fb) ||
        !grow_float(su, cu, fb) ||
        !grow_float(so, co, ob))
        return false;

    const bool pinned_input =
        grow_pinned(&host_x_staging, &host_x_capacity, xb);
    const bool pinned_output =
        grow_pinned(&host_out_staging, &host_out_capacity, ob);
    static bool staging_reported = false;
    if (!staging_reported) {
        fprintf(stderr,
            "resident-ffn: pinned host staging input=%d output=%d\n",
            (int)pinned_input, (int)pinned_output);
        staging_reported = true;
    }

    if (!stream_initialized) {
        if (!CUDA_OK(cudaStreamCreateWithFlags(
                &stream, cudaStreamNonBlocking)))
            return false;
        stream_initialized = true;
    }

    auto dp4a_bytes_per_256 = [](int t) -> size_t {
        return t == 8 ? 272 : t == 12 ? 144 : t == 13 ? 176 :
               t == 14 ? 210 : 0;
    };

    const int db = D / 256;
    const int fb256 = FF / 256;

    const size_t gbytes = dp4a_bytes_per_256(gt);
    const size_t ubytes = dp4a_bytes_per_256(ut);
    const size_t dbytes = dp4a_bytes_per_256(dt);

    // Gate/up share the D-wide quantized activation. Down has its own
    // FF-wide activation quantization and independent eligibility.
    bool dp4a_gu = dp4a_requested() && db > 0 &&
        D % 256 == 0 && gbytes && ubytes &&
        grb == (size_t)db * gbytes &&
        urb == (size_t)db * ubytes;

    bool dp4a_down = dp4a_requested() && fb256 > 0 &&
        FF % 256 == 0 && dbytes &&
        drb == (size_t)fb256 * dbytes;

    if (dp4a_gu) {
        dp4a_gu =
            grow_i8(sqx, cqx, (size_t)nb * D) &&
            grow_float(sxs, cxs, (size_t)nb * db * sizeof(float));
    }

    if (dp4a_down) {
        dp4a_down =
            grow_i8(sqh, cqh, (size_t)nb * FF) &&
            grow_float(shs, chs, (size_t)nb * fb256 * sizeof(float));
    }

    static const bool use_old_kernel =
        getenv("EAHMMF_OLD_KERNEL") != nullptr;

    auto launch_fp32 = [&](const uint8_t* weights, int type,
                           int ne0, int ne1, size_t row_bytes,
                           const float* input, int input_stride,
                           float* output, int output_stride) {
        if (!use_old_kernel &&
            tensor_core_eligible(ne0, ne1, nb, row_bytes, type)) {
            static bool announced = false;
            if (!announced) {
                fprintf(stderr,
                    "cuda-tensor-cores: experimental FP16 WMMA expert FFN path enabled\n");
                announced = true;
            }
            dim3 grid((ne1 + 15) / 16, (nb + 15) / 16);
            expert_matmul_tensor_core_kernel<<<grid, 32, 0, stream>>>(
                weights, input, output, type, ne0, ne1,
                row_bytes, nb, input_stride, output_stride);
        } else if (use_old_kernel || (ne0 & 255)) {
            const int total = ne1 * nb;
            expert_matmul_kernel<<<
                (total + 127) / 128, 128, 0, stream>>>(
                weights, input, output, type, ne0, ne1,
                row_bytes, nb, input_stride, output_stride);
        } else {
            constexpr int TB = 8;
            dim3 grid((ne1 + 3) / 4, (nb + TB - 1) / TB);
            expert_matmul_warp_kernel<TB><<<grid, 128, 0, stream>>>(
                weights, input, output, type, ne0, ne1,
                row_bytes, nb, input_stride, output_stride);
        }
        return CUDA_OK(cudaGetLastError());
    };

    profiler.mark(10, stream);
    profiler.mark(0, stream);
    const float* input_source = host_x;
    if (pinned_input) {
        memcpy(host_x_staging, host_x, xb);
        input_source = host_x_staging;
    }
    bool ok = CUDA_OK(cudaMemcpyAsync(
        sx, input_source, xb, cudaMemcpyHostToDevice, stream));
    profiler.mark(1, stream);
    profiler.mark(2, stream);

    // Gate and up.
    if (ok && dp4a_gu) {
        quantize_q8k_activations_kernel<<<nb * db, 256, 0, stream>>>(
            sx, sqx, sxs, D, ldx, db);
        ok = CUDA_OK(cudaGetLastError());

        if (ok) {
            const int total = gne1 * nb;
            quantized_dp4a_matmul_kernel<<<
                (total + 127) / 128, 128, 0, stream>>>(
                gw, sqx, sxs, sg, gt, D, FF, grb, nb, FF, db);
            ok = CUDA_OK(cudaGetLastError());
        }

        if (ok) {
            const int total = une1 * nb;
            quantized_dp4a_matmul_kernel<<<
                (total + 127) / 128, 128, 0, stream>>>(
                uw, sqx, sxs, su, ut, D, FF, urb, nb, FF, db);
            ok = CUDA_OK(cudaGetLastError());
        }
    } else if (ok) {
        ok = launch_fp32(gw, gt, D, FF, grb, sx, ldx, sg, FF);
        if (ok)
            ok = launch_fp32(uw, ut, D, FF, urb, sx, ldx, su, FF);
    }

    profiler.mark(3, stream);
    profiler.mark(4, stream);

    // SiLU(gate) * up, in-place in sg.
    if (ok) {
        const int total = nb * FF;
        silu_mul_inplace_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
            sg, su, total);
        ok = CUDA_OK(cudaGetLastError());
    }

    profiler.mark(5, stream);
    profiler.mark(6, stream);

    // Down projection consumes the GPU-resident FF-wide activation.
    if (ok && dp4a_down) {
        quantize_q8k_activations_kernel<<<nb * fb256, 256, 0, stream>>>(
            sg, sqh, shs, FF, FF, fb256);
        ok = CUDA_OK(cudaGetLastError());

        if (ok) {
            const int total = dne1 * nb;
            quantized_dp4a_matmul_kernel<<<
                (total + 127) / 128, 128, 0, stream>>>(
                dw, sqh, shs, so, dt, FF, D, drb, nb, ldo, fb256);
            ok = CUDA_OK(cudaGetLastError());
        }
    } else if (ok) {
        ok = launch_fp32(dw, dt, FF, D, drb, sg, FF, so, ldo);
    }

    profiler.mark(7, stream);
    profiler.mark(8, stream);

    if (ok) {
        float* output_destination =
            pinned_output ? host_out_staging : host_out;
        ok = CUDA_OK(cudaMemcpyAsync(
            output_destination, so, ob, cudaMemcpyDeviceToHost, stream));
    }

    profiler.mark(9, stream);
    profiler.mark(11, stream);

    // Always drain queued work before returning. The caller releases its
    // ExpertLease only after this function returns.
    const cudaError_t sync_status = cudaStreamSynchronize(stream);
    if (sync_status != cudaSuccess) ok = false;
    if (ok && pinned_output)
        memcpy(host_out, host_out_staging, ob);
    if (ok) profiler.collect();

    return ok;
}
