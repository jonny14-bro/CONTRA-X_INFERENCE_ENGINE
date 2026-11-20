// qwen3moe.cpp - standalone CPU inference engine for Qwen3-MoE GGUF models.
// Target: Qwen3-Coder-30B-A3B-Instruct-UD-Q5_K_XL.gguf (arch "qwen3moe").
//
//   * GGUF v3 parser (mmap, zero-copy weights)
//   * byte-level BPE tokenizer with the Qwen2 pre-tokenizer (hand-written, exact Unicode L/N tables)
//   * dequant kernels: F32 F16 BF16 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q4_K Q5_K Q6_K
//   * transformer: GQA + QK-norm + NeoX RoPE, 128-expert top-8 MoE (softmax -> top-k -> renorm)
//   * batched prefill (weights are dequantized once per row and reused across the batch)
//   * sampler: repetition penalty, top-k, top-p, min-p, temperature, greedy
//   * ChatML chat mode with persistent KV cache
//
// Build: make        Run: ./qwen3moe -m model.gguf -i
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <ctime>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "eahmmf.h"
#include "os_memory_monitor.h"
#include "etf.h"
#include "eahmmf_optimizer.h"
#include "pwea_client.h"
#include "pwea/native/include/pwea_predictor.h"
#include "pwea_placement.h"
#include "unicode_tables.h"
#include "q8k_activation_capture.h"
#include "q6_avxvnni.h"
#include "q6_groupwise_q8_checked.h"
#include "q6_groupwise_q8_v2.h"

#define MAXB 64  // max tokens per forward batch

[[noreturn]] static void die(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "error: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    exit(1);
}

// ============================================================================
// fp16 / dequantization
// ============================================================================
static float g_f16[65536];
static float f16_slow(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    float v;
    if (e == 0) v = ldexpf((float)m, -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = ldexpf((float)(m | 0x400), (int)e - 25);
    return s ? -v : v;
}
static inline float rd_f16(const uint8_t* p) {
    uint16_t h;
    memcpy(&h, p, 2);
    return g_f16[h];
}

typedef void (*dq_fn)(const uint8_t*, float*);
struct TypeInfo {
    int blck = 0;
    size_t bytes = 0;
    dq_fn dq = nullptr;
    const char* name = "?";
};
static TypeInfo g_types[40];

static void dq_f32(const uint8_t* p, float* y) { memcpy(y, p, 128); }
static void dq_f16(const uint8_t* p, float* y) {
    for (int i = 0; i < 32; i++) y[i] = rd_f16(p + 2 * i);
}
static void dq_bf16(const uint8_t* p, float* y) {
    for (int i = 0; i < 32; i++) {
        uint16_t h;
        memcpy(&h, p + 2 * i, 2);
        uint32_t u = (uint32_t)h << 16;
        memcpy(&y[i], &u, 4);
    }
}
static void dq_q4_0(const uint8_t* b, float* y) {
    float d = rd_f16(b);
    const uint8_t* qs = b + 2;
    for (int j = 0; j < 16; j++) {
        y[j] = ((qs[j] & 15) - 8) * d;
        y[j + 16] = ((qs[j] >> 4) - 8) * d;
    }
}
static void dq_q4_1(const uint8_t* b, float* y) {
    float d = rd_f16(b), m = rd_f16(b + 2);
    const uint8_t* qs = b + 4;
    for (int j = 0; j < 16; j++) {
        y[j] = (qs[j] & 15) * d + m;
        y[j + 16] = (qs[j] >> 4) * d + m;
    }
}
static void dq_q5_0(const uint8_t* b, float* y) {
    float d = rd_f16(b);
    uint32_t qh;
    memcpy(&qh, b + 2, 4);
    const uint8_t* qs = b + 6;
    for (int j = 0; j < 16; j++) {
        int xh0 = ((qh >> j) & 1) << 4, xh1 = ((qh >> (j + 16)) & 1) << 4;
        y[j] = (((qs[j] & 15) | xh0) - 16) * d;
        y[j + 16] = (((qs[j] >> 4) | xh1) - 16) * d;
    }
}
static void dq_q5_1(const uint8_t* b, float* y) {
    float d = rd_f16(b), m = rd_f16(b + 2);
    uint32_t qh;
    memcpy(&qh, b + 4, 4);
    const uint8_t* qs = b + 8;
    for (int j = 0; j < 16; j++) {
        int xh0 = ((qh >> j) & 1) << 4, xh1 = ((qh >> (j + 16)) & 1) << 4;
        y[j] = ((qs[j] & 15) | xh0) * d + m;
        y[j + 16] = ((qs[j] >> 4) | xh1) * d + m;
    }
}
static void dq_q8_0(const uint8_t* b, float* y) {
    float d = rd_f16(b);
    const int8_t* q = (const int8_t*)(b + 2);
    for (int i = 0; i < 32; i++) y[i] = d * q[i];
}
static inline void scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}
static void dq_q4_k(const uint8_t* b, float* y) {
    float d = rd_f16(b), mn = rd_f16(b + 2);
    const uint8_t* sc = b + 4;
    const uint8_t* q = b + 16;
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
        uint8_t s, m;
        scale_min_k4(is + 0, sc, &s, &m);
        float d1 = d * s, m1 = mn * m;
        scale_min_k4(is + 1, sc, &s, &m);
        float d2 = d * s, m2 = mn * m;
        for (int l = 0; l < 32; l++) *y++ = d1 * (q[l] & 0xF) - m1;
        for (int l = 0; l < 32; l++) *y++ = d2 * (q[l] >> 4) - m2;
        q += 32;
        is += 2;
    }
}
static void dq_q5_k(const uint8_t* b, float* y) {
    float d = rd_f16(b), mn = rd_f16(b + 2);
    const uint8_t* sc = b + 4;
    const uint8_t* qh = b + 16;
    const uint8_t* ql = b + 48;
    int is = 0;
    uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
        uint8_t s, m;
        scale_min_k4(is + 0, sc, &s, &m);
        float d1 = d * s, m1 = mn * m;
        scale_min_k4(is + 1, sc, &s, &m);
        float d2 = d * s, m2 = mn * m;
        for (int l = 0; l < 32; l++) *y++ = d1 * ((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
        for (int l = 0; l < 32; l++) *y++ = d2 * ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
        ql += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}
static void dq_q6_k(const uint8_t* b, float* y) {
    const uint8_t* ql = b;
    const uint8_t* qh = b + 128;
    const int8_t* sc = (const int8_t*)(b + 192);
    float d = rd_f16(b + 208);
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; l++) {
            int is = l / 16;
            int q1 = (int8_t)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            int q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            int q3 = (int8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            int q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l] = d * sc[is + 0] * q1;
            y[l + 32] = d * sc[is + 2] * q2;
            y[l + 64] = d * sc[is + 4] * q3;
            y[l + 96] = d * sc[is + 6] * q4;
        }
        y += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

static void init_types() {
    for (int i = 0; i < 65536; i++) g_f16[i] = f16_slow((uint16_t)i);
    auto set = [](int id, const char* n, int blck, size_t bytes, dq_fn f) { g_types[id] = {blck, bytes, f, n}; };
    set(0, "F32", 32, 128, dq_f32);
    set(1, "F16", 32, 64, dq_f16);
    set(2, "Q4_0", 32, 18, dq_q4_0);
    set(3, "Q4_1", 32, 20, dq_q4_1);
    set(6, "Q5_0", 32, 22, dq_q5_0);
    set(7, "Q5_1", 32, 24, dq_q5_1);
    set(8, "Q8_0", 32, 34, dq_q8_0);
    set(12, "Q4_K", 256, 144, dq_q4_k);
    set(13, "Q5_K", 256, 176, dq_q5_k);
    set(14, "Q6_K", 256, 210, dq_q6_k);
    set(30, "BF16", 32, 64, dq_bf16);
}

// ============================================================================
// GGUF
// ============================================================================
struct KV {
    int64_t i = 0;
    double f = 0;
    std::string s;
    bool is_arr = false;
    uint32_t elem_type = 0;
    std::vector<std::string> as;
    std::vector<int64_t> ai;
    std::vector<double> af;
};
struct Tensor {
    std::string name;
    uint32_t type = 0;
    int nd = 0;
    int64_t ne[4] = {1, 1, 1, 1};
    uint64_t off = 0;
    uint64_t file_off = 0;  // absolute offset in the GGUF file
    const uint8_t* data = nullptr;
    size_t row_bytes = 0;
    int64_t nelem() const { return ne[0] * ne[1] * ne[2] * ne[3]; }
};
struct Reader {
    const uint8_t *p, *end;
    template <class T> T get() {
        if (p + sizeof(T) > end) die("GGUF truncated");
        T v;
        memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    std::string str() {
        uint64_t n = get<uint64_t>();
        if (p + n > end) die("GGUF truncated (string)");
        std::string s((const char*)p, n);
        p += n;
        return s;
    }
};
static void read_scalar(Reader& r, uint32_t t, KV& v) {
    switch (t) {
        case 0: v.i = r.get<uint8_t>(); break;
        case 1: v.i = r.get<int8_t>(); break;
        case 2: v.i = r.get<uint16_t>(); break;
        case 3: v.i = r.get<int16_t>(); break;
        case 4: v.i = r.get<uint32_t>(); break;
        case 5: v.i = r.get<int32_t>(); break;
        case 6: v.f = r.get<float>(); break;
        case 7: v.i = r.get<uint8_t>(); break;
        case 8: v.s = r.str(); break;
        case 10: v.i = (int64_t)r.get<uint64_t>(); break;
        case 11: v.i = r.get<int64_t>(); break;
        case 12: v.f = r.get<double>(); break;
        default: die("unsupported GGUF value type %u", t);
    }
}

struct GGUF {
    uint8_t* map = nullptr;
    size_t size = 0;
    std::map<std::string, KV> kv;
    std::map<std::string, Tensor> tensors;

    void load(const char* path, bool populate) {
        int fd = open(path, O_RDONLY);
        if (fd < 0) die("cannot open %s", path);
        struct stat st;
        fstat(fd, &st);
        size = (size_t)st.st_size;
        map = (uint8_t*)mmap(nullptr, size, PROT_READ, MAP_PRIVATE | (populate ? MAP_POPULATE : 0), fd, 0);
        if (map == MAP_FAILED) die("mmap failed");
        close(fd);
        Reader r{map, map + size};
        if (memcmp(map, "GGUF", 4) != 0) die("not a GGUF file");
        r.p += 4;
        uint32_t ver = r.get<uint32_t>();
        if (ver < 2 || ver > 3) die("unsupported GGUF version %u", ver);
        uint64_t nt = r.get<uint64_t>(), nkv = r.get<uint64_t>();
        for (uint64_t i = 0; i < nkv; i++) {
            std::string key = r.str();
            uint32_t t = r.get<uint32_t>();
            KV v;
            if (t == 9) {
                v.is_arr = true;
                v.elem_type = r.get<uint32_t>();
                uint64_t n = r.get<uint64_t>();
                if (v.elem_type == 9) die("nested arrays unsupported");
                for (uint64_t k = 0; k < n; k++) {
                    if (v.elem_type == 8) v.as.push_back(r.str());
                    else {
                        KV e;
                        read_scalar(r, v.elem_type, e);
                        if (v.elem_type == 6 || v.elem_type == 12) v.af.push_back(e.f);
                        else v.ai.push_back(e.i);
                    }
                }
            } else read_scalar(r, t, v);
            kv[key] = std::move(v);
        }
        std::vector<Tensor> ts(nt);
        for (auto& t : ts) {
            t.name = r.str();
            t.nd = (int)r.get<uint32_t>();
            if (t.nd < 1 || t.nd > 4) die("bad tensor dims");
            for (int d = 0; d < t.nd; d++) t.ne[d] = (int64_t)r.get<uint64_t>();
            t.type = r.get<uint32_t>();
            t.off = r.get<uint64_t>();
        }
        uint64_t align = 32;
        auto it = kv.find("general.alignment");
        if (it != kv.end()) align = (uint64_t)it->second.i;
        uint64_t pos = (uint64_t)(r.p - map);
        uint64_t data_start = (pos + align - 1) / align * align;
        for (auto& t : ts) {
            if (t.type >= 40 || !g_types[t.type].dq)
                die("tensor '%s' uses unsupported ggml type %u", t.name.c_str(), t.type);
            const TypeInfo& ti = g_types[t.type];
            if (t.ne[0] % ti.blck) die("tensor '%s': row length %lld not multiple of block %d", t.name.c_str(), (long long)t.ne[0], ti.blck);
            t.row_bytes = (size_t)(t.ne[0] / ti.blck) * ti.bytes;
            size_t nbytes = t.row_bytes * (size_t)(t.ne[1] * t.ne[2] * t.ne[3]);
            if (data_start + t.off + nbytes > size) die("tensor '%s' out of file bounds", t.name.c_str());
            t.file_off = data_start + t.off;
            t.data = map + t.file_off;
            tensors[t.name] = t;
        }
    }
    const KV* find(const std::string& k) const {
        auto it = kv.find(k);
        return it == kv.end() ? nullptr : &it->second;
    }
    int64_t geti(const std::string& k, int64_t def, bool required = false) const {
        auto p = find(k);
        if (!p) { if (required) die("missing metadata key %s", k.c_str()); return def; }
        return p->i;
    }
    double getf(const std::string& k, double def) const {
        auto p = find(k);
        if (!p) return def;
        return (p->elem_type == 6 || p->elem_type == 12 || p->f != 0) ? p->f : (double)p->i;
    }
    std::string gets(const std::string& k, const std::string& def = "") const {
        auto p = find(k);
        return p ? p->s : def;
    }
    const Tensor* tensor(const std::string& n, bool required = true) const {
        auto it = tensors.find(n);
        if (it == tensors.end()) { if (required) die("missing tensor %s", n.c_str()); return nullptr; }
        return &it->second;
    }
};

// ============================================================================
// Math kernels
// ============================================================================
static inline float dotf(const float* a, const float* b, int n) {
    float s = 0;
#pragma omp simd reduction(+ : s)
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

using eahmmf::WeightView;
static WeightView view_of(const Tensor& t) {
    WeightView v;
    v.data = t.data;
    v.bytes = t.row_bytes * (size_t)(t.ne[1] * t.ne[2] * t.ne[3]);
    v.type = t.type;
    v.ne0 = t.ne[0];
    v.ne1 = t.ne[1] * t.ne[2] * t.ne[3];
    v.row_bytes = t.row_bytes;
    return v;
}

// out[t*ldo + r] = dot(W.row(r), x[t*ldx ..]) for r in [0,W.ne1), t in [0,nb).
// The kernel only sees a WeightView: it does not know whether the bytes live in mmap, RAM or VRAM.
static void matmul(const WeightView& W, const float* x,
                   int nb, int ldx, float* out, int ldo) {
    const TypeInfo& ti = g_types[W.type];
    const int blck = ti.blck;
    const int64_t nblk = W.ne0 / blck;
    const int64_t nrows = W.ne1;
    const int64_t row0 = 0;

    // Automatic CPU dispatch priority: Q8G16 -> Q8K -> FP32.
    // Q8G16 is selected whenever its kernel and operation are compatible.

    const char* q8g16_trace_env = std::getenv("Q8G16_TRACE");
    const bool q8g16_trace =
        q8g16_trace_env != nullptr &&
        std::strcmp(q8g16_trace_env, "1") == 0;

    const char* q8g16_disable_env =
        std::getenv("Q8G16_DISABLE");
    const bool q8g16_disabled =
        q8g16_disable_env != nullptr &&
        std::strcmp(q8g16_disable_env, "1") == 0;

    const bool q8g16_eligible =
        !q8g16_disabled &&
        W.type == 14 && nb > 0 && nb <= MAXB &&
        W.ne0 > 0 && W.ne0 % 256 == 0 &&
        blck == 256 && ldx >= W.ne0 &&
        q6vnni::available() &&
        q6g16_simd::available();

    const char* q8k_disable_env = std::getenv("Q8K_DISABLE");
    const bool q8k_disabled =
        q8k_disable_env != nullptr &&
        std::strcmp(q8k_disable_env, "1") == 0;

    const bool q8k_eligible =
        !q8k_disabled &&
        W.type == 14 && nb > 0 && nb <= MAXB &&
        W.ne0 > 0 && W.ne0 % 256 == 0 &&
        blck == 256 && ldx >= W.ne0 &&
        q6vnni::available();

    bool q8g16_rejected = false;

    if (q8g16_eligible) {

        std::vector<q6g16::Q8G16> q8g(
            static_cast<size_t>(nb) * static_cast<size_t>(nblk)
        );

        // Validate and quantize every activation before computing output.
        bool valid = true;
        size_t q8g16_quantized_blocks = 0;

        for (int t = 0; t < nb && valid; ++t) {
            for (int64_t b = 0; b < nblk; ++b) {
                const float* xb =
                    x + static_cast<size_t>(t) * ldx +
                    static_cast<size_t>(b) * 256;

                if (!q6g16_checked::quantize_simd(
                        xb, q8g[static_cast<size_t>(t) * nblk + b])) {
                    valid = false;
                    break;
                }

                ++q8g16_quantized_blocks;
            }
        }

        if (valid) {
            #pragma omp parallel
            {
                #pragma omp for schedule(dynamic, 16)
                for (int64_t r = 0; r < nrows; ++r) {
                    const uint8_t* wr =
                        W.data + static_cast<size_t>(row0 + r) * W.row_bytes;

                    float acc[MAXB];
                    for (int t = 0; t < nb; ++t)
                        acc[t] = 0.0f;

                    for (int64_t b = 0; b < nblk; ++b) {
                        const uint8_t* wb =
                            wr + static_cast<size_t>(b) * ti.bytes;

                        q6vnni::PredecodedQ6 decoded{};
                        q6vnni::decode_q6(wb, rd_f16(wb + 208), decoded);

                        for (int t = 0; t < nb; ++t) {
                            const q6g16::Q8G16& qb =
                                q8g[static_cast<size_t>(t) * nblk + b];

                            acc[t] +=
                                q6g16_v2::dot_q6_g16_v2(decoded, qb);
                        }
                    }

                    for (int t = 0; t < nb; ++t)
                        out[static_cast<size_t>(t) * ldo + r] = acc[t];
                }
            }


            // Optional numerical shadow against the original FP32 Q6_K path.
            // This is diagnostic-only: the Q8G16 output remains authoritative.
            const char* q8g16_shadow_env = std::getenv("Q8G16_SHADOW");
            const bool q8g16_shadow =
                q8g16_shadow_env != nullptr &&
                std::strcmp(q8g16_shadow_env, "1") == 0;

            if (q8g16_shadow) {
                const int64_t sample_rows = nrows < 16 ? nrows : 16;

                double sum_sq_error = 0.0;
                double sum_sq_ref = 0.0;
                double sum_sq_out = 0.0;
                double sum_ref_out = 0.0;
                double max_abs_error = 0.0;

                uint64_t compared_values = 0;
                uint64_t nonfinite_ref = 0;
                uint64_t nonfinite_q8g16 = 0;

                for (int64_t i = 0; i < sample_rows; ++i) {
                    const int64_t r = sample_rows <= 1
                        ? 0
                        : (i * (nrows - 1)) / (sample_rows - 1);

                    const uint8_t* wr =
                        W.data + static_cast<size_t>(row0 + r) * W.row_bytes;

                    for (int t = 0; t < nb; ++t) {
                        float ref_acc = 0.0f;
                        float tmp[256];

                        // Match the original FP32 fallback's block order,
                        // dequantizer, dot function and FP32 accumulation.
                        for (int64_t b = 0; b < nblk; ++b) {
                            const uint8_t* wb =
                                wr + static_cast<size_t>(b) * ti.bytes;

                            ti.dq(wb, tmp);

                            const float* xb =
                                x + static_cast<size_t>(t) * ldx +
                                static_cast<size_t>(b) * 256;

                            ref_acc += dotf(tmp, xb, 256);
                        }

                        const float actual =
                            out[static_cast<size_t>(t) * ldo + r];

                        if (!std::isfinite(ref_acc)) {
                            ++nonfinite_ref;
                            continue;
                        }

                        if (!std::isfinite(actual)) {
                            ++nonfinite_q8g16;
                            continue;
                        }

                        const double ref = ref_acc;
                        const double got = actual;
                        const double err = got - ref;
                        const double abs_err = std::fabs(err);

                        if (abs_err > max_abs_error)
                            max_abs_error = abs_err;

                        sum_sq_error += err * err;
                        sum_sq_ref += ref * ref;
                        sum_sq_out += got * got;
                        sum_ref_out += ref * got;
                        ++compared_values;
                    }
                }

                const double rmse = compared_values
                    ? std::sqrt(sum_sq_error / compared_values)
                    : 0.0;

                const double nrmse = sum_sq_ref > 0.0
                    ? std::sqrt(sum_sq_error / sum_sq_ref)
                    : 0.0;

                const double cosine_denom =
                    std::sqrt(sum_sq_ref * sum_sq_out);

                const double cosine = cosine_denom > 0.0
                    ? sum_ref_out / cosine_denom
                    : 0.0;

                std::fprintf(
                    stderr,
                    "[Q8G16_SHADOW] rows=%lld sampled_rows=%lld "
                    "tokens=%d compared=%llu "
                    "max_abs=%.9g rmse=%.9g nrmse=%.9g "
                    "cosine=%.12g nonfinite_ref=%llu "
                    "nonfinite_q8g16=%llu\n",
                    static_cast<long long>(nrows),
                    static_cast<long long>(sample_rows),
                    nb,
                    static_cast<unsigned long long>(compared_values),
                    max_abs_error,
                    rmse,
                    nrmse,
                    cosine,
                    static_cast<unsigned long long>(nonfinite_ref),
                    static_cast<unsigned long long>(nonfinite_q8g16)
                );
            }

            if (q8g16_trace) {
                const uint64_t dot_calls =
                    static_cast<uint64_t>(nrows) *
                    static_cast<uint64_t>(nb) *
                    static_cast<uint64_t>(nblk);

                std::fprintf(
                    stderr,
                    "[Q8G16_TRACE] event=dispatch kernel=V2 "
                    "rows=%lld tokens=%d ne0=%lld blocks_per_row=%lld "
                    "activation_blocks=%zu dot_calls=%llu\n",
                    static_cast<long long>(nrows), nb,
                    static_cast<long long>(W.ne0),
                    static_cast<long long>(nblk),
                    q8g16_quantized_blocks,
                    static_cast<unsigned long long>(dot_calls)
                );
            }

            return;
        }

        // Reject the complete operation. Do not pass invalid inputs to Q8K.
        q8g16_rejected = true;

        if (q8g16_trace) {
            std::fprintf(
                stderr,
                "[Q8G16_TRACE] event=rejection "
                "rows=%lld tokens=%d ne0=%lld "
                "quantized_blocks=%zu total_activation_blocks=%zu "
                "fallback=FP32\n",
                static_cast<long long>(nrows), nb,
                static_cast<long long>(W.ne0),
                q8g16_quantized_blocks, q8g.size()
            );
        }
    }

    if (W.type == 14 && !q8g16_eligible && q8g16_trace) {
        std::fprintf(
            stderr,
            "[Q8G16_TRACE] event=ineligible "
            "rows=%lld tokens=%d ne0=%lld type=%d fallback=Q8K_OR_FP32\n",
            static_cast<long long>(nrows), nb,
            static_cast<long long>(W.ne0), W.type
        );
    }

    // Q6_K-only optimized CPU path. All other types and unsupported
    // dimensions/CPUs retain the original FP32 implementation below.
    if (!q8g16_rejected && q8k_eligible) {

        std::vector<q6vnni::Q8K> q8(
            static_cast<size_t>(nb) * static_cast<size_t>(nblk)
        );

        // Quantize each token/block once, before parallel row processing.
        for (int t = 0; t < nb; ++t) {
            for (int64_t b = 0; b < nblk; ++b) {
                const float* xb =
                    x + static_cast<size_t>(t) * ldx +
                    static_cast<size_t>(b) * 256;

                q6vnni::quantize_q8k(
                    xb,
                    q8[static_cast<size_t>(t) * nblk + b]
                );
            }
        }

        if (q8g16_trace) {
            std::fprintf(
                stderr,
                "[Q8_DISPATCH_TRACE] event=dispatch kernel=Q8K "
                "rows=%lld tokens=%d ne0=%lld blocks_per_row=%lld\\n",
                static_cast<long long>(nrows),
                nb,
                static_cast<long long>(W.ne0),
                static_cast<long long>(nblk)
            );
        }

        #pragma omp parallel
        {
            #pragma omp for schedule(dynamic, 16)
            for (int64_t r = 0; r < nrows; ++r) {
                const uint8_t* wr =
                    W.data + static_cast<size_t>(row0 + r) * W.row_bytes;

                float acc[MAXB];
                for (int t = 0; t < nb; ++t)
                    acc[t] = 0.0f;

                for (int64_t b = 0; b < nblk; ++b) {
                    const uint8_t* wb =
                        wr + static_cast<size_t>(b) * ti.bytes;

                    q6vnni::PredecodedQ6 decoded{};
                    q6vnni::decode_q6(wb, rd_f16(wb + 208), decoded);

                    for (int t = 0; t < nb; ++t) {
                        const q6vnni::Q8K& qb =
                            q8[static_cast<size_t>(t) * nblk + b];

                        acc[t] += q6vnni::dot_q6_hadd(decoded, qb);
                    }
                }

                for (int t = 0; t < nb; ++t)
                    out[static_cast<size_t>(t) * ldo + r] = acc[t];
            }
        }

        return;
    }

    // Original FP32 path: unchanged fallback for Q5_K and all other types.
    if (q8g16_trace) {
        std::fprintf(
            stderr,
            "[Q8_DISPATCH_TRACE] event=dispatch kernel=FP32 "
            "rows=%lld tokens=%d ne0=%lld type=%d\n",
            static_cast<long long>(nrows),
            nb,
            static_cast<long long>(W.ne0),
            W.type
        );
    }

    #pragma omp parallel
    {
        float tmp[256];

        #pragma omp for schedule(dynamic, 16)
        for (int64_t r = 0; r < nrows; r++) {
            const uint8_t* wr =
                W.data + static_cast<size_t>(row0 + r) * W.row_bytes;

            float acc[MAXB];
            for (int t = 0; t < nb; t++)
                acc[t] = 0.f;

            for (int64_t b = 0; b < nblk; b++) {
                ti.dq(wr + static_cast<size_t>(b) * ti.bytes, tmp);
                const float* xb = x + b * blck;

                for (int t = 0; t < nb; t++)
                    acc[t] += dotf(
                        tmp, xb + static_cast<size_t>(t) * ldx, blck
                    );
            }

            for (int t = 0; t < nb; t++)
                out[static_cast<size_t>(t) * ldo + r] = acc[t];
        }
    }
}

#ifdef EAHMMF_CUDA_COMPUTE
extern "C" bool eahmmf_cuda_matmul(const uint8_t*, int, int, int, size_t,
                                   const float*, int, int, float*, int);
extern "C" bool eahmmf_cuda_matmul_pair(const uint8_t*, int, int, int, size_t, const uint8_t*, int, int, int, size_t, const float*, int, int, float*, int, float*, int);
extern "C" bool eahmmf_cuda_moe_ffn(const uint8_t*, int, int, int, size_t, const uint8_t*, int, int, int, size_t, const uint8_t*, int, int, int, size_t, const float*, int, int, float*, int);
static int g_shadow = 0;
static int g_cuda_compute = 0;
static int g_cuda_resident_ffn = 0;
static int g_cuda_tensor_cores = 0;
static long g_sh_calls = 0, g_sh_bad = 0, g_sh_skip = 0;
static double g_sh_max_rel = 0;
struct ShadowReport {
    ~ShadowReport() {
        if (g_shadow)
            fprintf(stderr, "cuda-shadow: calls=%ld bad=%ld skipped=%ld max_rel_err=%.3g\n",
                    g_sh_calls, g_sh_bad, g_sh_skip, g_sh_max_rel);
    }
} g_shadow_report;

static void matmul_shadow(const WeightView& W, const float* x, int nb, int ldx, float* out, int ldo) {
    matmul(W, x, nb, ldx, out, ldo);  // CPU result stays authoritative
    if (W.type != 8 && W.type != 12 && W.type != 13 && W.type != 14) { g_sh_skip++; return; }
    uint8_t* dw = nullptr;
    if (cudaMalloc(&dw, W.bytes) != cudaSuccess) { g_sh_skip++; return; }
    std::vector<float> g((size_t)nb * ldo, 0.f);
    bool ok = cudaMemcpy(dw, W.data, W.bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
              eahmmf_cuda_matmul(dw, (int)W.type, (int)W.ne0, (int)W.ne1, W.row_bytes,
                                 x, nb, ldx, g.data(), ldo);
    cudaFree(dw);
    g_sh_calls++;
    if (!ok) { g_sh_bad++; return; }
    double maxerr = 0, maxout = 1e-3;
    for (int t = 0; t < nb; t++)
        for (int r = 0; r < (int)W.ne1; r++) {
            double c = out[(size_t)t * ldo + r], d = g[(size_t)t * ldo + r];
            maxerr = std::max(maxerr, std::fabs(c - d));
            maxout = std::max(maxout, std::fabs(c));
        }
    double rel = maxerr / maxout;
    g_sh_max_rel = std::max(g_sh_max_rel, rel);
    if (!(rel < 1e-4)) g_sh_bad++;
}
#endif

#ifdef EAHMMF_CUDA_COMPUTE
static long g_gpu_calls = 0, g_gpu_fallbacks = 0, g_pair_calls = 0;
static long g_resident_ffn_calls = 0;

// Optional complete CPU-vs-GPU FFN numerical validation.
static long g_resident_ffn_shadow_calls = 0;
static unsigned long long g_resident_ffn_shadow_values = 0;
static unsigned long long g_resident_ffn_shadow_nonfinite = 0;
static double g_resident_ffn_shadow_max_abs = 0.0;
static double g_resident_ffn_shadow_max_rel = 0.0;
static double g_resident_ffn_shadow_sse = 0.0;
static double g_resident_ffn_shadow_ref_sse = 0.0;
struct ComputeReport {
    ~ComputeReport() {
        if (g_cuda_compute)
            fprintf(stderr, "cuda-compute: gpu_calls=%ld cpu_fallbacks=%ld pair_calls=%ld resident_ffn_calls=%ld\n", g_gpu_calls, g_gpu_fallbacks, g_pair_calls, g_resident_ffn_calls);
    }
} g_compute_report;

struct ResidentFfnShadowReport {
    ~ResidentFfnShadowReport() {
        const char* enabled = getenv("EAHMMF_GPU_RESIDENT_FFN_SHADOW");
        if (!enabled || !*enabled || strcmp(enabled, "0") == 0)
            return;

        double rms = g_resident_ffn_shadow_ref_sse > 0.0
            ? sqrt(g_resident_ffn_shadow_sse /
                   g_resident_ffn_shadow_ref_sse)
            : 0.0;

        fprintf(stderr,
            "resident-ffn-shadow: samples=%ld values=%llu "
            "nonfinite=%llu max_abs=%.9g max_rel=%.9g normalized_rms=%.9g\\n",
            g_resident_ffn_shadow_calls,
            g_resident_ffn_shadow_values,
            g_resident_ffn_shadow_nonfinite,
            g_resident_ffn_shadow_max_abs,
            g_resident_ffn_shadow_max_rel,
            rms);
    }
} g_resident_ffn_shadow_report;

static bool gpu_matmul(const WeightView& W, const float* x, int nb, int ldx, float* out, int ldo) {
    if (!W.dev || !(W.type == 8 || W.type == 12 || W.type == 13 || W.type == 14)) return false;
    return eahmmf_cuda_matmul(W.dev, (int)W.type, (int)W.ne0, (int)W.ne1, W.row_bytes, x, nb, ldx, out, ldo);
}

static bool gpu_matmul_pair(
    const WeightView& G, const WeightView& U,
    const float* x, int nb, int ldx,
    float* gout, int ldg, float* uout, int ldu)
{
    auto supported = [](const WeightView& W) {
        return W.dev &&
            (W.type == 8 || W.type == 12 || W.type == 13 || W.type == 14);
    };
    if (!supported(G) || !supported(U)) return false;
    if (G.ne0 != U.ne0 || G.ne1 != U.ne1) return false;
    return eahmmf_cuda_matmul_pair(
        G.dev, (int)G.type, (int)G.ne0, (int)G.ne1, G.row_bytes,
        U.dev, (int)U.type, (int)U.ne0, (int)U.ne1, U.row_bytes,
        x, nb, ldx, gout, ldg, uout, ldu);
}


static bool gpu_moe_ffn(
    const WeightView& G, const WeightView& U, const WeightView& Down,
    const float* x, int nb, int ldx, float* out, int ldo)
{
    auto supported = [](const WeightView& W) {
        return W.dev &&
            (W.type == 8 || W.type == 12 || W.type == 13 || W.type == 14);
    };
    if (!supported(G) || !supported(U) || !supported(Down))
        return false;

    if (G.ne0 != U.ne0 || G.ne1 != U.ne1 ||
        Down.ne0 != G.ne1 || Down.ne1 != G.ne0 ||
        ldx < G.ne0 || ldo != Down.ne1)
        return false;

    return eahmmf_cuda_moe_ffn(
        G.dev, (int)G.type, (int)G.ne0, (int)G.ne1, G.row_bytes,
        U.dev, (int)U.type, (int)U.ne0, (int)U.ne1, U.row_bytes,
        Down.dev, (int)Down.type, (int)Down.ne0, (int)Down.ne1,
        Down.row_bytes, x, nb, ldx, out, ldo);
}


// Complete CPU reference for the GPU-resident expert FFN.
// WeightView.data refers to the CPU-readable GGUF mapping in the CUDA
// execution configuration; the ExpertLease must remain alive throughout.
static void resident_ffn_shadow_compare_cpu(
    const WeightView& G, const WeightView& U, const WeightView& Down,
    const float* x, int nb, int ldx,
    const float* gpu_out, int ldo)
{
    const char* enabled = getenv("EAHMMF_GPU_RESIDENT_FFN_SHADOW");
    if (!enabled || !*enabled || strcmp(enabled, "0") == 0)
        return;

    static long max_samples = []() {
        const char* p = getenv("EAHMMF_GPU_RESIDENT_FFN_SHADOW_MAX");
        if (!p) return 16L;
        long n = strtol(p, nullptr, 10);
        return n > 0 ? n : 16L;
    }();

    // One sample per distinct gate/up/down quantization combination.
    static bool seen[40][40][40] = {};

    if (G.type >= 40 || U.type >= 40 || Down.type >= 40)
        return;

    if (seen[G.type][U.type][Down.type])
        return;

    if (g_resident_ffn_shadow_calls >= max_samples)
        return;

    seen[G.type][U.type][Down.type] = true;

    const int FF = (int)G.ne1;
    const int D = (int)Down.ne1;

    std::vector<float> gate((size_t)nb * FF, 0.0f);
    std::vector<float> up((size_t)nb * FF, 0.0f);
    std::vector<float> reference((size_t)nb * D, 0.0f);

    // Independent CPU implementation of the complete FFN.
    matmul(G, x, nb, ldx, gate.data(), FF);
    matmul(U, x, nb, ldx, up.data(), FF);

    for (size_t i = 0; i < gate.size(); ++i) {
        const float gv = gate[i];
        gate[i] = gv / (1.0f + expf(-gv)) * up[i];
    }

    matmul(Down, gate.data(), nb, FF, reference.data(), D);

    double max_abs = 0.0;
    double max_ref = 1e-3;
    double sample_sse = 0.0;
    double sample_ref_sse = 0.0;
    unsigned long long finite_values = 0;
    unsigned long long nonfinite = 0;

    for (int t = 0; t < nb; ++t) {
        for (int r = 0; r < D; ++r) {
            const double a = reference[(size_t)t * D + r];
            const double b = gpu_out[(size_t)t * ldo + r];

            if (!std::isfinite(a) || !std::isfinite(b)) {
                ++nonfinite;
                continue;
            }

            const double diff = a - b;
            const double adiff = std::fabs(diff);

            max_abs = std::max(max_abs, adiff);
            max_ref = std::max(max_ref, std::fabs(a));
            sample_sse += diff * diff;
            sample_ref_sse += a * a;
            ++finite_values;
        }
    }

    const double max_rel = max_abs / max_ref;
    const double sample_rms = sample_ref_sse > 0.0
        ? sqrt(sample_sse / sample_ref_sse)
        : 0.0;

    ++g_resident_ffn_shadow_calls;
    g_resident_ffn_shadow_values += finite_values;
    g_resident_ffn_shadow_nonfinite += nonfinite;

    g_resident_ffn_shadow_max_abs =
        std::max(g_resident_ffn_shadow_max_abs, max_abs);
    g_resident_ffn_shadow_max_rel =
        std::max(g_resident_ffn_shadow_max_rel, max_rel);

    g_resident_ffn_shadow_sse += sample_sse;
    g_resident_ffn_shadow_ref_sse += sample_ref_sse;

    fprintf(stderr,
        "[resident-ffn-shadow] sample=%ld formats=G%u/U%u/D%u "
        "batch=%d max_abs=%.9g max_rel=%.9g normalized_rms=%.9g "
        "nonfinite=%llu\n",
        g_resident_ffn_shadow_calls,
        G.type, U.type, Down.type, nb,
        max_abs, max_rel, sample_rms, nonfinite);
}

// --cuda-compute + --cuda-shadow: the GPU result is in use; check it against the CPU kernel.
static void shadow_compare_cpu(const WeightView& W, const float* x, int nb, int ldx, const float* gpu_out, int ldo) {
    std::vector<float> c((size_t)nb * ldo, 0.f);
    matmul(W, x, nb, ldx, c.data(), ldo);
    double maxerr = 0, maxout = 1e-3;
    for (int t = 0; t < nb; t++)
        for (int r = 0; r < (int)W.ne1; r++) {
            double a = c[(size_t)t * ldo + r], b = gpu_out[(size_t)t * ldo + r];
            maxerr = std::max(maxerr, std::fabs(a - b));
            maxout = std::max(maxout, std::fabs(a));
        }
    double rel = maxerr / maxout;
    g_sh_calls++;
    g_sh_max_rel = std::max(g_sh_max_rel, rel);
    if (!(rel < 1e-4)) g_sh_bad++;
}
#endif

// Expert FFN matmuls go through here; with --cuda-shadow the GPU kernel is cross-checked against the CPU result.
static void expert_matmul(
    const WeightView& W, const float* x, int nb, int ldx,
    float* out, int ldo, int layer, int expert, const char* operation)
{
#ifdef EAHMMF_CUDA_COMPUTE
    if (g_cuda_compute) {
        if (gpu_matmul(W, x, nb, ldx, out, ldo)) {
            q8k_activation_capture().record(
                W.type, W.ne0, x, nb, ldx, operation, layer, expert, "cuda");
            g_gpu_calls++;
            if (g_shadow) shadow_compare_cpu(W, x, nb, ldx, out, ldo);
            return;
        }
        g_gpu_fallbacks++;
    }
    if (g_shadow) {
        q8k_activation_capture().record(
            W.type, W.ne0, x, nb, ldx, operation, layer, expert, "cpu");
        matmul_shadow(W, x, nb, ldx, out, ldo);
        return;
    }
#endif
    q8k_activation_capture().record(
        W.type, W.ne0, x, nb, ldx, operation, layer, expert, "cpu");
    matmul(W, x, nb, ldx, out, ldo);
}

static void rmsnorm(const float* x, const float* w, float* out, int n, float eps) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float sc = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * sc * w[i];
}

static std::vector<float> tensor_f32(const Tensor& t) {
    const TypeInfo& ti = g_types[t.type];
    std::vector<float> out((size_t)t.nelem());
    int64_t rows = t.ne[1] * t.ne[2] * t.ne[3], nblk = t.ne[0] / ti.blck;
    for (int64_t r = 0; r < rows; r++)
        for (int64_t b = 0; b < nblk; b++)
            ti.dq(t.data + r * t.row_bytes + b * ti.bytes, &out[(size_t)(r * t.ne[0] + b * ti.blck)]);
    return out;
}

// ============================================================================
// Tokenizer (GPT-2 byte-level BPE + Qwen2 pre-tokenizer)
// ============================================================================
static bool in_ranges(const uint32_t (*r)[2], size_t n, uint32_t c) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (c < r[mid][0]) hi = mid;
        else if (c > r[mid][1]) lo = mid + 1;
        else return true;
    }
    return false;
}
static bool is_letter(uint32_t c) {
    if (c < 128) return (c | 32) >= 'a' && (c | 32) <= 'z';
    return in_ranges(UNI_L, UNI_L_N, c);
}
static bool is_number(uint32_t c) {
    if (c < 128) return c >= '0' && c <= '9';
    return in_ranges(UNI_N, UNI_N_N, c);
}
static bool is_space(uint32_t c) {
    return (c >= 9 && c <= 13) || c == 0x20 || c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) ||
           c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}
static uint32_t utf8_next(const std::string& s, size_t i, int& len) {
    uint8_t c = (uint8_t)s[i];
    int n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
    if (n == 0 || i + n > s.size()) { len = 1; return 0xFFFD; }
    uint32_t cp = n == 1 ? c : c & (0xFF >> (n + 1));
    for (int k = 1; k < n; k++) {
        uint8_t cc = (uint8_t)s[i + k];
        if ((cc & 0xC0) != 0x80) { len = 1; return 0xFFFD; }
        cp = (cp << 6) | (cc & 0x3F);
    }
    len = n;
    return cp;
}
static std::string utf8_encode(uint32_t c) {
    std::string o;
    if (c < 0x80) o += (char)c;
    else if (c < 0x800) { o += (char)(0xC0 | (c >> 6)); o += (char)(0x80 | (c & 0x3F)); }
    else if (c < 0x10000) { o += (char)(0xE0 | (c >> 12)); o += (char)(0x80 | ((c >> 6) & 0x3F)); o += (char)(0x80 | (c & 0x3F)); }
    else { o += (char)(0xF0 | (c >> 18)); o += (char)(0x80 | ((c >> 12) & 0x3F)); o += (char)(0x80 | ((c >> 6) & 0x3F)); o += (char)(0x80 | (c & 0x3F)); }
    return o;
}

// Equivalent of the Qwen2 regex:
// (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
static void pretokenize(const std::string& s, std::vector<std::pair<size_t, size_t>>& out) {
    std::vector<uint32_t> cp;
    std::vector<size_t> off;
    for (size_t i = 0; i < s.size();) {
        int len;
        cp.push_back(utf8_next(s, i, len));
        off.push_back(i);
        i += len;
    }
    off.push_back(s.size());
    const size_t n = cp.size();
    auto L = [&](size_t k) { return k < n && is_letter(cp[k]); };
    auto N = [&](size_t k) { return k < n && is_number(cp[k]); };
    auto WS = [&](size_t k) { return k < n && is_space(cp[k]); };
    auto NL = [&](size_t k) { return k < n && (cp[k] == '\r' || cp[k] == '\n'); };
    auto SYM = [&](size_t k) { return k < n && !is_space(cp[k]) && !is_letter(cp[k]) && !is_number(cp[k]); };
    auto lower = [](uint32_t c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; };
    size_t i = 0;
    while (i < n) {
        size_t j = 0;
        if (cp[i] == '\'' && i + 1 < n) {  // alt 1
            uint32_t c1 = lower(cp[i + 1]);
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') j = i + 2;
            else if (i + 2 < n) {
                uint32_t c2 = lower(cp[i + 2]);
                if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) j = i + 3;
            }
        }
        if (!j) {  // alt 2
            size_t k = 0;
            if (L(i)) k = i;
            else if (!NL(i) && !N(i) && L(i + 1)) k = i + 1;
            if (k || L(i)) {
                while (L(k)) k++;
                j = k;
            }
        }
        if (!j && N(i)) j = i + 1;  // alt 3
        if (!j) {                   // alt 4
            size_t k = i;
            bool ok = false;
            if (cp[i] == ' ' && SYM(i + 1)) { k = i + 1; ok = true; }
            else if (SYM(i)) ok = true;
            if (ok) {
                while (SYM(k)) k++;
                while (NL(k)) k++;
                j = k;
            }
        }
        if (!j && WS(i)) {  // alts 5,6,7
            size_t e = i;
            while (WS(e)) e++;
            size_t ln = 0;
            for (size_t k = e; k > i; k--) if (NL(k - 1)) { ln = k; break; }
            if (ln) j = ln;
            else if (e == n) j = e;
            else if (e - i >= 2) j = e - 1;
            else j = e;
        }
        if (!j) j = i + 1;
        out.push_back({off[i], off[j]});
        i = j;
    }
}

struct Tokenizer {
    std::vector<std::string> vocab;
    std::unordered_map<std::string, int> tok2id;
    std::unordered_map<uint64_t, std::pair<int, int>> merges;  // (a<<32|b) -> (rank, merged id)
    std::vector<std::pair<std::string, int>> specials;
    std::vector<std::string> piece;  // decoded bytes per id
    std::string byte2str[256];
    int eos = -1, im_start = -1, im_end = -1, endoftext = -1, nl = -1;

    void load(const GGUF& g) {
        const KV* toks = g.find("tokenizer.ggml.tokens");
        if (!toks || !toks->is_arr) die("no tokenizer.ggml.tokens in GGUF");
        std::string model = g.gets("tokenizer.ggml.model");
        if (model != "gpt2") fprintf(stderr, "warning: tokenizer model '%s' (expected gpt2)\n", model.c_str());
        std::string pre = g.gets("tokenizer.ggml.pre");
        if (pre != "qwen2" && pre != "qwen35" && !pre.empty()) fprintf(stderr, "warning: tokenizer pre-type '%s' (using qwen2 splitter)\n", pre.c_str());
        vocab = toks->as;
        const KV* types = g.find("tokenizer.ggml.token_type");
        std::unordered_map<uint32_t, int> cp2byte;
        {  // GPT-2 bytes_to_unicode
            std::vector<int> bs;
            for (int b = 33; b <= 126; b++) bs.push_back(b);
            for (int b = 161; b <= 172; b++) bs.push_back(b);
            for (int b = 174; b <= 255; b++) bs.push_back(b);
            bool in[256] = {false};
            for (int b : bs) in[b] = true;
            int n = 0;
            for (int b = 0; b < 256; b++) {
                uint32_t c = in[b] ? (uint32_t)b : (uint32_t)(256 + n++);
                byte2str[b] = utf8_encode(c);
                cp2byte[c] = b;
            }
        }
        piece.resize(vocab.size());
        for (size_t i = 0; i < vocab.size(); i++) {
            tok2id.emplace(vocab[i], (int)i);
            int ty = (types && i < types->ai.size()) ? (int)types->ai[i] : 1;
            if (ty == 3 || ty == 4) specials.push_back({vocab[i], (int)i});
            std::string out;
            bool ok = true;
            for (size_t k = 0; k < vocab[i].size();) {
                int len;
                uint32_t c = utf8_next(vocab[i], k, len);
                auto it = cp2byte.find(c);
                if (it == cp2byte.end()) { ok = false; break; }
                out += (char)it->second;
                k += len;
            }
            piece[i] = ok ? out : vocab[i];
        }
        std::sort(specials.begin(), specials.end(), [](auto& a, auto& b) { return a.first.size() > b.first.size(); });
        const KV* mg = g.find("tokenizer.ggml.merges");
        if (mg) {
            for (size_t r = 0; r < mg->as.size(); r++) {
                const std::string& m = mg->as[r];
                size_t sp = m.find(' ', 1);
                if (sp == std::string::npos) continue;
                auto a = tok2id.find(m.substr(0, sp)), b = tok2id.find(m.substr(sp + 1));
                auto c = tok2id.find(m.substr(0, sp) + m.substr(sp + 1));
                if (a == tok2id.end() || b == tok2id.end() || c == tok2id.end()) continue;
                merges.emplace(((uint64_t)a->second << 32) | (uint32_t)b->second, std::make_pair((int)r, c->second));
            }
        }
        auto lookup = [&](const char* s) { auto it = tok2id.find(s); return it == tok2id.end() ? -1 : it->second; };
        im_start = lookup("<|im_start|>");
        im_end = lookup("<|im_end|>");
        endoftext = lookup("<|endoftext|>");
        nl = lookup(byte2str[10].c_str());
        eos = (int)g.geti("tokenizer.ggml.eos_token_id", -1);
        for (int b = 0; b < 256; b++)
            if (!tok2id.count(byte2str[b])) die("vocab lacks byte token for 0x%02x", b);
    }

    void bpe(const std::string& s, size_t a, size_t b, std::vector<int>& out) const {
        std::vector<int> sy;
        for (size_t i = a; i < b; i++) sy.push_back(tok2id.at(byte2str[(uint8_t)s[i]]));
        while (sy.size() > 1) {
            int best = INT32_MAX;
            size_t bi = 0;
            int bid = 0;
            for (size_t i = 0; i + 1 < sy.size(); i++) {
                auto it = merges.find(((uint64_t)sy[i] << 32) | (uint32_t)sy[i + 1]);
                if (it != merges.end() && it->second.first < best) { best = it->second.first; bi = i; bid = it->second.second; }
            }
            if (best == INT32_MAX) break;
            sy[bi] = bid;
            sy.erase(sy.begin() + bi + 1);
        }
        out.insert(out.end(), sy.begin(), sy.end());
    }
    void encode_plain(const std::string& s, std::vector<int>& out) const {
        std::vector<std::pair<size_t, size_t>> parts;
        pretokenize(s, parts);
        for (auto& p : parts) bpe(s, p.first, p.second, out);
    }
    std::vector<int> encode(const std::string& s, bool parse_special) const {
        std::vector<int> out;
        if (!parse_special) { encode_plain(s, out); return out; }
        size_t pos = 0;
        while (pos < s.size()) {
            size_t best = std::string::npos;
            int bid = -1;
            size_t blen = 0;
            for (auto& sp : specials) {
                size_t f = s.find(sp.first, pos);
                if (f != std::string::npos && (f < best || (f == best && sp.first.size() > blen))) { best = f; bid = sp.second; blen = sp.first.size(); }
            }
            if (best == std::string::npos) { encode_plain(s.substr(pos), out); break; }
            if (best > pos) encode_plain(s.substr(pos, best - pos), out);
            out.push_back(bid);
            pos = best + blen;
        }
        return out;
    }
};

// ============================================================================
// Engine
// ============================================================================
struct Layer {
    std::vector<float> attn_norm, q_norm, k_norm, ffn_norm;
    WeightView wq, wk, wv, wo, router;
    const Tensor *gate, *up, *down;  // stacked expert tensors (source of ExpertDescriptors)
};

struct Engine {
    GGUF g;
    int n_layer, D, n_head, n_kv, hd, QD, KD, n_expert, n_used, FF, n_vocab, n_ctx;
    float eps, rope_base;
    const Tensor *tok_embd, *output;
    WeightView out_v;
    std::vector<float> out_norm, inv_freq;
    std::vector<Layer> layers;
    float *kc = nullptr, *vc = nullptr;
    // activations
    std::vector<float> x, xn, q, k, v, att, tmp, rl, gbuf, ubuf, ybuf, xs, rcos, rsin;
    // EAHMMF expert residency layer (dev must outlive store)
    std::unique_ptr<eahmmf::DeviceBackend> dev;
    std::unique_ptr<eahmmf::ExpertStore> store;
    etf::TelemetryCollector telemetry{2};
    eahmmf::AdaptiveOptimizer optimizer;
    uint64_t active_prompt_id = 0;
    bool etf_collection_enabled = true;
    // routing telemetry (separate from physical residency telemetry)
    bool trace_routing = false;
    std::vector<int> trace_ids;
    std::vector<float> trace_w;

    ~Engine() { free(kc); free(vc); }

    // Reconfiguration requires external synchronization with inference.
    // Outstanding expert pins are rejected before either owner is replaced.
    bool init_experts(const eahmmf::Config& cfg, std::unique_ptr<eahmmf::DeviceBackend> d = nullptr) {
        if (store && store->stats().pinned_now != 0) {
            fprintf(stderr,
                    "eahmmf: cannot reinitialize expert store with %llu outstanding pin(s)\n",
                    static_cast<unsigned long long>(store->stats().pinned_now));
            return false;
        }

        store.reset();
        dev = d ? std::move(d) : std::unique_ptr<eahmmf::DeviceBackend>(new eahmmf::SimDevice());
        std::vector<std::array<eahmmf::TensorRef, 3>> refs(n_layer);
        for (int l = 0; l < n_layer; l++) {
            const Tensor* ts[3] = {layers[l].gate, layers[l].up, layers[l].down};
            for (int c = 0; c < 3; c++) {
                eahmmf::TensorRef& r = refs[l][c];
                r.file_offset = ts[c]->file_off;
                r.type = ts[c]->type;
                r.ne0 = ts[c]->ne[0];
                r.ne1 = ts[c]->ne[1];
                r.ne2 = ts[c]->ne[2];
                r.row_bytes = ts[c]->row_bytes;
            }
        }

        static bool printed_expert_types = false;
        if (!printed_expert_types) {
            for (int l = 0; l < n_layer; ++l) {
                const Tensor* ts[3] = {
                    layers[l].gate, layers[l].up, layers[l].down
                };

                for (int c = 0; c < 3; ++c) {
                    const TypeInfo& ti = g_types[ts[c]->type];

                    fprintf(stderr,
                            "[expert-format] layer=%d component=%s "
                            "type=%s id=%u ne0=%lld ne1=%lld row_bytes=%zu\n",
                            l,
                            c == 0 ? "gate" : c == 1 ? "up" : "down",
                            ti.name,
                            ts[c]->type,
                            (long long)ts[c]->ne[0],
                            (long long)ts[c]->ne[1],
                            ts[c]->row_bytes);
                }
            }
            printed_expert_types = true;
        }


        // TEMPORARY: diagnose invalid expert tensor references.
        for (int l = 0; l < n_layer; ++l) {
            const Tensor* ts[3] = {
                layers[l].gate, layers[l].up, layers[l].down
            };
            const char* names[3] = {"gate", "up", "down"};

            for (int c = 0; c < 3; ++c) {
                const Tensor& t = *ts[c];
                const uint64_t expert_bytes =
                    static_cast<uint64_t>(t.ne[1]) * t.row_bytes;
                const uint64_t end =
                    t.file_off + static_cast<uint64_t>(n_expert) * expert_bytes;

                if (t.ne[2] < n_expert || t.file_off > g.size ||
                    end > g.size) {
                    fprintf(stderr,
                        "[descriptor-invalid] layer=%d component=%s name=%s "
                        "nd=%d ne=[%lld,%lld,%lld,%lld] "
                        "file_off=%llu row_bytes=%zu expert_bytes=%llu "
                        "all_experts_end=%llu file_size=%zu\n",
                        l, names[c], t.name.c_str(), t.nd,
                        (long long)t.ne[0], (long long)t.ne[1],
                        (long long)t.ne[2], (long long)t.ne[3],
                        (unsigned long long)t.file_off, t.row_bytes,
                        (unsigned long long)expert_bytes,
                        (unsigned long long)end, g.size);
                }
            }
        }

        store.reset(new eahmmf::ExpertStore(g.map, g.size, n_layer, n_expert, refs, cfg, dev.get()));
        optimizer.prime(*store, n_layer, n_expert);
        return true;
    }

    void load(const char* path, int ctx, bool populate) {
        g.load(path, populate);
        std::string arch = g.gets("general.architecture");
        if (arch != "qwen3moe") die("unsupported architecture '%s' (need qwen3moe)", arch.c_str());
        const std::string a = "qwen3moe.";
        n_layer = (int)g.geti(a + "block_count", 0, true);
        D = (int)g.geti(a + "embedding_length", 0, true);
        n_head = (int)g.geti(a + "attention.head_count", 0, true);
        n_kv = (int)g.geti(a + "attention.head_count_kv", 0, true);
        hd = (int)g.geti(a + "attention.key_length", D / n_head);
        n_expert = (int)g.geti(a + "expert_count", 0, true);
        n_used = (int)g.geti(a + "expert_used_count", 0, true);
        FF = (int)g.geti(a + "expert_feed_forward_length", 0, true);
        eps = (float)g.getf(a + "attention.layer_norm_rms_epsilon", 1e-6);
        rope_base = (float)g.getf(a + "rope.freq_base", 1e6);
        int n_ctx_train = (int)g.geti(a + "context_length", 4096);
        if (g.find(a + "rope.scaling.type") && g.gets(a + "rope.scaling.type") != "none")
            fprintf(stderr, "warning: rope scaling '%s' present in GGUF but not implemented\n", g.gets(a + "rope.scaling.type").c_str());
        QD = n_head * hd;
        KD = n_kv * hd;
        n_ctx = std::min(ctx, n_ctx_train);
        if (hd % 2 || n_head % n_kv) die("bad head config");
        if (D % 32 || FF % 32 || QD % 32) die("dimensions must be multiples of 32");

        tok_embd = g.tensor("token_embd.weight");
        output = g.tensor("output.weight", false);
        if (!output) output = tok_embd;  // tied embeddings
        out_v = view_of(*output);
        n_vocab = (int)output->ne[1];
        out_norm = tensor_f32(*g.tensor("output_norm.weight"));
        layers.resize(n_layer);
        for (int l = 0; l < n_layer; l++) {
            char b[96];
            auto T = [&](const char* s) { snprintf(b, sizeof b, "blk.%d.%s", l, s); return g.tensor(b); };
            Layer& L = layers[l];
            L.attn_norm = tensor_f32(*T("attn_norm.weight"));
            L.q_norm = tensor_f32(*T("attn_q_norm.weight"));
            L.k_norm = tensor_f32(*T("attn_k_norm.weight"));
            L.ffn_norm = tensor_f32(*T("ffn_norm.weight"));
            L.wq = view_of(*T("attn_q.weight"));
            L.wk = view_of(*T("attn_k.weight"));
            L.wv = view_of(*T("attn_v.weight"));
            L.wo = view_of(*T("attn_output.weight"));
            L.router = view_of(*T("ffn_gate_inp.weight"));
            L.gate = T("ffn_gate_exps.weight");
            L.up = T("ffn_up_exps.weight");
            L.down = T("ffn_down_exps.weight");
            if (L.gate->ne[2] != n_expert || L.gate->ne[1] != FF || L.down->ne[1] != D) die("unexpected expert tensor shape in layer %d", l);
        }
        inv_freq.resize(hd / 2);
        for (int i = 0; i < hd / 2; i++) inv_freq[i] = (float)pow((double)rope_base, -2.0 * i / hd);
        size_t kvn = (size_t)n_layer * n_ctx * KD;
        kc = (float*)calloc(kvn, sizeof(float));
        vc = (float*)calloc(kvn, sizeof(float));
        if (!kc || !vc) die("cannot allocate KV cache (%.2f GiB)", 2.0 * kvn * 4 / 1073741824.0);
        x.resize(MAXB * D); xn.resize(MAXB * D); q.resize(MAXB * QD); k.resize(MAXB * KD); v.resize(MAXB * KD);
        att.resize(MAXB * QD); tmp.resize(MAXB * D); rl.resize(MAXB * n_expert);
        gbuf.resize(MAXB * FF); ubuf.resize(MAXB * FF); ybuf.resize(MAXB * D); xs.resize(MAXB * D);
        rcos.resize(MAXB * hd / 2); rsin.resize(MAXB * hd / 2);
 #ifdef EAHMMF_TESTING
       if (!init_experts(eahmmf::Config()))
           die("cannot initialize default expert store for tests");
#else
       // ExpertStore is initialized once in main() with the final runtime configuration.
#endif
        fprintf(stderr, "model: %d layers, d=%d, heads=%d/%d (hd=%d), experts=%d (top-%d, ff=%d), vocab=%d, ctx=%d (train %d), KV=%.2f GiB\n",
                n_layer, D, n_head, n_kv, hd, n_expert, n_used, FF, n_vocab, n_ctx, n_ctx_train, 2.0 * kvn * 4 / 1073741824.0);
    }

    void embed_row(int id, float* out) const {
        if (id < 0 || id >= tok_embd->ne[1]) die("token id %d out of range", id);
        const TypeInfo& ti = g_types[tok_embd->type];
        const uint8_t* wr = tok_embd->data + (size_t)id * tok_embd->row_bytes;
        for (int64_t b = 0; b < tok_embd->ne[0] / ti.blck; b++) ti.dq(wr + b * ti.bytes, out + b * ti.blck);
    }

    // Evaluate nb<=MAXB tokens at positions p0..p0+nb-1. If logits != null, writes logits of the last token.
    void forward(const int* toks, int nb, int p0, float* logits) {
        if (nb < 1 || nb > MAXB) die("bad batch size");
        if (p0 + nb > n_ctx) die("context overflow (%d > %d)", p0 + nb, n_ctx);
        const int half = hd / 2, grp = n_head / n_kv;
        const float scale = 1.0f / sqrtf((float)hd);
        for (int t = 0; t < nb; t++) embed_row(toks[t], &x[(size_t)t * D]);
        for (int t = 0; t < nb; t++)
            for (int i = 0; i < half; i++) {
                double ang = (double)(p0 + t) * inv_freq[i];
                rcos[t * half + i] = (float)cos(ang);
                rsin[t * half + i] = (float)sin(ang);
            }
        for (int l = 0; l < n_layer; l++) {
            Layer& L = layers[l];
            // ---- attention ----
            for (int t = 0; t < nb; t++) rmsnorm(&x[(size_t)t * D], L.attn_norm.data(), &xn[(size_t)t * D], D, eps);
            q8k_activation_capture().record(L.wq.type, L.wq.ne0, xn.data(), nb, D, "attn_q", l, -1, "cpu");
            matmul(L.wq, xn.data(), nb, D, q.data(), QD);
            q8k_activation_capture().record(L.wk.type, L.wk.ne0, xn.data(), nb, D, "attn_k", l, -1, "cpu");
            matmul(L.wk, xn.data(), nb, D, k.data(), KD);
            q8k_activation_capture().record(L.wv.type, L.wv.ne0, xn.data(), nb, D, "attn_v", l, -1, "cpu");
            matmul(L.wv, xn.data(), nb, D, v.data(), KD);
            for (int t = 0; t < nb; t++) {
                for (int h = 0; h < n_head; h++) {
                    float* p = &q[(size_t)t * QD + h * hd];
                    rmsnorm(p, L.q_norm.data(), p, hd, eps);
                    for (int i = 0; i < half; i++) {
                        float a = p[i], b = p[i + half], c = rcos[t * half + i], s = rsin[t * half + i];
                        p[i] = a * c - b * s;
                        p[i + half] = a * s + b * c;
                    }
                }
                for (int h = 0; h < n_kv; h++) {
                    float* p = &k[(size_t)t * KD + h * hd];
                    rmsnorm(p, L.k_norm.data(), p, hd, eps);
                    for (int i = 0; i < half; i++) {
                        float a = p[i], b = p[i + half], c = rcos[t * half + i], s = rsin[t * half + i];
                        p[i] = a * c - b * s;
                        p[i + half] = a * s + b * c;
                    }
                }
                size_t o = ((size_t)l * n_ctx + p0 + t) * KD;
                memcpy(kc + o, &k[(size_t)t * KD], KD * sizeof(float));
                memcpy(vc + o, &v[(size_t)t * KD], KD * sizeof(float));
            }
            const float* lk = kc + (size_t)l * n_ctx * KD;
            const float* lv = vc + (size_t)l * n_ctx * KD;
#pragma omp parallel
            {
                std::vector<float> sc(p0 + nb);
#pragma omp for collapse(2) schedule(dynamic)
                for (int t = 0; t < nb; t++)
                    for (int h = 0; h < n_head; h++) {
                        const int np = p0 + t + 1, kvh = h / grp;
                        const float* qh = &q[(size_t)t * QD + h * hd];
                        float mx = -INFINITY;
                        for (int j = 0; j < np; j++) {
                            float s = dotf(qh, lk + (size_t)j * KD + kvh * hd, hd) * scale;
                            sc[j] = s;
                            mx = std::max(mx, s);
                        }
                        float sum = 0;
                        for (int j = 0; j < np; j++) { sc[j] = expf(sc[j] - mx); sum += sc[j]; }
                        float inv = 1.0f / sum;
                        float* o = &att[(size_t)t * QD + h * hd];
                        for (int d = 0; d < hd; d++) o[d] = 0;
                        for (int j = 0; j < np; j++) {
                            float w = sc[j] * inv;
                            const float* vj = lv + (size_t)j * KD + kvh * hd;
#pragma omp simd
                            for (int d = 0; d < hd; d++) o[d] += w * vj[d];
                        }
                    }
            }
            q8k_activation_capture().record(L.wo.type, L.wo.ne0, att.data(), nb, QD, "attn_output", l, -1, "cpu");
            matmul(L.wo, att.data(), nb, QD, tmp.data(), D);
            for (int i = 0; i < nb * D; i++) x[i] += tmp[i];

            // ---- MoE FFN ----
            for (int t = 0; t < nb; t++) rmsnorm(&x[(size_t)t * D], L.ffn_norm.data(), &xn[(size_t)t * D], D, eps);
            q8k_activation_capture().record(L.router.type, L.router.ne0, xn.data(), nb, D, "router", l, -1, "cpu");
            matmul(L.router, xn.data(), nb, D, rl.data(), n_expert);
            std::vector<std::vector<std::pair<int, float>>> by_exp(n_expert);
            std::vector<int> idx(n_expert);
            for (int t = 0; t < nb; t++) {
                float* r = &rl[(size_t)t * n_expert];
                float mx = *std::max_element(r, r + n_expert), sum = 0;
                for (int e = 0; e < n_expert; e++) { r[e] = expf(r[e] - mx); sum += r[e]; }
                std::iota(idx.begin(), idx.end(), 0);
                std::partial_sort(idx.begin(), idx.begin() + n_used, idx.end(), [&](int a, int b) { return r[a] > r[b] || (r[a] == r[b] && a < b); });
                float tot = 0;
                for (int i = 0; i < n_used; i++) tot += r[idx[i]];
                for (int i = 0; i < n_used; i++) {
                    const float route_weight = r[idx[i]] / tot;
                   by_exp[idx[i]].push_back({t, route_weight});
                   if (etf_collection_enabled)
                       telemetry.record_activation(active_prompt_id, l, p0 + t, idx[i], route_weight);
                    if (trace_routing) { trace_ids.push_back(idx[i]); trace_w.push_back(r[idx[i]] / tot); }
                }
            }
            memset(tmp.data(), 0, sizeof(float) * nb * D);
            for (int e = 0; e < n_expert; e++) {
                auto& lst = by_exp[e];
                if (lst.empty()) continue;
                int m = (int)lst.size();
                for (int i = 0; i < m; i++) memcpy(&xs[(size_t)i * D], &xn[(size_t)lst[i].first * D], D * sizeof(float));
                // Lease owns the pin through all weight-consuming operations.
                eahmmf::ExpertLease lease = store->acquire_lease(l, e);
                if (!lease) die("expert (layer %d, expert %d) is unavailable (see --expert-log)", l, e);
                const eahmmf::ExpertViews& ev = lease.views();
                bool resident_ffn = false;
#ifdef EAHMMF_CUDA_COMPUTE
                // Keep --cuda-shadow on the established per-projection path.
                if (g_cuda_compute &&
                    (g_cuda_resident_ffn ||
                     getenv("EAHMMF_GPU_RESIDENT_FFN") != nullptr) &&
                    !g_shadow) {
                    resident_ffn = gpu_moe_ffn(
                        ev.gate, ev.up, ev.down,
                        xs.data(), m, D, ybuf.data(), D);
                    if (resident_ffn) {
                        // Validation only; the fused GPU output remains authoritative.
                        // The expert lease is still held during this comparison.
                        resident_ffn_shadow_compare_cpu(
                            ev.gate, ev.up, ev.down,
                            xs.data(), m, D, ybuf.data(), D);

                        q8k_activation_capture().record(
                            ev.gate.type, ev.gate.ne0, xs.data(), m, D,
                            "moe_gate", l, e, "gpu_resident_cuda");
                        q8k_activation_capture().record(
                            ev.up.type, ev.up.ne0, xs.data(), m, D,
                            "moe_up", l, e, "gpu_resident_cuda");
                        // The down activation remains device-resident; do not
                        // fabricate a host-side activation-capture record.
                        g_resident_ffn_calls++;
                        g_gpu_calls += 3;
                    } else {
                        g_gpu_fallbacks++;
                    }
                }
#endif
                if (!resident_ffn) {
                    bool paired = false;
#ifdef EAHMMF_CUDA_COMPUTE
                    if (g_cuda_compute && getenv("EAHMMF_DISABLE_PAIR") == nullptr) {
                        paired = gpu_matmul_pair(
                            ev.gate, ev.up, xs.data(), m, D,
                            gbuf.data(), FF, ubuf.data(), FF);
                        if (paired) {
                            q8k_activation_capture().record(
                                ev.gate.type, ev.gate.ne0, xs.data(), m, D,
                                "moe_gate", l, e, "paired_cuda");
                            q8k_activation_capture().record(
                                ev.up.type, ev.up.ne0, xs.data(), m, D,
                                "moe_up", l, e, "paired_cuda");
                            g_pair_calls++;
                            g_gpu_calls += 2;
                            if (g_shadow) {
                                shadow_compare_cpu(
                                    ev.gate, xs.data(), m, D, gbuf.data(), FF);
                                shadow_compare_cpu(
                                    ev.up, xs.data(), m, D, ubuf.data(), FF);
                            }
                        }
                    }
#endif
                    if (!paired) {
                        expert_matmul(ev.gate, xs.data(), m, D, gbuf.data(), FF, l, e, "moe_gate");
                        expert_matmul(ev.up, xs.data(), m, D, ubuf.data(), FF, l, e, "moe_up");
                    }
                    for (int i = 0; i < m * FF; i++) {
                        float gv = gbuf[i];
                        gbuf[i] = gv / (1.0f + expf(-gv)) * ubuf[i];
                    }
                    expert_matmul(ev.down, gbuf.data(), m, FF, ybuf.data(), D, l, e, "moe_down");
                }
                lease.reset();  // Preserve the previous unpin point.
                for (int i = 0; i < m; i++) {
                    float w = lst[i].second;
                    float* dst = &tmp[(size_t)lst[i].first * D];
                    const float* src = &ybuf[(size_t)i * D];
                    for (int d = 0; d < D; d++) dst[d] += w * src[d];
                }
            }
            for (int i = 0; i < nb * D; i++) x[i] += tmp[i];
        }
        if (logits) {
            rmsnorm(&x[(size_t)(nb - 1) * D], out_norm.data(), xn.data(), D, eps);
            q8k_activation_capture().record(out_v.type, out_v.ne0, xn.data(), 1, D, "output_projection", n_layer, -1, "cpu");
            matmul(out_v, xn.data(), 1, D, logits, n_vocab);
        }
    }

    // Evaluate an arbitrary number of tokens starting at position p0 (chunked).
    void eval(const std::vector<int>& toks, int p0, int n_batch, float* logits) {
        for (size_t i = 0; i < toks.size(); i += n_batch) {
            int nb = (int)std::min<size_t>(n_batch, toks.size() - i);
            bool last = i + nb == toks.size();
            forward(&toks[i], nb, p0 + (int)i, last ? logits : nullptr);
        }
    }
};

// ============================================================================
// Sampler
// ============================================================================
struct Sampler {
    float temp = 0.7f, top_p = 0.8f, min_p = 0.0f, rep = 1.05f;
    int top_k = 20, rep_n = 64;
    std::mt19937_64 rng;

    int sample(float* lg, int n, const std::vector<int>& hist) {
        if (rep != 1.0f) {
            size_t from = hist.size() > (size_t)rep_n ? hist.size() - rep_n : 0;
            for (size_t i = from; i < hist.size(); i++) {
                int t = hist[i];
                if (t >= 0 && t < n) lg[t] = lg[t] > 0 ? lg[t] / rep : lg[t] * rep;
            }
        }
        if (temp <= 0) return (int)(std::max_element(lg, lg + n) - lg);
        std::vector<std::pair<float, int>> c(n);
        for (int i = 0; i < n; i++) c[i] = {lg[i], i};
        size_t keep = (top_k > 0 && top_k < n) ? (size_t)top_k : (size_t)n;
        std::partial_sort(c.begin(), c.begin() + keep, c.end(), [](auto& a, auto& b) { return a.first > b.first; });
        c.resize(keep);
        float mx = c[0].first, sum = 0;
        std::vector<float> p(keep);
        for (size_t i = 0; i < keep; i++) { p[i] = expf((c[i].first - mx) / temp); sum += p[i]; }
        for (auto& v : p) v /= sum;
        size_t cut = keep;
        if (min_p > 0) {
            cut = 0;
            while (cut < keep && p[cut] >= min_p * p[0]) cut++;
        }
        if (top_p < 1.0f) {
            float cum = 0;
            size_t k = 0;
            while (k < cut) { cum += p[k++]; if (cum >= top_p) break; }
            cut = k;
        }
        float tot = 0;
        for (size_t i = 0; i < cut; i++) tot += p[i];
        float r = std::uniform_real_distribution<float>(0.f, tot)(rng), acc = 0;
        for (size_t i = 0; i < cut; i++) { acc += p[i]; if (r <= acc) return c[i].second; }
        return c[cut - 1].second;
    }
};

// ============================================================================
// CLI
// ============================================================================
[[maybe_unused]] static void usage() {
    fprintf(stderr,
            "usage: qwen3moe -m model.gguf [options]\n"
            "  -p TEXT          prompt (one-shot chat unless --raw); omit for interactive\n"
            "  -i               interactive chat\n"
            "  -s TEXT          system prompt\n"
            "  --workload-id ID workload identity for ETF (default unknown)\n"
            "  --language NAME  workload language (default unknown)\n"
            "  --framework NAME workload framework (default unknown)\n"
            "  --task-category NAME workload category (default unknown)\n"
            "  --workload-phase NAME workload phase (default unknown)\n"
            "  --etf-log FILE   append workload metadata and expert activations as JSONL\n"
            "  --no-etf-collection disable ETF collection and historical feedback\n"
            "  --raw            no chat template: prompt is tokenized as-is (special tokens parsed)\n"
            "  -n N             max new tokens per reply (default 1024)\n"
            "  -c N             context length (default 8192)\n"
            "  -t N             threads (default: all cores)\n"
            "  -b N             prefill batch (1..%d, default 32)\n"
            "  --temp F --top-k N --top-p F --min-p F --repeat-penalty F --seed N\n"
            "  --cuda-shadow    (CUDA=1 build) run each expert matmul on the GPU too and report max error vs CPU\n"
            "  --cuda-compute   (CUDA=1; needs --device cuda) run expert matmuls on VRAM-resident experts\n"
            "  --cuda-resident-ffn   keep supported MoE expert FFN intermediates on the GPU\n"
            "  --cuda-tensor-cores   opt in to experimental FP16 WMMA expert matmuls\n"
            "  --populate       pre-fault the mmap'd model\n"
            "  expert residency (EAHMMF):\n"
            "  --expert-tier ssd|ram|vram   tier experts are staged to before execution (default ssd = zero-copy mmap)\n"
            "  --ram-budget-mb N / --vram-budget-mb N   capacity of the managed RAM / VRAM expert pools (default unlimited)\n"
            "  --expert-policy lru|frequency   RAM admission policy (default lru)\n"
            "  --expert-aging-interval N   halve frequency counters every N acquires (default 4096; 0 disables)\n"
            "  --vram-reserve-mb N   keep N MiB of physical VRAM free for CUDA/runtime overhead (default 0)\n"
            "  --pwea-placement   enable bounded prediction-driven placement (requires finite --ram-budget-mb)\n"
            "  --pwea-prediction-only   use PWEA prediction without ETF history (requires --pwea-placement)\n"
            "  --pwea-shadow-diagnostics   compare PWEA-only and adaptive plans (diagnostic overhead; adaptive mode only)\n"
            "  --pwea-max-migrations N   maximum protected migrations per prompt (default 2)\n"
            "  --expert-verify  checksum every migration against the GGUF   --expert-strict  fail instead of falling back\n"
            "  --expert-log FILE  residency event log (acquire/migrate/pin)  --expert-stats  print residency summary\n"
           "  --expert-stats-per-prompt  append per-prompt EAHMMF snapshots and deltas as JSONL\n"
           "  --expert-stats-per-prompt-file FILE  JSONL output (default benchmarks/prompt_stats.jsonl)\n"
           "  --os-memory-stats  collect Linux/process/cgroup/model mmap memory snapshots\n"
           "  --os-memory-stats-file FILE  OS memory JSONL output (default benchmarks/os_memory/os_memory.jsonl)\n"
            "  --device sim|cuda  VRAM backend (sim = simulated device; cuda needs a build with CUDA=1)\n"
            "  --tokenize TEXT  debug: print token ids   --pretok TEXT  debug: print pre-tokenizer split\n"
            "  --logits-test 1,2,3  debug: print logits for token ids\n",
            MAXB);
    exit(1);
}

#ifndef QWEN3MOE_NO_MAIN

static uint64_t prompt_stats_delta(uint64_t after, uint64_t before) {
    return after >= before ? after - before : 0;
}

static void prompt_stats_array(
    FILE* f, const uint64_t* values, const uint64_t* before, size_t count
) {
    fprintf(f, "[");
    for (size_t i = 0; i < count; ++i) {
        if (i) fprintf(f, ",");
        const uint64_t value = before
            ? prompt_stats_delta(values[i], before[i]) : values[i];
        fprintf(f, "%llu", (unsigned long long)value);
    }
    fprintf(f, "]");
}

static void prompt_stats_counters(
    FILE* f, const eahmmf::Stats& s, const eahmmf::Stats* base
) {
    auto u64 = [&](uint64_t value, uint64_t previous = 0) {
        return base ? prompt_stats_delta(value, previous) : value;
    };

    fprintf(f, "{");
    fprintf(f, "\"acquires\":%llu,",
        (unsigned long long)u64(s.acquires, base ? base->acquires : 0));
    fprintf(f, "\"successful_acquires\":%llu,",
        (unsigned long long)u64(
            s.successful_acquires,
            base ? base->successful_acquires : 0));

    const uint64_t failed =
        s.acquires >= s.successful_acquires
            ? s.acquires - s.successful_acquires : 0;
    const uint64_t previous_failed = base &&
        base->acquires >= base->successful_acquires
            ? base->acquires - base->successful_acquires : 0;

    fprintf(f, "\"failed_acquires\":%llu,",
        (unsigned long long)(base
            ? prompt_stats_delta(failed, previous_failed) : failed));

    fprintf(f, "\"releases\":%llu,",
        (unsigned long long)u64(s.releases, base ? base->releases : 0));
    fprintf(f, "\"pins\":%llu,",
        (unsigned long long)u64(s.pins, base ? base->pins : 0));
    fprintf(f, "\"unpins\":%llu,",
        (unsigned long long)u64(s.unpins, base ? base->unpins : 0));

    fprintf(f, "\"served_by_tier_ssd_ram_vram\":");
    prompt_stats_array(f, s.served, base ? base->served : nullptr, 3);

    fprintf(f, ",\"hits_by_tier_ssd_ram_vram\":");
    prompt_stats_array(f, s.hit, base ? base->hit : nullptr, 3);

    fprintf(f, ",\"misses_by_tier_ssd_ram_vram\":");
    prompt_stats_array(f, s.miss, base ? base->miss : nullptr, 3);

    fprintf(f, ",\"migration_count_by_path\":");
    prompt_stats_array(f, s.mig_n, base ? base->mig_n : nullptr, 5);

    fprintf(f, ",\"migration_failures_by_path\":");
    prompt_stats_array(f, s.mig_fail, base ? base->mig_fail : nullptr, 5);

    fprintf(f, ",\"migration_bytes_by_path\":");
    prompt_stats_array(f, s.mig_bytes, base ? base->mig_bytes : nullptr, 5);

    fprintf(f, ",\"admission_rejects_by_path\":");
    prompt_stats_array(f, s.admission_reject,
        base ? base->admission_reject : nullptr, 5);

    fprintf(f, ",\"allocation_failures_by_path\":");
    prompt_stats_array(f, s.alloc_fail,
        base ? base->alloc_fail : nullptr, 5);

    fprintf(f, ",\"copy_failures_by_path\":");
    prompt_stats_array(f, s.copy_fail,
        base ? base->copy_fail : nullptr, 5);

    fprintf(f, ",\"migration_duration_us_by_path\":[");
    for (size_t i = 0; i < 5; ++i) {
        if (i) fprintf(f, ",");
        const double value = base
            ? s.mig_us[i] - base->mig_us[i] : s.mig_us[i];
        fprintf(f, "%.3f", value >= 0.0 ? value : 0.0);
    }
    fprintf(f, "]");

    fprintf(f, ",\"fallbacks\":%llu",
        (unsigned long long)u64(s.fallbacks,
            base ? base->fallbacks : 0));
    fprintf(f, ",\"refused_evictions\":%llu",
        (unsigned long long)u64(s.refused_evictions,
            base ? base->refused_evictions : 0));
    fprintf(f, ",\"vram_alloc_failures\":%llu",
        (unsigned long long)u64(s.vram_alloc_failures,
            base ? base->vram_alloc_failures : 0));
    fprintf(f, ",\"vram_alloc_retries\":%llu",
        (unsigned long long)u64(s.vram_alloc_retries,
            base ? base->vram_alloc_retries : 0));
    fprintf(f, ",\"vram_recovery_evictions\":%llu",
        (unsigned long long)u64(s.vram_recovery_evictions,
            base ? base->vram_recovery_evictions : 0));
    fprintf(f, ",\"vram_recovered_allocations\":%llu",
        (unsigned long long)u64(s.vram_recovered_allocations,
            base ? base->vram_recovered_allocations : 0));
    fprintf(f, ",\"vram_unrecovered_allocations\":%llu",
        (unsigned long long)u64(s.vram_unrecovered_allocations,
            base ? base->vram_unrecovered_allocations : 0));

    fprintf(f, ",\"acquire_duration_us\":%.3f",
        base ? std::max(0.0, s.acquire_us - base->acquire_us)
             : s.acquire_us);
    fprintf(f, "}");
}

static void prompt_stats_snapshot(
    FILE* f, const eahmmf::ResidencySnapshot& snap
) {
    fprintf(f, "{\"counters\":");
    prompt_stats_counters(f, snap.stats, nullptr);

    // Gauges are snapshots, not counter deltas.
    fprintf(f, ",\"gauges\":{");
    fprintf(f, "\"acquire_duration_max_us\":%.3f,",
        snap.stats.acquire_us_max);
    fprintf(f, "\"pinned_now\":%llu,",
        (unsigned long long)snap.stats.pinned_now);
    fprintf(f, "\"pinned_max\":%llu",
        (unsigned long long)snap.stats.pinned_max);
    fprintf(f, "}");

    const bool ram_unlimited =
        snap.ram_capacity == static_cast<size_t>(-1);
    const bool vram_unlimited =
        snap.vram_capacity == static_cast<size_t>(-1);

    fprintf(f, ",\"memory_bytes\":{");
    fprintf(f, "\"ram_used\":%zu,\"ram_capacity\":",
        snap.ram_used);
    if (ram_unlimited) fputs("null", f);
    else fprintf(f, "%zu", snap.ram_capacity);
    fprintf(f, ",\"ram_capacity_unlimited\":%s",
        ram_unlimited ? "true" : "false");

    fprintf(f, ",\"vram_used\":%zu,\"vram_capacity\":",
        snap.vram_used);
    if (vram_unlimited) fputs("null", f);
    else fprintf(f, "%zu", snap.vram_capacity);
    fprintf(f, ",\"vram_capacity_unlimited\":%s",
        vram_unlimited ? "true" : "false");
    fprintf(f, "}");

    fprintf(f, ",\"experts\":{");
    fprintf(f, "\"ssd\":%llu,\"ram\":%llu,",
        (unsigned long long)snap.ssd_experts,
        (unsigned long long)snap.ram_experts);
    fprintf(f, "\"vram\":%llu,\"transitional\":%llu",
        (unsigned long long)snap.vram_experts,
        (unsigned long long)snap.transitional_experts);
    fprintf(f, "}}");
}


static void prompt_stats_json_string(FILE* f, const std::string& value) {
    fputc('"', f);
    for (unsigned char c : value) {
        switch (c) {
            case '"':  fputs("\\\"", f); break;
            case '\\': fputs("\\\\", f); break;
            case '\b': fputs("\\b", f); break;
            case '\f': fputs("\\f", f); break;
            case '\n': fputs("\\n", f); break;
            case '\r': fputs("\\r", f); break;
            case '\t': fputs("\\t", f); break;
            default:
                if (c < 0x20) fprintf(f, "\\u%04x", c);
                else fputc(c, f);
        }
    }
    fputc('"', f);
}

static bool append_prompt_stats(
    const std::string& output_path,
    uint64_t prompt_id,
    bool completed,
    const etf::WorkloadMetadata& workload,
    double prediction_seconds,
    double placement_seconds,
    double inference_seconds,
    double total_seconds,
    const eahmmf::ResidencySnapshot& s0,
    const eahmmf::ResidencySnapshot& s1,
    const eahmmf::ResidencySnapshot& s2
) {
    FILE* f = nullptr;
    try {
        const std::filesystem::path output(output_path);
        if (!output.parent_path().empty())
            std::filesystem::create_directories(output.parent_path());

        f = fopen(output_path.c_str(), "a");
        if (!f) return false;

        fprintf(f,
            "{\"schema_version\":1,\"prompt_id\":%llu,"
            "\"completed\":%s,",
            (unsigned long long)prompt_id,
            completed ? "true" : "false");

        fputs("\"workload\":{\"schema_version\":", f);
        fprintf(f, "%u,\"workload_id\":", workload.schema_version);
        prompt_stats_json_string(f, workload.workload_id);
        fputs(",\"language\":", f);
        prompt_stats_json_string(f, workload.language);
        fputs(",\"framework\":", f);
        prompt_stats_json_string(f, workload.framework);
        fputs(",\"task_category\":", f);
        prompt_stats_json_string(f, workload.task_category);
        fputs(",\"workload_phase\":", f);
        prompt_stats_json_string(f, workload.workload_phase);
        fputs("},", f);


        fprintf(f,
            "\"tier_order\":[\"SSD\",\"RAM\",\"VRAM\"],"
            "\"migration_path_order\":["
            "\"SSD->RAM\",\"RAM->VRAM\",\"VRAM->RAM\","
            "\"RAM->SSD\",\"VRAM->SSD\"],");

        fprintf(f,
            "\"timing_seconds\":{"
            "\"prediction\":%.6f,\"placement\":%.6f,"
            "\"inference\":%.6f,\"total\":%.6f},",
            prediction_seconds, placement_seconds,
            inference_seconds, total_seconds);

        fprintf(f, "\"S0\":");
        prompt_stats_snapshot(f, s0);
        fprintf(f, ",\"S1\":");
        prompt_stats_snapshot(f, s1);
        fprintf(f, ",\"S2\":");
        prompt_stats_snapshot(f, s2);

        fprintf(f, ",\"deltas\":{\"S0_to_S1\":");
        prompt_stats_counters(f, s1.stats, &s0.stats);
        fprintf(f, ",\"S1_to_S2\":");
        prompt_stats_counters(f, s2.stats, &s1.stats);
        fprintf(f, ",\"S0_to_S2\":");
        prompt_stats_counters(f, s2.stats, &s0.stats);
        fprintf(f, "}}\n");

        const bool ok = fflush(f) == 0;
        const bool close_ok = fclose(f) == 0;
        f = nullptr;
        return ok && close_ok;
    } catch (...) {
        if (f) fclose(f);
        return false;
    }
}


int main(int argc, char** argv) {
    init_types();
    std::string model, prompt, sys = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
    std::string dbg_tok, dbg_pre, dbg_logits;
    etf::WorkloadMetadata workload_metadata;
    std::string etf_log_path;
    std::string prompt_stats_path = "benchmarks/prompt_stats.jsonl";
    bool prompt_stats_enabled = false;
    std::string os_memory_stats_path = "benchmarks/os_memory/os_memory.jsonl";
    bool os_memory_stats_enabled = false;
    bool interactive = false, raw = false, populate = false, have_prompt = false, have_dbg = false;
    bool etf_collection_enabled = true;
    int n_new = 1024, n_ctx = 8192, n_threads = 0, n_batch = 32;
    eahmmf::Config xcfg;
    bool xstats = false;
    bool pwea_placement_enabled = false;
    bool pwea_prediction_only = false;
    bool pwea_shadow_diagnostics = false;
    bool pwea_max_migrations_explicit = false;
    size_t pwea_max_migrations = 2;
    std::string device = "sim";
    Sampler sp;
    uint64_t seed = (uint64_t)time(nullptr);
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto nxt = [&]() -> const char* { if (i + 1 >= argc) usage(); return argv[++i]; };
        if (a == "-m") model = nxt();
        else if (a == "-p") { prompt = nxt(); have_prompt = true; }
        else if (a == "-i") interactive = true;
        else if (a == "-s") sys = nxt();
        else if (a == "--workload-id") workload_metadata.workload_id = nxt();
        else if (a == "--language") workload_metadata.language = nxt();
        else if (a == "--framework") workload_metadata.framework = nxt();
        else if (a == "--task-category") workload_metadata.task_category = nxt();
        else if (a == "--workload-phase") workload_metadata.workload_phase = nxt();
        else if (a == "--etf-log") etf_log_path = nxt();
        else if (a == "--no-etf-collection") etf_collection_enabled = false;
        else if (a == "--raw") raw = true;
        else if (a == "-n") n_new = atoi(nxt());
        else if (a == "-c") n_ctx = atoi(nxt());
        else if (a == "-t") n_threads = atoi(nxt());
        else if (a == "-b") n_batch = atoi(nxt());
        else if (a == "--temp") sp.temp = (float)atof(nxt());
        else if (a == "--top-k") sp.top_k = atoi(nxt());
        else if (a == "--top-p") sp.top_p = (float)atof(nxt());
        else if (a == "--min-p") sp.min_p = (float)atof(nxt());
        else if (a == "--repeat-penalty") sp.rep = (float)atof(nxt());
        else if (a == "--seed") seed = strtoull(nxt(), nullptr, 10);
        else if (a == "--cuda-compute") {
#ifdef EAHMMF_CUDA_COMPUTE
            g_cuda_compute = 1;
#else
            die("--cuda-compute needs a build with CUDA=1");
#endif
        }
        else if (a == "--cuda-resident-ffn") {
#ifdef EAHMMF_CUDA_COMPUTE
            g_cuda_resident_ffn = 1;
#else
            die("--cuda-resident-ffn needs a build with CUDA=1");
#endif
        }
        else if (a == "--cuda-tensor-cores") {
#ifdef EAHMMF_CUDA_COMPUTE
            if (setenv("EAHMMF_TENSOR_CORES", "1", 1) != 0)
                die("could not enable CUDA Tensor Core path");
            g_cuda_tensor_cores = 1;
#else
            die("--cuda-tensor-cores needs a build with CUDA=1");
#endif
        }
        else if (a == "--cuda-shadow") {
#ifdef EAHMMF_CUDA_COMPUTE
            g_shadow = 1;
#else
            die("--cuda-shadow needs a build with CUDA=1");
#endif
        }
        else if (a == "--populate") populate = true;
        else if (a == "--expert-tier") {
            std::string v = nxt();
            if (v == "ssd") xcfg.exec_tier = eahmmf::Residency::SSD;
            else if (v == "ram") xcfg.exec_tier = eahmmf::Residency::RAM;
            else if (v == "vram") xcfg.exec_tier = eahmmf::Residency::VRAM;
            else usage();
        }
        else if (a == "--expert-policy") {
            std::string v = nxt();
            if (v == "lru") xcfg.expert_policy = eahmmf::ExpertPolicy::LRU;
            else if (v == "frequency") xcfg.expert_policy = eahmmf::ExpertPolicy::FREQUENCY;
            else usage();
        }
        else if (a == "--expert-aging-interval") {
            xcfg.expert_aging_interval = strtoull(nxt(), nullptr, 10);
        }
        else if (a == "--ram-budget-mb") xcfg.ram_budget = (size_t)(atof(nxt()) * 1048576);
        else if (a == "--vram-budget-mb") xcfg.vram_budget = (size_t)(atof(nxt()) * 1048576);
        else if (a == "--vram-reserve-mb") xcfg.vram_reserve = (size_t)(atof(nxt()) * 1048576);
        else if (a == "--pwea-placement") pwea_placement_enabled = true;
        else if (a == "--pwea-prediction-only") pwea_prediction_only = true;
        else if (a == "--pwea-shadow-diagnostics") pwea_shadow_diagnostics = true;
        else if (a == "--pwea-max-migrations") {
            const char* value = nxt();
            char* end = nullptr;
            const unsigned long long parsed = strtoull(value, &end, 10);
            if (!value[0] || value[0] == '-' || !end || *end != '\0' || parsed == 0 ||
                parsed > static_cast<unsigned long long>(SIZE_MAX))
                die("--pwea-max-migrations requires a positive integer");
            pwea_max_migrations = static_cast<size_t>(parsed);
            pwea_max_migrations_explicit = true;
        }
        else if (a == "--expert-verify") xcfg.verify = true;
        else if (a == "--expert-strict") xcfg.strict = true;
        else if (a == "--expert-log") xcfg.log_path = nxt();
        else if (a == "--expert-stats") xstats = true;
        else if (a == "--expert-stats-per-prompt") prompt_stats_enabled = true;
        else if (a == "--expert-stats-per-prompt-file") prompt_stats_path = nxt();
        else if (a == "--os-memory-stats") os_memory_stats_enabled = true;
        else if (a == "--os-memory-stats-file") os_memory_stats_path = nxt();
        else if (a == "--device") device = nxt();
        else if (a == "--tokenize") { dbg_tok = nxt(); have_dbg = true; }
        else if (a == "--pretok") { dbg_pre = nxt(); have_dbg = true; }
        else if (a == "--logits-test") { dbg_logits = nxt(); have_dbg = true; }
        else usage();
    }
    if (model.empty()) usage();

    if (pwea_placement_enabled && xcfg.ram_budget == SIZE_MAX)
        die("--pwea-placement requires a finite --ram-budget-mb N");

    if (pwea_max_migrations_explicit && !pwea_placement_enabled)
        die("--pwea-max-migrations requires --pwea-placement");


    if (pwea_prediction_only && !pwea_placement_enabled)
        die("--pwea-prediction-only requires --pwea-placement");

    if (pwea_shadow_diagnostics && !pwea_placement_enabled)
        die("--pwea-shadow-diagnostics requires --pwea-placement");

    if (pwea_shadow_diagnostics && pwea_prediction_only)
        die("--pwea-shadow-diagnostics cannot be combined with --pwea-prediction-only");

#ifdef EAHMMF_CUDA_COMPUTE
    if ((g_cuda_resident_ffn || g_cuda_tensor_cores) && !g_cuda_compute)
        die("--cuda-resident-ffn and --cuda-tensor-cores require --cuda-compute");
    if (g_cuda_compute && device != "cuda")
        die("--cuda-compute needs --device cuda");
#endif

    n_batch = std::max(1, std::min(n_batch, MAXB));
    if (n_threads > 0) omp_set_num_threads(n_threads);
    sp.rng.seed(seed);

    if (!dbg_pre.empty()) {  // needs no model weights beyond tables
        std::vector<std::pair<size_t, size_t>> parts;
        pretokenize(dbg_pre, parts);
        for (auto& p : parts) printf("%s\x1f", dbg_pre.substr(p.first, p.second - p.first).c_str());
        printf("\n");
        return 0;
    }

    Engine E;
    E.etf_collection_enabled = etf_collection_enabled;
    E.load(model.c_str(), n_ctx, populate);
    {
        std::unique_ptr<eahmmf::DeviceBackend> d;
        if (device == "cuda") {
#ifdef EAHMMF_CUDA
            d.reset(new eahmmf::CudaDevice());
#else
            die("this binary was built without CUDA (rebuild with: make CUDA=1)");
#endif
        } else if (device != "sim") usage();
        #ifdef EAHMMF_CUDA_COMPUTE
        if (g_cuda_compute && device != "cuda")
            die("--cuda-compute needs --device cuda");
        xcfg.gpu_compute = g_cuda_compute != 0;
#endif
        if (!E.init_experts(xcfg, std::move(d)))
            die("cannot reinitialize expert store while leases are active");
        if (xcfg.exec_tier != eahmmf::Residency::SSD)
            fprintf(stderr, "experts: exec tier=%s device=%s ram_budget=%s vram_budget=%s\n", eahmmf::res_name(xcfg.exec_tier), E.dev->name(),
                    xcfg.ram_budget == SIZE_MAX ? "unlimited" : "set", xcfg.vram_budget == SIZE_MAX ? "unlimited" : "set");
    }
    Tokenizer tk;
    tk.load(E.g);
    fprintf(stderr, "tokenizer: %zu tokens, %zu merges, im_start=%d im_end=%d eos=%d, threads=%d\n", tk.vocab.size(), tk.merges.size(),
            tk.im_start, tk.im_end, tk.eos, omp_get_max_threads());
    if (have_dbg) {
        if (!dbg_tok.empty()) {
            for (int t : tk.encode(dbg_tok, true)) printf("%d ", t);
            printf("\n");
        }
        if (!dbg_logits.empty()) {
            std::vector<int> ids;
            for (char* s = strtok(&dbg_logits[0], ","); s; s = strtok(nullptr, ",")) ids.push_back(atoi(s));
            std::vector<float> lg(E.n_vocab);
            E.eval(ids, 0, n_batch, lg.data());
            for (int i = 0; i < std::min(E.n_vocab, 16); i++) printf("%.6f ", lg[i]);
            printf("\nargmax %d\n", (int)(std::max_element(lg.begin(), lg.end()) - lg.begin()));
        }
        return 0;
    }
    if (!have_prompt) interactive = true;

    std::vector<float> logits(E.n_vocab);
    std::vector<int> pending, hist;
    int n_past = 0;
    bool first_turn = true;
    auto stop_tok = [&](int t) { return t == tk.im_end || t == tk.endoftext || t == tk.eos; };

    auto generate = [&]() {
        auto t0 = std::chrono::steady_clock::now();
        int n_prompt = (int)pending.size(), n_gen = 0;
        double t_prefill = 0;
        for (int step = 0; step < n_new; step++) {
            if (n_past + (int)pending.size() > E.n_ctx) { fprintf(stderr, "\n[context full]\n"); pending.clear(); return false; }
            E.eval(pending, n_past, n_batch, logits.data());
            n_past += (int)pending.size();
            pending.clear();
            if (step == 0) t_prefill = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            int tok = sp.sample(logits.data(), E.n_vocab, hist);
            hist.push_back(tok);
            if (stop_tok(tok)) {
                if (!raw) { pending = {tk.im_end, tk.nl}; }
                break;
            }
            const std::string& pc = tk.piece[tok];
            fwrite(pc.data(), 1, pc.size(), stdout);
            fflush(stdout);
            n_gen++;
            pending = {tok};
            if (step == n_new - 1 && !raw) pending = {tok, tk.im_end, tk.nl};
        }
        // Flush final generated/assistant-closing tokens inside this prompt's
        // telemetry window so their expert activations are not attributed
        // to the next user prompt.
        if (!pending.empty()) {
            E.eval(pending, n_past, n_batch, logits.data());
            n_past += (int)pending.size();
            pending.clear();
        }

        double tt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "\n[prompt %d tok in %.2fs (%.1f t/s) | gen %d tok in %.2fs (%.2f t/s)]\n", n_prompt, t_prefill,
                n_prompt / std::max(t_prefill, 1e-9), n_gen, tt - t_prefill, n_gen / std::max(tt - t_prefill, 1e-9));
        return true;
    };

    std::optional<etf::TelemetrySnapshot> telemetry_window;
    etf::JsonlExporter etf_exporter;
    auto export_snapshot = [&](const etf::TelemetrySnapshot& snapshot) {
        if (etf_log_path.empty()) return;
        std::string error;
        if (!etf_exporter.append(etf_log_path, snapshot, &error))
            die("ETF JSONL export failed: %s", error.c_str());
        fprintf(stderr, "[ETF] exported %llu prompt record(s) to %s\n",
                (unsigned long long)snapshot.completed_prompts,
                etf_log_path.c_str());
    };
   // Native PWEA model is loaded once, outside the prompt loop.
   std::optional<pwea::model::Model> pwea_native_model;
   std::string pwea_model_error;
   std::optional<pwea::Prediction> pwea_prediction_for_prompt;

   if (pwea_placement_enabled) {
       try {
           pwea_native_model.emplace(
               pwea::model::load_model(
                   "pwea/native/models/pwea_model_v1.bin"));

           fprintf(stderr,
                   "[PWEA] Native model loaded | samples=%u | "
                   "input=%u | output=%u\n",
                   pwea::model::N_SAMPLES,
                   pwea::model::INPUT_DIM,
                   pwea::model::OUTPUT_DIM);
       } catch (const std::exception& error) {
           pwea_model_error = error.what();
           fprintf(stderr,
                   "[PWEA] Native model load failed: %s\n",
                   pwea_model_error.c_str());
       }
   }

        eahmmf::osmem::RamBudgetConfig ram_budget_config;
        // Advisory capacity is independent of the static EAHMMF
        // placement budget. This manager never changes ExpertStore limits.
        ram_budget_config.hard_ceiling_bytes = ~uint64_t{0};

        eahmmf::osmem::RamBudgetManager ram_budget_manager(
            ram_budget_config);

        uint64_t prompt_stats_sequence = 0;
auto run_prompt = [&](const std::string& original_prompt) {
                const auto prompt_start = std::chrono::steady_clock::now();
        uint64_t prompt_stats_id = ++prompt_stats_sequence;
        eahmmf::ResidencySnapshot prompt_s0, prompt_s1, prompt_s2;
        eahmmf::osmem::StageSnapshot os_s0, os_s1, os_s2;

        auto capture_os_stage = [&](bool inspect_model_mapping) {
            uint64_t ram_bytes = 0;
            uint64_t vram_bytes = 0;
            if (E.store) {
                const auto residency = E.store->snapshot();
                ram_bytes = residency.ram_used;
                vram_bytes = residency.vram_used;
            }
            auto stage = eahmmf::osmem::capture_stage(
                E.g.map, E.g.size, ram_bytes, vram_bytes,
                inspect_model_mapping);

            stage.ram_budget = ram_budget_manager.observe(
                stage.system, stage.cgroup, ram_bytes,
                stage.system.memory_pressure);

            return stage;
        };

        auto apply_ram_admission = [&](const eahmmf::osmem::StageSnapshot& stage,
                                       const char* phase) {
            if (!E.store || !stage.ram_budget.has_recommendation)
                return;

            const uint64_t recommended =
                stage.ram_budget.recommended_capacity_bytes;
            const size_t cap = recommended > static_cast<uint64_t>(SIZE_MAX)
                ? SIZE_MAX : static_cast<size_t>(recommended);

            E.store->set_adaptive_ram_cap(cap);
            fprintf(stderr,
                    "[EAHMMF RAM] adaptive admission | phase=%s "
                    "cap=%.2f MiB static=%.2f MiB used=%.2f MiB\n",
                    phase,
                    E.store->adaptive_ram_total() / 1048576.0,
                    E.store->ram_total() / 1048576.0,
                    E.store->ram_used() / 1048576.0);
        };

        os_s0 = capture_os_stage(os_memory_stats_enabled);
        apply_ram_admission(os_s0, "S0");
        if (prompt_stats_enabled && E.store)
            prompt_s0 = E.store->snapshot();
pwea_prediction_for_prompt.reset();

       pwea::Prediction prediction;
       std::string pwea_error;

              const auto prediction_start = std::chrono::steady_clock::now();
if (pwea_placement_enabled) {
           if (pwea_native_model) {
               try {
                   const auto native_values =
                       pwea::predictor::predict_prompt(
                           *pwea_native_model, original_prompt);

                   if (native_values.size() != prediction.values.size()) {
                       pwea_error = "Native prediction output size mismatch";
                   } else {
                       for (size_t i = 0; i < native_values.size(); ++i) {
                           prediction.values[i] = native_values[i];
                       }

                       pwea_prediction_for_prompt.emplace(
                           std::move(prediction));
                   }
               } catch (const std::exception& error) {
                   pwea_error = error.what();
               }
           } else {
               pwea_error = pwea_model_error.empty()
                   ? "Native PWEA model is unavailable"
                   : pwea_model_error;
           }

           if (pwea_prediction_for_prompt) {
               fprintf(stderr,
                       "[PWEA] Native prediction ready | layers=%zu | experts=%zu\n",
                       pwea::N_LAYER, pwea::N_EXPERT);
           } else {
               fprintf(stderr,
                       "[PWEA] Prediction unavailable: %s | "
                       "continuing without prediction-based placement\n",
                       pwea_error.c_str());
           }
       } else {
           fprintf(stderr,
                   "[PWEA] Prediction disabled (normal baseline mode)\n");
       }

              const double prediction_seconds = std::chrono::duration<double>(
           std::chrono::steady_clock::now() - prediction_start).count();
       const auto placement_start = std::chrono::steady_clock::now();
// Prediction-driven placement is opt-in and runs at the prompt
        // boundary, before ETF starts measuring actual inference activity.
        if (pwea_placement_enabled && pwea_prediction_for_prompt && E.store) {
            // Snapshot the latest ETF -> EAHMMF activation history.
            std::vector<double> historical_activation;
            historical_activation.reserve(
                static_cast<size_t>(E.n_layer) *
                static_cast<size_t>(E.n_expert));

            double history_max = 0.0;
            for (int layer = 0; layer < E.n_layer; ++layer) {
                for (int expert = 0; expert < E.n_expert; ++expert) {
                    const double value =
                        E.optimizer.historical_activation(layer, expert);
                    historical_activation.push_back(value);
                    history_max = std::max(history_max, value);
                }
            }

            eahmmf::PWEAPlacementConfig adaptive_config;
            adaptive_config.gpu_compute = xcfg.gpu_compute;
            if (pwea_prediction_only) {
                adaptive_config.prediction_weight = 1.0;
                adaptive_config.history_weight = 0.0;
            }
            const bool history_active = history_max > 0.0;
            const std::vector<double>* placement_history =
                pwea_prediction_only ? nullptr : &historical_activation;
            fprintf(stderr,
                "[PWEA->EAHMMF] placement mode=%s | prediction_weight=%.2f "
                "history_weight=%.2f | ETF history=%s | history_max=%.6f\n",
                pwea_prediction_only ? "prediction-only" : "adaptive",
                adaptive_config.prediction_weight,
                adaptive_config.history_weight,
                pwea_prediction_only ? "ignored" :
                    (history_active ? "active" : "warming_up"),
                history_max);


            const size_t shadow_total =
                static_cast<size_t>(E.n_layer) *
                static_cast<size_t>(E.n_expert);
            eahmmf::PWEAPlacementReport adaptive_shadow;

            if (pwea_shadow_diagnostics) {
            // Same-live-state shadow comparison.
            // Both planners are read-only. Only the existing adaptive
            // execute_pwea_placement() call below performs migrations.
const size_t shadow_ram_before = E.store->ram_used();
            const size_t shadow_vram_before = E.store->vram_used();

            auto prediction_only_shadow =
                eahmmf::plan_pwea_placement(
                    *E.store,
                    *pwea_prediction_for_prompt,
                    E.n_layer,
                    E.n_expert,
                    xcfg.vram_reserve,
                    adaptive_config,
                    nullptr);

            adaptive_shadow = eahmmf::plan_pwea_placement(
                    *E.store,
                    *pwea_prediction_for_prompt,
                    E.n_layer,
                    E.n_expert,
                    xcfg.vram_reserve,
                    adaptive_config,
                    placement_history);

            const size_t shadow_ram_after = E.store->ram_used();
            const size_t shadow_vram_after = E.store->vram_used();

            fprintf(stderr,
                "[PWEA shadow] PWEA-only valid=%d evaluated=%zu "
                "feasible=%zu rejected=%zu\n",
                prediction_only_shadow.valid,
                prediction_only_shadow.evaluated,
                prediction_only_shadow.dry_run_feasible,
                prediction_only_shadow.rejected);

            fprintf(stderr,
                "[PWEA shadow] Adaptive valid=%d evaluated=%zu "
                "feasible=%zu rejected=%zu\n",
                adaptive_shadow.valid,
                adaptive_shadow.evaluated,
                adaptive_shadow.dry_run_feasible,
                adaptive_shadow.rejected);

            fprintf(stderr,
                "[PWEA shadow] Store unchanged during planning: "
                "RAM %zu -> %zu | VRAM %zu -> %zu\n",
                shadow_ram_before, shadow_ram_after,
                shadow_vram_before, shadow_vram_after);

            if (prediction_only_shadow.valid && adaptive_shadow.valid &&
                prediction_only_shadow.placements.size() == shadow_total &&
                adaptive_shadow.placements.size() == shadow_total) {

                using ShadowRec =
                    eahmmf::PWEAPlacementRecommendation;

                std::vector<const ShadowRec*> prediction_by_index(
                    shadow_total, nullptr);
                std::vector<const ShadowRec*> adaptive_by_index(
                    shadow_total, nullptr);

                std::vector<int> prediction_rank(shadow_total, -1);
                std::vector<int> adaptive_rank(shadow_total, -1);

                auto shadow_index = [&](const ShadowRec& rec) {
                    return static_cast<size_t>(rec.layer) *
                               static_cast<size_t>(E.n_expert) +
                           static_cast<size_t>(rec.expert_id);
                };

                bool shadow_indices_valid = true;

                for (size_t rank = 0;
                     rank < prediction_only_shadow.placements.size();
                     ++rank) {
                    const auto& rec =
                        prediction_only_shadow.placements[rank];
                    const size_t index = shadow_index(rec);

                    if (index >= shadow_total ||
                        prediction_by_index[index] != nullptr) {
                        shadow_indices_valid = false;
                        break;
                    }

                    prediction_by_index[index] = &rec;
                    prediction_rank[index] = static_cast<int>(rank + 1);
                }

                for (size_t rank = 0;
                     rank < adaptive_shadow.placements.size() &&
                     shadow_indices_valid;
                     ++rank) {
                    const auto& rec = adaptive_shadow.placements[rank];
                    const size_t index = shadow_index(rec);

                    if (index >= shadow_total ||
                        adaptive_by_index[index] != nullptr) {
                        shadow_indices_valid = false;
                        break;
                    }

                    adaptive_by_index[index] = &rec;
                    adaptive_rank[index] = static_cast<int>(rank + 1);
                }

                if (shadow_indices_valid) {
                    size_t rank_changed = 0;
                    size_t class_changed = 0;
                    size_t target_changed = 0;
                    size_t status_changed = 0;
                    size_t current_mismatch = 0;
                    size_t feasible_added = 0;
                    size_t feasible_removed = 0;

                    for (size_t i = 0; i < shadow_total; ++i) {
                        const auto* a = prediction_by_index[i];
                        const auto* b = adaptive_by_index[i];

                        if (!a || !b) {
                            shadow_indices_valid = false;
                            break;
                        }

                        if (prediction_rank[i] != adaptive_rank[i])
                            ++rank_changed;

                        if (a->classification != b->classification)
                            ++class_changed;

                        if (a->target != b->target)
                            ++target_changed;

                        if (a->status != b->status)
                            ++status_changed;

                        if (a->current != b->current)
                            ++current_mismatch;

                        const bool a_feasible =
                            a->status ==
                            eahmmf::PWEAPlanStatus::DRY_RUN_FEASIBLE;

                        const bool b_feasible =
                            b->status ==
                            eahmmf::PWEAPlanStatus::DRY_RUN_FEASIBLE;

                        if (!a_feasible && b_feasible)
                            ++feasible_added;
                        else if (a_feasible && !b_feasible)
                            ++feasible_removed;
                    }

                    if (shadow_indices_valid) {
                        fprintf(stderr,
                            "[PWEA shadow] Same-state comparison | "
                            "rank_changed=%zu/%zu | class_changed=%zu | "
                            "target_changed=%zu | status_changed=%zu\n",
                            rank_changed, shadow_total,
                            class_changed, target_changed, status_changed);

                        fprintf(stderr,
                            "[PWEA shadow] Residency agreement=%s "
                            "mismatches=%zu | feasible_added=%zu "
                            "feasible_removed=%zu\n",
                            current_mismatch == 0 ? "YES" : "NO",
                            current_mismatch,
                            feasible_added, feasible_removed);

                        auto migration_slots =
                            [&](const eahmmf::PWEAPlacementReport& report) {
                                std::vector<size_t> slots;
                                for (const auto& rec : report.placements) {
                                    if (rec.status ==
                                        eahmmf::PWEAPlanStatus::DRY_RUN_FEASIBLE) {
                                        slots.push_back(shadow_index(rec));
                                        if (slots.size() >= pwea_max_migrations)
                                            break;
                                    }
                                }
                                return slots;
                            };

                        const auto prediction_slots =
                            migration_slots(prediction_only_shadow);
                        const auto adaptive_slots =
                            migration_slots(adaptive_shadow);

                        auto print_slots = [&](const char* label,
                                               const std::vector<size_t>& slots) {
                            fprintf(stderr, "[PWEA shadow] %s:", label);
                            if (slots.empty()) fprintf(stderr, " none");

                            for (size_t index : slots) {
                                fprintf(stderr, " L%zuE%zu",
                                    index / static_cast<size_t>(E.n_expert),
                                    index % static_cast<size_t>(E.n_expert));
                            }
                            fprintf(stderr, "\n");
                        };

                        print_slots("PWEA-only migration slots",
                                    prediction_slots);
                        print_slots("Adaptive migration slots",
                                    adaptive_slots);
                    }
                }

                if (!shadow_indices_valid) {
                    fprintf(stderr,
                        "[PWEA shadow] ERROR: invalid or duplicate expert "
                        "indices in planner reports\n");
                }
            } else {
                fprintf(stderr,
                    "[PWEA shadow] Comparison unavailable: invalid planner "
                    "report or unexpected expert count\n");
                if (!prediction_only_shadow.valid)
                    fprintf(stderr,
                        "[PWEA shadow] PWEA-only error: %s\n",
                        prediction_only_shadow.error.c_str());
                if (!adaptive_shadow.valid)
                    fprintf(stderr,
                        "[PWEA shadow] Adaptive error: %s\n",
                        adaptive_shadow.error.c_str());
            }

                        }

auto placement = eahmmf::execute_pwea_placement(
                *E.store,
                *pwea_prediction_for_prompt,
                E.n_layer,
                E.n_expert,
                xcfg.vram_reserve,
                pwea_max_migrations,
                adaptive_config,
                placement_history);


            if (pwea_shadow_diagnostics && placement.valid && adaptive_shadow.valid &&
                placement.plan.placements.size() == shadow_total &&
                adaptive_shadow.placements.size() == shadow_total) {

                std::vector<const eahmmf::PWEAPlacementRecommendation*>
                    adaptive_shadow_by_index(shadow_total, nullptr);

                std::vector<const eahmmf::PWEAPlacementRecommendation*>
                    executed_plan_by_index(shadow_total, nullptr);

                auto report_index = [&](const auto& rec) {
                    return static_cast<size_t>(rec.layer) *
                               static_cast<size_t>(E.n_expert) +
                           static_cast<size_t>(rec.expert_id);
                };

                bool match_indices_valid = true;

                for (const auto& rec : adaptive_shadow.placements) {
                    const size_t index = report_index(rec);
                    if (index >= shadow_total ||
                        adaptive_shadow_by_index[index]) {
                        match_indices_valid = false;
                        break;
                    }
                    adaptive_shadow_by_index[index] = &rec;
                }

                for (const auto& rec : placement.plan.placements) {
                    const size_t index = report_index(rec);
                    if (index >= shadow_total ||
                        executed_plan_by_index[index]) {
                        match_indices_valid = false;
                        break;
                    }
                    executed_plan_by_index[index] = &rec;
                }

                size_t plan_mismatches = 0;

                if (match_indices_valid) {
                    for (size_t i = 0; i < shadow_total; ++i) {
                        const auto* a = adaptive_shadow_by_index[i];
                        const auto* b = executed_plan_by_index[i];

                        if (!a || !b ||
                            a->current != b->current ||
                            a->target != b->target ||
                            a->classification != b->classification ||
                            a->status != b->status ||
                            a->normalized_score != b->normalized_score) {
                            ++plan_mismatches;
                        }
                    }
                }

                fprintf(stderr,
                    "[PWEA shadow] Adaptive shadow vs execution plan: "
                    "mismatches=%zu/%zu | match=%s\n",
                    plan_mismatches, shadow_total,
                    match_indices_valid && plan_mismatches == 0
                        ? "YES" : "NO");
            } else {
                fprintf(stderr,
                    "[PWEA shadow] Adaptive shadow vs execution plan: "
                    "comparison unavailable\n");
            }

            if (!placement.valid) {
                fprintf(stderr,
                        "[PWEA->EAHMMF] placement aborted: %s\n",
                        placement.error.c_str());
            } else {
                fprintf(stderr,
                        "[PWEA->EAHMMF] evaluated=%zu HOT=%zu WARM=%zu COLD=%zu "
                        "feasible=%zu attempted=%zu migrated=%zu failed=%zu skipped=%zu\n",
                        placement.plan.evaluated,
                        placement.plan.hot,
                        placement.plan.warm,
                        placement.plan.cold,
                        placement.plan.dry_run_feasible,
                        placement.attempted,
                        placement.migrated,
                        placement.failed,
                        placement.skipped);

                for (const auto& item : placement.results) {
                    fprintf(stderr,
                            "[PWEA->EAHMMF] L%d E%d %s -> %s: %s\n",
                            item.layer,
                            item.expert_id,
                            eahmmf::res_name(item.from),
                            eahmmf::res_name(item.to),
                            item.reason.c_str());
                }
            }
        }

                const double placement_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - placement_start).count();
        if (prompt_stats_enabled && E.store)
            prompt_s1 = E.store->snapshot();
        os_s1 = capture_os_stage(false);
        apply_ram_admission(os_s1, "S1");
// Prediction and placement complete before the ETF measurement window.
        if (E.etf_collection_enabled) {
            E.active_prompt_id = E.telemetry.begin_prompt(workload_metadata);
            if (E.active_prompt_id == 0) die("ETF: a prompt is already active");
            prompt_stats_id = E.active_prompt_id;
        }

                const auto inference_start = std::chrono::steady_clock::now();
const bool completed = generate();
        const double inference_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - inference_start).count();
        if (prompt_stats_enabled && E.store)
            prompt_s2 = E.store->snapshot();
        if (os_memory_stats_enabled)
            os_s2 = capture_os_stage(true);
        const double total_prompt_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - prompt_start).count();
        if (prompt_stats_enabled && E.store) {
            if (!append_prompt_stats(prompt_stats_path, prompt_stats_id,
                    completed, workload_metadata,
                    prediction_seconds, placement_seconds,
                    inference_seconds, total_prompt_seconds,
                    prompt_s0, prompt_s1, prompt_s2)) {
                fprintf(stderr, "[prompt-stats] ERROR: failed to append %s\n",
                        prompt_stats_path.c_str());
            }
        }
        if (os_memory_stats_enabled) {
            std::string os_error;
            if (!eahmmf::osmem::append_prompt_record(
                    os_memory_stats_path, prompt_stats_id,
                    completed, model, os_s0, os_s1, os_s2,
                    &os_error)) {
                fprintf(stderr,
                    "[OS memory] ERROR: %s (%s)\n",
                    os_error.c_str(), os_memory_stats_path.c_str());
            }
        }

        if (E.etf_collection_enabled && completed) {
            auto snapshot = E.telemetry.complete_prompt(E.active_prompt_id);
            if (snapshot) {
                telemetry_window = std::move(snapshot);
                export_snapshot(*telemetry_window);
                if (E.store) {
                    auto report = E.optimizer.observe(
                        *telemetry_window, *E.store, E.n_layer, E.n_expert);
                    fprintf(stderr,
                            "[EAHMMF optimizer] %s | prompts=%llu | experts=%zu | dropped=%llu\n",
                            report.status.c_str(),
                            (unsigned long long)report.completed_prompts,
                            report.experts_observed,
                            (unsigned long long)report.dropped_events);
                }
            }
        } else if (E.etf_collection_enabled) {
            E.telemetry.cancel_prompt(E.active_prompt_id);
        }
        E.active_prompt_id = 0;
    };

    auto finalize_telemetry = [&]() {
        if (!E.etf_collection_enabled) return;
        auto snapshot = E.telemetry.flush();
        if (!snapshot) return;
        export_snapshot(*snapshot);
        if (E.store) {
            auto report = E.optimizer.observe(
                *snapshot, *E.store, E.n_layer, E.n_expert);
            fprintf(stderr,
                    "[EAHMMF optimizer] %s | prompts=%llu | experts=%zu | dropped=%llu\n",
                    report.status.c_str(),
                    (unsigned long long)report.completed_prompts,
                    report.experts_observed,
                    (unsigned long long)report.dropped_events);
        }
    };

    auto add_turn = [&](const std::string& user) {
        auto app = [&](const std::vector<int>& v) { pending.insert(pending.end(), v.begin(), v.end()); };
        if (first_turn) {
            pending.push_back(tk.im_start);
            app(tk.encode("system\n" + sys, false));
            pending.push_back(tk.im_end);
            pending.push_back(tk.nl);
            first_turn = false;
        }
        pending.push_back(tk.im_start);
        app(tk.encode("user\n" + user, false));
        pending.push_back(tk.im_end);
        pending.push_back(tk.nl);
        pending.push_back(tk.im_start);
        app(tk.encode("assistant\n", false));
    };

    if (have_prompt) {
        if (raw) pending = tk.encode(prompt, true);
        else add_turn(prompt);
        run_prompt(prompt);
        printf("\n");
        if (!interactive) {
            finalize_telemetry();
            if (xstats) E.store->print_summary(stderr);
            return 0;
        }
    }
    char* line = nullptr;
    size_t cap = 0;
    while (interactive) {
        fprintf(stderr, "\n> ");
        ssize_t len = getline(&line, &cap, stdin);
        if (len <= 0) break;
        std::string u(line, len);
        while (!u.empty() && (u.back() == '\n' || u.back() == '\r')) u.pop_back();
        if (u == "/exit" || u == "/quit") break;
        if (u == "/reset") { n_past = 0; pending.clear(); hist.clear(); first_turn = true; fprintf(stderr, "[reset]\n"); continue; }
        if (u.empty()) continue;
        if (raw) pending = tk.encode(u, true);
        else add_turn(u);
        run_prompt(u);
        printf("\n");
    }
    finalize_telemetry();
    if (xstats) E.store->print_summary(stderr);
    return 0;
}
#endif  // QWEN3MOE_NO_MAIN
