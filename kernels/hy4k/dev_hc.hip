/* dev_hc.hip -- card rows of hy4_hc_pre, hy4_hc_post and hy4_mla_kv (docs/OPS-HY4.md).
 * Same arithmetic as ref_hc.cpp and ref_mla.cpp, in FP32 with one rounding at the output; the sums
 * run in another order (block reduction), within the tests' tolerance. */
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "dev_common.h"

namespace hy4k {
namespace {

constexpr int kThreads = 256;

/* One block per row. The row of streams (hc·n ≤ 16384 values) is read once into LDS as f32. */
__global__ __launch_bounds__(kThreads) void hc_pre_k(const void* h, int hdt, long hp, long hs, const float* fn,
                                                   long fp, const float* base, long bs, const float* scale, long ss,
                                                   void* x, int xdt, long xp, long xs, float* post, long pp, long ps,
                                                   int n, int hc, int head, float eps, float heps, float mag) {
    /* The row goes to LDS if it fits (up to 14336 values); else it is read from memory each time. */
    extern __shared__ float flat[];
    __shared__ float red[kThreads / 32];
    const bool lds = hc * n <= 14336;
#define HV(i) (lds ? flat[(i)] : dld(h, hdt, r * hp + (long)(i) * hs))
    __shared__ float coef[32];
    __shared__ float pre[16];
    const long r = blockIdx.x;
    const int W = hc * n, mix = head ? hc : 2 * hc;
    float sq = 0;
    for (int i = threadIdx.x; i < W; i += kThreads) {
        const float v = dld(h, hdt, r * hp + (long)i * hs);
        if (lds) flat[i] = v;
        sq += v * v;
    }
    sq = block_sum<kThreads>(sq, red);
    const float inv = 1.0f / sqrtf(sq / (float)W + eps);
    for (int j = 0; j < mix; ++j) {
        float d = 0;
        for (int i = threadIdx.x; i < W; i += kThreads) d += HV(i) * fn[(long)j * fp + i];
        d = block_sum<kThreads>(d, red);
        if (threadIdx.x == 0) coef[j] = d * inv;
    }
    __syncthreads();
    if (threadIdx.x < hc) {
        const int j = threadIdx.x;
        const float s0 = scale[0];
        pre[j] = dsigmoid(coef[j] * s0 + base[(long)j * bs]) + heps;   // pre
        if (!head)
            post[r * pp + (long)j * ps] = mag * dsigmoid(coef[hc + j] * scale[ss] + base[(long)(hc + j) * bs]) + heps;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += kThreads) {
        float v = 0;
        for (int j = 0; j < hc; ++j) v += pre[j] * HV(j * n + i);
        dst(x, xdt, r * xp + (long)i * xs, v);
    }
#undef HV
}

/* The same row in one pass: each thread accumulates the norm and the MIX dot products from one read
 * of the row, and one reduction closes all MIX + 1 sums. hc_pre_k reads the row MIX times with MIX
 * block reductions in a row; with the 1-8 rows of a decode step (1-8 blocks on the whole card) that
 * was 21% of the step. 256 threads and the same per-thread order, shuffle and cross-wave sum as
 * hc_pre_k, so the output is bit-identical: 512 threads was 3% faster but moved the model's KL
 * against FP32 from 0.177 to 0.188, a reordering this model amplifies through its layers. */
constexpr int kThreads1 = 256;
template <int MIX>
__global__ __launch_bounds__(kThreads1) void hc_pre1_k(const void* h, int hdt, long hp, long hs, const float* fn,
                                                     long fp, const float* base, long bs, const float* scale, long ss,
                                                     void* x, int xdt, long xp, long xs, float* post, long pp, long ps,
                                                     int n, int hc, int head, float eps, float heps, float mag) {
    extern __shared__ float flat[];
    __shared__ float red[kThreads1 / 32][MIX + 1];
    __shared__ float tot[MIX + 1];
    __shared__ float pre[16];
    const long r = blockIdx.x;
    const int W = hc * n;
    const bool lds = W <= 14336;
    const int t = threadIdx.x, lane = t & 31, wave = t >> 5;
    float acc[MIX + 1];
#pragma unroll
    for (int j = 0; j <= MIX; ++j) acc[j] = 0.0f;
    for (int i = t; i < W; i += kThreads1) {
        const float v = dld(h, hdt, r * hp + (long)i * hs);
        if (lds) flat[i] = v;
        acc[MIX] += v * v;
#pragma unroll
        for (int j = 0; j < MIX; ++j) acc[j] += v * fn[(long)j * fp + i];
    }
#pragma unroll
    for (int j = 0; j <= MIX; ++j)
        for (int d = 16; d; d >>= 1) acc[j] += __shfl_xor(acc[j], d, 32);
    if (lane == 0)
#pragma unroll
        for (int j = 0; j <= MIX; ++j) red[wave][j] = acc[j];
    __syncthreads();
    if (t <= MIX) {
        float v = 0.0f;
        for (int w = 0; w < kThreads1 / 32; ++w) v += red[w][t];
        tot[t] = v;
    }
    __syncthreads();
    const float inv = 1.0f / sqrtf(tot[MIX] / (float)W + eps);
    if (t < hc) {
        pre[t] = dsigmoid(tot[t] * inv * scale[0] + base[(long)t * bs]) + heps;
        if (!head)
            post[r * pp + (long)t * ps] = mag * dsigmoid(tot[hc + t] * inv * scale[ss] + base[(long)(hc + t) * bs]) + heps;
    }
    __syncthreads();
    for (int i = t; i < n; i += kThreads1) {
        float v = 0;
        for (int j = 0; j < hc; ++j) v += pre[j] * (lds ? flat[j * n + i] : dld(h, hdt, r * hp + (long)(j * n + i) * hs));
        dst(x, xdt, r * xp + (long)i * xs, v);
    }
}

/* hc_pre1_k split over the card, for the few rows of a decode step (1-8 rows were 1-8 blocks on a
 * 64-CU card, latency-bound). The bits are hc_pre1_k's because the work is cut where hc_pre1_k's
 * arithmetic already has independent pieces, never inside one of its sums:
 *   - each of its MIX + 1 sums is 256 per-thread chains (thread t adds i = t, t + 256, ... in order),
 *     a xor butterfly inside each wave of 32, and an in-order sum of the 8 wave results;
 *   - so a chain (virtual thread t, sum j) is a unit of work: hc_pre_part_k gives one wave of 32 the
 *     32 chains of virtual wave w for sum j, the same lanes the butterfly mixes, runs the chains in
 *     the same order and the same butterfly, and stores the 8 wave results of each sum;
 *   - hc_pre_fin_k adds the 8 in hc_pre1_k's order (from 0.0f, w = 0..7) and finishes as it does.
 * With -ffp-contract=off (CMakeLists.txt) each step is a rounded product then a rounded add in both
 * kernels; reading ahead does not change which adds happen or their order. Per row that is
 * 8·(MIX + 1) waves instead of 8 on one CU: 72 for Hy4. It serves every row count: on the R9700
 * (tests/bench_hc.py) it took 1 row from 24.0 to 7.5 us, 8 from 23.6 to 7.9, 48 from 24.1 to 14.0
 * and 2048 from 593 to ~390, although it reads the row MIX + 1 times (from L2) instead of once. */
constexpr int kVWaves = kThreads1 / 32;   // hc_pre1_k's waves: the virtual waves of the split

template <int R, int U, int DT, bool SQ>
__device__ inline void hc_chains(const void* h, const long (&ro)[R], long hs, const float* f, int t, int W,
                                 float (&acc)[R]) {
    int i = t;
    /* U steps of the chain at a time: the loads of the U steps go out together (one latency instead
     * of U), then the adds run in the chain's order. */
    for (; i + (U - 1) * kThreads1 < W; i += U * kThreads1) {
        float fv[U], hv[U][R];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            if (!SQ) fv[u] = f[i + u * kThreads1];
#pragma unroll
            for (int q = 0; q < R; ++q) hv[u][q] = dld(h, DT, ro[q] + (long)(i + u * kThreads1) * hs);
        }
#pragma unroll
        for (int u = 0; u < U; ++u)
#pragma unroll
            for (int q = 0; q < R; ++q) acc[q] += hv[u][q] * (SQ ? hv[u][q] : fv[u]);
    }
    for (; i < W; i += kThreads1) {
        const float fv = SQ ? 0.0f : f[i];
#pragma unroll
        for (int q = 0; q < R; ++q) {
            const float v = dld(h, DT, ro[q] + (long)i * hs);
            acc[q] += v * (SQ ? v : fv);
        }
    }
}

/* grid (8·(MIX + 1), ceil(rows / R)), one wave a block: block b is virtual wave b % 8 of sum b / 8
 * (sum MIX is the norm) for rows [R·y, R·y + R). The R rows share the fn loads. part[row][j][w]. */
template <int MIX, int R, int DT>
__global__ __launch_bounds__(32) void hc_pre_part_k(const void* h, long hp, long hs, const float* fn, long fp,
                                                   float* part, int rows, int W) {
    constexpr int U = R == 1 ? 16 : R == 2 ? 8 : 4;   // read-ahead, ~40 VGPRs at every R
    const int w = blockIdx.x % kVWaves, j = blockIdx.x / kVWaves, lane = threadIdx.x;
    const int t = w * 32 + lane;                       // hc_pre1_k's thread
    const int r0 = blockIdx.y * R;
    long ro[R];
    float acc[R];
#pragma unroll
    for (int q = 0; q < R; ++q) {
        ro[q] = (long)min(r0 + q, rows - 1) * hp;      // rows past the end repeat the last, unstored
        acc[q] = 0.0f;
    }
    if (j == MIX) hc_chains<R, U, DT, true>(h, ro, hs, nullptr, t, W, acc);
    else hc_chains<R, U, DT, false>(h, ro, hs, fn + (long)j * fp, t, W, acc);
#pragma unroll
    for (int q = 0; q < R; ++q)
        for (int d = 16; d; d >>= 1) acc[q] += __shfl_xor(acc[q], d, 32);
    if (lane == 0)
#pragma unroll
        for (int q = 0; q < R; ++q)
            if (r0 + q < rows) part[((long)(r0 + q) * (MIX + 1) + j) * kVWaves + w] = acc[q];
}

/* grid (ceil(n / 256), rows): each block closes the row's sums (the same few adds in every block of
 * the row) and writes 256 columns of x; block 0 writes post. */
template <int MIX>
__global__ __launch_bounds__(kThreads1) void hc_pre_fin_k(const void* h, int hdt, long hp, long hs, const float* part,
                                                        const float* base, long bs, const float* scale, long ss,
                                                        void* x, int xdt, long xp, long xs, float* post, long pp,
                                                        long ps, int n, int hc, int head, float eps, float heps,
                                                        float mag) {
    __shared__ float tot[MIX + 1];
    __shared__ float pre[16];
    const long r = blockIdx.y;
    const int W = hc * n, t = threadIdx.x;
    if (t <= MIX) {
        const float* p = part + (r * (MIX + 1) + t) * kVWaves;
        float v = 0.0f;
        for (int w = 0; w < kVWaves; ++w) v += p[w];
        tot[t] = v;
    }
    __syncthreads();
    const float inv = 1.0f / sqrtf(tot[MIX] / (float)W + eps);
    if (t < hc) {
        pre[t] = dsigmoid(tot[t] * inv * scale[0] + base[(long)t * bs]) + heps;
        if (!head && blockIdx.x == 0)
            post[r * pp + (long)t * ps] = mag * dsigmoid(tot[hc + t] * inv * scale[ss] + base[(long)(hc + t) * bs]) + heps;
    }
    __syncthreads();
    const int i = blockIdx.x * kThreads1 + t;
    if (i < n) {
        float v = 0;
        for (int j = 0; j < hc; ++j) v += pre[j] * dld(h, hdt, r * hp + (long)(j * n + i) * hs);
        dst(x, xdt, r * xp + (long)i * xs, v);
    }
}

template <int MIX, int R>
void (*part_kernel(int dt))(const void*, long, long, const float*, long, float*, int, int) {
    return dt == 1 ? hc_pre_part_k<MIX, R, 1> : dt == 2 ? hc_pre_part_k<MIX, R, 2> : hc_pre_part_k<MIX, R, 0>;
}

__global__ void hc_post_k(void* h, int hdt, long hp, long hs, const void* y, int ydt, long yp, long ys,
                          const float* post, long pp, long ps, int n, int hc) {
    const long r = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hc * n) return;
    const int j = i / n, c = i - j * n;
    const long o = r * hp + (long)i * hs;
    dst(h, hdt, o, post[r * pp + (long)j * ps] * dld(y, ydt, r * yp + (long)c * ys) + dld(h, hdt, o));
}

__global__ void mla_kv_k(const void* kvx, int xdt, long xp, long xs, const void* kr, int rdt, long rp, long rs,
                         void* k, int kdt, long kp, long ks, void* v, int vdt, long vp, long vs, int H, int nope,
                         int rope, int vd) {
    const long r = blockIdx.y;
    const int per = nope + vd, kw = rope + nope;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < H * kw) {
        const int hh = i / kw, c = i - hh * kw;
        const float val = c < rope ? dld(kr, rdt, r * rp + (long)c * rs)
                                   : dld(kvx, xdt, r * xp + ((long)hh * per + (c - rope)) * xs);
        dst(k, kdt, r * kp + (long)i * ks, val);
    }
    if (i < H * vd) {
        const int hh = i / vd, c = i - hh * vd;
        dst(v, vdt, r * vp + (long)i * vs, dld(kvx, xdt, r * xp + ((long)hh * per + nope + c) * xs));
    }
}

}  // namespace
}  // namespace hy4k

using namespace hy4k;

namespace {
/* Which hc_pre form serves a call: 0 the default (the split while the scratch holds it), 1 hc_pre1_k
 * (the form before the split, for an A/B), 2 the split. HY4_HC_PRE=old or =split, read once;
 * the parameter `hc_form` (1 or 2), which the graph never passes, lets tests/test_dev.py compare
 * both forms in one process. */
int hc_form(const RadArgs* a) {
    static const int env = [] {
        const char* v = std::getenv("HY4_HC_PRE");
        if (!v || !*v) return 0;
        if (!std::strcmp(v, "old")) return 1;
        if (!std::strcmp(v, "split")) return 2;
        std::fprintf(stderr, "hy4k: HY4_HC_PRE=%s is not old or split: the split is used\n", v);
        return 0;
    }();
    const int64_t p = pi(a, "hc_form", 0);
    return p == 1 || p == 2 ? (int)p : env;
}
int64_t hc_part_bytes(int64_t rows, int64_t mix) { return rows * (mix + 1) * kVWaves * (int64_t)sizeof(float); }
}  // namespace

/* Scratch for the split: the 8 wave results of each sum, 288 bytes a row with Hy4's 8 products (590 KB
 * at 2048 rows), sized for the band's upper row count. Without it the launch falls back to hc_pre1_k,
 * the same bits. */
int64_t hy4_dev_hc_pre_scratch(const RadArgs* a) {
    const int f = hc_form(a);
    if (f == 1) return 0;
    const int64_t hc = pi(a, "hc"), rows = pi(a, "M", 1), mix = pi(a, "head", 0) ? hc : 2 * hc;
    if (mix != 8 && mix != 4) return 0;
    return hc_part_bytes(rows, mix);
}

int hy4_dev_hc_pre(const RadArgs* a, RadStream s) {
    const int64_t n = pi(a, "n"), hc = pi(a, "hc"), head = pi(a, "head", 0);
    const float eps = (float)pf(a, "eps", 1e-5), heps = (float)pf(a, "hc_eps", 1e-6);
    const float mag = (float)pf(a, "magnitude", 2.0);
    /* The host's limits (ref_hc.cpp). The row goes to LDS if it fits (14336 values, 56 KiB). */
    if (n < 1 || hc < 1 || hc > 16 || hc * n > kMaxHcWidth || (head != 0 && head != 1) || !rad_args_have(a, 5))
        return RAD_E_SHAPE;
    const RadTensor &h = a->t[0], &fn = a->t[1], &base = a->t[2], &scale = a->t[3], &x = a->t[4];
    const RadTensor* post = head ? nullptr : rad_arg_in(a, 5);
    const int64_t width = hc * n, mix = head ? hc : 2 * hc;
    if (!row_width(h, width) || !is_float(h)) return RAD_E_SHAPE;
    const int64_t rows = h.shape[0];
    if (rows < 1 || !is_rank(fn, 2, width) || fn.shape[0] != mix || fn.dtype != RAD_F32 || fn.stride[1] != 1 ||
        !is_rank(base, 1) || base.shape[0] != mix || base.dtype != RAD_F32 || !is_rank(scale, 1) ||
        scale.shape[0] != (head ? 1 : 2) || scale.dtype != RAD_F32 || !is_rank(x, 2, n) || x.shape[0] != rows ||
        !is_float(x))
        return RAD_E_SHAPE;
    if (!head && (!post || !is_rank(*post, 2, hc) || post->shape[0] != rows || post->dtype != RAD_F32))
        return RAD_E_SHAPE;
    const uint32_t lds = width <= 14336 ? (uint32_t)(width * sizeof(float)) : 0u;
    /* Hy4 has hc 4: 8 products, 4 with head. Other widths keep hc_pre_k. */
    const int form = hc_form(a);
    if ((mix == 8 || mix == 4) && form != 1 && a->scratch &&
        a->scratch_bytes >= hc_part_bytes(rows, mix)) {
        /* rows a wave carries: 1 up to 8 rows (most waves), 4 above (shares the fn loads; 48 rows
         * 16.3 -> 14.0 us). `hc_r` overrides it for tests/bench_hc.py. */
        const int64_t rq = pi(a, "hc_r", rows <= 8 ? 1 : 4);
        const int R = rq == 4 ? 4 : rq == 2 ? 2 : 1;
        float* part = (float*)a->scratch;
        auto pk = mix == 8 ? (R == 1 ? part_kernel<8, 1>(dcode(h)) : R == 2 ? part_kernel<8, 2>(dcode(h)) : part_kernel<8, 4>(dcode(h)))
                           : (R == 1 ? part_kernel<4, 1>(dcode(h)) : R == 2 ? part_kernel<4, 2>(dcode(h)) : part_kernel<4, 4>(dcode(h)));
        int rc = emit(s, pk, dim3((unsigned)(kVWaves * (mix + 1)), (unsigned)((rows + R - 1) / R)), dim3(32), 0u, h.data,
                      h.stride[0], h.stride[h.rank - 1], (const float*)fn.data, fn.stride[0], part, (int)rows,
                      (int)width);
        if (rc != RAD_OK) return rc;
        return emit(s, mix == 8 ? hc_pre_fin_k<8> : hc_pre_fin_k<4>,
                    dim3((unsigned)((n + kThreads1 - 1) / kThreads1), (unsigned)rows), dim3(kThreads1), 0u, h.data,
                    dcode(h), h.stride[0], h.stride[h.rank - 1], (const float*)part, (const float*)base.data,
                    base.stride[0], (const float*)scale.data, scale.stride[0], x.data, dcode(x), x.stride[0],
                    x.stride[1], head ? nullptr : (float*)post->data, head ? 0 : post->stride[0],
                    head ? 0 : post->stride[1], (int)n, (int)hc, (int)head, eps, heps, mag);
    }
    if (mix == 8 || mix == 4)
        return emit(s, mix == 8 ? hc_pre1_k<8> : hc_pre1_k<4>, dim3((unsigned)rows), dim3(kThreads1), lds, h.data,
                     dcode(h), h.stride[0], h.stride[h.rank - 1], (const float*)fn.data, fn.stride[0],
                     (const float*)base.data, base.stride[0], (const float*)scale.data, scale.stride[0], x.data,
                     dcode(x), x.stride[0], x.stride[1], head ? nullptr : (float*)post->data,
                     head ? 0 : post->stride[0], head ? 0 : post->stride[1], (int)n, (int)hc, (int)head, eps, heps,
                     mag);
    return emit(s, hc_pre_k, dim3((unsigned)rows), dim3(kThreads), lds, h.data,
                 dcode(h), h.stride[0], h.stride[h.rank - 1], (const float*)fn.data, fn.stride[0],
                 (const float*)base.data, base.stride[0], (const float*)scale.data, scale.stride[0], x.data, dcode(x),
                 x.stride[0], x.stride[1], head ? nullptr : (float*)post->data, head ? 0 : post->stride[0],
                 head ? 0 : post->stride[1], (int)n, (int)hc, (int)head, eps, heps, mag);
}

int hy4_dev_hc_post(const RadArgs* a, RadStream s) {
    const int64_t n = pi(a, "n"), hc = pi(a, "hc");
    if (n < 1 || hc < 1 || !rad_args_have(a, 3)) return RAD_E_SHAPE;
    const RadTensor &h = a->t[0], &y = a->t[1], &post = a->t[2];
    if (!row_width(h, hc * n) || !is_float(h)) return RAD_E_SHAPE;
    const int64_t rows = h.shape[0];
    if (rows < 1 || !is_rank(y, 2, n) || y.shape[0] != rows || !is_float(y) || !is_rank(post, 2, hc) ||
        post.shape[0] != rows || post.dtype != RAD_F32)
        return RAD_E_SHAPE;
    const unsigned bx = (unsigned)((hc * n + 255) / 256);
    return emit(s, hc_post_k, dim3(bx, (unsigned)rows), dim3(256), 0u, h.data, dcode(h), h.stride[0],
                 h.stride[h.rank - 1], y.data, dcode(y), y.stride[0], y.stride[1], (const float*)post.data,
                 post.stride[0], post.stride[1], (int)n, (int)hc);
}

int hy4_dev_mla_kv(const RadArgs* a, RadStream s) {
    const int64_t H = pi(a, "heads"), nope = pi(a, "nope"), rope = pi(a, "rope"), vd = pi(a, "v_dim");
    if (H < 1 || nope < 1 || rope < 1 || vd < 1 || !rad_args_have(a, 4)) return RAD_E_SHAPE;
    const RadTensor &kvx = a->t[0], &kr = a->t[1], &k = a->t[2], &v = a->t[3];
    const int64_t per = nope + vd;
    if (!row_width(kvx, H * per) || !is_rank(kr, 2, rope) || !row_width(k, H * (rope + nope)) ||
        !row_width(v, H * vd) || !is_float(kvx) || !is_float(kr) || !is_float(k) || !is_float(v))
        return RAD_E_SHAPE;
    const int64_t rows = kvx.shape[0];
    if (rows < 1 || kr.shape[0] != rows || k.shape[0] != rows || v.shape[0] != rows) return RAD_E_SHAPE;
    const int64_t w = std::max(H * (rope + nope), H * vd);
    return emit(s, mla_kv_k, dim3((unsigned)((w + 255) / 256), (unsigned)rows), dim3(256), 0u, kvx.data,
                 dcode(kvx), kvx.stride[0], kvx.stride[kvx.rank - 1], kr.data, dcode(kr), kr.stride[0],
                 kr.stride[1], k.data, dcode(k), k.stride[0], k.stride[k.rank - 1], v.data, dcode(v), v.stride[0],
                 v.stride[v.rank - 1], (int)H, (int)nope, (int)rope, (int)vd);
}
