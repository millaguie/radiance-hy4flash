/* dev_idx.hip -- the indexer's card rows: hy4_idx_pool, hy4_idx_score and hy4_idx_select
 * (docs/OPS-HY4.md). Same arithmetic as ref_idx.cpp. Selection uses no heap: it finds the exact
 * top-k threshold by radix over the float's sortable key, and breaks ties by lower index, so it
 * picks the same set as the host. */
#include "dev_common.h"

#include <cstdlib>

namespace hy4k {
namespace {

/* The rope's inv_freq, computed on the host the same way as ref_idx.cpp's rope_tail (std::pow in
 * float): the card's powf is off by 1-2 ulp and the phase p·inv multiplies that by the position. */
/* `c`: the M-RoPE component each frequency reads (0 without `sections`, 1-D position). */
/* Frequencies by value and the two M-RoPE sections; each frequency's component is derived from
 * the sections in rope_pair. With one more int[64] table, idx_pool_k's argument segment went over
 * the backend's 768 bytes (hidden ones included) and the card refused the op with images. */
struct InvFreq {
    float v[64];
    int s1, s2;   // 0, 0: 1-D positions, every frequency goes to component 0
};
/* -1: malformed `sections`. */
inline int inv_freq(const RadArgs* a, double theta, int64_t rd, InvFreq* t) {
    *t = InvFreq{};
    for (int64_t i = 0; i < rd / 2 && i < 64; ++i)
        t->v[i] = 1.0f / std::pow((float)theta, (float)(2 * i) / (float)rd);
    const char* v = rad_args_gets(a, "sections");
    if (!v || !*v) return 0;
    int64_t sec[3] = {};
    int ns = 0;
    for (const char* q = v; *q && ns < 3;) {
        while (*q == ' ' || *q == ',') ++q;
        if (!*q) break;
        char* e = nullptr;
        const long x = std::strtol(q, &e, 10);
        if (e == q || x < 0) return -1;
        sec[ns++] = x;
        q = e;
    }
    if (ns != 3) return -1;
    t->s1 = (int)sec[1];
    t->s2 = (int)sec[2];
    return 1;
}

/* The adjacent-pair rope of the last `rd` channels, like ref_idx.cpp's rope_tail; `p` holds the
 * three components of the position (equal for text). */
__device__ inline void rope_pair(float& a, float& b, int i, const InvFreq& inv_t, const long* p, bool bf16) {
    const int comp = (i % 3 == 1 && i < 3 * inv_t.s1) ? 1 : (i % 3 == 2 && i < 3 * inv_t.s2) ? 2 : 0;
    const float f = (float)p[comp] * inv_t.v[i];
    float c = cosf(f), s = sinf(f);
    if (bf16) {
        c = bf2f(f2bf(c));
        s = bf2f(f2bf(s));
    }
    const float x = a, y = b;
    a = x * c - y * s;
    b = y * c + x * s;
}

/* One block of n threads (n ≤ 256) per group: thread o computes output o. */
__global__ void idx_pool_k(const void* stg, int sdt, long sp, long ss, const float* wv, long wvp, const float* wg,
                           long wgp, const float* ape, long ap, const void* wn, int wnd, long wns, const void* page,
                           int pit, long pgs, const void* bpos, int bit, long bps, long bpc, int mc, void* cache,
                           int cdt, long cp,
                           long pages, int n, int ratio, int rd, InvFreq inv_t, float eps) {
    __shared__ float k[16 * 256];
    __shared__ float red[8];
    __shared__ float out[256];
    const long b = blockIdx.x;
    const long pg = dldi(page, pit, b * pgs);
    long bp3[3];
    for (int c = 0; c < 3; ++c) bp3[c] = dldi(bpos, bit, b * bps + (mc ? (long)c * bpc : 0));
    const long bp = bp3[0];
    if (pg < 0 || pg >= pages || bp < 0) return;   // the host refuses bpos < 0 with a page
    const int o = threadIdx.x;
    for (int j = 0; j < ratio; ++j)
        if (o < n) k[j * n + o] = dld(stg, sdt, (b * ratio + j) * sp + (long)o * ss);
    __syncthreads();
    float val = 0;
    if (o < n) {
        float vv[16], gg[16], mx = -INFINITY;
        for (int j = 0; j < ratio; ++j) {
            float dv = 0, dg = 0;
            for (int i = 0; i < n; ++i) {
                const float x = k[j * n + i];
                dv += x * wv[(long)o * wvp + i];
                dg += x * wg[(long)o * wgp + i];
            }
            vv[j] = dv;
            gg[j] = dg + ape[(long)j * ap + o];
            mx = fmaxf(mx, gg[j]);
        }
        float den = 0, num = 0;
        for (int j = 0; j < ratio; ++j) {
            const float e = expf(gg[j] - mx);
            den += e;
            num += vv[j] * e;
        }
        val = num / den;
    }
    float sq = o < n ? val * val : 0.0f;
    for (int d = 16; d; d >>= 1) sq += __shfl_xor(sq, d, 32);
    if ((o & 31) == 0) red[o >> 5] = sq;
    __syncthreads();
    float tot = 0;
    for (int i = 0; i < (int)(blockDim.x + 31) / 32; ++i) tot += red[i];
    const float inv = 1.0f / sqrtf(tot / (float)n + eps);
    if (o < n) {
        /* rounding to the cache's dtype before the rope, and after */
        dst(cache, cdt, pg * cp + o, val * inv * dld(wn, wnd, (long)o * wns));
        out[o] = dld(cache, cdt, pg * cp + o);
    }
    __syncthreads();
    const int off = n - rd, half = rd / 2;
    if (o < half) {
        float a = out[off + 2 * o], c = out[off + 2 * o + 1];
        rope_pair(a, c, o, inv_t, bp3, cdt == 1);
        out[off + 2 * o] = a;
        out[off + 2 * o + 1] = c;
    }
    __syncthreads();
    if (o < n) dst(cache, cdt, pg * cp + o, out[o]);
}

/* One 256-thread block per row: rotated q into LDS, then one group per thread. */
__global__ __launch_bounds__(256) void idx_score_k(const void* q, int qdt, long qp, long qh, long qs, const void* w,
                                                   int wdt, long wp, long wsc, const void* pos, int pit, long ps,
                                                   long ps1, int mcv,
                                                   const void* cache, int cdt, long cp, long pages, const void* bt,
                                                   int bit, long btp, long btw, const void* cu, int cit, long cs,
                                                   int S, const void* nc, int nit, long ns, float* score, long scp,
                                                   long scs, int blocks, int n, int H, int rd, InvFreq inv_t, float esc,
                                                   int rows) {
    /* mcv: bit 0, 3-D positions; bit 1, 8-wide bf16 key loads. One int because the kernel's
     * arguments, hidden ones included, may not exceed the backend's 768 bytes (tools/kernarg.sh). */
    const int mc = mcv & 1, vec = mcv >> 1;
    __shared__ float qq[32 * 256];
    __shared__ float wh[32];
    const long r = blockIdx.x;
    int s = -1;
    for (int i = 0; i < S; ++i) {
        const long a = dldi(cu, cit, (long)i * cs), b = dldi(cu, cit, (long)(i + 1) * cs);
        if (a >= 0 && b <= rows && r >= a && r < b) { s = i; break; }
    }
    if (s < 0) return;  // row outside cu's slices: nothing is written
    const bool bf16 = qdt == 1;
    for (int i = threadIdx.x; i < H * n; i += 256) {
        const int h = i / n, c = i - h * n;
        qq[i] = dld(q, qdt, r * qp + (long)h * qh + (long)c * qs);
    }
    if (threadIdx.x < H) wh[threadIdx.x] = dld(w, wdt, r * wp + (long)threadIdx.x * wsc);
    __syncthreads();
    long p3[3];
    for (int c = 0; c < 3; ++c) p3[c] = mc ? dldi(pos, pit, (long)c * ps + r * ps1) : dldi(pos, pit, r * ps);
    const int half = rd / 2, off = n - rd;
    for (int i = threadIdx.x; i < H * half; i += 256) {
        const int h = i / half, j = i - h * half;
        float a = qq[h * n + off + 2 * j], b = qq[h * n + off + 2 * j + 1];
        rope_pair(a, b, j, inv_t, p3, bf16);
        qq[h * n + off + 2 * j] = a;
        qq[h * n + off + 2 * j + 1] = b;
    }
    __syncthreads();
    if (bf16)
        for (int i = threadIdx.x; i < H * n; i += 256) qq[i] = bf2f(f2bf(qq[i]));
    __syncthreads();
    long ncs = dldi(nc, nit, (long)s * ns);
    if (ncs > btw) ncs = btw;
    /* Each key group is read once and the H heads accumulate together: the old loop read the group
     * once per head (16x). Every head still sums over i in ascending order, so the bits are the old
     * kernel's. With few rows, gridDim.y blocks share a row's groups, which changes no sum. */
    /* the launch's split (hy4_dev_idx_score): reading gridDim would add hidden arguments */
    const int gy = rows >= 64 ? 1 : min(16, max(1, (blocks + 255) / 256));
#if defined(__GFX12__)
    /* Hy4's geometry (16 heads, n 128, bf16 keys in 16-byte rows) on gfx12 BF16 WMMA: per 16-group
     * tile, S = Q K^T with the heads as the matrix rows (8 steps of 16x16x16), then relu and the
     * weighted sum over heads. A wave reads a group's key once for all 16 heads in fragments instead
     * of a thread walking it alone; 46 ms a 2048-row call at 56K context was 22% of the prefill.
     * The products are the same bf16 values, but the sums run in another order, so the scores (and
     * any selection tie) can differ from the scalar loop's in the last bits. */
    if (vec && bf16 && H == 16 && n == 128) {
        typedef __bf16 v8bfi __attribute__((ext_vector_type(8)));
        typedef float v8fi __attribute__((ext_vector_type(8)));
        const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
        v8bfi af[8];
#pragma unroll
        for (int kb = 0; kb < 8; ++kb) {   // A fragment: row = head lane % 16, k = 8*(e>>2) + 4*(lane>>4) + (e&3)
            union { uint16_t h[8]; v8bfi f; } u;
            const float* qrow = qq + (lane & 15) * n + kb * 16 + 4 * (lane >> 4);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                u.h[e] = f2bf(qrow[e]);
                u.h[4 + e] = f2bf(qrow[8 + e]);
            }
            af[kb] = u.f;
        }
        float wl[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) wl[e] = wh[8 * (lane >> 4) + e] * esc;
        const int tiles = (blocks + 15) / 16;
        for (int t = blockIdx.y * 8 + wave; t < tiles; t += 8 * gy) {
            const int g = t * 16 + (lane & 15);
            const long pg = g < ncs && g < blocks ? dldi(bt, bit, (long)s * btp + g) : -1;
            const bool ok = pg >= 0 && pg < pages;
            const uint16_t* krow = (const uint16_t*)cache + (ok ? pg * cp : 0) + 4 * (lane >> 4);
            v8fi dot = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int kb = 0; kb < 8; ++kb) {
                union { uint2 w[2]; v8bfi f; } u;
                const uint2* pw = (const uint2*)(krow + kb * 16);
                u.w[0] = ok ? pw[0] : make_uint2(0, 0);
                u.w[1] = ok ? pw[2] : make_uint2(0, 0);
                dot = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(af[kb], u.f, dot);
            }
            float part = 0.0f;
#pragma unroll
            for (int e = 0; e < 8; ++e) part += wl[e] * (dot[e] < 0.0f ? 0.0f : dot[e]);   // NaN propagates
            const float val = part + __shfl_xor(part, 16, 32);   // heads 0-7, then 8-15
            if (lane < 16 && g < blocks) score[r * scp + (long)g * scs] = ok ? val : -INFINITY;
        }
        return;
    }
#endif
    for (int g = blockIdx.y * 256 + threadIdx.x; g < blocks; g += 256 * gy) {
        float val = -INFINITY;
        const long pg = g < ncs ? dldi(bt, bit, (long)s * btp + g) : -1;
        if (pg >= 0 && pg < pages) {
            float dot[32];
#pragma unroll
            for (int h = 0; h < 32; ++h) dot[h] = 0.0f;
            if (vec) {
                const uint4* kp = (const uint4*)((const uint16_t*)cache + pg * cp);
                for (int i0 = 0; i0 < n; i0 += 8) {
                    const uint4 u = kp[i0 / 8];
                    const uint32_t wv[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        const float k = __uint_as_float(e & 1 ? wv[e / 2] & 0xffff0000u : wv[e / 2] << 16);
#pragma unroll
                        for (int h = 0; h < 32; ++h)
                            if (h < H) dot[h] += qq[h * n + i0 + e] * k;
                    }
                }
            } else {
                for (int i = 0; i < n; ++i) {
                    const float k = dld(cache, cdt, pg * cp + i);
#pragma unroll
                    for (int h = 0; h < 32; ++h)
                        if (h < H) dot[h] += qq[h * n + i] * k;
                }
            }
            val = 0;
#pragma unroll
            for (int h = 0; h < 32; ++h)
                if (h < H) val += wh[h] * esc * (dot[h] < 0.0f ? 0.0f : dot[h]);   // like std::max: NaN propagates
        }
        score[r * scp + (long)g * scs] = val;
    }
}

/* Sortable key of a float (larger float, larger key); -0 and +0 share one. */
__device__ inline uint32_t sort_key(float v) {
    if (v == 0.0f) v = 0.0f;
    const uint32_t u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

/* One 256-thread block per row. Candidates g < min(nc, pos/ratio, blocks) with page ≥ 0 and a
 * finite score. The best k = topk-1 are picked (higher score; at equal score, lower index), in
 * index order, and the row's own group at the end. */
__global__ __launch_bounds__(256) void idx_select_k(const float* score, long scp, long scs, const void* bt, int bit,
                                                    long btp, long btw, const void* nc, int nit, long ns,
                                                    const void* pos, int pit, long ps, const void* cu, int cit,
                                                    long cs, int S, void* sel, int sit, long slp, long sls,
                                                    void* nsel, int nsit, long nss, void* used, int uit, long us,
                                                    int blocks, int topk, int ratio, int rows) {
    __shared__ uint32_t cnt[256];
    __shared__ uint32_t thresh_s;
    __shared__ int pre_gt[257], pre_eq[257];
    const long r = blockIdx.x;
    const int t = threadIdx.x;
    int s = -1;
    for (int i = 0; i < S; ++i) {
        const long a = dldi(cu, cit, (long)i * cs), b = dldi(cu, cit, (long)(i + 1) * cs);
        if (a >= 0 && b <= rows && r >= a && r < b) { s = i; break; }
    }
    if (s < 0) return;
    const long p = dldi(pos, pit, r * ps);
    const long own = p >= 0 ? p / ratio : -1;
    long ncs = dldi(nc, nit, (long)s * ns);
    if (ncs < 0) ncs = 0;
    long nca = ncs < own ? ncs : own;
    if (nca > blocks) nca = blocks;
    if (nca > btw) nca = btw;
    if (nca < 0) nca = 0;
    const int K = topk - 1;
    auto valid = [&](long g, uint32_t* key) -> bool {
        const float v = score[r * scp + g * scs];
        if (!isfinite(v) || dldi(bt, bit, (long)s * btp + g) < 0) return false;
        *key = sort_key(v);
        return true;
    };
    /* how many are valid */
    uint32_t nv = 0;
    for (long g = t; g < nca; g += 256) { uint32_t k; nv += valid(g, &k); }
    cnt[t] = nv;
    __syncthreads();
    for (int d = 128; d; d >>= 1) { if (t < d) cnt[t] += cnt[t + d]; __syncthreads(); }
    const uint32_t total = cnt[0];
    __syncthreads();
    /* threshold: the largest key T with #(key ≥ T) ≥ K; with fewer than K valid, all (T = 0) */
    uint32_t T = 0;
    if (total > (uint32_t)K) {
        for (int bitn = 31; bitn >= 0; --bitn) {
            const uint32_t cand = T | (1u << bitn);
            uint32_t c = 0;
            for (long g = t; g < nca; g += 256) { uint32_t k; if (valid(g, &k) && k >= cand) ++c; }
            cnt[t] = c;
            __syncthreads();
            for (int d = 128; d; d >>= 1) { if (t < d) cnt[t] += cnt[t + d]; __syncthreads(); }
            if (cnt[0] >= (uint32_t)K) T = cand;
            __syncthreads();
        }
    }
    if (t == 0) thresh_s = T;
    __syncthreads();
    T = thresh_s;
    /* Contiguous slices per thread, to write in index order. */
    const long span = (nca + 255) / 256, g0 = t * span, g1 = g0 + span < nca ? g0 + span : nca;
    int ngt = 0, neq = 0;
    for (long g = g0; g < g1; ++g) {
        uint32_t k;
        if (valid(g, &k)) { if (k > T) ++ngt; else if (k == T) ++neq; }
    }
    pre_gt[t + 1] = ngt;
    pre_eq[t + 1] = neq;
    if (t == 0) { pre_gt[0] = 0; pre_eq[0] = 0; }
    __syncthreads();
    if (t == 0)
        for (int i = 1; i <= 256; ++i) { pre_gt[i] += pre_gt[i - 1]; pre_eq[i] += pre_eq[i - 1]; }
    __syncthreads();
    const int tot_gt = pre_gt[256], tot_eq = pre_eq[256];
    /* with total ≤ K all get in: T = 0 and every valid one is > 0 or == 0 */
    const int eq_quota = total > (uint32_t)K ? K - tot_gt : tot_eq;
    const int chosen = total > (uint32_t)K ? K : (int)total;
    /* output position of each picked one: those picked before this thread's slice */
    int eq_before = pre_eq[t];
    int out_i = pre_gt[t] + (eq_before < eq_quota ? eq_before : eq_quota);
    for (long g = g0; g < g1; ++g) {
        uint32_t k;
        if (!valid(g, &k)) continue;
        bool takes = k > T;
        if (k == T) { takes = eq_before < eq_quota; ++eq_before; }
        if (takes) dsti(sel, sit, r * slp + (long)(out_i++) * sls, dldi(bt, bit, (long)s * btp + g));
    }
    if (t == 0) {
        const long own_pg = own >= 0 && own < btw ? dldi(bt, bit, (long)s * btp + own) : -1;
        dsti(sel, sit, r * slp + (long)chosen * sls, own_pg);
        dsti(nsel, nsit, r * nss, chosen + 1);
        dsti(used, uit, r * us, (long)chosen * ratio + (p >= 0 ? p % ratio : 0) + 1);
    }
    for (int i = chosen + 1 + t; i < topk; i += 256) dsti(sel, sit, r * slp + (long)i * sls, -1);
}

}  // namespace
}  // namespace hy4k

using namespace hy4k;

int hy4_dev_idx_pool(const RadArgs* a, RadStream s) {
    const int64_t n = pi(a, "n"), ratio = pi(a, "ratio", 4), rd = pi(a, "rotary_dim");
    const int64_t bs = pi(a, "block_size", ratio);
    const double theta = pf(a, "theta", 1e7);
    const float eps = (float)pf(a, "eps", 1e-5);
    if (n < 1 || n > 256 || ratio < 1 || ratio > 16 || bs != ratio || rd < 2 || rd > n || rd > 128 || rd % 2 ||
        !rad_args_have(a, 8))
        return RAD_E_SHAPE;
    const RadTensor &stg = a->t[0], &wv = a->t[1], &wg = a->t[2], &ape = a->t[3], &wn = a->t[4];
    const RadTensor &page = a->t[5], &bpos = a->t[6], &cache = a->t[7];
    if (!is_rank(stg, 2, n) || !is_float(stg) || !is_rank(wv, 2, n) || wv.shape[0] != n || wv.dtype != RAD_F32 ||
        wv.stride[1] != 1 || !is_rank(wg, 2, n) || wg.shape[0] != n || wg.dtype != RAD_F32 || wg.stride[1] != 1 ||
        !is_rank(ape, 2, n) || ape.shape[0] != ratio || ape.dtype != RAD_F32 || ape.stride[1] != 1 ||
        !is_rank(wn, 1, n) || !is_float(wn) || !is_rank(page, 1) || !is_int(page) ||
        !is_int(bpos) || !cache_ok(cache, n) || !is_float(cache))
        return RAD_E_SHAPE;
    InvFreq inv_t;
    const int sc = inv_freq(a, theta, rd, &inv_t);
    if (sc < 0 || bpos.rank > 2 || (bpos.rank == 2 && bpos.shape[1] != 1 && (sc != 1 || bpos.shape[1] < 3)))
        return RAD_E_SHAPE;
    const int mc = sc == 1 && bpos.rank == 2 && bpos.shape[1] >= 3;
    if (!mc) inv_t.s1 = inv_t.s2 = 0;
    const int64_t M = page.shape[0];
    if (M < 1 || bpos.shape[0] != M || stg.shape[0] < M * ratio) return RAD_E_SHAPE;
    const unsigned threads = (unsigned)((n + 31) / 32 * 32);
    return emit(s, idx_pool_k, dim3((unsigned)M), dim3(threads), 0u, stg.data, dcode(stg), stg.stride[0],
                 stg.stride[1], (const float*)wv.data, wv.stride[0], (const float*)wg.data, wg.stride[0],
                 (const float*)ape.data, ape.stride[0], wn.data, dcode(wn), wn.stride[0], page.data, icode(page),
                 page.stride[0], bpos.data, icode(bpos), bpos.stride[0], mc ? bpos.stride[1] : 0, mc, cache.data,
                 dcode(cache), cache.stride[0], cache.shape[0], (int)n, (int)ratio, (int)rd, inv_t, eps);
}

int hy4_dev_idx_score(const RadArgs* a, RadStream s) {
    const int64_t n = pi(a, "n"), H = pi(a, "heads"), rd = pi(a, "rotary_dim");
    const double theta = pf(a, "theta", 1e7);
    if (n < 1 || n > 256 || H < 1 || H > 32 || H * n > 32 * 256 || rd < 2 || rd > n || rd > 128 || rd % 2 ||
        !rad_args_have(a, 8))
        return RAD_E_SHAPE;
    const RadTensor &q = a->t[0], &w = a->t[1], &pos = a->t[2], &cache = a->t[3];
    const RadTensor &bt = a->t[4], &cu = a->t[5], &nc = a->t[6], &score = a->t[7];
    const bool q3 = q.rank == 3;
    if (!(q3 ? is_rank(q, 3, n) && q.shape[1] == H : is_rank(q, 2, H * n)) || !is_float(q) || !is_rank(w, 2, H) ||
        !is_float(w) || !is_int(pos) || !cache_ok(cache, n) || !is_float(cache) ||
        !is_rank(bt, 2) || !is_int(bt) || bt.stride[1] != 1 || !is_rank(nc, 1) || !is_int(nc) || !is_rank(score, 2) ||
        score.dtype != RAD_F32 || !is_present(cu) || !is_int(cu) || cu.rank != 1)
        return RAD_E_SHAPE;
    InvFreq inv_t;
    const int sc = inv_freq(a, theta, rd, &inv_t);
    if (sc < 0 || pos.rank > 2 || (pos.rank == 2 && (sc != 1 || pos.shape[0] < 3))) return RAD_E_SHAPE;
    const int mc = sc == 1 && pos.rank == 2;
    if (!mc) inv_t.s1 = inv_t.s2 = 0;
    const int64_t pM = pos.rank == 2 ? pos.shape[1] : pos.shape[0];
    const int64_t M = q.shape[0], blocks = score.shape[1];
    if (M < 1 || w.shape[0] != M || pM != M || score.shape[0] != M) return RAD_E_SHAPE;
    const int64_t S = cu.shape[0] - 1;
    if (S < 1 || bt.shape[0] < S || nc.shape[0] < S) return RAD_E_SHAPE;
    const int64_t qs = q.stride[q.rank - 1], qh = q3 ? q.stride[1] : n * qs;
    /* bf16 keys in 16-byte rows load 8 values at a time. With few rows (decode), up to 16 blocks a row
     * share the groups: one block over 49K groups at 196K context was the indexer's decode cost. */
    const int vec = cache.dtype == RAD_BF16 && n % 8 == 0 && cache.stride[0] % 8 == 0 &&
                    ((uintptr_t)cache.data & 15u) == 0 && cache.stride[cache.rank - 1] == 1;
    const unsigned gy = M >= 64 ? 1u : (unsigned)std::min<int64_t>(16, std::max<int64_t>(1, (blocks + 255) / 256));
    return emit(s, idx_score_k, dim3((unsigned)M, gy), dim3(256), 0u, q.data, dcode(q), q.stride[0], qh, qs, w.data,
                 dcode(w), w.stride[0], w.stride[1], pos.data, icode(pos), pos.stride[0], mc ? pos.stride[1] : 0, (int)(mc | (vec << 1)),
                 cache.data, dcode(cache),
                 cache.stride[0], cache.shape[0], bt.data, icode(bt), bt.stride[0], bt.shape[1], cu.data, icode(cu),
                 cu.stride[0], (int)S, nc.data, icode(nc), nc.stride[0], (float*)score.data, score.stride[0],
                 score.stride[1], (int)blocks, (int)n, (int)H, (int)rd, inv_t,
                 (float)(1.0 / std::sqrt((double)H) / std::sqrt((double)n)), (int)M);
}

int hy4_dev_idx_select(const RadArgs* a, RadStream s) {
    const int64_t topk = pi(a, "topk"), ratio = pi(a, "ratio", 4);
    if (topk < 2 || topk > kMaxTopk || ratio < 1 || !rad_args_have(a, 8)) return RAD_E_SHAPE;
    const RadTensor &score = a->t[0], &bt = a->t[1], &nc = a->t[2], &pos = a->t[3], &cu = a->t[4];
    const RadTensor &sel = a->t[5], &nsel = a->t[6], &used = a->t[7];
    if (!is_rank(score, 2) || score.dtype != RAD_F32 || !is_rank(bt, 2) || !is_int(bt) || bt.stride[1] != 1 ||
        !is_rank(nc, 1) || !is_int(nc) || !is_rank(pos, 1) || !is_int(pos) || !is_rank(sel, 2, topk) || !is_int(sel) ||
        !is_rank(nsel, 1) || !is_int(nsel) || !is_rank(used, 1) || !is_int(used) || !is_present(cu) || !is_int(cu) ||
        cu.rank != 1)
        return RAD_E_SHAPE;
    const int64_t M = score.shape[0], blocks = score.shape[1];
    if (M < 1 || pos.shape[0] != M || sel.shape[0] != M || nsel.shape[0] != M || used.shape[0] != M)
        return RAD_E_SHAPE;
    const int64_t S = cu.shape[0] - 1;
    if (S < 1 || bt.shape[0] < S || nc.shape[0] < S) return RAD_E_SHAPE;
    return emit(s, idx_select_k, dim3((unsigned)M), dim3(256), 0u, (const float*)score.data, score.stride[0],
                 score.stride[1], bt.data, icode(bt), bt.stride[0], bt.shape[1], nc.data, icode(nc), nc.stride[0],
                 pos.data, icode(pos), pos.stride[0], cu.data, icode(cu), cu.stride[0], (int)S, sel.data, icode(sel),
                 sel.stride[0], sel.stride[1], nsel.data, icode(nsel), nsel.stride[0], used.data, icode(used),
                 used.stride[0], (int)blocks, (int)topk, (int)ratio, (int)M);
}
