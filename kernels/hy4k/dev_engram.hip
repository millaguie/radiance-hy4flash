/* dev_engram.hip -- card rows of hy4_ngram_ids, hy4_rows_rmsnorm and hy4_engram_gate
 * (docs/OPS-HY4.md). Same arithmetic as ref_engram.cpp; the reductions run in another order. */
#include "dev_common.h"

namespace hy4k {
namespace {

constexpr int kThreads = 256;

/* E4M3 with round to nearest even, by value: |q| ≤ 448 already clipped. Same as
 * rad_f32_to_fp8e4m3 for those inputs (subnormals in steps of 2^-9, normals with 3 mantissa
 * bits). */
__device__ inline float e4m3(float q) {
    const float a = fabsf(q);
    if (a == 0.0f) return q;
    float r;
    if (a < 0.015625f) {                 // < 2^-6: subnormal
        r = rintf(a * 512.0f) * (1.0f / 512.0f);
    } else {
        int ex;
        frexpf(a, &ex);                  // a = f·2^ex, f in [0.5, 1)
        const float pitch = ldexpf(1.0f, ex - 1 - 3);
        r = rintf(a / pitch) * pitch;
    }
    r = fminf(r, 448.0f);
    return copysignf(r, q);
}

/* One 128-thread block per (row, head); each thread carries up to 4 values of the slice. */
template <class Ch>
__global__ __launch_bounds__(128) void rows_rmsnorm_k(const void* x, int xdt, long xp, long xs, const void* w, int wdt,
                                                      long wh, long wi, void* y, int ydt, long yp, long ys, int d,
                                                      int fp8, float eps) {
    HY4_CHAIN_BODY {
    __shared__ float red[4];
    const long r = blockIdx.x;
    const int h = blockIdx.y;
    float v[4];
    float amax = 0;
    for (int k = 0; k < 4; ++k) {
        const int i = threadIdx.x + k * 128;
        v[k] = i < d ? dld(x, xdt, r * xp + ((long)h * d + i) * xs) : 0.0f;
        amax = fmaxf(amax, fabsf(v[k]));
    }
    if (fp8) {
        for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor(amax, o, 32));
        lds_barrier();
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = amax;
        lds_barrier();
        amax = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
        const float sc = fmaxf(amax, 1e-12f) / 448.0f;
        for (int k = 0; k < 4; ++k) {
            const float q = fminf(448.0f, fmaxf(-448.0f, v[k] / sc));
            v[k] = v[k] + (e4m3(q) * sc - v[k]);
        }
    }
    float sq = 0;
    for (int k = 0; k < 4; ++k) sq += v[k] * v[k];
    sq = block_sum<128>(sq, red);
    const float inv = 1.0f / sqrtf(sq / (float)d + eps);
    for (int k = 0; k < 4; ++k) {
        const int i = threadIdx.x + k * 128;
        if (i < d) dst(y, ydt, r * yp + ((long)h * d + i) * ys, v[k] * inv * dld(w, wdt, (long)h * wh + (long)i * wi));
    }
    } HY4_CHAIN_END
}

/* One block per row. */
template <class Ch>
__global__ __launch_bounds__(kThreads) void engram_gate_k(void* h, int hdt, long hp, long hs, const void* tok, int tit,
                                                        long ts, const void* k, int kdt, long kp, long ks, const void* v,
                                                        int vdt, long vp, long vs, const void* wk, int wkd, long wkr,
                                                        long wkc, const void* wq, int wqd, long wqr, long wqc,
                                                        const void* gam, int gd, long gs, const void* per, int pd,
                                                        long pr, long pc, int n, int hc, long pad, long skip,
                                                        long skip2, float eps, float geps) {
    HY4_CHAIN_BODY {
    __shared__ float red[kThreads / 32];
    __shared__ float gate[16];
    const long r = blockIdx.x;
    {
        const long tr = dldi(tok, tit, r * ts);
        if ((pad >= 0 && tr == pad) || (skip >= 0 && tr == skip) || (skip2 >= 0 && tr == skip2)) return;
    }
    /* rms of v, and of k_j and h_j per stream */
    float sv = 0;
    for (int i = threadIdx.x; i < n; i += kThreads) {
        const float x = dld(v, vdt, r * vp + (long)i * vs);
        sv += x * x;
    }
    const float iv = 1.0f / sqrtf(block_sum<kThreads>(sv, red) / (float)n + eps);
    for (int j = 0; j < hc; ++j) {
        float sk = 0, sh = 0;
        for (int i = threadIdx.x; i < n; i += kThreads) {
            const float a = dld(k, kdt, r * kp + ((long)j * n + i) * ks);
            const float b = dld(h, hdt, r * hp + ((long)j * n + i) * hs);
            sk += a * a;
            sh += b * b;
        }
        const float ik = 1.0f / sqrtf(block_sum<kThreads>(sk, red) / (float)n + eps);
        const float ih = 1.0f / sqrtf(block_sum<kThreads>(sh, red) / (float)n + eps);
        float dot = 0;
        for (int i = threadIdx.x; i < n; i += kThreads) {
            const float key = dld(k, kdt, r * kp + ((long)j * n + i) * ks) * ik * dld(wk, wkd, (long)j * wkr + (long)i * wkc);
            const float qry = dld(h, hdt, r * hp + ((long)j * n + i) * hs) * ih * dld(wq, wqd, (long)j * wqr + (long)i * wqc);
            dot += key * qry;
        }
        dot = block_sum<kThreads>(dot, red);
        if (threadIdx.x == 0) {
            const float dd = dot / sqrtf((float)n);
            const float sg = (float)((dd > 0) - (dd < 0));
            gate[j] = dsigmoid(sqrtf(fmaxf(fabsf(dd), geps)) * sg);
        }
    }
    lds_barrier();
    for (int i = threadIdx.x; i < n; i += kThreads) {
        const float s = dld(gam, gd, (long)i * gs) * (dld(v, vdt, r * vp + (long)i * vs) * iv);
        for (int j = 0; j < hc; ++j) {
            const long o = r * hp + ((long)j * n + i) * hs;
            dst(h, hdt, o, dld(h, hdt, o) + (gate[j] * s + dld(per, pd, (long)j * pr + (long)i * pc) * s));
        }
    }
    } HY4_CHAIN_END
}

/* cu[0..s] does not decrease: if an earlier one exceeds cu[s], the slices would overlap. */
__device__ inline bool cu_ascending(const void* cu, int cit, long cs, int s) {
    const long r0 = dldi(cu, cit, (long)s * cs);
    for (int i = 0; i < s; ++i)
        if (dldi(cu, cit, (long)i * cs) > r0) return false;
    return true;
}

__device__ inline int64_t floor_mod_d(int64_t a, int64_t p) {
    const int64_t m = a % p;
    return (m != 0 && ((m < 0) != (p < 0))) ? m + p : m;
}

/* One thread per row of sequence blockIdx.y. It reads the state (the previous step's tokens) and
 * does not write it: ngram_state_k does that afterwards, on the same stream. */
template <class Ch>
__global__ void ngram_ids_k(const void* tok, int tit, long ts, const void* pos, int pit, long ps, const void* cu,
                            int cit, long cs, int rows, const void* st, int sit, long st_r, long st_c, long slots,
                            const void* sidx, int xit, long sx, const void* map, int mit, long V, const int64_t* mult,
                            const int64_t* vsz, const int64_t* off, void* ids, int iit, long s_e, long s_r, long s_h,
                            int E, int heads, int ngram, long pad, void* aids, long a_e, long a_r, long a_h, int A,
                            const int32_t* nacc, long ns, int W) {
    HY4_CHAIN_BODY {
    const int s = blockIdx.y;
    const long r0 = dldi(cu, cit, (long)s * cs), r1 = dldi(cu, cit, (long)(s + 1) * cs);
    /* cu increasing: a slice that goes backwards is skipped whole (the host refuses it) */
    if (r0 < 0 || r1 > rows || r1 < r0 || !cu_ascending(cu, cit, cs, s)) return;
    const long r = r0 + (long)blockIdx.x * blockDim.x + threadIdx.x;
    /* the last sequence runs on into the next prompt chunk (A tokens after the step in `tok`),
     * whose ids go to `aids` for the Engram read-ahead; its positions continue the sequence's */
    const long reach = r1 + (A > 0 && r1 == rows && r1 > r0 ? A : 0);   // the one non-empty sequence ending last
    if (r >= reach) return;
    const bool future = r >= r1;
    const long slot = dldi(sidx, xit, (long)s * sx);
    if (slot >= slots) return;        // slot outside state: the sequence is skipped (the host refuses it)
    const int hist = ngram - 1, per = heads / hist;
    /* the committed window, oldest first, at num_accepted - 1 (the host's rule, ref_engram.cpp) */
    const long o = nacc ? (long)nacc[(long)s * ns] - 1 : 0;
    if (o < 0 || o + hist > W) return;   // a read past the window: the sequence is skipped (the host refuses)
    int64_t P[8];
    for (int k = 0; k < hist; ++k) P[k] = slot >= 0 ? dldi(st, sit, slot * st_r + (o + k) * st_c) : pad;
    const long p = future ? dldi(pos, pit, (r1 - 1) * ps) + (r - r1 + 1) : dldi(pos, pit, r * ps);
    int64_t t[8];
    for (int kk = 0; kk < ngram; ++kk) {
        if (p - kk < 0) t[kk] = pad;
        else if (r - kk >= r0) t[kk] = dldi(tok, tit, (r - kk) * ts);
        else {
            const long back = kk - 1 - (r - r0);   // 0: the last committed id
            t[kk] = slot >= 0 ? P[hist - 1 - back] : pad;
        }
    }
    for (int e = 0; e < E; ++e) {
        const int64_t pad_c = dldi(map, mit, (long)e * V + pad);
        uint64_t mixed[8];
        for (int kk = 0; kk < ngram; ++kk) {
            const int64_t c = (t[kk] >= 0 && t[kk] < V) ? dldi(map, mit, (long)e * V + t[kk]) : pad_c;
            mixed[kk] = (uint64_t)c * (uint64_t)mult[(long)e * ngram + kk];
        }
        for (int b = 0; b < hist; ++b) {
            uint64_t mix = mixed[0];
            for (int kk = 1; kk <= b + 1; ++kk) mix ^= mixed[kk];
            for (int j = 0; j < per; ++j) {
                const int h = b * per + j;
                const int64_t pr = vsz[(long)e * heads + h];
                if (pr < 1) continue;     // vocab_sizes < 1: the host refuses it; nothing is written here
                const int64_t id = floor_mod_d((int64_t)mix, pr) + off[(long)e * heads + h];
                if (future) dsti(aids, iit, e * a_e + (r - r1) * a_r + h * a_h, id);
                else dsti(ids, iit, e * s_e + r * s_r + h * s_h, id);
            }
        }
    }
    } HY4_CHAIN_END
}

/* One thread per sequence: the last `hist` unmapped tokens, from the step or the previous state. */
template <class Ch>
__global__ void ngram_state_k(const void* tok, int tit, long ts, const void* cu, int cit, long cs, int rows, int S,
                              void* st, int sit, long st_r, long st_c, long slots, const void* sidx, int xit, long sx,
                              int hist, const int32_t* nacc, long ns, int W) {
    HY4_CHAIN_BODY {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= S) return;
    const long r0 = dldi(cu, cit, (long)s * cs), r1 = dldi(cu, cit, (long)(s + 1) * cs);
    if (r0 < 0 || r1 > rows || r1 <= r0 || !cu_ascending(cu, cit, cs, s)) return;
    const long slot = dldi(sidx, xit, (long)s * sx);
    if (slot < 0 || slot >= slots) return;
    const long n = r1 - r0;
    const long o = nacc ? (long)nacc[(long)s * ns] - 1 : 0;
    if (o < 0 || o + hist > W || (nacc && hist - 1 + n > W)) return;   // the host refuses these
    /* Fixed-index loops (hist <= 7): a runtime index into P would put it in scratch. */
    int64_t P[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) P[k] = k < hist ? dldi(st, sit, slot * st_r + (o + k) * st_c) : 0;
    if (nacc) {   // decode layout: the read window shifted down by one, then every id of the step
#pragma unroll
        for (int i = 0; i < 7; ++i)
            if (i + 1 < hist) dsti(st, sit, slot * st_r + i * st_c, P[i + 1]);
        for (long t = 0; t < n; ++t) dsti(st, sit, slot * st_r + (hist - 1 + t) * st_c, dldi(tok, tit, (r0 + t) * ts));
    } else {      // the step's last hist ids, oldest first, at offset zero: (P ++ step)[n + i]
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (i >= hist) continue;
            int64_t v = 0;
            if (n + i >= hist) v = dldi(tok, tit, (r0 + n + i - hist) * ts);
            else {
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    if (j == n + i) v = P[j];
            }
            dsti(st, sit, slot * st_r + i * st_c, v);
        }
    }
    } HY4_CHAIN_END
}

}  // namespace
}  // namespace hy4k

using namespace hy4k;

int hy4_dev_rows_rmsnorm(const RadArgs* a, RadStream s) {
    const int64_t H = pi(a, "heads"), d = pi(a, "d"), fp8 = pi(a, "fp8", 0);
    const float eps = (float)pf(a, "eps", 1.1920929e-07);
    if (H < 1 || d < 1 || d > 512 || (fp8 != 0 && fp8 != 1) || !rad_args_have(a, 3)) return RAD_E_SHAPE;
    const RadTensor &x = a->t[0], &w = a->t[1], &y = a->t[2];
    if (!row_width(x, H * d) || !row_width(y, H * d) || !is_matrix(w, H, d) || !is_float(x) || !is_float(y) ||
        !is_float(w))
        return RAD_E_SHAPE;
    const int64_t rows = x.shape[0];
    if (rows < 1 || y.shape[0] != rows) return RAD_E_SHAPE;
    const long wi = w.stride[w.rank - 1], wh = w.rank == 1 ? d * wi : w.stride[0];
    return emit(s, HY4_CHAINED(a, rows_rmsnorm_k), dim3((unsigned)rows, (unsigned)H), dim3(128), 0u, x.data, dcode(x), x.stride[0],
                 x.stride[x.rank - 1], w.data, dcode(w), wh, wi, y.data, dcode(y), y.stride[0], y.stride[y.rank - 1],
                 (int)d, (int)fp8, eps);
}

int hy4_dev_engram_gate(const RadArgs* a, RadStream s) {
    const int64_t n = pi(a, "n"), hc = pi(a, "hc"), pad = pi(a, "pad", -1);
    const int64_t skip = pi(a, "skip", -1), skip2 = pi(a, "skip2", -1);
    const float eps = (float)pf(a, "eps", 1.1920929e-07), geps = (float)pf(a, "gate_eps", 1e-6);
    if (n < 1 || n > kMaxN || hc < 1 || hc > 16 || !rad_args_have(a, 8)) return RAD_E_SHAPE;
    const RadTensor &h = a->t[0], &tok = a->t[1], &k = a->t[2], &v = a->t[3];
    const RadTensor &wk = a->t[4], &wq = a->t[5], &gam = a->t[6], &per = a->t[7];
    if (!row_width(h, hc * n) || !row_width(k, hc * n) || !is_rank(v, 2, n) || !is_rank(tok, 1) || !is_int(tok) ||
        !is_matrix(wk, hc, n) || !is_matrix(wq, hc, n) || !is_rank(gam, 1, n) || !is_matrix(per, hc, n) || !is_float(h) ||
        !is_float(k) || !is_float(v) || !is_float(wk) || !is_float(wq) || !is_float(gam) || !is_float(per))
        return RAD_E_SHAPE;
    const int64_t rows = h.shape[0];
    if (rows < 1 || k.shape[0] != rows || v.shape[0] != rows || tok.shape[0] != rows) return RAD_E_SHAPE;
    auto rc2 = [&](const RadTensor& t, long* rr, long* cc) {
        *cc = t.stride[t.rank - 1];
        *rr = t.rank == 1 ? n * *cc : t.stride[0];
    };
    long wkr, wkc, wqr, wqc, pr, pc;
    rc2(wk, &wkr, &wkc);
    rc2(wq, &wqr, &wqc);
    rc2(per, &pr, &pc);
    return emit(s, HY4_CHAINED(a, engram_gate_k), dim3((unsigned)rows), dim3(kThreads), 0u, h.data, dcode(h), h.stride[0],
                 h.stride[h.rank - 1], tok.data, icode(tok), tok.stride[0], k.data, dcode(k), k.stride[0],
                 k.stride[k.rank - 1], v.data, dcode(v), v.stride[0], v.stride[1], wk.data, dcode(wk), wkr, wkc,
                 wq.data, dcode(wq), wqr, wqc, gam.data, dcode(gam), gam.stride[0], per.data, dcode(per), pr, pc,
                 (int)n, (int)hc, (long)pad, (long)skip, (long)skip2, eps, geps);
}

int hy4_dev_ngram_ids(const RadArgs* a, RadStream s) {
    const int64_t E = pi(a, "engram_layers"), heads = pi(a, "heads"), ngram = pi(a, "ngram", 3);
    const int64_t pad = pi(a, "pad");
    if (E < 1 || heads < 1 || ngram < 2 || ngram > 8 || heads % (ngram - 1) || !rad_args_have(a, 10))
        return RAD_E_SHAPE;
    const RadTensor &tok = a->t[0], &pos = a->t[1], &cu = a->t[2], &state = a->t[3], &sidx = a->t[4];
    const RadTensor &map = a->t[5], &mult = a->t[6], &vsz = a->t[7], &off = a->t[8], &ids = a->t[9];
    const int64_t hist = ngram - 1;
    if (!is_rank(tok, 1) || !is_int(tok) || !is_rank(pos, 1) || !is_int(pos) ||
        !(is_rank(ids, 3, heads) || is_rank(ids, 2, heads)) || ids.shape[0] % E != 0 || !is_int(ids) ||
        (ids.rank == 3 && ids.shape[0] != E) || !is_present(state) || state.rank < 2 ||
        state.shape[state.rank - 1] < hist || !is_int(state) || !is_present(sidx) || !is_int(sidx) ||
        !is_present(map) || !is_int(map) || map.rank > 2 || !is_present(cu) || !is_int(cu) || cu.rank != 1 ||
        !is_matrix(mult, E, ngram) || mult.dtype != RAD_I64 || !is_matrix(vsz, E, heads) || vsz.dtype != RAD_I64 ||
        !is_matrix(off, E, heads) || off.dtype != RAD_I64 || mult.stride[mult.rank - 1] != 1 ||
        vsz.stride[vsz.rank - 1] != 1 || off.stride[off.rank - 1] != 1 || map.stride[map.rank - 1] != 1 ||
        /* the kernel reads them as dense planes [E, …] */
        (mult.rank == 2 && mult.stride[0] != ngram) || (vsz.rank == 2 && vsz.stride[0] != heads) ||
        (off.rank == 2 && off.stride[0] != heads))
        return RAD_E_SHAPE;
    for (uint32_t i = 1; i + 1 < state.rank; ++i)
        if (state.shape[i] != 1) return RAD_E_SHAPE;
    const int64_t M = pos.shape[0];   // the step's rows; `tok` may carry the next chunk after them
    const int64_t R = ids.rank == 3 ? ids.shape[1] : ids.shape[0] / E;
    const int64_t s_e = ids.rank == 3 ? ids.stride[0] : R * ids.stride[0];
    const int64_t s_r = ids.rank == 3 ? ids.stride[1] : ids.stride[0];
    const int64_t s_h = ids.rank == 3 ? ids.stride[2] : ids.stride[1];
    if (M < 1 || tok.shape[0] < M || R < M) return RAD_E_SHAPE;
    const RadTensor* ahead = a->n_t > 10 && a->t[10].data ? &a->t[10] : nullptr;
    int64_t A = 0, a_e = 0, a_r = 0, a_h = 0;
    if (ahead) {
        if (!(is_rank(*ahead, 3, heads) || is_rank(*ahead, 2, heads)) || !is_int(*ahead) ||
            (ahead->rank == 3 && ahead->shape[0] != E) || (ahead->rank == 2 && ahead->shape[0] % E) ||
            icode(*ahead) != icode(ids))
            return RAD_E_SHAPE;
        const int64_t AR = ahead->rank == 3 ? ahead->shape[1] : ahead->shape[0] / E;
        a_e = ahead->rank == 3 ? ahead->stride[0] : AR * ahead->stride[0];
        a_r = ahead->rank == 3 ? ahead->stride[1] : ahead->stride[0];
        a_h = ahead->rank == 3 ? ahead->stride[2] : ahead->stride[1];
        A = std::min<int64_t>(AR, tok.shape[0] - M);
    }
    const int64_t S = cu.shape[0] - 1;
    if (S < 1 || sidx.shape[0] < S) return RAD_E_SHAPE;
    const int64_t V = map.rank == 1 ? map.shape[0] / E : map.shape[1];
    if (V < 1 || (map.rank == 1 ? map.shape[0] != E * V : map.shape[0] != E) || pad < 0 || pad >= V) return RAD_E_SHAPE;
    if (map.rank == 2 && map.stride[0] != V) return RAD_E_SHAPE;
    const long sx = sidx.stride[0];
    /* num_accepted (operand 11): i32, one a sequence; the window must hold the read and the step */
    const RadTensor* nacc = a->n_t > 11 && a->t[11].data ? &a->t[11] : nullptr;
    if (nacc && (nacc->dtype != RAD_I32 || nacc->shape[0] < S)) return RAD_E_SHAPE;
    const long ns = nacc ? (long)nacc->stride[0] : 0;
    const unsigned tiles = (unsigned)((M + A + 255) / 256);
    int rc = emit(s, HY4_CHAINED(a, ngram_ids_k), dim3(tiles, (unsigned)S), dim3(256), 0u, tok.data, icode(tok), tok.stride[0],
                   pos.data, icode(pos), pos.stride[0], cu.data, icode(cu), cu.stride[0], (int)M, state.data,
                   icode(state), state.stride[0], state.stride[state.rank - 1], state.shape[0], sidx.data, icode(sidx),
                   sx, map.data, icode(map), (long)V, (const int64_t*)mult.data, (const int64_t*)vsz.data,
                   (const int64_t*)off.data, ids.data, icode(ids), (long)s_e, (long)s_r, (long)s_h, (int)E, (int)heads,
                   (int)ngram, (long)pad, ahead ? ahead->data : nullptr, (long)a_e, (long)a_r, (long)a_h, (int)A,
                   nacc ? (const int32_t*)nacc->data : nullptr, ns, (int)state.shape[state.rank - 1]);
    if (rc != RAD_OK) return rc;
    return emit(s, HY4_CHAINED(a, ngram_state_k), dim3((unsigned)((S + 63) / 64)), dim3(64), 0u, tok.data, icode(tok), tok.stride[0],
                 cu.data, icode(cu), cu.stride[0], (int)M, (int)S, state.data, icode(state), state.stride[0],
                 state.stride[state.rank - 1], state.shape[0], sidx.data, icode(sidx), sx, (int)hist,
                 nacc ? (const int32_t*)nacc->data : nullptr, ns, (int)state.shape[state.rank - 1]);
}
