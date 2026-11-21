/* dev_vit.hip -- card rows of Hy4's vision ops (ref_vit.cpp, ref_rope.cpp).
 *
 * The coordinates and the segment boundaries live on the card: the host validates shapes and the
 * kernel skips what does not fit (a destination outside the output, a segment without a break
 * column), like the rest of hy4k (dev_common.h). */
#include "dev_common.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

using namespace hy4k;

namespace {

constexpr int kThreads = 256;

/* Segment of patch p (cu on the card, S+1 entries): linear search, S is small. */
__device__ inline int segment_of(const void* cu, int cit, long cs, int S, long p) {
    int s = 0;
    while (s + 1 < S && dldi(cu, cit, (long)(s + 1) * cs) <= p) ++s;
    return s;
}

/* cuc[s]: patches without the break column before segment s (height·(width-merge) of each). */
__device__ inline long cuc_of(const void* cu, int cit, long cs, const void* crd, int rit, long r0, long r1, int s,
                              int merge) {
    long acc = 0;
    for (int j = 0; j < s; ++j) {
        const long p = dldi(cu, cit, (long)j * cs);
        const long h = dldi(crd, rit, 2 * r0 + p * r1), w = dldi(crd, rit, 3 * r0 + p * r1);
        acc += h * (w - merge);
    }
    return acc;
}

__global__ __launch_bounds__(kThreads) void vit_compact_k(const void* pix, int pdt, long pp, long ps, const void* crd,
                                                        int rit, long r0, long r1, const void* cu, int cit, long cs,
                                                        int S, void* pc, int cdt, long cp, long css, void* crdc, int qit,
                                                        long q0, long q1, void* cuc, int uit, long us, int dim,
                                                        int merge, long Pc) {
    const long p = blockIdx.x;
    if (p == 0 && threadIdx.x == 0) {
        long acc = 0;
        dsti(cuc, uit, 0, 0);
        for (int j = 0; j < S; ++j) {
            const long a0 = dldi(cu, cit, (long)j * cs);
            const long h = dldi(crd, rit, 2 * r0 + a0 * r1), w = dldi(crd, rit, 3 * r0 + a0 * r1);
            acc += h * (w - merge);
            dsti(cuc, uit, (long)(j + 1) * us, acc);
        }
    }
    const int s = segment_of(cu, cit, cs, S, p);
    const long r = dldi(crd, rit, p * r1), c = dldi(crd, rit, r0 + p * r1);
    const long h = dldi(crd, rit, 2 * r0 + p * r1), w = dldi(crd, rit, 3 * r0 + p * r1);
    if (w <= merge || c >= w - merge || r < 0 || r >= h || c < 0) return;
    const long mm = (long)merge * merge, wb = w / merge;
    const long k = p - dldi(cu, cit, (long)s * cs), blk = k / mm, inb = k % mm;
    const long br = blk / wb, bc = blk % wb;
    const long dest = cuc_of(cu, cit, cs, crd, rit, r0, r1, s, merge) + (br * (wb - 1) + bc) * mm + inb;
    if (dest < 0 || dest >= Pc) return;
    for (int i = threadIdx.x; i < dim; i += kThreads)
        dst(pc, cdt, dest * cp + (long)i * css, dld(pix, pdt, p * pp + (long)i * ps));
    if (threadIdx.x == 0) {
        dsti(crdc, qit, dest * q0, r);
        dsti(crdc, qit, dest * q0 + q1, c);
        dsti(crdc, qit, dest * q0 + 2 * q1, h);
        dsti(crdc, qit, dest * q0 + 3 * q1, w - merge);
    }
}

__device__ inline void src_taps(long dst_i, long extent, int side, long* i0, long* i1, float* l) {
    const float esc = (float)(1.0 / (((double)extent + 0.1) / (double)side));
    float s = esc * ((float)dst_i + 0.5f) - 0.5f;
    if (s < 0) s = 0;
    long a0 = (long)s;
    if (a0 > side - 1) a0 = side - 1;
    *i0 = a0;
    *i1 = a0 < side - 1 ? a0 + 1 : a0;
    *l = s - (float)a0;
}

__global__ __launch_bounds__(kThreads) void vit_pos_k(void* x, int xdt, long xp, long xs, const void* tab, int tdt,
                                                    long tp, long ts, const void* crd, int rit, long q0, long q1, int n,
                                                    int side) {
    const long p = blockIdx.x;
    const long r = dldi(crd, rit, p * q0), c = dldi(crd, rit, p * q0 + q1);
    const long h = dldi(crd, rit, p * q0 + 2 * q1), w = dldi(crd, rit, p * q0 + 3 * q1);
    if (h < 1 || w < 1 || r < 0 || r >= h || c < 0 || c >= w) return;
    long y0, y1, x0, x1;
    float ly, lx;
    src_taps(r, h, side, &y0, &y1, &ly);
    src_taps(c, w, side, &x0, &x1, &lx);
    const float hy = 1.0f - ly, hx = 1.0f - lx;
    const long t00 = (1 + y0 * side + x0) * tp, t01 = (1 + y0 * side + x1) * tp;
    const long t10 = (1 + y1 * side + x0) * tp, t11 = (1 + y1 * side + x1) * tp;
    for (int i = threadIdx.x; i < n; i += kThreads) {
        const long ci = (long)i * ts;
        float v = hy * (hx * dld(tab, tdt, t00 + ci) + lx * dld(tab, tdt, t01 + ci)) +
                  ly * (hx * dld(tab, tdt, t10 + ci) + lx * dld(tab, tdt, t11 + ci));
        if (xdt == 1) v = bf2f(f2bf(v));
        else if (xdt == 2) v = (float)(_Float16)v;
        const long o = p * xp + (long)i * xs;
        dst(x, xdt, o, dld(x, xdt, o) + v);
    }
}

__global__ __launch_bounds__(kThreads) void vit_nl_k(const void* y, int ydt, long yp, long ys, const void* nl, int ndt,
                                                   long ns, const void* crd, int rit, long r0, long r1, const void* cu,
                                                   int cit, long cs, const void* cuc, int uit, long us, int S, void* out,
                                                   int odt, long op, long os, int n, int merge, long Rc) {
    const long r = blockIdx.x, mm = (long)merge * merge, p = r * mm;
    const int s = segment_of(cu, cit, cs, S, p);
    const long f = dldi(crd, rit, p * r1), c = dldi(crd, rit, r0 + p * r1);
    const long h = dldi(crd, rit, 2 * r0 + p * r1), w = dldi(crd, rit, 3 * r0 + p * r1);
    long k = -1;
    bool bad = h % merge || w % merge || w <= merge || f % merge || c % merge || f >= h || c >= w;
    if (!bad) {
        const long wb = w / merge, bc = c / merge;
        if (bc != wb - 1) {
            k = dldi(cuc, uit, (long)s * us) / mm + (f / merge) * (wb - 1) + bc;
            bad = k < 0 || k >= Rc;
        }
    }
    if (bad) {   // leaves no bytes from another pass
        for (int i = threadIdx.x; i < n; i += kThreads) dst(out, odt, r * op + (long)i * os, 0.0f);
        return;
    }
    for (int i = threadIdx.x; i < n; i += kThreads)
        dst(out, odt, r * op + (long)i * os, k < 0 ? dld(nl, ndt, (long)i * ns) : dld(y, ydt, k * yp + (long)i * ys));
}

/* The frequency ladder, computed on the host as the oracle does (pow in float) and passed by
 * value, at most 64 (Hy4 uses rotary_dim 64). The kernel derives each frequency's component from
 * the sections. The backend accepts 768 bytes of arguments, hidden ones included (~256): with two
 * tables of 128 the card refused the op at emit. */
struct FreqLadder {
    float inv[64];
};
// The backend's room is 768 bytes, and it also counts HIP's ~256 bytes of hidden arguments.
static_assert(sizeof(FreqLadder) + 128 + 256 <= 768, "rope_k arguments exceed the backend's room");

__global__ __launch_bounds__(kThreads) void rope_k(void* x, int xdt, long xp, const void* pos, int pit, long p0, long p1,
                                                 int multi, int hd, int half, float div, int im, int s1, int s2,
                                                 FreqLadder e) {
    const long m = blockIdx.x;
    const int h = blockIdx.y;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const int comp = !im ? 0 : (i % 3 == 1 && i < 3 * s1) ? 1 : (i % 3 == 2 && i < 3 * s2) ? 2 : 0;
        const float p = (float)(multi ? dldi(pos, pit, (long)comp * p0 + m * p1) : dldi(pos, pit, m * p0)) / div;
        const float ang = p * e.inv[i];
        const float c = cosf(ang), sn = sinf(ang);
        const long b = m * xp + (long)h * hd + 2 * i;
        const float x0 = dld(x, xdt, b), x1 = dld(x, xdt, b + 1);
        dst(x, xdt, b, x0 * c - x1 * sn);
        dst(x, xdt, b + 1, x1 * c + x0 * sn);
    }
}

}  // namespace

int hy4_dev_vit_compact(const RadArgs* a, RadStream st) {
    const int64_t dim = pi(a, "dim"), merge = pi(a, "merge");
    if (dim < 1 || merge < 1 || !rad_args_have(a, 6)) return RAD_E_SHAPE;
    const RadTensor &pix = a->t[0], &crd = a->t[1], &cu = a->t[2], &pc = a->t[3], &crdc = a->t[4], &cuc = a->t[5];
    if (!is_rank(pix, 2, dim) || !is_float(pix)) return RAD_E_SHAPE;
    const int64_t P = pix.shape[0];
    if (P < 1 || !is_rank(crd, 2) || !is_int(crd) || crd.shape[0] != 4 || crd.shape[1] != P || !is_rank(cu, 1) ||
        !is_int(cu) || cu.shape[0] < 2)
        return RAD_E_SHAPE;
    const int64_t S = cu.shape[0] - 1;
    if (!is_rank(cuc, 1) || !is_int(cuc) || cuc.shape[0] != S + 1 || !is_rank(pc, 2, dim) || !is_float(pc) ||
        !is_rank(crdc, 2, 4) || !is_int(crdc) || crdc.shape[0] != pc.shape[0] || pc.shape[0] < 1)
        return RAD_E_SHAPE;
    return emit(st, vit_compact_k, dim3((unsigned)P), dim3(kThreads), 0u, pix.data, dcode(pix), pix.stride[0],
                 pix.stride[1], crd.data, icode(crd), crd.stride[0], crd.stride[1], cu.data, icode(cu), cu.stride[0],
                 (int)S, pc.data, dcode(pc), pc.stride[0], pc.stride[1], crdc.data, icode(crdc), crdc.stride[0],
                 crdc.stride[1], cuc.data, icode(cuc), cuc.stride[0], (int)dim, (int)merge, (long)pc.shape[0]);
}

int hy4_dev_vit_pos(const RadArgs* a, RadStream st) {
    const int64_t n = pi(a, "n"), side = pi(a, "side");
    if (n < 1 || side < 2 || !rad_args_have(a, 3)) return RAD_E_SHAPE;
    const RadTensor &x = a->t[0], &tab = a->t[1], &crd = a->t[2];
    if (!is_rank(x, 2, n) || !is_float(x) || !is_rank(tab, 2, n) || tab.shape[0] != side * side + 1 || !is_float(tab) ||
        !is_rank(crd, 2, 4) || !is_int(crd) || crd.shape[0] != x.shape[0] || x.shape[0] < 1)
        return RAD_E_SHAPE;
    return emit(st, vit_pos_k, dim3((unsigned)x.shape[0]), dim3(kThreads), 0u, x.data, dcode(x), x.stride[0],
                 x.stride[1], tab.data, dcode(tab), tab.stride[0], tab.stride[1], crd.data, icode(crd), crd.stride[0],
                 crd.stride[1], (int)n, (int)side);
}

int hy4_dev_vit_newline(const RadArgs* a, RadStream st) {
    const int64_t n = pi(a, "n"), merge = pi(a, "merge");
    if (n < 1 || merge < 1 || !rad_args_have(a, 6)) return RAD_E_SHAPE;
    const RadTensor &y = a->t[0], &nl = a->t[1], &crd = a->t[2], &cu = a->t[3], &cuc = a->t[4], &out = a->t[5];
    const int64_t mm = merge * merge;
    if (!is_rank(out, 2, n) || !is_float(out) || !is_rank(y, 2, n) || !is_float(y) || !is_rank(nl, 1) ||
        nl.shape[0] != n || !is_float(nl))
        return RAD_E_SHAPE;
    const int64_t R = out.shape[0], P = R * mm;
    if (R < 1 || y.shape[0] < 1 || !is_rank(crd, 2) || !is_int(crd) || crd.shape[0] != 4 || crd.shape[1] != P ||
        !is_rank(cu, 1) || !is_int(cu) || cu.shape[0] < 2 || !is_rank(cuc, 1) || !is_int(cuc) ||
        cuc.shape[0] != cu.shape[0])
        return RAD_E_SHAPE;
    return emit(st, vit_nl_k, dim3((unsigned)R), dim3(kThreads), 0u, y.data, dcode(y), y.stride[0], y.stride[1],
                 nl.data, dcode(nl), nl.stride[0], crd.data, icode(crd), crd.stride[0], crd.stride[1], cu.data,
                 icode(cu), cu.stride[0], cuc.data, icode(cuc), cuc.stride[0], (int)(cu.shape[0] - 1), out.data,
                 dcode(out), out.stride[0], out.stride[1], (int)n, (int)merge, (long)y.shape[0]);
}

int hy4_dev_rope(const RadArgs* a, RadStream st) {
    const int64_t hd = pi(a, "head_dim"), nh = pi(a, "n_head"), nkv = pi(a, "n_head_kv", 0);
    const int64_t rot = pi(a, "rotary_dim", hd);
    const double theta = pf(a, "theta", 10000.0), esc = pf(a, "scale", 1.0);
    const char* mode = rad_args_gets(a, "mode");
    if (hd < 2 || nh < 1 || nkv < 0 || rot < 2 || rot > hd || (rot & 1) || rot / 2 > 64 || !mode ||
        !rad_args_have(a, 2))
        return RAD_E_SHAPE;
    const bool im = !std::strcmp(mode, "imrope");
    if (!im && std::strcmp(mode, "gptj")) return RAD_E_UNSUPPORTED;
    int64_t sec[3] = {};
    int ns = 0;
    if (const char* v = rad_args_gets(a, "sections")) {
        for (const char* q = v; *q && ns < 3;) {
            while (*q == ' ' || *q == ',') ++q;
            if (!*q) break;
            char* e = nullptr;
            const long x = std::strtol(q, &e, 10);
            if (e == q || x < 0) return RAD_E_SHAPE;
            sec[ns++] = x;
            q = e;
        }
    }
    if (im && ns != 3) return RAD_E_SHAPE;
    const RadTensor &x = a->t[0], &pos = a->t[1];
    if (!is_present(x) || !is_float(x) || x.rank != 2 || x.shape[1] < (nh + nkv) * hd || x.stride[1] != 1)
        return RAD_E_SHAPE;
    const int64_t M = x.shape[0];
    const bool multi = pos.rank == 2;
    if (M < 1 || !is_int(pos) || (multi && (!im || pos.shape[0] < 3 || pos.shape[1] != M)) ||
        (!multi && (pos.rank != 1 || pos.shape[0] != M)))
        return RAD_E_SHAPE;
    FreqLadder e{};
    const int64_t half = rot / 2;
    for (int64_t i = 0; i < half; ++i) {
        e.inv[i] = 1.0f / std::pow((float)theta, (float)(2 * i) / (float)rot);
    }
    const unsigned threads = (unsigned)std::min<int64_t>(kThreads, (half + 31) / 32 * 32);
    return emit(st, rope_k, dim3((unsigned)M, (unsigned)(nh + nkv)), dim3(threads), 0u, x.data, dcode(x), x.stride[0],
                 pos.data, icode(pos), pos.stride[0], multi ? pos.stride[1] : 0, (int)multi, (int)hd, (int)half,
                 (float)(esc != 0.0 ? esc : 1.0), (int)(im ? 1 : 0), (int)sec[1], (int)sec[2], e);
}
