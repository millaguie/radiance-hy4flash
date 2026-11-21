/* ref_vit.cpp -- Hy4's vision tower (HF HYV4FlashVisionTransformer), the three ops radiance does
 * not carry: FP32 host kernels. Specification: docs/OPS-HY4.md, «Vision».
 *
 * The processor (the core's Hy4 family) puts `merge` extra patch columns in each row: the
 * line-break column. The tower does not look at them: hy4_vit_compact removes them before the
 * first block, hy4_vit_pos adds the learned positions to the remaining patches and hy4_vit_newline
 * places the adapter's output on the grid with the break column, `image_newline` in it. */
#include "hy4k.h"
#include "host_util.h"

#include <cmath>
#include <cstring>

using namespace hy4k;

namespace {

/* Segment of patch p according to cu ([S+1], boundaries in patches). */
inline int64_t segment_of(const RadTensor& cu, int64_t S, int64_t p) {
    int64_t s = 0;
    while (s + 1 < S && ldi(cu, vi(cu, s + 1)) <= p) ++s;
    return s;
}
/* Plane k (row, column, height, width) of patch p in coordinates [4, P] laid out by plane. */
inline int64_t crd_at(const RadTensor& c, int64_t k, int64_t p) { return ldi(c, k * c.stride[0] + p * c.stride[1]); }

bool cu_ok(const RadTensor& cu, int64_t total) {
    if (!is_rank(cu, 1) || !is_int(cu) || cu.shape[0] < 2) return false;
    const int64_t S = cu.shape[0] - 1;
    if (ldi(cu, 0) != 0 || ldi(cu, vi(cu, S)) != total) return false;
    for (int64_t s = 0; s < S; ++s)
        if (ldi(cu, vi(cu, s + 1)) < ldi(cu, vi(cu, s))) return false;
    return true;
}

}  // namespace

/* hy4_vit_compact: the patches whose column comes before the last `merge` of their segment, in
 * the same order; their coordinates [Pc, 4] (row, column, height, width without the break column)
 * and the segment boundaries already without it. */
int hy4_ref_vit_compact(const RadArgs* a, RadStream) {
    const int64_t dim = pi(a, "dim"), merge = pi(a, "merge");
    if (dim < 1 || merge < 1 || !rad_args_have(a, 6)) return RAD_E_SHAPE;
    const RadTensor &pix = a->t[0], &crd = a->t[1], &cu = a->t[2], &pixc = a->t[3], &crdc = a->t[4],
                    &cuc = a->t[5];
    if (!is_rank(pix, 2, dim) || !is_float(pix)) return RAD_E_SHAPE;
    const int64_t P = pix.shape[0];
    if (P < 1 || !is_rank(crd, 2) || !is_int(crd) || crd.shape[0] != 4 || crd.shape[1] != P || !cu_ok(cu, P))
        return RAD_E_SHAPE;
    const int64_t S = cu.shape[0] - 1;
    if (!is_rank(cuc, 1) || !is_int(cuc) || cuc.shape[0] != S + 1 || !is_rank(pixc, 2, dim) || !is_float(pixc) ||
        !is_rank(crdc, 2, 4) || !is_int(crdc) || crdc.shape[0] != pixc.shape[0])
        return RAD_E_SHAPE;
    /* Counts and validates first; writes nothing if the count does not match the output. */
    int64_t keep = 0;
    for (int64_t p = 0; p < P; ++p) {
        const int64_t r = crd_at(crd, 0, p), c = crd_at(crd, 1, p), h = crd_at(crd, 2, p), w = crd_at(crd, 3, p);
        if (h < 1 || w <= merge || r < 0 || r >= h || c < 0 || c >= w) return RAD_E_SHAPE;
        keep += c < w - merge;
    }
    if (keep != pixc.shape[0]) return RAD_E_SHAPE;
    int64_t q = 0;
    sti(cuc, 0, 0);
    for (int64_t s = 0; s < S; ++s) {
        for (int64_t p = ldi(cu, vi(cu, s)); p < ldi(cu, vi(cu, s + 1)); ++p) {
            const int64_t w = crd_at(crd, 3, p), c = crd_at(crd, 1, p);
            if (c >= w - merge) continue;
            for (int64_t i = 0; i < dim; ++i)
                st(pixc, row(pixc, q) + col(pixc, i), ld(pix, row(pix, p) + col(pix, i)));
            sti(crdc, row(crdc, q) + col(crdc, 0), crd_at(crd, 0, p));
            sti(crdc, row(crdc, q) + col(crdc, 1), c);
            sti(crdc, row(crdc, q) + col(crdc, 2), crd_at(crd, 2, p));
            sti(crdc, row(crdc, q) + col(crdc, 3), w - merge);
            ++q;
        }
        sti(cuc, vi(cuc, s + 1), q);
    }
    return RAD_OK;
}

/* hy4_vit_pos: x += learned positions, the table [side², n] (without row 0, the class token's)
 * resampled bilinearly to each patch's (height, width) grid as HF does:
 * F.interpolate(scale_factor=(height+0.1)/side, align_corners=False) in f32, rounded to x's type
 * and added in that type. */
int hy4_ref_vit_pos(const RadArgs* a, RadStream) {
    const int64_t n = pi(a, "n"), side = pi(a, "side");
    if (n < 1 || side < 2 || !rad_args_have(a, 3)) return RAD_E_SHAPE;
    const RadTensor &x = a->t[0], &tab = a->t[1], &crd = a->t[2];
    if (!is_rank(x, 2, n) || !is_float(x) || !is_rank(tab, 2, n) || tab.shape[0] != side * side + 1 ||
        !is_float(tab) || !is_rank(crd, 2, 4) || !is_int(crd) || crd.shape[0] != x.shape[0])
        return RAD_E_SHAPE;
    const int64_t P = x.shape[0];
    if (P < 1) return RAD_E_SHAPE;
    for (int64_t p = 0; p < P; ++p) {
        const int64_t r = ldi(crd, row(crd, p)), c = ldi(crd, row(crd, p) + col(crd, 1));
        const int64_t h = ldi(crd, row(crd, p) + col(crd, 2)), w = ldi(crd, row(crd, p) + col(crd, 3));
        if (h < 1 || w < 1 || r < 0 || r >= h || c < 0 || c >= w) return RAD_E_SHAPE;
    }
#pragma omp parallel for schedule(static) if (P > 64)
    for (int64_t p = 0; p < P; ++p) {
        const int64_t r = ldi(crd, row(crd, p)), c = ldi(crd, row(crd, p) + col(crd, 1));
        const int64_t h = ldi(crd, row(crd, p) + col(crd, 2)), w = ldi(crd, row(crd, p) + col(crd, 3));
        /* area_pixel_compute_source_index with the given scale: src = (dst + 0.5) / factor - 0.5, in
         * f32 with 1/factor computed in double, like PyTorch's upsample_bilinear2d on CPU. */
        auto src_taps = [&](int64_t dst, int64_t extent, int64_t* i0, int64_t* i1, float* l) {
            const float esc = (float)(1.0 / (((double)extent + 0.1) / (double)side));
            float s = esc * ((float)dst + 0.5f) - 0.5f;
            if (s < 0) s = 0;
            int64_t a0 = (int64_t)s;
            if (a0 > side - 1) a0 = side - 1;
            *i0 = a0;
            *i1 = a0 < side - 1 ? a0 + 1 : a0;
            *l = s - (float)a0;
        };
        int64_t y0, y1, x0, x1;
        float ly, lx;
        src_taps(r, h, &y0, &y1, &ly);
        src_taps(c, w, &x0, &x1, &lx);
        const float hy = 1.0f - ly, hx = 1.0f - lx;
        const int64_t t00 = row(tab, 1 + y0 * side + x0), t01 = row(tab, 1 + y0 * side + x1);
        const int64_t t10 = row(tab, 1 + y1 * side + x0), t11 = row(tab, 1 + y1 * side + x1);
        const int64_t xb = row(x, p);
        for (int64_t i = 0; i < n; ++i) {
            const int64_t ci = col(tab, i);
            const float v = hy * (hx * ld(tab, t00 + ci) + lx * ld(tab, t01 + ci)) +
                            ly * (hx * ld(tab, t10 + ci) + lx * ld(tab, t11 + ci));
            /* the resample is rounded to x's type before it is added (HF: .to(embeddings.dtype)) */
            float vx = v;
            if (x.dtype != RAD_F32) {
                uint16_t b16 = 0;
                rad_store_f32(&b16, x.dtype, 0, v);
                vx = rad_load_f32(&b16, x.dtype, 0);
            }
            st(x, xb + col(x, i), ld(x, xb + col(x, i)) + vx);
        }
    }
    return RAD_OK;
}

/* hy4_vit_newline: the output [Pn / merge², n] on the processor's grid (block order, with the
 * break column): the row of a block in the last column is `nl`; the others, the adapter row that
 * falls to it in `y` [Pc / merge², n]. `crd` and `cu` are the processor's (with the column) and
 * `cuc` the boundaries without it (hy4_vit_compact). */
int hy4_ref_vit_newline(const RadArgs* a, RadStream) {
    const int64_t n = pi(a, "n"), merge = pi(a, "merge");
    if (n < 1 || merge < 1 || !rad_args_have(a, 6)) return RAD_E_SHAPE;
    const RadTensor &y = a->t[0], &nl = a->t[1], &crd = a->t[2], &cu = a->t[3], &cuc = a->t[4], &out = a->t[5];
    const int64_t mm = merge * merge;
    if (!is_rank(out, 2, n) || !is_float(out) || !is_rank(y, 2, n) || !is_float(y) || !is_rank(nl, 1) ||
        nl.shape[0] != n || !is_float(nl))
        return RAD_E_SHAPE;
    const int64_t R = out.shape[0], Rc = y.shape[0], P = R * mm;
    if (R < 1 || !is_rank(crd, 2) || !is_int(crd) || crd.shape[0] != 4 || crd.shape[1] != P || !cu_ok(cu, P) ||
        !is_rank(cuc, 1) || !is_int(cuc) || cuc.shape[0] != cu.shape[0] || !cu_ok(cuc, Rc * mm))
        return RAD_E_SHAPE;
    const int64_t S = cu.shape[0] - 1;
    for (int64_t s = 0; s <= S; ++s)
        if (ldi(cu, vi(cu, s)) % mm || ldi(cuc, vi(cuc, s)) % mm) return RAD_E_SHAPE;
    /* Row of y that goes to row r of the output, or -1 for a line break. */
    auto source_row = [&](int64_t r) -> int64_t {
        const int64_t p = r * mm, s = segment_of(cu, S, p);
        const int64_t f = crd_at(crd, 0, p), c = crd_at(crd, 1, p), h = crd_at(crd, 2, p), w = crd_at(crd, 3, p);
        if (h % merge || w % merge || w <= merge || f % merge || c % merge || f >= h || c >= w) return -2;
        const int64_t wb = w / merge, bc = c / merge;
        if (bc == wb - 1) return -1;
        const int64_t k = ldi(cuc, vi(cuc, s)) / mm + (f / merge) * (wb - 1) + bc;
        return k >= 0 && k < Rc ? k : -2;
    };
    for (int64_t r = 0; r < R; ++r)
        if (source_row(r) < -1) return RAD_E_SHAPE;
#pragma omp parallel for schedule(static) if (R > 64)
    for (int64_t r = 0; r < R; ++r) {
        const int64_t k = source_row(r), ob = row(out, r);
        for (int64_t i = 0; i < n; ++i)
            st(out, ob + col(out, i), k < 0 ? ld(nl, vi(nl, i)) : ld(y, row(y, k) + col(y, i)));
    }
    return RAD_OK;
}
