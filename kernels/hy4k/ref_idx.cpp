/* ref_idx.cpp -- Hy4's compressed indexer (HF HYV4FlashCompressedIndexer): group pooling, score
 * and selection. FP32 host kernels. The rope of the LAST `rotary_dim` channels (adjacent pairs, as
 * in GPT-J) is done inside the ops. Specification: docs/OPS-HY4.md. */
#include "hy4k.h"
#include "host_util.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <vector>

using namespace hy4k;

namespace {
/* Adjacent-pair rope over the last `rd` of x[0..n), at position p. As in HF: inv_freq
 * = 1 / base^(2i/rd) and the phase p·inv_freq in float; cos and sin are rounded to the
 * activation's dtype (`cos.to(dtype=x.dtype)`, HF:131) if it is BF16. */
/* With images (interleaved M-RoPE, `sections` s0 s1 s2), frequency i reads component 1 of the
 * position if i % 3 == 1 and i < 3·s1, component 2 if i % 3 == 2 and i < 3·s2, component 0
 * otherwise; a null `comp` is the 1-D position. */
void rope_tail(float* x, int64_t n, int64_t rd, double theta, const int64_t* p, const int* comp, bool bf16) {
    const int64_t o = n - rd;
    for (int64_t i = 0; i < rd / 2; ++i) {
        const float inv = 1.0f / std::pow((float)theta, (float)(2 * i) / (float)rd);
        const float f = (float)p[comp ? comp[i] : 0] * inv;
        float c = std::cos(f), s = std::sin(f);
        if (bf16) {
            c = rad_bf16_to_f32(rad_f32_to_bf16(c));
            s = rad_bf16_to_f32(rad_f32_to_bf16(s));
        }
        const float a = x[o + 2 * i], b = x[o + 2 * i + 1];
        x[o + 2 * i] = a * c - b * s;
        x[o + 2 * i + 1] = b * c + a * s;
    }
}
/* M-RoPE component of each frequency, from `sections` (without the parameter: 0, 1-D position). */
int mrope_components(const RadArgs* a, int64_t rd, int* comp) {
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
    for (int64_t i = 0; i < rd / 2; ++i)
        comp[i] = (i % 3 == 1 && i < 3 * sec[1]) ? 1 : (i % 3 == 2 && i < 3 * sec[2]) ? 2 : 0;
    return 1;
}
}  // namespace

/* hy4_idx_pool -- see docs/OPS-HY4.md. `block_size` (optional) exists to give the key group a page
 * size and must equal `ratio`. */
int hy4_ref_idx_pool(const RadArgs* a, RadStream) {
    const int64_t n = pi(a, "n"), ratio = pi(a, "ratio", 4), rd = pi(a, "rotary_dim");
    const int64_t bs = pi(a, "block_size", ratio);
    const double theta = pf(a, "theta", 1e7);
    const float eps = (float)pf(a, "eps", 1e-5);
    if (n < 1 || n > kMaxNIdx || ratio < 1 || ratio > kMaxRatio || bs != ratio || rd < 2 || rd > n || rd % 2 ||
        !rad_args_have(a, 8))
        return RAD_E_SHAPE;
    const RadTensor &stg = a->t[0], &wv = a->t[1], &wg = a->t[2], &ape = a->t[3], &wn = a->t[4];
    const RadTensor &page = a->t[5], &bpos = a->t[6], &cache = a->t[7];
    if (!is_rank(stg, 2, n) || !is_float(stg) || !is_rank(wv, 2, n) || wv.shape[0] != n || wv.dtype != RAD_F32 ||
        !is_rank(wg, 2, n) || wg.shape[0] != n || wg.dtype != RAD_F32 || !is_rank(ape, 2, n) ||
        ape.shape[0] != ratio || ape.dtype != RAD_F32 || !is_rank(wn, 1, n) || !is_rank(page, 1) ||
        !is_int(page) || !is_int(bpos) || !cache_ok(cache, n))
        return RAD_E_SHAPE;
    /* bpos: [M] (text) or [M, 3] with `sections` (the three components of the group's first token,
     * as qsa_work writes them with the rotary planes) */
    int comp[kMaxNIdx / 2];
    const int sc = mrope_components(a, rd, comp);
    /* a step without rotary planes carries column 0 alone ([M, 1]) or [M]: 1-D position */
    if (sc < 0 || bpos.rank > 2 || (bpos.rank == 2 && bpos.shape[1] != 1 && (sc != 1 || bpos.shape[1] < 3)))
        return RAD_E_SHAPE;
    const int mc = sc == 1 && bpos.rank == 2 && bpos.shape[1] >= 3;
    auto bp = [&](int64_t b, int c) { return bpos.rank == 2 ? ldi(bpos, row(bpos, b) + col(bpos, c)) : ldi(bpos, vi(bpos, b)); };
    const int64_t M = page.shape[0], pages = cache.shape[0];
    if (M < 1 || bpos.shape[0] != M || stg.shape[0] < M * ratio) return RAD_E_SHAPE;
    for (int64_t b = 0; b < M; ++b) {
        const int64_t pg = ldi(page, vi(page, b));
        if (pg >= pages || (pg >= 0 && bp(b, 0) < 0)) return RAD_E_INVAL;
        for (int64_t c = 0; c < b; ++c)  // unique pages: otherwise two groups write the same one
            if (pg >= 0 && ldi(page, vi(page, c)) == pg) return RAD_E_INVAL;
    }
    const int64_t ss = stg.stride[stg.rank - 1];
    const bool bf16 = cache.dtype == RAD_BF16;
#pragma omp parallel for schedule(static) if (M > 4)
    for (int64_t b = 0; b < M; ++b) {
        const int64_t pg = ldi(page, vi(page, b));
        if (pg < 0) continue;
        float k[kMaxRatio * kMaxNIdx], vv[kMaxRatio * kMaxNIdx], gg[kMaxRatio * kMaxNIdx], out[kMaxNIdx];
        for (int64_t j = 0; j < ratio; ++j)
            for (int64_t i = 0; i < n; ++i) k[(size_t)(j * n + i)] = ld(stg, row(stg, b * ratio + j) + i * ss);
        for (int64_t j = 0; j < ratio; ++j)
            for (int64_t o = 0; o < n; ++o) {
                float dv = 0, dg = 0;
                for (int64_t i = 0; i < n; ++i) {
                    const float x = k[(size_t)(j * n + i)];
                    dv += x * ld(wv, row(wv, o) + col(wv, i));
                    dg += x * ld(wg, row(wg, o) + col(wg, i));
                }
                vv[(size_t)(j * n + o)] = dv;
                gg[(size_t)(j * n + o)] = dg + ld(ape, row(ape, j) + col(ape, o));
            }
        float sq = 0;
        for (int64_t o = 0; o < n; ++o) {
            float mx = -std::numeric_limits<float>::infinity();
            for (int64_t j = 0; j < ratio; ++j) mx = std::max(mx, gg[(size_t)(j * n + o)]);
            float den = 0, num = 0;
            for (int64_t j = 0; j < ratio; ++j) {
                const float e = std::exp(gg[(size_t)(j * n + o)] - mx);
                den += e;
                num += vv[(size_t)(j * n + o)] * e;
            }
            out[(size_t)o] = num / den;
            sq += out[(size_t)o] * out[(size_t)o];
        }
        const float inv = 1.0f / std::sqrt(sq / (float)n + eps);
        const int64_t cb = row(cache, pg);
        for (int64_t o = 0; o < n; ++o) {
            /* rounding to the cache's dtype before the rope (HF:393) and after */
            st(cache, cb + o, out[(size_t)o] * inv * ld(wn, col(wn, o)));
            out[(size_t)o] = ld(cache, cb + o);
        }
        const int64_t p3[3] = { bp(b, 0), mc ? bp(b, 1) : 0, mc ? bp(b, 2) : 0 };
        rope_tail(out, n, rd, theta, p3, mc ? comp : nullptr, bf16);
        for (int64_t o = 0; o < n; ++o) st(cache, cb + o, out[(size_t)o]);
    }
    return RAD_OK;
}

/* hy4_idx_score -- see docs/OPS-HY4.md. */
int hy4_ref_idx_score(const RadArgs* a, RadStream) {
    const int64_t n = pi(a, "n"), H = pi(a, "heads"), rd = pi(a, "rotary_dim"), cap = pi(a, "seqs", 0);
    const double theta = pf(a, "theta", 1e7);
    if (n < 1 || n > kMaxNIdx || H < 1 || H > kMaxHIdx || rd < 2 || rd > n || rd % 2 || !rad_args_have(a, 8))
        return RAD_E_SHAPE;
    const RadTensor &q = a->t[0], &w = a->t[1], &pos = a->t[2], &cache = a->t[3];
    const RadTensor &bt = a->t[4], &cu = a->t[5], &nc = a->t[6], &score = a->t[7];
    /* q: [M, heads, n] or [M, heads·n] (the architecture passes it 2-D, a row of its buffer) */
    const bool q3 = q.rank == 3;
    if (!(q3 ? is_rank(q, 3, n) && q.shape[1] == H : is_rank(q, 2, H * n)) || !is_float(q) || !is_rank(w, 2, H) || !is_float(w) ||
        !is_int(pos) || !cache_ok(cache, n) || !is_rank(bt, 2) || !is_int(bt) ||
        !is_rank(nc, 1) || !is_int(nc) || !is_rank(score, 2) || score.dtype != RAD_F32)
        return RAD_E_SHAPE;
    /* pos: [M] (text) or [3, M] by component with `sections` (RadBatch::rope_pos) */
    int comp[kMaxNIdx / 2];
    const int sc = mrope_components(a, rd, comp);
    if (sc < 0 || pos.rank > 2 || (pos.rank == 2 && (sc != 1 || pos.shape[0] < 3))) return RAD_E_SHAPE;
    const int mc = sc == 1 && pos.rank == 2;
    const int64_t pM = pos.rank == 2 ? pos.shape[1] : pos.shape[0];
    const int64_t M = q.shape[0], blocks = score.shape[1], pages = cache.shape[0];
    if (M < 1 || w.shape[0] != M || pM != M || score.shape[0] != M)
        return RAD_E_SHAPE;
    const int64_t S = count_seqs(cu, cap, M);
    if (S < 1 || bt.shape[0] < S || nc.shape[0] < S) return RAD_E_SHAPE;
    for (int64_t s = 0; s < S; ++s) {
        const int64_t ncs = ldi(nc, vi(nc, s));
        /* The plane may be wider than the table (it is declared by max_ctx); nc may not. */
        if (ncs < 0 || ncs > bt.shape[1]) return RAD_E_INVAL;
        for (int64_t g = 0; g < std::min(ncs, blocks); ++g)
            if (ldi(bt, row(bt, s) + col(bt, g)) >= pages) return RAD_E_INVAL;
    }
    const int64_t qs = q.stride[q.rank - 1];
    const bool bf16 = q.dtype == RAD_BF16;
    const float esc = (float)(1.0 / std::sqrt((double)H) / std::sqrt((double)n));
    for (int64_t s = 0; s < S; ++s) {
        const int64_t r0 = ldi(cu, vi(cu, s)), r1 = ldi(cu, vi(cu, s + 1)), ncs = ldi(nc, vi(nc, s));
#pragma omp parallel for schedule(static) if (r1 - r0 > 4)
        for (int64_t r = r0; r < r1; ++r) {
            float qq[kMaxHIdx * kMaxNIdx];
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t i = 0; i < n; ++i)
                    qq[(size_t)(h * n + i)] = ld(q, row(q, r) + h * (q3 ? q.stride[1] : n * qs) + i * qs);
                int64_t p3[3] = { 0, 0, 0 };
                for (int c = 0; c < (mc ? 3 : 1); ++c)
                    p3[c] = mc ? ldi(pos, c * pos.stride[0] + r * pos.stride[1]) : ldi(pos, vi(pos, r));
                rope_tail(&qq[(size_t)(h * n)], n, rd, theta, p3, mc ? comp : nullptr, bf16);
            }
            /* HF rotates in FP32 and goes back to q's dtype before the score */
            if (bf16)
                for (int64_t i = 0; i < H * n; ++i) qq[i] = rad_bf16_to_f32(rad_f32_to_bf16(qq[i]));
            for (int64_t g = 0; g < blocks; ++g) {
                float val = -std::numeric_limits<float>::infinity();
                const int64_t pg = g < ncs ? ldi(bt, row(bt, s) + col(bt, g)) : -1;
                if (pg >= 0) {
                    const int64_t cb = row(cache, pg);
                    val = 0;
                    for (int64_t h = 0; h < H; ++h) {
                        float dot = 0;
                        for (int64_t i = 0; i < n; ++i) dot += qq[(size_t)(h * n + i)] * ld(cache, cb + i);
                        val += ld(w, row(w, r) + col(w, h)) * esc * std::max(dot, 0.0f);
                    }
                }
                st(score, row(score, r) + col(score, g), val);
            }
        }
    }
    return RAD_OK;
}

/* hy4_idx_select -- see docs/OPS-HY4.md. Candidates with page < 0 or a non-finite score do not
 * count. */
int hy4_ref_idx_select(const RadArgs* a, RadStream) {
    const int64_t topk = pi(a, "topk"), ratio = pi(a, "ratio", 4), cap = pi(a, "seqs", 0);
    if (topk < 2 || topk > kMaxTopk || ratio < 1 || !rad_args_have(a, 8)) return RAD_E_SHAPE;
    const RadTensor &score = a->t[0], &bt = a->t[1], &nc = a->t[2], &pos = a->t[3], &cu = a->t[4];
    const RadTensor &sel = a->t[5], &nsel = a->t[6], &used = a->t[7];
    if (!is_rank(score, 2) || score.dtype != RAD_F32 || !is_rank(bt, 2) || !is_int(bt) || !is_rank(nc, 1) ||
        !is_int(nc) || !is_rank(pos, 1) || !is_int(pos) || !is_rank(sel, 2, topk) || !is_int(sel) ||
        !is_rank(nsel, 1) || !is_int(nsel) || !is_rank(used, 1) || !is_int(used))
        return RAD_E_SHAPE;
    const int64_t M = score.shape[0], blocks = score.shape[1];
    if (M < 1 || pos.shape[0] != M || sel.shape[0] != M || nsel.shape[0] != M || used.shape[0] != M)
        return RAD_E_SHAPE;
    const int64_t S = count_seqs(cu, cap, M);
    if (S < 1 || bt.shape[0] < S || nc.shape[0] < S) return RAD_E_SHAPE;
    for (int64_t s = 0; s < S; ++s) {
        const int64_t r0 = ldi(cu, vi(cu, s)), r1 = ldi(cu, vi(cu, s + 1));
        if (ldi(nc, vi(nc, s)) < 0) return RAD_E_INVAL;
        for (int64_t r = r0; r < r1; ++r) {  // the row's own group must be in both tables
            const int64_t p = ldi(pos, vi(pos, r));
            if (p < 0 || p / ratio >= bt.shape[1]) return RAD_E_INVAL;
        }
    }
    for (int64_t s = 0; s < S; ++s) {
        const int64_t r0 = ldi(cu, vi(cu, s)), r1 = ldi(cu, vi(cu, s + 1)), ncs = ldi(nc, vi(nc, s));
#pragma omp parallel for schedule(static) if (r1 - r0 > 4)
        for (int64_t r = r0; r < r1; ++r) {
            const int64_t p = ldi(pos, vi(pos, r)), own = p / ratio;
            const int64_t ncand = std::min(std::min(ncs, own), blocks);
            /* The best k in a fixed-size heap whose root is the worst one kept: better =
             * higher score, or equal score and lower index. No memory allocation. */
            const int64_t k = topk - 1;
            float hv[kMaxTopk];
            int64_t hg[kMaxTopk];
            int64_t nh = 0;
            auto worse = [&](int64_t i, int64_t j) {  // is element i worse than j?
                return hv[i] < hv[j] || (hv[i] == hv[j] && hg[i] > hg[j]);
            };
            auto sift_down = [&](int64_t i) {
                for (;;) {
                    int64_t m = i, l = 2 * i + 1, d = l + 1;
                    if (l < nh && worse(l, m)) m = l;
                    if (d < nh && worse(d, m)) m = d;
                    if (m == i) return;
                    std::swap(hv[i], hv[m]); std::swap(hg[i], hg[m]); i = m;
                }
            };
            for (int64_t g = 0; g < ncand; ++g) {
                const float v = ld(score, row(score, r) + col(score, g));
                if (!std::isfinite(v) || ldi(bt, row(bt, s) + col(bt, g)) < 0) continue;
                if (nh < k) {
                    hv[nh] = v; hg[nh] = g;
                    for (int64_t i = nh++; i > 0 && worse(i, (i - 1) / 2); i = (i - 1) / 2) {
                        std::swap(hv[i], hv[(i - 1) / 2]); std::swap(hg[i], hg[(i - 1) / 2]);
                    }
                } else if (v > hv[0] || (v == hv[0] && g < hg[0])) {
                    hv[0] = v; hg[0] = g; sift_down(0);
                }
            }
            std::sort(hg, hg + nh);
            const int64_t sb = row(sel, r);
            int64_t i = 0;
            for (; i < nh; ++i) sti(sel, sb + col(sel, i), ldi(bt, row(bt, s) + col(bt, hg[i])));
            sti(sel, sb + col(sel, i), ldi(bt, row(bt, s) + col(bt, own)));
            for (++i; i < topk; ++i) sti(sel, sb + col(sel, i), -1);
            sti(nsel, vi(nsel, r), nh + 1);
            sti(used, vi(used, r), nh * ratio + p % ratio + 1);
        }
    }
    return RAD_OK;
}
