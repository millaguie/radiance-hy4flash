/* ref_engram.cpp -- Hy4's Engram (HF HYV4FlashEngramLayer): n-gram hash, an FP8 round trip and
 * per-head RMSNorm, and a per-stream gate. FP32 host kernels. Specification: docs/OPS-HY4.md. */
#include "hy4k.h"
#include "host_util.h"

#include <algorithm>
#include <vector>

using namespace hy4k;

namespace {
inline int64_t floor_mod(int64_t x, int64_t m) {
    const int64_t r = x % m;
    return r < 0 ? r + m : r;
}
}  // namespace

/* hy4_ngram_ids -- see docs/OPS-HY4.md. Checks before writing: cu against the rows, each slot of
 * sidx against the rows of state, the width of state (= ngram-1), and the shapes of the weights. */
int hy4_ref_ngram_ids(const RadArgs* a, RadStream) {
    const int64_t E = pi(a, "engram_layers"), heads = pi(a, "heads"), ngram = pi(a, "ngram", 3);
    const int64_t pad = pi(a, "pad"), cap = pi(a, "seqs", 0);
    if (E < 1 || heads < 1 || ngram < 2 || ngram > 8 || heads % (ngram - 1) || !rad_args_have(a, 10))
        return RAD_E_SHAPE;
    const RadTensor &tok = a->t[0], &pos = a->t[1], &cu = a->t[2], &state = a->t[3], &sidx = a->t[4];
    const RadTensor &map = a->t[5], &mult = a->t[6], &vsz = a->t[7], &off = a->t[8], &ids = a->t[9];
    const int64_t hist = ngram - 1;  // previous tokens the state keeps
    if (!is_rank(tok, 1) || !is_int(tok) || !is_rank(pos, 1) || !is_int(pos) ||
        /* ids: [E, R, heads], or [E·R, heads] (each layer's gather reads its slice as [T, heads]) */
        !(is_rank(ids, 3, heads) || is_rank(ids, 2, heads)) || ids.shape[0] % E != 0 || !is_int(ids) ||
        (ids.rank == 3 && ids.shape[0] != E) ||
        /* state: [slots, …, W], W = ngram-1 (+ the drafts a verify step may carry); the cache of
         * a RAD_KV_LINEAR group carries intermediate axes of size 1. The slot steps along the first
         * axis. */
        !is_present(state) || state.rank < 2 || state.shape[state.rank - 1] < hist || !is_int(state) ||
        !is_present(sidx) || !is_int(sidx) || !is_present(map) || !is_int(map) ||
        !is_matrix(mult, E, ngram) || mult.dtype != RAD_I64 ||
        !is_matrix(vsz, E, heads) || vsz.dtype != RAD_I64 ||
        !is_matrix(off, E, heads) || off.dtype != RAD_I64)
        return RAD_E_SHAPE;
    /* The intermediate axes of state ([slots, 1, 1, ngram-1] in a linear group) must be 1: the
     * kernel only reads and writes the first subrow of each slot. */
    for (uint32_t i = 1; i + 1 < state.rank; ++i)
        if (state.shape[i] != 1) return RAD_E_SHAPE;
    /* The step's rows are pos's; `tok` may carry `A` more tokens after them (RadBatch::n_ahead), the
     * next prefill chunk of the last sequence, whose ids go to the optional `ahead_ids` (operand 10)
     * so the Engram gathers can read their rows a step early. */
    const int64_t M = pos.shape[0];
    const int64_t R = ids.rank == 3 ? ids.shape[1] : ids.shape[0] / E;
    const int64_t s_e = ids.rank == 3 ? ids.stride[0] : R * ids.stride[0];
    const int64_t s_r = ids.rank == 3 ? ids.stride[1] : ids.stride[0];
    const int64_t s_h = ids.rank == 3 ? ids.stride[2] : ids.stride[1];
    if (M < 1 || tok.shape[0] < M || R < M) return RAD_E_SHAPE;  /* R may be max_tok */
    const RadTensor* ahead = a->n_t > 10 && a->t[10].data ? &a->t[10] : nullptr;
    int64_t A = 0, a_e = 0, a_r = 0, a_h = 0;
    if (ahead) {
        if (!(is_rank(*ahead, 3, heads) || is_rank(*ahead, 2, heads)) || !is_int(*ahead) ||
            (ahead->rank == 3 && ahead->shape[0] != E) || (ahead->rank == 2 && ahead->shape[0] % E))
            return RAD_E_SHAPE;
        const int64_t AR = ahead->rank == 3 ? ahead->shape[1] : ahead->shape[0] / E;
        a_e = ahead->rank == 3 ? ahead->stride[0] : AR * ahead->stride[0];
        a_r = ahead->rank == 3 ? ahead->stride[1] : ahead->stride[0];
        a_h = ahead->rank == 3 ? ahead->stride[2] : ahead->stride[1];
        A = std::min<int64_t>(AR, tok.shape[0] - M);
    }
    /* num_accepted (operand 11, optional, [n_seq] one-based) as radiance's ngram_ids has it: the
     * state is a chronological window of W = ngram-1 + max_spec committed ids per slot, read at
     * num_accepted - 1; with it present (a verify step whose drafts may be rejected) the write-back
     * shifts the read window down by one and appends every id of the step, so a next step that kept
     * k reads at k - 1. Absent, the step's last ngram-1 ids land at offset zero. */
    const RadTensor* nacc = a->n_t > 11 && a->t[11].data ? &a->t[11] : nullptr;
    const int64_t W = state.shape[state.rank - 1];
    const int64_t S = count_seqs(cu, cap, M);
    if (S < 1 || sidx.shape[0] < S) return RAD_E_SHAPE;
    /* map: [E, V] or flattened [E·V] */
    if (map.rank > 2) return RAD_E_SHAPE;
    const int64_t V = map.rank == 1 ? map.shape[0] / E : map.shape[1];
    if (V < 1 || (map.rank == 1 ? map.shape[0] != E * V : map.shape[0] != E)) return RAD_E_SHAPE;
    const int64_t per = heads / (ngram - 1), slots = state.shape[0];
    if (pad < 0 || pad >= V) return RAD_E_SHAPE;
    for (int64_t s = 0; s < S; ++s)
        if (ldi(sidx, col0(sidx, s)) >= slots) return RAD_E_INVAL;
    /* every sequence's window checked before any is written, as ref_ngram.cpp does */
    if (nacc && (!is_int(*nacc) || nacc->shape[0] < S)) return RAD_E_SHAPE;
    for (int64_t s = 0; s < S; ++s) {
        const int64_t o = nacc ? ldi(*nacc, vi(*nacc, s)) - 1 : 0;
        const int64_t r0 = ldi(cu, vi(cu, s)), r1 = ldi(cu, vi(cu, s + 1));
        if (o < 0 || o + hist > W) return RAD_E_INVAL;
        if (nacc && ldi(sidx, col0(sidx, s)) >= 0 && hist - 1 + (r1 - r0) > W) return RAD_E_SHAPE;
    }
    for (int64_t e = 0; e < E; ++e)
        for (int64_t h = 0; h < heads; ++h)
            if (ldi(vsz, at2(vsz, e, h, heads)) < 1) return RAD_E_INVAL;
    for (int64_t s = 0; s < S; ++s) {
        const int64_t r0 = ldi(cu, vi(cu, s)), r1 = ldi(cu, vi(cu, s + 1));
        const int64_t slot = ldi(sidx, col0(sidx, s));
        /* the committed window P (oldest first) and prev[k], k back from the last committed id */
        const int64_t o = nacc ? ldi(*nacc, vi(*nacc, s)) - 1 : 0;   // checked above
        int64_t P[16], prev[8];
        for (int64_t k = 0; k < hist; ++k) P[k] = slot >= 0 ? ldi(state, row(state, slot) + col(state, o + k)) : pad;
        for (int64_t k = 0; k < hist; ++k) prev[k] = P[hist - 1 - k];
        /* rows r1 .. r1+A-1 of the last sequence are the prompt after the step: positions run on */
        const int64_t reach = r1 + (A > 0 && r1 == M && r1 > r0 ? A : 0);   // the one non-empty sequence ending last
        for (int64_t r = r0; r < reach; ++r) {
            const bool future = r >= r1;
            const int64_t p = future ? ldi(pos, vi(pos, r1 - 1)) + (r - r1 + 1) : ldi(pos, vi(pos, r));
            int64_t t[8];
            for (int64_t k = 0; k < ngram; ++k) {
                if (p - k < 0) t[k] = pad;
                else if (r - k >= r0) t[k] = ldi(tok, vi(tok, r - k));
                else {
                    const int64_t back = k - 1 - (r - r0);  // 0: the last one of the previous step
                    t[k] = slot >= 0 ? prev[back] : pad;
                }
            }
            for (int64_t e = 0; e < E; ++e) {
                const int64_t pad_c = ldi(map, at2(map, e, pad, V));
                uint64_t mixed[8];
                for (int64_t k = 0; k < ngram; ++k) {
                    const int64_t c = (t[k] >= 0 && t[k] < V) ? ldi(map, at2(map, e, t[k], V)) : pad_c;
                    mixed[k] = (uint64_t)c * (uint64_t)ldi(mult, at2(mult, e, k, ngram));
                }
                for (int64_t b = 0; b < ngram - 1; ++b) {
                    uint64_t mix = mixed[0];
                    for (int64_t k = 1; k <= b + 1; ++k) mix ^= mixed[k];
                    for (int64_t j = 0; j < per; ++j) {
                        const int64_t h = b * per + j;
                        const int64_t pr = ldi(vsz, at2(vsz, e, h, heads));
                        const int64_t id = floor_mod((int64_t)mix, pr) + ldi(off, at2(off, e, h, heads));
                        if (future) sti(*ahead, e * a_e + (r - r1) * a_r + h * a_h, id);
                        else sti(ids, e * s_e + r * s_r + h * s_h, id);
                    }
                }
            }
        }
        if (slot >= 0 && r1 > r0) {
            const int64_t n_step = r1 - r0;
            if (nacc) {   // decode layout: the read window shifted down by one, then every id of the step
                for (int64_t i = 0; i + 1 < hist; ++i) sti(state, row(state, slot) + col(state, i), P[i + 1]);
                for (int64_t t = 0; t < n_step; ++t)
                    sti(state, row(state, slot) + col(state, hist - 1 + t), ldi(tok, vi(tok, r0 + t)));
            } else {      // the step's last hist ids (oldest first) at offset zero, older ones from P
                for (int64_t i = 0; i < hist; ++i) {
                    const int64_t src = n_step - hist + i;
                    const int64_t v = src >= 0 ? ldi(tok, vi(tok, r0 + src)) : P[src + hist];
                    sti(state, row(state, slot) + col(state, i), v);
                }
            }
        }
    }
    return RAD_OK;
}

/* hy4_rows_rmsnorm: x [M, heads·d], w [heads, d] → y [M, heads·d]. With fp8=1, each slice goes
 * through E4M3 with scale amax/448 before the norm (HF _fp8_round_trip, in FP32), the way Engram
 * reads its table. */
int hy4_ref_rows_rmsnorm(const RadArgs* a, RadStream) {
    const int64_t H = pi(a, "heads"), d = pi(a, "d"), fp8 = pi(a, "fp8", 0);
    const float eps = (float)pf(a, "eps", 1.1920929e-07);
    if (H < 1 || d < 1 || d > kMaxD || (fp8 != 0 && fp8 != 1) || !rad_args_have(a, 3)) return RAD_E_SHAPE;
    const RadTensor &x = a->t[0], &w = a->t[1], &y = a->t[2];
    if (!row_width(x, H * d) || !row_width(y, H * d) || !is_matrix(w, H, d) ||
        !is_float(x) || !is_float(y) || !is_float(w))
        return RAD_E_SHAPE;
    const int64_t rows = x.shape[0], xs = x.stride[x.rank - 1], ys = y.stride[y.rank - 1];
    if (rows < 1 || y.shape[0] != rows) return RAD_E_SHAPE;
#pragma omp parallel for schedule(static) if (rows > 16)
    for (int64_t r = 0; r < rows; ++r) {
        float v[kMaxD];
        for (int64_t h = 0; h < H; ++h) {
            const int64_t xb = row(x, r) + h * d * xs, yb = row(y, r) + h * d * ys;
            for (int64_t i = 0; i < d; ++i) v[(size_t)i] = ld(x, xb + i * xs);
            if (fp8) {
                float amax = 0;
                for (int64_t i = 0; i < d; ++i) amax = std::max(amax, std::fabs(v[(size_t)i]));
                const float sc = std::max(amax, 1e-12f) / 448.0f;
                for (int64_t i = 0; i < d; ++i) {
                    float q = v[(size_t)i] / sc;
                    q = std::min(448.0f, std::max(-448.0f, q));
                    const float qq = rad_fp8e4m3_to_f32(rad_f32_to_fp8e4m3(q)) * sc;
                    /* HF returns x + (q - x) (straight-through gradient), which in floating
                     * point is not always q: it is reproduced as is. */
                    v[(size_t)i] = v[(size_t)i] + (qq - v[(size_t)i]);
                }
            }
            float sq = 0;
            for (int64_t i = 0; i < d; ++i) sq += v[(size_t)i] * v[(size_t)i];
            const float inv = 1.0f / std::sqrt(sq / (float)d + eps);
            for (int64_t i = 0; i < d; ++i)
                st(y, yb + i * ys, v[(size_t)i] * inv * ld(w, at2(w, h, i, d)));
        }
    }
    return RAD_OK;
}

/* hy4_engram_gate -- see docs/OPS-HY4.md. */
int hy4_ref_engram_gate(const RadArgs* a, RadStream) {
    const int64_t n = pi(a, "n"), hc = pi(a, "hc"), pad = pi(a, "pad", -1);
    /* skip/skip2: more ids whose rows are left untouched (the image and video markers: HF keeps
     * them out of Engram with engram_mask) */
    const int64_t skip = pi(a, "skip", -1), skip2 = pi(a, "skip2", -1);
    const float eps = (float)pf(a, "eps", 1.1920929e-07), geps = (float)pf(a, "gate_eps", 1e-6);
    if (n < 1 || n > kMaxN || hc < 1 || hc > 16 || !rad_args_have(a, 8)) return RAD_E_SHAPE;
    const RadTensor &h = a->t[0], &tok = a->t[1], &k = a->t[2], &v = a->t[3];
    const RadTensor &wk = a->t[4], &wq = a->t[5], &gam = a->t[6], &per = a->t[7];
    if (!row_width(h, hc * n) || !row_width(k, hc * n) || !is_rank(v, 2, n) || !is_rank(tok, 1) || !is_int(tok) ||
        !is_matrix(wk, hc, n) || !is_matrix(wq, hc, n) ||
        !is_rank(gam, 1, n) || !is_matrix(per, hc, n) || !is_float(h) || !is_float(k) ||
        !is_float(v))
        return RAD_E_SHAPE;
    const int64_t rows = h.shape[0];
    if (rows < 1 || k.shape[0] != rows || v.shape[0] != rows || tok.shape[0] != rows) return RAD_E_SHAPE;
    const int64_t hs = h.stride[h.rank - 1], ks = k.stride[k.rank - 1], vs = v.stride[v.rank - 1];
    auto rms = [&](const RadTensor& t, int64_t base, int64_t s) {
        float sq = 0;
        for (int64_t i = 0; i < n; ++i) { const float x = ld(t, base + i * s); sq += x * x; }
        return 1.0f / std::sqrt(sq / (float)n + eps);
    };
#pragma omp parallel for schedule(static) if (rows > 16)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t tr = ldi(tok, vi(tok, r));
        if ((pad >= 0 && tr == pad) || (skip >= 0 && tr == skip) || (skip2 >= 0 && tr == skip2)) continue;
        const int64_t vb = row(v, r);
        const float iv = rms(v, vb, vs);
        float sig[kMaxN];
        for (int64_t i = 0; i < n; ++i) sig[(size_t)i] = ld(gam, col(gam, i)) * (ld(v, vb + i * vs) * iv);
        float g[16];
        for (int64_t j = 0; j < hc; ++j) {
            const int64_t kb = row(k, r) + j * n * ks, hb = row(h, r) + j * n * hs;
            const float ik = rms(k, kb, ks), ih = rms(h, hb, hs);
            float dot = 0;
            for (int64_t i = 0; i < n; ++i) {
                const float key = ld(k, kb + i * ks) * ik * ld(wk, at2(wk, j, i, n));
                const float qry = ld(h, hb + i * hs) * ih * ld(wq, at2(wq, j, i, n));
                dot += key * qry;
            }
            const float dd = dot / std::sqrt((float)n);
            const float sg = (float)((dd > 0) - (dd < 0));
            g[j] = sigmoid(std::sqrt(std::max(std::fabs(dd), geps)) * sg);
        }
        for (int64_t j = 0; j < hc; ++j) {
            const int64_t hb = row(h, r) + j * n * hs;
            for (int64_t i = 0; i < n; ++i) {
                const float s = sig[(size_t)i];
                st(h, hb + i * hs, ld(h, hb + i * hs) + (g[j] * s + ld(per, at2(per, j, i, n)) * s));
            }
        }
    }
    return RAD_OK;
}
