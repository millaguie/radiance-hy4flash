/* ref_mla.cpp -- hy4_mla_kv: assembles per-head K and V from the output of kv_b and the shared
 * k_rope (HF HYV4FlashAttention.expand_kv). The rope goes FIRST in K: [k_rope | k_nope_h], because
 * radiance's `rope` op rotates the first `rotary_dim` columns; the rows of q_b are permuted the same
 * way at conversion, so q·k does not change (docs/DESIGN.md, «Why the rope goes in front»). */
#include "hy4k.h"
#include "host_util.h"

#include <algorithm>
#include <cstring>

using namespace hy4k;

int hy4_ref_mla_kv(const RadArgs* a, RadStream) {
    const int64_t H = pi(a, "heads"), nope = pi(a, "nope"), rope = pi(a, "rope"), vd = pi(a, "v_dim");
    if (H < 1 || nope < 1 || rope < 1 || vd < 1 || !rad_args_have(a, 4)) return RAD_E_SHAPE;
    const RadTensor &kvx = a->t[0], &kr = a->t[1], &k = a->t[2], &v = a->t[3];
    const int64_t per = nope + vd;
    if (!row_width(kvx, H * per) || !is_rank(kr, 2, rope) || !row_width(k, H * (rope + nope)) ||
        !row_width(v, H * vd) || !is_float(kvx) || !is_float(kr) || !is_float(k) || !is_float(v))
        return RAD_E_SHAPE;
    const int64_t rows = kvx.shape[0];
    if (rows < 1 || kr.shape[0] != rows || k.shape[0] != rows || v.shape[0] != rows) return RAD_E_SHAPE;
    const int64_t xs = kvx.stride[kvx.rank - 1], rs = kr.stride[kr.rank - 1];
    const int64_t ks = k.stride[k.rank - 1], vs = v.stride[v.rank - 1];
#pragma omp parallel for schedule(static) if (rows > 16)
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t h = 0; h < H; ++h) {
            const int64_t xb = row(kvx, r) + h * per * xs;
            const int64_t kb = row(k, r) + h * (rope + nope) * ks;
            const int64_t vb = row(v, r) + h * vd * vs;
            for (int64_t i = 0; i < rope; ++i) st(k, kb + i * ks, ld(kr, row(kr, r) + i * rs));
            for (int64_t i = 0; i < nope; ++i) st(k, kb + (rope + i) * ks, ld(kvx, xb + i * xs));
            for (int64_t i = 0; i < vd; ++i) st(v, vb + i * vs, ld(kvx, xb + (nope + i) * xs));
        }
    }
    return RAD_OK;
}

/* hy4_mla_attn: Hy4's attention over the latent cache (absorbed form, docs/OPS-HY4.md). One KV
 * head shared by all: token j's row is [lat_j (latent) | rope_j (rope)], kv_store's at head_dim
 * (latent+rope)/2 (half in K, half in V; together in memory). The query arrives absorbed:
 * q [M, H, latent] and its rotated part q_rope [M, H, rope]. Score
 * s_j = scale·(q·lat_j + q_rope·rope_j), softmax with the head's sink as one more logit with no
 * value, output Σ p_j lat_j [M, H, latent]. Visibility is attn_paged_sink's: with cu, varlen;
 * without cu, rows split evenly across the sequences of seqused (the form with selection is
 * q_len 1 and one "sequence" per row). Causal, with window `window` (0: no window). A row with no
 * visible keys is left at zero. Optional block keys (bkeys, bcu: the DFly2 drafter) add, in the
 * selection form, each row's own block rows bcu[d] .. i to its keys. Oracle in double. */
int hy4_ref_mla_attn(const RadArgs* a, RadStream) {
    /* cu (5) is optional, and so are bkeys (8) and bcu (9), after the output so that a caller
     * without them issues the 8 operands it always did */
    if (!rad_args_have(a, 5) || !rad_arg_in(a, 6) || !rad_arg_in(a, 7)) return RAD_E_INVAL;
    const RadTensor &q = a->t[0], &qr = a->t[1], &kv = a->t[2], &bt = a->t[3], &su = a->t[4];
    const RadTensor* cus = rad_arg_in(a, 5);
    const RadTensor *bk = rad_arg_in(a, 8), *bcu = rad_arg_in(a, 9);
    const RadTensor &sk = a->t[6], &o = a->t[7];
    const int64_t H = pi(a, "heads"), L = pi(a, "latent"), R = pi(a, "rope");
    const int64_t bs = pi(a, "block_size"), window = pi(a, "window", 0);
    /* the card row's limits; the accumulator lives on the stack (a launch does not allocate memory) */
    constexpr int64_t kMaxL = 512, kMaxR = 64;
    if (H < 1 || L < 1 || L > kMaxL || R < 1 || R > kMaxR || bs < 1 || window < 0) return RAD_E_SHAPE;
    if (!is_float(q) || !is_float(qr) || !is_float(kv) || !is_float(o) || sk.dtype != RAD_F32) return RAD_E_SHAPE;
    /* [M, heads, width] or [M, heads·width] (contiguous heads); a 2-D q_rope carries head h in
     * columns [h·rope_pitch, h·rope_pitch + rope) (a view of the unabsorbed q, [rope | nope] per
     * head). */
    const int64_t rp = pi(a, "rope_pitch", R);
    if (!is_heads_view(q, H, L, L) || !is_heads_view(qr, H, R, rp) || !is_heads_view(o, H, L, L) || rp < R)
        return RAD_E_SHAPE;
    if (kv.rank != 4 || kv.shape[1] != 1 || kv.shape[2] != bs || kv.shape[3] < L + R) return RAD_E_SHAPE;
    const int64_t rows = q.shape[0];
    const int64_t q_ds = q.stride[q.rank - 1], q_hs = q.rank == 3 ? q.stride[1] : L * q_ds;
    const int64_t r_ds = qr.stride[qr.rank - 1], r_hs = qr.rank == 3 ? qr.stride[1] : rp * r_ds;
    const int64_t o_ds = o.stride[o.rank - 1], o_hs = o.rank == 3 ? o.stride[1] : L * o_ds;
    if (rows < 1 || qr.shape[0] != rows || o.shape[0] < rows || !is_int(su) || !is_int(bt) || bt.rank > 2)
        return RAD_E_SHAPE;
    int64_t nsk = 1;
    for (uint32_t i = 0; i < sk.rank; ++i) nsk *= sk.shape[i];
    if (nsk != H) return RAD_E_SHAPE;
    int64_t n_seq = 1;
    for (uint32_t i = 0; i < su.rank; ++i) n_seq *= su.shape[i];
    if (n_seq < 1) return RAD_E_SHAPE;
    if (cus && count_seqs(*cus, 0, rows) != n_seq) return RAD_E_SHAPE;
    if (!cus && rows % n_seq) return RAD_E_SHAPE;
    int64_t btn = 1;
    for (uint32_t i = 0; i < bt.rank; ++i) btn *= bt.shape[i];
    if (bt.rank == 2 && bt.shape[0] < n_seq) return RAD_E_SHAPE;
    const int64_t max_blocks = bt.rank == 2 ? bt.shape[1] : btn / n_seq;
    if (max_blocks < 1) return RAD_E_SHAPE;
    const int64_t bt_rs = bt.rank == 2 ? bt.stride[0] : max_blocks * bt.stride[0];
    const int64_t bt_cs = bt.rank == 2 ? bt.stride[1] : bt.stride[0];
    const double scale = pf(a, "scale", 1.0 / std::sqrt((double)(L + R)));
    const int64_t pages = kv.shape[0];
    /* Block keys (the DFly2 drafter, docs/DFLY2.md): only in the selection form with one row a
     * "sequence" of seqused, both operands or neither, and no window (a selection's positions are
     * not contiguous, and the drafter's layers are sparse ones, which run without). Row i of block
     * sequence d also sees bkeys[bcu[d] .. i]. */
    int64_t n_bseq = 0;
    if (bk || bcu) {
        if (!bk || !bcu || cus || window > 0) return RAD_E_INVAL;
        if (n_seq != rows || !is_rank(*bk, 2) || bk->shape[0] < rows || bk->shape[1] < L + R || !is_float(*bk))
            return RAD_E_SHAPE;
        n_bseq = count_seqs(*bcu, pi(a, "max_seqs", 0), rows);
        if (n_bseq < 1) return RAD_E_SHAPE;
    }
#pragma omp parallel for collapse(2) schedule(dynamic) if (rows * H > 8)
    for (int64_t tok = 0; tok < rows; ++tok) {
        for (int64_t h = 0; h < H; ++h) {
            int64_t s, i, qlen;
            if (cus) {
                s = 0;
                while (s + 1 < n_seq && ldi(*cus, (s + 1) * cus->stride[0]) <= tok) ++s;
                const int64_t beg = ldi(*cus, s * cus->stride[0]), end = ldi(*cus, (s + 1) * cus->stride[0]);
                if (tok < beg || tok >= end) continue;   // outside cu's slices: nothing is written
                qlen = end - beg;
                i = tok - beg;
            } else {
                qlen = rows / n_seq;
                s = tok / qlen;
                i = tok - s * qlen;
            }
            /* the block keys this row sees: [b0, tok] of its block sequence; outside bcu's spans the
             * row is not written */
            int64_t b0 = 0, b1 = -1;
            if (bk) {
                int64_t d = 0;
                while (d + 1 < n_bseq && ldi(*bcu, (d + 1) * bcu->stride[0]) <= tok) ++d;
                b0 = ldi(*bcu, d * bcu->stride[0]);
                if (tok < b0 || tok >= ldi(*bcu, (d + 1) * bcu->stride[0])) continue;
                b1 = tok;
            }
            const int64_t ob = tok * o.stride[0] + h * o_hs;
            const int64_t ctx = ldi(su, s * su.stride[0]);
            const int64_t pos = ctx - qlen + i;
            int64_t j0 = 0;
            const int64_t j1 = std::min(pos, ctx - 1);
            if (window > 0) j0 = std::max<int64_t>(j0, pos - window + 1);
            /* key j: its offset in the cache, or -1 if it is not there */
            auto key = [&](int64_t j) -> int64_t {
                if (j / bs >= max_blocks) return -1;
                const int64_t blk = ldi(bt, s * bt_rs + (j / bs) * bt_cs);
                return blk < 0 || blk >= pages ? -1 : blk * kv.stride[0] + (j % bs) * kv.stride[2];
            };
            /* a key row: `t` (the cache, or bkeys), its offset and column stride */
            auto score_of = [&](const RadTensor& t, int64_t k, int64_t cs) {
                double d = 0.0;
                for (int64_t c = 0; c < L; ++c)
                    d += (double)ld(q, tok * q.stride[0] + h * q_hs + c * q_ds) * ld(t, k + c * cs);
                for (int64_t c = 0; c < R; ++c)
                    d += (double)ld(qr, tok * qr.stride[0] + h * r_hs + c * r_ds) * ld(t, k + (L + c) * cs);
                return d * scale;
            };
            const int64_t kcs = kv.stride[3], bcs = bk ? bk->stride[1] : 0;
            /* two passes: the maximum, then weights and sum; without storing the scores */
            const double sink = ld(sk, h * sk.stride[sk.rank - 1]);
            double m = sink;
            bool any_valid = false;
            for (int64_t j = j0; j <= j1; ++j) {
                const int64_t k = key(j);
                if (k < 0) continue;
                m = std::max(m, score_of(kv, k, kcs));
                any_valid = true;
            }
            for (int64_t j = b0; j <= b1; ++j) {
                m = std::max(m, score_of(*bk, j * bk->stride[0], bcs));
                any_valid = true;
            }
            if (!any_valid) {
                for (int64_t d = 0; d < L; ++d) st(o, ob + d * o_ds, 0.0f);
                continue;
            }
            double acc[kMaxL] = {};
            double den = std::exp(sink - m);
            for (int64_t j = j0; j <= j1; ++j) {
                const int64_t k = key(j);
                if (k < 0) continue;
                const double w = std::exp(score_of(kv, k, kcs) - m);
                den += w;
                for (int64_t c = 0; c < L; ++c) acc[c] += w * ld(kv, k + c * kcs);
            }
            for (int64_t j = b0; j <= b1; ++j) {
                const int64_t k = j * bk->stride[0];
                const double w = std::exp(score_of(*bk, k, bcs) - m);
                den += w;
                for (int64_t c = 0; c < L; ++c) acc[c] += w * ld(*bk, k + c * bcs);
            }
            for (int64_t c = 0; c < L; ++c) st(o, ob + c * o_ds, (float)(acc[c] / den));
        }
    }
    return RAD_OK;
}

/* hy4_mla_project: the oracle of the card row, libref's ref_mla_project with the same view rules
 * ([M, heads*width] taken as [M, heads, width]), summing in f32 in ascending j. */
int hy4_ref_mla_project(const RadArgs* a, RadStream) {
    if (a->n_t < 3) return RAD_E_INVAL;
    RadTensor x = a->t[0], y = a->t[2];
    const RadTensor& w = a->t[1];
    const int64_t h = pi(a, "heads"), d = pi(a, "head_dim"), l = pi(a, "latent_dim");
    const char* mode = rad_args_gets(a, "mode");
    const bool absorb = mode && !std::strcmp(mode, "absorb"), expand = mode && !std::strcmp(mode, "expand");
    auto heads_view = [&](RadTensor& t, int64_t width) {
        if (h > 0 && width > 0 && t.rank == 2 && t.shape[1] == h * width) {
            t.rank = 3;
            t.shape[1] = h;
            t.shape[2] = width;
            t.stride[2] = t.stride[1];
            t.stride[1] *= width;
        }
    };
    heads_view(x, absorb ? d : l);
    heads_view(y, absorb ? l : d);
    if ((!absorb && !expand) || h < 1 || d < 1 || l < 1 || x.rank != 3 || w.rank != 2 || y.rank != 3 ||
        x.shape[1] != h || x.shape[2] != (absorb ? d : l) || w.shape[0] != h * 2 * d || w.shape[1] != l ||
        y.shape[0] != x.shape[0] || y.shape[1] != h || y.shape[2] != (absorb ? l : d))
        return RAD_E_SHAPE;
    if (!x.data || !w.data || !y.data) return RAD_E_INVAL;
#pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < x.shape[0]; ++r)
        for (int64_t head = 0; head < h; ++head)
            for (int64_t col = 0; col < (absorb ? l : d); ++col) {
                float sum = 0;
                for (int64_t j = 0; j < (absorb ? d : l); ++j) {
                    const int64_t wr = head * 2 * d + (absorb ? j : d + col), wc = absorb ? col : j;
                    sum += ld(x, r * x.stride[0] + head * x.stride[1] + j * x.stride[2]) *
                           ld(w, wr * w.stride[0] + wc * w.stride[1]);
                }
                st(y, r * y.stride[0] + head * y.stride[1] + col * y.stride[2], sum);
            }
    return RAD_OK;
}
