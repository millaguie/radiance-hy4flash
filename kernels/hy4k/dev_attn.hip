/* dev_attn.hip -- card row of attn_paged_sink (docs/OPS.md) for Hy4's geometry, which libr4d
 * does not serve: head_dim 256, gqa 1, pages of 4. The semantics are libavx/avx_attn.cpp's
 * (attn_paged_rows): row `tok` of sequence `s`, position `seqused[s] - qlen + i`, causal keys
 * `j ≤ pos` within the window `pos - j < window`, page `bt[s][j / block]`, the sink at the end as
 * one more logit with no value. With a `seqused` of one entry per row and no `cu` it is the form
 * with selection (QSA): each row is its own sequence of q_len 1 and `bt` its list of pages.
 *
 * One 256-thread block per (row, query head); thread d carries dimension d of the output. Keys
 * come in strips of 32: each wave scores 4 keys (8 lanes per key), and the online softmax is
 * libavx's. A simple, correct form without WMMA: speed is left for later (plan step 10). */
#include "dev_common.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace hy4k {
namespace {

constexpr int kStrip = 32;

template <class Ch>
__global__ __launch_bounds__(256) void attn_sink_k(const void* q, int qdt, long q_rs, long q_hs, long q_ds,
                                                   const void* kv, int kvdt, long sb, long sh, long so, long sd, long pages,
                                                   const void* bt, int bit, long bt_rs, long bt_cs, long max_blocks,
                                                   const void* su, int sit, long sus, const void* cus, int cit,
                                                   long cs, int n_seq, int q_len, const float* sinks, long sks,
                                                   void* o, int odt, long o_rs, long o_hs, long o_ds, int head_dim,
                                                   int gqa, int block_size, int window, float scale) {
    HY4_CHAIN_BODY {
    __shared__ float qd[256];
    __shared__ float sc[kStrip];
    __shared__ long kb[kStrip];
    const long tok = blockIdx.x;
    const int h = blockIdx.y, kh = h / gqa;
    const int t = threadIdx.x, lane = t & 31, wave = t >> 5;
    long s, i, qlen;
    if (cus) {
        s = 0;
        while (s + 1 < n_seq && dldi(cus, cit, (s + 1) * cs) <= tok) ++s;
        const long beg = dldi(cus, cit, s * cs), end = dldi(cus, cit, (s + 1) * cs);
        if (tok < beg || tok >= end) return;   // row outside cu's slices
        qlen = end - beg;
        i = tok - beg;
    } else {
        s = tok / q_len;
        i = tok - s * q_len;
        qlen = q_len;
    }
    const long ctx = dldi(su, sit, s * sus);
    const long pos = ctx - qlen + i;
    if (t < head_dim) qd[t] = dld(q, qdt, tok * q_rs + (long)h * q_hs + (long)t * q_ds);
    lds_barrier();
    long j0 = 0, j1 = pos < ctx - 1 ? pos : ctx - 1;   // causal: j ≤ pos, and j < ctx
    if (window > 0 && pos - window + 1 > j0) j0 = pos - window + 1;
    float m = -INFINITY, l = 0.0f, acc = 0.0f;
    for (long base = j0; base <= j1; base += kStrip) {
        /* the strip's pages */
        if (t < kStrip) {
            const long j = base + t;
            long k = -1;
            if (j <= j1 && j / block_size < max_blocks) {
                const long blk = dldi(bt, bit, s * bt_rs + (j / block_size) * bt_cs);
                if (blk >= 0 && blk < pages) k = blk * sb + (long)kh * sh + (j % block_size) * so;
            }
            kb[t] = k;
        }
        lds_barrier();
        /* scores: wave w scores keys 4w..4w+3, 8 lanes per key */
        {
            const int c = wave * 4 + (lane >> 3), sub = lane & 7;
            const long k = kb[c];
            float part = 0.0f;
            if (k >= 0)
                for (int d = sub; d < head_dim; d += 8) part += qd[d] * dld(kv, kvdt, k + (long)d * sd);
            for (int off = 4; off; off >>= 1) part += __shfl_xor(part, off, 8);
            if (sub == 0) sc[c] = k >= 0 ? part * scale : -INFINITY;
        }
        lds_barrier();
        float tmax = -INFINITY;
        for (int c = 0; c < kStrip; ++c) tmax = fmaxf(tmax, sc[c]);
        if (tmax > m) {
            const float r = m == -INFINITY ? 0.0f : expf(m - tmax);
            acc *= r;
            l *= r;
            m = tmax;
        }
        if (m != -INFINITY) {
            for (int c = 0; c < kStrip; ++c) {
                const long k = kb[c];
                if (k < 0) continue;
                const float w = expf(sc[c] - m);
                l += w;
                if (t < head_dim) acc += w * dld(kv, kvdt, k + (long)(head_dim + t) * sd);
            }
        }
        lds_barrier();
    }
    if (t >= head_dim) return;
    const long ob = tok * o_rs + (long)h * o_hs + (long)t * o_ds;
    if (m == -INFINITY) {   // no visible key: the row is left at zero, without the sink
        dst(o, odt, ob, 0.0f);
        return;
    }
    if (sinks) {
        const float sk = sinks[(long)h * sks];
        if (sk > m) {
            const float r = expf(m - sk);
            acc *= r;
            l = l * r + 1.0f;
        } else {
            l += expf(sk - m);
        }
    }
    dst(o, odt, ob, acc * (l > 0.0f ? 1.0f / l : 0.0f));
    } HY4_CHAIN_END
}


/* hy4_mla_attn: the same form over the latent cache (one KV head for all; row [lat | rope]).
 * One 256-thread block per (row, head); thread t carries dimensions t and t+256 of the latent
 * output (L ≤ 512). The absorbed query [q | q_rope] goes to LDS; each wave scores 4 keys of the
 * strip with 8 lanes per key over the L+R columns. Simple and correct: sharing one strip of keys
 * among all the heads of the row is left for when it is measured (plan step 10). */
constexpr int kLat = 512, kRopeMax = 64;

template <class Ch>
__global__ __launch_bounds__(256) void mla_attn_k(const void* q, int qdt, long q_rs, long q_hs, long q_ds,
                                                  const void* qr, int qrdt, long qr_rs, long qr_hs, long qr_ds,
                                                  const void* kv, int kvdt, long sb, long so, long sd, long pages,
                                                  const void* bt, int bit, long bt_rs, long bt_cs, long max_blocks,
                                                  const void* su, int sit, long sus, const void* cus, int cit,
                                                  long cs, int n_seq, int q_len, const float* sinks, long sks,
                                                  void* o, int odt, long o_rs, long o_hs, long o_ds, int L, int R,
                                                  int block_size, int window, float scale) {
    HY4_CHAIN_BODY {
    __shared__ float qd[kLat + kRopeMax];
    __shared__ float sc[kStrip];
    __shared__ long kb[kStrip];
    const long tok = blockIdx.x;
    const int h = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, wave = t >> 5;
    const int D = L + R;
    long s, i, qlen;
    if (cus) {
        s = 0;
        while (s + 1 < n_seq && dldi(cus, cit, (s + 1) * cs) <= tok) ++s;
        const long beg = dldi(cus, cit, s * cs), end = dldi(cus, cit, (s + 1) * cs);
        if (tok < beg || tok >= end) return;   // row outside cu's slices: nothing is written
        qlen = end - beg;
        i = tok - beg;
    } else {
        s = tok / q_len;
        i = tok - s * q_len;
        qlen = q_len;
    }
    const long ctx = dldi(su, sit, s * sus);
    const long pos = ctx - qlen + i;
    for (int d = t; d < L; d += 256) qd[d] = dld(q, qdt, tok * q_rs + (long)h * q_hs + (long)d * q_ds);
    if (t < R) qd[L + t] = dld(qr, qrdt, tok * qr_rs + (long)h * qr_hs + (long)t * qr_ds);
    lds_barrier();
    long j0 = 0, j1 = pos < ctx - 1 ? pos : ctx - 1;
    if (window > 0 && pos - window + 1 > j0) j0 = pos - window + 1;
    float m = -INFINITY, l = 0.0f, a0 = 0.0f, a1 = 0.0f;
    const int d0 = t, d1 = t + 256;
    for (long base = j0; base <= j1; base += kStrip) {
        if (t < kStrip) {
            const long j = base + t;
            long k = -1;
            if (j <= j1 && j / block_size < max_blocks) {
                const long blk = dldi(bt, bit, s * bt_rs + (j / block_size) * bt_cs);
                if (blk >= 0 && blk < pages) k = blk * sb + (j % block_size) * so;
            }
            kb[t] = k;
        }
        lds_barrier();
        {
            const int c = wave * 4 + (lane >> 3), sub = lane & 7;
            const long k = kb[c];
            float part = 0.0f;
            if (k >= 0)
                for (int d = sub; d < D; d += 8) part += qd[d] * dld(kv, kvdt, k + (long)d * sd);
            for (int off = 4; off; off >>= 1) part += __shfl_xor(part, off, 8);
            if (sub == 0) sc[c] = k >= 0 ? part * scale : -INFINITY;
        }
        lds_barrier();
        float tmax = -INFINITY;
        for (int c = 0; c < kStrip; ++c) tmax = fmaxf(tmax, sc[c]);
        if (tmax > m) {
            const float r = m == -INFINITY ? 0.0f : expf(m - tmax);
            a0 *= r;
            a1 *= r;
            l *= r;
            m = tmax;
        }
        if (m != -INFINITY) {
            for (int c = 0; c < kStrip; ++c) {
                const long k = kb[c];
                if (k < 0) continue;
                const float w = expf(sc[c] - m);
                l += w;
                if (d0 < L) a0 += w * dld(kv, kvdt, k + (long)d0 * sd);
                if (d1 < L) a1 += w * dld(kv, kvdt, k + (long)d1 * sd);
            }
        }
        lds_barrier();
    }
    const long ob = tok * o_rs + (long)h * o_hs;
    if (m == -INFINITY) {
        if (d0 < L) dst(o, odt, ob + (long)d0 * o_ds, 0.0f);
        if (d1 < L) dst(o, odt, ob + (long)d1 * o_ds, 0.0f);
        return;
    }
    if (sinks) {
        const float sk = sinks[(long)h * sks];
        if (sk > m) {
            const float r = expf(m - sk);
            a0 *= r;
            a1 *= r;
            l = l * r + 1.0f;
        } else {
            l += expf(sk - m);
        }
    }
    const float inv = l > 0.0f ? 1.0f / l : 0.0f;
    if (d0 < L) dst(o, odt, ob + (long)d0 * o_ds, a0 * inv);
    if (d1 < L) dst(o, odt, ob + (long)d1 * o_ds, a1 * inv);
    } HY4_CHAIN_END
}

/* hy4_mla_attn, form 2: libr4d's `attend_paged` (`libr4d/r4d_mla.hip`) with what Hy4 adds (the sink,
 * the window, the selection form). BF16, latent 512, rope 64. One wave per slice of a (row, head)'s
 * history, kMlaW slices; a lane holds 16 latent and 2 rope columns, so a score is one wave sum. The
 * slices meet in LDS in a fixed order, and the split depends only on the row's visible keys, never on
 * the batch: a row gives the same bits alone or in a batch (up to 32 rows, where form 3 takes over).
 * Form 1 gave every (row, head) a 256-thread block over 32-key strips: 3.5x slower here.
 * Form 2g (mla_attn2g_k, below form 3; opt-in, HY4_MLA_ATTN=g) serves these calls when it is on; this
 * per-head form is the default and its fallback (heads not a multiple of 16, no scratch). */
constexpr int kMlaW = 8;
__device__ __forceinline__ void load16(const uint16_t* p, float (&v)[16]) {
    const uint4 a = ((const uint4*)p)[0], b = ((const uint4*)p)[1];
    const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        v[2 * k] = __uint_as_float(w[k] << 16);
        v[2 * k + 1] = __uint_as_float(w[k] & 0xffff0000u);
    }
}
/* kBk: the DFly2 drafter's block keys (docs/DFLY2.md). Row `tok` of block sequence d (bcu) also sees
 * bk rows bcu[d] .. tok, after its cached keys in the slices' order, so the history keys keep the
 * indices they have without block keys; the instance without them is the plain form 2, whose bits
 * stay as they were. */
template <class Ch, bool kBk>
__global__ __launch_bounds__(kMlaW * 32) void mla_attn2_k(const uint16_t* q, long q_rs, long q_hs,
                                                          const uint16_t* qr, long qr_rs, long qr_hs,
                                                          const uint16_t* kv, long sb, long so, long pages,
                                                          const void* bt, int bit, long bt_rs, long bt_cs,
                                                          long max_blocks, const void* su, int sit, long sus,
                                                          const void* cus, int cit, long cs, int n_seq, int q_len,
                                                          const float* sinks, long sks, uint16_t* o, long o_rs,
                                                          long o_hs, int block_size, int window, float scale,
                                                          const uint16_t* bk, long bk_rs, const void* bcu, int bcit,
                                                          long bcs, int n_bseq) {
    HY4_CHAIN_BODY {
    __shared__ float partial[kMlaW][kLat];
    __shared__ float pm[kMlaW], pd[kMlaW];
    __shared__ int pc[kMlaW];   // keys the slice scored: with a NaN, pd cannot tell
    __shared__ int pv[kMlaW];   // keys on a valid page, whatever they score: the oracle's `any_valid`
    const long tok = blockIdx.x;
    const int h = blockIdx.y;
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    long s, i, qlen;
    if (cus) {
        s = 0;
        while (s + 1 < n_seq && dldi(cus, cit, (s + 1) * cs) <= tok) ++s;
        const long beg = dldi(cus, cit, s * cs), end = dldi(cus, cit, (s + 1) * cs);
        if (tok < beg || tok >= end) return;
        qlen = end - beg;
        i = tok - beg;
    } else {
        s = tok / q_len;
        i = tok - s * q_len;
        qlen = q_len;
    }
    long b0 = 0, nb = 0;   // the block keys: bk rows b0 .. b0 + nb - 1
    if (kBk) {
        long d = 0;
        while (d + 1 < n_bseq && dldi(bcu, bcit, (d + 1) * bcs) <= tok) ++d;
        b0 = dldi(bcu, bcit, d * bcs);
        if (b0 < 0 || tok < b0 || tok >= dldi(bcu, bcit, (d + 1) * bcs)) return;   // outside bcu: not written
        nb = tok - b0 + 1;
    }
    const long ctx = dldi(su, sit, s * sus);
    const long pos = ctx - qlen + i;
    long j0 = 0, j1 = pos < ctx - 1 ? pos : ctx - 1;
    if (window > 0 && pos - window + 1 > j0) j0 = pos - window + 1;
    if (j1 > max_blocks * block_size - 1) j1 = max_blocks * block_size - 1;   // a corrupt seqused cannot spin the loop
    const long nh = j1 >= j0 ? j1 - j0 + 1 : 0;
    /* the slices split the union: the nh cached keys first, then the nb block keys */
    const long n = nh + nb;
    const long begin = j0 + n * wave / kMlaW, end = j0 + n * (wave + 1) / kMlaW;
    float query[16], value[16];
    load16(q + tok * q_rs + (long)h * q_hs + lane * 16, query);
    const uint16_t* qp = qr + tok * qr_rs + (long)h * qr_hs + lane * 2;
    const float rq0 = bf2f(qp[0]), rq1 = bf2f(qp[1]);
#pragma unroll
    for (int k = 0; k < 16; ++k) value[k] = 0.0f;
    float mx = -INFINITY, den = 0.0f;
    int cnt = 0, vis = 0;
    for (long j = begin; j < end; ++j) {
        const uint16_t* row;
        if (kBk && j - j0 >= nh) {
            row = bk + (b0 + (j - j0 - nh)) * bk_rs;
        } else {
            const long page = dldi(bt, bit, s * bt_rs + (j / block_size) * bt_cs);
            if (page < 0 || page >= pages) continue;
            row = kv + page * sb + (j % block_size) * so;
        }
        float key[16];
        load16(row + lane * 16, key);
        float sc = rq0 * bf2f(row[kLat + lane * 2]) + rq1 * bf2f(row[kLat + lane * 2 + 1]);
#pragma unroll
        for (int k = 0; k < 16; ++k) sc += query[k] * key[k];
        for (int off = 16; off; off >>= 1) sc += __shfl_xor(sc, off, 32);
        sc *= scale;
        ++vis;
        if (sc == -INFINITY) continue;   // weight 0, as in the oracle (and no -inf - -inf)
        ++cnt;
        const bool rises = sc > mx;
        const float w = expf(rises ? mx - sc : sc - mx);
        if (rises) {
#pragma unroll
            for (int k = 0; k < 16; ++k) value[k] = value[k] * w + key[k];
            den = den * w + 1.0f;
            mx = sc;
        } else {
#pragma unroll
            for (int k = 0; k < 16; ++k) value[k] += key[k] * w;
            den += w;
        }
    }
#pragma unroll
    for (int k = 0; k < 16; ++k) partial[wave][lane * 16 + k] = value[k];
    if (!lane) { pm[wave] = mx; pd[wave] = den; pc[wave] = cnt; pv[wave] = vis; }
    lds_barrier();
    if (wave) return;
    /* A NaN in q or in a cache row leaves den NaN in its slice: it enters the merge and the output is
     * NaN, as in the oracle. */
    float m = -INFINITY;
    bool nan = false;
    for (int w = 0; w < kMlaW; ++w)
        if (pc[w] > 0) {
            if (pm[w] != pm[w] || pd[w] != pd[w]) nan = true;   // an all-NaN slice leaves mx at -inf
            else m = fmaxf(m, pm[w]);
        }
    /* The oracle takes M = max(sink, scores) over the visible keys: an infinite M (a +inf score or
     * sink, or every score and the sink -inf) makes its exp(. - M) NaN. A row with no visible key
     * is zero whatever the sink. */
    int any = 0;
    for (int w = 0; w < kMlaW; ++w) any += pv[w];
    const float skv = sinks ? sinks[(long)h * sks] : -INFINITY;
    const float big = fmaxf(m, skv);
    if (any && (big == INFINITY || big == -INFINITY)) nan = true;
    if (nan) m = NAN;
    uint16_t* out = o + tok * o_rs + (long)h * o_hs + lane * 16;
    if (!any || m == -INFINITY) {   // no visible key, or only -inf scores beside a finite sink: zeros
#pragma unroll
        for (int k = 0; k < 16; ++k) out[k] = 0;
        return;
    }
    float acc[16], l = 0.0f;
#pragma unroll
    for (int k = 0; k < 16; ++k) acc[k] = 0.0f;
    for (int w = 0; w < kMlaW; ++w) {
        if (pc[w] == 0) continue;
        const float e = expf(pm[w] - m);
        l += pd[w] * e;
#pragma unroll
        for (int k = 0; k < 16; ++k) acc[k] += partial[w][lane * 16 + k] * e;
    }
    if (sinks) {   // one more logit, with no value
        const float sk = sinks[(long)h * sks];
        if (sk > m) {
            const float r = expf(m - sk);
#pragma unroll
            for (int k = 0; k < 16; ++k) acc[k] *= r;
            l = l * r + 1.0f;
        } else {
            l += expf(sk - m);
        }
    }
    const float inv = !(l <= 0.0f) ? 1.0f / l : 0.0f;   // a NaN sink makes l NaN: let it through
#pragma unroll
    for (int k = 0; k < 16; ++k) out[k] = f2bf(acc[k] * inv);
    } HY4_CHAIN_END
}

/* hy4_mla_attn, form 3: libr4d's `attend_matrix` (`libr4d/r4d_mla.hip`, without PAIR, UNION or splits)
 * for batches of more than 32 rows, i.e. prefill. One 4-wave block per (row, 16 heads): the heads are
 * the matrix rows, so each key is loaded once for 16 heads instead of once per head. Per 16-key tile:
 * S = Q K^T with 16x16x16 BF16 WMMA over all 576 columns (latent and rope), an online f32 softmax per
 * head, P split into high and low BF16 halves (about 16 significant bits instead of 8, which a single
 * BF16 P would keep), and O += P V with V the latent part of the same keys. Hy4 adds the rope columns,
 * the sink, the window and the selection form; a NaN reaches the output. A row does not give form 2's
 * bits here: the choice is by batch size (> 32 rows), as libr4d's, which keeps batch invariance only
 * up to 32 rows. Within form 3 a row's bits depend only on the row (its tiles start at its own first
 * key), so a row gives the same bits in any batch of more than 32.
 *
 * Speed (2026-11-21, one R9700, tests/bench_attn.py): a 2048-row window-512 chunk at 32 heads went
 * from 6.5 to 2.7 ms, 16 heads from 3.8 to 1.5 ms, the 256-row selection of 2048 keys from 1.85 to
 * 0.65 ms. Taking parts out one at a time (timing only) showed where the time went:
 *   - Q was copied to LDS with 2-byte loads, a fixed 1 ms. Now each lane loads its A fragments: wave w
 *     keeps the nine of columns 144w .. 144w + 143 in registers (kSplit), and the 18 KB of LDS Q took
 *     are gone, which lets more blocks share a WGP.
 *   - The softmax ran on 16 threads, 16 keys each: 2 ms. Now 128 threads, two keys a lane.
 *   - S ran on wave 0 alone, a chain of 36 WMMAs with three waves waiting. Now each wave scores its
 *     column quarter (nine WMMAs) and the softmax adds the four in a fixed order. This is the one step
 *     that changes the bits (a different f32 grouping of the same products: about 0.1% of outputs move
 *     by one BF16 rounding, L2 5e-5 against the one-wave form). It is opt-in (HY4_MLA_ATTN=4): the
 *     default keeps the one-wave score (kSplit false: the old bits, 3.3 ms on the chunk above), because
 *     the split moved the full model's KL tails just past the rounding envelope.
 *   - The key loads are not the bound: without them the chunk takes 8% less, since a chunk's keys
 *     (3 MB) stay in the card's caches. Sharing a key tile between rows in LDS would save part of that
 *     8% and cost a block's LDS per extra row; not done.
 * Tried and dropped: 32-key tiles (two waves scoring; 62 KB of LDS and register spills, slower), the
 * softmax in every wave's registers to save a barrier and the P round trip (4x the exponentials,
 * slower), each wave owning a column slice of the tile in its own LDS so only two barriers remain
 * (the same time: the barriers are not the bound either). What is left is spread over the S and P V
 * WMMAs, the transposed V reads and the softmax, none above 15%.
 * gfx11 has no gfx12 WMMA and runs the same arithmetic without matrices: correct, slow. */
constexpr int kQP = 584;   // 576 columns and 8 of padding: 16-byte LDS rows off the bank stride
#if defined(__GFX12__)
typedef __bf16 v8bf3 __attribute__((ext_vector_type(8)));
typedef float v8f3 __attribute__((ext_vector_type(8)));
typedef short v8s3 __attribute__((ext_vector_type(8)));
/* A gfx12 A/B fragment (row = lane % 16; k = 8*(e>>2) + 4*(lane>>4) + (e&3), libr4d/r4d_gdn_wmma.h)
 * from an LDS source [row][k] at `pitch`: two runs of four, 16 bytes apart. */
__device__ __forceinline__ v8bf3 frag3(const uint16_t* src, int pitch, int base, int kbase, int lane) {
    const uint2* pw = (const uint2*)(src + (base + (lane & 15)) * pitch + 4 * (lane >> 4) + kbase);
    union { uint2 w[2]; v8bf3 f; } u;
    u.w[0] = pw[0];
    u.w[1] = pw[2];
    return u.f;
}
__device__ __forceinline__ v8f3 mma3(v8bf3 a, v8bf3 b, v8f3 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}
#endif
template <class Ch, bool kSplit>
__global__ __launch_bounds__(128) void mla_attn3_k(const uint16_t* q, long q_rs, long q_hs,
                                                   const uint16_t* qr, long qr_rs, long qr_hs,
                                                   const uint16_t* kv, long sb, long so, long pages,
                                                   const void* bt, int bit, long bt_rs, long bt_cs,
                                                   long max_blocks, const void* su, int sit, long sus,
                                                   const void* cus, int cit, long cs, int n_seq, int q_len,
                                                   const float* sinks, long sks, uint16_t* o, long o_rs,
                                                   long o_hs, int block_size, int window, float scale) {
    HY4_CHAIN_BODY {
    const long tok = blockIdx.x;
    const int h0 = blockIdx.y * 16;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    long s, i, qlen;
    if (cus) {
        s = 0;
        while (s + 1 < n_seq && dldi(cus, cit, (s + 1) * cs) <= tok) ++s;
        const long beg = dldi(cus, cit, s * cs), end = dldi(cus, cit, (s + 1) * cs);
        if (tok < beg || tok >= end) return;
        qlen = end - beg;
        i = tok - beg;
    } else {
        s = tok / q_len;
        i = tok - s * q_len;
        qlen = q_len;
    }
    const long ctx = dldi(su, sit, s * sus);
    const long pos = ctx - qlen + i;
    long j0 = 0, j1 = pos < ctx - 1 ? pos : ctx - 1;
    if (window > 0 && pos - window + 1 > j0) j0 = pos - window + 1;
    if (j1 > max_blocks * block_size - 1) j1 = max_blocks * block_size - 1;
#if defined(__GFX12__)
    __shared__ __align__(16) uint16_t keys[16 * kQP], prob_hi[16 * 24], prob_lo[16 * 24];
    __shared__ __align__(16) float score[kSplit ? 4 : 1][16 * 16];
    __shared__ float alpha[16], fin[16];
    __shared__ long source[2][16];
    __shared__ int rowvis, rescale;   // rowvis: a key on a valid page was seen (the oracle's `any_valid`)
    const int m = lane & 15, hh = lane >> 4;   // the lane's fragment row or column, and half
    /* Q's A fragments come straight from memory (the row's 18 KB stay in the near caches): in LDS they
     * took 18 KB of the block's 40 and held the card at fewer blocks */
    const uint16_t* qa = q + tok * q_rs + (long)(h0 + m) * q_hs + 4 * hh;
    const uint16_t* qb = qr + tok * qr_rs + (long)(h0 + m) * qr_hs + 4 * hh - kLat;
    auto qfrag = [&](int d) -> v8bf3 {
        const uint16_t* p = d < kLat ? qa + d : qb + d;
        union { uint2 w[2]; v8bf3 f; } A;
        A.w[0] = *(const uint2*)p;
        A.w[1] = *(const uint2*)(p + 8);
        return A.f;
    };
    v8bf3 qf[kSplit ? 9 : 1];   // kSplit: wave w keeps columns 144w .. 144w + 143 in registers
    if (kSplit) {
#pragma unroll
        for (int k = 0; k < 9; ++k) qf[k] = qfrag(wave * 144 + k * 16);
    }
    /* positions fit in 32 bits (the host checks max_blocks * block_size): no 64-bit division */
    auto key_row = [&](long j) -> long {
        if (j > j1) return -1;
        const unsigned ju = (unsigned)j, bs = (unsigned)block_size, blk = ju / bs;
        const long page = dldi(bt, bit, s * bt_rs + (long)blk * bt_cs);
        return page >= 0 && page < pages ? page * sb + (long)(ju - blk * bs) * so : -1;
    };
    /* the page lookups run a tile ahead of the key loads: `ahead` holds the tile after the next */
    long ahead = -1;
    if (tid < 16) {
        source[0][tid] = key_row(j0 + tid);
        ahead = key_row(j0 + 16 + tid);
    }
    if (!tid) rowvis = 0;
    /* the online softmax state of head tid >> 3, the same in its 8 lanes */
    float maximum = -INFINITY, denominator = 0.0f;
    int nan_seen = 0;
    v8f3 value[8];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) value[nt] = (v8f3){0, 0, 0, 0, 0, 0, 0, 0};
    lds_barrier();
    /* The next tile's keys are fetched into registers while the current one is multiplied, as
     * libr4d's attend_rows does: 16 keys x 72 groups of 8 columns, nine 16-byte loads a thread. */
    uint4 pre[9];
    auto fetch = [&](const long* src) {
#pragma unroll
        for (int k = 0; k < 9; ++k) {
            const int f = tid + k * 128, key = f / 72, c8 = f % 72;
            const long at = src[key];
            pre[k] = at >= 0 ? *(const uint4*)(kv + at + c8 * 8) : make_uint4(0, 0, 0, 0);
        }
    };
    fetch(source[0]);
    int cur = 0;
    for (long start = j0; start <= j1; start += 16, cur ^= 1) {
#pragma unroll
        for (int k = 0; k < 9; ++k) {
            const int f = tid + k * 128, key = f / 72, c8 = f % 72;
            *(uint4*)(keys + key * kQP + c8 * 8) = pre[k];
        }
        const bool more = start + 16 <= j1;
        if (tid < 16) {
            source[cur ^ 1][tid] = ahead;
            ahead = key_row(start + 32 + tid);
        }
        if (!tid) rescale = 0;
        lds_barrier();
        if (more) fetch(source[cur ^ 1]);
        /* S = Q K^T over the 576 columns (latent and rope) */
        if (kSplit) {   // wave w: columns 144w .. 144w + 143, raw dots
            v8f3 dot = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int k = 0; k < 9; ++k) dot = mma3(qf[k], frag3(keys, kQP, 0, wave * 144 + k * 16, lane), dot);
            if (!wave && lane < 16 && source[cur][lane] >= 0) rowvis = 1;
#pragma unroll
            for (int j = 0; j < 8; ++j) score[wave][(hh * 8 + j) * 16 + m] = dot[j];
        } else if (!wave) {   // one wave, all columns in order
            v8f3 dot = {0, 0, 0, 0, 0, 0, 0, 0};
            for (int d0 = 0; d0 < kLat; d0 += 128) {   // in four runs: unrolled whole, the loads took 30 VGPRs more
#pragma unroll
                for (int d = d0; d < d0 + 128; d += 16) dot = mma3(qfrag(d), frag3(keys, kQP, 0, d, lane), dot);
            }
#pragma unroll
            for (int d = kLat; d < kLat + kRopeMax; d += 16) dot = mma3(qfrag(d), frag3(keys, kQP, 0, d, lane), dot);
            const bool ok = source[cur][m] >= 0;
            if (lane < 16 && ok) rowvis = 1;
#pragma unroll
            for (int j = 0; j < 8; ++j) score[0][(hh * 8 + j) * 16 + m] = ok ? dot[j] * scale : -INFINITY;
        }
        lds_barrier();
        /* The softmax on all 128 threads: 8 consecutive lanes per head, two keys a lane. The maximum is
         * exact in any order; the denominator adds the 16 weights in key order (the shuffles broadcast
         * them), so the bits are those of one thread walking the keys. P is split into high and low
         * BF16 halves. */
        {
            const int head = tid >> 3, g = tid & 7, i0 = head * 16 + 2 * g;
            float2 v = *(const float2*)(score[0] + i0);
            if (kSplit) {   // the four column quarters, added in a fixed order
                const float2 v1 = *(const float2*)(score[1] + i0), v2 = *(const float2*)(score[2] + i0),
                             v3 = *(const float2*)(score[3] + i0);
                v.x = source[cur][2 * g] >= 0 ? (((v.x + v1.x) + v2.x) + v3.x) * scale : -INFINITY;
                v.y = source[cur][2 * g + 1] >= 0 ? (((v.y + v1.y) + v2.y) + v3.y) * scale : -INFINITY;
            }
            const float s0 = v.x, s1 = v.y;
            float mx = maximum;
            int bad = (s0 != s0) | (s1 != s1);
            if (s0 == s0) mx = fmaxf(mx, s0);
            if (s1 == s1) mx = fmaxf(mx, s1);
#pragma unroll
            for (int off = 1; off < 8; off <<= 1) {
                mx = fmaxf(mx, __shfl_xor(mx, off, 8));
                bad |= __shfl_xor(bad, off, 8);
            }
            const float a = mx > -INFINITY ? expf(maximum - mx) : 1.0f;
            const float p0 = s0 > -INFINITY ? expf(s0 - mx) : 0.0f;   // -inf and NaN weigh 0 (NaN is flagged apart)
            const float p1 = s1 > -INFINITY ? expf(s1 - mx) : 0.0f;
            const uint16_t hi0 = f2bf(p0), hi1 = f2bf(p1);
            *(uint32_t*)(prob_hi + head * 24 + 2 * g) = hi0 | (uint32_t)hi1 << 16;
            *(uint32_t*)(prob_lo + head * 24 + 2 * g) = f2bf(p0 - bf2f(hi0)) | (uint32_t)f2bf(p1 - bf2f(hi1)) << 16;
            float den = denominator * a;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                den += __shfl(p0, k, 8);
                den += __shfl(p1, k, 8);
            }
            maximum = mx;
            denominator = den;
            nan_seen |= bad;
            if (!g) {
                alpha[head] = a;
                if (a != 1.0f) rescale = 1;
            }
        }
        lds_barrier();
        /* O += P V, V being the keys' latent part read transposed from the tile (B[col][key] =
         * keys[key][col], eight LDS reads a fragment); an in-place transpose costs two barriers. Once
         * the maxima settle most tiles leave every alpha at 1, and the rescale is skipped (x * 1 = x:
         * the same bits). */
        {
            const v8bf3 hi = frag3(prob_hi, 24, 0, 0, lane), lo = frag3(prob_lo, 24, 0, 0, lane);
            if (rescale) {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float aj = alpha[hh * 8 + j];
#pragma unroll
                    for (int nt = 0; nt < 8; ++nt) value[nt][j] *= aj;
                }
            }
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                const int idx = wave * 128 + nt * 16 + m, k0 = 4 * hh;
                union { uint16_t h[8]; v8bf3 f; } u;
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    u.h[e] = keys[(k0 + e) * kQP + idx];
                    u.h[4 + e] = keys[(k0 + 8 + e) * kQP + idx];
                }
                value[nt] = mma3(hi, u.f, value[nt]);
                value[nt] = mma3(lo, u.f, value[nt]);
            }
        }
        lds_barrier();
    }
    if ((tid & 7) == 0) {   // each head's output factor with the sink: 0 with no key, NaN after a NaN
        const int head = tid >> 3;
        const float mx = maximum, den = denominator;
        float f;
        const float skv = sinks ? sinks[(long)(h0 + head) * sks] : -INFINITY;
        const float big = fmaxf(mx, skv);   // the oracle's M = max(sink, scores)
        if (!rowvis) f = 0.0f;
        else if (nan_seen || den != den || big == INFINITY || big == -INFINITY) f = NAN;   // exp(inf - inf)
        else if (!(mx > -INFINITY) || !(den > 0.0f)) f = 0.0f;
        else if (sinks) f = skv > mx ? expf(mx - skv) / (den * expf(mx - skv) + 1.0f) : 1.0f / (den + expf(skv - mx));
        else f = 1.0f / den;
        fin[head] = f;
    }
    lds_barrier();
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        const int col = wave * 128 + nt * 16 + m;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int row = hh * 8 + j;
            const float f = fin[row];
            o[tok * o_rs + (long)(h0 + row) * o_hs + col] = f2bf(f == 0.0f ? 0.0f : value[nt][j] * f);
        }
    }
#else
    /* gfx11: each wave takes 4 of the 16 heads over all keys, with form 2's arithmetic */
    for (int hh = 0; hh < 4; ++hh) {
        const int h = h0 + wave * 4 + hh;
        float query[16], value[16];
        load16(q + tok * q_rs + (long)h * q_hs + lane * 16, query);
        const uint16_t* qp = qr + tok * qr_rs + (long)h * qr_hs + lane * 2;
        const float rq0 = bf2f(qp[0]), rq1 = bf2f(qp[1]);
#pragma unroll
        for (int k = 0; k < 16; ++k) value[k] = 0.0f;
        float mx = -INFINITY, den = 0.0f;
        bool bad = false, vis = false;
        for (long j = j0; j <= j1; ++j) {
            const long page = dldi(bt, bit, s * bt_rs + (j / block_size) * bt_cs);
            if (page < 0 || page >= pages) continue;
            vis = true;
            const uint16_t* row = kv + page * sb + (j % block_size) * so;
            float key[16];
            load16(row + lane * 16, key);
            float sc = rq0 * bf2f(row[kLat + lane * 2]) + rq1 * bf2f(row[kLat + lane * 2 + 1]);
#pragma unroll
            for (int k = 0; k < 16; ++k) sc += query[k] * key[k];
            for (int off = 16; off; off >>= 1) sc += __shfl_xor(sc, off, 32);
            sc *= scale;
            if (sc != sc) { bad = true; continue; }
            if (sc == -INFINITY) continue;
            const float nm = fmaxf(mx, sc), a = mx > -INFINITY ? expf(mx - nm) : 0.0f, w = expf(sc - nm);
#pragma unroll
            for (int k = 0; k < 16; ++k) value[k] = value[k] * a + key[k] * w;
            den = den * a + w;
            mx = nm;
        }
        float f;
        const float big = fmaxf(mx, sinks ? sinks[(long)h * sks] : -INFINITY);   // the oracle's M
        if (!vis) f = 0.0f;
        else if (bad || big == INFINITY || big == -INFINITY) f = NAN;
        else if (!(mx > -INFINITY)) f = 0.0f;
        else if (sinks) {
            const float sk = sinks[(long)h * sks];
            f = sk > mx ? expf(mx - sk) / (den * expf(mx - sk) + 1.0f) : 1.0f / (den + expf(sk - mx));
        } else f = 1.0f / den;
        uint16_t* out = o + tok * o_rs + (long)h * o_hs + lane * 16;
#pragma unroll
        for (int k = 0; k < 16; ++k) out[k] = f2bf(f == 0.0f ? 0.0f : value[k] * f);
    }
#endif
    } HY4_CHAIN_END
}

/* hy4_mla_attn, form 2g: up to 32 rows (decode, DFly2 verify) and the drafter's block keys, with the
 * heads grouped as in libr4d's `attend_rows` (`libr4d/r4d_mla.hip`, v2.2.0) and the history split as in
 * its `attend_paged` + `merge_paged`. Form 2 gave each (row, head) a block, so every head read the
 * row's whole latent cache again; here a 4-wave block takes one row, 16 heads and one slice of the
 * row's visible keys, and each 16-key tile is loaded once for the 16 heads.
 *   - S^T = K Q^T on gfx12 BF16 WMMA: the keys are the matrix rows and the heads its columns, so a head
 *     is a lane and its running maximum and denominator stay in registers. Wave w scores the column
 *     quarter 144w .. 144w + 143 of the 576 (latent and rope; nine 16-wide steps), the four partial dots
 *     meet in LDS and every wave adds them in the same order, so all four hold the same scores.
 *   - P is split into high and low BF16 halves (as form 3), swapped between the half-waves into the
 *     second product's B layout, and wave w accumulates O^T for latent columns 128w .. 128w + 127, the
 *     keys read transposed from the tile. The next tile is fetched into registers while the current
 *     one is multiplied (double-buffered LDS tile): two LDS-only barriers a tile.
 *   - The slices: `splits` per (row, 16 heads), a function of the head count alone (mla_splits), so a
 *     row's bits do not depend on the batch: a verify row gives the decode row's bits. Each slice writes
 *     f32 partials (512 values, maximum, denominator, a visible key, a NaN seen) to the scratch, and
 *     mla_merge_k adds them in slice order with the sink, as form 2 merged its waves.
 * The oracle's special cases: a -inf score weighs 0; a NaN score, a NaN denominator or an infinite
 * max(sink, scores) gives NaN; a row with no visible key gives zeros; rows outside cu (bcu) are not
 * written (every slice marks them, and the merge skips them). Bits differ from form 2 (another
 * summation order); opt-in (HY4_MLA_ATTN=g) because its KL tail sat just past the envelope
 * (docs/OPS-HY4.md). gfx11 has no gfx12 WMMA and runs the
 * slices with form 2's per-key arithmetic, four heads a wave: correct, slow. */
constexpr int kGP = 584;           // LDS pitch of a key tile: 576 columns and 8 of padding
constexpr int kPart = kLat + 4;    // a slice's partial: 512 values, then max, denominator, visible, NaN
template <class Ch, bool kBk>
__global__ __launch_bounds__(128) void mla_attn2g_k(const uint16_t* q, long q_rs, long q_hs,
                                                    const uint16_t* qr, long qr_rs, long qr_hs,
                                                    const uint16_t* kv, long sb, long so, long pages,
                                                    const void* bt, int bit, long bt_rs, long bt_cs,
                                                    long max_blocks, const void* su, int sit, long sus,
                                                    const void* cus, int cit, long cs, int n_seq, int q_len,
                                                    int block_size, int window, float scale,
                                                    const uint16_t* bk, long bk_rs, const void* bcu, int bcit,
                                                    long bcs, int n_bseq, float* part, int H, int S) {
    HY4_CHAIN_BODY {
    const long tok = blockIdx.x;
    const int h0 = blockIdx.y * 16, split = blockIdx.z;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    /* this slice's partial of head h0 + c */
    auto slot = [&](int c) { return part + (((long)tok * H + h0 + c) * S + split) * kPart; };
    long s = 0, i = 0, qlen = 1;
    bool skip = false;
    if (cus) {
        while (s + 1 < n_seq && dldi(cus, cit, (s + 1) * cs) <= tok) ++s;
        const long beg = dldi(cus, cit, s * cs), end = dldi(cus, cit, (s + 1) * cs);
        if (tok < beg || tok >= end) skip = true;
        qlen = end - beg;
        i = tok - beg;
    } else {
        s = tok / q_len;
        i = tok - s * q_len;
        qlen = q_len;
    }
    long b0 = 0, nb = 0;   // the block keys: bk rows b0 .. b0 + nb - 1
    if (kBk && !skip) {
        long d = 0;
        while (d + 1 < n_bseq && dldi(bcu, bcit, (d + 1) * bcs) <= tok) ++d;
        b0 = dldi(bcu, bcit, d * bcs);
        if (b0 < 0 || tok < b0 || tok >= dldi(bcu, bcit, (d + 1) * bcs)) skip = true;
        nb = tok - b0 + 1;
    }
    if (skip) {   // outside cu (bcu): the merge reads the mark and writes nothing
        if (tid < 16) slot(tid)[kLat + 2] = -1.0f;
        return;
    }
    const long ctx = dldi(su, sit, s * sus);
    const long pos = ctx - qlen + i;
    long j0 = 0, j1 = pos < ctx - 1 ? pos : ctx - 1;
    if (window > 0 && pos - window + 1 > j0) j0 = pos - window + 1;
    if (j1 > max_blocks * block_size - 1) j1 = max_blocks * block_size - 1;
    const long nh = j1 >= j0 ? j1 - j0 + 1 : 0;
    /* the union in the order form 2 walks it: the nh cached keys, then the nb block keys; the slice
     * takes indices [begin, end) of it */
    const long n = nh + nb;
    const long begin = n * split / S, end = n * (split + 1) / S;
    /* key t of the union: its row, or null (past the slice, or a page outside the cache). Positions
     * fit in 32 bits (the host checks max_blocks * block_size). */
    auto key_row = [&](long t) -> const uint16_t* {
        if (t >= end) return nullptr;
        if (kBk && t >= nh) return bk + (b0 + (t - nh)) * bk_rs;
        const unsigned ju = (unsigned)(j0 + t), bs = (unsigned)block_size, blk = ju / bs;
        const long page = dldi(bt, bit, s * bt_rs + (long)blk * bt_cs);
        return page >= 0 && page < pages ? kv + page * sb + (long)(ju - blk * bs) * so : nullptr;
    };
#if defined(__GFX12__)
    __shared__ __align__(16) uint16_t keys[2][16 * kGP];
    __shared__ __align__(16) float dots[4][32 * 8];
    __shared__ int visible[2][16];
    const int m = lane & 15, hh = lane >> 4;   // the lane's head (column), and half
    /* Q as the B operand: B[k][n] = q[head h0 + n][column k], wave w's nine 16-column steps. Read from
     * memory at each tile (the row's 18 KB stay in the near caches): kept in registers they took 36
     * VGPRs and the kernel to 256 */
    const uint16_t* qa = q + tok * q_rs + (long)(h0 + m) * q_hs + 4 * hh;
    const uint16_t* qb = qr + tok * qr_rs + (long)(h0 + m) * qr_hs + 4 * hh - kLat;
    auto qfrag = [&](int d) -> v8bf3 {
        const uint16_t* p = d < kLat ? qa + d : qb + d;
        union { uint2 w[2]; v8bf3 f; } A;
        A.w[0] = *(const uint2*)p;
        A.w[1] = *(const uint2*)(p + 8);
        return A.f;
    };
    /* a tile is 16 keys x 72 pieces of 8 columns: nine 16-byte loads a thread */
    uint4 staged[9];
    unsigned staged_ok = 0;
    auto fetch = [&](long start) {
#pragma unroll
        for (int k = 0; k < 9; ++k) {
            const int f = tid + k * 128, key = f / 72, c8 = f % 72;
            const uint16_t* row = key_row(start + key);
            staged[k] = row ? *(const uint4*)(row + c8 * 8) : make_uint4(0, 0, 0, 0);
            staged_ok = row ? staged_ok | 1u << k : staged_ok & ~(1u << k);
        }
    };
    auto stash = [&](int b) {
#pragma unroll
        for (int k = 0; k < 9; ++k) {
            const int f = tid + k * 128, key = f / 72, c8 = f % 72;
            *(uint4*)(keys[b] + key * kGP + c8 * 8) = staged[k];
            if (!c8) visible[b][key] = (int)(staged_ok >> k & 1u);
        }
    };
    v8f3 value[8];
#pragma unroll
    for (int mt = 0; mt < 8; ++mt) value[mt] = (v8f3){0, 0, 0, 0, 0, 0, 0, 0};
    float run_max = -INFINITY, run_den = 0.0f;
    int bad = 0, seen = 0;
    if (begin < end) {
        fetch(begin);
        stash(0);
    }
    lds_barrier();
    int buf = 0;
    for (long start = begin; start < end; start += 16) {
        const bool more = start + 16 < end;
        if (more) fetch(start + 16);
        const uint16_t* tile = keys[buf];
        v8f3 dot = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int k = 0; k < 9; ++k)
            dot = mma3(frag3(tile, kGP, 0, wave * 144 + k * 16, lane), qfrag(wave * 144 + k * 16), dot);
        *(v8f3*)&dots[wave][lane * 8] = dot;
        lds_barrier();
        /* the four column quarters in a fixed order: every wave holds the same scores. Lane: head m,
         * keys 8hh .. 8hh + 7 of the tile. */
        v8f3 sc = *(const v8f3*)&dots[0][lane * 8];
        sc += *(const v8f3*)&dots[1][lane * 8];
        sc += *(const v8f3*)&dots[2][lane * 8];
        sc += *(const v8f3*)&dots[3][lane * 8];
        float sv[8], chunk_max = -INFINITY;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int vis = visible[buf][8 * hh + e];
            const float x = vis ? sc[e] * scale : -INFINITY;
            seen |= vis;
            bad |= vis & (x != x);
            sv[e] = x;
            chunk_max = fmaxf(chunk_max, x);   // a NaN is flagged, not taken
        }
        chunk_max = fmaxf(chunk_max, __shfl_xor(chunk_max, 16, 32));
        const float next_max = fmaxf(run_max, chunk_max);
        const float alpha = next_max > -INFINITY ? expf(run_max - next_max) : 1.0f;
        float p[8], sum = 0.0f;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            p[e] = sv[e] > -INFINITY ? expf(sv[e] - next_max) : 0.0f;   // -inf and NaN weigh 0
            sum += p[e];
        }
        /* both halves add keys 0-7 before 8-15, so the 16 lanes of a head agree */
        const float other = __shfl_xor(sum, 16, 32);
        run_den = run_den * alpha + (hh ? other + sum : sum + other);
        run_max = next_max;
        /* A B lane holds its head's keys 0-3 and 8-11 (lanes 0-15) or 4-7 and 12-15 (16-31); a score
         * lane holds 0-7 or 8-15. Each half sends the other the four it needs. */
        uint32_t hi[4], lo[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint16_t a0 = f2bf(p[2 * k]), a1 = f2bf(p[2 * k + 1]);
            hi[k] = (uint32_t)a0 | (uint32_t)a1 << 16;
            lo[k] = (uint32_t)f2bf(p[2 * k] - bf2f(a0)) | (uint32_t)f2bf(p[2 * k + 1] - bf2f(a1)) << 16;
        }
        const bool low = !hh;
        const uint32_t hx0 = (uint32_t)__shfl_xor((int)(low ? hi[2] : hi[0]), 16, 32),
                       hx1 = (uint32_t)__shfl_xor((int)(low ? hi[3] : hi[1]), 16, 32),
                       lx0 = (uint32_t)__shfl_xor((int)(low ? lo[2] : lo[0]), 16, 32),
                       lx1 = (uint32_t)__shfl_xor((int)(low ? lo[3] : lo[1]), 16, 32);
        union { uint32_t w[4]; v8bf3 f; } ph, pl;
        ph.w[0] = low ? hi[0] : hx0;
        ph.w[1] = low ? hi[1] : hx1;
        ph.w[2] = low ? hx0 : hi[2];
        ph.w[3] = low ? hx1 : hi[3];
        pl.w[0] = low ? lo[0] : lx0;
        pl.w[1] = low ? lo[1] : lx1;
        pl.w[2] = low ? lx0 : lo[2];
        pl.w[3] = low ? lx1 : lo[3];
        /* O^T += V^T P^T: A[col][key] = tile[key][col], eight LDS reads a fragment */
#pragma unroll
        for (int mt = 0; mt < 8; ++mt) {
            const int idx = wave * 128 + mt * 16 + m, k0 = 4 * hh;
            union { uint16_t h[8]; v8bf3 f; } u;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                u.h[e] = tile[(k0 + e) * kGP + idx];
                u.h[4 + e] = tile[(k0 + 8 + e) * kGP + idx];
            }
            value[mt] *= alpha;
            value[mt] = mma3(u.f, ph.f, value[mt]);
            value[mt] = mma3(u.f, pl.f, value[mt]);
        }
        if (more) stash(buf ^ 1);
        lds_barrier();
        buf ^= 1;
    }
    bad |= __shfl_xor(bad, 16, 32);
    seen |= __shfl_xor(seen, 16, 32);
    float* dst = slot(m);
#pragma unroll
    for (int mt = 0; mt < 8; ++mt) {
        float4* d4 = (float4*)(dst + wave * 128 + mt * 16 + 8 * hh);
        d4[0] = make_float4(value[mt][0], value[mt][1], value[mt][2], value[mt][3]);
        d4[1] = make_float4(value[mt][4], value[mt][5], value[mt][6], value[mt][7]);
    }
    if (!wave && lane < 16) {
        dst[kLat] = run_max;
        dst[kLat + 1] = run_den;
        dst[kLat + 2] = seen ? 1.0f : 0.0f;
        dst[kLat + 3] = bad ? 1.0f : 0.0f;
    }
#else
    /* gfx11: wave w takes heads h0 + 4w .. h0 + 4w + 3 over the slice, with form 2's per-key arithmetic */
    for (int c = wave * 4; c < wave * 4 + 4; ++c) {
        const int h = h0 + c;
        float query[16], value[16];
        load16(q + tok * q_rs + (long)h * q_hs + lane * 16, query);
        const uint16_t* qp = qr + tok * qr_rs + (long)h * qr_hs + lane * 2;
        const float rq0 = bf2f(qp[0]), rq1 = bf2f(qp[1]);
#pragma unroll
        for (int k = 0; k < 16; ++k) value[k] = 0.0f;
        float mx = -INFINITY, den = 0.0f;
        int bad = 0, seen = 0;
        for (long t = begin; t < end; ++t) {
            const uint16_t* row = key_row(t);
            if (!row) continue;
            seen = 1;
            float key[16];
            load16(row + lane * 16, key);
            float sc = rq0 * bf2f(row[kLat + lane * 2]) + rq1 * bf2f(row[kLat + lane * 2 + 1]);
#pragma unroll
            for (int k = 0; k < 16; ++k) sc += query[k] * key[k];
            for (int off = 16; off; off >>= 1) sc += __shfl_xor(sc, off, 32);
            sc *= scale;
            if (sc != sc) { bad = 1; continue; }
            if (sc == -INFINITY) continue;
            const bool rises = sc > mx;
            const float w = expf(rises ? mx - sc : sc - mx);
            if (rises) {
#pragma unroll
                for (int k = 0; k < 16; ++k) value[k] = value[k] * w + key[k];
                den = den * w + 1.0f;
                mx = sc;
            } else {
#pragma unroll
                for (int k = 0; k < 16; ++k) value[k] += key[k] * w;
                den += w;
            }
        }
        float* dst = slot(c);
#pragma unroll
        for (int k = 0; k < 16; ++k) dst[lane * 16 + k] = value[k];
        if (!lane) {
            dst[kLat] = mx;
            dst[kLat + 1] = den;
            dst[kLat + 2] = seen ? 1.0f : 0.0f;
            dst[kLat + 3] = bad ? 1.0f : 0.0f;
        }
    }
#endif
    } HY4_CHAIN_END
}

/* Form 2g's merge: a wave a (row, head), a lane 16 latent columns. The slices are added in their
 * order, as form 2 added its waves; the sink and the special cases are form 2's. */
template <class Ch>
__global__ __launch_bounds__(128) void mla_merge_k(const float* part, const float* sinks, long sks, uint16_t* o,
                                                   long o_rs, long o_hs, int H, int S) {
    HY4_CHAIN_BODY {
    const int lane = threadIdx.x & 31, h = blockIdx.y * 4 + (threadIdx.x >> 5);
    const long tok = blockIdx.x;
    if (h >= H) return;
    const float* src = part + ((long)tok * H + h) * S * kPart;
    if (src[kLat + 2] < 0.0f) return;   // a row outside cu (bcu): not written
    float m = -INFINITY;
    bool nan = false, any = false;
    for (int w = 0; w < S; ++w) {
        const float* hd = src + (long)w * kPart + kLat;
        any |= hd[2] > 0.0f;
        nan |= hd[3] != 0.0f || hd[1] != hd[1];
        m = fmaxf(m, hd[0]);
    }
    /* the oracle's M = max(sink, scores) over the visible keys: infinite with a visible key gives NaN */
    const float skv = sinks ? sinks[(long)h * sks] : -INFINITY;
    const float big = fmaxf(m, skv);
    if (any && (big == INFINITY || big == -INFINITY)) nan = true;
    uint4* out = (uint4*)(o + tok * o_rs + (long)h * o_hs + lane * 16);
    if (!any || (!nan && m == -INFINITY)) {   // no visible key, or only -inf scores beside a finite sink
        out[0] = out[1] = make_uint4(0, 0, 0, 0);
        return;
    }
    if (nan) {
        out[0] = out[1] = make_uint4(0x7fc07fc0u, 0x7fc07fc0u, 0x7fc07fc0u, 0x7fc07fc0u);
        return;
    }
    float acc[16], l = 0.0f;
#pragma unroll
    for (int k = 0; k < 16; ++k) acc[k] = 0.0f;
    for (int w = 0; w < S; ++w) {
        const float* sl = src + (long)w * kPart;
        if (!(sl[kLat] > -INFINITY)) continue;   // a slice with no weighed key
        const float e = expf(sl[kLat] - m);
        l += sl[kLat + 1] * e;
        const float4* v4 = (const float4*)(sl + lane * 16);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float4 v = v4[k];
            acc[4 * k] += v.x * e;
            acc[4 * k + 1] += v.y * e;
            acc[4 * k + 2] += v.z * e;
            acc[4 * k + 3] += v.w * e;
        }
    }
    if (sinks) {   // one more logit, with no value
        const float sk = sinks[(long)h * sks];
        if (sk > m) {
            const float r = expf(m - sk);
#pragma unroll
            for (int k = 0; k < 16; ++k) acc[k] *= r;
            l = l * r + 1.0f;
        } else {
            l += expf(sk - m);
        }
    }
    const float inv = !(l <= 0.0f) ? 1.0f / l : 0.0f;   // a NaN sink makes l NaN: let it through
    uint32_t w8[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) w8[k] = (uint32_t)f2bf(acc[2 * k] * inv) | (uint32_t)f2bf(acc[2 * k + 1] * inv) << 16;
    out[0] = make_uint4(w8[0], w8[1], w8[2], w8[3]);
    out[1] = make_uint4(w8[4], w8[5], w8[6], w8[7]);
    } HY4_CHAIN_END
}

/* Form 2g's slices per (row, 16 heads): a function of the head count alone (batch invariance).
 * 256 / heads gives 16 at Hy4's sparse layers (16 heads a rank) and 8 at the window layers (32):
 * 16 blocks a row and layer either way. `mla_splits` (1-64), which the graph never passes, overrides
 * it for tests/test_dev.py and tests/bench_attn.py. */
constexpr int64_t kMla2gRows = 64;   // rows the scratch serves (form 2g: <= 32 plain, <= 64 block keys)
int mla_splits(const RadArgs* a, int64_t H) {
    const int64_t p = pi(a, "mla_splits", 0);
    if (p >= 1 && p <= 64) return (int)p;
    return (int)std::min<int64_t>(32, std::max<int64_t>(1, 256 / std::max<int64_t>(H, 1)));
}
int64_t mla2g_bytes(int64_t rows, int64_t H, int S) { return rows * H * S * kPart * (int64_t)sizeof(float); }

/* hy4_mla_project on the card: radiance's mla_project as a batched GEMM per head, for the prefill's
 * large batches (libr4d's mla_project_bf16 loads each weight element once for up to 8 rows and has no
 * matrix-core path: 2.6 ms for a 2048-row absorb, 17% of an 18K prefill). One 4-wave block per
 * (32 rows, 64 columns, head); a wave owns 16 rows by 32 columns, two 16x16 accumulators, with A from
 * the activation rows and B from the weight: expand reads B rows K-contiguous, absorb reads them
 * strided by the weight pitch (the weight stores each head's key block [head_dim][latent]). Each
 * output sums K in one fixed order, so a row gives the same bits in any batch this row serves; libr4d's
 * kernel (decode) gives other bits, as forms 2 and 3 of hy4_mla_attn do. gfx11 has no gfx12 WMMA and
 * runs the same per-element sums without matrices. */
template <class Ch>
__global__ __launch_bounds__(128) void mla_proj_k(const uint16_t* x, long xr, long xh, const uint16_t* w, long wp,
                                                  uint16_t* y, long yr, long yh, int M, int d, int absorb) {
    HY4_CHAIN_BODY {
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int head = blockIdx.z, row0 = blockIdx.x * 32 + (wave & 1) * 16, col0 = blockIdx.y * 64 + (wave >> 1) * 32;
    const int K = absorb ? d : kLat;
    const long wbase = (long)head * 2 * d * wp;
#if defined(__GFX12__)
    const int ar = min(row0 + (lane & 15), M - 1);   // a row past M reads the last row and stores nothing
    const uint16_t* arow = x + (long)ar * xr + (long)head * xh + 4 * (lane >> 4);
    v8f3 acc[2] = {{0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}};
    for (int k0 = 0; k0 < K; k0 += 16) {
        union { uint2 u[2]; v8bf3 f; } A;
        A.u[0] = *(const uint2*)(arow + k0);
        A.u[1] = *(const uint2*)(arow + k0 + 8);
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            const int n = col0 + t * 16 + (lane & 15), kk = k0 + 4 * (lane >> 4);
            union { uint2 u[2]; uint16_t h[8]; v8bf3 f; } B;
            if (absorb) {   // B[n][k] = W[head*2d + k][n]
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    B.h[e] = w[wbase + (long)(kk + e) * wp + n];
                    B.h[4 + e] = w[wbase + (long)(kk + 8 + e) * wp + n];
                }
            } else {        // B[n][k] = W[head*2d + d + n][k]
                const uint16_t* br = w + wbase + (long)(d + n) * wp + kk;
                B.u[0] = *(const uint2*)br;
                B.u[1] = *(const uint2*)(br + 8);
            }
            acc[t] = mma3(A.f, B.f, acc[t]);
        }
    }
#pragma unroll
    for (int t = 0; t < 2; ++t)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int m = row0 + 8 * (lane >> 4) + e, n = col0 + t * 16 + (lane & 15);
            if (m < M) y[(long)m * yr + (long)head * yh + n] = f2bf(acc[t][e]);
        }
#else
    for (int t = 0; t < 2; ++t)
        for (int e = 0; e < 8; ++e) {
            const int m = row0 + 8 * (lane >> 4) + e, n = col0 + t * 16 + (lane & 15);
            if (m >= M) continue;
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                const float wv = bf2f(absorb ? w[wbase + (long)k * wp + n] : w[wbase + (long)(d + n) * wp + k]);
                sum += bf2f(x[(long)m * xr + (long)head * xh + k]) * wv;
            }
            y[(long)m * yr + (long)head * yh + n] = f2bf(sum);
        }
#endif
    } HY4_CHAIN_END
}

int64_t numel(const RadTensor& t) {
    int64_t n = 1;
    for (uint32_t i = 0; i < t.rank; ++i) n *= t.shape[i];
    return t.rank ? n : 0;
}

}  // namespace
}  // namespace hy4k

using namespace hy4k;

int hy4_dev_attn_sink(const RadArgs* a, RadStream stream) {
    if (a->n_t < 9) return RAD_E_INVAL;
    const RadTensor &q = a->t[0], &kv = a->t[1], &bt = a->t[2], &su = a->t[3], &sinks = a->t[7], &o = a->t[8];
    const RadTensor* kdes = rad_arg_in(a, 4);
    const RadTensor* vdes = rad_arg_in(a, 5);
    const RadTensor* cus = rad_arg_in(a, 6);
    if (kdes || vdes) return RAD_E_UNSUPPORTED;   // no descale: bf16 cache
    if (!q.data || !kv.data || !bt.data || !su.data || !o.data || !sinks.data) return RAD_E_INVAL;
    const int64_t head_dim = pi(a, "head_dim", q.rank >= 3 ? q.shape[q.rank - 1] : 0);
    const int64_t gqa = pi(a, "gqa", 1);
    const int64_t block_size = pi(a, "block_size", kv.rank == 4 ? kv.shape[2] : 0);
    const int64_t causal = pi(a, "causal", 1), window = pi(a, "window", 0);
    if (head_dim <= 0 || head_dim > 256 || gqa <= 0 || block_size <= 0 || causal != 1 || window < 0)
        return RAD_E_SHAPE;
    if (kv.rank != 4 || !is_float(kv) || !is_float(q) || !is_float(o) || sinks.dtype != RAD_F32) return RAD_E_SHAPE;
    const int64_t kv_heads = kv.shape[1], q_heads = gqa * kv_heads;
    if (kv_heads <= 0 || kv.shape[2] != block_size || kv.shape[3] < 2 * head_dim) return RAD_E_SHAPE;
    if (q.rank >= 3 && q.shape[q.rank - 2] != q_heads) return RAD_E_SHAPE;
    /* The rows come from q's extent: its last axis must be exactly the geometry's. */
    if (q.rank > 3 || o.rank > 3) return RAD_E_SHAPE;   // the kernel indexes [rows, heads, d] or [rows, heads·d]
    if (q.rank < 2 || (q.rank >= 3 ? q.shape[q.rank - 1] != head_dim : q.shape[1] != q_heads * head_dim))
        return RAD_E_SHAPE;
    if (o.rank < 2 || (o.rank >= 3 ? o.shape[o.rank - 1] != head_dim : o.shape[1] != q_heads * head_dim))
        return RAD_E_SHAPE;
    if (bt.rank > 2) return RAD_E_SHAPE;   // [S, width] or [S·width]
    if (numel(sinks) != q_heads) return RAD_E_SHAPE;
    const int64_t n_seq = numel(su), rows = numel(q) / (q_heads * head_dim);
    if (n_seq <= 0 || rows <= 0 || !is_int(su) || !is_int(bt)) return RAD_E_SHAPE;
    if (!cus && rows % n_seq != 0) return RAD_E_SHAPE;
    if (cus && (numel(*cus) != n_seq + 1 || !is_int(*cus))) return RAD_E_SHAPE;
    const int64_t q_len = cus ? 0 : rows / n_seq;
    if (numel(o) < rows * q_heads * head_dim) return RAD_E_SHAPE;
    const int64_t max_blocks = bt.rank >= 2 ? bt.shape[bt.rank - 1] : numel(bt) / n_seq;
    if (bt.rank >= 2 && bt.shape[0] < n_seq) return RAD_E_SHAPE;
    const int64_t bt_rs = bt.rank >= 2 ? bt.stride[bt.rank - 2] : max_blocks;
    const int64_t bt_cs = bt.rank >= 2 ? bt.stride[bt.rank - 1] : 1;
    const int64_t q_ds = q.rank >= 2 ? q.stride[q.rank - 1] : 1;
    const int64_t q_hs = q.rank >= 3 ? q.stride[q.rank - 2] : head_dim * q_ds;
    const int64_t q_rs = q.rank >= 2 ? q.stride[0] : q_heads * head_dim;
    const int64_t o_ds = o.rank >= 2 ? o.stride[o.rank - 1] : 1;
    const int64_t o_hs = o.rank >= 3 ? o.stride[o.rank - 2] : head_dim * o_ds;
    const int64_t o_rs = o.rank >= 2 ? o.stride[0] : q_heads * head_dim;
    const float scale = (float)pf(a, "scale", 1.0 / std::sqrt((double)head_dim));
    return emit(stream, HY4_CHAINED(a, attn_sink_k), dim3((unsigned)rows, (unsigned)q_heads), dim3(256), 0u, q.data, dcode(q), q_rs,
                 q_hs, q_ds, kv.data, dcode(kv), kv.stride[0], kv.stride[1], kv.stride[2], kv.stride[3], kv.shape[0], bt.data,
                 icode(bt), bt_rs, bt_cs, max_blocks, su.data, icode(su), su.stride[0], cus ? cus->data : nullptr,
                 cus ? icode(*cus) : 0, cus ? cus->stride[0] : 0, (int)n_seq, (int)q_len, (const float*)sinks.data,
                 sinks.stride[0], o.data, dcode(o), o_rs, o_hs, o_ds, (int)head_dim, (int)gqa, (int)block_size,
                 (int)window, scale);
}

/* HY4_MLA_ATTN, read once: 1 forces form 1, 2 the per-head form 2 at every row count (never 3 nor 2g),
 * 4 form 3 with its score split in four column quarters (20% faster, but on the full model its
 * per-position KL tails left the rounding envelope: p99 |dKL| 2.53 against 2.4, 861 top-1 changes
 * against 850), g form 2g below 33 rows and for block keys (decode 11-14% faster at 18K of context,
 * but in the KL scored in 32-row steps its p99 |dKL| was 2.43-2.62 against 2.4, where the per-head
 * form's own legal reorders gave 2.26-2.49: docs/OPS-HY4.md). h (and unset): form 3 above 32 rows,
 * the per-head form 2 below and for block keys. Both g and 4 are opt-in until a KL or BFCL run says
 * otherwise. Codes: 0 default, 1, 2, 4, 5 (h), 6 (g). */
namespace {
int mla_forced() {
    static const int forced = [] {
        const char* v = std::getenv("HY4_MLA_ATTN");
        if (!v || !*v) return 0;
        const int f = v[1] ? 0 : v[0] == 'h' ? 5 : v[0] == 'g' ? 6 : (v[0] == '1' || v[0] == '2' || v[0] == '4') ? v[0] - '0' : 0;
        /* a value that is none of these would silently mean "default": say so once */
        if (!f) std::fprintf(stderr, "hy4k: HY4_MLA_ATTN=%s is not 1, 2, 4, g or h: the default forms are used\n", v);
        return f;
    }();
    return forced;
}
/* form 2g on for this call: HY4_MLA_ATTN=g, or the test parameter mla_form 6 (5: off) */
bool mla_2g(const RadArgs* a) {
    const int64_t p = pi(a, "mla_form", 0);
    return p == 6 || (p != 5 && mla_forced() == 6);
}
}  // namespace

/* Scratch for form 2g's slice partials: rows (up to kMla2gRows) x heads x slices x 2064 bytes, at the
 * geometry form 2g takes (heads a multiple of 16, latent 512, rope 64): 16.9 MB at 32 rows, 33.8 MB at
 * 64, at 16 heads x 16 slices or 32 x 8. Nothing unless form 2g is on (HY4_MLA_ATTN=g). Without it the
 * launch runs the per-head form 2. */
int64_t hy4_dev_mla_attn_scratch(const RadArgs* a) {
    if (!mla_2g(a)) return 0;
    const int64_t H = pi(a, "heads", 0), L = pi(a, "latent", 0), R = pi(a, "rope", 0);
    if (H < 16 || H % 16 != 0 || L != kLat || R != kRopeMax) return 0;
    const int64_t rows = std::min<int64_t>(std::max<int64_t>(pi(a, "M", 1), 1), kMla2gRows);
    return mla2g_bytes(rows, H, mla_splits(a, H));
}

/* hy4_mla_attn (docs/OPS-HY4.md); the oracle is hy4_ref_mla_attn (ref_mla.cpp). */
int hy4_dev_mla_attn(const RadArgs* a, RadStream stream) {
    if (a->n_t < 8) return RAD_E_INVAL;
    const RadTensor &q = a->t[0], &qr = a->t[1], &kv = a->t[2], &bt = a->t[3], &su = a->t[4], &sk = a->t[6],
                    &o = a->t[7];
    const RadTensor* cus = rad_arg_in(a, 5);
    const RadTensor *bk = rad_arg_in(a, 8), *bcu = rad_arg_in(a, 9);   // optional, after the output
    if (!q.data || !qr.data || !kv.data || !bt.data || !su.data || !sk.data || !o.data) return RAD_E_INVAL;
    const int64_t H = pi(a, "heads"), L = pi(a, "latent"), R = pi(a, "rope");
    const int64_t bs = pi(a, "block_size"), window = pi(a, "window", 0);
    if (H < 1 || L < 1 || L > kLat || R < 1 || R > kRopeMax || bs < 1 || window < 0) return RAD_E_SHAPE;
    if (!is_float(q) || !is_float(qr) || !is_float(kv) || !is_float(o) || sk.dtype != RAD_F32) return RAD_E_SHAPE;
    const int64_t rp = pi(a, "rope_pitch", R);
    if (!is_heads_view(q, H, L, L) || !is_heads_view(qr, H, R, rp) || !is_heads_view(o, H, L, L) || rp < R)
        return RAD_E_SHAPE;
    if (kv.rank != 4 || kv.shape[1] != 1 || kv.shape[2] != bs || kv.shape[3] < L + R) return RAD_E_SHAPE;
    const int64_t rows = q.shape[0];
    const int64_t q_ds = q.stride[q.rank - 1], q_hs = q.rank == 3 ? q.stride[1] : L * q_ds;
    const int64_t r_ds = qr.stride[qr.rank - 1], r_hs = qr.rank == 3 ? qr.stride[1] : rp * r_ds;
    const int64_t o_ds = o.stride[o.rank - 1], o_hs = o.rank == 3 ? o.stride[1] : L * o_ds;
    if (qr.shape[0] != rows || o.shape[0] < rows || !is_int(su) || !is_int(bt) || bt.rank > 2) return RAD_E_SHAPE;
    if (numel(sk) != H) return RAD_E_SHAPE;
    const int64_t n_seq = numel(su);
    if (n_seq <= 0 || rows <= 0) return RAD_E_SHAPE;
    if (!cus && rows % n_seq != 0) return RAD_E_SHAPE;
    if (cus && (numel(*cus) != n_seq + 1 || !is_int(*cus))) return RAD_E_SHAPE;
    const int64_t q_len = cus ? 0 : rows / n_seq;
    const int64_t max_blocks = bt.rank >= 2 ? bt.shape[bt.rank - 1] : numel(bt) / n_seq;
    if (bt.rank >= 2 && bt.shape[0] < n_seq) return RAD_E_SHAPE;
    const int64_t bt_rs = bt.rank >= 2 ? bt.stride[0] : max_blocks * bt.stride[0];
    const int64_t bt_cs = bt.rank >= 2 ? bt.stride[1] : bt.stride[0];
    const float scale = (float)pf(a, "scale", 1.0 / std::sqrt((double)(L + R)));
    /* Block keys: the oracle's rules (selection form, both or neither, no window), and only form 2
     * serves them, whatever the row count: the drafter's rows are few and need no batch invariance. */
    if ((bk || bcu) && (!bk || !bcu || cus || window > 0)) return RAD_E_INVAL;
    if (bk && (n_seq != rows || bk->rank != 2 || bk->shape[0] < rows || bk->shape[1] < L + R || !is_int(*bcu) ||
               bcu->rank != 1 || bcu->shape[0] < 2))
        return RAD_E_SHAPE;
    /* Forms 2 and 3 take Hy4's geometry only: BF16, latent 512, rope 64, contiguous columns and rows
     * aligned for their 8-value loads. Anything else stays on form 1. */
    const int forced = mla_forced();
    const bool form1 = forced == 1;
    /* Form 2g (grouped heads, mla_attn2g_k + mla_merge_k) serves form 2's calls when it is on
     * (HY4_MLA_ATTN=g, or the test parameter mla_form 6) and its scratch holds the launch */
    const bool per_head = forced == 2 || !mla_2g(a);
    const int S = mla_splits(a, H);
    const auto al = [](const void* p) { return ((uintptr_t)p & 15u) == 0; };
    if (!form1 && q.dtype == RAD_BF16 && qr.dtype == RAD_BF16 && kv.dtype == RAD_BF16 && o.dtype == RAD_BF16 &&
        L == kLat && R == kRopeMax && q_ds == 1 && r_ds == 1 && o_ds == 1 && kv.stride[3] == 1 &&
        q.stride[0] % 8 == 0 && q_hs % 8 == 0 && o.stride[0] % 8 == 0 && o_hs % 8 == 0 && kv.stride[0] % 8 == 0 &&
        kv.stride[2] % 8 == 0 && al(q.data) && al(o.data) && al(kv.data) && (qr.stride[0] | r_hs) % 2 == 0) {
        /* forms 3 and 2g read q_rope in 4-value runs */
        const bool wide = H % 16 == 0 && qr.stride[0] % 4 == 0 && r_hs % 4 == 0 &&
                          max_blocks * bs < (int64_t(1) << 31) && ((uintptr_t)qr.data & 7u) == 0;
        const bool grouped = !per_head && wide && rows <= kMla2gRows && a->scratch &&
                             a->scratch_bytes >= mla2g_bytes(rows, H, S);
        auto form2g = [&](bool with_bk, int64_t n_bseq) {
            float* part = (float*)a->scratch;
            const int rc = emit(stream, with_bk ? HY4_CHAINED_T(a, mla_attn2g_k, true) : HY4_CHAINED_T(a, mla_attn2g_k, false),
                                dim3((unsigned)rows, (unsigned)(H / 16), (unsigned)S), dim3(128), 0u,
                                (const uint16_t*)q.data, q.stride[0], q_hs, (const uint16_t*)qr.data, qr.stride[0], r_hs,
                                (const uint16_t*)kv.data, kv.stride[0], kv.stride[2], kv.shape[0], bt.data, icode(bt),
                                bt_rs, bt_cs, max_blocks, su.data, icode(su), su.stride[0],
                                cus && !with_bk ? cus->data : nullptr, cus && !with_bk ? icode(*cus) : 0,
                                cus && !with_bk ? cus->stride[0] : 0, (int)n_seq, (int)q_len, (int)bs, (int)window, scale,
                                with_bk ? (const uint16_t*)bk->data : nullptr, with_bk ? bk->stride[0] : 0,
                                with_bk ? bcu->data : nullptr, with_bk ? icode(*bcu) : 0, with_bk ? bcu->stride[0] : 0,
                                (int)n_bseq, part, (int)H, S);
            if (rc != RAD_OK) return rc;
            return emit(stream, HY4_CHAINED(a, mla_merge_k), dim3((unsigned)rows, (unsigned)((H + 3) / 4)), dim3(128), 0u,
                        (const float*)part, (const float*)sk.data, sk.stride[sk.rank - 1], (uint16_t*)o.data,
                        o.stride[0], o_hs, (int)H, S);
        };
        if (bk) {
            /* bk rows are read like cache rows: bf16, contiguous columns, 16-byte aligned */
            if (bk->dtype != RAD_BF16 || bk->stride[1] != 1 || bk->stride[0] % 8 != 0 || !al(bk->data))
                return RAD_E_INVAL;
            const int64_t n_bseq = bcu->shape[0] - 1;
            if (pi(a, "max_seqs", 0) > 0 && n_bseq > pi(a, "max_seqs", 0)) return RAD_E_SHAPE;
            if (grouped) return form2g(true, n_bseq);
            return emit(stream, HY4_CHAINED_T(a, mla_attn2_k, true), dim3((unsigned)rows, (unsigned)H), dim3(kMlaW * 32), 0u,
                         (const uint16_t*)q.data, q.stride[0], q_hs, (const uint16_t*)qr.data, qr.stride[0], r_hs,
                         (const uint16_t*)kv.data, kv.stride[0], kv.stride[2], kv.shape[0], bt.data, icode(bt), bt_rs,
                         bt_cs, max_blocks, su.data, icode(su), su.stride[0], nullptr, 0, 0L, (int)n_seq, (int)q_len,
                         (const float*)sk.data, sk.stride[sk.rank - 1], (uint16_t*)o.data, o.stride[0], o_hs, (int)bs,
                         (int)window, scale, (const uint16_t*)bk->data, bk->stride[0], bcu->data, icode(*bcu),
                         bcu->stride[0], (int)n_bseq);
        }
        if (forced != 2 && rows > 32 && wide)
            return emit(stream, forced == 4 ? HY4_CHAINED_T(a, mla_attn3_k, true) : HY4_CHAINED_T(a, mla_attn3_k, false), dim3((unsigned)rows, (unsigned)(H / 16)),
                         dim3(128), 0u, (const uint16_t*)q.data, q.stride[0], q_hs, (const uint16_t*)qr.data,
                         qr.stride[0], r_hs, (const uint16_t*)kv.data, kv.stride[0], kv.stride[2], kv.shape[0], bt.data,
                         icode(bt), bt_rs, bt_cs, max_blocks, su.data, icode(su), su.stride[0], cus ? cus->data : nullptr,
                         cus ? icode(*cus) : 0, cus ? cus->stride[0] : 0, (int)n_seq, (int)q_len,
                         (const float*)sk.data, sk.stride[sk.rank - 1], (uint16_t*)o.data, o.stride[0], o_hs,
                         (int)bs, (int)window, scale);
        if (grouped && rows <= 32) return form2g(false, 0);
        return emit(stream, HY4_CHAINED_T(a, mla_attn2_k, false), dim3((unsigned)rows, (unsigned)H), dim3(kMlaW * 32), 0u,
                     (const uint16_t*)q.data, q.stride[0], q_hs, (const uint16_t*)qr.data, qr.stride[0], r_hs,
                     (const uint16_t*)kv.data, kv.stride[0], kv.stride[2], kv.shape[0], bt.data, icode(bt), bt_rs,
                     bt_cs, max_blocks, su.data, icode(su), su.stride[0], cus ? cus->data : nullptr,
                     cus ? icode(*cus) : 0, cus ? cus->stride[0] : 0, (int)n_seq, (int)q_len,
                     (const float*)sk.data, sk.stride[sk.rank - 1], (uint16_t*)o.data, o.stride[0], o_hs, (int)bs,
                     (int)window, scale, nullptr, 0L, nullptr, 0, 0L, 0);
    }
    /* forms 1 and 3 have no block keys: with them (HY4_MLA_ATTN=1, or a geometry form 2 does not
     * take) the call is refused */
    if (bk) return RAD_E_INVAL;
    return emit(stream, HY4_CHAINED(a, mla_attn_k), dim3((unsigned)rows, (unsigned)H), dim3(256), 0u, q.data, dcode(q), q.stride[0],
                 q_hs, q_ds, qr.data, dcode(qr), qr.stride[0], r_hs, r_ds, kv.data,
                 dcode(kv), kv.stride[0], kv.stride[2], kv.stride[3], kv.shape[0], bt.data, icode(bt), bt_rs, bt_cs,
                 max_blocks, su.data, icode(su), su.stride[0], cus ? cus->data : nullptr, cus ? icode(*cus) : 0,
                 cus ? cus->stride[0] : 0, (int)n_seq, (int)q_len, (const float*)sk.data,
                 sk.stride[sk.rank - 1], o.data, dcode(o), o.stride[0], o_hs, o_ds, (int)L, (int)R,
                 (int)bs, (int)window, scale);
}

/* hy4_mla_project, the wide form (2026-11-23): mla_proj_k's sums in the Direct-B style of gostar's
 * radiance PR #586 (libr4d/r4d_moe_nvfp4.hip, "Direct-B f16 WMMA prefill": every wave reads its B
 * fragments straight from global memory with 16-byte loads, no LDS stage). A block of WR x WC waves
 * takes (32 WR rows, 32 WC columns) of one head; a wave owns 32 rows by 32 columns, four 16x16
 * accumulators, so each A fragment serves two B fragments and each B fragment two A fragments (mla_proj_k:
 * one A for two B). The next K step's fragments are loaded before this step's WMMAs (one step ahead in
 * registers); a load past K reads the last step again (always a valid address, no branch around it).
 *
 * The absorb B is the weight's key block read across rows: B[n][k] = W[head*2d + k][n], which mla_proj_k
 * gathered with eight 2-byte loads a lane. Here it is one global_load_tr_b128 (gfx12; libr4d's
 * load_tr_b128, r4d_common.h): in each group of eight lanes, lane i names row k of the fragment it owes
 * (k = 4*(lane>>4) + (i&3) + 8*(i>>2)) at the group's 8-column base, and lane j receives column base+j of
 * those eight rows, in lane order: exactly the fragment mla_proj_k built, so no copy of the weight is
 * needed (a pre-transposed key block would cost ~230 MB a card). The expand B was already K-contiguous.
 *
 * Every WMMA gets the fragments mla_proj_k gave it, for the same (row, column, K step), and K is walked
 * in the same order from a zero accumulator: the output is mla_proj_k's, bit for bit (tests/bench_proj.py
 * compares the two). gfx11 runs the per-element sums of mla_proj_k. */
template <class Ch, int WR, int WC>
__global__ __launch_bounds__(WR * WC * 32) void mla_proj2_k(const uint16_t* x, long xr, long xh, const uint16_t* w,
                                                           long wp, uint16_t* y, long yr, long yh, int M, int d,
                                                           int mode) {
    HY4_CHAIN_BODY {
    /* mode bit 0: absorb; bit 1: the grid is (heads, column tiles, row tiles), so the blocks in flight
     * share row tiles across heads instead of walking one head's rows */
    const int absorb = mode & 1, hm = mode & 2;
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, hl = lane >> 4;
    const int head = hm ? blockIdx.x : blockIdx.z, rb = hm ? blockIdx.z : blockIdx.x;
    const int row0 = rb * (32 * WR) + (wave % WR) * 32;
    const int col0 = blockIdx.y * (32 * WC) + (wave / WR) * 32;
    const int K = absorb ? d : kLat;
    const long wbase = (long)head * 2 * d * wp;
#if defined(__GFX12__)
    const uint16_t* arow[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int ar = min(row0 + 16 * i + (lane & 15), M - 1);   // a row past M reads the last row, stores nothing
        arow[i] = x + (long)ar * xr + (long)head * xh + 4 * hl;
    }
    /* B sources: absorb, the row this lane names for load_tr and its group's column base; expand, the
     * weight row of the lane's column */
    const uint16_t* bsrc[2];
#pragma unroll
    for (int t = 0; t < 2; ++t) {
        if (absorb) {
            const int kr = 4 * hl + (lane & 3) + 8 * ((lane >> 2) & 1);
            bsrc[t] = w + wbase + (long)kr * wp + col0 + 16 * t + 8 * ((lane >> 3) & 1);
        } else {
            bsrc[t] = w + wbase + (long)(d + col0 + 16 * t + (lane & 15)) * wp + 4 * hl;
        }
    }
    union Fr { uint2 u[2]; uint4 q; v8bf3 f; };
    auto load = [&](int k0, Fr (&a)[2], Fr (&b)[2]) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            a[i].u[0] = *(const uint2*)(arow[i] + k0);
            a[i].u[1] = *(const uint2*)(arow[i] + k0 + 8);
        }
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            if (absorb) {
                b[t].q = __builtin_bit_cast(uint4, __builtin_amdgcn_global_load_tr_b128_v8i16(
                    (__attribute__((address_space(1))) v8s3*)(void*)const_cast<uint16_t*>(bsrc[t] + (long)k0 * wp)));
            } else {
                b[t].u[0] = *(const uint2*)(bsrc[t] + k0);
                b[t].u[1] = *(const uint2*)(bsrc[t] + k0 + 8);
            }
        }
    };
    v8f3 acc[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int t = 0; t < 2; ++t) acc[i][t] = v8f3{0, 0, 0, 0, 0, 0, 0, 0};
    Fr a[2], b[2], an[2], bn[2];
    load(0, a, b);
    for (int k0 = 0; k0 < K; k0 += 16) {
        load(min(k0 + 16, K - 16), an, bn);
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int t = 0; t < 2; ++t) acc[i][t] = mma3(a[i].f, b[t].f, acc[i][t]);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            a[i] = an[i];
            b[i] = bn[i];
        }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int m = row0 + 16 * i + 8 * hl + e, n = col0 + 16 * t + (lane & 15);
                if (m < M) y[(long)m * yr + (long)head * yh + n] = f2bf(acc[i][t][e]);
            }
#else
    for (int i = 0; i < 2; ++i)
        for (int t = 0; t < 2; ++t)
            for (int e = 0; e < 8; ++e) {
                const int m = row0 + 16 * i + 8 * hl + e, n = col0 + 16 * t + (lane & 15);
                if (m >= M) continue;
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    const float wv = bf2f(absorb ? w[wbase + (long)k * wp + n] : w[wbase + (long)(d + n) * wp + k]);
                    sum += bf2f(x[(long)m * xr + (long)head * xh + k]) * wv;
                }
                y[(long)m * yr + (long)head * yh + n] = f2bf(sum);
            }
#endif
    } HY4_CHAIN_END
}

/* Which projection kernel serves a call: the wide form unless HY4_MLA_PROJ=old (read once) or the
 * test parameter proj_form = 1 asks for mla_proj_k; proj_form = 2 forces the wide form, and proj_wr /
 * proj_wc (tests/bench_proj.py) pick its block shape. */
static int proj_form(const RadArgs* a) {
    static const int env = [] {
        const char* v = std::getenv("HY4_MLA_PROJ");
        return v && !std::strcmp(v, "old") ? 1 : 2;
    }();
    const int64_t p = pi(a, "proj_form", 0);
    return p == 1 || p == 2 ? (int)p : env;
}

/* hy4_mla_project (docs/OPS-HY4.md); the oracle is hy4_ref_mla_project (ref_mla.cpp). */
int hy4_dev_mla_project(const RadArgs* a, RadStream stream) {
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
    if ((!absorb && !expand) || h < 1 || d != 256 || l != kLat || x.rank != 3 || w.rank != 2 || y.rank != 3 ||
        x.shape[1] != h || x.shape[2] != (absorb ? d : l) || w.shape[0] != h * 2 * d || w.shape[1] != l ||
        y.shape[0] != x.shape[0] || y.shape[1] != h || y.shape[2] != (absorb ? l : d))
        return RAD_E_SHAPE;
    if (x.dtype != RAD_BF16 || w.dtype != RAD_BF16 || y.dtype != RAD_BF16) return RAD_E_SHAPE;
    if (!x.data || !w.data || !y.data) return RAD_E_INVAL;
    /* the fragment loads read 4-value runs: unit column strides and 8-byte aligned rows */
    if (x.stride[2] != 1 || y.stride[2] != 1 || w.stride[1] != 1 || x.stride[0] % 4 || x.stride[1] % 4 ||
        w.stride[0] % 4 || ((uintptr_t)x.data & 7u) || ((uintptr_t)w.data & 7u))
        return RAD_E_SHAPE;
    const int64_t M = x.shape[0], N = absorb ? l : d;
    /* the wide form's absorb B is a 16-byte transposed load: 16-byte aligned weight rows */
    const bool wide_ok = !absorb || ((w.stride[0] % 8) == 0 && ((uintptr_t)w.data & 15u) == 0);
    if (proj_form(a) == 2 && wide_ok) {
        /* 4 x 4 waves (128 rows by 128 columns) on the head-major grid: tests/bench_proj.py on the R9700,
         * 32 heads, 2048 rows: absorb 2.13 -> 0.41 ms, expand 2.25 -> 1.55 ms; 4096 rows: 12.5 -> 0.80
         * and 7.40 -> 2.80 ms (mla_proj_k -> this). The row-major grid, mla_proj_k's, gave 4096 rows 4-6
         * times 2048's time in either kernel; walking the heads first takes that away. */
        const int64_t wr = pi(a, "proj_wr", 4), wc = pi(a, "proj_wc", 4), hm = pi(a, "proj_hm", 1) ? 1 : 0;
        const unsigned rbs = (unsigned)((M + 32 * wr - 1) / (32 * wr));
        const dim3 grid = hm ? dim3((unsigned)h, (unsigned)(N / (32 * wc)), rbs) : dim3(rbs, (unsigned)(N / (32 * wc)), (unsigned)h);
#define HY4_PROJ2(WR, WC)                                                                                       \
    if (wr == WR && wc == WC)                                                                                   \
        return emit(stream, HY4_CHAINED_T(a, mla_proj2_k, WR, WC), grid, dim3(WR * WC * 32), 0u,                \
                    (const uint16_t*)x.data, x.stride[0], x.stride[1], (const uint16_t*)w.data, w.stride[0],    \
                    (uint16_t*)y.data, y.stride[0], y.stride[1], (int)M, (int)d, (int)(absorb | (hm << 1)));
        HY4_PROJ2(2, 4)
        HY4_PROJ2(4, 2)
        HY4_PROJ2(2, 2)
        HY4_PROJ2(4, 4)
        HY4_PROJ2(8, 2)
#undef HY4_PROJ2
        return RAD_E_SHAPE;
    }
    return emit(stream, HY4_CHAINED(a, mla_proj_k), dim3((unsigned)((M + 31) / 32), (unsigned)(N / 64), (unsigned)h), dim3(128), 0u,
                 (const uint16_t*)x.data, x.stride[0], x.stride[1], (const uint16_t*)w.data, w.stride[0],
                 (uint16_t*)y.data, y.stride[0], y.stride[1], (int)M, (int)d, (int)absorb);
}
