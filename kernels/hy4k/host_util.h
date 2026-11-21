/* host_util.h -- per-element access and operand validation for hy4k's host kernels.
 * Uses only the public header rad_plugin.h (libref's helpers are not installed).
 *
 * Rules (docs/PLUGIN.md §3): the number of rows and of sequences comes from the operands' extent
 * (`cu` has S+1 entries), never from a declared parameter; every index that comes in an operand
 * (page, slot, position, count) is checked against the extent of the operand it points into
 * before the first write; zero rows is not a valid call. */
#ifndef HY4K_HOST_UTIL_H
#define HY4K_HOST_UTIL_H

#include "rad_abi.h"
#include "rad_plugin.h"

#include <cmath>
#include <cstdint>

namespace hy4k {

inline int64_t row(const RadTensor& t, int64_t r) { return r * t.stride[0]; }
inline int64_t col(const RadTensor& t, int64_t i) { return i * t.stride[t.rank - 1]; }
inline float ld(const RadTensor& t, int64_t off) { return rad_load_f32(t.data, t.dtype, off); }
inline void st(const RadTensor& t, int64_t off, float v) { rad_store_f32(t.data, t.dtype, off, v); }
inline int64_t pi(const RadArgs* a, const char* k, int64_t d = 0) { return rad_args_geti_or(a, k, d); }
inline double pf(const RadArgs* a, const char* k, double d) { return rad_args_getf_or(a, k, d); }
inline float sigmoid(float z) { return 1.0f / (1.0f + std::exp(-z)); }

/* A launch cannot allocate memory (rad_abi.h, RadLaunchFn): each kernel uses fixed-size arrays on
 * the stack and refuses a geometry that does not fit in them. The limits cover Hy4 with margin. */
constexpr int64_t kMaxHcWidth = 16384;   // hc·n of a row of streams (Hy4: 4·3072)
constexpr int64_t kMaxN       = 8192;    // width of one stream (Hy4: 3072)
constexpr int64_t kMaxD       = 512;     // width of one rows_rmsnorm slice (Hy4: 96)
constexpr int64_t kMaxNIdx    = 256;     // width of one indexer key (Hy4: 128)
constexpr int64_t kMaxRatio   = 16;      // tokens per group (Hy4: 4)
constexpr int64_t kMaxHIdx    = 32;      // indexer heads (Hy4: 16)
constexpr int64_t kMaxTopk    = 1024;    // width of the selection (Hy4: 512)

inline bool is_present(const RadTensor& t) { return t.data != nullptr && t.rank >= 1; }
inline bool is_float(const RadTensor& t) {
    return t.dtype == RAD_F32 || t.dtype == RAD_BF16 || t.dtype == RAD_F16;
}
inline bool is_int(const RadTensor& t) { return t.dtype == RAD_I32 || t.dtype == RAD_I64; }

/* Operand of rank `rank` with last dimension `last` (0: any). */
inline bool is_rank(const RadTensor& t, uint32_t rank, int64_t last = 0) {
    return is_present(t) && t.rank == rank && (last == 0 || t.shape[rank - 1] == last);
}
/* An operand [rows, a] or [rows, a, b] whose total width per row is `width` and whose columns are
 * contiguous within the row (the row views the engine hands out are). */
inline bool row_width(const RadTensor& t, int64_t width) {
    if (!is_present(t) || t.rank < 2) return false;
    int64_t w = 1;
    for (uint32_t i = 1; i < t.rank; ++i) w *= t.shape[i];
    if (w != width) return false;
    int64_t want_stride = 1;
    for (int i = (int)t.rank - 1; i >= 1; --i) {
        if (t.shape[i] > 1 && t.stride[i] != want_stride) return false;
        want_stride *= t.shape[i];
    }
    return true;
}
inline int64_t ldi(const RadTensor& t, int64_t off) {
    if (t.dtype == RAD_I64) return ((const int64_t*)t.data)[off];
    return ((const int32_t*)t.data)[off];
}
inline void sti(const RadTensor& t, int64_t off, int64_t v) {
    if (t.dtype == RAD_I64) ((int64_t*)t.data)[off] = v;
    else ((int32_t*)t.data)[off] = (int32_t)v;
}
/* Entry `i` of a rank-1 operand, or column 0 of row `i` of a rank-2 one (the committed slot of a
 * state_index of [S, pitch]), like libref's col0(). */
inline int64_t col0(const RadTensor& t, int64_t i) { return t.rank >= 2 ? i * t.stride[0] : i * t.stride[0]; }
/* Element i of a rank-1 vector (with its stride). */
inline int64_t vi(const RadTensor& t, int64_t i) { return i * t.stride[0]; }

/* A logical weight [R, C] that may arrive as [R, C] or flattened as [R·C] (a concatenation of
 * 1-D tensors in the container gives a 1-D tensor). */
inline bool is_matrix(const RadTensor& t, int64_t R, int64_t C) {
    if (!is_present(t)) return false;
    if (t.rank == 1) return t.shape[0] == R * C;
    return t.rank == 2 && t.shape[0] == R && t.shape[1] == C;
}
inline int64_t at2(const RadTensor& t, int64_t r, int64_t c, int64_t C) {
    return t.rank == 1 ? (r * C + c) * t.stride[0] : r * t.stride[0] + c * t.stride[1];
}

/* Number of sequences according to `cu` (S+1 entries), checked against the ceiling `cap` (the `seqs` param) and
 * with 0 <= cu[s] <= cu[s+1] <= rows. -1 if it does not fit. */
inline int64_t count_seqs(const RadTensor& cu, int64_t cap, int64_t rows) {
    if (!is_rank(cu, 1) || !is_int(cu) || cu.shape[0] < 2) return -1;
    const int64_t S = cu.shape[0] - 1;
    if (cap > 0 && S > cap) return -1;
    int64_t prev = 0;
    for (int64_t s = 0; s <= S; ++s) {
        const int64_t c = ldi(cu, vi(cu, s));
        if (c < prev || c > rows) return -1;
        prev = c;
    }
    if (ldi(cu, vi(cu, 0)) < 0) return -1;
    return S;
}

/* The DFly2 ring (hy4_ring_store, hy4_ring_load): i32 state [slots, …, ring, W] or, as a
 * RAD_KV_LINEAR group hands it out, [slots, …, ring·W]; every other middle axis is 1. Strides in
 * words: of a slot, of an entry and of a word within the entry. */
struct RingView {
    int64_t slots, slot_stride, entry_stride, word_stride, W;
};
inline bool ring_view(const RadTensor& t, int64_t ring, RingView* v) {
    if (!is_present(t) || t.dtype != RAD_I32 || t.rank < 2 || ring < 1) return false;
    const uint32_t last = t.rank - 1;
    /* the entry axis is the one before the last when its extent is the ring's (any ring, 1
     * included: both readings agree there); otherwise the last axis holds the entries back to back */
    const bool split = t.rank >= 3 && t.shape[last - 1] == ring;
    for (uint32_t i = 1; i < (split ? last - 1 : last); ++i)
        if (t.shape[i] != 1) return false;
    v->slots = t.shape[0];
    v->slot_stride = t.stride[0];
    v->word_stride = t.stride[last];
    if (split) {
        v->W = t.shape[last];
        v->entry_stride = t.stride[last - 1];
    } else {
        if (t.shape[last] % ring) return false;
        v->W = t.shape[last] / ring;
        v->entry_stride = v->W * t.stride[last];
    }
    return v->slots >= 1 && v->W >= 1;
}
/* The ring entry of position p: p mod ring, non-negative. */
inline int64_t ring_mod(int64_t p, int64_t ring) {
    const int64_t m = p % ring;
    return m < 0 ? m + ring : m;
}

/* The group key cache: one page per group, n contiguous values at the start of the page. */
/* A view of heads: exactly [M, heads, width], or [M, ≥ (heads-1)·pitch + width] with head h in
 * columns [h·pitch, h·pitch + width). */
inline bool is_heads_view(const RadTensor& t, int64_t heads, int64_t width, int64_t pitch) {
    if (t.rank == 3) return t.shape[1] == heads && t.shape[2] == width;
    return t.rank == 2 && t.shape[1] >= (heads - 1) * pitch + width;
}

inline bool cache_ok(const RadTensor& c, int64_t n) {
    if (!is_present(c) || c.rank < 2 || !is_float(c)) return false;
    /* The page (axes 1..rank-1) must be dense: the kernel reads it as n consecutive values. An
     * axis of size 1 is never stepped, so its stride does not count (as in row_width). */
    int64_t per_page = 1;
    for (uint32_t i = c.rank - 1; i >= 1; --i) {
        if (c.shape[i] > 1 && c.stride[i] != per_page) return false;
        per_page *= c.shape[i];
    }
    return per_page >= n && c.stride[0] >= per_page;
}

}  // namespace hy4k

#endif
