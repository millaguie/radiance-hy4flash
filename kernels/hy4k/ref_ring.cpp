/* ref_ring.cpp -- hy4_ring_store and hy4_ring_load: the DFly2 drafter's per-sequence ring of
 * depth + 1 entries (docs/DFLY2.md, docs/OPS-HY4.md). Copies only, so the card row gives the same
 * bits. State handling as hy4_ngram_ids': state [slots, ..., W] with size-1 middle axes, the slot
 * from sidx per sequence, the sequences from cu, every pointer and extent checked before the first
 * write. */
#include "hy4k.h"
#include "host_util.h"

#include <cstring>

using namespace hy4k;

namespace {

/* One word of a source row as the i32 the entry keeps: an integer truncated to 32 bits, a float
 * as the bits of its f32 value (bf16 and f16 widen exactly, so host and card agree). */
int32_t src_word(const RadTensor& t, int64_t off) {
    if (t.dtype == RAD_I32 || t.dtype == RAD_I64) return (int32_t)ldi(t, off);
    const float f = ld(t, off);
    int32_t w;
    std::memcpy(&w, &f, 4);
    return w;
}

/* A destination word: dst is i32 or f32, both 4 bytes, and gets the bits as they are. */
void dst_word(const RadTensor& t, int64_t off, int32_t w) { ((int32_t*)t.data)[off] = w; }

bool src_ok(const RadTensor& t) {
    return t.dtype == RAD_I32 || t.dtype == RAD_I64 || is_float(t);
}

/* Offset of word c of row i of a [rows, ≥ c] operand, or of entry i of a rank-1 one (c is 0). */
int64_t at_rc(const RadTensor& t, int64_t i, int64_t c) {
    return t.rank == 1 ? i * t.stride[0] : i * t.stride[0] + c * t.stride[1];
}

/* A source or destination of `width` words a row: [rows, ≥ width], or [rows] when width is 1. */
bool rows_of(const RadTensor& t, int64_t width) {
    if (!is_present(t)) return false;
    if (t.rank == 1) return width == 1;
    return t.rank == 2 && t.shape[1] >= width;
}

/* The sequence whose span of cu holds row r, or -1 (a row outside every span is not touched).
 * cu was checked by count_seqs. */
int64_t seq_of(const RadTensor& cu, int64_t S, int64_t r) {
    for (int64_t s = 0; s < S; ++s)
        if (ldi(cu, vi(cu, s)) <= r && r < ldi(cu, vi(cu, s + 1))) return s;
    return -1;
}

enum { kPos, kRow, kOut, kDense };

}  // namespace

/* hy4_ring_store -- see docs/OPS-HY4.md. Operands: src?, row_ids, pos, cu, sidx, state:inout.
 * The rows are written in list order, so of two listed rows on the same (slot, entry) the later
 * one stays; the card row does the same on purpose (dev_ring.hip). */
int hy4_ref_ring_store(const RadArgs* a, RadStream) {
    if (!a || a->n_t < 6) return RAD_E_INVAL;
    const int64_t ring = pi(a, "ring"), width = pi(a, "width"), offset = pi(a, "offset", 0);
    const int64_t cap = pi(a, "seqs", 0), block = pi(a, "block", 0);
    const char* ms = rad_args_gets(a, "mode");
    int mode = -1;
    if (ms && !std::strcmp(ms, "pos")) mode = kPos;
    else if (ms && !std::strcmp(ms, "row")) mode = kRow;
    else if (ms && !std::strcmp(ms, "out")) mode = kOut;
    else if (ms && !std::strcmp(ms, "dense")) mode = kDense;
    if (mode < 0 || ring < 1 || width < 1 || offset < 0 || (mode == kPos && width != 1) ||
        (mode == kDense && (block < 1 || width < 2)))
        return RAD_E_SHAPE;
    const RadTensor* src = rad_arg_in(a, 0);
    const RadTensor &rid = a->t[1], &pos = a->t[2], &cu = a->t[3], &sidx = a->t[4], &state = a->t[5];
    RingView v;
    if (!is_rank(rid, 1) || !is_int(rid) || !is_rank(pos, 1) || !is_int(pos) || !is_present(sidx) ||
        !is_int(sidx) || !ring_view(state, ring, &v) || offset + width > v.W)
        return RAD_E_SHAPE;
    /* the source by mode: none for `pos`; per token, per logit row, or a block table [seqs, cols] */
    if (mode == kPos ? src != nullptr : !src || !src_ok(*src)) return RAD_E_SHAPE;
    if ((mode == kRow || mode == kOut) && !rows_of(*src, width)) return RAD_E_SHAPE;
    if (mode == kDense && (src->rank != 2 || !is_int(*src))) return RAD_E_SHAPE;
    const int64_t M = rid.shape[0], rows = pos.shape[0];
    if (M < 1 || rows < 1) return RAD_E_SHAPE;
    const int64_t S = count_seqs(cu, cap, rows);
    if (S < 1 || sidx.shape[0] < S) return RAD_E_SHAPE;
    if (mode == kOut && src->shape[0] < M) return RAD_E_SHAPE;
    /* every index the data carries, before the first write */
    for (int64_t k = 0; k < M; ++k) {
        const int64_t r = ldi(rid, vi(rid, k));
        if (r < 0 || r >= rows) return RAD_E_INVAL;
        const int64_t s = seq_of(cu, S, r);
        if (s < 0) continue;
        const int64_t slot = ldi(sidx, col0(sidx, s));
        if (slot >= v.slots) return RAD_E_INVAL;
        if (slot < 0) continue;
        if (mode == kRow && r >= src->shape[0]) return RAD_E_INVAL;
        if (mode == kDense && s >= src->shape[0]) return RAD_E_INVAL;
    }
    for (int64_t k = 0; k < M; ++k) {
        const int64_t r = ldi(rid, vi(rid, k));
        const int64_t s = seq_of(cu, S, r);
        if (s < 0) continue;
        const int64_t slot = ldi(sidx, col0(sidx, s));
        if (slot < 0) continue;
        const int64_t p = ldi(pos, vi(pos, r));
        const int64_t base = slot * v.slot_stride + ring_mod(p, ring) * v.entry_stride;
        /* dense: the last page that holds a position <= p (none before position 0) */
        const int64_t last = p >= 0 ? p / (block > 0 ? block : 1) : -1;
        for (int64_t c = 0; c < width; ++c) {
            int32_t w;
            switch (mode) {
            case kPos: w = (int32_t)p; break;
            case kRow: w = src_word(*src, at_rc(*src, r, c)); break;
            case kOut: w = src_word(*src, at_rc(*src, k, c)); break;
            default:   /* dense: seqused = p + 1, then the sequence's first pages, -1 past the last */
                if (c == 0) w = (int32_t)(p + 1);
                else w = (c - 1 <= last && c - 1 < src->shape[1]) ? src_word(*src, at_rc(*src, s, c - 1)) : -1;
            }
            ((int32_t*)state.data)[base + (offset + c) * v.word_stride] = w;
        }
    }
    return RAD_OK;
}

/* hy4_ring_load -- see docs/OPS-HY4.md. Operands: pos, cu?, sidx, state -> dst. Param `fill`
 * (default -1): the word written where there is no entry to read; the plugin passes 0 for the f32
 * `h` plane, where -1 would read as a NaN. */
int hy4_ref_ring_load(const RadArgs* a, RadStream) {
    if (!a || a->n_t < 5) return RAD_E_INVAL;
    const int64_t ring = pi(a, "ring"), width = pi(a, "width"), offset = pi(a, "offset", 0);
    const int64_t cap = pi(a, "seqs", 0), fill = pi(a, "fill", -1);
    if (ring < 1 || width < 1 || offset < 0 || fill < INT32_MIN || fill > INT32_MAX) return RAD_E_SHAPE;
    const RadTensor* cu = rad_arg_in(a, 1);
    const RadTensor &pos = a->t[0], &sidx = a->t[2], &state = a->t[3], &dst = a->t[4];
    RingView v;
    if (!is_rank(pos, 1) || !is_int(pos) || !is_present(sidx) || !is_int(sidx) || !ring_view(state, ring, &v) ||
        offset + width > v.W || !rows_of(dst, width) || (dst.dtype != RAD_I32 && dst.dtype != RAD_F32))
        return RAD_E_SHAPE;
    const int64_t rows = pos.shape[0];
    if (rows < 1) return RAD_E_SHAPE;
    /* with cu, pos is per row and every row of a sequence gets its entry; without, per sequence */
    const int64_t S = cu ? count_seqs(*cu, cap, rows) : rows;
    if (S < 1 || (!cu && cap > 0 && S > cap) || sidx.shape[0] < S || dst.shape[0] < (cu ? rows : S))
        return RAD_E_SHAPE;
    for (int64_t s = 0; s < S; ++s)
        if (ldi(sidx, col0(sidx, s)) >= v.slots) return RAD_E_INVAL;
    const int32_t fw = (int32_t)fill;
    /* with cu, a row outside every span gets the fill, as on the card, where a block per row writes
     * every row: a destination row is never left with what it held */
    if (cu)
        for (int64_t r = 0; r < rows; ++r)
            if (seq_of(*cu, S, r) < 0)
                for (int64_t c = 0; c < width; ++c) dst_word(dst, at_rc(dst, r, c), fw);
    for (int64_t s = 0; s < S; ++s) {
        const int64_t r0 = cu ? ldi(*cu, vi(*cu, s)) : s, r1 = cu ? ldi(*cu, vi(*cu, s + 1)) : s + 1;
        if (r1 <= r0) continue;   // an empty sequence: no block row, nothing to write
        const int64_t slot = ldi(sidx, col0(sidx, s));
        const int64_t p0 = ldi(pos, vi(pos, r0));
        /* the entry of the last committed row, p0 - 1, if it is the one stored there */
        int64_t base = -1;
        if (slot >= 0 && p0 >= 1) {
            const int64_t b = slot * v.slot_stride + ring_mod(p0 - 1, ring) * v.entry_stride;
            if ((int64_t)((const int32_t*)state.data)[b] == p0 - 1) base = b;
        }
        for (int64_t r = r0; r < r1; ++r)
            for (int64_t c = 0; c < width; ++c)
                dst_word(dst, at_rc(dst, r, c),
                         base >= 0 ? ((const int32_t*)state.data)[base + (offset + c) * v.word_stride] : fw);
    }
    return RAD_OK;
}
