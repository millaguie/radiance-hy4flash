/* dev_ring.hip -- card rows of hy4_ring_store and hy4_ring_load (docs/OPS-HY4.md), the DFly2
 * drafter's ring. Copies only: the same bits as ref_ring.cpp. The host launcher checks shapes; the
 * indices the data carries (row ids, cu, slots) are checked in the kernel: a store skips the bad
 * entry instead of writing out of bounds, a load writes the fill (the host row refuses the call). */
#include "dev_common.h"
#include <cstring>

namespace hy4k {
namespace {

constexpr int kThreads = 256;
enum { kPos, kRow, kOut, kDense };
/* source kinds: 0 i32, 1 i64, 2 f32, 3 bf16, 4 f16 */
inline int scode(const RadTensor& t) {
    switch (t.dtype) {
    case RAD_I32: return 0;
    case RAD_I64: return 1;
    case RAD_BF16: return 3;
    case RAD_F16: return 4;
    default: return 2;
    }
}

/* One word of a source as the i32 the entry keeps (ref_ring.cpp's src_word). */
__device__ inline int32_t src_word_d(const void* p, int sk, long i) {
    if (sk == 0) return ((const int32_t*)p)[i];
    if (sk == 1) return (int32_t)((const int64_t*)p)[i];
    const float f = dld(p, sk == 3 ? 1 : sk == 4 ? 2 : 0, i);
    return (int32_t)__float_as_uint(f);
}

__device__ inline long ring_mod_d(long p, long ring) {
    const long m = p % ring;
    return m < 0 ? m + ring : m;
}

/* The sequence whose valid span of cu holds row r, or -1. A span that goes backwards or past the
 * rows is no sequence (the host refuses such a cu). */
__device__ inline long seq_of_d(const void* cu, int cit, long cs, int S, long rows, long r) {
    long prev = 0;
    for (int s = 0; s < S; ++s) {
        const long a = dldi(cu, cit, (long)s * cs), b = dldi(cu, cit, (long)(s + 1) * cs);
        if (a < prev || b < a || b > rows) return -1;   // cu must not decrease up to the row's span
        if (a <= r && r < b) return s;
        prev = a;
    }
    return -1;
}

/* Where listed row k writes: false when the card skips it (the rows the host skips or refuses),
 * else its row, sequence, slot and position. */
__device__ inline bool store_at_d(long k, int mode, long src_rows, const void* rid, int rit, long ris,
                                  const void* pos, int pit, long ps, long rows, const void* cu, int cit, long cs,
                                  int S, const void* sidx, int xit, long sx, long slots, long* r_, long* s_,
                                  long* slot_, long* p_) {
    const long r = dldi(rid, rit, k * ris);
    if (r < 0 || r >= rows) return false;
    const long s = seq_of_d(cu, cit, cs, S, rows, r);
    if (s < 0) return false;
    const long slot = dldi(sidx, xit, s * sx);
    if (slot < 0 || slot >= slots) return false;
    if ((mode == kRow && r >= src_rows) || (mode == kDense && s >= src_rows)) return false;
    *r_ = r;
    *s_ = s;
    *slot_ = slot;
    *p_ = dldi(pos, pit, r * ps);
    return true;
}

/* One block per listed row k; the threads stride over the entry's words. Two listed rows that land
 * on the same (slot, entry) would race here, while the host writes them in list order: so a block
 * first looks for a later row k' > k that is written to the same place and, if there is one,
 * leaves the entry to it. The last listed wins on both sides. The search is O(M) a block, strided
 * over the threads, and only a k' whose position falls on the same entry pays the sequence lookup;
 * M is the step's logit rows, a few hundred at most. */
__global__ __launch_bounds__(kThreads) void ring_store_k(int mode, const void* src, int sk, long src_rs, long src_cs,
                                                         long src_rows, long src_cols, const void* rid, int rit,
                                                         long ris, long M, const void* pos, int pit, long ps,
                                                         long rows, const void* cu, int cit, long cs, int S,
                                                         const void* sidx, int xit, long sx, int32_t* st,
                                                         long slot_stride, long entry_stride, long word_stride,
                                                         long slots, long ring, int width, int offset, long block) {
    const long k = blockIdx.x;
    long r, s, slot, p;
    /* uniform over the block: every thread returns together, before the barrier */
    if (!store_at_d(k, mode, src_rows, rid, rit, ris, pos, pit, ps, rows, cu, cit, cs, S, sidx, xit, sx, slots, &r,
                    &s, &slot, &p))
        return;
    const long e = ring_mod_d(p, ring);
    int later = 0;
    for (long k2 = k + 1 + threadIdx.x; k2 < M && !later; k2 += kThreads) {
        const long r2 = dldi(rid, rit, k2 * ris);
        if (r2 < 0 || r2 >= rows || ring_mod_d(dldi(pos, pit, r2 * ps), ring) != e) continue;
        long rr, ss, slot2, pp;
        later = store_at_d(k2, mode, src_rows, rid, rit, ris, pos, pit, ps, rows, cu, cit, cs, S, sidx, xit, sx,
                           slots, &rr, &ss, &slot2, &pp) &&
                slot2 == slot;
    }
    if (__syncthreads_or(later)) return;
    int32_t* base = st + slot * slot_stride + e * entry_stride;
    const long last = p >= 0 ? p / block : -1;
    for (int c = threadIdx.x; c < width; c += kThreads) {
        int32_t w;
        if (mode == kPos) w = (int32_t)p;
        else if (mode == kRow) w = src_word_d(src, sk, r * src_rs + (long)c * src_cs);
        else if (mode == kOut) w = src_word_d(src, sk, k * src_rs + (long)c * src_cs);
        else if (c == 0) w = (int32_t)(p + 1);
        else w = (c - 1 <= last && c - 1 < src_cols) ? src_word_d(src, sk, s * src_rs + (long)(c - 1) * src_cs) : -1;
        base[(long)(offset + c) * word_stride] = w;
    }
}

/* One block per destination row: with cu, row r of its sequence; without, sequence r. Every
 * destination row is written: the entry, or `fill` when there is none to read (slot < 0, p0 < 1, a
 * stale entry), and also where the host refuses or writes the fill instead (a row outside cu's
 * spans, a cu that is no partition, a slot outside the state), so dst never keeps old bytes. */
__global__ __launch_bounds__(kThreads) void ring_load_k(const void* pos, int pit, long ps, long rows, const void* cu,
                                                        int cit, long cs, int S, const void* sidx, int xit, long sx,
                                                        const int32_t* st, long slot_stride, long entry_stride,
                                                        long word_stride, long slots, long ring, int width, int offset,
                                                        int32_t fill, int32_t* dst, long d_rs, long d_cs) {
    const long r = blockIdx.x;
    const int32_t* base = nullptr;
    long s = r, r0 = r;
    if (cu) {
        s = seq_of_d(cu, cit, cs, S, rows, r);
        if (s >= 0) r0 = dldi(cu, cit, s * cs);
    }
    if (s >= 0) {
        const long slot = dldi(sidx, xit, s * sx);
        const long p0 = dldi(pos, pit, r0 * ps);
        if (slot >= 0 && slot < slots && p0 >= 1) {
            const int32_t* b = st + slot * slot_stride + ring_mod_d(p0 - 1, ring) * entry_stride;
            if ((long)b[0] == p0 - 1) base = b;
        }
    }
    for (int c = threadIdx.x; c < width; c += kThreads)
        dst[r * d_rs + (long)c * d_cs] = base ? base[(long)(offset + c) * word_stride] : fill;
}

}  // namespace
}  // namespace hy4k

using namespace hy4k;

/* The same shape checks as hy4_ref_ring_store; the data's indices are the kernel's. */
int hy4_dev_ring_store(const RadArgs* a, RadStream stream) {
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
        !is_int(sidx) || !ring_view(state, ring, &v) || offset + width > v.W || !is_rank(cu, 1) || !is_int(cu) ||
        cu.shape[0] < 2)
        return RAD_E_SHAPE;
    if (mode == kPos ? src != nullptr
                     : !src || !(src->dtype == RAD_I32 || src->dtype == RAD_I64 || is_float(*src)))
        return RAD_E_SHAPE;
    if ((mode == kRow || mode == kOut) &&
        !(src->rank == 2 ? src->shape[1] >= width : src->rank == 1 && width == 1))
        return RAD_E_SHAPE;
    if (mode == kDense && (src->rank != 2 || !is_int(*src))) return RAD_E_SHAPE;
    const int64_t M = rid.shape[0], rows = pos.shape[0], S = cu.shape[0] - 1;
    if (M < 1 || rows < 1 || (cap > 0 && S > cap) || sidx.shape[0] < S) return RAD_E_SHAPE;
    if (mode == kOut && src->shape[0] < M) return RAD_E_SHAPE;
    const long src_rs = src ? src->stride[0] : 0, src_cs = src && src->rank == 2 ? src->stride[1] : 0;
    return emit(stream, ring_store_k, dim3((unsigned)M), dim3(kThreads), 0u, mode, src ? src->data : nullptr,
                src ? scode(*src) : 0, src_rs, src_cs, src ? src->shape[0] : 0,
                src && src->rank == 2 ? src->shape[1] : 1, rid.data, icode(rid), rid.stride[0], (long)M, pos.data,
                icode(pos), pos.stride[0], rows, cu.data, icode(cu), cu.stride[0], (int)S, sidx.data, icode(sidx),
                sidx.stride[0], (int32_t*)state.data, v.slot_stride, v.entry_stride, v.word_stride, v.slots, ring,
                (int)width, (int)offset, block > 0 ? block : 1);
}

int hy4_dev_ring_load(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 5) return RAD_E_INVAL;
    const int64_t ring = pi(a, "ring"), width = pi(a, "width"), offset = pi(a, "offset", 0);
    const int64_t cap = pi(a, "seqs", 0), fill = pi(a, "fill", -1);
    if (ring < 1 || width < 1 || offset < 0 || fill < INT32_MIN || fill > INT32_MAX) return RAD_E_SHAPE;
    const RadTensor* cu = rad_arg_in(a, 1);
    const RadTensor &pos = a->t[0], &sidx = a->t[2], &state = a->t[3], &dst = a->t[4];
    RingView v;
    if (!is_rank(pos, 1) || !is_int(pos) || !is_present(sidx) || !is_int(sidx) || !ring_view(state, ring, &v) ||
        offset + width > v.W || !is_present(dst) || (dst.dtype != RAD_I32 && dst.dtype != RAD_F32) ||
        !(dst.rank == 2 ? dst.shape[1] >= width : dst.rank == 1 && width == 1) ||
        (cu && (cu->rank != 1 || !is_int(*cu) || cu->shape[0] < 2)))
        return RAD_E_SHAPE;
    const int64_t rows = pos.shape[0];
    const int64_t S = cu ? cu->shape[0] - 1 : rows;
    const int64_t n_dst = cu ? rows : S;
    if (rows < 1 || (cap > 0 && S > cap) || sidx.shape[0] < S || dst.shape[0] < n_dst) return RAD_E_SHAPE;
    return emit(stream, ring_load_k, dim3((unsigned)n_dst), dim3(kThreads), 0u, pos.data, icode(pos), pos.stride[0],
                rows, cu ? cu->data : nullptr, cu ? icode(*cu) : 0, cu ? cu->stride[0] : 0, (int)S, sidx.data,
                icode(sidx), sidx.stride[0], (const int32_t*)state.data, v.slot_stride, v.entry_stride,
                v.word_stride, v.slots, ring, (int)width, (int)offset, (int32_t)fill, (int32_t*)dst.data, dst.stride[0],
                dst.rank == 2 ? dst.stride[1] : 0);
}
