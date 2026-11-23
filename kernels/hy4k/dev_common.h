/* dev_common.h -- helpers for hy4k's card kernels (HIP rows, plan step 6).
 *
 * Launches go through hipLaunchKernel with the arguments already converted to the parameter's
 * type, as libr4d's r4d_emit does: radiance puts its own runtime under the HIP calls, and the
 * image does not carry ROCm's. Shapes are validated on the host with host_util.h before launch;
 * indices that come in the data (cu, pages, slots) are checked inside the kernel, which skips the
 * bad entry instead of writing out of bounds. */
#ifndef HY4K_DEV_COMMON_H
#define HY4K_DEV_COMMON_H

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <tuple>
#include <cstdlib>
#include <climits>

#include "hy4k.h"
#include "host_util.h"

namespace hy4k {

/* dtype of an activation: 0 f32, 1 bf16, 2 f16 (the ones `is_float` accepts). */
inline int dcode(const RadTensor& t) { return t.dtype == RAD_BF16 ? 1 : t.dtype == RAD_F16 ? 2 : 0; }
/* integer: 0 i32, 1 i64 */
inline int icode(const RadTensor& t) { return t.dtype == RAD_I64 ? 1 : 0; }

__device__ inline float bf2f(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }
__device__ inline uint16_t f2bf(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40);  // NaN
    u += 0x7fffu + ((u >> 16) & 1u);                                          // round to nearest even
    return (uint16_t)(u >> 16);
}
__device__ inline float dld(const void* p, int dt, int64_t i) {
    if (dt == 1) return bf2f(((const uint16_t*)p)[i]);
    if (dt == 2) return (float)((const _Float16*)p)[i];
    return ((const float*)p)[i];
}
__device__ inline void dst(void* p, int dt, int64_t i, float v) {
    if (dt == 1) ((uint16_t*)p)[i] = f2bf(v);
    else if (dt == 2) ((_Float16*)p)[i] = (_Float16)v;
    else ((float*)p)[i] = v;
}
__device__ inline int64_t dldi(const void* p, int it, int64_t i) {
    return it ? ((const int64_t*)p)[i] : (int64_t)((const int32_t*)p)[i];
}
__device__ inline void dsti(void* p, int it, int64_t i, int64_t v) {
    if (it) ((int64_t*)p)[i] = v;
    else ((int32_t*)p)[i] = (int32_t)v;
}
__device__ inline float dsigmoid(float z) { return 1.0f / (1.0f + expf(-z)); }

/* A barrier that orders LDS traffic only. From radiance's libr4d (`lds_barrier()`, r4d_common.h,
 * v2.2.0): __syncthreads() is a workgroup acquire-release fence over every address space, so on gfx12
 * it emits `global_inv scope:SCOPE_SE` on both sides and throws away the L0 lines the next loads would
 * hit. Scoping the fence to the local address space drops the invalidate. Use it only where no thread
 * reads global memory another thread of the block wrote before the barrier; the arithmetic is the same,
 * so the bits are too. */
__device__ __forceinline__ void lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

/* Sum over a block of `kThreads` threads (a multiple of 32); every thread receives the total. The
 * barriers order the LDS slots only. */
template <int kThreads>
__device__ inline float block_sum(float v, float* lds) {
    for (int d = 16; d; d >>= 1) v += __shfl_xor(v, d, 32);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    lds_barrier();
    if (l == 0) lds[w] = v;
    lds_barrier();
    float t = 0;
    for (int i = 0; i < kThreads / 32; ++i) t += lds[i];
    return t;
}

/* ------------------------------------------------------------------ chained launches
 * From radiance's libr4d/r4d_chain.h (v2.2.0, Apache-2.0) and the Inkling 2 plugin
 * (radiance-inkling2-plugin, kernels/ink2k/dev_common.h, round 7); the contract is abi/rad_abi.h,
 * RadKernelChainFn. A chained launch loses the queue's barrier bit: the engine writes a record at byte
 * RAD_CHAIN_KARG_OFF of the argument slot, and the kernel's chained form waits, before its first read
 * of an operand, until the queue's completion counter reaches the record's target, then adds one a
 * block after its last write. Every kernel is a template on a chain tag (its first template
 * parameter): `r4dchain` instantiates the chained form and puts the name the backend recognises into
 * the symbol, `Unchained` the plain form, which never reads the record.
 *
 * Every block must add exactly once, after all its waves have written. hy4k's bodies return early
 * per thread, per wave (mla_attn2_k's `if (wave) return;`, attn_sink_k's `t >= head_dim`) or per
 * block, so each kernel runs its body as a lambda between chain_wait and chain_done (the body's
 * returns leave the lambda and every lane reaches chain_done; the arithmetic, and so the bits, is the
 * plain form's). libr4d ends with a block barrier and one add; here a wave that left the body early
 * would meet that barrier while the others are still at one of the body's own barriers, and the
 * hardware would let both through (an exited wave is out of the count, a waiting one is not), so the
 * add could come before the last write. chain_done therefore counts the waves instead: each one
 * releases its own writes at agent scope and adds one to a counter in LDS, and the wave that brings it
 * to the block's wave count adds the block's one to the queue counter. chain_wait zeroes the counters
 * and counts the block's waves between its two barriers, where every wave is still in. The wait ends with s_dcache_inv behind the
 * agent-scope acquire (a scalar load may be served from the scalar data cache, which still holds what
 * was there before the launch in front wrote), so the rows answer 2 (exports.cpp). */
struct r4dchain  { static constexpr bool on = true; };
struct Unchained { static constexpr bool on = false; };

struct ChainRec {
    unsigned long long  magic, target;
    unsigned long long* ctr;
    unsigned long long  check;
};
/* The record, or null where it is not the engine's (a launch from tests/test_dev.py): run without
 * waiting and do not add. Read at both ends, so nothing is held across the body. */
__device__ __forceinline__ const ChainRec* chain_rec() {
    const ChainRec* r =
        (const ChainRec*)((const char*)__builtin_amdgcn_kernarg_segment_ptr() + RAD_CHAIN_KARG_OFF);
    const unsigned long long magic = r->magic, target = r->target, check = r->check;
    const unsigned long long* ctr = r->ctr;
    if (magic != RAD_CHAIN_MAGIC ||
        check != (magic ^ target ^ (unsigned long long)(uintptr_t)ctr ^ RAD_CHAIN_CHECK))
        return nullptr;
    return r;
}
/* The block's wave count is counted, not read. gridDim/blockDim are hidden kernel arguments (reading them
 * took idx_score_k's chained form from 544 to 800 bytes of arguments, over the backend's 768), and the
 * dispatch packet (__builtin_amdgcn_dispatch_ptr) is no source either: with chained launches the packet
 * processor runs far ahead of the kernels, the host reuses a packet's ring slot once it is read, and a
 * block that starts or ends late reads another launch's packet. So chain_wait has every wave add one to
 * an LDS counter between two barriers, while all of them are in. */
__device__ __forceinline__ unsigned& chain_waves() {
    __shared__ unsigned n;
    return n;
}
__device__ __forceinline__ unsigned& chain_waves_done() {
    __shared__ unsigned n;
    return n;
}
/* libr4d's polling periods: fast for small grids (a decode step is mostly those), slower for wide
 * ones so a long wait does not crowd one L2 channel. */
template <class Ch>
__device__ __forceinline__ void chain_wait() {
    if constexpr (Ch::on) {
        const ChainRec* r = chain_rec();
        if (!r) return;
        const bool first = threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0;
        if (first) {
            chain_waves_done() = 0;
            chain_waves() = 0;
        }
        lds_barrier();
        if (__lane_id() == 0)
            __hip_atomic_fetch_add(&chain_waves(), 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
        if (first) {
            /* libr4d's periods: fast polls first, then fast for the first 16 blocks, its middle period up
             * to 64 and its slow one past that, so a wide grid waiting behind a long kernel does not crowd
             * the counter's L2 channel. libr4d picks the period by the grid's size, a hidden argument;
             * the sum of the block's indices is an SGPR sum, and a grid of n blocks has at most 16 and 64
             * blocks below those bounds in every dimension (a decode grid of a few blocks keeps the fast
             * period, as there). */
            const unsigned long long target = r->target;
            unsigned long long* ctr = r->ctr;
            const unsigned at = blockIdx.x + blockIdx.y + blockIdx.z;
            int n = 0;
            while (__hip_atomic_load(ctr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < target) {
                if (n < 8) {
                    __builtin_amdgcn_s_sleep(1);
                    ++n;
                } else if (at < 16) {
                    __builtin_amdgcn_s_sleep(1);
                } else if (at < 64) {
                    __builtin_amdgcn_s_sleep(4);
                } else {
                    __builtin_amdgcn_s_sleep(16);
                }
            }
        }
        __syncthreads();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        asm volatile("s_dcache_inv" ::: "memory");
    }
}
template <class Ch>
__device__ __forceinline__ void chain_done() {
    if constexpr (Ch::on) {
        const ChainRec* r = chain_rec();
        if (!r) return;
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
        if (__lane_id() == 0) {
            const unsigned waves = chain_waves();
            const unsigned before = __hip_atomic_fetch_add(&chain_waves_done(), 1u, __ATOMIC_ACQ_REL,
                                                           __HIP_MEMORY_SCOPE_WORKGROUP);
            if (before + 1 == waves)
                __hip_atomic_fetch_add(r->ctr, 1ull, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        }
    }
}
/* A kernel's chained bracket: HY4_CHAIN_BODY { ...body... } HY4_CHAIN_END. */
#define HY4_CHAIN_BODY \
    chain_wait<Ch>();  \
    [&]()
#define HY4_CHAIN_END \
    ();               \
    chain_done<Ch>();

/* A launcher's choice: the chained form when the engine resolved the row with chain = 1
 * (core/plugin/select.cpp sets the parameter where chaining is on). HY4_CHAINED(a, K) for a kernel
 * whose only template parameter is the tag, HY4_CHAINED_T(a, K, ...) for one with more (the tag
 * first). */
/* Chained forms only for launches of up to HY4K_CHAIN_ROWS rows (default 64: decode
 * and a DFly2 verify of up to 8 sequences of 6 rows). Measured on 2x R9700 (docs/DESIGN.md, "Concurrent
 * and chained hy4k launches"): chained everywhere, decode gained 3-7 % and prefill lost 7-9 % at 5.7K,
 * with a first 18K prompt at a third of the rate; a launch over the bound takes the plain form, which
 * the engine orders with the barrier bit as before. */
inline int64_t chain_rows() {
    static const int64_t n = [] {
        const char* v = std::getenv("HY4K_CHAIN_ROWS");
        return v && *v ? (int64_t)std::atoll(v) : (int64_t)64;
    }();
    return n;
}
/* The launch's rows: the first operand's leading extent (every hy4k op's first present operand is
 * row-major in the step's rows, or in fewer: the indexer's group stage). Not the parameter `M`, which
 * the engine resolves to the band's bound (2048 in a decode pass too). */
inline int64_t launch_rows(const RadArgs* a) {
    for (int i = 0; i < a->n_t; ++i)
        if (a->t[i].data && a->t[i].rank >= 1) return a->t[i].shape[0];
    return INT64_MAX;
}
inline bool chain_param(const RadArgs* a) {
    return pi(a, "chain", 0) != 0 && launch_rows(a) <= chain_rows();
}
#define HY4_CHAINED(a, K) (chain_param(a) ? K<r4dchain> : K<Unchained>)
#define HY4_CHAINED_T(a, K, ...) (chain_param(a) ? K<r4dchain, __VA_ARGS__> : K<Unchained, __VA_ARGS__>)

template <class... P, class... A>
inline int emit(RadStream stream, void (*kernel)(P...), dim3 grid, dim3 block, uint32_t lds,
                 const A&... args) {
    static_assert(sizeof...(P) == sizeof...(A), "kernel arguments");
    std::tuple<P...> vals(static_cast<P>(args)...);
    void* params[sizeof...(P) + 1];
    std::apply([&](P&... v) {
        int i = 0;
        ((params[i++] = (void*)&v), ...);
    }, vals);
    params[sizeof...(P)] = nullptr;
    return hipLaunchKernel((const void*)kernel, grid, block, params, lds, (hipStream_t)stream) == hipSuccess
               ? RAD_OK : RAD_E_DEVICE;
}

}  // namespace hy4k

#endif
