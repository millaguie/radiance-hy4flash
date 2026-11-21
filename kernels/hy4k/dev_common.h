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

/* Sum over a block of `kThreads` threads (a multiple of 32); every thread receives the total. */
template <int kThreads>
__device__ inline float block_sum(float v, float* lds) {
    for (int d = 16; d; d >>= 1) v += __shfl_xor(v, d, 32);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) lds[w] = v;
    __syncthreads();
    float t = 0;
    for (int i = 0; i < kThreads / 32; ++i) t += lds[i];
    return t;
}

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
