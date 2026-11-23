/* dev_quant.hip -- card row of had_quant_act_fp8 (ref_quant.cpp, libhalo's semantics): one wave
 * per (row, group of 128), FWHT in registers, amax, scale amax/448 and E4M3. */
#include "dev_common.h"

namespace hy4k {
namespace {

/* E4M3 (bias 7, 3 mantissa bits, no infinities), round to nearest even, |v| ≤ 448 already clipped. */
__device__ inline uint8_t e4m3_byte(float v) {
    const uint32_t sgn = (__float_as_uint(v) >> 24) & 0x80u;
    const float a = fabsf(v);
    if (a < 0.015625f) {                                   // subnormal: step 2^-9
        const int m = (int)rintf(a * 512.0f);              // 0..8; 8 is the first normal (2^-6)
        return (uint8_t)(sgn | (m == 8 ? 0x08u : (uint32_t)m));
    }
    int ex;
    frexpf(a, &ex);                                        // a = f·2^ex, f in [0.5, 1)
    int e = ex - 1;                                        // a in [2^e, 2^(e+1))
    int m = (int)rintf(ldexpf(a, -e) * 8.0f) - 8;          // mantissa 0..8
    if (m == 8) { m = 0; ++e; }
    const int ef = e + 7;
    if (ef > 15 || (ef == 15 && m > 6)) return (uint8_t)(sgn | 0x7eu);   // ±448
    return (uint8_t)(sgn | ((uint32_t)ef << 3) | (uint32_t)m);
}

/* One 32-lane wave per (row, group of 128); lane l carries columns 4l..4l+3. The FWHT runs in
 * registers (like libhalo's halo_fwht_wave): stages 1 and 2 within the lane, 4..64 across lanes
 * with __shfl_xor. Each butterfly puts a+b in the low element and a-b in the high one, the same
 * order as the oracle, so the result is bit-identical. Four waves per block. */
template <class Ch>
__global__ __launch_bounds__(128) void had_quant_k(const void* x, int xdt, long xp, long xs, uint8_t* q, long qp,
                                                   long qs, float* s, long sp, long ss, int ngrp) {
    HY4_CHAIN_BODY {
    const long r = blockIdx.x;
    const int lane = threadIdx.x & 31, g = blockIdx.y * 4 + (threadIdx.x >> 5);
    if (g >= ngrp) return;
    const long c0 = (long)g * 128 + lane * 4;
    float v[4];
    for (int j = 0; j < 4; ++j) v[j] = dld(x, xdt, r * xp + (c0 + j) * xs);
    for (int h = 1; h < 4; h <<= 1)
        for (int j = 0; j < 4; ++j)
            if ((j & h) == 0) {
                const float a = v[j], b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            }
    for (int mask = 1; mask < 32; mask <<= 1)
        for (int j = 0; j < 4; ++j) {
            const float o = __shfl_xor(v[j], mask, 32);
            v[j] = (lane & mask) ? o - v[j] : v[j] + o;
        }
    float m = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor(m, o, 32));
    const float sc = m > 0.0f ? m * (1.0f / 448.0f) : 1.0f, inv = 1.0f / sc;
    if (lane == 0) s[r * sp + (long)g * ss] = sc;
    for (int j = 0; j < 4; ++j) q[r * qp + (c0 + j) * qs] = e4m3_byte(fminf(448.0f, fmaxf(-448.0f, v[j] * inv)));
    } HY4_CHAIN_END
}

}  // namespace
}  // namespace hy4k

using namespace hy4k;

int hy4_dev_had_quant_act_fp8(const RadArgs* a, RadStream st) {
    const int64_t n = pi(a, "n"), group = pi(a, "group", 128);
    if (n < 1 || group != 128 || n % 128 || !rad_args_have(a, 3)) return RAD_E_SHAPE;
    const RadTensor &x = a->t[0], &q = a->t[1], &s = a->t[2];
    if (!row_width(x, n) || !is_float(x) || !row_width(q, n) || q.dtype != RAD_F8E4M3 || !is_rank(s, 2, n / 128) ||
        s.dtype != RAD_F32)
        return RAD_E_SHAPE;
    const int64_t rows = x.shape[0];
    if (rows < 1 || q.shape[0] != rows || s.shape[0] != rows) return RAD_E_SHAPE;
    const int ngrp = (int)(n / 128);
    return emit(st, HY4_CHAINED(a, had_quant_k), dim3((unsigned)rows, (unsigned)((ngrp + 3) / 4)), dim3(128), 0u, x.data, dcode(x),
                 x.stride[0], x.stride[x.rank - 1], (uint8_t*)q.data, q.stride[0], q.stride[q.rank - 1],
                 (float*)s.data, s.stride[0], s.stride[1], ngrp);
}
