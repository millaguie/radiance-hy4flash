/* ref_quant.cpp -- had_quant_act_fp8: a 128-point Hadamard rotation and E4M3 quantisation per
 * (row, 128 columns), the activation libr4d's w4nl64a8h experts read. In radiance 2.1.0 and 2.2.0 only
 * libhalo declares it (halo_quant_act_fp8.hip), and libhalo does not load on the R9700, so hy4k
 * carries it with the same semantics: unnormalised FWHT in natural order (the factor 128 goes in
 * the weight's scale), scale amax/448 (1 if the group is all zeros), E4M3 codes rounded to nearest
 * even and clipped to ±448, `q` row-major and `scale` [M, n/128] f32. */
#include "hy4k.h"
#include "host_util.h"

#include <algorithm>

using namespace hy4k;

namespace {
void fwht128(float* v) {
    for (int h = 1; h < 128; h <<= 1)
        for (int i = 0; i < 128; i += 2 * h)
            for (int j = i; j < i + h; ++j) {
                const float a = v[j], b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            }
}
}  // namespace

int hy4_ref_had_quant_act_fp8(const RadArgs* a, RadStream) {
    const int64_t n = pi(a, "n"), group = pi(a, "group", 128);
    if (n < 1 || group != 128 || n % 128 || !rad_args_have(a, 3)) return RAD_E_SHAPE;
    const RadTensor &x = a->t[0], &q = a->t[1], &s = a->t[2];
    if (!row_width(x, n) || !is_float(x) || !row_width(q, n) || q.dtype != RAD_F8E4M3 || !is_rank(s, 2, n / 128) ||
        s.dtype != RAD_F32)
        return RAD_E_SHAPE;
    const int64_t rows = x.shape[0];
    if (rows < 1 || q.shape[0] != rows || s.shape[0] != rows) return RAD_E_SHAPE;
    const int64_t xs = x.stride[x.rank - 1], qs = q.stride[q.rank - 1];
#pragma omp parallel for schedule(static) if (rows > 16)
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t g = 0; g < n / 128; ++g) {
            float v[128];
            for (int i = 0; i < 128; ++i) v[i] = ld(x, row(x, r) + (g * 128 + i) * xs);
            fwht128(v);
            float m = 0;
            for (int i = 0; i < 128; ++i) m = std::max(m, std::fabs(v[i]));
            const float sc = m > 0.0f ? m * (1.0f / 448.0f) : 1.0f, inv = 1.0f / sc;
            st(s, row(s, r) + col(s, g), sc);
            uint8_t* qd = (uint8_t*)q.data;
            for (int i = 0; i < 128; ++i)
                qd[row(q, r) + (g * 128 + i) * qs] = rad_f32_to_fp8e4m3(std::min(448.0f, std::max(-448.0f, v[i] * inv)));
        }
    }
    return RAD_OK;
}
