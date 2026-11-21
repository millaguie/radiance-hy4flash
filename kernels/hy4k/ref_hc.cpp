/* ref_hc.cpp -- Hy4's independent hyper-connections (HF HYV4FlashHyperConnection and
 * HYV4FlashHyperHead), FP32 host kernels. Specification: docs/OPS-HY4.md. */
#include "hy4k.h"
#include "host_util.h"

#include <vector>

using namespace hy4k;

int hy4_ref_hc_pre(const RadArgs* a, RadStream) {
    const int64_t n = pi(a, "n"), hc = pi(a, "hc"), head = pi(a, "head", 0);
    const float eps = (float)pf(a, "eps", 1e-5), heps = (float)pf(a, "hc_eps", 1e-6);
    const float mag = (float)pf(a, "magnitude", 2.0);
    if (n < 1 || hc < 1 || hc > 16 || hc * n > kMaxHcWidth || (head != 0 && head != 1) || !rad_args_have(a, 5))
        return RAD_E_SHAPE;
    const RadTensor &h = a->t[0], &fn = a->t[1], &base = a->t[2], &scale = a->t[3], &x = a->t[4];
    const RadTensor* post = head ? nullptr : rad_arg_in(a, 5);
    const int64_t width = hc * n, mix = head ? hc : 2 * hc;
    if (!row_width(h, width) || !is_float(h)) return RAD_E_SHAPE;
    const int64_t rows = h.shape[0];
    if (rows < 1 || !is_rank(fn, 2, width) || fn.shape[0] != mix || fn.dtype != RAD_F32 ||
        !is_rank(base, 1) || base.shape[0] != mix || !is_rank(scale, 1) || scale.shape[0] != (head ? 1 : 2) ||
        !is_rank(x, 2, n) || x.shape[0] != rows || !is_float(x))
        return RAD_E_SHAPE;
    if (!head && (!post || !is_rank(*post, 2, hc) || post->shape[0] != rows || post->dtype != RAD_F32))
        return RAD_E_SHAPE;
    const int64_t hs = h.stride[h.rank - 1];
#pragma omp parallel for schedule(static) if (rows > 16)
    for (int64_t r = 0; r < rows; ++r) {
        float flat[kMaxHcWidth];
        const int64_t hb = row(h, r);
        float sq = 0;
        for (int64_t i = 0; i < width; ++i) {
            flat[(size_t)i] = ld(h, hb + i * hs);
            sq += flat[(size_t)i] * flat[(size_t)i];
        }
        const float inv = 1.0f / std::sqrt(sq / (float)width + eps);
        float m[32];
        for (int64_t j = 0; j < mix; ++j) {
            const int64_t fb = row(fn, j);
            float dot = 0;
            for (int64_t i = 0; i < width; ++i) dot += flat[(size_t)i] * ld(fn, fb + col(fn, i));
            m[j] = dot * inv;
        }
        const float s0 = ld(scale, 0);
        float pre[16];
        for (int64_t j = 0; j < hc; ++j) pre[j] = sigmoid(m[j] * s0 + ld(base, col(base, j))) + heps;
        if (!head) {
            const float s1 = ld(scale, col(scale, 1));
            for (int64_t j = 0; j < hc; ++j)
                st(*post, row(*post, r) + col(*post, j),
                   mag * sigmoid(m[hc + j] * s1 + ld(base, col(base, hc + j))) + heps);
        }
        const int64_t xb = row(x, r);
        for (int64_t i = 0; i < n; ++i) {
            float v = 0;
            for (int64_t j = 0; j < hc; ++j) v += pre[j] * flat[(size_t)(j * n + i)];
            st(x, xb + col(x, i), v);
        }
    }
    return RAD_OK;
}

int hy4_ref_hc_post(const RadArgs* a, RadStream) {
    const int64_t n = pi(a, "n"), hc = pi(a, "hc");
    if (n < 1 || hc < 1 || !rad_args_have(a, 3)) return RAD_E_SHAPE;
    const RadTensor &h = a->t[0], &y = a->t[1], &post = a->t[2];
    if (!row_width(h, hc * n) || !is_float(h)) return RAD_E_SHAPE;
    const int64_t rows = h.shape[0], hs = h.stride[h.rank - 1];
    if (rows < 1 || !is_rank(y, 2, n) || y.shape[0] != rows || !is_float(y) || !is_rank(post, 2, hc) ||
        post.shape[0] != rows || post.dtype != RAD_F32)
        return RAD_E_SHAPE;
#pragma omp parallel for schedule(static) if (rows > 16)
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t hb = row(h, r), yb = row(y, r), pb = row(post, r);
        for (int64_t j = 0; j < hc; ++j) {
            const float p = ld(post, pb + col(post, j));
            for (int64_t i = 0; i < n; ++i) {
                const int64_t o = hb + (j * n + i) * hs;
                st(h, o, p * ld(y, yb + col(y, i)) + ld(h, o));
            }
        }
    }
    return RAD_OK;
}
