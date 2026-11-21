/* ref_rope.cpp -- Hy4's rotary with images: GPT-J pairs (2i, 2i+1) and Qwen3-VL's interleaved
 * M-RoPE sections (HF HYV4FlashRotaryEmbedding.recomposition_frequencies + Hy4's
 * apply_rotary_pos_emb, which repeats each frequency across its pair). radiance has the two pieces
 * separately (`rope` mode=gptj and imrope, the latter with NeoX pairs), not together.
 *
 * Same schema as `rope` (declared with decl_rope and op="hy4_rope"): mode `gptj` with positions
 * [M] (text, same as `rope`), or mode `imrope` with positions [3, M] and `sections` (s0 s1 s2):
 * frequency i reads component 1 if i % 3 == 1 and i < 3·s1, component 2 if i % 3 == 2 and
 * i < 3·s2, and component 0 otherwise. It rotates the first `rotary_dim` channels of each Q and K
 * head; V is left untouched. */
#include "hy4k.h"
#include "host_util.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

using namespace hy4k;

namespace {

int read_sections(const RadArgs* a, int64_t* sec) {
    const char* v = rad_args_gets(a, "sections");
    if (!v || !*v) return 0;
    int n = 0;
    for (const char* q = v; *q && n < 3;) {
        while (*q == ' ' || *q == ',') ++q;
        if (!*q) break;
        char* e = nullptr;
        const long x = std::strtol(q, &e, 10);
        if (e == q || x < 0) return -1;
        sec[n++] = x;
        q = e;
    }
    return n;
}

}  // namespace

int hy4_ref_rope(const RadArgs* a, RadStream) {
    const int64_t hd = pi(a, "head_dim"), nh = pi(a, "n_head"), nkv = pi(a, "n_head_kv", 0);
    const int64_t rot = pi(a, "rotary_dim", hd);
    const double theta = pf(a, "theta", 10000.0), esc = pf(a, "scale", 1.0);
    const char* mode = rad_args_gets(a, "mode");
    if (hd < 2 || nh < 1 || nkv < 0 || rot < 2 || rot > hd || (rot & 1) || rot / 2 > 256 || !mode ||
        !rad_args_have(a, 2))
        return RAD_E_SHAPE;
    const bool im = !std::strcmp(mode, "imrope");
    if (!im && std::strcmp(mode, "gptj")) return RAD_E_UNSUPPORTED;
    int64_t sec[3] = {};
    const int ns = read_sections(a, sec);
    if (im && ns != 3) return RAD_E_SHAPE;
    const RadTensor &x = a->t[0], &pos = a->t[1];
    const int64_t width = (nh + nkv) * hd;
    if (!is_present(x) || !is_float(x) || x.rank != 2 || x.shape[1] < width || x.stride[1] != 1) return RAD_E_SHAPE;
    const int64_t M = x.shape[0];
    const bool multi = pos.rank == 2;
    if (M < 1 || !is_int(pos) || (multi && (!im || pos.shape[0] < 3 || pos.shape[1] != M)) ||
        (!multi && (pos.rank != 1 || pos.shape[0] != M)))
        return RAD_E_SHAPE;
    const int64_t half = rot / 2;
    float inv[256];
    int comp[256];
    for (int64_t i = 0; i < half; ++i) {
        /* radiance's `rope` frequency ladder (ref_pow in float): a text without an image gives the
         * same as radiance's rope */
        inv[i] = 1.0f / std::pow((float)theta, (float)(2 * i) / (float)rot);
        comp[i] = 0;
        if (im) {
            if (i % 3 == 1 && i < 3 * sec[1]) comp[i] = 1;
            else if (i % 3 == 2 && i < 3 * sec[2]) comp[i] = 2;
        }
    }
    const float div = esc != 0.0 ? (float)esc : 1.0f;
#pragma omp parallel for schedule(static) if (M > 16)
    for (int64_t m = 0; m < M; ++m) {
        float p[3];
        for (int c = 0; c < 3; ++c)
            p[c] = (float)(multi ? ldi(pos, c * pos.stride[0] + m * pos.stride[1]) : ldi(pos, vi(pos, m))) / div;
        const int64_t fb = row(x, m);
        for (int64_t h = 0; h < nh + nkv; ++h) {
            const int64_t b = fb + h * hd;
            for (int64_t i = 0; i < half; ++i) {
                /* in f32, like radiance's `rope` (HF rounds cos/sin to x's dtype: in the BF16
                 * pass that is part of its noise floor, not of the model) */
                const float ang = p[comp[i]] * inv[i];
                const float c = std::cos(ang), s = std::sin(ang);
                const float x0 = ld(x, b + 2 * i), x1 = ld(x, b + 2 * i + 1);
                st(x, b + 2 * i, x0 * c - x1 * s);
                st(x, b + 2 * i + 1, x1 * c + x0 * s);
            }
        }
    }
    return RAD_OK;
}
