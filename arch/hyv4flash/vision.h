/* vision.h -- Hy4-flash's vision tower (HF HYV4FlashVisionTransformer), inside hyv4flash.cpp.
 *
 * It needs the Hy4 family of radiance's processor (core/mm/processor.cpp, branch hy4-vision): it
 * gives the patches in merge×merge block order with `merge` extra columns per row, where HF puts
 * the learned line break. Per encoder pass (RadBatch::enc):
 *
 *   pc   = hy4_vit_compact(patches)             drops the break column (the tower does not see it)
 *   x    = pc · Wp^T + bp                        HF's 16×16 conv as a GEMM
 *   x   += hy4_vit_pos(table 128×128)            learned positions, resampled (bilinear)
 *   27 times: x += o(attn(qkv(LN1 x))) ; x += fc2(gelu(fc1(LN2 x)))   no RoPE, full attention
 *                                                within each image, exact GELU
 *   m    = RMSNorm(x) viewed as [Pc/4, 4·1152]  one 2×2 block per row (the processor's order)
 *   y    = proj_2(gelu(proj_0(m)))               `perceive`'s 2×2 and 1×1 convs as GEMMs
 *   out  = hy4_vit_newline(y, image_newline)     the grid with the break column
 *
 * proj_0 is a conv [4608, 1152, 2, 2]: as a GEMM over [p00 | p01 | p10 | p11] it wants its columns
 * in (ky, kx, c) order, which a radiance map cannot permute; conv/trim.py writes it already
 * permuted into the --add constants (`hy4.vit_merge_w`). The rest are read from the checkpoint.
 * Every card runs the whole tower with replicated weights (as rad_block_vit.h does). */
#ifndef HY4_VISION_H
#define HY4_VISION_H

static const char* const kVitMergeW = "hy4.vit_merge_w";

struct Vision {
    bool on = false;
    int64_t depth = 0, width = 0, heads = 0, hd = 0, ff = 0, patch = 0, merge = 0, side = 0, n_embd = 0;
    int64_t pdim = 0, P = 0;
    double eps = 1e-5;
    struct Blk {
        rad_weight n1w = 0, n1b = 0, qkv_w = 0, qkv_b = 0, o_w = 0, o_b = 0, n2w = 0, n2b = 0;
        rad_weight f1_w = 0, f1_b = 0, f2_w = 0, f2_b = 0;
        rad_op ln1 = 0, qkv = 0, attn = 0, o = 0, ln2 = 0, f1 = 0, f2 = 0;
    };
    std::vector<Blk> L;
    rad_weight w_patch = 0, b_patch = 0, w_pos = 0, w_mn = 0, w_m1 = 0, b_m1 = 0, w_m2 = 0, b_m2 = 0, w_nl = 0;
    rad_op op_compact = 0, op_patch = 0, op_pos = 0, op_mn = 0, op_m1 = 0, op_m2 = 0, op_nl = 0;
    rad_buf pixc = 0, crdc = 0, cuc = 0, x = 0, hb = 0, qkv = 0, att = 0, fb = 0, mh = 0, mf = 0, y = 0, out = 0;

    static bool is_present(const RadModelMeta* m) { return rad_meta_geti(m, "vision_config.num_hidden_layers", 0) > 0; }

    int declare(RadBuilder* b, const RadModelMeta* m, const RadBuildCtx* ctx, Names& nm, const Geom& g,
                int64_t model_embd, int first_group) {
        on = false;
        if (!is_present(m) || ctx->max_enc_patches <= 0) return RAD_OK;
        depth  = rad_meta_geti(m, "vision_config.num_hidden_layers", 0);
        width  = rad_meta_geti(m, "vision_config.hidden_size", 0);
        heads  = rad_meta_geti(m, "vision_config.num_attention_heads", 0);
        ff     = rad_meta_geti(m, "vision_config.intermediate_size", 0);
        patch  = rad_meta_geti(m, "vision_config.patch_size", 0);
        merge  = rad_meta_geti(m, "vision_config.spatial_merge_size", 0);
        n_embd = rad_meta_geti(m, "vision_config.out_hidden_size", 0);
        side   = rad_meta_geti(m, "vision_config.max_image_size", 0) / std::max<int64_t>(patch, 1);
        eps    = rad_meta_getf(m, "vision_config.rms_norm_eps", 1e-5);
        const char* act = rad_meta_gets(m, "vision_config.hidden_act", "gelu");
        if (depth <= 0 || width <= 0 || heads <= 0 || width % heads || ff <= 0 || patch <= 0 || merge != 2 ||
            side <= 1 || n_embd != model_embd || rad_meta_geti(m, "vision_config.temporal_patch_size", 1) != 1 ||
            std::strcmp(act, "gelu")) {
            std::fprintf(stderr, "radiance: hyv4flash: vision_config this tower does not serve (layers %lld, "
                                 "width %lld, heads %lld, patch %lld, merge %lld, output %lld, act %s)\n",
                         (long long)depth, (long long)width, (long long)heads, (long long)patch, (long long)merge,
                         (long long)n_embd, act);
            return RAD_E_UNSUPPORTED;
        }
        hd = width / heads;
        pdim = 3 * patch * patch;
        const int64_t mm = merge * merge, mw = mm * width;
        P = ctx->max_enc_patches;
        if (ctx->shape_probe) P = std::max<int64_t>(mm, std::min<int64_t>(P, ctx->max_tok) / mm * mm);
        if (P % mm) return RAD_E_SHAPE;
        const int64_t R = P / mm, max_seg = ctx->max_enc_patches;
        const int ACC = RAD_ACCESS_PER_REQUEST;

        auto bw = [&](const char* dn, const char* src, uint32_t dt, std::initializer_list<int64_t> shp,
                      RadWeightGroup grp) -> rad_weight {
            if (map_copy(b, dn, src) < 0) return 0;
            return decl_w(b, dn, dt, shp, ACC, RAD_SHARD_NONE, grp);
        };
        w_patch = bw(nm.f("v.patch.weight"), nm.ckpt("vit.embeddings.patch_embedding.weight"), RAD_BF16,
                     { width, pdim }, grp_model());
        b_patch = bw(nm.f("v.patch.bias"), nm.ckpt("vit.embeddings.patch_embedding.bias"), RAD_BF16, { width },
                     grp_model());
        w_pos = bw(nm.f("v.pos.weight"), nm.ckpt("vit.embeddings.position_embedding.weight"), RAD_BF16,
                   { side * side + 1, width }, grp_model());
        L.assign((size_t)depth, Blk{});
        for (int64_t i = 0; i < depth; ++i) {
            Blk& k = L[(size_t)i];
            const RadWeightGroup grp = grp_layer(first_group + (int)i);
            const int li = (int)i;
            const char* p = nm.ckpt("vit.layers.%d", li);
            const std::string s(p);
            auto ck = [&](const char* suf) { return nm.ckpt("%s.%s", s.c_str(), suf); };
            k.n1w = bw(nm.f("v.blk.%d.ln1.weight", li), ck("input_layernorm.weight"), RAD_BF16, { width }, grp);
            k.n1b = bw(nm.f("v.blk.%d.ln1.bias", li), ck("input_layernorm.bias"), RAD_BF16, { width }, grp);
            HY_TRY(map_concat(b, nm.f("v.blk.%d.qkv.weight", li),
                              { ck("self_attn.q_proj.weight"), ck("self_attn.k_proj.weight"), ck("self_attn.v_proj.weight") }));
            k.qkv_w = decl_w(b, nm.f("v.blk.%d.qkv.weight", li), RAD_BF16, { 3 * width, width }, ACC, RAD_SHARD_NONE, grp);
            HY_TRY(map_concat(b, nm.f("v.blk.%d.qkv.bias", li),
                              { ck("self_attn.q_proj.bias"), ck("self_attn.k_proj.bias"), ck("self_attn.v_proj.bias") }));
            k.qkv_b = decl_w(b, nm.f("v.blk.%d.qkv.bias", li), RAD_BF16, { 3 * width }, ACC, RAD_SHARD_NONE, grp);
            k.o_w = bw(nm.f("v.blk.%d.o.weight", li), ck("self_attn.o_proj.weight"), RAD_BF16, { width, width }, grp);
            k.o_b = bw(nm.f("v.blk.%d.o.bias", li), ck("self_attn.o_proj.bias"), RAD_BF16, { width }, grp);
            k.n2w = bw(nm.f("v.blk.%d.ln2.weight", li), ck("post_attention_layernorm.weight"), RAD_BF16, { width }, grp);
            k.n2b = bw(nm.f("v.blk.%d.ln2.bias", li), ck("post_attention_layernorm.bias"), RAD_BF16, { width }, grp);
            k.f1_w = bw(nm.f("v.blk.%d.fc1.weight", li), ck("mlp.dense_h_to_4h.weight"), RAD_BF16, { ff, width }, grp);
            k.f1_b = bw(nm.f("v.blk.%d.fc1.bias", li), ck("mlp.dense_h_to_4h.bias"), RAD_BF16, { ff }, grp);
            k.f2_w = bw(nm.f("v.blk.%d.fc2.weight", li), ck("mlp.dense_4h_to_h.weight"), RAD_BF16, { width, ff }, grp);
            k.f2_b = bw(nm.f("v.blk.%d.fc2.bias", li), ck("mlp.dense_4h_to_h.bias"), RAD_BF16, { width }, grp);
            if (!k.n1w || !k.n1b || !k.qkv_w || !k.qkv_b || !k.o_w || !k.o_b || !k.n2w || !k.n2b || !k.f1_w ||
                !k.f1_b || !k.f2_w || !k.f2_b)
                return fail(__LINE__);
        }
        HY_TRY(map_copy(b, nm.f("v.merge.norm.weight"), nm.ckpt("vit.perceive.norm.weight")));
        w_mn = decl_w(b, nm.f("v.merge.norm.weight"), RAD_F32, { width }, ACC, RAD_SHARD_NONE, grp_model());
        w_m1 = bw(nm.f("v.merge.fc1.weight"), kVitMergeW, RAD_BF16, { mw, mw }, grp_model());
        b_m1 = bw(nm.f("v.merge.fc1.bias"), nm.ckpt("vit.perceive.proj.0.bias"), RAD_BF16, { mw }, grp_model());
        w_m2 = bw(nm.f("v.merge.fc2.weight"), nm.ckpt("vit.perceive.proj.2.weight"), RAD_BF16, { n_embd, mw },
                  grp_model());
        b_m2 = bw(nm.f("v.merge.fc2.bias"), nm.ckpt("vit.perceive.proj.2.bias"), RAD_BF16, { n_embd }, grp_model());
        w_nl = bw(nm.f("v.newline"), nm.ckpt("vit.perceive.image_newline"), RAD_BF16, { n_embd }, grp_model());
        if (!w_patch || !b_patch || !w_pos || !w_mn || !w_m1 || !b_m1 || !w_m2 || !b_m2 || !w_nl) return fail(__LINE__);

        pixc = decl_b(b, nm.f("v.pixc"), RAD_BF16, { P, pdim });
        crdc = decl_b(b, nm.f("v.crdc"), RAD_I32, { P, 4 });
        cuc  = decl_b(b, nm.f("v.cuc"), RAD_I32, { P / mm + 1 });
        x    = decl_b(b, nm.f("v.x"), RAD_BF16, { P, width });
        hb   = decl_b(b, nm.f("v.h"), RAD_BF16, { P, width });
        qkv  = decl_b(b, nm.f("v.qkv"), RAD_BF16, { P, 3 * width });
        att  = decl_b(b, nm.f("v.att"), RAD_BF16, { P, width });
        fb   = decl_b(b, nm.f("v.ff"), RAD_BF16, { P, ff });
        mh   = decl_b(b, nm.f("v.mh"), RAD_BF16, { R, mw });
        mf   = decl_b(b, nm.f("v.mf"), RAD_BF16, { R, mw });
        y    = decl_b(b, nm.f("v.y"), RAD_BF16, { R, n_embd });
        out  = decl_b(b, nm.f("v.out"), RAD_BF16, { R, n_embd });
        if (!pixc || !crdc || !cuc || !x || !hb || !qkv || !att || !fb || !mh || !mf || !y || !out) return fail(__LINE__);

        auto gemm = [&](int64_t rows, int64_t N, int64_t K, const char* a, rad_weight w, rad_weight bias) {
            return RAD_OP(b, "gemm_nt_bias",
                          RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", N), RAD_INT("K", K),
                                     RAD_STR("dtype", "bf16"), RAD_STR("act", a)),
                          RAD_WEIGHTS(w, bias));
        };
        auto lnorm = [&](rad_weight w, rad_weight bias) {
            return RAD_OP(b, "layernorm",
                          RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", width), RAD_F64("eps", eps),
                                     RAD_STR("dtype", "bf16")),
                          RAD_WEIGHTS(w, bias));
        };
        op_compact = rw(b, RAD_OP(b, "hy4_vit_compact",
                                  RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("dim", pdim), RAD_INT("merge", merge)),
                                  RAD_NOWEIGHTS),
                        {}, { pixc, crdc, cuc });
        op_patch = rw(b, gemm(P, width, pdim, "none", w_patch, b_patch), { pixc }, { x });
        op_pos = rw(b, RAD_OP(b, "hy4_vit_pos",
                              RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("n", width), RAD_INT("side", side),
                                         RAD_STR("dtype", "bf16")),
                              RAD_WEIGHTS(w_pos)),
                    { x, crdc }, { x });
        for (Blk& k : L) {
            k.ln1 = rw(b, lnorm(k.n1w, k.n1b), { x }, { hb });
            k.qkv = rw(b, gemm(P, 3 * width, width, "none", k.qkv_w, k.qkv_b), { hb }, { qkv });
            k.attn = rw(b, RAD_OP(b, "attn_dense",
                                  RAD_PARAMS(RAD_RANGE("M", 1, P), RAD_INT("head_dim", hd), RAD_INT("gqa", 1),
                                             RAD_INT("causal", 0), RAD_STR("dtype", "bf16"),
                                             RAD_F64("scale", 1.0 / std::sqrt((double)hd)),
                                             RAD_INT("max_seqlen", max_seg)),
                                  RAD_NOWEIGHTS),
                        { qkv, cuc }, { att });
            k.o = rw(b, gemm(P, width, width, "none", k.o_w, k.o_b), { att, x }, { x });
            k.ln2 = rw(b, lnorm(k.n2w, k.n2b), { x }, { hb });
            k.f1 = rw(b, gemm(P, ff, width, "gelu", k.f1_w, k.f1_b), { hb }, { fb });
            k.f2 = rw(b, gemm(P, width, ff, "none", k.f2_w, k.f2_b), { fb, x }, { x });
        }
        op_mn = rw(b, op_rmsnorm(b, g, P, width, eps, w_mn), { x }, { mh });
        op_m1 = rw(b, gemm(R, mw, mw, "gelu", w_m1, b_m1), { mh }, { mf });
        op_m2 = rw(b, gemm(R, n_embd, mw, "none", w_m2, b_m2), { mf }, { y });
        op_nl = rw(b, RAD_OP(b, "hy4_vit_newline",
                             RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", n_embd), RAD_INT("merge", merge),
                                        RAD_STR("dtype", "bf16")),
                             RAD_WEIGHTS(w_nl)),
                   { y, cuc }, { out });
        if (!op_compact || !op_patch || !op_pos || !op_mn || !op_m1 || !op_m2 || !op_nl) return fail(__LINE__);
        for (const Blk& k : L)
            if (!k.ln1 || !k.qkv || !k.attn || !k.o || !k.ln2 || !k.f1 || !k.f2) return fail(__LINE__);

        if (!ctx->shape_probe) {
            /* The band's largest image (square, side sqrt(longest_edge)) with the break column: if it
             * does not fit in one pass, the processor shrinks it and the model sees a different
             * image than in HF. */
            const int64_t band = rad_meta_geti(m, "processor.image_processor.size.longest_edge",
                                               rad_meta_geti(m, "processor.image_processor.max_pixels", 2048 * 2048));
            const int64_t f2 = patch * merge;
            const int64_t edge = (int64_t)std::sqrt((double)band) / f2 * f2 / patch;
            const int64_t needed = edge * (edge + merge);
            if (max_seg < needed)
                std::fprintf(stderr, "radiance: hyv4flash: mm-max-patches %lld is not enough even for the band's "
                                     "largest square image (%lld patches with the break column; elongated "
                                     "ones need more): those are shrunk relative to HF\n", (long long)max_seg, (long long)needed);
            RadEncoderDecl e{};
            e.name        = "vision";
            e.modalities  = RAD_MM_IMAGE;
            e.patch_dim   = pdim;
            e.merge       = mm;
            e.max_patches = P;
            e.n_embd      = n_embd;
            e.out         = out;
            HY_TRY(rad_declare_encoder(b, &e));
            rad_note(b, "vision tower: %lld blocks of %lld (%lld heads of %lld), ff %lld, patch %lld, "
                        "merge %lld, %lld patches per pass",
                     (long long)depth, (long long)width, (long long)heads, (long long)hd, (long long)ff,
                     (long long)patch, (long long)merge, (long long)P);
        }
        on = true;
        return RAD_OK;
    }

    void step(RadCtx* c, const RadBatch* batch) const {
        const int64_t Pn = batch->enc_n_patch, S = batch->enc_n_seg, mm = merge * merge;
        if (!on || Pn <= 0 || Pn > P || Pn % mm || S <= 0 || !batch->enc_seg_h || !batch->enc_seg_w) {
            rad_step_fail(c, "hyv4flash: an encoder pass this tower does not serve");
            return;
        }
        /* Patches without the break column: the host knows them from each segment's grid. */
        int64_t Pc = 0, Pt = 0;
        for (int64_t s = 0; s < S; ++s) {
            const int64_t h = batch->enc_seg_h[s], w = batch->enc_seg_w[s];
            if (w <= merge || h % merge || w % merge) {
                rad_step_fail(c, "hyv4flash: a segment without the break column or with a grid that is not a "
                                 "multiple of merge (a processor other than Hy4's)");
                return;
            }
            Pc += h * (w - merge);
            Pt += h * w;
        }
        /* the grids must match the pass's patches, or the tower would read stale rows */
        if (Pt != Pn || Pc % mm) {
            rad_step_fail(c, "hyv4flash: the segment grids do not match the pass's patches");
            return;
        }
        const int64_t Rn = Pn / mm, Rc = Pc / mm;
        const RadOperand pix = praw2(batch->enc_pixels, RAD_BF16, Pn, pdim);
        const RadOperand crd = praw2(batch->enc_coord, RAD_I32, 4, Pn);
        const RadOperand cu = praw(batch->enc_cu, RAD_I32, S + 1);
        RadOperand cuc_s = brows(cuc, S + 1);

        RAD_ISSUE_N(c, op_compact, Pn, pix, crd, cu, brows(pixc, Pc), brows(crdc, Pc), cuc_s);
        RAD_ISSUE_N(c, op_patch, Pc, brows(pixc, Pc), RAD_W(w_patch), RAD_W(b_patch), RAD_NONE, brows(x, Pc));
        RAD_ISSUE_N(c, op_pos, Pc, brows(x, Pc), RAD_W(w_pos), brows(crdc, Pc));
        for (const Blk& k : L) {
            RAD_ISSUE_N(c, k.ln1, Pc, brows(x, Pc), RAD_W(k.n1w), RAD_W(k.n1b), brows(hb, Pc));
            RAD_ISSUE_N(c, k.qkv, Pc, brows(hb, Pc), RAD_W(k.qkv_w), RAD_W(k.qkv_b), RAD_NONE, brows(qkv, Pc));
            RAD_ISSUE_N(c, k.attn, Pc, bcol(qkv, 0, width, Pc), bcol(qkv, width, width, Pc),
                        bcol(qkv, 2 * width, width, Pc), cuc_s, brows(att, Pc));
            RAD_ISSUE_N(c, k.o, Pc, brows(att, Pc), RAD_W(k.o_w), RAD_W(k.o_b), brows(x, Pc), brows(x, Pc));
            RAD_ISSUE_N(c, k.ln2, Pc, brows(x, Pc), RAD_W(k.n2w), RAD_W(k.n2b), brows(hb, Pc));
            RAD_ISSUE_N(c, k.f1, Pc, brows(hb, Pc), RAD_W(k.f1_w), RAD_W(k.f1_b), RAD_NONE, brows(fb, Pc));
            RAD_ISSUE_N(c, k.f2, Pc, brows(fb, Pc), RAD_W(k.f2_w), RAD_W(k.f2_b), brows(x, Pc), brows(x, Pc));
        }
        RAD_ISSUE_N(c, op_mn, Pc, brows(x, Pc), RAD_W(w_mn), brows(mh, Rc));
        RAD_ISSUE_N(c, op_m1, Rc, brows(mh, Rc), RAD_W(w_m1), RAD_W(b_m1), RAD_NONE, brows(mf, Rc));
        RAD_ISSUE_N(c, op_m2, Rc, brows(mf, Rc), RAD_W(w_m2), RAD_W(b_m2), RAD_NONE, brows(y, Rc));
        RAD_ISSUE_N(c, op_nl, Rn, brows(y, Rc), RAD_W(w_nl), crd, cu, cuc_s, brows(out, Rn));
    }
};

#endif
