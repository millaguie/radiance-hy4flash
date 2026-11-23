/* hyv4flash.cpp -- architecture plugin for Tencent Hy4-flash (text only) for radiance 2.2.0.
 *
 * Design: docs/DESIGN.md. New ops: the hy4k kernel plugin (docs/OPS-HY4.md).
 *
 * BF16 trunk (BF16 or w4nl64 experts), tp 1 or 2. The MLA in two forms (HY4_MLA, docs/DESIGN.md):
 * absorbed (a one-head latent cache; the serving one) or expanded (K and V per head; the one for
 * plan step 4's parity). Without the variable, whichever the container allows.
 *
 * Per layer (HF HYV4FlashDecoderLayer):
 *   [Engram]  ids (once per step) -> embed_lookup_q -> rows_rmsnorm(fp8) -> value/key_i ->
 *             engram_gate (sum into the 4 streams)
 *   hc_pre(attn) -> rmsnorm -> MLA (+ indexer in the sparse ones) -> rmsnorm(post_proj) -> hc_post
 *   hc_pre(ffn)  -> rmsnorm -> dense MLP (layer 0) or latent MoE + shared -> rmsnorm -> hc_post
 * Closing: gather_rows -> hc_pre(head) -> rmsnorm -> logits_gemm.
 *
 * Debug tap: HY4_DUMP_DIR=<dir> writes, in the layers of HY4_DUMP_LAYERS (default 0,1,2), the
 * same names as ref/reference.py (layer.out, attn.out, …) as raw files
 * `<dir>/s<step>.<name>.L<layer>.<dtype>` with the step's rows. It synchronises the stream.
 */
#include <arch/rad_arch.h>
#include <arch/rad_fp8.h>
#include <arch/rad_block_moe_fp8.h>
#include <arch/rad_block_vit.h>      /* MediaRows: the encoder rows over the embedding */
/* The copy of rad_fp8.h with the rotated clamp (vendor/) must be the one that got in: if a change
 * in include order lets the installed one through, this fails here and not when a rotated container
 * is declared. */
#ifndef HY4_RAD_FP8_VENDOR
#error "hyv4flash needs vendor/arch/rad_fp8.h before the installed header (hyv4flash's CMakeLists)"
#endif
#include "rad_abi_layout.h"

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace hyv4flash {
using namespace rad::arch;

/* A declare failure says which plugin line it happens at: the core only reports the code. */
static int fail(int line, int st = RAD_E_INVAL) {
    std::fprintf(stderr, "radiance: hyv4flash: declare fails at hyv4flash.cpp:%d (%d)\n", line, st);
    return st;
}
#define HY_TRY(expr) do { const int st_ = (expr); if (st_ < 0) return fail(__LINE__, st_); } while (0)

/* ------------------------------------------------------------------ model geometry */
struct ModelShape {
    int64_t n_layer = 0, n_embd = 0, hc = 4, vocab = 0;
    int64_t q_lora = 0, kv_lora = 0, nope = 0, rope = 0, v_dim = 0;
    int64_t heads_sparse = 0, heads_win = 0;
    int64_t window = 0;
    double  theta_sparse = 0, theta_win = 0, eps = 0, hc_eps = 0, hc_mag = 0;
    int64_t ff_dense = 0, ff_exp = 0, latent = 0, n_expert = 0, top_k = 0;
    double  route_scale = 0, swiglu_limit = 0;
    /* indexer */
    int64_t idx_heads = 0, idx_dim = 0, idx_ratio = 0, idx_topk_groups = 0, idx_candidate = 0;
    /* Engram */
    std::vector<int> engram_layers;  /* 0-based layer indices */
    int64_t engram_pad = 0, engram_ngram = 3, engram_heads = 16, engram_d = 96, engram_map = 0;
    std::vector<bool> sparse;        /* per layer: sparse (indexed) or window attention */
    std::vector<bool> moe;           /* per layer: MoE or dense MLP */
    int first_moe() const {
        for (size_t i = 0; i < moe.size(); ++i)
            if (moe[i]) return (int)i;
        return -1;
    }
};

static int64_t gi(const RadModelMeta* m, const char* k, int64_t d = INT64_MIN) {
    return meta_text_geti(m, k, d);
}
static double gf(const RadModelMeta* m, const char* k, double d = NAN) {
    return meta_text_getf(m, k, d);
}

/* Reads the geometry from text_config. Everything that changes the arithmetic is required: a wrong
 * default gives fluent, worse text. */
static int read_shape(const RadModelMeta* m, ModelShape& f) {
    f.n_layer = gi(m, "num_hidden_layers");
    f.n_embd = gi(m, "hidden_size");
    f.hc = gi(m, "hc_mult");
    f.vocab = gi(m, "vocab_size");
    f.q_lora = gi(m, "q_lora_rank");
    f.kv_lora = gi(m, "kv_lora_rank");
    f.nope = gi(m, "qk_nope_head_dim");
    f.rope = gi(m, "qk_rope_head_dim");
    f.v_dim = gi(m, "swa_v_head_dim");
    f.heads_sparse = gi(m, "num_attention_heads");
    f.heads_win = gi(m, "swa_num_attention_heads");
    f.window = gi(m, "sliding_window");
    f.theta_sparse = gf(m, "rope_parameters.deepseek_sparse_attention.rope_theta");
    f.theta_win = gf(m, "rope_parameters.sliding_attention.rope_theta");
    f.eps = gf(m, "rms_norm_eps");
    f.hc_eps = gf(m, "hc_eps");
    f.hc_mag = gf(m, "hc_magnitude");
    f.ff_dense = gi(m, "intermediate_size");
    f.ff_exp = gi(m, "moe_intermediate_size");
    f.latent = gi(m, "routed_expert_hidden_size");
    f.n_expert = gi(m, "n_routed_experts");
    f.top_k = gi(m, "num_experts_per_tok");
    f.route_scale = gf(m, "routed_scaling_factor");
    f.swiglu_limit = gf(m, "swiglu_limit");
    f.idx_heads = gi(m, "compressed_indexer_num_attention_heads");
    f.idx_dim = gi(m, "compressed_indexer_head_dim");
    f.idx_ratio = gi(m, "compressed_indexer_compress_ratio");
    f.idx_topk_groups = gi(m, "compressed_indexer_group_topk");
    f.idx_candidate = gi(m, "compressed_indexer_candidate_topk");
    f.engram_pad = gi(m, "engram_pad_id");
    f.engram_ngram = gi(m, "engram_max_ngram_size");
    const int64_t per = gi(m, "engram_n_head_per_ngram");
    const int64_t emb = gi(m, "engram_n_embed_per_ngram");
    const int64_t vals[] = { f.n_layer, f.n_embd, f.hc, f.vocab, f.q_lora, f.kv_lora, f.nope, f.rope,
                             f.v_dim, f.heads_sparse, f.heads_win, f.window, f.ff_dense, f.ff_exp,
                             f.latent, f.n_expert, f.top_k, f.idx_heads, f.idx_dim, f.idx_ratio,
                             f.idx_topk_groups, f.idx_candidate, f.engram_pad, f.engram_ngram, per,
                             emb };
    for (int64_t v : vals)
        if (v == INT64_MIN || v <= 0) {
            std::fprintf(stderr, "radiance: hyv4flash: a text_config key is missing or <= 0\n");
            return fail(__LINE__);
        }
    const double fv[] = { f.theta_sparse, f.theta_win, f.eps, f.hc_eps, f.hc_mag, f.route_scale,
                          f.swiglu_limit };
    for (double v : fv)
        if (!(v > 0)) {
            std::fprintf(stderr, "radiance: hyv4flash: a real text_config value is missing (theta, "
                                 "eps, hc_eps, hc_magnitude, routed_scaling_factor or swiglu_limit)\n");
            return fail(__LINE__);
        }
    f.engram_heads = per * (f.engram_ngram - 1);
    f.engram_d = emb / per;
    /* What the plugin assumes and does not read as geometry: if the config says otherwise, it is
     * refused instead of serving another model. Both layer kinds share nope/rope/v
     * (Hy4: 192/64/256). */
    struct IntCheck { const char* key; int64_t want; };
    const IntCheck int_checks[] = {
        { "v_head_dim", f.v_dim }, { "swa_qk_nope_head_dim", f.nope }, { "swa_qk_rope_head_dim", f.rope },
        { "n_shared_experts", 1 }, { "n_group", 1 }, { "topk_group", 1 },
    };
    for (const IntCheck& q : int_checks) {
        const int64_t v = gi(m, q.key);
        if (v != q.want) {
            std::fprintf(stderr, "radiance: hyv4flash: %s = %lld, the plugin only serves %lld\n", q.key,
                         (long long)v, (long long)q.want);
            return RAD_E_UNSUPPORTED;
        }
    }
    struct StrCheck { const char* key; const char* want; };
    const StrCheck str_checks[] = { { "hidden_act", "silu" }, { "norm_topk_prob", "1" }, { "engram_fp8_embedding", "1" } };
    for (const StrCheck& q : str_checks) {
        const char* v = meta_text_gets(m, q.key, nullptr);
        if (!v || std::strcmp(v, q.want) != 0) {
            std::fprintf(stderr, "radiance: hyv4flash: %s = %s, the plugin only serves %s\n", q.key, v ? v : "(missing)",
                         q.want);
            return RAD_E_UNSUPPORTED;
        }
    }
    if (f.nope + f.rope != f.v_dim) {
        /* The cache keeps K and V with the same head_dim (kv_store); in Hy4, 192+64 = 256 = v. */
        std::fprintf(stderr, "radiance: hyv4flash: nope+rope (%lld) != v (%lld)\n",
                     (long long)(f.nope + f.rope), (long long)f.v_dim);
        return RAD_E_UNSUPPORTED;
    }
    /* The config's lists arrive as one string with the elements separated by spaces
     * (as layer_types in embgemma2). */
    auto words = [](const char* s) {
        std::vector<std::string> v;
        for (const char* p = s ? s : ""; *p;) {
            while (*p == ' ') ++p;
            const char* t = p;
            while (*p && *p != ' ') ++p;
            if (p > t) v.emplace_back(t, (size_t)(p - t));
        }
        return v;
    };
    const auto lt = words(meta_text_gets(m, "layer_types", ""));
    const auto mt = words(meta_text_gets(m, "mlp_layer_types", ""));
    if ((int64_t)lt.size() != f.n_layer || (int64_t)mt.size() != f.n_layer) {
        std::fprintf(stderr, "radiance: hyv4flash: layer_types (%zu) and mlp_layer_types (%zu) must name "
                             "all %lld layers\n", lt.size(), mt.size(), (long long)f.n_layer);
        return fail(__LINE__);
    }
    f.sparse.assign((size_t)f.n_layer, false);
    f.moe.assign((size_t)f.n_layer, false);
    for (int64_t l = 0; l < f.n_layer; ++l) {
        const std::string& t = lt[(size_t)l];
        if (t == "deepseek_sparse_attention") f.sparse[(size_t)l] = true;
        else if (t != "sliding_attention") {
            std::fprintf(stderr, "radiance: hyv4flash: layer_types[%lld] = '%s'\n", (long long)l, t.c_str());
            return RAD_E_UNSUPPORTED;
        }
        const std::string& u = mt[(size_t)l];
        if (u == "sparse") f.moe[(size_t)l] = true;
        else if (u != "dense") {
            std::fprintf(stderr, "radiance: hyv4flash: mlp_layer_types[%lld] = '%s'\n", (long long)l, u.c_str());
            return RAD_E_UNSUPPORTED;
        }
    }
    for (const std::string& w : words(meta_text_gets(m, "engram_layer_ids", ""))) {
        const long long id = std::strtoll(w.c_str(), nullptr, 10);
        if (id < 1 || id > f.n_layer) return fail(__LINE__);
        f.engram_layers.push_back((int)id - 1);  /* the config counts from 1 (HF: layer_idx + 1) */
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ blocks */
struct Mla {
    int layer = 0;
    bool sparse = false;
    int64_t H = 0;      /* heads on this card */
    int64_t Hall = 0;   /* heads of the model */
    rad_kvgroup kv = 0;
    LinearFP8 q_a{}, q_b{}, kv_a{}, kv_b{}, gate{}, o{};
    rad_weight w_qa_norm = 0, w_kv_norm = 0, w_sink = 0, w_post = 0;
    rad_weight w_kvb_abs = 0;   /* absorbed MLA: per-head kv_b [zeros(rope) | k_nope | v] (decl_kvb_abs) */
    rad_op op_absorb = 0, op_expand = 0;
    rad_op op_absorb_m = 0, op_expand_m = 0;   /* hy4_mla_project: the same, for steps over 32 rows */
    rad_op op_qa_norm = 0, op_kv_norm = 0, op_rope_q = 0, op_rope_k = 0, op_kv = 0, op_store = 0;
    rad_op op_attn = 0, op_attn_sel = 0, op_sig = 0, op_mul = 0, op_post = 0;
    rad_op op_ar = 0;   /* all-reduce of o_proj's output, with tp > 1 */
    /* IQ8R trunk fusions (decl_mla): q_a_norm writes q_b's int8 pair itself (rmsnorm_had_quant_i8g),
     * and op_gq is sigmoid + mul + o_proj's quantiser in one launch (gate_had_quant_i8g). */
    bool qa_q8 = false;
    rad_op op_gq = 0;
    /* indexer */
    LinearFP8 tk{}, iq{}, ihw{};
    rad_weight w_tk_norm = 0, w_ival = 0, w_igate = 0, w_iape = 0, w_inorm = 0;
    rad_op op_tk_norm = 0, op_work = 0, op_pool = 0, op_tail = 0, op_score = 0, op_select = 0;
};

struct Engram {
    int layer = 0, e = 0;
    rad_weight w_table = 0, w_norms = 0, w_val = 0, w_val_b = 0, w_keys = 0, w_keys_b = 0;
    rad_weight w_table_scale = 0;   /* the E4M3 table's one scale; 0 for the bf16 table */
    rad_weight w_nk = 0, w_nq = 0, w_gamma = 0, w_per = 0;
    rad_op op_lookup = 0, op_cast = 0, op_norm = 0, op_val = 0, op_key = 0, op_gate = 0;
};

struct Ffn {
    int layer = 0;
    bool moe = false;
    rad_weight w_norm = 0, w_out_norm = 0;
    rad_op op_norm = 0, op_out_norm = 0, op_join = 0;
    /* op_norm is rmsnorm_had_quant_i8g: it writes the int8 pair the first linear (ffn_latent_in, or
     * layer 0's gate_up) reads, under an IQ8R trunk (declare) */
    bool norm_q8 = false;
    rad_op op_ar = 0;                /* all-reduce of the routed arm, with tp > 1 */
    LinearFP8 gu{}, down{};          /* dense MLP, or the shared expert (replicated) */
    GateQuantFP8 act{};
    LinearFP8 lat_in{}, lat_out{};   /* latent MoE */
    QuantFP8 lat_q{};                /* the latent to E4M3 for the quantised experts */
    MoeFP8 moe_blk{};
};

struct Hc {
    rad_weight fn = 0, base = 0, scale = 0;
    rad_op op = 0;
};

struct Layer {
    Hc hc_attn{}, hc_ffn{};
    rad_weight w_in_norm = 0;
    rad_op op_in_norm = 0, op_post_attn = 0, op_post_ffn = 0;
    /* op_in_norm is rmsnorm_had_quant_i8g: it writes the int8 pair attn_q_a and attn_gate read */
    bool in_q8 = false;
    Mla attn{};
    Ffn ffn{};
    int engram = -1;
    int dfly = -1;   /* the DFly2 drafter layer that reads this layer's cache and selection, or -1 */
};

/* Attention buffers with the exact width of one layer kind: the projections (LinearFP8) assume an
 * output of N columns, and the sparse layers have 32 heads and the window ones 64. */
struct AttnBufs {
    rad_buf q = 0;                 /* [T, H·256] */
    rad_buf kvx = 0;               /* [T, H·(nope+v)] */
    rad_buf kbuf = 0, vbuf = 0;    /* [T, H·256] */
    rad_buf gt = 0;                /* [T, H·v] gate */
    rad_buf qlat = 0, olat = 0;    /* [T, H·kv_lora]: absorbed query, latent output (absorbed MLA) */
    ActFP8  o{};                   /* [T, H·v] attention output */
};

/* Where one attention's names, buffers and row counts come from: a trunk layer's, or a DFly2
 * drafter layer's (docs/DFLY2.md). The drafter's attention is a trunk sparse one without the
 * indexer and without a cache of its own: it reads its trunk layer's kv_full and its block's own
 * keys (bkeys), so decl_mla declares only the absorbed selection form for it. */
struct MlaSite {
    const char* wp = nullptr;      /* declared name prefix, unscoped: "blk.L" or "dfly.D" */
    const char* cp = nullptr;      /* checkpoint prefix of the attention module */
    int gl = 0;                    /* the weights' group layer */
    const Geom* g = nullptr;
    const ActFP8* a = nullptr;     /* normalised input */
    const ActFP8* qa = nullptr;
    const ActFP8* lat = nullptr;   /* the expanded form's normed latent (trunk only) */
    rad_buf kva = 0, y = 0;
    const AttnBufs* B = nullptr;
    rad_buf sel = 0, sequ = 0;     /* the selection's page lists and seqused (trunk only) */
    int64_t sel_rows = 0;          /* rows of one selection launch */
    bool drafter = false;
    /* The input norm already wrote a's int8 pair (Layer::in_q8): q_a declares no quantiser. Trunk
     * only; the drafter's site leaves it false. */
    bool a_fed = false;
};

/* ------------------------------------------------------------------ DFly2 drafter (docs/DFLY2.md) */
struct DflyLayer {
    int target = 0;                /* trunk layer t_d: the kv_full cache and the selection it reads */
    Mla at{};
    rad_weight w_in_norm = 0, w_ffn_norm = 0;
    rad_op op_in_norm = 0, op_add_attn = 0, op_ffn_norm = 0, op_act = 0, op_add_ffn = 0;
    LinearFP8 gu{}, down{};
    /* Trunk side, right after layer `target`'s selection: this layer's history words in the ring
     * entry, from the selection (seqused, then the pages) or, in a step without one, dense. */
    rad_op op_st_sequ = 0, op_st_sel = 0, op_st_dense = 0;
};

/* One of the five sequential draft steps. Each has its own ops: one set reissued five times would
 * end every buffer's liveness at its first use (the planner reads the declaration order), so the
 * normed block rows could be reused under the second step's gather. */
struct DflyStep {
    rad_op op_gy = 0, op_embed = 0, op_embed_ar = 0, op_nh = 0, op_ne = 0, op_act = 0;
    rad_op op_add = 0, op_head = 0, op_topk = 0, op_gath = 0, op_prev = 0, op_prop = 0;
    /* the hidden correction's linears: one weight each, this step's own launches (dfly_relaunch) */
    LinearFP8 gu{}, down{};
};

struct Dfly {
    bool on = false;
    int64_t n_layers = 0, block = 0, depth = 0;
    int64_t rows = 0;              /* block rows a query pass carries at most: max_seqs · block */
    int64_t ffw = 0, hcw = 0;      /* the FFN's width, the hidden correction's */
    int64_t entry = 0;             /* i32 words of a ring entry: 1 + n + layers · (1 + topk) */
    int32_t mask = -1;
    /* Converting: dfly_plan declared the name maps of the weights it measured (final_norm, layer
     * 0's ffn_down, the correction's down), which decl_dfly must not declare again. */
    bool premapped = false;
    /* The container holds the drafter's linears in IQ8R (dfly_plan reads it from their stored
     * format): its geometry is then an int8 one (Geom::w_i8h) whatever the trunk's, and the inputs of
     * its linears carry int8 code pairs. Otherwise they are bf16, as the checkpoint holds them. */
    bool i8 = false;
    std::vector<int> targets;
    Geom g{};                      /* the trunk's geometry at the drafter's rows and linears' format */
    rad_kvgroup kv = 0;            /* kv_dfly: the ring, RAD_KV_LINEAR, bound to targets[0] */
    std::vector<DflyLayer> layers;
    std::vector<DflyStep> steps;
    /* weights */
    rad_weight w_final = 0, w_enorm = 0, w_hnorm = 0, w_gate = 0;
    rad_weight w_cnh = 0, w_cne = 0;
    LinearFP8 eh{};                /* the fusion's eh_proj, [n, 2n], at max_seqs rows */
    /* buffers: [rows, ·], one row a block row, or [max_seqs, ·], one row a sequence */
    rad_buf x = 0;                 /* the residual stream; the block's embedding at first */
    ActFP8 a{}, qa{}, ff{};
    rad_buf kva = 0;               /* [normed latent | roped key]: the layer's bkeys */
    AttnBufs B{};
    rad_buf y = 0, gu = 0, yn = 0; /* a sub-block's output, gate|up, the final norm's output */
    rad_buf hf = 0, hb = 0;        /* the ring's h, f32 bits as loaded and as bf16 */
    rad_buf hist = 0;              /* i32 [rows, layers · (1 + topk)]: each layer's seqused | pages */
    rad_buf e0 = 0, h0 = 0, fz = 0;   /* [max_seqs, n]: slot 0's embedding, its h, the fused term */
    ActFP8 cat{};                  /* [max_seqs, 2n]: [enorm(e0) | hnorm(h0)], then each step's z */
    rad_buf yk = 0, ep = 0, cgu = 0, cd = 0, cc = 0;
    ActFP8 cff{};
    rad_buf logits = 0;            /* bf16 [max_seqs, this rank's vocabulary] */
    rad_buf cand = 0, cval = 0, pairs = 0, gath = 0, prev = 0;
    rad_buf proposal = 0;          /* i32 [max_seqs, depth], PERSIST: RadDrafterDecl::proposal */
    /* ops */
    rad_op op_embed = 0, op_embed_ar = 0, op_load_h = 0, op_load_hist = 0, op_cast_h = 0;
    rad_op op_g_e0 = 0, op_g_h0 = 0, op_enorm = 0, op_hnorm = 0, op_gate = 0, op_fuse = 0;
    rad_op op_final = 0;
    rad_op op_st_h = 0, op_st_pos = 0;   /* trunk side, after the head norm */
};

static rad_op op_rmsnorm(RadBuilder* b, const Geom& g, int64_t rows, int64_t n, double eps, rad_weight w);
#include "vision.h"

struct Model {
    Names nm{};
    Geom g{};          /* base geometry (theta of the sparse layers) */
    Geom gw{};         /* the window layers' one: another theta */
    ModelShape f{};
    rad_kvgroup kv_full = 0, kv_win = 0, kv_idx = 0, kv_tail = 0, kv_tok = 0;
    /* Absorbed MLA (HY4_MLA, docs/DESIGN.md «Absorbed MLA»): a one-head latent cache
     * [lat | k_rope], and the query and the output go through the latent. `mla_both`: the
     * expanded form's kv_b is declared too, so that one conversion leaves the container servable
     * in both forms. */
    bool mla_abs = false, mla_both = false;
    /* `hy4.engram_codes=rowmax448` (decl_engram): the container's Engram tables are
     * conv/engram_codes.py's E4M3 codes, row maxima 448, row scales dropped. */
    bool engram_codes = false;
    /* buffers */
    rad_buf h = 0;                 /* [T, hc·n] the streams */
    rad_buf xin = 0;               /* [T, n] hc_pre's mix */
    rad_buf post = 0;              /* [T, hc] f32 */
    ActFP8  a{};                   /* [T, n] normalised input of a sub-block */
    ActFP8  qa{};                  /* [T, q_lora] */
    rad_buf kva = 0;               /* [T, kv_lora + rope] */
    ActFP8  lat{};                 /* [T, kv_lora] */
    AttnBufs ab[2]{};               /* 0: sparse layers, 1: window layers */
    rad_buf y = 0;                 /* [T, n] output of a sub-block */
    /* indexer */
    ActFP8  tkn{};                 /* [T, idx_dim] per-token keys */
    rad_buf iq = 0;                /* [T, idx_heads·idx_dim] */
    rad_buf ihw = 0;               /* [T, idx_heads] */
    rad_buf stage = 0, page = 0, bpos = 0, nc = 0;
    rad_buf score = 0, sel = 0, nsel = 0, sequ = 0;
    /* {0, plane_rows}: the single sequence of a window of a step wider than the score plane
     * (as in QSA, rad_qsa.h). It is uploaded from the pinned copy `win_cu_h` in each step that
     * uses it. */
    rad_buf win_cu = 0;
    int32_t* win_cu_h = nullptr;
    bool win_cu_tried = false;
    int64_t max_work = 0, plane_rows = 0, max_groups = 0, sel_rows = 0;
    /* Prefill span pass: Code000, radiance e4541c70 (qwen4exp_fp8's piece_rows, span_on and
     * pass_step). `piece_rows` is one routed pass without a span, 2048 rows or the step if it is
     * shorter. `span_on`: the deployment plans a step wider than a piece, which is then a single
     * sequence (--prefill-span, RadBuildCtx::max_mixed_tok). `pass_step`: --prefill-span-pass
     * step, so the routed block's pass is the whole step up to 65536 slot rows (`pass_cap` in
     * declare) and each expert's weight is read once a pass. A sizing declare takes both from the real one. */
    int64_t piece_rows = 0;
    bool span_on = false, pass_step = false;
    /* Engram */
    rad_buf eids = 0;              /* [E·T, heads] i32, host, PERSIST */
    int64_t max_spec = 0;          /* drafts a verify step may carry (RadBuildCtx::max_spec) */
    rad_buf eids_ahead = 0;        /* [E·T, heads] i32, host: the next prefill chunk's ids (read-ahead) */
    rad_buf erows = 0;             /* [T, heads·d] host */
    rad_buf erows_dev = 0;         /* [T, heads·d] */
    rad_buf enorm = 0;             /* [T, heads·d] */
    rad_buf ev = 0;                /* [T, n] */
    rad_buf ek = 0;                /* [T, hc·n] */
    rad_weight w_map = 0, w_mult = 0, w_vsz = 0, w_off = 0;
    rad_op op_ids = 0;
    /* FFN */
    rad_buf gu = 0;                /* [T, 2·ff_dense] dense MLP */
    ActFP8  ff{};                  /* [T, ff_dense] */
    rad_buf gu_s = 0;              /* [T, 2·ff_exp] shared expert */
    ActFP8  ff_s{};                /* [T, ff_exp] */
    ActFP8  latm{};                /* [T, latent] */
    rad_buf routed = 0, sout = 0;  /* [T, n] */
    MoeFP8::Wire mw{};
    /* closing */
    rad_buf hout = 0, xout = 0, xnorm = 0, logits = 0;
    rad_weight w_token = 0, w_norm = 0, w_lm = 0;
    Hc hc_head{};
    rad_op op_embed = 0, op_embed_ar = 0, op_enter = 0, op_gather = 0, op_head_norm = 0, op_logits = 0;
    /* the DFly2 drafter: only if the container carries it and the build speculates (dfly_plan) */
    Dfly dfly{};
    /* vision (vision.h): only if the container carries the tower and the deployment wants media */
    bool media = false;
    /* image and video markers: Engram leaves their rows alone (HF engram_mask); -1 without tower */
    int64_t tok_img = -1, tok_vid = -1;
    Vision vis{};
    MediaRows mrows{};
    std::vector<Layer> layers;
    std::vector<Engram> engram;
    /* Each layer's MoeFP8 sources, probed before the buffers (as in nemotronh): the experts' format
     * decides whether the latent carries E4M3 codes. */
    std::vector<MoeFP8::Src> msrc;
    bool moe_q = false;     /* quantised experts (not bf16): the routed arm reads E4M3 */
    bool moe_rot = false;   /* rotated experts (fwht128): a separate rotated latent */
};

static Model g_model[MAX_SLOTS];
/* Where sizing declares go (RadBuildCtx::shape_probe): the engine declares again at smaller step sizes
 * to size the activation arena's levels, lends what is left to the expert slab and arms the prefill
 * stager. Those declares must not touch the state step() reads, so they fill this model instead, as
 * glm5next's probe_model. Without shape_probe_ok the arena was not lent: decode ran at 10.6 tok/s
 * against 24 with it, on the GPTQ container at tp 2. */
static thread_local Model g_probe;   /* the engine runs sizing declares in parallel */

/* ------------------------------------------------------------------ debug tap */
struct Tap {
    const char* dir = nullptr;
    bool layer_on[256] = {};
};
static Tap g_tap;

static void tap_init() {
    g_tap.dir = std::getenv("HY4_DUMP_DIR");
    if (!g_tap.dir || !*g_tap.dir) { g_tap.dir = nullptr; return; }
    const char* ls = std::getenv("HY4_DUMP_LAYERS");
    if (!ls || !*ls) ls = "0,1,2";
    for (const char* p = ls; *p;) {
        char* e = nullptr;
        const long v = std::strtol(p, &e, 10);
        if (e == p) break;
        if (v >= 0 && v < 256) g_tap.layer_on[v] = true;
        p = (*e == ',') ? e + 1 : e;
    }
}

/* Copies `rows` rows of `cols` elements from `b` and writes them. Only with the tap on. */
static void tap(RadCtx* c, const RadBatch* batch, rad_buf b, int64_t rows, int64_t cols, int esize,
                const char* name, int layer, int64_t row0 = 0, const char* ext = nullptr) {
    if (!g_tap.dir || layer < 0 || layer >= 256 || !g_tap.layer_on[layer]) return;
    char* p = (char*)rad_buf_ptr(c, b);
    if (!p) return;
    p += row0 * cols * esize;
    const int64_t bytes = rows * cols * esize;
    std::string tmp((size_t)bytes, '\0');
    if (rad_memcpy_async(tmp.data(), p, bytes, rad_stream(c)) < 0) return;
    rad_stream_sync(rad_stream(c));
    char path[512];
    std::snprintf(path, sizeof path, "%s/s%d.%s.L%02d.%s", g_tap.dir, batch->step, name, layer,
                  ext ? ext : esize == 4 ? "f32" : "bf16");
    if (FILE* fp = std::fopen(path, "wb")) {
        std::fwrite(tmp.data(), 1, tmp.size(), fp);
        std::fclose(fp);
    }
}

/* The same from a batch pointer (the block table), `bytes` contiguous bytes. */
static void tap_raw(RadCtx* c, const RadBatch* batch, const void* p, int64_t bytes, const char* name, int layer) {
    if (!g_tap.dir || !p || layer < 0 || layer >= 256 || !g_tap.layer_on[layer]) return;
    std::string tmp((size_t)bytes, '\0');
    if (rad_memcpy_async(tmp.data(), p, bytes, rad_stream(c)) < 0) return;
    rad_stream_sync(rad_stream(c));
    char path[512];
    std::snprintf(path, sizeof path, "%s/s%d.%s.L%02d.i32", g_tap.dir, batch->step, name, layer);
    if (FILE* fp = std::fopen(path, "wb")) {
        std::fwrite(tmp.data(), 1, tmp.size(), fp);
        std::fclose(fp);
    }
}

/* ------------------------------------------------------------------ declare */
static int decl_norm(RadBuilder* b, Names& nm, const char* name, const char* src, int64_t n, int l,
                     rad_weight* w) {
    HY_TRY(map_copy(b, name, src));
    *w = decl_w(b, name, RAD_F32, { n }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                l >= 0 ? grp_layer(l) : grp_model());
    (void)nm;
    return *w ? RAD_OK : fail(__LINE__);
}

/* An in-place all-reduce of [rows, n_embd], or 0 with one card (as in nemotronh). `exact` for
 * the embedding's: it sums disjoint rows at the model's input, and a lossy wire would change the
 * text (rad_fp8.h, ArFold::kArExact; rad_glm_tp.h). */
static rad_op op_reduce(RadBuilder* b, const Geom& g, rad_buf x, bool exact = false) {
    if (g.world <= 1) return 0;
    return rw(b, RAD_OP(b, "all_reduce",
                        RAD_PARAMS(RAD_INT("world_size", g.world), RAD_RANGE("numel", g.n_embd, g.max_tok * g.n_embd),
                                   RAD_STR("dtype", g.dtype), RAD_INT("exact", exact ? 1 : g.wire_exact),
                                   RAD_INT("min_bytes", g.wire_min_bytes)),
                        RAD_NOWEIGHTS),
              { x }, { x });
}

static rad_op op_rmsnorm(RadBuilder* b, const Geom& g, int64_t rows, int64_t n, double eps, rad_weight w) {
    return RAD_OP(b, "rmsnorm",
                  RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_F64("eps", eps),
                             RAD_F64("wadd", 0.0), RAD_STR("dtype", g.dtype)),
                  RAD_WEIGHTS(w));
}

/* ---- IQ8R trunk fusions (docs/PERF-OPTIONS-2026-11-21.md, row 2)
 *
 * Under an IQ8R trunk every int8 linear's input was a bf16 producer followed by the linear's own
 * had_quant_act_i8g at group 256. libr4d serves three producers that write that int8 pair on their
 * own pass, at group 256 (its `_iq8r` rows of rmsnorm_had_quant_i8g, gated_had_quant_i8g and
 * gate_had_quant_i8g): one launch less each, and the same bytes. They round the produced value to
 * bf16 first, as the separate op stores it, then rotate each 128-wide half with the same in-register
 * r4d_fwht_wave<128>, take one amax over the 256 columns and round as had_quant_act_iq8r does. The
 * norm keeps rmsnorm_bf16's sum-of-squares partition at the same workgroup width; the gate rounds
 * the sigmoid to bf16 before the multiply, as the separate sigmoid and mul do.
 *
 * A producer fuses only where every linear reading its codes is stored IQ8R (rotated, group 256):
 * the probe below reads the stored format before anything is declared, because the norm is declared
 * before the linears that read it and the planner takes liveness from declaration order. */
static constexpr int64_t kIq8rGroup = 256;

/* Whether `L` (`base`, from `src`) is stored IQ8R. Declares its name map (LinearFP8::probe), so its
 * declare() must follow with the same base. */
static bool probe_iq8r(RadBuilder* b, Names& nm, const Geom& g, LinearFP8& L, const char* base,
                       std::initializer_list<const char*> src) {
    if (L.probe(b, nm, g, base, src) != RAD_OK) return false;
    RadEncoding enc{};
    return weight_enc(b, nm.f("%s.weight", base), &enc) && linear_is_iq8r(enc);
}

/* An input fed by a fusing producer at the trunk's group: `consumer_iq8r` is what probe_iq8r (or
 * the encoding of a row-mapped weight) said, `op` the fused op, which some loaded library must
 * declare (radiance's CPU kernels do not). */
static bool q8_fusable(RadBuilder* b, const Geom& g, const ActFP8& a, int64_t n, bool consumer_iq8r,
                       const char* op) {
    return g.w_i8h && consumer_iq8r && a.q8 && a.s8 && a.g8 == kIq8rGroup && n % kIq8rGroup == 0 &&
           rad_op_declared(b, op);
}

/* The copy of `a` that a linear reading codes a fused producer wrote is declared with: no
 * quantiser of its own (ActFP8::q8_fed), rotated codes (q8_rot). */
static ActFP8 fed(const ActFP8& a) {
    ActFP8 r = a;
    r.q8_fed = r.q8_rot = true;
    return r;
}

/* RMS norm of `src` into a.x plus a's int8 pair, at the pair's group. `src` may be a.x (in place,
 * as the plain norm of q_a's output): the kernel reads the row before its barrier and each lane
 * writes only the columns it read. */
static rad_op op_rmsnorm_q8(RadBuilder* b, const Geom& g, int64_t rows, int64_t n, double eps, rad_weight w,
                            rad_buf src, const ActFP8& a) {
    return rw(b, RAD_OP(b, "rmsnorm_had_quant_i8g",
                        RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_F64("eps", eps),
                                   RAD_INT("group", a.g8), RAD_STR("dtype", g.dtype), RAD_F64("wadd", 0.0)),
                        RAD_WEIGHTS(w)),
              { src }, { a.q8, a.s8, a.x });
}

static void step_rmsnorm_q8(RadCtx* c, rad_op op, int64_t T, rad_buf src, rad_weight w, const ActFP8& a) {
    RAD_ISSUE_N(c, op, T, brows(src, T), RAD_NONE, RAD_W(w), brows(a.q8, T), brows(a.s8, T), brows(a.x, T));
}

static int decl_hc(RadBuilder* b, Model& m, Hc& hc, const char* pre, const char* name, int l, bool head) {
    Names& nm = m.nm;
    const ModelShape& f = m.f;
    const int64_t mix = head ? f.hc : 2 * f.hc, width = f.hc * f.n_embd;
    const auto grp = l >= 0 ? grp_layer(l) : grp_model();
    HY_TRY(map_copy(b, nm.f("%s.fn", name), nm.ckpt("%s_fn", pre)));
    HY_TRY(map_copy(b, nm.f("%s.base", name), nm.ckpt("%s_base", pre)));
    HY_TRY(map_copy(b, nm.f("%s.scale", name), nm.ckpt("%s_scale", pre)));
    hc.fn = decl_w(b, nm.f("%s.fn", name), RAD_F32, { mix, width }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
    hc.base = decl_w(b, nm.f("%s.base", name), RAD_F32, { mix }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
    hc.scale = decl_w(b, nm.f("%s.scale", name), RAD_F32, { head ? 1 : 2 }, RAD_ACCESS_PER_TOKEN,
                      RAD_SHARD_NONE, grp);
    if (!hc.fn || !hc.base || !hc.scale) return fail(__LINE__);
    const int64_t rows = head ? m.g.max_logit_rows : m.g.max_tok;
    hc.op = rw(b, RAD_OP(b, "hy4_hc_pre",
                         RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", f.n_embd), RAD_INT("hc", f.hc),
                                    RAD_INT("head", head ? 1 : 0), RAD_F64("eps", f.eps),
                                    RAD_F64("hc_eps", f.hc_eps), RAD_F64("magnitude", f.hc_mag),
                                    RAD_STR("dtype", m.g.dtype)),
                         RAD_WEIGHTS(hc.fn, hc.base, hc.scale)),
               { head ? m.hout : m.h }, { head ? m.xout : m.xin, m.post });
    return hc.op ? RAD_OK : fail(__LINE__);
}

static rad_op op_hc_post(RadBuilder* b, const Model& m) {
    return rw(b, RAD_OP(b, "hy4_hc_post",
                        RAD_PARAMS(RAD_RANGE("M", 1, m.g.max_tok), RAD_INT("n", m.f.n_embd),
                                   RAD_INT("hc", m.f.hc), RAD_STR("dtype", m.g.dtype)),
                        RAD_NOWEIGHTS),
              { m.h, m.y, m.post }, { m.h });
}

/* The absorbed form's kv_b, `blk.L.attn_kv_b_abs.weight` [Hall·2·256, kv_lora]: per head, the key
 * rows in q's order ([rope | nope]: zeros in the rope part, which q·W must not see, then kv_b's
 * k_nope rows) and after them the value rows. It is `mla_project`'s layout (head_dim 256 in both
 * halves), which libr4d serves on the R9700. The zeros come from `hy4.ceros_rope` [rope, kv_lora],
 * which conv/trim.py writes into the --add constants. Chained RAD_MAP_ROWS maps, 2 heads
 * (6 pieces) per map. */
static const char* const kZerosRope = "hy4.ceros_rope";
static int decl_kvb_abs_map(RadBuilder* b, Model& m, const char* decl, const char* src, int64_t Hall) {
    const ModelShape& f = m.f;
    const int64_t per = f.nope + f.v_dim;
    for (int64_t h0 = 0; h0 < Hall; h0 += 2) {
        RadNameMap mp = map_base(decl, RAD_MAP_ROWS);
        for (int64_t h = h0; h < std::min(h0 + 2, Hall); ++h) {
            mp.src[mp.n_src] = kZerosRope; mp.src_index[mp.n_src] = 0; mp.src_rows[mp.n_src] = f.rope; ++mp.n_src;
            mp.src[mp.n_src] = src; mp.src_index[mp.n_src] = (int)(h * per); mp.src_rows[mp.n_src] = f.nope; ++mp.n_src;
            mp.src[mp.n_src] = src; mp.src_index[mp.n_src] = (int)(h * per + f.nope);
            mp.src_rows[mp.n_src] = f.v_dim; ++mp.n_src;
        }
        const int r = rad_decl_name_map(b, &mp);
        if (r != RAD_OK) return r;
    }
    return RAD_OK;
}
static int decl_kvb_abs_map(RadBuilder* b, Model& m, int l, int64_t Hall) {
    return decl_kvb_abs_map(b, m, m.nm.f("blk.%d.attn_kv_b_abs.weight", l),
                            m.nm.ckpt("model.layers.%d.self_attn.kv_b_proj.weight", l), Hall);
}

/* A trunk layer's site: the model's shared buffers, by layer kind. */
static MlaSite mla_site_trunk(Model& m, const Mla& at, int l) {
    MlaSite s{};
    s.wp = m.nm.ckpt("blk.%d", l);
    s.cp = m.nm.ckpt("model.layers.%d.self_attn", l);
    s.gl = l;
    s.g = at.sparse ? &m.g : &m.gw;
    s.a = &m.a;
    s.qa = &m.qa;
    s.lat = &m.lat;
    s.kva = m.kva;
    s.y = m.y;
    s.B = &m.ab[at.sparse ? 0 : 1];
    s.sel = m.sel;
    s.sequ = m.sequ;
    s.sel_rows = m.sel_rows;
    return s;
}

static int decl_mla(RadBuilder* b, Model& m, Mla& at, const MlaSite& s) {
    const AttnBufs& B = *s.B;
    Names& nm = m.nm;
    const ModelShape& f = m.f;
    const Geom& g = *s.g;
    const int64_t T = g.max_tok, H = at.H, qk = f.nope + f.rope;
    const char* pre = s.cp;
    const char* wp = s.wp;
    const int l = s.gl;
    const auto grp = grp_layer(l);
    /* The drafter serves only the absorbed selection form (bkeys exist in no other). */
    if (s.drafter && (!m.mla_abs || !at.sparse)) return fail(__LINE__);

    /* s.a_fed: the input norm wrote the pair (Layer::in_q8), and q_a only reads it. */
    const ActFP8 a_in = s.a_fed ? fed(*s.a) : *s.a;
    HY_TRY(at.q_a.declare(b, nm, g, nm.f("%s.attn_q_a", wp), f.q_lora, f.n_embd, RAD_SHARD_NONE,
                                grp, { nm.ckpt("%s.q_a_proj.weight", pre) }, a_in, s.qa->x));
    /* Under an IQ8R trunk q_a's own had_quant_act_i8g writes the int8 pair of the normed input,
     * and the gate, which reads the same rows unchanged (nothing in between writes `a`), reads
     * that pair as fed instead of quantising them again into the same buffers: the same codes in
     * one launch less a layer (as museglimmer_ct's q/gate/k/v). Only where q_a quantises rotated
     * itself or the input norm wrote the pair, so the gate never reads codes nobody wrote; not for
     * the drafter, whose site keeps the declarations it had. */
    ActFP8 a_q = a_in;
    if (at.q_a.i8 && at.q_a.i8_rot && at.q_a.op_q) a_q = fed(*s.a);
    HY_TRY(decl_norm(b, nm, nm.f("%s.attn_q_a_norm.weight", wp), nm.ckpt("%s.q_a_layernorm.weight", pre),
                           f.q_lora, l, &at.w_qa_norm));

    /* q_b with each head's rows reordered to [rope | nope] (docs/DESIGN.md, «Why the rope
     * goes in front»): chained RAD_MAP_ROWS maps, 8 pieces per map, 2 per head. */
    {
        const char* decl = nm.f("%s.attn_q_b.weight", wp);
        const char* src = nm.ckpt("%s.q_b_proj.weight", pre);
        /* The map is the whole logical weight's (all heads); with tp, each card takes its slice
         * of heads by rows (RAD_SHARD_ROW). */
        for (int64_t h0 = 0; h0 < at.Hall; h0 += 4) {
            RadNameMap mp = map_base(decl, RAD_MAP_ROWS);
            for (int64_t h = h0; h < std::min(h0 + 4, at.Hall); ++h) {
                mp.src[mp.n_src] = src; mp.src_index[mp.n_src] = (int)(h * qk + f.nope);
                mp.src_rows[mp.n_src] = f.rope; ++mp.n_src;
                mp.src[mp.n_src] = src; mp.src_index[mp.n_src] = (int)(h * qk);
                mp.src_rows[mp.n_src] = f.nope; ++mp.n_src;
            }
            HY_TRY(rad_decl_name_map(b, &mp));
        }
        /* With q_b stored IQ8R, q_a_norm writes q_b's int8 pair (op_rmsnorm_q8), in place in qa as
         * the plain norm; the indexer's iq reads the bf16 plane, which it still writes. The map is
         * declared above, so the format is known before the norm, which stays where it was. */
        RadEncoding enc{};
        at.qa_q8 = !s.drafter && g.w_i8h && q8_fusable(b, g, *s.qa, f.q_lora,
                                                       weight_enc(b, decl, &enc) && linear_is_iq8r(enc),
                                                       "rmsnorm_had_quant_i8g");
    }
    /* A library may declare the fused op's name without a row for these params (another group):
     * then the plain norm, and q_b quantises its input itself. */
    at.op_qa_norm = at.qa_q8 ? op_rmsnorm_q8(b, g, T, f.q_lora, f.eps, at.w_qa_norm, s.qa->x, *s.qa) : 0;
    if (!at.op_qa_norm) {
        at.qa_q8 = false;
        at.op_qa_norm = rw(b, op_rmsnorm(b, g, T, f.q_lora, f.eps, at.w_qa_norm), { s.qa->x }, { s.qa->x });
    }
    if (!at.op_qa_norm) return fail(__LINE__);
    at.q_b.mapped = true;  /* the row map above is already this weight's */
    HY_TRY(at.q_b.declare(b, nm, g, nm.f("%s.attn_q_b", wp), H * qk, f.q_lora, RAD_SHARD_ROW,
                                grp, { nm.ckpt("%s.q_b_proj.weight", pre) }, at.qa_q8 ? fed(*s.qa) : *s.qa, B.q));
    HY_TRY(at.kv_a.declare(b, nm, g, nm.f("%s.attn_kv_a", wp), f.kv_lora + f.rope, f.n_embd,
                                 RAD_SHARD_NONE, grp, { nm.ckpt("%s.kv_a_proj_with_mqa.weight", pre) }, *s.a,
                                 s.kva));
    HY_TRY(decl_norm(b, nm, nm.f("%s.attn_kv_a_norm.weight", wp),
                           nm.ckpt("%s.kv_a_layernorm.weight", pre), f.kv_lora, l, &at.w_kv_norm));
    at.op_kv_norm = m.mla_abs ? rw(b, op_rmsnorm(b, g, T, f.kv_lora, f.eps, at.w_kv_norm), { s.kva }, { s.kva })
                              : rw(b, op_rmsnorm(b, g, T, f.kv_lora, f.eps, at.w_kv_norm), { s.kva }, { s.lat->x });
    const char* rop = m.media ? "hy4_rope" : "rope";
    at.op_rope_q = rw(b, decl_rope(b, g, T, qk, H, 0, f.rope, "gptj", rop), { B.q }, { B.q });
    at.op_rope_k = rw(b, decl_rope(b, g, T, f.rope, 1, 0, f.rope, "gptj", rop), { s.kva }, { s.kva });
    if (!s.drafter && (!m.mla_abs || m.mla_both))
        HY_TRY(at.kv_b.declare(b, nm, g, nm.f("%s.attn_kv_b", wp), H * (f.nope + f.v_dim), f.kv_lora,
                                     RAD_SHARD_ROW, grp, { nm.ckpt("%s.kv_b_proj.weight", pre) }, *s.lat, B.kvx));
    if (!m.mla_abs)
        at.op_kv = rw(b, RAD_OP(b, "hy4_mla_kv",
                                RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("heads", H), RAD_INT("nope", f.nope),
                                           RAD_INT("rope", f.rope), RAD_INT("v_dim", f.v_dim),
                                           RAD_STR("dtype", g.dtype)),
                                RAD_NOWEIGHTS),
                      { B.kvx, s.kva }, { B.kbuf, B.vbuf });
    if (m.mla_abs) {
        /* q (absorbed) = q_nope · W_k per head; the latent output is expanded with W_v (mla_project,
         * head_dim 256: q's rope part multiplies the zero rows). */
        at.w_kvb_abs = decl_w(b, nm.f("%s.attn_kv_b_abs.weight", wp), RAD_BF16, { H * 2 * qk, f.kv_lora },
                              RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp);
        if (!at.w_kvb_abs || f.v_dim != qk) return fail(__LINE__);
        auto proj = [&](const char* mode, const char* op = "mla_project") {
            return RAD_OP(b, op,
                          RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("heads", H), RAD_INT("head_dim", qk),
                                     RAD_INT("latent_dim", f.kv_lora), RAD_STR("mode", mode), RAD_STR("dtype", "bf16")),
                          RAD_WEIGHTS(at.w_kvb_abs));
        };
        at.op_absorb = rw(b, proj("absorb"), { B.q }, { B.qlat });
        at.op_expand = rw(b, proj("expand"), { B.olat }, { B.o.x });
        /* Prefill takes hy4k's matrix-core row of the same projection (hy4_mla_project): libr4d's has
         * no WMMA path and was 17% of an 18K prefill. Decode keeps libr4d's, batch-invariant up to 32
         * rows; the two give a row different bits, as the attention's forms 2 and 3 do. */
        at.op_absorb_m = rw(b, proj("absorb", "hy4_mla_project"), { B.q }, { B.qlat });
        at.op_expand_m = rw(b, proj("expand", "hy4_mla_project"), { B.olat }, { B.o.x });
        if (!at.op_absorb || !at.op_expand || !at.op_absorb_m || !at.op_expand_m) return fail(__LINE__);
    }
    HY_TRY(map_copy(b, nm.f("%s.attn_sinks", wp), nm.ckpt("%s.learnable_sink_param", pre)));
    at.w_sink = decl_w(b, nm.f("%s.attn_sinks", wp), RAD_F32, { H }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_ROW, grp);
    if (!at.w_sink) return fail(__LINE__);

    /* Sparse attention pages its group 4 at a time (the indexer's group), so the dense attention
     * of those layers declares block_size 4 too. */
    /* Pages of 4 in both groups, on purpose. A group fixes its block_size with the first op
     * declared after it that carries one: for kv_full and kv_win that is layer 0's sparse
     * attention (rad_kv_block_size does not answer before that op; in kv_store, further down, it
     * does). The sparse layers need 4 (one page per indexer group). The window layers could go
     * to 16 by declaring kv_win right before their first attention, but hy4k's card row
     * (dev_attn.hip) serves any page and libr4d serves head_dim 256 with sink at none. */
    const int64_t bs = f.idx_ratio;
    auto attn = [&](int64_t qlen, int64_t seqs) {
        return RAD_OP(b, "attn_paged_sink",
                      RAD_PARAMS(RAD_RANGE("q_len", 1, qlen), RAD_INT("head_dim", qk), RAD_INT("gqa", 1),
                                 RAD_INT("causal", 1), RAD_INT("window", at.sparse ? 0 : f.window),
                                 RAD_INT("n_head", H), RAD_INT("block_size", bs),
                                 RAD_INT("max_seqs", seqs), RAD_INT("max_ctx", g.max_ctx),
                                 RAD_STR("q_dtype", g.dtype), RAD_STR("kv_dtype", g.kv_dtype),
                                 RAD_F64("scale", 1.0 / std::sqrt((double)qk))),
                      RAD_WEIGHTS(at.w_sink));
    };
    /* The absorbed form: hy4_mla_attn over the latent cache; q's rope is read from B.q uncopied. */
    auto attn_lat = [&](int64_t rows, int64_t seqs) {
        return RAD_OP(b, "hy4_mla_attn",
                      RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("heads", H), RAD_INT("latent", f.kv_lora),
                                 RAD_INT("rope", f.rope), RAD_INT("block_size", bs),
                                 RAD_INT("window", at.sparse ? 0 : f.window),
                                 RAD_F64("scale", 1.0 / std::sqrt((double)qk)), RAD_INT("rope_pitch", qk),
                                 RAD_INT("max_seqs", seqs), RAD_INT("max_ctx", g.max_ctx), RAD_STR("dtype", g.dtype)),
                      RAD_WEIGHTS(at.w_sink));
    };
    const int64_t lat_hd = (f.kv_lora + f.rope) / 2;   /* row [lat | k_rope] in the K, V halves */
    if (s.drafter) {
        /* Selection form only, over its trunk layer's kv_full, which it never writes: s.sel holds
         * every drafter layer's history (seqused | pages, as the ring stores them) and s.kva is the
         * layer's bkeys. One launch over the pass's block rows. */
        at.op_attn_sel = rw(b, attn_lat(T, g.max_seqs), { B.qlat, B.q, s.sel, s.kva }, { B.olat });
        if (!at.op_attn_sel) return fail(__LINE__);
    } else if (m.mla_abs) {
        at.op_attn = rw(b, attn_lat(T, g.max_seqs), { B.qlat, B.q }, { B.olat });
        at.op_store = rw(b, RAD_OP(b, "kv_store",
                                   RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", lat_hd), RAD_INT("n_head_kv", 1),
                                              RAD_INT("block_size", rad_kv_block_size(b, at.kv)),
                                              RAD_STR("kv_dtype", g.kv_dtype)),
                                   RAD_NOWEIGHTS),
                         { s.kva, s.kva }, {});
        if (at.sparse) at.op_attn_sel = rw(b, attn_lat(s.sel_rows, s.sel_rows), { B.qlat, B.q, s.sel, s.sequ }, { B.olat });
    } else {
        at.op_attn = rw(b, attn(T, g.max_seqs), { B.q }, { B.o.x });
        at.op_store = rw(b, RAD_OP(b, "kv_store",
                                   RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("head_dim", qk), RAD_INT("n_head_kv", H),
                                              RAD_INT("block_size", rad_kv_block_size(b, at.kv)),
                                              RAD_STR("kv_dtype", g.kv_dtype)),
                                   RAD_NOWEIGHTS),
                         { B.kbuf, B.vbuf }, {});
        if (at.sparse)  /* one query per row: q_len 1 and as many "sequences" as rows */
            at.op_attn_sel = rw(b, attn(1, s.sel_rows), { B.q, s.sel, s.sequ }, { B.o.x });
    }

    HY_TRY(at.gate.declare(b, nm, g, nm.f("%s.attn_gate", wp), H * f.v_dim, f.n_embd, RAD_SHARD_ROW,
                                 grp, { nm.ckpt("%s.linear_gate.weight", pre) }, a_q, B.gt));
    /* With o_proj stored IQ8R, sigmoid(gate) * o and o_proj's quantiser are one launch over
     * (token, head) rows of v_dim (gate_had_quant_i8g, as rad_block_attn_gated_fp8.h): the [T * H,
     * v_dim / 256] scale plane is the [T, H * v_dim / 256] o_proj reads. Nothing reads the bf16
     * product, so it is not written. The gate is rounded to bf16 before the multiply, as the separate
     * sigmoid stores it, and the product is commutative: the same bytes as the three launches. */
    bool gate_q8 = !s.drafter && g.w_i8h &&
                         q8_fusable(b, g, B.o, f.v_dim,
                                    probe_iq8r(b, nm, g, at.o, nm.f("%s.attn_output", wp),
                                               { nm.ckpt("%s.o_proj.weight", pre) }),
                                    "gate_had_quant_i8g");
    if (gate_q8) {
        at.op_gq = rw(b, RAD_OP(b, "gate_had_quant_i8g",
                                RAD_PARAMS(RAD_RANGE("M", 1, T * H), RAD_INT("n", f.v_dim), RAD_INT("group", B.o.g8),
                                           RAD_STR("dtype", g.dtype), RAD_STR("act", "sigmoid")),
                                RAD_NOWEIGHTS),
                      { B.gt, B.o.x }, { B.o.q8, B.o.s8 });
        /* declared by name but no row for these params: the three separate launches */
        if (!at.op_gq) gate_q8 = false;
    }
    if (!gate_q8) {
        at.op_sig = rw(b, RAD_OP(b, "sigmoid", RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", H * f.v_dim),
                                                         RAD_STR("dtype", g.dtype)), RAD_NOWEIGHTS),
                       { B.gt }, { B.gt });
        at.op_mul = rw(b, RAD_OP(b, "mul", RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", H * f.v_dim),
                                                     RAD_STR("dtype", g.dtype)), RAD_NOWEIGHTS),
                       { B.o.x, B.gt }, { B.o.x });
    }
    HY_TRY(at.o.declare(b, nm, g, nm.f("%s.attn_output", wp), f.n_embd, H * f.v_dim, RAD_SHARD_COL,
                              grp, { nm.ckpt("%s.o_proj.weight", pre) }, gate_q8 ? fed(B.o) : B.o, s.y));
    /* With tp, each card holds the partial sum of its heads: they are summed before the norm. */
    at.op_ar = op_reduce(b, g, s.y);
    if (g.world > 1 && !at.op_ar) return fail(__LINE__);
    HY_TRY(decl_norm(b, nm, nm.f("%s.attn_post_norm.weight", wp), nm.ckpt("%s.post_proj_norm.weight", pre),
                           f.n_embd, l, &at.w_post));
    at.op_post = rw(b, op_rmsnorm(b, g, T, f.n_embd, f.eps, at.w_post), { s.y }, { s.y });

    if (!at.sparse || s.drafter) return RAD_OK;

    /* ---- indexer */
    const char* ip = nm.ckpt("%s.compressed_indexer", pre);
    const int64_t n = f.idx_dim;
    HY_TRY(at.tk.declare(b, nm, g, nm.f("blk.%d.idx_tk", l), n, f.n_embd, RAD_SHARD_NONE, grp,
                               { nm.ckpt("%s.token_key_projection.weight", ip) }, m.a, m.tkn.x));
    HY_TRY(decl_norm(b, nm, nm.f("blk.%d.idx_tk_norm.weight", l), nm.ckpt("%s.token_key_norm.weight", ip), n,
                           l, &at.w_tk_norm));
    at.op_tk_norm = rw(b, op_rmsnorm(b, g, T, n, f.eps, at.w_tk_norm), { m.tkn.x }, { m.tkn.x });
    HY_TRY(rad_bind_layer_kv(b, l, m.kv_tail));
    at.op_work = rw(b, RAD_OP(b, "qsa_work",
                              RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", n), RAD_INT("ratio", f.idx_ratio),
                                         RAD_INT("ring", f.idx_ratio + m.max_spec), RAD_INT("seqs", g.max_seqs),
                                         RAD_INT("work", m.max_work), RAD_STR("dtype", g.dtype)),
                              RAD_NOWEIGHTS),
                    { m.tkn.x }, { m.stage, m.page, m.bpos, m.nc });
    HY_TRY(map_copy(b, nm.f("blk.%d.idx_value.weight", l), nm.ckpt("%s.key_value_projection.weight", ip)));
    HY_TRY(map_copy(b, nm.f("blk.%d.idx_gate.weight", l), nm.ckpt("%s.key_gate_projection.weight", ip)));
    HY_TRY(map_copy(b, nm.f("blk.%d.idx_ape", l), nm.ckpt("%s.intra_group_ape", ip)));
    at.w_ival = decl_w(b, nm.f("blk.%d.idx_value.weight", l), RAD_F32, { n, n }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
    at.w_igate = decl_w(b, nm.f("blk.%d.idx_gate.weight", l), RAD_F32, { n, n }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
    at.w_iape = decl_w(b, nm.f("blk.%d.idx_ape", l), RAD_F32, { f.idx_ratio, n }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
    HY_TRY(decl_norm(b, nm, nm.f("blk.%d.idx_key_norm.weight", l), nm.ckpt("%s.key_norm.weight", ip), n, l,
                           &at.w_inorm));
    if (!at.w_ival || !at.w_igate || !at.w_iape) return fail(__LINE__);
    /* The group-key group, declared right before the op that fixes its block_size (ratio), like
     * QSA's (rad_qsa.h). One head and head_dim = n / (2·ratio): each page holds exactly one key
     * of n values. All the sparse layers share it. */
    if (!m.kv_idx) {
        if (n % (2 * f.idx_ratio) != 0) return fail(__LINE__);
        RadKVGroupDecl d{};
        d.kind = RAD_KV_FULL; d.dtype = RAD_BF16; d.n_head_kv = 1; d.head_dim = n / (2 * f.idx_ratio);
        m.kv_idx = rad_decl_kv_group(b, nm.f("kv_idx_keys"), &d);
        if (!m.kv_idx) return fail(__LINE__);
        /* The score reads every closed group of the history each step: the group is not chunked. */
        HY_TRY(rad_kv_group_dense_read(b, m.kv_idx));
    }
    /* `sections` (3-D positions) only with the tower: without it the op is declared as before */
    at.op_pool = rw(b, m.media
        ? RAD_OP(b, "hy4_idx_pool",
                 RAD_PARAMS(RAD_RANGE("M", 1, m.max_work), RAD_INT("n", n), RAD_INT("ratio", f.idx_ratio),
                            RAD_INT("rotary_dim", f.rope), RAD_INT("block_size", f.idx_ratio),
                            RAD_F64("theta", f.theta_sparse), RAD_F64("eps", f.eps), RAD_STR("dtype", g.dtype),
                            RAD_STR("sections", m.g.rope_sections)),
                 RAD_WEIGHTS(at.w_ival, at.w_igate, at.w_iape, at.w_inorm))
        : RAD_OP(b, "hy4_idx_pool",
                 RAD_PARAMS(RAD_RANGE("M", 1, m.max_work), RAD_INT("n", n), RAD_INT("ratio", f.idx_ratio),
                            RAD_INT("rotary_dim", f.rope), RAD_INT("block_size", f.idx_ratio),
                            RAD_F64("theta", f.theta_sparse), RAD_F64("eps", f.eps), RAD_STR("dtype", g.dtype)),
                 RAD_WEIGHTS(at.w_ival, at.w_igate, at.w_iape, at.w_inorm)),
                    { m.stage, m.page, m.bpos }, {});
    HY_TRY(rad_bind_layer_kv(b, l, m.kv_idx));
    at.op_tail = rw(b, RAD_OP(b, "qsa_tail_store",
                              RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", n), RAD_INT("ratio", f.idx_ratio),
                                         RAD_INT("ring", f.idx_ratio + m.max_spec), RAD_INT("seqs", g.max_seqs),
                                         RAD_STR("dtype", g.dtype)),
                              RAD_NOWEIGHTS),
                    { m.tkn.x }, {});
    HY_TRY(at.iq.declare(b, nm, g, nm.f("blk.%d.idx_q", l), f.idx_heads * n, f.q_lora, RAD_SHARD_NONE, grp,
                               { nm.ckpt("%s.query_projection.weight", ip) }, m.qa, m.iq));
    HY_TRY(at.ihw.declare(b, nm, g, nm.f("blk.%d.idx_hw", l), f.idx_heads, f.n_embd, RAD_SHARD_NONE, grp,
                                { nm.ckpt("%s.head_weight_projection.weight", ip) }, m.a, m.ihw));
    at.op_score = rw(b, m.media
        ? RAD_OP(b, "hy4_idx_score",
                 RAD_PARAMS(RAD_RANGE("M", 1, m.plane_rows), RAD_INT("n", n), RAD_INT("heads", f.idx_heads),
                            RAD_INT("rotary_dim", f.rope), RAD_F64("theta", f.theta_sparse),
                            RAD_INT("blocks", m.max_groups), RAD_INT("seqs", g.max_seqs), RAD_STR("dtype", g.dtype),
                            RAD_STR("sections", m.g.rope_sections)),
                 RAD_NOWEIGHTS)
        : RAD_OP(b, "hy4_idx_score",
                 RAD_PARAMS(RAD_RANGE("M", 1, m.plane_rows), RAD_INT("n", n), RAD_INT("heads", f.idx_heads),
                            RAD_INT("rotary_dim", f.rope), RAD_F64("theta", f.theta_sparse),
                            RAD_INT("blocks", m.max_groups), RAD_INT("seqs", g.max_seqs), RAD_STR("dtype", g.dtype)),
                 RAD_NOWEIGHTS),
                     { m.iq, m.ihw, m.nc }, { m.score });
    at.op_select = rw(b, RAD_OP(b, "hy4_idx_select",
                                RAD_PARAMS(RAD_RANGE("M", 1, m.plane_rows), RAD_INT("blocks", m.max_groups),
                                           RAD_INT("topk", f.idx_topk_groups), RAD_INT("ratio", f.idx_ratio),
                                           RAD_INT("seqs", g.max_seqs)),
                                RAD_NOWEIGHTS),
                      { m.score, m.nc }, { m.sel, m.nsel, m.sequ });
    return at.op_tk_norm && at.op_work && at.op_pool && at.op_tail && at.op_score && at.op_select
               ? RAD_OK : fail(__LINE__);
}

static const char* const kEngramCodesKey = "hy4.engram_codes";
static const char* const kEngramCodesRowmax = "rowmax448";

static int decl_engram(RadBuilder* b, Model& m, Engram& en) {
    Names& nm = m.nm;
    const ModelShape& f = m.f;
    const Geom& g = m.g;
    const int l = en.layer, T = (int)g.max_tok;
    const int64_t H = f.engram_heads, d = f.engram_d, w = H * d, n = f.n_embd;
    const char* pre = nm.ckpt("model.layers.%d.engram_layer", l);
    const auto grp = grp_layer(l);

    /* The table: a model weight (grp_model) that only embed_lookup_q reads, so that the core treats
     * it as an n-gram table and applies ngram-placement to it (core/place/planner.cpp:305-307).
     *
     * Two forms. The checkpoint's bf16 table (31.7 GB a layer), or the E4M3 codes Engram reads it
     * as: conv/engram_codes.py writes each row's codes under the row's own scale (amax/448, the
     * runtime round trip below) as `<pre>...embedding.shard_0.weight` with one bf16 scale of 1 in
     * `<pre>...embedding.weight_scale`, which rad-convert keeps byte for byte as
     * `fp8_e4m3*bf16[*x*]` (a sharded table under one scale, core/format/checkpoint.cpp) and
     * embed_lookup_q serves with no change. Dropping the row scales is exact: the round trip is
     * equivariant to a row's scale and the per-head RMSNorm after it cancels it, so a row of codes
     * gives today's rows to the eps (conv/engram_codes.py verifies this), in half the bytes. The
     * codes come first, so a conversion that is given them (`--add`) takes them and one that is
     * not reads the bf16 table as before. */
    const char* tn = nm.f("blk.%d.engram_table.weight", l);
    HY_TRY(map_copy_alt(b, tn, nm.ckpt("%s.multi_head_embedding.embedding.shard_0.weight", pre),
                        nm.ckpt("%s.multi_head_embedding.embedding.weight", pre)));
    RadEncoding enc{};
    int64_t shp[RAD_MAX_RANK] = {};
    uint32_t rk = 0;
    int64_t rows = 0;
    if (rad_weight_encoding(b, tn, &enc, shp, &rk) == RAD_OK && rk == 2)
        rows = shp[0];
    if (rows <= 0) {
        std::fprintf(stderr, "radiance: hyv4flash: the size of layer %d's Engram table is unknown\n", l);
        return fail(__LINE__);
    }
    /* The codes path needs both the container key and the shipped encoding: E4M3 codes under one bf16
     * scale for the whole table. The key is what says the codes are conv/engram_codes.py's (each
     * row's amax/448, the row scales dropped): an E4M3 table without it is some other quantisation
     * of the bf16 table (one scale for all of it, say), which the round trip below would then round
     * a second time, so it is refused, as is the key over a table in another form. Without the key
     * and with no E4M3 table the table is declared bf16 as before, so a bf16 container declares
     * exactly what it did, and a third form is refused by the kernels' layout hooks as it was. */
    const RadEncPlane* tsc = rad_enc_plane(&enc, "scale");
    const bool f8 = enc.n_planes >= 1 && enc.plane[0].dtype == RAD_F8E4M3;
    const bool e4m3 = f8 && rad_enc_is(&enc, "affine") && enc.n_planes == 2 && tsc && tsc->dtype == RAD_BF16 &&
                      tsc->block[0] == 0 && tsc->block[1] == 0;
    if (m.engram_codes != e4m3 || (f8 && !e4m3)) {
        if (m.engram_codes)
            std::fprintf(stderr, "radiance: hyv4flash: the container says %s=%s, but layer %d's Engram table is "
                         "not E4M3 codes under one bf16 scale\n", kEngramCodesKey, kEngramCodesRowmax, l);
        else
            std::fprintf(stderr, "radiance: hyv4flash: layer %d's Engram table is E4M3, but the container has no "
                         "%s=%s: these are not the codes Engram reads (conv/engram_codes.py), so they are not "
                         "served. A container converted with those codes takes the key in place "
                         "(deploy/set-engram-key.sh)\n", l, kEngramCodesKey, kEngramCodesRowmax);
        return fail(__LINE__);
    }
    if (e4m3) {
        en.w_table = decl_view(b, tn, nullptr, "codes", RAD_F8E4M3, { rows, d }, RAD_ACCESS_VOCAB, RAD_SHARD_NONE,
                               grp_model());
        en.w_table_scale = decl_view(b, nm.f("blk.%d.engram_table.scale", l), tn, "scale", RAD_BF16, { 1 },
                                     RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
        if (!en.w_table_scale) return fail(__LINE__);
    } else {
        en.w_table = decl_w(b, tn, RAD_BF16, { rows, d }, RAD_ACCESS_VOCAB, RAD_SHARD_NONE, grp_model());
    }
    /* The 16 per-head norms, together in [H, d]. */
    /* The 16 per-head norms, concatenated: [H·d] (the kernels read it as [H, d]). */
    HY_TRY(map_concat_shards(b, nm.f("blk.%d.engram_norms", l), nm.ckpt("%s.lookup_per_head_norms.", pre), ".weight",
                             (int)H));
    en.w_norms = decl_w(b, nm.f("blk.%d.engram_norms", l), RAD_F32, { H * d }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
    HY_TRY(map_copy(b, nm.f("blk.%d.engram_val.weight", l), nm.ckpt("%s.value_proj.weight", pre)));
    HY_TRY(map_copy(b, nm.f("blk.%d.engram_val.bias", l), nm.ckpt("%s.value_proj.bias", pre)));
    en.w_val = decl_w(b, nm.f("blk.%d.engram_val.weight", l), RAD_BF16, { n, w }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
    en.w_val_b = decl_w(b, nm.f("blk.%d.engram_val.bias", l), RAD_BF16, { n }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
    /* The 4 key_projs as a single [hc·n, w] (and its bias [hc·n]): one GEMM. */
    {
        RadNameMap mp = map_base(nm.f("blk.%d.engram_keys.weight", l), RAD_MAP_CONCAT);
        RadNameMap mb = map_base(nm.f("blk.%d.engram_keys.bias", l), RAD_MAP_CONCAT);
        for (int64_t j = 0; j < f.hc; ++j) {
            mp.src[mp.n_src++] = nm.ckpt("%s.key_projs.%lld.weight", pre, (long long)j);
            mb.src[mb.n_src++] = nm.ckpt("%s.key_projs.%lld.bias", pre, (long long)j);
        }
        HY_TRY(rad_decl_name_map(b, &mp));
        HY_TRY(rad_decl_name_map(b, &mb));
    }
    en.w_keys = decl_w(b, nm.f("blk.%d.engram_keys.weight", l), RAD_BF16, { f.hc * n, w }, RAD_ACCESS_PER_TOKEN,
                       RAD_SHARD_NONE, grp);
    en.w_keys_b = decl_w(b, nm.f("blk.%d.engram_keys.bias", l), RAD_BF16, { f.hc * n }, RAD_ACCESS_PER_TOKEN,
                         RAD_SHARD_NONE, grp);
    auto cat4 = [&](const char* decl, const char* fmt) -> int {
        RadNameMap mp = map_base(decl, RAD_MAP_CONCAT);
        for (int64_t j = 0; j < f.hc; ++j) mp.src[mp.n_src++] = nm.ckpt(fmt, pre, (long long)j);
        return rad_decl_name_map(b, &mp);
    };
    HY_TRY(cat4(nm.f("blk.%d.engram_nk", l), "%s.norm_key.%lld.weight"));
    HY_TRY(cat4(nm.f("blk.%d.engram_nq", l), "%s.norm_query.%lld.weight"));
    HY_TRY(cat4(nm.f("blk.%d.engram_per", l), "%s.unified_per_hc_scales.%lld"));
    HY_TRY(map_copy(b, nm.f("blk.%d.engram_gamma", l), nm.ckpt("%s.unified_pre_gamma", pre)));
    en.w_nk = decl_w(b, nm.f("blk.%d.engram_nk", l), RAD_F32, { f.hc * n }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
    en.w_nq = decl_w(b, nm.f("blk.%d.engram_nq", l), RAD_F32, { f.hc * n }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
    en.w_per = decl_w(b, nm.f("blk.%d.engram_per", l), RAD_F32, { f.hc * n }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
    en.w_gamma = decl_w(b, nm.f("blk.%d.engram_gamma", l), RAD_F32, { n }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp);
    if (!en.w_table || !en.w_norms || !en.w_val || !en.w_val_b || !en.w_keys || !en.w_keys_b || !en.w_nk ||
        !en.w_nq || !en.w_per || !en.w_gamma)
        return fail(__LINE__);

    {
        const rad_weight wt[2] = { en.w_table, en.w_table_scale };   /* the bf16 table has no scale */
        en.op_lookup = rw(b, RAD_OP(b, "embed_lookup_q",
                                    RAD_PARAMS(RAD_RANGE("M", 1, (int64_t)T * H), RAD_INT("n_embd", d),
                                               RAD_INT("n_vocab", rows), RAD_STR("dtype", g.dtype)),
                                    wt, en.w_table_scale ? 2 : 1),
                          { m.eids, m.eids_ahead }, { m.erows });
    }
    en.op_cast = rw(b, RAD_OP(b, "cast", RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", w),
                                                    RAD_STR("from", g.dtype), RAD_STR("to", g.dtype)),
                              RAD_NOWEIGHTS),
                    { m.erows }, { m.erows_dev });
    en.op_norm = rw(b, RAD_OP(b, "hy4_rows_rmsnorm",
                              RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("heads", H), RAD_INT("d", d), RAD_INT("fp8", 1),
                                         RAD_F64("eps", 1.1920929e-07), RAD_STR("dtype", g.dtype)),
                              RAD_WEIGHTS(en.w_norms)),
                    { m.erows_dev }, { m.enorm });
    en.op_val = rw(b, RAD_OP(b, "gemm_nt_bias",
                             RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("N", n), RAD_INT("K", w), RAD_STR("dtype", g.dtype)),
                             RAD_WEIGHTS(en.w_val, en.w_val_b)),
                   { m.enorm }, { m.ev });
    en.op_key = rw(b, RAD_OP(b, "gemm_nt_bias",
                             RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("N", f.hc * n), RAD_INT("K", w),
                                        RAD_STR("dtype", g.dtype)),
                             RAD_WEIGHTS(en.w_keys, en.w_keys_b)),
                   { m.enorm }, { m.ek });
    /* `skip`/`skip2` (image and video markers) only with the tower */
    en.op_gate = rw(b, m.media
        ? RAD_OP(b, "hy4_engram_gate",
                 RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", n), RAD_INT("hc", f.hc), RAD_INT("pad", f.engram_pad),
                            RAD_INT("skip", m.tok_img), RAD_INT("skip2", m.tok_vid), RAD_F64("eps", 1.1920929e-07),
                            RAD_F64("gate_eps", 1e-6), RAD_STR("dtype", g.dtype)),
                 RAD_WEIGHTS(en.w_nk, en.w_nq, en.w_gamma, en.w_per))
        : RAD_OP(b, "hy4_engram_gate",
                 RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", n), RAD_INT("hc", f.hc), RAD_INT("pad", f.engram_pad),
                            RAD_F64("eps", 1.1920929e-07), RAD_F64("gate_eps", 1e-6), RAD_STR("dtype", g.dtype)),
                 RAD_WEIGHTS(en.w_nk, en.w_nq, en.w_gamma, en.w_per)),
                    { m.h, m.ek, m.ev }, { m.h });
    return en.op_lookup && en.op_cast && en.op_norm && en.op_val && en.op_key && en.op_gate ? RAD_OK : fail(__LINE__);
}

static int decl_ffn(RadBuilder* b, Model& m, Ffn& ff, int l) {
    Names& nm = m.nm;
    const ModelShape& f = m.f;
    const Geom& g = m.g;
    const int64_t T = g.max_tok, n = f.n_embd;
    const char* pre = nm.ckpt("model.layers.%d.mlp", l);
    const auto grp = grp_layer(l);
    /* The SiLU product in front of a down projection. With the down projection stored IQ8R it writes
     * that linear's int8 pair on its own pass (gated_had_quant_i8g at group 256: silu(gate) * up
     * rounded to bf16, then had_quant_act_iq8r's rotation and codes) and no bf16 plane, which only the
     * down projection read; the down projection declares no quantiser. Otherwise (and with a bf16
     * trunk) the product is written in bf16 alone, and an int8 down projection quantises that plane
     * itself. `a_down` gets the activation the down projection is declared with. */
    const auto decl_act = [&](const char* down, const char* src, rad_buf gate_up, const ActFP8& a, int64_t w,
                              ActFP8* a_down) -> int {
        const bool q8 = g.w_i8h &&
                        q8_fusable(b, g, a, w, probe_iq8r(b, nm, g, ff.down, down, { src }), "gated_had_quant_i8g");
        if (q8) {
            *a_down = fed(a);
            /* declared by name but unresolved for these params: the bf16 product below */
            if (ff.act.declare(b, g, gate_up, *a_down, w, "silu", 0, false) == RAD_OK) return RAD_OK;
        }
        Geom gact = g;
        gact.w_bf16 = true;
        *a_down = a;
        return ff.act.declare(b, gact, gate_up, a, w);
    };
    /* ff.norm_q8: the FFN's input norm wrote m.a's int8 pair (declare), so its first linear (and the
     * shared expert's gate_up) read it as fed. */
    const ActFP8 a_in = ff.norm_q8 ? fed(m.a) : m.a;
    ActFP8 a_down{};
    if (!ff.moe) {
        HY_TRY(ff.gu.declare(b, nm, g, nm.f("blk.%d.ffn_gate_up", l), 2 * f.ff_dense, n, RAD_SHARD_NONE, grp,
                                   { nm.ckpt("%s.gate_proj.weight", pre), nm.ckpt("%s.up_proj.weight", pre) },
                                   a_in, m.gu, 0, { f.ff_dense, f.ff_dense }));
        const char* dn = nm.f("blk.%d.ffn_down", l);
        const char* ds = nm.ckpt("%s.down_proj.weight", pre);
        HY_TRY(decl_act(dn, ds, m.gu, m.ff, f.ff_dense, &a_down));
        HY_TRY(ff.down.declare(b, nm, g, dn, n, f.ff_dense, RAD_SHARD_NONE, grp, { ds }, a_down, m.y));
        HY_TRY(decl_norm(b, nm, nm.f("blk.%d.ffn_out_norm.weight", l), nm.ckpt("%s.post_fc2_norm.weight", pre), n,
                               l, &ff.w_out_norm));
        ff.op_out_norm = rw(b, op_rmsnorm(b, g, T, n, f.eps, ff.w_out_norm), { m.y }, { m.y });
        return ff.op_out_norm ? RAD_OK : fail(__LINE__);
    }
    /* Latent MoE (like nemotronh's LatentMoe): down to the latent, MoeFP8 over the latent with the
     * router reading the full-width input, back up, plus the shared expert at full width. */
    HY_TRY(ff.lat_in.declare(b, nm, g, nm.f("blk.%d.ffn_latent_in", l), f.latent, n, RAD_SHARD_NONE, grp,
                                   { nm.ckpt("%s.routed_expert_down_proj.weight", pre) }, a_in, m.latm.x));
    /* The shared expert's gate/up reads the same normed input as the latent projection, unchanged
     * by the routed arm in between (the router reads only its bf16 plane): under an IQ8R trunk it
     * reads ffn_latent_in's int8 pair as fed (or the norm's, with ff.norm_q8), as the attention gate
     * reads q_a's (decl_mla). */
    ActFP8 a_lat = a_in;
    if (ff.lat_in.i8 && ff.lat_in.i8_rot && ff.lat_in.op_q) a_lat = fed(m.a);
    MoeFP8::Src& ms = m.msrc[(size_t)l];   /* probed in declare, before the buffers */
    Geom ge = g;
    ge.n_embd = f.latent;
    ge.n_ff_all = f.ff_exp;
    ge.w_bf16 = !m.moe_q;   /* the trunk is BF16; the routed arm reads E4M3 if the experts are not */
    /* With rotated experts, MoeFP8 reads its own rotated latent (`hr`, which it writes itself) and
     * the latent's unrotated codes go unread: they are only declared (MoeFP8 requires the pair). */
    if (m.moe_q && !m.moe_rot) HY_TRY(ff.lat_q.declare(b, ge, m.latm, f.latent));
    MoeFP8::Wire mw = m.mw;
    mw.h = m.latm;
    mw.route = m.a.x;
    HY_TRY(ff.moe_blk.declare(b, nm, ge, l, mw, ms, false, false));
    HY_TRY(ff.lat_out.declare(b, nm, g, nm.f("blk.%d.ffn_latent_out", l), n, f.latent, RAD_SHARD_NONE, grp,
                                    { nm.ckpt("%s.routed_expert_up_proj.weight", pre) }, m.latm, m.routed));
    /* The up projection is linear, without bias: the sum of the partial ups is the up of the sum. */
    ff.op_ar = op_reduce(b, g, m.routed);
    if (g.world > 1 && !ff.op_ar) return fail(__LINE__);
    const char* sp = nm.ckpt("%s.shared_experts", pre);
    HY_TRY(ff.gu.declare(b, nm, g, nm.f("blk.%d.ffn_gate_up_shexp", l), 2 * f.ff_exp, n, RAD_SHARD_NONE, grp,
                               { nm.ckpt("%s.gate_proj.weight", sp), nm.ckpt("%s.up_proj.weight", sp) }, a_lat, m.gu_s, 0,
                               { f.ff_exp, f.ff_exp }));
    {
        const char* dn = nm.f("blk.%d.ffn_down_shexp", l);
        const char* ds = nm.ckpt("%s.down_proj.weight", sp);
        HY_TRY(decl_act(dn, ds, m.gu_s, m.ff_s, f.ff_exp, &a_down));
        HY_TRY(ff.down.declare(b, nm, g, dn, n, f.ff_exp, RAD_SHARD_NONE, grp, { ds }, a_down, m.sout));
    }
    ff.op_join = rw(b, RAD_OP(b, "add", RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", n), RAD_STR("dtype", g.dtype)),
                              RAD_NOWEIGHTS),
                    { m.routed, m.sout }, { m.y });
    HY_TRY(decl_norm(b, nm, nm.f("blk.%d.ffn_out_norm.weight", l), nm.ckpt("%s.combine_output_norm.weight", pre),
                           n, l, &ff.w_out_norm));
    ff.op_out_norm = rw(b, op_rmsnorm(b, g, T, n, f.eps, ff.w_out_norm), { m.y }, { m.y });
    return ff.op_join && ff.op_out_norm ? RAD_OK : fail(__LINE__);
}

/* ------------------------------------------------------------------ DFly2: declare (docs/DFLY2.md) */
/* The block the drafter was trained at: the bonus token and five masks. A shorter block would be an
 * exact prefix (the attention is causal and the steps sequential), but radiance drives a drafter
 * at depth == max_spec, so only a build at max_spec 5 serves it. */
static constexpr int64_t kDflyBlock = 6;

/* A config list of integers (space separated, as read_shape's lists), or `dflt` when absent. */
static std::vector<int> meta_ints(const RadModelMeta* meta, const char* key, std::initializer_list<int> dflt) {
    const char* s = rad_meta_gets(meta, key, nullptr);
    if (!s) {
        char k[192];
        std::snprintf(k, sizeof k, "text_config.%s", key);
        s = rad_meta_gets(meta, k, nullptr);
    }
    if (!s) return std::vector<int>(dflt);
    std::vector<int> v;
    for (const char* p = s; *p;) {
        char* e = nullptr;
        const long x = std::strtol(p, &e, 10);
        if (e == p) { ++p; continue; }
        v.push_back((int)x);
        p = e;
    }
    return v;
}

/* The block's mask token: Hy4-flash's, the one the drafter was trained with. */
static constexpr int32_t kDflyMask = 120009;

/* A drafter config value, under its name or text_config's (as meta_ints), or null when absent. */
static const char* dfly_meta(const RadModelMeta* meta, const char* key) {
    const char* s = rad_meta_gets(meta, key, nullptr);
    if (s) return s;
    char k[192];
    std::snprintf(k, sizeof k, "text_config.%s", key);
    return rad_meta_gets(meta, k, nullptr);
}

/* An integer drafter config value (a boolean reads 1 or 0), `dflt` when absent. A value that is
 * not an integer reads as LLONG_MIN, so a check against the expected value refuses it. */
static long long dfly_meta_geti(const RadModelMeta* meta, const char* key, long long dflt) {
    const char* s = dfly_meta(meta, key);
    if (!s || !*s) return dflt;
    char* e = nullptr;
    const long long v = std::strtoll(s, &e, 0);
    return e != s && *e == 0 ? v : LLONG_MIN;
}

/* The shape of a drafter weight by its declared name. A container answers that name as it is;
 * a checkpoint (converting) only through the name map, which is then declared here, once:
 * decl_dfly does not declare it again (Dfly::premapped). */
static bool dfly_shape(RadBuilder* b, Model& m, const char* decl, const char* ckpt, int64_t* shp,
                       RadEncoding* enc = nullptr) {
    RadEncoding local{};
    RadEncoding& e = enc ? *enc : local;
    uint32_t rk = 0;
    for (int64_t i = 0; i < RAD_MAX_RANK; ++i) shp[i] = 0;
    const char* dn = m.nm.f("%s", decl);
    if (m.dfly.premapped && map_copy(b, dn, ckpt) != RAD_OK) return false;
    return rad_weight_encoding(b, dn, &e, shp, &rk) == RAD_OK && rk >= 1;
}

/* Whether this build serves the DFly2 drafter, filled into m.dfly: the container (or, converting,
 * the checkpoint) carries it, the build speculates at the drafter's depth, and HY4_DFLY is not 0
 * (prompt lookup alone over a container that carries the drafter: radiance refuses --spec-ngram
 * once a drafter is declared). It declares nothing: a build that does not serve the drafter
 * declares exactly what it did before. A drafter the build cannot serve is a note, not a failure:
 * the trunk still serves, by prompt lookup. */
static int dfly_plan(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx, Model& m) {
    Dfly& df = m.dfly;
    df = Dfly{};
    if (ctx->max_spec <= 0) return RAD_OK;
    const char* ev = std::getenv("HY4_DFLY");
    if (ev && ev[0] == '0') return RAD_OK;
    const ModelShape& f = m.f;
    int64_t shp[RAD_MAX_RANK] = {};
    {
        RadEncoding e{};
        df.premapped = !weight_enc(b, m.nm.f("token_embd.weight"), &e);   /* before its map: converting */
    }
    if (!dfly_shape(b, m, "dfly.final_norm.weight", "model.dfly_layers.final_norm.weight", shp)) {
        df = Dfly{};
        return RAD_OK;
    }
    /* A drafter this build cannot serve: the trunk serves without it. Said once a process, on
     * stderr too (declare runs per rank and level). */
    auto skip = [&](const char* why) {
        rad_note(b, "DFly2 drafter in the container, not served: %s", why);
        static std::atomic<bool> said{false};
        if (!ctx->shape_probe && !said.exchange(true))
            std::fprintf(stderr, "hyv4flash: DFly2 drafter in the container, not served: %s\n", why);
        const bool pm = df.premapped;
        df = Dfly{};
        df.premapped = pm;   /* the maps declared stay declared */
        return RAD_OK;
    };
    df.block = kDflyBlock;
    df.depth = kDflyBlock - 1;
    if (ctx->max_spec != df.depth) {
        char w[160];
        std::snprintf(w, sizeof w, "max_spec %d, and it proposes %lld a step (radiance drives a drafter at depth == "
                                   "max_spec)", ctx->max_spec, (long long)df.depth);
        return skip(w);
    }
    /* The drafter config keys that change the graph. The wiring below is the reference's at
     * Hy4-flash's values (docs/DFLY2.md), so a checkpoint that states another value is served
     * without the drafter rather than with a graph that is not its own. A key the config does not
     * state takes the reference's default; a null one is not in the container at all (radiance
     * flattens config.json without its nulls), so `dfly_context_local_window` refuses only when it
     * holds a value. */
    const char* fusion = dfly_meta(meta, "dfly_mtp_fusion");
    if (fusion && std::strcmp(fusion, "slot0"))
        return skip("dfly_mtp_fusion is not slot0 (only slot 0's fusion is wired)");
    if (dfly_meta_geti(meta, "dfly_enable_hidden_correction", 1) == 0)
        return skip("dfly_enable_hidden_correction is off (the correction is wired in)");
    if (dfly_meta(meta, "dfly_context_local_window"))
        return skip("dfly_context_local_window is set (the drafter attends to the whole selection)");
    const long long topk = dfly_meta_geti(meta, "dfly_target_index_topk", f.idx_topk_groups * f.idx_ratio);
    if (topk != f.idx_topk_groups * f.idx_ratio)
        return skip("dfly_target_index_topk is not the trunk's selection (idx_topk_groups * idx_ratio tokens)");
    static const std::vector<int> kTargets = { 25, 29, 33, 37, 41 };
    df.targets = meta_ints(meta, "dfly_target_kv_layers", { 25, 29, 33, 37, 41 });
    if (df.targets != kTargets) return skip("dfly_target_kv_layers is not 25 29 33 37 41");
    /* The reference reads the history of the same layers it reads the cache of. */
    if (meta_ints(meta, "dfly_target_index_layer", { 25, 29, 33, 37, 41 }) != df.targets)
        return skip("dfly_target_index_layer differs from dfly_target_kv_layers");
    df.n_layers = (int64_t)df.targets.size();
    if (dfly_meta_geti(meta, "dfly_num_hidden_layers", df.n_layers) != df.n_layers)
        return skip("dfly_num_hidden_layers is not 5, the target layers' count");
    for (int t : df.targets)
        if (t < 0 || t >= f.n_layer || !f.sparse[(size_t)t]) return skip("a target layer is not a sparse trunk layer");
    /* the block's masks, as declared to radiance (RadDrafterDecl::mask_token) and in docs/DFLY2.md */
    if (dfly_meta_geti(meta, "dfly_mask_token_id", kDflyMask) != kDflyMask) return skip("dfly_mask_token_id is not 120009");
    df.mask = kDflyMask;
    if (df.mask >= f.vocab) return skip("dfly_mask_token_id is not a token");
    /* bkeys exist only in hy4_mla_attn's absorbed selection form */
    if (!m.mla_abs) return skip("the MLA is served expanded (HY4_MLA)");
    /* The draft head is the trunk's lm_head as a bf16 gemm_nt (row_topk reads bf16 logits). */
    if (weight_codes_dtype(b, m.nm.f("output.weight"), RAD_BF16) != RAD_BF16) return skip("lm_head is not bf16");
    if (m.g.world > 1 && !vocab_head_sharded(m.g)) return skip("lm_head does not split evenly by vocabulary");
    if (m.g.n_vocab % 8) return skip("the vocabulary slice is not a multiple of 8 (row_topk)");
    RadEncoding e_ffn{}, e_cor{};
    if (!dfly_shape(b, m, "dfly.0.ffn_down.weight", "model.dfly_layers.0.mlp.down_proj.weight", shp, &e_ffn) ||
        shp[0] != f.n_embd)
        return skip("the FFN's width is unknown");
    df.ffw = shp[1];
    if (!dfly_shape(b, m, "dfly.correction.down.weight", "model.dfly_layers.hidden_correction.down_proj.weight", shp,
                    &e_cor) ||
        shp[0] != f.n_embd)
        return skip("the hidden correction's width is unknown");
    df.hcw = shp[1];
    /* The drafter's format, from what the container (or, converting, the recipe) holds for its
     * linears, as LinearFP8::probe learns a trunk's: IQ8R in any of them makes the drafter an int8
     * one (int8 geometry and input pairs). Per linear, LinearFP8 still serves whichever form it finds
     * stored (attn_kv_a stays bf16, as the trunk's), so a recipe may keep any of them bf16. Every
     * linear is looked at, not one or two: a recipe that quantised only some would otherwise leave
     * the drafter in the bf16 geometry with an int8 weight, a load failure instead of a served model.
     * No container key: unlike the trunk's, no producer here writes codes.
     * Only when serving: converting (premapped), dfly_shape creates the name map of each weight it
     * probes, which decl_dfly then declares again, so there the two probes above (the recipe's
     * format for them) decide. */
    df.i8 = linear_is_iq8r(e_ffn) || linear_is_iq8r(e_cor);
    if (!df.premapped) {
        static const char* const per_layer[][2] = {
            {"attn_q_a", "self_attn.q_a_proj"}, {"attn_q_b", "self_attn.q_b_proj"},
            {"attn_gate", "self_attn.linear_gate"}, {"attn_output", "self_attn.o_proj"},
            {"ffn_gate_up", "mlp.gate_proj"}};
        int64_t shp2[RAD_MAX_RANK];   /* dfly_shape clears RAD_MAX_RANK entries */
        for (int64_t d = 0; d < df.n_layers && !df.i8; ++d)
            for (const auto& w : per_layer) {
                RadEncoding e{};
                if (dfly_shape(b, m, m.nm.f("dfly.%lld.%s.weight", (long long)d, w[0]),
                               m.nm.ckpt("model.dfly_layers.%lld.%s.weight", (long long)d, w[1]), shp2, &e) &&
                    linear_is_iq8r(e)) { df.i8 = true; break; }
            }
        static const char* const once[][2] = {
            {"dfly.fusion.eh.weight", "model.dfly_layers.mtp_fusion.eh_proj.weight"},
            {"dfly.correction.gate_up.weight", "model.dfly_layers.hidden_correction.gate_proj.weight"}};
        for (const auto& w : once) {
            RadEncoding e{};
            if (!df.i8 && dfly_shape(b, m, w[0], w[1], shp2, &e) && linear_is_iq8r(e)) df.i8 = true;
        }
    }
    df.entry = 1 + f.n_embd + df.n_layers * (1 + f.idx_topk_groups);
    df.rows = std::min<int64_t>(m.g.max_seqs * df.block, m.g.max_tok);
    if (df.rows < df.block) return skip("max_tok holds no block");
    df.layers.resize((size_t)df.n_layers);
    for (size_t d = 0; d < df.layers.size(); ++d) df.layers[d].target = df.targets[d];
    df.on = true;
    return RAD_OK;
}

/* hy4_ring_store, at the trunk's logit rows. */
static rad_op op_ring_store(RadBuilder* b, const Model& m, int64_t width, int64_t offset, const char* mode) {
    const Geom& g = m.g;
    const int64_t ring = m.dfly.depth + 1;
    if (!std::strcmp(mode, "dense"))
        return RAD_OP(b, "hy4_ring_store",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("ring", ring), RAD_INT("width", width),
                                 RAD_INT("offset", offset), RAD_INT("seqs", g.max_seqs), RAD_STR("mode", mode),
                                 RAD_INT("block", m.f.idx_ratio)),
                      RAD_NOWEIGHTS);
    return RAD_OP(b, "hy4_ring_store",
                  RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("ring", ring), RAD_INT("width", width),
                             RAD_INT("offset", offset), RAD_INT("seqs", g.max_seqs), RAD_STR("mode", mode)),
                  RAD_NOWEIGHTS);
}

/* The ring words of drafter layer d's history, after trunk layer t_d's selection. */
static int decl_dfly_store(RadBuilder* b, Model& m, int d) {
    DflyLayer& D = m.dfly.layers[(size_t)d];
    const int64_t K = m.f.idx_topk_groups, off = 1 + m.f.n_embd + d * (1 + K);
    D.op_st_sequ = rw(b, op_ring_store(b, m, 1, off, "row"), { m.sequ }, {});
    D.op_st_sel = rw(b, op_ring_store(b, m, K, off + 1, "row"), { m.sel }, {});
    D.op_st_dense = rw(b, op_ring_store(b, m, 1 + K, off, "dense"), {}, {});
    return D.op_st_sequ && D.op_st_sel && D.op_st_dense ? RAD_OK : fail(__LINE__);
}

/* A bf16 drafter's hidden-correction linear: replicated, its weight alone (each draft step declares
 * its own `gemm_nt`, dfly_relaunch). The container must hold it bf16. */
static rad_weight dfly_weight(RadBuilder* b, Names& nm, const char* name, std::initializer_list<const char*> src,
                              int64_t N, int64_t K, int gl, bool mapped = false) {
    if (!mapped && map_linear(b, name, src) != RAD_OK) return 0;
    RadEncoding e{};
    if (weight_enc(b, name, &e) && !(rad_enc_is(&e, "plain") && e.n_planes == 1 && e.plane[0].dtype == RAD_BF16)) {
        std::fprintf(stderr, "radiance: hyv4flash: '%s' must be stored bf16 (the drafter's linears are)\n", name);
        return 0;
    }
    return decl_w(b, name, RAD_BF16, { N, K }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                  gl >= 0 ? grp_layer(gl) : grp_model());
}
static rad_op op_gemm(RadBuilder* b, const Geom& g, int64_t rows, int64_t N, int64_t K, rad_weight w) {
    return RAD_OP(b, "gemm_nt",
                  RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", N), RAD_INT("K", K), RAD_STR("dtype", g.dtype)),
                  RAD_WEIGHTS(w));
}

/* Draft step k's own launch of a hidden-correction linear `L` that another step declared (bf16, or
 * IQ8R: the weight `wn` the container holds), over `a` into `y` at up to `rows` rows: the same
 * weight views, new ops at this step's place in the declaration order (DflyStep), with the ops
 * LinearFP8::declare gives that format: `gemm_nt`, or had_quant_act_i8g into `a`'s int8 pair and
 * gemm_nt_q at i8a8h. */
static int dfly_relaunch(RadBuilder* b, const Geom& g, const LinearFP8& L, const char* wn, const ActFP8& a, rad_buf y,
                         int64_t rows, LinearFP8* out) {
    *out = L;
    out->op = out->op_q = 0;
    if (L.bf16) {
        out->op = rw(b, op_gemm(b, g, rows, L.N, L.K, L.w), { a.x }, { y });
        return out->op ? RAD_OK : fail(__LINE__);
    }
    RadEncoding e{};
    if (!L.i8 || !weight_enc(b, wn, &e) || !linear_is_iq8r(e) || a.q8_fed || !a.q8 || !a.s8 ||
        a.g8 != RAD_I8_GROUP_WIDE) {
        std::fprintf(stderr, "radiance: hyv4flash: '%s' must be stored bf16 or IQ8R (the drafter's linears are)\n", wn);
        return RAD_E_DTYPE;
    }
    out->op_q = rw(b, RAD_OP(b, "had_quant_act_i8g",
                             RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", L.K), RAD_INT("group", a.g8),
                                        RAD_STR("dtype", "bf16")),
                             RAD_NOWEIGHTS),
                   { a.x }, { a.q8, a.s8 });
    out->op = rw(b, RAD_OP(b, "gemm_nt_q",
                           RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("N", L.N), RAD_INT("K", L.K),
                                      RAD_INT("group", a.g8), RAD_STR("dtype", "i8a8h")),
                           RAD_WEIGHTS(L.w, L.ws)),
                 { a.q8, a.s8 }, { y });
    return out->op_q && out->op ? RAD_OK : fail(__LINE__);
}
static rad_op op_elem(RadBuilder* b, const Geom& g, const char* name, int64_t rows, int64_t n) {
    return RAD_OP(b, name, RAD_PARAMS(RAD_RANGE("M", 1, rows), RAD_INT("n", n), RAD_STR("dtype", g.dtype)),
                  RAD_NOWEIGHTS);
}

/* The drafter (docs/DFLY2.md, steps 1-6), declared after the trunk's closing: its buffers live only
 * in its own passes. Geometry: the trunk's at the drafter's rows, with the linears in the format the
 * container holds them (Dfly::i8), whatever the trunk's (dfly_plan said it is served). */
static int decl_dfly(RadBuilder* b, Model& m) {
    Dfly& df = m.dfly;
    Names& nm = m.nm;
    const ModelShape& f = m.f;
    const Geom& tg = m.g;
    df.g = tg;
    df.g.max_tok = df.rows;
    df.g.w_bf16 = !df.i8;
    df.g.w_i8h = df.i8;
    df.g.w_i8_group = RAD_I8_GROUP_WIDE;
    const Geom& g = df.g;
    const int64_t n = f.n_embd, S = tg.max_seqs, R = df.rows, W = tg.world, qk = f.nope + f.rope;
    const int64_t H = f.heads_sparse / W, K = f.idx_topk_groups, HW = df.n_layers * (1 + K);
    const int64_t V = tg.n_vocab;   /* this rank's slice of the vocabulary */
    constexpr int64_t kTopR = 2;    /* a rank's candidates: the gather moves 16-byte words (pair_argmax) */
    const int gl0 = (int)f.n_layer;   /* the drafter's weight groups, past the trunk's layers */

    /* ---- buffers */
    df.x = decl_b(b, nm.f("dfly_x"), RAD_BF16, { R, n });
    df.a.x = decl_b(b, nm.f("dfly_a"), RAD_BF16, { R, n });
    df.qa.x = decl_b(b, nm.f("dfly_qa"), RAD_BF16, { R, f.q_lora });
    df.kva = decl_b(b, nm.f("dfly_kva"), RAD_BF16, { R, f.kv_lora + f.rope });
    df.B.q = decl_b(b, nm.f("dfly_q"), RAD_BF16, { R, H * qk });
    df.B.qlat = decl_b(b, nm.f("dfly_qlat"), RAD_BF16, { R, H * f.kv_lora });
    df.B.olat = decl_b(b, nm.f("dfly_olat"), RAD_BF16, { R, H * f.kv_lora });
    df.B.gt = decl_b(b, nm.f("dfly_gt"), RAD_BF16, { R, H * f.v_dim });
    df.B.o.x = decl_b(b, nm.f("dfly_o"), RAD_BF16, { R, H * f.v_dim });
    df.y = decl_b(b, nm.f("dfly_y"), RAD_BF16, { R, n });
    df.gu = decl_b(b, nm.f("dfly_gate_up"), RAD_BF16, { R, 2 * df.ffw });
    df.ff.x = decl_b(b, nm.f("dfly_ffn"), RAD_BF16, { R, df.ffw });
    df.yn = decl_b(b, nm.f("dfly_yn"), RAD_BF16, { R, n });
    df.hf = decl_b(b, nm.f("dfly_hf"), RAD_F32, { R, n });
    df.hb = decl_b(b, nm.f("dfly_hb"), RAD_BF16, { R, n });
    df.hist = decl_b(b, nm.f("dfly_hist"), RAD_I32, { R, HW });
    df.e0 = decl_b(b, nm.f("dfly_e0"), RAD_BF16, { S, n });
    df.h0 = decl_b(b, nm.f("dfly_h0"), RAD_BF16, { S, n });
    df.fz = decl_b(b, nm.f("dfly_fz"), RAD_BF16, { S, n });
    df.cat.x = decl_b(b, nm.f("dfly_cat"), RAD_BF16, { S, 2 * n });
    df.yk = decl_b(b, nm.f("dfly_yk"), RAD_BF16, { S, n });
    df.ep = decl_b(b, nm.f("dfly_ep"), RAD_BF16, { S, n });
    df.cgu = decl_b(b, nm.f("dfly_cgu"), RAD_BF16, { S, 2 * df.hcw });
    df.cff.x = decl_b(b, nm.f("dfly_cff"), RAD_BF16, { S, df.hcw });
    df.cd = decl_b(b, nm.f("dfly_cd"), RAD_BF16, { S, n });
    df.cc = decl_b(b, nm.f("dfly_cc"), RAD_BF16, { S, n });
    df.logits = decl_b(b, nm.f("dfly_logits"), RAD_BF16, { S, V });
    df.cand = decl_b(b, nm.f("dfly_cand"), RAD_I32, { S, kTopR });
    df.cval = decl_b(b, nm.f("dfly_cval"), RAD_F32, { S, kTopR });
    df.pairs = decl_b(b, nm.f("dfly_pairs"), RAD_F32, { S, kTopR, 2 });
    if (W > 1) df.gath = decl_b(b, nm.f("dfly_gath"), RAD_F32, { S, W * kTopR, 2 });
    df.prev = decl_b(b, nm.f("dfly_prev"), RAD_I32, { S });
    /* PERSIST: the core reads the proposal after the pass, past the last op that writes it */
    df.proposal = decl_b(b, nm.f("dfly_proposal"), RAD_I32, { S, df.depth }, RAD_BUF_PERSIST);
    {
        const rad_buf all[] = { df.x, df.a.x, df.qa.x, df.kva, df.B.q, df.B.qlat, df.B.olat, df.B.gt, df.B.o.x, df.y,
                                df.gu, df.ff.x, df.yn, df.hf, df.hb, df.hist, df.e0, df.h0, df.fz, df.cat.x, df.yk,
                                df.ep, df.cgu, df.cff.x, df.cd, df.cc, df.logits, df.cand, df.cval, df.pairs,
                                df.prev, df.proposal };
        for (rad_buf x : all)
            if (!x) return fail(__LINE__);
        if (W > 1 && !df.gath) return fail(__LINE__);
    }
    /* An IQ8R drafter: each linear's input gets an int8 code pair, which the linear's own quantiser
     * writes from the bf16 plane (no producer here writes codes), as the trunk's i8pair. A bf16
     * drafter declares none, so its buffers are the ones it always had. */
    if (df.i8) {
        const auto pair = [&](ActFP8& a, const char* name, int64_t width, int64_t rows) {
            return a.declare_qs(b, nm, g, name, width, rows, true, false, g.w_i8_group);
        };
        HY_TRY(pair(df.a, "dfly_a", n, R));
        HY_TRY(pair(df.qa, "dfly_qa", f.q_lora, R));
        HY_TRY(pair(df.B.o, "dfly_o", H * f.v_dim, R));
        HY_TRY(pair(df.ff, "dfly_ffn", df.ffw, R));
        HY_TRY(pair(df.cat, "dfly_cat", 2 * n, S));
        HY_TRY(pair(df.cff, "dfly_cff", df.hcw, S));
    }

    /* ---- step 1: the block's embedding (the trunk's table, vocab-sharded and summed as the
     * trunk's), and the ring entry of p0 - 1 for every block row */
    df.op_embed = rw(b, RAD_OP(b, "embed_lookup",
                               RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n_embd", n), RAD_INT("n_vocab", tg.n_vocab),
                                          RAD_STR("dtype", g.dtype), RAD_INT("vocab_offset", tg.vocab_off)),
                               RAD_WEIGHTS(m.w_token)),
                     {}, { df.x });
    df.op_embed_ar = op_reduce(b, g, df.x, true);
    if (!df.op_embed || (W > 1 && !df.op_embed_ar)) return fail(__LINE__);
    /* `fill`: what a row gets when its sequence has no entry to read (no slot, p0 < 1, a stale
     * entry). -1 in the histories, where it reads as an empty selection's count and absent pages;
     * 0 in the f32 h plane, where -1 would be a NaN that the fusion and the hidden correction carry
     * into every drafted row. Either way the proposals are garbage and the verify rejects them. */
    auto load = [&](int64_t width, int64_t offset, int64_t fill) {
        return RAD_OP(b, "hy4_ring_load",
                      RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("ring", df.depth + 1), RAD_INT("width", width),
                                 RAD_INT("offset", offset), RAD_INT("seqs", S), RAD_INT("fill", fill)),
                      RAD_NOWEIGHTS);
    };
    df.op_load_h = rw(b, load(n, 1, 0), {}, { df.hf });
    df.op_load_hist = rw(b, load(HW, 1 + n, -1), {}, { df.hist });
    /* h was stored as the f32 bits of a bf16 value: narrowing it back is exact */
    df.op_cast_h = rw(b, RAD_OP(b, "cast", RAD_PARAMS(RAD_RANGE("M", 1, R), RAD_INT("n", n), RAD_STR("from", "f32"),
                                                      RAD_STR("to", "bf16")),
                                RAD_NOWEIGHTS),
                      { df.hf }, { df.hb });

    /* ---- step 2: slot 0's fusion. Each sequence's slot 0 is its first block row, cu_seqlens[s]:
     * gathered by it, fused, and added back by it (scatter_add_rows rounds the sum once, as the
     * reference's bf16 `anchors + gate * fused`). e0 keeps the unfused embedding of b, step 6's
     * first embed(prev). */
    auto gather = [&]() {
        return RAD_OP(b, "gather_rows", RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n", n), RAD_STR("dtype", g.dtype)),
                      RAD_NOWEIGHTS);
    };
    df.op_g_e0 = rw(b, gather(), { df.x }, { df.e0 });
    df.op_g_h0 = rw(b, gather(), { df.hb }, { df.h0 });
    HY_TRY(decl_norm(b, nm, nm.f("dfly.fusion.enorm.weight"), "model.dfly_layers.mtp_fusion.enorm.weight", n, -1,
                     &df.w_enorm));
    HY_TRY(decl_norm(b, nm, nm.f("dfly.fusion.hnorm.weight"), "model.dfly_layers.mtp_fusion.hnorm.weight", n, -1,
                     &df.w_hnorm));
    HY_TRY(decl_norm(b, nm, nm.f("dfly.fusion.gate"), "model.dfly_layers.mtp_fusion.gate", n, -1, &df.w_gate));
    df.op_enorm = rw(b, op_rmsnorm(b, g, S, n, f.eps, df.w_enorm), { df.e0 }, { df.cat.x });
    df.op_hnorm = rw(b, op_rmsnorm(b, g, S, n, f.eps, df.w_hnorm), { df.h0 }, { df.cat.x });
    /* bf16 at a bf16 geometry: the same weight and gemm_nt over cat's bf16 plane as a plain one */
    HY_TRY(df.eh.declare(b, nm, g, nm.f("dfly.fusion.eh"), n, 2 * n, RAD_SHARD_NONE, grp_model(),
                         { "model.dfly_layers.mtp_fusion.eh_proj.weight" }, df.cat, df.fz, S));
    df.op_gate = rw(b, RAD_OP(b, "mul_weight", RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n", n), RAD_STR("dtype", g.dtype)),
                              RAD_WEIGHTS(df.w_gate)),
                    { df.fz }, { df.fz });
    df.op_fuse = rw(b, op_elem(b, g, "scatter_add_rows", S, n), { df.fz, df.x }, { df.x });
    if (!df.op_load_h || !df.op_load_hist || !df.op_cast_h || !df.op_g_e0 || !df.op_g_h0 || !df.op_enorm ||
        !df.op_hnorm || !df.op_gate || !df.op_fuse)
        return fail(__LINE__);

    /* ---- step 3: five layers, pre-norm with residual. The attention is the trunk's sparse one
     * (decl_mla, site `dfly.D`) over trunk layer t_d's cache, heads split by tp as the trunk's. */
    for (int d = 0; d < (int)df.n_layers; ++d) {
        DflyLayer& D = df.layers[(size_t)d];
        const int gl = gl0 + d;
        const char* lp = nm.ckpt("model.dfly_layers.%d", d);
        HY_TRY(decl_norm(b, nm, nm.f("dfly.%d.attn_norm.weight", d), nm.ckpt("%s.input_layernorm.weight", lp), n, gl,
                         &D.w_in_norm));
        D.op_in_norm = rw(b, op_rmsnorm(b, g, R, n, f.eps, D.w_in_norm), { df.x }, { df.a.x });
        D.at.layer = D.target;
        D.at.sparse = true;
        D.at.Hall = f.heads_sparse;
        D.at.H = H;
        D.at.kv = m.kv_full;
        HY_TRY(decl_kvb_abs_map(b, m, nm.f("dfly.%d.attn_kv_b_abs.weight", d),
                                nm.ckpt("%s.self_attn.kv_b_proj.weight", lp), f.heads_sparse));
        MlaSite s{};
        s.wp = nm.ckpt("dfly.%d", d);
        s.cp = nm.ckpt("%s.self_attn", lp);
        s.gl = gl;
        s.g = &df.g;
        s.a = &df.a;
        s.qa = &df.qa;
        s.kva = df.kva;
        s.y = df.y;
        s.B = &df.B;
        s.sel = df.hist;
        s.sel_rows = R;
        s.drafter = true;
        HY_TRY(decl_mla(b, m, D.at, s));
        D.op_add_attn = rw(b, op_elem(b, g, "add", R, n), { df.x, df.y }, { df.x });
        HY_TRY(decl_norm(b, nm, nm.f("dfly.%d.ffn_norm.weight", d), nm.ckpt("%s.post_attention_layernorm.weight", lp), n,
                         gl, &D.w_ffn_norm));
        D.op_ffn_norm = rw(b, op_rmsnorm(b, g, R, n, f.eps, D.w_ffn_norm), { df.x }, { df.a.x });
        HY_TRY(D.gu.declare(b, nm, g, nm.f("dfly.%d.ffn_gate_up", d), 2 * df.ffw, n, RAD_SHARD_NONE, grp_layer(gl),
                            { nm.ckpt("%s.mlp.gate_proj.weight", lp), nm.ckpt("%s.mlp.up_proj.weight", lp) }, df.a, df.gu,
                            0, { df.ffw, df.ffw }));
        /* DeepseekV2MLP's SiluAndMul: no clamp, the gate first */
        D.op_act = rw(b, op_elem(b, g, "silu_mul", R, df.ffw), { df.gu }, { df.ff.x });
        D.down.mapped = d == 0 && df.premapped;
        HY_TRY(D.down.declare(b, nm, g, nm.f("dfly.%d.ffn_down", d), n, df.ffw, RAD_SHARD_NONE, grp_layer(gl),
                              { nm.ckpt("%s.mlp.down_proj.weight", lp) }, df.ff, df.y));
        D.op_add_ffn = rw(b, op_elem(b, g, "add", R, n), { df.x, df.y }, { df.x });
        if (!D.op_in_norm || !D.op_add_attn || !D.op_ffn_norm || !D.op_act || !D.op_add_ffn) return fail(__LINE__);
    }

    /* ---- step 5: the final norm over the block rows */
    if (!df.premapped) HY_TRY(map_copy(b, nm.f("dfly.final_norm.weight"), "model.dfly_layers.final_norm.weight"));
    df.w_final = decl_w(b, nm.f("dfly.final_norm.weight"), RAD_F32, { n }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
    if (!df.w_final) return fail(__LINE__);
    df.op_final = rw(b, op_rmsnorm(b, g, R, n, f.eps, df.w_final), { df.x }, { df.yn });

    /* ---- step 6: five sequential steps over one row a sequence: c = y_k + down(silu(gate(z)) ·
     * up(z)), z = [hidden_norm(y_k) | embed_norm(embed(prev))], prev = argmax(lm_head(c)). The head
     * is the trunk's lm_head over this rank's vocabulary; each rank's best two (row_topk, global ids)
     * meet in pair_argmax, ties to the lower id, so every rank proposes the same tokens. */
    const char* cp = "model.dfly_layers.hidden_correction";
    HY_TRY(decl_norm(b, nm, nm.f("dfly.correction.hidden_norm.weight"), nm.ckpt("%s.hidden_norm.weight", cp), n, -1,
                     &df.w_cnh));
    HY_TRY(decl_norm(b, nm, nm.f("dfly.correction.embed_norm.weight"), nm.ckpt("%s.embed_norm.weight", cp), n, -1,
                     &df.w_cne));
    /* The correction's two weights, which every step launches. bf16: declared here, as plain weights
     * (`cgu`, `cdown` carry them to dfly_relaunch). IQ8R: step 1 declares each through LinearFP8,
     * which reads its stored format (and its views), and steps 2-5 relaunch it. */
    const char* n_cgu = nm.f("dfly.correction.gate_up.weight");
    const char* n_cdown = nm.f("dfly.correction.down.weight");
    LinearFP8 cgu{}, cdown{};
    if (!df.i8) {
        cgu.bf16 = cdown.bf16 = true;
        cgu.N = 2 * df.hcw;
        cgu.K = 2 * n;
        cdown.N = n;
        cdown.K = df.hcw;
        cgu.w = dfly_weight(b, nm, n_cgu, { nm.ckpt("%s.gate_proj.weight", cp), nm.ckpt("%s.up_proj.weight", cp) },
                            cgu.N, cgu.K, -1);
        cdown.w = dfly_weight(b, nm, n_cdown, { nm.ckpt("%s.down_proj.weight", cp) }, cdown.N, cdown.K, -1,
                              df.premapped);
        if (!cgu.w || !cdown.w) return fail(__LINE__);
    }
    df.steps.resize((size_t)df.depth);
    for (int64_t k = 1; k <= df.depth; ++k) {
        DflyStep& C = df.steps[(size_t)(k - 1)];
        C.op_gy = rw(b, gather(), { df.yn }, { df.yk });
        if (k > 1) {
            C.op_embed = rw(b, RAD_OP(b, "embed_lookup",
                                      RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n_embd", n), RAD_INT("n_vocab", tg.n_vocab),
                                                 RAD_STR("dtype", g.dtype), RAD_INT("vocab_offset", tg.vocab_off)),
                                      RAD_WEIGHTS(m.w_token)),
                            { df.prev }, { df.ep });
            C.op_embed_ar = op_reduce(b, g, df.ep, true);
            if (!C.op_embed || (W > 1 && !C.op_embed_ar)) return fail(__LINE__);
        }
        C.op_nh = rw(b, op_rmsnorm(b, g, S, n, f.eps, df.w_cnh), { df.yk }, { df.cat.x });
        C.op_ne = rw(b, op_rmsnorm(b, g, S, n, f.eps, df.w_cne), { k > 1 ? df.ep : df.e0 }, { df.cat.x });
        if (df.i8 && k == 1) {
            HY_TRY(C.gu.declare(b, nm, g, nm.f("dfly.correction.gate_up"), 2 * df.hcw, 2 * n, RAD_SHARD_NONE,
                                grp_model(), { nm.ckpt("%s.gate_proj.weight", cp), nm.ckpt("%s.up_proj.weight", cp) },
                                df.cat, df.cgu, S));
            cgu = C.gu;
        } else {
            HY_TRY(dfly_relaunch(b, g, cgu, n_cgu, df.cat, df.cgu, S, &C.gu));
        }
        C.op_act = rw(b, op_elem(b, g, "silu_mul", S, df.hcw), { df.cgu }, { df.cff.x });
        if (df.i8 && k == 1) {
            C.down.mapped = df.premapped;
            HY_TRY(C.down.declare(b, nm, g, nm.f("dfly.correction.down"), n, df.hcw, RAD_SHARD_NONE, grp_model(),
                                  { nm.ckpt("%s.down_proj.weight", cp) }, df.cff, df.cd, S));
            cdown = C.down;
        } else {
            HY_TRY(dfly_relaunch(b, g, cdown, n_cdown, df.cff, df.cd, S, &C.down));
        }
        C.op_add = rw(b, op_elem(b, g, "add", S, n), { df.yk, df.cd }, { df.cc });
        C.op_head = rw(b, op_gemm(b, g, S, V, n, m.w_lm), { df.cc }, { df.logits });
        C.op_topk = rw(b, W > 1 ? RAD_OP(b, "row_topk",
                                         RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("N", V), RAD_INT("R", kTopR),
                                                    RAD_INT("vocab_off", tg.vocab_off), RAD_STR("dtype", g.dtype)),
                                         RAD_NOWEIGHTS)
                                : RAD_OP(b, "row_topk",
                                         RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("N", V), RAD_INT("R", kTopR),
                                                    RAD_STR("dtype", g.dtype)),
                                         RAD_NOWEIGHTS),
                        { df.logits }, { df.cand, df.cval, df.pairs });
        if (W > 1) {
            /* row-interleaved: a row's pairs from every rank side by side */
            C.op_gath = rw(b, RAD_OP(b, "all_gather",
                                     RAD_PARAMS(RAD_INT("world_size", W), RAD_INT("numel", S * kTopR * 2),
                                                RAD_STR("dtype", "f32"), RAD_INT("row", kTopR * 2)),
                                     RAD_NOWEIGHTS),
                           { df.pairs }, { df.gath });
            if (!C.op_gath) return fail(__LINE__);
        }
        const rad_buf pp = W > 1 ? df.gath : df.pairs;
        auto pick = [&]() {
            return RAD_OP(b, "pair_argmax",
                          RAD_PARAMS(RAD_RANGE("M", 1, S), RAD_INT("n", W * kTopR), RAD_STR("dtype", "f32")),
                          RAD_NOWEIGHTS);
        };
        /* the next step's embedding reads a contiguous id column; the proposal's is pitched */
        C.op_prev = rw(b, pick(), { pp }, { df.prev });
        C.op_prop = rw(b, pick(), { pp }, { df.proposal });
        if (!C.op_gy || !C.op_nh || !C.op_ne || !C.op_act || !C.op_add || !C.op_head ||
            !C.op_topk || !C.op_prev || !C.op_prop)
            return fail(__LINE__);
    }

    RadDrafterDecl drf{};
    drf.name = "dfly2";
    drf.kind = RAD_DRAFT_BLOCK;
    drf.depth = (int)df.depth;
    drf.block = df.block;
    drf.proposal = df.proposal;
    drf.proposal_pitch = df.depth;
    drf.mask_token = df.mask;
    drf.window = 0;   /* no cache of its own: nothing to window */
    HY_TRY(rad_declare_drafter(b, &drf));
    rad_note(b, "DFly2 drafter: %lld layers over trunk layers %d..%d, block %lld (mask %d), ffn %lld, correction %lld, "
                "a ring of %lld entries of %lld words",
             (long long)df.n_layers, df.targets.front(), df.targets.back(), (long long)df.block, (int)df.mask,
             (long long)df.ffw, (long long)df.hcw, (long long)(df.depth + 1), (long long)df.entry);
    return RAD_OK;
}

static int declare(RadBuilder* b, const RadModelMeta* meta, const RadBuildCtx* ctx) {
    if (!b || !meta || !ctx || rad_build_slot(ctx) >= MAX_SLOTS) return fail(__LINE__);
    Model& m = ctx->shape_probe ? g_probe : g_model[rad_build_slot(ctx)];
    /* The reset loses the windows' pinned copy: it is freed first (each redeclaration). */
    /* g_probe never holds one: step() does not run on it. */
    if (m.win_cu_h) rad_dev_free(m.win_cu_h, RAD_MEM_HOST_PINNED);
    m = Model{};
    m.nm = Names(ctx->scope);
    if (!ctx->shape_probe) tap_init();   /* g_tap is global and sizing declares run in parallel */
    HY_TRY(read_shape(meta, m.f));
    ModelShape& f = m.f;
    Names& nm = m.nm;

    /* The base geometry: geom_from over a copy of the metadata with the fields the core does not get
     * from this config (nested in text_config, rope per layer kind). The base heads are the sparse
     * layers'; the window layers' go in gw. head_dim = qk = v (unabsorbed form). */
    RadModelMeta mm = *meta;
    mm.n_layers = (int)f.n_layer;
    mm.n_embd = f.n_embd;
    mm.n_head = f.heads_sparse;
    mm.n_head_kv = f.heads_sparse;
    mm.head_dim = f.nope + f.rope;
    mm.n_ff = f.ff_dense;
    mm.n_vocab = f.vocab;
    mm.rms_eps = (float)f.eps;
    mm.rope_theta = (float)f.theta_sparse;
    HY_TRY(geom_from(m.g, &mm, ctx, "bf16", RAD_BF16));
    /* tp: the attention heads and the experts' width are split; everything else is replicated. */
    const int64_t W = m.g.world;
    if (W != 1 && W != 2) {
        std::fprintf(stderr, "radiance: hyv4flash: serves tp 1 or 2, not %lld\n", (long long)W);
        return RAD_E_UNSUPPORTED;
    }
    if (f.heads_sparse % W || f.heads_win % W || (f.ff_exp / 128) % W || f.ff_exp % 128) {
        std::fprintf(stderr, "radiance: hyv4flash: heads (%lld, %lld) or experts of %lld do not split over %lld\n",
                     (long long)f.heads_sparse, (long long)f.heads_win, (long long)f.ff_exp, (long long)W);
        return RAD_E_UNSUPPORTED;
    }
    /* Text: the 3 M-RoPE positions coincide and radiance's 1-D rope (gptj) is enough. With the tower
     * served, images separate them: hy4_rope (GPT-J pairs with the interleaved sections, which
     * radiance does not have together) over RadBatch::rope_pos. */
    m.media = Vision::is_present(meta) && ctx->max_enc_patches > 0;
    m.g.rope_mc = m.media;
    if (m.media) {
        const char* sec = rad_meta_gets(meta, "text_config.rope_parameters.sliding_attention.mrope_section",
                                        rad_meta_gets(meta, "rope_parameters.sliding_attention.mrope_section", ""));
        /* the sparse layers' table: `deepseek_sparse_attention` in the checkpoint (HF calls it
         * indexed_attention) */
        const char* sec2 = rad_meta_gets(meta, "text_config.rope_parameters.deepseek_sparse_attention.mrope_section",
                           rad_meta_gets(meta, "rope_parameters.deepseek_sparse_attention.mrope_section",
                           rad_meta_gets(meta, "text_config.rope_parameters.indexed_attention.mrope_section", "")));
        if (!sec || !*sec || std::strcmp(sec, sec2)) {
            std::fprintf(stderr, "radiance: hyv4flash: the tower needs mrope_section (and the same one in both tables)\n");
            return RAD_E_UNSUPPORTED;
        }
        m.tok_img = rad_meta_geti(meta, "image_token_id", -1);
        m.tok_vid = rad_meta_geti(meta, "video_token_id", -1);
        std::snprintf(m.g.rope_mode, sizeof m.g.rope_mode, "imrope");
        std::snprintf(m.g.rope_sections, sizeof m.g.rope_sections, "%s", sec);
    }
    /* The trunk's linears are bf16 unless the container was converted with --set hy4.trunk_i8=1
     * (data/recipes/hy4flash-w4nl64-gptq-iq8r.recipe): then they are IQ8R (int8 rotated by fwht128,
     * a u8 scale per 256 columns under a bf16 one a row), the int8 format this build serves (block
     * fp8 is refused), and every activation buffer in front of one declares its int8 code pair
     * (Geom::w_i8h). A container-wide key, read the same at convert and at serve time, because
     * producers and consumers must agree. The bf16 trunk was 15.7 ms of a ~28 ms decode step on the
     * GPTQ container at tp 2. */
    const bool trunk_i8 = rad_meta_geti(meta, "hy4.trunk_i8", 0) != 0;
    /* The Engram tables' form (decl_engram): a container-wide key, as trunk_i8, set by the conversion
     * that wrote conv/engram_codes.py's codes (deploy/conv-engram-e4m3.sh). Any other value is
     * refused here, so a later codes form cannot be served as this one. */
    {
        const char* ec = rad_meta_gets(meta, kEngramCodesKey, "");
        if (*ec && std::strcmp(ec, kEngramCodesRowmax)) {
            std::fprintf(stderr, "radiance: hyv4flash: %s=%s is not a form this build reads (only %s)\n",
                         kEngramCodesKey, ec, kEngramCodesRowmax);
            return fail(__LINE__);
        }
        m.engram_codes = *ec != 0;
    }
    m.g.w_bf16 = !trunk_i8;
    m.g.w_i8h = trunk_i8;
    m.g.w_i8_group = 256;
    /* Speculative verify (prompt lookup, or a drafter): a rejected draft must not stay in any
     * per-sequence state. The indexer's tail ring holds ratio + max_spec rows (rad_qsa.h), and
     * Engram's n-gram window ngram-1 + max_spec ids, read at num_accepted - 1 (hy4_ngram_ids, as
     * radiance's ngram_ids). The attention caches are paged by position and the next verify
     * overwrites a rejected row. */
    m.max_spec = std::max<int64_t>(ctx->max_spec, 0);
    if (ctx->kv_dtype != RAD_DT_INVALID && ctx->kv_dtype != RAD_BF16) {
        std::fprintf(stderr, "radiance: hyv4flash: this version serves a BF16 cache\n");
        return RAD_E_UNSUPPORTED;
    }
    m.g.kv_dtype = "bf16";
    m.gw = m.g;
    m.gw.theta = (float)f.theta_win;
    m.gw.n_head = m.gw.n_head_kv = f.heads_win / W;
    m.gw.n_head_all = m.gw.n_head_kv_all = f.heads_win;
    Geom& g = m.g;
    const int64_t T = g.max_tok, n = f.n_embd, qk = f.nope + f.rope;

    /* ---- MLA form (docs/DESIGN.md, «Absorbed MLA»). HY4_MLA=absorbed|expanded|both; without
     * the variable, absorbed if the container (or the conversion, with hy4.ceros_rope in --add) has
     * the absorbed kv_b. "both" is for converting: it declares both kv_b and serves the
     * absorbed form. */
    {
        const char* mv = std::getenv("HY4_MLA");
        const std::string mode = mv && *mv ? mv : "";
        if (!mode.empty() && mode != "absorbed" && mode != "expanded" && mode != "both") {
            std::fprintf(stderr, "radiance: hyv4flash: HY4_MLA=%s: must be absorbed, expanded or both\n", mode.c_str());
            return RAD_E_INVAL;
        }
        /* expanded over an absorbed-only container: kv_b is missing, and the weight load says so. */
        if (mode != "expanded") {
            for (int l = 0; l < (int)f.n_layer; ++l)
                HY_TRY(decl_kvb_abs_map(b, m, l, f.sparse[(size_t)l] ? f.heads_sparse : f.heads_win));
            /* The absorbed kv_b is built by copying rows of kv_b_proj: that only works if it is
             * BF16 (a recipe that quantised it would copy bytes of another format). It is checked
             * when the source is known (when converting); when serving, the weight is already
             * written. */
            RadEncoding src_enc{};
            if (weight_enc(b, m.nm.ckpt("model.layers.%d.self_attn.kv_b_proj.weight", 0), &src_enc) &&
                (src_enc.n_planes != 1 || src_enc.plane[0].dtype != RAD_BF16)) {
                std::fprintf(stderr, "radiance: hyv4flash: kv_b_proj is not BF16; the absorbed kv_b is built by copying "
                                     "its rows (HY4_MLA=expanded, or keep kv_b_proj BF16 in the recipe)\n");
                return RAD_E_UNSUPPORTED;
            }
            RadEncoding e{};
            const bool has_abs = weight_enc(b, m.nm.f("blk.%d.attn_kv_b_abs.weight", 0), &e);
            if (!mode.empty() && !has_abs) {
                std::fprintf(stderr, "radiance: hyv4flash: HY4_MLA=%s needs the absorbed kv_b and the container does not "
                                     "have it (convert with hy4.ceros_rope in --add and HY4_MLA=both)\n", mode.c_str());
                return RAD_E_UNSUPPORTED;
            }
            m.mla_abs = has_abs;
            m.mla_both = mode == "both";
        }
        static std::atomic<bool> announced{false};   /* once a process: declare runs per rank and level */
        if (!ctx->shape_probe && !announced.exchange(true)) {
            std::fprintf(stderr, "hyv4flash: MLA %s%s\n", m.mla_abs ? "absorbed" : "expanded",
                         m.mla_both ? " and expanded kv_b (both: for converting only; when serving, it loads a kv_b "
                                       "it does not use)" : "");
        }
    }

    /* ---- the DFly2 drafter: whether this build serves it (declares nothing yet) */
    HY_TRY(dfly_plan(b, meta, ctx, m));

    /* ---- KV groups */
    {
        /* Absorbed: one head [lat | k_rope] in the K and V halves, replicated on each rank. */
        const int64_t kv_heads_d = m.mla_abs ? 1 : f.heads_sparse / W, kv_heads_w = m.mla_abs ? 1 : f.heads_win / W;
        const int64_t kv_hd = m.mla_abs ? (f.kv_lora + f.rope) / 2 : qk;
        RadKVGroupDecl d{};
        d.kind = RAD_KV_FULL; d.dtype = RAD_BF16; d.n_head_kv = kv_heads_d; d.head_dim = kv_hd;
        m.kv_full = rad_decl_kv_group(b, nm.f("kv_full"), &d);
        /* A cut-down model may have no window or Engram layers: their groups only if something
         * uses them (the core refuses a group without layers). */
        const bool has_win = std::count(f.sparse.begin(), f.sparse.end(), false) > 0;
        RadKVGroupDecl w = d;
        w.kind = RAD_KV_WINDOW; w.window = f.window; w.n_head_kv = kv_heads_w;
        if (has_win) m.kv_win = rad_decl_kv_group(b, nm.f("kv_win"), &w);
        RadKVGroupDecl t{};
        t.kind = RAD_KV_LINEAR; t.dtype = RAD_BF16; t.n_head_kv = 1;
        t.state_dim[0] = f.idx_ratio + m.max_spec; t.state_dim[1] = f.idx_dim;   /* ring: ratio + max_spec */
        m.kv_tail = rad_decl_kv_group(b, nm.f("kv_idx_tail"), &t);
        RadKVGroupDecl s{};
        s.kind = RAD_KV_LINEAR; s.dtype = RAD_I32; s.n_head_kv = 1;
        s.state_dim[0] = 1; s.state_dim[1] = f.engram_ngram - 1 + m.max_spec;   /* the n-gram window */
        if (!f.engram_layers.empty()) m.kv_tok = rad_decl_kv_group(b, nm.f("kv_engram_tok"), &s);
        if (!m.kv_full || (has_win && !m.kv_win) || !m.kv_tail || (!f.engram_layers.empty() && !m.kv_tok))
            return fail(__LINE__);
        /* The drafter's ring (docs/DFLY2.md): one entry per position mod depth + 1, i32 words. Last,
         * so the trunk's groups keep their order. */
        if (m.dfly.on) {
            RadKVGroupDecl r{};
            r.kind = RAD_KV_LINEAR; r.dtype = RAD_I32; r.n_head_kv = 1;
            r.state_dim[0] = m.dfly.depth + 1; r.state_dim[1] = m.dfly.entry;
            m.dfly.kv = rad_decl_kv_group(b, nm.f("kv_dfly"), &r);
            if (!m.dfly.kv) return fail(__LINE__);
            HY_TRY(rad_bind_layer_kv(b, m.dfly.targets[0], m.dfly.kv));
        }
    }

    /* ---- MoE configuration and probing of each layer's experts */
    MoeFP8::Config mc{};
    mc.n_expert_all = mc.n_expert = f.n_expert;
    mc.expert_base = 0;
    /* With tp, each card carries a slice of columns of every expert (as in nemotronh) and a partial
     * sum; the routed arm's all-reduce goes after the latent's up projection. */
    /* Split by columns, not by experts: every card has every expert and resolves the encoding with
     * the same one. */
    mc.probed_expert = 0;
    mc.top_k = f.top_k;
    mc.n_ff_exp = f.ff_exp / W;
    /* Calibration at tp 2: the tap needs the Gram of each expert's whole K, and with split columns
     * each rank would see a slice (MoeFP8 refuses it). While recording, each rank carries whole
     * experts, [r E/W, (r+1) E/W), as in nemotronh. The routed arm's all-reduce sums the same way:
     * each rank contributes its experts' rows. */
    if (W > 1 && calib_dir()) {
        if (f.n_expert % W) {
            std::fprintf(stderr, "radiance: hyv4flash: %lld experts do not split over %lld ranks for calibration\n",
                         (long long)f.n_expert, (long long)W);
            return RAD_E_UNSUPPORTED;
        }
        mc.n_expert = f.n_expert / W;
        mc.expert_base = (int64_t)m.g.rank * mc.n_expert;
        mc.n_ff_exp = f.ff_exp;
        mc.probed_expert = mc.expert_base;   /* probe() maps the first one this rank has */
    }
    mc.norm_topk = 1;
    mc.sigmoid_router = 1;
    /* The selection and the scatter in one launch (router_sigmoid_topk_scatter) wherever a pass fits
     * the sort's single workgroup (rows * top_k <= 1024: decode and drafter verifies); wider passes keep
     * the separate ops. Byte-identical either way (rad_block_moe_fp8.h), 41 launches fewer a step. */
    mc.router_fuse = 1;
    mc.route_scale = f.route_scale;
    mc.swiglu_limit = f.swiglu_limit;
    mc.router_k = n;
    mc.no_reduce = 1;
    /* Prefill span pass: Code000, radiance e4541c70. Without --prefill-span-pass step a routed
     * pass is 2048 rows, as it always was here, and a wider step takes several (MoeFP8::step_rows).
     * With it, on a deployment that spans, the pass is the whole step: this plugin runs every step
     * serially (it pipelines nothing; the routed arm's all-reduce is its own, over the step's
     * rows), so nothing needs the pass to be a piece. The pass then reads each expert's weight and
     * sorts the slots once a step (the PCIe sweep of the pooled experts is already once a step,
     * whatever the passes). A calibration run keeps its pieces: its statistics are taken a pass
     * at a time. Not bit-identical to 2048-row passes (an expert's longer run moves rows between
     * the grouped GEMM's short-run and tiled paths); deterministic. */
    m.piece_rows = std::min<int64_t>(T, 2048);
    if (ctx->shape_probe) {
        m.span_on = g_model[rad_build_slot(ctx)].span_on;
        m.pass_step = g_model[rad_build_slot(ctx)].pass_step;
    } else {
        m.span_on = ctx->max_mixed_tok > 0 && ctx->max_mixed_tok < T && T > m.piece_rows;
        m.pass_step = ctx->span_pass_step != 0 && m.span_on && !calib_dir();
        if (ctx->span_pass_step != 0 && !m.pass_step && g.rank == 0)
            rad_note(b, "--prefill-span-pass step: the routed pass stays 2048 rows (%s)",
                     !m.span_on ? "this deployment has no step wider than that"
                                : "a calibration run takes its statistics a pass at a time");
    }
    /* The step pass is capped at 65536 slot rows (4096 tokens at top-16), the tallest pass run on
     * the cards: span 2's. A wider step (span 3 or 4) takes passes that tall, so the slot-row
     * planes and the routed ops' declared ranges stay where they were validated on every step,
     * decode included. The first span-4 run with an 8192-token pass (131072 slot rows) hung in
     * decode in libr4d's chained moe_down_dec_kernel; whether that range was the cause is not
     * known. A deployment without a span, and span 2, declare what they did before the cap. */
    const int64_t pass_cap = std::max<int64_t>(m.piece_rows, 65536 / std::max<int64_t>(f.top_k, 1));
    mc.rows = m.pass_step ? std::min(T, pass_cap) : m.piece_rows;
    if (m.pass_step && T > pass_cap && !ctx->shape_probe && g.rank == 0)
        rad_note(b, "--prefill-span-pass step: a %lld-token step takes routed passes of %lld tokens "
                    "(65536 expert-slot rows, the tallest pass validated)", (long long)T, (long long)pass_cap);
    m.msrc.resize((size_t)f.n_layer);
    for (int l = 0; l < (int)f.n_layer; ++l) {
        if (!f.moe[(size_t)l]) continue;
        const char* pre = nm.ckpt("model.layers.%d.mlp", l);
        MoeFP8::Src& ms = m.msrc[(size_t)l];
        ms.cfg = mc;
        ms.router = nm.ckpt("%s.gate.weight", pre);
        ms.router_bias = nm.ckpt("%s.gate.e_score_correction_bias", pre);
        ms.experts_gate_up = nm.ckpt("%s.experts.gate_up_proj", pre);
        ms.experts_down = nm.ckpt("%s.experts.down_proj", pre);
        ms.cfg.calib = calib_dir() && calib_layer_on(l) ? 1 : 0;
        HY_TRY(MoeFP8::probe(b, nm, l, ms, &ms.cfg));
        const bool q = !ms.cfg.expert_bf16;
        if (l == f.first_moe()) { m.moe_q = q; m.moe_rot = ms.cfg.expert_rot != 0; }
        const char* dbg = std::getenv("HY4_DEBUG_MOE");
        if (dbg && *dbg && !ctx->shape_probe)
            std::fprintf(stderr, "hyv4flash: layer %d experts bf16=%d rot=%d nl=%d g64=%d w4=%d\n", l,
                         ms.cfg.expert_bf16, ms.cfg.expert_rot, ms.cfg.expert_nl, ms.cfg.expert_g64, ms.cfg.expert_w4);
        if (q != m.moe_q || (ms.cfg.expert_rot != 0) != m.moe_rot) {
            std::fprintf(stderr, "radiance: hyv4flash: layer %d stores its experts in a different format than the first\n", l);
            return RAD_E_UNSUPPORTED;
        }
    }

    /* ---- buffers */
    m.max_work = (T + f.idx_ratio - 1) / f.idx_ratio + g.max_seqs;
    m.max_groups = (g.max_ctx + f.idx_ratio - 1) / f.idx_ratio + 1;
    /* The score plane is [plane_rows, max_groups] f32. A step wider than 1024 rows is scored by
     * windows only if the deployment guarantees it is a single sequence
     * (RadBuildCtx::max_mixed_tok); otherwise the plane covers the whole step. */
    m.plane_rows = T;
    /* With images the score reads the step's 3-D positions whole (RadBatch::rope_pos cannot be
     * chunked by windows): the plane covers the step. */
    const bool win1024 = !m.media && T > 1024 && ctx->max_mixed_tok > 0 && ctx->max_mixed_tok <= 1024;
    if (win1024) m.plane_rows = 1024;
    /* Prefill span pass: Code000, radiance e4541c70 (qwen4exp_fp8's qsa_rows). Under a span whose
     * shared budget is a piece, the plane stays a piece tall and the spanned step, a single
     * sequence, is scored in windows: the plane is the one buffer whose size is the product of two
     * deployment bounds (at 196K of context 384 MiB a rank for every 2048 rows), and a window
     * writes each row the bytes the whole plane would. HY4_SPAN_PLANE=step grows it back to the
     * step after either rule (one score launch a step; for measuring). */
    else if (!m.media && m.span_on && ctx->max_mixed_tok <= m.piece_rows) m.plane_rows = m.piece_rows;
    if (m.plane_rows < T) {
        const char* sp = std::getenv("HY4_SPAN_PLANE");
        if (sp && !std::strcmp(sp, "step")) m.plane_rows = T;
    }
    /* The plane's limit, as in qwen4exp (512 Mi f32 elements, 2 GiB). */
    if (m.plane_rows * m.max_groups > (int64_t)512 << 20) {
        std::fprintf(stderr, "radiance: hyv4flash: the score plane [%lld, %lld] exceeds 512 Mi elements: lower "
                             "max-model-len or max-num-batched-tokens\n",
                     (long long)m.plane_rows, (long long)m.max_groups);
        return RAD_E_UNSUPPORTED;
    }
    /* Rows per launch of the attention with selection (scratch budget, as in QSA). */
    m.sel_rows = std::max<int64_t>(g.max_seqs, 256);
    m.h = decl_b(b, nm.f("h"), RAD_BF16, { T, f.hc * n });
    m.xin = decl_b(b, nm.f("xin"), RAD_BF16, { T, n });
    m.post = decl_b(b, nm.f("post"), RAD_F32, { T, f.hc });
    m.a.x = decl_b(b, nm.f("a"), RAD_BF16, { T, n });
    m.qa.x = decl_b(b, nm.f("qa"), RAD_BF16, { T, f.q_lora });
    /* An IQ8R trunk (Geom::w_i8h): the first int8 linear reading an input quantises it from the
     * bf16 plane into an int8 code pair of the input's own, at the trunk's scale group, and the
     * others reading the same rows take that pair as fed (decl_mla's gate, decl_ffn's shared
     * expert). Where libr4d fuses it, the producer writes the pair itself: the input norms and
     * q_a_norm (rmsnorm_had_quant), the attention gate (gate_had_quant) and the SiLU products
     * (gated_had_quant); the pair carries no E4M3 plane. */
    const auto i8pair = [&](ActFP8& a, const std::string& name, int64_t width) {
        return g.w_i8h ? a.declare_qs(b, nm, g, name.c_str(), width, T, true, false, g.w_i8_group) : RAD_OK;
    };
    HY_TRY(i8pair(m.a, "a", n));
    HY_TRY(i8pair(m.qa, "qa", f.q_lora));
    m.kva = decl_b(b, nm.f("kva"), RAD_BF16, { T, f.kv_lora + f.rope });
    m.lat.x = decl_b(b, nm.f("lat"), RAD_BF16, { T, f.kv_lora });
    for (int k = 0; k < 2; ++k) {
        const int64_t Hk = (k == 0 ? f.heads_sparse : f.heads_win) / W;
        const char* sf = k == 0 ? "sparse" : "win";
        AttnBufs& B = m.ab[k];
        B.q = decl_b(b, nm.f("q_%s", sf), RAD_BF16, { T, Hk * qk });
        if (!m.mla_abs || m.mla_both)
            B.kvx = decl_b(b, nm.f("kvx_%s", sf), RAD_BF16, { T, Hk * (f.nope + f.v_dim) });
        if (!m.mla_abs) {
            B.kbuf = decl_b(b, nm.f("kbuf_%s", sf), RAD_BF16, { T, Hk * qk });
            B.vbuf = decl_b(b, nm.f("vbuf_%s", sf), RAD_BF16, { T, Hk * f.v_dim });
        } else {
            B.qlat = decl_b(b, nm.f("qlat_%s", sf), RAD_BF16, { T, Hk * f.kv_lora });
            B.olat = decl_b(b, nm.f("olat_%s", sf), RAD_BF16, { T, Hk * f.kv_lora });
        }
        B.gt = decl_b(b, nm.f("gt_%s", sf), RAD_BF16, { T, Hk * f.v_dim });
        B.o.x = decl_b(b, nm.f("o_%s", sf), RAD_BF16, { T, Hk * f.v_dim });
        HY_TRY(i8pair(B.o, std::string("o_") + sf, Hk * f.v_dim));
    }
    m.y = decl_b(b, nm.f("y"), RAD_BF16, { T, n });
    m.tkn.x = decl_b(b, nm.f("tkn"), RAD_BF16, { T, f.idx_dim });
    /* tkn feeds the indexer's tail store, not an int8 linear: no pair */
    m.iq = decl_b(b, nm.f("iq"), RAD_BF16, { T, f.idx_heads * f.idx_dim });
    m.ihw = decl_b(b, nm.f("ihw"), RAD_BF16, { T, f.idx_heads });
    m.stage = decl_b(b, nm.f("idx_stage"), RAD_BF16, { m.max_work * f.idx_ratio, f.idx_dim });
    m.page = decl_b(b, nm.f("idx_page"), RAD_I32, { m.max_work });
    /* With images, the three components of each group's first token (qsa_work with the planes). */
    m.bpos = m.media ? decl_b(b, nm.f("idx_bpos"), RAD_I32, { m.max_work, 3 })
                     : decl_b(b, nm.f("idx_bpos"), RAD_I32, { m.max_work });
    m.nc = decl_b(b, nm.f("idx_nc"), RAD_I32, { g.max_seqs }, RAD_BUF_PERSIST);
    m.score = decl_b(b, nm.f("idx_score"), RAD_F32, { m.plane_rows, m.max_groups });
    m.sel = decl_b(b, nm.f("idx_sel"), RAD_I32, { T, f.idx_topk_groups }, RAD_BUF_PERSIST);
    m.nsel = decl_b(b, nm.f("idx_nsel"), RAD_I32, { T }, RAD_BUF_PERSIST);
    m.sequ = decl_b(b, nm.f("idx_sequ"), RAD_I32, { T }, RAD_BUF_PERSIST);
    m.win_cu = decl_b(b, nm.f("idx_win_cu"), RAD_I32, { 2 }, RAD_BUF_PERSIST);
    const int64_t E = (int64_t)f.engram_layers.size();
    if (E > 0) {
        m.eids = decl_b(b, nm.f("engram_ids"), RAD_I32, { E * T, f.engram_heads }, RAD_BUF_PERSIST, RAD_DOMAIN_HOST);
        /* The next prefill chunk's ids (RadBatch::n_ahead, one chunk at most), which each Engram gather
         * hands to libavx's reader thread as `ahead`: with the table on disk a fresh prompt's chunk
         * waited on its row reads (18K: 674 tok/s on the first pass, 1800 once cached). */
        m.eids_ahead = decl_b(b, nm.f("engram_ids_ahead"), RAD_I32, { E * T, f.engram_heads }, RAD_BUF_PERSIST,
                              RAD_DOMAIN_HOST);
        m.erows = decl_b(b, nm.f("engram_rows"), RAD_BF16, { T, f.engram_heads * f.engram_d }, RAD_BUF_TRANSIENT,
                         RAD_DOMAIN_HOST);
        m.erows_dev = decl_b(b, nm.f("engram_rows_dev"), RAD_BF16, { T, f.engram_heads * f.engram_d });
        m.enorm = decl_b(b, nm.f("engram_norm"), RAD_BF16, { T, f.engram_heads * f.engram_d });
        m.ev = decl_b(b, nm.f("engram_v"), RAD_BF16, { T, n });
        m.ek = decl_b(b, nm.f("engram_k"), RAD_BF16, { T, f.hc * n });
    }
    m.gu = decl_b(b, nm.f("gate_up"), RAD_BF16, { T, 2 * f.ff_dense });
    m.ff.x = decl_b(b, nm.f("ffn"), RAD_BF16, { T, f.ff_dense });
    HY_TRY(i8pair(m.ff, "ffn", f.ff_dense));
    /* The shared expert with its own buffers: LinearFP8 assumes outputs of exactly N columns. */
    m.gu_s = decl_b(b, nm.f("gate_up_shexp"), RAD_BF16, { T, 2 * f.ff_exp });
    m.ff_s.x = decl_b(b, nm.f("ffn_shexp"), RAD_BF16, { T, f.ff_exp });
    HY_TRY(i8pair(m.ff_s, "ffn_shexp", f.ff_exp));
    m.latm.x = decl_b(b, nm.f("moe_lat"), RAD_BF16, { T, f.latent });
    Geom glat = g;
    glat.n_embd = f.latent;
    glat.w_bf16 = !m.moe_q;
    HY_TRY(m.latm.declare_qs(b, nm, glat, "moe_lat", f.latent));
    HY_TRY(i8pair(m.latm, "moe_lat", f.latent));   /* after the experts' E4M3 pair: declare_qs resets g8 */
    m.routed = decl_b(b, nm.f("moe_routed"), RAD_BF16, { T, n });
    m.sout = decl_b(b, nm.f("shared_out"), RAD_BF16, { T, n });
    m.hout = decl_b(b, nm.f("h_out"), RAD_BF16, { g.max_logit_rows, f.hc * n });
    m.xout = decl_b(b, nm.f("x_out"), RAD_BF16, { g.max_logit_rows, n });
    m.logits = decl_b(b, nm.f("logits"), RAD_F32, { g.max_logit_rows, g.n_vocab });
    HY_TRY(rad_declare_logits(b, m.logits));
    {
        /* A pass's sorted planes: a row per (token, expert slot) of one pass, so a step tall under
         * --prefill-span-pass step (prefill span pass: Code000, radiance e4541c70). */
        const int64_t rows_moe = mc.rows, S = rows_moe * f.top_k;
        m.mw.logits = decl_b(b, nm.f("moe_logits"), RAD_F32, { T, f.n_expert });
        m.mw.ids = decl_b(b, nm.f("moe_ids"), RAD_I32, { T, f.top_k });
        m.mw.ew = decl_b(b, nm.f("moe_weights"), RAD_F32, { T, f.top_k });
        m.mw.eoff = decl_b(b, nm.f("moe_offsets"), RAD_I32, { f.n_expert + 1 });
        m.mw.ecnt = decl_b(b, nm.f("moe_counts"), RAD_I32, { f.n_expert }, RAD_BUF_PERSIST);
        m.mw.sorted = decl_b(b, nm.f("moe_sorted"), RAD_I32, { S });
        /* expert width on this card: ff_exp/W with split columns, all of ff_exp when calibrating */
        const int64_t ffr = mc.n_ff_exp;
        m.mw.egu = decl_b(b, nm.f("moe_gate_up"), RAD_BF16, { S, 2 * ffr });
        m.mw.eff.x = decl_b(b, nm.f("moe_ffn"), RAD_BF16, { S, ffr });
        HY_TRY(m.mw.eff.declare_qs(b, nm, glat, "moe_ffn", ffr, S));
        if (m.moe_rot) {
            m.mw.hr.q = decl_b(b, nm.f("moe_lat_rot.q"), RAD_F8E4M3, { T, f.latent });
            m.mw.hr.s = decl_b(b, nm.f("moe_lat_rot.s"), RAD_F32, { T, f.latent / RAD_FP8_BLOCK });
            if (!m.mw.hr.q || !m.mw.hr.s) return fail(__LINE__);
        }
        m.mw.edn = decl_b(b, nm.f("moe_down"), RAD_BF16, { S, f.latent });
    }

    /* ---- embedding and Engram (ids once per step) */
    HY_TRY(map_copy(b, nm.f("token_embd.weight"), nm.ckpt("model.embed_tokens.weight")));
    m.w_token = decl_w(b, nm.f("token_embd.weight"), RAD_BF16, { g.n_vocab, n }, RAD_ACCESS_VOCAB, RAD_SHARD_ROW,
                       grp_model());
    m.op_embed = rw(b, RAD_OP(b, "embed_lookup",
                              RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n_embd", n), RAD_INT("n_vocab", g.n_vocab),
                                         RAD_STR("dtype", g.dtype), RAD_INT("vocab_offset", g.vocab_off)),
                              RAD_WEIGHTS(m.w_token)),
                    {}, { m.xin });
    /* The embedding is split by vocabulary: each card writes its rows and zero in the others. */
    m.op_embed_ar = op_reduce(b, g, m.xin, true);
    if (g.world > 1 && !m.op_embed_ar) return fail(__LINE__);
    /* The encoder rows replace the lookup's before its reduction (MediaRows). */
    if (m.media) HY_TRY(m.mrows.declare(b, T, n, m.xin));
    m.op_enter = rw(b, RAD_OP(b, "hc_enter", RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("n", n), RAD_INT("hc", f.hc),
                                                        RAD_STR("dtype", g.dtype)), RAD_NOWEIGHTS),
                    { m.xin }, { m.h });
    if (E > 0) {
        std::vector<const char*> smap, smult, svsz, soff;
        for (int li : f.engram_layers) {
            const char* p = nm.ckpt("model.layers.%d.engram_layer", li);
            smap.push_back(nm.ckpt("%s.lookup_table", p));
            smult.push_back(nm.ckpt("%s.hash_multipliers", p));
            svsz.push_back(nm.ckpt("%s.head_vocab_sizes", p));
            soff.push_back(nm.ckpt("%s.head_offsets", p));
        }
        auto cat = [&](const char* decl, const std::vector<const char*>& srcs) -> int {
            RadNameMap mp = map_base(decl, RAD_MAP_CONCAT);
            for (const char* s : srcs) mp.src[mp.n_src++] = s;
            return rad_decl_name_map(b, &mp);
        };
        HY_TRY(cat(nm.f("engram_map"), smap));
        HY_TRY(cat(nm.f("engram_mult"), smult));
        HY_TRY(cat(nm.f("engram_vsz"), svsz));
        HY_TRY(cat(nm.f("engram_off"), soff));
        RadEncoding enc{};
        int64_t shp[RAD_MAX_RANK] = {};
        uint32_t rk = 0;
        int64_t V = 0;
        if (rad_weight_encoding(b, nm.f("engram_map"), &enc, shp, &rk) == RAD_OK && rk >= 1)
            V = (rk == 1 ? shp[0] : shp[0] * shp[1]) / E;
        if (V <= 0) return fail(__LINE__);
        f.engram_map = V;
        m.w_map = decl_w(b, nm.f("engram_map"), RAD_I32, { E * V }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_model());
        m.w_mult = decl_w(b, nm.f("engram_mult"), RAD_I64, { E * f.engram_ngram }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                          grp_model());
        m.w_vsz = decl_w(b, nm.f("engram_vsz"), RAD_I64, { E * f.engram_heads }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                         grp_model());
        m.w_off = decl_w(b, nm.f("engram_off"), RAD_I64, { E * f.engram_heads }, RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE,
                         grp_model());
        if (!m.w_map || !m.w_mult || !m.w_vsz || !m.w_off) return fail(__LINE__);
        HY_TRY(rad_bind_layer_kv(b, f.engram_layers[0], m.kv_tok));
        m.op_ids = rw(b, RAD_OP(b, "hy4_ngram_ids",
                                RAD_PARAMS(RAD_RANGE("M", 1, T), RAD_INT("engram_layers", E),
                                           RAD_INT("heads", f.engram_heads), RAD_INT("ngram", f.engram_ngram),
                                           RAD_INT("pad", f.engram_pad), RAD_INT("seqs", g.max_seqs)),
                                RAD_WEIGHTS(m.w_map, m.w_mult, m.w_vsz, m.w_off)),
                      {}, { m.eids, m.eids_ahead });
        if (!m.op_ids) return fail(__LINE__);
    }

    /* ---- layers */

    m.layers.resize((size_t)f.n_layer);
    for (int l = 0; l < (int)f.n_layer; ++l) {
        Layer& L = m.layers[(size_t)l];
        const char* pre = nm.ckpt("model.layers.%d", l);
        for (size_t e = 0; e < f.engram_layers.size(); ++e)
            if (f.engram_layers[e] == l) {
                L.engram = (int)m.engram.size();
                Engram en{};
                en.layer = l;
                en.e = (int)e;
                m.engram.push_back(en);
                HY_TRY(decl_engram(b, m, m.engram.back()));
            }
        HY_TRY(decl_hc(b, m, L.hc_attn, nm.ckpt("%s.hc_attn_layer.hc_pre.hc", pre), nm.f("blk.%d.hc_attn", l), l, false));
        HY_TRY(decl_norm(b, nm, nm.f("blk.%d.attn_norm.weight", l), nm.ckpt("%s.input_layernorm.weight", pre), n, l,
                               &L.w_in_norm));
        L.attn.layer = l;
        L.attn.sparse = f.sparse[(size_t)l];
        L.attn.Hall = L.attn.sparse ? f.heads_sparse : f.heads_win;
        L.attn.H = L.attn.Hall / W;
        L.attn.kv = L.attn.sparse ? m.kv_full : m.kv_win;
        MlaSite site = mla_site_trunk(m, L.attn, l);
        /* The input norm writes the int8 pair q_a (and the gate, decl_mla) read when q_a is stored
         * IQ8R: probed here, since the norm is declared first. kv_a and the indexer's linears read
         * the bf16 plane, which it still writes. */
        L.in_q8 = g.w_i8h &&
                  q8_fusable(b, g, m.a, n,
                             probe_iq8r(b, nm, g, L.attn.q_a, nm.f("%s.attn_q_a", site.wp),
                                        { nm.ckpt("%s.q_a_proj.weight", site.cp) }),
                             "rmsnorm_had_quant_i8g");
        /* the fused op unresolved for these params: the plain norm, and q_a quantises itself */
        L.op_in_norm = L.in_q8 ? op_rmsnorm_q8(b, g, T, n, f.eps, L.w_in_norm, m.xin, m.a) : 0;
        if (!L.op_in_norm) {
            L.in_q8 = false;
            L.op_in_norm = rw(b, op_rmsnorm(b, g, T, n, f.eps, L.w_in_norm), { m.xin }, { m.a.x });
        }
        site.a_fed = L.in_q8;
        HY_TRY(rad_bind_layer_kv(b, l, L.attn.kv));
        HY_TRY(decl_mla(b, m, L.attn, site));
        for (size_t d = 0; d < m.dfly.layers.size(); ++d)
            if (m.dfly.layers[d].target == l) {
                L.dfly = (int)d;
                HY_TRY(decl_dfly_store(b, m, (int)d));
            }
        L.op_post_attn = op_hc_post(b, m);
        HY_TRY(decl_hc(b, m, L.hc_ffn, nm.ckpt("%s.hc_mlp_layer.hc_pre.hc", pre), nm.f("blk.%d.hc_ffn", l), l, false));
        HY_TRY(decl_norm(b, nm, nm.f("blk.%d.ffn_norm.weight", l), nm.ckpt("%s.post_attention_layernorm.weight", pre),
                               n, l, &L.ffn.w_norm));
        L.ffn.layer = l;
        L.ffn.moe = f.moe[(size_t)l];
        /* The same for the FFN's input: the first linear reading it is ffn_latent_in (the shared
         * expert's gate_up shares its pair, decl_ffn), or layer 0's gate_up; the router reads the
         * bf16 plane. */
        {
            const char* mp = nm.ckpt("%s.mlp", pre);
            L.ffn.norm_q8 =
                g.w_i8h &&
                q8_fusable(b, g, m.a, n,
                           L.ffn.moe ? probe_iq8r(b, nm, g, L.ffn.lat_in, nm.f("blk.%d.ffn_latent_in", l),
                                                  { nm.ckpt("%s.routed_expert_down_proj.weight", mp) })
                                     : probe_iq8r(b, nm, g, L.ffn.gu, nm.f("blk.%d.ffn_gate_up", l),
                                                  { nm.ckpt("%s.gate_proj.weight", mp),
                                                    nm.ckpt("%s.up_proj.weight", mp) }),
                           "rmsnorm_had_quant_i8g");
        }
        L.ffn.op_norm = L.ffn.norm_q8 ? op_rmsnorm_q8(b, g, T, n, f.eps, L.ffn.w_norm, m.xin, m.a) : 0;
        if (!L.ffn.op_norm) {   /* unresolved for these params: the plain norm */
            L.ffn.norm_q8 = false;
            L.ffn.op_norm = rw(b, op_rmsnorm(b, g, T, n, f.eps, L.ffn.w_norm), { m.xin }, { m.a.x });
        }
        HY_TRY(decl_ffn(b, m, L.ffn, l));
        L.op_post_ffn = op_hc_post(b, m);
        if (!L.op_in_norm || !L.op_post_attn || !L.ffn.op_norm || !L.op_post_ffn) return fail(__LINE__);
    }

    /* ---- closing */
    m.op_gather = rw(b, RAD_OP(b, "gather_rows",
                               RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n", f.hc * n),
                                          RAD_STR("dtype", g.dtype)),
                               RAD_NOWEIGHTS),
                     { m.h }, { m.hout });
    HY_TRY(decl_hc(b, m, m.hc_head, nm.ckpt("model.hc_head.hc_head"), nm.f("hc_head"), -1, true));
    HY_TRY(decl_norm(b, nm, nm.f("output_norm.weight"), nm.ckpt("model.norm.weight"), n, -1, &m.w_norm));
    m.op_head_norm = rw(b, op_rmsnorm(b, g, g.max_logit_rows, n, f.eps, m.w_norm), { m.xout }, { m.xout });
    HY_TRY(map_copy(b, nm.f("output.weight"), nm.ckpt("lm_head.weight")));
    m.w_lm = decl_w(b, nm.f("output.weight"), weight_codes_dtype(b, nm.f("output.weight"), RAD_BF16), { g.n_vocab, n },
                    RAD_ACCESS_VOCAB, RAD_SHARD_ROW, grp_model());
    m.op_logits = rw(b, RAD_OP(b, "logits_gemm",
                               RAD_PARAMS(RAD_RANGE("M", 1, g.max_logit_rows), RAD_INT("n_vocab", g.n_vocab),
                                          RAD_INT("n_embd", n), RAD_STR("dtype", g.dtype)),
                               RAD_WEIGHTS(m.w_lm)),
                     { m.xout }, { m.logits });
    if (!m.op_embed || !m.op_enter || !m.op_gather || !m.op_head_norm || !m.op_logits) return fail(__LINE__);
    if (m.dfly.on) {
        /* the ring entry's h (the head norm's output, lm_head's input), and its position word last */
        m.dfly.op_st_h = rw(b, op_ring_store(b, m, n, 1, "out"), { m.xout }, {});
        m.dfly.op_st_pos = rw(b, op_ring_store(b, m, 1, 0, "pos"), {}, {});
        if (!m.dfly.op_st_h || !m.dfly.op_st_pos) return fail(__LINE__);
        HY_TRY(decl_dfly(b, m));
    }
    /* The tower goes last: its activations share the arena with everything before. */
    if (m.media) HY_TRY(m.vis.declare(b, meta, ctx, nm, g, n, (int)f.n_layer + 64));

    rad_note(b, "Hy4-flash: %lld layers (%lld sparse), MLA %s, %lld experts top-%lld at width %lld, "
                "Engram in %lld layers, tp %lld, BF16",
             (long long)f.n_layer, (long long)std::count(f.sparse.begin(), f.sparse.end(), true),
             m.mla_abs ? "absorbed" : "expanded",
             (long long)f.n_expert, (long long)f.top_k, (long long)f.latent, (long long)E, (long long)g.world);
    return RAD_OK;
}

/* ------------------------------------------------------------------ step */
static void step_mla(RadCtx* c, const RadBatch* batch, const Model& m, const Mla& at, bool selected,
                     const int32_t* win_cu) {
    const AttnBufs& B = m.ab[at.sparse ? 0 : 1];
    const ModelShape& f = m.f;
    const int64_t T = batch->n_tok, S = batch->n_seq, H = at.H, qk = f.nope + f.rope;
    const RadKVGroupBatch* kvb = kv_batch(batch, at.kv);
    at.q_a.step(c, m.a, m.qa.x, T);
    if (at.qa_q8)
        step_rmsnorm_q8(c, at.op_qa_norm, T, m.qa.x, at.w_qa_norm, m.qa);
    else
        RAD_ISSUE(c, at.op_qa_norm, brows(m.qa.x, T), RAD_W(at.w_qa_norm), brows(m.qa.x, T));
    at.q_b.step(c, m.qa, B.q, T);
    at.kv_a.step(c, m.a, m.kva, T);
    const Geom& g = at.sparse ? m.g : m.gw;
    const int64_t L = f.kv_lora, lat_hd = (f.kv_lora + f.rope) / 2;
    if (m.mla_abs) {
        /* the cache row [normed lat | rotated k_rope] is built in place in kva */
        RAD_ISSUE(c, at.op_kv_norm, bcol(m.kva, 0, L, T), RAD_W(at.w_kv_norm), bcol(m.kva, 0, L, T));
        RAD_ISSUE(c, at.op_rope_q, bcol(B.q, 0, H * qk, T), rope_posmc(g, batch, T));
        RAD_ISSUE(c, at.op_rope_k, bcol(m.kva, L, f.rope, T), rope_posmc(g, batch, T));
        /* A step over 2048 rows projects in pieces of 2048 (a spanned step, or any step of a larger
         * max-num-batched-tokens without a span): hy4_mla_project is per row, so the bytes are one
         * launch's, but one launch over 4096 rows measured 3.7 times one over 2048 (profile at 18K,
         * 2026-11-22), which made the span slower than none. Up to 2048 rows: one launch, as before. */
        if (T <= m.piece_rows)
            RAD_ISSUE(c, T > 32 ? at.op_absorb_m : at.op_absorb, bcol(B.q, 0, H * qk, T), RAD_W(at.w_kvb_abs),
                      bcol(B.qlat, 0, H * L, T));
        else
            for (int64_t o = 0; o < T; o += m.piece_rows) {
                const int64_t nr = std::min(m.piece_rows, T - o);
                RAD_ISSUE(c, at.op_absorb_m, bcol_at(B.q, o, H * qk, 0, H * qk, nr), RAD_W(at.w_kvb_abs),
                          bcol_at(B.qlat, o, H * L, 0, H * L, nr));
            }
        RAD_ISSUE(c, at.op_store, bcol(m.kva, 0, lat_hd, T), bcol(m.kva, lat_hd, lat_hd, T),
                  praw(kvb ? kvb->slot_mapping : nullptr, RAD_I32, T), kv_cache(at.kv, at.layer));
    } else {
        RAD_ISSUE(c, at.op_kv_norm, bcol(m.kva, 0, L, T), RAD_W(at.w_kv_norm), brows(m.lat.x, T));
        RAD_ISSUE(c, at.op_rope_q, bcol(B.q, 0, H * qk, T), rope_posmc(g, batch, T));
        RAD_ISSUE(c, at.op_rope_k, bcol(m.kva, L, f.rope, T), rope_posmc(g, batch, T));
        at.kv_b.step(c, m.lat, B.kvx, T);
        RAD_ISSUE(c, at.op_kv, bcol(B.kvx, 0, H * (f.nope + f.v_dim), T), bcol(m.kva, L, f.rope, T),
                  bcol(B.kbuf, 0, H * qk, T), bcol(B.vbuf, 0, H * f.v_dim, T));
        RAD_ISSUE(c, at.op_store, bcol(B.kbuf, 0, H * qk, T), bcol(B.vbuf, 0, H * f.v_dim, T),
                  praw(kvb ? kvb->slot_mapping : nullptr, RAD_I32, T), kv_cache(at.kv, at.layer));
    }

    if (at.sparse) {
        /* indexer: always store keys */
        const RadKVGroupBatch* tlb = kv_batch(batch, m.kv_tail);
        const RadKVGroupBatch* bkb = kv_batch(batch, m.kv_idx);
        at.tk.step(c, m.a, m.tkn.x, T);
        RAD_ISSUE(c, at.op_tk_norm, brows(m.tkn.x, T), RAD_W(at.w_tk_norm), brows(m.tkn.x, T));
        const int64_t tw = tlb && tlb->state_index_pitch > 0 ? tlb->state_index_pitch : 1;
        const int64_t work = (T + f.idx_ratio - 1) / f.idx_ratio + S;
        /* With images (as in rad_qsa.h): qsa_work writes each group's three rotary components into
         * bpos [work, 3]; without planes in the step, only column 0. */
        const bool mc = m.g.rope_mc && batch->rope_pos;
        RadOperand bpos = brows(m.bpos, work);
        if (m.media) {
            bpos = RAD_B_COL(m.bpos, 0, mc ? 3 : 1);
            bpos.rows = work;
        }
        RAD_ISSUE_N(c, at.op_work, T, brows(m.tkn.x, T), praw(batch->positions, RAD_I32, T),
                    praw(batch->cu_seqlens, RAD_I32, S + 1), kv_cache(m.kv_tail, at.layer),
                    praw2(tlb ? tlb->state_index : nullptr, RAD_I32, S, tw),
                    praw2(bkb ? bkb->block_table : nullptr, RAD_I32, S, bkb ? bkb->block_table_pitch : 0),
                    mc ? praw2(batch->rope_pos, RAD_I32, 3, T) : RAD_NONE, brows(m.stage, work * f.idx_ratio),
                    brows(m.page, work), bpos, brows(m.nc, S));
        RAD_ISSUE_N(c, at.op_pool, work, brows(m.stage, work * f.idx_ratio), RAD_W(at.w_ival), RAD_W(at.w_igate),
                    RAD_W(at.w_iape), RAD_W(at.w_inorm), brows(m.page, work), bpos,
                    kv_cache(m.kv_idx, at.layer));
        RAD_ISSUE_N(c, at.op_tail, T, brows(m.tkn.x, T), praw(batch->positions, RAD_I32, T),
                    praw(batch->cu_seqlens, RAD_I32, S + 1), kv_cache(m.kv_tail, at.layer),
                    praw2(tlb ? tlb->state_index : nullptr, RAD_I32, S, tw));
        if (selected) {
            at.iq.step(c, m.qa, m.iq, T);
            at.ihw.step(c, m.a, m.ihw, T);
            const int64_t iqw = f.idx_heads * f.idx_dim, K = f.idx_topk_groups;
            const RadOperand bt_keys = praw2(bkb ? bkb->block_table : nullptr, RAD_I32, S,
                                             bkb ? bkb->block_table_pitch : 0);
            const RadOperand bt_attn = praw2(kvb ? kvb->block_table : nullptr, RAD_I32, S,
                                             kvb ? kvb->block_table_pitch : 0);
            /* Only the columns of the live history (as in QSA, rad_qsa.h): the kernels size their
             * grid by the operand's width. A query has at most (pos+1)/ratio closed groups, and
             * pos < max_ctx_len + n_tok. */
            int64_t live = ((int64_t)batch->max_ctx_len + T + f.idx_ratio) / f.idx_ratio + 1;
            if (live > m.max_groups) live = m.max_groups;
            auto score_select = [&](int64_t w0, int64_t R, RadOperand cu) {
                /* q's rope at the rotary position: 3-D with images (plane covers the step: w0 = 0) */
                const RadOperand qpos = mc && w0 == 0 && R == T ? praw2(batch->rope_pos, RAD_I32, 3, T)
                                                                : praw(batch->positions + w0, RAD_I32, R);
                /* A window's rows end at w0 + R, so they have no closed group past that: the same
                 * bound as `live` with the window's end for the step's (the whole step when the
                 * plane covers it). A spanned step's first windows then score only their own
                 * history, not the groups the later windows close. The select reads columns below a
                 * row's own group (idx_select_k), so the selection is the same. */
                const int64_t lw = std::min<int64_t>(live, ((int64_t)batch->max_ctx_len + w0 + R + f.idx_ratio) /
                                                               f.idx_ratio + 1);
                RAD_ISSUE_N(c, at.op_score, R, brow_slice(m.iq, w0, R, iqw), brow_slice(m.ihw, w0, R, f.idx_heads),
                            qpos, kv_cache(m.kv_idx, at.layer), bt_keys, cu,
                            brows(m.nc, S), bcol_at(m.score, 0, m.max_groups, 0, lw, R));
                RAD_ISSUE_N(c, at.op_select, R, bcol_at(m.score, 0, m.max_groups, 0, lw, R), bt_attn, brows(m.nc, S),
                            praw(batch->positions + w0, RAD_I32, R), cu, brow_slice(m.sel, w0, R, K),
                            brow_slice(m.nsel, w0, R, 1), brow_slice(m.sequ, w0, R, 1));
            };
            /* A step wider than the plane is a single sequence (plane_rows): it is scored by windows
             * from the start, the last one ends at the end of the step and overlaps the one before.
             * Each row reads its own (query, position) and the step's table and nc, so a row in two
             * windows is written twice with the same bytes. */
            const int64_t PR = m.plane_rows;
            if (T > PR) {
                if (S != 1 || !win_cu) {
                    rad_step_fail(c, "hyv4flash: a step wider than the score plane must be a single "
                                     "sequence (RadBuildCtx::max_mixed_tok)");
                    return;
                }
                for (int64_t w0 = 0;;) {
                    score_select(w0, PR, praw(win_cu, RAD_I32, 2));
                    if (w0 + PR >= T) break;
                    w0 = w0 + PR + PR <= T ? w0 + PR : T - PR;
                }
            } else {
                score_select(0, T, praw(batch->cu_seqlens, RAD_I32, S + 1));
            }
            tap(c, batch, m.sel, T, K, 4, "index.sel_pages", at.layer, 0, "i32");
            if (kvb && kvb->block_table)
                tap_raw(c, batch, kvb->block_table, (int64_t)kvb->block_table_pitch * 4, "index.bt", at.layer);
            /* One query per row (q_len 1), no cu: its table is its row of `sel` (as in QSA). */
            const int64_t qp = H * qk, vp = H * f.v_dim;
            for (int64_t off0 = 0; off0 < T; off0 += m.sel_rows) {
                /* hy4_mla_attn picks its form by the launch's rows (form 3 above 32): a step over 32
                 * rows ends with a full window, overlapping the one before, so no tail of 32 rows or
                 * fewer runs on form 2 and a row's bits do not depend on its offset in the step.
                 * Form 3 is per row, so the overlapped rows are written again with the same bits. */
                const int64_t off = T > 32 && off0 + m.sel_rows > T ? std::max<int64_t>(T - m.sel_rows, 0) : off0;
                const int64_t rows = std::min(m.sel_rows, T - off);
                if (m.mla_abs) {
                    RAD_ISSUE_N(c, at.op_attn_sel, rows, bcol_at(B.qlat, off, H * L, 0, H * L, rows),
                                bcol_at(B.q, off, qp, 0, H * qk, rows), kv_cache(at.kv, at.layer),
                                brow_slice(m.sel, off, rows, K), brow_slice(m.sequ, off, rows, 1), RAD_NONE,
                                RAD_W(at.w_sink), bcol_at(B.olat, off, H * L, 0, H * L, rows));
                    continue;
                }
                RAD_ISSUE_N(c, at.op_attn_sel, 1, bcol_at(B.q, off, qp, 0, H * qk, rows), kv_cache(at.kv, at.layer),
                            brow_slice(m.sel, off, rows, K), brow_slice(m.sequ, off, rows, 1), RAD_NONE, RAD_NONE,
                            RAD_NONE, RAD_W(at.w_sink), bcol_at(B.o.x, off, vp, 0, H * f.v_dim, rows));
            }
        }
    }
    if (at.sparse && selected) {
    } else if (m.mla_abs) {
        RAD_ISSUE_N(c, at.op_attn, T, bcol(B.qlat, 0, H * L, T), bcol(B.q, 0, H * qk, T), kv_cache(at.kv, at.layer),
                    praw2(kvb ? kvb->block_table : nullptr, RAD_I32, S, kvb ? kvb->block_table_pitch : 0),
                    praw(kvb ? kvb->seqused : nullptr, RAD_I32, S), praw(batch->cu_seqlens, RAD_I32, S + 1),
                    RAD_W(at.w_sink), bcol(B.olat, 0, H * L, T));
    } else {
        RAD_ISSUE_N(c, at.op_attn, batch->max_q_len, bcol(B.q, 0, H * qk, T), kv_cache(at.kv, at.layer),
                    praw2(kvb ? kvb->block_table : nullptr, RAD_I32, S, kvb ? kvb->block_table_pitch : 0),
                    praw(kvb ? kvb->seqused : nullptr, RAD_I32, S), RAD_NONE, RAD_NONE,
                    praw(batch->cu_seqlens, RAD_I32, S + 1), RAD_W(at.w_sink), bcol(B.o.x, 0, H * f.v_dim, T));
    }
    if (m.mla_abs && T <= m.piece_rows)
        RAD_ISSUE(c, T > 32 ? at.op_expand_m : at.op_expand, bcol(B.olat, 0, H * L, T), RAD_W(at.w_kvb_abs),
                  bcol(B.o.x, 0, H * f.v_dim, T));
    else if (m.mla_abs)   /* in pieces, as the absorb above */
        for (int64_t o = 0; o < T; o += m.piece_rows) {
            const int64_t nr = std::min(m.piece_rows, T - o);
            RAD_ISSUE(c, at.op_expand_m, bcol_at(B.olat, o, H * L, 0, H * L, nr), RAD_W(at.w_kvb_abs),
                      bcol_at(B.o.x, o, H * f.v_dim, 0, H * f.v_dim, nr));
        }
    at.gate.step(c, m.a, B.gt, T);
    if (at.op_gq) {
        /* a row a (token, head): M is T * H, the operands the whole [T, H * v_dim] planes */
        RAD_ISSUE_N(c, at.op_gq, T * H, brows(B.gt, T), brows(B.o.x, T), brows(B.o.q8, T), brows(B.o.s8, T),
                    RAD_NONE);
    } else {
        RAD_ISSUE(c, at.op_sig, bcol(B.gt, 0, H * f.v_dim, T), bcol(B.gt, 0, H * f.v_dim, T));
        RAD_ISSUE(c, at.op_mul, bcol(B.o.x, 0, H * f.v_dim, T), bcol(B.gt, 0, H * f.v_dim, T),
                  bcol(B.o.x, 0, H * f.v_dim, T));
    }
    at.o.step(c, B.o, m.y, T);
    if (at.op_ar) RAD_ISSUE_N(c, at.op_ar, T * f.n_embd, brows(m.y, T), RAD_NONE);
    RAD_ISSUE(c, at.op_post, brows(m.y, T), RAD_W(at.w_post), brows(m.y, T));
}

/* ------------------------------------------------------------------ DFly2: step (docs/DFLY2.md) */
/* The ring operands every store and load shares; false if the batch carries no kv_dfly slots. */
struct RingOps {
    RadOperand pos, cu, sidx, state;
};
static bool ring_ops(const RadBatch* batch, const Model& m, RingOps* r) {
    const Dfly& df = m.dfly;
    const RadKVGroupBatch* kb = kv_batch(batch, df.kv);
    if (!kb || !kb->state_index || !batch->positions || !batch->cu_seqlens) return false;
    const int64_t tw = kb->state_index_pitch > 0 ? kb->state_index_pitch : 1;
    r->pos = praw(batch->positions, RAD_I32, batch->n_tok);
    r->cu = praw(batch->cu_seqlens, RAD_I32, batch->n_seq + 1);
    r->sidx = praw2(kb->state_index, RAD_I32, batch->n_seq, tw);
    r->state = kv_cache(df.kv, df.targets[0]);
    return true;
}

/* Trunk step, right after layer t_d's attention: drafter layer d's history for every logit row,
 * while idx_sel and idx_sequ still hold this layer's selection (the next sparse layer overwrites
 * them). A step without selection stores the dense form from kv_full's block table. */
static void dfly_store_hist(RadCtx* c, const RadBatch* batch, const Model& m, int d, bool selected) {
    const DflyLayer& D = m.dfly.layers[(size_t)d];
    const int64_t T = batch->n_tok, S = batch->n_seq, R = batch->n_out;
    RingOps r;
    if (R <= 0 || !batch->out_ids || !ring_ops(batch, m, &r)) return;
    const RadOperand ids = praw(batch->out_ids, RAD_I32, R);
    if (selected) {
        RAD_ISSUE_N(c, D.op_st_sequ, R, brows(m.sequ, T), ids, r.pos, r.cu, r.sidx, r.state);
        RAD_ISSUE_N(c, D.op_st_sel, R, brows(m.sel, T), ids, r.pos, r.cu, r.sidx, r.state);
        return;
    }
    const RadKVGroupBatch* kvb = kv_batch(batch, m.kv_full);
    if (!kvb || !kvb->block_table) return;
    RAD_ISSUE_N(c, D.op_st_dense, R, praw2(kvb->block_table, RAD_I32, S, kvb->block_table_pitch), ids, r.pos, r.cu,
                r.sidx, r.state);
}

/* Trunk step, after the head norm: h, then the position word that makes the entry valid. */
static void dfly_store_tail(RadCtx* c, const RadBatch* batch, const Model& m) {
    const Dfly& df = m.dfly;
    const int64_t R = batch->n_out;
    RingOps r;
    if (R <= 0 || !batch->out_ids || !ring_ops(batch, m, &r)) return;
    const RadOperand ids = praw(batch->out_ids, RAD_I32, R);
    RAD_ISSUE_N(c, df.op_st_h, R, brows(m.xout, R), ids, r.pos, r.cu, r.sidx, r.state);
    RAD_ISSUE_N(c, df.op_st_pos, R, RAD_NONE, ids, r.pos, r.cu, r.sidx, r.state);
}

/* A drafter layer's attention: the trunk's sparse attention in the absorbed selection form, its
 * history the ring's (dfly_hist) and its block's own keys (bkeys = kva, bcu = cu_seqlens), over
 * trunk layer t_d's kv_full, which it only reads. */
static void step_dfly_attn(RadCtx* c, const RadBatch* batch, const Model& m, const DflyLayer& D, int d) {
    const Dfly& df = m.dfly;
    const Mla& at = D.at;
    const AttnBufs& B = df.B;
    const ModelShape& f = m.f;
    const Geom& g = df.g;
    const int64_t T = batch->n_tok, S = batch->n_seq, H = at.H, qk = f.nope + f.rope, L = f.kv_lora;
    const int64_t K = f.idx_topk_groups, h0 = (int64_t)d * (1 + K);
    at.q_a.step(c, df.a, df.qa.x, T);
    RAD_ISSUE(c, at.op_qa_norm, brows(df.qa.x, T), RAD_W(at.w_qa_norm), brows(df.qa.x, T));
    at.q_b.step(c, df.qa, B.q, T);
    at.kv_a.step(c, df.a, df.kva, T);
    RAD_ISSUE(c, at.op_kv_norm, bcol(df.kva, 0, L, T), RAD_W(at.w_kv_norm), bcol(df.kva, 0, L, T));
    RAD_ISSUE(c, at.op_rope_q, bcol(B.q, 0, H * qk, T), rope_posmc(g, batch, T));
    RAD_ISSUE(c, at.op_rope_k, bcol(df.kva, L, f.rope, T), rope_posmc(g, batch, T));
    RAD_ISSUE(c, T > 32 ? at.op_absorb_m : at.op_absorb, bcol(B.q, 0, H * qk, T), RAD_W(at.w_kvb_abs),
              bcol(B.qlat, 0, H * L, T));
    /* operands: q, q_rope, kv_cache, block_table, seqused, cu_seqlens?, sinks, out, bkeys, bcu */
    RAD_ISSUE_N(c, at.op_attn_sel, T, bcol(B.qlat, 0, H * L, T), bcol(B.q, 0, H * qk, T), kv_cache(m.kv_full, D.target),
                bcol(df.hist, h0 + 1, K, T), bcol(df.hist, h0, 1, T), RAD_NONE, RAD_W(at.w_sink),
                bcol(B.olat, 0, H * L, T), brows(df.kva, T), praw(batch->cu_seqlens, RAD_I32, S + 1));
    RAD_ISSUE(c, T > 32 ? at.op_expand_m : at.op_expand, bcol(B.olat, 0, H * L, T), RAD_W(at.w_kvb_abs),
              bcol(B.o.x, 0, H * f.v_dim, T));
    at.gate.step(c, df.a, B.gt, T);
    RAD_ISSUE(c, at.op_sig, bcol(B.gt, 0, H * f.v_dim, T), bcol(B.gt, 0, H * f.v_dim, T));
    RAD_ISSUE(c, at.op_mul, bcol(B.o.x, 0, H * f.v_dim, T), bcol(B.gt, 0, H * f.v_dim, T), bcol(B.o.x, 0, H * f.v_dim, T));
    at.o.step(c, B.o, df.y, T);
    if (at.op_ar) RAD_ISSUE_N(c, at.op_ar, T * f.n_embd, brows(df.y, T), RAD_NONE);
    RAD_ISSUE(c, at.op_post, brows(df.y, T), RAD_W(at.w_post), brows(df.y, T));
}

/* The query pass (RadBatch::draft_pass > 0): every drafting sequence's block [b, MASK x 5] at
 * positions p0 .. p0 + 5, `block` rows each, in sequence order; its proposals to dfly_proposal. */
static void step_dfly(RadCtx* c, const RadBatch* batch, const Model& m) {
    const Dfly& df = m.dfly;
    const int64_t T = batch->n_tok, S = batch->n_seq, n = m.f.n_embd;
    if (S <= 0) return;
    /* Slot k of sequence s is row cu_seqlens[s] + k: every sequence carries a whole block. Only
     * the total is checked here: cu_seqlens is device memory (abi/rad_runtime.h), rewritten every
     * step at one address so a recorded pass replays it, and on a step issued ahead of the one
     * still running it does not yet hold this step's spans. Reading it would need a copy and a
     * wait on the stream, on every query pass. The scheduler builds the pass from whole blocks
     * (RAD_DRAFT_BLOCK); a span of another length makes the drafter read the wrong rows and
     * propose garbage, which the verify rejects (acceptance, never a token). */
    if (T != S * df.block || T > df.rows || S > m.g.max_seqs) {
        rad_step_fail(c, "hyv4flash: a DFly2 query pass must carry whole blocks of 6 rows, within max_seqs");
        return;
    }
    RingOps r;
    if (!ring_ops(batch, m, &r) || !batch->token_ids) {
        rad_step_fail(c, "hyv4flash: the DFly2 query pass carries no kv_dfly slots");
        return;
    }
    const RadOperand slot0 = praw(batch->cu_seqlens, RAD_I32, S);
    /* 1. the block's embedding; the ring entry of p0 - 1 in every row of the sequence */
    RAD_ISSUE(c, df.op_embed, praw(batch->token_ids, RAD_I32, T), RAD_W(m.w_token), brows(df.x, T));
    if (df.op_embed_ar) RAD_ISSUE_N(c, df.op_embed_ar, T * n, brows(df.x, T), RAD_NONE);
    RAD_ISSUE_N(c, df.op_load_h, T, r.pos, r.cu, r.sidx, r.state, brows(df.hf, T));
    RAD_ISSUE_N(c, df.op_load_hist, T, r.pos, r.cu, r.sidx, r.state, brows(df.hist, T));
    RAD_ISSUE_N(c, df.op_cast_h, T, brows(df.hf, T), brows(df.hb, T));
    /* 2. slot 0: e0 + gate · eh_proj([enorm(e0) | hnorm(h)]) */
    RAD_ISSUE_N(c, df.op_g_e0, S, brows(df.x, T), slot0, brows(df.e0, S));
    RAD_ISSUE_N(c, df.op_g_h0, S, brows(df.hb, T), slot0, brows(df.h0, S));
    RAD_ISSUE_N(c, df.op_enorm, S, brows(df.e0, S), RAD_W(df.w_enorm), bcol(df.cat.x, 0, n, S));
    RAD_ISSUE_N(c, df.op_hnorm, S, brows(df.h0, S), RAD_W(df.w_hnorm), bcol(df.cat.x, n, n, S));
    df.eh.step(c, df.cat, df.fz, S);
    RAD_ISSUE_N(c, df.op_gate, S, brows(df.fz, S), RAD_W(df.w_gate), brows(df.fz, S));
    RAD_ISSUE_N(c, df.op_fuse, S, brows(df.fz, S), slot0, brows(df.x, T));
    /* 3. five layers */
    for (size_t d = 0; d < df.layers.size(); ++d) {
        const DflyLayer& D = df.layers[d];
        RAD_ISSUE(c, D.op_in_norm, brows(df.x, T), RAD_W(D.w_in_norm), brows(df.a.x, T));
        step_dfly_attn(c, batch, m, D, (int)d);
        RAD_ISSUE(c, D.op_add_attn, brows(df.x, T), brows(df.y, T), brows(df.x, T));
        RAD_ISSUE(c, D.op_ffn_norm, brows(df.x, T), RAD_W(D.w_ffn_norm), brows(df.a.x, T));
        D.gu.step(c, df.a, df.gu, T);
        RAD_ISSUE(c, D.op_act, brows(df.gu, T), brows(df.ff.x, T));
        D.down.step(c, df.ff, df.y, T);
        RAD_ISSUE(c, D.op_add_ffn, brows(df.x, T), brows(df.y, T), brows(df.x, T));
    }
    /* 5. the final norm */
    RAD_ISSUE(c, df.op_final, brows(df.x, T), RAD_W(df.w_final), brows(df.yn, T));
    /* 6. the sequential steps: slot k's row of every sequence, then the head's argmax */
    for (int64_t k = 1; k <= df.depth; ++k) {
        const DflyStep& C = df.steps[(size_t)(k - 1)];
        RAD_ISSUE_N(c, C.op_gy, S, bcol_at(df.yn, k, n, 0, n, T - k), slot0, brows(df.yk, S));
        if (k > 1) {
            RAD_ISSUE_N(c, C.op_embed, S, brows(df.prev, S), RAD_W(m.w_token), brows(df.ep, S));
            if (C.op_embed_ar) RAD_ISSUE_N(c, C.op_embed_ar, S * n, brows(df.ep, S), RAD_NONE);
        }
        RAD_ISSUE_N(c, C.op_nh, S, brows(df.yk, S), RAD_W(df.w_cnh), bcol(df.cat.x, 0, n, S));
        RAD_ISSUE_N(c, C.op_ne, S, brows(k > 1 ? df.ep : df.e0, S), RAD_W(df.w_cne), bcol(df.cat.x, n, n, S));
        C.gu.step(c, df.cat, df.cgu, S);
        RAD_ISSUE_N(c, C.op_act, S, brows(df.cgu, S), brows(df.cff.x, S));
        C.down.step(c, df.cff, df.cd, S);
        RAD_ISSUE_N(c, C.op_add, S, brows(df.yk, S), brows(df.cd, S), brows(df.cc, S));
        RAD_ISSUE_N(c, C.op_head, S, brows(df.cc, S), RAD_W(m.w_lm), brows(df.logits, S));
        RAD_ISSUE_N(c, C.op_topk, S, brows(df.logits, S), brows(df.cand, S), brows(df.cval, S), brows(df.pairs, S));
        if (C.op_gath) RAD_ISSUE(c, C.op_gath, brows(df.pairs, S), brows(df.gath, S));
        const rad_buf pp = C.op_gath ? df.gath : df.pairs;
        RAD_ISSUE_N(c, C.op_prev, S, brows(pp, S), brows(df.prev, S), RAD_NONE);
        RAD_ISSUE_N(c, C.op_prop, S, brows(pp, S), bcol(df.proposal, k - 1, 1, S), RAD_NONE);
    }
}

/* win_cu of a step wider than the score plane; null if the step fits or there is no pinned copy. */
static const int32_t* window_bounds(RadCtx* c, const RadBatch* batch, Model& m) {
    if (batch->n_tok <= m.plane_rows || !m.win_cu) return nullptr;
    if (!m.win_cu_tried) {
        m.win_cu_tried = true;
        m.win_cu_h = (int32_t*)rad_dev_alloc(2 * sizeof(int32_t), RAD_MEM_HOST_PINNED);
        if (m.win_cu_h) {
            m.win_cu_h[0] = 0;
            m.win_cu_h[1] = (int32_t)m.plane_rows;
        } else {
            std::fprintf(stderr, "radiance: hyv4flash: no pinned memory for the score windows\n");
        }
    }
    if (!m.win_cu_h) return nullptr;
    int32_t* d = (int32_t*)rad_buf_ptr(c, m.win_cu);
    if (rad_memcpy_async(d, m.win_cu_h, 2 * sizeof(int32_t), rad_stream(c)) != RAD_OK) return nullptr;
    return d;
}

static void step(RadCtx* c, const RadBatch* batch) {
    Model& m = g_model[rad_slot(c)];
    const ModelShape& f = m.f;
    /* An encoder pass runs only the tower (RadBatch::enc). */
    if (batch->enc) {
        m.vis.step(c, batch);
        return;
    }
    /* A drafter pass runs only the DFly2 drafter: positive, the query pass proposing the block;
     * negative, the context pass, which has nothing to do (the drafter keeps no cache: it reads the
     * trunk's, and the ring the trunk step filled). */
    if (batch->draft_pass != 0) {
        if (!m.dfly.on) {
            rad_step_fail(c, "hyv4flash: a drafter pass, and this build declares no drafter");
            return;
        }
        if (batch->draft_pass > 0) step_dfly(c, batch, m);
        return;
    }
    /* Speculation is a chain verify within the declared depth: the n-gram window and the indexer
     * ring are sized for max_spec rows past the committed one, and the window's write-back assumes
     * each row follows the one before (no tree). */
    if (batch->n_spec > m.max_spec || batch->spec_parent) {
        rad_step_fail(c, "hyv4flash: this step verifies more drafts than declared, or a tree");
        return;
    }
    const int64_t T = batch->n_tok, S = batch->n_seq, n = f.n_embd;
    RAD_ISSUE(c, m.op_embed, praw(batch->token_ids, RAD_I32, T), RAD_W(m.w_token), brows(m.xin, T));
    if (m.media) m.mrows.step(c, batch, m.xin, T);
    if (m.op_embed_ar) RAD_ISSUE_N(c, m.op_embed_ar, T * n, brows(m.xin, T), RAD_NONE);
    RAD_ISSUE(c, m.op_enter, brows(m.xin, T), brows(m.h, T));
    if (m.op_ids) {
        const RadKVGroupBatch* tk = kv_batch(batch, m.kv_tok);
        const int64_t tw = tk && tk->state_index_pitch > 0 ? tk->state_index_pitch : 1;
        /* The next prompt chunk after the step (in token_ids past n_tok), one chunk at most: the row
         * cache holds about four chunks of one table's rows, and libavx hands over a quarter of it. */
        const int64_t A = std::min<int64_t>(batch->n_ahead, m.g.max_tok);
        const int64_t EM = (int64_t)f.engram_layers.size() * m.g.max_tok;
        /* A verify step's decode sequences read their n-gram window at num_accepted - 1 (their drafts
         * may be rejected); its prefill chunks commit every token and carry the read-ahead. Two
         * launches over the two sub-ranges (batch_split: decode sequences come first), rows absolute
         * through cu; the decode launch covers only the decode rows. */
        int64_t Sd = 0, Td = 0;
        if (batch->num_accepted) batch_split(batch, &Sd, &Td);
        Sd = std::min<int64_t>(Sd, S);
        const int32_t* sidx = tk ? tk->state_index : nullptr;
        if (Sd > 0)
            RAD_ISSUE_N(c, m.op_ids, Td, praw(batch->token_ids, RAD_I32, Td), praw(batch->positions, RAD_I32, Td),
                        praw(batch->cu_seqlens, RAD_I32, Sd + 1), kv_cache(m.kv_tok, f.engram_layers[0]),
                        praw2(sidx, RAD_I32, Sd, tw), RAD_W(m.w_map), RAD_W(m.w_mult), RAD_W(m.w_vsz), RAD_W(m.w_off),
                        brows(m.eids, EM), RAD_NONE, praw(batch->num_accepted, RAD_I32, Sd));
        if (S > Sd)
            RAD_ISSUE_N(c, m.op_ids, T, praw(batch->token_ids, RAD_I32, T + A), praw(batch->positions, RAD_I32, T),
                        praw(batch->cu_seqlens + Sd, RAD_I32, S - Sd + 1), kv_cache(m.kv_tok, f.engram_layers[0]),
                        praw2(sidx ? sidx + Sd * tw : nullptr, RAD_I32, S - Sd, tw), RAD_W(m.w_map), RAD_W(m.w_mult),
                        RAD_W(m.w_vsz), RAD_W(m.w_off), brows(m.eids, EM), A > 0 ? brows(m.eids_ahead, EM) : RAD_NONE,
                        RAD_NONE);
    }
    /* Indexer selection: only if some context in the step exceeds candidate_topk tokens. */
    const bool selected = (int64_t)batch->max_ctx_len + batch->max_q_len > f.idx_candidate;
    const int32_t* win_cu = selected ? window_bounds(c, batch, m) : nullptr;
    for (size_t li = 0; li < m.layers.size(); ++li) {
        const Layer& L = m.layers[li];
        const int l = (int)li;
        if (L.engram >= 0) {
            const Engram& en = m.engram[(size_t)L.engram];
            const int64_t H = f.engram_heads, w = H * f.engram_d;
            const int64_t A = std::min<int64_t>(batch->n_ahead, m.g.max_tok);
            RAD_ISSUE_N(c, en.op_lookup, T * H,
                        brow_slice(m.eids, (int64_t)en.e * m.g.max_tok, T, H), RAD_W(en.w_table),
                        en.w_table_scale ? RAD_W(en.w_table_scale) : RAD_NONE,   /* bf16 table: no scale */
                        brows(m.erows, T),
                        A > 0 ? brow_slice(m.eids_ahead, (int64_t)en.e * m.g.max_tok, A, H) : RAD_NONE);
            RAD_ISSUE(c, en.op_cast, brows(m.erows, T), brows(m.erows_dev, T));
            tap(c, batch, m.h, T, f.hc * n, 2, "engram.in", l);
            tap(c, batch, m.eids, T, H, 4, "engram.ids", l, (int64_t)en.e * m.g.max_tok, "i32");
            RAD_ISSUE(c, en.op_norm, brows(m.erows_dev, T), RAD_W(en.w_norms), brows(m.enorm, T));
            tap(c, batch, m.enorm, T, w, 2, "engram.rows_norm", l);
            RAD_ISSUE(c, en.op_val, brows(m.enorm, T), RAD_W(en.w_val), RAD_W(en.w_val_b), RAD_NONE, brows(m.ev, T));
            RAD_ISSUE(c, en.op_key, brows(m.enorm, T), RAD_W(en.w_keys), RAD_W(en.w_keys_b), RAD_NONE, brows(m.ek, T));
            RAD_ISSUE(c, en.op_gate, brows(m.h, T), praw(batch->token_ids, RAD_I32, T), brows(m.ek, T), brows(m.ev, T),
                      RAD_W(en.w_nk), RAD_W(en.w_nq), RAD_W(en.w_gamma), RAD_W(en.w_per));
        }
        tap(c, batch, m.h, T, f.hc * n, 2, "layer.in", l);
        RAD_ISSUE(c, L.hc_attn.op, brows(m.h, T), RAD_W(L.hc_attn.fn), RAD_W(L.hc_attn.base), RAD_W(L.hc_attn.scale),
                  brows(m.xin, T), brows(m.post, T));
        tap(c, batch, m.xin, T, n, 2, "ihc_attn.mix", l);
        tap(c, batch, m.post, T, f.hc, 4, "ihc_attn.post", l);
        if (L.in_q8)
            step_rmsnorm_q8(c, L.op_in_norm, T, m.xin, L.w_in_norm, m.a);
        else
            RAD_ISSUE(c, L.op_in_norm, brows(m.xin, T), RAD_W(L.w_in_norm), brows(m.a.x, T));
        step_mla(c, batch, m, L.attn, selected, win_cu);
        if (L.dfly >= 0) dfly_store_hist(c, batch, m, L.dfly, selected);
        tap(c, batch, m.y, T, n, 2, "attn.out", l);
        RAD_ISSUE(c, L.op_post_attn, brows(m.h, T), brows(m.y, T), brows(m.post, T));
        tap(c, batch, m.h, T, f.hc * n, 2, "layer.after_attn", l);
        RAD_ISSUE(c, L.hc_ffn.op, brows(m.h, T), RAD_W(L.hc_ffn.fn), RAD_W(L.hc_ffn.base), RAD_W(L.hc_ffn.scale),
                  brows(m.xin, T), brows(m.post, T));
        tap(c, batch, m.xin, T, n, 2, "ihc_ffn.mix", l);
        tap(c, batch, m.post, T, f.hc, 4, "ihc_ffn.post", l);
        if (L.ffn.norm_q8)
            step_rmsnorm_q8(c, L.ffn.op_norm, T, m.xin, L.ffn.w_norm, m.a);
        else
            RAD_ISSUE(c, L.ffn.op_norm, brows(m.xin, T), RAD_W(L.ffn.w_norm), brows(m.a.x, T));
        const Ffn& ff = L.ffn;
        if (!ff.moe) {
            ff.gu.step(c, m.a, m.gu, T);
            ff.act.step(c, m.gu, T);
            ff.down.step(c, m.ff, m.y, T);
        } else {
            ff.lat_in.step(c, m.a, m.latm.x, T);
            ff.lat_q.step(c, T);
            ff.moe_blk.step(c, batch);
            tap(c, batch, m.mw.ids, T, f.top_k, 4, "router.experts", l, 0, "i32");
            tap(c, batch, m.mw.ew, T, f.top_k, 4, "router.weights", l);
            ff.lat_out.step(c, m.latm, m.routed, T);
            if (ff.op_ar) RAD_ISSUE_N(c, ff.op_ar, T * n, brows(m.routed, T), RAD_NONE);
            ff.gu.step(c, m.a, m.gu_s, T);
            ff.act.step(c, m.gu_s, T);
            ff.down.step(c, m.ff_s, m.sout, T);
            RAD_ISSUE(c, ff.op_join, brows(m.routed, T), brows(m.sout, T), brows(m.y, T));
        }
        RAD_ISSUE(c, ff.op_out_norm, brows(m.y, T), RAD_W(ff.w_out_norm), brows(m.y, T));
        tap(c, batch, m.y, T, n, 2, "ffn.out", l);
        RAD_ISSUE(c, L.op_post_ffn, brows(m.h, T), brows(m.y, T), brows(m.post, T));
        tap(c, batch, m.h, T, f.hc * n, 2, "layer.out", l);
    }
    if (batch->n_out > 0) {
        const int64_t R = batch->n_out;
        RAD_ISSUE_N(c, m.op_gather, R, brows(m.h, T), praw(batch->out_ids, RAD_I32, R), brows(m.hout, R));
        RAD_ISSUE_N(c, m.hc_head.op, R, brows(m.hout, R), RAD_W(m.hc_head.fn), RAD_W(m.hc_head.base),
                    RAD_W(m.hc_head.scale), brows(m.xout, R), RAD_NONE);
        RAD_ISSUE_N(c, m.op_head_norm, R, brows(m.xout, R), RAD_W(m.w_norm), brows(m.xout, R));
        RAD_ISSUE_N(c, m.op_logits, R, brows(m.xout, R), RAD_W(m.w_lm), brows(m.logits, R));
        if (m.dfly.on) dfly_store_tail(c, batch, m);
    }
}

/* What the engine asks before declaring: declare() honours shape_probe (g_probe, above). */
static int probe(const RadModelMeta* meta, RadArchProbe* out) {
    if (!meta || !out) return RAD_E_INVAL;
    out->shape_probe_ok = 1;
    /* Chained launches (RadArchProbe::chain_pays): a decode step issues ~1750 launches, and chained
     * they ran 33.9 -> 35.9 tok/s single stream and 236 -> 245 at 8 clients (A/B/A on the GPTQ
     * container at tp 2; prefill at 5.7K unchanged). HY4_CHAIN=0 turns them off, to compare. */
    const char* chain = std::getenv("HY4_CHAIN");
    if (!(chain && chain[0] == '0')) out->chain_pays = RAD_CHAIN_ALWAYS;
    /* The tower takes whole merge x merge blocks (vision.h): with --mm-max-patches auto the engine
     * rounds its pass to a multiple of this. */
    if (Vision::is_present(meta)) {
        const int64_t k = rad_meta_geti(meta, "vision_config.spatial_merge_size", 0);
        if (k > 0) out->enc_merge = k * k;
    }
    return RAD_OK;
}

}  // namespace hyv4flash

extern "C" const char* rad_arch_id(void) { return "hyv4flash"; }
RAD_ARCH_MULTI_INSTANCE()
RAD_ARCH_PROBE(hyv4flash)
extern "C" const char* rad_arch_quant(void) { return ""; }
extern "C" int rad_arch_declare(RadBuilder* b, const RadModelMeta* m, const RadBuildCtx* c) {
    return hyv4flash::declare(b, m, c);
}
extern "C" void rad_arch_step(struct RadCtx* c, const struct RadBatch* batch) {
    hyv4flash::step(c, batch);
}
extern "C" uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
RAD_PLUGIN_ABI_LAYOUT_EXPORT
extern "C" const RadPluginInfo* rad_plugin_info(void) {
    static const RadPluginInfo info = {
        RAD_PLUGIN_ARCH, "hyv4flash", "0.1.0",
        "Tencent Hy4-flash: iHC, MLA with sink, compressed indexer, Engram, latent MoE", ""
    };
    return &info;
}
