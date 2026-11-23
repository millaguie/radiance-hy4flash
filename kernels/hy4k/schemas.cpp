/* schemas.cpp -- schemas of Hy4-flash's new ops and their host rows (docs/OPS-HY4.md). */
#include "hy4k.h"

int hy4_ref_hc_pre(const RadArgs* a, RadStream s);
int hy4_ref_hc_post(const RadArgs* a, RadStream s);
int hy4_ref_mla_kv(const RadArgs* a, RadStream s);
int hy4_ref_mla_attn(const RadArgs* a, RadStream s);
int hy4_ref_ngram_ids(const RadArgs* a, RadStream s);
int hy4_ref_rows_rmsnorm(const RadArgs* a, RadStream s);
int hy4_ref_engram_gate(const RadArgs* a, RadStream s);
int hy4_ref_idx_pool(const RadArgs* a, RadStream s);
int hy4_ref_idx_score(const RadArgs* a, RadStream s);
int hy4_ref_idx_select(const RadArgs* a, RadStream s);
int hy4_ref_had_quant_act_fp8(const RadArgs* a, RadStream s);
int hy4_ref_vit_compact(const RadArgs* a, RadStream s);
int hy4_ref_vit_pos(const RadArgs* a, RadStream s);
int hy4_ref_vit_newline(const RadArgs* a, RadStream s);
int hy4_ref_rope(const RadArgs* a, RadStream s);
int hy4_ref_mla_project(const RadArgs* a, RadStream s);
int hy4_ref_ring_store(const RadArgs* a, RadStream s);
int hy4_ref_ring_load(const RadArgs* a, RadStream s);
#ifdef HY4K_WITH_DEVICE
int hy4_dev_had_quant_act_fp8(const RadArgs* a, RadStream s);
int hy4_dev_hc_pre(const RadArgs* a, RadStream s);
int64_t hy4_dev_hc_pre_scratch(const RadArgs* a);
int hy4_dev_hc_post(const RadArgs* a, RadStream s);
int hy4_dev_mla_kv(const RadArgs* a, RadStream s);
int hy4_dev_ngram_ids(const RadArgs* a, RadStream s);
int hy4_dev_rows_rmsnorm(const RadArgs* a, RadStream s);
int hy4_dev_engram_gate(const RadArgs* a, RadStream s);
int hy4_dev_idx_pool(const RadArgs* a, RadStream s);
int hy4_dev_idx_score(const RadArgs* a, RadStream s);
int hy4_dev_idx_select(const RadArgs* a, RadStream s);
int hy4_dev_attn_sink(const RadArgs* a, RadStream s);
int hy4_dev_mla_attn(const RadArgs* a, RadStream s);
int64_t hy4_dev_mla_attn_scratch(const RadArgs* a);
int hy4_dev_vit_compact(const RadArgs* a, RadStream s);
int hy4_dev_vit_pos(const RadArgs* a, RadStream s);
int hy4_dev_vit_newline(const RadArgs* a, RadStream s);
int hy4_dev_rope(const RadArgs* a, RadStream s);
int hy4_dev_mla_project(const RadArgs* a, RadStream s);
int hy4_dev_ring_store(const RadArgs* a, RadStream s);
int hy4_dev_ring_load(const RadArgs* a, RadStream s);
#endif

namespace {

#define P_INT(k) { k, RAD_P_INT, RAD_REQUIRED, 0 }
#define P_OPT(k) { k, RAD_P_INT, RAD_OPTIONAL, 0 }
#define P_F64(k) { k, RAD_P_F64, RAD_OPTIONAL, 0 }
#define P_STR(k) { k, RAD_P_STR, RAD_REQUIRED, 0 }
#define IN(k)    { k, RAD_OPD_IN, 0 }
#define INOUT(k) { k, RAD_OPD_INOUT, 0 }
#define OUT(k)   { k, RAD_OPD_OUT, 0 }
#define OUTO(k)  { k, RAD_OPD_OUT, 1 }
#define W(k)     { k, RAD_OPD_WEIGHT, 0 }

const RadParamSpec pHcPre[] = { P_INT("M"), P_INT("n"), P_INT("hc"), P_OPT("head"), P_F64("eps"),
                                P_F64("hc_eps"), P_F64("magnitude"), P_STR("dtype") };
const RadOperandSpec oHcPre[] = { IN("h"), W("fn"), W("base"), W("scale"), OUT("x"), OUTO("post") };
const RadParamSpec pHcPost[] = { P_INT("M"), P_INT("n"), P_INT("hc"), P_STR("dtype") };
const RadOperandSpec oHcPost[] = { INOUT("h"), IN("y"), IN("post") };
const RadParamSpec pMlaKv[] = { P_INT("M"), P_INT("heads"), P_INT("nope"), P_INT("rope"), P_INT("v_dim"),
                                P_STR("dtype") };
const RadOperandSpec oMlaKv[] = { IN("kvx"), IN("k_rope"), OUT("k"), OUT("v") };
#define INO(k)   { k, RAD_OPD_IN, 1 }
const RadParamSpec pMlaAttn[] = { P_INT("M"), P_INT("heads"), P_INT("latent"), P_INT("rope"), P_INT("block_size"),
                                  P_OPT("window"), P_F64("scale"), P_OPT("rope_pitch"), P_OPT("max_seqs"), P_OPT("max_ctx"),
                                  P_STR("dtype") };
const RadOperandSpec oMlaAttn[] = { IN("q"), IN("q_rope"), IN("kv_cache"), IN("block_table"), IN("seqused"),
                                    INO("cu_seqlens"), W("sinks"), OUT("out"),
                                    INO("bkeys"), INO("bcu") };
const RadParamSpec pNgram[] = { P_INT("M"), P_INT("engram_layers"), P_INT("heads"), P_INT("ngram"),
                                P_INT("pad"), P_INT("seqs") };
const RadOperandSpec oNgram[] = { IN("tok"), IN("pos"), IN("cu"), INOUT("state"), IN("sidx"), W("map"),
                                  W("mult"), W("vocab_sizes"), W("offsets"), OUT("ids"), OUTO("ahead_ids"), INO("num_accepted") };
const RadParamSpec pRowsNorm[] = { P_INT("M"), P_INT("heads"), P_INT("d"), P_OPT("fp8"), P_F64("eps"),
                                   P_STR("dtype") };
const RadOperandSpec oRowsNorm[] = { IN("x"), W("w"), OUT("y") };
const RadParamSpec pEngGate[] = { P_INT("M"), P_INT("n"), P_INT("hc"), P_OPT("pad"), P_OPT("skip"), P_OPT("skip2"),
                                   P_F64("eps"),
                                  P_F64("gate_eps"), P_STR("dtype") };
const RadOperandSpec oEngGate[] = { INOUT("h"), IN("tok"), IN("k"), IN("v"), W("norm_key"),
                                    W("norm_query"), W("pre_gamma"), W("per_hc") };
const RadParamSpec pIdxPool[] = { P_INT("M"), P_INT("n"), P_INT("ratio"), P_INT("rotary_dim"),
                                  P_OPT("block_size"), P_F64("theta"), P_F64("eps"), P_STR("dtype"),
                                  { "sections", RAD_P_STR, RAD_OPTIONAL, 0 } };
/* had_quant_act_fp8: libhalo's schema (halo_rows.cpp, the same in 2.1.0 and 2.2.0); radiance does
 * not load libhalo on the R9700. */
const RadParamSpec pHadQ[] = { P_INT("M"), P_INT("n"), P_INT("group"), P_STR("dtype") };
const RadOperandSpec oHadQ[] = { IN("x"), OUT("q"), OUT("scale") };
const RadOperandSpec oIdxPool[] = { IN("stage"), W("value"), W("gate"), W("ape"), W("norm"), IN("page"),
                                    IN("bpos"), INOUT("cache") };
const RadParamSpec pIdxScore[] = { P_INT("M"), P_INT("n"), P_INT("heads"), P_INT("rotary_dim"),
                                   P_F64("theta"), P_INT("blocks"), P_INT("seqs"), P_STR("dtype"),
                                   { "sections", RAD_P_STR, RAD_OPTIONAL, 0 } };
const RadOperandSpec oIdxScore[] = { IN("q"), IN("w"), IN("pos"), IN("cache"), IN("bt"), IN("cu"), IN("nc"),
                                     OUT("score") };
const RadParamSpec pIdxSelect[] = { P_INT("M"), P_INT("blocks"), P_INT("topk"), P_INT("ratio"),
                                    P_INT("seqs") };
const RadOperandSpec oIdxSelect[] = { IN("score"), IN("bt"), IN("nc"), IN("pos"), IN("cu"), OUT("sel"),
                                      OUT("nsel"), OUT("seqused") };

/* Vision (ref_vit.cpp). */
const RadParamSpec pVitCompact[] = { P_INT("M"), P_INT("dim"), P_INT("merge") };
const RadOperandSpec oVitCompact[] = { IN("pixels"), IN("coord"), IN("cu"), OUT("pixels_c"), OUT("coord_c"),
                                       OUT("cu_c") };
const RadParamSpec pVitPos[] = { P_INT("M"), P_INT("n"), P_INT("side"), P_STR("dtype") };
const RadOperandSpec oVitPos[] = { INOUT("x"), W("table"), IN("coord_c") };
const RadParamSpec pVitNl[] = { P_INT("M"), P_INT("n"), P_INT("merge"), P_STR("dtype") };
const RadOperandSpec oVitNl[] = { IN("y"), W("newline"), IN("coord"), IN("cu"), IN("cu_c"), OUT("out") };

/* Hy4's rotary with images (ref_rope.cpp): the schema of `rope`. */
const RadParamSpec pRope[] = { P_INT("M"), P_INT("head_dim"), P_INT("n_head"), P_OPT("n_head_kv"), P_F64("theta"),
                               P_F64("scale"), P_STR("mode"), P_OPT("rotary_dim"),
                               { "sections", RAD_P_STR, RAD_OPTIONAL, 0 } };
const RadOperandSpec oRope[] = { INOUT("qk"), IN("positions") };
/* hy4_mla_project: radiance's mla_project, field for field (libref/ref_registry.cpp), so the plugin can
 * issue either op on the same operands. */
const RadParamSpec pMlaProj[] = { P_INT("M"), P_INT("heads"), P_INT("head_dim"), P_INT("latent_dim"), P_STR("mode"),
                                  P_STR("dtype") };
const RadOperandSpec oMlaProj[] = { IN("a"), W("b"), OUT("y") };

/* The DFly2 drafter's ring (ref_ring.cpp, docs/DFLY2.md). */
const RadParamSpec pRingStore[] = { P_INT("M"), P_INT("ring"), P_INT("width"), P_INT("offset"), P_INT("seqs"),
                                    P_STR("mode"), P_OPT("block") };
const RadOperandSpec oRingStore[] = { INO("src"), IN("row_ids"), IN("pos"), IN("cu"), IN("sidx"), INOUT("state") };
const RadParamSpec pRingLoad[] = { P_INT("M"), P_INT("ring"), P_INT("width"), P_INT("offset"), P_INT("seqs"),
                                   P_OPT("fill") };
const RadOperandSpec oRingLoad[] = { IN("pos"), INO("cu"), IN("sidx"), IN("state"), OUT("dst") };

#define ARR(x) x, (int)(sizeof(x) / sizeof(x[0]))
const RadOpSchema kSchemas[] = {
    { "hy4_hc_pre", ARR(pHcPre), ARR(oHcPre),
      "Hy4 independent hyper-connection, read half: x = sum_j pre_j h_j with pre/post gates from "
      "(flat h . fn) * rsqrt(mean(flat^2)+eps); head=1 is the final collapse (pre only)." },
    { "hy4_hc_post", ARR(pHcPost), ARR(oHcPost),
      "Hy4 independent hyper-connection, write half: h_j = post_j * y + h_j in f32." },
    { "hy4_mla_attn", ARR(pMlaAttn), ARR(oMlaAttn),
      "Hy4 attention over the latent cache (absorbed MLA): one shared KV head whose row is [latent | rope], "
      "score scale*(q.lat + q_rope.rope), softmax with a sink logit a head, out = sum p lat; causal, optional "
      "window; varlen with cu_seqlens, else rows split evenly over seqused (selection: one row a sequence); "
      "optional block keys (bkeys [rows, latent+rope], bcu) a selection row also attends to, causally." },
    { "hy4_mla_kv", ARR(pMlaKv), ARR(oMlaKv),
      "Hy4 MLA, non-absorbed: per head k = [k_rope | k_nope_h], v_h, from kv_b's output and the shared "
      "rotated k_rope." },
    { "hy4_ngram_ids", ARR(pNgram), ARR(oNgram),
      "Hy4 Engram hash ids for all Engram layers in one pass: token map, pad before the sequence start, "
      "xor of c_k*mult_k, floor_mod by each head's prime plus its offset; keeps the 2 previous raw tokens "
      "per sequence in `state`." },
    { "hy4_rows_rmsnorm", ARR(pRowsNorm), ARR(oRowsNorm),
      "RMS norm of each d-wide slice of a row with its own gain (Engram's per-head norms); with fp8=1 "
      "each slice first goes through E4M3 at scale amax/448, as Engram reads its table." },
    { "hy4_engram_gate", ARR(pEngGate), ARR(oEngGate),
      "Hy4 Engram write: per stream gate sigma(sign(d)*sqrt(max(|d|,gate_eps))) of the normed key/query "
      "dot, plus a per-stream branch, added to h; rows whose token is `pad` (or `skip`, `skip2`: the image and "
      "video markers) are left untouched." },
    { "hy4_idx_pool", ARR(pIdxPool), ARR(oIdxPool),
      "Hy4 compressed indexer: gated softmax pooling of `ratio` token keys with a learned APE, RMS norm, "
      "RoPE of the trailing rotary_dim at the block's first position, into the block-key cache." },
    { "hy4_idx_score", ARR(pIdxScore), ARR(oIdxScore),
      "Hy4 compressed indexer score: sum_h w_h * relu(rope(q_h) . k_g) / sqrt(heads*n) over the "
      "sequence's complete blocks; -inf past nc." },
    { "hy4_idx_select", ARR(pIdxSelect), ARR(oIdxSelect),
      "Hy4 compressed indexer selection: top (topk-1) blocks before the query's own (ties to the lower "
      "index), ascending, then the own block; sel are attention pages, seqused the per-query length." },
    { "had_quant_act_fp8", ARR(pHadQ), ARR(oHadQ),
      "a 128-wide Walsh-Hadamard along K (unnormalised, natural order), then per-(row, 128) symmetric E4M3: "
      "scale amax/448 (1 for a zero group), codes round-to-nearest-even clamped to +-448, q row-major, scale "
      "[M, n/128] f32. Pairs with a weight stored rotated (w4nl64a8h): the factor 128 is in its scale." },
    { "hy4_vit_compact", ARR(pVitCompact), ARR(oVitCompact),
      "Hy4 vision: drop each segment's last `merge` patch columns (the newline column the processor adds), "
      "keeping order; coord_c [Pc, 4] (row, col, rows, cols without the column) and cu_c the new boundaries." },
    { "hy4_vit_pos", ARR(pVitPos), ARR(oVitPos),
      "Hy4 vision: x += the learned [side^2 (+1 class row), n] position table resampled bilinearly to each "
      "patch's grid, HF's interpolate(scale_factor=(size+0.1)/side, align_corners=False) in f32, rounded to x." },
    { "hy4_vit_newline", ARR(pVitNl), ARR(oVitNl),
      "Hy4 vision: the merged rows laid on the processor's grid (merge-block order, newline column last): "
      "a row of the last block column is `newline`, every other the adapter row it maps to." },
    { "hy4_rope", ARR(pRope), ARR(oRope),
      "Hy4 rotary: GPT-J pairs (2i, 2i+1) on the first rotary_dim channels of every Q and K head; mode gptj "
      "with [M] positions, or imrope with [3, M] positions and sections s0 s1 s2 (Qwen3-VL's interleaved "
      "components: 1 where i%3==1 and i<3*s1, 2 where i%3==2 and i<3*s2, else 0)." },
    { "hy4_mla_project", ARR(pMlaProj), ARR(oMlaProj),
      "radiance's mla_project (absorb: a[M,h,head_dim] . Wk_h; expand: a[M,h,latent] . Wv_h^T, b [h*2*head_dim, "
      "latent] with each head's key rows then its value rows), f32 accumulate, one rounding to bf16. A second "
      "op so a plugin can route large batches to a matrix-core row and keep libr4d's for decode." },
    { "hy4_ring_store", ARR(pRingStore), ARR(oRingStore),
      "DFly2 ring write: for each listed batch row, `width` i32 words at state[sidx[s]][pos mod ring][offset..] "
      "(the last listed row wins an entry) by mode: pos (the position), row (src row r), out (src row k), dense (pos+1, then the sequence's block "
      "table pages up to pos/block, -1 past them); floats as their f32 bits." },
    { "hy4_ring_load", ARR(pRingLoad), ARR(oRingLoad),
      "DFly2 ring read: the entry of p0-1 (p0 = pos of a sequence's first row), words offset..offset+width-1, "
      "to every row of the sequence (with cu) or to row s (without); `fill` (default -1) when the slot is "
      "negative, p0 < 1 or the entry's position word is not p0-1, and in the rows outside cu's spans." },
};

#define ROW(nm, op, fam, what, fn) \
    { nm, op, fam, what, "any", "bf16 f32, per operand", RAD_DOMAIN_HOST, 0, \
      nullptr, 0, nullptr, 0, nullptr, nullptr, nullptr, fn, nullptr, nullptr, nullptr, nullptr }
const RadKernelInfo kKernels[] = {
    ROW("hy4_ref_hc_pre", "hy4_hc_pre", "norm", "Hy4 iHC read half, f32 host", hy4_ref_hc_pre),
    ROW("hy4_ref_hc_post", "hy4_hc_post", "norm", "Hy4 iHC write half, f32 host", hy4_ref_hc_post),
    ROW("hy4_ref_mla_kv", "hy4_mla_kv", "attn", "Hy4 MLA K/V assembly, host", hy4_ref_mla_kv),
    ROW("hy4_ref_mla_attn", "hy4_mla_attn", "attn", "Hy4 latent attention with sink, host (double)", hy4_ref_mla_attn),
    ROW("hy4_ref_ngram_ids", "hy4_ngram_ids", "embed", "Hy4 Engram hash ids, host", hy4_ref_ngram_ids),
    ROW("hy4_ref_rows_rmsnorm", "hy4_rows_rmsnorm", "norm", "per-slice RMS norm, f32 host", hy4_ref_rows_rmsnorm),
    ROW("hy4_ref_engram_gate", "hy4_engram_gate", "norm", "Hy4 Engram gate, f32 host", hy4_ref_engram_gate),
    ROW("hy4_ref_idx_pool", "hy4_idx_pool", "attn", "Hy4 indexer pooling, f32 host", hy4_ref_idx_pool),
    ROW("hy4_ref_idx_score", "hy4_idx_score", "attn", "Hy4 indexer score, f32 host", hy4_ref_idx_score),
    ROW("hy4_ref_idx_select", "hy4_idx_select", "attn", "Hy4 indexer selection, host", hy4_ref_idx_select),
    ROW("hy4_ref_had_quant_act_fp8", "had_quant_act_fp8", "quant", "rotated E4M3 activation, f32 host",
        hy4_ref_had_quant_act_fp8),
    ROW("hy4_ref_vit_compact", "hy4_vit_compact", "embed", "Hy4 vision newline-column drop, host", hy4_ref_vit_compact),
    ROW("hy4_ref_vit_pos", "hy4_vit_pos", "embed", "Hy4 vision position resample, f32 host", hy4_ref_vit_pos),
    ROW("hy4_ref_vit_newline", "hy4_vit_newline", "embed", "Hy4 vision newline layout, host", hy4_ref_vit_newline),
    ROW("hy4_ref_rope", "hy4_rope", "rope", "Hy4 GPT-J rotary with interleaved M-RoPE, f32 host", hy4_ref_rope),
    ROW("hy4_ref_mla_project", "hy4_mla_project", "attn", "absorbed MLA projections, f32 host", hy4_ref_mla_project),
    ROW("hy4_ref_ring_store", "hy4_ring_store", "kv", "DFly2 ring write, host", hy4_ref_ring_store),
    ROW("hy4_ref_ring_load", "hy4_ring_load", "kv", "DFly2 ring read, host", hy4_ref_ring_load),
};

#ifdef HY4K_WITH_DEVICE
/* The band each card row serves: selection picks by these constraints, and a row that meets them
 * must not refuse at launch (docs/PLUGIN.md). Attention is narrowed to Hy4's geometry so as not to
 * take from libr4d the ones it does serve (h128 gqa 8/16, page 16). */
/* hc ≤ 4 and n ≤ 4096 keep hc·n ≤ 16384 (constraints cannot express products); Hy4: 4 and 3072. */
const RadConstraint cHcPre[] = { RAD_CLE("hc", 4), RAD_CLE("n", 4096) };
const RadConstraint cNgram[] = { RAD_CLE("ngram", 8), RAD_CGE("ngram", 2) };
const RadConstraint cRowsNorm[] = { RAD_CLE("d", 512) };
const RadConstraint cGate[] = { RAD_CLE("n", 8192), RAD_CLE("hc", 16) };
/* rotary_dim ≤ 64 ≤ n keeps rd ≤ n; Hy4: n 128, rd 64. */
const RadConstraint cPool[] = { RAD_CLE("n", 256), RAD_CGE("n", 64), RAD_CLE("ratio", 16), RAD_CLE("rotary_dim", 64) };
const RadConstraint cScore[] = { RAD_CLE("n", 256), RAD_CGE("n", 64), RAD_CLE("heads", 32), RAD_CLE("rotary_dim", 64) };
const RadConstraint cSelect[] = { RAD_CLE("topk", 1024), RAD_CGE("topk", 2) };
const RadConstraint cHadQ[] = { RAD_CEQ("group", 128), RAD_CDIV("n", 128) };
const RadConstraint cMlaProj[] = { RAD_CEQ("head_dim", 256), RAD_CEQ("latent_dim", 512), RAD_CIN("dtype", "bf16") };
const RadConstraint cMlaAttn[] = { RAD_CLE("latent", 512), RAD_CLE("rope", 64), RAD_CIN("dtype", "bf16") };
const RadConstraint cAttn[] = { RAD_CEQ("head_dim", 256), RAD_CEQ("gqa", 1), RAD_CEQ("causal", 1),
                                RAD_CIN("q_dtype", "bf16"), RAD_CIN("kv_dtype", "bf16") };
#define NC(c) c, (int)(sizeof(c) / sizeof(c[0]))
#define ROWD(nm, op, fam, what, cons, fn) \
    { nm, op, fam, what, "any in the constraints", "bf16 f32 f16 activations, f32 gains", RAD_DOMAIN_DEVICE, 100, \
      NC(cons), nullptr, 0, nullptr, nullptr, nullptr, fn, nullptr, nullptr, nullptr, nullptr }
const RadKernelInfo kDev[] = {
    /* scratch: the split form's wave results (dev_hc.hip, hc_pre_part_k) */
    { "hy4_dev_hc_pre", "hy4_hc_pre", "norm", "Hy4 iHC read half, f32 accumulate, card", "any in the constraints",
      "bf16 f32 f16 activations, f32 gains", RAD_DOMAIN_DEVICE, 100, NC(cHcPre), nullptr, 0, nullptr, nullptr, nullptr,
      hy4_dev_hc_pre, hy4_dev_hc_pre_scratch, nullptr, nullptr, nullptr },
    { "hy4_dev_hc_post", "hy4_hc_post", "norm", "Hy4 iHC write half, card", "any", "bf16 f32 f16, f32 post",
      RAD_DOMAIN_DEVICE, 100, nullptr, 0, nullptr, 0, nullptr, nullptr, nullptr, hy4_dev_hc_post, nullptr, nullptr,
      nullptr, nullptr },
    { "hy4_dev_mla_kv", "hy4_mla_kv", "attn", "Hy4 MLA K/V assembly, card", "any", "bf16 f32 f16",
      RAD_DOMAIN_DEVICE, 100, nullptr, 0, nullptr, 0, nullptr, nullptr, nullptr, hy4_dev_mla_kv, nullptr, nullptr,
      nullptr, nullptr },
    ROWD("hy4_dev_ngram_ids", "hy4_ngram_ids", "embed", "Hy4 Engram hash ids, one thread a row, card", cNgram,
         hy4_dev_ngram_ids),
    ROWD("hy4_dev_rows_rmsnorm", "hy4_rows_rmsnorm", "norm", "per-slice RMS norm with E4M3 round trip, card", cRowsNorm,
         hy4_dev_rows_rmsnorm),
    ROWD("hy4_dev_engram_gate", "hy4_engram_gate", "norm", "Hy4 Engram gate, a block a row, card", cGate,
         hy4_dev_engram_gate),
    ROWD("hy4_dev_idx_pool", "hy4_idx_pool", "attn", "Hy4 indexer pooling, a block a group, card", cPool,
         hy4_dev_idx_pool),
    ROWD("hy4_dev_idx_score", "hy4_idx_score", "attn", "Hy4 indexer score, a block a row, card", cScore,
         hy4_dev_idx_score),
    ROWD("hy4_dev_idx_select", "hy4_idx_select", "attn", "Hy4 indexer top-k by radix threshold, card", cSelect,
         hy4_dev_idx_select),
    ROWD("hy4_dev_had_quant_act_fp8", "had_quant_act_fp8", "quant", "128-wide Hadamard + E4M3 per (row, 128), card",
         cHadQ, hy4_dev_had_quant_act_fp8),
    /* attn_paged_sink at head_dim 256 and gqa 1, any page, causal: Hy4's geometry, which libr4d does
     * not serve (h128 gqa 8/16 and page 16 only). Schema from libref/libr4d. */
    /* scratch: form 2g's slice partials (dev_attn.hip, mla_attn2g_k) */
    { "hy4_dev_mla_attn", "hy4_mla_attn", "attn", "Hy4 latent attention with sink, heads grouped by 16 on WMMA, card",
      "any in the constraints", "bf16 f32 f16 activations, f32 gains", RAD_DOMAIN_DEVICE, 100, NC(cMlaAttn), nullptr, 0,
      nullptr, nullptr, nullptr, hy4_dev_mla_attn, hy4_dev_mla_attn_scratch, nullptr, nullptr, nullptr },
    ROWD("hy4_dev_mla_project", "hy4_mla_project", "attn", "absorbed MLA projections, a WMMA tile a block, card",
         cMlaProj, hy4_dev_mla_project),
    /* Vision (dev_vit.hip): the tower and the rotary with M-RoPE. */
    { "hy4_dev_vit_compact", "hy4_vit_compact", "embed", "Hy4 vision newline-column drop, a block a patch, card",
      "any", "bf16 f32 f16", RAD_DOMAIN_DEVICE, 100, nullptr, 0, nullptr, 0, nullptr, nullptr, nullptr,
      hy4_dev_vit_compact, nullptr, nullptr, nullptr, nullptr },
    { "hy4_dev_vit_pos", "hy4_vit_pos", "embed", "Hy4 vision position resample, a block a patch, card", "any",
      "bf16 f32 f16", RAD_DOMAIN_DEVICE, 100, nullptr, 0, nullptr, 0, nullptr, nullptr, nullptr, hy4_dev_vit_pos,
      nullptr, nullptr, nullptr, nullptr },
    { "hy4_dev_vit_newline", "hy4_vit_newline", "embed", "Hy4 vision newline layout, a block a row, card", "any",
      "bf16 f32 f16", RAD_DOMAIN_DEVICE, 100, nullptr, 0, nullptr, 0, nullptr, nullptr, nullptr,
      hy4_dev_vit_newline, nullptr, nullptr, nullptr, nullptr },
    { "hy4_dev_rope", "hy4_rope", "rope", "Hy4 GPT-J rotary with interleaved M-RoPE, a block a (row, head), card",
      "any", "bf16 f32 f16", RAD_DOMAIN_DEVICE, 100, nullptr, 0, nullptr, 0, nullptr, nullptr, nullptr, hy4_dev_rope,
      nullptr, nullptr, nullptr, nullptr },
    { "hy4_dev_attn_sink", "attn_paged_sink", "attn",
      "paged attention with a sink logit a head, causal, any window, any page size; a block a (row, head)",
      "head_dim 256, gqa 1, any block_size, varlen or per-row selection", "bf16 q/kv/out, f32 sinks",
      RAD_DOMAIN_DEVICE, 100, NC(cAttn), nullptr, 0, nullptr, nullptr, nullptr, hy4_dev_attn_sink, nullptr,
      nullptr, nullptr, nullptr },
    /* The DFly2 ring (dev_ring.hip): copies, the same bits as the host. */
    { "hy4_dev_ring_store", "hy4_ring_store", "kv", "DFly2 ring write, a block a listed row, card", "any",
      "i32 state, i32 i64 f32 bf16 f16 sources", RAD_DOMAIN_DEVICE, 100, nullptr, 0, nullptr, 0, nullptr, nullptr,
      nullptr, hy4_dev_ring_store, nullptr, nullptr, nullptr, nullptr },
    { "hy4_dev_ring_load", "hy4_ring_load", "kv", "DFly2 ring read, a block a destination row, card", "any",
      "i32 state, i32 f32 destination", RAD_DOMAIN_DEVICE, 100, nullptr, 0, nullptr, 0, nullptr, nullptr, nullptr,
      hy4_dev_ring_load, nullptr, nullptr, nullptr, nullptr },
};
constexpr int kNDev = (int)(sizeof(kDev) / sizeof(kDev[0]));
#else
constexpr int kNDev = 0;
#endif
constexpr int kNHost = (int)(sizeof(kKernels) / sizeof(kKernels[0]));

}  // namespace

int hy4k_schema_count() { return (int)(sizeof(kSchemas) / sizeof(kSchemas[0])); }
const RadOpSchema* hy4k_schema_at(int i) {
    return i >= 0 && i < hy4k_schema_count() ? &kSchemas[i] : nullptr;
}
/* Rows: the card ones first (if any), then the host oracles. */
int hy4k_kernel_count() { return kNDev + kNHost; }
const RadKernelInfo* hy4k_kernel_at(int i) {
    if (i < 0 || i >= kNDev + kNHost) return nullptr;
#ifdef HY4K_WITH_DEVICE
    if (i < kNDev) return &kDev[i];
#endif
    return &kKernels[i - kNDev];
}
int hy4k_kernel_is_reference(int i) { return i >= kNDev ? 1 : 0; }
