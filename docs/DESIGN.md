# Design of the Hy4-flash plugin for radiance 2.2.0

Plan step 3 of the project plan (the "plan steps" cited across this repository). Which ops are
reused, which are new, how each layer is emitted and what state is kept between steps. Numeric
reference: the code of Transformers PR #50903 at commit `c2ec850`
(`ref/.venv/lib/python3.12/site-packages/transformers/models/hy_v4_flash/modeling_hy_v4_flash.py`),
cited as `HF:line`. The radiance citations are from a radiance checkout at v2.1.0 (with commit
b264e9b reverted in libr4d; what moved in 2.2.0 is in "Moving to radiance 2.2.0", at the end).

The MLA comes in two forms: the **expanded** one (K and V per head), used for parity on the host
backend (plan steps 4 and 5), and the **absorbed** one (latent cache), used to serve long context
("Absorbed MLA", below).

## Two plugins and a recipe

| Piece | Type | What it holds |
|---|---|---|
| `hy4k` | `kernels` | The new ops: host kernel (oracle) now; HIP kernel in plan step 6 |
| `hyv4flash` | `architectures` | `rad_arch_declare`, `rad_arch_step`, the calibration tap and the debug tap |
| `data/recipes/hy4flash-w4nl64-rtn.recipe` | recipe (done) | `w4nl64` experts by rounding with the searched scale (`rule=search`), E4M3 scale per 64 below 2^-13 (the kernel constant); trunk and Engram table in BF16. Tested on the trimmed model: host, one R9700 and two |
| `data/recipes/hy4flash-w4nl64-i8.recipe` | recipe (pending) | `w4nl64` experts with `calib=$CALIB`; int8 trunk **except** the 5 Engram projections per layer (`value_proj`, `key_projs`), which stay in BF16 (94 M parameters in total, 188 MB) so they can use `gemm_nt_bias`; Engram table in BF16 as is (the FP8 round trip happens in `hy4_rows_rmsnorm`); head not quantised; no expert protected in BF16 |

## Hy4 constants that fix the ops

| Constant | Value | Where |
|---|---|---|
| `rms_norm_eps` | 1e-5 | every RMSNorm except Engram; also the inner norm of the iHC over the `hc·n` = 12288 columns (HF:810, 824) |
| eps of the Engram norms | 1.1920929e-07 | `nn.RMSNorm` without eps (HF:737-743); set the same way in `ref/reference.py` |
| `hc_eps` | 1e-6 | added to `pre` and `post` of the iHC (HF:841, 844) |
| `hc_magnitude` | 2.0 | multiplies `post` (HF:844) |
| Engram `gate_eps` | 1e-6 | `gate.abs().clamp_min(1e-6).sqrt()·sign(gate)`, with `sign(0) = 0` (HF:793) |
| rope | **`gptj`** (pairs `2i, 2i+1`), 64 dims, theta 1e7 in the sparse layers and 1e4 in the window layers | `rotate_half` of adjacent pairs and `repeat_interleave(2)` of cos/sin (HF:290-294, 322-323). Interleaved mrope: in text the 3 positions are equal and it is equivalent to the 1-D rope (the op test confirms it) |
| window | the current key + the 511 before it (`pos - j < 512`) | HF:1430-1433; `attn_paged_sink` uses the same rule |
| indexer | groups of 4, 16 heads of 128, top 511 + the own group, only when the context exceeds 2048 | HF:344-436, 627 |
| `routed_scaling_factor` | 3.998 | HF:246 |
| `swiglu_limit` | 10 | HF:204-206 |

## Reused ops

| Hy4 piece | Op | Notes |
|---|---|---|
| Embedding | `embed_lookup` | |
| Embedding to 4 streams | `hc_enter` | copies `x` into the `hc` streams (`docs/OPS.md:198`; HF:1443) |
| Engram gather | `embed_lookup_q`, BF16 table without scale (`rad_block_ple.h:208-214`), declared as a model weight (`grp_model()`, like `rad_block_ple.h:212-219`; the planner only takes the n-gram branch with `mu.layer < 0`, `core/place/planner.cpp:305-307`) | Only this way does the core treat it as an n-gram table (`core/place/planner.h:442-444`) and apply `ngram-placement`. Several tables per model work (`libavx/avx_ngram.cpp:382-399`). `ids` and the rows are `RAD_DOMAIN_HOST` buffers, as in `rad_block_ple.h:309-320`, so that the host gather reads in place with `disk` |
| Every RMSNorm with a per-row weight (`input_layernorm`, `post_attention_layernorm`, `post_proj_norm`, `post_fc2_norm`, `combine_output_norm`, `q_a_layernorm`, `kv_a_layernorm`, the indexer's `token_key_norm`, the final `norm`) | `rmsnorm` | eps 1e-5 |
| Projections without bias (q_a, q_b, kv_a, kv_b, o_proj, gate_proj, MLP, indexer) | In the BF16 container (parity), `gemm_nt`. In the RTN/GPTQ with int8 trunk, the quantised linears of qwen4exp_fp8 (`gemm_nt_q` with its activation quantisation and its scales) | |
| Engram projections **with bias** (`value_proj` and the 4 `key_projs`) | `gemm_nt_bias` (`docs/OPS.md:791`) | The biases are in the checkpoint (`engram_layer.value_proj.bias`, `key_projs.N.bias`). The recipe leaves these weights in BF16: `gemm_nt_bias` takes no scales |
| Attention output gate | `sigmoid` + `mul` | Like `AttnGatedBF16` (`rad_block_attn_gated.h:220-233`) |
| Rope | `rope` with `mode gptj`, `rotary_dim 64`, `theta` per layer type | Two `Geom` (or direct `RAD_OP`), because `decl_rope` takes `g.theta` (`rad_arch.h:1443-1453`) |
| Attention, window layers | `kv_store` + `attn_paged_sink`, `head_dim 256`, `gqa 1`, `window 512`, group `RAD_KV_WINDOW` | flint pattern (`flint.cpp:169-180`, `flint_attention.h:143-153`) |
| Attention, sparse layers up to 2048 | the same, `window 0`, group `RAD_KV_FULL` | |
| Attention, sparse layers with selection | `attn_paged_sink` with `sel` as a per-query page table, pages of 4 | QSA pattern (`rad_block_attn_gated_fp8.h:67-81`; `docs/QSA.md:140-155`): **one logical sequence per query**, `q_len = 1`, with its own `cu_seqlens` (not the batch one) and `seqused` per query; it is emitted in chunks sized by the declared scratch. The selection buffers are declared before the attention and the indexer after it, as in QSA. The host accepts any `block_size` |
| Indexer state | `qsa_work`, `qsa_tail_store` | `qsa_work` before `qsa_tail_store` (`docs/QSA.md`) |
| Routed MoE | `MoeFP8` like nemotronh's `LatentMoe` (`nemotronh.cpp:270-345`, 810-857) | `sigmoid_router`, correction bias only for the choice, `top_k 16`, `norm_topk`, `route_scale 3.998`, `swiglu_limit 10`, `router_k = 3072`, `no_reduce`. Experts with `RAD_ACCESS_CONDITIONAL`: `expert_tiered` and the disk tier work |
| Shared expert (full width) and dense MLP of layer 0 | `MlpFP8` (`rad_block_mlp_fp8.h:27`) with its `silu_mul` activation, **without clamp** (`HYV4FlashMLP` uses `silu` without a limit, HF:164; the clamp at 10 is only for the routed experts). In the BF16 container, `gemm_nt` + `silu_mul` + `gemm_nt` | Outside the block, like nemotronh |
| Head | `gather_rows` + `logits_gemm` | `gather_rows` keeps one row per sampled position before `logits_gemm` (`docs/OPS.md:2686-2690`; like `nemotronh.cpp:991-1003`). `lm_head` is `RAD_ACCESS_VOCAB` and the sampler fuses with `logits_gemm` |
| Calibration tap | `Config::calib` of `MoeFP8` with `calib_layer_on(l)` | Restrictions below |

## New ops (plugin `hy4k`)

All of them with a host kernel in FP32 (oracle). Each one has its specification in
`docs/OPS-HY4.md` in the format of `radiance/docs/OPS.md` and an op test against the HF code in
numpy/torch, also fed with the `ref/` dumps of layers 0-2.

| Op | Inputs → outputs | What it computes |
|---|---|---|
| `hy4_hc_pre` | `h [T,hc,n]`; weights `fn [2hc,hc·n]` f32, `base [2hc]`, `scale [2]` → `x [T,n]`, `post [T,hc]` f32. With `head=1`: `fn [hc,hc·n]`, `base [hc]`, `scale [1]`, no `post` | `flat` = the `hc` streams in FP32; `mix = (flat·fnᵀ)·rsqrt(mean(flat²)+1e-5)`; `pre = σ(mix·s0+b)+hc_eps`; `post = 2·σ(mix'·s1+b')+hc_eps`; `x = Σ pre_i·h_i`, rounded to the dtype of `h`. HF:831-852, 927-938 |
| `hy4_hc_post` | `h [T,hc,n]` inout, `y [T,n]`, `post [T,hc]` | `h_i = post_i·y + h_i` in FP32, rounded to the dtype of `h`. HF:897, 903 |
| `hy4_mla_kv` | `kvx [T,H·(192+256)]` (kv_b output), `k_rope [T,64]` (already rotated) → `k [T,H,256]`, `v [T,H,256]` | Builds `k_h = [k_rope | k_nope_h]` (rope in front, see below) and splits off `v_h`. HF:541-546, 652-662 |
| `hy4_ngram_ids` | `tok`, `pos`, `cu`, `state` inout (group `RAD_KV_LINEAR`, `[slots, 1, 1, 2]`), `state_index`; weights of the `E` layers with Engram (`E` = length of `engram_layer_ids` in the config: 4 in the model, 1 in the trimmed model): `map [E,120818]` i32, `mult [E,3]` i64, `vocab_sizes [E,16]` i64, `offsets [E,16]` i64 → `ids [E·max_tok,16]` i32 (2-D: each layer's gather reads its chunk as `[T,16]`) (`RAD_BUF_PERSIST`: it is read layers later, like `rad_block_ple.h:300-303`), (an `ahead_ids` for the read-ahead is left for later: today it is not in the schema) | **One emission per step**, at the start, for the `E` layers: this way the state is written only once. See "Engram state". Per layer `e`: `c_k = map_e[t(p-k)]` if `t < 120818`, otherwise `map_e[120002]`, with `t(p-k)` the `pad` id if `pos-k < 0`. Heads **0-7: bigrams** (`n = 2`), **8-15: trigrams** (`n = 3`), in that order (largest n-gram, like `hash_ngrams`, HF:765-771): `mix_n = c_0·mult_0 ⊕ … ⊕ c_{n-1}·mult_{n-1}` (int64 with overflow; `c_0` the current token); `id_h = mix_{n(h)} mod p_{e,h} + off_{e,h}`, with `p_{e,h}` and `off_{e,h}` in the order of the flattened `engram_layer_head_vocab_sizes[e]` (HF:719-723). HF:757-772, 1470-1486 |
| `hy4_rows_rmsnorm` | `x [T,16,96]`, weight `[16,96]`, `fp8` → `[T,1536]` | With `fp8=1`, first the E4M3 round trip of each chunk with scale amax/448 (`_fp8_round_trip`, HF:688-694, 780-781), at run time; then RMSNorm per head with its own weight, eps 1.1920929e-07. HF:780-783 |
| `hy4_engram_gate` | `h [T,hc,n]` inout, `tok [T]`, `k [T,hc,n]`, `v [T,n]`; weights `norm_key [hc,n]`, `norm_query [hc,n]`, `pre_gamma [n]`, `per_hc [hc,n]` | `g_i = σ(sign(d)·√max(|d|,1e-6))`, `d = ⟨rms(k_i)·wk_i, rms(h_i)·wq_i⟩/√n`; `s = pre_gamma·rms(v)`; `h_i += g_i·s + per_hc_i·s`; Engram mask from `tok` (HF:1894-1898). eps of the rms 1.1920929e-07. With `tok == 120002` the **increment** is 0 and `h` stays untouched (HF:800). HF:785-800 |
| `hy4_idx_pool` | `stage` (per-token keys of the groups that close), `page`, `bpos` (from `qsa_work`); weights `value [128,128]` f32, `gate [128,128]` f32, `ape [4,128]` f32, `norm [128]` → writes the group key into `kv_idx_keys` | In FP32: `Σ_j v_j·softmax_j(g_j+ape_j)`; RMSNorm (eps 1e-5); rounding to the activation dtype (HF:393); `gptj` rope on the last 64 at the position of the group's first token, theta 1e7. HF:378-392, 413-416 |
| `hy4_idx_score` | `q [T,16,128]` **without rope**, `w [T,16]` (unscaled), `pos`, `kv_idx_keys`, `bt_keys`, `cu`, `nc` → `score [T,G]` f32 | Rotates `q` at `pos` (last 64 channels, adjacent pairs); `Σ_h w_h·16^-½·128^-½·relu(q_h·k_g)` for the `nc` groups of the sequence. It does not mask per query: `hy4_idx_select` does that, since it has `pos`. **By row windows**: score and select are emitted together in chunks of `plane_rows` queries (≤ 1024), as in QSA (`rad_qsa.h`, `plane_rows`), and the plane is `[plane_rows, max_ctx/4]` f32: at 196,608 tokens, 1024 × 49,152 × 4 B = 192 MiB, below QSA's 512 MiB cap. **Geometry as in QSA** (`rad_qsa.h:370-400`): `RadBuildCtx::max_mixed_tok ≤ plane_rows`, so a step wider than the plane has a single sequence; the windows are only used in that case and carry their own `win_cu = {0, rows}`, with `q`, `pos` and the output offset to the chunk. HF:417-427 |
| `hy4_idx_select` | `score`, `bt_attn`, `nc`, `pos`, `cu` → `sel [T,512]`, `nsel [T]`, `seqused [T]` | Per query, candidates `g < n`, `n = min(nc, pos/4)` (integer division: the own group is left out, unlike QSA's `(pos+1)/ratio`); top-511 of those with signed ranking (the head weights can be negative); the own group (`bt_attn[pos/4]`) always last; `seqused = (nsel-1)·4 + pos%4 + 1`. **Tie-break**: on equal score the group with the lower index wins (as in QSA). **Output order**: the 511 by ascending group index and the own group at the end, so that the attention sums the pages in key order, like the HF mask, and two equal selections give the same bytes (`docs/QSA.md:167-174`). A NaN score counts as −∞. Below 2048 of context it is not emitted and the attention is dense. HF:426-433, 627 |

### Why the rope goes in front

`rope` rotates the **first** `rotary_dim` columns of each head (`libref/ref_attn.cpp:1000`;
`docs/OPS.md`, "the head past rotary_dim is untouched") and Hy4 rotates the **last** 64. The product
`q·k` does not change if q and k permute their columns the same way. So:

- `rad-convert` reorders the `q_b` rows per head to `[rope 64 | nope 192]`. It is declared with
  chained `RAD_MAP_ROWS` maps on the same name (8 pieces per map, `rad_arch.h:459-462`), two pieces
  per head (rope, then nope), as `bailing_hybrid.cpp:244-255` does;
- `rope` rotates `q` in place with `head_dim 256`, `rotary_dim 64`;
- `hy4_mla_kv` builds `k_h = [k_rope | k_nope_h]`.

The indexer does not need it: its ops are its own, so `hy4_idx_pool` and `hy4_idx_score` rotate
the last 64 channels themselves (of the group keys and of the queries), with the frequencies in
float as in HF. `query_projection` stays as it is.

## Flow of a layer (HF:871-907)

```
h = hc_enter(embed_lookup(tok))                         (once)
ids = hy4_ngram_ids(tok, ...)  -> [E,T,16]               (once, writes the state)
per layer:
  if the layer has Engram (1, 11, 21, 31; index e = 0..E-1):
      rows = embed_lookup_q(ids[e])       (BF16 table as is; contiguous chunk, offset e·max_tok·16: the libavx gather
                                           only uses its row cache with direct reads if `tok`
                                           is contiguous, libavx/avx_gemm.cpp:1313)
      e   = hy4_rows_rmsnorm(rows, fp8=1) -> v = gemm_nt_bias(value), k_i = gemm_nt_bias(key_i)
      hy4_engram_gate(h, tok, k, v, ...)
  x, post = hy4_hc_pre(h, attn_hc);  x = rmsnorm(x, input_layernorm)
  y = attention (below);              y = rmsnorm(y, post_proj_norm)
  hy4_hc_post(h, y, post)
  x, post = hy4_hc_pre(h, ffn_hc);   x = rmsnorm(x, post_attention_layernorm)
  y = layer 0:    rmsnorm(MLP(x), post_fc2_norm)
      layers 1-41: rmsnorm(up(MoeFP8(down(x))) + shared(x), combine_output_norm)
  hy4_hc_post(h, y, post)
close: x = gather_rows(h, sampled positions); x = hy4_hc_pre(x, head=1); x = rmsnorm(x, norm); logits_gemm(x, lm_head)
```

### Attention, non-absorbed form

```
qr  = rmsnorm(gemm(x, q_a), q_a_layernorm)
q   = gemm(qr, q_b)                -> [T,H,256] with the rope in front; rope(q, gptj, theta of the type)
kv  = gemm(x, kv_a)                -> lat [T,512] -> rmsnorm(kv_a_layernorm); k_rope [T,64] -> rope
kvx = gemm(lat, kv_b)              -> [T, H·(192+256)]
k, v = hy4_mla_kv(kvx, k_rope);  kv_store(k, v)
sparse layers > 2048:  indexer (below) -> sel, seqused
o   = attn_paged_sink(q, kv, sinks, window | sel)
o   = o · sigmoid(gemm(x, gate_proj))        (per channel, [T, H·256])
y   = gemm(o, o_proj)
```

Indexer (sparse layers). The key store is emitted **always**, also with short context; only the
scoring and the selection depend on exceeding 2048 (like `QsaIndexer::step`, which stores the keys
before it looks at `emit_sel`, `rad_qsa.h:346-354`). Otherwise, on crossing 2048 the keys of the
earlier groups would be missing:

```
always:
  tk = rmsnorm(gemm(x, token_key_projection), token_key_norm)      -> per-token keys [T,128]
  qsa_work(...) -> stage, page, bpos, nc;  hy4_idx_pool(stage, ...) -> kv_idx_keys;  qsa_tail_store(...)
only if the context exceeds 2048:
  qi = gemm(qr, query_projection) -> [T,16,128]   (no rope: hy4_idx_score applies it at `pos`)
  w  = gemm(x, head_weight_projection)  (FP32)
  in chunks of plane_rows: score = hy4_idx_score(qi, w, kv_idx_keys, ...); sel = hy4_idx_select(score, ...)
```

The non-absorbed KV takes `H·(256+256)·2` B per token and layer: 32 KB in the sparse layers and
64 KB in the window layers (only 512 tokens). It is good for parity on the host and for short
contexts.

### Absorbed MLA

`HY4_MLA` picks the form: `absorbed`, `expanded` or `both` (to convert). Without the variable,
the plugin serves the absorbed form if the container has `blk.L.attn_kv_b_abs.weight`, and the
expanded one otherwise (tested: the trimmed models converted earlier go to expanded and the one
converted with `both` to absorbed). The form is fixed in the container: only one converted with
`both` (what `tools/convert-hip.sh` does by default) serves in both; the others, in their own.
`kv_b_proj` has to be in BF16, because the absorbed kv_b is built by copying its rows; the plugin
checks this at conversion time.

```
qr, q, kva as above;  lat = rmsnorm(kva[:, :512]) in place;  rope(q), rope(kva[:, 512:])
q_lat = mla_project(q, kv_b_abs, absorb)          -> [T, H·512]   (= q_nope · W_k per head)
kv_store(kva[:, :288], kva[:, 288:])               -> cache row [lat | k_rope], one head
sparse layers > 2048: indexer same as above -> sel, seqused
o_lat = hy4_mla_attn(q_lat, q (rope_pitch 256), cache, sinks, window | sel)   -> [T, H·512]
o   = mla_project(o_lat, kv_b_abs, expand)         -> [T, H·256]  (= o_lat · W_vᵀ)
gate, o_proj and all-reduce, the same
```

`kv_b_abs` is `[H·512, 512]`: per head, 256 key rows in the order of `q` (`[rope | nope]`, so the
64 rope rows are zeros and then come the 192 `k_nope` rows of `kv_b`) and 256 value rows. It is the
shape of `mla_project` (`head_dim` 256 in both halves) that libr4d serves on the R9700; with a
`k_nope` of 192 and a `v` of 256 it would not fit any other way. It is built at conversion time
with row maps from `kv_b` and `hy4.ceros_rope` (`conv/trim.py`, in the `--add` constants). The
zeros change nothing: the rope part of `q` multiplies zeros and is summed in f32.

Cache: 576 values per token and layer (1152 B in BF16), the same on both cards with tp 2, against
`H·512` per card for the expanded form: 13.5 KiB per token in the 12 sparse layers against
192 KiB. With tp 2 each card keeps the whole cache, because the attention of its heads reads all of
it.


## State between steps

| Group | Type | Layers | What it keeps |
|---|---|---|---|
| `kv_full` | `RAD_KV_FULL` | 12 sparse | K and V per head |
| `kv_win` | `RAD_KV_WINDOW`, `window 512` | 30 window | K and V per head |
| `kv_idx_tail` | `RAD_KV_LINEAR`, `{4 + max_spec, 128}` per sparse layer | 12 sparse | per-token keys of the open group (`qsa_tail_store`) |
| `kv_idx_keys` | `RAD_KV_FULL`, block 4, one page per group | 12 sparse | group keys, declared right before the op that fixes `block_size` (`docs/QSA.md`, "Block-key storage") |
| `kv_engram_tok` | `RAD_KV_LINEAR` i32, `state_dim {1, 2 + max_spec}`: one id window per sequence | one for the `E` layers (written by the single emission of `hy4_ngram_ids`) | the 2 previous tokens **unmapped**: each layer applies its own map |
| `kv_dfly` | `RAD_KV_LINEAR` i32, `state_dim {6, 5638}`, only with the DFly2 drafter | bound to layer 25 | the drafter's ring: per logit row, `h` and layers 25-41's selections (docs/DFLY2.md) |

### Engram state

- `hy4_ngram_ids` is emitted once per step, before the first layer, and computes the ids of the
  4 layers. This way the state is read and written once (if each layer emitted it, layers 11, 21
  and 31 would read the state already overwritten by layer 1).
- The state is a chronological window of `ngram - 1 + max_spec` unmapped ids per sequence,
  radiance's `ngram_ids` contract (libref `ref_ngram.cpp`): it is read at `num_accepted - 1`; a
  step without `num_accepted` leaves its last `ngram - 1` ids, oldest first, at offset zero, and a
  verify step (with it) shifts the read window down by one and appends every id it ran, so a next
  step that kept k reads at k - 1. The 4 `lookup_table` of the checkpoint are identical today, but
  the op does not assume it.
- `hy4_ngram_ids` reads `t(p-k)` from the step itself if it falls inside the chunk; otherwise from
  the state, and only if `pos-k ≥ 0`; if `pos-k < 0`, it uses the `pad` id.
- Cold (new sequence), the state is not read: `pos-k < 0` covers it.
- The `state_index` operand comes from the `RadKVGroupBatch`, as in `rad_block_ple.h:337-339`: a
  sequence never reads another one's state, and the prefix cache saves and restores the state with
  the group (spec §7.3).
- Speculative verify: in a step with decode sequences the plugin issues `hy4_ngram_ids` twice, over
  the decode sequences with `num_accepted` and over the prefill chunks without it, since a chunk
  commits every token and its rows do not fit the decode window. The indexer's tail ring holds
  `ratio + max_spec` rows (`rad_qsa.h`); the attention caches are paged by position and a later
  verify overwrites a rejected row.
- `ahead_ids` (optional): the next prefill chunk's ids (`RadBatch::n_ahead`, one chunk at most),
  which each Engram gather hands to libavx's reader thread as `ahead`. A fresh 18K prefill went from
  674 to 1937 tok/s on its first pass with the table on the test machine's NVMe.
- Tests: two sequences in continuous batching, where the first one retires and another one enters
  with a cached prefix; and a 1-token decode that compares the ids of the 4 layers with the
  reference.

### Engram table as E4M3 codes

- `conv/engram_codes.py` writes each table row as the E4M3 codes `hy4_rows_rmsnorm`'s round trip
  computes from it (scale amax/448 in FP32, row scales dropped, one bf16 scale of 1 a table), and
  `deploy/conv-engram-e4m3.sh` puts them in the container in place with
  `--set hy4.engram_codes=rowmax448`: 63.4 GB in place of 126.7 GB. `decl_engram` takes the codes
  path only with that key and an `affine` table of E4M3 codes under one bf16 scale; an E4M3 table
  without the key (some other quantisation, which the round trip would round a second time) and the
  key over any other table are refused. A container converted before the key existed takes it in
  place with `deploy/set-engram-key.sh` (an `--in-place` run that keeps every weight).
- Dropping the row scales is exact only because the per-head RMSNorm cancels them, and only at the
  eps the plugin uses, 1.1920929e-07 (as `ref/reference.py`): a nonzero row of codes has mean(c²) of
  at least 448²/96 = 2091, so the codes path normalises as if eps were 0, and the bf16 path differs
  from that by about eps/(2·mean(x²)); `engram_codes.py` checks that on every row (at most 1e-6),
  with the codes' row max (448, or a zero row) and finiteness.
- So the codes form fixes the eps. HF's `nn.RMSNorm` with `eps=None` takes the eps of the input
  dtype, 0.0078 in bf16; at that eps the codes path would differ from the bf16 table's
  (measured: 0.5% in mean and 3.6% at most a head). A container with the bf16 table can still change
  the eps (it is an op parameter); a container with the codes cannot, and a different eps means
  converting the bf16 table again.

## Trunk format and launches (plan step 13)

- **IQ8R trunk.** A container converted with `--set hy4.trunk_i8=1` and
  `data/recipes/hy4flash-w4nl64-gptq-iq8r.recipe` (applicable in place, `rad-convert --in-place
  --replace`) stores the attention q_a/q_b, gate and output, the dense FFN, the MoE latent
  projections and the shared expert as IQ8R; the key sets `Geom::w_bf16 = false`, `w_i8h = true`,
  group 256, and the first int8 linear reading an input quantises it into the input's int8 code pair
  (no producer feeds one); the others reading the same rows read that pair as fed (`ActFP8::q8_fed`):
  the attention gate reads `attn_q_a`'s and the shared expert's gate/up reads `ffn_latent_in`'s,
  the same codes with 83 `had_quant_act_i8g` launches a decode token less (334 → 251), and libr4d's fused producers (`rmsnorm_had_quant`, `gate_had_quant`, `gated_had_quant`) write the pairs at five sites a layer: about 1,770 → 1,518 launches a decode token, decode +3-4%, the same bits (KL at all 10,851 positions and greedy text). `attn_kv_a`
  (N 576) and `idx_hw` (N 16) stay bf16, since libr4d has no int8 row above 64 rows for those
  widths, and so does the rest of the indexer (its top-k is a hard choice). On the
  cards: decode 35.8 → 48.3 tok/s; model KL against FP32 unchanged in mean (0.183 → 0.181), at the
  edge of the rounding envelope per position. Without the key the container is served bf16 as
  before. The DFly2 drafter's linears take the same format on their own terms (below).
- **Chained launches.** `probe()` sets `chain_pays = RAD_CHAIN_ALWAYS` (decode 33.9 → 35.9 tok/s
  single stream); `HY4_CHAIN=0` turns it off.
- **Batch-size forms.** Steps over 32 rows run attention form 3 and `hy4_mla_project` (WMMA); up to
  32 rows, form 2 and libr4d's `mla_project`, batch-invariant there. A row's bits differ between the
  two, as libr4d's do. Under speculation a decode step carries `1 + n_spec` rows a sequence, so with
  `spec-ngram: 4` seven sequences already cross 32 rows: greedy text can then differ from a solo run,
  as it can without speculation at 33 sequences.

## Tensor parallelism (`tp: 2`, plan steps 6b and 8)

Done and tested on two R9700 with the 3-layer trimmed model, in BF16 and in RTN `w4nl64`
(2026-11-19): top-5 of the 4 prompts the same as at tp 1 and as the FP32 reference, decode equal to
prefill.

- The new ops are replicated on each rank (small or per-row data).
- Attention: heads split by rows (q_b with its row map over all heads, kv_b, gate_proj, sinks),
  cache with H/world heads; `o_proj` by columns and an all-reduce before the `post_proj` norm.
- MoE: each expert split by columns (`n_ff_exp = 1536 / world`, like nemotronh, with
  `probed_expert = 0` because every card has every expert), `no_reduce` in the block and an
  all-reduce of the routed arm after the latent up-projection (linear and without bias). The shared
  expert and the dense MLP of layer 0 are replicated, and are added afterwards.
- Embedding split by vocabulary with an **exact** all-reduce (disjoint rows at the model input;
  rad_fp8.h ArFold::kArExact). Final head by vocabulary.
- Engram, iHC, norms, `q_a`, `kv_a`, the latent up- and down-projection and the indexer:
  replicated.
- `rad_route_table` with one row per MoE layer, so that the file stager reads the counts of its
  layer (`abi/rad_builder.h:826-838`; `qwen4exp_fp8.cpp:929`).

## DFly2 drafter (plan step 14)

The checkpoint's block drafter (`model.dfly_layers.*`, 0.85B bf16, 5 layers) proposes 5 tokens a
step as a radiance `RAD_DRAFT_BLOCK` drafter. The contract with the `hy4k` ops and the wiring as built
are in [DFLY2.md](DFLY2.md). In short:

- Served only when the container carries `dfly.*` and the build speculates at `max_spec == 5`
  (`num-speculative-tokens: 5`); otherwise nothing is declared and the trunk is unchanged.
  `HY4_DFLY=0` leaves it out, for prompt lookup alone over a container that carries it.
- No cache of its own: drafter layer `d` reads trunk layer `25 + 4d`'s `kv_full` (never writes it)
  through the trunk's attention declaration (`decl_mla`, drafter site), plus its block's own keys
  (`hy4_mla_attn`'s `bkeys`). What the trunk knew at the last accepted row (`h` and the selections)
  travels in the `kv_dfly` ring, written by every trunk step with logits (`hy4_ring_store`) and read
  by the query pass (`hy4_ring_load`). The context pass does nothing.
- Weights under `dfly.*`, appended in place by `deploy/conv-dfly.sh`: bf16 as the checkpoint holds
  them, or with their linears in IQ8R (`DFLY_I8=1`, `hy4flash-w4nl64-gptq-iq8r-dfly.recipe`,
  `--replace 'dfly.*'`). The plugin follows the stored format of `dfly.0.ffn_down` and
  `dfly.correction.down` (`Dfly::i8`; no container key, since no producer here writes codes), not
  the trunk's: an IQ8R drafter gets the trunk's int8 geometry and the int8 input pairs, a bf16 one
  declares exactly what it did before. `attn_kv_a` (N 576) and `attn_kv_b_abs` stay bf16 for the
  trunk's reasons, and so do the norms and the fusion's gate vector. The bf16 drafter's linears were
  11% of a decode step. The head is the trunk's `lm_head`, vocab-sharded at tp 2 like the trunk's.

## Calibration (plan step 9)

`MoeFP8` rejects the tap in `declare` in three cases. The plugin avoids them:

- With experts split per rank (`rad_block_moe_fp8.h:1426-1431`): with calibration and `world > 1`,
  whole experts per rank, like `nemotronh.cpp:844-855`.
- With the routed arm fed in int8 (`w.eff.q8_fed`, `:1084-1092`, which also rejects
  `swiglu_limit > 0`): the latent and the gated product go in E4M3 (`w4nl64a8h`, not
  `…a8i8h`), even when the trunk is int8.
- With some expert protected in BF16 (`:855-858`): the recipe protects none.

Smoke test of the tap on the 3-layer trimmed model, in `declare`, with the BF16 container and with
the RTN one.

## Debug tap

`HY4_DUMP_DIR`, like nemotronh's `dump()` (`nemotronh.cpp:1017-1034`): `rad_buf_ptr`,
`rad_memcpy_async`, `rad_stream_sync`, `fwrite`, after each sub-block of the requested layers
(`HY4_DUMP_LAYERS`, 0-2 by default). It writes the same names as `ref/reference.py`, so
`ref/compare.py` compares the two without translating. The transient arena buffers can be
overwritten by the end of the step: the tap copies right after the op that writes them. When off
it emits nothing.

## Chat template

The checkpoint template (`chat_template.jinja`) builds its markers with
`'<｜hy_start{}｜>'.format(HYTK)`, and radiance's Jinja (the llama.cpp one) has no `.format()`: the
container ends up without a template and `/v1/chat/completions` returns 501.
`data/chat_template.jinja` is the same template with those 22 lines rewritten with `~`.
`tools/convert-hip.sh` always embeds it (`--chat-template`); an already converted container gets it
with `override-chat-template`. radiance derives the response format from it (reasoning
`<think:opensource>`, calls `<tool_calls:opensource>` with
`<arg_key:opensource>`/`<arg_value:opensource>`).

`tests/test_chat_template.py` compares it with the original rendered by HF (thirteen conversations:
system with and without tools, tool result, parallel calls with reversed results, ids that do not
match, non-ASCII text, and `reasoning_effort` through kwargs and through the request body) and
passes responses through `rad-chatparse` (reasoning, one and two calls, and no tags with
`no_think`). With `--no-override` it uses the template the container carries. Differences from HF,
all of them from the `chat.cpp` and the llama.cpp Jinja that radiance uses, not from the template:

- content in parts is joined with `\n` (for every template that does not index the list); HF joins
  it with nothing;
- numbers: an integral float loses its `.0` (`1.0` comes out as `1`) and an integer ≥ 2^63
  overflows, in the call arguments of the history and in the tool schemas;
- a tool without `description` or without `parameters` gets them empty; `defer_loading: true` is
  lost (the tool shows up in `<tools>`, HF hides it);
- from the assistant message only `reasoning_content` is read, not `reasoning`; the `developer` role
  is merged into the system one; an assistant message at the end is treated as a prefill (without
  `hy_end`);
- an assistant turn with empty `arguments` renders without arguments; with broken JSON (that of a
  cut-off call the client sends back in the history) the request returns 400, without reaching the
  template;
- when reading responses, a call cut off mid-generation gives broken JSON arguments in the stream,
  and a `["string","null"]` parameter with `null` comes out as the string `"null"`.

`reasoning_effort`: the template only accepts `no_think` and `high` (Tencent dropped `max` from it on
2026-11-19; the copy in `data/` follows), and the OpenAI levels (`low`, `medium`) return 400. It is
served with `reasoning-effort-map: {low: high, medium: high, xhigh: high, max: high, minimal: no_think,
none: no_think}`
(`deploy/example-tp2.yaml`; `tests/test_reasoning_map.py` checks it against a live server: each
level gives HF's prompt tokens with the target level, levels with the same target give the same
logprobs and `high` and `no_think` give others; an explicit `null` goes to `high`).

## Vision (plan step 11)

Images, no video. It needs the `Hy4` family of the radiance core processor (local branch
`hy4-vision`, `core/mm/processor.cpp`): the Qwen-VL one (smart_resize with factor 32, mean and
standard deviation 0.5, band from `processor_config.json`), with `merge` more patch columns per
row. That column is where HF puts the learned line break, so the number of markers
(`h/2 · (w/2 + 1)`), the M-RoPE grid (width `w/2 + 1`) and the text advance after the image
(`max(h/2, w/2 + 1)`) come out as in HF without touching anything else in the core.

The plugin side, only with `mm-max-patches` (without it, the text path does not change at all):

- The tower (`vision.h`): `hy4_vit_compact` removes the line-break column, patches by GEMM, learned
  positions (`hy4_vit_pos`), 27 ViT blocks without RoPE with exact GELU, the adapter (RMSNorm, the
  2×2 conv as a GEMM with its weight permuted in the constants, `hy4.vit_merge_w`, GELU and the 1×1
  conv) and `hy4_vit_newline`. Weights replicated on each card, like `rad_block_vit.h`.
- The text: `hy4_rope` (GPT-J pairs with the interleaved M-RoPE sections) over `RadBatch::rope_pos`
  in the MLA, the 3-D positions in the indexer, and Engram without writing into the image and video
  rows. With images, the indexer score plane covers the whole step (`rope_pos` is not chunked): with
  `max-num-batched-tokens` 2048 and 196K of context that is 384 MiB of f32 per card instead of 192.

Pending (review, V2): an indexer group whose first token arrived in an earlier step takes its
phase from `qsa_work` as `pos0 + offset of the step's first row`, which is only exact if that token
is text with the same offset. It can fail when resuming from the prefix cache in the middle of a
group right after an image, with more than 2048 tokens of context. Not tested against HF.

## Known differences from the reference

- The router reads its input in BF16 (`gemm_nt_f32` with BF16 activation); HF casts it to FP32
  first. It falls within the BF16 noise floor.
- The Engram mask of the reference (`ref/reference.py`) is all ones; no prompt has token 120002,
  so it matches the plugin's in the comparisons. HF also excludes the image and video tokens; the
  plugin is text only and does not handle them.
- The MLA absorption (plan step 6) changes the order of the sums against the non-absorbed form.

## `w4nl64` experts in radiance 2.1.0 and 2.2.0 (plan step 7)

Two engine gaps for rotated experts with clamped SwiGLU, like the Hy4 ones:

- **`had_quant_act_fp8`** (the rotated E4M3 activation) is only declared by libhalo: `hy4k`
  provides it (`docs/OPS-HY4.md`).
- **`GateQuantFP8`** (`arch/common/rad_fp8.h`) rejects `rot` with clamp.
  `arch/hyv4flash/vendor/arch/rad_fp8.h` is a copy of the 2.2.0 header with one change in that
  branch (search for "hyv4flash (vendor)"): the clamped product (`swiglu_clamped`) goes to the bf16
  plane and `had_quant_act_fp8` rotates and quantises it. It is included before the installed one
  (`target_include_directories(... BEFORE ...)`); the include guard keeps the original out,
  `hyv4flash.cpp` checks with `HY4_RAD_FP8_VENDOR` that the copy got in, and the root CMake stops
  with any ABI other than 43 (2.2.0). When moving to a new radiance version: redo the copy from the
  new header, or remove it if upstream already serves that case.

The plugin probes the experts of each layer before the buffers (like nemotronh): with quantised
experts it declares the E4M3 codes of the latent (MoeFP8 requires them) and the rotated latent
`hr`, which the block itself writes; with unrotated experts, also the latent quantiser.

Conversion, in the build image (with the card plugins):

    tools/convert-hip.sh <checkpoint in /models/...> -o /m/....rad --recipe /hy4data/recipes/hy4flash-w4nl64-rtn.recipe \
        --add /m/full-consts/engram-consts.safetensors --kernels hy4k,libr4d,libavx,libref --max-spec 0 --max-ctx 65536

`--max-spec 0`: the plugin rejects speculation. `--max-ctx`: the score plane at 1M of context goes
over the cap.

**Speed.** The first version of `had_quant_act_fp8` (FWHT in LDS, 14 barriers, one block per
group) made the RTN trimmed model more than 6× slower than BF16 on one R9700 (`largo` 22.8 s
against 3.6). With the FWHT in registers (fa783b7): `medio` 0.70 s (BF16 0.65) and `largo` 4.2 s
(BF16 3.6). The rest is measured in plan step 10.

## Moving to radiance 2.2.0

The plugins build against 2.2.0's installed headers only (ABI 43, `RAD_ABI_VERSION`; the loader
refuses a plugin of another ABI). The 2.1.0 tree is the snapshot taken before the move. Builds:
`build-22` (host, against `prefix-2.2`, radiance v2.2.0 installed with `RAD_WITH_HIP=OFF`) and
`build-hip-22` (`tools/build-hip.sh`, in `radiance-build:2.2.0`: the `build` stage of radiance's
own Dockerfile at v2.2.0, GPU_TARGETS gfx1201). `deploy/run-hy4.sh` takes the image and
the libr4d from `HY4_IMAGE` and `HY4_LIBR4D` (`stilldeadcode/radiance:2.2.0` and the 2.2.0 libr4d
without b264e9b, built by <https://github.com/millaguie/radiance-wave-rescale>).

The conversion scripts (`deploy/conv-clean.sh`, `conv-dfly.sh`, `conv-engram-e4m3.sh`,
`set-engram-key.sh`) run 2.2.0's `rad-convert` with the plugins of the checkout's `build-hip-22`
(`HY4_IMAGE`, `HY4_PLUGINS` and `HY4_REPO` override them). Every one states the Engram tables as
the served container holds them, E4M3 codes: `--set hy4.engram_codes=rowmax448` and the codes
overlay as a source, the full verified one in `conv-clean.sh`, `conv-dfly.sh` and
`conv-engram-e4m3.sh`, and its header-only stand-in in `set-engram-key.sh`, on purpose (that run
keeps every weight, so nothing is read from it). `conv-clean.sh` writes the publishing copy to a directory of its own
(`~/models/rad-pub/hy4-flash-w4nl64-gptq-iq8r-dfly2-e4m3.rad`; its first run wrote it to
`~/models/rad/`, before the output had its own directory): 140.21 GiB of weights, the served
container's 43630 weights less the 42 expanded `attn_kv_b`. It checks in the run's last line that
every weight was copied, and with `rad-info` that the output holds 43588. `--plan-only` does not
match weights against the `--reuse` container (rad-convert stops before), so that was checked by
comparing the verbose plan with `rad-info -v` of the served container: the 43588 planned weights
have the encoding, quantiser, options and size it holds.
`conv-dfly.sh` without `DFLY_I8=1` fails at declare against this container (`dfly.fusion.eh.weight`
declared twice, on 2.1.0 as well). That is a plugin bug in `dfly_plan` (the premapped probes are
declared twice), fixed separately, not a recipe mismatch; and only `DFLY_I8=1` applies to the
served container now, which already holds the drafter in IQ8R.

What 2.2.0 changed that the plugin reads, and what the plugin does with it:

- **`GateQuantFP8::declare`** (`rad_fp8.h`): the clamped branch now serves rotated int8 codes
  (`had_quant_act_i8g`, for an i8a8h consumer), but still refuses a rotated E4M3 consumer (`rot`).
  So the vendored copy stays, redone from 2.2.0's file: the 2.2.0 int8 form is kept as upstream
  has it, and `rot` (not fed int8, not bf16) takes `had_quant_act_fp8` at `a.cg()`, which is
  `RAD_FP8_BLOCK` when the codes are not int8. No other part of the copy differs from 2.2.0.
- **`MoeFP8`** (`rad_block_moe_fp8.h`): a six-bit codebook (`w6nl64a8h`), a trellis gate_up with a
  codebook down, the shared down projection in front of the routed one (`sh_down_first_*`, off by
  default) and the `split` operand for a checkpoint inside the step. The down GEMM now reads the
  codes of the gated quantiser's own pair (`gq_exp.a`), which for `w4nl64a8h` experts is the same
  pair as before. None of these is turned on by the plugin: Hy4 has no linear state, so
  `RadArchProbe::ckpt_snap_ok` stays 0 and `RadBatch::ckpt_cut` is never set, and the shared arm is
  the plugin's own (it does not go through `MoeFP8`'s).
- **`RadBuildCtx::span_pass_step`**, **`RadArchProbe::ckpt_snap_ok`**, **`RadBatch::ckpt_cut`** and
  **`rad_ckpt_snap`** (ABI 42 and 43) are appended fields and calls: zero-initialised, the plugin
  keeps 2.1.0's behaviour. `RAD_DRAFT_HISTORY` is a new drafter kind; DFly2 stays `RAD_DRAFT_BLOCK`
  and `RadDrafterDecl` did not change.
- **`RAD_ARCH_TRY`** (`rad_arch.h`) now prints where a declare status crossed. The plugin uses its
  own `HY_TRY`, so its messages are as before.
- The op contracts the plugin declares (`docs/OPS.md`, `docs/QSA.md`) did not change; QSA's
  "below the bound" forms are `QsaIndexer`'s, which the plugin does not use (its indexer is
  `hy4_idx_*`). `had_quant_act_fp8` is still declared only by libhalo, so `hy4k` keeps its own.
  `RadTensor`, `RadParam`, `RadArgs` and `RadKernelInfo` did not change (`tests/rad_ctypes.py`).
- The YAML keys the example configs use are all in 2.2.0 under the same names (2.2.0 only adds
  keys: `draft-verify`, `draft-depth*`, `draft-suffix*`, `prefill-span-*`, ...).


Checked on 2026-11-21 (the test machine, 2x R9700, the GPTQ container `hy4-flash-w4nl64-gptq.rad`, MLA
absorbed). The hy4k device code `build-hip-22` builds is byte for byte `build-hip2`'s (2.1.0): only
`hyv4flash.so` differs.

- Host: `test_ops` 18, `test_ops_extra` 63, `test_hc` 3 + layer 0, `test_rope_mc` 8 cases, all
  OK; `test_chat_template` gives the same 18 ok and the same one FAIL as on 2.1.0 (the static
  check against the trimmed checkpoint's template, which the data/ template no longer matches since
  `max` was taken out of `reasoning_effort`). `tools/kernarg.sh`: 92 kernels, largest argument
  segment 736 bytes, no kernel at 256 VGPRs, no scratch.
- Card: `tests/test_dev.py` 190 cases OK.
- KL (`perf-kl-22.yaml`, 210 W) is bit for bit 2.1.0's: mean 0.18608, every one of the 10851
  positions equal, top-1 88.19 %, no top-1 change.
- DFly2 (`num-speculative-tokens: 5`, 210 W): the greedy texts with and without the drafter are the
  same bytes, and the same bytes as 2.1.0's without it (short and long prompt).
- Speed (330 W, a short `bench_throughput` run; decode at 1 / 4 / 8 clients, prefill at 5.7K / 18K tok/s):

| | decode 1 | 4 | 8 | prefill 5.7K | 18K |
|---|---|---|---|---|---|
| no drafter, 2.1.0 | 56.5 | 193.5 | 338.6 | 3,054 | 2,753 |
| no drafter, 2.2.0 | 57.5 | 196.2 | 341.0 | 2,936 | 2,738 |
| DFly2 adaptive, 2.1.0, same session | 109.3 | 361.5 | 632.0 | 2,926 | 2,672 |
| DFly2 adaptive, 2.2.0 (two runs) | 110.9 / 110.8 | 360.7 / 364.5 | 422.0 / 415.3 | 2,962 / 2,939 | 2,690 / 2,693 |
| DFly2 adaptive, 2.2.0, `RADIANCE_AQL_CHAIN_FENCE=1` (two runs) | 109.6 / 110.0 | 327.1 / 360.0 | 602.9 / 512.9 | 2,937 / 2,961 | 2,659 / 2,682 |

  With the drafter at 8 clients 2.2.0 loses a third against 2.1.0. Its startup report says why it
  can: with hy4k loaded every chainable row says 2 (hy4k has none), so 2.2.0 issues the chained
  launches without the packet processor's acquire ("their packets take no acquire", `aql:
  01f08152`). Keeping the acquire (`RADIANCE_AQL_CHAIN_FENCE=1`, which `deploy/run-hy4.sh` passes
  through) brings 8 clients back into 2.1.0's spread (2.1.0 gave 528 in the morning run and 632
  here). Until more runs settle it, a 2.2.0 deployment with DFly2 sets it.
