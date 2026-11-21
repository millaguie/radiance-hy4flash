# `hy4k` ops

Specification of the new ops of the `hy4k` kernel plugin, in the format of
`radiance/docs/OPS.md`. Each op has an FP32 host kernel (`kernels/hy4k/ref_*.cpp`), which serves
as the oracle, and a HIP card row (`kernels/hy4k/dev_*.hip`, plan step 6) that
`tests/test_dev.py` compares with it on the local GPU. See "Card rows" at the end.

Tests: `tests/test_hc.py`, `tests/test_ops.py` and `tests/test_ops_extra.py` call each kernel with `ctypes` and compare with
the HF code (Transformers PR #50903, commit `c2ec850`) and with the dumps of
`ref/reference.py`. Tolerance in FP32: relative L2 < 1e-5, which is the rounding of long sums in
another order. Integer results (ids, selection) must be equal.

## Table

| op | params | operands |
|---|---|---|
| `hy4_hc_pre` | `M` `n` `hc` `head`/opt `eps`:f64 `hc_eps`:f64 `magnitude`:f64 `dtype` | `h`, `fn`w, `base`w, `scale`w → `x`, `post`? |
| `hy4_hc_post` | `M` `n` `hc` `dtype` | `h`:inout, `y`, `post` |
| `hy4_mla_kv` | `M` `heads` `nope` `rope` `v_dim` `dtype` | `kvx`, `k_rope` → `k`, `v` |
| `hy4_mla_attn` | `M` `heads` `latent` `rope` `block_size` `window`/opt `scale`:f64 `rope_pitch`/opt `max_seqs`/opt `max_ctx`/opt `dtype` | `q`, `q_rope`, `kv_cache`, `block_table`, `seqused`, `cu_seqlens`?, `sinks`w → `out`, `bkeys`?, `bcu`? |
| `hy4_mla_project` | radiance's `mla_project`: `M` `heads` `head_dim` `latent_dim` `mode` `dtype` | `a`, `b`w → `y` |
| `hy4_ngram_ids` | `M` `engram_layers` `heads` `ngram` `pad` `seqs` | `tok`, `pos`, `cu`, `state`:inout, `sidx`, `map`w, `mult`w, `vocab_sizes`w, `offsets`w → `ids`, `ahead_ids`?, `num_accepted`? |
| `hy4_rows_rmsnorm` | `M` `heads` `d` `fp8`/opt `eps`:f64 `dtype` | `x`, `w`w → `y` |
| `hy4_engram_gate` | `M` `n` `hc` `pad`/opt `skip`/opt `skip2`/opt `eps`:f64 `gate_eps`:f64 `dtype` | `h`:inout, `tok`, `k`, `v`, `norm_key`w, `norm_query`w, `pre_gamma`w, `per_hc`w |
| `hy4_idx_pool` | `M` `n` `ratio` `rotary_dim` `block_size`/opt `theta`:f64 `eps`:f64 `dtype` `sections`/opt | `stage`, `value`w, `gate`w, `ape`w, `norm`w, `page`, `bpos`, `cache`:inout |
| `hy4_idx_score` | `M` `n` `heads` `rotary_dim` `theta`:f64 `blocks` `seqs` `dtype` `sections`/opt | `q`, `w`, `pos`, `cache`, `bt`, `cu`, `nc` → `score` |
| `hy4_idx_select` | `M` `blocks` `topk` `ratio` `seqs` | `score`, `bt`, `nc`, `pos`, `cu` → `sel`, `nsel`, `seqused` |
| `had_quant_act_fp8` (standard op) | `M` `n` `group` `dtype` | `x` → `q`, `scale` |
| `hy4_rope` | the `rope` schema: `M` `head_dim` `n_head` `n_head_kv`/opt `theta`:f64 `scale`:f64 `mode` `rotary_dim`/opt `sections`/opt | `qk`:inout, `positions` |
| `hy4_vit_compact` | `M` `dim` `merge` | `pixels`, `coord`, `cu` → `pixels_c`, `coord_c`, `cu_c` |
| `hy4_vit_pos` | `M` `n` `side` `dtype` | `x`:inout, `table`w, `coord_c` |
| `hy4_vit_newline` | `M` `n` `merge` `dtype` | `y`, `newline`w, `coord`, `cu`, `cu_c` → `out` |
| `hy4_ring_store` | `M` `ring` `width` `offset` `seqs` `mode`:str `block`/opt | `src`?, `row_ids`, `pos`, `cu`, `sidx`, `state`:inout |
| `hy4_ring_load` | `M` `ring` `width` `offset` `seqs` `fill`/opt | `pos`, `cu`?, `sidx`, `state` → `dst` |

## Common rules

- The row count comes from the operands' extent, and the sequence count from `cu`'s
  (`S + 1` entries). `seqs` is only a ceiling (`docs/PLUGIN.md` §3): a step with fewer live
  sequences than `max_seqs` is the normal case.
- Before the first write, everything an operand carries that points into another is checked:
  `0 ≤ cu[s] ≤ cu[s+1] ≤ rows`, each `sidx` slot against the rows of `state`, each page against
  the cache's, the row's own group against the width of `bt`, non-negative counts, shapes, widths
  and dtypes. If anything does not fit, the op returns an error (`RAD_E_SHAPE` or `RAD_E_INVAL`)
  and does not write.
- Zero rows is not a valid call.
- `cu` may be a span of the step's offsets (`cu + n_seq_decode`, `docs/PLUGIN.md`
  §3 and §8): it need not start at 0 nor end at the row count. Rows outside the
  spans of `cu` are not written.
- A launch does not allocate memory (`rad_abi.h`, `RadLaunchFn`): each kernel uses fixed arrays on
  the stack and rejects the geometry that does not fit (limits in `kernels/hy4k/host_util.h`).
- Computation is in FP32 with a single rounding to the output dtype, the contract of radiance's
  reference kernels (`docs/PLUGIN.md` §6). HF materialises some intermediates in BF16 when the
  model runs in BF16 (the RMSNorms, Engram's FP8 round trip); those differences are not reproduced
  and are part of the BF16 noise floor that the BF16 pass of `ref/reference.py` measures.
- `sidx` may be `[S]` or `[S, pitch]`: column 0 of each row is read (the committed slot).
- The host rows are oracles: they are marked as reference row by row (`rad_kernel_reference`), so
  the engine only serves them with `server.debug-accept-reference-kernels`. The card rows are
  production rows (`rad_plugin_reference` = 0).

## `hy4_hc_pre` and `hy4_hc_post`

Hy4's independent hyper-connections (HF `HYV4FlashHyperConnection`, `HYV4FlashHyperHead`).
`h` holds the row's `hc` streams, `[M, hc·n]`. All in FP32:

```
flat = h as [hc·n]
inv  = 1 / sqrt(mean(flat²) + eps)                  eps = rms_norm_eps (1e-5)
m_j  = (flat · fn_j) · inv                          j < 2·hc  (hc with head=1)
pre_j  = σ(m_j·scale[0] + base[j]) + hc_eps
post_j = magnitude·σ(m_{hc+j}·scale[1] + base[hc+j]) + hc_eps     (without head)
x    = Σ_j pre_j · h_j                              rounded to x's dtype
```

`fn` is f32 `[2·hc, hc·n]`; `post` is f32 `[M, hc]`. With `head=1` it is the model's closing step:
`fn` `[hc, hc·n]`, a single `scale`, no `post`.

`hy4_hc_post` writes a sub-block's output into the streams: `h_j = post_j·y + h_j` in FP32,
rounded to `h`'s dtype.

## `hy4_mla_kv`

Builds K and V per head in the unabsorbed form of the MLA. `kvx` is the output of `kv_b`,
`[M, heads·(nope + v_dim)]`, per head `k_nope` and then `v`. `k_rope` is the shared plane of 64,
already rotated, `[M, rope]`. Outputs: `k [M, heads, rope + nope] = [k_rope | k_nope_h]` and `v [M, heads,
v_dim]`. The rope goes **first** because radiance's `rope` rotates the first `rotary_dim` columns;
the rows of `q_b` are permuted the same way at conversion.

## `hy4_mla_attn`

Hy4's attention in the absorbed form of the MLA, over the latent cache. A single KV head for all
the query heads: token `j`'s row is `[lat_j | rope_j]`, the normed latent (`latent`, 512) and the
rotated part of the key (`rope`, 64), as `kv_store` writes it with `head_dim`
`(latent + rope) / 2` and `n_head_kv` 1 (one half in K and the other in V, contiguous). `kv_cache` is
`[pages, 1, block_size, ≥ latent + rope]`.

The query arrives absorbed: `q [M, heads, latent]` is each head's `q_nope · W_k` (with
`mla_project` `mode=absorb`) and `q_rope [M, heads, rope]` its rotated part, each with its own row
and head stride. 2-D works too, `[M, heads·width]` with the heads one after another; a 2-D `q_rope`
carries head `h` in columns `[h·rope_pitch, h·rope_pitch + rope)`, so the plugin passes the unabsorbed
`q` (`[rope | nope]` per head, 256) with `rope_pitch` 256 and copies nothing. The score is `s_j = scale · (q · lat_j + q_rope · rope_j)` with `scale` from the head's
original width (`(nope + rope)^-0.5`), and `sinks` `[heads]` f32 enters the softmax as one more
logit with no value: `out = Σ_j p_j lat_j`, `[M, heads, latent]`. The output is expanded afterwards with
`mla_project` `mode=expand`. A row with no visible key gives zero.

Visibility is `attn_paged_sink`'s: with `cu_seqlens`, sequence `s` has rows `[cu[s],
cu[s+1])` and row `i` is at position `seqused[s] - qlen + i`; without `cu`, the rows are split
evenly among the sequences of `seqused` (the selection form is one row per "sequence" and
`block_table` its page list). Causal; with `window` > 0 only `pos - j < window`. Rows outside
the spans of `cu` are not written (common rule). `latent` ≤ 512 and `rope` ≤ 64 on the host too: the
oracle accumulates in a fixed stack array, in two passes (maximum, then weights), without allocating.

It is exact against the unabsorbed form: `tests/test_ops.py` compares the oracle (double) with HF's
attention assembled by hand (`k = [k_rope | W_k·lat]`, `v = W_v·lat`) in the dense, windowed and
selection forms (L2 3·10⁻⁸).

**Block keys** (operands 8 and 9, the DFly2 drafter, `docs/DFLY2.md`). They come after `out`, as
trailing optional operands, so a caller without them issues the 8 operands it always did. `bkeys [rows, ≥ latent + rope]`
holds the block rows' own `[lat | rope]` in batch row order, `bcu [S + 1]` (i32 or i64) the block rows
of each drafter sequence. Row `i` of block sequence `d` attends to its cached keys, as without them, and
to `bkeys[bcu[d] .. i]` (causal), one softmax over the union with the sink. Rules, checked before
writing:

- both operands or neither (`RAD_E_INVAL`);
- the selection form only: no `cu_seqlens` and one `seqused` entry per row (each row is its own
  "sequence" of `block_table`), otherwise `RAD_E_INVAL`/`RAD_E_SHAPE`. The cached part is then tokens
  `< seqused[i]` of the row's page list, an empty history being `seqused` 0;
- no `window` (`RAD_E_INVAL`): a selection's positions are not contiguous, so a window over the union
  has no meaning, and the drafter's layers are sparse ones, which run without;
- `bcu` follows the common rule for `cu` (`count_seqs`, ceiling `max_seqs` when given): it may be a span,
  and rows outside its spans are not written.

A row with block keys always sees at least its own, so it never gives the "no visible key" zero.
`tests/test_ops_extra.py` compares the oracle with the union computed in double from the contract
(`tests/dfly_ref.py`), with histories from 0 to 40 tokens, invalid pages, causality, the sink at
+50 and −∞, and the refusals (L2 2·10⁻⁸).

## `hy4_ngram_ids`

The ids of the Engram tables of every layer with Engram (`engram_layers` = `E`), in a single
emission per step. `ids` is `[E, R, heads]` or `[E·R, heads]` i32 with `R ≥ M` (the architecture
declares it 2-D with `R = max_tok`, and each layer's gather reads its span as `[T, heads]`); rows
`0..M-1` of each span are written.

Per sequence `s` (rows `cu[s]..cu[s+1]`) and row `r` at position `p`:

- `t(p-k)` is `tok[r-k]` if that row belongs to the same sequence and the same step; otherwise the
  state (`state[sidx[s]] = [t(last), t(last-1)]` from the previous step); and `pad` if `p-k < 0`.
- `c_k = map_e[t]` if `t < V` (`V` = 120818), otherwise `map_e[pad]`.
- Heads `0..heads/(ngram-1)-1` are bigrams, the following ones trigrams (in the model, 0-7 and 8-15).
- `mix_n = c_0·mult_{e,0} ⊕ … ⊕ c_{n-1}·mult_{e,n-1}`, int64 products with overflow.
- `id_h = floor_mod(mix_{n(h)}, vocab_sizes[e,h]) + offsets[e,h]`.

`state` is a chronological window of `W ≥ ngram − 1` unmapped ids per slot, as radiance's
`ngram_ids` keeps it. The committed ids before the step are `state[sidx[s]][o .. o + ngram − 2]`, oldest
first, with `o = num_accepted[s] − 1` when the optional `num_accepted` (operand 11, i32, one-based) is
given and 0 otherwise. Afterwards, without `num_accepted`, the step's last `ngram − 1` ids (older ones
from the window if the step is shorter) land at offset zero; with it (a verify step), the read window
shifts down by one and every id of the step follows, so a next step that kept k reads at k − 1; the
window must hold `ngram − 2 + rows`. A sequence with `sidx < 0` neither reads nor writes state.

`tok` may carry `A` more tokens after the step's rows (`M` is `pos`'s extent): the next prefill chunk
of the last non-empty sequence (`RadBatch::n_ahead`). With the optional output `ahead_ids` (operand
10, the layout of `ids`) their ids are written there as the step that runs them will compute them,
positions continuing the sequence's; the state advances by the step's rows only.

## `hy4_ring_store` and `hy4_ring_load`

The DFly2 drafter's ring (`docs/DFLY2.md`, "The ring"): per sequence, `ring` (= depth + 1 = 6) entries
of `W` i32 words, entry `e` holding what the trunk produced for the row at a position `≡ e (mod
ring)`. Word 0 of an entry is the position stored there; the layout of the other words is the
plugin's (`h` as f32 bits, then `seqused` and the pages of each drafter layer). Copies only: the host
and the card give the same bits.

`state` is i32, `[slots, …, ring, W]`, or `[slots, …, ring·W]` as a `RAD_KV_LINEAR` group hands it out
(entries back to back in the last axis); every other middle axis is 1. The axis before the last is the
entry axis when its extent is `ring`, otherwise the last axis is split in `ring` equal entries. The slot
of sequence `s` is `sidx[s]` (column 0 of `[S, pitch]`), as in `hy4_ngram_ids`; a negative slot neither
reads nor writes.

**`hy4_ring_store`.** `row_ids [M]` lists batch rows; `pos [rows]` and `cu [S + 1]` are the step's
(rows from `pos`'s extent, `cu` may be a span). For each `k`: `r = row_ids[k]`, `s` the sequence whose
span holds `r` (none: skipped), `slot = sidx[s]`, `e = pos[r] mod ring` (non-negative). It writes words
`offset .. offset + width − 1` of `state[slot][e]` by `mode`:

| `mode` | words | `src` |
|---|---|---|
| `pos` | `pos[r]` (`width` 1) | absent |
| `row` | row `r` of `src` (per token: `idx_sel`, the selection's `seqused`) | `[rows, ≥ width]`, or `[rows]` with `width` 1 |
| `out` | row `k` of `src` (per logit row: the head norm's output) | `[≥ M, ≥ width]`, or `[M]` with `width` 1 |
| `dense` | `pos[r] + 1`, then columns `0 .. width − 2` of `src[s]`, −1 past column `pos[r] / block` (and past the table) | a block table `[seqs, cols]`, i32/i64; `block` ≥ 1 required, `width` ≥ 2 |

A source may be i32, i64 (truncated to 32 bits), f32 (its bits), bf16 or f16 (widened to f32, exact,
then its bits). Checked before the first write: `row_ids` within `pos`, `cu` (`count_seqs`), each slot
used against `state`'s slots, the source's rows (`r` for `row`, `s` for `dense`, `M` for `out`),
`offset + width ≤ W`, `src` present exactly when the mode takes one. Two listed rows that write the
same `(slot, entry)` in one call (more positions of a sequence than `ring`, or a repeated row id): the
later listed wins, on both sides. The host writes in list order; the card, where the rows run in
parallel, skips row `k` when a later `k' > k` that is written (not skipped) lands on the same slot
and entry, so no two blocks write one entry. In the plugin's wiring a sequence has at most 6 logit
rows, so this does not happen; it is defined so host and card cannot disagree.

**`hy4_ring_load`.** With `cu`, `pos` is per row and `p0[s] = pos[cu[s]]`, the position of the
sequence's first block row; every row of `[cu[s], cu[s+1])` gets the words in `dst[r]`, and every
row `< rows` outside the spans gets `fill` (an empty sequence has no rows). Without `cu`, `pos [S]` is
per sequence and `dst[s]` gets them. The entry read is `e = (p0 − 1) mod ring`, words `offset .. offset
+ width − 1`. When `slot < 0`, `p0 < 1`, or the entry's position word is not `p0 − 1` (a stale or
never-written entry), the words are `fill`: the drafter then proposes from garbage and the verify
rejects it (acceptance lost, never a token). `fill` is an optional i32 param, −1 by default (refused
outside i32); `dst` is i32 or f32 and gets the stored bits, so `h` comes back as it was stored, and a
fill of −1 in an f32 plane is a NaN: the plugin passes `fill = 0` for the `h` load (0.0) and keeps −1
for the histories (an empty count, no pages). `dst` is `[≥ rows, ≥ width]`, or `[rows]` with `width`
1; rows past `rows` (or `S`) are not written. A slot outside `state` is an error on the host (nothing
written); the card writes the fill in that sequence's rows and the entries of the others.

`tests/test_ops_extra.py` checks both against the contract's model (`tests/dfly_ref.py`): every mode,
both state layouts, a span `cu`, a sequence without slot, the fill cases (−1, `fill` 0 into f32, rows
outside `cu`), duplicate entries (13 rows of one sequence, ring 6) and the refusals (state
untouched).

Decisions the spec (`docs/DFLY2.md`) left open, taken here:

- `dense` pages "past page `pos[r] / block`": the page holding position `pos[r]` is kept, the next
  ones are −1, so the word count `pos[r] + 1` and the pages agree; columns past the table's width are
  −1 too. A count larger than `(width − 1)·block` is stored as is (the attention cannot read past the
  page list, `j1` is clamped to it).
- `out` and `row` with float sources: bf16 and f16 are accepted and widened to f32 before taking the
  bits (the head norm's output is the trunk's dtype); `h` therefore always lands as f32 bits.
- `pos` with a `src` is refused rather than ignored, so a wiring mistake shows.
- `ring_load` takes the position word at word 0 of the entry, whatever `offset` is.

## `hy4_rows_rmsnorm`

RMSNorm of each `d`-column span of the row with its own weight: `x [M, heads·d]`,
`w [heads, d]`, eps 1.1920929e-07 by default (the one of the Engram norms).

With `fp8 = 1`, before the norm each span goes through E4M3 the way Engram reads its table (HF
`_fp8_round_trip`): `scale = max(amax, 1e-12)/448`; `q = e4m3(clamp(x/scale, ±448))·scale`
(round to nearest even); the result is `x + (q − x)`, as in HF, which in floating point is not
always `q`. All in FP32. The Engram table is stored as is in BF16 and the FP8 round trip happens
here, at run time: that way it is exact against the FP32 reference. Against the BF16 one it is not
(HF rounds the scale, the division and the product to BF16); the difference goes into the BF16
noise floor.

## `hy4_engram_gate`

Engram's write into the streams. `k [M, hc, n]` are the outputs of the `key_projs` (with bias,
not normalised), `v [M, n]` that of `value_proj` (with bias). Per row:

```
key_i   = rms(k_i)·norm_key_i;  query_i = rms(h_i)·norm_query_i
d_i     = ⟨key_i, query_i⟩ / sqrt(n)
g_i     = σ(sign(d_i)·sqrt(max(|d_i|, gate_eps)))                  sign(0) = 0
s       = pre_gamma · rms(v)
h_i    += g_i·s + per_hc_i·s
```

`rms` has no weight and eps 1.1920929e-07; `gate_eps` is 1e-6. If `tok` is `pad`, `skip` or
`skip2`, the row does not change.

HF leaves image and video positions out of Engram (`engram_mask`, `image_token_id` 120120 and
`video_token_id` 120124): the plugin passes those two ids in `skip` and `skip2` when it serves the
tower.

## `hy4_idx_pool`

The indexer's group keys. `stage [M·ratio, n]` holds the per-token keys, already normalised
(`token_key_norm`), of the `M` groups that close, from `qsa_work`. Per group `b` with
`page[b] ≥ 0`, in FP32:

```
v_j = value · k_j;   g_j = gate · k_j + ape_j                j < ratio
pooled = Σ_j v_j · softmax_j(g)                               per channel
key = rmsnorm(pooled, eps) · norm   -> rounded to the cache dtype
rope of adjacent pairs of the last rotary_dim at bpos[b], theta    -> cache[page[b]]
```

The rope uses `inv_freq = 1 / theta^(2i/rotary_dim)` and the phase `bpos·inv_freq`, both in float,
as HF does. If the cache is BF16, cos and sin are rounded to BF16 before multiplying, as HF does
(`cos.to(dtype=x.dtype)`). `block_size` (optional) exists so that the `kv_idx_keys` group takes its
page size from this op (`core/build/rad_builder.cpp:385-407`); it must equal `ratio`.
Pages must be unique within a call.

## `hy4_idx_score`

`q [M, heads, n]` (or `[M, heads·n]`) are the indexer's queries **unrotated**; the op rotates them at `pos` with the same
rope (cos and sin in BF16 if `q` is BF16) and rounds them to `q`'s dtype. `w [M, heads]` are the unscaled head weights. For each
group `g < nc[s]` with page `bt[s][g] ≥ 0`:

```
score[r][g] = Σ_h w_h · heads^-½ · n^-½ · relu(q_h · k_g)
```

and −∞ elsewhere. It does not mask per query. The `score` plane may have more columns than
`bt` (it is declared by `max_ctx`); an `nc` larger than the width of `bt` is an error (`RAD_E_INVAL`).

With images, `hy4_idx_pool` and `hy4_idx_score` receive `sections` (those of `mrope_section`) and
3-component rotary positions: `bpos [M, 3]` (what `qsa_work` writes with the step's planes) and
`pos [3, M]` (`RadBatch::rope_pos`). Frequency `i` reads component 1 if `i % 3 == 1` and
`i < 3·s1`, component 2 if `i % 3 == 2` and `i < 3·s2`, and component 0 otherwise. With `bpos [M]`,
`[M, 1]` or `pos [M]` the position is 1-D, even with `sections`. `hy4_idx_select` keeps the token's
position (the own group comes from the slot, not from the rotary).

## `hy4_idx_select`

Per query at `pos`: candidates `g < min(nc[s], pos / ratio)`, that is, the groups closed before
the query's own. A candidate with a negative page in `bt` or a non-finite score (−∞, NaN) does
not count. It picks the `topk - 1` with the highest score; on equal score the lower index wins. Output in ascending group index order, with the own group (`pos / ratio`) at
the end, as pages of `bt` (the attention group's table). The rest of `sel` is set to −1.
`nsel = chosen + 1`; `seqused = (nsel − 1)·ratio + pos % ratio + 1`.

The op does not look at the 2048 threshold: the architecture emits it only when the context
passes it.

## Vision

The vision tower (`arch/hyv4flash/vision.h`) uses radiance ops (`gemm_nt_bias`, `layernorm`,
non-causal `attn_dense`, `rmsnorm`) and these three. The processor (the core's Hy4 family, branch
`hy4-vision`) gives the patches in `merge×merge` block order with `merge` extra columns per row:
the line-break column.

- `hy4_vit_compact`: copies the patches whose column is less than their segment's `width − merge`,
  in the same order; `coord_c [Pc, 4]` is (row, column, height, width without the column) and `cu_c`
  the new boundaries. It rejects a segment without the column (width ≤ merge) and an output of
  another size.
- `hy4_vit_pos`: `x += interp(table)`, the table `[side² + 1, n]` without row 0 (the class token's)
  resampled bilinearly to each patch's grid as HF does:
  `F.interpolate(scale_factor=(height + 0.1)/side, align_corners=False)`, that is
  `src = (dst + 0.5) · side/(height + 0.1) − 0.5`, with 0 as the minimum, in f32. The resampled
  value is rounded to `x`'s dtype before it is added.
- `hy4_vit_newline`: `out [Pn/merge², n]` on the processor's grid; the row of a block in the last
  column is `newline`, the others take the row of `y` that falls to them.

How the tower is checked against HF (`ref/compare_vision.py`): with `RADIANCE_DEBUG_MM_DUMP=<dir>`
(a core diagnostic, `core/engine_mm.cpp` in radiance 2.1.0), radiance dumps the patches of each
image in block order as `mm_<key>_<t>x<h>x<w>.pix.bf16` and the tower's rows as `….bf16`, raw
BF16 with no header (`[patches, 768]` and `[rows, 3072]`; `w` includes the line-break column). The
script compares them with `ref/vision_ref.py` and runs HF's tower on radiance's pixels to separate
the processor's error from the tower's. Result of 2026-11-20 on the one-layer trimmed model (L2
against HF FP32, HF's floor in BF16 in brackets): landscape 1.55e-2 (1.66e-2), wide 1.59e-2
(1.71e-2), small 2.11e-2 (1.11e-2); processor ≤ 5e-4.

The processor patch shrinks an image until the grid with the line-break column fits in one pass,
and says so in the log with the original and resulting sizes. 16640 (128 × 130) is what the
largest square image of the band asks for; an elongated one of the same area asks for more
(8192×512: 512 × 34 = 17408), so the warning at tower startup is a minimum and the processor's is
the one that counts.

`hy4_rope`: Hy4's rotary with images. GPT-J pairs `(2i, 2i+1)` over the first `rotary_dim`
channels of each Q and K head; `mode=gptj` with positions `[M]`, or `mode=imrope` with `[3, M]` and
`sections`, with the same component rule as the indexer. radiance has `gptj` and `imrope`
separately (its `imrope` pairs like NeoX); HF repeats each frequency within its pair. With 1-D
positions it gives the same bytes as libref's `rope` (`mode=gptj`) in BF16
(`tests/test_rope_mc.py`): serving with the tower does not change the text.

## Card rows

One HIP row per op (`kernels/hy4k/dev_*.hip`), built with `tools/build-hip.sh` in the radiance 2.2.0
build image (gfx1201 and gfx11-generic). They compute the same as the oracle, in FP32 with one
rounding at the output; only the order of the reductions changes. What they do differently:

- **Indices that come in the data.** The host checks them before writing and returns an error. The
  card cannot read the data at launch: the kernel skips the bad entry and does not write out of
  bounds. This is how it treats a `cu` that goes backwards or past the rows, `sidx` slots outside
  `state` (the sequence is skipped), negative or out-of-cache pages, a negative `bpos` and
  `vocab_sizes` < 1.
- **Caller preconditions** the card does not check: unique pages in `hy4_idx_pool` (`qsa_work`
  provides them).
- **Shapes.** f32 weights with the last dimension contiguous; `bt` with contiguous columns.
- **inv_freq of the indexer's rope.** The host computes it the same way as the oracle (`std::pow` in
  float) and passes it to the kernel by value: the card's `powf` is 1-2 ulp off, and the phase
  `p·inv_freq` multiplies that by the position.
- **`constraints`.** Each row declares its band (`schemas.cpp`), chosen so that every geometry
  inside it is served (for example `hc ≤ 4` and `n ≤ 4096` instead of a limit on the product, which
  constraints cannot express). What is not geometry but operand form (sinks present in the
  attention, no K/V descale, strides) is checked at launch, as in any row.

`attn_paged_sink` (standard op, libref's schema): row `hy4_dev_attn_sink` for the Hy4 geometry that
libr4d does not serve, limited to `head_dim` 256, `gqa` 1, `causal` 1 and a BF16 cache, with any
`block_size` and window. Semantics of `libavx/avx_attn.cpp` (`attn_paged_rows`): varlen with
`cu_seqlens`, or the selection form (no `cu`, `seqused` per row and `bt` = the row's page list). A
row with no visible key gives zero, with no sink. One 256-thread block per (row, head), no WMMA:
correct, not fast.

`hy4_mla_attn`: card row `hy4_dev_mla_attn`, in three forms.

- **Form 3** (`mla_attn3_k`), batches of more than 32 rows (prefill): libr4d's `attend_matrix`
  (`libr4d/r4d_mla.hip`) with the rope columns, the sink, the window and the selection form. One
  4-wave block per (row, 16 heads); the heads are the matrix rows, so a key is loaded once for 16
  heads. S = Q K^T and O += P V on gfx12 BF16 WMMA, P split into high and low BF16 halves. On gfx11
  it runs the same arithmetic without matrices. 3.3 ms for a 2048-row window chunk at 32 heads on one
  R9700 (6.5 ms before 2026-11-21, same bits), against 26 ms for form 2 and 93 ms for form 1
  (`tests/bench_attn.py`). `HY4_MLA_ATTN=4` splits the score in four column quarters added in a fixed
  order: 2.7 ms, but other bits, and on the full model its per-position KL tails left the rounding
  envelope (p99 |dKL| 2.53 against 2.4, 861 top-1 changes against 850, mean 0.1808 -> 0.1793), so it
  is opt-in.
- **Form 2** (`mla_attn2_k`), up to 32 rows (decode): libr4d's `attend_paged` with the same additions.
  One 8-wave block per (row, head), each wave a slice of the visible keys (a lane holds 16 latent and
  2 rope columns, so a score is one wave sum), the slices merged in LDS in a fixed order. The split
  depends only on the row, so a row gives the same bits alone or in a batch of up to 32
  (`tests/test_dev.py`). Forms 2 and 3 give different bits for the same row, as libr4d's do.
- **Form 1** (`mla_attn_k`): everything else (f32, f16, other widths, unaligned views). One
  256-thread block per (row, head) over 32-key strips, as `attn_paged_sink`.

`HY4_MLA_ATTN=1` forces form 1 and `=2` form 2, to compare them. In forms 2 and 3 a -inf score weighs
0 and a NaN reaches the output, as in the oracle.

With block keys (`bkeys`, `bcu`) the launcher runs form 2 whatever the row count (`mla_attn2_k<true>`):
the drafter's rows are few and need no batch invariance across forms. The slices split the union, the
row's cached keys first and then its block keys, so a row gives the same bits as form 2 over the same
keys laid out as a plain selection, and the same bits alone as in a batch (`tests/test_dev.py`). The
instance without block keys (`mla_attn2_k<false>`) is the form 2 of before: an old and a new build give
the same output bytes in forms 1, 2 and 3. Block keys need form 2's geometry (bf16, `bkeys` with
contiguous columns, rows a multiple of 8 values, 16-byte aligned); the card row refuses them with
`RAD_E_INVAL` anywhere else, and under `HY4_MLA_ATTN=1`. The host row serves them in every dtype.

`hy4_ring_store` and `hy4_ring_load`: card rows `hy4_dev_ring_store` (a 256-thread block per listed
row, the threads striding over the words) and `hy4_dev_ring_load` (a block per destination row), the
same bits as the host (`tests/test_dev.py`, at Hy4's entry width 5638, with duplicate entries: 13
rows of one sequence in ring 6, every mode, last listed wins). Before writing, a store block searches
the later listed rows for one on its entry (strided over the threads, `__syncthreads_or`). As in the
other rows, the card skips what the host refuses on a store: a row id outside `pos`, a row outside a
valid `cu` span, a slot outside `state` (the sequence is skipped and the others are written). A load
block always writes its row: the entry or the fill, also for a slot outside `state` or a `cu` that is
no partition, so `dst` never keeps stale bytes.

`hy4_hc_pre`: with `hc` 4 (8 products, 4 with `head`), `hc_pre1_k` reads the row once and accumulates
the norm and the products together, with `hc_pre_k`'s 256-thread split and summation order, so the
bits are the same in one pass instead of eight. Since 2026-11-21 that sum is split over the card
(`hc_pre_part_k` + `hc_pre_fin_k`) at every row count, and the bits are still `hc_pre1_k`'s: each of
its sums is 256 per-thread chains, a xor butterfly inside each wave of 32 and an in-order add of the
8 wave results, so one wave of `hc_pre_part_k` runs the 32 chains of one of those waves for one sum
(8·(MIX + 1) = 72 waves a row instead of 8 on one CU), and `hc_pre_fin_k` adds the 8 results from
0.0f in the same order and closes the row as `hc_pre1_k` does. The wave results go to the row's
scratch (`scratch` hook, 288 bytes a row); without scratch the launch is `hc_pre1_k`.
`HY4_HC_PRE=old` serves `hc_pre1_k` for an A/B (`=split` the split); the parameter `hc_form` (1 old,
2 split), which the graph does not pass, does the same per call for `tests/test_dev.py`, which
checks the split against `hc_pre1_k` bit for bit (x and post, with and without `head`) at 1, 2, 7, 8,
33 and 2048 rows, f32 streams and a 16000 row. On the R9700 (`tests/bench_hc.py`, a graph of 100
calls) a call went from 24.0 to 7.5 us at 1 row, 23.6 to 7.9 at 8, 24.1 to 14.0 at 48 and 595 to
358 at 2048. `hy4_idx_score` reads each key group once for all
heads (it used to read it once per head); under 64 rows it spreads a row's groups over up to 16
blocks. At Hy4's geometry (bf16, 16 heads, n 128) on gfx12 it runs on WMMA: the products are the
same bf16 values but the sums run in another order, so a score can differ in its last bits from the
scalar loop's, and a group on the top-k threshold can swap with a tie. That is accepted: the model's
KL stays inside the rounding envelope (half the positions identical; plan step 13), and the oracle
comparison holds at L2 ~1e-7. Elsewhere the scalar loop keeps the old kernel's bits.

A kernel's arguments, hidden ones included, may not exceed 768 bytes: the radiance backend refuses
the launch and the op fails on the card. `tests/test_dev.py` cannot see it (it launches through
HIP); `tools/kernarg.sh` checks it in the `.so`, along with any register spill to scratch. Reading
`gridDim` adds hidden arguments.

## `had_quant_act_fp8`

A standard radiance op (the schema is `libhalo/halo_rows.cpp`'s, identical field by field) that in
2.1.0 and 2.2.0 only libhalo declares, a library that does not load on the R9700 nor on the host. It is the
activation that libr4d's `w4nl64a8h` experts read, so `hy4k` brings it (a host oracle and a card
row, identical bit for bit). Semantics of `libhalo/halo_quant_act_fp8.hip`:

- per row and group of 128 columns (`group` = 128, `n` a multiple of 128): an unnormalised FWHT of
  128 in natural order; the factor 128 goes into the weight's scale (`docs/MOE-W4.md`);
- `scale = amax/448` (1 if the group is all zeros), f32 `[M, n/128]`;
- `q = e4m3(clamp(v/scale, ±448))`, round to nearest even, in row order.

If hy4k shares a `RADIANCE_HOME` with libhalo, the schema matches and the engine accepts it.
`tests/test_ops_extra.py` compares the oracle with torch (`float8_e4m3fn`).
