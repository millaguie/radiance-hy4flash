# DFly2 drafter (plan step 14, phase B)

Hy4-flash ships a block drafter, `model.dfly_layers.*` (0.85B, bf16), that proposes 5 tokens in one
pass. This file is the contract between the plugin wiring (`arch/hyv4flash`) and the new `hy4k` ops.
Reference: SGLang `hunyuan_v4_flash_dfly.py`, `dfly_utils.py`, `dfly_worker_v2.py`.

## What the reference computes

One proposal per sequence and step, for the block `[b, MASK x 5]` at positions `p0 .. p0 + 5`. `b` is
the bonus token the last verify (or the prefill) chose, `MASK` is token 120009, and `p0 - 1` is the
position of the last committed row `a` (the row whose logits chose `b`).

1. `e = embed(block)` with the trunk's `embed_tokens`, `[6, 3072]`.
2. Slot 0 only: `e0 = e0 + gate * eh_proj([enorm(e0); hnorm(h)])`, where `h` is the trunk's output for
   row `a` after `hc_head` and the final norm (the input of `lm_head`). `eh_proj` is `[3072, 6144]`,
   `gate` a `[3072]` vector.
3. Five layers, each a trunk sparse-layer attention (32 heads, MLA latent 512 + rope 64, q_lora 1024,
   sink, `linear_gate`, `post_proj_norm`) followed by a dense SwiGLU FFN of width 9216, pre-norm with
   residual (`input_layernorm`, `post_attention_layernorm`). Drafter layer `d` reads the KV cache of
   trunk layer `t_d` = 25, 29, 33, 37, 41 (all sparse layers, `kv_full`).
4. The keys of block row `j` in drafter layer `d`:
   - the history: the tokens trunk layer `t_d` selected for row `a` (its top-512 groups of 4), or every
     cached position `< p0` when that step ran no selection (context within `candidate_topk`);
   - the block's own keys, rows `0 .. j` (causal), computed by this drafter layer from its own input.
   The cache holds rejected drafts at positions `>= p0`: no cached key at or past `p0` is read.
5. `y = final_norm(x)` over the 6 rows.
6. Sequentially, `prev = b`; for `k = 1 .. 5`:
   `c = y_k + down(silu(gate(z)) * up(z))`, `z = [hidden_norm(y_k); embed_norm(embed(prev))]`
   (`gate`, `up`: `[3072, 6144]`, `down`: `[3072, 3072]`); `prev = argmax(lm_head(c))`, the k-th draft.
   `confidence_head` is not used.

## Radiance mapping

- Declared as a `RAD_DRAFT_BLOCK` drafter: `block = 6`, `depth = 5`, `mask_token = 120009`, `proposal`
  an i32 buffer `[max_seqs, 5]`. The query pass (`RadBatch::draft_pass > 0`) carries the block rows
  and runs steps 1-6. The context pass (`draft_pass < 0`) does nothing: the drafter keeps no cache.
- **The ring.** The trunk does not know, when it runs a verify, which row will be the last accepted.
  So every row with logits (`out_ids`: all rows of a verify, the last one of a prefill chunk) stores
  `h` and the 5 selections in a per-sequence ring of `depth + 1` entries indexed by `position %
  (depth + 1)`. The query pass reads the entry of `p0 - 1`. The ring is a `RAD_KV_LINEAR` group
  (`kv_dfly`), so its slot follows the sequence (`state_index`).
- Entry layout, i32 words: `[0]` the position stored (or -1), `[1 .. 3072]` `h` as f32 bits, then per
  drafter layer `d` the history in `hy4_mla_attn`'s selection form: `seqused` and 512 pages. With a
  selection, those are row `a`'s `idx_sel` row and `seqused` (`hy4_idx_select`: ascending groups, own
  group last, so the keys are exactly the positions `<= p0 - 1` it chose). Without one (context within
  `candidate_topk`, at most 512 pages), `seqused = p0` and the sequence's first pages of `kv_full`'s
  block table. Entry width `W_e = 1 + 3072 + 5 * 513 = 5638`.

## New ops (`hy4k`, host oracle `hy4_ref_*` and card `hy4_dev_*`, as the existing ones)

### `hy4_ring_store`

| Params | Operands |
|---|---|
| `M`(range) `ring` `width` `offset` `seqs` `mode`:str | `src`?, `row_ids`, `pos`, `cu`, `sidx`, `state`:inout |

For each `k < M`: `r = row_ids[k]` is a batch row, `s` its sequence (`cu`), `slot = sidx[s]` (skip if
`< 0`), `e = pos[r] mod ring`. It writes `width` i32 words at `state[slot][e][offset ..]` (of two
listed rows on the same entry the later listed wins, host and card alike), by `mode`:
- `pos`: `pos[r]` (`width` 1, no `src`);
- `row`: `src[r]`, a per-token buffer (`idx_sel`, the selection's `seqused`);
- `out`: `src[k]`, a per-logit-row buffer (the head norm's output); an f32 source as its bits;
- `dense`: `pos[r] + 1`, then the first `width - 1` words of `src[s]` (a block table `[seqs, cols]`),
  those past page `pos[r] / block` set to -1 (`block` = 4, the ratio; a param `block`).
The state is `[slots, ring, W_e]` with any size-1 middle axes, as the other `RAD_KV_LINEAR` groups.
Copies only: host and card give the same bits. Every operand pointing into another is checked
before the first write (common rule).

### `hy4_ring_load`

| Params | Operands |
|---|---|
| `M`(range) `ring` `width` `offset` `seqs` `fill`? | `pos`, `cu`?, `sidx`, `state` → `dst` |

`p0[s] = pos[cu[s]]`, the position of sequence `s`'s first block row; `e = (p0[s] - 1) mod ring`. With
`cu`, every row `r` of sequence `s` gets `state[slot][e][offset .. offset + width - 1]` in `dst[r]`;
without `cu`, `pos` is per sequence and `dst[s]` gets it. When `slot < 0`, `p0 < 1`, or the entry's
position word is not `p0 - 1`, the words are `fill` instead (-1 by default; rows outside `cu`'s
spans get it too): the drafter then computes from garbage and the verify rejects what it proposes
(acceptance lost, never a token). The `h` load passes `fill = 0`, since -1 is a NaN as f32.

### `hy4_mla_attn` with block keys

Two optional operands at the very end, after the output (order: `q`, `q_rope`, `kv_cache`,
`block_table`, `seqused`, `cu_seqlens`?, `sinks`w, `out`, `bkeys`?, `bcu`?): radiance lets a caller
omit only trailing operands, so every existing call stays valid. Used by the drafter in the selection form (no `cu_seqlens`; per row
`block_table` = its page list and `seqused`):
- `bkeys` `[rows, 576]` bf16: the block rows' own `[normed latent | roped key]`, in batch row order;
- `bcu` `[seqs + 1]` i32: the block rows of each sequence.
Row `i` of sequence `s` attends to its cached selection (tokens `< seqused[i]` of its page list) and
to `bkeys[bcu[s] .. i]` (causal), with the sink, one softmax over the union. Form 2 only: the drafter's
rows are few and need no batch invariance.

### Argmax

The draft head reuses the trunk's `lm_head` and radiance's ops, at both tp: `gemm_nt` over this
rank's vocabulary into bf16 logits, `row_topk` (R = 2, global ids, `pairs`), at tp 2 `all_gather`,
then `pair_argmax` (ties to the lower id). No new op.

*Deviation (plugin):* not `sample_argmax` at tp 1. The device `row_topk` reads bf16 logits, so the
sharded head needs them anyway, and one path gives the same proposals at tp 1 and 2; `sample_argmax`
also takes the sampler's `params` operand and writes a contiguous token. `pair_argmax` writes a
pitched column, so each step writes its proposal column directly and a second launch writes the
contiguous `prev` the next step embeds. Cost: the argmax runs over bf16-rounded logits where the
reference's are f32, so a near-tie can pick the other token; that costs acceptance, never a token.

## Plugin wiring (as built, `arch/hyv4flash/hyv4flash.cpp`)

1. When: `dfly_plan` serves the drafter only if the container (or, converting, the checkpoint)
   carries `dfly.final_norm.weight`, `RadBuildCtx::max_spec == 5` (radiance requires `depth ==
   max_spec`; a build at another depth serves prompt lookup alone and says so), the MLA is absorbed,
   `lm_head` is bf16 and splits evenly by vocabulary, and `HY4_DFLY` is not `0`. `HY4_DFLY=0` serves
   such a container by prompt lookup (radiance refuses `--spec-ngram` once a drafter is declared).
   The drafter config must also describe the graph wired here, or the drafter is not served (a
   note, the trunk serves): `dfly_mtp_fusion` `slot0`, `dfly_enable_hidden_correction` on, no
   `dfly_context_local_window`, `dfly_target_index_topk` = `idx_topk_groups * idx_ratio` (2048),
   `dfly_target_kv_layers` and `dfly_target_index_layer` = 25 29 33 37 41,
   `dfly_num_hidden_layers` 5, `dfly_mask_token_id` 120009. A key the config does not state takes
   that reference default (a null is not in the container: radiance's flattening drops it).
   Otherwise nothing is declared and the trunk is the one it was.
2. Weights: `dfly.D.*` from `model.dfly_layers.D.*` through the trunk's attention declaration
   (`decl_mla` with a drafter site: absorbed selection form only, no indexer, no `kv_store`) and a
   dense FFN; `dfly.fusion.*`, `dfly.correction.*`, `dfly.final_norm.weight`. *Deviation:* not
   `blk.dfly.D.*`. A recipe pattern is a glob in which `*` crosses dots, so the IQ8R recipe's
   `blk.*.attn_q_a.weight` (and the other trunk rules) would quantise the drafter's linears; under
   `dfly.` only a rule that names `dfly.` reaches them. Conversion in place: `deploy/conv-dfly.sh`
   (`--max-spec 5`).
   - *Format.* The drafter's linears are bf16 (`gemm_nt`) as the checkpoint holds them, or IQ8R,
     whatever the trunk's format. `dfly_plan` reads it from the stored format of
     `dfly.0.ffn_down.weight` and `dfly.correction.down.weight` (`Dfly::i8`), at convert time from
     the recipe and at serve time from the container, as `LinearFP8::probe` learns a trunk's; no
     container key, since no producer in the drafter writes codes. IQ8R gives the drafter the
     trunk's int8 geometry (`Geom::w_i8h`, group 256) and an int8 pair on each linear's input
     (`dfly_a`, `dfly_qa`, `dfly_o`, `dfly_ffn`, `dfly_cat`, `dfly_cff`), written by the linear's
     own `had_quant_act_i8g`; a bf16 drafter declares exactly what it did before. Each linear still
     serves the form it finds stored.
   - *IQ8R in `hy4flash-w4nl64-gptq-iq8r-dfly.recipe`:* `attn_q_a`, `attn_q_b`, `attn_gate`,
     `attn_output`, `ffn_gate_up`, `ffn_down`, `fusion.eh`, `correction.gate_up`,
     `correction.down` (every N a multiple of 128, every K of 256). Kept bf16, as the trunk keeps
     them: `attn_kv_a` (N 576: no int8 row above 64 rows, and a query pass carries 6 rows a
     sequence) and `attn_kv_b_abs` (`mla_project` is bf16); the norms, the fusion's `gate` vector
     and the sinks. Rewriting a bf16 drafter in place: `DFLY_I8=1 deploy/conv-dfly.sh`
     (`--replace 'dfly.*'`). The bf16 linears were 11% of a decode step with the drafter on.
   - The hidden correction's two linears run in each of the five steps: step 1 declares the weight,
     steps 2-5 launch it with ops of their own (`dfly_relaunch`, `DflyStep::gu`/`down`), for the
     liveness reason in step 4.
3. Trunk step: `hy4_ring_store` over the `out_ids` rows. Right after layer `t_d`'s attention (while
   `idx_sel` and `idx_sequ` still hold its selection): with a selection, mode `row` for `seqused`
   (width 1) and for the pages (width 512); without one, mode `dense` (width 513, `block` 4) from
   `kv_full`'s block table, so `seqused = p0` as the entry layout says (not a count of -1). After the
   head norm, mode `out` for `h` (width 3072), then mode `pos` for the position word, last.
4. Query pass (`step_dfly`): `hy4_ring_load` *with* `cu` (every block row gets the entry; the
   per-sequence form would need each sequence's `p0` in a contiguous array, which the batch does
   not have), twice: `h` into an f32 plane (narrowed to bf16, exact; `fill = 0`, so a row without
   an entry reads 0.0, not the NaN of -1) and the five histories (`fill` -1) into one
   i32 plane `[rows, 5 * 513]` whose column views are each layer's `seqused` and page list. Slot 0
   of sequence `s` is row `cu_seqlens[s]` (every sequence carries a whole block; only `T = 6 S`
   is checked, `cu_seqlens` being device data): `gather_rows` by `cu_seqlens` takes slot 0's embedding and `h`, and `scatter_add_rows` adds the
   fused term back; step 6's row `k` is `gather_rows` by `cu_seqlens` over the view starting at row
   `k`. Each of the five steps has its own ops, so buffer liveness follows the issue order.
   `hy4_mla_attn` gets `bkeys` = the layer's `kva` rows and `bcu` = `cu_seqlens`, one launch over
   the pass's rows, against `kv_cache(kv_full, t_d)`, read only.
5. The context pass (`draft_pass < 0`) returns at once. The ring's position words start at 0 (the
   engine zeroes a slot at admission), which matters only for `p0 = 1`, an entry the prefill wrote.

## Serving

Measured on 2 R9700 at 330 W, rewrite task, 400 tokens a request, greedy (one client and
eight clients against the HTTP API):

| | no speculation | DFly2, fixed block | DFly2, `draft-window: adaptive` |
|---|---|---|---|
| 1 client, 1.4K context | 41.8 tok/s | 77.3 tok/s | 72.3 tok/s |
| 1 client, 6.6K context | 40.7 tok/s | 78.7 tok/s | 73.8 tok/s |
| 8 clients, 8 x 400 tokens | 28.3 s | 29.9 s | 28.3 s |

Acceptance: 93% of the drafts on a verbatim copy, 54% on code rewriting. With many sequences a
verify step carries 6 rows each, crosses the 32-row forms and reads many more experts from host
memory, so the gain vanishes; the adaptive window gives that case back and keeps most of the
single-client gain. Serve with `num-speculative-tokens: 5` and `draft-window: adaptive`. Greedy text
is the same as without the drafter (checked on a 1.4K and a 6.6K-token prompt).
