# Radiance optimisations not yet used by Hy4-flash (survey, 2026-11-21)

Hy4 already uses recording and replay, chained launches, the prefill stager, the targeted host join for Engram, n-gram read-ahead, the exact embedding all-reduce, shared int8 quantisers, an IQ8R trunk and the DFly2 drafter. It still leaves out most of radiance's launch-count fusions, weight prefetch, the lossy wire and wider prefill steps, and the prefill numbers point to the expert link as the bottleneck.

**Why 4096-token chunks tested the same as 2048.** The plugin declares `kv_idx_tail` and `kv_engram_tok` as `RAD_KV_LINEAR`, so the core treats the model as hybrid (`core/sched/geometry.cpp:49-58`). Prefill chunks are then capped at `checkpoint-interval`, which defaults to 2048 (`geometry.cpp:147`, `205-219`). The startup line `chunk geometry: ... (the checkpoint interval, not the budget)` will confirm it. This matters because each chunk sweeps every pooled expert (GUIDE §4.5). With ~19.5 GB of pool per card, a 2048-token chunk's sweep at PCIe Gen5 x8 takes about 0.7 s; it overlaps compute (a chunk takes about max(compute, sweep)), so it bounds short prompts near the 2,800 tok/s seen at 5.7K, while from 18K on compute is already longer (see corrections).

Plugin lines are from the file as it is on disk now; it is being edited, so they will drift.

| # | Mechanism (radiance ref) | Where in `hyv4flash.cpp` | Expected gain | Effort | Risk |
|---|---|---|---|---|---|
| 1 | Wider prefill steps: `prefill-span: 2` with `max-num-batched-tokens: 4096`, or `checkpoint-interval: 4096` (GUIDE §4.5 "prefill-span"; Flash-Next tiered: +9% at 64K, +49% at 250K) | Config. Raise MoE pass rows at 1808 and 1938 (`min(T,2048)`); generalise `plane_rows` at 1842 to `max_mixed_tok` | 0-10% at 18-73K (the sweep overlaps compute; see corrections); measure first | S | Arena takes VRAM from experts; ties may differ |
| 2 | Launch fusions that libr4d already has. `NormQuantFP8` → `rmsnorm_had_quant_iq8r` (`arch/common/rad_fp8.h:396-533`, `r4d_rows.cpp:9287`) for in_norm+q_a, ffn_norm+lat_in, qa_norm+q_b. `gate_had_quant_iq8r` (`r4d_rows.cpp:9368`, as `rad_block_attn_gated_fp8.h:449`) replaces sigmoid+mul+o quant. `gated_had_quant_iq8r` (9330) for shared/dense SiLU + down quant | 2037/2054 with 629-638 and 998-1004; 641 with 662; 766-775 and step 2242-2245; drop `gact.w_bf16=true` at 980 | About −250 of ~1750 launches; +5-8% decode | S-M, plugin only | Low; byte-identical per OPS.md except possibly the sigmoid gate |
| 3 | `mc.router_fuse = 1` → `router_sigmoid_topk_scatter` (`rad_block_moe_fp8.h:1665-1700`; row limits ≤512 experts, top_k ≤32, M·k ≤1024 fit; GLM sets it at `glm5next.cpp:424`) | Next to 1805 | −41 launches, ~+1.5%, decode and verify only (M·k ≤ 1024; wider passes keep the separate ops) | 1 line | None (byte-identical) |
| 4 | Weight `prefetch` beside idle-memory stretches: `decl_prefetch` / `prefetch_beside` (`rad_arch.h:649-682`, OPS.md `prefetch`, qwen4exp `kAttnOutPrefetchBytes`) | Declare after o/gate at 774; issue before attention (~2229) and beside the two all_reduces | +1-3% decode (VRAM-resident weights read into the 64 MB on-die cache beside a collective) | S | None (values unchanged) |
| 5 | Split-K `hy4_hc_pre` (one workgroup per row today; GLM's `mhc_pre_partial`/`finish` pattern, OPS.md "mHC") | `kernels/hy4k/dev_hc.hip:12-174` | 7% → ~2%, +4-5% decode | M | Can stay bit-identical by keeping the 256 virtual-lane partition |
| 6 | Fold the post-norm into `hy4_hc_post` (no upstream op; `ar_hc_write` cannot be used, see below) | `op_hc_post` 561; drop the norms at 781 and 1034 | −84 launches, +2-4% | S-M | Low if the bf16 rounding of the norm is matched |
| 7 | `tp-wire: wht6` (GUIDE §4.7/§4.11; production Flash-Next runs it); the plugin already passes `wire_exact`/`min_bytes` (520-528) | Config | +2-3% prefill; +1-2% decode at ≥4 clients or with DFly2 rows | Config | Lossy, so check against the KL envelope |
| 8 | Run the shared expert beside the routed arm: interleaved like `rad_block_moe_fp8.h:1804-1811`, or on lane 1 with a `rad_lane1_window` (PLUGIN.md §11) | Step 2529-2539 | +2-4% decode (routed GEMMs wait on PCIe while VRAM bandwidth sits idle) | M | Lane-1 buffer rules |
| 9 | Split the shared expert and the dense layer 0 across ranks (qwen4exp `shared_place`/`block_slice`, 285-300 and 452-469); then add before one all_reduce | 985-990, 1020-1031, step 2534-2539 | ~71 MB a card freed, half the shared expert's compute; +1-2% decode | S-M | Summation order changes (envelope) |
| 10 | DFly2 short block: `rad_drafter_short_block` plus `draft-short-from: 8` (`abi/rad_builder.h:850-862`, GUIDE §4.6: 27B 653→697 at 8 streams, 752→946 at 16) | After `rad_declare_drafter` at 1596, plus the ops for the shorter pass | +5-15% at ≥8 clients | M | Greedy text may change at ties |
| 11 | Hy4-specific fusions: kv_norm+rope_k+kv_store, rope_q folded into absorb/attention, clamped SwiGLU + rotated E4M3 quant (the vendor `GateQuantFP8` clipped+rot branch is two ops today) | 669-673, 747, step 2113-2119; `vendor/arch/rad_fp8.h` ~624 | −~170 launches, +2-4% | M (hy4k kernels) | Low |
| 12 | IQR experts (`expert_iqr`, `moe_gemm_iqr`, docs/IQR.md; Flash-Next's container is `iq4r-moe`) | Recipe and probe at 1820 | Fewer PCIe bytes: prefill +20-30%, decode +5-10% | L (re-GPTQ, clamp on the i8a8h path) | Quality |
| 13 | Engram value/key projections in IQ8R, with the bias folded into `hy4_engram_gate` | 898-916, 948-956 | ~1% | S | Low |

Rows 2, 3, 6 and 11 together remove about 540 launches; that is about +8-14% decode, not 30% (see corrections), and still the biggest decode lever. Row 2 overlaps the quantiser sharing you have in progress, so check it against that work first.

**Considered and rejected**
- **Decode expert prefetch at tp ≥ 2.** It is off by design: one stream went from 10.1 ms to 20.5 ms a step with it (GUIDE §4.3).
- **`expert-profile` / `expert-pin-top`.** Already measured, no change.
- **`ar_hc_write` / `ar_gather_hc_write` (Flash-Next's all-reduce folded into the residual write).** Hy4 has an RMS norm between the reduce and `hc_post`, and a norm is not linear.
- **`ar_rmsnorm_*_iq8r`.** These emit int8 codes for a following GEMM, but Hy4's post-norms feed `hc_post`. They would only fit the FFN tail (residual = `sout`) with unused codes, as an alternative to row 6.
- **`kv-cache-dtype: i8`.** Attention reads are capped at 2048 selected tokens and the 512 window (~47 MB/token), so it buys capacity, not speed. It would also need i8 forms of `hy4_mla_attn` and `kv_store`, and the plugin refuses non-bf16 at 1680.
- **`ngram-placement: ram` / `vram`.** The 118 GiB table, or 61 GiB stored E4M3, does not fit 64 GB of RAM next to a 39 GB pool. The community 4-bit table needs a new gather op and changes the model.
- **Async host op for Engram.** There is none in 2.1.0. A host op runs on the issuing thread after a join on the last device op that touched the host arena (`core/runtime/issue.cpp:1023-1051`, replay at `ctx.cpp:1302-1320`). The device keeps running what is queued, so only the layer-1 lookup is exposed, by max(0, gather time − layer-0 time). Merging the four lookups would not help, because the later ones already overlap device work.
- **Tree verify, `draft-coupling`.**
  - Trees: the indexer ring and the Engram window assume a chain.
  - `draft-coupling`: only MTP and DFlash2 heads (DFlash2 is Gemma's drafter, not DFly2).
  - (The adaptive window was listed here by mistake: it applies to block drafters; see corrections.)
- **Two-piece pipelined prefill.** It is `qwen35`-only and gave +3-7% there; all_reduce is about 5% of Hy4 prefill, for high effort.
- **`tp-attn-split: rows`.** Hy4's heads divide over two ranks, and the absorbed cache is replicated anyway.
- **Hyper-connection `mix: e4m3`.** `hc_pre` is limited by latency, not bytes.
- **`mla_project` in int8.** libr4d serves bf16 only (`r4d_rows.cpp:6716`).

## Corrections after review

- **Adaptive draft window: not a rejection.** `core/sched/scheduler.cpp:256-268` enables it for
  `RAD_DRAFT_BLOCK`; `docs/DFLY2.md` § Serving measured it on DFly2 (1 client 72-74 tok/s, 8 clients
  28.3 s, the fixed block's 8-client loss gone). The survey followed a stale table row in GUIDE.
- **Row 1, wider prefill: expect 0-10% at 18-73K, mainly below 16K, and measure before any plugin
  change.** The pool sweep overlaps compute (a chunk takes about max(compute, sweep)): at 5.7K, 3 chunks
  x 0.7 s of sweep against 2.04 s measured; at 18K a chunk already takes 0.84 s and experts are 15% of
  the profile, so long prompts are compute-bound. A span engages only from 16K and never on the first
  interval (GUIDE §4.5); Flash-Next tiered got +2.3% at 18K, +9% at 64K.
- **Row 3, `router_fuse`:** the fused op covers `M * top_k <= 1024` (64 rows at top-16); wider passes
  fall back to the separate ops by construction (`rad_block_moe_fp8.h:1672`, `1831`), so prefill is
  not broken: decode and verify only.
- **Row 4, prefetch:** the weights are VRAM-resident; `prefetch` reads them into the 64 MB on-die cache
  beside a collective (OPS.md § `prefetch`), it does not move them from host. Gain nearer +1-3%, not
  visible under the syncing profiler.
- **Row 9:** the replicated dense layer 0 and shared expert are about 142 MB a card in int8, so a split
  frees about 71 MB a card (not 290 MB/token); the gain is half the shared expert's compute. Low.
- **Rows 2, 3, 6, 11 together:** about +8-14% decode, not 30%: a launch is ~11 µs on average and the
  fused work remains.
- **Missing:** `kv-cache-dtype: i8` deserves a number before rejecting it: VRAM freed here becomes
  expert residency (speed). The Engram row cache is fixed at 2^19 rows a table
  (`libavx/avx_ngram.cpp:286`, `RowCache::kRows`): no knob in 2.1.0; enlarging it is a libavx change.

## Done on 2026-11-22: LDS-only barriers and grouped-head decode attention

From `owndoc/knowledge/gpu-r9700-rdna4/optimizaciones.md` (its #4 and #1). Details in
`docs/OPS-HY4.md` § Card rows.

- **LDS-only barriers, on by default.** Every `__syncthreads()` in hy4k that only orders LDS traffic is
  now libr4d's `lds_barrier()` (all of them but `ring_store_k`'s `__syncthreads_or`), which drops the
  `global_inv` that gfx12 put on both sides of each barrier (6 for 12 barriers in `mla_attn3_k`, 9 for
  18 in `engram_gate_k`, 3 for 6 in `idx_score_k`; none left). **Bit for bit**: the KL is final-22's at
  all 10851 positions, scored in 32-row steps it equals the 2.2 plugins' 32-row run, and the greedy
  texts are the 2.2 plugins' bytes with and without DFly2. **Speed: no measurable change** (form 3
  stays at 3.30 ms on a 2048-row chunk; full model within the session's spread, table below).
- **Form 2g of `hy4_mla_attn`, opt-in (`HY4_MLA_ATTN=g`).** Heads grouped by 16 on BF16 WMMA, the
  history split in slices fixed by the head count (batch-invariant: a verify row gives the decode row's
  bits, and the greedy texts are the same bytes with and without DFly2). The attention call is 1.5-3.9x
  faster at decode geometry (sparse layer, 1 row 152 -> 42 µs, 8 rows 261 -> 84 µs; window layer
  8 rows 151 -> 48 µs), and decode after an 18K prompt is +11% (1 client) to +14% (4 clients). It
  stays opt-in because in the KL scored in 32-row steps (the only KL that reaches it) its p99 |dKL|
  was 2.43-2.62 in three legal samples against the envelope's 2.4, where the per-head form's own
  reorders gave 2.26-2.49 (`docs/OPS-HY4.md`). With DFly2 at 1 client and a short prompt it lost 8%
  (108.8 -> 99.7 tok/s, three runs each, acceptance unchanged at 0.665 against 0.667) and 1-5% at 4
  clients; not isolated. The suspect is the drafter's block-key attention, where a short history
  leaves form 2g's fixed cost (16 slices and the merge launch) uncovered: a build with the block keys
  back on the per-head form (`if (grouped) return form2g(true, ...)` off) was built but not measured.

Speed on the test machine, one session (330 W, NVMe composite <= 60 °C before each run, ~/hy4/cola/opt/opt.sh
and opt2.sh, results in ~/hy4/cola/resultados/opt/). base = the 2.2 plugins (home-22); lds = the new
hy4k with form 2g off, i.e. the default of this build; 2g = the new hy4k with form 2g on (the build
that was measured had it on by default, the same kernels as `HY4_MLA_ATTN=g` now). Decode 1/4/8 is
`bench_throughput` (short prompt, 256 tokens); "18K decode" is `largo.py` (256 tokens after an ~18K
prompt, two runs each); prefill is `prefill_fijo` p50.

| | decode 1 | 4 | 8 | prefill 5.7K | 18K | 18K decode 1 client | 4 clients |
|---|---|---|---|---|---|---|---|
| base | 57.5 / 57.4 | 196.5 / 191.7 | 341.5 / 338.5 | 3,066 / 2,912 | 2,765 / 2,739 | 41.3-43.3 / 41.2-43.2 | 107.8-108.2 / 107.2-108.9 |
| lds | 57.6 | 200.3 | 340.2 | 2,934 | 2,679 | 42.5-44.5 | 111.7-115.8 |
| 2g | 58.0 / 57.9 | 197.7 / 197.2 | 364.6 / 365.0 | 3,017 / 2,988 | 2,743 / 2,697 | 45.9-48.8 / 45.8-48.7 | 122.4-124.6 / 119.6-124.7 |
| base, DFly2 (3 runs) | 108.9 / 108.7 / 108.8 | 361.6 / 363.4 / 363.7 | 449.6 / 458.1 / 455.3 | 2,928 / 3,004 / 2,982 | 2,661 / 2,730 / 2,691 | 79.0-84.7 | 154-184 |
| lds, DFly2 | 110.0 | 362.8 | 424.5 | 2,956 | 2,680 | 80.3-86.0 | 150-195 |
| 2g, DFly2 (3 runs) | 99.8 / 99.7 / 97.6 | 344.0 / 345.3 / 328.6 | 562.1 / 566.0 / 522.5 | 2,995 / 2,953 / 2,614 | 2,714 / 2,745 / 2,476 | 80.8-82.6 | 161-208 |

DFly2 at 8 clients varies run to run on 2.2 (425-603 in `docs/BENCH-2026-11-21.md`), so 2g's
522-566 against 450-458 is suggestive, not settled. The third 2g DFly2 run was slower in prefill too
(2,614 / 2,476), so part of its decode drop is that run's. Next steps for form 2g: an envelope measured
in 32-row steps (several legal per-head reorders) or a BFCL run, to decide the default; and the
block-key variant above for DFly2 at 1 client.

## Done on 2026-11-23: concurrent and chained hy4k launches

From the Inkling 2 plugin (rounds 4 and 7) and the #586 survey. Details and the table are in
`docs/DESIGN.md`, "Concurrent and chained hy4k launches".

- `rad_kernel_concurrent` (every card row): within 1-2 %, bit for bit.
- `rad_kernel_chain` = 2, with chained forms only for launches of up to 64 rows:
  - Without DFly2: decode +6.5 % at one client, +4 % at 4 and 8, chat +5 %.
  - With DFly2: +3.5 % / +4.5 % / +1 %.
  - Prefill unchanged.
  - KL bit for bit, both in 2048-row chunks and in 32-row steps.
- Chaining prefill launches too lost 7-9 % of prefill, so they stay plain.
- `RADIANCE_AQL_CHAIN_FENCE=1` is no longer needed with DFly2: 8 clients gave 462 tok/s without it
  and 456 with it, with the same drafter acceptance.

- **Wide `hy4_mla_project`** (`mla_proj2_k`: Direct-B loads after gostar's #586, `global_load_tr_b128`
  for the absorb weight, 4x4-wave tiles, a step-ahead prefetch, head-major grid):
  - Prefill +9-11 % at 5.7K, +11 % at 18K, +7-9 % at 73K. Decode unchanged.
  - Bit for bit with `mla_proj_k`, and the 4096-row superlinear cost is gone. Details in `docs/DESIGN.md`,
    "The wide mla_project".

Next on the list: row 3 (`router_fuse`), which is still a one-liner, and the int8 head for the DFly2
drafter.
