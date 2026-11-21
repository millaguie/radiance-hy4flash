# Hy4-flash on Strix Halo (gfx1151)

Status: not tested on the machine. Coverage was reviewed in the radiance 2.1.0 code
(`libhalo/halo_rows.cpp`, `arch/common/rad_block_moe_fp8.h`) on 2026-11-19 and corrected on
2026-11-20: no op is missing.

On Strix Halo radiance uses `libhalo` instead of `libr4d` (`libr4d/CMakeLists.txt`: gfx12 only). The
`hy4k` rows already build for `gfx11-generic` (`tools/build-hip.sh`), which covers gfx1151, and are
tested on a Radeon 890M iGPU (gfx1150, also RDNA 3.5).

## Coverage

| Model ops | Who serves them on gfx1151 |
|---|---|
| `hy4_*` (iHC, Engram, indexer, `hy4_mla_kv`, `hy4_mla_attn`), `attn_paged_sink` at `head_dim` 256 | `hy4k` |
| `had_quant_act_fp8` | `libhalo` (and `hy4k`) |
| embedding, `embed_lookup_q`, `hc_enter`, `rmsnorm`, `rope`, `kv_store`, `gemm_nt` BF16, `gemm_nt_bias`, `gemm_nt_f32` (N ≤ 512), `sigmoid`, `mul`, `add`, `cast`, `silu_mul`, `act_mul`, `qsa_work`, `qsa_tail_store`, `gather_rows`, `logits_gemm`, sampling | `libhalo` |
| Absorbed MLA: `mla_project` (`head_dim` 256, `latent_dim` 512) | `libhalo` |
| Sigmoid router (≤ 512 experts, top-k ≤ 32), `moe_scatter_inv` (≤ 512), `moe_gather`/`moe_gather_inv`, `swiglu_clamped`, `swiglu_clamped_quant_fp8`, `quant_act_fp8`, `gram_accum` | `libhalo` |
| 4-bit expert GEMM (`moe_gemm_q` over `w4nl64a8h`) | `libhalo`, row `moe_gemm_w4a8` |

**Correction (2026-11-20).** The first version of this page said that nothing served the expert GEMM
in `w4nl64a8h`. That is wrong: `libhalo` 2.1.0 has the `moe_gemm_w4a8` row of the `moe_gemm_q` op
(`libhalo/halo_rows.cpp:8073`, constraints `cMoeGemmW4` with `dtype` in `w4a8 w4a8h w4nla8h w4nl64a8h
w4nl32a8h w5nl64a8h`), with its layout hook (`halo_rad_layout_moe_w4`). The earlier review only
looked at the `fp8a8_f32scale` and `mxfp4a8` rows. With BF16 experts there is also `moe_gemm_bf16`,
but that is 297 GB. `moe_scatter` (without `_inv`) stops at 256 experts; the block uses
`moe_scatter_inv` when it is declared, and that one reaches 512.

## What is missing

1. **Test it.** On paper no op is missing: `hy4k` for what is specific to Hy4 (built for
   `gfx11-generic`) and `libhalo` for the rest, experts included. Without the machine we do not know
   whether something rejects at launch (a `constraints` band, a shape) or how fast it runs. The 890M
   iGPU (gfx1150) does not work for this: `libhalo` builds for gfx1151.
2. **Optional: an int8 trunk.** The BF16 trunk is ~14 GB read on every token: at Strix Halo's
   ~256 GB/s, the ceiling is about 15-18 tok/s. `libhalo` does have `gemm_nt_q` in `w8a16`; we still
   need to check which int8 trunk recipe matches that row.

## Memory (128 GB unified, tp 1)

| | GB |
|---|---|
| `w4nl64` experts | ~84 |
| BF16 trunk | ~14 |
| Absorbed MLA cache at 196K | ~2.6 |
| Engram | on disk (`ngram-placement: disk`), 118 GB on the NVMe |

About 100 GB resident: it fits, with little headroom for the system. No host pool (the memory is
already the same). Configuration: `kernels: hy4k,libhalo,libavx`, `tp: 1`, absorbed MLA, Engram on
disk.
