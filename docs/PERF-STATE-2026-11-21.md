# Hy4-flash on 2x R9700: performance state, 2026-11-21 01:45

Context for a round of improvement proposals. Model: Tencent Hy4-flash (42 layers, hidden 3072, 4
residual streams with iHC hyper-connections; MLA attention, latent 512 + rope 64, served absorbed;
12 sparse layers with an indexer selecting top 2048 tokens in 4-token groups, 30 sliding-window layers
of 512; 512 routed experts top-16 + 1 shared, latent width 1536, GPTQ 4-bit `w4nl64`; an "Engram"
n-gram embedding table of 118 GiB read from NVMe on the host). Engine: radiance 2.1.0 with this
repo's plugins (`arch/hyv4flash`, `kernels/hy4k`), tensor parallel 2 on two AMD Radeon R9700 (gfx1201,
RDNA4, 32 GB each, PCIe), 64 GB host RAM, 39 GB pinned host pool for experts, 196K context, 8
sequences, power cap lifted to 330 W.

## Numbers

| | decode 1 / 4 / 8 clients (tok/s) | prefill 5.7K / 18K / 73K (tok/s) |
|---|---|---|
| start of the night | 10.6 / 44 / 52 | 159 / 125 / — |
| + `shape_probe_ok` (arena lent to experts, stager armed) | 24.0 / 90 / 179 | 349 / 277 / — |
| now | 33.9 / 123 / 236 | 2259 / 1831 / ~1100 |
| Qwen3.8-Flash-Next on the same cards (all weights in VRAM) | 201 / 574 / 986 | 6400 / 6700 / 6500 |

The first prefill pass of a fresh prompt is much slower than later passes (18K: 648 vs 1595 tok/s
before the last kernels): the Engram rows of a new text come from NVMe (row cache with io_uring direct
reads, ~50% misses).

## What was done on 2026-11-21 (each tested against a host oracle, KL envelope, bench)

- `hy4_hc_pre` in one pass (bit-identical): decode +25%.
- MLA attention: form 2 (libr4d `attend_paged` port, <= 32 rows) and form 3 (libr4d
  `attend_matrix` port with gfx12 BF16 WMMA, > 32 rows), with sink, window and selection.
- Indexer score: loop interchange, then gfx12 WMMA (16 heads x 16 groups x 128).
- `hy4_mla_project` (absorb/expand per head) as a WMMA GEMM for > 32 rows.

## Where the time goes now

`profile-ops` syncs every op, so shares are hints; device GEMM times are device events.

- Decode (short context, per token ~28 ms unprofiled): `gemm_nt` (the dense trunk in BF16:
  attn_output 26%, attn_gate 22%, ffn_gate_up_shexp 11%, attn_q_b 10% of it) 15.7 ms of device
  time per token; Engram host lookup (3 per step, host op, the step's issuing thread blocks);
  routed experts ~6-7 ms (40% of each layer's experts are read from host RAM over PCIe in place;
  the decode expert prefetch of radiance is off at tp > 1); ~1770 ops per step (passes are
  recorded and replayed).
- Prefill 18K: Engram 26% (host, inflated by the profiler's drains), attention 20%,
  `mla_project` 17% (now replaced), experts 11%, trunk GEMMs 9%.
- Prefill 56K: Engram 32%, indexer 22% (now WMMA, ~20x faster), attention 12%.

## Known constraints

- radiance 2.1.0 on RDNA4 refuses block-FP8 linears; int8 (IQ8R: int8 rotated by fwht128, u8 scale a
  256 columns under a bf16 scale a row) is served. Converting the trunk to IQ8R needs a new
  container or `rad-convert --in-place --replace`; the test machine has 95 GB free on its local disk; the
  container is 220 GB.
- 64 GB host RAM cannot hold the 118 GiB Engram table (`ngram-placement: ram`); the community
  quantises it to 4 bits (33 GiB).
- The checkpoint ships a DFly2 drafter (5 layers, 0.85B) that the plugin does not load; the plugin
  refuses speculative decoding (the indexer's tail ring and Engram's state are written without
  `num_accepted`).
- Kernel arguments must stay <= 768 bytes with hidden ones (`tools/kernarg.sh`).
- A row must give the same bits within each attention form (batch invariance up to 32 rows).
