---
license: other
license_name: tencent-hy-license
license_link: LICENSE
base_model: tencent/Hy4-flash
library_name: radiance
tags:
- radiance
- rocm
- amd
- gptq
- moe
- speculative-decoding
---

# Hy4-flash for radiance: W4NL64 experts, IQ8R trunk, DFly2 drafter

**This is a modified version of [tencent/Hy4-flash](https://huggingface.co/tencent/Hy4-flash): the
weights were quantised and repacked. It is not Tencent's release and Tencent does not endorse it.**

The file here is a **radiance container** (`.rad`, 140 GiB), not a Transformers or GGUF model. It is
served by [radiance](https://codeberg.org/StillDeadcode/radiance) 2.2.0, the inference server for
AMD GPUs, and **needs the Hy4-flash plugins**, which are not part of radiance:
<https://github.com/millaguie/radiance-hy4flash>. Without them radiance does not know the
architecture and refuses the file.

Text only: the container has no vision tower.

## What was changed from the original weights

Converted from the BF16 checkpoint of `tencent/Hy4-flash` (revision
`d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959`) with radiance's `rad-convert` and the plugins:

| part | in the original | here |
|---|---|---|
| routed experts (512 a layer, 41 layers) | bf16 | **GPTQ, 4 bits**: codes over the `w4nl` codebook, rotated by a 128-wide Hadamard, an E4M3 scale per 64 weights (`w4nl64`) |
| trunk: attention projections, the dense FFN, the shared expert, the experts' latent projections | bf16 | **IQ8R**: rotated int8, round to nearest with a scale search, no calibration |
| DFly2 drafter (`dfly_layers`) linears | bf16 | **IQ8R** |
| Engram tables (4 layers) | bf16, 118 GiB | **E4M3 codes**, one byte a value, 59 GiB: the codes the model's own per-row FP8 round trip produces, with the row scales dropped (the RMSNorm that follows cancels them) |
| indexer, norms, router, hyper-connections, `lm_head`, the MLA's `kv_a` | bf16 | unchanged |
| the MLA's `kv_b` | bf16 | bf16, stored in the absorbed form (per head, for a latent KV cache) |
| vision tower | bf16 | **not included** |
| chat template | uses `.format()` | the same template with its 22 `.format(HYTK)` calls rewritten as concatenation, which radiance's Jinja renders; the prompt is the same |

Also added: the Engram hash constants, which the checkpoint does not ship and its code computes at
load. Maximum context 65,536 tokens; drafter depth 5.

The GPTQ calibration used conversations of tool-using agents in Spanish and BigCodeBench-hard. That
corpus is private and is not distributed.

## Metrics

Measured on 2x AMD Radeon AI PRO R9700 (gfx1201, 32 GB each), tensor parallel 2, 64 GB of host RAM,
radiance 2.2.0, power cap 330 W a card, a throughput bench against the HTTP API with 256 generated
tokens a request. The model does not fit in 64 GB of VRAM: the experts are tiered between the cards
and pinned host memory, and the Engram tables are read from an NVMe drive.

| | decode, 1 client | 4 clients | 8 clients | prefill, 5.7K tokens | 18K tokens |
|---|---|---|---|---|---|
| no speculation | 61.4 tok/s | 207.6 | 355 | 3,290 tok/s | 3,000 |

The decode figures at 4 and 8 clients are the sum over the clients. Without speculation the bench
prompt does not matter for decode speed.

**The DFly2 numbers of the first release were inflated.** The bench sends a raw completion prompt,
and the model loops on it (only 29.5 % of the word 6-grams of the output are distinct). A drafter
predicts a loop very well, so the bench gave 109.8 tok/s at one client. With chat prompts the gain
depends on the text. One client, time to the first token included, tok/s:

| chat prompt | no speculation | DFly2 | drafted tokens accepted |
|---|---|---|---|
| code | 52.0 | 79.0 | 66 % |
| tool calls | 48.1 | 83.8 | 97 % |
| Spanish prose | 55.0 | 54.2 | 31 % |
| six chats at once (sum) | 128 | 109 | 51 % |

DFly2 pays off for code and tool calls at one or two clients. For prose, and for several clients
at once, serve without it (`num-speculative-tokens: 0`). The six-chat row was measured on the
previous kernels; the other rows on the current ones.

Quality:

- BFCL v4, `non_live`: **79.56 %** (8 clients, DFly2 on).
- Per-position KL divergence against an FP32 reference pass of the original weights: mean **0.186**
  (10,851 positions, top-1 agreement 88.19 %).
- With greedy sampling the text is the same bytes with and without the drafter.

For comparison, on the same machine:

| | prefill | decode, 1 client | 4 clients | 8 clients | BFCL v4 `non_live` |
|---|---|---|---|---|---|
| llama.cpp (PR #32517), AngelSlim Q4_K_M GGUF, tuned | 360 tok/s | 13.75 tok/s | — | — | — |
| this container, no speculation | 3,290 tok/s | 61.4 tok/s | 207.6 | 355 | 79.56 % (with DFly2) |
| Qwen3.8-Flash-Next on radiance (all weights in VRAM) | ~6,400 tok/s | 201 tok/s | 581 | 972 | 84.17 % |

## How to run it

The full steps are in the
[GitHub README](https://github.com/millaguie/radiance-hy4flash#how-to-run-it). In short:

1. `docker pull stilldeadcode/radiance:2.2.0`
2. Build the plugins from <https://github.com/millaguie/radiance-hy4flash> against radiance 2.2.0
   (`tools/build-hip.sh`) and copy `build-hip-22/radiance_home` to `~/hy4/home`, with
   `data/chat_template.jinja` and `deploy/example-tp2-dfly2.yaml` beside it in `~/hy4`.
3. Download the container. Hugging Face takes files of at most 50 GB, so the 140 GiB `.rad` is in
   four parts: download them, join them and check the result.

   ```sh
   hf download millaguie/Hy4-flash-radiance-W4NL64-IQ8R-DFly2 --local-dir ~/models/rad
   cd ~/models/rad && cat hy4-flash-w4nl64-gptq-iq8r-dfly2-e4m3.rad.part* > hy4-flash-w4nl64-gptq-iq8r-dfly2-e4m3.rad && sha256sum -c SHA256SUMS
   ```
4. Start the server:

```sh
docker run -d --name hy4-flash --network host --init --ulimit memlock=-1 \
    --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined \
    --group-add "$(getent group render | cut -d: -f3)" \
    --group-add "$(getent group video | cut -d: -f3)" \
    -v ~/hy4:/hy4 -v ~/models/rad:/models:ro \
    stilldeadcode/radiance:2.2.0 --config /hy4/example-tp2-dfly2.yaml
```

The config's keys that matter for this container:

```yaml
server:
  kernels: hy4k,libr4d,libavx
  radiance-home: /hy4/home:/opt/radiance/share/radiance   # the plugins ahead of the image's own
models:
  hy4-flash:
    model: /models/hy4-flash-w4nl64-gptq-iq8r-dfly2-e4m3.rad
    tp: 2
    placement: expert_tiered
    host-pool-mib: 39000
    ngram-placement: disk
    max-model-len: 65536
    max-num-seqs: 8
    mm-max-patches: 0
    num-speculative-tokens: 5
    draft-window: adaptive
    override-chat-template: /hy4/chat_template.jinja
    reasoning-effort-map: {low: high, medium: high, xhigh: high, max: high, minimal: no_think, none: no_think}
```

Requirements: two 32 GB cards radiance has kernels for (tested on the R9700 only), 64 GB of host
RAM, the container on an NVMe drive. The API is OpenAI-compatible (`/v1/chat/completions`), with
reasoning (`high` or `no_think`) and tool calls.

## Conversion recipe

The recipes and scripts are in the GitHub repository (`data/recipes/`, `deploy/`, `conv/`), and
its `docs/DESIGN.md` explains each choice. The steps:

1. `hy4flash-w4nl64-rtn.recipe`: a round-to-nearest container, served to record the GPTQ Grams of
   the 41 MoE layers.
2. `hy4flash-w4nl64-gptq.recipe`: the experts by GPTQ against those Grams.
3. `hy4flash-w4nl64-gptq-iq8r-dfly.recipe`: the trunk and the drafter in IQ8R. Its rules:

```
blk.*.ffn_*_exps.*.weight   gptq table=w4nl group=64 scale=fp8_e4m3 scale2=f32 block2=*x* scale2_value=0.0001220703125 transform=fwht128 rule=search cd=3 calib=$CALIB
blk.*.<trunk linear>.weight rtn  codes=i8 group=256 scale=u8 scale2=bf16 block2=1x* transform=fwht128 clamp=sym rule=search
dfly.*.<linear>.weight      rtn  codes=i8 group=256 scale=u8 scale2=bf16 block2=1x* transform=fwht128 clamp=sym rule=search
```

4. `conv/engram_codes.py`: the Engram tables as E4M3 codes, every row checked against the bf16 row
   it replaces; the container carries the key `hy4.engram_codes=rowmax448`.
5. `deploy/conv-clean.sh`: this file, every weight copied from the container that was measured.

## License and attribution

Hy4-flash is developed by the **Tencent Hy Team** and released under the **Tencent Hy License,
Version 1.0**. This container is a Derivative Work of it (a quantised version of the model weights)
and is distributed under the same license: a copy is in [`LICENSE`](LICENSE) in this repository,
unchanged. Nothing here removes or limits its Section 10 (additional commercial terms for large
Model-as-a-Service businesses), which applies to every recipient of this file as it does to the
original.

Notice of changes, as Section 4(b) asks: the weights in this file were modified from Tencent's
release by Millaguie in November 2026, as the table under "What was changed from the original
weights" lists; the chat template inside it was modified as that table says. The original model
card, with its documentation and citation, is at <https://huggingface.co/tencent/Hy4-flash>.

The plugins that serve it are a separate work under Apache-2.0
(<https://github.com/millaguie/radiance-hy4flash>); radiance is Apache-2.0, copyright Deadcode and
the radiance contributors.
