# radiance-hy4flash

Plugins for [radiance](https://codeberg.org/StillDeadcode/radiance), the inference server for AMD
GPUs, that serve Tencent's [Hy4-flash](https://huggingface.co/tencent/Hy4-flash). There are two, both
built out of tree and loaded by the official radiance image:

- `hyv4flash`, an architecture plugin: the 42 layers with their four residual streams and
  hyper-connections, the MLA attention (served in its absorbed form), the sparse layers' indexer,
  the routed experts, the Engram n-gram embedding and the DFly2 block drafter;
- `hy4k`, an op library: the ops radiance does not have (`docs/OPS-HY4.md`), each with an FP32 host
  kernel that serves as the oracle and a HIP kernel for the card.

Nothing in radiance itself is changed. The support may be proposed upstream:
[radiance issue #520](https://codeberg.org/StillDeadcode/radiance/issues/520).

## Status

- radiance **2.2.0** (plugin ABI 43). The build stops on any other version.
- Tested on one machine: 2x AMD Radeon AI PRO R9700 (gfx1201, 32 GB each), tensor parallel 2, 64 GB
  of host RAM, the container on an NVMe drive. The HIP kernels also build for `gfx11-generic`;
  Strix Halo is not tested ([`docs/STRIX-HALO.md`](docs/STRIX-HALO.md)).
- **Text only** in the released form. The vision tower is written and was tested on radiance 2.1
  with a patched core (the official image has no Hy4 image processor); see "Vision" in
  [`docs/DESIGN.md`](docs/DESIGN.md).
- Hy4-flash does not fit in 64 GB of VRAM: the experts are tiered between the cards and pinned host
  memory, and the Engram tables are read from the drive.

## The container

A converted container is on Hugging Face:
[millaguie/Hy4-flash-radiance-W4NL64-IQ8R-DFly2](https://huggingface.co/millaguie/Hy4-flash-radiance-W4NL64-IQ8R-DFly2)
(140 GiB). It holds:

| part | format |
|---|---|
| routed experts (512 a layer, 41 layers) | GPTQ, 4-bit codes over the `w4nl` codebook, an E4M3 scale per 64 weights, rotated (`w4nl64`) |
| trunk: attention, the dense FFN, the shared expert, the experts' latent projections | IQ8R: rotated int8, round to nearest with a scale search |
| DFly2 drafter (5 layers, drafts 5 tokens a pass) | IQ8R linears |
| Engram tables (4 layers) | the E4M3 codes the model itself rounds each row to, 59 GiB in place of 118 GiB of bf16 |
| indexer, norms, router, hyper-connections, `lm_head` | bf16, as in the checkpoint |

It has no vision tower. It was converted with `--max-ctx 65536` and `--max-spec 5`.

## Metrics

radiance 2.2.0 with these plugins on the test machine, power cap 330 W a card, a throughput bench
against the HTTP API with 256 generated tokens a request
([`docs/BENCH-2026-11-21.md`](docs/BENCH-2026-11-21.md)):

| | decode, 1 client | 4 clients | 8 clients | prefill, 5.7K tokens | 18K tokens |
|---|---|---|---|---|---|
| no speculation | 57.7 tok/s | 196.2 | 360.6 | 3,036 tok/s | 2,706 |
| DFly2, adaptive window, `RADIANCE_AQL_CHAIN_FENCE=1` | 109.8 tok/s | 341.3 | 425 (425 to 603 across runs) | 2,976 tok/s | — |

The decode figures at 4 and 8 clients are the sum over the clients.

Quality of this container:

- BFCL v4, `non_live`: 79.56 % (8 clients, DFly2 on).
- Per-position KL divergence against an FP32 reference pass of the original weights: mean 0.186
  (10,851 positions, top-1 agreement 88.19 %).
- With greedy sampling the text is the same bytes with and without the drafter.

For comparison, on the same machine:

| | prefill | decode, 1 client | 4 clients | 8 clients | BFCL v4 `non_live` |
|---|---|---|---|---|---|
| llama.cpp (PR #32517), AngelSlim Q4_K_M GGUF, tuned (`-ub 2048 -fa 1`, 19 expert layers on the cards) | 360 tok/s | 13.75 tok/s | — | — | — |
| Hy4-flash, radiance 2.2.0 + these plugins, DFly2 | 2,976 tok/s (3,036 without the drafter) | 109.8 tok/s | 341.3 | 425 | 79.56 % |
| Qwen3.8-Flash-Next on radiance (all weights in VRAM) | ~6,400 tok/s | 201 tok/s | 581 | 972 | 84.17 % |

The gap to llama.cpp is mostly placement: it keeps the experts of 22 layers on the CPU and reads
the Engram table through mmap.

## How to run it

You need two 32 GB cards radiance has kernels for (tested: R9700), 64 GB of host RAM, about 151 GB
on an NVMe drive, Docker, and an `amdgpu` driver that supports ROCm 10.1 (the host does not need
ROCm). The steps use `~/hy4` as the working directory and `~/models/rad` for the container.

**1. Get radiance 2.2.0.**

```sh
docker pull stilldeadcode/radiance:2.2.0
```

**2. Build the plugins.** They must be compiled against the headers of the radiance they load in.
`tools/build-hip.sh` does it inside the `build` stage of radiance's own Dockerfile, which you build
once:

```sh
git clone --branch v2.2.0 --depth 1 https://codeberg.org/StillDeadcode/radiance.git
docker buildx build --target build --build-arg GPU_TARGETS=gfx1201 -t radiance-build:2.2.0 radiance

git clone https://github.com/millaguie/radiance-hy4flash.git && cd radiance-hy4flash
tools/build-hip.sh            # writes build-hip-22/radiance_home/{architectures/hyv4flash.so,kernels/hy4k.so}
tools/kernarg.sh              # optional: checks every kernel against the backend's dispatch limits
```

Without Docker, against an installed radiance 2.2.0 (`cmake --install` of radiance into a prefix):

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -Dradiance_DIR=<prefix>/lib/cmake/radiance "-DRAD_GPU_TARGETS=gfx1201"
cmake --build build           # writes build/radiance_home/
```

A radiance built without HIP gives the host kernels only, which is enough for the tests in `tests/`
but not for serving on a card.

**3. Put the plugins on the plugin search path.** A radiance home is a directory with
`architectures/`, `kernels/` and `quantizers/`; `RADIANCE_HOME` is a search path of homes separated
by `:`, the first winning, so the plugins go ahead of the image's own without copying it
(radiance's `docs/PLUGIN.md` §13). The build tree's `radiance_home/` is such a home:

```sh
mkdir -p ~/hy4 && cp -r build-hip-22/radiance_home ~/hy4/home
cp data/chat_template.jinja deploy/example-tp2-dfly2.yaml ~/hy4/
```

The config names the path as `server.radiance-home` (the same as setting `RADIANCE_HOME`) and lists
the kernel libraries to load, `hy4k` first:

```yaml
server:
  kernels: hy4k,libr4d,libavx
  radiance-home: /hy4/home:/opt/radiance/share/radiance
```

To check that both load:

```sh
docker run --rm -v ~/hy4/home:/plugins:ro -e RADIANCE_HOME=/plugins:/opt/radiance/share/radiance \
    --entrypoint rad-info stilldeadcode/radiance:2.2.0 --plugins
```

**4. Download the container** (one `.rad` file, 140 GiB) to a local NVMe drive. Hugging Face takes
files of at most 50 GB, so it is uploaded in four parts; join them and check the result:

```sh
hf download millaguie/Hy4-flash-radiance-W4NL64-IQ8R-DFly2 --include 'hy4-flash-w4nl64-gptq-iq8r-dfly2-e4m3.rad.part*' 'SHA256SUMS' --local-dir ~/models/rad
cd ~/models/rad
cat hy4-flash-w4nl64-gptq-iq8r-dfly2-e4m3.rad.part* > hy4-flash-w4nl64-gptq-iq8r-dfly2-e4m3.rad
sha256sum -c SHA256SUMS      # every part and the joined file must say OK
rm hy4-flash-w4nl64-gptq-iq8r-dfly2-e4m3.rad.part*                 # once the check passes: 140 GiB back
```

**5. The config.** [`deploy/example-tp2-dfly2.yaml`](deploy/example-tp2-dfly2.yaml), copied to
`~/hy4` in step 3; set `model:` to the file you downloaded:

```yaml
server:
  host: 127.0.0.1
  port: 8000
  kernels: hy4k,libr4d,libavx
  radiance-home: /hy4/home:/opt/radiance/share/radiance
memory:
  gpu-headroom-mib: 128
models:
  hy4-flash:
    aliases: [tencent/Hy4-flash]
    model: /models/hy4-flash-w4nl64-gptq-iq8r-dfly2-e4m3.rad
    tp: 2
    placement: expert_tiered          # experts tiered between VRAM and pinned host memory
    host-pool-mib: 39000              # 39 GB of a 64 GB host
    ngram-placement: disk             # the Engram tables stay on the drive
    max-model-len: 65536
    max-num-batched-tokens: 2048
    max-num-seqs: 8
    mm-max-patches: 0                 # text only
    num-speculative-tokens: 5         # DFly2
    draft-window: adaptive
    override-chat-template: /hy4/chat_template.jinja
    reasoning-effort-map: {low: high, medium: high, xhigh: high, max: high, minimal: no_think, none: no_think}
```

Hy4-flash's template knows two reasoning levels, `high` and `no_think`; the map translates the
OpenAI ones. Without the drafter, set `num-speculative-tokens: 0` and drop `draft-window`.

**6. Start it.**

```sh
docker run -d --name hy4-flash --network host --init --ulimit memlock=-1 \
    --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined \
    --group-add "$(getent group render | cut -d: -f3)" \
    --group-add "$(getent group video | cut -d: -f3)" \
    -e RADIANCE_AQL_CHAIN_FENCE=1 \
    -v ~/hy4:/hy4 -v ~/models/rad:/models:ro \
    stilldeadcode/radiance:2.2.0 --config /hy4/example-tp2-dfly2.yaml
docker logs -f hy4-flash      # ready at "server: listening"

curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model": "hy4-flash", "messages": [{"role": "user", "content": "Hello"}]}'
```

`RADIANCE_AQL_CHAIN_FENCE=1` is for DFly2 with several clients on radiance 2.2.0: without it the
8-client throughput falls by about a third ("Moving to radiance 2.2.0" in
[`docs/DESIGN.md`](docs/DESIGN.md)). The other `docker run` options are explained in radiance's
`docs/DOCKER.md`. [`deploy/run-hy4.sh`](deploy/run-hy4.sh) wraps the same line for test runs.

**Optional: a libr4d without commit b264e9b**, about 15 % more generation speed on these cards.
[millaguie/radiance-wave-rescale](https://github.com/millaguie/radiance-wave-rescale) builds it for
each radiance version; take the `libr4d.so` of its release `v2.2.0` (it only works with the image
of the same version) and mount it over the image's:

```sh
    -v ~/hy4/libr4d.so:/opt/radiance/share/radiance/kernels/libr4d.so:ro \
```

## Convert it yourself

The conversion starts from Tencent's BF16 checkpoint (about 410 GiB on disk) and runs `rad-convert`
with the plugins loaded. [`tools/convert-hip.sh`](tools/convert-hip.sh) runs it in the build image;
the recipes are in [`data/recipes/`](data/recipes), and [`docs/DESIGN.md`](docs/DESIGN.md) describes
each step ("Calibration", "`w4nl64` experts", "Trunk format and launches", "Engram table as E4M3
codes", "Moving to radiance 2.2.0"). In order:

1. `conv/trim.py` with all 42 layers writes `engram-consts.safetensors`, the Engram hash constants
   the checkpoint does not ship, which every conversion adds with `--add`.
2. `hy4flash-w4nl64-rtn.recipe`: a round-to-nearest container, served to record the GPTQ Grams
   (`deploy/calib-batches.sh`, with a corpus from `ref/corpus_calib.py`).
3. `hy4flash-w4nl64-gptq.recipe`: the experts by GPTQ against those Grams.
4. `hy4flash-w4nl64-gptq-iq8r.recipe`: the trunk in IQ8R; `hy4flash-w4nl64-gptq-iq8r-dfly.recipe`
   and `deploy/conv-dfly.sh`: the drafter.
5. `conv/engram_codes.py` and `deploy/conv-engram-e4m3.sh`: the Engram tables as E4M3 codes.
6. `deploy/conv-clean.sh`: a clean copy with every weight copied, the published file.

The calibration corpus used for the published container (conversations of tool-using agents, in
Spanish, and BigCodeBench-hard for code) holds personal data and is not published; bring your own. The scripts in `deploy/` take their paths from variables documented at the top of each.

## Repository layout

| path | what |
|---|---|
| `arch/hyv4flash/` | the architecture plugin (`hyv4flash.cpp`, `vision.h`) and its vendored `rad_fp8.h` |
| `kernels/hy4k/` | the op library: schemas, FP32 host kernels (`ref_*.cpp`), HIP kernels (`dev_*.hip`) |
| `conv/` | conversion helpers: the trimmed checkpoint and Engram constants, the Engram E4M3 codes |
| `data/` | the chat template radiance renders and the conversion recipes |
| `deploy/` | example server configs, the launcher, the calibration and conversion scripts |
| `tools/` | builds in the radiance build image, `rad-convert` with the plugins, the kernel argument check, a Python with torch |
| `ref/` | the layer-by-layer FP32/BF16 reference from the Transformers code, and the comparison scripts |
| `tests/` | op tests against the reference (host), card kernels against the host oracle, template and server tests |
| `hf/` | the Hugging Face model card and the Tencent Hy License that go beside the container |
| `docs/` | the documents below |

## Documentation

- [`docs/DESIGN.md`](docs/DESIGN.md): the design. Which radiance ops are reused and which are new,
  how each layer is emitted, the state kept between steps, the absorbed MLA, quantisation, vision,
  the move to radiance 2.2.0.
- [`docs/OPS-HY4.md`](docs/OPS-HY4.md): the contract of every `hy4k` op.
- [`docs/DFLY2.md`](docs/DFLY2.md): the DFly2 drafter.
- [`docs/BENCH-2026-11-21.md`](docs/BENCH-2026-11-21.md): the measurements.
- [`docs/PERF-STATE-2026-11-21.md`](docs/PERF-STATE-2026-11-21.md) and
  [`docs/PERF-OPTIONS-2026-11-21.md`](docs/PERF-OPTIONS-2026-11-21.md): where the time went and what
  is left to try.
- [`docs/STRIX-HALO.md`](docs/STRIX-HALO.md): what serving on Strix Halo would need (not tested).

## License

The plugins are [Apache-2.0](LICENSE), radiance's own license. Keep [`NOTICE`](NOTICE) with any copy
or derivative: it names the file copied from radiance and the two copied from the model repository.

The Hy4-flash weights are Tencent's, released under the
[Tencent Hy License](https://huggingface.co/tencent/Hy4-flash/blob/main/LICENSE) (a copy is in
[`hf/LICENSE`](hf/LICENSE)). They are not in this repository, and this license does not cover them:
a converted container is a derivative of the weights and stays under the Tencent Hy License,
including the additional commercial terms of its Section 10. Hy4-flash is developed by the Tencent
Hy Team; this project is not affiliated with or endorsed by Tencent.
