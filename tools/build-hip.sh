#!/bin/sh
# Builds the plugins with the HIP rows (gfx1201) inside the `build` stage of the radiance 2.2.0
# image: radiance's Dockerfile at v2.2.0, built with `--target build -t radiance-build:2.2.0` and
# GPU_TARGETS=gfx1201. The plugins must be compiled against the headers of the radiance they load in
# (ABI 43), so the 2.1.0 image (radiance-build:latest) no longer serves. Output in build-hip-22/;
# the 2.1.0 trees (build-hip, build-hip2) are left alone.
#   tools/build-hip.sh [build dir]   (default build-hip-22)
set -e
cd "$(dirname "$0")/.."
B=${1:-build-hip-22}
docker run --rm -u "$(id -u):$(id -g)" -v "$PWD:/hy4" -w /hy4 -e B="$B" radiance-build:2.2.0 sh -c '
  cmake -S . -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -Dradiance_DIR=/stage/opt/radiance/lib/cmake/radiance \
    "-DRAD_GPU_TARGETS=gfx1201;gfx11-generic" \
    -DCMAKE_HIP_COMPILER=/opt/rocm-repeatable/lib/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm \
    -DCMAKE_HIP_FLAGS=--rocm-path=/opt/rocm -DCMAKE_INSTALL_PREFIX=/hy4/prefix-hip >/dev/null &&
  cmake --build "$B" -j16'
