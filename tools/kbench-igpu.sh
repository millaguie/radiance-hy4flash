#!/bin/sh
# rad-kbench of the trimmed model on a Radeon 890M iGPU (gfx1150 with gfx11-generic code), with
# build-hip's hy4k ahead of the image's libraries. Usage: tools/kbench-igpu.sh [rad-kbench options]
# HY4_DIR (default $HOME/hy4) is mounted at /m and holds recorte3.rad (the trimmed model,
# conv/trim.py) and kbench/, where the results go.
cd "$(dirname "$0")/.."
exec docker run --rm --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined \
  -u "$(id -u):$(id -g)" --group-add "$(getent group render | cut -d: -f3)" \
  -v "$PWD/build-hip/radiance_home:/hy4home:ro" -v "${HY4_DIR:-$HOME/hy4}:/m" --entrypoint sh radiance-build:latest -c '
  export RADIANCE_HOME=/hy4home:/stage/opt/radiance/share/radiance; cd /m/kbench
  /stage/opt/radiance/bin/rad-kbench -m /m/recorte3.rad --kernels hy4k:libr4d:libavx:libref "$@"' sh "$@"
