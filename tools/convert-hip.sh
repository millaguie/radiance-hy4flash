#!/bin/bash
# rad-convert inside the radiance 2.2.0 build image, with the plugins from build-hip-22 (the card ones,
# as they will be served; the image's host backend converts the same).
# Usage: [CALIB=/m/...] tools/convert-hip.sh <rad-convert arguments with /m/... or /models/... paths>
# Mounts (each a variable):
#   HY4_DIR       the working directory, mounted at /m: outputs, extra sources, Grams   ($HOME/hy4)
#   HY4_MODELS    optional: a models directory mounted at /models, for example the Hugging Face
#                 cache that holds the checkpoint ($HOME/.cache/huggingface/hub; a snapshot's files
#                 are symlinks into its blobs/, so mount the cache, not the snapshot)
#   HY4_BUILD     the build tree with the plugins (build-hip-22)
# The GPTQ recipe reads CALIB (the Grams dir, as an in-container path). HY4_MLA=both leaves the container
# servable in both forms of the MLA (docs/DESIGN.md, "Absorbed MLA").
# Always puts in data/chat_template.jinja: the checkpoint's template uses `.format()`, which radiance's
# Jinja does not have, and without it /v1/chat/completions returns 501.
cd "$(dirname "$0")/.." || exit 1
export HY4_MLA="${HY4_MLA:-both}"   # by default, the container serves both forms of the MLA
DIR=${HY4_DIR:-$HOME/hy4}
[ -d "$DIR" ] || { echo "convert-hip: no working directory at $DIR (set HY4_DIR)" >&2; exit 1; }
MODELS=()
if [ -n "${HY4_MODELS:-}" ]; then
  [ -d "$HY4_MODELS" ] || { echo "convert-hip: no models directory at $HY4_MODELS" >&2; exit 1; }
  MODELS=(-v "$HY4_MODELS:/models")
fi
exec docker run --rm -e HY4_DEBUG_MOE -e RADIANCE_LOG -e CALIB -e HY4_MLA -u "$(id -u):$(id -g)" --security-opt seccomp=unconfined \
  -v "$PWD/${HY4_BUILD:-build-hip-22}/radiance_home:/hy4home:ro" -v "$PWD/data:/hy4data:ro" -v "$DIR:/m" \
  ${MODELS[@]+"${MODELS[@]}"} --entrypoint sh radiance-build:2.2.0 -c '
  export RADIANCE_HOME=/hy4home:/stage/opt/radiance/share/radiance
  /stage/opt/radiance/bin/rad-convert --chat-template /hy4data/chat_template.jinja "$@"' sh "$@"
