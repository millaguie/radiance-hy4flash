#!/bin/bash
# Launches Hy4-flash on a machine with one or two cards, with a radiance image and, optionally, a
# libr4d of your own (Hy4-flash plan steps 6b and 8). Only with any other radiance on the cards
# stopped.
# Usage: [CFG=hy4-full.yaml] [HY4_IMAGE=...] [HY4_LIBR4D=...] run-hy4.sh [cards: 0, 1 or 0,1 (both
# by default)] [dump dir]
#
# Paths (each a variable):
#   HY4_DIR     the working directory, mounted at /hy4: holds the config ($CFG), chat_template.jinja
#               (a copy of the repo's data/chat_template.jinja) and the plugins the config's
#               radiance-home names (/hy4/home: a copy of build-hip-22/radiance_home)   ($HOME/hy4)
#   HY4_MODELS  the directory with the containers, mounted at /models          ($HOME/models/rad)
#   HY4_LIBR4D  a libr4d.so mounted over the image's (for example the one without commit b264e9b,
#               from https://github.com/millaguie/radiance-wave-rescale). Unset, the image's own is used.
#               A libr4d only works with the image of its own version, so the two go together.
#   HY4_IMAGE   the radiance image (default stilldeadcode/radiance:2.2.0, the version these plugins
#               build against: a plugin of another ABI is refused at load).
# With no dump dir (and no HY4_DUMP_DIR exported) the debug tap stays off. With one, each layer in
# HY4_DUMP_LAYERS copies its activations to pageable memory and synchronises: no use for measuring.
# A config with tp 2 needs both cards; with tp 1, one.
# If the config carries the line "# template-sha256: <hash>", it first checks that
# $HY4_DIR/chat_template.jinja is that one. HY4_MLA (absorbed|expanded) goes to the plugin: choosing
# between the two needs a container converted with HY4_MLA=both (what convert-hip.sh does).
# RADIANCE_CALIB_* (GPTQ calibration, deploy/calib-batches.sh) pass through unchanged, and so does
# RADIANCE_AQL_CHAIN_FENCE (2.2.0: 1 keeps the packet processor's acquire on chained launches).
set -eu
CARDS=${1:-0,1}; DUMP=${2:-${HY4_DUMP_DIR:-}}; CFG=${CFG:-hy4.yaml}
IMAGE=${HY4_IMAGE:-stilldeadcode/radiance:2.2.0}
# The scripts here are run with sudo: the defaults are under the home of the user who called sudo.
H=$(getent passwd "${SUDO_USER:-$(id -un)}" | cut -d: -f6)
DIR=${HY4_DIR:-$H/hy4}
MODELS=${HY4_MODELS:-$H/models/rad}
[ -f "$DIR/$CFG" ] || { echo "no config at $DIR/$CFG (set HY4_DIR, CFG)" >&2; exit 1; }
[ -d "$MODELS" ] || { echo "no models directory at $MODELS (set HY4_MODELS)" >&2; exit 1; }
LIBR4D=()
if [ -n "${HY4_LIBR4D:-}" ]; then
  [ -f "$HY4_LIBR4D" ] || { echo "no libr4d at $HY4_LIBR4D" >&2; exit 1; }
  LIBR4D=(-v "$HY4_LIBR4D":/opt/radiance/share/radiance/kernels/libr4d.so:ro)
else
  echo "run-hy4: HY4_LIBR4D is not set: the image's own libr4d" >&2
fi
WANT=$(sed -n 's/^# template-sha256: *\([0-9a-f]*\).*/\1/p' "$DIR/$CFG")
if [ -n "$WANT" ]; then
  HAVE=$(sha256sum "$DIR/chat_template.jinja" | cut -d' ' -f1)
  [ "$HAVE" = "$WANT" ] || { echo "chat_template.jinja is not the one in the repo ($HAVE, the config asks for $WANT)" >&2; exit 1; }
fi
VID=$(getent group video | cut -d: -f3); RID=$(getent group render | cut -d: -f3)
docker rm -f hy4-test 2>/dev/null || true
docker run -d --name hy4-test --network host --init --ulimit memlock=-1 \
  --device /dev/kfd --device /dev/dri --group-add "$RID" --group-add "$VID" \
  --security-opt seccomp=unconfined -e ROCR_VISIBLE_DEVICES=$CARDS -e HY4_MLA -e HY4_MLA_ATTN -e HY4_CHAIN -e HY4_DFLY \
  -e RADIANCE_CALIB_DIR -e RADIANCE_CALIB_LAYERS -e RADIANCE_CALIB_TOKENS -e RADIANCE_AQL_CHAIN_FENCE \
  -e HY4_DUMP_DIR=$DUMP -e HY4_DUMP_LAYERS=${HY4_DUMP_LAYERS:-0,1,2} \
  -v "$DIR":/hy4 -v "$MODELS":/models:ro \
  ${LIBR4D[@]+"${LIBR4D[@]}"} \
  "$IMAGE" --config /hy4/$CFG
