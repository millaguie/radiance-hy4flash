#!/bin/bash
# Hy4-flash launcher for a throughput bench that drives a served model by container name and port:
# like run-hy4.sh, but the container takes the name the bench expects ($HY4_NAME) and the config is
# deploy/example-bench.yaml (port 8000, the model under its served name). Hy4-flash plan step 10.
# It removes a running container of that name first, so only when that one may be stopped.
#   CFG=example-bench.yaml HY4_MLA=absorbed ./run-hy4-bench.sh
# HY4_DIR, HY4_MODELS, HY4_IMAGE and HY4_LIBR4D as in run-hy4.sh (default image radiance 2.2.0; the
# config's radiance-home must hold plugins of the image's ABI). HY4_NAME: the container's name
# (default radiance-hy4-bench).
set -eu
NAME=${HY4_NAME:-radiance-hy4-bench}
CFG=${CFG:-example-bench.yaml}
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
  echo "run-hy4-bench: HY4_LIBR4D is not set: the image's own libr4d" >&2
fi
WANT=$(sed -n 's/^# template-sha256: *\([0-9a-f]*\).*/\1/p' "$DIR/$CFG")
if [ -n "$WANT" ]; then
  HAVE=$(sha256sum "$DIR/chat_template.jinja" | cut -d' ' -f1)
  [ "$HAVE" = "$WANT" ] || { echo "chat_template.jinja is not the one in the repo" >&2; exit 1; }
fi
VID=$(getent group video | cut -d: -f3); RID=$(getent group render | cut -d: -f3)
docker rm -f "$NAME" 2>/dev/null || true
while docker ps -a --format "{{.Names}}" | grep -qx "$NAME"; do sleep 2; done
docker run -d --name "$NAME" --network host --init --ulimit memlock=-1 \
  --device /dev/kfd --device /dev/dri --group-add "$RID" --group-add "$VID" \
  --security-opt seccomp=unconfined -e HY4_MLA \
  -v "$DIR":/hy4 -v "$MODELS":/models:ro \
  ${LIBR4D[@]+"${LIBR4D[@]}"} \
  "$IMAGE" --config /hy4/$CFG
