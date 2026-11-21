#!/bin/bash
# Adds the container key hy4.engram_codes=rowmax448 to the served GPTQ container, in place, without
# writing a weight: its Engram tables already are conv/engram_codes.py's E4M3 codes
# (deploy/conv-engram-e4m3.sh ran before the key existed), and the plugin now serves an E4M3 table
# only with the key (decl_engram). Run on the machine that holds it, with the engine stopped (rad-convert refuses a
# container another process has open). Usage: set-engram-key.sh [--plan-only] [-v]
#
# How: rad-convert has no header-only edit, but an --in-place run that keeps every weight is one.
# It re-derives the container's metadata from the checkpoint and the --set keys (only radiance.*
# keys are carried over), so this is conv-engram-e4m3.sh's run again, the same checkpoint, recipe,
# extra sources, environment and keys, plus the new key, and without --replace: every weight then
# plans as the container holds it and is kept where it is (add_existing), and a weight that would
# plan otherwise is refused (`--in-place: '...' is in ... as ...`), not written. What is written is
# the new tables past the end (a few MB) and the header; <container>.pre-append keeps the old
# header and length, so the run is undone by restoring it (as for every in-place run).
#
# The codes overlay is given as its header-only form (conv/engram_codes.py --header-only: the
# names, dtypes and shapes, its codes zeros): those are all the plan reads of it, since the only
# weights it supplies are the four tables the container holds, and a held weight is kept or
# refused, never read from its source. The full overlay would do as well; the header-only one is
# a megabyte, needs none of the 63.4 GB, and cannot put a byte of itself in the container here.
# The run's closing `--in-place:` line must say `0 added past its end`; anything else means the
# declare changed since the conversion. The script keeps the log ($LOG) and checks that line: when
# it says otherwise it exits 1, and since the header is rewritten by then, the run is undone by
# restoring the header from the .pre-append (the copy made aside before the run holds the header
# before that).
#
# The plugin must be one that knows the key (decl_engram reads hy4.engram_codes). The image and the
# plugins are radiance 2.2.0's (stilldeadcode/radiance:2.2.0 and this checkout's
# build-hip-22/radiance_home, which tools/build-hip.sh writes), and can be overridden: HY4_IMAGE,
# HY4_PLUGINS (a radiance_home built against the image's radiance), HY4_REPO (holds conv/, data/
# and tools/); PY is a python3 with torch, numpy and safetensors (tools/py-docker.sh runs one from
# an image). Later in-place runs
# over this container must pass the key again (conv-engram-e4m3.sh and conv-dfly.sh do), or the
# plugin refuses the result.
#
# rad-convert --in-place overwrites <container>.pre-append with the header it replaces, so the one
# there now is copied aside first, under a new name: never overwritten.
#
# Paths (each a variable; the defaults are under $HOME):
#   HY4_CONTAINER    the served container                 ($HOME/models/rad/hy4-flash-w4nl64-gptq.rad)
#   HY4_CKPT         the tencent/Hy4-flash checkpoint: a snapshot directory of a Hugging Face cache
#                    (its blobs/ are mounted with it) or a plain directory
#                    ($HOME/.cache/huggingface/hub/models--tencent--Hy4-flash/snapshots/<revision>)
#   HY4_DIR          the working directory: holds full-consts/engram-consts.safetensors, and the
#                    logs go there                        ($HOME/hy4)
#   HY4_CALIB        the GPTQ Grams the recipe names      ($HY4_DIR/calib-gptq)
#   HY4_OVERLAY_DIR  where the Engram codes overlay lives ($HOME/models/engram-e4m3)
#   HY4_REPO         this repository: conv/, data/, tools/ (the checkout this script is in)
#   HY4_PLUGINS      a radiance_home with architectures/ and kernels/, built against the image's
#                    radiance                             ($HY4_REPO/build-hip-22/radiance_home)
#   HY4_IMAGE        the radiance image                   (stilldeadcode/radiance:2.2.0)
#   PY               a python3 with torch, numpy and safetensors ($HY4_REPO/tools/py-docker.sh)
set -eu
C=${HY4_CONTAINER:-$HOME/models/rad/hy4-flash-w4nl64-gptq.rad}
S=${HY4_CKPT:-$HOME/.cache/huggingface/hub/models--tencent--Hy4-flash/snapshots/d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959}
DIR=${HY4_DIR:-$HOME/hy4}
CALIB=${HY4_CALIB:-$DIR/calib-gptq}
REPO=${HY4_REPO:-$(cd "$(dirname "$0")/.." && pwd)}
IMAGE=${HY4_IMAGE:-stilldeadcode/radiance:2.2.0}
PLUGINS=${HY4_PLUGINS:-$REPO/build-hip-22/radiance_home}
PY=${PY:-$REPO/tools/py-docker.sh}
OVDIR=${HY4_OVERLAY_DIR:-$HOME/models/engram-e4m3}
HO=$OVDIR/hy4-flash-engram-e4m3.header-only.safetensors
RECIPE=hy4flash-w4nl64-gptq-iq8r-dfly.recipe
TAG=before-engram-key

PLAN=0
for a in "$@"; do
  case "$a" in
    --plan-only) PLAN=1 ;;
    -v|--verbose) ;;
    *) echo "set-engram-key: no '$a' here: the run on the served container is fixed; only --plan-only and -v" >&2; exit 1 ;;
  esac
done

[ -e "$C" ] || { echo "set-engram-key: no container at $C (set HY4_CONTAINER)" >&2; exit 1; }
[ -d "$S" ] || { echo "set-engram-key: no checkpoint at $S (set HY4_CKPT)" >&2; exit 1; }
[ -d "$DIR" ] || { echo "set-engram-key: no working directory at $DIR (set HY4_DIR)" >&2; exit 1; }
[ -d "$CALIB" ] || { echo "set-engram-key: no Grams directory at $CALIB (set HY4_CALIB)" >&2; exit 1; }
# A snapshot of a Hugging Face cache is symlinks into ../../blobs: the model's cache directory is
# mounted whole, so they resolve inside the container.
S=$(cd "$S" && pwd)
CKROOT=$(cd "$S/../.." && pwd)
[ -d "$CKROOT/blobs" ] || CKROOT=$S
CKPT=/ckpt${S#"$CKROOT"}
# what $PY has to see when it is tools/py-docker.sh (mounted at the same paths)
export HY4_PY_RO="${HY4_PY_RO:+$HY4_PY_RO:}$CKROOT:$REPO" HY4_PY_RW="${HY4_PY_RW:+$HY4_PY_RW:}$OVDIR"
[ -x "$PY" ] || { echo "set-engram-key: no python at $PY (set PY)" >&2; exit 1; }
[ -f "$PLUGINS/architectures/hyv4flash.so" ] || { echo "set-engram-key: no plugins at $PLUGINS (set HY4_PLUGINS)" >&2; exit 1; }
[ -f "$REPO/data/recipes/$RECIPE" ] || { echo "set-engram-key: no recipe under $REPO/data (set HY4_REPO)" >&2; exit 1; }
mkdir -p "$OVDIR"
[ -e "$HO" ] || "$PY" "$REPO/conv/engram_codes.py" "$S" "$HO" --header-only

if [ "$PLAN" = 0 ] && [ -e "$C.pre-append" ]; then
  B="$C.pre-append.$TAG.$(date +%Y%m%d-%H%M%S)"
  [ -e "$B" ] && { echo "set-engram-key: $B already exists" >&2; exit 1; }
  cp --no-clobber --preserve=all "$C.pre-append" "$B"
  cmp -s "$C.pre-append" "$B" || { echo "set-engram-key: the copy $B differs from $C.pre-append" >&2; exit 1; }
  echo "set-engram-key: $C.pre-append copied to $B"
fi

run() {
  sudo docker run --rm -u "$(id -u):$(id -g)" --security-opt seccomp=unconfined \
    -e RADIANCE_HOME=/hy4home:/opt/radiance/share/radiance -e CALIB=/calib -e HY4_MLA=both \
    -v "$PLUGINS:/hy4home:ro" -v "$REPO/data:/hy4data:ro" \
    -v "$DIR:/m" -v "$(dirname "$C"):/models" -v "$CKROOT:/ckpt:ro" -v "$CALIB:/calib:ro" -v "$OVDIR:/ov:ro" \
    --entrypoint /opt/radiance/bin/rad-convert "$IMAGE" "$CKPT" \
    --chat-template /hy4data/chat_template.jinja --recipe "/hy4data/recipes/$RECIPE" \
    --reuse "/models/$(basename "$C")" --in-place \
    --add /m/full-consts/engram-consts.safetensors --add "/ov/$(basename "$HO")" \
    --max-spec 5 --max-ctx 65536 --set hy4.trunk_i8=1 --set hy4.engram_codes=rowmax448 "$@"
}
if [ "$PLAN" = 1 ]; then run "$@"; exit; fi

LOG=${LOG:-$DIR/set-engram-key.$(date +%Y%m%d-%H%M%S).log}
[ -e "$LOG" ] && { echo "set-engram-key: $LOG already exists" >&2; exit 1; }
set -o pipefail
run "$@" 2>&1 | tee "$LOG"
grep -- '--in-place: [0-9]* weight(s)' "$LOG" | tail -1 | grep -q -- '; 0 added past its end$' || {
  echo "set-engram-key: the log ($LOG) does not say '0 added past its end': weights were written to $C," >&2
  echo "set-engram-key: and its header is already the new one. Undo the run by restoring the header and" >&2
  echo "set-engram-key: length from $C.pre-append (docs: rad-convert --in-place) before serving it." >&2
  exit 1
}
echo "set-engram-key: every weight of $C kept where it was, the key added (log: $LOG)"
