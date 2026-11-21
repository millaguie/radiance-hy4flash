#!/bin/bash
# Appends the DFly2 drafter (docs/DFLY2.md) in place to the served GPTQ container, as an earlier
# in-place run appended the IQ8R trunk. Run on the machine that holds the container.
# Usage: [DFLY_I8=1] conv-dfly.sh [--plan-only] [-v]
#
# The served container holds the drafter, in IQ8R, since 2026-11-21: both forms of this script have
# run on it, and only DFLY_I8=1 applies to it now (its recipe is the one the container was last
# extended with; a run that changes no rule keeps every weight). The form without DFLY_I8 is how a
# container with no drafter gets one: against the served one its recipe, with no rule for
# `dfly.*`, would plan the drafter's linears as bf16, which --in-place refuses.
#
# The drafter is declared only at max_spec 5 (radiance drives a drafter at depth == max_spec), hence
# --max-spec 5 (new weights need no --replace: --in-place writes them past the end); the trunk declares the same weights at any max_spec, so every trunk weight is kept.
# The recipe is the one the container was last extended with (a kept weight must match its rule):
# no rule names `dfly.*`, so the drafter goes in bf16 as the checkpoint holds it, whatever the trunk's
# format.
#
# DFLY_I8=1: rewrites the drafter the container already holds, in place, with its linears in IQ8R
# (data/recipes/hy4flash-w4nl64-gptq-iq8r-dfly.recipe, docs/DFLY2.md): that recipe and
# --replace 'dfly.*', so every drafter weight whose rule changed is written again past the end (the
# norms, attn_kv_a and attn_kv_b_abs keep their encoding and stay where they are). The plugin reads
# the drafter's format from the weights, so no key changes. Run it with --plan-only first.
#
# The container's Engram tables are conv/engram_codes.py's E4M3 codes since
# deploy/conv-engram-e4m3.sh, so this run states them as that one did: the codes overlay as one more
# source (--add) and --set hy4.engram_codes=rowmax448. An in-place run re-derives the metadata from
# the checkpoint and the --set keys, so without the key the container would lose it, and without
# the overlay the tables would plan as bf16, which --in-place refuses. A real run takes only the
# full overlay `engram_codes.py --check-verified --rehash` accepts (a read of 63.4 GB): the tables
# are kept where they are, but a table the container did not hold would be written from the
# overlay. With --plan-only and no overlay, the header-only one stands in.
#
# The image and the plugins are radiance 2.2.0's (stilldeadcode/radiance:2.2.0 and this checkout's
# build-hip-22/radiance_home, which tools/build-hip.sh writes), and can be overridden: HY4_IMAGE,
# HY4_PLUGINS (built against the image's radiance: a plugin of another ABI is refused at load),
# HY4_REPO (holds conv/, data/ and tools/). PY is a python3 with torch,
# numpy and safetensors (tools/py-docker.sh runs one from an image).
#
# rad-convert --in-place overwrites <container>.pre-append with the header it replaces, so the one
# there now (written by the previous in-place run) is copied aside first, under a new name: never
# overwritten. A --plan-only run writes nothing and copies nothing.
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
OV=$OVDIR/hy4-flash-engram-e4m3.safetensors
RECIPE=hy4flash-w4nl64-gptq-iq8r.recipe
TAG=before-dfly
NEED_GB=4   # the drafter's planes and a margin

PLAN=0
for a in "$@"; do
  case "$a" in
    --plan-only) PLAN=1 ;;
    -v|--verbose) ;;
    *) echo "conv-dfly: no '$a' here: the run on the served container is fixed; only --plan-only and -v" >&2; exit 1 ;;
  esac
done

if [ "${DFLY_I8:-0}" = 1 ]; then
  RECIPE=hy4flash-w4nl64-gptq-iq8r-dfly.recipe
  TAG=before-dfly-i8
  set -- --replace 'dfly.*' "$@"   # ahead of the caller's arguments (--plan-only)
fi

[ -e "$C" ] || { echo "conv-dfly: no container at $C (set HY4_CONTAINER)" >&2; exit 1; }
[ -d "$S" ] || { echo "conv-dfly: no checkpoint at $S (set HY4_CKPT)" >&2; exit 1; }
[ -d "$DIR" ] || { echo "conv-dfly: no working directory at $DIR (set HY4_DIR)" >&2; exit 1; }
[ -d "$CALIB" ] || { echo "conv-dfly: no Grams directory at $CALIB (set HY4_CALIB)" >&2; exit 1; }
# A snapshot of a Hugging Face cache is symlinks into ../../blobs: the model's cache directory is
# mounted whole, so they resolve inside the container.
S=$(cd "$S" && pwd)
CKROOT=$(cd "$S/../.." && pwd)
[ -d "$CKROOT/blobs" ] || CKROOT=$S
CKPT=/ckpt${S#"$CKROOT"}
# what $PY has to see when it is tools/py-docker.sh (mounted at the same paths)
export HY4_PY_RO="${HY4_PY_RO:+$HY4_PY_RO:}$CKROOT:$REPO" HY4_PY_RW="${HY4_PY_RW:+$HY4_PY_RW:}$OVDIR"
[ -x "$PY" ] || { echo "conv-dfly: no python at $PY (set PY)" >&2; exit 1; }
[ -f "$PLUGINS/architectures/hyv4flash.so" ] || { echo "conv-dfly: no plugins at $PLUGINS (set HY4_PLUGINS)" >&2; exit 1; }
[ -f "$REPO/data/recipes/$RECIPE" ] || { echo "conv-dfly: no recipe under $REPO/data (set HY4_REPO)" >&2; exit 1; }

USE=$OV
if [ ! -e "$OV" ]; then
  [ "$PLAN" = 1 ] || { echo "conv-dfly: no codes overlay at $OV: conv/engram_codes.py writes it" >&2; exit 1; }
  USE=$OVDIR/hy4-flash-engram-e4m3.header-only.safetensors
  mkdir -p "$OVDIR"
  [ -e "$USE" ] || "$PY" "$REPO/conv/engram_codes.py" "$S" "$USE" --header-only
fi
if [ "$USE" = "$OV" ]; then
  if [ "$PLAN" = 1 ]; then REHASH=; else REHASH=--rehash; fi
  "$PY" "$REPO/conv/engram_codes.py" --check-verified "$OV" $REHASH ||
    { echo "conv-dfly: $OV is not a full, verified overlay; nothing converted" >&2; exit 1; }
fi

if [ "$PLAN" = 0 ]; then
  # the drafter appended: 0.75 GiB of IQ8R linears with DFLY_I8=1, about 1.6 GiB all in bf16
  F=$(df --output=avail -B1G "$(dirname "$C")" | tail -1 | tr -d ' ')
  [ "$F" -ge "$NEED_GB" ] || { echo "conv-dfly: $F GB free beside $C, $NEED_GB needed for the new planes" >&2; exit 1; }
fi

if [ "$PLAN" = 0 ] && [ -e "$C.pre-append" ]; then
  B="$C.pre-append.$TAG.$(date +%Y%m%d-%H%M%S)"
  [ -e "$B" ] && { echo "conv-dfly: $B already exists" >&2; exit 1; }
  cp --no-clobber --preserve=all "$C.pre-append" "$B"
  cmp -s "$C.pre-append" "$B" || { echo "conv-dfly: the copy $B differs from $C.pre-append" >&2; exit 1; }
  echo "conv-dfly: $C.pre-append copied to $B"
fi

exec sudo docker run --rm -u "$(id -u):$(id -g)" --security-opt seccomp=unconfined \
  -e RADIANCE_HOME=/hy4home:/opt/radiance/share/radiance -e CALIB=/calib -e HY4_MLA=both \
  -v "$PLUGINS:/hy4home:ro" -v "$REPO/data:/hy4data:ro" \
  -v "$DIR:/m" -v "$(dirname "$C"):/models" -v "$CKROOT:/ckpt:ro" -v "$CALIB:/calib:ro" -v "$OVDIR:/ov:ro" \
  --entrypoint /opt/radiance/bin/rad-convert "$IMAGE" "$CKPT" \
  --chat-template /hy4data/chat_template.jinja --recipe "/hy4data/recipes/$RECIPE" \
  --reuse "/models/$(basename "$C")" --in-place \
  --add /m/full-consts/engram-consts.safetensors --add "/ov/$(basename "$USE")" \
  --max-spec 5 --max-ctx 65536 --set hy4.trunk_i8=1 --set hy4.engram_codes=rowmax448 "$@"
