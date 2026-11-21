#!/bin/bash
# Rewrites the four Engram tables of the served GPTQ container, in place, as the E4M3 codes Engram
# reads them as (conv/engram_codes.py): 63.4 GB in place of 126.7 GB of bf16, the same rows to the
# norm's eps, so the first pass of a new prompt reads half as much from the drive. Run on the
# machine that holds the container.
# Usage: conv-engram-e4m3.sh [--plan-only] [-v]
#
# 1. conv/engram_codes.py writes the codes from the checkpoint to $OV (63.4 GB),
#    as a sharded FP8 table under one bf16 scale of 1 a layer, checking every row as it goes, and
#    renames it from .partial and writes $OV.verified only after reading it back. An $OV already
#    there is used only if `engram_codes.py --check-verified` accepts it (the marker matches its size
#    and mtime, and its header is a full run's); anything else stops the script, and a stale $OV is
#    moved aside by hand. With --plan-only and no $OV, a header-only one (sparse, its codes zeros) is
#    written beside it instead, for the plan alone: its name (.header-only.safetensors) is one
#    --check-verified refuses, so a real run never takes it.
# 2. rad-convert --in-place over the container with $OV as one more source (--add), with
#    --set hy4.engram_codes=rowmax448: the key that tells the plugin the E4M3 tables are these codes
#    (decl_engram refuses an E4M3 table without it, and the key over any other table). The plugin's
#    name map takes the codes when they are there (hyv4flash.cpp, decl_engram), rad-convert keeps
#    them byte for byte as `fp8_e4m3*bf16[*x*]` (no recipe rule names the table), and
#    --replace 'blk.*.engram_table.weight' writes the four tables again past the end; every other
#    weight is kept. The recipe is the one the container was last extended with (a kept weight
#    must match its rule), unchanged.
#
# The image and the plugins are radiance 2.2.0's (stilldeadcode/radiance:2.2.0 and this checkout's
# build-hip-22/radiance_home, which tools/build-hip.sh writes), and can be overridden: HY4_IMAGE,
# HY4_PLUGINS (built against the image's radiance: a plugin of another ABI is refused at load),
# HY4_REPO (holds conv/, data/ and tools/). The plugin that serves the
# result must be one that reads the E4M3 table (an older plugin declares the table bf16 and refuses
# the container). PY is a python3 with torch, numpy and safetensors (tools/py-docker.sh runs one
# from an image).
#
# Disk: $OV 63.4 GB, then 63.4 GB more on the container (59.0 GiB of new planes; the plan reports
# the whole container, not what --in-place appends). The old bf16 planes (126.7 GB) stay in the
# file as unreachable bytes: `rad-info --unreferenced` lists them, and `fallocate --punch-hole`
# over those ranges gives the disk back. $OV is not needed once the container holds the codes.
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
OV=$OVDIR/hy4-flash-engram-e4m3.safetensors
RECIPE=hy4flash-w4nl64-gptq-iq8r-dfly.recipe
TAG=before-engram-e4m3
NEED_GB=130   # $OV and the appended planes, 126.8 GB, and a margin

PLAN=0
for a in "$@"; do
  case "$a" in
    --plan-only) PLAN=1 ;;
    -v|--verbose) ;;
    *) echo "conv-engram-e4m3: no '$a' here: the run on the served container is fixed; only --plan-only and -v" >&2; exit 1 ;;
  esac
done

[ -e "$C" ] || { echo "conv-engram-e4m3: no container at $C (set HY4_CONTAINER)" >&2; exit 1; }
[ -d "$S" ] || { echo "conv-engram-e4m3: no checkpoint at $S (set HY4_CKPT)" >&2; exit 1; }
[ -d "$DIR" ] || { echo "conv-engram-e4m3: no working directory at $DIR (set HY4_DIR)" >&2; exit 1; }
[ -d "$CALIB" ] || { echo "conv-engram-e4m3: no Grams directory at $CALIB (set HY4_CALIB)" >&2; exit 1; }
# A snapshot of a Hugging Face cache is symlinks into ../../blobs: the model's cache directory is
# mounted whole, so they resolve inside the container.
S=$(cd "$S" && pwd)
CKROOT=$(cd "$S/../.." && pwd)
[ -d "$CKROOT/blobs" ] || CKROOT=$S
CKPT=/ckpt${S#"$CKROOT"}
# what $PY has to see when it is tools/py-docker.sh (mounted at the same paths)
export HY4_PY_RO="${HY4_PY_RO:+$HY4_PY_RO:}$CKROOT:$REPO" HY4_PY_RW="${HY4_PY_RW:+$HY4_PY_RW:}$OVDIR"
[ -x "$PY" ] || { echo "conv-engram-e4m3: no python at $PY (set PY)" >&2; exit 1; }
[ -f "$PLUGINS/architectures/hyv4flash.so" ] || { echo "conv-engram-e4m3: no plugins at $PLUGINS (set HY4_PLUGINS)" >&2; exit 1; }
[ -f "$REPO/data/recipes/$RECIPE" ] || { echo "conv-engram-e4m3: no recipe under $REPO/data (set HY4_REPO)" >&2; exit 1; }
mkdir -p "$OVDIR"

free_gb() { df --output=avail -B1G "$1" | tail -1 | tr -d ' '; }

USE=$OV
if [ ! -e "$OV" ]; then
  if [ "$PLAN" = 1 ]; then
    USE=$OVDIR/hy4-flash-engram-e4m3.header-only.safetensors
    [ -e "$USE" ] || "$PY" "$REPO/conv/engram_codes.py" "$S" "$USE" --header-only
  else
    for f in "$OV.partial" "$OV.verified"; do
      [ -e "$f" ] && { echo "conv-engram-e4m3: $f is there without $OV: a failed run's, move it aside" >&2; exit 1; }
    done
    F=$(free_gb "$OVDIR")
    [ "$F" -ge "$NEED_GB" ] || { echo "conv-engram-e4m3: $F GB free under $OVDIR, $NEED_GB needed" >&2; exit 1; }
    "$PY" "$REPO/conv/engram_codes.py" "$S" "$OV"
  fi
fi
if [ "$USE" = "$OV" ]; then
  if [ "$PLAN" = 1 ]; then REHASH=; else REHASH=--rehash; fi   # a plan checks size and mtime only
  "$PY" "$REPO/conv/engram_codes.py" --check-verified "$OV" $REHASH ||
    { echo "conv-engram-e4m3: $OV is not a full, verified overlay; nothing converted" >&2; exit 1; }
fi
if [ "$PLAN" = 0 ]; then
  F=$(free_gb "$(dirname "$C")")
  [ "$F" -ge 65 ] || { echo "conv-engram-e4m3: $F GB free beside $C, 65 needed for the new planes" >&2; exit 1; }
fi

if [ "$PLAN" = 0 ] && [ -e "$C.pre-append" ]; then
  B="$C.pre-append.$TAG.$(date +%Y%m%d-%H%M%S)"
  [ -e "$B" ] && { echo "conv-engram-e4m3: $B already exists" >&2; exit 1; }
  cp --no-clobber --preserve=all "$C.pre-append" "$B"
  cmp -s "$C.pre-append" "$B" || { echo "conv-engram-e4m3: the copy $B differs from $C.pre-append" >&2; exit 1; }
  echo "conv-engram-e4m3: $C.pre-append copied to $B"
fi

exec sudo docker run --rm -u "$(id -u):$(id -g)" --security-opt seccomp=unconfined \
  -e RADIANCE_HOME=/hy4home:/opt/radiance/share/radiance -e CALIB=/calib -e HY4_MLA=both \
  -v "$PLUGINS:/hy4home:ro" -v "$REPO/data:/hy4data:ro" \
  -v "$DIR:/m" -v "$(dirname "$C"):/models" -v "$CKROOT:/ckpt:ro" -v "$CALIB:/calib:ro" -v "$OVDIR:/ov:ro" \
  --entrypoint /opt/radiance/bin/rad-convert "$IMAGE" "$CKPT" \
  --chat-template /hy4data/chat_template.jinja --recipe "/hy4data/recipes/$RECIPE" \
  --reuse "/models/$(basename "$C")" --in-place --replace 'blk.*.engram_table.weight' \
  --add /m/full-consts/engram-consts.safetensors --add "/ov/$(basename "$USE")" \
  --max-spec 5 --max-ctx 65536 --set hy4.trunk_i8=1 --set hy4.engram_codes=rowmax448 "$@"
