#!/bin/bash
# A clean copy of the served container, for publishing: the same weights (GPTQ experts, IQ8R
# trunk, IQ8R DFly2 drafter, the Engram tables as conv/engram_codes.py's E4M3 codes), without the
# planes the in-place runs left unreachable, without the expanded kv_b the absorbed MLA does not read
# (HY4_MLA=absorbed), and with the current chat template (data/chat_template.jinja: Tencent's of
# 2026-11-19, `.format()` rewritten). Every weight is copied from the served container (--reuse, no
# --in-place): nothing is quantised again, so its bits are the served ones.
# Run on the machine that holds the served container; writes a NEW file in a directory of its own
# ($OUTDIR; a local disk, since the output is 150.6 GB). Usage: conv-clean.sh [--plan-only] [-v]
#
# The run is the served container's last one (deploy/set-engram-key.sh) to a new file: the same
# checkpoint, recipe, extra sources and keys, so every weight plans as the container holds it.
# - The codes overlay ($OV) goes in as one more source (--add): the plugin's name map takes the
#   codes when they are there (hyv4flash.cpp, decl_engram) and rad-convert keeps them as
#   `fp8_e4m3*bf16[*x*]`; without it the tables would plan as the checkpoint's bf16, which the
#   container does not hold, and be written again as bf16 (126.7 GB, and not what is served).
# - --set hy4.engram_codes=rowmax448: decl_engram refuses an E4M3 table without the key, and a
#   --set key is seen by declare and written into the container.
# The overlay must be the full, verified one: `engram_codes.py --check-verified` gates it (with
# --rehash, a read of 63.4 GB, before a real run; by size and mtime for a plan). With --plan-only
# and no overlay, the header-only one stands in, as in conv-engram-e4m3.sh; a real run never takes it.
#
# Paths (each a variable; the defaults are under $HOME):
#   HY4_CONTAINER    the served container, the source     ($HOME/models/rad/hy4-flash-w4nl64-gptq.rad)
#   HY4_OUTDIR       where the clean copy is written      ($HOME/models/rad-pub)
#   HY4_CKPT         the tencent/Hy4-flash checkpoint: a snapshot directory of a Hugging Face cache
#                    (its blobs/ are mounted with it) or a plain directory
#                    ($HOME/.cache/huggingface/hub/models--tencent--Hy4-flash/snapshots/<revision>)
#   HY4_DIR          the working directory: holds full-consts/engram-consts.safetensors, and the
#                    log goes there                       ($HOME/hy4)
#   HY4_CALIB        the GPTQ Grams the recipe names      ($HY4_DIR/calib-gptq)
#   HY4_OVERLAY_DIR  where the Engram codes overlay lives ($HOME/models/engram-e4m3)
#   HY4_REPO         this repository: conv/, data/, tools/ (the checkout this script is in)
#   HY4_PLUGINS      a radiance_home with architectures/ and kernels/, built against the image's
#                    radiance                             ($HY4_REPO/build-hip-22/radiance_home)
#   HY4_IMAGE        the radiance image                   (stilldeadcode/radiance:2.2.0)
#   PY               a python3 with torch, numpy and safetensors ($HY4_REPO/tools/py-docker.sh)
# The served container is mounted read-only, as a single file, and the only writable mount is
# $OUTDIR, which is not the served container's directory: nothing inside the run can write to the
# served container. The run is fixed: only --plan-only and -v pass through to rad-convert.
#
# --plan-only stops before rad-convert matches the weights against the --reuse container, so the
# plan does not say what is copied; a real run does, in its last line, and this script reads it
# from the log it keeps ($LOG): anything but `the other 0 were written` means some weight was
# converted again instead of copied, and the script says so and exits 1 (the output is left there:
# move it aside, it is not the served weights). It then counts the output's weights with rad-info
# (its directory section): $WEIGHTS, the served container's 43630 less the 42 expanded attn_kv_b.
set -eu
REPO=${HY4_REPO:-$(cd "$(dirname "$0")/.." && pwd)}
IMAGE=${HY4_IMAGE:-stilldeadcode/radiance:2.2.0}
PLUGINS=${HY4_PLUGINS:-$REPO/build-hip-22/radiance_home}
PY=${PY:-$REPO/tools/py-docker.sh}
DIR=${HY4_DIR:-$HOME/hy4}
CALIB=${HY4_CALIB:-$DIR/calib-gptq}
SRC=${HY4_CONTAINER:-$HOME/models/rad/hy4-flash-w4nl64-gptq.rad}
OUTDIR=${HY4_OUTDIR:-$HOME/models/rad-pub}
OUT=$OUTDIR/hy4-flash-w4nl64-gptq-iq8r-dfly2-e4m3.rad
OLDOUT=$(dirname "$SRC")/$(basename "$OUT")   # where the first run of this script wrote it
WEIGHTS=43588
S=${HY4_CKPT:-$HOME/.cache/huggingface/hub/models--tencent--Hy4-flash/snapshots/d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959}
OVDIR=${HY4_OVERLAY_DIR:-$HOME/models/engram-e4m3}
OV=$OVDIR/hy4-flash-engram-e4m3.safetensors
RECIPE=hy4flash-w4nl64-gptq-iq8r-dfly.recipe
NEED_GB=165   # the output, 150.6 GB (the plan's 140.21 GiB of weights), and a margin

die() { echo "conv-clean: $*" >&2; exit 1; }

PLAN=0
for a in "$@"; do
  case "$a" in
    --plan-only) PLAN=1 ;;
    -v|--verbose) ;;
    *) die "no '$a' here: the run is fixed (a new file, every weight copied from $SRC); only --plan-only and -v" ;;
  esac
done

[ -e "$OUT" ] && die "$OUT already exists"
[ -e "$OLDOUT" ] && die "$OLDOUT already exists (an earlier run's output): move it to $OUTDIR or aside"
[ -e "$SRC" ] || die "no container at $SRC (set HY4_CONTAINER)"
[ -d "$S" ] || die "no checkpoint at $S (set HY4_CKPT)"
[ -d "$DIR" ] || die "no working directory at $DIR (set HY4_DIR)"
[ -d "$CALIB" ] || die "no Grams directory at $CALIB (set HY4_CALIB)"
# A snapshot of a Hugging Face cache is symlinks into ../../blobs: the model's cache directory is
# mounted whole, so they resolve inside the container.
S=$(cd "$S" && pwd)
CKROOT=$(cd "$S/../.." && pwd)
[ -d "$CKROOT/blobs" ] || CKROOT=$S
CKPT=/ckpt${S#"$CKROOT"}
# what $PY has to see when it is tools/py-docker.sh (mounted at the same paths)
export HY4_PY_RO="${HY4_PY_RO:+$HY4_PY_RO:}$CKROOT:$REPO" HY4_PY_RW="${HY4_PY_RW:+$HY4_PY_RW:}$OVDIR"
[ -f "$PLUGINS/architectures/hyv4flash.so" ] || die "no plugins at $PLUGINS (set HY4_PLUGINS)"
[ -f "$REPO/data/recipes/$RECIPE" ] || die "no recipe under $REPO/data (set HY4_REPO)"
[ -x "$PY" ] || die "no python at $PY (set PY)"

USE=$OV
if [ ! -e "$OV" ]; then
  [ "$PLAN" = 1 ] || die "no codes overlay at $OV: conv/engram_codes.py writes it (deploy/conv-engram-e4m3.sh)"
  USE=$OVDIR/hy4-flash-engram-e4m3.header-only.safetensors
  mkdir -p "$OVDIR"
  [ -e "$USE" ] || "$PY" "$REPO/conv/engram_codes.py" "$S" "$USE" --header-only
fi
if [ "$USE" = "$OV" ]; then
  if [ "$PLAN" = 1 ]; then REHASH=; else REHASH=--rehash; fi
  "$PY" "$REPO/conv/engram_codes.py" --check-verified "$OV" $REHASH ||
    die "$OV is not a full, verified overlay; nothing converted"
fi
mkdir -p "$OUTDIR"   # ours; docker would create a missing mount source as root
if [ "$PLAN" = 0 ]; then
  F=$(df --output=avail -B1G "$OUTDIR" | tail -1 | tr -d ' ')
  [ "$F" -ge "$NEED_GB" ] || die "$F GB free under $OUTDIR, $NEED_GB needed"
fi

run() {
  sudo docker run --rm -u "$(id -u):$(id -g)" --security-opt seccomp=unconfined \
    -e RADIANCE_HOME=/hy4home:/opt/radiance/share/radiance -e CALIB=/calib -e HY4_MLA=absorbed \
    -v "$PLUGINS:/hy4home:ro" -v "$REPO/data:/hy4data:ro" \
    -v "$DIR:/m:ro" -v "$OUTDIR:/out" -v "$SRC:/models/$(basename "$SRC"):ro" \
    -v "$CKROOT:/ckpt:ro" -v "$CALIB:/calib:ro" -v "$OVDIR:/ov:ro" \
    --entrypoint /opt/radiance/bin/rad-convert "$IMAGE" "$CKPT" \
    --chat-template /hy4data/chat_template.jinja --recipe "/hy4data/recipes/$RECIPE" \
    --reuse "/models/$(basename "$SRC")" -o "/out/$(basename "$OUT")" \
    --add /m/full-consts/engram-consts.safetensors --add "/ov/$(basename "$USE")" \
    --max-spec 5 --max-ctx 65536 --set hy4.trunk_i8=1 --set hy4.engram_codes=rowmax448 "$@"
}
if [ "$PLAN" = 1 ]; then run "$@"; exit; fi

LOG=${LOG:-$DIR/conv-clean.$(date +%Y%m%d-%H%M%S).log}
[ -e "$LOG" ] && die "$LOG already exists"
set -o pipefail
run "$@" 2>&1 | tee "$LOG"
# rad-convert's closing summary: the last `--reuse:` line of the log, not any line of it
grep -- '--reuse: ' "$LOG" | tail -1 |
  grep -q -- '--reuse: [0-9]* weight(s), .* copied from .*; the other 0 were written$' ||
  die "the log ($LOG) does not say every weight was copied from $SRC: $OUT is not the served weights, move it aside"
N=$(sudo docker run --rm -u "$(id -u):$(id -g)" -v "$OUT:/out.rad:ro" \
      --entrypoint /opt/radiance/bin/rad-info "$IMAGE" /out.rad | awk '$1 == "directory" { print $NF; exit }')
[ "$N" = "$WEIGHTS" ] || die "$OUT holds '$N' weights, not $WEIGHTS: move it aside"
echo "conv-clean: the $N weights of $OUT were all copied from $SRC (log: $LOG)"
