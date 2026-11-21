#!/bin/bash
# python3 with torch (float8), numpy and safetensors, from a container image, for a machine with no
# such Python of its own: conv/engram_codes.py needs one, and the deploy/conv-*.sh scripts call it
# through $PY, whose default is this script. Usage: tools/py-docker.sh script.py [arguments]
#
# The paths a run reads and writes are mounted at the same paths, so arguments pass unchanged:
#   - $HOME and the current directory, writable;
#   - HY4_PY_RO: directories mounted read-only, separated by ':' (a checkpoint outside $HOME);
#   - HY4_PY_RW: directories mounted writable, separated by ':' (an output directory outside $HOME).
# The deploy/conv-*.sh scripts set both for the paths they pass. A directory named there must exist:
# docker would create a missing one as root.
#   HY4_PY_IMAGE   an image whose python3 has torch (with float8_e4m3fn), numpy and safetensors
#                  (default magiccodingman/vllm-radiance:1.0.394, which holds the three)
#   HY4_DOCKER     how docker is called (default "sudo docker"; "docker" for a user in its group)
set -eu
IMAGE=${HY4_PY_IMAGE:-magiccodingman/vllm-radiance:1.0.394}
read -r -a DOCKER <<< "${HY4_DOCKER:-sudo docker}"
MOUNTS=(); declare -A SEEN=()
add() {   # add <list> <suffix>: each directory once, the first mention winning (docker refuses a
          # mount point given twice)
  local d; local IFS=:
  for d in $1; do
    [ -n "$d" ] || continue
    [ -d "$d" ] || { echo "py-docker: no directory $d" >&2; exit 1; }
    [ -n "${SEEN[$d]:-}" ] && continue
    SEEN[$d]=1; MOUNTS+=(-v "$d:$d$2")
  done
}
add "$HOME:$PWD:${HY4_PY_RW:-}" ""
add "${HY4_PY_RO:-}" :ro
exec "${DOCKER[@]}" run --rm -u "$(id -u):$(id -g)" "${MOUNTS[@]}" -w "$PWD" --entrypoint python3 "$IMAGE" "$@"
