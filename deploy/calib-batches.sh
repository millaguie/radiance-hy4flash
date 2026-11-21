#!/bin/bash
# Plan step 9: records the GPTQ Grams of the whole model, in batches of layers (what a batch
# accumulates has to fit in RAM). Each batch starts the RTN with the tap on its layers, sends it the corpus
# (send_calib.py) until the tap writes, stops the server and moves the Grams to $HY4_CALIB.
# Usage (on the GPU machine, with nothing else on the cards):
#   sudo nohup ./calib-batches.sh [batches...] > calib.log 2>&1 &
# With no arguments, all of them: MoE layers 1-41 in groups of 4.
# Paths: HY4_DIR (default $HOME/hy4) is the working directory: it holds run-hy4.sh, send_calib.py
# (ref/), hy4-calib.yaml and corpus-calib/calib.jsonl (ref/corpus_calib.py), and the logs go there.
# HY4_CALIB (default $HY4_DIR/calib-gptq) is where the Grams end up. run-hy4.sh reads HY4_DIR,
# HY4_MODELS, HY4_IMAGE and HY4_LIBR4D.
set -u
# The scripts here are run with sudo: the defaults are under the home of the user who called sudo.
H=$(getent passwd "${SUDO_USER:-$(id -un)}" | cut -d: -f6)
export HY4_DIR=${HY4_DIR:-$H/hy4}
cd "$HY4_DIR" || exit 1
BATCHES=("$@")
# In groups of 4: with 5 the RAM peak was 55.5 of 61 GB (~2.4 GB accumulated per layer plus the 39 GB pool).
[ ${#BATCHES[@]} -eq 0 ] && BATCHES=(1-4 5-8 9-12 13-16 17-20 21-24 25-28 29-32 33-36 37-40 41-41)
# A single copy: another one still running would remove this one's container when it finished.
exec 9>/run/lock/calib-batches.lock
flock -n 9 || { echo "another calib-batches.sh is already running"; exit 1; }
CALIB=${HY4_CALIB:-$HY4_DIR/calib-gptq}
TOK=${TOK:-262144}
mkdir -p "$CALIB"
for T in "${BATCHES[@]}"; do
  echo "=== batch $T: $(date)"
  mkdir -p calib
  if ls calib/gram.* >/dev/null 2>&1; then echo "calib/ is not empty; stopping"; exit 1; fi
  RADIANCE_CALIB_DIR=/hy4/calib RADIANCE_CALIB_LAYERS=$T RADIANCE_CALIB_TOKENS=$TOK \
    CFG=hy4-calib.yaml ./run-hy4.sh 0,1 /hy4/nodump >/dev/null
  CID=$(docker inspect -f '{{.Id}}' hy4-test)
  docker logs -f hy4-test > "calib-$T.log" 2>&1 &
  LP=$!
  for i in $(seq 300); do
    grep -qE 'listening|startup stopped' "calib-$T.log" 2>/dev/null && break
    docker ps -q -f name=hy4-test | grep -q . || break
    sleep 5
  done
  if ! grep -q listening "calib-$T.log"; then echo "it did not start"; grep -E '^(E|W) ' "calib-$T.log" | tail -5; kill $LP; exit 1; fi
  python3 send_calib.py --corpus corpus-calib/calib.jsonl --tokens $((TOK + 40000)) --url http://127.0.0.1:18093 \
    --log "calib-$T.log" | tail -2
  for i in $(seq 120); do
    grep -q 'calibration wrote' "calib-$T.log" && break
    [ "$(docker inspect -f '{{.State.Running}}' "$CID" 2>/dev/null)" = true ] || break
    sleep 5
  done
  grep 'calibration wrote' "calib-$T.log" | tail -1
  echo "container: $(docker inspect -f 'running={{.State.Running}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}} {{.State.Error}}' "$CID" 2>&1)"
  free -g | head -2
  docker rm -f "$CID" >/dev/null; kill $LP 2>/dev/null
  n=$(ls calib | wc -l); echo "Grams: $n files, $(du -sh calib | cut -f1)"
  [ "$n" -gt 0 ] || { echo "the tap did not write"; exit 1; }
  rsync -a --remove-source-files calib/ "$CALIB/" && echo "moved to $CALIB"
done
echo "=== end: $(date)"
