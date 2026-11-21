#!/bin/bash
# Absorbed against expanded MLA on the R9700s (a finding of the review of the absorbed MLA; Hy4-flash
# plan, "Absorbed MLA cache"). On the 3-layer trimmed model converted with HY4_MLA=both
# ($HY4_MODELS/recorte3-abs.rad), at tp 1 and tp 2:
#   1. KL against HF's FP32 of the trimmed model (kld-ref, $HY4_DIR/kld-fp32-recorte), both forms;
#   2. cache capacity (the `kv:` line at startup) with the same config;
#   3. speed: decode of 256 tokens after `medio` and prefill of `largo` (8448 tokens).
# Only with any other radiance on the cards stopped. Output in $HY4_DIR/mla-card/.
# Usage: sudo ./test-mla.sh (run-hy4.sh with no dump dir: the debug tap off)
# Paths: HY4_DIR (default $HOME/hy4) is the working directory: it holds run-hy4.sh, vel.py and the
# kld reference. run-hy4.sh reads HY4_DIR, HY4_MODELS, HY4_IMAGE and HY4_LIBR4D.
set -u
# The scripts here are run with sudo: the defaults are under the home of the user who called sudo.
H=$(getent passwd "${SUDO_USER:-$(id -un)}" | cut -d: -f6)
export HY4_DIR=${HY4_DIR:-$H/hy4}
DIR=$HY4_DIR
OUT=$DIR/mla-card
mkdir -p "$OUT"
cd "$DIR" || exit 1

wait_exit() {   # the kld-ref container exits on its own once it writes the json
  while docker ps --format '{{.Names}}' | grep -qx hy4-test; do sleep 5; done
}
wait_ready() {
  for _ in $(seq 1 180); do
    docker logs hy4-test 2>&1 | grep -q 'server: listening' && return 0
    docker ps --format '{{.Names}}' | grep -qx hy4-test || return 1
    sleep 5
  done
  return 1
}

for TP in 1 2; do
  CARDS=$([ $TP = 1 ] && echo 0 || echo 0,1)
  for MODE in absorbed expanded; do
    ET=tp$TP-$MODE
    # KL
    cat > "$DIR/mla-kld.yaml" <<EOF
server:
  kernels: hy4k,libr4d,libavx
  radiance-home: /hy4/home:/opt/radiance/share/radiance
models:
  recorte3:
    model: /models/recorte3-abs.rad
    tp: $TP
    max-model-len: 16384
    max-num-batched-tokens: 2048
    max-num-seqs: 4
    num-speculative-tokens: 0
    no-prefix-cache: true
    ngram-placement: disk
    kld-ref: /hy4/kld-fp32-recorte
    kld-out: /hy4/mla-card/kld-$ET.json
EOF
    HY4_MLA=$MODE CFG=mla-kld.yaml ./run-hy4.sh $CARDS >/dev/null
    wait_exit
    docker logs hy4-test > "$OUT/kld-$ET.log" 2>&1 || true
    # serve: capacity and speed
    cat > "$DIR/mla-srv.yaml" <<EOF
server:
  host: 127.0.0.1
  port: 18093
  kernels: hy4k,libr4d,libavx
  radiance-home: /hy4/home:/opt/radiance/share/radiance
memory:
  gpu-headroom-mib: 128
models:
  hy4-flash:
    model: /models/recorte3-abs.rad
    tp: $TP
    max-model-len: 65536
    max-num-batched-tokens: 2048
    max-num-seqs: 1
    num-speculative-tokens: 0
    no-prefix-cache: true
    ngram-placement: disk
EOF
    HY4_MLA=$MODE CFG=mla-srv.yaml ./run-hy4.sh $CARDS >/dev/null
    if wait_ready; then
      docker logs hy4-test 2>&1 | grep -E '^I kv:|MLA|vram_kv|kv cache' > "$OUT/kv-$ET.txt"
      ok=1
      for i in 1 2; do python3 vel.py medio 256 || ok=0; done > "$OUT/vel-$ET.txt" 2>&1
      python3 vel.py largo 1 >> "$OUT/vel-$ET.txt" 2>&1 || ok=0
      [ $ok = 1 ] || echo "FAILURE: vel.py did not finish in $ET (see srv-$ET.log)" | tee -a "$OUT/failures.txt"
    else
      echo "FAILURE: the server did not start in $ET" | tee -a "$OUT/failures.txt"
    fi
    docker logs hy4-test > "$OUT/srv-$ET.log" 2>&1   # always, also if it dies mid-test
    docker rm -f hy4-test >/dev/null 2>&1
  done
done
echo "done $(date +%H:%M)" > "$OUT/done.txt"
