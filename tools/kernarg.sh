#!/bin/sh
# Checks every kernel in a plugin against two limits the radiance backend enforces at dispatch and
# plain HIP does not, so tests/test_dev.py (which launches through HIP directly) cannot see them:
#   - 768 bytes of kernel arguments, hidden ones included ("its kernel argument segment is N bytes
#     and the backend's slot holds 768"); reading gridDim adds hidden arguments;
#   - 256 VGPRs or more: a kernel at the limit spills on the next compiler change;
#   - a private (scratch) segment: a kernel that spills registers to scratch is reported, since the
#     spill costs speed on the step path and the backend's scratch support is not ours to assume.
# Fails if a library is missing, holds no code object, or a code object holds no kernel, so an empty
# check cannot pass. Usage: tools/kernarg.sh [lib.so ...] (default: build-hip-22's plugins).
cd "$(dirname "$0")/.."
if [ $# -eq 0 ]; then
  for d in build-hip-22; do
    for f in $d/radiance_home/kernels/hy4k.so $d/radiance_home/architectures/hyv4flash.so; do
      [ -f "$f" ] && set -- "$@" "$f"
    done
  done
fi
[ $# -gt 0 ] || { echo "kernarg: no library to check" >&2; exit 1; }
exec docker run --rm -u "$(id -u):$(id -g)" -v "$PWD:/hy4" -w /hy4 radiance-build:2.2.0 sh -c '
L=/opt/rocm-repeatable/lib/llvm/bin; T=$(mktemp -d); fail=0
for so in "$@"; do
  [ -f "$so" ] || { echo "kernarg: $so: missing"; fail=1; continue; }
  # objcopy writes its output to /dev/null so the checked library is never rewritten
  if ! $L/llvm-objcopy --dump-section=.hip_fatbin=$T/fb.bin "$so" /dev/null 2>$T/err; then
    case "$so" in *architectures/*) continue ;; esac   # an arch plugin may carry no device code
    echo "kernarg: $so: no .hip_fatbin ($(cat $T/err))"; fail=1; continue
  fi
  # the fatbin concatenates one bundle per source file: split it at each code object ELF header
  n=0
  for o in $(grep -obUaP "\x7fELF\x02\x01\x01\x40" $T/fb.bin | cut -d: -f1); do
    n=$((n + 1)); dd if=$T/fb.bin of=$T/o$n.elf bs=1 skip=$o status=none
  done
  [ $n -gt 0 ] || { echo "kernarg: $so: no code object in the fatbin"; fail=1; continue; }
  k=0; max=0
  for f in $T/o*.elf; do
    $L/llvm-readelf --notes "$f" > $T/notes || { echo "kernarg: $so: llvm-readelf failed"; fail=1; continue; }
    tgt=$(awk "/amdhsa.target:/ { print \$2; exit }" $T/notes)   # printed after the kernels
    out=$(awk -v tgt="$tgt" "
      /\.kernarg_segment_size:/ { ka = \$2 }
      /\.private_segment_fixed_size:/ { pr = \$2 }
      /\.vgpr_count:/ { if (\$2 + 0 >= 256) { printf \"BAD %s %s: %d VGPRs, at the limit (the next live value spills)\\n\", tgt, nm, \$2; bad = 1 } }
      /\.name:/ { nm = \$2 }
      /\.symbol:/ { n++; if (ka + 0 > m) m = ka + 0
                    if (ka + 0 > 768) { printf \"BAD %s %s: %d bytes of arguments (> 768)\n\", tgt, nm, ka; bad = 1 }
                    if (pr + 0 > 0) { printf \"BAD %s %s: %d bytes of scratch (registers spill)\n\", tgt, nm, pr; bad = 1 }
                    ka = 0; pr = 0 }
      END { printf \"COUNT %d %d\n\", n, m; exit bad }" $T/notes) || fail=1
    echo "$out" | grep "^BAD" | sed "s|^BAD|kernarg: $so:|"
    c=$(echo "$out" | awk "/^COUNT/ { print \$2 }"); x=$(echo "$out" | awk "/^COUNT/ { print \$3 }")
    [ "$c" -gt 0 ] || { echo "kernarg: $so: a code object with no kernel"; fail=1; }
    k=$((k + c)); [ "$x" -gt "$max" ] && max=$x
  done
  echo "kernarg: $so: $n code objects, $k kernels, largest argument segment $max bytes"
  rm -f $T/o*.elf
done
exit $fail' sh "$@"
