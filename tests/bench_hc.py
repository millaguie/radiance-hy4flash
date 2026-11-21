#!/usr/bin/env python3
"""Times hy4_hc_pre on the card at Hy4's geometry (bf16 streams, n 3072, hc 4, 8 products), the form
before the split (hc_pre1_k, `hc_form` 1 = HY4_HC_PRE=old) against the split (hc_pre_part_k +
hc_pre_fin_k, `hc_form` 2) and the default by row count, at 1, 8, 48 and 2048 rows. The launches
are recorded into a HIP graph, as the engine replays a step, so the time is the card's and not
Python's. --rows-a-wave also times the split's rows-a-wave choices (`hc_r`).
Usage: python3 tests/bench_hc.py [--so ...] [--rows-a-wave]"""
import argparse
import ctypes as C
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import rad_ctypes as RC  # noqa: E402
from test_dev import Dev, ROOT, bf, hip  # noqa: E402

hip.hipDeviceSynchronize.argtypes = []
hip.hipStreamCreate.argtypes = [C.POINTER(C.c_void_p)]
hip.hipStreamBeginCapture.argtypes = [C.c_void_p, C.c_int]
hip.hipStreamEndCapture.argtypes = [C.c_void_p, C.POINTER(C.c_void_p)]
hip.hipGraphInstantiate.argtypes = [C.POINTER(C.c_void_p), C.c_void_p, C.c_void_p, C.c_void_p, C.c_size_t]
hip.hipGraphLaunch.argtypes = [C.c_void_p, C.c_void_p]
hip.hipStreamSynchronize.argtypes = [C.c_void_p]
hip.hipGraphExecDestroy.argtypes = [C.c_void_p]
hip.hipGraphDestroy.argtypes = [C.c_void_p]

N, HC = 3072, 4
PER_GRAPH, REPS = 100, 20


def measure(pl, stream, rows, form, r=None):
    """Microseconds a call: PER_GRAPH calls in a graph, REPS replays, best of 3."""
    h = Dev(bf(torch.randn(rows, HC * N)))
    fn = Dev(torch.randn(2 * HC, HC * N) * 0.01)
    base, scale = Dev(torch.randn(2 * HC) * 0.5), Dev(torch.tensor([0.7, 0.9]))
    x, post = Dev(bf(torch.zeros(rows, N))), Dev(torch.zeros(rows, HC))
    p = {"M": rows, "n": N, "hc": HC, "eps": 1e-5, "hc_eps": 1e-6, "magnitude": 2.0, "dtype": "bf16",
         "hc_form": form}
    if r:
        p["hc_r"] = r
    ts = [h, fn, base, scale, x, post]
    nb = pl.scratch_bytes("hy4_dev_hc_pre", ts, p)
    sc = Dev(torch.zeros(max(nb, 4), dtype=torch.uint8))
    a, keep = pl.args(ts, p, (sc.p, nb) if nb else None)
    k = pl.kernels["hy4_dev_hc_pre"]
    assert k.launch(C.byref(a), stream) == 0
    assert hip.hipStreamSynchronize(stream) == 0
    g, ex = C.c_void_p(), C.c_void_p()
    assert hip.hipStreamBeginCapture(stream, 0) == 0
    for _ in range(PER_GRAPH):
        k.launch(C.byref(a), stream)
    assert hip.hipStreamEndCapture(stream, C.byref(g)) == 0
    assert hip.hipGraphInstantiate(C.byref(ex), g, None, None, 0) == 0
    best = float("inf")
    for _ in range(3):
        hip.hipGraphLaunch(ex, stream)
        hip.hipStreamSynchronize(stream)
        t = time.perf_counter()
        for _ in range(REPS):
            hip.hipGraphLaunch(ex, stream)
        hip.hipStreamSynchronize(stream)
        best = min(best, (time.perf_counter() - t) / (REPS * PER_GRAPH) * 1e6)
    hip.hipGraphExecDestroy(ex)
    hip.hipGraphDestroy(g)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", default=str(ROOT / "build-hip2/radiance_home/kernels/hy4k.so"))
    ap.add_argument("--rows-a-wave", action="store_true")
    a = ap.parse_args()
    pl = RC.Plugin(Path(a.so))
    stream = C.c_void_p()
    assert hip.hipStreamCreate(C.byref(stream)) == 0
    torch.manual_seed(3)
    print(f"hy4_hc_pre, bf16, n {N}, hc {HC}: us a call (old = hc_pre1_k, split = part + fin)")
    for rows in (1, 8, 48, 2048):
        old, split, dflt = (measure(pl, stream, rows, f) for f in (1, 2, 0))
        print(f"  {rows:5d} rows: old {old:8.2f}  split {split:8.2f}  default {dflt:8.2f}  (old/split {old / split:.2f}x)")
        if a.rows_a_wave and rows != 2048:
            print("         split by rows a wave: " +
                  "  ".join(f"{r}: {measure(pl, stream, rows, 2, r):.2f}" for r in (1, 2, 4)))


if __name__ == "__main__":
    main()
