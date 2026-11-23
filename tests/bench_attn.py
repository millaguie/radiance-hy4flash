#!/usr/bin/env python3
"""Times hy4_mla_attn on the card at the geometry of a Hy4 prefill chunk. HY4_MLA_ATTN picks the form
(unset: form 3 above 32 rows; =4 form 3 with the score split in four column quarters; =2 form 2; =1 form 1). Usage: HY4_MLA_ATTN=2 python3 tests/bench_attn.py
--decode: Hy4's decode geometry (1-32 rows; 16 heads over a 2048-key selection, 32 heads over a 512
window at 4000 of context), the per-head form 2 against form 2g at several slice counts, each call
with the scratch its hook asks for, timed over a graph-free loop of 200 calls."""
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


def measure(pl, ins, outs, p, n=10):
    de = [Dev(t) if t is not None else None for t in ins]
    ds = [Dev(t) for t in outs]
    nb = pl.scratch_bytes("hy4_dev_mla_attn", de + ds, p)
    sc = Dev(torch.zeros(max(nb, 4), dtype=torch.uint8)) if nb else None
    scr = (sc.p, nb) if sc else None
    assert pl.call("hy4_dev_mla_attn", de + ds, p, scratch=scr) == 0
    hip.hipDeviceSynchronize()
    t = time.perf_counter()
    for _ in range(n):
        pl.call("hy4_dev_mla_attn", de + ds, p, scratch=scr)
    hip.hipDeviceSynchronize()
    return (time.perf_counter() - t) / n * 1e3


def decode(pl):
    I32, L, R, bs, P = torch.int32, 512, 64, 4, 3000
    cache = bf(torch.randn(P, 1, bs, L + R))
    forms = [("per-head form 2", {"mla_form": 5})] + [(f"2g, {S} slices", {"mla_form": 6, "mla_splits": S}) for S in (4, 8, 16, 32)] + \
            [("2g, default slices", {"mla_form": 6})]
    for rows in (1, 4, 8, 32):
        sel = torch.stack([torch.randperm(P)[:512] for _ in range(rows)]).to(I32)
        bt = torch.randint(0, P, (rows, 1000), dtype=I32)
        geo = [("16 heads, selection 2048", 16, 0, [sel, torch.full((rows,), 2048, dtype=I32), None]),
               ("32 heads, window 512", 32, 512, [bt, torch.full((rows,), 4000, dtype=I32),
                                                  torch.arange(rows + 1, dtype=I32)])]
        for name, H, w, extra in geo:
            q, qr = bf(torch.randn(rows, H, L) / 8), bf(torch.randn(rows, H, R) / 8)
            line = []
            for fname, kw in forms:
                p = {"M": rows, "heads": H, "latent": L, "rope": R, "block_size": bs, "window": w, "scale": 1 / 16,
                     "dtype": "bf16", **kw}
                us = measure(pl, [q, qr, cache] + extra + [torch.randn(H)], [bf(torch.zeros(rows, H, L))], p, n=200) * 1e3
                line.append(f"{fname} {us:.1f}")
            print(f"{rows:>2} rows, {name}: " + " | ".join(line) + " us", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", default=str(ROOT / "build-hip-22/radiance_home/kernels/hy4k.so"))
    ap.add_argument("--decode", action="store_true", help="decode geometry: per-head form 2 against form 2g")
    args = ap.parse_args()
    pl = RC.Plugin(Path(args.so))
    if args.decode:
        decode(pl)
        return
    I32, L, R, bs = torch.int32, 512, 64, 4
    P = 3000
    cache = bf(torch.randn(P, 1, bs, L + R))
    # window: 2048 rows, 32 heads, window 512, context 6000
    T, H, ctx = 2048, 32, 6000
    bt = torch.randperm(P)[:1500].view(1, 1500).to(I32)
    p = {"M": T, "heads": H, "latent": L, "rope": R, "block_size": bs, "window": 512, "scale": 1 / 16, "dtype": "bf16"}
    ms = measure(pl, [bf(torch.randn(T, H, L) / 8), bf(torch.randn(T, H, R) / 8), cache, bt, torch.tensor([ctx], dtype=I32),
                   torch.tensor([0, T], dtype=I32), torch.randn(H)], [bf(torch.zeros(T, H, L))], p)
    print(f"window 512, 2048 rows x 32 heads: {ms:.2f} ms")
    for T2, w2, H2 in ((1024, 512, 32), (2048, 256, 32), (2048, 512, 16), (2048, 64, 32)):
        p2 = {**p, "M": T2, "window": w2, "heads": H2}
        ms = measure(pl, [bf(torch.randn(T2, H2, L) / 8), bf(torch.randn(T2, H2, R) / 8), cache, bt,
                       torch.tensor([ctx], dtype=I32), torch.tensor([0, T2], dtype=I32), torch.randn(H2)],
                  [bf(torch.zeros(T2, H2, L))], p2)
        print(f"  {T2} rows, window {w2}, {H2} heads: {ms:.2f} ms")
    # selection: 256 rows, 16 heads, 2048 keys a row
    T, H = 256, 16
    sel = torch.stack([torch.randperm(P)[:512] for _ in range(T)]).to(I32)
    p = {**p, "M": T, "heads": H, "window": 0}
    ms = measure(pl, [bf(torch.randn(T, H, L) / 8), bf(torch.randn(T, H, R) / 8), cache, sel,
                   torch.full((T,), 2048, dtype=I32), None, torch.randn(H)], [bf(torch.zeros(T, H, L))], p)
    print(f"selection, 256 rows x 16 heads x 2048 keys: {ms:.2f} ms")


if __name__ == "__main__":
    main()
