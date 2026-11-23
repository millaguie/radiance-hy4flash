#!/usr/bin/env python3
"""hy4_mla_project on the card: the wide form (mla_proj2_k) against mla_proj_k, bit for bit, and their
times at Hy4's prefill geometry (head_dim 256, latent 512, 32 and 16 heads a rank) for 2048 and 4096
rows and a few ragged row counts. Each wide block shape (proj_wr x proj_wc waves) is checked and timed.
Usage: python3 tests/bench_proj.py [--so build-hip-22/radiance_home/kernels/hy4k.so]"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import rad_ctypes as RC  # noqa: E402
from test_dev import Dev, ROOT, bf, hip  # noqa: E402

hip.hipDeviceSynchronize.argtypes = []


def run(pl, x, w, M, H, mode, extra, n):
    L, d = 512, 256
    out = Dev(bf(torch.zeros(M, H * (L if mode == "absorb" else d))))
    p = {"M": M, "heads": H, "head_dim": d, "latent_dim": L, "mode": mode, "dtype": "bf16", **extra}
    assert pl.call("hy4_dev_mla_project", [x, w, out], p) == 0
    hip.hipDeviceSynchronize()
    t = time.perf_counter()
    for _ in range(n):
        pl.call("hy4_dev_mla_project", [x, w, out], p)
    hip.hipDeviceSynchronize()
    return (time.perf_counter() - t) / max(n, 1) * 1e3, out.to_host()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", default=str(ROOT / "build-hip-22/radiance_home/kernels/hy4k.so"))
    a = ap.parse_args()
    pl = RC.Plugin(Path(a.so))
    torch.manual_seed(11)
    shapes = [(2, 4, 0), (4, 4, 0), (4, 4, 1), (2, 4, 1), (8, 2, 0), (8, 2, 1)]
    fails = 0
    for H in (32, 16):
        w = Dev(bf(torch.randn(H * 2 * 256, 512) / 16))
        for M in (37, 300, 2048, 4096):
            n = 20 if M >= 2048 else 3
            for mode in ("absorb", "expand"):
                x = Dev(bf(torch.randn(M, H * (256 if mode == "absorb" else 512))))
                t_old, y_old = run(pl, x, w, M, H, mode, {"proj_form": 1}, n)
                line = [f"old {t_old:.3f}"]
                for wr, wc, hm in shapes:
                    t_new, y_new = run(pl, x, w, M, H, mode, {"proj_form": 2, "proj_wr": wr, "proj_wc": wc, "proj_hm": hm}, n)
                    same = torch.equal(y_old.view(torch.int16), y_new.view(torch.int16))
                    fails += not same
                    line.append(f"{wr}x{wc}{'h' if hm else ''} {t_new:.3f}{'' if same else ' DIFFERENT BITS'}")
                print(f"H {H:2d} M {M:4d} {mode:6s} ms: " + " | ".join(line), flush=True)
    print("FAIL" if fails else "OK: every wide shape gives mla_proj_k's bits")


if __name__ == "__main__":
    main()
