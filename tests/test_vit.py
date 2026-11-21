#!/usr/bin/env python3
"""Vision tower ops (hy4_vit_compact, hy4_vit_pos, hy4_vit_newline) against the HF reference of
ref/vision_ref.py (~/hy4/vision-fp32).

The radiance processor (Hy4 family) gives the patches in merge×merge block order and with `merge`
extra columns per row; HF gives them in row order and without that column. Here the radiance input
is built from the HF patches and the test checks that:
- compact returns the HF patches in block order, without the column;
- patches · W + b + pos gives the input to the first HF block (`emb`);
- newline places the HF adapter rows and the line break where HF puts them (`out`).
Plus rejection cases (counts that do not add up, missing line-break column).

Usage: tests/test_vit.py [--ref ~/hy4/vision-fp32] [--snapshot S]
--snapshot defaults to the first snapshot of tencent/Hy4-flash in the Hugging Face cache under $HOME.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).parent))
import rad_ctypes as RC  # noqa: E402

I32 = torch.int32
MERGE, SIDE, DIM, WIDTH = 2, 128, 768, 1152
FAILURES = []


def report(what, ok, extra=""):
    print(("ok   " if ok else "FAIL "), what, extra)
    if not ok:
        FAILURES.append(what)


def l2(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / max(b.norm(), 1e-30))


def blocks(gh, gw):
    """(row, column) indices of a gh×gw grid in merge×merge block order."""
    out = []
    for bh in range(gh // MERGE):
        for bw in range(gw // MERGE):
            for mh in range(MERGE):
                for mw in range(MERGE):
                    out.append((bh * MERGE + mh, bw * MERGE + mw))
    return out


def radiance_input(pv, gh, gw):
    """Patches as the Hy4 family gives them: block order over the grid with the extra column
    (zeros), and their coordinates [4, P] (row, column, height, width with the column)."""
    gwe = gw + MERGE
    order = blocks(gh, gwe)
    pix = torch.zeros(len(order), DIM)
    crd = torch.zeros(4, len(order), dtype=I32)
    for k, (r, c) in enumerate(order):
        if c < gw:
            pix[k] = pv[r * gw + c]
        crd[:, k] = torch.tensor([r, c, gh, gwe], dtype=I32)
    return pix, crd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, default=Path.home() / "hy4/vision-fp32")
    ap.add_argument("--snapshot", type=Path, default=None)
    ap.add_argument("--consts", type=Path, default=Path.home() / "hy4/full-consts/engram-consts.safetensors")
    a = ap.parse_args()
    snap = a.snapshot or sorted((Path.home() / ".cache/huggingface/hub/models--tencent--Hy4-flash/snapshots").iterdir())[0]
    wmap = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]

    def weight(k):
        with safe_open(str(snap / wmap[k]), framework="pt") as f:
            return f.get_tensor(k).float()

    wp = weight("vit.embeddings.patch_embedding.weight").reshape(WIDTH, DIM)
    bp = weight("vit.embeddings.patch_embedding.bias")
    tab = weight("vit.embeddings.position_embedding.weight")
    nl = weight("vit.perceive.image_newline")
    pl = RC.Plugin(Path(__file__).resolve().parent.parent / "build/radiance_home/kernels/hy4k.so")

    # two images in one pass: the segments are chained
    pv_all, crd_all, cu, expected, emb_ref, out_ref = [], [], [0], [], [], []
    for name in ("pequena", "ancha"):
        t = load_file(str(a.ref / f"{name}.safetensors"))
        _, gh, gw = t["grid_thw"][0].tolist()
        pix, crd = radiance_input(t["pixel_values"], gh, gw)
        pv_all.append(pix)
        crd_all.append(crd)
        cu.append(cu[-1] + pix.shape[0])
        order = [r * gw + c for r, c in blocks(gh, gw)]
        expected.append(t["pixel_values"][order])
        emb_ref.append(t["emb"][order])
        out_ref.append(t["out"])
    pix = torch.cat(pv_all)
    crd = torch.cat(crd_all, 1).contiguous()
    cu = torch.tensor(cu, dtype=I32)
    Pc = sum(e.shape[0] for e in expected)

    pixc = torch.empty(Pc, DIM)
    crdc = torch.empty(Pc, 4, dtype=I32)
    cuc = torch.empty(3, dtype=I32)
    rc = pl.call("hy4_ref_vit_compact", [pix, crd, cu, pixc, crdc, cuc], {"M": pix.shape[0], "dim": DIM, "merge": MERGE})
    e = l2(pixc, torch.cat(expected))
    report("compact: the HF patches in block order", rc == 0 and e == 0, f"L2 {e:.1e}, cu_c {cuc.tolist()}")

    x = pixc @ wp.T + bp
    rc = pl.call("hy4_ref_vit_pos", [x, tab, crdc], {"M": Pc, "n": WIDTH, "side": SIDE, "dtype": "f32"})
    e = l2(x, torch.cat(emb_ref))
    report("pos: patches + positions = HF emb (f32)", rc == 0 and e < 1e-6, f"L2 {e:.2e}")

    # newline: the HF adapter rows without the line breaks, in the order of the merged grid
    y = []
    for t_out, crd_i in zip(out_ref, crd_all):
        gh, gwe = int(crd_i[2, 0]), int(crd_i[3, 0])
        wb = gwe // MERGE
        rows = t_out.reshape(gh // MERGE, wb, -1)
        y.append(rows[:, : wb - 1].reshape(-1, rows.shape[-1]))
    y = torch.cat(y)
    out = torch.empty(pix.shape[0] // (MERGE * MERGE), y.shape[1])
    rc = pl.call("hy4_ref_vit_newline", [y, nl, crd, cu, cuc, out], {"M": out.shape[0], "n": y.shape[1], "merge": MERGE, "dtype": "f32"})
    e = l2(out, torch.cat(out_ref))
    report("newline: the HF output with its line breaks", rc == 0 and e < 1e-7, f"L2 {e:.2e}")

    # adapter (`perceive`) as vision.h does it: RMSNorm, view [Pc/4, 4·1152] in block order, GEMM
    # with `hy4.vit_merge_w` (the permuted 2x2 conv that conv/trim.py writes into the constants),
    # GELU, the 1x1 conv as a GEMM; against the HF output without the line breaks. Tests the weight
    # permutation and the order of the view.
    import torch.nn.functional as F
    from safetensors.torch import load_file as lf
    cts = lf(str(a.consts))
    wm = cts["hy4.vit_merge_w"].float()
    w0 = weight("vit.perceive.proj.0.weight")
    report("vit_merge_w = proj.0 permuted (ky, kx, c)", torch.equal(wm, w0.permute(0, 2, 3, 1).reshape(w0.shape[0], -1)))
    bm, w2, b2 = weight("vit.perceive.proj.0.bias"), weight("vit.perceive.proj.2.weight"), weight("vit.perceive.proj.2.bias")
    nw = weight("vit.perceive.norm.weight")
    for name in ("pequena", "ancha"):
        t = load_file(str(a.ref / f"{name}.safetensors"))
        _, gh, gw = t["grid_thw"][0].tolist()
        x = t["h.L26"][[r * gw + c for r, c in blocks(gh, gw)]]
        xn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * nw
        m = xn.reshape(-1, 4 * WIDTH)
        yv = F.gelu(m @ wm.T + bm) @ w2.reshape(w2.shape[0], -1).T + b2
        wb = gw // MERGE + 1
        ref = t["out"].reshape(gh // MERGE, wb, -1)[:, : wb - 1].reshape(-1, yv.shape[1])
        e = l2(yv, ref)
        report(f"adapter with vit_merge_w ({name}) against HF FP32", e < 1e-5, f"L2 {e:.2e}")
        bad = F.gelu(m @ w0.permute(0, 3, 2, 1).reshape(w0.shape[0], -1).T + bm) @ w2.reshape(w2.shape[0], -1).T + b2
        report(f"a wrong permutation shows ({name})", l2(bad, ref) > 1e-2, f"L2 {l2(bad, ref):.2e}")

    # rejections
    rc = pl.call("hy4_ref_vit_compact", [pix, crd, cu, torch.empty(Pc + 4, DIM), torch.empty(Pc + 4, 4, dtype=I32), cuc],
                 {"M": pix.shape[0], "dim": DIM, "merge": MERGE})
    report("compact with output of another size: rejects", rc != 0)
    bad = crd.clone()
    bad[3] = MERGE
    rc = pl.call("hy4_ref_vit_compact", [pix, bad, cu, pixc, crdc, cuc], {"M": pix.shape[0], "dim": DIM, "merge": MERGE})
    report("compact without line-break column: rejects", rc != 0)
    rc = pl.call("hy4_ref_vit_newline", [y[:-1], nl, crd, cu, cuc, out], {"M": out.shape[0], "n": y.shape[1], "merge": MERGE, "dtype": "f32"})
    report("newline with one row too few: rejects", rc != 0)
    print("\n" + ("all good" if not FAILURES else f"{len(FAILURES)} failures: {FAILURES}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
