#!/usr/bin/env python3
"""Compares what radiance dumps (RADIANCE_DEBUG_MM_DUMP) with the HF reference (ref/vision_ref.py).

For each image:
- processor: the radiance patches (block order, with the skip column) against the HF ones, already
  in the same order and without the column;
- tower: the radiance output against the HF FP32 one, with the HF BF16 floor next to it;
- tower alone: HF in FP32 on the radiance pixels, to separate the tower error from the processor
  error (--tower).

Usage: compare_vision.py --dump DIR --ref ~/hy4/vision-fp32 --noise ~/hy4/vision-bf16 [--tower --snapshot S]
"""
import argparse
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

MERGE, DIM, N_EMBD = 2, 768, 3072


def l2(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / max(b.norm(), 1e-30))


def bf16(path, cols):
    t = torch.frombuffer(bytearray(Path(path).read_bytes()), dtype=torch.bfloat16)
    return t.view(-1, cols).float()


def blocks(gh, gw):
    return [(bh * MERGE + mh, bw * MERGE + mw) for bh in range(gh // MERGE) for bw in range(gw // MERGE)
            for mh in range(MERGE) for mw in range(MERGE)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--noise", type=Path)
    ap.add_argument("--tower", action="store_true")
    ap.add_argument("--snapshot", type=Path)
    a = ap.parse_args()
    refs = {}
    for f in a.ref.glob("*.safetensors"):
        t = load_file(str(f))
        _, gh, gw = t["grid_thw"][0].tolist()
        refs[(gh, gw)] = (f.stem, t)
    tower = None
    if a.tower:
        sys.path.insert(0, str(Path(__file__).parent))
        from transformers import AutoConfig
        import vision_ref as V
        cfg = AutoConfig.from_pretrained(a.snapshot)
        cfg.vision_config._attn_implementation = "eager"
        tower = V.load_tower(a.snapshot, cfg, torch.float32)
    for pix in sorted(a.dump.glob("mm_*.pix.bf16")):
        m = re.search(r"_1x(\d+)x(\d+)\.pix\.bf16$", pix.name)
        gh, gwe = int(m.group(1)), int(m.group(2))
        gw = gwe - MERGE
        if (gh, gw) not in refs:
            print(pix.name, "no HF reference")
            continue
        name, t = refs[(gh, gw)]
        rp = bf16(pix, DIM)
        order = blocks(gh, gwe)
        kept = [k for k, (r, c) in enumerate(order) if c < gw]
        rp_c = rp[kept]
        hf_ord = [r * gw + c for r, c in blocks(gh, gw)]
        hp = t["pixel_values"][hf_ord]
        diff = (rp_c - hp).abs()
        print(f"{name} ({gh}x{gw}): processor L2 {l2(rp_c, hp):.2e}, max |diff| {diff.max():.4f}, "
              f"different values {(diff > 0).float().mean() * 100:.2f} %")
        out = bf16(str(pix).replace(".pix.bf16", ".bf16"), N_EMBD)
        ho = t["out"]
        line = f"{name}: tower vs HF FP32 L2 {l2(out, ho):.2e}"
        if a.noise and (a.noise / f"{name}.safetensors").exists():
            hb = load_file(str(a.noise / f"{name}.safetensors"))["out"]
            line += f" (HF BF16: {l2(hb, ho):.2e})"
        print(line)
        if tower is not None:
            raster = torch.empty(gh * gw, DIM)
            for k, (r, c) in zip(kept, [order[q] for q in kept]):
                raster[r * gw + c] = rp[k]
            with torch.no_grad():
                r = tower(raster, grid_thw=torch.tensor([[1, gh, gw]]))
            ref_rp = r.pooler_output.squeeze(0)
            print(f"{name}: tower vs HF FP32 on the radiance pixels L2 {l2(out, ref_rp):.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
