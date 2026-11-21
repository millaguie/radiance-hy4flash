#!/usr/bin/env python3
"""hy4_hc_pre / hy4_hc_post against HF's HYV4FlashHyperConnection and HYV4FlashHyperHead, and
against layer 0 of the FP32 reference pass (if present).

HY4_CKPT is the checkpoint (default: the first snapshot of tencent/Hy4-flash in the Hugging Face
cache under $HOME); HY4_REF the FP32 reference pass ref/reference.py wrote (default
~/hy4/ref/fp32-1)."""
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from rad_ctypes import Plugin  # noqa: E402
from transformers import AutoConfig  # noqa: E402
from transformers.models.hy_v4_flash import modeling_hy_v4_flash as M  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SO = ROOT / "home/kernels/hy4k.so"
SNAP = Path(os.environ["HY4_CKPT"]) if os.environ.get("HY4_CKPT") else sorted((Path.home() / ".cache/huggingface/hub/models--tencent--Hy4-flash/snapshots").iterdir())[0]
REF = Path(os.environ.get("HY4_REF") or Path.home() / "hy4/ref/fp32-1")


# FP32: the 12,288-term dot product summed in a different order than torch's BLAS gives a relative
# error of about sqrt(n)·eps ≈ 7e-6. Above 1e-5 it is no longer rounding.
TOL_FP32 = 1e-5


def l2(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / a.norm().clamp_min(1e-30)).item()


def main() -> int:
    torch.manual_seed(0)
    pl = Plugin(SO)
    cfg = AutoConfig.from_pretrained(SNAP).text_config
    hc, n = cfg.hc_mult, cfg.hidden_size
    ok = True
    for dt in (torch.float32, torch.bfloat16):
        mod = M.HYV4FlashHyperConnection(cfg)
        with torch.no_grad():
            mod.fn.normal_(0, 0.02); mod.base.normal_(0, 0.5); mod.scale.normal_(0, 0.5)
        h = (torch.randn(37, hc, n) * 3).to(dt)
        with torch.no_grad():
            post_ref, x_ref = mod(h.unsqueeze(0))
            post_ref, x_ref = post_ref[0], x_ref[0]
        x = torch.empty(37, n, dtype=dt)
        post = torch.empty(37, hc, dtype=torch.float32)
        rc = pl.call("hy4_ref_hc_pre", [h.reshape(37, hc * n), mod.fn.data.float().contiguous(), mod.base.data,
                                         mod.scale.data, x, post],
                     {"M": 37, "n": n, "hc": hc, "eps": float(cfg.rms_norm_eps), "hc_eps": float(cfg.hc_eps),
                      "magnitude": float(cfg.hc_magnitude), "dtype": "bf16" if dt == torch.bfloat16 else "f32"})
        e1, e2 = l2(x_ref.float(), x.float()), l2(post_ref.float(), post)
        tol = TOL_FP32 if dt == torch.float32 else 4e-4
        print(f"hc_pre {dt}: rc {rc}, x L2 {e1:.2e}, post L2 {e2:.2e}")
        ok &= rc == 0 and e1 < tol and e2 < TOL_FP32
        # post: h_j = post_j * y + h_j, compared with the HF expression (HF:897)
        y = torch.randn(37, n).to(dt)
        ref_h = (post.float().unsqueeze(-1) * y.float().unsqueeze(-2) + h.float()).to(dt)
        h2 = h.clone()
        rc = pl.call("hy4_ref_hc_post", [h2.reshape(37, hc * n), y, post],
                     {"M": 37, "n": n, "hc": hc, "dtype": "f32"})
        e = l2(ref_h.float(), h2.float())
        print(f"hc_post {dt}: rc {rc}, L2 {e:.2e} equal={torch.equal(ref_h, h2)}")
        ok &= rc == 0 and torch.equal(ref_h, h2)
        # head (head=1)
        hh = M.HYV4FlashHyperHead(cfg)
        with torch.no_grad():
            hh.hc_fn.normal_(0, 0.02); hh.hc_base.normal_(0, 0.5); hh.hc_scale.normal_(0, 0.5)
            xh_ref = hh(h.unsqueeze(0))[0]
        xh = torch.empty(37, n, dtype=dt)
        rc = pl.call("hy4_ref_hc_pre", [h.reshape(37, hc * n), hh.hc_fn.data.float().contiguous(), hh.hc_base.data,
                                         hh.hc_scale.data, xh, None],
                     {"M": 37, "n": n, "hc": hc, "head": 1, "eps": float(cfg.rms_norm_eps),
                      "hc_eps": float(cfg.hc_eps), "dtype": "f32"})
        e = l2(xh_ref.float(), xh.float())
        print(f"hc_head {dt}: rc {rc}, L2 {e:.2e}")
        ok &= rc == 0 and e < tol

    layer0 = REF / "corto.L00.safetensors"
    if layer0.exists():
        from safetensors.torch import load_file
        from safetensors import safe_open
        import json
        sys.path.insert(0, str(ROOT / "ref"))
        from dump_names import canonical_keys
        v = canonical_keys(load_file(str(layer0)))
        idx = json.loads((SNAP / "model.safetensors.index.json").read_text())["weight_map"]
        def w(k):
            with safe_open(str(SNAP / idx[k]), framework="pt") as f:
                return f.get_tensor(k).float()
        fn = w("model.layers.0.hc_attn_layer.hc_pre.hc_fn").contiguous()
        base = w("model.layers.0.hc_attn_layer.hc_pre.hc_base")
        scale = w("model.layers.0.hc_attn_layer.hc_pre.hc_scale")
        h = v["layer.in"].float()
        t = h.shape[0]
        x = torch.empty(t, n); post = torch.empty(t, hc)
        rc = pl.call("hy4_ref_hc_pre", [h.reshape(t, hc * n), fn, base, scale, x, post],
                     {"M": t, "n": n, "hc": hc, "eps": float(cfg.rms_norm_eps), "hc_eps": float(cfg.hc_eps),
                      "magnitude": float(cfg.hc_magnitude), "dtype": "f32"})
        e1, e2 = l2(v["ihc_attn.mix"], x), l2(v["ihc_attn.post"], post)
        print(f"real layer 0 (corto): rc {rc}, mix L2 {e1:.2e}, post L2 {e2:.2e}")
        ok &= rc == 0 and e1 < TOL_FP32 and e2 < TOL_FP32
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
