#!/usr/bin/env python3
"""hy4_rope (GPT-J pairs with the M-RoPE sections interleaved) against the HF rotary of Hy4
(HYV4FlashRotaryEmbedding + apply_rotary_pos_emb), with 3-D positions that differ per plane like
those of an image, in both tables (window, theta 1e4; sparse, theta 1e7). With the three planes
equal it must give the same as mode=gptj with 1-D positions.

HY4_CKPT is the checkpoint, read for its config (default: the first snapshot of tencent/Hy4-flash
in the Hugging Face cache under $HOME)."""
import os
import sys
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.models.hy_v4_flash import modeling_hy_v4_flash as M

sys.path.insert(0, str(Path(__file__).parent))
import rad_ctypes as RC  # noqa: E402

SNAP = Path(os.environ["HY4_CKPT"]) if os.environ.get("HY4_CKPT") else sorted((Path.home() / ".cache/huggingface/hub/models--tencent--Hy4-flash/snapshots").iterdir())[0]
pl = RC.Plugin(Path(__file__).resolve().parent.parent / "build/radiance_home/kernels/hy4k.so")
cfg = AutoConfig.from_pretrained(SNAP).text_config
rot = M.HYV4FlashRotaryEmbedding(cfg)
failures = []


def l2(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm())


torch.manual_seed(0)
Mtok, H, D = 96, 4, 64
pos = torch.stack([torch.full((Mtok,), 7), 7 + torch.arange(Mtok) // 12, 7 + torch.arange(Mtok) % 12]).to(torch.int32)
for ltype, theta in (("sliding_attention", 1e4), ("indexed_attention", 1e7)):
    sec = " ".join(str(s) for s in cfg.rope_parameters[ltype]["mrope_section"])
    x = torch.randn(Mtok, H * D)
    cos, sin = rot(x, pos.long().view(3, 1, Mtok), layer_type=ltype)
    q = x.view(1, Mtok, H, D).transpose(1, 2)
    ref, _ = M.apply_rotary_pos_emb(q, q, cos, sin, unsqueeze_dim=1)
    ref = ref.transpose(1, 2).reshape(Mtok, H * D)
    y = x.clone()
    rc = pl.call("hy4_ref_rope", [y, pos.contiguous()], {"M": Mtok, "head_dim": D, "n_head": H, "n_head_kv": 0,
                 "theta": theta, "scale": 1.0, "mode": "imrope", "rotary_dim": D, "sections": sec})
    e = l2(y, ref)
    ok = rc == 0 and e < 1e-6
    print("ok  " if ok else "FAIL", f"imrope {ltype}: L2 {e:.2e}")
    if not ok:
        failures.append(ltype)
    # equal planes == gptj 1-D
    p1 = torch.arange(Mtok, dtype=torch.int32)
    y3, y1 = x.clone(), x.clone()
    pl.call("hy4_ref_rope", [y3, p1.repeat(3, 1).contiguous()], {"M": Mtok, "head_dim": D, "n_head": H, "theta": theta,
            "scale": 1.0, "mode": "imrope", "rotary_dim": D, "sections": sec})
    pl.call("hy4_ref_rope", [y1, p1], {"M": Mtok, "head_dim": D, "n_head": H, "theta": theta, "scale": 1.0,
            "mode": "gptj", "rotary_dim": D})
    e = l2(y3, y1)
    print("ok  " if e == 0 else "FAIL", f"equal planes = gptj ({ltype}): L2 {e:.1e}")
    if e != 0:
        failures.append(ltype + " 1-D")
# 1-D: hy4_rope (gptj) against radiance's `rope` (libref), the one that runs without the tower.
# Turning on mm-max-patches switches op in the MLA: text must give the same bytes.
libref = RC.Plugin(Path(__file__).resolve().parent.parent / "prefix/share/radiance/kernels/libref.so")
for dt in (torch.float32, torch.bfloat16):
    for theta in (1e4, 1e7):
        x = torch.randn(Mtok, H * 256).to(dt)
        p1 = (torch.arange(Mtok, dtype=torch.int32) + 70000)
        a_, b_ = x.clone(), x.clone()
        pr = {"M": Mtok, "head_dim": 256, "n_head": H, "n_head_kv": 0, "theta": theta, "scale": 1.0, "mode": "gptj",
              "rotary_dim": 64}
        r1 = pl.call("hy4_ref_rope", [a_, p1], pr)
        r2 = libref.call("ref_rope", [b_, p1], pr)
        # BF16 (what is served): the same bytes. f32: libref uses its own pow/cos (ref_pow, ref_cos),
        # so f32 rounding is allowed.
        if dt == torch.bfloat16:
            ok = r1 == 0 and r2 == 0 and torch.equal(a_, b_)
        else:
            ok = r1 == 0 and r2 == 0 and l2(a_, b_) < 1e-6
        print("ok  " if ok else "FAIL", f"1-D same as radiance's rope ({dt}, theta {theta:g}):",
              f"rc {r1} {r2}, differing {int((a_ != b_).sum())}, L2 {l2(a_, b_):.1e}")
        if not ok:
            failures.append(f"rope 1-D {dt} {theta}")
print("all good" if not failures else f"failures: {failures}")
sys.exit(1 if failures else 0)
