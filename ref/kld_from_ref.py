#!/usr/bin/env python3
"""A radiance `kld-record` reference directory built from the logits of ref/reference.py.

radiance compares a model against a reference recorded with `kld-ref: DIR` (docs/TOOLS.md, format
in core/kld.h, `radiance-kld-1`). Here the reference is not another container but the HF FP32
pass: this script writes that directory from `<prompt>.logits.safetensors` and `<prompt>.ids`, and
radiance scores its own model on the same tokens (it reads them from the directory, without
retokenizing).

With `--noise DIR` (the HF BF16 pass) it also computes the floor: the per-position KL of the BF16
pass against the FP32 one, its top-1 agreement and its percentiles. It is the criterion of plan
step 5.

Usage: kld_from_ref.py --ref ~/hy4/ref/fp32-1 --out DIR [--noise ~/hy4/ref/bf16-1]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dump_names import canonical_keys  # noqa: E402

PROMPTS = ["corto", "medio", "tools", "largo"]
TOP = 16          # kKldTop from core/kld.h
CHUNK = 1024


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out")
    ap.add_argument("--noise")
    ap.add_argument("--model", default="HF Hy4-flash FP32 (ref/reference.py)")
    ap.add_argument("--trim", action="store_true",
                    help="the logits of the 3-layer trimmed model (logits_trim of <prompt>.L02) instead of "
                         "the model ones")
    a = ap.parse_args()

    if a.out:
        os.makedirs(a.out, exist_ok=False)   # radiance does not overwrite a reference either
        f_tok = open(os.path.join(a.out, "tokens.i32"), "wb")
        f_logp = open(os.path.join(a.out, "logp.f16"), "wb")
        f_top = open(os.path.join(a.out, "top.bin"), "wb")
        f_rows = open(os.path.join(a.out, "rows.f32"), "wb")
    docs, row0, n_vocab = [], 0, None
    kl_all, top1_all = [], []
    for p in PROMPTS:
        ids = load_file(os.path.join(a.ref, f"{p}.ids.safetensors"))["ids"].to(torch.int64)
        def logits(d):
            if a.trim:
                return canonical_keys(load_file(os.path.join(d, f"{p}.L02.safetensors")))["logits_trim"].float()
            return load_file(os.path.join(d, f"{p}.logits.safetensors"))["logits"].float()
        lg = logits(a.ref)
        n = ids.numel()
        n_vocab = lg.shape[1]
        rows = n - 1                          # position p predicts token p+1
        lr = None
        if a.noise:
            lr = logits(a.noise)
        if a.out:
            f_tok.write(ids.to(torch.int32).numpy().tobytes())
        for r0 in range(0, rows, CHUNK):
            r1 = min(rows, r0 + CHUNK)
            lp = torch.log_softmax(lg[r0:r1].double(), dim=-1)
            if a.out:
                f_logp.write(lp.to(torch.float16).numpy().tobytes())
                tv, ti = lp.topk(TOP, dim=-1)
                rec = np.zeros((r1 - r0, TOP), dtype=[("id", "<i4"), ("lp", "<f4")])
                rec["id"] = ti.numpy().astype(np.int32)
                rec["lp"] = tv.numpy().astype(np.float32)
                f_top.write(rec.tobytes())
                nxt = ids[r0 + 1:r1 + 1]
                nll = -lp.gather(1, nxt.view(-1, 1)).squeeze(1)
                ins = -(lp.exp() * lp).sum(-1)
                f_rows.write(torch.stack([nll, ins], 1).to(torch.float32).numpy().tobytes())
            if lr is not None:
                lq = torch.log_softmax(lr[r0:r1].double(), dim=-1)
                kl = (lp.exp() * (lp - lq)).sum(-1)
                kl_all.append(kl.numpy())
                top1_all.append((lp.argmax(-1) == lq.argmax(-1)).numpy())
        docs.append({"source": p, "tokens": n, "score_from": 0, "row0": row0})
        row0 += rows
        print(f"{p}: {n} tokens, {rows} rows", file=sys.stderr)
    if a.out:
        for f in (f_tok, f_logp, f_top, f_rows):
            f.flush()
            os.fsync(f.fileno())
            f.close()
        m = {"format": "radiance-kld-1", "model": a.model, "corpus": "ref/prompts.json", "n_vocab": n_vocab,
             "top": TOP, "rows": row0, "docs": docs}
        with open(os.path.join(a.out, "kld.json.tmp"), "w") as f:
            json.dump(m, f, indent=1)
        os.replace(os.path.join(a.out, "kld.json.tmp"), os.path.join(a.out, "kld.json"))
        print(f"reference written to {a.out}: {row0} rows", file=sys.stderr)
    if kl_all:
        kl = np.concatenate(kl_all)
        t1 = np.concatenate(top1_all)
        q = np.percentile(kl, [50, 99, 99.9, 99.99])
        print(f"BF16 floor (HF BF16 vs HF FP32), {kl.size} positions: mean KL {kl.mean():.5f}, "
              f"median {q[0]:.5f}, p99 {q[1]:.4f}, p99.9 {q[2]:.4f}, p99.99 {q[3]:.4f}, "
              f"max {kl.max():.4f}; top-1 agreement {t1.mean():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
