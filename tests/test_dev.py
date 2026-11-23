#!/usr/bin/env python3
"""hy4k card rows against their host oracles, on this machine's GPU (plan step 6).

Loads the hy4k.so with HIP rows (build-hip-22, tools/build-hip.sh), which carries both row classes, and
the ROCm HIP runtime (/opt/rocm/lib/libamdhip64.so). Each case runs the same data through the host
row (host memory) and the card row (GPU memory) and compares: integers equal; floats with relative
L2 < 1e-5 if the output is f32, and < 4e-3 with 99.5 % of the values within one BF16 rounding if it
is BF16 (the sums go in another order and a value on the edge falls to the other side).

Usage: ref/.venv/bin/python tests/test_dev.py [--so build-hip-22/radiance_home/kernels/hy4k.so]
"""
import argparse
import ctypes as C
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import rad_ctypes as RC  # noqa: E402

RC.DT["e4m3"] = 4   # RAD_F8E4M3 (abi/rad_types.h)

ROOT = Path(__file__).resolve().parent.parent
hip = C.CDLL("/opt/rocm/lib/libamdhip64.so")
hip.hipMalloc.argtypes = [C.POINTER(C.c_void_p), C.c_size_t]
hip.hipMemcpy.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int]
hip.hipMemset.argtypes = [C.c_void_p, C.c_int, C.c_size_t]
hip.hipFree.argtypes = [C.c_void_p]
H2D, D2H = 1, 2
results = []


class Dev:
    """A tensor on the GPU with the interface rad_ctypes.tensor() uses."""

    def __init__(self, t: torch.Tensor):
        self.t = t.contiguous()
        self.n = self.t.numel() * self.t.element_size()
        p = C.c_void_p()
        assert hip.hipMalloc(C.byref(p), max(self.n, 4)) == 0
        self.p = p.value
        assert hip.hipMemcpy(self.p, self.t.data_ptr(), self.n, H2D) == 0
        self.dtype = self.t.dtype
        self.shape = self.t.shape

    def data_ptr(self):
        return self.p

    def dim(self):
        return self.t.dim()

    def stride(self, i):
        return self.t.stride(i)

    def to_host(self) -> torch.Tensor:
        assert hip.hipDeviceSynchronize() == 0
        out = torch.empty_like(self.t)
        assert hip.hipMemcpy(out.data_ptr(), self.p, self.n, D2H) == 0
        return out

    def __del__(self):
        hip.hipFree(self.p)


class F8:
    """A uint8 tensor passed to the plugin as E4M3."""

    def __init__(self, t):
        self.t = t
        self.dtype = "e4m3"
        self.shape = t.shape

    def data_ptr(self):
        return self.t.data_ptr()

    def dim(self):
        return self.t.dim()

    def stride(self, i):
        return self.t.stride(i)


class F8D(F8):
    def __init__(self, d):
        self.t = d
        self.dtype = "e4m3"
        self.shape = d.shape

    def data_ptr(self):
        return self.t.data_ptr()


def report(name, ok, detail=""):
    results.append(bool(ok))
    print(("OK  " if ok else "FAIL"), name, detail)


def compare(name, host: torch.Tensor, dev: torch.Tensor):
    if not torch.is_floating_point(host):
        d = int((host != dev).sum())
        report(name, d == 0, f"{d} differ out of {host.numel()}")
        return
    h, d = host.double(), dev.double()
    finite = torch.isfinite(h)
    nan_h, nan_d = torch.isnan(h), torch.isnan(d)
    inf_h = ~finite & ~nan_h
    if not torch.equal(finite, torch.isfinite(d)) or not torch.equal(nan_h, nan_d) or not torch.equal(h[inf_h], d[inf_h]):
        report(name, False, "non-finite values differ")
        return
    e = float((h[finite] - d[finite]).norm() / max(float(h[finite].norm()), 1e-30))
    if host.dtype == torch.float32:
        report(name, e < 1e-5, f"L2 {e:.2e}")
    else:
        ulp = (h[finite].abs() * 2.0 ** -7).clamp_min(1e-30)
        within = float(((h[finite] - d[finite]).abs() <= ulp).float().mean())
        report(name, e < 4e-3 and within > 0.995, f"L2 {e:.2e}, within one rounding {within:.4f}")


def run_both(pl, name, inputs, outputs, params):
    """Calls hy4_ref_<name> and hy4_dev_<name>. `inputs` and `outputs` are lists of host
    tensors in operand order (None for a gap). Returns (host outputs, card outputs)."""
    sh = [t.clone() if t is not None else None for t in outputs]
    rc_h = pl.call(f"hy4_ref_{name}", list(inputs) + sh, params)
    de = [Dev(t) if t is not None else None for t in inputs]
    ds = [Dev(t.clone()) if t is not None else None for t in outputs]
    rc_d = dev_call(pl, f"hy4_dev_{name}", de + ds, params)
    out = [x.to_host() if x is not None else None for x in ds]
    if rc_h != 0 or rc_d != 0:
        report(f"{name}: rc", False, f"host {rc_h}, card {rc_d}")
    return sh, out, de


def bf(t):
    return t.to(torch.bfloat16)


def dev_call(pl, kernel, tensors, params, stream=None):
    """A card row's launch with the scratch its hook asks for, as the engine gives it (zeroed here:
    the engine's is whatever the last op left, so a row must not read scratch it did not write)."""
    nb = pl.scratch_bytes(kernel, tensors, params)
    sc = Dev(torch.zeros(max(nb, 4), dtype=torch.uint8)) if nb else None
    rc = pl.call(kernel, tensors, params, scratch=(sc.p, nb) if sc else None, stream=stream)
    if sc:
        assert hip.hipDeviceSynchronize() == 0   # the scratch is freed on return
    return rc


def main():
    ap = argparse.ArgumentParser()
    # build-hip-22, the radiance 2.2.0 build (tools/build-hip.sh); the 2.1.0 trees are not looked at,
    # so a run cannot pick up a plugin of the other ABI because it happens to be newer
    cands = [ROOT / d / "radiance_home/kernels/hy4k.so" for d in ("build-hip-22",)]
    cands = [c for c in cands if c.exists()]
    ap.add_argument("--so", default=str(max(cands, key=lambda c: c.stat().st_mtime) if cands else cands))
    # --chain: every card call carries chain = 1, as the engine resolves a row where chaining is on, so
    # the chained forms (r4dchain) run. HIP writes no chain record, so they neither wait nor count: the
    # run checks that the chained forms compute what the plain ones do.
    ap.add_argument("--chain", action="store_true")
    a = ap.parse_args()
    pl = RC.Plugin(Path(a.so))
    if a.chain:
        call0 = pl.call
        pl.call = lambda kernel, tensors, params, **kw: call0(
            kernel, tensors, {**params, "chain": 1} if kernel.startswith("hy4_dev_") else params, **kw)
    torch.manual_seed(7)
    I32, I64 = torch.int32, torch.int64

    # hc_pre / hc_post / mla_kv
    rows, n, hc = 37, 3072, 4
    h = bf(torch.randn(rows, hc * n))
    fn = torch.randn(2 * hc, hc * n) * 0.01
    base, scale = torch.randn(2 * hc) * 0.5, torch.tensor([0.7, 0.9])
    p = {"M": rows, "n": n, "hc": hc, "eps": 1e-5, "hc_eps": 1e-6, "magnitude": 2.0, "dtype": "bf16"}
    hs, ds, _ = run_both(pl, "hc_pre", [h, fn, base, scale], [bf(torch.zeros(rows, n)), torch.zeros(rows, hc)], p)
    compare("hc_pre x", hs[0], ds[0])
    compare("hc_pre post", hs[1], ds[1])
    hs, ds, _ = run_both(pl, "hc_pre", [h, fn[:hc].contiguous(), base[:hc].contiguous(), scale[:1].contiguous()],
                    [bf(torch.zeros(rows, n))], {**p, "head": 1})
    compare("hc_pre head", hs[0], ds[0])
    y, post = bf(torch.randn(rows, n)), torch.rand(rows, hc) * 2
    hh = h.clone()
    pl.call("hy4_ref_hc_post", [hh, y, post], {"M": rows, "n": n, "hc": hc, "dtype": "bf16"})
    dh = Dev(h)
    rc = pl.call("hy4_dev_hc_post", [dh, Dev(y), Dev(post)], {"M": rows, "n": n, "hc": hc, "dtype": "bf16"})
    compare("hc_post", hh, dh.to_host()) if rc == 0 else report("hc_post rc", False, str(rc))
    H, nope, rope, vd = 32, 192, 64, 256
    kvx, kr = bf(torch.randn(rows, H * (nope + vd))), bf(torch.randn(rows, rope))
    hs, ds, _ = run_both(pl, "mla_kv", [kvx, kr], [bf(torch.zeros(rows, H * (rope + nope))), bf(torch.zeros(rows, H * vd))],
                    {"M": rows, "heads": H, "nope": nope, "rope": rope, "v_dim": vd, "dtype": "bf16"})
    compare("mla_kv k", hs[0], ds[0])
    compare("mla_kv v", hs[1], ds[1])

    # ngram_ids: 2 layers, 2 sequences, linear group state, ids [E·R, 16]
    E, V, heads, R = 2, 1000, 16, 40
    wmap = torch.randint(0, 500, (E, V), dtype=I32)
    mult = torch.randint(1, 2 ** 40, (E, 3), dtype=I64)
    vsz = torch.tensor([[97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173]] * E, dtype=I64)
    off = torch.cat([torch.zeros(E, 1, dtype=I64), vsz.cumsum(1)[:, :-1]], 1)
    tok = torch.randint(0, 1100, (20,), dtype=I32)
    pos = torch.tensor(list(range(5, 12)) + list(range(0, 13)), dtype=I32)
    cu, sidx = torch.tensor([0, 7, 20], dtype=I32), torch.tensor([[2, 9], [0, 9]], dtype=I32)
    st = torch.randint(0, 900, (4, 1, 1, 2), dtype=I32)
    pn = {"M": 20, "engram_layers": E, "heads": heads, "ngram": 3, "pad": 5, "seqs": 4}
    sth, ids_h = st.clone(), torch.full((E * R, heads), -3, dtype=I32)
    rc_h = pl.call("hy4_ref_ngram_ids", [tok, pos, cu, sth, sidx, wmap, mult, vsz, off, ids_h], pn)
    dst, dids = Dev(st), Dev(torch.full((E * R, heads), -3, dtype=I32))
    rc_d = pl.call("hy4_dev_ngram_ids", [Dev(tok), Dev(pos), Dev(cu), dst, Dev(sidx), Dev(wmap), Dev(mult), Dev(vsz),
                                         Dev(off), dids], pn)
    if rc_h or rc_d:
        report("ngram_ids rc", False, f"{rc_h} {rc_d}")
    compare("ngram_ids ids", ids_h, dids.to_host())
    compare("ngram_ids state", sth, dst.to_host())
    # ahead_ids (the Engram read-ahead): one sequence of 20 tokens as one step, against its first 10
    # with the other 10 ahead; the ahead ids must be the ones the next step will compute, on the host
    # and on the card, and the state must advance by the step's 10 tokens only.
    tokA = torch.randint(0, 1100, (20,), dtype=I32)
    posA = torch.arange(40, 60, dtype=I32)
    stA = torch.randint(0, 900, (4, 1, 1, 2), dtype=I32)
    sidxA = torch.tensor([[1, 9]], dtype=I32)
    pA = {"M": 20, "engram_layers": E, "heads": heads, "ngram": 3, "pad": 5, "seqs": 4}
    full = torch.full((E * 20, heads), -3, dtype=I32)
    st_full = stA.clone()
    pl.call("hy4_ref_ngram_ids", [tokA, posA, torch.tensor([0, 20], dtype=I32), st_full, sidxA, wmap, mult, vsz, off,
                                   full], pA)
    p10 = {**pA, "M": 10}
    for side in ("host", "card"):
        st10 = stA.clone()
        ids10 = torch.full((E * 10, heads), -3, dtype=I32)
        ahd = torch.full((E * 10, heads), -3, dtype=I32)
        args = [tokA, posA[:10].contiguous(), torch.tensor([0, 10], dtype=I32), st10, sidxA, wmap, mult, vsz, off, ids10, ahd]
        if side == "host":
            rc = pl.call("hy4_ref_ngram_ids", args, p10)
            o10, oah, ost = ids10, ahd, st10
        else:
            dv = [Dev(x) for x in args]
            rc = pl.call("hy4_dev_ngram_ids", dv, p10)
            o10, oah, ost = dv[9].to_host(), dv[10].to_host(), dv[3].to_host()
        fv, av = full.view(E, 20, heads), oah.view(E, 10, heads)
        report(f"ngram_ids ahead ({side}): ids of the step", rc == 0 and torch.equal(o10.view(E, 10, heads), fv[:, :10]))
        report(f"ngram_ids ahead ({side}): ahead ids = those of the next step", torch.equal(av, fv[:, 10:]))
        st_10ref = stA.clone()
        pl.call("hy4_ref_ngram_ids", [tokA[:10].contiguous(), posA[:10].contiguous(), torch.tensor([0, 10], dtype=I32),
                                       st_10ref, sidxA, wmap, mult, vsz, off, torch.zeros(E * 10, heads, dtype=I32)], p10)
        report(f"ngram_ids ahead ({side}): the state advances by the step only", torch.equal(ost, st_10ref))
    # num_accepted (speculative verify): a 10-token prefill, a verify step [x0, x1, x2] of which two
    # commit, then a step [y0, y1] read with num_accepted 2. Its ids must equal a plain run's over the
    # committed text c0..c9 x0 x1 y0 y1, on the host and on the card; the window is 2 + 3 wide.
    c, x, y = torch.randint(0, 1100, (10,), dtype=I32), torch.randint(0, 1100, (3,), dtype=I32), torch.randint(0, 1100, (2,), dtype=I32)
    plain = torch.cat([c, x[:2], y])
    pp = {"M": 14, "engram_layers": E, "heads": heads, "ngram": 3, "pad": 5, "seqs": 4}
    st_p = torch.zeros(4, 1, 1, 2, dtype=I32)
    ids_p = torch.full((E * 14, heads), -3, dtype=I32)
    pl.call("hy4_ref_ngram_ids", [plain, torch.arange(14, dtype=I32), torch.tensor([0, 14], dtype=I32), st_p,
                                   torch.tensor([[2, 9]], dtype=I32), wmap, mult, vsz, off, ids_p], pp)
    for side in ("host", "card"):
        stw = torch.zeros(4, 1, 1, 5, dtype=I32)
        def run_step(tk, ps, nacc, M):
            ids_ = torch.full((E * M, heads), -3, dtype=I32)
            args = [tk, ps, torch.tensor([0, M], dtype=I32), stw, torch.tensor([[2, 9]], dtype=I32), wmap, mult, vsz,
                    off, ids_, None, nacc]
            pr = {**pp, "M": M}
            if side == "host":
                rc = pl.call("hy4_ref_ngram_ids", args, pr)
                return rc, ids_, stw
            dv = [Dev(t) if t is not None else None for t in args]
            rc = pl.call("hy4_dev_ngram_ids", dv, pr)
            return rc, dv[9].to_host(), dv[3].to_host()
        r1, _, stw = run_step(c, torch.arange(10, dtype=I32), None, 10)
        r2, _, stw = run_step(x, torch.arange(10, 13, dtype=I32), torch.tensor([1], dtype=I32), 3)
        r3, idsy, stw = run_step(y, torch.arange(12, 14, dtype=I32), torch.tensor([2], dtype=I32), 2)
        report(f"ngram_ids num_accepted ({side}): ids after the rejection = those of the run without drafts",
               r1 == 0 and r2 == 0 and r3 == 0 and torch.equal(idsy.view(E, 2, heads), ids_p.view(E, 14, heads)[:, 12:]))
    # Two sequences in the same calls: a prefill, then three verify rounds of 1 + K rows with random
    # acceptance (every round's accepted prefix committed, one round all rejected), the window at its
    # exact width hist + K. Each round's ids must equal a plain run's over each sequence's committed
    # text, and the host and the card must end with the same state.
    K, hist = 3, 2
    W = hist + K
    for side in ("host", "card"):
        g = torch.Generator().manual_seed(7)   # the same draws on both sides
        st2 = torch.zeros(4, 1, 1, W, dtype=I32)
        sidx2 = torch.tensor([[1, 9], [3, 9]], dtype=I32)
        commit = [torch.randint(0, 1100, (6,), dtype=I32, generator=g), torch.randint(0, 1100, (9,), dtype=I32, generator=g)]
        def call_step(toks, poss, nacc):
            lens = [len(t) for t in toks]
            cu_ = torch.tensor([0, lens[0], lens[0] + lens[1]], dtype=I32)
            M = sum(lens)
            ids_ = torch.full((E * M, heads), -3, dtype=I32)
            args = [torch.cat(toks), torch.cat(poss), cu_, st2, sidx2, wmap, mult, vsz, off, ids_, None, nacc]
            pr = {"M": M, "engram_layers": E, "heads": heads, "ngram": 3, "pad": 5, "seqs": 4}
            if side == "host":
                rc = pl.call("hy4_ref_ngram_ids", args, pr)
                return rc, ids_, st2
            dv = [Dev(t) if t is not None else None for t in args]
            rc = pl.call("hy4_dev_ngram_ids", dv, pr)
            return rc, dv[9].to_host(), dv[3].to_host()
        rc, _, st2 = call_step(commit, [torch.arange(6, dtype=I32), torch.arange(9, dtype=I32)], None)
        ok = rc == 0
        nacc_prev = torch.tensor([1, 1], dtype=I32)
        for round_i, acc in enumerate(([2, 4], [1, 1], [3, 2])):   # rows kept this round, per sequence
            rows = [torch.randint(0, 1100, (1 + K,), dtype=I32, generator=g) for _ in range(2)]
            poss = [torch.arange(len(commit[i]), len(commit[i]) + 1 + K, dtype=I32) for i in range(2)]
            rc, ids_r, st2 = call_step(rows, poss, nacc_prev)
            ok = ok and rc == 0
            for i in range(2):   # the plain run over the committed text followed by this round's rows
                plain2 = torch.cat([commit[i], rows[i]])
                idp = torch.full((E * len(plain2), heads), -3, dtype=I32)
                pl.call("hy4_ref_ngram_ids", [plain2, torch.arange(len(plain2), dtype=I32),
                                               torch.tensor([0, len(plain2)], dtype=I32), torch.zeros(4, 1, 1, 2, dtype=I32),
                                               torch.tensor([[2, 9]], dtype=I32), wmap, mult, vsz, off, idp],
                        {"M": len(plain2), "engram_layers": E, "heads": heads, "ngram": 3, "pad": 5, "seqs": 4})
                r0 = 0 if i == 0 else 1 + K
                got = ids_r.view(E, 2 * (1 + K), heads)[:, r0:r0 + 1 + K]
                ok = ok and torch.equal(got, idp.view(E, len(plain2), heads)[:, len(commit[i]):])
                commit[i] = torch.cat([commit[i], rows[i][:acc[i]]])
            nacc_prev = torch.tensor(acc, dtype=I32)
        report(f"ngram_ids speculation, 2 sequences and 3 rounds ({side})", ok)
        if side == "host":
            st_host = st2.clone()
        else:
            report("ngram_ids speculation: the same final state on host and card", torch.equal(st_host, st2))

    # rows_rmsnorm with and without fp8
    x = bf(torch.randn(23, 16 * 96) * 3)
    w = torch.randn(16 * 96)
    for fp8 in (0, 1):
        hs, ds, _ = run_both(pl, "rows_rmsnorm", [x, w], [bf(torch.zeros(23, 16 * 96))],
                        {"M": 23, "heads": 16, "d": 96, "fp8": fp8, "eps": 1.1920929e-07, "dtype": "bf16"})
        compare(f"rows_rmsnorm fp8={fp8}", hs[0], ds[0])

    # engram_gate, with pad rows
    rows = 11
    h = bf(torch.randn(rows, hc * n))
    tk = torch.randint(0, 100, (rows,), dtype=I32)
    tk[3] = 77
    k, v = bf(torch.randn(rows, hc * n)), bf(torch.randn(rows, n))
    wk, wq, gam, per = torch.randn(hc * n), torch.randn(hc * n), torch.randn(n) * 0.1, torch.randn(hc * n) * 0.1
    pe = {"M": rows, "n": n, "hc": hc, "pad": 77, "eps": 1.1920929e-07, "gate_eps": 1e-6, "dtype": "bf16"}
    hh = h.clone()
    pl.call("hy4_ref_engram_gate", [hh, tk, k, v, wk, wq, gam, per], pe)
    dh = Dev(h)
    rc = pl.call("hy4_dev_engram_gate", [dh, Dev(tk), Dev(k), Dev(v), Dev(wk), Dev(wq), Dev(gam), Dev(per)], pe)
    compare("engram_gate", hh, dh.to_host()) if rc == 0 else report("engram_gate rc", False, str(rc))
    # image and video markers (skip, skip2): their rows stay as they are, on host and on card
    tk2 = tk.clone()
    tk2[1], tk2[6] = 120120, 120124
    pe2 = dict(pe, skip=120120, skip2=120124)
    hh2 = h.clone()
    pl.call("hy4_ref_engram_gate", [hh2, tk2, k, v, wk, wq, gam, per], pe2)
    dh2 = Dev(h)
    rc = pl.call("hy4_dev_engram_gate", [dh2, Dev(tk2), Dev(k), Dev(v), Dev(wk), Dev(wq), Dev(gam), Dev(per)], pe2)
    untouched = all(torch.equal(hh2[i], h[i]) for i in (1, 3, 6)) and not torch.equal(hh2[0], h[0])
    report("engram_gate skip: image, video and pad rows untouched on host", untouched)
    compare("engram_gate skip", hh2, dh2.to_host()) if rc == 0 else report("engram_gate skip rc", False, str(rc))

    # idx_pool into a [pages, 1, 4, 32] bf16 cache
    nI, ratio, M = 128, 4, 10
    stg = bf(torch.randn(M * ratio, nI))
    wv, wg, ape, wn = torch.randn(nI, nI) * 0.1, torch.randn(nI, nI) * 0.1, torch.randn(ratio, nI), torch.randn(nI)
    page = torch.tensor([3, 7, -1, 0, 12, 5, 9, -1, 15, 1], dtype=I32)
    bpos = torch.arange(M, dtype=I32) * 4 + 100000
    cache = bf(torch.randn(20, 1, 4, 32))
    pp = {"M": M, "n": nI, "ratio": ratio, "rotary_dim": 64, "block_size": 4, "theta": 1e7, "eps": 1e-5, "dtype": "bf16"}
    ch = cache.clone()
    pl.call("hy4_ref_idx_pool", [stg, wv, wg, ape, wn, page, bpos, ch], pp)
    dc = Dev(cache)
    rc = pl.call("hy4_dev_idx_pool", [Dev(stg), Dev(wv), Dev(wg), Dev(ape), Dev(wn), Dev(page), Dev(bpos), dc], pp)
    compare("idx_pool", ch, dc.to_host()) if rc == 0 else report("idx_pool rc", False, str(rc))

    # idx_score: 2 sequences, 2-D q, plane wider than bt
    Hq, rows = 16, 9
    q = bf(torch.randn(rows, Hq * nI))
    wh = bf(torch.randn(rows, Hq))
    pos = torch.tensor([30, 31, 32, 33, 50, 51, 52, 53, 54], dtype=I32)
    kc = bf(torch.randn(20, 1, 4, 32))
    bt = torch.stack([torch.randperm(20)[:10], torch.randperm(20)[:10]]).to(I32)
    bt[1, 4] = -1
    cu, nc = torch.tensor([0, 4, 9], dtype=I32), torch.tensor([7, 10], dtype=I32)
    ps_ = {"M": rows, "n": nI, "heads": Hq, "rotary_dim": 64, "theta": 1e7, "blocks": 12, "seqs": 4, "dtype": "bf16"}
    hs, ds, _ = run_both(pl, "idx_score", [q, wh, pos, kc, bt, cu, nc], [torch.full((rows, 12), 7.0)], ps_)
    compare("idx_score", hs[0], ds[0])
    # Hy4's geometry at scale (16 heads, n 128, bf16): the gfx12 WMMA path, with few rows (several
    # blocks a row) and many (one block a row), a -1 page, a NaN in one key, and a group count that
    # is not a multiple of 16. (An out-of-range page makes the host oracle refuse the call.)
    for rows in (3, 70):
        G = 1001
        q = bf(torch.randn(rows, Hq * nI))
        wh = bf(torch.randn(rows, Hq))
        pos = torch.arange(4100, 4100 + rows, dtype=I32)
        kc = bf(torch.randn(1100, 1, 4, 32))
        kc[17, 0, 2, 5] = float("nan")
        bt = torch.stack([torch.randperm(1100)[:G + 3] for _ in range(2)]).to(I32)
        bt[0, 9] = -1
        half = rows // 2
        cu, nc = torch.tensor([0, half, rows], dtype=I32), torch.tensor([G, G - 7], dtype=I32)
        p_ = {**ps_, "M": rows, "blocks": G}
        hs, ds, _ = run_both(pl, "idx_score", [q, wh, pos, kc, bt, cu, nc], [torch.full((rows, G), 7.0)], p_)
        nan_h, nan_d = torch.isnan(hs[0]), torch.isnan(ds[0])
        report(f"idx_score WMMA {rows} rows: NaN where the host has it", torch.equal(nan_h, nan_d))
        finite = torch.isfinite(hs[0])
        report(f"idx_score WMMA {rows} rows: -inf where the host has it", torch.equal(torch.isneginf(hs[0]), torch.isneginf(ds[0])))
        compare(f"idx_score WMMA {rows} rows x {G} groups (finite)", hs[0][finite], ds[0][finite])

    # idx_select: 600 groups, lots of ties, topk 512, 2 sequences
    rows, G = 6, 600
    sc = torch.randint(0, 40, (rows, G)).float()
    sc[:, 100:120] = float("-inf")
    sc[2, 7] = float("nan")
    btg = torch.stack([torch.randperm(700)[:G + 4], torch.randperm(700)[:G + 4]]).to(I32)
    btg[0, 50] = -1
    pos = torch.tensor([2000, 2100, 2301, 1500, 2399, 3], dtype=I32)
    cu, nc = torch.tensor([0, 3, 6], dtype=I32), torch.tensor([590, 600], dtype=I32)
    pz = {"M": rows, "blocks": G, "topk": 512, "ratio": 4, "seqs": 4}
    hs, ds, _ = run_both(pl, "idx_select", [sc, btg, nc, pos, cu],
                    [torch.full((rows, 512), -9, dtype=I32), torch.zeros(rows, dtype=I32), torch.zeros(rows, dtype=I32)],
                    pz)
    compare("idx_select sel", hs[0], ds[0])
    compare("idx_select nsel", hs[1], ds[1])
    compare("idx_select seqused", hs[2], ds[2])

    # Review of plan step 6 (findings F1, F5, F2): rope at high positions in f32, NaN in the score, and
    # hc_pre with a row that does not fit in LDS.
    for p0 in (100000, 131071, 1000000):
        rows = 4
        q = torch.randn(rows, Hq * nI)
        wh = torch.randn(rows, Hq)
        pos = torch.arange(rows, dtype=I32) + p0
        kc = torch.randn(20, 1, 4, 32)
        bt = torch.randperm(20)[:12].view(1, 12).to(I32)
        cu, nc = torch.tensor([0, rows], dtype=I32), torch.tensor([12], dtype=I32)
        ps2 = {"M": rows, "n": nI, "heads": Hq, "rotary_dim": 64, "theta": 1e7, "blocks": 12, "seqs": 1, "dtype": "f32"}
        hs, ds, _ = run_both(pl, "idx_score", [q, wh, pos, kc, bt, cu, nc], [torch.zeros(rows, 12)], ps2)
        compare(f"idx_score f32 pos {p0}", hs[0], ds[0])
        stg = torch.randn(M * ratio, nI)
        bpos = torch.arange(M, dtype=I32) * 4 + p0
        cache = torch.randn(20, 1, 4, 32)
        ch = cache.clone()
        pp2 = {**pp, "dtype": "f32"}
        pl.call("hy4_ref_idx_pool", [stg, wv, wg, ape, wn, page, bpos, ch], pp2)
        dc = Dev(cache)
        rc = pl.call("hy4_dev_idx_pool", [Dev(stg), Dev(wv), Dev(wg), Dev(ape), Dev(wn), Dev(page), Dev(bpos), dc], pp2)
        compare(f"idx_pool f32 bpos {p0}", ch, dc.to_host()) if rc == 0 else report("idx_pool rc", False, str(rc))
        if p0 != 100000:
            continue
        # M-RoPE (images): 3-component positions; with all three equal, the same as 1-D
        sec = {"sections": "12 10 10"}
        p3 = torch.stack([pos, pos + 3, pos + 7]).contiguous()
        hs, ds, _ = run_both(pl, "idx_score", [q, wh, p3, kc, bt, cu, nc], [torch.zeros(rows, 12)], {**ps2, **sec})
        compare("idx_score 3-D", hs[0], ds[0])
        h1, _, _ = run_both(pl, "idx_score", [q, wh, pos.repeat(3, 1).contiguous(), kc, bt, cu, nc], [torch.zeros(rows, 12)],
                       {**ps2, **sec})
        h0, _, _ = run_both(pl, "idx_score", [q, wh, pos, kc, bt, cu, nc], [torch.zeros(rows, 12)], ps2)
        report("idx_score 3-D with equal planes = 1-D", torch.equal(h1[0], h0[0]))
        b3 = torch.stack([bpos, bpos + 5, bpos + 9], 1).contiguous()
        ch, dc = cache.clone(), Dev(cache)
        rh = pl.call("hy4_ref_idx_pool", [stg, wv, wg, ape, wn, page, b3, ch], {**pp2, **sec})
        rc = pl.call("hy4_dev_idx_pool", [Dev(stg), Dev(wv), Dev(wg), Dev(ape), Dev(wn), Dev(page), Dev(b3), dc],
                     {**pp2, **sec})
        compare("idx_pool 3-D", ch, dc.to_host()) if rc == 0 and rh == 0 else report("idx_pool 3-D rc", False, f"{rh} {rc}")
        ci, c1 = cache.clone(), cache.clone()
        pl.call("hy4_ref_idx_pool", [stg, wv, wg, ape, wn, page, bpos.view(-1, 1).repeat(1, 3).contiguous(), ci],
                {**pp2, **sec})
        pl.call("hy4_ref_idx_pool", [stg, wv, wg, ape, wn, page, bpos, c1], pp2)
        report("idx_pool 3-D with equal planes = 1-D", torch.equal(ci, c1))
    q = torch.randn(2, Hq * nI)
    q[0, 5] = float("nan")
    hs, ds, _ = run_both(pl, "idx_score", [q, torch.randn(2, Hq), torch.tensor([40, 41], dtype=I32), torch.randn(20, 1, 4, 32),
                                      torch.randperm(20)[:6].view(1, 6).to(I32), torch.tensor([0, 2], dtype=I32),
                                      torch.tensor([6], dtype=I32)],
                    [torch.zeros(2, 6)], {**ps2, "M": 2, "blocks": 6})
    report("idx_score propagates NaN like the host", bool(torch.isnan(hs[0][0]).all() and torch.isnan(ds[0][0]).all()))
    compare("idx_score with NaN (row 1 and NaN in place)", hs[0], ds[0])
    rows, n2 = 3, 4000
    h = bf(torch.randn(rows, hc * n2))
    fn2 = torch.randn(2 * hc, hc * n2) * 0.01
    p2 = {"M": rows, "n": n2, "hc": hc, "eps": 1e-5, "hc_eps": 1e-6, "magnitude": 2.0, "dtype": "bf16"}
    hs, ds, _ = run_both(pl, "hc_pre", [h, fn2, base, scale], [bf(torch.zeros(rows, n2)), torch.zeros(rows, hc)], p2)
    compare("hc_pre with a 16000 row (no LDS)", hs[0], ds[0])
    # hc 3 (6 products): hc_pre_k, which stays for widths other than 4 and 8
    h3 = bf(torch.randn(5, 3 * 1024))
    fn3, base3 = torch.randn(6, 3 * 1024) * 0.01, torch.randn(6) * 0.5
    p3 = {"M": 5, "n": 1024, "hc": 3, "eps": 1e-5, "hc_eps": 1e-6, "magnitude": 2.0, "dtype": "bf16"}
    hs, ds, _ = run_both(pl, "hc_pre", [h3, fn3, base3, scale], [bf(torch.zeros(5, 1024)), torch.zeros(5, 3)], p3)
    compare("hc_pre hc 3 (old form)", hs[0], ds[0])

    # had_quant_act_fp8 (128 rotation + E4M3): codes and scales equal bit for bit
    xq = bf(torch.randn(17, 1536) * 3)
    xq[3, 128:256] = 0                           # zero group: scale 1
    qh, qd = torch.zeros(17, 1536, dtype=torch.uint8), Dev(torch.zeros(17, 1536, dtype=torch.uint8))
    sh, sd = torch.zeros(17, 12), Dev(torch.zeros(17, 12))
    ph = {"M": 17, "n": 1536, "group": 128, "dtype": "bf16"}
    rc_h = pl.call("hy4_ref_had_quant_act_fp8", [xq, F8(qh), sh], ph)
    rc_d = pl.call("hy4_dev_had_quant_act_fp8", [Dev(xq), F8D(qd), sd], ph)
    if rc_h or rc_d:
        report("had_quant_act_fp8 rc", False, f"{rc_h} {rc_d}")
    else:
        qd = qd.to_host()
        sd = sd.to_host()
        report("had_quant_act_fp8 exact codes and scales", torch.equal(qh, qd) and torch.equal(sh, sd),
               f"codes differing {int((qh != qd).sum())}")
    # More cases, all compared bit for bit (exact codes and scales):
    # - wide range, with groups of one large value among small ones to reach the E4M3 subnormals;
    # - n = 640 and 128 (5 and 1 groups: the partial block of 4 waves and a lone group), f32 and bf16.
    def had_case(name, x, min_sub=0):
        M, n = x.shape
        qh2, qd2 = torch.zeros(M, n, dtype=torch.uint8), Dev(torch.zeros(M, n, dtype=torch.uint8))
        sh2, sd2 = torch.zeros(M, n // 128), Dev(torch.zeros(M, n // 128))
        pq = {"M": M, "n": n, "group": 128, "dtype": "bf16" if x.dtype == torch.bfloat16 else "f32"}
        rc_h2 = pl.call("hy4_ref_had_quant_act_fp8", [x, F8(qh2), sh2], pq)
        rc_d2 = pl.call("hy4_dev_had_quant_act_fp8", [Dev(x), F8D(qd2), sd2], pq)
        if rc_h2 or rc_d2:
            report(f"had_quant_act_fp8 {name}: rc", False, f"{rc_h2} {rc_d2}")
            return
        qd2, sd2 = qd2.to_host(), sd2.to_host()
        sub = int((((qh2 & 0x78) == 0) & ((qh2 & 0x07) != 0)).sum())   # exponent 0, nonzero mantissa
        report(f"had_quant_act_fp8 {name}: exact codes and scales",
               torch.equal(qh2, qd2) and torch.equal(sh2, sd2) and sub >= min_sub,
               f"codes differing {int((qh2 != qd2).sum())}, subnormals {sub}")
    # x = H·y/128 per group: the rotation gives back y (H·H = 128·I), so the rotated domain has the
    # wide range of y and reaches the E4M3 subnormals. In f32 so that x is not rounded.
    Hm = torch.ones(1, 1)
    while Hm.shape[0] < 128:
        Hm = torch.cat([torch.cat([Hm, Hm], 1), torch.cat([Hm, -Hm], 1)], 0)
    yw = torch.exp(torch.randn(9, 4, 128) * 3) * torch.sign(torch.randn(9, 4, 128))
    had_case("wide range f32 (y in the rotated domain)", (yw @ Hm / 128).reshape(9, 512), min_sub=100)
    had_case("n=640 f32", torch.randn(3, 640) * 2)
    had_case("n=128 bf16", bf(torch.randn(3, 128)))

    hc_pre_split(pl)
    # attn_paged_sink against a torch reference (the hy4k host does not have one)
    attention(pl)
    latent_attention(pl)
    latent_attention_2g(pl)
    dfly(pl)
    print("OK" if all(results) else "FAIL")
    return 0 if all(results) else 1


def hc_pre_split(pl):
    """hy4_hc_pre's split form (hc_pre_part_k + hc_pre_fin_k) against the one-block-a-row form it
    replaces (hc_pre1_k, `hc_form` 1, what HY4_HC_PRE=old serves): the same bits, x and
    post, at decode, verify and prefill row counts, with and without head, every rows-per-wave
    choice, bf16 and f32 streams, and a row too wide for hc_pre1_k's LDS. Plus the host oracle."""
    n, hc = 3072, 4
    base, scale = torch.randn(2 * hc) * 0.5, torch.tensor([0.7, 0.9])
    p0 = {"n": n, "hc": hc, "eps": 1e-5, "hc_eps": 1e-6, "magnitude": 2.0}

    DTN = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float32: "f32"}

    def card(h, fn, b, sc, rows, head, form, width_n=n, r=None, width_hc=hc):
        p = {**p0, "M": rows, "n": width_n, "hc": width_hc, "head": head, "hc_form": form, "dtype": DTN[h.dtype]}
        if r:
            p["hc_r"] = r
        x = Dev(torch.zeros(rows, width_n, dtype=h.dtype))
        post = None if head else Dev(torch.zeros(rows, width_hc))
        rc = dev_call(pl, "hy4_dev_hc_pre", [Dev(h), Dev(fn), Dev(b), Dev(sc), x, post], p)
        return rc, x.to_host(), post.to_host() if post else None

    def same(name, a, b):
        ok = a[0] == 0 and b[0] == 0 and torch.equal(a[1], b[1]) and (a[2] is None or torch.equal(a[2], b[2]))
        d = int((a[1] != b[1]).sum()) if a[1].shape == b[1].shape else -1
        report(name, ok, f"rc {a[0]} {b[0]}, x differing {d}")

    for rows in (1, 2, 7, 8, 33, 2048):
        h = bf(torch.randn(rows, hc * n))
        fn = torch.randn(2 * hc, hc * n) * 0.01
        old = card(h, fn, base, scale, rows, 0, 1)
        same(f"hc_pre split = old, {rows} rows", card(h, fn, base, scale, rows, 0, 2), old)
        same(f"hc_pre default = old, {rows} rows", card(h, fn, base, scale, rows, 0, 0), old)
        if rows in (7, 33, 2048):
            for r in (1, 2, 4):
                same(f"hc_pre split = old, {rows} rows, {r} rows a wave", card(h, fn, base, scale, rows, 0, 2, r=r), old)
        hd = (fn[:hc].contiguous(), base[:hc].contiguous(), scale[:1].contiguous())
        same(f"hc_pre head split = old, {rows} rows", card(h, *hd, rows, 1, 2), card(h, *hd, rows, 1, 1))
        if rows in (1, 8):
            pr = {**p0, "M": rows, "dtype": "bf16", "hc_form": 2}
            hs, ds, _ = run_both(pl, "hc_pre", [h, fn, base, scale], [bf(torch.zeros(rows, n)), torch.zeros(rows, hc)], pr)
            compare(f"hc_pre split x vs host, {rows} rows", hs[0], ds[0])
            compare(f"hc_pre split post vs host, {rows} rows", hs[1], ds[1])
    # f32 streams, and a width hc_pre1_k reads from memory instead of LDS (hc·n 16000 > 14336)
    h = torch.randn(5, hc * n)
    fn = torch.randn(2 * hc, hc * n) * 0.01
    same("hc_pre split = old, f32, 5 rows", card(h, fn, base, scale, 5, 0, 2), card(h, fn, base, scale, 5, 0, 1))
    n2 = 4000
    h = bf(torch.randn(3, hc * n2))
    fn = torch.randn(2 * hc, hc * n2) * 0.01
    same("hc_pre split = old, 16000 row (no LDS)", card(h, fn, base, scale, 3, 0, 2, width_n=n2),
         card(h, fn, base, scale, 3, 0, 1, width_n=n2))
    # without scratch the split is not taken: the old form, not a refusal
    h = bf(torch.randn(2, hc * n))
    fn = torch.randn(2 * hc, hc * n) * 0.01
    x = Dev(torch.zeros(2, n, dtype=torch.bfloat16))
    rc = pl.call("hy4_dev_hc_pre", [Dev(h), Dev(fn), Dev(base), Dev(scale), x, Dev(torch.zeros(2, hc))],
                 {**p0, "M": 2, "dtype": "bf16", "hc_form": 2})
    same("hc_pre without scratch = old", (rc, x.to_host(), None), card(h, fn, base, scale, 2, 0, 1))
    # n 4000: hc·n 16000 is no multiple of 256, so the chains differ in length (virtual threads
    # 0-127 add 63 values, 128-255 add 62) and the last read-ahead group of a chain is a partial one;
    # 33 rows are 8 waves of 4 and a tail wave of 1, at every rows-a-wave choice
    h = bf(torch.randn(33, hc * n2))
    fn = torch.randn(2 * hc, hc * n2) * 0.01
    old = card(h, fn, base, scale, 33, 0, 1, width_n=n2)
    for r in (1, 2, 4):
        same(f"hc_pre split = old, n 4000, 33 rows, {r} rows a wave",
             card(h, fn, base, scale, 33, 0, 2, width_n=n2, r=r), old)
    # f16 streams (dld/dst's third code), with one and four rows a wave
    h = torch.randn(9, hc * n).half()
    fn = torch.randn(2 * hc, hc * n) * 0.01
    old = card(h, fn, base, scale, 9, 0, 1)
    for r in (1, 4):
        same(f"hc_pre split = old, f16, 9 rows, {r} rows a wave", card(h, fn, base, scale, 9, 0, 2, r=r), old)
    # a scratch that is there but 4 bytes short: the old form, the same bits, and the scratch left as
    # it was (0xff bytes: the split would have written its partial sums over them)
    rows = 5
    h = bf(torch.randn(rows, hc * n))
    fn = torch.randn(2 * hc, hc * n) * 0.01
    x, post = Dev(torch.zeros(rows, n, dtype=torch.bfloat16)), Dev(torch.zeros(rows, hc))
    ts = [Dev(h), Dev(fn), Dev(base), Dev(scale), x, post]
    ps = {**p0, "M": rows, "dtype": "bf16", "hc_form": 2}
    need = pl.scratch_bytes("hy4_dev_hc_pre", ts, ps)
    sc = Dev(torch.full((need,), 0xFF, dtype=torch.uint8))
    rc = pl.call("hy4_dev_hc_pre", ts, ps, scratch=(sc.p, need - 4))
    assert hip.hipDeviceSynchronize() == 0
    same("hc_pre with a short scratch = old", (rc, x.to_host(), post.to_host()), card(h, fn, base, scale, rows, 0, 1))
    report("hc_pre with a short scratch leaves it untouched", torch.equal(sc.to_host(), torch.full((need,), 0xFF,
           dtype=torch.uint8)), f"scratch {need} bytes, {need - 4} given")
    # and the full size does take the split (it writes the scratch), so the case above is a fallback
    sc = Dev(torch.full((need,), 0xFF, dtype=torch.uint8))
    rc = pl.call("hy4_dev_hc_pre", ts, ps, scratch=(sc.p, need))
    assert hip.hipDeviceSynchronize() == 0
    same("hc_pre with the scratch it asks for = old", (rc, x.to_host(), post.to_host()),
         card(h, fn, base, scale, rows, 0, 1))
    left = int((sc.to_host().view(torch.int32) == -1).sum())   # a word still 0xffffffff (a NaN)
    report("hc_pre with the scratch it asks for writes all of it", left == 0, f"{left} words left at 0xffffffff")
    # hc 2 without head: 4 products, the post path of the MIX = 4 instantiation (with head, hc 4
    # covers MIX = 4's x path above, and hc 2 with head is 2 products: hc_pre_k)
    hc2 = 2
    base2, scale2 = torch.randn(2 * hc2) * 0.5, torch.tensor([0.7, 0.9])
    for rows in (1, 8, 33):
        h = bf(torch.randn(rows, hc2 * n))
        fn = torch.randn(2 * hc2, hc2 * n) * 0.01
        old = card(h, fn, base2, scale2, rows, 0, 1, width_hc=hc2)
        for r in ((1, 2, 4) if rows == 33 else (None,)):
            same(f"hc_pre split = old, hc 2, {rows} rows" + (f", {r} rows a wave" if r else ""),
                 card(h, fn, base2, scale2, rows, 0, 2, r=r, width_hc=hc2), old)


def attention(pl):
    """Dense with 2 sequences (cu), with and without window, and the selection form.
    Reference in f64."""
    I32 = torch.int32
    hd, H, bs, P = 256, 4, 4, 64
    kvc = bf(torch.randn(P, H, bs, 2 * hd))
    sinks = torch.randn(H)
    scale = 1.0 / 16

    def ref(qrow, s_pages, ctx, pos, window, h):
        ks, vs = [], []
        for j in range(ctx):
            if j > pos or (window > 0 and pos - j >= window):
                continue
            pg = s_pages[j // bs]
            if pg < 0 or pg >= P:
                continue
            ks.append(kvc[pg, h, j % bs, :hd].double())
            vs.append(kvc[pg, h, j % bs, hd:].double())
        if not ks:
            return torch.zeros(hd, dtype=torch.float64)
        K, Vv = torch.stack(ks), torch.stack(vs)
        lg = (K @ qrow.double()) * scale
        lg = torch.cat([lg, sinks[h:h + 1].double()])
        pr = torch.softmax(lg, 0)[:-1]
        return pr @ Vv

    # dense: sequence 0 with 5 rows (context 23), sequence 1 with 3 rows (context 9)
    cu = torch.tensor([0, 5, 8], dtype=I32)
    su = torch.tensor([23, 9], dtype=I32)
    btab = torch.stack([torch.randperm(P)[:8], torch.randperm(P)[:8]]).to(I32)
    q = bf(torch.randn(8, H * hd))
    for window in (0, 7):
        o = Dev(bf(torch.zeros(8, H * hd)))
        p = {"q_len": 8, "head_dim": hd, "gqa": 1, "causal": 1, "window": window, "n_head": H, "block_size": bs,
             "max_seqs": 2, "max_ctx": 64, "q_dtype": "bf16", "kv_dtype": "bf16", "scale": scale}
        rc = pl.call("hy4_dev_attn_sink", [Dev(q), Dev(kvc), Dev(btab), Dev(su), None, None, Dev(cu), Dev(sinks), o], p)
        if rc:
            report(f"attn dense window={window} rc", False, str(rc))
            continue
        out = o.to_host().float().view(8, H, hd)
        refo = torch.zeros(8, H, hd, dtype=torch.float64)
        for s in range(2):
            for i in range(int(cu[s + 1] - cu[s])):
                r = int(cu[s]) + i
                qlen = int(cu[s + 1] - cu[s])
                pos = int(su[s]) - qlen + i
                for h in range(H):
                    refo[r, h] = ref(q[r].view(H, hd)[h].float(), btab[s].tolist(), int(su[s]), pos, window, h)
        compare(f"attn dense window={window}", refo.to(torch.bfloat16), out.to(torch.bfloat16))
    # selection: 6 rows, each with its list of 10 pages and its seqused
    rows, K = 6, 10
    sel = torch.randint(0, P, (rows, K), dtype=I32)
    sel[2, 3] = -1
    sel[4, 1] = P + 5          # page outside the cache: counts as missing
    sequ = torch.tensor([37, 40, 21, 1, 33, 39], dtype=I32)
    q = bf(torch.randn(rows, H * hd))
    o = Dev(bf(torch.zeros(rows, H * hd)))
    p = {"q_len": 1, "head_dim": hd, "gqa": 1, "causal": 1, "window": 0, "n_head": H, "block_size": bs,
         "max_seqs": 256, "max_ctx": 64, "q_dtype": "bf16", "kv_dtype": "bf16", "scale": scale}
    rc = pl.call("hy4_dev_attn_sink", [Dev(q), Dev(kvc), Dev(sel), Dev(sequ), None, None, None, Dev(sinks), o], p)
    if rc:
        report("attn selection rc", False, str(rc))
        return
    out = o.to_host().float().view(rows, H, hd)
    refo = torch.zeros(rows, H, hd, dtype=torch.float64)
    for r in range(rows):
        for h in range(H):
            refo[r, h] = ref(q[r].view(H, hd)[h].float(), sel[r].tolist(), int(sequ[r]), int(sequ[r]) - 1, 0, h)
    compare("attn selection", refo.to(torch.bfloat16), out.to(torch.bfloat16))


def latent_attention(pl):
    """hy4_mla_attn: card against the host oracle (double), in bf16. Dense varlen with and without
    window, selection, and a long case (2048 keys, 16 heads) that walks many strips."""
    I32 = torch.int32
    L, R, bs = 512, 64, 4

    def case(name, H, P, q_rows, extra, window, ctx_note=""):
        cache = bf(torch.randn(P, 1, bs, L + R))
        sinks = torch.randn(H)
        q = bf(torch.randn(q_rows, H, L) / 8)
        qr = bf(torch.randn(q_rows, H, R) / 8)
        o = bf(torch.zeros(q_rows, H, L))
        p = {"M": q_rows, "heads": H, "latent": L, "rope": R, "block_size": bs, "window": window,
             "scale": 1.0 / 16, "dtype": "bf16"}
        sh, out, _ = run_both(pl, "mla_attn", [q, qr, cache] + extra + [sinks], [o], p)
        compare(f"mla_attn {name}", sh[0], out[0])

    cu = torch.tensor([0, 5, 8], dtype=I32)
    su = torch.tensor([23, 9], dtype=I32)
    P = 64
    bt = torch.stack([torch.randperm(P)[:8], torch.randperm(P)[:8]]).to(I32)
    for w in (0, 7):
        case(f"dense window={w}", 4, P, 8, [bt, su, cu], w)
    sel = torch.randint(0, P, (6, 10), dtype=I32)
    sel[2, 3] = -1
    sel[4, 1] = P + 5
    case("selection", 4, P, 6, [sel, torch.tensor([37, 40, 21, 1, 33, 39], dtype=I32), None], 0)
    # 2-D: q [T, H·L] and q_rope as a view of a q [T, H·256] with rope_pitch 256
    H = 4
    cache = bf(torch.randn(P, 1, bs, L + R))
    sinks = torch.randn(H)
    q2 = bf(torch.randn(8, H * L) / 8)
    qf = bf(torch.randn(8, H * 256) / 8)
    o2 = bf(torch.zeros(8, H * L))
    p2 = {"M": 8, "heads": H, "latent": L, "rope": R, "block_size": bs, "window": 7, "rope_pitch": 256,
          "scale": 1.0 / 16, "dtype": "bf16"}
    sh, out, _ = run_both(pl, "mla_attn", [q2, qf, cache, bt, su, cu, sinks], [o2], p2)
    compare("mla_attn 2-D with rope_pitch", sh[0], out[0])
    P2 = 600
    sel2 = torch.stack([torch.randperm(P2)[:512] for _ in range(3)]).to(I32)
    case("long selection (2048 keys, 16 heads)", 16, P2, 3, [sel2, torch.tensor([2048, 2048, 2000], dtype=I32), None], 0)
    bt3 = torch.randperm(P2)[:160].view(1, 160).to(I32)
    case("window 512 in 640 of context, 32 rows", 8, P2, 32,
         [bt3, torch.tensor([640], dtype=I32), torch.tensor([0, 32], dtype=I32)], 512)

    # Forms 2 and 3 of the card row at Hy4's scale: window edges off a page boundary, mixed batches, bad
    # pages, a sink above every score, peaked scores, and a row giving the same bits alone as in a batch.
    def case2(name, H, cache, q, qr, extra, window, sinks):
        rows = q.shape[0]
        o = bf(torch.zeros(rows, H, L))
        p = {"M": rows, "heads": H, "latent": L, "rope": R, "block_size": bs, "window": window,
             "scale": 1.0 / 16, "dtype": "bf16"}
        sh, out, _ = run_both(pl, "mla_attn", [q, qr, cache] + extra + [sinks], [o], p)
        compare(f"mla_attn {name}", sh[0], out[0])
        return out[0]

    P3 = 700
    cache3 = bf(torch.randn(P3, 1, bs, L + R))
    H3 = 32
    sk3 = torch.randn(H3)
    n3, ctx3 = 1024, 2051            # the chunk starts at position 1027, not a multiple of 4
    bt4 = torch.randperm(P3)[:520].view(1, 520).to(I32)
    q3, qr3 = bf(torch.randn(n3, H3, L) / 8), bf(torch.randn(n3, H3, R) / 8)
    big = case2("32 heads, 1024 rows, window 512, start at 1027", H3, cache3, q3, qr3,
                [bt4, torch.tensor([ctx3], dtype=I32), torch.tensor([0, n3], dtype=I32)], 512, sk3)
    r0 = 777
    # Batch invariance within each form: 3 (more than 32 rows) and 2 (up to 32). The bits differ between
    # the two, as in libr4d, which is batch-invariant only up to 32 rows.
    def chunk(a, b):   # rows a..b-1 of the 1024 chunk as a batch of their own
        return case2(f"rows {a}-{b - 1} of the chunk", H3, cache3, q3[a:b].contiguous(), qr3[a:b].contiguous(),
                     [bt4, torch.tensor([ctx3 - n3 + b], dtype=I32), torch.tensor([0, b - a], dtype=I32)], 512, sk3)
    t40 = chunk(760, 800)
    report("mla_attn batch-invariant, form 3 (window)", torch.equal(big[r0].view(-1), t40[r0 - 760].view(-1)))
    t20, t1 = chunk(770, 790), chunk(777, 778)
    report("mla_attn batch-invariant, form 2 (window)", torch.equal(t20[r0 - 770].view(-1), t1[0].view(-1)))
    # The window's lower edge where it leaves key 0: positions 495-534 (form 3, 40 rows) and 500-519
    # (form 2), so rows at 511, 512 and 513 and every residue mod 4 cross it.
    for a0, n0 in ((495, 40), (500, 20)):
        case2(f"window 512 at positions {a0}-{a0 + n0 - 1}", H3, cache3, q3[:n0].contiguous(),
              qr3[:n0].contiguous(), [bt4, torch.tensor([a0 + n0], dtype=I32), torch.tensor([0, n0], dtype=I32)],
              512, sk3)
    # a mixed batch: one-row (decode) sequences beside a prefill chunk, context under the window, a table
    # of -1 pages only (a row with no key) and out-of-range entries
    btm = torch.stack([torch.randperm(P3)[:200] for _ in range(4)]).to(I32)
    btm[3] = -1
    btm[0, 10] = P3 + 3
    btm[2, 50] = -1
    cum = torch.tensor([0, 1, 2, 66, 67], dtype=I32)
    sum_ = torch.tensor([300, 17, 500, 9], dtype=I32)
    qm, qrm = bf(torch.randn(67, H3, L) / 8), bf(torch.randn(67, H3, R) / 8)
    case2("mixed batch, bad pages, row with no keys", H3, cache3, qm, qrm, [btm, sum_, cum], 512, sk3)
    case2("sink above everything", H3, cache3, qm, qrm, [btm, sum_, cum], 512, torch.full((H3,), 50.0))
    case2("peaked scores", H3, cache3, bf(torch.randn(67, H3, L) * 2), bf(torch.randn(67, H3, R) * 2),
          [btm, sum_, cum], 512, sk3)
    # selection at scale (2048 keys a row, 16 heads, 64 rows) and its invariance
    sel3 = torch.stack([torch.randperm(P3)[:512] for _ in range(64)]).to(I32)
    sq3 = torch.full((64,), 2048, dtype=I32)
    q5, qr5 = bf(torch.randn(64, 16, L) / 8), bf(torch.randn(64, 16, R) / 8)
    big5 = case2("selection, 64 rows of 2048 keys", 16, cache3, q5, qr5, [sel3, sq3, None], 0, sk3[:16].contiguous())
    def sel_chunk(a, b):
        return case2(f"selection, rows {a}-{b - 1}", 16, cache3, q5[a:b].contiguous(), qr5[a:b].contiguous(),
                     [sel3[a:b].contiguous(), sq3[a:b].contiguous(), None], 0, sk3[:16].contiguous())
    s40, s20, s1 = sel_chunk(10, 50), sel_chunk(30, 50), sel_chunk(41, 42)
    report("mla_attn batch-invariant, form 3 (selection)", torch.equal(big5[41].view(-1), s40[31].view(-1)))
    report("mla_attn batch-invariant, form 2 (selection)", torch.equal(s20[11].view(-1), s1[0].view(-1)))
    # a row with few keys (n < 8: empty slices) alone and in a mixed batch
    alone6 = case2("decode row with 9 keys, alone", H3, cache3, qm[66:67].contiguous(), qrm[66:67].contiguous(),
                  [btm[3:4].clone().copy_(btm[0:1]), sum_[3:4].clone().fill_(9), torch.tensor([0, 1], dtype=I32)],
                  512, sk3)
    # the same mixed batch cut to 19 rows (form 2), the 9-key row last
    btm6 = btm.clone(); btm6[3] = btm[0]
    cu19 = torch.tensor([0, 1, 2, 18, 19], dtype=I32)
    q19 = torch.cat([qm[:18], qm[66:67]]).contiguous()
    qr19 = torch.cat([qrm[:18], qrm[66:67]]).contiguous()
    batch6 = case2("mixed batch of 19 rows with that row", H3, cache3, q19, qr19,
                  [btm6, torch.tensor([300, 17, 16, 9], dtype=I32), cu19], 512, sk3)
    report("mla_attn batch-invariant, form 2 (9 keys)", torch.equal(batch6[18].view(-1), alone6.view(-1)))
    # rows outside cu are not written (sentinel output)
    oc = bf(torch.full((6, 4, L), 7.0))
    pc_ = {"M": 6, "heads": 4, "latent": L, "rope": R, "block_size": bs, "window": 0, "scale": 1.0 / 16, "dtype": "bf16"}
    cache4 = bf(torch.randn(64, 1, bs, L + R))
    _, outc, _ = run_both(pl, "mla_attn", [bf(torch.randn(6, 4, L) / 8), bf(torch.randn(6, 4, R) / 8), cache4,
                                      bt, su, torch.tensor([0, 2, 4], dtype=I32), torch.randn(4)], [oc], pc_)
    report("mla_attn does not write the rows outside cu", bool((outc[0].float()[4:] == 7.0).all()))
    # a NaN in a cache row: rows that see it output NaN, as on the host
    cache5 = cache4.clone()
    cache5[int(bt[0, 1]), 0, 2, 5] = float("nan")
    qn = bf(torch.randn(8, 4, L) / 8)
    shn, outn, _ = run_both(pl, "mla_attn", [qn, bf(torch.randn(8, 4, R) / 8), cache5, bt, su,
                                        torch.tensor([0, 5, 8], dtype=I32), torch.randn(4)], [bf(torch.zeros(8, 4, L))],
                       {**pc_, "M": 8})
    report("mla_attn propagates NaN like the host",
           torch.equal(torch.isnan(shn[0].float()), torch.isnan(outn[0].float())) and bool(torch.isnan(outn[0].float()).any()))
    # NaN through form 3 (40 rows, 16 heads)
    bt40 = torch.randperm(P3)[:20].view(1, 20).to(I32)
    c40 = cache3.clone()
    c40[int(bt40[0, 3]), 0, 1, 7] = float("nan")
    shn3, outn3, _ = run_both(pl, "mla_attn", [bf(torch.randn(40, 16, L) / 8), bf(torch.randn(40, 16, R) / 8), c40, bt40,
                                          torch.tensor([60], dtype=I32), torch.tensor([0, 40], dtype=I32),
                                          torch.randn(16)], [bf(torch.zeros(40, 16, L))], {**pc_, "M": 40, "heads": 16})
    report("mla_attn form 3 propagates NaN like the host",
           torch.equal(torch.isnan(shn3[0].float()), torch.isnan(outn3[0].float())) and bool(torch.isnan(outn3[0].float()).any()))
    compare("mla_attn form 3, rows without NaN", torch.nan_to_num(shn3[0].float()).to(torch.bfloat16),
            torch.nan_to_num(outn3[0].float()).to(torch.bfloat16))
    # Infinities, as the oracle treats them: with a visible key, a +inf score or a +inf sink gives NaN
    # (exp(inf - inf)); a -inf sink weighs nothing. Forms 2 (8 rows) and 3 (40 rows).
    for nr, Hh in ((8, 4), (40, 16)):
        qi, qri = bf(torch.randn(nr, Hh, L) / 8), bf(torch.randn(nr, Hh, R) / 8)
        qi[1, 0, 3] = float("inf")
        ski = torch.randn(Hh)
        ski[Hh - 1] = float("inf")
        if Hh > 2:
            ski[2] = float("-inf")
        cui = torch.tensor([0, nr // 2, nr], dtype=I32) if nr == 8 else torch.tensor([0, nr], dtype=I32)
        sui = torch.tensor([30, 9], dtype=I32) if nr == 8 else torch.tensor([60], dtype=I32)
        bti = bt if nr == 8 else torch.randperm(64)[:20].view(1, 20).to(I32)
        shi, outi, _ = run_both(pl, "mla_attn", [qi, qri, cache4, bti, sui, cui, ski], [bf(torch.zeros(nr, Hh, L))],
                           {**pc_, "M": nr, "heads": Hh})
        report(f"mla_attn {nr} rows: +inf and sink +inf give NaN like the host",
               torch.equal(torch.isnan(shi[0].float()), torch.isnan(outi[0].float())))
        finite = ~torch.isnan(shi[0].float())
        compare(f"mla_attn {nr} rows with infinities (finite)", shi[0][finite], outi[0][finite])
    # hy4_mla_project: both modes at Hy4's geometry (head_dim 256, latent 512, 32 heads), with a row
    # count that is not a multiple of 32, in the [M, heads*width] view the plugin passes.
    Hp, dp = 32, 256
    wproj = bf(torch.randn(Hp * 2 * dp, L) / 16)
    for mode, win, wout in (("absorb", dp, L), ("expand", L, dp)):
        for Mp in (70, 300):
            xp = bf(torch.randn(Mp, Hp * win))
            pp = {"M": Mp, "heads": Hp, "head_dim": dp, "latent_dim": L, "mode": mode, "dtype": "bf16"}
            shp, outp, _ = run_both(pl, "mla_project", [xp, wproj], [bf(torch.zeros(Mp, Hp * wout))], pp)
            compare(f"mla_project {mode}, {Mp} rows", shp[0], outp[0])
    # Every visible score -inf (q -inf against an all-positive cache) beside an infinite sink: the
    # oracle's M = max(sink, scores) is infinite and the output NaN; with a finite sink, zeros.
    cpos = bf(torch.rand(64, 1, bs, L + R) + 0.5)
    for nr, Hh in ((8, 4), (40, 16)):
        qn2 = bf(torch.randn(nr, Hh, L) / 8)
        qn2[:, :, 0] = float("-inf")
        sk2 = torch.randn(Hh)
        sk2[0] = float("inf")
        sk2[1] = float("-inf")
        cu2 = torch.tensor([0, nr // 2, nr], dtype=I32) if nr == 8 else torch.tensor([0, nr], dtype=I32)
        su2 = torch.tensor([30, 9], dtype=I32) if nr == 8 else torch.tensor([60], dtype=I32)
        bt2 = bt if nr == 8 else torch.randperm(64)[:20].view(1, 20).to(I32)
        sh2, out2, _ = run_both(pl, "mla_attn", [qn2, bf(torch.zeros(nr, Hh, R)), cpos, bt2, su2, cu2, sk2],
                           [bf(torch.full((nr, Hh, L), 7.0))], {**pc_, "M": nr, "heads": Hh})
        report(f"mla_attn {nr} rows: -inf scores and infinite sink like the host",
               torch.equal(torch.isnan(sh2[0].float()), torch.isnan(out2[0].float())) and
               torch.equal(torch.nan_to_num(sh2[0].float()), torch.nan_to_num(out2[0].float())))
    # form 1 (f32): the fallback for what forms 2 and 3 do not take
    cf = torch.randn(64, 1, bs, L + R)
    qf32, qrf32 = torch.randn(8, 4, L) / 8, torch.randn(8, 4, R) / 8
    shf, outf, _ = run_both(pl, "mla_attn", [qf32, qrf32, cf, bt, su, torch.tensor([0, 5, 8], dtype=I32), torch.randn(4)],
                       [torch.zeros(8, 4, L)], {**pc_, "M": 8, "window": 7, "dtype": "f32"})
    compare("mla_attn form 1 in f32", shf[0], outf[0])

    # vision: two images (grids 4x6 and 6x8, with the line-break column: 4x8 and 6x10)
    MERGE, DIMP, W_, NE = 2, 768, 64, 96
    crds, cu_l = [], [0]
    for gh, gwe in ((4, 8), (6, 10)):
        order = [(bh * 2 + mh, bw * 2 + mw) for bh in range(gh // 2) for bw in range(gwe // 2)
                 for mh in range(2) for mw in range(2)]
        crds.append(torch.tensor([[r, c, gh, gwe] for r, c in order], dtype=I32).T)
        cu_l.append(cu_l[-1] + len(order))
    crd = torch.cat(crds, 1).contiguous()
    cu = torch.tensor(cu_l, dtype=I32)
    Pn = crd.shape[1]
    Pc = 4 * 6 + 6 * 8
    pix = bf(torch.randn(Pn, DIMP))
    sh, out, _ = run_both(pl, "vit_compact", [pix, crd, cu],
                     [bf(torch.zeros(Pc, DIMP)), torch.zeros(Pc, 4, dtype=I32), torch.zeros(3, dtype=I32)],
                     {"M": Pn, "dim": DIMP, "merge": MERGE})
    compare("vit_compact patches", sh[0], out[0])
    compare("vit_compact coordinates", sh[1], out[1])
    compare("vit_compact boundaries", sh[2], out[2])
    tab = bf(torch.randn(16 * 16 + 1, W_))
    x0 = bf(torch.randn(Pc, W_))
    xh, xd = x0.clone(), Dev(x0.clone())
    rh = pl.call("hy4_ref_vit_pos", [xh, tab, sh[1]], {"M": Pc, "n": W_, "side": 16, "dtype": "bf16"})
    rd = pl.call("hy4_dev_vit_pos", [xd, Dev(tab), Dev(sh[1])], {"M": Pc, "n": W_, "side": 16, "dtype": "bf16"})
    compare("vit_pos", xh, xd.to_host()) if rh == 0 and rd == 0 else report("vit_pos rc", False, f"{rh} {rd}")
    y = bf(torch.randn(Pc // 4, NE))
    nl = bf(torch.randn(NE))
    sh3, out3, _ = run_both(pl, "vit_newline", [y, nl, crd, cu, sh[2]], [bf(torch.zeros(Pn // 4, NE))],
                       {"M": Pn // 4, "n": NE, "merge": MERGE, "dtype": "bf16"})
    compare("vit_newline", sh3[0], out3[0])
    # hy4_rope: 3-D positions of an image, and 1-D
    Mr, Hr = 40, 3
    q = bf(torch.randn(Mr, Hr * 256))
    p3 = torch.stack([torch.full((Mr,), 5), 5 + torch.arange(Mr) // 6, 5 + torch.arange(Mr) % 6]).to(I32).contiguous()
    for name, pos, rope_mode in (("3-D", p3, "imrope"), ("1-D", torch.arange(Mr, dtype=I32) + 4000, "gptj")):
        qh, qd = q.clone(), Dev(q.clone())
        pr = {"M": Mr, "head_dim": 256, "n_head": Hr, "theta": 1e7, "scale": 1.0, "mode": rope_mode, "rotary_dim": 64,
              "sections": "12 10 10"}
        rh = pl.call("hy4_ref_rope", [qh, pos], pr)
        rd = pl.call("hy4_dev_rope", [qd, Dev(pos)], pr)
        compare(f"hy4_rope {name}", qh, qd.to_host()) if rh == 0 and rd == 0 else report("hy4_rope rc", False, f"{rh} {rd}")


def latent_attention_2g(pl):
    """hy4_mla_attn's form 2g (opt-in: HY4_MLA_ATTN=g, here the parameter mla_form 6; heads grouped by 16
    on WMMA, history split in slices fixed by the head count, mla_attn2g_k + mla_merge_k) at Hy4's decode geometry: 16 heads over a 2048-key selection and
    32 heads over a 512 window at 4000 of context, 1 to 32 rows. Against the oracle; a row gives the same
    bits alone as in any batch of up to 32; the slice count (`mla_splits`) changes only the rounding;
    rows with fewer keys than slices, no key, NaN and infinities as the oracle; rows outside cu not
    written; without scratch the call runs the per-head form 2 (`mla_form` 5), with its own bits."""
    I32 = torch.int32
    L, R, bs, P = 512, 64, 4, 1200
    cache = bf(torch.randn(P, 1, bs, L + R))

    def params(rows, H, window, **kw):   # form 2g on (mla_form 6, what HY4_MLA_ATTN=g serves) unless kw says
        return {"M": rows, "heads": H, "latent": L, "rope": R, "block_size": bs, "window": window, "scale": 1.0 / 16,
                "dtype": "bf16", "mla_form": 6, **kw}

    def card(q, qr, extra, H, window, sinks, scratch=True, out=None, cache_=None, **kw):
        rows = q.shape[0]
        d = Dev(bf(torch.zeros(rows, H, L)) if out is None else out.clone())
        ins = [Dev(t) if t is not None else None for t in [q, qr, cache if cache_ is None else cache_] + extra + [sinks]]
        p = params(rows, H, window, **kw)
        rc = dev_call(pl, "hy4_dev_mla_attn", ins + [d], p) if scratch else pl.call("hy4_dev_mla_attn", ins + [d], p)
        return rc, d.to_host()

    def oracle(name, q, qr, extra, H, window, sinks, cache_=None, **kw):
        o = bf(torch.zeros(q.shape[0], H, L))
        sh, out, _ = run_both(pl, "mla_attn", [q, qr, cache if cache_ is None else cache_] + extra + [sinks], [o],
                              params(q.shape[0], H, window, **kw))
        compare(f"mla_attn 2g {name}", sh[0], out[0])
        return sh[0], out[0]

    def qs(rows, H, mul=1 / 8):
        return bf(torch.randn(rows, H, L) * mul), bf(torch.randn(rows, H, R) * mul)

    # sparse layers: 16 heads, one selection of 512 pages (2048 keys) a row
    H16, sk16 = 16, torch.randn(16)
    sel = torch.stack([torch.randperm(P)[:512] for _ in range(32)]).to(I32)
    sq = torch.full((32,), 2048, dtype=I32)
    sq[3], sq[7], sq[9] = 1500, 5, 0          # a short history, fewer keys than slices, none at all
    q16, qr16 = qs(32, H16)
    for rows in (1, 4, 8, 32):
        oracle(f"selection, 16 heads, {rows} rows", q16[:rows].contiguous(), qr16[:rows].contiguous(),
               [sel[:rows].contiguous(), sq[:rows].contiguous(), None], H16, 0, sk16)
    _, b32 = card(q16, qr16, [sel, sq, None], H16, 0, sk16)
    same = all(torch.equal(card(q16[r:r + 1].contiguous(), qr16[r:r + 1].contiguous(),
                                [sel[r:r + 1].contiguous(), sq[r:r + 1].contiguous(), None], H16, 0, sk16)[1][0], b32[r])
               for r in (0, 3, 7, 9, 31))
    report("mla_attn 2g batch-invariant (selection, 1 row = in 32)", same)
    report("mla_attn 2g: a row with no key gives zeros", bool((b32[9].float() == 0).all()))
    # window layers: 32 heads, window 512 at 4000 of context; decode rows and a 6-row verify sequence
    H32, sk32 = 32, torch.randn(32)
    btw = torch.randint(0, P, (6, 1000), dtype=I32)
    btw[2, 900] = -1                            # a bad page inside the window
    btw[4, 990] = P + 7
    suw = torch.tensor([4000, 3999, 3001, 600, 2002, 4000], dtype=I32)
    cuw = torch.tensor([0, 1, 2, 3, 4, 5, 11], dtype=I32)   # five decode rows, then a verify of 6
    q32, qr32 = qs(11, H32)
    _, w11 = oracle("window 512, 32 heads, 5 decode rows and a 6-row verify", q32, qr32, [btw, suw, cuw], H32, 512, sk32)
    # the verify's row 3 (position 3997) alone: a sequence of one row at seqused 3998
    _, w1 = card(q32[8:9].contiguous(), qr32[8:9].contiguous(),
                 [btw[5:6].contiguous(), torch.tensor([3998], dtype=I32), torch.tensor([0, 1], dtype=I32)], H32, 512, sk32)
    report("mla_attn 2g: a verify row = the decode row at its position (bits)", torch.equal(w1[0], w11[8]))
    # the slice count only moves the rounding: 1, 3 and 64 slices against the oracle, and different bits
    for S in (1, 3, 64):
        oracle(f"selection, {S} slices", q16[:8].contiguous(), qr16[:8].contiguous(),
               [sel[:8].contiguous(), sq[:8].contiguous(), None], H16, 0, sk16, mla_splits=S)
    # the per-head form 2 (mla_form 5, HY4_MLA_ATTN=h): both close to the oracle, other bits
    _, g8 = card(q16[:8].contiguous(), qr16[:8].contiguous(), [sel[:8].contiguous(), sq[:8].contiguous(), None], H16, 0, sk16)
    rc5, h8 = card(q16[:8].contiguous(), qr16[:8].contiguous(), [sel[:8].contiguous(), sq[:8].contiguous(), None], H16, 0,
                   sk16, mla_form=5)
    nd = int((g8.view(torch.int16) != h8.view(torch.int16)).sum())
    report("mla_attn 2g against the per-head form 2: other rounding only", rc5 == 0 and nd > 0,
           f"{nd} of {g8.numel()} values differ")
    rcn, n8 = card(q16[:8].contiguous(), qr16[:8].contiguous(), [sel[:8].contiguous(), sq[:8].contiguous(), None], H16, 0,
                   sk16, scratch=False)
    report("mla_attn 2g without scratch: the per-head form 2 (bits)", rcn == 0 and torch.equal(n8, h8))
    report("mla_attn 2g: scratch only at form 2g's geometry",
           pl.scratch_bytes("hy4_dev_mla_attn", [], params(8, 16, 0)) == 8 * 16 * 16 * 516 * 4 and
           pl.scratch_bytes("hy4_dev_mla_attn", [], params(8, 8, 0)) == 0 and
           pl.scratch_bytes("hy4_dev_mla_attn", [], params(2048, 32, 0)) == 64 * 32 * 8 * 516 * 4 and
           pl.scratch_bytes("hy4_dev_mla_attn", [], params(8, 16, 0, mla_form=5)) == 0 and
           pl.scratch_bytes("hy4_dev_mla_attn", [], params(8, 16, 0, mla_form=0)) == 0)   # off by default
    # rows outside cu are not written
    _, oc = card(q32[:6].contiguous(), qr32[:6].contiguous(), [btw[:3].contiguous(), suw[:3].contiguous(),
                 torch.tensor([0, 2, 4, 4], dtype=I32)], H32, 512, sk32, out=bf(torch.full((6, H32, L), 7.0)))
    report("mla_attn 2g does not write the rows outside cu", bool((oc.float()[4:] == 7.0).all()) and
           not bool((oc.float()[:4] == 7.0).any()))
    # sink above every score, and -inf
    oracle("sink above everything", q32, qr32, [btw, suw, cuw], H32, 512, torch.full((H32,), 50.0))
    oracle("sink -inf", q32, qr32, [btw, suw, cuw], H32, 512, torch.full((H32,), float("-inf")))
    oracle("peaked scores", *qs(11, H32, 2.0), [btw, suw, cuw], H32, 512, sk32)
    # a NaN in a cache row: the rows that see it give NaN, as the oracle
    cn = cache.clone()
    cn[int(sel[1, 100]), 0, 2, 5] = float("nan")
    shn, outn = oracle("NaN in the cache (finite rows)", q16[:4].contiguous(), qr16[:4].contiguous(),
                       [sel[:4].contiguous(), sq[:4].contiguous(), None], H16, 0, sk16, cache_=cn)
    report("mla_attn 2g propagates NaN like the host",
           torch.equal(torch.isnan(shn.float()), torch.isnan(outn.float())) and bool(torch.isnan(outn.float()).any()))
    # +inf in q and a +inf sink give NaN; a -inf sink weighs nothing
    qi, qri = qs(8, H16)
    qi[1, 0, 3] = float("inf")
    ski = torch.randn(H16)
    ski[H16 - 1], ski[2] = float("inf"), float("-inf")
    shi, outi = oracle("infinities", qi, qri, [sel[:8].contiguous(), sq[:8].contiguous(), None], H16, 0, ski)
    report("mla_attn 2g: +inf and sink +inf give NaN like the host",
           torch.equal(torch.isnan(shi.float()), torch.isnan(outi.float())))
    # every visible score -inf beside an infinite sink (NaN) and a finite one (zeros)
    cpos = bf(torch.rand(P, 1, bs, L + R) + 0.5)
    qn2 = bf(torch.randn(8, H16, L) / 8)
    qn2[:, :, 0] = float("-inf")
    sk2 = torch.randn(H16)
    sk2[0], sk2[1] = float("inf"), float("-inf")
    o7 = bf(torch.full((8, H16, L), 7.0))
    p8 = params(8, H16, 0)
    ins = [qn2, bf(torch.zeros(8, H16, R)), cpos, sel[:8].contiguous(), sq[:8].contiguous(), None, sk2]
    sh2, out2, _ = run_both(pl, "mla_attn", ins, [o7], p8)
    report("mla_attn 2g: -inf scores and infinite sink like the host",
           torch.equal(torch.isnan(sh2[0].float()), torch.isnan(out2[0].float())) and
           torch.equal(torch.nan_to_num(sh2[0].float()), torch.nan_to_num(out2[0].float())))


def dfly(pl):
    """The DFly2 ops (docs/DFLY2.md): hy4_ring_store and hy4_ring_load bit for bit against the host, and
    hy4_mla_attn's block keys (form 2 only) at Hy4's geometry against the host oracle with the form-2
    tolerance, against the union built by hand, causal, with empty histories and with the sink."""
    import dfly_ref as D
    I32, I64 = torch.int32, torch.int64
    torch.manual_seed(21)
    # --- ring: state [slots, 1, 1, ring * W] as a linear group hands it out, Hy4's entry width
    ring, W, slots, rows = 6, 5638, 4, 13
    init = torch.randint(-9, 9, (slots, 1, 1, ring * W), dtype=I32)
    pos = torch.tensor([7, 500, 501, 502, 503, 504, 505, 2047, 2048, 2049, 0, 1, 2], dtype=I32)
    cu = torch.tensor([1, 7, 10, 13], dtype=I32)        # row 0 outside; three sequences
    sidx = torch.tensor([[3, 9], [0, 9], [-1, 9]], dtype=I32)   # [S, pitch]: column 0
    sel = torch.randint(-1, 3000, (rows, 512), dtype=I32)
    nsel = torch.randint(1, 2048, (rows,), dtype=I64)
    hn = bf(torch.randn(6, 3072))
    btab = torch.randint(0, 9000, (3, 700), dtype=I32)
    steps = [("pos", None, torch.arange(rows, dtype=I32), 1, 0, None),
             ("row", nsel, torch.arange(rows, dtype=I32), 1, 3073, None),
             ("row", sel, torch.arange(rows, dtype=I32), 512, 3074, None),
             ("out", hn, torch.tensor([6, 9, 12, 1, 2, 3], dtype=I32), 3072, 1, None),
             ("dense", btab, torch.arange(rows, dtype=I64), 513, 3073 + 513, 4)]
    sh, sd = init.clone(), Dev(init.clone())
    for mode, src, rid, width, off, block in steps:
        p = {"M": len(rid), "ring": ring, "width": width, "offset": off, "seqs": 8, "mode": mode}
        if block:
            p["block"] = block
        rh = pl.call("hy4_ref_ring_store", [src, rid, pos, cu, sidx, sh], p)
        rd = pl.call("hy4_dev_ring_store", [Dev(src) if src is not None else None, Dev(rid), Dev(pos), Dev(cu),
                                            Dev(sidx), sd], p)
        if rh != 0 or rd != 0:
            report(f"ring_store {mode} rc", False, f"host {rh}, card {rd}")
        compare(f"ring_store {mode} (state, bit for bit)", sh, sd.to_host())
    st3 = sh.view(slots, ring, W)
    ref3 = init.clone().view(slots, ring, W)
    for mode, src, rid, width, off, block in steps:
        D.ring_store(ref3, mode, src, rid, pos, cu, sidx[:, :1].reshape(-1), ring, width, off, block or 1)
    report("ring_store host = the contract's model", torch.equal(st3, ref3))
    # loads: the next step (block rows) of the three sequences, one stale, plus the per-sequence form
    pos2 = torch.tensor([506, 507, 508, 2050, 2051, 3, 4], dtype=I32)
    cu2 = torch.tensor([0, 3, 5, 7], dtype=I32)
    for name, ps, cu_, width, off, dt in (("history with cu", pos2, cu2, 513, 3074 - 1, I32),
                                          ("h with cu, f32", pos2, cu2, 3072, 1, torch.float32),
                                          ("stale entry", torch.tensor([900, 901, 902, 2050, 2051, 3, 4], dtype=I32),
                                           cu2, 513, 3586, I32),
                                          ("per sequence", torch.tensor([506, 2050, 3], dtype=I32), None, 513, 3586, I32)):
        n_dst = len(ps)
        dh = torch.full((n_dst, width), 55, dtype=dt)
        p = {"M": n_dst, "ring": ring, "width": width, "offset": off, "seqs": 8}
        dd = Dev(dh.clone())
        rh = pl.call("hy4_ref_ring_load", [ps, cu_, sidx, sh, dh], p)
        rd = pl.call("hy4_dev_ring_load", [Dev(ps), Dev(cu_) if cu_ is not None else None, Dev(sidx), Dev(sh), dd], p)
        if rh != 0 or rd != 0:
            report(f"ring_load {name} rc", False, f"host {rh}, card {rd}")
        out = dd.to_host()
        compare(f"ring_load {name} (bit for bit)", dh.view(I32), out.view(I32))
        if name == "h with cu, f32":
            report("ring_load: h back as stored (row 505's head norm)", torch.equal(out[0], hn[0].float()))
    # a slot outside the state: the host refuses, the card skips that sequence and writes the rest
    bad = torch.tensor([[3, 0], [7, 0], [0, 0]], dtype=I32)
    sd2 = Dev(sh.clone())
    rh = pl.call("hy4_ref_ring_store", [None, torch.arange(rows, dtype=I32), pos, cu, bad, sh.clone()],
                 {"M": rows, "ring": ring, "width": 1, "offset": 0, "seqs": 8, "mode": "pos"})
    rd = pl.call("hy4_dev_ring_store", [None, Dev(torch.arange(rows, dtype=I32)), Dev(pos), Dev(cu), Dev(bad), sd2],
                 {"M": rows, "ring": ring, "width": 1, "offset": 0, "seqs": 8, "mode": "pos"})
    after = sd2.to_host().view(slots, ring, W)
    report("ring_store slot outside state: host refuses, card writes only the valid sequences",
           rh != 0 and rd == 0 and torch.equal(after[1:3], st3[1:3]))
    # two listed rows on one (slot, entry): 13 rows of one sequence, ring 6, listed out of position
    # order; the card keeps the last listed, as the host does (every mode, Hy4's widths)
    pd = torch.arange(700, 713, dtype=I32)
    cud = torch.tensor([0, 13], dtype=I32)
    sxd = torch.tensor([[2, 9]], dtype=I32)
    ridd = torch.tensor([12, 0, 5, 6, 1, 11, 7, 2, 3, 10, 4, 9, 8], dtype=I32)
    btd = torch.randint(0, 9000, (1, 700), dtype=I32)
    hnd = bf(torch.randn(13, 3072))
    dups = [("pos", None, 1, 0, None), ("row", nsel, 1, 3073, None), ("row", sel, 512, 3074, None),
            ("out", hnd, 3072, 1, None), ("dense", btd, 513, 3073 + 513, 4)]
    sh4, sd4 = init.clone(), Dev(init.clone())
    for mode, src, width, off, block in dups:
        p = {"M": 13, "ring": ring, "width": width, "offset": off, "seqs": 8, "mode": mode}
        if block:
            p["block"] = block
        rh = pl.call("hy4_ref_ring_store", [src, ridd, pd, cud, sxd, sh4], p)
        rd = pl.call("hy4_dev_ring_store", [Dev(src) if src is not None else None, Dev(ridd), Dev(pd), Dev(cud),
                                            Dev(sxd), sd4], p)
        if rh != 0 or rd != 0:
            report(f"ring_store duplicates {mode} rc", False, f"host {rh}, card {rd}")
        compare(f"ring_store duplicates {mode}: last listed wins (bit for bit)", sh4, sd4.to_host())
    e4 = sh4.view(slots, ring, W)
    lastp = [[int(pd[r]) for r in ridd.tolist() if int(pd[r]) % ring == e][-1] for e in range(ring)]
    report("ring_store duplicates: each entry holds its last listed row's position",
           [int(e4[2, e, 0]) for e in range(ring)] == lastp)

    # loads that find no entry or no sequence: the fill, on both sides. Rows 0 and 7 outside cu, a
    # sequence without slot, a stale one; fill -1 (default) and fill 0 into f32 (the plugin's h)
    pos5 = torch.tensor([3, 506, 507, 900, 901, 4, 5, 6], dtype=I32)
    cu5 = torch.tensor([1, 3, 5, 7], dtype=I32)
    for name, width, off, dt, fv in (("history, fill -1", 513, 3073, I32, None),
                                     ("h, f32, fill 0", 3072, 1, torch.float32, 0)):
        dh = torch.full((8, width), 55, dtype=dt)
        dd = Dev(dh.clone())
        p = {"M": 8, "ring": ring, "width": width, "offset": off, "seqs": 8}
        if fv is not None:
            p["fill"] = fv
        rh = pl.call("hy4_ref_ring_load", [pos5, cu5, sidx, sh, dh], p)
        rd = pl.call("hy4_dev_ring_load", [Dev(pos5), Dev(cu5), Dev(sidx), Dev(sh), dd], p)
        if rh != 0 or rd != 0:
            report(f"ring_load {name} rc", False, f"host {rh}, card {rd}")
        out = dd.to_host()
        compare(f"ring_load {name}, rows outside cu (bit for bit)", dh.view(I32), out.view(I32))
        f_ = -1 if fv is None else 0
        report(f"ring_load {name}: rows outside cu, no slot and stale get the fill",
               all(bool((out[i].view(I32) == f_).all()) for i in (0, 3, 4, 5, 6, 7)) and
               not bool((out[1].view(I32) == f_).all()))
    # a slot outside the state on a load: the host refuses, the card writes the fill there
    dd = Dev(torch.full((7, 513), 55, dtype=I32))
    rh = pl.call("hy4_ref_ring_load", [pos2, cu2, bad, sh, torch.full((7, 513), 55, dtype=I32)],
                 {"M": 7, "ring": ring, "width": 513, "offset": 3073, "seqs": 8})
    rd = pl.call("hy4_dev_ring_load", [Dev(pos2), Dev(cu2), Dev(bad), Dev(sh), dd],
                 {"M": 7, "ring": ring, "width": 513, "offset": 3073, "seqs": 8})
    out = dd.to_host()
    report("ring_load slot outside state: host refuses, card writes the fill for that sequence",
           rh != 0 and rd == 0 and bool((out[3:5] == -1).all()) and not bool((out[0:3] == -1).all()))

    # --- hy4_mla_attn with block keys: 3 block sequences of 6 rows (18 rows), then 7 (42 rows, over the
    # 32 at which plain calls go to form 3). Selections of 512 pages, histories from 0 to 2048 tokens.
    L, R, bs, P, H = 512, 64, 4, 700, 32
    cache = bf(torch.randn(P, 1, bs, L + R))
    sinks = torch.randn(H)
    scale = 1.0 / 16

    def mk(nrows, sequ=None):
        q, qr = bf(torch.randn(nrows, H, L) / 8), bf(torch.randn(nrows, H, R) / 8)
        sel = torch.stack([torch.randperm(P)[:512] for _ in range(nrows)]).to(I32)
        if sequ is None:
            sequ = torch.randint(0, 2049, (nrows,), dtype=I32)
            sequ[0], sequ[1], sequ[2] = 0, 2048, 1
        bk = bf(torch.randn(nrows, L + R))
        bcu = torch.arange(0, nrows + 1, 6, dtype=I32)
        return q, qr, sel, sequ, bk, bcu

    def attn(name, q, qr, sel, sequ, bk, bcu, sk, out=None):
        nrows = q.shape[0]
        o = bf(torch.zeros(nrows, H, L)) if out is None else out
        p = {"M": nrows, "heads": H, "latent": L, "rope": R, "block_size": bs, "window": 0, "scale": scale,
             "max_seqs": 64, "dtype": "bf16"}
        shh, outd, _ = run_both(pl, "mla_attn", [q, qr, cache, sel, sequ, None, sk], [o, bk, bcu], p)
        if name:
            compare(f"mla_attn bkeys {name}", shh[0], outd[0])
        return shh[0], outd[0]

    for nrows in (18, 42):
        q, qr, sel, sequ, bk, bcu = mk(nrows)
        hst, crd = attn(f"{nrows} rows (form 2)", q, qr, sel, sequ, bk, bcu, sinks)
        if nrows == 18:
            # rows 0-5 against the union built by hand in double
            want, _ = D.union_attn(q[:6], qr[:6], cache, sel[:6], sequ[:6], bk, bcu[:2], sinks, scale, bs)
            compare("mla_attn bkeys: rows 0-5 = the union by hand", bf(want), crd[:6])
            # the union laid out as a plain selection (history tokens, then the block keys, in new pages):
            # the same keys in the same order, so form 2 without block keys gives the same bits
            r = 4
            n_u = int(sequ[r]) + (r - 0) + 1
            pages = (n_u + bs - 1) // bs
            cu_cache = bf(torch.zeros(P + pages, 1, bs, L + R))
            cu_cache[:P] = cache
            tokens = [cache[int(sel[r, j // bs]), 0, j % bs] for j in range(int(sequ[r]))] + [bk[b] for b in range(0, r + 1)]
            for t, row in enumerate(tokens):
                cu_cache[P + t // bs, 0, t % bs] = row
            sel_u = torch.arange(P, P + pages, dtype=I32).view(1, -1)
            o1 = bf(torch.zeros(1, H, L))
            p1 = {"M": 1, "heads": H, "latent": L, "rope": R, "block_size": bs, "window": 0, "scale": scale, "dtype": "bf16"}
            d1 = Dev(o1.clone())
            # with the scratch the engine gives, as the batch above (the per-head form by default; form 2g
            # under HY4_MLA_ATTN=g, which latent_attention_2g covers)
            rc1 = dev_call(pl, "hy4_dev_mla_attn", [Dev(q[r:r + 1].contiguous()), Dev(qr[r:r + 1].contiguous()),
                                                    Dev(cu_cache), Dev(sel_u), Dev(torch.tensor([n_u], dtype=I32)), None,
                                                    Dev(sinks), d1], p1)
            report("mla_attn bkeys: a row = form 2 over its union as a plain selection (bits)",
                   rc1 == 0 and torch.equal(d1.to_host()[0].view(-1), crd[r].view(-1)))
            # block causality: later block keys changed, earlier rows keep their bits; the changed ones move
            bk2 = bk.clone()
            bk2[3] = bf(torch.randn(L + R))
            bk2[13] = bf(torch.randn(L + R))
            _, crd2 = attn(None, q, qr, sel, sequ, bk2, bcu, sinks)
            # bk row 3 is seen by rows 3-5 and bk row 13 by rows 13-17 (sequence 2 is rows 12-17)
            same = torch.equal(crd2[:3], crd[:3]) and torch.equal(crd2[6:13], crd[6:13])
            moved = all(not torch.equal(crd2[r_], crd[r_]) for r_ in (3, 5, 13, 17))
            report("mla_attn bkeys: block causality (card, bits)", same and moved)
            # a row gives the same bits alone (its block sequence cut to it) as in the batch: row 2 of
            # sequence 0 sees bk rows 0..2
            _, crd1 = attn(None, q[:3].contiguous(), qr[:3].contiguous(), sel[:3].contiguous(), sequ[:3].contiguous(),
                           bk[:3].contiguous(), torch.tensor([0, 3], dtype=I32), sinks)
            report("mla_attn bkeys: a row alone = in the batch (bits)", torch.equal(crd1[2].view(-1), crd[2].view(-1)))
    # empty history everywhere (seqused 0: only the block keys), against the host and by hand
    q, qr, sel, sequ, bk, bcu = mk(12, torch.zeros(12, dtype=I32))
    _, crd = attn("empty history (seqused 0)", q, qr, sel, sequ, bk, bcu, sinks)
    want, _ = D.union_attn(q, qr, cache, sel, sequ, bk, bcu, sinks, scale, bs)
    compare("mla_attn bkeys: empty history = by hand", bf(want), crd)
    # the sink: above every score, and -inf (weighs nothing)
    q, qr, sel, sequ, bk, bcu = mk(12)
    attn("sink above everything", q, qr, sel, sequ, bk, bcu, torch.full((H,), 50.0))
    attn("sink -inf", q, qr, sel, sequ, bk, bcu, torch.full((H,), float("-inf")))
    _, crd = attn("random sink", q, qr, sel, sequ, bk, bcu, sinks)
    want, _ = D.union_attn(q[6:], qr[6:], cache, sel[6:], sequ[6:], bk[6:], torch.tensor([0, 6], dtype=I32), sinks,
                           scale, bs)
    compare("mla_attn bkeys: rows 6-11 with the sink = by hand", bf(want), crd[6:])
    # rows outside bcu are not written (a span that starts at row 2)
    _, crd = attn(None, q, qr, sel, sequ, bk, torch.tensor([2, 6, 12], dtype=I32), sinks, out=bf(torch.full((12, H, L), 7.0)))
    report("mla_attn bkeys: rows outside bcu not written", bool((crd[:2].float() == 7.0).all()) and
           not bool((crd[2].float() == 7.0).all()))
    # forms 1 and 3 have no block keys: an f32 call (form 1's) is refused on the card, served on the host
    o32 = torch.zeros(6, 4, L)
    args32 = [torch.randn(6, 4, L) / 8, torch.randn(6, 4, R) / 8, torch.randn(64, 1, bs, L + R),
              torch.randint(0, 64, (6, 8), dtype=I32), torch.randint(0, 32, (6,), dtype=I32), None, torch.randn(4)]
    bk32 = [torch.randn(6, L + R), torch.tensor([0, 6], dtype=I32)]
    p32 = {"M": 6, "heads": 4, "latent": L, "rope": R, "block_size": bs, "window": 0, "scale": scale, "dtype": "f32"}
    rh = pl.call("hy4_ref_mla_attn", args32 + [o32] + bk32, p32)
    rd = pl.call("hy4_dev_mla_attn", [Dev(t) if t is not None else None for t in args32 + [o32] + bk32], p32)
    report("mla_attn bkeys outside form 2 (f32): refused on the card, served on the host", rh == 0 and rd != 0,
           f"host {rh}, card {rd}")


if __name__ == "__main__":
    sys.exit(main())
