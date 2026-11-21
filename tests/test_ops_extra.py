#!/usr/bin/env python3
"""Cases tests/test_ops.py does not cover (the review of plan step 3, F1-F10): Engram's FP8 path
from the raw table, errors on bad indices or shapes, several sequences, sidx with pitch, selection
on all rows with order and padding, tie at the cutoff, score with nc < blocks, rope in BF16 against
HF."""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import test_ops as T  # noqa: E402  (reuses plugin, config and helpers)
from transformers.models.hy_v4_flash import modeling_hy_v4_flash as M  # noqa: E402

pl, cfg, w, l2, report, REF, SNAP, IDX = T.pl, T.cfg, T.w, T.l2, T.report, T.REF, T.SNAP, T.IDX
import rad_ctypes as RC  # noqa: E402
RC.DT["e4m3"] = 4   # RAD_F8E4M3
I32 = torch.int32


def test_fp8_from_table():
    from safetensors import safe_open
    v = T.load_ref(REF / "corto.L01.safetensors")
    ids = v["engram.ids"]
    key = "model.layers.1.engram_layer.multi_head_embedding.embedding.weight"
    with safe_open(str(SNAP / IDX[key]), framework="pt") as f:
        sl = f.get_slice(key)
        rows = torch.stack([sl[int(i):int(i) + 1][0] for i in ids.flatten().tolist()]).float()
    t = ids.shape[0]
    rows = rows.reshape(t, -1)
    wn = torch.stack([w(f"model.layers.1.engram_layer.lookup_per_head_norms.{h}.weight").float() for h in range(16)])
    y = torch.empty_like(rows)
    rc = pl.call("hy4_ref_rows_rmsnorm", [rows, wn, y], {"M": t, "heads": 16, "d": 96, "fp8": 1, "eps": 1.1920929e-07, "dtype": "f32"})
    e = l2(v["engram.rows_norm"].reshape(t, -1), y)
    report("rows_rmsnorm fp8=1 from the raw table", rc == 0 and e < 1e-6, f"L2 {e:.2e}")


def test_negatives():
    p = {"engram_layers": 1, "heads": 16, "ngram": 3, "pad": 5, "seqs": 4}
    wmap = torch.arange(10, dtype=I32).view(1, -1)
    mult = torch.tensor([[3, 5, 7]], dtype=torch.int64)
    vsz = torch.full((1, 16), 13, dtype=torch.int64)
    off = torch.zeros(1, 16, dtype=torch.int64)
    tok = torch.tensor([1, 2, 3], dtype=I32)
    ids = torch.empty(1, 3, 16, dtype=I32)
    st = torch.zeros(2, 2, dtype=I32)
    # cu of one sequence with a ceiling of 4: valid (the count comes from cu)
    rc = pl.call("hy4_ref_ngram_ids", [tok, torch.arange(3, dtype=I32), torch.tensor([0, 3], dtype=I32), st,
                 torch.tensor([0], dtype=I32), wmap, mult, vsz, off, ids], {"M": 3, **p})
    report("ngram_ids with seqs larger than cu: accepts", rc == 0)
    rc = pl.call("hy4_ref_ngram_ids", [tok, torch.arange(3, dtype=I32), torch.tensor([0, 3], dtype=I32), st,
                 torch.tensor([5], dtype=I32), wmap, mult, vsz, off, ids], {"M": 3, **p})
    report("ngram_ids with slot outside state: rejects", rc != 0)
    rc = pl.call("hy4_ref_ngram_ids", [tok, torch.arange(3, dtype=I32), torch.tensor([0, 9], dtype=I32), st,
                 torch.tensor([0], dtype=I32), wmap, mult, vsz, off, ids], {"M": 3, **p})
    report("ngram_ids with cu past the rows: rejects", rc != 0)
    # ngram=3 and heads=16 valid; only the state width (1 instead of 2) is wrong
    rc = pl.call("hy4_ref_ngram_ids", [tok, torch.arange(3, dtype=I32), torch.tensor([0, 3], dtype=I32),
                 torch.zeros(2, 1, dtype=I32), torch.tensor([0], dtype=I32), wmap, mult, vsz, off, ids], {"M": 3, **p})
    report("ngram_ids with state narrower than ngram-1: rejects", rc != 0)
    # pool with a page outside the cache and with repeated pages
    n = 128
    stg = torch.randn(8, n)
    wv, wg = torch.randn(n, n), torch.randn(n, n)
    ape, wn = torch.randn(4, n), torch.ones(n)
    cache = torch.zeros(2, n)
    pp = {"n": n, "ratio": 4, "rotary_dim": 64, "theta": 1e7, "eps": 1e-5, "dtype": "f32"}
    rc = pl.call("hy4_ref_idx_pool", [stg, wv, wg, ape, wn, torch.tensor([0, 4], dtype=I32), torch.tensor([0, 4], dtype=I32), cache], {"M": 2, **pp})
    report("idx_pool with page outside the cache: rejects", rc != 0 and cache.abs().sum() == 0)
    rc = pl.call("hy4_ref_idx_pool", [stg, wv, wg, ape, wn, torch.tensor([1, 1], dtype=I32), torch.tensor([0, 4], dtype=I32), cache], {"M": 2, **pp})
    report("idx_pool with repeated pages: rejects", rc != 0)
    rc = pl.call("hy4_ref_idx_pool", [stg, wv, wg, ape, wn, torch.tensor([0, 1], dtype=I32), torch.tensor([0, 4], dtype=I32), cache], {"M": 2, **pp, "block_size": 16})
    report("idx_pool with block_size != ratio: rejects", rc != 0)
    # select with the own group outside bt
    sc = torch.randn(1, 2)
    sel, ns, us = torch.empty(1, 4, dtype=I32), torch.empty(1, dtype=I32), torch.empty(1, dtype=I32)
    rc = pl.call("hy4_ref_idx_select", [sc, torch.tensor([[7, 8]], dtype=I32), torch.tensor([2], dtype=I32),
                 torch.tensor([17], dtype=I32), torch.tensor([0, 1], dtype=I32), sel, ns, us], {"M": 1, "blocks": 2, "topk": 4, "ratio": 4, "seqs": 1})
    report("idx_select with own group outside bt: rejects", rc != 0)
    sel.fill_(-5)
    rc = pl.call("hy4_ref_idx_select", [sc, torch.tensor([[7, 8, 9, 10, 11]], dtype=I32), torch.tensor([-1], dtype=I32),
                 torch.tensor([5], dtype=I32), torch.tensor([0, 1], dtype=I32), sel, ns, us], {"M": 1, "blocks": 2, "topk": 4, "ratio": 4, "seqs": 1})
    report("idx_select with negative nc: rejects without writing", rc != 0 and bool((sel == -5).all()))
    # hc_pre with wrong h width
    rc = pl.call("hy4_ref_hc_pre", [torch.randn(3, 10), torch.randn(4, 12), torch.randn(4), torch.randn(2),
                 torch.empty(3, 3), torch.empty(3, 2)], {"M": 3, "n": 3, "hc": 2, "dtype": "f32"})
    report("hc_pre with h of wrong width: rejects", rc != 0)


def test_ngram_multi_seq():
    layer = cfg.engram_layer_ids[0] - 1
    toks = json.loads((T.ROOT / "ref/prompts.json").read_text())["prompts"]
    a, b = toks["tools"][:40], toks["corto"][:33]
    ra, eng, sizes = T.ids_hf(layer, a)
    rb, _, _ = T.ids_hf(layer, b)
    wmap = eng.lookup_table.to(I32).view(1, -1)
    mult = eng.hash_multipliers.view(1, -1)
    vsz = torch.tensor([sizes], dtype=torch.int64)
    off = torch.tensor([0] + sizes[:-1]).cumsum(0).view(1, -1)
    p = {"engram_layers": 1, "heads": 16, "ngram": 3, "pad": cfg.engram_pad_id, "seqs": 8}
    st = torch.full((5, 2), -9, dtype=I32)
    sidx = torch.tensor([[3, 0], [1, 4]], dtype=I32)  # [S, pitch 2]: column 0 = committed slot
    oa, ob = torch.empty(len(a), 16, dtype=torch.long), torch.empty(len(b), 16, dtype=torch.long)
    ia = ib = 0
    chunks = [(3, 2), (1, 5), (7, 1), (4, 4)]
    k = 0
    while ia < len(a) or ib < len(b):
        ca, cb = chunks[k % 4]; k += 1
        ca, cb = min(ca, len(a) - ia), min(cb, len(b) - ib)
        tok = torch.tensor(a[ia:ia + ca] + b[ib:ib + cb], dtype=I32)
        pos = torch.tensor(list(range(ia, ia + ca)) + list(range(ib, ib + cb)), dtype=I32)
        ids = torch.empty(1, ca + cb, 16, dtype=I32)
        rc = pl.call("hy4_ref_ngram_ids", [tok, pos, torch.tensor([0, ca, ca + cb], dtype=I32), st, sidx,
                     wmap, mult, vsz, off, ids], {"M": ca + cb, **p})
        oa[ia:ia + ca] = ids[0, :ca]
        ob[ib:ib + cb] = ids[0, ca:]
        ia, ib = ia + ca, ib + cb
    others = st[[0, 2, 4]].eq(-9).all()
    report("ngram_ids with 2 sequences, chunks and sidx of pitch 2", rc == 0 and torch.equal(oa, ra) and torch.equal(ob, rb) and bool(others))


def test_select_full():
    v = T.load_ref(REF / "largo.L01.safetensors")
    sc = v["index.logits"].float().contiguous()
    t, G = sc.shape
    bt = (torch.arange(G + 1, dtype=I32) * 3 + 7).view(1, -1)
    sel, ns, us = torch.empty(t, 512, dtype=I32), torch.empty(t, dtype=I32), torch.empty(t, dtype=I32)
    rc = pl.call("hy4_ref_idx_select", [sc, bt, torch.tensor([G], dtype=I32), torch.arange(t, dtype=I32),
                 torch.tensor([0, t], dtype=I32), sel, ns, us], {"M": t, "blocks": G, "topk": 512, "ratio": 4, "seqs": 1})
    hf = v["index.selection"].long()
    bad = 0
    for r in range(2048, t):
        k = int(ns[r])
        pages = sel[r, :k].long()
        groups = (pages - 7) // 3
        ok = set(groups[:-1].tolist()) == set(hf[r][:-1][hf[r][:-1] >= 0].tolist()) and int(groups[-1]) == r // 4
        ok &= bool((groups[:-1].diff() > 0).all()) and bool((sel[r, k:] == -1).all())
        ok &= int(us[r]) == (k - 1) * 4 + r % 4 + 1
        bad += not ok
    report("idx_select on all rows with selection", rc == 0 and bad == 0, f"bad rows {bad}")
    # tie at the cutoff: with topk 3 and 4 candidates [1, 5, 5, 5] the lower index wins
    sc = torch.tensor([[1.0, 5.0, 5.0, 5.0, 0.0]])
    sel, ns, us = torch.empty(1, 3, dtype=I32), torch.empty(1, dtype=I32), torch.empty(1, dtype=I32)
    rc = pl.call("hy4_ref_idx_select", [sc, torch.arange(6, dtype=I32).view(1, -1), torch.tensor([5], dtype=I32),
                 torch.tensor([17], dtype=I32), torch.tensor([0, 1], dtype=I32), sel, ns, us], {"M": 1, "blocks": 5, "topk": 3, "ratio": 4, "seqs": 1})
    report("idx_select breaks ties toward the lower index", rc == 0 and sel[0].tolist() == [1, 2, 4])
    # candidates with a negative page or score -inf do not count
    sc = torch.tensor([[float("-inf"), 1.0, 2.0, 0.0]])
    sel, ns, us = torch.empty(1, 6, dtype=I32), torch.empty(1, dtype=I32), torch.empty(1, dtype=I32)
    rc = pl.call("hy4_ref_idx_select", [sc, torch.tensor([[-1, 11, 12, 13]], dtype=I32), torch.tensor([3], dtype=I32),
                 torch.tensor([13], dtype=I32), torch.tensor([0, 1], dtype=I32), sel, ns, us], {"M": 1, "blocks": 4, "topk": 6, "ratio": 4, "seqs": 1})
    report("idx_select ignores negative pages and -inf", rc == 0 and sel[0].tolist() == [11, 12, 13, -1, -1, -1] and int(ns[0]) == 3 and int(us[0]) == 10)


def test_score_two_seqs():
    torch.manual_seed(3)
    n, H = 128, 16
    cache = torch.randn(20, n)
    q = torch.randn(9, H, n)
    wh = torch.randn(9, H)
    pos = torch.tensor([30, 31, 32, 33, 50, 51, 52, 53, 54], dtype=I32)
    bt = torch.stack([torch.randperm(20)[:10], torch.randperm(20)[:10]]).to(I32)
    nc = torch.tensor([7, 10], dtype=I32)
    cu = torch.tensor([0, 4, 9], dtype=I32)
    score = torch.empty(9, 10)
    rc = pl.call("hy4_ref_idx_score", [q, wh, pos, cache, bt, cu, nc, score],
                 {"M": 9, "n": n, "heads": H, "rotary_dim": 64, "theta": 1e7, "blocks": 10, "seqs": 4, "dtype": "f32"})
    ref = torch.full((9, 10), float("-inf"))
    for s in range(2):
        for r in range(int(cu[s]), int(cu[s + 1])):
            qr = torch.stack([T.rope_hf(q[r:r + 1, h], pos[r:r + 1].long())[0] for h in range(H)])
            for g in range(int(nc[s])):
                k = cache[int(bt[s, g])]
                ref[r, g] = (wh[r] * (16 ** -0.5 * 128 ** -0.5) * (qr @ k).clamp_min(0)).sum()
    finite = torch.isfinite(ref)
    e = l2(ref[finite], score[finite])
    report("idx_score with 2 sequences and nc < blocks", rc == 0 and e < 1e-5 and torch.equal(finite, torch.isfinite(score)), f"L2 {e:.2e}")
    # The architecture's shapes: 2-D q [M, heads·n] and a plane wider than bt.
    width = torch.full((9, 12), 7.0)
    p2 = {"M": 9, "n": n, "heads": H, "rotary_dim": 64, "theta": 1e7, "blocks": 12, "seqs": 4, "dtype": "f32"}
    rc = pl.call("hy4_ref_idx_score", [q.reshape(9, H * n).contiguous(), wh, pos, cache, bt, cu, nc, width], p2)
    report("idx_score with 2-D q and plane wider than bt", rc == 0 and torch.equal(width[:, :10], score)
           and bool(torch.isinf(width[:, 10:]).all()))
    other = torch.full((9, 12), 7.0)
    rc = pl.call("hy4_ref_idx_score", [q, wh, pos, cache, bt, cu, torch.tensor([7, 11], dtype=I32), other], p2)
    report("idx_score with nc larger than bt width: rejects without writing", rc != 0 and bool(other.eq(7.0).all()))


def test_rope_bf16():
    torch.manual_seed(4)
    n = 128
    stg = torch.randn(8, n).bfloat16()
    wv, wg = torch.randn(n, n) * 0.1, torch.randn(n, n) * 0.1
    ape, wn = torch.randn(4, n), torch.ones(n)
    cache = torch.zeros(2, n).bfloat16()
    bpos = torch.tensor([149996, 150000], dtype=I32)
    rc = pl.call("hy4_ref_idx_pool", [stg, wv, wg, ape, wn, torch.tensor([0, 1], dtype=I32), bpos, cache],
                 {"M": 2, "n": n, "ratio": 4, "rotary_dim": 64, "theta": 1e7, "eps": 1e-5, "dtype": "bf16"})
    ix = M.HYV4FlashCompressedIndexer(cfg)
    with torch.no_grad():
        ix.key_value_projection.weight.copy_(wv); ix.key_gate_projection.weight.copy_(wg)
        ix.intra_group_ape.copy_(ape); ix.key_norm.weight.copy_(wn)
        # As it ends up after loading in BF16: pooling and rope (inv_freq) stay in FP32; an
        # ix.to(bfloat16) would also lower inv_freq, which HF's loading does not do.
        gk = ix.pool_groups(stg.unsqueeze(0))[0]
        rot = M.HYV4FlashRotaryEmbedding(cfg)
        p3 = bpos.long().view(1, 1, -1).expand(3, 1, -1)
        c, s = rot(gk.unsqueeze(0), p3, layer_type="indexed_attention")
        ref = M._apply_partial_rotary_pos_emb(gk.unsqueeze(0).unsqueeze(2), c, s, 64, unsqueeze_dim=2)[0, :, 0]
    e = l2(ref.float(), cache.float())
    equal_frac = (ref == cache).float().mean().item()
    report("idx_pool in BF16 vs HF (rope with cos/sin in BF16)", rc == 0 and torch.equal(ref, cache), f"L2 {e:.2e}, equal {equal_frac:.0%}")


def test_ngram_shapes():
    """New shapes of hy4_ngram_ids (findings F12, F13): 2-D ids [E·R, heads] with R > M, weights of
    2 layers with the flattened map, state of a linear group [slots, 1, 1, 2], and the rejections."""
    layer = cfg.engram_layer_ids[0] - 1
    a = json.loads((T.ROOT / "ref/prompts.json").read_text())["prompts"]["tools"][:5]
    ra, eng, sizes = T.ids_hf(layer, a)
    E, R, M = 2, 9, len(a)
    wmap = eng.lookup_table.to(I32).repeat(E)                      # [E·V] flattened
    mult = eng.hash_multipliers.view(1, -1).repeat(E, 1)
    vsz = torch.tensor([sizes] * E, dtype=torch.int64)
    off = torch.tensor([0] + sizes[:-1]).cumsum(0).view(1, -1).repeat(E, 1)
    p = {"engram_layers": E, "heads": 16, "ngram": 3, "pad": cfg.engram_pad_id, "seqs": 1, "M": M}
    tok, pos, cu = torch.tensor(a, dtype=I32), torch.arange(M, dtype=I32), torch.tensor([0, M], dtype=I32)

    def call_ids(ids, st):
        return pl.call("hy4_ref_ngram_ids", [tok, pos, cu, st, torch.tensor([0], dtype=I32), wmap, mult, vsz,
                                             off, ids], p)

    ids = torch.full((E * R, 16), -7, dtype=I32)
    rc = call_ids(ids, torch.full((2, 1, 1, 2), -9, dtype=I32))
    t = ids.view(E, R, 16)
    good = rc == 0 and all(torch.equal(t[e, :M].long(), ra) for e in range(E)) and bool(t[:, M:].eq(-7).all())
    report("ngram_ids with ids [E·R, heads], R > M, flattened map and state [slots,1,1,2]", good)
    rc = call_ids(torch.empty(E * R + 1, 16, dtype=I32), torch.zeros(2, 1, 1, 2, dtype=I32))
    report("ngram_ids with id rows not a multiple of E: rejects", rc != 0)
    rc = call_ids(torch.empty(E * (M - 1), 16, dtype=I32), torch.zeros(2, 1, 1, 2, dtype=I32))
    report("ngram_ids with R < M: rejects", rc != 0)
    st = torch.full((2, 3, 2), -9, dtype=I32)
    rc = call_ids(torch.empty(E * R, 16, dtype=I32), st)
    report("ngram_ids with state middle axis 3: rejects without writing", rc != 0 and bool(st.eq(-9).all()))
    p3 = dict(p)
    rc = pl.call("hy4_ref_ngram_ids", [tok, pos, cu, torch.zeros(2, 1, 1, 2, dtype=I32), torch.tensor([0], dtype=I32),
                                       wmap.view(E, -1, 1).expand(E, wmap.numel() // E, 2).contiguous(), mult, vsz, off,
                                       torch.empty(E * R, 16, dtype=I32)], p3)
    report("ngram_ids with rank-3 map: rejects", rc != 0)


def test_had_quant_torch():
    """had_quant_act_fp8 (host oracle) against a torch reference: 128-point FWHT by the Sylvester
    matrix in natural order, scale amax/448 (1 if the group is zero), float8_e4m3fn. Wide range
    (subnormals and saturation in the rotated domain), integers and eighths (ties) and a zero group."""
    H = torch.ones(1, 1)
    while H.shape[0] < 128:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    g = torch.Generator().manual_seed(11)
    cases = {
        "wide range": torch.exp(torch.randn(8, 512, generator=g) * 3) * torch.sign(torch.randn(8, 512, generator=g)),
        "eighths": torch.randint(-64, 64, (8, 512), generator=g).float() / 8,
    }
    cases["wide range"][2, 128:256] = 0
    for name, x in cases.items():
        xb = x.to(torch.bfloat16)
        q = torch.zeros(8, 512, dtype=torch.uint8)
        sc = torch.zeros(8, 4)
        rc = pl.call("hy4_ref_had_quant_act_fp8", [xb, T.F8(q) if hasattr(T, "F8") else F8(q), sc],
                     {"M": 8, "n": 512, "group": 128, "dtype": "bf16"})
        # butterfly in f32, the same summation order as the oracle (x @ H sums in another order and
        # moves the last ulp of amax with this range); the H matrix only checks the natural order
        v = xb.float().view(8, 4, 128).clone()
        h = 1
        while h < 128:
            v = v.view(8, 4, 128 // (2 * h), 2, h)
            a_, b_ = v[..., 0, :].clone(), v[..., 1, :].clone()
            v = torch.stack([a_ + b_, a_ - b_], dim=-2).view(8, 4, 128)
            h *= 2
        assert torch.allclose(v, xb.float().view(8, 4, 128) @ H, rtol=1e-4, atol=1e-3 * float(v.abs().max()))
        am = v.abs().amax(-1)
        esc = torch.where(am > 0, am * (1.0 / 448.0), torch.ones_like(am))
        codes = (v / esc.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8).view(8, 512)
        report(f"had_quant_act_fp8 vs torch ({name})", rc == 0 and torch.equal(q, codes) and torch.equal(sc, esc),
               f"differing codes {int((q != codes).sum())}")


class F8:
    """A uint8 tensor passed to the plugin as E4M3."""

    def __init__(self, t):
        self.t, self.dtype, self.shape = t, "e4m3", t.shape

    def data_ptr(self):
        return self.t.data_ptr()

    def dim(self):
        return self.t.dim()

    def stride(self, i):
        return self.t.stride(i)


def test_ring():
    """hy4_ring_store and hy4_ring_load (the DFly2 ring) against tests/dfly_ref.py: every mode, a cu span
    that does not start at 0, a sequence without slot, both state layouts, the -1 cases of the load and
    the refusals (state untouched)."""
    import dfly_ref as D
    torch.manual_seed(11)
    ring, W, slots = 6, 24, 3
    rows = 9
    pos = torch.tensor([99, 10, 11, 12, 13, 3, 4, 5, 6], dtype=I32)   # row 0 outside cu
    cu = torch.tensor([1, 5, 9], dtype=I32)
    sidx = torch.tensor([2, 0], dtype=I32)
    sel = torch.randint(-1, 500, (rows, 7), dtype=I32)
    nsel = torch.randint(0, 2000, (rows,), dtype=torch.int64)
    h_bf = torch.randn(2, 5).to(torch.bfloat16)
    h_f32 = torch.randn(2, 3)
    btab = torch.randint(0, 900, (2, 5), dtype=I32)
    steps = [  # (mode, src, row_ids, width, offset, block)
        ("pos", None, torch.arange(rows, dtype=I32), 1, 0, None),
        ("row", sel, torch.arange(rows, dtype=I32), 7, 1, None),
        ("row", nsel, torch.arange(rows, dtype=I32), 1, 8, None),
        ("out", h_bf, torch.tensor([4, 8], dtype=I32), 5, 9, None),
        ("out", h_f32, torch.tensor([4, 8], dtype=I32), 3, 14, None),
        ("dense", btab, torch.tensor([2, 4, 5, 8], dtype=torch.int64), 7, 17, 4),
    ]
    init = torch.randint(-50, 50, (slots, ring * W), dtype=I32)
    for layout in ("flat", "split"):
        st = init.clone().view(slots, 1, 1, ring * W) if layout == "flat" else init.clone().view(slots, ring, W)
        ref = init.clone().view(slots, ring, W)
        ok = True
        for mode, src, rid, width, off, block in steps:
            p = {"M": len(rid), "ring": ring, "width": width, "offset": off, "seqs": 4, "mode": mode}
            if block:
                p["block"] = block
            rc = pl.call("hy4_ref_ring_store", [src, rid, pos, cu, sidx, st], p)
            D.ring_store(ref, mode, src, rid, pos, cu, sidx, ring, width, off, block or 1)
            ok = ok and rc == 0 and torch.equal(st.reshape(slots, ring, W), ref)
        report(f"ring_store, every mode ({layout} state)", ok)
    # slot 1 (no sequence) and row 0 (outside cu) untouched; the sequence with sidx -1 writes nothing
    report("ring_store leaves slot 1 untouched", torch.equal(st.view(slots, -1)[1], init[1]))
    st2 = init.clone().view(slots, ring, W)
    rc = pl.call("hy4_ref_ring_store", [None, torch.arange(rows, dtype=I32), pos, cu, torch.tensor([-1, 0], dtype=I32),
                                        st2], {"M": rows, "ring": ring, "width": 1, "offset": 0, "seqs": 4, "mode": "pos"})
    report("ring_store: sidx -1 writes nothing for that sequence",
           rc == 0 and torch.equal(st2[2], init.view(slots, ring, W)[2]) and not torch.equal(st2[0], init.view(slots, ring, W)[0]))
    # the dense words: count p+1, the pages up to p/4, -1 past them (row 2: p 11, pages 0..2)
    e = st.reshape(slots, ring, W)[2, 11 % ring, 17:24]
    report("ring_store dense: count and pages", e.tolist() == [12] + btab[0, :3].tolist() + [-1, -1, -1])
    # h kept as f32 bits
    hb = st.reshape(slots, ring, W)[2, 13 % ring, 9:14].clone().view(torch.float32)
    report("ring_store out: bf16 kept as its f32 bits", torch.equal(hb, h_bf[0].float()))

    # loads. Next step: sequence 0 at p0 14 (entry of 13 there), sequence 1 at p0 20 (entry of 19:
    # stale), sequence 2 without slot, sequence 3 at p0 0; then without cu.
    stl = st.reshape(slots, ring, W)
    pos2 = torch.tensor([14, 15, 16, 20, 21, 5, 6, 0, 1], dtype=I32)
    cu2 = torch.tensor([0, 3, 5, 7, 9], dtype=I32)
    sidx2 = torch.tensor([2, 0, -1, 0], dtype=I32)
    pl_ = {"M": 9, "ring": ring, "width": 14, "offset": 1, "seqs": 4}
    dst = torch.full((9, 14), 77, dtype=I32)
    rc = pl.call("hy4_ref_ring_load", [pos2, cu2, sidx2, st, dst], pl_)
    want = torch.full((9, 14), 77, dtype=I32)
    D.ring_load(stl, want, pos2, cu2, sidx2, ring, 14, 1)
    report("ring_load with cu", rc == 0 and torch.equal(dst, want) and bool((dst[3:] == -1).all()) and
           torch.equal(dst[0], stl[2, 13 % ring, 1:15]))
    pos3 = torch.tensor([14, 7, 12], dtype=I32)          # per sequence; seq 1 (slot 0) at 7: entry of 6
    sidx3 = torch.tensor([2, 0, 2], dtype=I32)
    dst3 = torch.zeros(3, 5, dtype=torch.float32)
    rc = pl.call("hy4_ref_ring_load", [pos3, None, sidx3, st, dst3], {**pl_, "M": 3, "width": 5, "offset": 9})
    want3 = torch.zeros(3, 5, dtype=I32)
    D.ring_load(stl, want3, pos3, None, sidx3, ring, 5, 9)
    report("ring_load without cu, f32 destination (bits)", rc == 0 and torch.equal(dst3.view(I32), want3) and
           torch.equal(dst3[0], h_bf[0].float()))
    # a store then a load gives back what was stored (the h of row a = p0 - 1)
    report("ring_load reads what ring_store wrote", torch.equal(dst3[2].view(I32), stl[2, 11 % ring, 9:14]))
    # fill: 0 in an f32 plane gives 0.0 where there is no entry (the plugin's h load), not a NaN
    dfz = torch.full((9, 14), 7.5, dtype=torch.float32)
    rc = pl.call("hy4_ref_ring_load", [pos2, cu2, sidx2, st, dfz], {**pl_, "fill": 0})
    wantz = torch.full((9, 14), 7, dtype=I32)
    D.ring_load(stl, wantz, pos2, cu2, sidx2, ring, 14, 1, fill=0)
    report("ring_load fill 0, f32: no entry reads 0.0", rc == 0 and torch.equal(dfz.view(I32), wantz) and
           bool((dfz[3:] == 0).all()) and not bool(dfz.isnan().any()))
    rc = pl.call("hy4_ref_ring_load", [pos2, cu2, sidx2, st, torch.zeros(9, 14, dtype=I32)], {**pl_, "fill": 1 << 31})
    report("ring_load fill outside i32: refused", rc != 0)
    # rows outside cu's spans get the fill too (row 0 and row 8 here), with -1 and with fill 5
    cu4 = torch.tensor([1, 3, 5, 7, 8], dtype=I32)
    for fv in (None, 5):
        d4 = torch.full((9, 14), 77, dtype=I32)
        rc = pl.call("hy4_ref_ring_load", [pos2, cu4, sidx2, st, d4], {**pl_, **({"fill": fv} if fv is not None else {})})
        w4 = torch.full((9, 14), 77, dtype=I32)
        D.ring_load(stl, w4, pos2, cu4, sidx2, ring, 14, 1, fill=-1 if fv is None else fv)
        f_ = -1 if fv is None else fv
        report(f"ring_load rows outside cu get the fill ({f_})", rc == 0 and torch.equal(d4, w4) and
               bool((d4[0] == f_).all()) and bool((d4[8] == f_).all()))

    # duplicates: rows of one sequence over more positions than the ring holds; the last listed wins
    std = init.clone().view(slots, ring, W)
    refd = init.clone().view(slots, ring, W)
    pd = torch.arange(100, 113, dtype=I32)                   # 13 rows, ring 6: every entry 2 or 3 times
    cud = torch.tensor([0, 13], dtype=I32)
    srcd = torch.randint(-1000, 1000, (13, 4), dtype=I32)
    ridd = torch.tensor([12, 0, 5, 6, 1, 11, 7, 2, 3, 10, 4, 9, 8], dtype=I32)   # not in position order
    pdd = {"M": 13, "ring": ring, "width": 4, "offset": 2, "seqs": 4, "mode": "row"}
    rc = pl.call("hy4_ref_ring_store", [srcd, ridd, pd, cud, torch.tensor([1], dtype=I32), std], pdd)
    D.ring_store(refd, "row", srcd, ridd, pd, cud, torch.tensor([1], dtype=I32), ring, 4, 2)
    lastw = all(torch.equal(std[1, e, 2:6], srcd[[int(r) for r in ridd if int(pd[r]) % ring == e][-1]])
                for e in range(ring))
    report("ring_store duplicates: the last listed row wins", rc == 0 and torch.equal(std, refd) and lastw)

    # refusals, with nothing written
    def refused(name, kern, ops, p):
        before = st.clone()
        rc = pl.call(kern, ops, p)
        report(f"{name}: refused, state untouched", rc != 0 and torch.equal(st, before), f"rc={rc}")
    ps = {"M": rows, "ring": ring, "width": 1, "offset": 0, "seqs": 4, "mode": "pos"}
    refused("ring_store slot outside state", "hy4_ref_ring_store",
            [None, torch.arange(rows, dtype=I32), pos, cu, torch.tensor([3, 0], dtype=I32), st], ps)
    refused("ring_store row id past the rows", "hy4_ref_ring_store",
            [None, torch.tensor([1, 9], dtype=I32), pos, cu, sidx, st], {**ps, "M": 2})
    refused("ring_store offset + width past the entry", "hy4_ref_ring_store",
            [sel, torch.arange(rows, dtype=I32), pos, cu, sidx, st], {**ps, "mode": "row", "width": 7, "offset": 18})
    refused("ring_store row without src", "hy4_ref_ring_store",
            [None, torch.arange(rows, dtype=I32), pos, cu, sidx, st], {**ps, "mode": "row"})
    refused("ring_store pos with src", "hy4_ref_ring_store",
            [nsel, torch.arange(rows, dtype=I32), pos, cu, sidx, st], ps)
    refused("ring_store cu past the rows", "hy4_ref_ring_store",
            [None, torch.arange(rows, dtype=I32), pos, torch.tensor([1, 5, 10], dtype=I32), sidx, st], ps)
    refused("ring_store dense block table without the sequence's row", "hy4_ref_ring_store",
            [btab[:1].contiguous(), torch.arange(rows, dtype=I32), pos, cu, sidx, st],
            {**ps, "mode": "dense", "width": 7, "offset": 17, "block": 4})
    refused("ring_store state not i32", "hy4_ref_ring_store",
            [None, torch.arange(rows, dtype=I32), pos, cu, sidx, torch.zeros(slots, ring * W, dtype=torch.int64)], ps)
    dl = torch.full((9, 14), 77, dtype=I32)
    rc = pl.call("hy4_ref_ring_load", [pos2, cu2, torch.tensor([2, 0, 3, 0], dtype=I32), st, dl], pl_)
    report("ring_load slot outside state: refused, nothing written", rc != 0 and bool((dl == 77).all()))
    rc = pl.call("hy4_ref_ring_load", [pos2, cu2, sidx2, st, torch.zeros(9, 14, dtype=torch.bfloat16)], pl_)
    report("ring_load bf16 destination: refused", rc != 0)


def test_mla_bkeys():
    """hy4_mla_attn with block keys (DFly2) against tests/dfly_ref.py's union in double: two block
    sequences of 6 rows, histories of every length (0 included), invalid pages, causality, the sink,
    rows outside bcu and the refusals."""
    import dfly_ref as D
    torch.manual_seed(12)
    H, L, R, bs, P = 3, 512, 64, 4, 40
    rows = 12
    cache = torch.randn(P, 1, bs, L + R)
    sel = torch.randint(0, P, (rows, 10), dtype=I32)
    sel[1, 2] = -1
    sel[7, 0] = P + 3
    sequ = torch.tensor([37, 40, 0, 1, 23, 39, 0, 0, 12, 40, 5, 33], dtype=I32)
    bkeys = torch.randn(rows, L + R)
    bcu = torch.tensor([0, 6, 12], dtype=I32)
    q, qr = torch.randn(rows, H, L) / 8, torch.randn(rows, H, R) / 8
    sinks = torch.randn(H)
    scale = 1.0 / 16
    p = {"M": rows, "heads": H, "latent": L, "rope": R, "block_size": bs, "window": 0, "scale": scale, "dtype": "f32"}

    def run(bk, bc, sk, su=sequ, o=None, cu=None, pp=p):
        o = torch.zeros(rows, H, L) if o is None else o
        return pl.call("hy4_ref_mla_attn", [q, qr, cache, sel, su, cu, sk, o, bk, bc], pp), o

    rc, o = run(bkeys, bcu, sinks)
    want, _ = D.union_attn(q, qr, cache, sel, sequ, bkeys, bcu, sinks, scale, bs)
    e = l2(want, o)
    report("mla_attn bkeys: the union of selection and block keys", rc == 0 and e < 1e-6, f"rc={rc} L2 {e:.2e}")
    bk2 = bkeys.clone()
    bk2[3] += 1.0
    bk2[9] += 1.0
    rc2, o2 = run(bk2, bcu, sinks)
    causal = torch.equal(o2[:3], o[:3]) and torch.equal(o2[6:9], o[6:9]) and \
        not torch.equal(o2[3], o[3]) and not torch.equal(o2[5], o[5]) and not torch.equal(o2[9], o[9])
    report("mla_attn bkeys: block causality", rc2 == 0 and causal)
    zero = torch.zeros(rows, dtype=I32)
    rc, o = run(bkeys, bcu, sinks, su=zero)
    want, _ = D.union_attn(q, qr, cache, sel, zero, bkeys, bcu, sinks, scale, bs)
    e = l2(want, o)
    report("mla_attn bkeys: empty history (seqused 0)", rc == 0 and e < 1e-6, f"L2 {e:.2e}")
    for name, sk in (("sink above everything", torch.full((H,), 50.0)), ("sink -inf", torch.full((H,), float("-inf")))):
        rc, o = run(bkeys, bcu, sk)
        want, _ = D.union_attn(q, qr, cache, sel, sequ, bkeys, bcu, sk, scale, bs)
        e = l2(want, o)
        report(f"mla_attn bkeys: {name}", rc == 0 and e < 1e-6, f"L2 {e:.2e}")
    rc, o = run(bkeys, torch.tensor([2, 6, 12], dtype=I32), sinks, o=torch.full((rows, H, L), 7.0))
    report("mla_attn bkeys: rows outside bcu untouched", rc == 0 and bool((o[:2] == 7.0).all()) and not bool((o[2] == 7.0).all()))
    bad = [("bkeys without bcu", dict(bk=bkeys, bc=None, sk=sinks)),
           ("bcu without bkeys", dict(bk=None, bc=bcu, sk=sinks)),
           ("with cu_seqlens", dict(bk=bkeys, bc=bcu, sk=sinks, cu=torch.arange(rows + 1, dtype=I32),
                                    su=torch.full((rows,), 4, dtype=I32))),
           ("with a window", dict(bk=bkeys, bc=bcu, sk=sinks, pp={**p, "window": 8})),
           ("bkeys narrower than latent + rope", dict(bk=bkeys[:, :L].contiguous(), bc=bcu, sk=sinks)),
           ("bcu past the rows", dict(bk=bkeys, bc=torch.tensor([0, 6, 13], dtype=I32), sk=sinks))]
    for name, kw in bad:
        o0 = torch.full((rows, H, L), 7.0)
        rc, o = run(o=o0, **kw)
        report(f"mla_attn bkeys {name}: refused, nothing written", rc != 0 and bool((o == 7.0).all()), f"rc={rc}")
    # without block keys a selection row of 1 "sequence" a row still needs seqused per row
    rc, _ = run(bkeys[:6].contiguous(), bcu, sinks)
    report("mla_attn bkeys with fewer rows than the batch: refused", rc != 0)


def main() -> int:
    test_fp8_from_table()
    test_negatives()
    test_ngram_multi_seq()
    test_ngram_shapes()
    test_select_full()
    test_score_two_seqs()
    test_rope_bf16()
    test_had_quant_torch()
    test_ring()
    test_mla_bkeys()
    print("OK" if all(T.results) else "FAIL")
    return 0 if all(T.results) else 1


if __name__ == "__main__":
    sys.exit(main())
