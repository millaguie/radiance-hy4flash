#!/usr/bin/env python3
"""hy4k ops against HF (random weights or from the checkpoint) and against layer 1 of the FP32
reference pass. FP32 tolerance: 1e-5 (rounding of long sums in a different order); integers must
be equal.

HY4_CKPT is the checkpoint (default: the first snapshot of tencent/Hy4-flash in the Hugging Face
cache under $HOME); HY4_REF the FP32 reference pass ref/reference.py wrote (default
~/hy4/ref/fp32-1)."""
import os
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ref"))
from rad_ctypes import Plugin  # noqa: E402
from dump_names import canonical_keys  # noqa: E402
from transformers import AutoConfig  # noqa: E402
from transformers.models.hy_v4_flash import modeling_hy_v4_flash as M  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SNAP = Path(os.environ["HY4_CKPT"]) if os.environ.get("HY4_CKPT") else sorted((Path.home() / ".cache/huggingface/hub/models--tencent--Hy4-flash/snapshots").iterdir())[0]
REF = Path(os.environ.get("HY4_REF") or Path.home() / "hy4/ref/fp32-1")
TOL = 1e-5
IDX = json.loads((SNAP / "model.safetensors.index.json").read_text())["weight_map"]
pl = Plugin(Path(os.environ["HY4K_SO"]) if os.environ.get("HY4K_SO") else ROOT / "home/kernels/hy4k.so")
cfg = AutoConfig.from_pretrained(SNAP).text_config
results = []


def w(k):
    with safe_open(str(SNAP / IDX[k]), framework="pt") as f:
        return f.get_tensor(k)


def load_ref(p):
    """A dump of the reference pass, with its names as they are now (it may predate the rename)."""
    return canonical_keys(load_file(str(p)))


def l2(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / a.norm().clamp_min(1e-30)).item()


def report(name, ok, detail=""):
    results.append(ok)
    print(f"{'OK   ' if ok else 'FAIL '} {name} {detail}")


def test_mla_kv():
    T, H, nope, rope, vd = 9, cfg.num_attention_heads, cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, 256
    kvx = torch.randn(T, H * (nope + vd))
    kr = torch.randn(T, rope)
    k = torch.empty(T, H, rope + nope)
    v = torch.empty(T, H, vd)
    rc = pl.call("hy4_ref_mla_kv", [kvx, kr, k, v], {"M": T, "heads": H, "nope": nope, "rope": rope, "v_dim": vd, "dtype": "f32"})
    x = kvx.view(T, H, nope + vd)
    ok = rc == 0 and torch.equal(k[..., :rope], kr[:, None].expand(T, H, rope)) and \
        torch.equal(k[..., rope:], x[..., :nope]) and torch.equal(v, x[..., nope:])
    report("mla_kv", ok)


def test_mla_attn():
    """hy4_mla_attn (absorbed form) against HF's unabsorbed attention in f64: k = [k_rope | Wk·lat],
    v = Wv·lat per head, sink, causal, window. The absorbed query is q_nope·Wk; the latent output is
    expanded with Wv. Dense varlen (two sequences, with and without window) and the selection form."""
    torch.manual_seed(1)
    H, L, R, nope, vd, bs, P = 3, 512, 64, 192, 256, 4, 40
    scale = 1.0 / (R + nope) ** 0.5
    Wk = torch.randn(H, nope, L, dtype=torch.float64) / L ** 0.5
    Wv = torch.randn(H, vd, L, dtype=torch.float64) / L ** 0.5
    sinks = torch.randn(H)
    lat = torch.randn(P, bs, L)
    kr = torch.randn(P, bs, R)
    cache = torch.cat([lat, kr], -1).view(P, 1, bs, L + R).contiguous()

    def ref(qrow, pages, ctx, pos, window, h):
        # qrow: [R + nope] in radiance's q order ([rope | nope])
        ks, vs = [], []
        for j in range(ctx):
            if j > pos or (window > 0 and pos - j >= window):
                continue
            pg = pages[j // bs] if j // bs < len(pages) else -1
            if pg < 0 or pg >= P:
                continue
            lat_j = lat[pg, j % bs].double()
            ks.append(torch.cat([kr[pg, j % bs].double(), Wk[h] @ lat_j]))
            vs.append(Wv[h] @ lat_j)
        if not ks:
            return torch.zeros(vd, dtype=torch.float64)
        K, V = torch.stack(ks), torch.stack(vs)
        lg = torch.cat([(K @ qrow.double()) * scale, sinks[h:h + 1].double()])
        return torch.softmax(lg, 0)[:-1] @ V

    def absorbed(q, rows, extra):
        qn = q[:, :, R:].double()
        qlat = torch.einsum("thn,hnl->thl", qn, Wk).float().contiguous()
        qrope = q[:, :, :R].contiguous()
        o = torch.zeros(rows, H, L)
        rc = pl.call("hy4_ref_mla_attn", [qlat, qrope, cache] + extra + [sinks, o],
                     {"M": rows, "heads": H, "latent": L, "rope": R, "block_size": bs, "window": extra_w[0],
                      "scale": scale, "dtype": "f32"})
        return rc, torch.einsum("thl,hvl->thv", o.double(), Wv)

    I32 = torch.int32
    cu = torch.tensor([0, 5, 8], dtype=I32)
    su = torch.tensor([23, 9], dtype=I32)
    bt = torch.stack([torch.randperm(P)[:8], torch.randperm(P)[:8]]).to(I32)
    q = torch.randn(8, H, R + nope)
    for window in (0, 7):
        extra_w = [window]
        rc, out = absorbed(q, 8, [bt, su, cu])
        refo = torch.zeros(8, H, vd, dtype=torch.float64)
        for s_ in range(2):
            qlen = int(cu[s_ + 1] - cu[s_])
            for i in range(qlen):
                r = int(cu[s_]) + i
                for h in range(H):
                    refo[r, h] = ref(q[r, h], bt[s_].tolist(), int(su[s_]), int(su[s_]) - qlen + i, window, h)
        e = l2(refo, out)
        report(f"mla_attn dense window={window}", rc == 0 and e < 1e-5, f"rc={rc} L2 {e:.2e}")
    # 2-D as in the plugin: q_lat [T, H·L] and q_rope a view of q [T, H·(R+nope)] with rope_pitch
    qn = q[:, :, R:].double()
    qlat2 = torch.einsum("thn,hnl->thl", qn, Wk).float().reshape(8, H * L).contiguous()
    qfull = q.reshape(8, H * (R + nope)).contiguous()
    o2 = torch.zeros(8, H * L)
    rc = pl.call("hy4_ref_mla_attn", [qlat2, qfull, cache, bt, su, cu, sinks, o2],
                 {"M": 8, "heads": H, "latent": L, "rope": R, "block_size": bs, "window": 7,
                  "rope_pitch": R + nope, "scale": scale, "dtype": "f32"})
    out2 = torch.einsum("thl,hvl->thv", o2.view(8, H, L).double(), Wv)
    e = l2(refo, out2)
    report("mla_attn 2-D with rope_pitch (window=7)", rc == 0 and e < 1e-5, f"rc={rc} L2 {e:.2e}")
    # a cu that does not cover all rows (span of a mixed step): rows outside are not touched
    cu_t = torch.tensor([2, 5], dtype=I32)
    su_t = torch.tensor([23], dtype=I32)
    qlat_t = torch.einsum("thn,hnl->thl", q[:, :, R:].double(), Wk).float().contiguous()
    o_t = torch.full((8, H, L), 7.0)
    rc = pl.call("hy4_ref_mla_attn", [qlat_t, q[:, :, :R].contiguous(), cache, bt[:1].contiguous(), su_t, cu_t, sinks, o_t],
                 {"M": 8, "heads": H, "latent": L, "rope": R, "block_size": bs, "window": 0, "scale": scale, "dtype": "f32"})
    outside = torch.cat([o_t[:2], o_t[5:]])
    report("mla_attn rows outside cu untouched", rc == 0 and bool((outside == 7.0).all()) and not bool((o_t[2:5] == 7.0).all()),
           f"rc={rc}")
    # validation: bt with fewer rows than sequences, and cu that goes past the rows
    o_m = torch.zeros(8, H, L)
    rc1 = pl.call("hy4_ref_mla_attn", [qlat_t, q[:, :, :R].contiguous(), cache, bt[:1].contiguous(), su, cu, sinks, o_m],
                  {"M": 8, "heads": H, "latent": L, "rope": R, "block_size": bs, "dtype": "f32"})
    rc2 = pl.call("hy4_ref_mla_attn", [qlat_t[:4].contiguous(), q[:4, :, :R].contiguous(), cache, bt[:1].contiguous(),
                                       su_t, torch.tensor([0, 9], dtype=I32), sinks, o_m[:4].contiguous()],
                  {"M": 4, "heads": H, "latent": L, "rope": R, "block_size": bs, "dtype": "f32"})
    report("mla_attn rejects short bt and long cu", rc1 != 0 and rc2 != 0, f"rc {rc1} {rc2}")
    rows, K = 6, 10
    sel = torch.randint(0, P, (rows, K), dtype=I32)
    sel[2, 3] = -1
    sel[4, 1] = P + 5
    sequ = torch.tensor([37, 40, 21, 1, 33, 39], dtype=I32)
    q = torch.randn(rows, H, R + nope)
    extra_w = [0]
    rc, out = absorbed(q, rows, [sel, sequ, None])
    refo = torch.zeros(rows, H, vd, dtype=torch.float64)
    for r in range(rows):
        for h in range(H):
            refo[r, h] = ref(q[r, h], sel[r].tolist(), int(sequ[r]), int(sequ[r]) - 1, 0, h)
    e = l2(refo, out)
    report("mla_attn selection", rc == 0 and e < 1e-5, f"rc={rc} L2 {e:.2e}")


def ids_hf(layer_idx, toks):
    """Flat HF ids for a whole sequence (layer with Engram `layer_idx`, 0-based)."""
    eng = M.HYV4FlashEngramLayer.__new__(M.HYV4FlashEngramLayer)
    torch.nn.Module.__init__(eng)
    eng.layer_id = layer_idx + 1
    eng.max_ngram_size = cfg.engram_max_ngram_size
    eng.n_head_per_ngram = cfg.engram_n_head_per_ngram
    eng.pad_id = cfg.engram_pad_id
    eng.seed = cfg.engram_seed
    sizes = [x for o in cfg.engram_layer_head_vocab_sizes[str(layer_idx + 1)] for x in o]
    eng.head_vocab_sizes = torch.tensor(sizes)
    eng.lookup_table = w(f"model.layers.{layer_idx}.engram_layer.lookup_table")
    eng.hash_multipliers = eng.draw_hash_multipliers(int(eng.lookup_table.max()) + 1)
    ids_t = torch.tensor([toks])
    ng = M.HYV4FlashTextModel.build_ngram_tokens(type("c", (), {"config": cfg})(), ids_t, None)
    h = eng.hash_ngrams(ng)[0]
    off = torch.tensor([0] + sizes[:-1]).cumsum(0)
    return h + off, eng, sizes


def test_ngram_ids():
    layers = [l - 1 for l in cfg.engram_layer_ids]
    toks = json.loads((ROOT / "ref/prompts.json").read_text())["prompts"]["tools"][:150]
    toks = toks + [120002, 125000, 5]  # the pad and an id outside the map
    ref, maps, mults, vszs, offs = [], [], [], [], []
    for li in layers:
        r, eng, sizes = ids_hf(li, toks)
        ref.append(r)
        maps.append(eng.lookup_table.to(torch.int32))
        mults.append(eng.hash_multipliers)
        vszs.append(torch.tensor(sizes, dtype=torch.int64))
        offs.append(torch.tensor([0] + sizes[:-1]).cumsum(0))
    ref = torch.stack(ref)
    wmap, mult, vsz, off = torch.stack(maps), torch.stack(mults), torch.stack(vszs), torch.stack(offs)
    E, T = len(layers), len(toks)
    p = {"engram_layers": E, "heads": 16, "ngram": 3, "pad": cfg.engram_pad_id, "seqs": 1}
    # (a) whole sequence in one step, no state
    ids = torch.empty(E, T, 16, dtype=torch.int32)
    st = torch.zeros(1, 2, dtype=torch.int32)
    rc = pl.call("hy4_ref_ngram_ids", [torch.tensor(toks, dtype=torch.int32), torch.arange(T, dtype=torch.int32),
                 torch.tensor([0, T], dtype=torch.int32), st, torch.tensor([0], dtype=torch.int32), wmap, mult, vsz, off, ids],
                 {"M": T, **p})
    report("ngram_ids whole", rc == 0 and torch.equal(ids.long(), ref))
    # (b) in chunks of 1, 2 and 5 tokens with state between steps
    st = torch.full((3, 2), -7, dtype=torch.int32)  # garbage: must not be read on a cold start
    out = torch.empty(E, T, 16, dtype=torch.int32)
    i, chunk = 0, [1, 2, 5]
    k = 0
    while i < T:
        c = min(chunk[k % 3], T - i); k += 1
        seg = torch.empty(E, c, 16, dtype=torch.int32)
        rc = pl.call("hy4_ref_ngram_ids", [torch.tensor(toks[i:i + c], dtype=torch.int32),
                     torch.arange(i, i + c, dtype=torch.int32), torch.tensor([0, c], dtype=torch.int32), st,
                     torch.tensor([2], dtype=torch.int32), wmap, mult, vsz, off, seg], {"M": c, **p})
        out[:, i:i + c] = seg
        i += c
    report("ngram_ids by chunks with state", rc == 0 and torch.equal(out.long(), ref))
    v = load_ref(REF / "tools.L01.safetensors")
    t = json.loads((ROOT / "ref/prompts.json").read_text())["prompts"]["tools"]
    ids1 = torch.empty(1, len(t), 16, dtype=torch.int32)
    rc = pl.call("hy4_ref_ngram_ids", [torch.tensor(t, dtype=torch.int32), torch.arange(len(t), dtype=torch.int32),
                 torch.tensor([0, len(t)], dtype=torch.int32), torch.zeros(1, 2, dtype=torch.int32),
                 torch.tensor([0], dtype=torch.int32), wmap[:1], mult[:1], vsz[:1], off[:1], ids1],
                 {"M": len(t), **p, "engram_layers": 1})
    report("ngram_ids vs the dump (tools, layer 1)", rc == 0 and torch.equal(ids1[0].long(), v["engram.ids"]))


def test_engram():
    v = load_ref(REF / "corto.L01.safetensors")
    T, n, hc = v["engram.in"].shape[0], cfg.hidden_size, cfg.hc_mult
    pre = "model.layers.1.engram_layer."
    wn = torch.stack([w(f"{pre}lookup_per_head_norms.{h}.weight").float() for h in range(16)])
    rows = v["engram.rows_fp8"].float().reshape(T, -1)
    y = torch.empty_like(rows)
    rc = pl.call("hy4_ref_rows_rmsnorm", [rows, wn, y], {"M": T, "heads": 16, "d": 96, "eps": 1.1920929e-07, "dtype": "f32"})
    e = l2(v["engram.rows_norm"].reshape(T, -1), y)
    report("rows_rmsnorm vs the dump", rc == 0 and e < TOL, f"L2 {e:.2e}")
    y1 = torch.empty_like(rows)  # concatenated 1-D weights [heads·d], as the architecture declares them
    rc = pl.call("hy4_ref_rows_rmsnorm", [rows, wn.reshape(-1).contiguous(), y1],
                 {"M": T, "heads": 16, "d": 96, "eps": 1.1920929e-07, "dtype": "f32"})
    report("rows_rmsnorm with 1-D weights, same as 2-D", rc == 0 and torch.equal(y1, y))
    e_n = v["engram.rows_norm"].reshape(T, -1).float()
    k = torch.stack([F.linear(e_n, w(f"{pre}key_projs.{j}.weight").float(), w(f"{pre}key_projs.{j}.bias").float())
                     for j in range(hc)], dim=1)
    vv = F.linear(e_n, w(f"{pre}value_proj.weight").float(), w(f"{pre}value_proj.bias").float())
    h = v["engram.in"].float().clone()
    h0 = h.clone()
    wk = torch.stack([w(f"{pre}norm_key.{j}.weight").float() for j in range(hc)])
    wq = torch.stack([w(f"{pre}norm_query.{j}.weight").float() for j in range(hc)])
    gam = w(f"{pre}unified_pre_gamma").float()
    per = torch.stack([w(f"{pre}unified_per_hc_scales.{j}").float() for j in range(hc)])
    ids = load_file(str(REF / "corto.ids.safetensors"))["ids"].to(torch.int32)
    rc = pl.call("hy4_ref_engram_gate", [h.reshape(T, hc * n), ids, k.reshape(T, hc * n).contiguous(), vv, wk, wq, gam, per],
                 {"M": T, "n": n, "hc": hc, "pad": cfg.engram_pad_id, "eps": 1.1920929e-07, "gate_eps": 1e-6, "dtype": "f32"})
    e = l2(v["engram.out"], h - h0)
    report("engram_gate vs the dump", rc == 0 and e < TOL, f"L2 {e:.2e}")
    h1 = h0.clone()  # concatenated 1-D weights, as the architecture declares them
    rc = pl.call("hy4_ref_engram_gate", [h1.reshape(T, hc * n), ids, k.reshape(T, hc * n).contiguous(), vv,
                                         wk.reshape(-1).contiguous(), wq.reshape(-1).contiguous(), gam,
                                         per.reshape(-1).contiguous()],
                 {"M": T, "n": n, "hc": hc, "pad": cfg.engram_pad_id, "eps": 1.1920929e-07, "gate_eps": 1e-6, "dtype": "f32"})
    report("engram_gate with 1-D weights, same as 2-D", rc == 0 and torch.equal(h1, h))
    # mask: with every token = pad, h does not change
    h2 = h0.clone()
    rc = pl.call("hy4_ref_engram_gate", [h2.reshape(T, hc * n), torch.full((T,), cfg.engram_pad_id, dtype=torch.int32),
                 k.reshape(T, hc * n).contiguous(), vv, wk, wq, gam, per],
                 {"M": T, "n": n, "hc": hc, "pad": cfg.engram_pad_id, "dtype": "f32"})
    report("engram_gate leaves h intact with pad", rc == 0 and torch.equal(h2, h0))


def rope_hf(x, pos):
    """Rope of the last 64 like HF's indexer (theta of the sparse layers)."""
    rot = M.HYV4FlashRotaryEmbedding(cfg)
    p3 = pos.view(1, 1, -1).expand(3, 1, -1)
    c, s = rot(x.unsqueeze(0), p3, layer_type="indexed_attention")
    return M._apply_partial_rotary_pos_emb(x.unsqueeze(0).unsqueeze(2), c, s, cfg.qk_rope_head_dim, unsqueeze_dim=2)[0, :, 0]


def test_idx_pool():
    v = load_ref(REF / "largo.L01.safetensors")
    tk = v["index.token_keys"].float()
    G = tk.shape[0] // 4
    pre = "model.layers.1.self_attn.compressed_indexer."
    cache = torch.zeros(G + 3, 128)
    page = torch.arange(G, dtype=torch.int32) + 2   # shifted pages to test `page`
    bpos = (torch.arange(G, dtype=torch.int32) * 4)
    rc = pl.call("hy4_ref_idx_pool", [tk[:G * 4].contiguous(), w(pre + "key_value_projection.weight").float(),
                 w(pre + "key_gate_projection.weight").float(), w(pre + "intra_group_ape").float(),
                 w(pre + "key_norm.weight").float(), page, bpos, cache],
                 {"M": G, "n": 128, "ratio": 4, "rotary_dim": cfg.qk_rope_head_dim, "theta": 1e7, "eps": float(cfg.rms_norm_eps), "dtype": "f32"})
    ref = rope_hf(v["index.group_keys"].float(), bpos.long())
    e = l2(ref, cache[2:G + 2])
    report("idx_pool vs the dump (largo, layer 1)", rc == 0 and e < TOL and cache[:2].abs().sum() == 0, f"L2 {e:.2e}")
    return cache, page, G


def test_idx_score_select(cache, page, G):
    v = load_ref(REF / "largo.L01.safetensors")
    T = v["index.query"].shape[0]
    pos = torch.arange(T, dtype=torch.int32)
    # unrotated q: the dump's rope is undone by rotating by −pos (FP32 rounding error)
    qrot = v["index.query"].float()
    q = torch.stack([rope_hf(qrot[:, h], -pos.long()) for h in range(16)], dim=1)
    wh = v["index.head_weights"].float() / (16 ** -0.5 * 128 ** -0.5)
    bt = page.view(1, -1).contiguous()
    score = torch.empty(T, G)
    rc = pl.call("hy4_ref_idx_score", [q, wh, pos, cache, bt, torch.tensor([0, T], dtype=torch.int32),
                 torch.tensor([G], dtype=torch.int32), score],
                 {"M": T, "n": 128, "heads": 16, "rotary_dim": cfg.qk_rope_head_dim, "theta": 1e7, "blocks": G, "seqs": 1, "dtype": "f32"})
    ref = v["index.logits"].float()
    valid = torch.isfinite(ref)
    e = l2(ref[valid], score[valid])
    report("idx_score vs the dump (largo, layer 1)", rc == 0 and e < 1e-5, f"L2 {e:.2e} (q un-rotated)")
    # selection over the dump's logits: same as HF's (without the -1 padding)
    sel = torch.empty(T, 512, dtype=torch.int32)
    nsel = torch.empty(T, dtype=torch.int32)
    used = torch.empty(T, dtype=torch.int32)
    bta = torch.arange(G + 1, dtype=torch.int32).view(1, -1)  # attention table = identity
    rc = pl.call("hy4_ref_idx_select", [ref.contiguous(), bta, torch.tensor([G], dtype=torch.int32), pos,
                 torch.tensor([0, T], dtype=torch.int32), sel, nsel, used],
                 {"M": T, "blocks": G, "topk": 512, "ratio": 4, "seqs": 1})
    hf = v["index.selection"].long()
    bad = 0
    for r in range(2100, T, 37):
        a = set(hf[r][hf[r] >= 0].tolist())
        b = set(sel[r, :int(nsel[r])].tolist())
        bad += a != b
    ok_used = all(int(used[r]) == (int(nsel[r]) - 1) * 4 + r % 4 + 1 for r in range(2100, T, 37))
    report("idx_select vs HF's selection", rc == 0 and bad == 0 and ok_used, f"differing rows {bad}")


def main() -> int:
    torch.manual_seed(0)
    test_mla_kv()
    test_mla_attn()
    test_ngram_ids()
    test_engram()
    cache, page, G = test_idx_pool()
    test_idx_score_select(cache, page, G)
    print("OK" if all(results) else "FAIL")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
