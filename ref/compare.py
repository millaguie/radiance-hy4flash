"""Compares the plugin tap dumps (HY4_DUMP_DIR) with those of ref/reference.py.

Usage: compare.py --dump DIR --ref DIR [--noise DIR] [--snap DIR] --prompt corto [--layers 0,1,2]

A prompt longer than one step arrives in several (s0, s1, ...): they are joined in step order.

For each name both have: relative L2 error ||a - r|| / ||r|| against the FP32 reference (`--ref`).
With `--noise` (the BF16 pass of reference.py) it also prints the BF16 noise floor of the same
name (the BF16 pass against the FP32 one) and the ratio: the plan step 4 criterion is
ratio <= 2. Integer dumps (ids) are compared for equality.

Selections (plan steps 4.3.5 and 4.3.7): the router experts and the indexer groups are compared as
sets per row. Each mismatch is measured by its distance to the cutoff in the FP32 reference: for
the router, |sigmoid(logit) + bias - cutoff| with the cutoff between the 16th and the 17th (the
bias comes from the checkpoint, `--snap`); for the indexer, |logit - cutoff_score|. With `--noise`,
the band is the largest of those distances between the BF16 pass and the FP32 one, and a mismatch
outside the band is a failure.

Exits with 1 if some criterion fails (ratio > 2, different ids, shapes, mismatches outside the
band, a dump the reference has and the plugin does not), with 2 if the criterion could not be
measured (missing `--noise`, or `--snap` with experts), and with 0 if it was measured and passes.
`--report-only` only reports and exits with 0.
"""
import glob
import json
import argparse
import os
import sys

import numpy as np
from safetensors.torch import load_file as _load_torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dump_names import canonical  # noqa: E402  (dumps written before the rename carry the old names)


def load_file(p):
    """safetensors to numpy; BF16 ones (the BF16 pass of reference.py) become float32."""
    out = {}
    for k, v in _load_torch(p).items():
        out[canonical(k)] = v.float().numpy() if v.is_floating_point() else v.numpy()
    return out


def tap_name(f):
    """The dump name of a tap file `s<step>.<name>.L<layer>.<dtype>`, under its current name."""
    return canonical(f.split(".", 1)[1].rsplit(".L", 1)[0])


def read_tap(fname):
    ext = fname.rsplit(".", 1)[1]
    if ext == "f32":
        a = np.fromfile(fname, dtype=np.float32)
    elif ext == "bf16":
        u = np.fromfile(fname, dtype=np.uint16).astype(np.uint32) << 16
        a = u.view(np.float32)
    elif ext == "i32":
        a = np.fromfile(fname, dtype=np.int32)
    else:
        raise ValueError(fname)
    return a


def join_steps(dump, name, layer, width):
    """The dumps of `name` in `layer` from all steps, in step order, as [rows, width]."""
    chunks = []
    for f in os.listdir(dump):
        if tap_name(f) == name and f".L{layer:02d}." in f:
            chunks.append((int(f.split(".", 1)[0][1:]), f))
    return np.concatenate([read_tap(os.path.join(dump, f)).reshape(-1, width) for _, f in sorted(chunks)])


def steps(dump, layer):
    """{step: (first row, rows)} from the size of layer.in (4-stream iHC, BF16)."""
    nbytes = {}
    for f in os.listdir(dump):
        if tap_name(f) == "layer.in" and f".L{layer:02d}." in f:
            nbytes[int(f.split(".", 1)[0][1:])] = os.path.getsize(os.path.join(dump, f)) // 2
    out, r0 = {}, 0
    for k in sorted(nbytes):
        n = nbytes[k] // layer_row_size
        out[k] = (r0, n)
        r0 += n
    return out


layer_row_size = 4 * 3072


def compare_selection(dump, layer, ref_sel, ref_full=None):
    """Groups picked per row: pages of `sel` mapped to groups with the step block table."""
    pp = steps(dump, layer)
    rows = differing = extra = 0
    worst_rows = []
    cutoff_dist = []
    for f in sorted(os.listdir(dump)):
        if not (tap_name(f) == "index.sel_pages" and f".L{layer:02d}." in f):
            continue
        k = int(f.split(".", 1)[0][1:])
        r0, n = pp[k]
        sel = read_tap(os.path.join(dump, f)).reshape(n, -1)
        lg, ct = (ref_full["index.logits"][r0:r0 + n], ref_full["index.cutoff_score"][r0:r0 + n]) \
            if ref_full is not None and "index.logits" in ref_full else (None, None)
        bt = os.path.join(dump, f"s{k}.index.bt.L{layer:02d}.i32")
        if not os.path.exists(bt):
            bt = os.path.join(dump, f"s{k}.indice.bt.L{layer:02d}.i32")   # written before the rename
        bt = read_tap(bt)
        group_of = {int(pg): g for g, pg in enumerate(bt) if pg >= 0}
        for i in range(n):
            ours = {group_of[int(pg)] for pg in sel[i] if pg >= 0}
            theirs = {int(g) for g in ref_sel[r0 + i] if g >= 0}
            rows += 1
            d = len(ours ^ theirs)
            if d and lg is not None:
                for g in ours ^ theirs:
                    cutoff_dist.append(abs(float(lg[i, g]) - float(ct[i])))
            if d:
                differing += 1
                extra += d
                worst_rows.append((d, r0 + i))
    worst_rows.sort(reverse=True)
    return rows, differing, extra, worst_rows[:5], cutoff_dist


def index_cutoff(ref, first_row, n):
    return ref["index.logits"][first_row:first_row + n], ref["index.cutoff_score"][first_row:first_row + n]


def router_bias(snap, layer):
    """e_score_correction_bias of the layer, from the checkpoint (safetensors index)."""
    from safetensors import safe_open
    idx = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
    k = f"model.layers.{layer}.mlp.gate.e_score_correction_bias"
    with safe_open(os.path.join(snap, idx[k]), "np") as f:
        return f.get_tensor(k).astype(np.float64)


def router_scores(logits, bias, k):
    """Selection score and cutoff per row (between the k-th and the k+1-th)."""
    sc = 1.0 / (1.0 + np.exp(-logits.astype(np.float64))) + bias
    o = -np.sort(-sc, axis=1)
    return sc, (o[:, k - 1] + o[:, k]) / 2


def router_distances(ours, theirs, sc, cutoff):
    """Distances to the cutoff of the experts that differ between two selections [rows, k]."""
    d = []
    for i in range(ours.shape[0]):
        for e in set(ours[i].tolist()) ^ set(theirs[i].tolist()):
            d.append(abs(sc[i, e] - cutoff[i]))
    return np.array(d)


def per_row(x, r):
    """Relative L2 error per row: median, p99 and max."""
    a, b = x.astype(np.float64), r.astype(np.float64)
    e = np.linalg.norm(a - b, axis=1) / np.maximum(np.linalg.norm(b, axis=1), 1e-30)
    return np.median(e), np.percentile(e, 99), e.max()


def l2rel(a, r):
    a = a.astype(np.float64).ravel()
    r = r.astype(np.float64).ravel()
    n = np.linalg.norm(r)
    return float(np.linalg.norm(a - r) / n) if n > 0 else float(np.linalg.norm(a - r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--noise")
    ap.add_argument("--snap", help="HF checkpoint, for the router bias")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--layers", default="0,1,2")
    ap.add_argument("--report-only", action="store_true", help="only reports; gives no verdict")
    ap.add_argument("--band-factor", type=float, default=2.0,
                    help="a different expert or group fails if its distance to the cutoff exceeds this factor "
                         "times the BF16 band (default 2, like the L2: on the trimmed model, the cases "
                         "between 1× and 2× give a KL within the HF BF16 noise)")
    a = ap.parse_args()
    unmeasured = []

    ids = load_file(os.path.join(a.ref, f"{a.prompt}.ids.safetensors"))["ids"]
    rows = len(ids)
    worst = 0.0
    measured = 0
    failures = []
    for layer in [int(x) for x in a.layers.split(",")]:
        ref = load_file(os.path.join(a.ref, f"{a.prompt}.L{layer:02d}.safetensors"))
        noise = None
        if a.noise:
            p = os.path.join(a.noise, f"{a.prompt}.L{layer:02d}.safetensors")
            noise = load_file(p) if os.path.exists(p) else None
            if noise is None:
                unmeasured.append(f"L{layer:02d} noise (missing {p})")
        names = sorted({tap_name(f) for f in os.listdir(a.dump) if f".L{layer:02d}." in f})
        tag = f"L{layer:02d}"
        # What the reference dumps and the plugin does not need to (HF internals without a tap).
        internal = {"attn.in", "ffn.in", "engram.key_norm", "engram.query_norm",
                    "engram.rows_fp8", "engram.hash_ids", "engram.gate", "engram.out", "engram.signal",
                    "engram.value", "router.logits", "logits_trim"}
        missing = [k for k in ref if k not in names and k not in internal and not k.startswith("index.")]
        if "index.selection" in ref and "index.sel_pages" not in names:
            missing.append("index.sel_pages")
        for k in missing:
            print(f"{tag} {k:24s} MISSING from the plugin dumps")
            failures.append(f"{tag} {k}: missing")
        # ---- indexer groups
        if "index.selection" in ref and "index.sel_pages" in names:
            nfs, dist, extra, worst_rows, dc = compare_selection(a.dump, layer, ref["index.selection"], ref)
            line = (f"{tag} {'index.selection':24s} rows {nfs}, different {dist}, extra or missing groups "
                     f"{extra}; worst (diff, row) {worst_rows}")
            if len(dc):
                line += f"; max distance to cutoff {max(dc):.4g}"
            if noise is not None and "index.selection" in noise and "index.logits" in ref:
                lg, ct = ref["index.logits"], ref["index.cutoff_score"]
                band = 0.0
                for r_ in range(rows):
                    a_ = {int(g) for g in noise["index.selection"][r_] if g >= 0}
                    b_ = {int(g) for g in ref["index.selection"][r_] if g >= 0}
                    for g in a_ ^ b_:
                        band = max(band, abs(float(lg[r_, g]) - float(ct[r_])))
                outside = int(sum(1 for d in dc if d > band))
                outside2 = int(sum(1 for d in dc if d > a.band_factor * band))
                line += (f"; BF16 band {band:.4g}, out of band {outside} "
                          f"(of {a.band_factor:g}× band: {outside2})")
                if outside2:
                    failures.append(f"{tag} index.selection: {outside2} outside {a.band_factor:g}× band")
            print(line)
        # ---- router experts
        if "router.experts" in ref and "router.experts" in names:
            k = ref["router.experts"].shape[1]
            ours = join_steps(a.dump, "router.experts", layer, k).astype(np.int64)
            theirs = ref["router.experts"].astype(np.int64)
            diff = sum(1 for i in range(rows) if set(ours[i].tolist()) != set(theirs[i].tolist()))
            line = f"{tag} {'router.experts':24s} rows with another set {diff} of {rows}"
            if "router.weights" in ref and "router.weights" in names:
                # weights matched by expert, on the rows with the same set
                pm = join_steps(a.dump, "router.weights", layer, k)
                xs, rs = [], []
                for i in range(rows):
                    if set(ours[i].tolist()) == set(theirs[i].tolist()):
                        o = {int(e): j for j, e in enumerate(ours[i])}
                        xs.append([pm[i, o[int(e)]] for e in theirs[i]])
                        rs.append(ref["router.weights"][i])
                if xs:
                    line += f"; weights (equal rows) L2 {l2rel(np.array(xs), np.array(rs)):.3e}"
            if not a.snap or noise is None:
                unmeasured.append(f"{tag} router.experts (missing --snap or --noise)")
            if a.snap:
                sc, cutoff = router_scores(ref["router.logits"], router_bias(a.snap, layer), k)
                d = router_distances(ours, theirs, sc, cutoff)
                if len(d):
                    line += f"; max distance to cutoff {d.max():.4g}"
                if noise is not None and "router.experts" in noise:
                    b = router_distances(noise["router.experts"].astype(np.int64), theirs, sc, cutoff)
                    band = float(b.max()) if len(b) else 0.0
                    outside = int((d > band).sum()) if len(d) else 0
                    outside2 = int((d > a.band_factor * band).sum()) if len(d) else 0
                    line += (f"; BF16 band {band:.4g}, out of band {outside} "
                              f"(of {a.band_factor:g}× band: {outside2})")
                    if outside2:
                        failures.append(f"{tag} router.experts: {outside2} outside {a.band_factor:g}× band")
            print(line)
        for name in names:
            if name.startswith("index.") or name in ("router.experts", "router.weights"):
                continue
            if name not in ref:
                print(f"{tag} {name:24s} no reference")
                continue
            r = ref[name].reshape(rows, -1)
            x = join_steps(a.dump, name, layer, r.shape[1])
            if x.shape != r.shape:
                print(f"{tag} {name:24s} shape {x.shape} vs {r.shape}")
                failures.append(f"{tag} {name}: shape")
                continue
            if np.issubdtype(r.dtype, np.integer):
                dist = int((x != r).sum())
                print(f"{tag} {name:24s} different {dist} of {r.size}")
                if dist:
                    failures.append(f"{tag} {name}: {dist} different")
                continue
            e = l2rel(x, r)
            med, p99, mx = per_row(x, r)
            line = f"{tag} {name:24s} L2 {e:.3e}  per row med {med:.2e} p99 {p99:.2e} max {mx:.2e}"
            if noise is not None and name in noise:
                rr = noise[name].reshape(rows, -1)
                s_ = l2rel(rr, r)
                smed, sp99, smx = per_row(rr, r)
                # Where the HF BF16 pass computes in f32 (iHC), its floor is 0 and the remaining error
                # is f32 summation order: the minimum floor is the FP32 tolerance of the plan step 3
                # ops (docs/OPS-HY4.md, relative L2 < 1e-5). A 0 error is a 0 ratio.
                ratio = 0.0 if e == 0 else e / max(s_, 1e-5)
                worst = max(worst, ratio)
                measured += 1
                line += (f"\n    BF16 floor L2 {s_:.3e}  per row med {smed:.2e} p99 {sp99:.2e} max {smx:.2e}"
                          f"   ratio {ratio:.2f}")
                if ratio > 2:
                    failures.append(f"{tag} {name}: ratio {ratio:.2f}")
            print(line)
    if a.noise and measured:
        print(f"worst ratio {worst:.2f} in {measured} comparisons (criterion <= 2)")
    elif a.noise:
        unmeasured.insert(0, "no ratio computed")
    else:
        unmeasured.insert(0, "ratios and bands (missing --noise)")
    if a.report_only:
        print("report only (--report-only)")
        return 0
    if failures:
        print("FAIL:", "; ".join(failures))
        return 1
    if unmeasured:
        print("NOT MEASURED:", "; ".join(unmeasured))
        return 2
    print("criterion measured and met")
    return 0


if __name__ == "__main__":
    sys.exit(main())
