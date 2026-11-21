#!/usr/bin/env python3
"""Per-position KL (kld-ref of the trimmed model) on the rows whose router picks another expert,
and on those outside the BF16 band. Usage: kl_bands.py <HF snapshot>. Measures whether
--band-factor 1 or 2 is better. HY4_REF_ROOT is the directory with the reference passes fp32-1 and
bf16-1 (default ~/hy4/ref); the kld rows and the plugin dumps are read from HY4_DIR (default ~/hy4)."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare as C
snap = sys.argv[1]
D = os.environ.get("HY4_DIR") or os.path.expanduser("~/hy4")
R = os.environ.get("HY4_REF_ROOT") or os.path.join(D, "ref")
kl = np.fromfile(os.path.join(D, "kld-recorte3-host.json.rows"), np.float32).reshape(-1, 4)[:, 0]
off = {"corto": 0, "medio": 283, "tools": 2082, "largo": 2404}
n = {"corto": 283, "medio": 1799, "tools": 322, "largo": 8447}
print(f"per-position KL, all: mean {kl.mean():.2e} p50 {np.median(kl):.2e} p99 {np.percentile(kl,99):.2e}")
changed = set(); outside = {}
for p in off:
    for layer in (1, 2):
        ref = C.load_file(f"{R}/fp32-1/{p}.L{layer:02d}.safetensors")
        noise_l = C.load_file(f"{R}/bf16-1/{p}.L{layer:02d}.safetensors")
        k = ref["router.experts"].shape[1]
        ours = C.join_steps(os.path.join(D, f"dump-{p}"), "router.experts", layer, k).astype(np.int64)
        theirs = ref["router.experts"].astype(np.int64)
        sc, cutoff = C.router_scores(ref["router.logits"], C.router_bias(snap, layer), k)
        band = C.router_distances(noise_l["router.experts"].astype(np.int64), theirs, sc, cutoff).max()
        for i in range(min(len(ours), n[p])):
            diff = set(ours[i].tolist()) ^ set(theirs[i].tolist())
            if diff:
                changed.add(off[p] + i)
                d = max(abs(sc[i, e] - cutoff[i]) for e in diff)
                if d > band:
                    outside[off[p] + i] = (p, layer, i, d / band)
print(f"positions with some different expert: {len(changed)}; their mean KL {kl[list(changed)].mean():.2e}")
same = np.setdiff1d(np.arange(len(kl)), list(changed))
print(f"positions without changes: mean KL {kl[same].mean():.2e}")
pct = lambda v: 100 * (kl < v).mean()
for r, (p, layer, i, x) in sorted(outside.items()):
    print(f"  out of band: {p} L{layer:02d} row {i} ({x:.2f}× band): KL {kl[r]:.2e}, percentile {pct(kl[r]):.0f}")
