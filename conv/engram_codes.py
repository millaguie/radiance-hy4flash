#!/usr/bin/env python3
"""Hy4-flash's Engram tables as the E4M3 codes Engram reads them as, for `rad-convert --add`.

Engram quantises each row it looks up (one head's 96 values) to E4M3 under the row's own scale
before the per-head RMSNorm (HF `_fp8_round_trip`, config `engram_fp8_embedding`); the plugin does
the same at run time on the bf16 table (`hy4_rows_rmsnorm` with fp8=1, docs/OPS-HY4.md). The
round trip is equivariant to a row's scale and the RMSNorm cancels it, so the codes alone, with the
row scales dropped and one scale of 1 for the table, give the rows the model reads, to the norm's
eps. Re-quantising the bf16 table under one scale for all of it would not: the model's own round
trip then rounds it a second time (4.2% error at the RMSNorm's output, where this is ~1e-7).

The codes are computed as the plugin's round trip computes them, in FP32 (kernels/hy4k/
ref_engram.cpp, `hy4_ref_rows_rmsnorm`): amax over the row, scale = max(amax, 1e-12) / 448,
e4m3(clamp(x / scale, +-448)), round to nearest even. A zero row gives zero codes, as there. HF
in bf16 rounds the scale and the division to bf16 first; the plugin chose FP32 (exact against the
FP32 reference), and these codes keep what the served container computes today.

The codes fix the Engram RMSNorm's eps. They give the rows the model reads only at the eps the
plugin uses, 1.1920929e-07 (`hy4_rows_rmsnorm`'s `eps` in hyv4flash.cpp, ENGRAM_EPS in
ref/reference.py): a nonzero row of codes has mean(c^2) of at least 448^2/96 = 2091, next to which
the eps is nothing (eps / (2 mean) < 3e-11), so the codes path normalises as if eps were 0, and the
bf16 path differs from that by about eps / (2 mean(x^2)) of the row, which this script requires to
be at most 1e-6 on every row. HF's `nn.RMSNorm`
with `eps=None` takes the eps of the input dtype, 0.0078 in bf16, which next to these rows' mean(x^2)
is not small: at that eps the codes path would differ from the bf16 one (measured 0.5% in mean
and 3.6% at most a head). A container with the bf16 table can still change the eps (an op parameter);
a container with the codes cannot, so a different eps means converting the bf16 table again.

Writes one safetensors file, per Engram layer L (from the checkpoint's index):
- `model.layers.L.engram_layer.multi_head_embedding.embedding.shard_0.weight` F8_E4M3 [rows, 96]
- `model.layers.L.engram_layer.multi_head_embedding.embedding.weight_scale` BF16 [1], 1.0
The names are the ones rad-convert reads as a sharded FP8 table under one scale
(core/format/checkpoint.cpp, Qwen3.8-Flash-Next-FP8's n-gram table), kept byte for byte as
`fp8_e4m3*bf16[*x*]` with no recipe rule; they differ from the bf16 table's name, so `--add`
does not collide with the checkpoint, and the plugin's name map takes the codes when they are
there and the bf16 table when not.

The tables are streamed in chunks of rows (never whole in memory) and written to <out>.partial.
Every row is checked as it is written (the equivalence above holds row by row, so a sample would
not do): all its values are finite, its codes' max is exactly 448 or the codes and the row are all
zero, and eps / (2 mean(x^2)) <= 1e-6 for a nonzero row. The first row that fails aborts the run
with its layer and index, leaving the .partial. Then <out>.partial is read back whole (its BLAKE2b
against the bytes written) and a sample of rows is checked again against codes computed afresh;
only then is it renamed to <out>, and for a full run <out>.verified is written beside it (size,
mtime, BLAKE2b). An existing <out>, <out>.partial or <out>.verified is refused, never overwritten.

A test or plan output cannot pass for the overlay: --header-only writes only to a name ending in
`.header-only.safetensors`, --rows N only to one ending in `.rowsN.safetensors`, a full run to
neither, and only a full run writes the .verified marker that deploy/conv-engram-e4m3.sh checks
(`--check-verified`) before rad-convert.

Usage:
  engram_codes.py <snapshot> <out.safetensors>               the overlay (63.4 GB for the 4 tables)
  engram_codes.py <snapshot> --verify [ROWS]                 checks only, ROWS sampled rows a table
  engram_codes.py <snapshot> <out.rowsN.safetensors> --rows N
                                                             only each table's first N rows (a test)
  engram_codes.py <snapshot> <out.header-only.safetensors> --header-only
                                                             the full header over a sparse file of
                                                             the full size, for `rad-convert
                                                             --plan-only` only: its codes are zeros
  engram_codes.py --check-verified <out> [--rehash]          exit 0 only for a full overlay whose
                                                             .verified marker matches it (--rehash:
                                                             its BLAKE2b too, a read of 63.4 GB)
The checks (--verify, and the sample read back after writing):
  (a) the plugin's runtime round trip + RMSNorm on the codes against the same on the bf16 rows;
  (b) the codes re-quantised per row are the codes (a row's max is 448 or the row is zero);
  (c) every code is E4M3's nearest value to x / scale (ties to even), against an independent
      search of the 256 codes, so the cast is the kernels' rounding;
  and, for information, how many codes HF's round trip in bf16 would choose differently.
"""
import hashlib
import json
import os
import re
import struct
import sys
from pathlib import Path

import numpy as np
import torch

F8 = torch.float8_e4m3fn
FMAX = 448.0
EPS = 1.1920929e-07          # the Engram norms' eps (hyv4flash.cpp, hy4_rows_rmsnorm's `eps`)
EPS_REL = 1e-6               # the largest eps / (2 mean(x^2)) a nonzero row may have
CHUNK = 1 << 20              # rows a chunk: 192 MiB of bf16 in, 96 MiB of codes out
NAME = "model.layers.{}.engram_layer.multi_head_embedding.embedding.weight"
CODES = "model.layers.{}.engram_layer.multi_head_embedding.embedding.shard_0.weight"
SCALE = "model.layers.{}.engram_layer.multi_head_embedding.embedding.weight_scale"
BF16_ONE = struct.pack("<H", 0x3F80)


def tables(snap: Path) -> list[tuple[int, Path, int, int, int]]:
    """(layer, file, data offset, rows, d) for each Engram table, from the index and the headers."""
    wm = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    pat = re.compile(r"model\.layers\.(\d+)\.engram_layer\.multi_head_embedding\.embedding\.weight$")
    out = []
    for name, fn in wm.items():
        m = pat.match(name)
        if not m:
            continue
        p = snap / fn
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        t = h[name]
        if t["dtype"] != "BF16" or len(t["shape"]) != 2:
            sys.exit(f"engram_codes: {name} is {t['dtype']} {t['shape']}, not a bf16 matrix")
        rows, d = t["shape"]
        if t["data_offsets"][1] - t["data_offsets"][0] != rows * d * 2:
            sys.exit(f"engram_codes: {name}'s byte range does not match its shape")
        out.append((int(m.group(1)), p, 8 + n + t["data_offsets"][0], rows, d))
    if not out:
        sys.exit("engram_codes: the checkpoint has no Engram table")
    return sorted(out)


def read_rows(p: Path, off: int, d: int, r0: int, n: int) -> torch.Tensor:
    """Rows [r0, r0 + n) of a bf16 table as f32."""
    with open(p, "rb") as f:
        f.seek(off + r0 * d * 2)
        raw = f.read(n * d * 2)
    if len(raw) != n * d * 2:
        sys.exit(f"engram_codes: short read in {p}")
    u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
    return torch.from_numpy(u.view(np.float32).reshape(n, d).copy())


def codes_of(x: torch.Tensor) -> torch.Tensor:
    """The plugin's round trip's codes, in FP32 (hy4_ref_rows_rmsnorm)."""
    amax = x.abs().amax(-1, keepdim=True)
    sc = torch.clamp(amax, min=1e-12) / FMAX
    return (x / sc).clamp(-FMAX, FMAX).to(F8)


def runtime(v: torch.Tensor) -> torch.Tensor:
    """hy4_ref_rows_rmsnorm with fp8=1 on one head's rows (the norm weight left out: it multiplies
    both paths alike)."""
    amax = v.abs().amax(-1, keepdim=True)
    sc = torch.clamp(amax, min=1e-12) / FMAX
    qq = (v / sc).clamp(-FMAX, FMAX).to(F8).to(torch.float32) * sc
    v = v + (qq - v)
    return v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + EPS)


E4M3_VALUES = torch.arange(256, dtype=torch.uint8).view(F8).to(torch.float32)


def check(x: torch.Tensor, c: torch.Tensor) -> dict:
    """The checks (a)-(c) on bf16 rows x (as f32) and their codes c."""
    cf = c.to(torch.float32)
    # (a) today's path on the bf16 rows against the path on the codes (gathered at scale 1: an
    # E4M3 value is exact in bf16, so embed_lookup_q's output is cf as it is)
    ref, new = runtime(x), runtime(cf)
    err = (new - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-30)
    # (b) re-quantised per row, the codes are the codes; a row's max is 448 or the row is zero
    amax = cf.abs().amax(-1)
    ident = bool(torch.equal(codes_of(cf).view(torch.uint8), c.view(torch.uint8)))
    max_ok = bool(((amax == FMAX) | (amax == 0)).all())
    # (c) the cast is round to nearest, ties to even, against a search of the 256 codes
    sc = torch.clamp(x.abs().amax(-1, keepdim=True), min=1e-12) / FMAX
    q = (x / sc).clamp(-FMAX, FMAX)
    finite = torch.isfinite(E4M3_VALUES)
    vals, order = torch.sort(E4M3_VALUES[finite])
    codes_sorted = torch.arange(256)[finite][order]
    i = torch.searchsorted(vals, q.reshape(-1)).clamp(1, len(vals) - 1)
    lo, hi = vals[i - 1], vals[i]
    qf = q.reshape(-1)
    pick_hi = (hi - qf) < (qf - lo)
    tie = (hi - qf) == (qf - lo)
    even_hi = (codes_sorted[i] & 1) == 0
    pick_hi = pick_hi | (tie & even_hi)
    near = torch.where(pick_hi, hi, lo)
    rne = bool((near == cf.reshape(-1)).all())   # by value: -0 and +0 are one
    # HF in bf16 (the dtype the reference model runs in): scale and division rounded to bf16
    xb = x.to(torch.bfloat16)
    sb = xb.abs().amax(-1, keepdim=True).clamp(min=1e-12) / FMAX
    hf = (xb / sb).clamp(-FMAX, FMAX).to(F8)
    hf_diff = float((hf.view(torch.uint8) != c.view(torch.uint8)).float().mean())
    return dict(rows=int(x.shape[0]), zero_rows=int((amax == 0).sum()),
                a_rel_err_mean=float(err.mean()), a_rel_err_max=float(err.max()),
                a_abs_err_max=float((new - ref).abs().max()), b_identity=ident, b_rowmax_448_or_0=max_ok,
                c_round_nearest_even=rne, hf_bf16_codes_differ=hf_diff)


def ok(r: dict) -> bool:
    return r["a_rel_err_max"] < 1e-6 and r["b_identity"] and r["b_rowmax_448_or_0"] and r["c_round_nearest_even"]


def sample_rows(rows: int, n: int, seed: int) -> list[tuple[int, int]]:
    """Runs of 4096 rows at random places, n rows in all (the first and last run included)."""
    run = 4096
    k = max(1, n // run)
    rng = np.random.default_rng(seed)
    starts = {0, max(0, rows - run)} | set(int(s) for s in rng.integers(0, max(1, rows - run), k))
    return [(s, min(run, rows - s)) for s in sorted(starts)][: k + 2]


def verify(snap: Path, n: int) -> bool:
    good = True
    for L, p, off, rows, d in tables(snap):
        xs = torch.cat([read_rows(p, off, d, s, c) for s, c in sample_rows(rows, n, L)])
        r = check(xs, codes_of(xs))
        good &= ok(r)
        print(f"layer {L}: {json.dumps(r)}", flush=True)
    return good


def kind_of(out: Path) -> tuple[str, int | None]:
    """What an output name may hold: ("header-only", None), ("rows", N) or ("full", None)."""
    if out.name.endswith(".header-only.safetensors"):
        return "header-only", None
    m = re.search(r"\.rows(\d+)\.safetensors$", out.name)
    if m:
        return "rows", int(m.group(1))
    if not out.name.endswith(".safetensors"):
        sys.exit(f"engram_codes: {out} does not end in .safetensors")
    return "full", None


class RowCheck:
    """The equivalence invariants on every row written, and their counts for the summary."""

    def __init__(self, L: int):
        self.L, self.rows, self.zero, self.ms_min, self.rel_max = L, 0, 0, float("inf"), 0.0

    def __call__(self, x: torch.Tensor, c: torch.Tensor, r0: int) -> None:
        def die(mask: torch.Tensor, what: str) -> None:
            r = r0 + int(mask.nonzero()[0, 0])
            sys.exit(f"engram_codes: layer {self.L}, row {r}: {what}; the run stops, its .partial stays")

        fin = torch.isfinite(x).all(-1)
        if not fin.all():
            die(~fin, "a value is not finite")
        cf = c.to(torch.float32)
        camax = cf.abs().amax(-1)
        czero = camax == 0
        if not ((camax == FMAX) | czero).all():
            die(~((camax == FMAX) | czero), "its codes' max is neither 448 nor 0")
        xzero = x.abs().amax(-1) == 0
        if not torch.equal(czero, xzero):
            die(czero != xzero, "its codes are all zero and the row is not (or the other way round)")
        ms = x.pow(2).mean(-1)
        rel = EPS / (2 * ms)
        far = ~xzero & ~(rel <= EPS_REL)   # a mean that underflows to 0 is caught too (inf)
        if far.any():
            r = r0 + int(far.nonzero()[0, 0])
            sys.exit(f"engram_codes: layer {self.L}, row {r}: mean(x^2) {float(ms[r - r0]):.3e} puts the "
                     f"Engram eps {EPS:.3e} within {float(rel[r - r0]):.2e} of it (> {EPS_REL:g}): the codes "
                     f"would not give this row; the run stops, its .partial stays")
        self.rows += int(x.shape[0])
        self.zero += int(xzero.sum())
        nz = ~xzero
        if nz.any():
            self.ms_min = min(self.ms_min, float(ms[nz].min()))
            self.rel_max = max(self.rel_max, float(rel[nz].max()))

    def summary(self) -> str:
        return (f"layer {self.L}: {self.rows} rows checked, {self.zero} zero, all finite, row max 448 "
                f"or zero; min mean(x^2) {self.ms_min:.3e}, max eps/(2 mean) {self.rel_max:.2e} (<= {EPS_REL:g})")


def marker_of(out: Path) -> Path:
    return out.with_name(out.name + ".verified")


def header_of(p: Path) -> tuple[dict, int]:
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def blake(p: Path) -> str:
    h = hashlib.blake2b(digest_size=32)
    with open(p, "rb") as f:
        while b := f.read(64 << 20):
            h.update(b)
    return h.hexdigest()


def check_verified(out: Path, rehash: bool) -> bool:
    """A full overlay whose marker matches it: what deploy/conv-engram-e4m3.sh feeds rad-convert."""
    def no(why: str) -> bool:
        print(f"engram_codes: {out} is not a verified overlay: {why}", file=sys.stderr)
        return False

    if kind_of(out)[0] != "full":
        return no("its name is a test's or a plan's")
    mk = marker_of(out)
    if not out.is_file() or not mk.is_file():
        return no(f"{'it' if not out.is_file() else mk.name} is not there")
    m = json.loads(mk.read_text())
    st = out.stat()
    if m.get("kind") != "full" or m.get("file") != out.name or m.get("size") != st.st_size or \
            m.get("mtime_ns") != st.st_mtime_ns:
        return no(f"{mk.name} does not match the file (size, mtime or name)")
    meta = header_of(out)[0].get("__metadata__", {})
    if "header_only" in meta or "rows" in meta:
        return no("its header says it is a test's or a plan's")
    if rehash and blake(out) != m.get("blake2b"):
        return no("its BLAKE2b differs from the marker's")
    print(f"engram_codes: {out} verified ({st.st_size} bytes{', BLAKE2b matches' if rehash else ''})")
    return True


def write(snap: Path, out: Path, limit: int | None, header_only: bool) -> bool:
    kind, nrows = kind_of(out)
    want = "header-only" if header_only else "rows" if limit is not None else "full"
    if kind != want or (kind == "rows" and nrows != limit):
        sys.exit(f"engram_codes: a {want} output goes to a name ending in "
                 + {"header-only": "'.header-only.safetensors'", "rows": f"'.rows{limit}.safetensors'",
                    "full": "'.safetensors' that is neither of those"}[want]
                 + f", not {out.name}: a test or plan output must not pass for the overlay")
    if header_only and limit is not None:
        sys.exit("engram_codes: --header-only and --rows together")
    part, mk = out.with_name(out.name + ".partial"), marker_of(out)
    for f in (out, part, mk):
        if f.exists():
            sys.exit(f"engram_codes: {f} already exists; nothing is overwritten")
    ts = tables(snap)
    hdr, pos, plan = {}, 0, []
    for L, p, off, rows, d in ts:
        n = rows if limit is None else min(rows, limit)
        hdr[CODES.format(L)] = {"dtype": "F8_E4M3", "shape": [n, d], "data_offsets": [pos, pos + n * d]}
        plan.append((L, p, off, n, d, pos))
        pos += n * d
    for L, *_ in ts:   # the scales after the tables, so every table starts 8-aligned
        hdr[SCALE.format(L)] = {"dtype": "BF16", "shape": [1], "data_offsets": [pos, pos + 2]}
        pos += 2
    meta = {"format": "pt", "source": "conv/engram_codes.py",
            "what": "Engram rows as E4M3 codes under each row's amax/448 (FP32), row scales dropped, scale 1",
            "engram_eps": repr(EPS)}
    if header_only:
        meta["header_only"] = "the codes are zeros: for rad-convert --plan-only, never for a container"
    if limit is not None:
        meta["rows"] = str(limit)
    hdr["__metadata__"] = meta
    hj = json.dumps(hdr, separators=(",", ":")).encode()
    hj += b" " * (-(8 + len(hj)) % 8)
    base = 8 + len(hj)
    total = base + pos
    print(f"engram_codes: {len(ts)} tables, {pos / 1e9:.2f} GB of codes into {part}", flush=True)
    hw = hashlib.blake2b(digest_size=32)
    checks = []
    with open(part, "xb") as f:
        head = struct.pack("<Q", len(hj)) + hj
        f.write(head)
        hw.update(head)
        if header_only:
            f.truncate(total)
            # the codes stay a hole (zeros); the scales are the overlay's, 1.0, so the stand-in
            # differs from it only in the codes
            f.seek(total - len(BF16_ONE) * len(ts))
            f.write(BF16_ONE * len(ts))
            assert f.tell() == total
        else:
            for L, p, off, n, d, at in plan:
                assert f.tell() == base + at
                ck = RowCheck(L)
                for r0 in range(0, n, CHUNK):
                    c = min(CHUNK, n - r0)
                    x = read_rows(p, off, d, r0, c)
                    q = codes_of(x)
                    ck(x, q, r0)
                    b = q.view(torch.uint8).numpy().tobytes()
                    f.write(b)
                    hw.update(b)
                    print(f"\r  layer {L}: {r0 + c}/{n} rows", end="", flush=True)
                print(flush=True)
                checks.append(ck)
            f.write(BF16_ONE * len(ts))
            hw.update(BF16_ONE * len(ts))
            assert f.tell() == total
        f.flush()
        os.fsync(f.fileno())
    if header_only:
        return publish(part, out, None)
    for ck in checks:
        print(f"engram_codes: {ck.summary()}")
    print(f"engram_codes: {sum(c.rows for c in checks)} rows checked, {sum(c.zero for c in checks)} zero, "
          f"no violation", flush=True)
    # Read the .partial back: whole, against the bytes written, and a sample against codes computed again.
    digest = hw.hexdigest()
    good = blake(part) == digest
    print(f"engram_codes: read back whole: {'matches' if good else 'DOES NOT MATCH'} what was written")
    with open(part, "rb") as f:
        for L, p, off, n, d, at in plan:
            for s, c in sample_rows(n, 1 << 16, L + 1):
                f.seek(base + at + s * d)
                got = torch.from_numpy(np.frombuffer(f.read(c * d), dtype=np.uint8).reshape(c, d).copy()).view(F8)
                x = read_rows(p, off, d, s, c)
                same = torch.equal(got.view(torch.uint8), codes_of(x).view(torch.uint8))
                good &= same and ok(check(x, got))
        f.seek(base + pos - 2 * len(ts))
        good &= f.read(2 * len(ts)) == BF16_ONE * len(ts)
    print(f"engram_codes: sample read back {'matches' if good else 'DOES NOT MATCH'}")
    if not good:
        print(f"engram_codes: {part} stays as it is, not renamed", file=sys.stderr)
        return False
    return publish(part, out, None if limit is not None else
                   {"kind": "full", "file": out.name, "size": total, "blake2b": digest, "snapshot": str(snap),
                    "rows": sum(c.rows for c in checks), "zero_rows": sum(c.zero for c in checks),
                    "max_eps_rel": max(c.rel_max for c in checks), "engram_eps": EPS})


def publish(part: Path, out: Path, marker: dict | None) -> bool:
    """<out>.partial to <out>, then (a full run) the marker, written whole and renamed in place."""
    if out.exists():
        sys.exit(f"engram_codes: {out} appeared while writing; the result stays in {part}")
    os.rename(part, out)
    if marker is None:
        print(f"engram_codes: {out} ({out.stat().st_size} bytes; not an overlay for a conversion)")
        return True
    marker["mtime_ns"] = out.stat().st_mtime_ns
    mk = marker_of(out)
    tmp = mk.with_name(mk.name + ".tmp")
    with open(tmp, "x") as f:
        f.write(json.dumps(marker, indent=1) + "\n")
        f.flush()
        os.fsync(f.fileno())
    if mk.exists():
        sys.exit(f"engram_codes: {mk} appeared while writing; the marker stays in {tmp}")
    os.rename(tmp, mk)
    print(f"engram_codes: {out} written and verified ({mk.name})")
    return True


def main() -> None:
    a = sys.argv[1:]
    if len(a) < 2:
        sys.exit(__doc__)
    if a[0] == "--check-verified":
        sys.exit(0 if check_verified(Path(a[1]), "--rehash" in a[2:]) else 1)
    torch.set_num_threads(os.cpu_count() or 1)
    snap = Path(a[0])
    if a[1] == "--verify":
        sys.exit(0 if verify(snap, int(a[2]) if len(a) > 2 else 1 << 18) else 1)
    out, rest = Path(a[1]), a[2:]
    limit = int(rest[rest.index("--rows") + 1]) if "--rows" in rest else None
    sys.exit(0 if write(snap, out, limit, "--header-only" in rest) else 1)


if __name__ == "__main__":
    main()
