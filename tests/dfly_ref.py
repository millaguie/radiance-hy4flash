"""Plain-Python models of the DFly2 ops (docs/OPS-HY4.md: hy4_ring_store, hy4_ring_load and
hy4_mla_attn's block keys), written from the contract, for tests/test_ops_extra.py (against the
host oracle) and tests/test_dev.py (hand-built cases for the card)."""
import torch


def words(t: torch.Tensor) -> torch.Tensor:
    """A source row as the i32 words the ring keeps: integers truncated, floats as f32 bits."""
    if t.dtype in (torch.int32, torch.int64):
        return t.to(torch.int32)
    return t.float().contiguous().view(torch.int32)


def seq_of(cu, r):
    for s in range(len(cu) - 1):
        if int(cu[s]) <= r < int(cu[s + 1]):
            return s
    return None


def ring_store(state3, mode, src, row_ids, pos, cu, sidx, ring, width, offset, block=1):
    """state3: [slots, ring, W] (a view of the state); writes in place."""
    for k, r in enumerate(row_ids.tolist()):
        s = seq_of(cu, r)
        if s is None:
            continue
        slot = int(sidx.view(len(sidx), -1)[s, 0])
        if slot < 0:
            continue
        p = int(pos[r])
        if mode == "pos":
            w = torch.tensor([p], dtype=torch.int32)
        elif mode == "row":
            w = words(src[r].reshape(-1)[:width])
        elif mode == "out":
            w = words(src[k].reshape(-1)[:width])
        else:
            last = p // block if p >= 0 else -1
            bt = words(src[s])
            pages = [int(bt[c]) if c <= last and c < len(bt) else -1 for c in range(width - 1)]
            w = torch.tensor([p + 1] + pages, dtype=torch.int32)
        state3[slot, p % ring, offset:offset + width] = w


def ring_load(state3, dst, pos, cu, sidx, ring, width, offset, fill=-1):
    """dst: [rows or S, width] i32 (f32 destinations are compared as their bits); writes in place.
    With cu, the rows outside every span get `fill` too."""
    S = len(cu) - 1 if cu is not None else len(pos)
    if cu is not None:
        for r in range(len(pos)):
            if seq_of(cu, r) is None:
                dst[r, :width] = fill
    for s in range(S):
        r0, r1 = (int(cu[s]), int(cu[s + 1])) if cu is not None else (s, s + 1)
        if r1 <= r0:
            continue
        slot = int(sidx.view(len(sidx), -1)[s, 0])
        p0 = int(pos[r0])
        ok = slot >= 0 and p0 >= 1 and int(state3[slot, (p0 - 1) % ring, 0]) == p0 - 1
        for r in range(r0, r1):
            dst[r, :width] = state3[slot, (p0 - 1) % ring, offset:offset + width] if ok else fill


def union_attn(q, qr, cache, sel, sequ, bkeys, bcu, sinks, scale, bs):
    """hy4_mla_attn in the selection form with block keys, in double: row i of block sequence d
    attends to its selection (tokens < sequ[i] of its page list, valid pages only) and to bkeys
    rows bcu[d] .. i, with the head's sink. Returns [rows, H, L] and the mask of rows inside bcu."""
    rows, H, L = q.shape
    P = cache.shape[0]
    out = torch.zeros(rows, H, L, dtype=torch.float64)
    inside = torch.zeros(rows, dtype=torch.bool)
    for i in range(rows):
        d = seq_of(bcu, i)
        if d is None:
            continue
        inside[i] = True
        keys = []
        for j in range(int(sequ[i])):
            pg = int(sel[i, j // bs]) if j // bs < sel.shape[1] else -1
            if 0 <= pg < P:
                keys.append(cache[pg, 0, j % bs].double())
        for b in range(int(bcu[d]), i + 1):
            keys.append(bkeys[b].double())
        K = torch.stack(keys)                      # [n, L + R]
        for h in range(H):
            qq = torch.cat([q[i, h].double(), qr[i, h].double()])
            lg = torch.cat([(K @ qq) * scale, sinks[h:h + 1].double()])
            out[i, h] = torch.softmax(lg, 0)[:-1] @ K[:, :L]
    return out, inside
