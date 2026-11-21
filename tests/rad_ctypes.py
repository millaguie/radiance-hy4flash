"""Call the host kernels of a radiance plugin from Python, with torch tensors.

Mirrors with ctypes the structs of abi/rad_types.h and abi/rad_abi.h (RadTensor, RadParam, RadArgs,
RadKernelInfo) of radiance 2.2.0, the same as 2.1.0's (ABI 43 appends to other structs). Used to
test each hy4k op against the HF code and against the dumps in ref/ without building a model.
"""
import ctypes as C
from pathlib import Path

import torch

MAX_RANK = 6
P_INT, P_STR, P_RANGE, P_F64 = 1, 2, 3, 4
DT = {torch.float32: 1, torch.float16: 2, torch.bfloat16: 3, torch.int32: 11, torch.int64: 12}


class RadTensor(C.Structure):
    _fields_ = [("data", C.c_void_p), ("dtype", C.c_uint32), ("rank", C.c_uint32),
                ("shape", C.c_int64 * MAX_RANK), ("stride", C.c_int64 * MAX_RANK)]


class RadParam(C.Structure):
    _fields_ = [("key", C.c_char_p), ("kind", C.c_int), ("ival", C.c_longlong), ("ihi", C.c_longlong),
                ("sval", C.c_char_p), ("dval", C.c_double)]


class RadArgs(C.Structure):
    _fields_ = [("t", C.POINTER(RadTensor)), ("n_t", C.c_int), ("p", C.POINTER(RadParam)), ("n_p", C.c_int),
                ("scratch", C.c_void_p), ("scratch_bytes", C.c_int64), ("instance", C.c_void_p),
                ("rank", C.c_int), ("world_size", C.c_int), ("kv_used_max_h", C.c_int32),
                ("wgp_count", C.c_int32)]


LAUNCH = C.CFUNCTYPE(C.c_int, C.POINTER(RadArgs), C.c_void_p)
SCRATCH = C.CFUNCTYPE(C.c_int64, C.POINTER(RadArgs))


class RadKernelInfo(C.Structure):
    # Only the fields up to `scratch`, which is what is used; the rest of the struct is not touched.
    _fields_ = [("name", C.c_char_p), ("op", C.c_char_p), ("family", C.c_char_p), ("computes", C.c_char_p),
                ("shape", C.c_char_p), ("dtypes", C.c_char_p), ("domain", C.c_int), ("priority", C.c_int),
                ("constraints", C.c_void_p), ("n_constraints", C.c_int),
                ("tunables", C.c_void_p), ("n_tunables", C.c_int), ("tune_valid", C.c_void_p),
                ("init", C.c_void_p), ("fini", C.c_void_p), ("launch", LAUNCH), ("scratch", SCRATCH)]


def tensor(t: torch.Tensor) -> RadTensor:
    r = RadTensor()
    if t is None:
        return r
    r.data = t.data_ptr()
    r.dtype = DT[t.dtype]
    r.rank = t.dim()
    for i in range(t.dim()):
        r.shape[i] = t.shape[i]
        r.stride[i] = t.stride(i)
    return r


class Plugin:
    def __init__(self, so: Path):
        self.lib = C.CDLL(str(so))
        self.lib.rad_kernel_count.restype = C.c_int
        self.lib.rad_kernel_at.restype = C.POINTER(RadKernelInfo)
        self.kernels = {}
        for i in range(self.lib.rad_kernel_count()):
            k = self.lib.rad_kernel_at(i).contents
            self.kernels[k.name.decode()] = k

    def call(self, kernel: str, tensors: list, params: dict, scratch=None, stream=None) -> int:
        """`scratch` is (device pointer, bytes) for a kernel with a scratch hook; `stream` a hipStream_t."""
        a, keep = self.args(tensors, params, scratch)
        return self.kernels[kernel].launch(C.byref(a), stream)

    def scratch_bytes(self, kernel: str, tensors: list, params: dict) -> int:
        """What the kernel's scratch hook asks for at this geometry (0 without a hook)."""
        k = self.kernels[kernel]
        if not k.scratch:
            return 0
        a, keep = self.args(tensors, params)
        return int(k.scratch(C.byref(a)))

    def args(self, tensors: list, params: dict, scratch=None):
        """A RadArgs and the objects it points into (keep both alive while it is used)."""
        ts = (RadTensor * len(tensors))(*[tensor(t) for t in tensors])
        ps = []
        keep = []
        for k, v in params.items():
            p = RadParam()
            kb = k.encode()
            keep.append(kb)
            p.key = kb
            if isinstance(v, float):
                p.kind, p.dval = P_F64, v
            elif isinstance(v, str):
                vb = v.encode()
                keep.append(vb)
                p.kind, p.sval = P_STR, vb
            else:
                p.kind, p.ival = P_INT, int(v)
            ps.append(p)
        pa = (RadParam * len(ps))(*ps)
        a = RadArgs(t=ts, n_t=len(tensors), p=pa, n_p=len(ps))
        if scratch is not None:
            a.scratch, a.scratch_bytes = scratch
        return a, (ts, pa, ps, keep)
