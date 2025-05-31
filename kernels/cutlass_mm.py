from pathlib import Path

import torch
import torch.utils.cpp_extension
from torch import Tensor

from ._lib import lib, lib_ops

CURRENT_DIR = Path(__file__).parent

_cutlass_mm = torch.utils.cpp_extension.load(
    "cutlass_int4mm",
    sources=[CURRENT_DIR / "cutlass_int4mm.cu"],
    extra_cuda_cflags=["-O3"],
    extra_include_paths=[str(CURRENT_DIR / "cutlass/include")],
    verbose=True,
)


lib.define("int4_mm(Tensor A, Tensor B) -> Tensor")


def int4_mm(A: Tensor, B: Tensor) -> Tensor:
    assert A.is_cuda and A.ndim == 2 and A.dtype is torch.int8 and A.is_contiguous()
    assert B.is_cuda and B.ndim == 2 and B.dtype is torch.int8 and B.T.is_contiguous()
    return lib_ops.int4_mm(A, B)


@torch.library.impl(lib, "int4_mm", "Meta")
def _(A: Tensor, B: Tensor) -> Tensor:
    return torch.empty((A.shape[0], B.shape[1]), device=A.device, dtype=torch.int32)


torch.library.impl(lib, "int4_mm", "CUDA")(_cutlass_mm.int4_mm)


lib.define("scaled_int4_mm(Tensor A, Tensor B, Tensor row_scale, Tensor col_scale) -> Tensor")


def scaled_int4_mm(A: Tensor, B: Tensor, row_scale: Tensor, col_scale: Tensor) -> Tensor:
    assert A.is_cuda and A.ndim == 2 and A.dtype is torch.int8 and A.is_contiguous()
    assert B.is_cuda and B.ndim == 2 and B.dtype is torch.int8 and B.T.is_contiguous()
    assert row_scale.dtype == col_scale.dtype == torch.bfloat16  # only support bfloat16 for now
    assert row_scale.squeeze().shape == (A.shape[0],)
    assert col_scale.squeeze().shape == (B.shape[1],)
    return lib_ops.scaled_int4_mm(A, B, row_scale, col_scale)


@torch.library.impl(lib, "scaled_int4_mm", "Meta")
def _(A: Tensor, B: Tensor, row_scale: Tensor, col_scale: Tensor) -> Tensor:
    return torch.empty((A.shape[0], B.shape[1]), device=A.device, dtype=row_scale.dtype)


torch.library.impl(lib, "scaled_int4_mm", "CUDA")(_cutlass_mm.scaled_int4_mm)
