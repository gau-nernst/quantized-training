from pathlib import Path

import torch
import torch.utils.cpp_extension
from torch import Tensor

from ._lib import lib, lib_ops

CURRENT_DIR = Path(__file__).parent

_cutlass_mm = torch.utils.cpp_extension.load(
    "cutlass_mm",
    sources=[CURRENT_DIR / "cutlass_mm.cu"],
    extra_cuda_cflags=["-O3"],
    extra_include_paths=[str(CURRENT_DIR / "cutlass/include")],
    verbose=True,
)


lib.define("int4_mm(Tensor A, Tensor B) -> Tensor")


def int4_mm(A: Tensor, B: Tensor) -> Tensor:
    assert A.is_cuda and A.ndim == 2 and A.dtype is torch.int32 and A.is_contiguous()
    assert B.is_cuda and B.ndim == 2 and B.dtype is torch.int32 and B.T.is_contiguous()
    return lib_ops.int4_mm(A, B)


@torch.library.impl(lib, "int4_mm", "Meta")
def _(A: Tensor, B: Tensor) -> Tensor:
    return torch.empty((A.shape[0], B.shape[1]), device=A.device, dtype=torch.int32)


torch.library.impl(lib, "int4_mm", "CUDA")(_cutlass_mm.int4_mm)


# TODO: fused output scaling to cutlass kernel
def int4_mm_dequant(A: Tensor, B: Tensor, row_scale: Tensor, col_scale: Tensor) -> Tensor:
    assert row_scale.dtype is col_scale.dtype
    assert row_scale.squeeze().shape == (A.shape[0],)
    assert col_scale.squeeze().shape == (B.shape[1],)
    return int4_mm(A, B) * col_scale.view(1, -1) * row_scale.view(-1, 1)
