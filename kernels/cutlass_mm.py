from pathlib import Path

import torch
import torch.utils.cpp_extension
from torch import Tensor

from ._lib import lib, lib_ops

CURRENT_DIR = Path(__file__).parent

extra_include_paths = [
    str(CURRENT_DIR / "cutlass/include"),
    str(CURRENT_DIR / "cutlass/tools/util/include"),
]

# TODO: figure out a way to remove default -gencode=...
torch.utils.cpp_extension.load(
    "cutlass_sm80",
    sources=[CURRENT_DIR / "cutlass_int4.cu"],
    extra_cuda_cflags=["-gencode=arch=compute_80,code=sm_80"],
    extra_include_paths=extra_include_paths,
    verbose=True,
    is_python_module=False,
)

if torch.cuda.get_device_capability() == (12, 0):
    torch.utils.cpp_extension.load(
        "cutlass_sm120a",
        sources=[CURRENT_DIR / "cutlass_fp4.cu"],
        extra_cuda_cflags=["-gencode=arch=compute_120a,code=sm_120a"],
        extra_include_paths=extra_include_paths,
        verbose=True,
        is_python_module=False,
    )


lib.define("int4_mm(Tensor A, Tensor B) -> Tensor")
lib.define("scaled_int4_mm(Tensor A, Tensor B, Tensor row_scale, Tensor col_scale) -> Tensor")
lib.define("nvfp4_mm(Tensor A, Tensor B, Tensor scale_A, Tensor scale_B) -> Tensor")
lib.define("mxfp4_mm(Tensor A, Tensor B, Tensor scale_A, Tensor scale_B) -> Tensor")


def int4_mm(A: Tensor, B: Tensor) -> Tensor:
    assert A.is_cuda and A.ndim == 2 and A.dtype is torch.int8 and A.is_contiguous()
    assert B.is_cuda and B.ndim == 2 and B.dtype is torch.int8 and B.T.is_contiguous()
    return lib_ops.int4_mm(A, B)


@torch.library.impl(lib, "int4_mm", "Meta")
def _(A: Tensor, B: Tensor) -> Tensor:
    return torch.empty((A.shape[0], B.shape[1]), device=A.device, dtype=torch.int32)


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


def nvfp4_mm(A: Tensor, B: Tensor, scale_A: Tensor, scale_B: Tensor) -> Tensor:
    assert A.is_cuda and A.ndim == 2 and A.dtype is torch.uint8 and A.is_contiguous()
    assert B.is_cuda and B.ndim == 2 and B.dtype is torch.uint8 and B.T.is_contiguous()
    assert A.shape[1] == B.shape[0]
    return lib_ops.nvfp4_mm(A, B, scale_A, scale_B)


def mxfp4_mm(A: Tensor, B: Tensor, scale_A: Tensor, scale_B: Tensor) -> Tensor:
    assert A.is_cuda and A.ndim == 2 and A.dtype is torch.uint8 and A.is_contiguous()
    assert B.is_cuda and B.ndim == 2 and B.dtype is torch.uint8 and B.T.is_contiguous()
    assert A.shape[1] == B.shape[0]
    return lib_ops.mxfp4_mm(A, B, scale_A, scale_B)


@torch.library.impl(lib, "nvfp4_mm", "Meta")
@torch.library.impl(lib, "mxfp4_mm", "Meta")
def _(A: Tensor, B: Tensor, scale_A: Tensor, scale_B: Tensor) -> Tensor:
    return torch.empty((A.shape[0], B.shape[1]), device=A.device, dtype=torch.bfloat16)
