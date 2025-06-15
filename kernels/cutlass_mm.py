import os
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
def load_extension(major: int, minor: int, a: bool = False):
    if torch.cuda.get_device_capability() >= (major, minor):
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        cc = f"{major}{minor}"

        if a:
            os.environ["TORCH_CUDA_ARCH_LIST"] += "a"
            cc += "a"

        os.environ["TORCH_CUDA_ARCH_LIST"] += "+PTX"

        torch.utils.cpp_extension.load(
            f"cutlass_sm{cc}",
            sources=list(CURRENT_DIR.glob(f"cutlass_sm{cc}_*.cu")),
            # extra_cuda_cflags=["-DCUTLASS_DEBUG_TRACE_LEVEL=1"],
            extra_include_paths=extra_include_paths,
            verbose=True,
            is_python_module=False,
        )

        del os.environ["TORCH_CUDA_ARCH_LIST"]


load_extension(8, 0)
load_extension(8, 9)
load_extension(12, 0, a=True)


lib.define("int4_mm(Tensor A, Tensor B) -> Tensor")
lib.define("scaled_int4_mm(Tensor A, Tensor B, Tensor row_scale, Tensor col_scale) -> Tensor")
lib.define("fp8_mm(Tensor A, Tensor B) -> Tensor")
lib.define("cutlass_fp8_mm(Tensor A, Tensor B) -> Tensor")
lib.define("cutlass_scaled_fp8_mm(Tensor A, Tensor B, Tensor scale_A, Tensor scale_B) -> Tensor")
lib.define("scaled_fp8_mm(Tensor A, Tensor B, Tensor row_scale, Tensor col_scale) -> Tensor")
lib.define("nvfp4_mm(Tensor A, Tensor B, Tensor scale_A, Tensor scale_B) -> Tensor")
lib.define("mxfp4_mm(Tensor A, Tensor B, Tensor scale_A, Tensor scale_B) -> Tensor")


def int4_mm(A: Tensor, B: Tensor) -> Tensor:
    assert A.ndim == 2 and A.dtype is torch.int8 and A.is_contiguous()
    assert B.ndim == 2 and B.dtype is torch.int8 and B.T.is_contiguous()
    return lib_ops.int4_mm(A, B)


@torch.library.impl(lib, "int4_mm", "Meta")
def _(A: Tensor, B: Tensor) -> Tensor:
    return torch.empty((A.shape[0], B.shape[1]), device=A.device, dtype=torch.int32)


def fp8_mm(A: Tensor, B: Tensor) -> Tensor:
    assert A.ndim == 2 and A.dtype is torch.float8_e4m3fn and A.is_contiguous()
    assert B.ndim == 2 and B.dtype is torch.float8_e4m3fn and B.T.is_contiguous()
    if torch.cuda.get_device_capability()[0] == 12:
        return lib_ops.cutlass_fp8_mm(A, B)
    else:
        return lib_ops.fp8_mm(A, B)


@torch.library.impl(lib, "fp8_mm", "Meta")
def _(A: Tensor, B: Tensor) -> Tensor:
    return torch.empty((A.shape[0], B.shape[1]), device=A.device, dtype=torch.bfloat16)


def scaled_int4_mm(A: Tensor, B: Tensor, row_scale: Tensor, col_scale: Tensor) -> Tensor:
    assert A.ndim == 2 and A.dtype is torch.int8 and A.is_contiguous()
    assert B.ndim == 2 and B.dtype is torch.int8 and B.T.is_contiguous()
    assert row_scale.dtype == col_scale.dtype == torch.bfloat16  # only support bfloat16 for now
    assert row_scale.squeeze().shape == (A.shape[0],)
    assert col_scale.squeeze().shape == (B.shape[1],)
    return lib_ops.scaled_int4_mm(A, B, row_scale, col_scale)


def scaled_fp8_mm(A: Tensor, B: Tensor, row_scale: Tensor, col_scale: Tensor) -> Tensor:
    assert A.ndim == 2 and A.is_contiguous()
    assert B.ndim == 2 and B.T.is_contiguous()
    assert A.dtype == B.dtype
    assert A.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    assert row_scale.dtype == col_scale.dtype == torch.float32  # only support float32 for now
    assert row_scale.squeeze().shape == (A.shape[0],)
    assert col_scale.squeeze().shape == (B.shape[1],)

    if torch.cuda.get_device_capability()[0] == 12:
        return lib_ops.cutlass_scaled_fp8_mm(A, B, row_scale, col_scale)
    else:
        return lib_ops.scaled_fp8_mm(A, B, row_scale, col_scale)


@torch.library.impl(lib, "scaled_int4_mm", "Meta")
@torch.library.impl(lib, "scaled_fp8_mm", "Meta")
def _(A: Tensor, B: Tensor, row_scale: Tensor, col_scale: Tensor) -> Tensor:
    return torch.empty((A.shape[0], B.shape[1]), device=A.device, dtype=row_scale.dtype)


# TODO: check scale_A/scale_B shapes
def nvfp4_mm(A: Tensor, B: Tensor, scale_A: Tensor, scale_B: Tensor) -> Tensor:
    assert A.ndim == 2 and A.dtype is torch.uint8 and A.is_contiguous()
    assert B.ndim == 2 and B.dtype is torch.uint8 and B.T.is_contiguous()
    assert A.shape[1] == B.shape[0]
    return lib_ops.nvfp4_mm(A, B, scale_A, scale_B)


def mxfp4_mm(A: Tensor, B: Tensor, scale_A: Tensor, scale_B: Tensor) -> Tensor:
    assert A.ndim == 2 and A.dtype is torch.uint8 and A.is_contiguous()
    assert B.ndim == 2 and B.dtype is torch.uint8 and B.T.is_contiguous()
    assert A.shape[1] == B.shape[0]
    return lib_ops.mxfp4_mm(A, B, scale_A, scale_B)


@torch.library.impl(lib, "cutlass_scaled_fp8_mm", "Meta")
@torch.library.impl(lib, "nvfp4_mm", "Meta")
@torch.library.impl(lib, "mxfp4_mm", "Meta")
def _(A: Tensor, B: Tensor, scale_A: Tensor, scale_B: Tensor) -> Tensor:
    return torch.empty((A.shape[0], B.shape[1]), device=A.device, dtype=torch.bfloat16)
