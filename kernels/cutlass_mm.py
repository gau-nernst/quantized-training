from pathlib import Path

import torch
import torch.utils.cpp_extension
from torch import Tensor

CURRENT_DIR = Path(__file__).parent

_cutlass_mm = torch.utils.cpp_extension.load(
    "cutlass_mm",
    sources=[CURRENT_DIR / "cutlass_mm.cu"],
    extra_cuda_cflags=["-O3"],
    extra_include_paths=[str(CURRENT_DIR / "cutlass/include")],
    verbose=True,
)


def _int4_mm(A: Tensor, B: Tensor) -> Tensor:
    assert A.is_cuda and A.is_contiguous()
    assert B.is_cuda and B.T.is_contiguous()
    return _cutlass_mm.int4_mm(A, B)
