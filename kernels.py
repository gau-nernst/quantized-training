import torch
import triton
import triton.language as tl
from torch import Tensor

_LIB_NAME = "qtrain"
lib = torch.library.Library(_LIB_NAME, "FRAGMENT")


# from "Good config for fp8 inputs."
# https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
configs = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128}, num_stages=3, num_warps=8),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128}, num_stages=3, num_warps=8),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 128}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 128}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64}, num_stages=4, num_warps=4),
]


def _grid(meta):
    return (triton.cdiv(meta["M"], meta["BLOCK_M"]) * triton.cdiv(meta["N"], meta["BLOCK_N"]),)


# templated matmul from pytorch
# https://github.com/pytorch/pytorch/blob/c2e2602ecdc2ec1f120e19198dfc18fc39f7bd09/torch/_inductor/kernel/mm.py
@triton.autotune(configs=configs, key=["M", "N", "K"])
@triton.jit
def _int8_mm_kernel(
    # fmt: off
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr = 8,
    EVEN_K: tl.constexpr = True,
    # fmt: on
):
    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    A = A_ptr + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B_ptr + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.0)
            b = tl.load(B, mask=rk[:, None] < k, other=0.0)
        # TODO: quant prologue
        acc += tl.dot(a, b)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # new code here: dequant epilogue
    # inductor generates a suffix
    xindex = idx_m * stride_cm + idx_n * stride_cn
    tl.store(C_ptr + tl.broadcast_to(xindex, mask.shape), acc, mask)


def _int8_mm(A: Tensor, B: Tensor, C: Tensor | None = None):
    assert A.dtype is torch.int8 and B.dtype is torch.int8
    assert A.shape[1] == B.shape[0]
    M, K = A.shape
    _, N = B.shape
    if C is None:
        C = torch.empty(M, N, dtype=torch.int32, device=A.device)
    else:
        assert C.shape == (M, N)

    # fmt: off
    _int8_mm_kernel[_grid](
        A, B, C, #
        M, N, K, #
        *A.stride(),
        *B.stride(),
        *C.stride(),
        EVEN_K=K % 2 == 0,
    )
    # fmt: on
    return C


lib.define("int8_mm(Tensor a, Tensor b) -> Tensor")
int8_mm = getattr(torch.ops, _LIB_NAME).int8_mm
torch.library.impl(lib, "int8_mm", "CUDA")(_int8_mm)


@torch.library.impl(lib, "int8_mm", "Meta")
def _(a: Tensor, b: Tensor):
    return torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.int32)


@triton.autotune(configs=configs, key=["M", "N", "K"])
@triton.jit
def _scaled_int8_mm_kernel(
    # fmt: off
    A_ptr, B_ptr, C_ptr,
    A_scale_rowwise_ptr,
    B_scale_colwise_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr = 8,
    EVEN_K: tl.constexpr = True,
    # fmt: on
):
    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    A = A_ptr + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B_ptr + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.0)
            b = tl.load(B, mask=rk[:, None] < k, other=0.0)
        # TODO: quant prologue
        acc += tl.dot(a, b)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    a_scale = tl.load(A_scale_rowwise_ptr + idx_m, mask=idx_m < M).to(tl.float32)
    b_scale = tl.load(B_scale_colwise_ptr + idx_n, mask=idx_n < N).to(tl.float32)
    acc = acc.to(tl.float32) * a_scale * b_scale

    # new code here: dequant epilogue
    # inductor generates a suffix
    xindex = idx_m * stride_cm + idx_n * stride_cn
    tl.store(C_ptr + tl.broadcast_to(xindex, mask.shape), acc, mask)


def _scaled_int8_mm(
    A: Tensor,
    B: Tensor,
    A_scale_rowwise: Tensor,
    B_scale_colwise: Tensor,
    C: Tensor | None = None,
    dtype: torch.dtype = torch.bfloat16,
):
    assert A.dtype is torch.int8 and B.dtype is torch.int8
    assert A.shape[1] == B.shape[0]
    M, K = A.shape
    _, N = B.shape
    assert A_scale_rowwise.shape == (M,)
    assert B_scale_colwise.shape == (N,)
    if C is None:
        C = torch.empty(M, N, device=A.device, dtype=dtype)
    else:
        assert C.shape == (M, N)

    # fmt: off
    _scaled_int8_mm_kernel[_grid](
        A, B, C, #
        A_scale_rowwise,
        B_scale_colwise,
        M, N, K, #
        *A.stride(),
        *B.stride(),
        *C.stride(),
        EVEN_K=K % 2 == 0,
    )
    # fmt: on
    return C


lib.define("scaled_int8_mm(Tensor a, Tensor b, Tensor a_scale, Tensor b_scale) -> Tensor")
scaled_int8_mm = getattr(torch.ops, _LIB_NAME).scaled_int8_mm
torch.library.impl(lib, "scaled_int8_mm", "CUDA")(_scaled_int8_mm)


@torch.library.impl(lib, "scaled_int8_mm", "Meta")
def _(a: Tensor, b: Tensor, a_scale: Tensor, b_scale: Tensor, dtype: torch.dtype = torch.bfloat16):
    return torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=dtype)
