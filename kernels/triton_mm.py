import torch
import triton
import triton.language as tl
from torch import Tensor

from ._lib import lib, lib_ops

# https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
# (BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps)
configs = [
    (128, 256, 64, 3, 8),
    (64, 256, 32, 4, 4),
    (128, 128, 32, 4, 4),
    (128, 64, 32, 4, 4),
    (64, 128, 32, 4, 4),
    (128, 32, 32, 4, 4),
    (64, 32, 32, 5, 2),
    (32, 64, 32, 5, 2),
    # Good config for fp8 inputs
    (128, 256, 128, 3, 8),
    (256, 128, 128, 3, 8),
    (256, 64, 128, 4, 4),
    (64, 256, 128, 4, 4),
    (128, 128, 128, 4, 4),
    (128, 64, 64, 4, 4),
    (64, 128, 64, 4, 4),
    (128, 32, 64, 4, 4),
    # https://github.com/pytorch/pytorch/blob/7868b65c4d4f34133607b0166f08e9fbf3b257c4/torch/_inductor/kernel/mm_common.py#L172
    (64, 64, 32, 2, 4),
    (64, 128, 32, 3, 4),
    (128, 64, 32, 3, 4),
    (64, 128, 32, 4, 8),
    (128, 64, 32, 4, 8),
    (64, 32, 32, 5, 8),
    (32, 64, 32, 5, 8),
    (128, 128, 32, 2, 8),
    (64, 64, 64, 3, 8),
    (128, 256, 128, 3, 8),
    (256, 128, 128, 3, 8),
]

configs = [
    triton.Config(dict(BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K), num_stages=num_stages, num_warps=num_warps)
    for BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps in configs
]


def _grid(meta):
    return (triton.cdiv(meta["M"], meta["BLOCK_M"]) * triton.cdiv(meta["N"], meta["BLOCK_N"]),)


# templated matmul from pytorch
# https://github.com/pytorch/pytorch/blob/c2e2602ecdc2ec1f120e19198dfc18fc39f7bd09/torch/_inductor/kernel/mm.py
# also re-tune when stride changes i.e. transpose configuration
@triton.autotune(configs=configs, key=["M", "N", "K", "stride_ak", "stride_bk"])
@triton.jit
def _matmul_kernel(
    # fmt: off
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    ACC_DTYPE: tl.constexpr,
    EVEN_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr = 8,
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.0)
            b = tl.load(B, mask=rk[:, None] < k, other=0.0)
        acc += tl.dot(a, b, out_dtype=ACC_DTYPE)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_m * stride_cm + idx_n * stride_cn
    tl.store(C_ptr + tl.broadcast_to(xindex, mask.shape), acc, mask)


lib.define("int8_mm(Tensor A, Tensor B) -> Tensor")


def int8_mm(A: Tensor, B: Tensor) -> Tensor:
    assert A.dtype is torch.int8 and B.dtype is torch.int8
    assert A.shape[1] == B.shape[0]
    return lib_ops.int8_mm(A, B)


@torch.library.impl(lib, "int8_mm", "Meta")
def _(a: Tensor, b: Tensor):
    return torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.int32)


@torch.library.impl(lib, "int8_mm", "CUDA")
def _(A: Tensor, B: Tensor):
    return _triton_mm(A, B, torch.int32, torch.int32)


def _triton_mm(A: Tensor, B: Tensor, out_dtype: torch.dtype, acc_dtype: torch.dtype):
    ACC_DTYPE_TRITON = {torch.float32: tl.float32, torch.float16: tl.float16, torch.int32: tl.int32}[acc_dtype]
    assert A.shape[1] == B.shape[0]
    M, K = A.shape
    _, N = B.shape
    EVEN_K = K % 2 == 0
    C = torch.empty(M, N, dtype=out_dtype, device=A.device)
    _matmul_kernel[_grid](A, B, C, M, N, K, *A.stride(), *B.stride(), *C.stride(), ACC_DTYPE_TRITON, EVEN_K)
    return C


@triton.autotune(configs=configs, key=["M", "N", "K", "stride_ak", "stride_bk"])
@triton.jit
def _scaled_mm_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    scale1_ptr,
    scale2_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    EVEN_K: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    SCALE1_TYPE: tl.constexpr,
    SCALE2_TYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr = 8,
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

    ACC_DTYPE = tl.int32 if COMPUTE_DTYPE == tl.int8 else tl.float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.0)
            b = tl.load(B, mask=rk[:, None] < k, other=0.0)

        # mixed mm
        if a.dtype != COMPUTE_DTYPE:
            a = a.to(COMPUTE_DTYPE)
        if b.dtype != COMPUTE_DTYPE:
            b = b.to(COMPUTE_DTYPE)

        acc += tl.dot(a, b)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    acc = acc.to(tl.float32)

    if SCALE1_TYPE == "row":
        acc *= tl.load(scale1_ptr + idx_m, mask=idx_m < M).to(tl.float32)
    elif SCALE1_TYPE == "column":
        acc *= tl.load(scale1_ptr + idx_n, mask=idx_n < N).to(tl.float32)
    elif SCALE1_TYPE == "tensor":
        acc *= tl.load(scale1_ptr).to(tl.float32)
    else:
        tl.static_assert(False, f"SCALE1_TYPE must be row, column, or tensor. Received {SCALE1_TYPE}")

    # optional
    if SCALE2_TYPE == "row":
        acc *= tl.load(scale2_ptr + idx_m, mask=idx_m < M).to(tl.float32)
    elif SCALE2_TYPE == "column":
        acc *= tl.load(scale2_ptr + idx_n, mask=idx_n < N).to(tl.float32)
    elif SCALE2_TYPE == "tensor":
        acc *= tl.load(scale2_ptr).to(tl.float32)
    else:
        tl.static_assert(
            SCALE2_TYPE == "none",
            f"SCALE2_TYPE must be row, column, tensor, or none. Received {SCALE2_TYPE}",
        )

    # inductor generates a suffix
    xindex = idx_m * stride_cm + idx_n * stride_cn
    tl.store(C_ptr + tl.broadcast_to(xindex, mask.shape), acc, mask)


lib.define("scaled_mm(Tensor A, Tensor B, Tensor scale1, Tensor? scale2) -> Tensor")


def scaled_mm(A: Tensor, B: Tensor, scale1: Tensor, scale2: Tensor | None) -> Tensor:
    """Perform `(A @ B) * scale1 * scale2`. `scale2` is optional. Broadcasting rules apply.

    When both A and B are INT8, INT8 tensor cores will be used. Otherwise, BF16 tensor cores
    are used.
    """
    assert A.dtype in (torch.int8, torch.bfloat16)
    assert B.dtype in (torch.int8, torch.bfloat16)
    assert A.shape[1] == B.shape[0]
    M, N = A.shape[0], B.shape[1]
    assert scale1.shape in ((M, 1), (1, N), ())
    assert scale1.is_contiguous()
    if scale2 is not None:
        assert scale2.shape in ((M, 1), (1, N), ())
        assert scale2.is_contiguous()

    if torch.compiler.is_compiling():
        return lib_ops.scaled_mm(A, B, scale1, scale2)

    else:
        # eager mode don't use triton kernel to avoid excessive autotune time during lm_eval
        # try to match numerics of the triton kernel
        if A.dtype == B.dtype == torch.int8:
            out = torch._int_mm(A, B)
        else:
            out = A.bfloat16() @ B.bfloat16()
        out = out.float() * scale1.float()
        if scale2 is not None:
            out = out * scale2.float()
        return out.to(scale1.dtype)


@torch.library.impl(lib, "scaled_mm", "Meta")
def _(A: Tensor, B: Tensor, scale1: Tensor, scale2: Tensor | None):
    return torch.empty((A.shape[0], B.shape[1]), device=A.device, dtype=scale1.dtype)


@torch.library.impl(lib, "scaled_mm", "CUDA")
def _(A: Tensor, B: Tensor, scale1: Tensor, scale2: Tensor | None):
    M, K = A.shape
    _, N = B.shape
    C = torch.empty(M, N, device=A.device, dtype=scale1.dtype)

    # this is faster than using a dictionary lookup table
    def get_scale_type(scale: Tensor | None):
        if scale is None:
            return "none"
        if scale.shape == (M, 1):
            return "row"
        if scale.shape == (1, N):
            return "column"
        if scale.shape == ():
            return "tensor"
        raise ValueError

    _scaled_mm_kernel[_grid](
        A,
        B,
        C,
        scale1,
        scale2,
        M,
        N,
        K,
        *A.stride(),
        *B.stride(),
        *C.stride(),
        EVEN_K=K % 2 == 0,
        COMPUTE_DTYPE=tl.int8 if A.dtype == B.dtype == torch.int8 else tl.bfloat16,
        SCALE1_TYPE=get_scale_type(scale1),
        SCALE2_TYPE=get_scale_type(scale2),
    )
    return C
