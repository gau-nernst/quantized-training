import triton
import triton.language as tl
import torch
from torch import Tensor


# https://github.com/pytorch/pytorch/blob/6dcd773c5711b0cd822b142227c8657e062a298d/torch/_inductor/kernel/conv.py#L61-L70
kernel_configs = [
    # "BLOCK_M", "BLOCK_N", "BLOCK_K", "num_stages", "num_warps"
    (64, 256, 16, 2, 4),
    (256, 64, 16, 2, 4),
    (1024, 16, 16, 1, 8),
    (128, 128, 32, 2, 8),
    (64, 64, 32, 2, 4),
    (64, 256, 32, 2, 8),
    (256, 64, 32, 2, 8),
]
configs = [
    triton.Config(
        dict(BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K),
        num_stages=num_stages,
        num_warps=num_warps,
    )
    for BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps in kernel_configs
]


# this is mostly codegen-ed from torch.compile
# NOTE: caller should make sure that inputs are channels-last for best performance
@triton.autotune(
    configs,
    key=["BATCH", "IN_C", "IN_H", "IN_W", "OUT_C", "OUT_H", "OUT_W"],
)
@triton.jit
def conv2d_kernel(
    X_ptr,
    W_ptr,
    out_ptr,
    # Tensor dimensions
    BATCH: int,
    IN_C: int,
    IN_H: int,
    IN_W: int,
    OUT_C: int,
    OUT_H: int,
    OUT_W: int,
    # Strides:
    stride_xn: int,
    stride_xc: int,
    stride_xh: int,
    stride_xw: int,
    stride_wc_out: int,
    stride_wc_in: int,
    stride_wh: int,
    stride_ww: int,
    stride_outn: int,
    stride_outc: int,
    stride_outh: int,
    stride_outw: int,
    #
    KERNEL_H: tl.constexpr,
    KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PADDING_H: tl.constexpr,
    PADDING_W: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    nhw = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = nhw % OUT_W
    nh = nhw // OUT_W
    idx_y_h = nh % OUT_H
    idx_n = nh // OUT_H
    idx_y_c = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    group = 0
    GROUP_IN_C = IN_C
    GROUP_OUT_C = OUT_C

    x_base = X_ptr + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:, None]
    w_base = W_ptr + (group * stride_wc_out * GROUP_OUT_C + idx_y_c * stride_wc_out)[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Could be simplified, but slightly slower:
    # for i in range(KERNEL_H):
    #     for j in range(KERNEL_W):
    #         for k in range(0, GROUP_IN_C, BLOCK_K):
    BLOCK_K_COUNT = (GROUP_IN_C + BLOCK_K - 1) // BLOCK_K
    for ijk in range(KERNEL_H * KERNEL_W * BLOCK_K_COUNT):
        k = (ijk % BLOCK_K_COUNT) * BLOCK_K
        ij = ijk // BLOCK_K_COUNT
        i = ij // KERNEL_W
        j = ij % KERNEL_W

        idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
        idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W
        idx_x_c = tl.arange(0, BLOCK_K) + k

        x_ptrs = x_base + (
            (idx_x_h * stride_xh)[:, None] + (idx_x_w * stride_xw)[:, None] + (idx_x_c * stride_xc)[None, :]
        )
        mask_x = (
            (idx_n < BATCH)[:, None]
            & (idx_x_h >= 0)[:, None]
            & (idx_x_h < IN_H)[:, None]
            & (idx_x_w >= 0)[:, None]
            & (idx_x_w < IN_W)[:, None]
            & (idx_x_c < GROUP_IN_C)[None, :]
        )
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = w_base + ((idx_x_c * stride_wc_in)[:, None] + (i * stride_wh) + (j * stride_ww))
        mask_w = (idx_x_c[:, None] < GROUP_IN_C) & (idx_y_c[None, :] < GROUP_OUT_C)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(matrix_x, matrix_w)

    mask = (
        (idx_n < BATCH)[:, None]
        & (idx_y_h < OUT_H)[:, None]
        & (idx_y_w < OUT_W)[:, None]
        & (idx_y_c < GROUP_OUT_C)[None, :]
    )
    idx_n = idx_n[:, None]
    idx_c = idx_y_c[None, :] + group * GROUP_OUT_C
    idx_h = idx_y_h[:, None]
    idx_w = idx_y_w[:, None]

    out_ptr_offsets = tl.broadcast_to(
        idx_n * stride_outn + idx_c * stride_outc + idx_h * stride_outh + idx_w * stride_outw, acc.shape
    )
    tl.store(out_ptr + out_ptr_offsets, acc, mask)


def conv2d_grid(n, c, h, w, meta):
    return (
        triton.cdiv(n * h * w, meta["BLOCK_M"]),
        triton.cdiv(c, meta["BLOCK_N"]),
        meta["GROUPS"],
    )


def conv2d(X: Tensor, W: Tensor, stride: tuple[int, int] = (1, 1), padding: tuple[int, int] = (0, 0)) -> None:
    BATCH, IN_C, IN_H, IN_W = X.shape
    OUT_C, _, KERNEL_H, KERNEL_W = W.shape

    # refer to https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html for equation
    OUT_H = (IN_H + 2 * padding[0] - KERNEL_H) // stride[0] + 1
    OUT_W = (IN_W + 2 * padding[1] - KERNEL_W) // stride[1] + 1

    out = torch.empty(
        BATCH,
        OUT_C,
        OUT_H,
        OUT_W,
        device=X.device,
        dtype=X.dtype,
        memory_format=torch.channels_last,
    )

    def grid(meta):
        return (
            triton.cdiv(BATCH * OUT_H * OUT_W, meta["BLOCK_M"]),
            triton.cdiv(OUT_C, meta["BLOCK_N"]),
        )

    conv2d_kernel[grid](
        X,
        W,
        out,
        BATCH,
        IN_C,
        IN_H,
        IN_W,
        OUT_C,
        OUT_H,
        OUT_W,
        *X.stride(),
        *W.stride(),
        *out.stride(),
        KERNEL_H,
        KERNEL_W,
        *stride,
        *padding,
    )

    return out

# TODO: int8 conv2d, scaled int8 conv2d
