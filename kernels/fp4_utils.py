import torch
from torch import Tensor


# https://github.com/NVIDIA/cutlass/blob/v3.9.2/media/docs/cpp/blackwell_functionality.md#scale-factor-layouts
def pack_block_scales(scales: Tensor):
    M, N = scales.shape
    out = scales.reshape(M // 128, 128, N // 4, 4).transpose(1, 2)  # [num_M_tiles, num_N_tiles, 128, 4]
    out = out.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return out.flatten()


# https://docs.nvidia.com/cuda/cublas/index.html#d-block-quantization
def absmax_to_mx_scales_nvidia(absmax: Tensor):
    assert absmax.dtype == torch.float32
    scales = absmax / 6.0
    bits = scales.view(torch.int32)
    exponent = bits >> 23  # x > 0, so don't need to mask sign bit
    mantissa = bits & 0x7F_FFFF
    return torch.where(
        (((exponent > 0) & (exponent < 254) & (mantissa > 0)) | ((exponent == 0) & (mantissa > 0x40_0000))),
        exponent + 1,
        exponent,
    )


# https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
def absmax_to_mx_scales_ocp(absmax: Tensor):
    assert absmax.dtype == torch.float32
    # return (absmax.view(torch.int32) >> 23).clip(2) - 2
    return ((absmax.view(torch.int32) & 0x7F80_0000).view(torch.float32) / 4.0).view(torch.int32) >> 23


def fp32_to_fp4e2m1x2(x: Tensor):
    assert x.dtype == torch.float32

    bits_f32 = x.view(torch.int32)
    sign = (bits_f32 >> 31) & 0x1

    x_abs = x.abs()
    nosign = torch.where(x_abs <= 5.0, 0b0110, 0b0111).int()
    nosign = torch.where(x_abs < 3.5, 0b0101, nosign)
    nosign = torch.where(x_abs <= 2.5, 0b0100, nosign)
    nosign = torch.where(x_abs < 1.75, 0b0011, nosign)
    nosign = torch.where(x_abs <= 1.25, 0b0010, nosign)
    nosign = torch.where(x_abs < 0.75, 0b0001, nosign)
    nosign = torch.where(x_abs <= 0.25, 0b0000, nosign)

    f4_e2m1 = (sign << 3) | nosign

    # pack to 32-bit register
    f4_e2m1x2 = (
        f4_e2m1[..., ::8]
        | (f4_e2m1[..., 1::8] << 4)
        | (f4_e2m1[..., 2::8] << 8)
        | (f4_e2m1[..., 3::8] << 12)
        | (f4_e2m1[..., 4::8] << 16)
        | (f4_e2m1[..., 5::8] << 20)
        | (f4_e2m1[..., 6::8] << 24)
        | (f4_e2m1[..., 7::8] << 28)
    )
    return f4_e2m1x2.view(torch.float4_e2m1fn_x2)


def quantize_mxfp4(x: Tensor):
    x_f32_blocks = x.float().unflatten(-1, (-1, 32))  # [M, N/32, 32]
    blocks_absmax = x_f32_blocks.abs().amax(dim=-1)  # [M, N/32]

    # block_scales_bits = absmax_to_mx_scales_nvidia(blocks_absmax)
    block_scales_bits = absmax_to_mx_scales_ocp(blocks_absmax)
    scales = block_scales_bits.to(torch.int8).view(torch.float8_e8m0fnu)

    # TODO: division by bit manipulation?
    x_f32_blocks = x_f32_blocks / (block_scales_bits.unsqueeze(-1) << 23).view(torch.float32)
    xq_f4x2_blocks = fp32_to_fp4e2m1x2(x_f32_blocks)
    xq_f4x2 = xq_f4x2_blocks.view(x.shape[0], -1).view(torch.float4_e2m1fn_x2)

    return xq_f4x2, scales


def dequantize_mxfp4(xq: Tensor, scales: Tensor):
    assert xq.dtype == torch.float4_e2m1fn_x2
    assert scales.dtype == torch.float8_e8m0fnu

    FP4E2M1_LUT = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=xq.device,
        dtype=torch.float32,
    )

    M = xq.shape[0]
    N = xq.shape[1] * 2

    # unpack
    xq_i32 = xq.view(torch.int32)
    xq_unpacked = torch.stack(
        [
            xq_i32 & 0xF,
            (xq_i32 >> 4) & 0xF,
            (xq_i32 >> 8) & 0xF,
            (xq_i32 >> 12) & 0xF,
            (xq_i32 >> 16) & 0xF,
            (xq_i32 >> 20) & 0xF,
            (xq_i32 >> 24) & 0xF,
            (xq_i32 >> 28) & 0xF,
        ],
        dim=-1,
    )
    x = FP4E2M1_LUT[xq_unpacked]

    scales_f32 = (scales.view(torch.uint8).to(torch.int32) << 23).view(torch.float32)
    x_scaled = x.view(M, N // 32, 32) * scales_f32.reshape(M, N // 32, 1)
    return x_scaled.view(M, N)


if __name__ == "__main__":
    x = torch.randn(512, 256)
    xq, scales = quantize_mxfp4(x)
    x_qdq = dequantize_mxfp4(xq, scales)
    assert x_qdq.shape == x.shape
