import torch
from torch import Tensor
import triton
import triton.language as tl


DTYPE_AMAX_LUT = {
    torch.float4_e2m1fn_x2: 6.0,
    torch.float8_e4m3fn: 448.0,
    torch.float8_e5m2: 57_344.0,
}

DTYPE_POW2_AMAX_LUT = {
    torch.float4_e2m1fn_x2: 4.0,
    torch.float8_e4m3fn: 256.0,
    torch.float8_e5m2: 32_768.0,
}


# https://github.com/NVIDIA/cutlass/blob/v3.9.2/media/docs/cpp/blackwell_functionality.md#scale-factor-layouts
def pack_block_scales(scales: Tensor):
    M, N = scales.shape
    assert M % 128 == 0 and N % 4 == 0  # don't support padding for now
    out = scales.reshape(M // 128, 128, N // 4, 4).transpose(1, 2)  # [num_M_tiles, num_N_tiles, 128, 4]
    out = out.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return out.flatten()


# https://docs.nvidia.com/cuda/cublas/index.html#d-block-quantization
def absmax_to_mx_scales_nvidia(absmax: Tensor, dtype: torch.dtype):
    assert absmax.dtype == torch.float32
    dtype_amax = DTYPE_AMAX_LUT[dtype]
    scales = absmax / dtype_amax
    bits = scales.view(torch.int32)
    exponent = bits >> 23  # x > 0, so don't need to mask sign bit
    mantissa = bits & 0x7F_FFFF
    return torch.where(
        (((exponent > 0) & (exponent < 254) & (mantissa > 0)) | ((exponent == 0) & (mantissa > 0x40_0000))),
        exponent + 1,
        exponent,
    )


# https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
def absmax_to_mx_scales_ocp(absmax: Tensor, dtype: torch.dtype):
    assert absmax.dtype == torch.float32
    # return (absmax.view(torch.int32) >> 23).clip(2) - 2
    dtype_pow2_amax = DTYPE_POW2_AMAX_LUT[dtype]
    return ((absmax.view(torch.int32) & 0x7F80_0000).view(torch.float32) / dtype_pow2_amax).view(torch.int32) >> 23


@triton.jit
def fp32x8_to_fp4e2m1x8_triton(x: tl.tensor):
    tl.static_assert(x.dtype == tl.int64)  # view fp32x2 as int64
    return tl.inline_asm_elementwise(
        asm="""
        {
        .reg .b32 lo, hi;
        .reg .b8 byte0, byte1, byte2, byte3;

        mov.b64 {lo, hi}, $1;
        cvt.rn.satfinite.e2m1x2.f32 byte0, hi, lo;
        mov.b64 {lo, hi}, $2;
        cvt.rn.satfinite.e2m1x2.f32 byte1, hi, lo;
        mov.b64 {lo, hi}, $3;
        cvt.rn.satfinite.e2m1x2.f32 byte2, hi, lo;
        mov.b64 {lo, hi}, $4;
        cvt.rn.satfinite.e2m1x2.f32 byte3, hi, lo;
        mov.b32 $0, {byte0, byte1, byte2, byte3};
        }
        """,
        constraints="=r,l,l,l,l",
        args=(x,),
        dtype=tl.int8,
        is_pure=True,
        pack=4,
    )


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


def quantize_mxfp8(x: Tensor, dtype: torch.dtype):
    assert dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    x_f32_blocks = x.float().unflatten(-1, (-1, 32))  # [M, N/32, 32]
    blocks_absmax = x_f32_blocks.abs().amax(dim=-1)  # [M, N/32]

    block_scales_bits = absmax_to_mx_scales_ocp(blocks_absmax, dtype)
    scales = block_scales_bits.to(torch.int8).view(torch.float8_e8m0fnu)

    # TODO: division by bit manipulation?
    x_f32_blocks = x_f32_blocks / (block_scales_bits.unsqueeze(-1) << 23).view(torch.float32)
    xq_f8 = x_f32_blocks.to(dtype).view(x.shape)

    return xq_f8, scales


def quantize_mxfp4(x: Tensor):
    x_f32_blocks = x.float().unflatten(-1, (-1, 32))  # [M, N/32, 32]
    blocks_absmax = x_f32_blocks.abs().amax(dim=-1)  # [M, N/32]

    dtype = torch.float4_e2m1fn_x2
    # block_scales_bits = absmax_to_mx_scales_nvidia(blocks_absmax, dtype)
    block_scales_bits = absmax_to_mx_scales_ocp(blocks_absmax, dtype)
    scales = block_scales_bits.to(torch.int8).view(torch.float8_e8m0fnu)

    # TODO: division by bit manipulation?
    x_f32_blocks = x_f32_blocks / (block_scales_bits.unsqueeze(-1) << 23).view(torch.float32)
    xq_f4x2_blocks = fp32_to_fp4e2m1x2(x_f32_blocks)
    xq_f4x2 = xq_f4x2_blocks.view(x.shape[0], -1).view(dtype)

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


# https://docs.nvidia.com/cuda/cublas/index.html#d-block-quantization
def quantize_nvfp4(x: Tensor, tensor_amax: Tensor | None = None):
    x_f32_blocks = x.float().unflatten(-1, (-1, 16))  # [M, N/16, 16]
    blocks_absmax = x_f32_blocks.abs().amax(dim=-1)  # [M, N/16]

    data_dtype = torch.float4_e2m1fn_x2
    scale_dtype = torch.float8_e4m3fn

    # tensor_amax can be provided (e.g. for activations)
    # or calculated on-the-fly
    if tensor_amax is None:
        tensor_amax = x_f32_blocks.amax()

    scale_in_D = DTYPE_AMAX_LUT[data_dtype] * DTYPE_AMAX_LUT[scale_dtype] / tensor_amax
    scales = (blocks_absmax * (DTYPE_AMAX_LUT[scale_dtype] / tensor_amax)).to(scale_dtype)

    x_f32_blocks = x_f32_blocks * (scale_in_D / scales.float().unsqueeze(-1))
    xq_f4x2_blocks = fp32_to_fp4e2m1x2(x_f32_blocks)
    xq_f4x2 = xq_f4x2_blocks.view(x.shape[0], -1).view(data_dtype)

    return xq_f4x2, scales, scale_in_D
