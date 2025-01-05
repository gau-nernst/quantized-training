import pandas as pd
import torch
from torch import Tensor

from benchmark_mm import bench_f, pack_int4, unpack_int4
from kernels import scaled_int4_mm, scaled_mm


@torch.compile(mode="max-autotune", dynamic=False)
def scaled_mm_inductor(A: Tensor, B: Tensor, scale_A: Tensor, scale_B: Tensor):
    if A.dtype == torch.int8:
        return torch._int_mm(A, B) * scale_B * scale_A
    else:
        return torch._scaled_mm(A, B, scale_A, scale_B, out_dtype=torch.bfloat16, use_fast_accum=True)


def scaled_mm_ref(A: Tensor, B: Tensor, scale_A: Tensor, scale_B: Tensor):
    return (A.float() * scale_A) @ (B.float() * scale_B)


def assert_close(actual: Tensor, expected: Tensor):
    error = (actual.float() - expected.float()).abs() / expected.float().abs().clip(1e-12)
    mean_error = error.mean().item()
    assert mean_error < 2e-2, mean_error


if __name__ == "__main__":
    torch.set_default_device("cuda")

    # we need to do this to force inductor to use triton's implementation of torch._scaled_mm() on devices with SMs<68
    torch._inductor.config.max_autotune_gemm_backends = "TRITON"
    torch._inductor.utils.is_big_gpu = lambda _: True
    torch._inductor.config.force_fuse_int_mm_with_mul = True

    data = []
    sizes = [1024, 2048, 4096]
    for sz in sizes:
        print(f"M=N=K={sz}")

        A_bf16 = torch.randn(sz, sz).bfloat16()
        B_bf16 = torch.randn(sz, sz).bfloat16()
        A_i8 = torch.randint(-128, 127, size=(sz, sz), dtype=torch.int8)
        B_i8 = torch.randint(-128, 127, size=(sz, sz), dtype=torch.int8)
        A_f8 = torch.randn(sz, sz).to(torch.float8_e4m3fn)
        B_f8 = torch.randn(sz, sz).to(torch.float8_e4m3fn)
        A_i4 = pack_int4(torch.randint(-8, 7, size=(sz, sz), dtype=torch.int8))
        B_i4 = pack_int4(torch.randint(-8, 7, size=(sz, sz), dtype=torch.int8))
        scale_A = torch.randn(sz, 1).bfloat16()
        scale_B = torch.randn(1, sz).bfloat16()

        bf16_time = bench_f(torch.mm, A_bf16, B_bf16.T)
        inductor_scaled_i8_time = bench_f(scaled_mm_inductor, A_i8, B_i8.T, scale_A, scale_B)
        triton_scaled_i8_time = bench_f(scaled_mm, A_i8, B_i8.T, scale_A, scale_B)
        cublas_tensor_scaled_f8_time = bench_f(
            torch._scaled_mm, A_f8, B_f8.T, scale_A[0, 0].float(), scale_B[0, 0].float(), out_dtype=scale_A.dtype
        )
        inductor_scaled_f8_time = bench_f(scaled_mm_inductor, A_f8, B_f8.T, scale_A.float(), scale_B.float())
        triton_scaled_f8_time = bench_f(scaled_mm, A_f8, B_f8.T, scale_A, scale_B)
        cutlass_scaled_i4_time = bench_f(scaled_int4_mm, A_i4, B_i4.T, scale_A, scale_B)

        assert_close(scaled_mm(A_i8, B_i8.T, scale_A, scale_B), scaled_mm_ref(A_i8, B_i8.T, scale_A, scale_B))
        assert_close(scaled_mm(A_f8, B_f8.T, scale_A, scale_B), scaled_mm_ref(A_f8, B_f8.T, scale_A, scale_B))
        assert_close(
            scaled_int4_mm(A_i4, B_i4.T, scale_A, scale_B),
            scaled_mm_ref(unpack_int4(A_i4), unpack_int4(B_i4).T, scale_A, scale_B),
        )

        scale_A = torch.randn(sz, sz // 128).bfloat16()
        scale_B = torch.randn(sz // 128, sz // 128).bfloat16()
        triton_tile_scaled_i8_time = bench_f(scaled_mm, A_i8, B_i8.T, scale_A, scale_B.T)
        triton_tile_scaled_f8_time = bench_f(scaled_mm, A_f8, B_f8.T, scale_A, scale_B.T)

        expanded_scale_A = scale_A.repeat_interleave(128, 1)
        expanded_scale_B = scale_B.repeat_interleave(128, 0).repeat_interleave(128, 1)
        assert_close(
            scaled_mm(A_i8, B_i8.T, scale_A, scale_B.T),
            scaled_mm_ref(A_i8, B_i8.T, expanded_scale_A, expanded_scale_B.T),
        )
        assert_close(
            scaled_mm(A_f8, B_f8.T, scale_A, scale_B.T),
            scaled_mm_ref(A_f8, B_f8.T, expanded_scale_A, expanded_scale_B.T),
        )

        data.append(
            [
                bf16_time / inductor_scaled_i8_time,
                bf16_time / triton_scaled_i8_time,
                bf16_time / triton_tile_scaled_i8_time,
                bf16_time / cublas_tensor_scaled_f8_time,
                bf16_time / inductor_scaled_f8_time,
                bf16_time / triton_scaled_f8_time,
                bf16_time / triton_tile_scaled_f8_time,
                bf16_time / cutlass_scaled_i4_time,
            ]
        )

    df = pd.DataFrame(
        data,
        index=sizes,
        columns=[
            "Inductor scaled INT8",
            "Triton scaled INT8",
            "Triton tile-scaled INT8",
            "CuBLAS tensor-scaled FP8",
            "Inductor scaled FP8",
            "Triton scaled FP8",
            "Triton tile-scaled FP8",
            "Cutlass scaled INT4",
        ],
    )
    print(df.round(2).T.to_markdown())
