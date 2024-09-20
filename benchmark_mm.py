import argparse

import pandas as pd
import torch
from triton.testing import do_bench

from kernels import _triton_mm, int4_mm, int8_mm


def bench_f(f, *args, **kwargs):
    return do_bench(lambda: f(*args, **kwargs), fast_flush=False, return_mode="median")


def pack_int4(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.int8
    return (x[:, ::2] << 4) | (x[:, 1::2] & 0xF)


def unpack_int4(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.int8
    # NOTE: do this way to handle sign-extension correctly
    return torch.stack([x >> 4, x << 4 >> 4], dim=1).view(x.shape[0], -1)


def to_layout(x: torch.Tensor, column_major: bool):
    return x.T.contiguous().T if column_major else x.contiguous()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--a_column_major", action="store_true")
    parser.add_argument("--b_column_major", action="store_true")
    args = parser.parse_args()

    torch.set_default_device("cuda")

    data = []
    sizes = [1024, 2048, 4096]
    for sz in sizes:
        print(f"M=N=K={sz}")

        A_bf16 = torch.randn(sz, sz).bfloat16()
        B_bf16 = torch.randn(sz, sz).bfloat16()
        A_f16 = torch.randn(sz, sz).half()
        B_f16 = torch.randn(sz, sz).half()
        A_i8 = torch.randint(-128, 127, size=(sz, sz), dtype=torch.int8)
        B_i8 = torch.randint(-128, 127, size=(sz, sz), dtype=torch.int8)
        A_f8 = torch.randn(sz, sz).to(torch.float8_e4m3fn)
        B_f8 = torch.randn(sz, sz).to(torch.float8_e4m3fn)

        A_bf16, A_f16, A_i8 = [to_layout(x, args.a_column_major) for x in [A_bf16, A_f16, A_i8]]
        B_bf16, B_f16, B_i8 = [to_layout(x, args.b_column_major) for x in [B_bf16, B_f16, B_i8]]

        bf16_time = bench_f(torch.mm, A_bf16, B_bf16)
        i8_pytorch_time = bench_f(torch._int_mm, A_i8, B_i8)
        i8_triton_time = bench_f(int8_mm, A_i8, B_i8)
        torch.testing.assert_close(torch._int_mm(A_i8, B_i8), int8_mm(A_i8, B_i8))

        if not args.a_column_major and args.b_column_major:
            A_i8_ref = torch.randint(-8, 7, size=(sz, sz), dtype=torch.int8)
            B_i8_ref = torch.randint(-8, 7, size=(sz, sz), dtype=torch.int8)
            A_i4 = pack_int4(A_i8_ref)
            B_i4 = pack_int4(B_i8_ref.T).contiguous().T
            i4_cutlass_time = bench_f(int4_mm, A_i4, B_i4)

            actual = int4_mm(A_i4, B_i4)
            expected = torch._int_mm(A_i8_ref, B_i8_ref.T.contiguous().T)
            torch.testing.assert_close(actual, expected)
        else:
            i4_cutlass_time = float("inf")

        if torch.cuda.get_device_capability() >= (8, 9):
            # TODO: add torch._scaled_mm()
            f8_triton_time = bench_f(_triton_mm, A_f8, B_f8, torch.bfloat16, torch.float32)
        f16_acc_f16_triton_time = bench_f(_triton_mm, A_f16, B_f16, torch.float16, torch.float16)

        data.append(
            [
                bf16_time / i8_pytorch_time,
                bf16_time / i8_triton_time,
                bf16_time / i4_cutlass_time,
                bf16_time / f8_triton_time,
                bf16_time / f16_acc_f16_triton_time,
            ]
        )

    df = pd.DataFrame(
        data,
        index=sizes,
        columns=[
            "CuBLAS INT8",
            "Triton INT8",
            "Cutlass INT4",
            "Triton FP8",
            "Triton FP16 w/ FP16 accumulate",
        ],
    )
    print(df.round(2).T.to_markdown())
