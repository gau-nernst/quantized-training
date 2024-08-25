import pandas as pd
import torch
from triton.testing import do_bench

from kernels import _int8_mm, _triton_mm


def bench_f(f, *args):
    return do_bench(lambda: f(*args), fast_flush=False, return_mode="median")


data = []
sizes = [1024, 2048, 4096]
transpose = [(0, 0), (0, 1), (1, 0)]
for sz in sizes:
    for ta, tb in transpose:
        layout_str = ("A.T" if ta else "A") + " @ " + ("B.T" if tb else "B")
        print(f"M=N=K={sz}, {layout_str}")

        A_bf16 = torch.randn(sz, sz).bfloat16().cuda().transpose(0, ta)
        B_bf16 = torch.randn(sz, sz).bfloat16().cuda().transpose(0, tb)
        A_f16 = torch.randn(sz, sz).half().cuda().transpose(0, ta)
        B_f16 = torch.randn(sz, sz).half().cuda().transpose(0, tb)
        A_i8 = torch.randint(-128, 127, size=(sz, sz), dtype=torch.int8).cuda().transpose(0, ta)
        B_i8 = torch.randint(-128, 127, size=(sz, sz), dtype=torch.int8).cuda().transpose(0, tb)
        A_f8 = torch.randn(sz, sz).to(torch.float8_e4m3fn).cuda().transpose(0, ta)
        B_f8 = torch.randn(sz, sz).to(torch.float8_e4m3fn).cuda().transpose(0, tb)

        bf16_time = bench_f(torch.mm, A_bf16, B_bf16)
        i8_time = bench_f(torch._int_mm, A_i8, B_i8)
        i8_triton_time = bench_f(_int8_mm, A_i8, B_i8)
        torch.testing.assert_close(torch._int_mm(A_i8, B_i8), _int8_mm(A_i8, B_i8))

        # TODO: add torch._scaled_mm()
        f8_triton_time = bench_f(_triton_mm, A_f8, B_f8, torch.bfloat16, torch.float32)
        f16_acc_f16_triton_time = bench_f(_triton_mm, A_f16, B_f16, torch.float16, torch.float16)

        data.append(
            [
                sz,
                layout_str,
                bf16_time / i8_time,
                bf16_time / i8_triton_time,
                bf16_time / f8_triton_time,
                bf16_time / f16_acc_f16_triton_time,
            ]
        )

df = pd.DataFrame(
    data, columns=["Size", "Layout", "PyTorch INT8", "Triton INT8", "Triton FP8", "Triton FP16 w/ FP16 accumulate"]
)
print(df.T)
