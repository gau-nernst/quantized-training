import pandas as pd
import torch
from triton.testing import do_bench

from kernels import _int8_mm

data = []
sizes = [1024, 2048, 4096]
transpose = [(0, 0), (0, 1), (1, 0)]
for sz in sizes:
    for ta, tb in transpose:
        layout_str = ("A.T" if ta else "A") + " @ " + ("B.T" if tb else "B")
        print(f"M=N=K={sz}, {layout_str}")

        A_bf16 = torch.randn(sz, sz).bfloat16().cuda().transpose(0, ta)
        B_bf16 = torch.randn(sz, sz).bfloat16().cuda().transpose(0, tb)
        A_i8 = torch.randint(-128, 127, size=(sz, sz), dtype=torch.int8).cuda().transpose(0, ta)
        B_i8 = torch.randint(-128, 127, size=(sz, sz), dtype=torch.int8).cuda().transpose(0, tb)

        bf16_time = do_bench(lambda: A_bf16 @ B_bf16, fast_flush=False, return_mode="median")
        i8_time = do_bench(lambda: torch._int_mm(A_i8, B_i8), fast_flush=False, return_mode="median")
        i8_triton_time = do_bench(lambda: _int8_mm(A_i8, B_i8), fast_flush=False, return_mode="median")
        torch.testing.assert_close(torch._int_mm(A_i8, B_i8), _int8_mm(A_i8, B_i8))

        data.append([sz, layout_str, bf16_time / i8_time, bf16_time / i8_triton_time])

df = pd.DataFrame(data, columns=["Size", "Layout", "PyTorch INT8", "Triton INT8"])
print(df.T)
