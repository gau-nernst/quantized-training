import pandas as pd
import torch
from triton.testing import do_bench

from gn_kernels import _triton_conv2d, int8_conv2d


def bench_f(f, *args, **kwargs):
    return do_bench(lambda: f(*args, **kwargs), fast_flush=False, return_mode="median")


if __name__ == "__main__":
    torch.set_default_device("cuda")

    data = []
    sizes = [
        # H, W, C
        # ResNet-50 (but with 256x256 inputs)
        (64, 64, 256),
        (32, 32, 512),
        (16, 16, 1024),
        (8, 8, 2048),
        # Flux VAE
        (1024, 1024, 128),
        (512, 512, 256),
        (256, 256, 512),
        (128, 128, 512),
    ]
    for H, W, C in sizes:
        print(f"{H=}, {W=}, {C=}")

        # might want to benchmark bs>1
        X_bf16 = torch.randn(1, C, H, W).bfloat16().to(memory_format=torch.channels_last)
        W_bf16 = torch.randn(C, C, 3, 3).bfloat16().to(memory_format=torch.channels_last)

        X_i8 = torch.randint(-128, 127, (1, C, H, W), dtype=torch.int8).to(memory_format=torch.channels_last)
        W_i8 = torch.randint(-128, 127, (C, C, 3, 3), dtype=torch.int8).to(memory_format=torch.channels_last)

        bf16_time = bench_f(torch.conv2d, X_bf16, W_bf16, padding=1)
        bf16_triton_time = bench_f(_triton_conv2d, X_bf16, W_bf16, padding=(1, 1))
        i8_triton_time = bench_f(int8_conv2d, X_i8, W_i8, padding=1)

        data.append(
            [
                bf16_time / bf16_triton_time,
                bf16_time / i8_triton_time,
            ]
        )

    df = pd.DataFrame(
        data,
        index=sizes,
        columns=[
            "Triton BF16",
            "Triton INT8",
        ],
    )
    print(df.round(2).to_markdown())
