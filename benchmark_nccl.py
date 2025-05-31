# torchrun --standalone --nproc_per_node=2 bench_nccl.py

import os
import time

import torch
import torch.distributed as dist


def all_reduce(x):
    dist.all_reduce(x, op=dist.ReduceOp.AVG)
    dist.barrier()


if __name__ == "__main__":
    dist.init_process_group(backend="nccl")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.set_default_device("cuda")
    torch.cuda.set_device(local_rank)

    x = torch.randn(1024, 1024)

    N = 100
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        all_reduce(x)
    torch.cuda.synchronize()
    latency = (time.perf_counter() - t0) / N

    if rank == 0:
        print(f"All-reduce: {x.numel() * x.itemsize / 1e9 / latency} GiB/s")

    dist.barrier()
    dist.destroy_process_group()
