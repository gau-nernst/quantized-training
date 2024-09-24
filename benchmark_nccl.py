# torchrun --standalone --nproc_per_node=2 bench_nccl.py

import os

import torch
import torch.distributed as dist

from benchmark_mm import bench_f


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

    all_reduce_time = bench_f(all_reduce, x)

    if rank == 0:
        print(f"All-reduce: {x.numel() * x.itemsize / 1e9 / all_reduce_time} GiB/s")

    dist.barrier()
    dist.destroy_process_group()
