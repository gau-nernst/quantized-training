import math
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch import Tensor
from torch.utils.data import IterableDataset

from llama_tokenizers import get_tokenizer


def get_dataset(type: str, seq_len: int, eval: bool, **kwargs):
    ds_cls = dict(token=TokenDataset, hf=HFDataset)[type]
    return ds_cls(seq_len=seq_len, eval=eval, **kwargs)


# datasets produced by tokenize_data.py
class TokenDataset(IterableDataset):
    def __init__(self, dataset_dir: str, seq_len: int, eval: bool) -> None:
        super().__init__()
        self.shards = sorted(Path(dataset_dir).glob("*.bin"))
        self.seq_len = seq_len
        self.eval = eval
        print(f"Found {len(self.shards)} shards of data")

    def _iter_shard(self, shard: Tensor):
        n_slices = math.floor(shard.shape[0] / (self.seq_len + 1))
        slice_indices = range(n_slices) if self.eval else torch.randperm(n_slices)

        for slice_idx in slice_indices:
            batch = shard[slice_idx * (self.seq_len + 1) : (slice_idx + 1) * (self.seq_len + 1)]
            yield batch[:-1].long(), batch[1:].long()

    def __iter__(self):
        while True:
            # NOTE: we don't split data across workers. just depend on workers having different
            # random seeds to select different slice of data.
            shard_indices = range(len(self.shards)) if self.eval else torch.randperm(len(self.shards))
            for shard_idx in shard_indices:
                # divide a shard into n slices of (seq_len + 1)

                # fast validation loss: calculate loss on consecutive, non-overlapping slices
                # the more correct way is to calculate loss for each token with full seq_len context (rolling window)

                shard_np = np.memmap(self.shards[shard_idx], dtype=np.uint16, mode="r")
                shard = torch.from_numpy(shard_np)

                for data in self._iter_shard(shard):
                    yield data

            if self.eval:
                break


# https://github.com/pytorch/torchtitan/blob/main/torchtitan/datasets/hf_datasets.py
# must have "text" column e.g.
# - allenai/c4
# - HuggingFaceFW/fineweb-edu
class HFDataset(IterableDataset):
    def __init__(self, dataset: str, subset: str, split: str, tokenizer: str, seq_len: int, eval: bool) -> None:
        self.ds = load_dataset(dataset, name=subset, split=split, streaming=True)
        self.tokenizer = get_tokenizer(tokenizer)
        self.seq_len = seq_len
        self.eval = eval

    def __iter__(self):
        buffer = []

        while True:
            if self.eval:
                ds = self.ds
            else:
                seed = torch.empty(1, dtype=torch.int64).random_().item()
                ds = self.ds.shuffle(seed, buffer_size=10_000)

            for sample in ds:
                buffer.extend(self.tokenizer(sample["text"]))
                while len(buffer) >= self.seq_len + 1:
                    tokens = torch.tensor(buffer[: self.seq_len + 1], dtype=torch.int64)
                    buffer = buffer[self.seq_len + 1 :]
                    yield tokens[:-1], tokens[1:]

            if self.eval:
                break


# TODO: timm/imagenet-1k-wds
