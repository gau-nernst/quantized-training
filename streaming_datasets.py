# https://github.com/pytorch/torchtitan/blob/main/torchtitan/datasets/hf_datasets.py

from typing import Callable

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset


class C4Dataset(IterableDataset):
    def __init__(self, subset: str, split: str, tokenizer: Callable[[str], list[int]], seq_len: int = 2048) -> None:
        self.ds = load_dataset("allenai/c4", name=subset, split=split, streaming=True)
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self):
        buffer = []

        while True:
            i64_info = torch.iinfo(torch.int64)
            seed = torch.empty(1, dtype=torch.int64).random_(i64_info.min, i64_info.max).item()

            for sample in self.ds.shuffle(seed):
                buffer.extend(self.tokenizer(sample["text"]))

                while len(buffer) >= self.seq_len + 1:
                    tokens = torch.tensor(buffer[: self.seq_len + 1], dtype=torch.int64)
                    buffer = buffer[self.seq_len + 1 :]
                    yield tokens[:-1], tokens[1:]


# TODO: timm/imagenet-1k-wds
