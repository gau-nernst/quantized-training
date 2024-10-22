import math
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info

from llama_tokenizers import get_tokenizer

from .utils import _get_dist_info


# datasets produced by tokenize_data.py
class TokenDataset(IterableDataset):
    def __init__(self, dataset_dir: str, seq_len: int, eval: bool, seed: int = 2024) -> None:
        super().__init__()
        self.shards = sorted(Path(dataset_dir).glob("*.bin"))
        self.seq_len = seq_len
        self.eval = eval
        print(f"Found {len(self.shards)} shards of data")

        # TODO: load and save state_dict
        self._generator = torch.Generator().manual_seed(seed)

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
            if self.eval:
                shard_indices = range(len(self.shards))
            else:
                shard_indices = torch.randperm(len(self.shards), generator=self._generator)

            for shard_idx in shard_indices:
                # divide a shard into n slices of (seq_len + 1)
                shard_np = np.memmap(self.shards[shard_idx], dtype=np.uint16, mode="r")
                shard = torch.from_numpy(shard_np)

                for data in self._iter_shard(shard):
                    yield data

            if self.eval:
                break


# adapted from https://github.com/pytorch/torchtitan/blob/main/torchtitan/datasets/hf_datasets.py
# must have "text" column e.g.
# - allenai/c4
# - HuggingFaceFW/fineweb-edu
class HFTextDataset(IterableDataset):
    def __init__(
        self,
        dataset: str,
        subset: str,
        split: str,
        tokenizer: str,
        seq_len: int,
        eval: bool,
        seed: int = 2024,
    ) -> None:
        self.ds = load_dataset(dataset, name=subset, split=split, streaming=True)
        self.tokenizer = get_tokenizer(tokenizer)
        self.seq_len = seq_len
        self.eval = eval

        self.ds = self.ds.select_columns("text")
        if not eval:  # only shuffle shards
            self.ds = self.ds.shuffle(seed=seed, buffer_size=1)
        rank, world_size = _get_dist_info()
        if world_size > 1:
            self.ds = split_dataset_by_node(self.ds, rank, world_size)
        self._epoch = 0
        self._buffer: list[int] = []

    def __iter__(self):
        # does HF datasets split samples among data workers automatically?
        worker_info = get_worker_info()
        if worker_info is not None:
            assert worker_info.num_workers == 1

        SAMPLE_LEN = self.seq_len + 1
        while True:
            self.ds.set_epoch(self._epoch)
            for sample in self.ds:
                new_tokens = self.tokenizer(sample["text"], add_bos=True, add_eos=True)
                self._buffer.extend(new_tokens)

                while len(self._buffer) >= SAMPLE_LEN:
                    sample = torch.tensor(self._buffer[:SAMPLE_LEN], dtype=torch.int32)
                    self._buffer = self._buffer[SAMPLE_LEN:]
                    yield sample[:-1], sample[1:]

            self._epoch += 1
            if self.eval:
                break

    def state_dict(self):
        ds_state_dict = self.ds.state_dict()
        if not self.eval and ds_state_dict["shard_example_idx"] > 0:
            ds_state_dict["shard_example_idx"] -= 1  # compensate for prefetch
        return dict(
            ds=ds_state_dict,
            _epoch=self._epoch,
            _buffer=list(self._buffer),  # make a copy
        )

    def load_state_dict(self, state_dict: dict):
        self.ds.load_state_dict(state_dict["ds"])
        self._epoch = state_dict["_epoch"]
        self._buffer = list(state_dict["_buffer"])  # make a copy
