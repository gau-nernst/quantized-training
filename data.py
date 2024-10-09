import math
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info

from llama_tokenizers import get_tokenizer


def get_dataset(type: str, eval: bool, **kwargs):
    ds_cls = dict(
        token=TokenDataset,
        hf_text=HFTextDataset,
        hf_image=HFImageDataset,
    )[type]
    return ds_cls(eval=eval, **kwargs)


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
        if dist.is_initialized():
            self.ds = split_dataset_by_node(
                dataset=self.ds,
                rank=dist.get_rank(),
                world_size=dist.get_world_size(),
            )
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


# must have "jpg" and "cls" keys, typically webdataset format e.g.
# - timm/imagenet-1k-wds
class HFImageDataset(IterableDataset):
    def __init__(self, dataset: str, split: str, eval: bool, transform=None) -> None:
        self.ds = load_dataset(dataset, split=split, streaming=True)
        self.eval = eval
        self.transform = transform

    def __iter__(self):
        while True:
            if self.eval:
                ds = self.ds
            else:
                seed = torch.empty(1, dtype=torch.int64).random_().item()
                ds = self.ds.shuffle(seed)  # large buffer size will slow down

            # TODO: support other keys
            # TODO: add batching here to support things like CutMix/MixUp?
            for sample in ds.select_columns(["jpg", "cls"]):
                # some images are RGBA, which will throw off torchvision
                img = sample["jpg"].convert("RGB")
                if self.transform is not None:
                    img = self.transform(img)

                yield img, sample["cls"]

            if self.eval:
                break


class ShuffleDataset(IterableDataset):
    def __init__(self, ds: IterableDataset, buffer_size: int = 1000, seed: int = 2024) -> None:
        self.ds = ds
        self.buffer_size = buffer_size

        self._generator = torch.Generator().manual_seed(seed)
        self._buffer1 = []
        self._buffer2 = []

    def __iter__(self):
        for sample in self.ds:
            # buffer2 is filled. once buffer2 is full, we shuffle and swap it with buffer1.
            # buffer2 should now be empty and buffer1 is full. we yield 1 sample from buffer1.
            # in subsequent iterations, we add 1 item to buffer2 and remove 1 item from buffer1,
            # thus maintaining the invariance that len(buffer1) + len(buffer2) = buffer_size - 1.
            self._buffer2.append(sample)
            if len(self._buffer2) == self.buffer_size:
                self._buffer2 = self._shuffle(self._buffer2)
                self._buffer1, self._buffer2 = self._buffer2, self._buffer1

            if len(self._buffer1):
                yield self._buffer1.pop()

        while len(self._buffer1):
            yield self._buffer1.pop()
        self._buffer2 = self._shuffle(self._buffer2)
        while len(self._buffer2):
            yield self._buffer2.pop()

    def _shuffle(self, buffer: list):
        indices = torch.randperm(len(buffer), generator=self._generator)
        return [buffer[idx] for idx in indices]

    def state_dict(self):
        return dict(
            ds=self.ds.state_dict(),
            _generator=self._generator.get_state(),
            _buffer1=list(self._buffer1),  # shallow copy
            _buffer2=list(self._buffer2),
        )

    def load_state_dict(self, state_dict: dict):
        self.ds.load_state_dict(state_dict["ds"])
        self._generator.set_state(state_dict["_generator"])
        self._buffer1 = list(state_dict["_buffer1"])  # shallow copy
        self._buffer2 = list(state_dict["_buffer2"])
