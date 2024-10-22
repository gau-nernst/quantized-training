import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info


def _get_dist_info(*, include_worker_info: bool = False):
    if dist.is_initialized():
        rank, world_size = dist.get_rank(), dist.get_world_size()
    else:
        rank, world_size = 0, 1

    worker_info = get_worker_info()
    if include_worker_info and worker_info is not None:
        rank = rank * worker_info.num_workers + worker_info.id
        world_size *= worker_info.num_workers
    return rank, world_size


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
        # instead of saving (and loading) the buffer, which can be huge, we can save the current index in the buffer
        # instead. we will then save the state of inner dataset when we swap the buffer (i.e. save every buffer_size
        # samples). then we can efficiently rewind the dataset.
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
