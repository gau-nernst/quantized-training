import logging
import tarfile

import huggingface_hub
import requests
import torch
import torchvision
from datasets import load_dataset
from torch.utils.data import IterableDataset

from .utils import _get_dist_info

logger = logging.getLogger(__name__)


def decode_image(data: bytes):
    # 0.20 supports jpeg, png, and webp
    return torchvision.io.decode_image(
        torch.frombuffer(data, dtype=torch.uint8),
        mode=torchvision.io.ImageReadMode.RGB,
        apply_exif_orientation=True,
    )


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


# technically this is generic, not limited to only image
class WebDataset(IterableDataset):
    def __init__(
        self,
        urls: list[str],
        columns: list[str] | None = None,
        transform: dict | None = None,
        eval: bool = True,
        seed: int = 2024,
    ) -> None:
        self.urls = urls
        self.columns = tuple(columns) if columns is not None else None  # shallow copy
        self.transform = dict(transform) if transform is not None else None
        self.eval = eval

        self._generator = torch.Generator().manual_seed(seed)
        self._sess = requests.Session()

    @staticmethod
    def from_hf(repo_id: str, **kwargs) -> "WebDataset":
        fs = huggingface_hub.HfFileSystem()
        urls = []
        for path in fs.glob(f"hf://datasets/{repo_id}/**/*.tar"):
            hf_file = fs.resolve_path(path)
            url = huggingface_hub.hf_hub_url(repo_id, hf_file.path_in_repo, repo_type="dataset")
            urls.append(url)
        urls.sort()
        return WebDataset(urls, **kwargs)

    @staticmethod
    def _get_headers(url: str) -> dict[str, str]:
        headers = dict()
        if url.startswith("https://huggingface.co/datasets"):
            token = huggingface_hub.utils.get_token()
            if token is not None:
                headers["Authorization"] = f"Bearer {token}"
        return headers

    def _url_iter(self):
        while True:
            if not self.eval:
                indices = torch.randperm(len(self.urls), generator=self._generator)
            else:
                indices = range(len(self.urls))

            for idx in indices:
                yield self.urls[idx]
            if self.eval:
                break

    def __iter__(self):
        rank, world_size = _get_dist_info(include_worker_info=True)

        # if all distributed processes use the same RNG seed, the infinite url sequence should be exactly identical.
        # thus, to evenly distribute them among processes, each process simply takes 1 shard every world_size.
        for shard_idx, url in enumerate(self._url_iter()):
            if shard_idx % world_size != rank:
                continue

            try:
                # TODO: might use smaller timeout. add retry for timeout/broken connection.
                resp = self._sess.get(
                    url,
                    headers=self._get_headers(url),
                    timeout=30,
                    stream=True,
                )
                tar = tarfile.open(fileobj=resp.raw, mode="r|")

                sample = dict()
                for tarinfo in tar:
                    key, ext = tarinfo.name.rsplit(".", 1)
                    if "__key__" in sample:
                        if sample["__key__"] != key:
                            if self.transform is not None:
                                for k, v in sample.items():
                                    if k in self.transform:
                                        sample[k] = self.transform[k](v)
                            yield sample
                            sample = dict(__key__=key)
                    else:
                        sample["__key__"] = key
                    if self.columns is None or ext in self.columns:
                        sample[ext] = tar.extractfile(tarinfo).read()
                yield sample

            except Exception as e:
                # when failure, we simply continue with the next shard
                logger.exception(f"Exception while reading {url=}. {e}")
