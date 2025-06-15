from .image import HFImageDataset, WebDataset, decode_image
from .text import HFTextDataset, TokenDataset


__all__ = [
    "HFImageDataset",
    "WebDataset",
    "decode_image",
    "HFTextDataset",
    "TokenDataset",
]


def get_dataset(type: str, eval: bool, **kwargs):
    ds_cls = dict(
        token=TokenDataset,
        hf_text=HFTextDataset,
        hf_image=HFImageDataset,
        wds=WebDataset,
    )[type]
    return ds_cls(eval=eval, **kwargs)
