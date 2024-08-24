import argparse
from pathlib import Path

import numpy as np
import sentencepiece as spm
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm


def _process_tinystories(tokenizer: spm.SentencePieceProcessor, split: str, save_dir: str, n_threads: int):
    # do everything in memory. we have enough RAM
    filepath = hf_hub_download("roneneldan/TinyStories", f"TinyStoriesV2-GPT4-{split}.txt", repo_type="dataset")
    stories = open(filepath).read().split("\n<|endoftext|>\n")

    tokens_list = []
    chunk_size = 10_000
    for i in tqdm(range(0, len(stories), chunk_size), desc=f"Tokenizing TinyStories {split}", dynamic_ncols=True):
        chunk = stories[i : min(i + chunk_size, len(stories))]
        tokens_list.extend(tokenizer.Encode(chunk, add_bos=True, add_eos=True, num_threads=n_threads))

    total_size = sum(len(x) for x in tokens_list)
    mmap_tokens = np.memmap(f"{save_dir}/data.bin", dtype=np.uint16, mode="w+", shape=total_size)
    i = 0
    for tokens in tokens_list:
        mmap_tokens[i : i + len(tokens)] = tokens
        i += len(tokens)
    mmap_tokens.flush()


def _process_c4_realnewslike(tokenizer: spm.SentencePieceProcessor, split: str, save_dir: str, n_threads: int):
    ds = load_dataset("allenai/c4", "realnewslike", split=split)

    toks_per_shard = 2e8  # 200M tokens -> 400 MiB w/ uint16
    shard_idx = 0

    def write_shard(tokens_list: list[int], shard_idx: int):
        save_path = f"{save_dir}/data_{shard_idx:04d}.bin"
        print(f"Write {save_path}")
        mmap_tokens = np.memmap(save_path, dtype=np.uint16, mode="w+", shape=len(tokens_list))
        mmap_tokens[:] = tokens_list
        mmap_tokens.flush()

    tokens_list = []
    chunk_size = 10_000
    for i in tqdm(range(0, len(ds), chunk_size), desc=f"Tokenizing C4 realnewslike {split}", dynamic_ncols=True):
        chunk = ds[i : min(i + chunk_size, len(ds))]["text"]
        chunk_tokens = tokenizer.Encode(chunk, add_bos=False, add_eos=False, num_threads=n_threads)

        for new_tokens in chunk_tokens:
            tokens_list.extend(new_tokens)
            if len(tokens_list) >= toks_per_shard:
                write_shard(tokens_list, shard_idx)
                tokens_list = []
                shard_idx += 1

    write_shard(tokens_list, shard_idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--n_threads", type=int, default=4)
    args = parser.parse_args()

    dataset = args.dataset
    split = args.split
    n_threads = args.n_threads

    save_dir = Path(f"{dataset}_{split}")
    marker = save_dir / "COMPLETE"
    if not marker.exists():
        tokenizer_path = hf_hub_download("meta-llama/Llama-2-7b", "tokenizer.model")
        tokenizer = spm.SentencePieceProcessor(tokenizer_path)
        assert tokenizer.vocab_size() < (1 << 16)  # make sure we can use uint16

        save_dir.mkdir(exist_ok=True)
        if dataset == "tinystories":
            _process_tinystories(tokenizer, split, save_dir, n_threads)
        elif dataset == "c4_realnewslike":
            _process_c4_realnewslike(tokenizer, split, save_dir, n_threads)
        else:
            raise ValueError(f"Unsupported {dataset=}")
        open(marker, "w").close()  # create an empty file
