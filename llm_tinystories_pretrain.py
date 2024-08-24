import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import wandb
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import LlamaConfig, LlamaForCausalLM

from train_utils import get_grad_norm, get_optim_cls, print_model_stats, quantize_model


def _process_tinystories(tokenizer: spm.SentencePieceProcessor, split: str, save_path: str):
    # do everything in memory. we have enough RAM
    filepath = hf_hub_download("roneneldan/TinyStories", f"TinyStoriesV2-GPT4-{split}.txt", repo_type="dataset")
    stories = open(filepath).read().split("\n<|endoftext|>\n")

    tokens_list = []
    chunk_size = 10_000
    for i in tqdm(range(0, len(stories), chunk_size), desc="Tokenizing TinyStories"):
        chunk = stories[i : min(i + chunk_size, len(stories))]
        tokens_list.extend(tokenizer.Encode(chunk, add_bos=True, add_eos=True, num_threads=4))

    total_size = sum(len(x) for x in tokens_list)
    mmap_tokens = np.memmap(save_path, dtype=np.uint16, mode="w+", shape=total_size)
    i = 0
    for tokens in tokens_list:
        mmap_tokens[i : i + len(tokens)] = tokens
        i += len(tokens)
    mmap_tokens.flush()


class TokenDataset(IterableDataset):
    def __init__(self, dataset: str, split: str, batch_size: int, seq_len: int) -> None:
        super().__init__()
        self.save_dir = Path(f"{dataset}_{split}")
        self.batch_size = batch_size
        self.seq_len = seq_len

        marker = self.save_dir / "COMPLETE"
        if not marker.exists():
            tokenizer_path = hf_hub_download("meta-llama/Llama-2-7b", "tokenizer.model")
            tokenizer = spm.SentencePieceProcessor(tokenizer_path)
            assert tokenizer.vocab_size() < (1 << 16)  # make sure we can use uint16

            self.save_dir.mkdir(exist_ok=True)
            if dataset == "tinystories":
                _process_tinystories(tokenizer, split, self.save_dir / "data.bin")
            else:
                raise ValueError(f"Unsupported {dataset=}")
            open(marker, "w").close()  # create an empty file

    def __iter__(self):
        shards = sorted(self.save_dir.glob("*.bin"))
        toks_per_batch = self.batch_size * (self.seq_len + 1)

        while True:
            # NOTE: we don't split data across workers. just depend on workers having different
            # random seeds to select different slice of data.
            shard_indices = torch.randperm(len(shards))
            for shard_idx in shard_indices:
                # divide a shard into n slices of toks_per_batch
                # to make sure the slices are slightly different everytime, we add a random offset
                # offset in np.memmap is in bytes, so need to times 2
                offset = torch.randint(0, toks_per_batch, size=(1,)).item()
                shard_np = np.memmap(shards[shard_idx], dtype=np.uint16, mode="r", offset=offset * 2)
                shard = torch.from_numpy(shard_np)

                n_slices = math.floor(shard.shape[0] / toks_per_batch)
                slice_indices = torch.randperm(n_slices)
                for slice_idx in slice_indices:
                    batch = shard[slice_idx * toks_per_batch : (slice_idx + 1) * toks_per_batch]
                    batch = batch.view(self.batch_size, self.seq_len + 1)
                    yield batch.long()


class CosineSchedule:
    def __init__(self, lr: float, total_steps: int, warmup: float = 0.05) -> None:
        self.lr = lr
        self.final_lr = 0
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup)

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.lr * step / self.warmup_steps
        if step < self.total_steps:
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.final_lr + 0.5 * (self.lr - self.final_lr) * (1 + math.cos(progress * math.pi))
        return self.final_lr


def get_loss(model: LlamaForCausalLM, batch: torch.Tensor):
    logits = model(batch[:, :-1].long()).logits.flatten(0, 1)
    labels = batch[:, 1:].long().flatten()
    return torch.nn.functional.cross_entropy(logits, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # default config is 470M
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--ffn_size", type=int, default=4096)
    parser.add_argument("--head_dim", type=int, default=64)

    parser.add_argument("--int8_mixed_precision", type=json.loads)
    parser.add_argument("--int8_quantized_training", type=json.loads)
    parser.add_argument("--quantize_lm_head", action="store_true")
    parser.add_argument("--activation_checkpointing", action="store_true")

    parser.add_argument("--dataset", default="tinystories")
    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--gradient_accumulation", type=int, default=1)

    parser.add_argument("--optim", default="optimizers.AdamW")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--optim_kwargs", type=json.loads, default=dict())
    parser.add_argument("--cosine_lr_schedule", action="store_true")

    parser.add_argument("--ckpt_interval", type=int, default=1000)
    parser.add_argument("--project", default="llm_pretraining")
    parser.add_argument("--run_name")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    assert args.batch_size % args.gradient_accumulation == 0
    if args.seed is not None:
        torch.manual_seed(args.seed)
    if args.profile:
        args.n_steps = 5

    config = LlamaConfig(
        hidden_size=args.d_model,
        intermediate_size=args.ffn_size,
        num_hidden_layers=args.depth,
        num_attention_heads=args.d_model // args.head_dim,
        max_position_embeddings=args.seq_len,
        use_cache=False,
    )
    model = LlamaForCausalLM(config).bfloat16().cuda()
    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()
    quantize_model(model.model, args.int8_mixed_precision, args.int8_quantized_training)
    if args.quantize_lm_head:
        quantize_model(model.lm_head, args.int8_mixed_precision, args.int8_quantized_training)
    print_model_stats(model)

    optim_cls = get_optim_cls(args.optim)
    optim = optim_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, **args.optim_kwargs)
    lr_schedule = CosineSchedule(args.lr, args.n_steps) if args.cosine_lr_schedule else None

    ds = TokenDataset(args.dataset, "train", args.batch_size, args.seq_len)
    dloader = iter(DataLoader(ds, batch_size=None, num_workers=1, pin_memory=True))

    save_dir = Path("runs/llm_pretrain") / f"{args.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        dir="/tmp", config=args, project=args.project, name=args.run_name, mode="disabled" if args.profile else None
    )

    step = 0
    log_interval = 50
    bsize = args.batch_size // args.gradient_accumulation
    pbar = tqdm(total=args.n_steps, dynamic_ncols=True)
    model.train()
    time0 = time.time()
    if args.profile:
        prof = torch.profiler.profile()

    while step < args.n_steps:
        for _ in range(args.gradient_accumulation):
            batch = next(dloader).cuda()
            loss = torch.compile(get_loss)(model, batch)
            loss.backward()

        if lr_schedule is not None:
            lr = lr_schedule.get_lr(step)
            for param_group in optim.param_groups:
                param_group["lr"] = lr

        if step % log_interval == 0:
            log_dict = dict(
                loss=loss.item(),
                grad_norm=get_grad_norm(model),
                lr=optim.param_groups[0]["lr"],
                num_tokens_seen_millions=args.batch_size * args.seq_len * step / 1e6,
                max_memory_allocated=torch.cuda.max_memory_allocated(),
            )
            if step > 0:
                time1 = time.time()
                log_dict["tokens_per_second"] = args.batch_size * args.seq_len * log_interval / (time1 - time0)
                time0 = time1
            run.log(log_dict, step=step)
            pbar.set_postfix(loss=log_dict["loss"])

        optim.step()
        optim.zero_grad()

        step += 1
        pbar.update()
        if args.profile and step == 1:
            prof.start()

        if args.ckpt_interval > 0 and step % args.ckpt_interval == 0:
            ckpt = dict(
                model=model.state_dict(),
                optim=optim.state_dict(),
                step=step,
            )
            torch.save(ckpt, save_dir / "last.pth")

    run.finish()
    if args.profile:
        prof.stop()
        prof.export_chrome_trace("trace.json.gz")
