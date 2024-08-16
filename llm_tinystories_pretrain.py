import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import wandb
from tqdm import tqdm
from transformers import LlamaConfig, LlamaForCausalLM

from train_utils import get_grad_norm, get_optim_cls, print_model_stats, quantize_model


def get_loss(model: LlamaForCausalLM, batch: torch.Tensor):
    return model(batch, labels=batch).loss


def get_tinystories(split: str):
    assert split in ("train", "valid")
    save_path = Path(f"tinystories_{split}.bin")

    if not save_path.exists():
        import sentencepiece as spm
        from huggingface_hub import hf_hub_download

        tokenizer_path = hf_hub_download("meta-llama/Llama-2-7b", "tokenizer.model")
        tokenizer = spm.SentencePieceProcessor(tokenizer_path)
        assert tokenizer.vocab_size() < (1 << 16)  # make sure we can use uint16

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

    tokens = np.memmap(save_path, dtype=np.uint16, mode="r")
    return torch.from_numpy(tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # default config is 470M
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--ffn_size", type=int, default=4096)
    parser.add_argument("--head_dim", type=int, default=64)

    parser.add_argument("--model_quantize")
    parser.add_argument("--activation_checkpointing", action="store_true")

    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--gradient_accumulation", type=int, default=1)

    parser.add_argument("--optim", default="optimizers.AdamW")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--optim_kwargs", type=json.loads, default=dict())

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
    quantize_model(model, args.model_quantize)
    print_model_stats(model)

    optim_cls = get_optim_cls(args.optim)
    optim = optim_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, **args.optim_kwargs)

    data = get_tinystories("train").cuda()

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
            # randomly select a continuous chunk, then reshape it
            idx = torch.randint(0, data.shape[0] - bsize * args.seq_len, (1,)).item()
            batch = data[idx : idx + bsize * args.seq_len].view(bsize, args.seq_len).long()

            loss = torch.compile(get_loss)(model, batch)
            loss.backward()

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
