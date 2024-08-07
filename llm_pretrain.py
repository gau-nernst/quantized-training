import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import wandb
from tqdm import tqdm
from transformers import LlamaConfig, LlamaForCausalLM

from train_utils import get_grad_norm, get_optim_cls, print_model_stats, quantize_model


def get_loss(model: LlamaForCausalLM, batch: torch.Tensor):
    return model(batch, labels=batch).loss


def get_tinystories():
    save_path = Path("tokenized_data.bin")

    if not save_path.exists():
        import sentencepiece as spm
        from huggingface_hub import hf_hub_download

        tokenizer_path = hf_hub_download("meta-llama/Llama-2-7b", "tokenizer.model")
        tokenizer = spm.SentencePieceProcessor(tokenizer_path)
        assert tokenizer.vocab_size() < (1 << 16)  # make sure we can use uint16

        # do everything in memory. we have enough RAM
        filepath = hf_hub_download("roneneldan/TinyStories", "TinyStoriesV2-GPT4-train.txt", repo_type="dataset")
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
    parser.add_argument("--d_model", default=1024)
    parser.add_argument("--depth", default=24)
    parser.add_argument("--ffn_size", default=4096)
    parser.add_argument("--head_dim", default=64)

    parser.add_argument("--model_quantize")

    parser.add_argument("--n_steps", default=1000)
    parser.add_argument("--batch_size", default=4)
    parser.add_argument("--seq_len", default=2048)

    parser.add_argument("--optim", default="torch.optim.AdamW")
    parser.add_argument("--lr", default=3e-4)
    parser.add_argument("--weight_decay", default=1e-2)
    parser.add_argument("--optim_kwargs", type=json.loads, default=dict())

    parser.add_argument("--project", default="llm_pretraining")
    parser.add_argument("--run_name")
    parser.add_argument("--seed")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    config = LlamaConfig(
        hidden_size=args.d_model,
        intermediate_size=args.ffn_size,
        num_hidden_layers=args.depth,
        num_attention_heads=args.d_model // args.head_dim,
        max_position_embeddings=args.seq_len,
        use_cache=False,
    )
    model = LlamaForCausalLM(config).bfloat16().cuda()
    quantize_model(model, args.model_quantize)
    print_model_stats(model)

    optim_cls = get_optim_cls(args.optim)
    optim = optim_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, **args.optim_kwargs)

    data = get_tinystories().cuda()

    run = wandb.init(dir="/tmp", config=args, project=args.project, name=args.run_name)
    step = 0
    pbar = tqdm(total=args.n_steps, dynamic_ncols=True)
    model.train()
    time0 = time.time()

    while step < args.n_steps:
        # select a continuous chunk, then reshape it
        idx = torch.randint(0, data.shape[0] - args.batch_size * args.seq_len, (1,)).item()
        batch = data[idx : idx + args.batch_size * args.seq_len].view(args.batch_size, args.seq_len).long()

        loss = torch.compile(get_loss)(model, batch)
        loss.backward()

        if step % 50 == 0:
            log_dict = dict(
                loss=loss.item(),
                grad_norm=get_grad_norm(model),
                lr=optim.param_groups[0]["lr"],
                max_memory_allocated=torch.cuda.max_memory_allocated(),
            )
            if step > 0:
                time1 = time.time()
                log_dict["tokens_per_second"] = args.batch_size * args.seq_len * 50 / (time1 - time0)
                time0 = time1
            run.log(log_dict, step=step)
            pbar.set_postfix(loss=log_dict["loss"])

        optim.step()
        optim.zero_grad()

        step += 1
        pbar.update()

    run.finish()
