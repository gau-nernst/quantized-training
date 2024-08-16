import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import torch
import wandb
from datasets import load_dataset
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_utils import get_grad_norm, get_optim_cls, print_model_stats, quantize_model


def _data_iter(tokens_list: list[Tensor], batch_size: int, seq_len_multiple: int = 256):
    n = len(tokens_list)

    while True:
        # shuffle
        tokens_list = [tokens_list[idx] for idx in torch.randperm(n)]

        for i in range(0, n - batch_size + 1, batch_size):
            tokens_batch = tokens_list[i : i + batch_size]
            length = max(math.ceil(x.shape[0] / seq_len_multiple) * seq_len_multiple for x in tokens_batch)

            inputs = torch.zeros(batch_size, length, dtype=torch.int64)
            labels = torch.full((batch_size, length), -100, dtype=torch.int64)
            for _i, tokens in enumerate(tokens_batch):
                n_toks = tokens.shape[0]
                inputs[_i, :n_toks] = tokens
                labels[_i, :n_toks] = tokens

            yield inputs.cuda(), labels.cuda()


def get_metamathqa(tokenizer_id: str, batch_size: int, max_seq_len: int, seq_len_multiple: int = 256):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = max_seq_len

    def apply_template(example):
        text = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{query}\n\n"
            "### Response: Let's think step by step. {response}"
        ).format(query=example["query"], response=example["response"])
        return tokenizer(text, truncation=True, return_attention_mask=False)

    ds = load_dataset("meta-math/MetaMathQA", split="train").with_format("torch")
    tokens_list = ds.map(apply_template, remove_columns=ds.features)["input_ids"]
    return _data_iter(tokens_list, batch_size, seq_len_multiple), len(ds)


def get_loss(model, inputs, labels):
    return model(inputs, labels=labels).loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--freeze_embedding_layer", action="store_true")

    parser.add_argument("--weight_quantize", default="none")
    parser.add_argument("--activation_quantize", default="none")
    parser.add_argument("--grad_weight_compute", default="none")
    parser.add_argument("--compile", action="store_true")

    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--seq_len_multiple", type=int, default=256)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_steps", type=int, default=1000)

    parser.add_argument("--optim", default="optimizers.AdamW")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--optim_kwargs", type=json.loads, default=dict())

    parser.add_argument("--ckpt_interval", type=int, default=1000)
    parser.add_argument("--project")
    parser.add_argument("--run_name", default="debug")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # NOTE: must set max_position_embeddings to not store excessive positional encodings buffer
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        max_position_embeddings=args.max_seq_len,
        use_cache=False,
    )
    model.gradient_checkpointing_enable()
    if args.freeze_embedding_layer:
        model.get_input_embeddings().requires_grad_(False)

    # don't quantize lm_head, since it might be weight-tied to input embeddings
    quantize_model(model.get_decoder(), args.weight_quantize, args.activation_quantize, args.grad_weight_compute)

    print(f"Vocab size: {model.vocab_size:,}")
    print_model_stats(model)

    optim_cls = get_optim_cls(args.optim)
    optim = optim_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, **args.optim_kwargs)

    train_data_iter, train_size = get_metamathqa(
        args.model,
        args.batch_size,
        args.max_seq_len,
        seq_len_multiple=args.seq_len_multiple,
    )
    print(f"Training dataset size: {train_size:,}")
    print(f"Each epoch will takes {train_size // args.batch_size:,} iters to finish")

    save_dir = Path("runs/llm_finetune") / f"{args.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(project=args.project, name=args.run_name, config=args, dir="/tmp")

    step = 0
    log_interval = 50
    pbar = tqdm(total=args.n_steps, dynamic_ncols=True)
    model.train()

    while step < args.n_steps:
        inputs, labels = next(train_data_iter)
        loss_fn = torch.compile(get_loss, fullgraph=True) if args.compile else get_loss
        loss = loss_fn(model, inputs, labels)
        loss.backward()

        if step % log_interval == 0:
            log_dict = dict(
                loss=loss.item(),
                grad_norm=get_grad_norm(model),
                lr=optim.param_groups[0]["lr"],
            )
            run.log(log_dict, step=step)
            pbar.set_postfix(loss=log_dict["loss"])

        optim.step()
        optim.zero_grad()

        step += 1
        pbar.update()

        if args.ckpt_interval > 0 and step % args.ckpt_interval == 0:
            # TODO: checkpoint optimizer
            ckpt = dict(
                model=model.state_dict(),
                step=step,
            )
            torch.save(ckpt, save_dir / "last.pth")

    max_memory = torch.cuda.max_memory_allocated()
    run.log(dict(max_memory=max_memory))
    print(f"Max memory allocated: {max_memory / 1e9:.2f} GiB")

    run.finish()
