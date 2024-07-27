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
from torch import Tensor, nn
from torchao.prototype import low_bit_optim
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def _data_iter(tokens_list: list[Tensor], batch_size: int, seq_len_multiple: int = 256):
    n = len(tokens_list)

    while True:
        # shuffle
        tokens_list = [tokens_list[idx] for idx in torch.randperm(n)]

        for i in range(0, n - batch_size + 1, batch_size):
            tokens_batch = tokens_list[i : i + batch_size]
            length = max(math.ceil(x.shape[0] / seq_len_multiple) * seq_len_multiple for x in tokens_batch)

            inputs = torch.zeros(batch_size, length, dtype=torch.int64)
            labels = torch.zeros(batch_size, length, dtype=torch.int64)
            for _i, tokens in enumerate(tokens_batch):
                n_toks = tokens.shape[0]
                inputs[_i, :n_toks] = tokens
                labels[_i, :n_toks] = tokens
                labels[_i, n_toks:] = -100

            yield inputs.cuda(), labels.cuda()


def create_dataset(
    model: str,
    dataset: str,
    split: str,
    batch_size: int,
    max_seq_len: int,
    question_key: str | None = None,
    answer_key: str | None = None,
):
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = max_seq_len

    def apply_chat_template(example):
        if "messages" in example:
            messages = example["messages"]
        else:
            messages = [
                dict(role="user", content=example[question_key]),
                dict(role="assistant", content=example[answer_key]),
            ]
        return dict(
            input_ids=tokenizer.apply_chat_template(
                messages, add_generation_prompt=False, tokenize=True, truncation=True
            )
        )

    ds = load_dataset(dataset, split=split).with_format("torch")
    if "messages" not in ds.features:
        assert question_key is not None and answer_key is not None

    tokens_list = ds.map(apply_chat_template, remove_columns=ds.features)["input_ids"]
    return _data_iter(tokens_list, batch_size), len(ds)


def get_loss(model, inputs, labels):
    return model(inputs, labels=labels).loss


def setup_fuse_backward_with_optim_step(
    model: nn.Module,
    optimizer: str,
    lr: float,
    weight_decay: float,
    optimizer_kwargs: dict,
):
    optim_cls = dict(
        AdamW=torch.optim.AdamW,
        AdamW8bit=low_bit_optim.AdamW8bit,
        AdamWFp8=low_bit_optim.AdamWFp8,  # requires PyTorch 2.4. embedding layer should be frozen.
    )[optimizer]

    optim_dict = {p: optim_cls([p], lr=lr, weight_decay=weight_decay, **optimizer_kwargs) for p in model.parameters()}

    def optim_hook(p):
        optim_dict[p].step()
        optim_dict[p].zero_grad()

    for p in model.parameters():
        if p.requires_grad:
            p.register_post_accumulate_grad_hook(optim_hook)

    return optim_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--max_seq_len", type=int, default=2048)

    parser.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--split", required=True)
    parser.add_argument("--question_key")
    parser.add_argument("--answer_key")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_steps", type=int, default=1000)

    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--optimizer_kwargs", type=json.loads, default=dict())

    parser.add_argument("--ckpt_interval", type=int, default=1000)
    parser.add_argument("--project")
    parser.add_argument("--run_name")
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
    )
    model.gradient_checkpointing_enable()
    model.get_input_embeddings().requires_grad_(False)
    print(f"Vocab size: {model.vocab_size:,}")
    print(f"No. of trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"No. of non-trainable params: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")
    print(f"No. of buffers: {sum(p.numel() for p in model.buffers()):,}")

    optim_dict = setup_fuse_backward_with_optim_step(
        model, args.optimizer, args.lr, args.weight_decay, args.optimizer_kwargs
    )

    train_data_iter, train_size = create_dataset(
        args.model,
        args.dataset,
        args.split,
        args.batch_size,
        args.max_seq_len,
        question_key=args.question_key,
        answer_key=args.answer_key,
    )
    print(f"Training dataset size: {train_size:,}")
    print(f"Each epoch will takes {train_size // args.batch_size:,} iters to finish")

    Path("wandb_logs").mkdir(exist_ok=True)
    wandb_run = wandb.init(project=args.project, name=args.run_name, dir="wandb_logs", config=args)

    ckpt_dir = Path("checkpoints") / args.dataset.replace("/", "_") / args.model.replace("/", "_")
    ckpt_dir = ckpt_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    pbar = tqdm(total=args.n_steps, dynamic_ncols=True)
    model.train()

    while step < args.n_steps:
        inputs, labels = next(train_data_iter)
        loss = torch.compile(get_loss, fullgraph=True)(model, inputs, labels)
        loss.backward()

        step += 1
        pbar.update()

        if step % 50 == 0:
            loss_python = loss.item()
            pbar.set_postfix(loss=loss_python)
            wandb_run.log(dict(loss=loss_python), step=step)

        if args.ckpt_interval > 0 and step % args.ckpt_interval == 0:
            # TODO: checkpoint optimizer
            ckpt = dict(
                model=model.state_dict(),
                step=step,
            )
            torch.save(ckpt, ckpt_dir / "last.pth")

    wandb_run.finish()